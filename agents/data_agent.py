"""
KA-MATS Cryptoz · Data Agent
Iknir Capital

Fetches OHLCV from Binance via CCXT and computes technical indicators.
Works with any CCXT-supported exchange — just change exchange name in settings.

No API keys needed for historical data (public endpoint).
API keys only needed for paper/live order execution.

install: pip install ccxt pandas-ta
"""

from __future__ import annotations

import time

import pandas as pd
import ta as ta_lib
from loguru import logger

from config.settings import CONFIG, DataConfig


class CryptoDataAgent:
    """
    Fetches OHLCV bars from a crypto exchange and computes indicators.

    Output: dict[symbol → pd.DataFrame] with columns:
        open, high, low, close, volume,
        ema_20, ema_50, ema_200, rsi_14, atr_14,
        bb_upper, bb_lower, bb_mid, adx, macd, macd_signal
    """

    def __init__(self, cfg: DataConfig = None) -> None:
        self.cfg = cfg or CONFIG.data
        self._exchange = None  # lazy-init

    def _get_exchange(self):
        if self._exchange is None:
            try:
                import ccxt

                self._exchange = ccxt.binance(
                    {
                        "enableRateLimit": True,
                        "options": {"defaultType": "spot"},
                    }
                )
                logger.info(f"[DataAgent] Connected to {self.cfg.exchange} (public OHLCV)")
            except ImportError:
                raise ImportError("ccxt not installed. Run: pip install ccxt")
        return self._exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = None,
        limit: int = 500,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV bars from exchange. Returns DataFrame or None on error."""
        tf = timeframe or self.cfg.timeframe
        exchange = self._get_exchange()
        try:
            raw = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df.astype(float)
            logger.debug(
                f"[DataAgent] {symbol} | {len(df)} {tf} bars | {df.index[0].date()} → {df.index[-1].date()}"
            )
            return df
        except Exception as e:
            logger.warning(f"[DataAgent] {symbol}: fetch failed — {e}")
            return None

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators needed by strategy agents."""
        df = df.copy()

        # EMAs
        df["ema_20"] = ta_lib.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema_50"] = ta_lib.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df["ema_200"] = ta_lib.trend.EMAIndicator(df["close"], window=200).ema_indicator()

        # RSI
        df["rsi_14"] = ta_lib.momentum.RSIIndicator(df["close"], window=14).rsi()

        # ATR
        df["atr_14"] = ta_lib.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        ).average_true_range()

        # Bollinger Bands
        bb = ta_lib.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()

        # ADX + Directional Indicators (needed for regime direction classification)
        adx_ind = ta_lib.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx_ind.adx()
        df["plus_di"] = adx_ind.adx_pos()
        df["minus_di"] = adx_ind.adx_neg()

        # MACD + histogram
        macd_ind = ta_lib.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()  # histogram = macd - signal

        # Volume ratio
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma20"].replace(0, float("nan"))
        df["dollar_volume_20d"] = (df["close"] * df["volume"]).rolling(20).median()

        # 20-bar highest high (needed by MomentumBreakout near-high check)
        df["high_20"] = df["high"].rolling(20).max()

        # Z-score
        roll_mean = df["close"].rolling(20).mean()
        roll_std = df["close"].rolling(20).std(ddof=0)
        df["zscore"] = (df["close"] - roll_mean) / roll_std.replace(0, float("nan"))

        # ── VWAP (rolling 24h anchor, resets every 6 bars on 4h = 1 calendar day) ──
        # Crypto trades 24/7 — VWAP resets at UTC midnight each day.
        # Accumulated (price × volume) / accumulated volume over each calendar day.
        df["vwap"] = self._compute_rolling_vwap(df)

        # ── Keltner Channels (for Squeeze Momentum detection) ──
        # KC uses EMA20 ± (ATR20 × multiplier). Squeeze = BB inside KC.
        kc_atr = ta_lib.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=20
        ).average_true_range()
        kc_mult = 1.5
        df["kc_upper"] = df["ema_20"] + kc_mult * kc_atr
        df["kc_lower"] = df["ema_20"] - kc_mult * kc_atr

        # ── Squeeze flag: BB is inside KC → volatility coiling ──
        df["squeeze_on"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])

        # ── Squeeze momentum: delta-momentum oscillator (LazyBear style) ──
        # Midpoint of KC range
        kc_mid = (df["kc_upper"] + df["kc_lower"]) / 2
        # Momentum = close - average(average(high,low), EMA20 midpoint)
        hl_mid = (df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2
        df["squeeze_val"] = df["close"] - ((hl_mid + kc_mid) / 2)
        # Smooth with linear regression slope (5-bar)
        df["squeeze_mom"] = (
            df["squeeze_val"]
            .rolling(5)
            .apply(
                lambda x: float(pd.Series(x).corr(pd.Series(range(len(x)))) * x.std()),
                raw=True,
            )
        )

        # ── Heikin Ashi candles (computed from raw OHLCV) ──
        df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        # HA open is cumulative — seed with first bar's midpoint, then EMA-style accumulation
        ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open.append((ha_open[-1] + df["ha_close"].iloc[i - 1]) / 2)
        df["ha_open"] = ha_open
        df["ha_high"] = df[["high", "ha_open", "ha_close"]].max(axis=1)
        df["ha_low"] = df[["low", "ha_open", "ha_close"]].min(axis=1)
        # HA body direction and size (needed by HeikinAshiTrendConfirm)
        df["ha_bullish"] = (df["ha_close"] > df["ha_open"]).astype(float)
        df["ha_body_pct"] = abs(df["ha_close"] - df["ha_open"]) / df["ha_close"].replace(0, float("nan"))
        # "No upper wick" on a bull bar = strong trend resumption signal.
        # Threshold: upper wick < 1.5% of close. Original 0.1% was too tight —
        # crypto 4h bars always have some upper wick noise even on strong bull bars.
        df["ha_no_upper_wick"] = (
            (df["ha_close"] > df["ha_open"])
            & ((df["ha_high"] - df["ha_close"]) / df["ha_close"].replace(0, float("nan")) < 0.015)
            & (df["ha_body_pct"] > 0.003)  # body must be > 0.3% (filter out doji bars)
        ).astype(float)

        return df

    @staticmethod
    def _compute_rolling_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Compute daily-anchored VWAP for 24/7 crypto markets.
        Resets at UTC midnight (every 6 bars on 4h timeframe).
        Returns a Series aligned with df.index.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = typical_price * df["volume"]

        vwap = pd.Series(index=df.index, dtype=float)
        cumtp = 0.0
        cumvol = 0.0
        prev_date = None

        for ts, row in df.iterrows():
            curr_date = ts.date() if hasattr(ts, "date") else ts
            if curr_date != prev_date:
                # New calendar day — reset accumulators
                cumtp = 0.0
                cumvol = 0.0
                prev_date = curr_date
            cumtp += tp_vol.loc[ts]
            cumvol += row["volume"]
            vwap.loc[ts] = cumtp / cumvol if cumvol > 0 else float("nan")

        return vwap

    def fetch_all(
        self,
        symbols: list[str],
        timeframe: str = None,
        limit: int = 500,
        delay_sec: float = 0.3,
    ) -> dict[str, pd.DataFrame]:
        """Fetch + compute indicators for all symbols. Returns {symbol: df}."""
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.fetch_ohlcv(sym, timeframe, limit)
            if df is not None and len(df) >= self.cfg.warmup_bars:
                df = self.compute_indicators(df)
                result[sym] = df
            else:
                logger.warning(f"[DataAgent] {sym}: insufficient bars, skipping")
            time.sleep(delay_sec)
        logger.info(f"[DataAgent] Fetched {len(result)}/{len(symbols)} symbols")
        return result
