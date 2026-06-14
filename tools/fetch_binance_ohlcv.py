"""
tools/fetch_binance_ohlcv.py
============================
Fetches OHLCV from Binance via CCXT (no API key required for public data)
and saves a parquet in the same format as the yfinance cache.

Symbol mapping: yfinance uses "BTC-USD", Binance uses "BTC/USDT".
Output schema: DatetimeIndex + columns [open, high, low, close, volume]
               — identical to what load_data() produces from yfinance.

Usage (standalone):
    python tools/fetch_binance_ohlcv.py                  # daily bars
    python tools/fetch_binance_ohlcv.py --timeframe 1h   # 1h bars (v12 intrabar resolution)

Output:
    results/crypto_backtest/ohlcv_binance_2019_2026_v1.parquet      (1d)
    results/crypto_backtest/ohlcv_binance_1h_2019_2026_v1.parquet   (1h)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ccxt
import pandas as pd
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────

FETCH_START = "2019-04-01"
FETCH_END = "2026-01-01"
OUTPUT_DIR = ROOT / "results" / "crypto_backtest"
OUTPUT_FILE = OUTPUT_DIR / "ohlcv_binance_2019_2026_v1.parquet"
OUTPUT_FILE_1H = OUTPUT_DIR / "ohlcv_binance_1h_2019_2026_v1.parquet"

# yfinance symbol → Binance symbol
SYMBOL_MAP = {
    "BTC-USD": "BTC/USDT",
    "ETH-USD": "ETH/USDT",
    "SOL-USD": "SOL/USDT",
    "BNB-USD": "BNB/USDT",
    "ADA-USD": "ADA/USDT",
    "AVAX-USD": "AVAX/USDT",
    "DOT-USD": "DOT/USDT",
    "ATOM-USD": "ATOM/USDT",
    "NEAR-USD": "NEAR/USDT",
    "LINK-USD": "LINK/USDT",
    "UNI-USD": "UNI/USDT",
    "AAVE-USD": "AAVE/USDT",
    "XRP-USD": "XRP/USDT",
    "LTC-USD": "LTC/USDT",
    "DOGE-USD": "DOGE/USDT",
    "MATIC-USD": "MATIC/USDT",
    "FIL-USD": "FIL/USDT",
    "ALGO-USD": "ALGO/USDT",
    "XLM-USD": "XLM/USDT",
    "VET-USD": "VET/USDT",
}

# Dead coins for survivorship-bias correction (Phase 0a step 2)
# These have a known collapse date — data is fetched up to that date
DEAD_COIN_MAP = {
    "LUNC-USD": ("LUNA/USDT", "2022-05-12"),  # LUNA → LUNC collapse
    "FTT-USD": ("FTT/USDT", "2022-11-09"),  # FTX collapse
}

RATE_LIMIT_SLEEP = 0.4  # seconds between requests (Binance allows 1200/min)

# ── Fetcher ────────────────────────────────────────────────────────────────────


def _fetch_symbol(
    exchange: ccxt.Exchange, binance_sym: str, start: str, end: str, timeframe: str = "1d"
) -> pd.DataFrame | None:
    """
    Fetch bars at `timeframe` for binance_sym from start → end (exclusive).
    Returns DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
    """
    since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = pd.Timestamp(end)
    rows = []

    while True:
        try:
            candles = exchange.fetch_ohlcv(binance_sym, timeframe=timeframe, since=since_ms, limit=1000)
        except ccxt.BadSymbol:
            logger.warning(f"  {binance_sym}: symbol not found on Binance")
            return None
        except Exception as e:
            logger.warning(f"  {binance_sym}: fetch error ({e}) — retrying in 3s")
            time.sleep(3)
            continue

        if not candles:
            break
        rows.extend(candles)
        last_ts = pd.Timestamp(candles[-1][0], unit="ms")
        if last_ts >= end_ts or len(candles) < 1000:
            break
        since_ms = candles[-1][0] + 1
        time.sleep(RATE_LIMIT_SLEEP)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df.index.name = None
    # Trim to [start, end)
    df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
    df = df.sort_index().dropna(subset=["close"])
    return df


def fetch_all(include_dead: bool = True, timeframe: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch all live coins + dead coins. Returns {yf_symbol: df}."""
    exchange = ccxt.binance({"enableRateLimit": True})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}

    for yf_sym, bn_sym in SYMBOL_MAP.items():
        logger.info(f"  {yf_sym} ({bn_sym}, {timeframe}) ...")
        df = _fetch_symbol(exchange, bn_sym, FETCH_START, FETCH_END, timeframe=timeframe)
        if df is None or len(df) < 50:
            logger.warning(f"  {yf_sym}: insufficient data, skipping")
            continue
        frames[yf_sym] = df
        logger.success(f"  {yf_sym}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
        time.sleep(RATE_LIMIT_SLEEP)

    if include_dead:
        for yf_sym, (bn_sym, cutoff) in DEAD_COIN_MAP.items():
            logger.info(f"  {yf_sym} ({bn_sym}, cutoff {cutoff}, {timeframe}) ...")
            df = _fetch_symbol(exchange, bn_sym, FETCH_START, cutoff, timeframe=timeframe)
            if df is None or len(df) < 20:
                logger.warning(f"  {yf_sym}: insufficient data for dead coin")
                continue
            frames[yf_sym] = df
            logger.success(f"  {yf_sym}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
            time.sleep(RATE_LIMIT_SLEEP)

    return frames


def save(frames: dict[str, pd.DataFrame], path: Path = OUTPUT_FILE) -> None:
    combined = pd.concat(frames, axis=1)
    combined.to_parquet(path)
    logger.success(f"Saved {len(frames)} symbols -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Binance OHLCV via public CCXT")
    parser.add_argument(
        "--timeframe",
        default="1d",
        choices=["1d", "4h", "1h"],
        help="bar timeframe (1h used for v12 intrabar resolution)",
    )
    args = parser.parse_args()

    out_path = (
        OUTPUT_FILE
        if args.timeframe == "1d"
        else (OUTPUT_DIR / f"ohlcv_binance_{args.timeframe}_2019_2026_v1.parquet")
    )
    logger.info(
        f"Fetching {len(SYMBOL_MAP)} live + {len(DEAD_COIN_MAP)} dead coins "
        f"from Binance CCXT ({args.timeframe}) ..."
    )
    frames = fetch_all(include_dead=True, timeframe=args.timeframe)
    save(frames, out_path)
    logger.info(f"Done. {len(frames)} coins fetched -> {out_path}")
