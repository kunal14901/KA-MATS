"""
KA-MATS Cryptoz · WebSocket Data Agent
Iknir Capital

Real-time market data via CCXT Pro WebSocket streams (watch_ohlcv).

Replaces REST polling (~200ms round-trip per symbol, sequential) with
push-based WebSocket updates (~5ms, all symbols concurrent).

Design:
  - REST warmup: backfill 300 bars per symbol once at startup so EMA200
    and other slow indicators converge immediately.
  - WebSocket steady-state: watch_ohlcv() pushes every bar update; we
    maintain a rolling buffer per symbol and recompute indicators only
    when a bar CLOSES (not on every intra-bar tick).
  - Graceful fallback: if the WebSocket stream errors repeatedly, the
    agent falls back to REST fetch for that symbol and keeps trying to
    re-establish the stream with exponential backoff.

Usage (standalone):
    agent = WebSocketDataAgent(symbols=["BTC/USDT", "ETH/USDT"])
    asyncio.run(agent.run(on_bar_close=my_callback))

Usage (bridged into the synchronous orchestrator):
    bridge = WSDataBridge(symbols=CRYPTO_SYMBOLS)
    bridge.start()                  # runs the event loop in a daemon thread
    df = bridge.get_dataframe("BTC/USDT")   # thread-safe snapshot
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import deque
from collections.abc import Awaitable, Callable

import pandas as pd
from loguru import logger

from agents.data_agent import CryptoDataAgent
from config.settings import CONFIG, DataConfig

BarCloseCallback = Callable[[str, pd.DataFrame], Awaitable[None] | None]


class WebSocketDataAgent:
    """
    Async OHLCV streaming via ccxt.pro with REST warmup and auto-reconnect.

    One watch loop per symbol runs concurrently. Bars accumulate in a
    bounded deque; indicator computation is delegated to CryptoDataAgent
    so the indicator definitions stay in exactly one place.
    """

    MAX_BUFFER = 500  # rolling bars kept per symbol
    RECONNECT_BASE_SEC = 2.0  # exponential backoff base
    RECONNECT_MAX_SEC = 300.0

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = None,
        cfg: DataConfig = None,
    ) -> None:
        self.cfg = cfg or CONFIG.data
        self.symbols = symbols
        self.timeframe = timeframe or self.cfg.timeframe

        self._rest_agent = CryptoDataAgent(cfg=self.cfg)
        self._exchange = None  # ccxt.pro instance, created inside the event loop

        # symbol → deque of [ts_ms, o, h, l, c, v]
        self._buffers: dict[str, deque[list]] = {sym: deque(maxlen=self.MAX_BUFFER) for sym in symbols}
        self._buffer_lock = threading.Lock()
        self._last_closed_ts: dict[str, int] = {}
        self._ws_failures: dict[str, int] = dict.fromkeys(symbols, 0)
        self._running = False

    # ── Exchange lifecycle ────────────────────────────────────

    def _create_pro_exchange(self):
        try:
            import ccxt.pro as ccxtpro
        except ImportError as e:
            raise ImportError("ccxt.pro not available. Upgrade ccxt: pip install -U ccxt") from e
        exchange_cls = getattr(ccxtpro, self.cfg.exchange, ccxtpro.binance)
        return exchange_cls(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )

    async def close(self) -> None:
        self._running = False
        if self._exchange is not None:
            with contextlib.suppress(Exception):
                await self._exchange.close()
            self._exchange = None

    # ── REST warmup ───────────────────────────────────────────

    def warmup(self, limit: int = 300) -> None:
        """Backfill buffers via REST so slow indicators converge at start."""
        for sym in self.symbols:
            df = self._rest_agent.fetch_ohlcv(sym, self.timeframe, limit=limit)
            if df is None or df.empty:
                logger.warning(f"[WSData] {sym}: warmup fetch failed")
                continue
            with self._buffer_lock:
                buf = self._buffers[sym]
                buf.clear()
                for ts, row in df.iterrows():
                    buf.append(
                        [
                            int(ts.value // 10**6),
                            row["open"],
                            row["high"],
                            row["low"],
                            row["close"],
                            row["volume"],
                        ]
                    )
                if buf:
                    self._last_closed_ts[sym] = buf[-1][0]
        logger.info(f"[WSData] Warmup complete for {len(self.symbols)} symbols")

    # ── Snapshot access (thread-safe) ─────────────────────────

    def get_dataframe(self, symbol: str, with_indicators: bool = True) -> pd.DataFrame | None:
        """Return a copy of the current buffer as an indicator-enriched DataFrame."""
        with self._buffer_lock:
            rows = list(self._buffers.get(symbol, []))
        if len(rows) < self.cfg.warmup_bars:
            return None
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index().astype(float)
        if with_indicators:
            df = self._rest_agent.compute_indicators(df)
        return df

    def last_price(self, symbol: str) -> float | None:
        with self._buffer_lock:
            buf = self._buffers.get(symbol)
            if not buf:
                return None
            return float(buf[-1][4])

    # ── WebSocket streaming ───────────────────────────────────

    async def _watch_symbol(self, symbol: str, on_bar_close: BarCloseCallback | None) -> None:
        """Watch one symbol forever; reconnect with exponential backoff on errors."""
        while self._running:
            try:
                candles = await self._exchange.watch_ohlcv(symbol, self.timeframe)
                self._ws_failures[symbol] = 0
                if not candles:
                    continue
                closed = self._ingest(symbol, candles)
                if closed and on_bar_close is not None:
                    df = self.get_dataframe(symbol)
                    if df is not None:
                        result = on_bar_close(symbol, df)
                        if asyncio.iscoroutine(result):
                            await result
            except asyncio.CancelledError:
                raise
            except Exception as e:
                fails = self._ws_failures[symbol] = self._ws_failures[symbol] + 1
                backoff = min(self.RECONNECT_BASE_SEC * (2**fails), self.RECONNECT_MAX_SEC)
                logger.warning(f"[WSData] {symbol}: stream error #{fails} ({e}) — retry in {backoff:.0f}s")
                # REST fallback so the buffer never goes stale during outage
                self._rest_fallback(symbol)
                await asyncio.sleep(backoff)

    def _ingest(self, symbol: str, candles: list) -> bool:
        """
        Merge incoming candles into the buffer.

        Returns True if a NEW bar closed (i.e. a candle with a timestamp newer
        than the last known one became final — meaning the previous bar is complete).
        """
        new_bar_closed = False
        with self._buffer_lock:
            buf = self._buffers[symbol]
            last_ts = self._last_closed_ts.get(symbol, 0)
            for c in candles:
                ts = int(c[0])
                if buf and buf[-1][0] == ts:
                    buf[-1] = list(c[:6])  # update the live (still-forming) bar
                elif ts > (buf[-1][0] if buf else 0):
                    if buf and buf[-1][0] > last_ts:
                        # previous bar is now final
                        self._last_closed_ts[symbol] = buf[-1][0]
                        new_bar_closed = True
                    buf.append(list(c[:6]))
        return new_bar_closed

    def _rest_fallback(self, symbol: str) -> None:
        """Refresh the buffer via REST when the WebSocket is down."""
        try:
            df = self._rest_agent.fetch_ohlcv(symbol, self.timeframe, limit=50)
            if df is None or df.empty:
                return
            with self._buffer_lock:
                buf = self._buffers[symbol]
                known = {row[0] for row in buf}
                for ts, row in df.iterrows():
                    ts_ms = int(ts.value // 10**6)
                    if ts_ms not in known:
                        buf.append([ts_ms, row["open"], row["high"], row["low"], row["close"], row["volume"]])
            logger.info(f"[WSData] {symbol}: REST fallback refreshed buffer")
        except Exception as e:
            logger.debug(f"[WSData] {symbol}: REST fallback failed — {e}")

    async def run(self, on_bar_close: BarCloseCallback | None = None) -> None:
        """
        Start streaming all symbols concurrently. Blocks until cancelled.

        on_bar_close(symbol, df) fires once per symbol per completed bar with
        the full indicator-enriched DataFrame.
        """
        self._running = True
        self._exchange = self._create_pro_exchange()
        logger.info(
            f"[WSData] Streaming {len(self.symbols)} symbols on "
            f"{self.cfg.exchange} {self.timeframe} via WebSocket"
        )
        try:
            await asyncio.gather(*(self._watch_symbol(sym, on_bar_close) for sym in self.symbols))
        finally:
            await self.close()


class WSDataBridge:
    """
    Synchronous facade over WebSocketDataAgent for the existing orchestrator.

    Runs the asyncio event loop in a daemon thread. The orchestrator keeps its
    bar-driven cadence but reads fresh, push-updated data instead of issuing
    REST calls — latency per bar drops from N×200ms to a dict lookup.
    """

    def __init__(self, symbols: list[str], timeframe: str = None) -> None:
        self.agent = WebSocketDataAgent(symbols=symbols, timeframe=timeframe)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, warmup: bool = True) -> None:
        if self._thread is not None:
            return
        if warmup:
            self.agent.warmup()

        def _run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self.agent.run())
            except Exception as e:
                logger.error(f"[WSData] Event loop terminated: {e}")

        self._thread = threading.Thread(target=_run_loop, daemon=True, name="ws-data")
        self._thread.start()
        logger.info("[WSData] Bridge thread started")

    def stop(self) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self.agent.close(), self._loop)
        self._thread = None

    def get_dataframe(self, symbol: str, with_indicators: bool = False) -> pd.DataFrame | None:
        # Default raw: the orchestrator computes indicators itself downstream.
        return self.agent.get_dataframe(symbol, with_indicators=with_indicators)

    def last_price(self, symbol: str) -> float | None:
        return self.agent.last_price(symbol)
