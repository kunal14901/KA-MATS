"""
KA-MATS Cryptoz · Exchange Connector Factory
Iknir Capital

Supported:
  binance_spot      — Binance spot market (real or testnet)
  bybit_spot        — Bybit spot market (real or testnet)
  binance_futures   — Binance USDT-M perpetuals (real or testnet) — SHORTS enabled
  bybit_futures     — Bybit USDT perpetuals (real or testnet) — SHORTS enabled

All connectors share the same interface:
  .market_buy(symbol, qty) → order dict
  .market_sell(symbol, qty) → order dict
  .limit_buy(symbol, qty, price) → order dict
  .limit_sell(symbol, qty, price) → order dict
  .get_balance(asset="USDT") → float
  .get_ticker(symbol) → float (last price)
  .cancel_all(symbol) → None

Futures connectors additionally provide:
  .supports_shorts == True
  .open_short(symbol, qty) → order dict   (market sell to open)
  .close_short(symbol, qty) → order dict  (reduce-only market buy)
  .set_leverage(symbol, leverage) → None  (default 1× for safety)

Testnet auto-detection:
  If api_key/secret contain "testnet" in env var name, sandbox mode is enabled.
"""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod

from loguru import logger


class ExchangeConnector(ABC):
    """Base interface for all exchange connectors."""

    #: True only for futures/margin connectors that can hold short positions.
    supports_shorts: bool = False

    #: True when the connector implements exchange-side protective exits
    #: (resting stop-loss / take-profit orders). Phase 1 measured intrabar
    #: exchange-side exits at +0.15 Sharpe vs daily-close local bookkeeping.
    supports_protective_exits: bool = False

    @abstractmethod
    def market_buy(self, symbol: str, qty: float) -> dict: ...

    @abstractmethod
    def market_sell(self, symbol: str, qty: float) -> dict: ...

    @abstractmethod
    def get_balance(self, asset: str = "USDT") -> float: ...

    @abstractmethod
    def get_ticker(self, symbol: str) -> float: ...

    def limit_buy(self, symbol: str, qty: float, price: float) -> dict:
        raise NotImplementedError

    def limit_sell(self, symbol: str, qty: float, price: float) -> dict:
        raise NotImplementedError

    def open_short(self, symbol: str, qty: float) -> dict:
        raise NotImplementedError("Shorts require a futures connector")

    def close_short(self, symbol: str, qty: float) -> dict:
        raise NotImplementedError("Shorts require a futures connector")

    # ── Protective exits (exchange-side stop-loss + take-profit) ──────────────
    # Returned dict shape (stored in the position state and passed back to the
    # other two methods):
    #   {"kind": "oco"|"pair", "stop_id": str, "tp_id": str,
    #    "stop_price": float, "tp_price": float, "list_id": str|None}

    def place_protective_exit(
        self, symbol: str, qty: float, stop_price: float, take_profit: float, is_long: bool = True
    ) -> dict | None:
        """Place resting stop-loss + take-profit orders for an open position.
        Returns None when the connector does not support protective exits."""
        return None

    def cancel_protective_exit(self, symbol: str, protective: dict) -> None:
        """Cancel resting protective orders (e.g. before a locally-decided exit)."""
        return None

    def fetch_order_safe(self, order_id: str, symbol: str) -> dict | None:
        """Fetch a single order's status; None when lookup fails."""
        return None

    def cancel_order(self, order_id: str, symbol: str) -> None:
        """Cancel a single resting order. Failure is non-fatal (logged upstream)."""
        raise NotImplementedError


class BinanceSpot(ExchangeConnector):
    """Binance spot connector via CCXT."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        import ccxt

        opts: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        self._exchange = ccxt.binance(opts)
        self._testnet = testnet
        if testnet:
            self._exchange.set_sandbox_mode(True)
            # Binance spot testnet does not support the SAPI currency metadata
            # endpoints CCXT may call during load_markets().
            self._exchange.options["fetchCurrencies"] = False
        self._exchange.load_markets()
        tag = "TESTNET" if testnet else "LIVE"
        logger.info(f"[Exchange] Binance {tag} connected")

    def market_buy(self, symbol: str, qty: float) -> dict:
        return self._retry(lambda: self._exchange.create_market_buy_order(symbol, qty))

    def market_sell(self, symbol: str, qty: float) -> dict:
        return self._retry(lambda: self._exchange.create_market_sell_order(symbol, qty))

    def limit_buy(self, symbol: str, qty: float, price: float) -> dict:
        return self._retry(lambda: self._exchange.create_limit_buy_order(symbol, qty, price))

    def limit_sell(self, symbol: str, qty: float, price: float) -> dict:
        return self._retry(lambda: self._exchange.create_limit_sell_order(symbol, qty, price))

    def cancel_order(self, order_id: str, symbol: str) -> None:
        self._retry(lambda: self._exchange.cancel_order(order_id, symbol))

    def get_balance(self, asset: str = "USDT") -> float:
        bal = self._retry(lambda: self._exchange.fetch_balance())
        return float(bal.get(asset, {}).get("free", 0.0))

    def get_ticker(self, symbol: str) -> float:
        t = self._retry(lambda: self._exchange.fetch_ticker(symbol))
        return float(t["last"])

    # ── Protective exits: Binance spot OCO (one-cancels-other) ────────────────
    # Spot balances are locked by resting sell orders, so stop + TP must be a
    # single OCO list — two independent sells would double-lock the coins.
    # The stop leg is a STOP_LOSS_LIMIT with the limit set slightly below the
    # trigger (bounded slippage); the TP leg is a plain LIMIT at target.

    supports_protective_exits = True
    STOP_LIMIT_OFFSET_PCT = 0.005  # stop-limit price 0.5% below stop trigger

    def place_protective_exit(
        self, symbol: str, qty: float, stop_price: float, take_profit: float, is_long: bool = True
    ) -> dict | None:
        if not is_long:
            return None  # spot cannot hold shorts
        ex = self._exchange
        market = ex.market(symbol)
        stop_limit = stop_price * (1.0 - self.STOP_LIMIT_OFFSET_PCT)
        request = {
            "symbol": market["id"],
            "side": "SELL",
            "quantity": ex.amount_to_precision(symbol, qty),
            "price": ex.price_to_precision(symbol, take_profit),
            "stopPrice": ex.price_to_precision(symbol, stop_price),
            "stopLimitPrice": ex.price_to_precision(symbol, stop_limit),
            "stopLimitTimeInForce": "GTC",
        }
        # Endpoint name moved across Binance/ccxt versions; try newest first.
        last_err: Exception | None = None
        for method_name in ("private_post_orderlist_oco", "private_post_order_oco"):
            method = getattr(ex, method_name, None)
            if method is None:
                continue
            try:
                resp = self._retry(lambda m=method: m(request))
                break
            except Exception as e:
                last_err = e
                resp = None
        else:
            resp = None
        if resp is None:
            raise RuntimeError(f"OCO placement failed for {symbol}: {last_err}")

        stop_id, tp_id = None, None
        for report in resp.get("orderReports", resp.get("orders", [])):
            otype = report.get("type", "")
            oid = str(report.get("orderId", ""))
            if "STOP" in otype:
                stop_id = oid
            elif otype in ("LIMIT_MAKER", "LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT"):
                tp_id = oid
        logger.info(
            f"[Exchange] OCO placed {symbol} | stop={stop_price:.6g} "
            f"tp={take_profit:.6g} | stop_id={stop_id} tp_id={tp_id}"
        )
        return {
            "kind": "oco",
            "stop_id": stop_id,
            "tp_id": tp_id,
            "stop_price": float(stop_price),
            "tp_price": float(take_profit),
            "list_id": str(resp.get("orderListId", "")) or None,
        }

    def cancel_protective_exit(self, symbol: str, protective: dict) -> None:
        # Cancelling either leg of an OCO cancels the whole list.
        for oid in (protective.get("stop_id"), protective.get("tp_id")):
            if not oid:
                continue
            try:
                self._retry(lambda o=oid: self._exchange.cancel_order(o, symbol))
                return
            except Exception as e:
                logger.debug(f"[Exchange] cancel {symbol} order {oid}: {e}")
        logger.warning(f"[Exchange] OCO cancel {symbol}: no leg could be cancelled (may already be filled)")

    def fetch_order_safe(self, order_id: str, symbol: str) -> dict | None:
        try:
            return self._retry(lambda: self._exchange.fetch_order(order_id, symbol))
        except Exception as e:
            logger.debug(f"[Exchange] fetch_order {symbol}/{order_id}: {e}")
            return None

    @staticmethod
    def _retry(fn, retries: int = 3, delay: float = 1.0):
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"[Exchange] Retry {attempt + 1}/{retries}: {e}")
                time.sleep(delay * (attempt + 1))


class BybitSpot(ExchangeConnector):
    """Bybit spot connector via CCXT."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        import ccxt

        opts: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if testnet:
            opts["options"]["sandboxMode"] = True
        self._exchange = ccxt.bybit(opts)
        self._testnet = testnet
        self._exchange.load_markets()
        tag = "TESTNET" if testnet else "LIVE"
        logger.info(f"[Exchange] Bybit {tag} connected")

    def market_buy(self, symbol: str, qty: float) -> dict:
        return self._retry(lambda: self._exchange.create_market_buy_order(symbol, qty))

    def market_sell(self, symbol: str, qty: float) -> dict:
        return self._retry(lambda: self._exchange.create_market_sell_order(symbol, qty))

    def limit_buy(self, symbol: str, qty: float, price: float) -> dict:
        return self._retry(lambda: self._exchange.create_limit_buy_order(symbol, qty, price))

    def limit_sell(self, symbol: str, qty: float, price: float) -> dict:
        return self._retry(lambda: self._exchange.create_limit_sell_order(symbol, qty, price))

    def cancel_order(self, order_id: str, symbol: str) -> None:
        self._retry(lambda: self._exchange.cancel_order(order_id, symbol))

    def fetch_order_safe(self, order_id: str, symbol: str) -> dict | None:
        try:
            return self._retry(lambda: self._exchange.fetch_order(order_id, symbol))
        except Exception as e:
            logger.debug(f"[Exchange] fetch_order {symbol}/{order_id}: {e}")
            return None

    def get_balance(self, asset: str = "USDT") -> float:
        bal = self._retry(lambda: self._exchange.fetch_balance())
        return float(bal.get(asset, {}).get("free", 0.0))

    def get_ticker(self, symbol: str) -> float:
        t = self._retry(lambda: self._exchange.fetch_ticker(symbol))
        return float(t["last"])

    @staticmethod
    def _retry(fn, retries: int = 3, delay: float = 1.0):
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"[Exchange] Retry {attempt + 1}/{retries}: {e}")
                time.sleep(delay * (attempt + 1))


class _FuturesBase(ExchangeConnector):
    """
    Shared implementation for USDT-margined perpetual futures connectors.

    Shorts are first-class: open_short() market-sells to open a short position,
    close_short() market-buys with reduceOnly so it can never flip into a long.
    Leverage defaults to 1× (same exposure as spot — shorts gain capability,
    not extra risk). Raise via set_leverage() only if you know what you're doing.
    """

    supports_shorts = True
    DEFAULT_LEVERAGE = 1

    _exchange = None  # set by subclass __init__
    _tag = "FUTURES"

    def market_buy(self, symbol: str, qty: float) -> dict:
        return self._retry(lambda: self._exchange.create_market_buy_order(symbol, qty))

    def market_sell(self, symbol: str, qty: float) -> dict:
        return self._retry(lambda: self._exchange.create_market_sell_order(symbol, qty))

    def limit_buy(self, symbol: str, qty: float, price: float) -> dict:
        return self._retry(lambda: self._exchange.create_limit_buy_order(symbol, qty, price))

    def limit_sell(self, symbol: str, qty: float, price: float) -> dict:
        return self._retry(lambda: self._exchange.create_limit_sell_order(symbol, qty, price))

    def cancel_order(self, order_id: str, symbol: str) -> None:
        self._retry(lambda: self._exchange.cancel_order(order_id, symbol))

    def open_short(self, symbol: str, qty: float) -> dict:
        """Open a short: market sell on a perpetual contract."""
        order = self._retry(lambda: self._exchange.create_market_sell_order(symbol, qty))
        logger.info(f"[Exchange:{self._tag}] OPEN SHORT {symbol} qty={qty:.6f}")
        return order

    def close_short(self, symbol: str, qty: float) -> dict:
        """Close a short: reduce-only market buy (cannot accidentally go long)."""
        order = self._retry(
            lambda: self._exchange.create_market_buy_order(symbol, qty, params={"reduceOnly": True})
        )
        logger.info(f"[Exchange:{self._tag}] CLOSE SHORT {symbol} qty={qty:.6f}")
        return order

    def set_leverage(self, symbol: str, leverage: int = None) -> None:
        lev = leverage or self.DEFAULT_LEVERAGE
        try:
            self._retry(lambda: self._exchange.set_leverage(lev, symbol))
            logger.info(f"[Exchange:{self._tag}] {symbol} leverage set to {lev}x")
        except Exception as e:
            logger.warning(f"[Exchange:{self._tag}] set_leverage({symbol}, {lev}) failed: {e}")

    # ── Protective exits: two reduce-only trigger orders ──────────────────────
    # Futures margin is position-based, so stop and TP can rest simultaneously
    # as independent reduce-only orders (no balance double-lock). There is no
    # OCO on futures — the surviving sibling is cancelled by the reconciler in
    # LiveExecution after one leg fills.

    supports_protective_exits = True

    def place_protective_exit(
        self, symbol: str, qty: float, stop_price: float, take_profit: float, is_long: bool = True
    ) -> dict | None:
        ex = self._exchange
        exit_side = "sell" if is_long else "buy"
        amount = float(ex.amount_to_precision(symbol, qty))

        stop_order = self._retry(
            lambda: ex.create_order(
                symbol,
                "market",
                exit_side,
                amount,
                None,
                {"stopLossPrice": float(ex.price_to_precision(symbol, stop_price)), "reduceOnly": True},
            )
        )
        try:
            tp_order = self._retry(
                lambda: ex.create_order(
                    symbol,
                    "market",
                    exit_side,
                    amount,
                    None,
                    {
                        "takeProfitPrice": float(ex.price_to_precision(symbol, take_profit)),
                        "reduceOnly": True,
                    },
                )
            )
        except Exception:
            # Don't leave a lone stop order in an inconsistent half-placed state
            with contextlib.suppress(Exception):
                ex.cancel_order(stop_order["id"], symbol)
            raise

        logger.info(
            f"[Exchange:{self._tag}] protective pair placed {symbol} | "
            f"stop={stop_price:.6g} tp={take_profit:.6g}"
        )
        return {
            "kind": "pair",
            "stop_id": str(stop_order.get("id", "")),
            "tp_id": str(tp_order.get("id", "")),
            "stop_price": float(stop_price),
            "tp_price": float(take_profit),
            "list_id": None,
        }

    def cancel_protective_exit(self, symbol: str, protective: dict) -> None:
        for oid in (protective.get("stop_id"), protective.get("tp_id")):
            if not oid:
                continue
            try:
                self._retry(lambda o=oid: self._exchange.cancel_order(o, symbol))
            except Exception as e:
                logger.debug(f"[Exchange:{self._tag}] cancel {symbol}/{oid}: {e}")

    def fetch_order_safe(self, order_id: str, symbol: str) -> dict | None:
        try:
            return self._retry(lambda: self._exchange.fetch_order(order_id, symbol))
        except Exception as e:
            logger.debug(f"[Exchange:{self._tag}] fetch_order {symbol}/{order_id}: {e}")
            return None

    def get_balance(self, asset: str = "USDT") -> float:
        bal = self._retry(lambda: self._exchange.fetch_balance())
        return float(bal.get(asset, {}).get("free", 0.0))

    def get_ticker(self, symbol: str) -> float:
        t = self._retry(lambda: self._exchange.fetch_ticker(symbol))
        return float(t["last"])

    @staticmethod
    def _retry(fn, retries: int = 3, delay: float = 1.0):
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"[Exchange] Retry {attempt + 1}/{retries}: {e}")
                time.sleep(delay * (attempt + 1))


class BinanceFutures(_FuturesBase):
    """Binance USDT-M perpetual futures connector via CCXT (shorts enabled)."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        import ccxt

        self._exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},  # USDT-M perpetuals
            }
        )
        if testnet:
            # ccxt >= 4.5 dropped futures sandbox mode (testnet.binancefuture.com).
            # Binance replaced it with Demo Trading (demo-fapi.binance.com) —
            # keys come from binance.com → Demo Trading → API Key tab.
            self._exchange.enable_demo_trading(True)
        self._exchange.load_markets()
        self._testnet = testnet
        self._tag = "Binance-Futures-" + ("DEMO" if testnet else "LIVE")
        logger.info(f"[Exchange] {self._tag} connected (shorts ENABLED, 1x default)")


class BybitFutures(_FuturesBase):
    """Bybit USDT perpetual futures connector via CCXT (shorts enabled)."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        import ccxt

        self._exchange = ccxt.bybit(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        if testnet:
            self._exchange.set_sandbox_mode(True)
        self._exchange.load_markets()
        self._testnet = testnet
        self._tag = "Bybit-Futures-" + ("TESTNET" if testnet else "LIVE")
        logger.info(f"[Exchange] {self._tag} connected (shorts ENABLED, 1x default)")


def create_exchange(mode: str, api_key: str, api_secret: str) -> ExchangeConnector | None:
    """Factory: create exchange connector from mode string.

    Modes:
        paper                    → None (no exchange needed)
        binance_testnet          → BinanceSpot(testnet=True)
        binance_live             → BinanceSpot(testnet=False)
        bybit_testnet            → BybitSpot(testnet=True)
        bybit_live               → BybitSpot(testnet=False)
        binance_futures_testnet  → BinanceFutures(testnet=True)   — shorts enabled
        binance_futures_live     → BinanceFutures(testnet=False)  — shorts enabled
        bybit_futures_testnet    → BybitFutures(testnet=True)     — shorts enabled
        bybit_futures_live       → BybitFutures(testnet=False)    — shorts enabled
    """
    if mode == "paper":
        return None

    if not api_key or not api_secret:
        raise ValueError(
            f"Mode '{mode}' requires API credentials. Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET in .env"
        )

    registry = {
        "binance_testnet": lambda: BinanceSpot(api_key, api_secret, testnet=True),
        "binance_live": lambda: BinanceSpot(api_key, api_secret, testnet=False),
        "bybit_testnet": lambda: BybitSpot(api_key, api_secret, testnet=True),
        "bybit_live": lambda: BybitSpot(api_key, api_secret, testnet=False),
        "binance_futures_testnet": lambda: BinanceFutures(api_key, api_secret, testnet=True),
        "binance_futures_live": lambda: BinanceFutures(api_key, api_secret, testnet=False),
        "bybit_futures_testnet": lambda: BybitFutures(api_key, api_secret, testnet=True),
        "bybit_futures_live": lambda: BybitFutures(api_key, api_secret, testnet=False),
    }
    factory = registry.get(mode)
    if factory is None:
        raise ValueError(f"Unknown execution mode: '{mode}'. Valid: paper, {', '.join(registry)}")
    return factory()
