"""
KA-MATS Cryptoz · Live Execution Agent
Iknir Capital

Hybrid execution: local position tracking + real exchange orders.

Architecture:
  - Extends CryptoPaperExecution (inherits ALL bookkeeping: stops, targets,
    trailing stops, max hold, adaptive ATR, funding, break-even)
  - Overrides execute() to also place a real market BUY on the exchange
  - Overrides _close_position() to also place a real market SELL
  - Exchange failures are non-fatal: position is tracked locally regardless

Why hybrid instead of pure exchange?
  1. Local bookkeeping drives trailing stops, max-hold, and the adaptive learner
  2. State persistence survives exchange outages
  3. Exchange-side protective orders (stop/TP) give intrabar exit fidelity —
     Phase 1 measured close-only exits as the single largest backtest/live gap
     (intrabar fills improved Sharpe 0.935 → 1.084 in the honest baseline)

Order flow:
  OPEN:   exchange market_buy → local bookkeeping at real fill →
          exchange-side protective stop/TP placed (spot OCO / futures pair)
  EXIT (exchange side):  stop or TP triggers intrabar on the exchange →
          reconciler detects the fill next poll → local close at real fill price
  EXIT (locally decided, e.g. max_hold):  cancel protective orders →
          exchange market exit → local bookkeeping at real fill
  TRAIL:  when the local trailing stop ratchets, protective orders are
          cancelled and re-placed at the new stop level
  If exchange fails: position is still tracked locally, alert is raised,
  and the next bar retry mechanism handles it.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from agents.execution_agent import CryptoPaperExecution
from config.settings import ExecutionConfig
from core.models import RiskDecision

if TYPE_CHECKING:
    from core.adaptive_learner import AdaptiveLearner
    from exchanges import ExchangeConnector


class LiveExecution(CryptoPaperExecution):
    """
    Real exchange execution with full local position lifecycle management.

    Inherits from CryptoPaperExecution:
      - Position tracking (_positions dict)
      - Stop-loss / take-profit checking
      - Trailing stop logic (adaptive ATR)
      - Break-even stop
      - Max hold period (per-strategy)
      - Funding rate simulation
      - Equity curve tracking
      - State persistence (save/load JSON)
      - Closed trade recording

    Adds:
      - Real market orders on entry/exit via ExchangeConnector
      - Exchange balance sync
      - Order failure handling with retry queues
    """

    def __init__(
        self,
        exchange: ExchangeConnector,
        initial_capital: float = None,
        cfg: ExecutionConfig = None,
        state_file: str = None,
        learner: AdaptiveLearner = None,
    ) -> None:
        super().__init__(
            initial_capital=initial_capital,
            cfg=cfg,
            state_file=state_file,
            learner=learner,
        )
        self._exchange = exchange
        # Track pending exits that failed — retry on next bar
        # symbol → {"qty": float, "is_long": bool}
        self._pending_sells: dict[str, dict] = {}
        # Symbols whose leverage has been pinned to 1x (futures only, once per symbol)
        self._leverage_set: set = set()
        # Max acceptable deviation between decision price and live ticker before
        # an entry is abandoned (protects against gaps between bar close and fill).
        self.max_entry_slippage_pct: float = 0.005  # 0.5%
        # Exchange-side protective stop/TP orders (intrabar exit fidelity).
        # Auto-enabled when the connector supports them; positions without
        # protective orders fall back to the legacy poll-based exit checks.
        self.exchange_protective_exits: bool = getattr(exchange, "supports_protective_exits", False)
        # Replace protective stop only when the trailing stop improved by more
        # than this fraction — avoids cancel/replace churn on tiny ratchets.
        self.min_stop_replace_pct: float = 0.001  # 0.1%

        # ── v15 maker entries (validates the 65%-fill backtest assumption) ────
        # Entry flow: limit buy at (last * (1 - offset)) → poll until timeout →
        # cancel + market-buy remainder on non-fill. Every entry's fill type is
        # appended to logs/fill_stats.jsonl so the paper phase MEASURES the
        # maker fill rate instead of assuming it.
        self.maker_entries: bool = True
        self.maker_offset_pct: float = 0.0005  # rest 0.05% below last
        self.maker_timeout_seconds: float = 90.0  # then fall back to market
        self.maker_poll_seconds: float = 5.0
        self._fill_log_path = Path("logs") / "fill_stats.jsonl"

        logger.info(
            "[LiveExecution] Initialized — real exchange orders enabled | "
            f"exchange-side stop/TP: {'ON' if self.exchange_protective_exits else 'OFF'} | "
            f"maker entries: {'ON' if self.maker_entries else 'OFF'}"
        )

    # ── v15 maker entry: limit → timeout → market fallback ───────────────────

    def _log_fill(
        self, symbol: str, fill_type: str, limit_price: float, fill_price: float, resting_seconds: float
    ) -> None:
        """Append one entry-fill record; the paper phase aggregates these to
        validate (or refute) the backtest's 65% maker-fill assumption."""
        rec = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "symbol": symbol,
            "fill_type": fill_type,  # maker | taker_fallback | mixed | market
            "limit_price": round(limit_price, 8),
            "fill_price": round(fill_price, 8),
            "resting_seconds": round(resting_seconds, 1),
        }
        try:
            self._fill_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._fill_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            logger.debug(f"[LiveExecution] fill log write failed: {e}")

    def _maker_entry(self, symbol: str, qty: float, ref_price: float) -> dict | None:
        """Long entry via resting limit order with market fallback.

        Returns the ccxt order dict of whichever order(s) achieved the fill.
        Falls back to a plain market buy on any error path (never silently
        skips an approved entry because of fill mechanics).
        """
        limit_price = ref_price * (1.0 - self.maker_offset_pct)
        t0 = time.monotonic()
        try:
            order = self._exchange.limit_buy(symbol, qty, limit_price)
            order_id = str(order.get("id", ""))
        except Exception as e:
            logger.warning(f"[LiveExecution] {symbol}: limit entry failed ({e}) — using market order")
            order = self._exchange.market_buy(symbol, qty)
            self._log_fill(symbol, "market", limit_price, float(order.get("average") or ref_price), 0.0)
            return order

        # Instant fill: some venues match marketable limits in the create response
        if (order.get("status") or "").lower() == "closed":
            fill_price = float(order.get("average") or limit_price)
            logger.info(f"[LiveExecution] {symbol}: MAKER fill (instant) @ {fill_price:.4f}")
            self._log_fill(symbol, "maker", limit_price, fill_price, 0.0)
            return order

        # Poll until filled or timeout
        while time.monotonic() - t0 < self.maker_timeout_seconds:
            time.sleep(self.maker_poll_seconds)
            status = self._exchange.fetch_order_safe(order_id, symbol)
            if status is None:
                continue
            state = (status.get("status") or "").lower()
            if state == "closed":
                elapsed = time.monotonic() - t0
                fill_price = float(status.get("average") or limit_price)
                logger.info(f"[LiveExecution] {symbol}: MAKER fill @ {fill_price:.4f} after {elapsed:.0f}s")
                self._log_fill(symbol, "maker", limit_price, fill_price, elapsed)
                return status

        # Timeout: cancel and market-buy the unfilled remainder
        elapsed = time.monotonic() - t0
        try:
            self._exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.debug(f"[LiveExecution] {symbol}: limit cancel: {e}")

        status = self._exchange.fetch_order_safe(order_id, symbol) or {}
        already = float(status.get("filled") or 0.0)
        remainder = max(qty - already, 0.0)

        if remainder <= qty * 0.001:  # effectively fully filled at cancel
            fill_price = float(status.get("average") or limit_price)
            self._log_fill(symbol, "maker", limit_price, fill_price, elapsed)
            return status

        mkt = self._exchange.market_buy(symbol, remainder)
        mkt_price = float(mkt.get("average") or mkt.get("price") or ref_price)
        if already > 0:  # partial maker + taker remainder
            maker_price = float(status.get("average") or limit_price)
            blended = (maker_price * already + mkt_price * remainder) / qty
            self._log_fill(symbol, "mixed", limit_price, blended, elapsed)
            return {"id": mkt.get("id"), "average": blended, "filled": qty}
        logger.info(
            f"[LiveExecution] {symbol}: maker timeout after {elapsed:.0f}s — TAKER fallback @ {mkt_price:.4f}"
        )
        self._log_fill(symbol, "taker_fallback", limit_price, mkt_price, elapsed)
        return mkt

    def execute(
        self,
        decision: RiskDecision,
        current_price: float = None,
        regime: str = "",
        atr_value: float = 0.0,
    ) -> bool:
        """Open position: local bookkeeping + real exchange buy order."""
        if not decision.approved:
            return False

        symbol = decision.symbol
        if symbol in self._positions:
            return False

        price = current_price or decision.entry_price
        is_long = decision.stop_loss < price if decision.signal_id else True
        shares = decision.position_size
        cost = price * shares

        # Pre-flight: check exchange balance
        try:
            exchange_balance = self._exchange.get_balance("USDT")
            if exchange_balance < cost * 0.95:  # 5% margin for fees
                logger.error(
                    f"[LiveExecution] {symbol}: insufficient exchange balance "
                    f"(need ${cost:.2f}, have ${exchange_balance:.2f})"
                )
                return False
        except Exception as e:
            logger.error(f"[LiveExecution] {symbol}: balance check failed: {e}")
            return False

        # Pre-flight: slippage guard — abandon entry if live price has gapped
        # away from the decision price (market order would fill far from plan).
        try:
            live_price = self._exchange.get_ticker(symbol)
            deviation = abs(live_price - price) / price if price > 0 else 0.0
            if deviation > self.max_entry_slippage_pct:
                logger.warning(
                    f"[LiveExecution] {symbol}: entry abandoned — live price "
                    f"{live_price:.4f} deviates {deviation:.2%} from decision "
                    f"price {price:.4f} (max {self.max_entry_slippage_pct:.2%})"
                )
                return False
            price = live_price  # size/risk math uses the freshest price
        except Exception as e:
            logger.debug(f"[LiveExecution] {symbol}: ticker check skipped ({e})")

        # Place real exchange order
        exchange_fill = None
        try:
            if is_long:
                if self.maker_entries:
                    exchange_fill = self._maker_entry(symbol, shares, price)
                else:
                    exchange_fill = self._exchange.market_buy(symbol, shares)
            elif getattr(self._exchange, "supports_shorts", False):
                # Futures/perpetual connector: pin leverage to 1x once, then
                # market sell opens the short (same notional exposure as spot)
                if symbol not in self._leverage_set:
                    self._exchange.set_leverage(symbol)
                    self._leverage_set.add(symbol)
                exchange_fill = self._exchange.open_short(symbol, shares)
            else:
                logger.warning(
                    f"[LiveExecution] {symbol}: SHORT requires a futures mode "
                    f"(binance_futures_* / bybit_futures_*) — skipped on spot"
                )
                return False

            fill_price = float(exchange_fill.get("average", exchange_fill.get("price", price)))
            fill_qty = float(exchange_fill.get("filled", shares))
            logger.info(
                f"[LiveExecution] EXCHANGE BUY {symbol} | "
                f"price={fill_price:.4f} | qty={fill_qty:.6f} | "
                f"order_id={exchange_fill.get('id', 'n/a')}"
            )
            # Use actual fill price for bookkeeping
            decision_copy = RiskDecision(
                symbol=decision.symbol,
                approved=True,
                signal_id=decision.signal_id,
                timestamp=decision.timestamp,
                strategy_name=decision.strategy_name,
                entry_price=fill_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                position_size=fill_qty,
                risk_amount=decision.risk_amount,
                veto_reason=None,
            )
            opened = super().execute(
                decision_copy, current_price=fill_price, regime=regime, atr_value=atr_value
            )
            if opened:
                self._place_protective(symbol)
            return opened

        except Exception as e:
            logger.error(f"[LiveExecution] EXCHANGE BUY FAILED {symbol}: {e} — position NOT opened")
            return False

    # ── Exchange-side protective stop/TP orders ──────────────────────────────

    def _place_protective(self, symbol: str) -> None:
        """Place resting stop-loss + take-profit orders for a just-opened position.

        Failure is non-fatal: the position falls back to legacy poll-based
        exit checks (the pre-Phase-1 behaviour), and a retry happens on the
        next bar via _sync_protective_orders().
        """
        if not self.exchange_protective_exits:
            return
        pos = self._positions.get(symbol)
        if pos is None:
            return
        is_long = pos.get("direction", "BUY") == "BUY"
        try:
            protective = self._exchange.place_protective_exit(
                symbol,
                qty=pos["shares"],
                stop_price=pos["stop_price"],
                take_profit=pos["target_price"],
                is_long=is_long,
            )
            if protective:
                pos["protective"] = protective
                self.save_state()
        except Exception as e:
            logger.error(
                f"[LiveExecution] {symbol}: protective stop/TP placement failed: {e} "
                f"— falling back to poll-based exits, will retry next bar"
            )

    def _reconcile_protective_fills(self, timestamp: datetime) -> None:
        """Detect protective orders that filled on the exchange since last poll
        and close the corresponding local positions at the real fill price."""
        if not self.exchange_protective_exits:
            return
        for sym in list(self._positions.keys()):
            pos = self._positions[sym]
            protective = pos.get("protective")
            if not protective:
                continue

            filled_leg = None  # ("stop_loss" | "take_profit", fill_price)
            for leg, reason in (("stop_id", "stop_loss"), ("tp_id", "take_profit")):
                oid = protective.get(leg)
                if not oid:
                    continue
                order = self._exchange.fetch_order_safe(oid, sym)
                if order is None:
                    continue
                status = (order.get("status") or "").lower()
                filled = float(order.get("filled") or 0.0)
                if status == "closed" and filled > 0:
                    fill_price = float(
                        order.get("average")
                        or order.get("price")
                        or protective.get("stop_price" if reason == "stop_loss" else "tp_price")
                    )
                    filled_leg = (reason, fill_price)
                    break

            if filled_leg is None:
                continue

            reason, fill_price = filled_leg
            logger.info(
                f"[LiveExecution] {sym}: exchange-side {reason} filled intrabar "
                f"@ {fill_price:.4f} — reconciling local state"
            )
            # Cancel surviving sibling (futures pair only; OCO auto-cancels,
            # in which case the cancel is a harmless no-op/debug log).
            if protective.get("kind") == "pair":
                try:
                    self._exchange.cancel_protective_exit(sym, protective)
                except Exception as e:
                    logger.warning(f"[LiveExecution] {sym}: sibling cancel failed: {e}")
            # Position is already flat on the exchange — close locally only.
            pos.pop("protective", None)
            CryptoPaperExecution._close_position(self, sym, fill_price, timestamp, reason)

    def _sync_protective_orders(self) -> None:
        """Keep exchange-side stops in line with local trailing/break-even stops,
        and retry placement for positions that have no protective orders yet."""
        if not self.exchange_protective_exits:
            return
        for sym in list(self._positions.keys()):
            pos = self._positions[sym]
            protective = pos.get("protective")

            if not protective:
                self._place_protective(sym)  # placement failed at entry — retry
                continue

            local_stop = float(pos["stop_price"])
            exchange_stop = float(protective.get("stop_price", 0.0))
            if exchange_stop <= 0:
                continue
            is_long = pos.get("direction", "BUY") == "BUY"
            improved = (
                local_stop > exchange_stop * (1 + self.min_stop_replace_pct)
                if is_long
                else local_stop < exchange_stop * (1 - self.min_stop_replace_pct)
            )
            if not improved:
                continue

            logger.info(
                f"[LiveExecution] {sym}: trailing stop moved "
                f"{exchange_stop:.4f} → {local_stop:.4f} — replacing protective orders"
            )
            try:
                self._exchange.cancel_protective_exit(sym, protective)
            except Exception as e:
                logger.warning(f"[LiveExecution] {sym}: protective cancel failed: {e}")
                continue  # don't double-place; retry next bar
            pos.pop("protective", None)
            self._place_protective(sym)

    def _close_position(self, sym: str, price: float, timestamp: datetime, reason: str):
        """Close position: real exchange exit order + local bookkeeping.

        LONG  → market sell
        SHORT → reduce-only market buy (futures connector)
        """
        pos = self._positions.get(sym)
        if pos is None:
            return

        shares = pos["shares"]
        is_long = pos.get("direction", "BUY") == "BUY"

        # Cancel resting protective orders first — on spot the OCO locks the
        # coins, so the market sell below would be rejected otherwise.
        protective = pos.pop("protective", None)
        if protective:
            try:
                self._exchange.cancel_protective_exit(sym, protective)
            except Exception as e:
                logger.warning(f"[LiveExecution] {sym}: protective cancel failed before exit: {e}")

        try:
            if is_long:
                order = self._exchange.market_sell(sym, shares)
            else:
                order = self._exchange.close_short(sym, shares)
            fill_price = float(order.get("average", order.get("price", price)))
            logger.info(
                f"[LiveExecution] EXCHANGE {'SELL' if is_long else 'COVER'} {sym} | "
                f"reason={reason} | price={fill_price:.4f} | "
                f"order_id={order.get('id', 'n/a')}"
            )
            # Use actual exchange fill price for PnL
            super()._close_position(sym, fill_price, timestamp, reason)
            # Clear any pending retry
            self._pending_sells.pop(sym, None)

        except Exception as e:
            logger.error(
                f"[LiveExecution] EXCHANGE EXIT FAILED {sym}: {e} — "
                f"queuing for retry, using local price for bookkeeping"
            )
            self._pending_sells[sym] = {"qty": shares, "is_long": is_long}
            # Still close locally so portfolio state doesn't stall
            super()._close_position(sym, price, timestamp, reason)

    def update_prices(self, prices: dict[str, float], timestamp: datetime) -> None:
        """Reconcile exchange-side exits, retry failed orders, then run the
        normal local lifecycle (trailing, max-hold, fallback stop/TP checks)."""
        # 1. Positions whose stop/TP already filled on the exchange intrabar:
        #    close locally at the real fill price before any local exit logic
        #    can act on a stale price.
        self._reconcile_protective_fills(timestamp)

        # 2. Retry pending exits first
        for sym in list(self._pending_sells.keys()):
            pending = self._pending_sells[sym]
            # Back-compat: older saved state stored a bare float qty (long sell)
            if isinstance(pending, dict):
                qty, is_long = pending["qty"], pending.get("is_long", True)
            else:
                qty, is_long = float(pending), True
            try:
                if is_long:
                    order = self._exchange.market_sell(sym, qty)
                else:
                    order = self._exchange.close_short(sym, qty)
                logger.info(
                    f"[LiveExecution] RETRY {'SELL' if is_long else 'COVER'} {sym} "
                    f"succeeded | order_id={order.get('id', 'n/a')}"
                )
                del self._pending_sells[sym]
            except Exception as e:
                logger.warning(f"[LiveExecution] RETRY EXIT {sym} still failing: {e}")

        # 3. Normal price update (trailing ratchet, max-hold, and poll-based
        #    stop/TP fallback for positions without protective orders).
        super().update_prices(prices, timestamp)

        # 4. Trailing/break-even may have moved local stops — bring the
        #    exchange-side orders in line, and retry any missing placements.
        self._sync_protective_orders()

    def sync_exchange_balance(self) -> float | None:
        """Fetch real exchange balance for monitoring/alerting."""
        try:
            bal = self._exchange.get_balance("USDT")
            logger.info(f"[LiveExecution] Exchange USDT balance: ${bal:,.2f}")
            return bal
        except Exception as e:
            logger.warning(f"[LiveExecution] Balance sync failed: {e}")
            return None
