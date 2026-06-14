"""
KA-MATS Cryptoz · Execution Agent
Iknir Capital

Two modes:
  paper           → local simulation (no exchange, for backtesting)
  binance_testnet → real Binance testnet orders (fake money, real API)

For testnet keys: https://testnet.binance.vision (free, no real money)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.adaptive_learner import AdaptiveLearner

# ── Position lifecycle constants ──────────────────────────────────────────────
# Crypto positions held indefinitely often sit in dead-money range between
# stop and target. Cap holding time to free capital for better setups.
MAX_HOLD_BARS: int = 20  # default: 20 × 4h = ~3.3 calendar days
MAX_HOLD_BARS_TREND: int = 40  # CryptoTrendPullback: 40 × 4h = ~6.7 days
# Trends need more time to develop — 20 bars
# was expiring winners before target hit
MAX_HOLD_BARS_BDR: int = 30  # BTCDominanceRotation: 30 × 4h = ~5 days
# BTC/ETH in volatile/bear regimes need more
# time to reach 6.5-7× ATR target than default 20

# Simulated perpetual funding rate (applies to LONG positions only).
# 0.01%/8h = 0.03%/day ≈ 11%/year — typical BTC/ETH longs in bull market.
# Charged every 2 bars on 4h timeframe (= 1 funding period = 8h).
FUNDING_RATE_PER_8H: float = 0.0001  # 0.01%
BARS_PER_FUNDING: int = 2  # 4h bars per 8h funding window

# Break-even stop: DISABLED in backtest (hurts TrendPullback R:R).
# Kept for reference. In live mode, consider enabling only for MomentumBreakout.
BREAKEVEN_ACTIVATE_ATR: float = 1.5  # (disabled — not applied in update_prices)
BREAKEVEN_BUFFER_ATR: float = 0.5

# Trailing stop: original validated values (2.0 activation, 1.0 trail)
TRAIL_ACTIVATE_ATR: float = 2.0  # activate trailing stop when up 2×ATR
TRAIL_DISTANCE_ATR: float = 1.0  # trail at 1×ATR below price

from loguru import logger

from config.settings import CONFIG, ExecutionConfig
from core.models import ClosedTrade, PortfolioState, PositionSide, RiskDecision

_STATE_FILE = Path("knowledge/.execution_state.json")


class CryptoPaperExecution:
    """
    Local paper trading — no exchange needed.
    Tracks positions, stops, and targets in memory.
    Used for backtesting and offline paper trading.
    """

    def __init__(
        self,
        initial_capital: float = None,
        cfg: ExecutionConfig = None,
        state_file: str = None,
        learner: AdaptiveLearner = None,
    ) -> None:
        self.cfg = cfg or CONFIG.execution
        self._learner: AdaptiveLearner | None = learner
        cap = initial_capital or self.cfg.initial_capital_usdt
        self._state_file = Path(state_file) if state_file else _STATE_FILE
        self.portfolio = PortfolioState(
            initial_capital=cap,
            cash=cap,
            peak_equity=cap,
            net_equity=cap,
        )
        # position dict: symbol → {shares, entry_price, stop_price, target_price,
        #                           strategy_name, direction, entry_time, value}
        self._positions: dict[str, dict] = {}
        # Equity curve for Sharpe / max-drawdown calculation
        # List of (timestamp, net_equity) sampled every update_prices() call
        self.equity_history: list = []

    @staticmethod
    def _json_default(obj):
        """Handle datetime and other non-serializable types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def save_state(self) -> None:
        """Persist execution state so open positions can be recovered after restart."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "saved_at": datetime.now(UTC).isoformat(),
                "portfolio": {
                    "initial_capital": self.portfolio.initial_capital,
                    "cash": self.portfolio.cash,
                    "peak_equity": self.portfolio.peak_equity,
                    "net_equity": self.portfolio.net_equity,
                },
                "positions": self._positions,
                "closed_trades": [t.model_dump(mode="json") for t in self.portfolio.closed_trades],
                "equity_history": [
                    (ts.isoformat() if isinstance(ts, datetime) else str(ts), eq)
                    for ts, eq in self.equity_history[-5000:]
                ],
            }
            self._state_file.write_text(
                json.dumps(state, indent=2, default=self._json_default),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[Execution] Failed to save state: {e}")

    def load_state(self) -> bool:
        """Restore execution state from disk. Returns True when state was loaded."""
        if not self._state_file.exists():
            return False
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))

            p = raw.get("portfolio", {})
            self.portfolio.initial_capital = float(p.get("initial_capital", self.portfolio.initial_capital))
            self.portfolio.cash = float(p.get("cash", self.portfolio.cash))
            self.portfolio.peak_equity = float(p.get("peak_equity", self.portfolio.peak_equity))
            self.portfolio.net_equity = float(p.get("net_equity", self.portfolio.net_equity))

            self._positions = raw.get("positions", {}) or {}
            self.portfolio.positions = self._positions

            self.portfolio.closed_trades = [
                ClosedTrade.model_validate(t) for t in (raw.get("closed_trades", []) or [])
            ]

            hist = []
            for ts, eq in raw.get("equity_history", []) or []:
                try:
                    hist.append((datetime.fromisoformat(ts), float(eq)))
                except Exception:
                    continue
            self.equity_history = hist

            logger.info(
                f"[Execution] State restored | open_positions={len(self._positions)} | "
                f"closed_trades={len(self.portfolio.closed_trades)}"
            )
            return True
        except Exception as e:
            logger.warning(f"[Execution] Failed to load state: {e}")
            return False

    def execute(
        self,
        decision: RiskDecision,
        current_price: float = None,
        regime: str = "",
        atr_value: float = 0.0,
    ) -> bool:
        """Open a position if decision is approved. Returns True on success.

        Args:
            decision:      Approved RiskDecision from CryptoRiskManager.
            current_price: Override entry price (backtest uses bar close).
            regime:        Regime string at entry — stored in ClosedTrade so
                           adaptive learner partitions by regime family (v17a).
            atr_value:     ATR at entry bar — required for trailing stops and
                           break-even activation.  Pass snapshot.indicators.atr_14.
        """
        if not decision.approved:
            return False

        symbol = decision.symbol
        if symbol in self._positions:
            return False

        price = current_price or decision.entry_price
        # Determine direction from stop-loss relationship to price
        if decision.signal_id:
            is_long = decision.stop_loss < price
        else:
            is_long = True
        fill = price * (1 + self.cfg.slippage_pct) if is_long else price * (1 - self.cfg.slippage_pct)

        shares = decision.position_size
        cost = fill * shares
        fee = cost * self.cfg.taker_fee_pct  # open (taker) fee

        if cost + fee > self.portfolio.cash:
            logger.warning(f"[Execution] {symbol}: insufficient cash")
            return False

        self.portfolio.cash -= cost + fee
        self._positions[symbol] = {
            "shares": shares,
            "entry_price": fill,
            "stop_price": decision.stop_loss,
            "target_price": decision.take_profit,
            "strategy_name": decision.strategy_name,
            "direction": "BUY" if is_long else "SELL",
            "entry_time": datetime.now(UTC),
            "value": cost,
            "entry_fee": fee,  # BUG-3: stored for accurate PnL (open fee was missing)
            "bars_held": 0,
            "funding_paid": 0.0,
            "regime": regime,  # BUG-2: stored so ClosedTrade.regime is set correctly
            "atr_at_entry": atr_value,  # enables break-even and trailing stop activation
        }
        # Mirror into portfolio.positions for risk manager checks
        self.portfolio.positions[symbol] = self._positions[symbol]

        logger.info(
            f"[Execution] OPEN {symbol} {'BUY' if is_long else 'SELL'} | "
            f"fill={fill:.4f} | shares={shares:.6f} | cost=${cost:.2f} | fee=${fee:.2f}"
        )
        self.save_state()
        return True

    def update_prices(self, prices: dict[str, float], timestamp: datetime) -> None:
        """
        Check stop-loss, take-profit, max hold period, and funding costs
        for all open positions.
        """
        to_close = []
        for sym, pos in self._positions.items():
            price = prices.get(sym)
            if price is None:
                continue

            # ── Increment bar counter ───────────────────────────────
            pos["bars_held"] = pos.get("bars_held", 0) + 1

            # ── Funding rate (longs only, every BARS_PER_FUNDING bars) ──
            if pos["direction"] == "BUY" and pos["bars_held"] % BARS_PER_FUNDING == 0:
                funding_cost = pos["value"] * FUNDING_RATE_PER_8H
                pos["funding_paid"] = pos.get("funding_paid", 0.0) + funding_cost
                self.portfolio.cash = max(0.0, self.portfolio.cash - funding_cost)

            # ── Max hold period (per-strategy) ─────────────────────
            sname = pos.get("strategy_name", "")
            if sname == "CryptoTrendPullback":
                max_hold = MAX_HOLD_BARS_TREND  # 40 bars
            elif sname == "BTCDominanceRotation":
                max_hold = MAX_HOLD_BARS_BDR  # 30 bars
            else:
                max_hold = MAX_HOLD_BARS  # 20 bars

            # ── Break-even stop: protect profitable positions ──────
            # When unrealized profit ≥ 1.2 ATR, move stop to entry ± 0.2 ATR.
            # Converts reversing winners to breakeven instead of full losses.
            atr_entry = pos.get("atr_at_entry", 0.0)
            if atr_entry > 0:
                if pos["direction"] == "BUY":
                    profit = price - pos["entry_price"]
                    if profit >= BREAKEVEN_ACTIVATE_ATR * atr_entry:
                        be_stop = pos["entry_price"] + BREAKEVEN_BUFFER_ATR * atr_entry
                        if be_stop > pos["stop_price"]:
                            pos["stop_price"] = be_stop
                else:  # SHORT
                    profit = pos["entry_price"] - price
                    if profit >= BREAKEVEN_ACTIVATE_ATR * atr_entry:
                        be_stop = pos["entry_price"] - BREAKEVEN_BUFFER_ATR * atr_entry
                        if be_stop < pos["stop_price"]:
                            pos["stop_price"] = be_stop

            # ── Trailing stop (both LONG and SHORT) ───────────────
            # Adaptive learner: widen/tighten trail per-symbol stop-hit rate
            if atr_entry > 0:
                atr_adj = 0.0
                if self._learner:
                    atr_adj = self._learner.atr_multiplier_adj(sym)
                eff_trail_activate = TRAIL_ACTIVATE_ATR + atr_adj
                eff_trail_distance = TRAIL_DISTANCE_ATR + atr_adj * 0.5
                if pos["direction"] == "BUY":
                    profit = price - pos["entry_price"]
                    if profit >= eff_trail_activate * atr_entry:
                        trail_level = price - eff_trail_distance * atr_entry
                        if trail_level > pos["stop_price"]:
                            pos["stop_price"] = trail_level
                else:  # SHORT
                    profit = pos["entry_price"] - price
                    if profit >= eff_trail_activate * atr_entry:
                        trail_level = price + eff_trail_distance * atr_entry
                        if trail_level < pos["stop_price"]:
                            pos["stop_price"] = trail_level

            if pos["bars_held"] >= max_hold:
                to_close.append((sym, price, "max_hold_expired"))
            elif pos["direction"] == "BUY":
                if price <= pos["stop_price"]:
                    to_close.append((sym, price, "stop_loss"))
                elif price >= pos["target_price"]:
                    to_close.append((sym, price, "take_profit"))
            else:
                if price >= pos["stop_price"]:
                    to_close.append((sym, price, "stop_loss"))
                elif price <= pos["target_price"]:
                    to_close.append((sym, price, "take_profit"))

        for sym, price, reason in to_close:
            self._close_position(sym, price, timestamp, reason)

        # Update peak equity and record for Sharpe/drawdown computation
        eq = self._net_equity(prices)
        self.portfolio.net_equity = eq
        if eq > self.portfolio.peak_equity:
            self.portfolio.peak_equity = eq
        self.equity_history.append((timestamp, eq))
        self.save_state()

    def _close_position(self, sym: str, price: float, timestamp: datetime, reason: str):
        pos = self._positions.pop(sym, None)
        if sym in self.portfolio.positions:
            del self.portfolio.positions[sym]
        if pos is None:
            return

        is_long = pos["direction"] == "BUY"
        fill = price * (1 - self.cfg.slippage_pct) if is_long else price * (1 + self.cfg.slippage_pct)
        close_fee = abs(fill * pos["shares"]) * self.cfg.taker_fee_pct
        open_fee = pos.get("entry_fee", 0.0)  # BUG-3 fix: open fee now included in PnL
        funding = pos.get("funding_paid", 0.0)  # already deducted from cash per-bar

        if is_long:
            # Full round-trip PnL: gross - open_fee - close_fee - funding
            pnl = (fill - pos["entry_price"]) * pos["shares"] - open_fee - close_fee - funding
        else:
            pnl = (pos["entry_price"] - fill) * pos["shares"] - open_fee - close_fee

        if is_long:
            self.portfolio.cash += fill * pos["shares"]
        else:
            # Release the reserved short notional plus net PnL. Without this,
            # paper cash moves opposite to the recorded short trade PnL.
            entry_notional = pos["entry_price"] * pos["shares"]
            gross_pnl = (pos["entry_price"] - fill) * pos["shares"]
            self.portfolio.cash += entry_notional + gross_pnl - close_fee

        trade = ClosedTrade(
            symbol=sym,
            strategy_name=pos["strategy_name"],
            regime=pos.get("regime", ""),  # BUG-2 fix: regime now correctly set
            side=PositionSide.LONG if is_long else PositionSide.SHORT,
            entry_price=pos["entry_price"],
            exit_price=fill,
            size=pos["shares"],
            pnl=round(pnl, 6),
            entry_time=pos["entry_time"],
            exit_time=timestamp,
            exit_reason=reason,
        )
        self.portfolio.closed_trades.append(trade)

        logger.info(
            f"[Execution] CLOSE {sym} | {reason} | "
            f"pnl={pnl:+.4f} USDT | "
            f"entry={pos['entry_price']:.4f} exit={fill:.4f}"
        )
        self.save_state()

    def _net_equity(self, prices: dict[str, float]) -> float:
        pos_val = 0.0
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos["entry_price"])
            if pos["direction"] == "BUY":
                pos_val += price * pos["shares"]
            else:
                pos_val += (2 * pos["entry_price"] - price) * pos["shares"]
        return self.portfolio.cash + pos_val

    def close_all(self, prices: dict[str, float], timestamp: datetime) -> None:
        for sym in list(self._positions.keys()):
            price = prices.get(sym, self._positions[sym]["entry_price"])
            self._close_position(sym, price, timestamp, "end_of_backtest")


class BinanceTestnetExecution:
    """
    Real Binance testnet paper trading.
    Requires API keys from https://testnet.binance.vision
    Set env vars: BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET
    """

    def __init__(self, cfg: ExecutionConfig = None) -> None:
        self.cfg = cfg or CONFIG.execution
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ccxt

                self._client = ccxt.binance(
                    {
                        "apiKey": self.cfg.api_key,
                        "secret": self.cfg.api_secret,
                        "enableRateLimit": True,
                        "options": {"defaultType": "spot", "sandboxMode": True},
                        "urls": {
                            "api": {
                                "public": "https://testnet.binance.vision/api",
                                "private": "https://testnet.binance.vision/api",
                            }
                        },
                    }
                )
                logger.info("[Execution] Connected to Binance TESTNET")
            except ImportError:
                raise ImportError("Run: pip install ccxt")
        return self._client

    def buy(self, symbol: str, usdt_amount: float) -> dict:
        client = self._get_client()
        try:
            ticker = client.fetch_ticker(symbol)
            qty = usdt_amount / ticker["last"]
            order = client.create_market_buy_order(symbol, qty)
            logger.info(f"[Testnet] BUY {symbol} | ${usdt_amount:.2f}")
            return order
        except Exception as e:
            logger.error(f"[Testnet] Buy {symbol} failed: {e}")
            return {}

    def sell(self, symbol: str, qty: float) -> dict:
        client = self._get_client()
        try:
            order = client.create_market_sell_order(symbol, qty)
            logger.info(f"[Testnet] SELL {symbol} | qty={qty:.6f}")
            return order
        except Exception as e:
            logger.error(f"[Testnet] Sell {symbol} failed: {e}")
            return {}

    def balance(self) -> dict:
        try:
            b = self._get_client().fetch_balance()
            logger.info(f"[Testnet] USDT free: {b.get('USDT', {}).get('free', 0):.2f}")
            return b
        except Exception as e:
            logger.error(f"[Testnet] Balance fetch failed: {e}")
            return {}
