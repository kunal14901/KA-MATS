"""Unit tests for LiveExecution — 17% → ~85%."""

from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agents.live_execution import LiveExecution
from core.models import RiskDecision


def _decision(
    symbol="BTC/USDT",
    entry=44500.0,
    stop=43500.0,
    target=47500.0,
    size=0.10,
):
    return RiskDecision(
        signal_id=uuid4(),
        symbol=symbol,
        timestamp=datetime.now(UTC),
        approved=True,
        position_size=size,
        position_value=size * entry,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        risk_amount=(entry - stop) * size,
        veto_reason="",
        strategy_name="TestStrategy",
    )


def _mock_exchange(balance=50000.0, fill_price=44500.0):
    """Spot-style connector mock: no short support, no protective exits.

    limit_buy reports an instant fill (status=closed) so the v15 maker entry
    path resolves without polling — taker-fallback behaviour is tested
    separately with an explicitly "open" limit order.
    """
    ex = MagicMock()
    ex.supports_shorts = False
    ex.supports_protective_exits = False
    ex.get_balance.return_value = balance
    ex.get_ticker.return_value = fill_price
    ex.market_buy.return_value = {"average": str(fill_price), "filled": "0.10", "id": "ord_001"}
    ex.market_sell.return_value = {"average": str(fill_price), "filled": "0.10", "id": "ord_002"}
    ex.limit_buy.return_value = {
        "average": str(fill_price),
        "filled": "0.10",
        "id": "lim_001",
        "status": "closed",
    }
    return ex


def _mock_protective_exchange(balance=50000.0, fill_price=44500.0, stop=43500.0, target=47500.0):
    """Spot-style connector mock with exchange-side stop/TP (OCO) support."""
    ex = _mock_exchange(balance=balance, fill_price=fill_price)
    ex.supports_protective_exits = True
    ex.place_protective_exit.return_value = {
        "kind": "oco",
        "stop_id": "sl_1",
        "tp_id": "tp_1",
        "stop_price": stop,
        "tp_price": target,
        "list_id": "lst_1",
    }
    ex.fetch_order_safe.return_value = {"status": "open", "filled": 0.0}
    ex.cancel_protective_exit.return_value = None
    return ex


def _mock_futures_exchange(balance=50000.0, fill_price=44500.0):
    """Futures-style connector mock: shorts enabled."""
    ex = _mock_exchange(balance=balance, fill_price=fill_price)
    ex.supports_shorts = True
    ex.supports_protective_exits = False
    ex.open_short.return_value = {"average": str(fill_price), "filled": "0.10", "id": "ord_s01"}
    ex.close_short.return_value = {"average": str(fill_price), "filled": "0.10", "id": "ord_s02"}
    ex.set_leverage.return_value = None
    return ex


@pytest.mark.unit
class TestLiveExecution:
    def test_init(self):
        ex = LiveExecution(exchange=_mock_exchange())
        assert ex._exchange is not None
        assert len(ex._pending_sells) == 0

    def test_execute_buy_success(self):
        exchange = _mock_exchange(balance=50000.0, fill_price=44500.0)
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision()
        result = ex.execute(decision, current_price=44500.0, regime="trending_up")
        assert result is True
        # v15 maker entry: limit order fills instantly, no market fallback needed
        exchange.limit_buy.assert_called_once()
        exchange.market_buy.assert_not_called()
        assert "BTC/USDT" in ex.portfolio.positions

    def test_execute_rejects_unapproved(self):
        exchange = _mock_exchange()
        ex = LiveExecution(exchange=exchange)
        decision = _decision()
        decision.approved = False
        result = ex.execute(decision, current_price=44500.0)
        assert result is False
        exchange.market_buy.assert_not_called()

    def test_execute_rejects_duplicate_position(self):
        exchange = _mock_exchange()
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        d1 = _decision()
        ex.execute(d1, current_price=44500.0, regime="trending_up")
        d2 = _decision()  # same symbol
        result = ex.execute(d2, current_price=44500.0)
        assert result is False

    def test_execute_insufficient_balance(self):
        exchange = _mock_exchange(balance=100.0)  # not enough
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision()
        result = ex.execute(decision, current_price=44500.0)
        assert result is False
        exchange.market_buy.assert_not_called()

    def test_execute_balance_check_fails(self):
        exchange = _mock_exchange()
        exchange.get_balance.side_effect = Exception("API error")
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision()
        result = ex.execute(decision, current_price=44500.0)
        assert result is False

    def test_execute_exchange_buy_fails(self):
        exchange = _mock_exchange(balance=50000.0)
        # Both the maker limit attempt and the market fallback are rejected
        exchange.limit_buy.side_effect = Exception("Order rejected")
        exchange.market_buy.side_effect = Exception("Order rejected")
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision()
        result = ex.execute(decision, current_price=44500.0)
        assert result is False
        assert "BTC/USDT" not in ex.portfolio.positions

    def test_maker_timeout_falls_back_to_market(self):
        exchange = _mock_exchange(balance=50000.0, fill_price=44500.0)
        # Limit order rests unfilled; never reports closed
        exchange.limit_buy.return_value = {"id": "lim_002", "status": "open"}
        exchange.fetch_order_safe.return_value = {"status": "open", "filled": 0.0}
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        ex.maker_timeout_seconds = 0.0  # immediate timeout for the test
        ex.maker_poll_seconds = 0.0
        decision = _decision()
        result = ex.execute(decision, current_price=44500.0, regime="trending_up")
        assert result is True
        exchange.cancel_order.assert_called_once()
        exchange.market_buy.assert_called_once()  # taker fallback
        assert "BTC/USDT" in ex.portfolio.positions

    def test_execute_short_not_supported_on_spot(self):
        exchange = _mock_exchange(balance=50000.0)
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        # Create a decision where stop > price → short signal
        decision = _decision(entry=44500.0, stop=45500.0, target=42000.0)
        result = ex.execute(decision, current_price=44500.0)
        assert result is False
        exchange.market_sell.assert_not_called()

    def test_execute_short_on_futures(self):
        exchange = _mock_futures_exchange(balance=50000.0)
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision(entry=44500.0, stop=45500.0, target=42000.0)
        result = ex.execute(decision, current_price=44500.0, regime="trending_down")
        assert result is True
        exchange.open_short.assert_called_once()
        exchange.set_leverage.assert_called_once()  # pinned to 1x before first short
        assert "BTC/USDT" in ex.portfolio.positions
        assert ex.portfolio.positions["BTC/USDT"]["direction"] == "SELL"

    def test_short_close_uses_reduce_only_cover(self):
        exchange = _mock_futures_exchange(balance=50000.0, fill_price=44500.0)
        # Entry (open_short) fills at 44500; cover fills lower at 41900 → profit
        exchange.close_short.return_value = {"average": "41900", "filled": "0.10", "id": "ord_s02"}
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision(entry=44500.0, stop=45500.0, target=42000.0)
        ex.execute(decision, current_price=44500.0, regime="trending_down")
        # Price hits the short target → close via reduce-only buy
        ex.update_prices({"BTC/USDT": 41900.0}, datetime.now(UTC))
        exchange.close_short.assert_called()
        exchange.market_sell.assert_not_called()
        assert "BTC/USDT" not in ex.portfolio.positions
        assert ex.portfolio.closed_trades[-1].pnl > 0  # short won as price fell

    def test_slippage_guard_abandons_gapped_entry(self):
        exchange = _mock_exchange(balance=50000.0)
        exchange.get_ticker.return_value = 46000.0  # gapped >0.5% above decision price
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision(entry=44500.0)
        result = ex.execute(decision, current_price=44500.0)
        assert result is False
        exchange.market_buy.assert_not_called()

    def test_close_position_success(self):
        exchange = _mock_exchange(balance=50000.0, fill_price=47000.0)
        exchange.get_ticker.return_value = 44500.0  # live price matches entry (no gap)
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision()
        ex.execute(decision, current_price=44500.0, regime="trending_up")

        ts = datetime.now(UTC)
        ex._close_position("BTC/USDT", 47000.0, ts, "take_profit")

        exchange.market_sell.assert_called_once()
        assert "BTC/USDT" not in ex._pending_sells

    def test_close_position_exchange_fails_queues_retry(self):
        exchange = _mock_exchange(balance=50000.0, fill_price=44500.0)
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision()
        ex.execute(decision, current_price=44500.0, regime="trending_up")

        exchange.market_sell.side_effect = Exception("Network error")
        ts = datetime.now(UTC)
        ex._close_position("BTC/USDT", 45000.0, ts, "stop_loss")

        assert "BTC/USDT" in ex._pending_sells

    def test_close_position_nonexistent(self):
        exchange = _mock_exchange()
        ex = LiveExecution(exchange=exchange)
        ts = datetime.now(UTC)
        # Should not raise
        ex._close_position("NONEXISTENT/USDT", 100.0, ts, "test")

    def test_update_prices_retries_pending_sells(self):
        exchange = _mock_exchange(balance=50000.0, fill_price=44500.0)
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)

        # Manually add a pending sell
        ex._pending_sells["ETH/USDT"] = 1.5

        prices = {"ETH/USDT": 3500.0}
        ts = datetime.now(UTC)
        ex.update_prices(prices, ts)

        exchange.market_sell.assert_called_once_with("ETH/USDT", 1.5)
        assert "ETH/USDT" not in ex._pending_sells

    def test_update_prices_retry_still_fails(self):
        exchange = _mock_exchange()
        exchange.market_sell.side_effect = Exception("Still down")
        ex = LiveExecution(exchange=exchange)
        ex._pending_sells["ETH/USDT"] = 1.5

        prices = {"ETH/USDT": 3500.0}
        ts = datetime.now(UTC)
        ex.update_prices(prices, ts)

        assert "ETH/USDT" in ex._pending_sells  # still queued

    def test_sync_exchange_balance(self):
        exchange = _mock_exchange(balance=25000.0)
        ex = LiveExecution(exchange=exchange)
        bal = ex.sync_exchange_balance()
        assert bal == 25000.0

    def test_sync_exchange_balance_fails(self):
        exchange = _mock_exchange()
        exchange.get_balance.side_effect = Exception("Timeout")
        ex = LiveExecution(exchange=exchange)
        bal = ex.sync_exchange_balance()
        assert bal is None


@pytest.mark.unit
class TestProtectiveExits:
    """Exchange-side stop/TP order lifecycle (Phase 1 execution fidelity)."""

    def _open(self, exchange, stop=43500.0, target=47500.0):
        ex = LiveExecution(exchange=exchange, initial_capital=50000.0)
        decision = _decision(stop=stop, target=target)
        assert ex.execute(decision, current_price=44500.0, regime="trending_up")
        return ex

    def test_disabled_when_connector_lacks_support(self):
        exchange = _mock_exchange()
        ex = self._open(exchange)
        assert ex.exchange_protective_exits is False
        exchange.place_protective_exit.assert_not_called()
        assert "protective" not in ex._positions["BTC/USDT"]

    def test_entry_places_protective_orders(self):
        exchange = _mock_protective_exchange()
        ex = self._open(exchange)
        exchange.place_protective_exit.assert_called_once()
        kwargs = exchange.place_protective_exit.call_args.kwargs
        assert kwargs["stop_price"] == 43500.0
        assert kwargs["take_profit"] == 47500.0
        assert kwargs["is_long"] is True
        assert ex._positions["BTC/USDT"]["protective"]["stop_id"] == "sl_1"

    def test_placement_failure_is_nonfatal_and_retried(self):
        exchange = _mock_protective_exchange()
        exchange.place_protective_exit.side_effect = Exception("API down")
        ex = self._open(exchange)
        # Position open despite failure, no protective stored
        assert "BTC/USDT" in ex._positions
        assert "protective" not in ex._positions["BTC/USDT"]

        # Next bar: placement retried and succeeds
        exchange.place_protective_exit.side_effect = None
        ex.update_prices({"BTC/USDT": 45000.0}, datetime.now(UTC))
        assert ex._positions["BTC/USDT"]["protective"]["tp_id"] == "tp_1"

    def test_reconcile_stop_fill_closes_locally(self):
        exchange = _mock_protective_exchange()
        ex = self._open(exchange)

        # Exchange reports the stop leg filled intrabar at 43400
        def order_status(oid, sym):
            if oid == "sl_1":
                return {"status": "closed", "filled": 0.10, "average": 43400.0}
            return {"status": "canceled", "filled": 0.0}

        exchange.fetch_order_safe.side_effect = order_status

        ex.update_prices({"BTC/USDT": 44000.0}, datetime.now(UTC))

        assert "BTC/USDT" not in ex._positions
        trade = ex.portfolio.closed_trades[-1]
        assert trade.exit_reason == "stop_loss"
        # Closed at the real exchange fill, not the polled bar price
        assert trade.exit_price == pytest.approx(43400.0, rel=0.01)
        # No duplicate market sell — position was already flat on exchange
        exchange.market_sell.assert_not_called()

    def test_reconcile_tp_fill_cancels_pair_sibling(self):
        exchange = _mock_protective_exchange()
        exchange.place_protective_exit.return_value = {
            "kind": "pair",
            "stop_id": "sl_1",
            "tp_id": "tp_1",
            "stop_price": 43500.0,
            "tp_price": 47500.0,
            "list_id": None,
        }
        ex = self._open(exchange)

        def order_status(oid, sym):
            if oid == "tp_1":
                return {"status": "closed", "filled": 0.10, "average": 47510.0}
            return {"status": "open", "filled": 0.0}

        exchange.fetch_order_safe.side_effect = order_status

        ex.update_prices({"BTC/USDT": 47000.0}, datetime.now(UTC))

        assert "BTC/USDT" not in ex._positions
        assert ex.portfolio.closed_trades[-1].exit_reason == "take_profit"
        # Futures pair: surviving stop leg must be cancelled manually
        exchange.cancel_protective_exit.assert_called()
        exchange.market_sell.assert_not_called()

    def test_local_exit_cancels_protective_before_market_sell(self):
        exchange = _mock_protective_exchange()
        ex = self._open(exchange)

        ex._close_position("BTC/USDT", 45000.0, datetime.now(UTC), "max_hold_expired")

        # Cancel must happen (spot OCO locks the coins) and then market sell
        exchange.cancel_protective_exit.assert_called_once()
        exchange.market_sell.assert_called_once()
        assert "BTC/USDT" not in ex._positions

    def test_trailing_stop_replaces_protective_orders(self):
        exchange = _mock_protective_exchange()
        ex = self._open(exchange)
        # Simulate the trailing logic having ratcheted the local stop up 2%
        ex._positions["BTC/USDT"]["stop_price"] = 44400.0

        ex.update_prices({"BTC/USDT": 46000.0}, datetime.now(UTC))

        exchange.cancel_protective_exit.assert_called_once()
        assert exchange.place_protective_exit.call_count == 2  # entry + replace

    def test_tiny_stop_move_does_not_churn_orders(self):
        exchange = _mock_protective_exchange()
        ex = self._open(exchange)
        # 0.05% improvement — below the 0.1% replace threshold
        ex._positions["BTC/USDT"]["stop_price"] = 43500.0 * 1.0005

        ex.update_prices({"BTC/USDT": 45000.0}, datetime.now(UTC))

        exchange.cancel_protective_exit.assert_not_called()
        assert exchange.place_protective_exit.call_count == 1  # entry only
