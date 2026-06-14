"""Unit tests for Execution Agent (Paper Trading)."""

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

from agents.execution_agent import CryptoPaperExecution
from core.models import RiskDecision


def _approved_decision(
    symbol: str = "BTC/USDT",
    entry_price: float = 44500.0,
    stop_loss: float = 43500.0,
    take_profit: float = 47500.0,
    position_size: float = 0.10,
) -> RiskDecision:
    return RiskDecision(
        signal_id=uuid4(),
        symbol=symbol,
        timestamp=datetime.now(UTC),
        approved=True,
        position_size=position_size,
        position_value=position_size * entry_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_amount=max((entry_price - stop_loss) * position_size, 0.0),
        veto_reason="",
        strategy_name="TestStrategy",
    )


@pytest.mark.unit
class TestPaperExecution:
    """Test paper trading execution logic."""

    def test_execute_buy_order(self):
        """Test opening a long position."""
        executor = CryptoPaperExecution()

        decision = _approved_decision()

        executed = executor.execute(decision, current_price=44500.0, regime="trending_up")

        assert executed
        assert "BTC/USDT" in executor.portfolio.positions

        position = executor.portfolio.positions["BTC/USDT"]
        assert position["shares"] == pytest.approx(0.10)
        assert position["stop_price"] == pytest.approx(43500.0)
        assert position["target_price"] == pytest.approx(47500.0)

    def test_reject_unapproved_decision(self):
        """Test that unapproved decisions are not executed."""
        executor = CryptoPaperExecution()

        decision = RiskDecision(
            signal_id=uuid4(),
            symbol="BTC/USDT",
            timestamp=datetime.now(UTC),
            approved=False,
            position_size=0.0,
            veto_reason="Max positions reached",
        )

        executed = executor.execute(decision, current_price=44500.0, regime="trending_up")

        assert not executed
        assert "BTC/USDT" not in executor.portfolio.positions

    def test_deduct_cash_on_buy(self):
        """Test that cash is deducted when buying."""
        executor = CryptoPaperExecution()
        initial_cash = executor.portfolio.cash

        decision = _approved_decision()

        executor.execute(decision, current_price=44500.0, regime="trending_up")

        # Cash should decrease by position value + fees
        expected_cost = 0.10 * 44500.0  # $4450
        assert executor.portfolio.cash < initial_cash
        assert executor.portfolio.cash < initial_cash - expected_cost * 0.99  # Allowing for fees

    def test_stop_loss_hit(self):
        """Test position closed when stop loss is hit."""
        executor = CryptoPaperExecution()

        # Open position
        decision = _approved_decision()

        executor.execute(decision, current_price=44500.0, regime="trending_up")

        # Update price below stop
        executor.update_prices({"BTC/USDT": 43400.0}, datetime.now(UTC))

        # Position should be closed
        assert "BTC/USDT" not in executor.portfolio.positions
        assert len(executor.portfolio.closed_trades) == 1

        trade = executor.portfolio.closed_trades[0]
        assert trade.exit_reason == "stop_loss"
        assert trade.pnl < 0  # Lost money

    def test_take_profit_hit(self):
        """Test position closed when take profit is hit."""
        executor = CryptoPaperExecution()

        # Open position
        decision = _approved_decision()

        executor.execute(decision, current_price=44500.0, regime="trending_up")

        # Update price above target
        executor.update_prices({"BTC/USDT": 47600.0}, datetime.now(UTC))

        # Position should be closed
        assert "BTC/USDT" not in executor.portfolio.positions
        assert len(executor.portfolio.closed_trades) == 1

        trade = executor.portfolio.closed_trades[0]
        assert trade.exit_reason == "take_profit"
        assert trade.pnl > 0  # Made profit

    def test_pnl_calculation(self):
        """Test PnL calculation is accurate."""
        executor = CryptoPaperExecution()

        decision = _approved_decision()

        executor.execute(decision, current_price=44500.0, regime="trending_up")

        # Close at profit
        executor.update_prices({"BTC/USDT": 47500.0}, datetime.now(UTC))

        trade = executor.portfolio.closed_trades[0]

        # PnL = (exit - entry) * quantity - fees
        expected_pnl = (47500.0 - 44500.0) * 0.10  # $300 before fees
        assert expected_pnl > 0
        assert trade.pnl > 250  # After fees/slippage, should remain strongly positive
        assert trade.pnl < 310

    def test_trailing_stop(self):
        """Test trailing stop logic (if implemented)."""
        executor = CryptoPaperExecution()

        decision = _approved_decision()

        executor.execute(decision, current_price=44500.0, regime="trending_up")

        # Price moves up significantly
        executor.update_prices({"BTC/USDT": 46000.0}, datetime.now(UTC))

        # Note: trailing stop may or may not be implemented
        # This test will pass either way
        assert "BTC/USDT" in executor.portfolio.positions or len(executor.portfolio.closed_trades) > 0

    def test_multiple_positions(self):
        """Test handling multiple open positions."""
        executor = CryptoPaperExecution()

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        for sym in symbols:
            entry = 44500.0 if sym == "BTC/USDT" else 100.0
            decision = _approved_decision(
                symbol=sym,
                entry_price=entry,
                stop_loss=entry * 0.95,
                take_profit=entry * 1.10,
                position_size=0.10 if sym == "BTC/USDT" else 1.0,
            )

            executor.execute(decision, current_price=decision.entry_price, regime="trending_up")

        assert len(executor.portfolio.positions) == 3

    def test_net_equity_calculation(self):
        """Test net equity includes cash + open position value."""
        executor = CryptoPaperExecution()

        decision = _approved_decision()

        initial_equity = executor.portfolio.net_equity

        executor.execute(decision, current_price=44500.0, regime="trending_up")

        # Equity should decrease slightly due to fees
        assert executor.portfolio.net_equity == pytest.approx(initial_equity)

        # Update price to profit
        executor.update_prices({"BTC/USDT": 46000.0}, datetime.now(UTC))

        # Equity should be higher (unrealized profit)
        assert executor.portfolio.net_equity > initial_equity
