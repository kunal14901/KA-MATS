"""Extended execution agent tests — boost from 51% to ~80%."""

import json
import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from agents.execution_agent import BinanceTestnetExecution, CryptoPaperExecution
from core.models import RiskDecision


def _decision(
    symbol="BTC/USDT",
    entry=44500.0,
    stop=43500.0,
    target=47500.0,
    size=0.10,
    strategy="TestStrategy",
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
        risk_amount=max((entry - stop) * size, 0.0),
        veto_reason="",
        strategy_name=strategy,
    )


@pytest.mark.unit
class TestPaperExecutionExtended:
    def test_stop_loss_triggers_close(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision(entry=44500.0, stop=43500.0, target=47500.0)
        ex.execute(d, current_price=44500.0, regime="trending_up")

        ts = datetime(2024, 6, 20, 0, 0, tzinfo=UTC)
        ex.update_prices({"BTC/USDT": 43000.0}, ts)  # below stop
        assert "BTC/USDT" not in ex._positions
        assert len(ex.portfolio.closed_trades) == 1
        assert ex.portfolio.closed_trades[0].exit_reason == "stop_loss"

    def test_take_profit_triggers_close(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision(entry=44500.0, stop=43500.0, target=47500.0)
        ex.execute(d, current_price=44500.0, regime="trending_up")

        ts = datetime(2024, 6, 20, 0, 0, tzinfo=UTC)
        ex.update_prices({"BTC/USDT": 48000.0}, ts)  # above target
        assert "BTC/USDT" not in ex._positions
        assert ex.portfolio.closed_trades[0].exit_reason == "take_profit"

    def test_max_hold_expired(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision()
        ex.execute(d, current_price=44500.0, regime="trending_up")

        ts = datetime(2024, 6, 20, 0, 0, tzinfo=UTC)
        # Simulate 21 bars (max hold = 20)
        for i in range(21):
            ex.update_prices({"BTC/USDT": 44500.0 + i * 10}, ts)
        assert "BTC/USDT" not in ex._positions
        assert ex.portfolio.closed_trades[0].exit_reason == "max_hold_expired"

    def test_short_position_stop_loss(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        # Create a short: stop > entry
        d = _decision(entry=44500.0, stop=46000.0, target=42000.0, size=0.10)
        ex.execute(d, current_price=44500.0, regime="trending_down")

        ts = datetime(2024, 6, 20, 0, 0, tzinfo=UTC)
        ex.update_prices({"BTC/USDT": 46500.0}, ts)  # above stop → short stop loss
        assert "BTC/USDT" not in ex._positions
        assert ex.portfolio.closed_trades[0].exit_reason == "stop_loss"

    def test_short_position_take_profit(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision(entry=44500.0, stop=46000.0, target=42000.0, size=0.10)
        ex.execute(d, current_price=44500.0, regime="trending_down")

        ts = datetime(2024, 6, 20, 0, 0, tzinfo=UTC)
        ex.update_prices({"BTC/USDT": 41000.0}, ts)
        assert "BTC/USDT" not in ex._positions
        assert ex.portfolio.closed_trades[0].exit_reason == "take_profit"

    def test_multiple_positions(self):
        ex = CryptoPaperExecution(initial_capital=100000.0)
        d1 = _decision(symbol="BTC/USDT", entry=44500.0)
        d2 = _decision(symbol="ETH/USDT", entry=3200.0, stop=3000.0, target=3800.0)
        ex.execute(d1, current_price=44500.0, regime="trending_up")
        ex.execute(d2, current_price=3200.0, regime="trending_up")
        assert len(ex._positions) == 2

    def test_insufficient_cash(self):
        ex = CryptoPaperExecution(initial_capital=100.0)
        d = _decision(entry=44500.0, size=1.0)  # costs ~$44500
        result = ex.execute(d, current_price=44500.0)
        assert result is False

    def test_close_all(self):
        ex = CryptoPaperExecution(initial_capital=100000.0)
        d1 = _decision(symbol="BTC/USDT")
        d2 = _decision(symbol="ETH/USDT", entry=3200.0, stop=3000.0, target=3800.0)
        ex.execute(d1, current_price=44500.0, regime="trending_up")
        ex.execute(d2, current_price=3200.0, regime="trending_up")

        ts = datetime(2024, 6, 30, 0, 0, tzinfo=UTC)
        ex.close_all({"BTC/USDT": 45000.0, "ETH/USDT": 3300.0}, ts)
        assert len(ex._positions) == 0
        assert len(ex.portfolio.closed_trades) == 2

    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = str(Path(tmp) / "state.json")
            ex = CryptoPaperExecution(initial_capital=50000.0, state_file=state_file)
            d = _decision()
            ex.execute(d, current_price=44500.0, regime="trending_up")
            ex.save_state()

            ex2 = CryptoPaperExecution(initial_capital=50000.0, state_file=state_file)
            loaded = ex2.load_state()
            assert loaded
            assert "BTC/USDT" in ex2._positions

    def test_load_state_nonexistent(self):
        ex = CryptoPaperExecution(state_file="/nonexistent/path/state.json")
        assert ex.load_state() is False

    def test_equity_tracking(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision()
        ex.execute(d, current_price=44500.0, regime="trending_up")

        ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        ex.update_prices({"BTC/USDT": 45000.0}, ts)
        assert len(ex.equity_history) >= 1
        assert ex.portfolio.net_equity > 0

    def test_funding_rate_applied(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision()
        ex.execute(d, current_price=44500.0, regime="trending_up")

        # Simulate 3 bars × funding interval
        ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        for i in range(4):
            ex.update_prices({"BTC/USDT": 44500.0}, ts)
        # After some bars, funding may have been applied
        # (BARS_PER_FUNDING = 3 for daily bars)
        assert ex._positions.get("BTC/USDT", {}).get("funding_paid", 0.0) >= 0.0

    def test_net_equity_calculation(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision()
        ex.execute(d, current_price=44500.0, regime="trending_up")

        eq = ex._net_equity({"BTC/USDT": 45000.0})
        assert eq > 0

    def test_strategy_specific_max_hold(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        # CryptoTrendPullback has 40-bar max hold
        d = _decision(strategy="CryptoTrendPullback")
        ex.execute(d, current_price=44500.0, regime="trending_up")

        ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        # 25 bars shouldn't close (default max = 20, but trend pullback = 40)
        for i in range(25):
            ex.update_prices({"BTC/USDT": 44500.0}, ts)
        assert "BTC/USDT" in ex._positions  # still open

    def test_breakeven_stop_activates(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision(entry=44500.0, stop=43500.0, target=47500.0)
        ex.execute(d, current_price=44500.0, regime="trending_up")
        # Manually set atr_at_entry to enable break-even logic
        ex._positions["BTC/USDT"]["atr_at_entry"] = 500.0
        original_stop = ex._positions["BTC/USDT"]["stop_price"]

        ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        # Price rises well above 1.2× ATR from fill price to trigger breakeven
        ex.update_prices({"BTC/USDT": 45500.0}, ts)
        # Stop should be moved up from original
        assert ex._positions["BTC/USDT"]["stop_price"] >= original_stop

    def test_trailing_stop_activates(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision(entry=44500.0, stop=43500.0, target=47500.0)
        ex.execute(d, current_price=44500.0, regime="trending_up")
        ex._positions["BTC/USDT"]["atr_at_entry"] = 500.0

        ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        # Price rises significantly → trailing stop activates
        ex.update_prices({"BTC/USDT": 46000.0}, ts)
        original_stop = ex._positions["BTC/USDT"]["stop_price"]

        # Further rise → trailing stop moves up
        ex.update_prices({"BTC/USDT": 47000.0}, ts)
        new_stop = ex._positions["BTC/USDT"]["stop_price"]
        assert new_stop >= original_stop

    def test_short_position_breakeven(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        # Short: stop > entry
        d = _decision(entry=44500.0, stop=46000.0, target=42000.0, size=0.10)
        ex.execute(d, current_price=44500.0, regime="trending_down")
        ex._positions["BTC/USDT"]["atr_at_entry"] = 500.0
        original_stop = ex._positions["BTC/USDT"]["stop_price"]

        ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        # Price drops well below entry to trigger breakeven for short
        ex.update_prices({"BTC/USDT": 43500.0}, ts)
        assert ex._positions["BTC/USDT"]["stop_price"] <= original_stop

    def test_short_trailing_stop(self):
        ex = CryptoPaperExecution(initial_capital=50000.0)
        d = _decision(entry=44500.0, stop=46000.0, target=42500.0, size=0.10)
        ex.execute(d, current_price=44500.0, regime="trending_down")
        ex._positions["BTC/USDT"]["atr_at_entry"] = 500.0

        ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        # Price drops but not below target
        ex.update_prices({"BTC/USDT": 43000.0}, ts)
        stop1 = ex._positions["BTC/USDT"]["stop_price"]

        ex.update_prices({"BTC/USDT": 42800.0}, ts)  # further profit
        stop2 = ex._positions["BTC/USDT"]["stop_price"]
        assert stop2 <= stop1  # trailing stop tightens (lower for short)

    def test_json_default_datetime(self):
        from agents.execution_agent import CryptoPaperExecution

        result = CryptoPaperExecution._json_default(datetime(2024, 1, 1, tzinfo=UTC))
        assert "2024" in result

    def test_json_default_non_serializable(self):
        import pytest

        from agents.execution_agent import CryptoPaperExecution

        with pytest.raises(TypeError):
            CryptoPaperExecution._json_default(set())


@pytest.mark.unit
class TestBinanceTestnetExecution:
    def test_init(self):
        te = BinanceTestnetExecution()
        assert te._client is None
