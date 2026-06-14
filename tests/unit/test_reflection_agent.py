"""Unit tests for ReflectionAgent — 73% → ~95%."""

from unittest.mock import MagicMock

import pytest

from core.reflection_agent import ReflectionAgent


def _trade(
    symbol="BTC/USDT",
    strategy_name="TestStrategy",
    regime="trending_up",
    pnl=100.0,
    exit_reason="take_profit",
    direction="BUY",
    hold_days=5,
    exit_time=None,
):
    t = MagicMock()
    t.symbol = symbol
    t.strategy_name = strategy_name
    t.regime = regime
    t.pnl = pnl
    t.exit_reason = exit_reason
    t.direction = direction
    t.hold_days = hold_days
    t.exit_time = exit_time
    return t


@pytest.mark.unit
class TestReflectionAgent:
    def test_init(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        assert agent._reflect_count == 0

    def test_reflect_win(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        agent.reflect(_trade(pnl=200.0, exit_reason="take_profit"))
        mem.add.assert_called_once()
        args = mem.add.call_args
        assert "BTC/USDT" in args.kwargs.get("situation", args[0][0] if args[0] else "")
        assert agent._reflect_count == 1

    def test_reflect_loss(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        agent.reflect(_trade(pnl=-150.0, exit_reason="stop_loss"))
        mem.add.assert_called_once()
        call_kwargs = mem.add.call_args
        outcome = call_kwargs[1].get("outcome", call_kwargs[0][1] if len(call_kwargs[0]) >= 2 else "")
        assert "LOSS" in outcome

    def test_reflect_increments_count(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        for _ in range(5):
            agent.reflect(_trade())
        assert agent._reflect_count == 5

    def test_reflect_error_doesnt_crash(self):
        mem = MagicMock()
        mem.add.side_effect = Exception("Write failed")
        agent = ReflectionAgent(mem)
        agent.reflect(_trade())  # should not raise
        assert agent._reflect_count == 0

    def test_situation_contains_strategy(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(strategy_name="CryptoMomentumBreakout"))
        assert "CryptoMomentumBreakout" in situation

    def test_situation_contains_regime(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(regime="volatile"))
        assert "volatile" in situation

    def test_situation_stop_hit(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(exit_reason="stop_loss"))
        assert "stop_hit true" in situation

    def test_situation_target_hit(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(exit_reason="take_profit"))
        assert "target_hit true" in situation

    def test_situation_signal_exit(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(exit_reason="reverse_signal"))
        assert "signal_exit true" in situation

    def test_situation_timeout_exit(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(exit_reason="max_hold_timeout"))
        assert "timeout_exit true" in situation

    def test_situation_holding_short(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(hold_days=1))
        assert "holding short" in situation

    def test_situation_holding_medium(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(hold_days=5))
        assert "holding medium" in situation

    def test_situation_holding_long(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(hold_days=15))
        assert "holding long" in situation

    def test_situation_no_hold_days(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        situation = agent._build_situation(_trade(hold_days=None))
        assert "holding" not in situation

    def test_outcome_win(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        outcome = agent._build_outcome(_trade(pnl=100.0, exit_reason="take_profit"))
        assert "WIN" in outcome
        assert "hit target cleanly" in outcome

    def test_outcome_loss_stop(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        outcome = agent._build_outcome(_trade(pnl=-50.0, exit_reason="stop_loss"))
        assert "LOSS" in outcome
        assert "stopped out" in outcome

    def test_outcome_win_other(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        outcome = agent._build_outcome(_trade(pnl=50.0, exit_reason="reverse_signal"))
        assert "profitable" in outcome

    def test_outcome_loss_other(self):
        mem = MagicMock()
        agent = ReflectionAgent(mem)
        outcome = agent._build_outcome(_trade(pnl=-30.0, exit_reason="max_hold"))
        assert "underperformed" in outcome
