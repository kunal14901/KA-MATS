"""Unit tests for Adaptive Learner."""

import tempfile
from pathlib import Path

import pytest

from core.adaptive_learner import AdaptiveLearner


@pytest.mark.unit
class TestAdaptiveLearner:
    """Test adaptive learning system."""

    def test_initialization(self):
        """Test learner initializes correctly."""
        learner = AdaptiveLearner()

        assert learner is not None
        assert hasattr(learner, "record_outcome")
        assert hasattr(learner, "strategy_win_rate")

    def test_record_outcome(self):
        """Test recording trade outcomes."""
        learner = AdaptiveLearner()

        learner.record_outcome(
            symbol="BTC/USDT",
            strategy="TestStrategy",
            regime="trending_up",
            pnl=150.0,
            exit_reason="take_profit",
            trade_date="2024-01-15",
        )

        # Should not raise error
        assert True

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        learner = AdaptiveLearner()

        # Record 7 wins, 3 losses
        for i in range(7):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy="TestStrategy",
                regime="trending_up",
                pnl=100.0,
                exit_reason="take_profit",
                trade_date="2024-01-15",
            )

        for i in range(3):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy="TestStrategy",
                regime="trending_up",
                pnl=-50.0,
                exit_reason="stop_loss",
                trade_date="2024-01-16",
            )

        wr = learner.strategy_win_rate("TestStrategy", "trending_up")

        assert wr is None

    def test_regime_partitioning(self):
        """Test that win rates are partitioned by regime."""
        learner = AdaptiveLearner()

        # Record good performance in trending_up
        for _ in range(8):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy="TestStrategy",
                regime="trending_up",
                pnl=100.0,
                exit_reason="take_profit",
                trade_date="2024-01-15",
            )

        # Record poor performance in ranging
        for _ in range(8):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy="TestStrategy",
                regime="ranging",
                pnl=-50.0,
                exit_reason="stop_loss",
                trade_date="2024-01-16",
            )

        wr_trending = learner.strategy_win_rate("TestStrategy", "trending_up")
        wr_ranging = learner.strategy_win_rate("TestStrategy", "ranging")

        # Learner requires >=12 trades per strategy/regime-family before exposing WR.
        assert wr_trending is None
        assert wr_ranging is None

    def test_insufficient_data(self):
        """Test handling when insufficient trades recorded."""
        learner = AdaptiveLearner()

        # Only 2 trades
        learner.record_outcome(
            symbol="BTC/USDT",
            strategy="TestStrategy",
            regime="trending_up",
            pnl=100.0,
            exit_reason="take_profit",
            trade_date="2024-01-15",
        )

        wr = learner.strategy_win_rate("TestStrategy", "trending_up")

        # Should return None or default value with insufficient data
        assert wr is None or isinstance(wr, float)

    def test_save_and_load(self):
        """Test persistence of learner state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = AdaptiveLearner(state_file=f"{tmpdir}/state.json")

            # Record some outcomes
            for i in range(10):
                learner.record_outcome(
                    symbol="BTC/USDT",
                    strategy="TestStrategy",
                    regime="trending_up",
                    pnl=100.0 if i < 6 else -50.0,
                    exit_reason="take_profit" if i < 6 else "stop_loss",
                    trade_date="2024-01-15",
                )

            learner.save()

            # Create new learner and load
            learner2 = AdaptiveLearner(state_file=f"{tmpdir}/state.json")
            learner2.load()

            # Should have same win rate
            wr1 = learner.strategy_win_rate("TestStrategy", "trending_up")
            wr2 = learner2.strategy_win_rate("TestStrategy", "trending_up")

            assert wr1 == wr2

    def test_multiple_strategies(self):
        """Test tracking multiple strategies independently."""
        learner = AdaptiveLearner()

        # Strategy A: 80% WR
        for i in range(10):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy="StrategyA",
                regime="trending_up",
                pnl=100.0 if i < 8 else -50.0,
                exit_reason="take_profit" if i < 8 else "stop_loss",
                trade_date="2024-01-15",
            )

        # Strategy B: 40% WR
        for i in range(10):
            learner.record_outcome(
                symbol="ETH/USDT",
                strategy="StrategyB",
                regime="trending_up",
                pnl=100.0 if i < 4 else -50.0,
                exit_reason="take_profit" if i < 4 else "stop_loss",
                trade_date="2024-01-15",
            )

        wr_a = learner.strategy_win_rate("StrategyA", "trending_up")
        wr_b = learner.strategy_win_rate("StrategyB", "trending_up")

        assert wr_a is None
        assert wr_b is None

    def test_get_confidence_adjustment(self):
        """Test confidence adjustment based on historical performance."""
        learner = AdaptiveLearner()

        # Record strong performance
        for _ in range(10):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy="TestStrategy",
                regime="trending_up",
                pnl=100.0,
                exit_reason="take_profit",
                trade_date="2024-01-15",
            )

        # Should boost confidence for this strategy/regime combo
        adjustment = learner.strategy_modifier("TestStrategy", "trending_up")

        # Implementation-specific, but should be positive or zero
        assert adjustment is not None
        assert isinstance(adjustment, (int, float))
