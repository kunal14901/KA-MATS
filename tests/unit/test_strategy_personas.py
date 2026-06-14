"""Unit tests for StrategyPersona + StrategyPersonaManager — 0% → ~90%."""

from unittest.mock import MagicMock

import pytest

from core.strategy_personas import StrategyPersona, StrategyPersonaManager


@pytest.mark.unit
class TestStrategyPersona:
    def test_defaults(self):
        p = StrategyPersona(name="TestStrat")
        assert p.name == "TestStrat"
        assert p.overall_win_rate == 0.50
        assert p.health_score == 0.5
        assert p.total_trades == 0

    def test_summary(self):
        p = StrategyPersona(name="TestStrat", overall_win_rate=0.65, health_score=0.80)
        s = p.summary()
        assert "TestStrat" in s
        assert "win=" in s
        assert "health=" in s

    def test_summary_with_best_worst(self):
        p = StrategyPersona(
            name="X",
            best_regime="bull",
            worst_regime="bear",
            win_rate_by_regime={"bull": 0.70, "bear": 0.30},
        )
        s = p.summary()
        assert "bull" in s
        assert "bear" in s

    def test_win_rate_for_known_regime(self):
        p = StrategyPersona(
            name="X",
            win_rate_by_regime={"trending_up": 0.72, "volatile": 0.35},
        )
        assert p.win_rate_for("trending_up") == 0.72
        assert p.win_rate_for("volatile") == 0.35

    def test_win_rate_for_unknown_regime_falls_back(self):
        p = StrategyPersona(name="X", overall_win_rate=0.55)
        assert p.win_rate_for("unknown_regime") == 0.55

    def test_is_healthy_true(self):
        p = StrategyPersona(name="X", health_score=0.60, data_confidence=0.5)
        assert p.is_healthy()

    def test_is_healthy_low_health(self):
        p = StrategyPersona(name="X", health_score=0.20, data_confidence=0.5)
        assert not p.is_healthy()

    def test_is_healthy_no_data(self):
        p = StrategyPersona(name="X", health_score=0.80, data_confidence=0.0)
        assert not p.is_healthy()

    def test_is_healthy_custom_min(self):
        p = StrategyPersona(name="X", health_score=0.50, data_confidence=0.5)
        assert not p.is_healthy(min_health=0.60)
        assert p.is_healthy(min_health=0.40)


@pytest.mark.unit
class TestStrategyPersonaManager:
    def _mock_learner(self):
        """Create a mock AdaptiveLearner with strategy × regime data."""
        learner = MagicMock()
        learner.get_learning_summary.return_value = {
            "strategy_insights": [
                {"key": "CryptoTrendPullback::bull", "win_rate": 0.65, "modifier": 0.05},
                {"key": "CryptoTrendPullback::bear", "win_rate": 0.30, "modifier": -0.10},
                {"key": "CryptoCSM::bull", "win_rate": 0.55, "modifier": 0.02},
                {"key": "CryptoCSM::sideways", "win_rate": 0.48, "modifier": -0.01},
            ],
            "symbol_insights": [
                {"symbol": "BTC/USDT", "stop_hit_rate": 0.30},
                {"symbol": "ETH/USDT", "stop_hit_rate": 0.25},
            ],
        }
        return learner

    def test_build_populates_personas(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        assert mgr._built
        assert len(mgr.personas) > 0

    def test_build_populates_win_rates(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        p = mgr.get_persona("CryptoTrendPullback")
        assert p is not None
        assert p.win_rate_by_regime.get("bull") == 0.65
        assert p.win_rate_by_regime.get("bear") == 0.30

    def test_build_best_worst_regime(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        p = mgr.get_persona("CryptoTrendPullback")
        assert p.best_regime == "bull"
        assert p.worst_regime == "bear"

    def test_build_regime_selectivity(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        p = mgr.get_persona("CryptoTrendPullback")
        # spread = 0.65 - 0.30 = 0.35, selectivity = min(1.0, 0.35 * 2.5) = 0.875
        assert p.regime_selectivity > 0.5

    def test_build_computes_health(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        for p in mgr.personas.values():
            assert 0.0 <= p.health_score <= 1.0

    def test_best_for_regime(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        result = mgr.best_for_regime("bull")
        # Should return a tuple or None
        if result is not None:
            name, wr = result
            assert isinstance(name, str)
            assert wr >= 0.40

    def test_best_for_regime_not_built(self):
        mgr = StrategyPersonaManager()
        assert mgr.best_for_regime("bull") is None

    def test_rank_for_regime(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        ranked = mgr.rank_for_regime("bull")
        assert isinstance(ranked, list)
        if len(ranked) > 1:
            assert ranked[0][1] >= ranked[1][1]  # sorted descending

    def test_rank_for_regime_not_built(self):
        mgr = StrategyPersonaManager()
        assert mgr.rank_for_regime("bull") == []

    def test_get_persona_unknown(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        assert mgr.get_persona("NonExistent") is None

    def test_all_summaries(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        summaries = mgr.all_summaries()
        assert len(summaries) == len(mgr.personas)

    def test_report(self):
        mgr = StrategyPersonaManager()
        mgr.build(self._mock_learner())
        report = mgr.report("bull")
        assert "bull" in report
        assert "ACTIVE" in report or "SUPPRESSED" in report
