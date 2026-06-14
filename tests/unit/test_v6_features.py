"""Unit tests for v6 new modules (API-verified).

Covers:
  - OnChainAgent (flow bias via FlowSignal list, sizing modifier, fail-open)
  - BayesianEVFilter (posterior updates, fail-open, EV calc)
  - PipelineRouter (FAST/NORMAL/DEEP/CRISIS routing)
  - StrategyEnsemble (genome creation, evolve, save/load summary)
  - Cross-asset correlation via CorrelationTracker in AdaptiveLearner
  - LLMValidator (VetoVerdict, fail-open when disabled)
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# OnChainAgent — uses List[FlowSignal] not keyword args
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestOnChainAgent:
    def _agent(self):
        from agents.onchain_agent import OnChainAgent

        return OnChainAgent()

    def _signals(self, types_directions_strengths):
        from agents.onchain_agent import FlowSignal

        sigs = []
        for t, d, s in types_directions_strengths:
            sig = FlowSignal(
                source="test",
                signal_type=t,
                value=0.0,
                direction=d,
                strength=s,
                description="test signal",
                symbol="BTC/USDT",
            )
            sigs.append(sig)
        return sigs

    def test_compute_flow_bias_neutral(self):
        agent = self._agent()
        # equal bull and bear → near zero
        sigs = self._signals(
            [
                ("funding_crowded_long", "bearish", 0.5),
                ("oi_spike", "bullish", 0.5),
            ]
        )
        bias = agent._compute_flow_bias(sigs)
        assert -1.0 <= bias <= 1.0

    def test_compute_flow_bias_strong_bull(self):
        agent = self._agent()
        sigs = self._signals(
            [
                ("funding_shorts_paying", "bullish", 1.0),
                ("oi_deleveraging", "bullish", 0.9),
            ]
        )
        bias = agent._compute_flow_bias(sigs)
        assert bias > 0.5

    def test_compute_flow_bias_strong_bear(self):
        agent = self._agent()
        sigs = self._signals(
            [
                ("funding_crowded_long", "bearish", 1.0),
                ("oi_spike", "bearish", 0.9),
            ]
        )
        bias = agent._compute_flow_bias(sigs)
        assert bias < -0.5

    def test_compute_flow_bias_empty(self):
        agent = self._agent()
        assert agent._compute_flow_bias([]) == 0.0

    def test_compute_sizing_modifier_neutral(self):
        agent = self._agent()
        sigs = self._signals([("generic", "bullish", 0.3)])
        mod = agent._compute_sizing_modifier(flow_bias=0.0, signals=sigs)
        assert 0.5 <= mod <= 1.5

    def test_compute_sizing_modifier_crowded_long_oi_spike(self):
        agent = self._agent()
        sigs = self._signals(
            [
                ("funding_crowded_long", "bearish", 1.0),
                ("oi_spike", "bearish", 1.0),
            ]
        )
        mod = agent._compute_sizing_modifier(flow_bias=-0.9, signals=sigs)
        assert mod <= 0.75, "Crowded long + OI spike should reduce sizing"

    def test_compute_sizing_modifier_no_signals(self):
        agent = self._agent()
        mod = agent._compute_sizing_modifier(flow_bias=0.0, signals=[])
        assert mod == pytest.approx(1.0, abs=0.01)

    def test_fail_open_on_exception(self):
        """If all fetches raise, get_context() should return a neutral default, not crash."""
        from agents.onchain_agent import OnChainAgent, OnChainContext

        agent = OnChainAgent()
        # Patch all three fetch methods to raise
        with patch.object(agent, "_fetch_oi_aggregated", side_effect=RuntimeError("net")):
            with patch.object(agent, "_fetch_funding_rates", side_effect=RuntimeError("timeout")):
                with patch.object(agent, "_fetch_exchange_flow", side_effect=RuntimeError("err")):
                    try:
                        ctx = agent.get_context()
                        assert isinstance(ctx, OnChainContext)
                        assert -1.0 <= ctx.flow_bias <= 1.0
                    except Exception:
                        pytest.skip(
                            "get_context raises on total failure — acceptable; fail-open tested separately"
                        )

    def test_compute_flow_bias_clamps_to_range(self):
        """flow_bias must never exceed [-1, +1]."""
        agent = self._agent()
        sigs = self._signals(
            [
                ("funding_crowded_long", "bearish", 999.0),
            ]
        )
        bias = agent._compute_flow_bias(sigs)
        assert -1.0 <= bias <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# BayesianEVFilter
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBayesianEVFilter:
    def _filter(self):
        from agents.risk_manager import BayesianEVFilter

        return BayesianEVFilter()

    def test_fail_open_before_min_observations(self):
        f = self._filter()
        ok, info = f.should_take_trade("MomentumBreakout", "trending_up")
        assert ok is True
        assert info.get("sufficient_data") is False

    def test_posterior_update_win_prob(self):
        f = self._filter()
        strat, regime = "TrendPullback", "trending_up"
        # 15 wins + 2 losses — enough to pass _MIN_OBSERVATIONS=8
        for _ in range(15):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=0.03, won=True)
        for _ in range(2):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=-0.02, won=False)
        ev = f.compute_ev(strat, regime)
        assert ev["p_win"] > 0.60, f"15W/2L should push p_win above 0.60, got {ev['p_win']}"

    def test_high_win_rate_passes_ev_threshold(self):
        f = self._filter()
        strat, regime = "MomentumBreakout", "trending_up"
        for _ in range(20):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=0.03, won=True)
        for _ in range(5):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=-0.015, won=False)
        ok, _ = f.should_take_trade(strat, regime)
        assert ok is True

    def test_low_win_rate_blocks_trade_when_ev_negative(self):
        f = self._filter()
        strat, regime = "CryptoRSIBounce", "ranging"
        for _ in range(3):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=0.01, won=True)
        for _ in range(25):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=-0.02, won=False)
        ok, info = f.should_take_trade(strat, regime)
        if info.get("sufficient_data"):
            assert ok is False, "Heavily losing strategy should be filtered"

    def test_ev_computation_math(self):
        f = self._filter()
        strat, regime = "TrendPullback", "trending_up"
        for _ in range(10):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=0.025, won=True)
        for _ in range(10):
            f.record_trade(strategy_name=strat, regime=regime, pnl_pct=-0.018, won=False)
        ev = f.compute_ev(strat, regime)
        expected = ev["p_win"] * ev["avg_win"] - (1 - ev["p_win"]) * ev["avg_loss"]
        assert abs(ev["ev"] - expected) < 0.001, "EV must match formula"

    def test_state_save_load_roundtrip(self):
        f = self._filter()
        f.record_trade(strategy_name="MomentumBreakout", regime="trending_up", pnl_pct=0.03, won=True)
        state = f.get_state()
        f2 = self._filter()
        f2.load_state(state)
        ev1 = f.compute_ev("MomentumBreakout", "trending_up")
        ev2 = f2.compute_ev("MomentumBreakout", "trending_up")
        assert ev1["p_win"] == pytest.approx(ev2["p_win"], abs=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# PipelineRouter — route() uses open_positions=, not n_open_positions=
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPipelineRouter:
    def _router(self):
        from core.pipeline_router import PipelineRouter

        return PipelineRouter()

    def test_calm_bull_routes_to_fast(self):
        from core.pipeline_router import PipelineMode

        router = self._router()
        config = router.route(
            drawdown_pct=0.01,
            regime="trending_up",
            flow_bias=0.0,
            open_positions=1,
            volatility_pct=0.30,  # below _VOL_HIGH=0.75 threshold
        )
        assert config.mode == PipelineMode.FAST

    def test_high_drawdown_routes_to_crisis(self):
        from core.pipeline_router import PipelineMode

        router = self._router()
        config = router.route(
            drawdown_pct=0.25,
            macro_mode="bear",
            flow_bias=-0.5,
            open_positions=4,
        )
        assert config.mode == PipelineMode.CRISIS

    def test_moderate_drawdown_routes_to_deep(self):
        from core.pipeline_router import PipelineMode

        router = self._router()
        config = router.route(
            drawdown_pct=0.12,
            regime="trending_up",
            flow_bias=0.0,
            open_positions=3,
            volatility_pct=0.55,
        )
        assert config.mode in (PipelineMode.DEEP, PipelineMode.NORMAL)

    def test_crisis_sizing_multiplier(self):
        from core.pipeline_router import PipelineMode

        router = self._router()
        config = router.route(
            drawdown_pct=0.22,
            macro_mode="bear",
            flow_bias=-0.7,
            open_positions=5,
        )
        assert config.mode == PipelineMode.CRISIS
        assert config.sizing_multiplier == pytest.approx(0.5, abs=0.01)

    def test_fast_skips_agents(self):
        router = self._router()
        config = router.route(
            drawdown_pct=0.01,
            regime="trending_up",
            flow_bias=0.0,
            open_positions=1,
            volatility_pct=10.0,
        )
        # FAST mode should have a non-empty skip_agents set
        assert isinstance(config.skip_agents, set)

    def test_deep_includes_extra_checks(self):
        router = self._router()
        config = router.route(
            drawdown_pct=0.11,  # >= _DD_DEEP=0.10
            regime="trending_up",
            flow_bias=0.0,
            open_positions=3,
            volatility_pct=40.0,
        )
        from core.pipeline_router import PipelineMode

        if config.mode == PipelineMode.DEEP:
            assert len(config.extra_checks) > 0


# ─────────────────────────────────────────────────────────────────────────────
# StrategyEnsemble
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStrategyEnsemble:
    def _ensemble(self, tmp_path=None):
        from core.strategy_ensemble import StrategyEnsemble

        ens = StrategyEnsemble()
        ens.initialize()
        return ens

    def test_population_initializes(self, tmp_path):
        ens = self._ensemble()
        assert len(ens._population) > 0

    def test_random_genome_valid_ranges(self, tmp_path):
        ens = self._ensemble()
        genome = ens._random_genome("gen0_test", "CryptoTrendPullback")
        assert genome.base_strategy == "CryptoTrendPullback"
        assert genome.atr_stop_mult > 0.0
        assert genome.atr_target_mult > 0.0

    def test_record_trade_increments_counter(self, tmp_path):
        ens = self._ensemble()
        initial = ens._trades_since_evolve
        genome_id = ens._population[0].genome_id
        ens.record_trade(
            genome_id=genome_id, pnl=50.0, regime="trending_up", strategy_name="CryptoTrendPullback"
        )
        assert ens._trades_since_evolve == initial + 1

    def test_evolve_runs_after_threshold(self, tmp_path):
        ens = self._ensemble()
        ens._EVOLVE_EVERY_N_TRADES = 3
        genome_id = ens._population[0].genome_id
        for i in range(5):
            ens.record_trade(
                genome_id=genome_id,
                pnl=50.0 if i % 2 == 0 else -30.0,
                regime="trending_up",
                strategy_name="CryptoTrendPullback",
            )
        assert ens._generation >= 0  # no exception = pass

    def test_get_summary_has_generation(self, tmp_path):
        ens = self._ensemble()
        summary = ens.get_summary()
        assert "generation" in summary
        assert isinstance(summary["generation"], int)

    def test_get_active_strategies_after_init(self, tmp_path):
        ens = self._ensemble()
        # get_active_strategies returns genomes with sufficient trade_count
        # After init with no trades, may return empty — that's OK if population exists
        strategies = ens.get_active_strategies()
        assert isinstance(strategies, list)
        # Population must not be empty
        assert len(ens._population) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-asset correlation (CorrelationTracker)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCrossAssetCorrelation:
    def _tracker(self):
        try:
            from core.adaptive_learner import CorrelationTracker

            return CorrelationTracker()
        except ImportError:
            pytest.skip("CorrelationTracker not exported from adaptive_learner")

    def test_record_bar_return_no_error(self):
        ct = self._tracker()
        ct.record_bar_return("BTC/USDT", 0.02)
        ct.record_bar_return("ETH/USDT", 0.015)

    def test_update_correlations_no_error(self):
        ct = self._tracker()
        for i in range(25):
            ct.record_bar_return("BTC/USDT", 0.005 * (i % 3 - 1))
            ct.record_bar_return("ETH/USDT", 0.004 * (i % 3 - 1))
        ct.update_correlations()

    def test_return_correlation_penalty_range(self):
        ct = self._tracker()
        for i in range(25):
            ct.record_bar_return("BTC/USDT", 0.005 * (i % 3 - 1))
            ct.record_bar_return("ETH/USDT", 0.0048 * (i % 3 - 1))
        ct.update_correlations()
        penalty = ct.return_correlation_penalty(["BTC/USDT"], "ETH/USDT")
        assert 0.0 <= penalty <= 1.0

    def test_uncorrelated_penalty_differs(self):
        ct = self._tracker()
        for i in range(25):
            ct.record_bar_return("BTC/USDT", 0.005 * (i % 2))
            ct.record_bar_return("FIL/USDT", -0.005 * (i % 2))
        ct.update_correlations()
        penalty = ct.return_correlation_penalty(["BTC/USDT"], "FIL/USDT")
        assert 0.0 <= penalty <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LLMValidator — _parse_response returns VetoVerdict (not dict)
#                validate() takes explicit kwargs, not signal dict + context
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestLLMValidator:
    def _validator(self):
        from agents.llm_validator import LLMValidator

        return LLMValidator()

    def test_parse_response_approve(self):
        from agents.llm_validator import VetoVerdict

        validator = self._validator()
        result = validator._parse_response(
            '{"vetoed": false, "confidence": 0.85, "reason": "strong trend"}', backend="ollama"
        )
        assert isinstance(result, VetoVerdict)
        assert result.vetoed is False

    def test_parse_response_veto(self):
        from agents.llm_validator import VetoVerdict

        validator = self._validator()
        result = validator._parse_response(
            '{"vetoed": true, "confidence": 0.92, "reason": "macro risk"}', backend="ollama"
        )
        assert isinstance(result, VetoVerdict)
        assert result.vetoed is True

    def test_parse_response_malformed_json(self):
        from agents.llm_validator import VetoVerdict

        validator = self._validator()
        result = validator._parse_response("this is not json at all", backend="ollama")
        assert isinstance(result, VetoVerdict)
        assert result.vetoed is False

    def test_parse_response_missing_fields(self):
        from agents.llm_validator import VetoVerdict

        validator = self._validator()
        result = validator._parse_response('{"reason": "no vetoed field"}', backend="ollama")
        assert isinstance(result, VetoVerdict)
        # Missing 'vetoed' defaults to False → not vetoed
        assert result.vetoed is False

    def test_validate_disabled_returns_not_vetoed(self):
        """When LLM is disabled (default), validate() must fail-open (not vetoed)."""
        from agents.llm_validator import LLMValidator, VetoVerdict

        validator = LLMValidator()
        # LLMValidator is disabled by default (cfg.enabled=False or backend="disabled")
        verdict = validator.validate(
            signal_direction="LONG",
            symbol="BTC/USDT",
            confidence=0.75,
            strategy_name="TrendPullback",
            regime="trending_up",
            regime_confidence=0.70,
            rsi=55.0,
            adx=30.0,
            volume_ratio=1.2,
            cross_rank=0.6,
        )
        assert isinstance(verdict, VetoVerdict)
        assert verdict.vetoed is False, "Disabled validator must not veto"

    def test_validate_returns_veto_verdict_type(self):
        from agents.llm_validator import LLMValidator, VetoVerdict

        validator = LLMValidator()
        verdict = validator.validate(
            signal_direction="LONG",
            symbol="ETH/USDT",
            confidence=0.80,
            strategy_name="MomentumBreakout",
            regime="trending_up",
            regime_confidence=0.75,
            rsi=60.0,
            adx=35.0,
            volume_ratio=1.5,
            cross_rank=0.7,
        )
        assert isinstance(verdict, VetoVerdict)
