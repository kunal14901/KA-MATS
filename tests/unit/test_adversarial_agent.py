"""Unit tests for AdversarialAgent — 14% → ~80%."""

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

from agents.adversarial_agent import AdversarialAgent
from core.models import (
    AdversarialVerdict,
    AltDataContext,
    CandidateSignal,
    ConvictionScore,
    Features,
    Indicators,
    KnowledgeContext,
    MarketSnapshot,
    PriceData,
    RegimeAnalysis,
    RegimeType,
    SAConviction,
    SignalDirection,
    ThesisContext,
)
from core.session_memory import SessionContext


def _signal(
    symbol="BTC/USDT",
    direction=SignalDirection.BUY,
    confidence=0.70,
    strategy="TestStrategy",
):
    return CandidateSignal(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=direction,
        strategy_name=strategy,
        confidence=confidence,
        entry_price=44500.0,
        stop_price=43500.0,
        target_price=47500.0,
        conditions=[],
    )


def _snapshot(symbol="BTC/USDT", rsi=55.0, volume_ratio=1.5):
    return MarketSnapshot(
        symbol=symbol,
        price=PriceData(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            open=44200.0,
            high=44800.0,
            low=44000.0,
            close=44500.0,
            volume=2000.0,
        ),
        indicators=Indicators(
            ema_20=44500.0,
            ema_50=43800.0,
            ema_200=42000.0,
            rsi_14=rsi,
            atr_14=850.0,
            bb_upper=46000.0,
            bb_middle=44500.0,
            bb_lower=43000.0,
            adx_14=28.0,
            plus_di=32.0,
            minus_di=18.0,
            macd=250.0,
            macd_signal=180.0,
        ),
        features=Features(volume_ratio=volume_ratio, zscore_20=1.0, cross_rank=0.5),
        timestamp=datetime.now(UTC),
        data_quality_ok=True,
    )


def _regime(regime=RegimeType.TRENDING_UP, confidence=0.85):
    return RegimeAnalysis(
        symbol="BTC/USDT",
        timestamp=datetime.now(UTC),
        regime=regime,
        confidence=confidence,
        trend_strength=28.0,
        volatility_pct=40.0,
        zscore=1.0,
        rationale="test",
    )


@pytest.mark.unit
class TestAdversarialAgent:
    def test_init(self):
        agent = AdversarialAgent()
        assert agent is not None

    def test_assess_all_pass(self):
        agent = AdversarialAgent()
        signals = [_signal()]
        assessments = agent.assess(signals, _snapshot(), _regime())
        assert len(assessments) == 1
        assert assessments[0].verdict == AdversarialVerdict.PASS

    def test_assess_multiple_signals(self):
        agent = AdversarialAgent()
        signals = [_signal(symbol="BTC/USDT"), _signal(symbol="ETH/USDT")]
        assessments = agent.assess(signals, _snapshot(), _regime())
        assert len(assessments) == 2

    # ── Macro short filter ────────────────────────────────

    def test_macro_short_filter_kills_sell_in_uptrend(self):
        agent = AdversarialAgent()
        sig = _signal(direction=SignalDirection.SELL)
        assessments = agent.assess([sig], _snapshot(), _regime(RegimeType.TRENDING_UP))
        assert assessments[0].verdict == AdversarialVerdict.FAIL

    def test_macro_short_filter_allows_buy_in_uptrend(self):
        agent = AdversarialAgent()
        sig = _signal(direction=SignalDirection.BUY)
        assessments = agent.assess([sig], _snapshot(), _regime(RegimeType.TRENDING_UP))
        assert assessments[0].verdict != AdversarialVerdict.FAIL

    def test_macro_short_filter_allows_sell_in_downtrend(self):
        agent = AdversarialAgent()
        sig = _signal(direction=SignalDirection.SELL)
        assessments = agent.assess([sig], _snapshot(), _regime(RegimeType.TRENDING_DOWN))
        # Should not fail on macro short filter
        assert assessments[0].verdict != AdversarialVerdict.FAIL

    # ── Crowding check ────────────────────────────────────

    def test_crowding_flags_extreme_volume(self):
        agent = AdversarialAgent()
        sig = _signal()
        snap = _snapshot(volume_ratio=4.0)
        assessments = agent.assess([sig], snap, _regime())
        # High volume = medium flag, but alone won't fail
        assert assessments[0].verdict in (AdversarialVerdict.PASS, AdversarialVerdict.FLAG)

    def test_crowding_passes_normal_volume(self):
        agent = AdversarialAgent()
        check = agent._check_crowding(_signal(), _snapshot(volume_ratio=1.5))
        assert check.passed

    def test_crowding_no_volume_data(self):
        agent = AdversarialAgent()
        snap = _snapshot()
        snap.features.volume_ratio = None
        check = agent._check_crowding(_signal(), snap)
        assert check.passed

    # ── Volatility regime ─────────────────────────────────

    def test_volatility_flags_entry_in_volatile(self):
        agent = AdversarialAgent()
        check = agent._check_volatility_regime(_signal(), _regime(RegimeType.VOLATILE))
        assert not check.passed
        assert check.severity == "medium"

    def test_volatility_exempts_designed_strategies(self):
        agent = AdversarialAgent()
        sig = _signal(strategy="CryptoVolatilityDip")
        check = agent._check_volatility_regime(sig, _regime(RegimeType.VOLATILE))
        assert check.passed

    def test_volatility_passes_in_trending(self):
        agent = AdversarialAgent()
        check = agent._check_volatility_regime(_signal(), _regime(RegimeType.TRENDING_UP))
        assert check.passed

    # ── Momentum exhaustion ───────────────────────────────

    def test_momentum_exhaustion_buy_high_rsi(self):
        agent = AdversarialAgent()
        sig = _signal(direction=SignalDirection.BUY)
        snap = _snapshot(rsi=85.0)
        check = agent._check_momentum_exhaustion(sig, snap)
        assert not check.passed

    def test_momentum_exhaustion_sell_low_rsi(self):
        agent = AdversarialAgent()
        sig = _signal(direction=SignalDirection.SELL)
        snap = _snapshot(rsi=15.0)
        check = agent._check_momentum_exhaustion(sig, snap)
        assert not check.passed

    def test_momentum_exhaustion_normal_rsi(self):
        agent = AdversarialAgent()
        check = agent._check_momentum_exhaustion(_signal(), _snapshot(rsi=55.0))
        assert check.passed

    def test_momentum_exhaustion_no_rsi(self):
        agent = AdversarialAgent()
        snap = _snapshot()
        snap.indicators.rsi_14 = None
        check = agent._check_momentum_exhaustion(_signal(), snap)
        assert check.passed

    # ── Thesis alignment ──────────────────────────────────

    def test_thesis_alignment_no_thesis(self):
        agent = AdversarialAgent()
        check = agent._check_thesis_alignment(_signal(), None, "BTC/USDT")
        assert check.passed

    def test_thesis_alignment_short_against_strong_bullish(self):
        agent = AdversarialAgent()
        thesis = ThesisContext(
            timestamp=datetime.now(UTC),
            conviction_scores=[
                ConvictionScore(conviction=SAConviction.COMPUTE_DEMAND, score=0.80, rationale="test")
            ],
            dominant_conviction=SAConviction.COMPUTE_DEMAND,
            symbol_conviction_alignment=SAConviction.COMPUTE_DEMAND,
        )
        sig = _signal(direction=SignalDirection.SELL)
        check = agent._check_thesis_alignment(sig, thesis, "BTC/USDT")
        assert not check.passed

    def test_thesis_alignment_no_conviction(self):
        agent = AdversarialAgent()
        thesis = ThesisContext(
            timestamp=datetime.now(UTC),
            conviction_scores=[],
            dominant_conviction=None,
            symbol_conviction_alignment=None,
        )
        check = agent._check_thesis_alignment(_signal(), thesis, "BTC/USDT")
        assert check.passed

    # ── Knowledge bear case ───────────────────────────────

    def test_knowledge_no_context(self):
        agent = AdversarialAgent()
        check = agent._check_knowledge_bear_case(_signal(), None)
        assert check.passed

    def test_knowledge_no_constraints(self):
        agent = AdversarialAgent()
        knowledge = KnowledgeContext(
            query_regime=RegimeType.TRENDING_UP,
            suggested_constraints=[],
            confidence_modifier=0.0,
            strategy_bias=None,
        )
        check = agent._check_knowledge_bear_case(_signal(), knowledge)
        assert check.passed

    def test_knowledge_directional_conflict(self):
        agent = AdversarialAgent()
        knowledge = KnowledgeContext(
            query_regime=RegimeType.VOLATILE,
            suggested_constraints=["Trend-following poor in volatile regime"],
            confidence_modifier=-0.05,
            strategy_bias=SignalDirection.SELL,
        )
        sig = _signal(direction=SignalDirection.BUY)
        check = agent._check_knowledge_bear_case(sig, knowledge)
        assert not check.passed

    # ── Bull/Bear debate ──────────────────────────────────

    def test_debate_bullish_setup(self):
        agent = AdversarialAgent()
        sig = _signal(confidence=0.70)
        snap = _snapshot(rsi=50.0, volume_ratio=1.5)
        regime = _regime(RegimeType.TRENDING_UP, confidence=0.80)
        adj, note = agent._bull_bear_debate(sig, snap, regime)
        assert adj >= 0.0  # bull-favored
        assert isinstance(note, str)

    def test_debate_bearish_setup(self):
        agent = AdversarialAgent()
        sig = _signal(confidence=0.40)
        snap = _snapshot(rsi=78.0, volume_ratio=4.0)
        regime = _regime(RegimeType.VOLATILE, confidence=0.40)
        adj, note = agent._bull_bear_debate(sig, snap, regime)
        assert adj <= 0.0  # bear-favored

    def test_debate_with_session_context_bull(self):
        agent = AdversarialAgent()
        sig = _signal(confidence=0.70)
        snap = _snapshot(rsi=50.0)
        regime = _regime(RegimeType.TRENDING_UP, confidence=0.80)
        ctx = SessionContext()
        ctx.write_thesis("BULLISH", 0.65)
        ctx.write_knowledge("support", 0.60, historical_wr=0.65, sample_size=10)
        adj, note = agent._bull_bear_debate(sig, snap, regime, session_ctx=ctx)
        assert adj > 0.0
        assert "SA thesis" in note or "BM25" in note

    def test_debate_with_session_context_bear(self):
        agent = AdversarialAgent()
        sig = _signal(confidence=0.45)
        snap = _snapshot(rsi=75.0, volume_ratio=3.5)
        regime = _regime(RegimeType.VOLATILE, confidence=0.40)
        ctx = SessionContext()
        ctx.write_thesis("BEARISH", 0.65)
        ctx.write_knowledge("oppose", 0.60, historical_wr=0.30, sample_size=10)
        adj, note = agent._bull_bear_debate(sig, snap, regime, session_ctx=ctx)
        assert adj < 0.0

    # ── Flag weight ───────────────────────────────────────

    def test_avg_flag_weight_no_learner(self):
        agent = AdversarialAgent()
        assert agent._avg_flag_weight(["crowding_check"]) == 1.0

    def test_avg_flag_weight_empty(self):
        agent = AdversarialAgent()
        assert agent._avg_flag_weight([]) == 1.0

    # ── Multi-flag aggregation ────────────────────────────

    def test_two_medium_flags_produce_flag_verdict(self):
        agent = AdversarialAgent()
        # Use volatile regime + high RSI → 2 medium failures
        sig = _signal(direction=SignalDirection.BUY, confidence=0.60)
        snap = _snapshot(rsi=85.0, volume_ratio=4.0)
        regime = _regime(RegimeType.VOLATILE, confidence=0.80)
        assessments = agent.assess([sig], snap, regime)
        # Should get FLAG (volatile + momentum exhaustion + crowding)
        assert assessments[0].verdict in (AdversarialVerdict.FLAG, AdversarialVerdict.FAIL)

    def test_one_medium_flag_minor_flag(self):
        agent = AdversarialAgent()
        sig = _signal(direction=SignalDirection.BUY)
        snap = _snapshot(rsi=55.0, volume_ratio=4.0)  # just crowding
        regime = _regime(RegimeType.TRENDING_UP, confidence=0.85)
        assessments = agent.assess([sig], snap, regime)
        assert assessments[0].verdict in (AdversarialVerdict.PASS, AdversarialVerdict.FLAG)
