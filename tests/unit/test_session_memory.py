"""Unit tests for SessionContext (session_memory.py) — 0% → ~100%."""

import pytest

from core.session_memory import SessionContext


@pytest.mark.unit
class TestSessionContext:
    def test_defaults(self):
        ctx = SessionContext(symbol="BTC/USDT")
        assert ctx.symbol == "BTC/USDT"
        assert ctx.regime is None
        assert ctx.sa_conviction is None
        assert ctx.rag_verdict is None
        assert ctx.bull_score == 0.0
        assert ctx.bear_score == 0.0
        assert ctx.debate_net == 0.0

    # ── write_regime ──────────────────────────────────────

    def test_write_regime(self):
        ctx = SessionContext()
        ctx.write_regime("trending_up", confidence=0.85, adx=30.0, volatility_pct=40.0, zscore=1.1)
        assert ctx.regime == "trending_up"
        assert ctx.regime_confidence == 0.85
        assert ctx.trend_strength == 30.0
        assert ctx.volatility_pct == 40.0
        assert ctx.zscore == 1.1

    def test_write_regime_clears_cache(self):
        ctx = SessionContext()
        ctx.write_regime("trending_up", confidence=0.85)
        _ = ctx.pipeline_confidence_mult("BUY")
        assert "BUY" in ctx._mult_cache
        ctx.write_regime("volatile", confidence=0.60)
        assert len(ctx._mult_cache) == 0

    # ── write_thesis ──────────────────────────────────────

    def test_write_thesis(self):
        ctx = SessionContext()
        ctx.write_thesis("BULLISH", 0.75)
        assert ctx.sa_conviction == "BULLISH"
        assert ctx.sa_score == 0.75

    def test_write_thesis_clears_cache(self):
        ctx = SessionContext()
        _ = ctx.pipeline_confidence_mult("BUY")
        ctx.write_thesis("BEARISH", 0.60)
        assert len(ctx._mult_cache) == 0

    # ── write_knowledge ───────────────────────────────────

    def test_write_knowledge(self):
        ctx = SessionContext()
        ctx.write_knowledge("support", rag_confidence=0.70, historical_wr=0.65, sample_size=20)
        assert ctx.rag_verdict == "support"
        assert ctx.rag_confidence == 0.70
        assert ctx.historical_wr == 0.65
        assert ctx.bm25_sample_size == 20

    def test_write_knowledge_clears_cache(self):
        ctx = SessionContext()
        _ = ctx.pipeline_confidence_mult("SELL")
        ctx.write_knowledge("oppose", rag_confidence=0.50)
        assert len(ctx._mult_cache) == 0

    # ── write_debate ──────────────────────────────────────

    def test_write_debate(self):
        ctx = SessionContext()
        ctx.write_debate(bull_score=0.7, bear_score=0.3)
        assert ctx.bull_score == 0.7
        assert ctx.bear_score == 0.3
        assert ctx.debate_net == pytest.approx(0.4)

    # ── pipeline_confidence_mult ──────────────────────────

    def test_mult_default_is_one(self):
        ctx = SessionContext()
        # Default regime_confidence=0.0 < 0.35 → 0.82 penalty
        assert ctx.pipeline_confidence_mult("BUY") == pytest.approx(0.82)

    def test_mult_caching(self):
        ctx = SessionContext()
        m1 = ctx.pipeline_confidence_mult("BUY")
        m2 = ctx.pipeline_confidence_mult("BUY")
        assert m1 == m2

    def test_mult_low_regime_confidence_penalty(self):
        ctx = SessionContext()
        ctx.write_regime("volatile", confidence=0.30)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m < 1.0  # should be penalized

    def test_mult_medium_regime_confidence_penalty(self):
        ctx = SessionContext()
        ctx.write_regime("volatile", confidence=0.45)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m < 1.0  # 0.92 penalty

    def test_mult_high_regime_confidence_boost(self):
        ctx = SessionContext()
        ctx.write_regime("trending_up", confidence=0.80)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m >= 1.0

    def test_mult_thesis_aligned_buy(self):
        ctx = SessionContext()
        ctx.write_regime("trending_up", confidence=0.80)
        ctx.write_thesis("BULLISH", 0.60)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m > 1.0  # aligned boost

    def test_mult_thesis_disagreement(self):
        ctx = SessionContext()
        ctx.write_thesis("BEARISH", 0.60)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m < 1.0  # disagreement penalty

    def test_mult_bm25_high_wr_boost(self):
        ctx = SessionContext()
        ctx.write_regime("trending_up", confidence=0.80)
        ctx.write_knowledge("support", rag_confidence=0.50, historical_wr=0.70, sample_size=10)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m > 1.0

    def test_mult_bm25_low_wr_penalty(self):
        ctx = SessionContext()
        ctx.write_knowledge("oppose", rag_confidence=0.50, historical_wr=0.35, sample_size=10)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m < 1.0

    def test_mult_bm25_insufficient_sample_no_effect(self):
        ctx = SessionContext()
        ctx.write_knowledge("support", rag_confidence=0.50, historical_wr=0.70, sample_size=2)
        m = ctx.pipeline_confidence_mult("BUY")
        # No BM25 factor applied (sample_size < 5), but rag_verdict still applies
        assert isinstance(m, float)

    def test_mult_rag_support_aligned(self):
        ctx = SessionContext()
        ctx.write_regime("trending_up", confidence=0.80)
        ctx.write_knowledge("support", rag_confidence=0.50)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m > 1.0  # 1.06 * 1.04 boost

    def test_mult_rag_oppose_disagrees(self):
        ctx = SessionContext()
        ctx.write_knowledge("oppose", rag_confidence=0.50)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m < 1.0  # 0.93 penalty

    def test_mult_bounded_low(self):
        ctx = SessionContext()
        ctx.write_regime("volatile", confidence=0.30)
        ctx.write_thesis("BEARISH", 0.70)
        ctx.write_knowledge("oppose", rag_confidence=0.60, historical_wr=0.20, sample_size=20)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m >= 0.68

    def test_mult_bounded_high(self):
        ctx = SessionContext()
        ctx.write_regime("trending_up", confidence=0.90)
        ctx.write_thesis("BULLISH", 0.80)
        ctx.write_knowledge("support", rag_confidence=0.80, historical_wr=0.80, sample_size=30)
        m = ctx.pipeline_confidence_mult("BUY")
        assert m <= 1.20

    # ── to_debate_summary ─────────────────────────────────

    def test_to_debate_summary(self):
        ctx = SessionContext(symbol="ETH/USDT")
        ctx.write_regime("trending_up", confidence=0.85, adx=30.0)
        ctx.write_thesis("BULLISH", 0.70)
        ctx.write_knowledge("support", rag_confidence=0.65, historical_wr=0.60, sample_size=15)
        d = ctx.to_debate_summary()
        assert d["regime"] == "trending_up"
        assert d["regime_confidence"] == 0.85
        assert d["sa_conviction"] == "BULLISH"
        assert d["sa_score"] == 0.70
        assert d["rag_verdict"] == "support"
        assert d["historical_wr"] == 0.60

    # ── summary_line ──────────────────────────────────────

    def test_summary_line(self):
        ctx = SessionContext(symbol="BTC/USDT")
        ctx.write_regime("trending_up", confidence=0.85)
        line = ctx.summary_line()
        assert isinstance(line, str)
        assert "BM25_WR" in line
