"""
KA-MATS · Session Working Memory
Iknir Capital — Phase I (v18 Enhancement)

Inspired by the shared-memory pattern in open-multi-agent:
  https://github.com/open-multi-agent/open-multi-agent

Key insight: in a linear assembly-line pipeline each agent only sees the
output of the one directly upstream.  A shared SessionContext lets every
agent deposit its findings so *all* downstream agents can synthesize from
the full picture — turning a one-way relay race into genuine collaboration.

Usage:
    # Orchestrator creates one context per bar / symbol evaluation
    ctx = SessionContext(symbol="NVDA", timestamp=ts)

    # MarketAnalyst writes
    ctx.write_regime("trending_up", confidence=0.82, adx=34.1)

    # ThesisAgent writes
    ctx.write_thesis("BULLISH", score=0.71)

    # KnowledgeAgent writes
    ctx.write_knowledge("support", rag_confidence=0.60, historical_wr=0.62)

    # Strategy layer queries the pipeline multiplier
    mult = ctx.pipeline_confidence_mult("BUY")   # e.g. 1.12

    # Adversarial reads the full context for richer debate
    debate_ctx = ctx.to_debate_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionContext:
    """
    Ephemeral, per-bar working memory shared across all pipeline agents.

    Reset each bar — never persisted.  BM25 / FAISS handle long-term memory.

    Namespace convention (mirrors open-multi-agent namespaced store):
      MarketAnalyst  → regime_*, trend_*, vol_*
      ThesisAgent    → sa_*
      KnowledgeAgent → rag_*, historical_*
      StrategyAgent  → n_signals_*, strategies_*
      Adversarial    → bull_*, bear_*, debate_*
    """

    symbol: str = ""
    timestamp: Any | None = None

    # ── MarketAnalyst namespace ────────────────────────────────────────────────
    regime: str | None = None  # e.g. "trending_up"
    regime_confidence: float = 0.0
    trend_strength: float | None = None  # ADX value
    volatility_pct: float | None = None
    zscore: float | None = None

    # ── ThesisAgent namespace ──────────────────────────────────────────────────
    sa_conviction: str | None = None  # "BULLISH" | "BEARISH" | "NEUTRAL"
    sa_score: float = 0.0  # 0.0 – 1.0 strength

    # ── KnowledgeAgent / BM25 namespace ───────────────────────────────────────
    rag_verdict: str | None = None  # "support" | "neutral" | "oppose"
    rag_confidence: float = 0.0
    historical_wr: float | None = None  # WR from BM25 for similar setups
    bm25_sample_size: int = 0  # n trades used to compute WR

    # ── StrategyAgent namespace ────────────────────────────────────────────────
    n_signals_generated: int = 0
    strategies_fired: list = field(default_factory=list)

    # ── Adversarial namespace ──────────────────────────────────────────────────
    bull_score: float = 0.0
    bear_score: float = 0.0
    debate_net: float = 0.0  # bull_score − bear_score ∈ [−1, +1]

    # ── Derived (written once by orchestrator after all inputs are available) ──
    _mult_cache: dict = field(default_factory=dict, repr=False)

    # ─────────────────────────────────────────────────────────────────────────
    #  WRITE HELPERS  (agents call these)
    # ─────────────────────────────────────────────────────────────────────────

    def write_regime(
        self,
        regime: str,
        confidence: float,
        adx: float | None = None,
        volatility_pct: float | None = None,
        zscore: float | None = None,
    ) -> None:
        self.regime = regime
        self.regime_confidence = confidence
        self.trend_strength = adx
        self.volatility_pct = volatility_pct
        self.zscore = zscore
        self._mult_cache.clear()

    def write_thesis(self, conviction: str | None, score: float) -> None:
        self.sa_conviction = conviction
        self.sa_score = score
        self._mult_cache.clear()

    def write_knowledge(
        self,
        verdict: str | None,
        rag_confidence: float,
        historical_wr: float | None = None,
        sample_size: int = 0,
    ) -> None:
        self.rag_verdict = verdict
        self.rag_confidence = rag_confidence
        self.historical_wr = historical_wr
        self.bm25_sample_size = sample_size
        self._mult_cache.clear()

    def write_debate(self, bull_score: float, bear_score: float) -> None:
        self.bull_score = bull_score
        self.bear_score = bear_score
        self.debate_net = bull_score - bear_score

    # ─────────────────────────────────────────────────────────────────────────
    #  CROSS-AGENT CONFIDENCE MULTIPLIER
    # ─────────────────────────────────────────────────────────────────────────

    def pipeline_confidence_mult(self, signal_direction: str) -> float:
        """
        Synthesize a multiplicative confidence adjustment from all agent findings.

        Design (inspired by open-multi-agent coordinator synthesis):
          - High cross-agent agreement  → up to 1.20×
          - Low / conflicting signals   → down to 0.68×
          - Applied BEFORE adversarial  so adversarial still runs on adjusted confidence

        Factors (each bounded, then combined multiplicatively):
          1. Regime confidence  — low-conf regime penalises all signals
          2. SA thesis alignment — thesis agrees with signal direction → boost
          3. BM25 historical WR  — similar setups had bad WR → penalise
          4. RAG verdict alignment — knowledge base opposes the signal → penalise

        Returns: multiplier ∈ [0.68, 1.20]
        """
        cache_key = signal_direction
        if cache_key in self._mult_cache:
            return self._mult_cache[cache_key]

        mult = 1.0

        # ── Factor 1: Regime confidence ────────────────────────────────────
        # If we can't confidently classify the regime, signals are noisier.
        rc = self.regime_confidence
        if rc < 0.35:
            mult *= 0.82  # very uncertain regime → strong penalty
        elif rc < 0.50:
            mult *= 0.92  # uncertain
        elif rc > 0.75:
            mult *= 1.06  # high-confidence regime → small boost

        # ── Factor 2: SA thesis alignment ─────────────────────────────────
        # Thesis adds forward-looking conviction; disagreement is a red flag.
        if self.sa_conviction is not None and self.sa_score > 0.30:
            bullish_signal = signal_direction == "BUY"
            bullish_thesis = self.sa_conviction == "BULLISH"
            if bullish_signal == bullish_thesis:
                mult *= 1.08  # aligned
            elif self.sa_score > 0.55:
                mult *= 0.88  # strong disagreement

        # ── Factor 3: BM25 historical win-rate ────────────────────────────
        # Only penalise / boost when we have a statistically meaningful sample.
        if self.historical_wr is not None and self.bm25_sample_size >= 5:
            if self.historical_wr >= 0.65:
                mult *= 1.06  # historically strong setup
            elif self.historical_wr < 0.38:
                mult *= 0.90  # historically poor setup

        # ── Factor 4: RAG verdict alignment ───────────────────────────────
        if self.rag_verdict is not None and self.rag_confidence > 0.40:
            bullish_signal = signal_direction == "BUY"
            if (self.rag_verdict == "support" and bullish_signal) or (
                self.rag_verdict == "oppose" and not bullish_signal
            ):
                mult *= 1.04  # knowledge base agrees
            elif (self.rag_verdict == "oppose" and bullish_signal) or (
                self.rag_verdict == "support" and not bullish_signal
            ):
                mult *= 0.93  # knowledge base disagrees

        # Cap to avoid extreme values
        result = max(0.68, min(1.20, mult))
        self._mult_cache[cache_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  DEBATE CONTEXT EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def to_debate_summary(self) -> dict:
        """
        Export a rich context dict for the Adversarial Agent's bull/bear debate.
        Mirrors open-multi-agent's getSummary() shared-memory export.
        """
        return {
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "trend_strength": self.trend_strength,
            "volatility_pct": self.volatility_pct,
            "zscore": self.zscore,
            "sa_conviction": self.sa_conviction,
            "sa_score": self.sa_score,
            "rag_verdict": self.rag_verdict,
            "rag_confidence": self.rag_confidence,
            "historical_wr": self.historical_wr,
            "bm25_sample_size": self.bm25_sample_size,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────

    def summary_line(self) -> str:
        """One-line diagnostic string for logging."""
        wr_str = f"BM25_WR={self.historical_wr:.0%}" if self.historical_wr else "BM25_WR=?"
        return (
            f"SessionCtx[{self.symbol}] "
            f"regime={self.regime}({self.regime_confidence:.2f}) "
            f"sa={self.sa_conviction}({self.sa_score:.2f}) "
            f"rag={self.rag_verdict} "
            f"{wr_str}"
        )
