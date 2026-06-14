"""
KA-MATS · Adversarial Agent
Iknir Capital — Phase I (v2.0 Blueprint)

ROLE: Devil's Advocate & Signal Stress Testing

Every candidate signal MUST survive adversarial scrutiny before reaching
the Risk Manager. This agent actively tries to find reasons to kill trades.

Decision outputs:
  PASS — Signal survives all checks → proceeds to Risk Manager unchanged
  FLAG — Concerns found → proceeds with confidence penalty applied
  FAIL — Fatal issue detected → dropped, never reaches Risk Manager

Adversarial checks (Phase I — 5 checks per signal):
  1. thesis_alignment      — Does signal direction conflict with dominant SA conviction?
  2. crowding_check        — Is volume ratio extreme? (potential front-running)
  3. volatility_regime     — Is entering a VOLATILE regime position prudent?
  4. momentum_exhaustion   — RSI at extremes = exhaustion risk
  5. knowledge_bear_case   — Does the knowledge base flag counter-evidence?

Verdict aggregation rules:
  - Any HIGH severity failure  → FAIL
  - 2+ medium failures         → FLAG (full penalty)
  - 1 medium failure           → FLAG (half penalty)
  - All passed                 → PASS

Phase II additions (implemented):
  - Bull/Bear mini-debate: structured rule-based debate → confidence adj ±0.15
Phase II additions (pending):
  - Crowded position detection (institutional filing lag proxy)
  - Cross-asset correlation stress test
  - Earnings / catalyst calendar check
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from core.adaptive_learner import AdaptiveLearner
    from core.session_memory import SessionContext

from core.models import (
    AdversarialCheck,
    AdversarialVerdict,
    AltDataContext,
    CandidateSignal,
    KnowledgeContext,
    MarketSnapshot,
    RegimeAnalysis,
    RegimeType,
    SignalAssessment,
    SignalDirection,
    ThesisContext,
)


class AdversarialAgent:
    """
    Adversarial Agent — stress-tests every candidate signal before Risk Manager.

    Architecture note:
      This agent deliberately tries to KILL trades.
      A signal that survives adversarial scrutiny has higher quality.
      FAIL verdicts are logged and never forwarded to the Risk Manager.
      FLAG verdicts pass with a confidence penalty applied to the signal.

    Safety: Adversarial Agent can only veto or flag — it cannot approve.
            The Risk Manager's absolute veto authority is unchanged.
    """

    # Strategies specifically designed for VOLATILE regime — exempt from volatility penalty
    VOLATILE_DESIGNED_STRATEGIES: frozenset = frozenset({"CryptoVolatilityDip", "BTCDominanceRotation"})

    # Volume ratio above this = potential crowding / momentum exhaustion
    CROWDING_THRESHOLD: float = 3.0

    # RSI levels at which momentum exhaustion is flagged
    RSI_EXTREME_HIGH: float = 80.0
    RSI_EXTREME_LOW: float = 20.0

    # Confidence penalties (original validated values)
    FULL_FLAG_PENALTY: float = -0.10
    HALF_FLAG_PENALTY: float = -0.05

    # SA conviction strength above which shorting an aligned symbol is flagged
    SA_ALIGNMENT_VETO_THRESHOLD: float = 0.60

    # Bull/Bear debate confidence adjustment cap
    DEBATE_MAX_ADJ: float = 0.15

    def __init__(self, learner: AdaptiveLearner = None) -> None:
        self._learner = learner
        logger.info("[AdversarialAgent] Initialized — signal stress-testing active")

    # ─────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────

    def assess(
        self,
        signals: list[CandidateSignal],
        snapshot: MarketSnapshot,
        regime: RegimeAnalysis,
        thesis: ThesisContext | None = None,
        knowledge: KnowledgeContext | None = None,
        alt_data: AltDataContext | None = None,
        session_ctx: SessionContext | None = None,
    ) -> list[SignalAssessment]:
        """
        Stress-test all candidate signals.

        Args:
            signals:   Candidate signals from Strategy Agent
            snapshot:  MarketSnapshot from Data Agent
            regime:    RegimeAnalysis from Market Analyst
            thesis:    ThesisContext from Thesis Agent (optional)
            knowledge: KnowledgeContext from Knowledge Agent (optional)
            alt_data:  AltDataContext from Alt Data Agent (optional)

        Returns:
            List[SignalAssessment] — one per signal, each PASS / FLAG / FAIL
        """
        assessments: list[SignalAssessment] = []

        for signal in signals:
            assessment = self._assess_signal(
                signal, snapshot, regime, thesis, knowledge, alt_data, session_ctx
            )
            assessments.append(assessment)

            emoji = {"pass": "✓", "flag": "⚑", "fail": "✗"}.get(assessment.verdict.value, "?")
            logger.info(
                f"[AdversarialAgent] {emoji} {signal.symbol} "
                f"{signal.direction.value} ({signal.strategy_name}) → "
                f"{assessment.verdict.value.upper()} | "
                f"{assessment.adversarial_note[:80]}"
            )

        passed = sum(1 for a in assessments if a.verdict == AdversarialVerdict.PASS)
        flagged = sum(1 for a in assessments if a.verdict == AdversarialVerdict.FLAG)
        failed = sum(1 for a in assessments if a.verdict == AdversarialVerdict.FAIL)
        logger.info(f"[AdversarialAgent] Summary: {passed} PASS | {flagged} FLAG | {failed} FAIL")

        return assessments

    # ─────────────────────────────────────────────────────────
    #  SIGNAL-LEVEL ASSESSMENT
    # ─────────────────────────────────────────────────────────

    def _assess_signal(
        self,
        signal: CandidateSignal,
        snapshot: MarketSnapshot,
        regime: RegimeAnalysis,
        thesis: ThesisContext | None,
        knowledge: KnowledgeContext | None,
        alt_data: AltDataContext | None,
        session_ctx: SessionContext | None = None,
    ) -> SignalAssessment:
        checks: list[AdversarialCheck] = [
            self._check_macro_short_filter(signal, regime),
            self._check_thesis_alignment(signal, thesis, snapshot.symbol),
            self._check_crowding(signal, snapshot),
            self._check_volatility_regime(signal, regime),
            self._check_momentum_exhaustion(signal, snapshot),
            self._check_knowledge_bear_case(signal, knowledge),
        ]

        # ── Verdict aggregation ────────────────────────────
        high_failures = [c for c in checks if not c.passed and c.severity == "high"]
        medium_failures = [c for c in checks if not c.passed and c.severity == "medium"]

        if high_failures:
            verdict = AdversarialVerdict.FAIL
            confidence_adj = 0.0
            note = f"FAIL — {high_failures[0].name}: {high_failures[0].description}"
        elif len(medium_failures) >= 2:
            verdict = AdversarialVerdict.FLAG
            base = self.FULL_FLAG_PENALTY
            avg_weight = self._avg_flag_weight([c.name for c in medium_failures])
            confidence_adj = base * avg_weight
            reasons = ", ".join(c.name for c in medium_failures)
            note = f"FLAG — {len(medium_failures)} concerns ({reasons})"
        elif len(medium_failures) == 1:
            verdict = AdversarialVerdict.FLAG
            base = self.HALF_FLAG_PENALTY
            flag_weight = self._avg_flag_weight([medium_failures[0].name])
            confidence_adj = base * flag_weight
            note = f"FLAG (minor) — {medium_failures[0].name}: {medium_failures[0].description}"
        else:
            verdict = AdversarialVerdict.PASS
            confidence_adj = 0.0
            note = "All adversarial checks passed."

        # ── Bull/Bear mini-debate (Phase II) ──────────────
        # Only run debate for signals that survive the main checks
        # Debate produces an additional confidence adj ∈ [-0.15, +0.15]
        # v18: debate receives full session context for richer arguments
        if verdict != AdversarialVerdict.FAIL:
            debate_adj, debate_note = self._bull_bear_debate(
                signal, snapshot, regime, session_ctx=session_ctx
            )
            if abs(debate_adj) > 0.01:
                confidence_adj = max(-0.25, min(0.15, confidence_adj + debate_adj))
                note = f"{note} | Debate: {debate_note}"
                logger.debug(
                    f"[AdversarialAgent] Bull/Bear debate: {signal.symbol} "
                    f"adj={debate_adj:+.3f} → total_adj={confidence_adj:+.3f}"
                )

        return SignalAssessment(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            verdict=verdict,
            checks=checks,
            confidence_adjustment=round(confidence_adj, 3),
            adversarial_note=note,
        )

    # ─────────────────────────────────────────────────────────
    #  INDIVIDUAL CHECKS
    # ─────────────────────────────────────────────────────────

    def _check_macro_short_filter(
        self,
        signal: CandidateSignal,
        regime: RegimeAnalysis,
    ) -> AdversarialCheck:
        """
        Kill SELL signals when the macro regime is trending_up.
        Shorting into a bull trend fights the tape and the SA thesis.
        Historical data (2022-2025) showed SHORT trades at 27% win rate
        vs LONG trades at 51% win rate — systematic short bleed in up-trends.
        """
        if signal.direction == SignalDirection.SELL and regime.regime == RegimeType.TRENDING_UP:
            return AdversarialCheck(
                name="macro_short_filter",
                passed=False,
                description=(
                    f"SELL blocked: regime={regime.regime.value} "
                    f"(conf={regime.confidence:.2f}) -- shorting against bull trend"
                ),
                severity="high",
            )
        return AdversarialCheck(
            name="macro_short_filter",
            passed=True,
            description=f"Short filter OK: regime {regime.regime.value}",
            severity="low",
        )

    def _check_thesis_alignment(
        self,
        signal: CandidateSignal,
        thesis: ThesisContext | None,
        symbol: str,
    ) -> AdversarialCheck:
        """
        Flag shorting a symbol with a strong SA conviction behind it.
        The SA thesis is a structural long thesis for aligned instruments.
        """
        if thesis is None or thesis.symbol_conviction_alignment is None:
            return AdversarialCheck(
                name="thesis_alignment",
                passed=True,
                description="Symbol not in SA conviction map — neutral",
                severity="low",
            )

        is_short = signal.direction == SignalDirection.SELL
        aligned_conviction = thesis.symbol_conviction_alignment
        conviction_strength = next(
            (s.score for s in thesis.conviction_scores if s.conviction == aligned_conviction),
            0.0,
        )

        if is_short and conviction_strength >= self.SA_ALIGNMENT_VETO_THRESHOLD:
            return AdversarialCheck(
                name="thesis_alignment",
                passed=False,
                description=(
                    f"SHORT on {symbol} conflicts with strong SA conviction "
                    f"[{aligned_conviction.value}] (strength={conviction_strength:.2f})"
                ),
                severity="medium",
            )

        direction_label = "LONG" if signal.direction == SignalDirection.BUY else "SHORT"
        return AdversarialCheck(
            name="thesis_alignment",
            passed=True,
            description=(
                f"{direction_label} aligned with SA conviction "
                f"[{aligned_conviction.value}] (strength={conviction_strength:.2f})"
            ),
            severity="low",
        )

    def _check_crowding(
        self,
        signal: CandidateSignal,
        snapshot: MarketSnapshot,
    ) -> AdversarialCheck:
        """
        Detect potential crowded entries.
        Very high volume ratio suggests institutions already moved.
        Entering after a crowd means chasing, not leading.
        """
        vol_ratio = snapshot.features.volume_ratio
        if vol_ratio is None:
            return AdversarialCheck(
                name="crowding_check",
                passed=True,
                description="Volume ratio unavailable — check skipped",
                severity="low",
            )

        if vol_ratio > self.CROWDING_THRESHOLD:
            return AdversarialCheck(
                name="crowding_check",
                passed=False,
                description=(
                    f"Volume ratio {vol_ratio:.1f}x exceeds {self.CROWDING_THRESHOLD}x "
                    f"— potential crowded entry or momentum exhaustion"
                ),
                severity="medium",
            )

        return AdversarialCheck(
            name="crowding_check",
            passed=True,
            description=f"Volume ratio {vol_ratio:.2f}x — within normal range",
            severity="low",
        )

    def _check_volatility_regime(
        self,
        signal: CandidateSignal,
        regime: RegimeAnalysis,
    ) -> AdversarialCheck:
        """
        Flag new position entries during a VOLATILE regime.
        High volatility widens spreads, increases slippage, and
        raises stop-hunting risk. New entries should be discouraged.

        EXEMPT: CryptoVolatilityDip and BTCDominanceRotation are specifically
        designed for the volatile regime — flagging them defeats their purpose.
        """
        if regime.regime == RegimeType.VOLATILE:
            if signal.strategy_name in self.VOLATILE_DESIGNED_STRATEGIES:
                return AdversarialCheck(
                    name="volatility_regime",
                    passed=True,
                    description=(
                        f"VOLATILE regime — {signal.strategy_name} is designed for this regime, "
                        f"volatility penalty waived"
                    ),
                    severity="low",
                )
            return AdversarialCheck(
                name="volatility_regime",
                passed=False,
                description=(
                    f"VOLATILE regime (conf={regime.confidence:.2f}) — "
                    f"new entries carry elevated slippage, gap, and stop-hunt risk"
                ),
                severity="medium",
            )

        return AdversarialCheck(
            name="volatility_regime",
            passed=True,
            description=f"Regime {regime.regime.value} — acceptable for new position entry",
            severity="low",
        )

    def _check_momentum_exhaustion(
        self,
        signal: CandidateSignal,
        snapshot: MarketSnapshot,
    ) -> AdversarialCheck:
        """
        Flag entries at RSI extremes.
        Buying into RSI > 80 or selling into RSI < 20 is momentum exhaustion risk.
        """
        rsi = snapshot.indicators.rsi_14
        if rsi is None:
            return AdversarialCheck(
                name="momentum_exhaustion",
                passed=True,
                description="RSI unavailable — check skipped",
                severity="low",
            )

        if signal.direction == SignalDirection.BUY and rsi > self.RSI_EXTREME_HIGH:
            return AdversarialCheck(
                name="momentum_exhaustion",
                passed=False,
                description=(
                    f"BUY with RSI={rsi:.1f} > {self.RSI_EXTREME_HIGH} — momentum may already be exhausted"
                ),
                severity="medium",
            )

        if signal.direction == SignalDirection.SELL and rsi < self.RSI_EXTREME_LOW:
            return AdversarialCheck(
                name="momentum_exhaustion",
                passed=False,
                description=(
                    f"SELL with RSI={rsi:.1f} < {self.RSI_EXTREME_LOW} — momentum may already be exhausted"
                ),
                severity="medium",
            )

        return AdversarialCheck(
            name="momentum_exhaustion",
            passed=True,
            description=f"RSI={rsi:.1f} — no exhaustion signal",
            severity="low",
        )

    def _check_knowledge_bear_case(
        self,
        signal: CandidateSignal,
        knowledge: KnowledgeContext | None,
    ) -> AdversarialCheck:
        """
        Check if the Knowledge Agent surfaced constraints that
        conflict with the signal direction.
        """
        if knowledge is None or not knowledge.suggested_constraints:
            return AdversarialCheck(
                name="knowledge_bear_case",
                passed=True,
                description="No knowledge constraints to evaluate",
                severity="low",
            )

        # Flag if knowledge has a directional bias opposite to the signal
        if (
            knowledge.strategy_bias is not None
            and signal.direction != knowledge.strategy_bias
            and signal.direction != SignalDirection.HOLD
        ):
            return AdversarialCheck(
                name="knowledge_bear_case",
                passed=False,
                description=(
                    f"Signal {signal.direction.value} conflicts with knowledge "
                    f"bias {knowledge.strategy_bias.value}: "
                    f"{knowledge.suggested_constraints[0]}"
                ),
                severity="medium",
            )

        return AdversarialCheck(
            name="knowledge_bear_case",
            passed=True,
            description=(
                f"{len(knowledge.suggested_constraints)} constraint(s) in knowledge base "
                f"— no directional conflict with signal"
            ),
            severity="low",
        )

    # ─────────────────────────────────────────────────────────
    #  BULL / BEAR MINI-DEBATE (Phase II)
    # ─────────────────────────────────────────────────────────

    def _bull_bear_debate(
        self,
        signal: CandidateSignal,
        snapshot: MarketSnapshot,
        regime: RegimeAnalysis,
        session_ctx: SessionContext | None = None,
    ) -> tuple[float, str]:
        """
        Rule-based Bull/Bear structured mini-debate for a candidate signal.

        v18: Enhanced with cross-agent session context (open-multi-agent pattern).
        Debate now synthesises findings from ALL upstream agents, not just
        the snapshot and regime passed in directly.

        Each side scores independent arguments ∈ [0, 1].
        Net score → confidence adjustment ∈ [-DEBATE_MAX_ADJ, +DEBATE_MAX_ADJ].

        Bull arguments (support the trade):
          + Regime aligns with signal direction           0.25
          + Regime confidence is high (≥0.75)             0.20
          + RSI in healthy zone (35-65)                   0.15
          + Signal confidence ≥ 0.65                      0.20
          + Volume ratio slightly elevated (1.2-2.5x)     0.10
          [v18 additions]
          + SA thesis explicitly agrees with direction     0.20
          + BM25 historical WR ≥ 60% for similar setup    0.15

        Bear arguments (oppose the trade):
          - Regime is VOLATILE (bad for entries)           0.30
          - RSI extended on entry side                     0.20
          - Signal confidence < 0.50                       0.15
          - Volume ratio extreme (> 3x) = crowding         0.15
          - Regime confidence < 0.50 = uncertain           0.10
          [v18 additions]
          - SA thesis explicitly contradicts direction     0.20
          - BM25 historical WR < 38% for similar setup    0.15

        Returns (adj, summary_note).
        """
        is_buy = signal.direction == SignalDirection.BUY
        rsi = snapshot.indicators.rsi_14
        vol_ratio = snapshot.features.volume_ratio if snapshot.features else None
        regime_type = regime.regime
        regime_conf = regime.confidence

        bull_score = 0.0
        bear_score = 0.0
        bull_args: list[str] = []
        bear_args: list[str] = []

        # ── Bull arguments ─────────────────────────────────
        # 1. Regime aligns with signal direction
        if is_buy and regime_type in (RegimeType.TRENDING_UP, RegimeType.MEAN_REVERTING, RegimeType.RANGING):
            bull_score += 0.25
            bull_args.append(f"regime {regime_type.value} supports long")
        elif not is_buy and regime_type in (RegimeType.TRENDING_DOWN, RegimeType.VOLATILE):
            bull_score += 0.20
            bull_args.append(f"regime {regime_type.value} supports short")

        # 2. High regime confidence
        if regime_conf >= 0.75:
            bull_score += 0.20
            bull_args.append(f"regime conf={regime_conf:.2f} strong")

        # 3. RSI in healthy zone
        if rsi is not None:
            if is_buy and 35 <= rsi <= 65:
                bull_score += 0.15
                bull_args.append(f"RSI={rsi:.0f} healthy BUY zone")
            elif not is_buy and 35 <= rsi <= 65:
                bull_score += 0.15
                bull_args.append(f"RSI={rsi:.0f} healthy SELL zone")

        # 4. High signal conviction
        if signal.confidence >= 0.65:
            bull_score += 0.20
            bull_args.append(f"signal conf={signal.confidence:.2f}")

        # 5. Moderate volume (healthy interest, not crowding)
        if vol_ratio is not None and 1.2 <= vol_ratio <= 2.5:
            bull_score += 0.10
            bull_args.append(f"vol_ratio={vol_ratio:.1f}x normal interest")

        # ── [v18] SA thesis alignment — from shared session memory ────────
        # Inspired by open-multi-agent cross-agent synthesis
        if session_ctx is not None and session_ctx.sa_conviction is not None:
            if session_ctx.sa_score >= 0.40:
                thesis_bullish = session_ctx.sa_conviction == "BULLISH"
                if is_buy and thesis_bullish:
                    bull_score += 0.20
                    bull_args.append(f"SA thesis BULLISH (score={session_ctx.sa_score:.2f}) confirms BUY")
                elif not is_buy and not thesis_bullish and session_ctx.sa_conviction == "BEARISH":
                    bull_score += 0.15
                    bull_args.append("SA thesis BEARISH confirms SELL")

        # ── [v18] BM25 historical WR — past similar setups won ────────────
        if (
            session_ctx is not None
            and session_ctx.historical_wr is not None
            and session_ctx.bm25_sample_size >= 5
        ) and session_ctx.historical_wr >= 0.60:
            bull_score += 0.15
            bull_args.append(
                f"BM25 WR={session_ctx.historical_wr:.0%} "
                f"(n={session_ctx.bm25_sample_size}) similar setups strong"
            )

        # ── Bear arguments ─────────────────────────────────
        # 1. Volatile regime (bad for all new entries)
        if regime_type == RegimeType.VOLATILE:
            bear_score += 0.30
            bear_args.append("VOLATILE regime raises gap/slippage risk")

        # 2. RSI extended
        if rsi is not None:
            if is_buy and rsi > 72:
                bear_score += 0.20
                bear_args.append(f"RSI={rsi:.0f} extended for BUY")
            elif not is_buy and rsi < 28:
                bear_score += 0.20
                bear_args.append(f"RSI={rsi:.0f} extended for SELL")

        # 3. Low signal conviction
        if signal.confidence < 0.50:
            bear_score += 0.15
            bear_args.append(f"low signal conf={signal.confidence:.2f}")

        # 4. Extreme volume ratio (likely crowding)
        if vol_ratio is not None and vol_ratio > 3.0:
            bear_score += 0.15
            bear_args.append(f"vol_ratio={vol_ratio:.1f}x extreme crowding risk")

        # 5. Low regime confidence (uncertain context)
        if regime_conf < 0.50:
            bear_score += 0.10
            bear_args.append(f"regime conf={regime_conf:.2f} uncertain")

        # ── [v18] SA thesis contradiction — from shared session memory ─────
        if session_ctx is not None and session_ctx.sa_conviction is not None:
            if session_ctx.sa_score >= 0.40:
                thesis_bullish = session_ctx.sa_conviction == "BULLISH"
                if is_buy and not thesis_bullish and session_ctx.sa_conviction == "BEARISH":
                    bear_score += 0.20
                    bear_args.append(f"SA thesis BEARISH (score={session_ctx.sa_score:.2f}) contradicts BUY")
                elif not is_buy and thesis_bullish:
                    bear_score += 0.15
                    bear_args.append("SA thesis BULLISH contradicts SELL")

        # ── [v18] BM25 historical WR — past similar setups lost ───────────
        if (
            session_ctx is not None
            and session_ctx.historical_wr is not None
            and session_ctx.bm25_sample_size >= 5
        ) and session_ctx.historical_wr < 0.38:
            bear_score += 0.15
            bear_args.append(
                f"BM25 WR={session_ctx.historical_wr:.0%} "
                f"(n={session_ctx.bm25_sample_size}) similar setups weak"
            )

        # ── Net score → adjustment ─────────────────────────
        # Net ∈ [-1, +1] → adj ∈ [-DEBATE_MAX_ADJ, +DEBATE_MAX_ADJ]
        max_bull = 1.25  # updated max with v18 additions (0.90 + 0.20 + 0.15)
        max_bear = 1.30  # updated max with v18 additions (0.90 + 0.20 + 0.15 + 0.05)
        norm_bull = min(1.0, bull_score / max_bull)
        norm_bear = min(1.0, bear_score / max_bear)
        net = norm_bull - norm_bear  # ∈ [-1, +1]
        adj = round(net * self.DEBATE_MAX_ADJ, 3)

        # Build summary note
        bull_str = "; ".join(bull_args) if bull_args else "no bull arguments"
        bear_str = "; ".join(bear_args) if bear_args else "no bear arguments"
        note = f"bull={norm_bull:.2f}({bull_str}) vs bear={norm_bear:.2f}({bear_str}) → {adj:+.3f}"

        return adj, note

    def _avg_flag_weight(self, flag_names: list[str]) -> float:
        """
        Phase II: Average learned penalty weight for a list of flag types.
        Uses AdaptiveLearner if available; defaults to 1.0 (no adjustment).
        """
        if not self._learner or not flag_names:
            return 1.0
        weights = [self._learner.flag_penalty_weight(name) for name in flag_names]
        avg = sum(weights) / len(weights)
        if abs(avg - 1.0) > 0.05:
            logger.debug(
                f"[AdversarialAgent] Adaptive flag weights {flag_names}: avg={avg:.3f} (vs baseline 1.0)"
            )
        return avg
