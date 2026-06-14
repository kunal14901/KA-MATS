"""
SwarmVoter — multi-agent consensus gate for KA-MATS.

Inspired by AutoHedge's swarm intelligence approach, but with a key difference:
KA-MATS agents produce deterministic, auditable votes (not probabilistic LLM opinions),
so the quorum result is reproducible and explainable.

Vote sources (5 agents):
  1. AdversarialAgent   — PASS=1.0  FLAG=0.5  FAIL=0.0
  2. LLMValidator       — not_vetoed=1.0  vetoed=0.0  (fail-open: 0.75 if disabled)
  3. BayesianEVFilter   — approved=1.0  rejected=0.0  (fail-open: 0.75 if insufficient data)
  4. RegimeAlignment    — trending_up+LONG or trending_down+SHORT = 1.0  else 0.5
  5. ConfidenceGate     — confidence >= 0.65 = 1.0  else confidence / 0.65

Quorum: weighted sum >= SWARM_QUORUM_THRESHOLD (default 3.0 / 5.0 = 60%).
Trades below quorum are logged and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SWARM_VOTING_ENABLED: bool = True  # master on/off
SWARM_QUORUM_THRESHOLD: float = 3.0  # votes needed (out of 5 max)
SWARM_LOG_ALL_VOTES: bool = True  # log every vote breakdown


@dataclass
class AgentVote:
    agent: str
    score: float  # 0.0 – 1.0
    reason: str
    raw: Any = field(default=None, repr=False)


@dataclass
class SwarmDecision:
    approved: bool
    total_score: float
    quorum: float
    votes: list[AgentVote]
    summary: str

    @property
    def vote_str(self) -> str:
        lines = [
            f"SwarmVoter [{self.total_score:.2f}/{self.quorum:.1f}] "
            f"{'APPROVE' if self.approved else 'REJECT'}"
        ]
        for v in self.votes:
            bar = "█" * int(v.score * 5) + "░" * (5 - int(v.score * 5))
            lines.append(f"  [{bar}] {v.score:.2f}  {v.agent:<22} {v.reason}")
        return "\n".join(lines)


class SwarmVoter:
    """
    Aggregates agent verdicts into a single APPROVE / REJECT decision.

    Usage (in orchestrator._process_symbol):
        decision = swarm_voter.vote(
            adversarial_verdict=assessment.verdict,
            llm_vetoed=veto_verdict.vetoed,
            llm_enabled=self.llm_validator.config.enabled,
            bayes_approved=bayes_result.approved,
            bayes_has_data=bayes_result.has_sufficient_data,
            regime=regime,
            signal_direction=sig.direction,
            confidence=sig.confidence,
        )
        if not decision.approved:
            logger.info(decision.vote_str)
            continue
    """

    def __init__(
        self,
        quorum: float = SWARM_QUORUM_THRESHOLD,
        enabled: bool = SWARM_VOTING_ENABLED,
    ) -> None:
        self.quorum = quorum
        self.enabled = enabled

    # ------------------------------------------------------------------
    def vote(
        self,
        adversarial_verdict: str,  # "pass" / "flag" / "fail"
        llm_vetoed: bool,
        llm_enabled: bool,
        bayes_approved: bool,
        bayes_has_data: bool,
        regime: str,  # "trending_up" / "trending_down" / "ranging" / ...
        signal_direction: str,  # "BUY" / "SELL"
        confidence: float,
    ) -> SwarmDecision:
        """Compute quorum from 5 agent votes and return an approve/reject decision."""

        if not self.enabled:
            return SwarmDecision(
                approved=True,
                total_score=5.0,
                quorum=self.quorum,
                votes=[],
                summary="swarm_voting_disabled",
            )

        votes: list[AgentVote] = []

        # ── Vote 1: Adversarial Agent ──────────────────────────────────
        adv_map = {"pass": 1.0, "flag": 0.5, "fail": 0.0}
        adv_score = adv_map.get(adversarial_verdict.lower(), 0.5)
        votes.append(
            AgentVote(
                agent="AdversarialAgent",
                score=adv_score,
                reason=adversarial_verdict.upper(),
            )
        )

        # ── Vote 2: LLM Validator ─────────────────────────────────────
        if not llm_enabled:
            llm_score = 0.75  # fail-open: slight positive bias (unknown = lean approve)
            llm_reason = "DISABLED (fail-open)"
        elif llm_vetoed:
            llm_score = 0.0
            llm_reason = "VETOED"
        else:
            llm_score = 1.0
            llm_reason = "APPROVED"
        votes.append(
            AgentVote(
                agent="LLMValidator",
                score=llm_score,
                reason=llm_reason,
            )
        )

        # ── Vote 3: Bayesian EV Filter ────────────────────────────────
        if not bayes_has_data:
            bayes_score = 0.75  # fail-open: insufficient data = lean approve
            bayes_reason = "INSUFFICIENT_DATA (fail-open)"
        elif bayes_approved:
            bayes_score = 1.0
            bayes_reason = "EV_POSITIVE"
        else:
            bayes_score = 0.0
            bayes_reason = "EV_NEGATIVE"
        votes.append(
            AgentVote(
                agent="BayesianEVFilter",
                score=bayes_score,
                reason=bayes_reason,
            )
        )

        # ── Vote 4: Regime Alignment ──────────────────────────────────
        aligned = (regime == "trending_up" and signal_direction == "BUY") or (
            regime == "trending_down" and signal_direction == "SELL"
        )
        neutral = regime in ("ranging", "volatile", "unknown")
        if aligned:
            regime_score = 1.0
            regime_reason = f"{regime}+{signal_direction} ALIGNED"
        elif neutral:
            regime_score = 0.6
            regime_reason = f"{regime} NEUTRAL"
        else:
            regime_score = 0.2
            regime_reason = f"{regime}+{signal_direction} MISALIGNED"
        votes.append(
            AgentVote(
                agent="RegimeAlignment",
                score=regime_score,
                reason=regime_reason,
            )
        )

        # ── Vote 5: Confidence Gate ───────────────────────────────────
        CONF_TARGET = 0.65
        conf_score = min(1.0, confidence / CONF_TARGET)
        conf_reason = f"conf={confidence:.3f} ({'OK' if confidence >= CONF_TARGET else 'WEAK'})"
        votes.append(
            AgentVote(
                agent="ConfidenceGate",
                score=conf_score,
                reason=conf_reason,
            )
        )

        # ── Quorum decision ───────────────────────────────────────────
        total = sum(v.score for v in votes)
        approved = total >= self.quorum

        summary = (
            f"quorum={'MET' if approved else 'FAILED'} "
            f"score={total:.2f}/{self.quorum:.1f} "
            f"adv={adversarial_verdict} conf={confidence:.3f}"
        )

        return SwarmDecision(
            approved=approved,
            total_score=round(total, 3),
            quorum=self.quorum,
            votes=votes,
            summary=summary,
        )
