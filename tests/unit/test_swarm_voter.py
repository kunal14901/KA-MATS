"""Unit tests for core/swarm_voter.py — SwarmVoter consensus gate."""

from __future__ import annotations

import pytest

from core.swarm_voter import (
    SWARM_QUORUM_THRESHOLD,
    AgentVote,
    SwarmDecision,
    SwarmVoter,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _vote(
    adv: str = "pass",
    llm_vetoed: bool = False,
    llm_enabled: bool = True,
    bayes_approved: bool = True,
    bayes_has_data: bool = True,
    regime: str = "trending_up",
    direction: str = "BUY",
    confidence: float = 0.75,
) -> SwarmDecision:
    voter = SwarmVoter()
    return voter.vote(
        adversarial_verdict=adv,
        llm_vetoed=llm_vetoed,
        llm_enabled=llm_enabled,
        bayes_approved=bayes_approved,
        bayes_has_data=bayes_has_data,
        regime=regime,
        signal_direction=direction,
        confidence=confidence,
    )


# ── SwarmVoter.vote — disabled mode ──────────────────────────────────────────


class TestSwarmVoterDisabled:
    def test_disabled_always_approves(self):
        voter = SwarmVoter(enabled=False)
        d = voter.vote("fail", True, True, False, True, "ranging", "BUY", 0.1)
        assert d.approved is True
        assert d.total_score == 5.0
        assert d.summary == "swarm_voting_disabled"
        assert d.votes == []

    def test_disabled_respects_quorum_param(self):
        voter = SwarmVoter(quorum=10.0, enabled=False)
        d = voter.vote("fail", True, True, False, True, "ranging", "BUY", 0.1)
        assert d.approved is True  # disabled ignores quorum


# ── Adversarial vote mapping ──────────────────────────────────────────────────


class TestAdversarialVote:
    def test_pass_gives_1(self):
        d = _vote(adv="pass")
        adv_vote = next(v for v in d.votes if v.agent == "AdversarialAgent")
        assert adv_vote.score == 1.0

    def test_flag_gives_half(self):
        d = _vote(adv="flag")
        adv_vote = next(v for v in d.votes if v.agent == "AdversarialAgent")
        assert adv_vote.score == 0.5

    def test_fail_gives_zero(self):
        d = _vote(adv="fail")
        adv_vote = next(v for v in d.votes if v.agent == "AdversarialAgent")
        assert adv_vote.score == 0.0

    def test_unknown_verdict_defaults_to_half(self):
        d = _vote(adv="UNKNOWN_VERDICT")
        adv_vote = next(v for v in d.votes if v.agent == "AdversarialAgent")
        assert adv_vote.score == 0.5

    def test_case_insensitive(self):
        d = _vote(adv="PASS")
        adv_vote = next(v for v in d.votes if v.agent == "AdversarialAgent")
        assert adv_vote.score == 1.0


# ── LLM Validator vote ────────────────────────────────────────────────────────


class TestLLMVote:
    def test_enabled_not_vetoed_approves(self):
        d = _vote(llm_enabled=True, llm_vetoed=False)
        llm = next(v for v in d.votes if v.agent == "LLMValidator")
        assert llm.score == 1.0
        assert "APPROVED" in llm.reason

    def test_enabled_vetoed_gives_zero(self):
        d = _vote(llm_enabled=True, llm_vetoed=True)
        llm = next(v for v in d.votes if v.agent == "LLMValidator")
        assert llm.score == 0.0
        assert "VETOED" in llm.reason

    def test_disabled_fail_open(self):
        d = _vote(llm_enabled=False, llm_vetoed=False)
        llm = next(v for v in d.votes if v.agent == "LLMValidator")
        assert llm.score == 0.75
        assert "DISABLED" in llm.reason


# ── Bayesian EV vote ──────────────────────────────────────────────────────────


class TestBayesVote:
    def test_approved_with_data(self):
        d = _vote(bayes_approved=True, bayes_has_data=True)
        bayes = next(v for v in d.votes if v.agent == "BayesianEVFilter")
        assert bayes.score == 1.0
        assert "EV_POSITIVE" in bayes.reason

    def test_rejected_with_data(self):
        d = _vote(bayes_approved=False, bayes_has_data=True)
        bayes = next(v for v in d.votes if v.agent == "BayesianEVFilter")
        assert bayes.score == 0.0
        assert "EV_NEGATIVE" in bayes.reason

    def test_insufficient_data_fail_open(self):
        d = _vote(bayes_has_data=False)
        bayes = next(v for v in d.votes if v.agent == "BayesianEVFilter")
        assert bayes.score == 0.75
        assert "INSUFFICIENT_DATA" in bayes.reason


# ── Regime Alignment vote ─────────────────────────────────────────────────────


class TestRegimeVote:
    def test_trending_up_buy_aligned(self):
        d = _vote(regime="trending_up", direction="BUY")
        reg = next(v for v in d.votes if v.agent == "RegimeAlignment")
        assert reg.score == 1.0
        assert "ALIGNED" in reg.reason

    def test_trending_down_sell_aligned(self):
        d = _vote(regime="trending_down", direction="SELL")
        reg = next(v for v in d.votes if v.agent == "RegimeAlignment")
        assert reg.score == 1.0

    def test_trending_up_sell_misaligned(self):
        d = _vote(regime="trending_up", direction="SELL")
        reg = next(v for v in d.votes if v.agent == "RegimeAlignment")
        assert reg.score == 0.2
        assert "MISALIGNED" in reg.reason

    def test_ranging_neutral(self):
        d = _vote(regime="ranging", direction="BUY")
        reg = next(v for v in d.votes if v.agent == "RegimeAlignment")
        assert reg.score == 0.6
        assert "NEUTRAL" in reg.reason

    def test_volatile_neutral(self):
        d = _vote(regime="volatile", direction="SELL")
        reg = next(v for v in d.votes if v.agent == "RegimeAlignment")
        assert reg.score == 0.6

    def test_unknown_regime_neutral(self):
        d = _vote(regime="unknown", direction="BUY")
        reg = next(v for v in d.votes if v.agent == "RegimeAlignment")
        assert reg.score == 0.6


# ── Confidence Gate vote ──────────────────────────────────────────────────────


class TestConfidenceVote:
    def test_high_confidence_capped_at_1(self):
        d = _vote(confidence=1.0)
        conf = next(v for v in d.votes if v.agent == "ConfidenceGate")
        assert conf.score == 1.0
        assert "OK" in conf.reason

    def test_exactly_threshold_gives_1(self):
        d = _vote(confidence=0.65)
        conf = next(v for v in d.votes if v.agent == "ConfidenceGate")
        assert conf.score == pytest.approx(1.0, rel=1e-3)

    def test_below_threshold_partial(self):
        d = _vote(confidence=0.325)  # 0.325 / 0.65 = 0.5
        conf = next(v for v in d.votes if v.agent == "ConfidenceGate")
        assert conf.score == pytest.approx(0.5, rel=1e-3)
        assert "WEAK" in conf.reason

    def test_zero_confidence_zero_score(self):
        d = _vote(confidence=0.0)
        conf = next(v for v in d.votes if v.agent == "ConfidenceGate")
        assert conf.score == 0.0


# ── Quorum / approval logic ───────────────────────────────────────────────────


class TestQuorum:
    def test_all_strong_votes_approve(self):
        d = _vote(
            adv="pass",
            llm_vetoed=False,
            llm_enabled=True,
            bayes_approved=True,
            bayes_has_data=True,
            regime="trending_up",
            direction="BUY",
            confidence=0.8,
        )
        assert d.approved is True
        assert d.total_score >= SWARM_QUORUM_THRESHOLD

    def test_all_weak_votes_reject(self):
        # adv=fail(0)+llm_vetoed(0)+bayes_rejected(0)+misaligned(0.2)+low_conf(0.1/0.65)
        d = _vote(
            adv="fail",
            llm_vetoed=True,
            llm_enabled=True,
            bayes_approved=False,
            bayes_has_data=True,
            regime="trending_up",
            direction="SELL",
            confidence=0.1,
        )
        assert d.approved is False

    def test_custom_quorum_higher_rejects_borderline(self):
        voter = SwarmVoter(quorum=4.5)
        # ranging neutral = 0.6; total ≈ 1.0+1.0+1.0+0.6+1.0 = 4.6 > 4.5 (still passes)
        voter.vote("pass", False, True, True, True, "ranging", "BUY", 0.7)
        # misaligned regime: 1.0+1.0+1.0+0.2+(0.3/0.65)≈0.46 = 3.66 < 4.5 → reject
        d2 = voter.vote("pass", False, True, True, True, "trending_down", "BUY", 0.3)
        assert d2.approved is False

    def test_vote_count_always_five(self):
        d = _vote()
        assert len(d.votes) == 5

    def test_total_score_matches_sum(self):
        d = _vote()
        assert d.total_score == pytest.approx(sum(v.score for v in d.votes), rel=1e-4)


# ── SwarmDecision.vote_str ────────────────────────────────────────────────────


class TestVoteStr:
    def test_vote_str_contains_approve(self):
        d = _vote()
        text = d.vote_str
        assert "APPROVE" in text or "REJECT" in text

    def test_vote_str_has_all_agent_names(self):
        d = _vote()
        for agent_name in (
            "AdversarialAgent",
            "LLMValidator",
            "BayesianEVFilter",
            "RegimeAlignment",
            "ConfidenceGate",
        ):
            assert agent_name in d.vote_str

    def test_vote_str_rejected_label(self):
        d = _vote(
            adv="fail",
            llm_vetoed=True,
            llm_enabled=True,
            bayes_approved=False,
            bayes_has_data=True,
            regime="trending_up",
            direction="SELL",
            confidence=0.1,
        )
        assert "REJECT" in d.vote_str


# ── AgentVote dataclass ───────────────────────────────────────────────────────


class TestAgentVote:
    def test_agent_vote_fields(self):
        v = AgentVote(agent="TestAgent", score=0.7, reason="TEST")
        assert v.agent == "TestAgent"
        assert v.score == 0.7
        assert v.reason == "TEST"
        assert v.raw is None

    def test_agent_vote_with_raw(self):
        v = AgentVote(agent="A", score=1.0, reason="OK", raw={"x": 1})
        assert v.raw == {"x": 1}


# ── SwarmDecision dataclass ───────────────────────────────────────────────────


class TestSwarmDecision:
    def test_approved_summary_contains_score(self):
        d = _vote()
        assert "score=" in d.summary
