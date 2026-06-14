"""Tests for Raya CEO Agent — the company orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from agents.raya import (
    AgentRole,
    CompanyPhase,
    DailyPlan,
    DailyReport,
    Message,
    RayaCEO,
)

# ─────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────


def _make_mock_portfolio(equity=100_000.0, positions=None, closed_trades=None):
    """Build a mock portfolio."""
    port = MagicMock()
    port.net_equity = equity
    port.peak_equity = equity
    port.cash = equity
    port.positions = positions or {}
    port.closed_trades = closed_trades or []
    return port


def _make_mock_orchestrator(equity=100_000.0):
    """Build a mock orchestrator with all agents."""
    orch = MagicMock()
    orch.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    orch.executor.portfolio = _make_mock_portfolio(equity)
    orch.executor._positions = {}

    # Data agent
    orch.data_agent.fetch_ohlcv.return_value = None
    orch.data_agent.compute_indicators.return_value = None

    # Market analyst
    regime = MagicMock()
    regime.regime.value = "trending_up"
    regime.confidence = 0.75
    orch.market_analyst.analyse.return_value = regime

    # Alt data
    alt_ctx = MagicMock()
    alt_ctx.data_quality_ok = True
    alt_ctx.signals = []
    orch.alt_data_agent.get_context.return_value = alt_ctx

    # Thesis
    thesis = MagicMock()
    thesis.dominant_conviction = MagicMock()
    thesis.dominant_conviction.value = "MOMENTUM"
    thesis.overall_thesis_strength = 0.7
    thesis.advisory_note = "Follow trend"
    orch.thesis_agent.score.return_value = thesis

    # Knowledge
    knowledge = MagicMock()
    knowledge.suggested_constraints = ["Use momentum filters"]
    knowledge.confidence_modifier = 0.05
    orch.knowledge_agent.query.return_value = knowledge

    # Learner
    orch.learner.get_learning_summary.return_value = {
        "total_trades_observed": 50,
        "strategy_insights": [
            {"strategy": "MomentumBreakout", "regime": "trending_up", "win_rate": 0.56, "trades": 30}
        ],
        "symbol_insights": [{"symbol": "BTC/USDT", "trades": 20}],
    }
    orch.learner.check_recovery_suppression.return_value = []

    # Shadow logger
    orch.shadow = MagicMock()

    # Last good snapshots
    orch._last_good_snapshot = {}

    return orch


@pytest.fixture
def mock_orch():
    return _make_mock_orchestrator()


@pytest.fixture
def raya(mock_orch, tmp_path):
    return RayaCEO(orchestrator=mock_orch, log_dir=str(tmp_path / "raya_logs"))


# ─────────────────────────────────────────────────────────────
#  TESTS: Basic initialization
# ─────────────────────────────────────────────────────────────


class TestRayaInit:
    def test_init(self, raya):
        assert raya._orch is not None
        assert raya._today is None
        assert raya._plan is None
        assert raya._scrum_done is False
        assert raya._review_done is False

    def test_phase_property(self, raya):
        phase = raya.phase
        assert isinstance(phase, CompanyPhase)
        assert phase.value in [
            "pre_market",
            "morning_scrum",
            "trading",
            "evening_review",
            "after_hours",
        ]

    def test_today_str(self, raya):
        today = raya.today_str
        assert len(today) == 10  # YYYY-MM-DD
        assert "-" in today


# ─────────────────────────────────────────────────────────────
#  TESTS: Message bus
# ─────────────────────────────────────────────────────────────


class TestMessageBus:
    def test_say_creates_message(self, raya):
        msg = raya._say("TestSpeaker", "Hello world")
        assert isinstance(msg, Message)
        assert msg.speaker == "TestSpeaker"
        assert msg.content == "Hello world"
        assert len(raya.messages) == 1

    def test_say_with_data(self, raya):
        msg = raya._say("Agent", "Data here", data={"key": "value"})
        assert msg.data == {"key": "value"}

    def test_messages_accumulate(self, raya):
        raya._say("A", "msg1")
        raya._say("B", "msg2")
        raya._say("C", "msg3")
        assert len(raya.messages) == 3


# ─────────────────────────────────────────────────────────────
#  TESTS: New day
# ─────────────────────────────────────────────────────────────


class TestNewDay:
    def test_new_day_resets_state(self, raya):
        raya._today = "2024-01-01"
        raya._scrum_done = True
        raya._review_done = True
        raya._messages = [Message(timestamp=datetime.now(UTC), speaker="test", phase="test", content="test")]

        raya._new_day()  # today is different from 2024-01-01
        assert raya._scrum_done is False
        assert raya._review_done is False

    def test_new_day_idempotent(self, raya):
        raya._new_day()
        msgs_count = len(raya.messages)
        raya._new_day()  # same day, should not reset
        assert len(raya.messages) == msgs_count


# ─────────────────────────────────────────────────────────────
#  TESTS: Morning Scrum
# ─────────────────────────────────────────────────────────────


class TestMorningScrum:
    def test_morning_scrum_returns_plan(self, raya):
        plan = raya.run_morning_scrum()
        assert isinstance(plan, DailyPlan)
        assert plan.date == raya.today_str
        assert plan.risk_stance in ["conservative", "normal", "aggressive"]
        assert isinstance(plan.active_strategies, list)

    def test_morning_scrum_sets_flags(self, raya):
        raya.run_morning_scrum()
        assert raya._scrum_done is True
        assert raya.plan is not None

    def test_morning_scrum_logs_messages(self, raya):
        raya.run_morning_scrum()
        speakers = {m.speaker for m in raya.messages}
        assert AgentRole.CEO in speakers
        assert AgentRole.DATA in speakers
        assert AgentRole.MARKET_ANALYST in speakers

    def test_morning_scrum_saves_plan_file(self, raya, tmp_path):
        raya.run_morning_scrum()
        plan_files = list((tmp_path / "raya_logs").glob("*_plan.json"))
        assert len(plan_files) == 1


# ─────────────────────────────────────────────────────────────
#  TESTS: Trading Session
# ─────────────────────────────────────────────────────────────


class TestTradingSession:
    def test_trading_session_runs_pipeline(self, raya, mock_orch):
        result = raya.run_trading_session()
        assert "equity" in result
        assert "positions" in result
        mock_orch.run_bar.assert_called_once()

    def test_trading_session_auto_runs_scrum(self, raya):
        assert raya._scrum_done is False
        raya.run_trading_session()
        assert raya._scrum_done is True  # auto-ran scrum


# ─────────────────────────────────────────────────────────────
#  TESTS: Evening Review
# ─────────────────────────────────────────────────────────────


class TestEveningReview:
    def test_evening_review_returns_report(self, raya):
        report = raya.run_evening_review()
        assert isinstance(report, DailyReport)
        assert report.date == raya.today_str
        assert report.equity > 0

    def test_evening_review_sets_flags(self, raya):
        raya.run_evening_review()
        assert raya._review_done is True
        assert raya.report is not None

    def test_evening_review_logs_messages(self, raya):
        raya.run_evening_review()
        speakers = {m.speaker for m in raya.messages}
        assert AgentRole.CEO in speakers
        assert AgentRole.REFLECTION in speakers
        assert AgentRole.RISK_MANAGER in speakers

    def test_evening_review_saves_report_file(self, raya, tmp_path):
        raya.run_evening_review()
        report_files = list((tmp_path / "raya_logs").glob("*_report.json"))
        assert len(report_files) == 1

    def test_evening_review_accumulates_history(self, raya):
        raya.run_evening_review()
        assert len(raya.daily_reports) == 1


# ─────────────────────────────────────────────────────────────
#  TESTS: Full Day
# ─────────────────────────────────────────────────────────────


class TestFullDay:
    def test_full_day_returns_report(self, raya, mock_orch):
        report = raya.run_full_day()
        assert isinstance(report, DailyReport)
        mock_orch.run_bar.assert_called_once()

    def test_full_day_runs_all_phases(self, raya):
        raya.run_full_day()
        {m.phase for m in raya.messages}
        # Should have messages from various phases
        assert len(raya.messages) > 10


# ─────────────────────────────────────────────────────────────
#  TESTS: Status
# ─────────────────────────────────────────────────────────────


class TestStatus:
    def test_get_status_structure(self, raya):
        status = raya.get_status()
        assert "phase" in status
        assert "equity" in status
        assert "drawdown_pct" in status
        assert "open_positions" in status
        assert "total_trades" in status
        assert "win_rate" in status
        assert "recent_messages" in status

    def test_get_status_after_scrum(self, raya):
        raya.run_morning_scrum()
        status = raya.get_status()
        assert status["scrum_done"] is True
        assert status["plan"] is not None

    def test_get_status_messages_limited(self, raya):
        for i in range(30):
            raya._say("Agent", f"Message {i}")
        status = raya.get_status()
        assert len(status["recent_messages"]) <= 20


# ─────────────────────────────────────────────────────────────
#  TESTS: Logging / Persistence
# ─────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_daily_log(self, raya, tmp_path):
        raya._say("A", "test message")
        raya._save_daily_log("2024-01-01")
        log_file = tmp_path / "raya_logs" / "2024-01-01.jsonl"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message" in content

    def test_save_plan(self, raya, tmp_path):
        plan = DailyPlan(date="2024-01-01")
        raya._save_plan(plan)
        plan_file = tmp_path / "raya_logs" / "2024-01-01_plan.json"
        assert plan_file.exists()

    def test_save_report(self, raya, tmp_path):
        report = DailyReport(date="2024-01-01", equity=100_000)
        raya._save_report(report)
        report_file = tmp_path / "raya_logs" / "2024-01-01_report.json"
        assert report_file.exists()


# ─────────────────────────────────────────────────────────────
#  TESTS: Edge Cases
# ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_scrum_with_no_data(self, raya, mock_orch):
        """All data fetches fail — scrum should still complete."""
        mock_orch.data_agent.fetch_ohlcv.return_value = None
        plan = raya.run_morning_scrum()
        assert isinstance(plan, DailyPlan)

    def test_review_with_no_trades(self, raya, mock_orch):
        """No trades closed — review should complete cleanly."""
        mock_orch.executor.portfolio.closed_trades = []
        report = raya.run_evening_review()
        assert report.trades_closed == 0
        assert report.win_rate is None

    def test_review_with_drawdown(self, raya, mock_orch):
        """High drawdown should trigger improvement suggestion."""
        mock_orch.executor.portfolio.peak_equity = 200_000.0
        mock_orch.executor.portfolio.net_equity = 150_000.0
        report = raya.run_evening_review()
        # 25% drawdown should trigger advice
        improvements = [i for i in report.improvements if "drawdown" in i.lower()]
        assert len(improvements) > 0

    def test_scrum_with_alt_data_failure(self, raya, mock_orch):
        """Alt data fails — should continue without crash."""
        mock_orch.alt_data_agent.get_context.side_effect = Exception("API down")
        plan = raya.run_morning_scrum()
        assert isinstance(plan, DailyPlan)

    def test_scrum_with_thesis_failure(self, raya, mock_orch):
        """Thesis agent fails — should continue."""
        mock_orch.thesis_agent.score.side_effect = Exception("Error")
        plan = raya.run_morning_scrum()
        assert isinstance(plan, DailyPlan)

    def test_scrum_with_knowledge_failure(self, raya, mock_orch):
        """Knowledge agent fails — should continue."""
        mock_orch.knowledge_agent.query.side_effect = Exception("Error")
        plan = raya.run_morning_scrum()
        assert isinstance(plan, DailyPlan)


# ─────────────────────────────────────────────────────────────
#  TESTS: Enums / Models
# ─────────────────────────────────────────────────────────────


class TestModels:
    def test_company_phase_values(self):
        assert CompanyPhase.PRE_MARKET.value == "pre_market"
        assert CompanyPhase.MORNING_SCRUM.value == "morning_scrum"
        assert CompanyPhase.TRADING.value == "trading"
        assert CompanyPhase.EVENING_REVIEW.value == "evening_review"
        assert CompanyPhase.AFTER_HOURS.value == "after_hours"

    def test_agent_role_values(self):
        assert AgentRole.CEO.value == "Raya (CEO)"
        assert AgentRole.DATA.value == "Data Agent"

    def test_daily_plan_defaults(self):
        plan = DailyPlan(date="2024-01-01")
        assert plan.conviction == "NEUTRAL"
        assert plan.risk_stance == "normal"
        assert plan.max_new_positions == 5

    def test_daily_report_defaults(self):
        report = DailyReport(date="2024-01-01")
        assert report.equity == 0.0
        assert report.win_rate is None
        assert report.improvements == []

    def test_message_creation(self):
        msg = Message(
            timestamp=datetime.now(UTC),
            speaker="test",
            phase="trading",
            content="hello",
        )
        assert msg.data is None
