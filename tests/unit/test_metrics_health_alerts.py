"""Unit tests for MetricsCollector + HealthMonitor + AlertManager — boost coverage."""

from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.alerts import Alert, AlertManager, AlertSeverity
from core.health import HealthCheck, HealthMonitor, HealthStatus
from core.metrics import MetricsCollector, get_metrics

# ═══════════════════════════════════════════════════════════
#  MetricsCollector
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMetricsCollector:
    def test_record_and_get_latest(self):
        m = MetricsCollector()
        m.record("equity", 10500.0)
        assert m.get_latest("equity") == 10500.0

    def test_get_latest_unknown(self):
        m = MetricsCollector()
        assert m.get_latest("nonexistent") is None

    def test_increment_counter(self):
        m = MetricsCollector()
        m.increment("trades_total")
        m.increment("trades_total", 5)
        assert m.get_counter("trades_total") == 6.0

    def test_get_counter_unknown(self):
        m = MetricsCollector()
        assert m.get_counter("nope") == 0.0

    def test_get_recent_values(self):
        m = MetricsCollector()
        for i in range(5):
            m.record("pnl", float(i * 100))
        recent = m.get_recent_values("pnl", last_n=3)
        assert len(recent) == 3
        assert recent[-1].value == 400.0

    def test_get_recent_values_unknown(self):
        m = MetricsCollector()
        assert m.get_recent_values("unknown") == []

    def test_record_trade_opened(self):
        m = MetricsCollector()
        m.record_trade_opened("BTC/USDT", "TestStrat", "trending_up", 0.1)
        assert m.get_counter("trades_total") == 1.0
        assert m.get_counter("trades_opened") == 1.0

    def test_record_trade_closed_win(self):
        m = MetricsCollector()
        m.record_trade_closed("BTC/USDT", "TestStrat", 500.0, "take_profit")
        assert m.get_counter("trades_won") == 1.0
        assert m.get_counter("trades_closed") == 1.0

    def test_record_trade_closed_loss(self):
        m = MetricsCollector()
        m.record_trade_closed("BTC/USDT", "TestStrat", -200.0, "stop_loss")
        assert m.get_counter("trades_lost") == 1.0

    def test_record_equity(self):
        m = MetricsCollector()
        m.record_equity(15000.0, 5.0)
        assert m.get_latest("equity") == 15000.0
        assert m.get_latest("drawdown_pct") == 5.0

    def test_record_open_positions(self):
        m = MetricsCollector()
        m.record_open_positions(3, 45.0)
        assert m.get_latest("open_positions") == 3.0

    def test_record_api_call(self):
        m = MetricsCollector()
        m.record_api_call("fetch_ticker", 150.0, True)
        m.record_api_call("fetch_ticker", 200.0, False)
        assert m.get_counter("api_calls_total") == 2.0
        assert m.get_counter("api_calls_success") == 1.0
        assert m.get_counter("api_calls_failed") == 1.0

    def test_record_data_quality(self):
        m = MetricsCollector()
        m.record_data_quality("BTC/USDT", 95, 100)
        assert m.get_latest("data_missing_pct") == 5.0

    def test_record_error(self):
        m = MetricsCollector()
        m.record_error("timeout", "data_agent")
        assert m.get_counter("errors_total") == 1.0

    def test_get_win_rate(self):
        m = MetricsCollector()
        m.increment("trades_won", 7)
        m.increment("trades_lost", 3)
        assert m.get_win_rate() == pytest.approx(0.7)

    def test_get_win_rate_no_trades(self):
        m = MetricsCollector()
        assert m.get_win_rate() is None

    def test_get_sharpe_ratio(self):
        m = MetricsCollector()
        pnls = [100.0, 200.0, -50.0, 150.0, -30.0, 80.0, 120.0, -10.0, 200.0, -60.0, 100.0]
        sharpe = m.get_sharpe_ratio(pnls)
        assert sharpe is not None
        assert isinstance(sharpe, float)

    def test_get_sharpe_ratio_insufficient_data(self):
        m = MetricsCollector()
        assert m.get_sharpe_ratio([10.0, 20.0]) is None

    def test_get_sharpe_ratio_zero_std(self):
        m = MetricsCollector()
        pnls = [100.0] * 15
        sharpe = m.get_sharpe_ratio(pnls)
        assert sharpe == 0.0

    def test_get_summary(self):
        m = MetricsCollector()
        m.record_equity(15000.0, 3.0)
        m.record_open_positions(2, 20.0)
        m.increment("trades_total", 10)
        s = m.get_summary()
        assert "trades" in s
        assert "portfolio" in s
        assert "system" in s
        assert s["portfolio"]["equity"] == 15000.0

    def test_success_rate(self):
        m = MetricsCollector()
        m.increment("api_calls_success", 90)
        m.increment("api_calls_failed", 10)
        rate = m._get_success_rate("api_calls")
        assert rate == pytest.approx(0.9)

    def test_success_rate_no_calls(self):
        m = MetricsCollector()
        assert m._get_success_rate("api_calls") is None

    def test_cleanup_old_metrics(self):
        m = MetricsCollector(retention_seconds=0)  # expire immediately
        m.record("test_metric", 1.0)
        # Next record triggers cleanup
        m.record("test_metric", 2.0)
        vals = m.get_recent_values("test_metric")
        # At least latest value should be present
        assert len(vals) >= 1

    def test_global_get_metrics(self):
        m = get_metrics()
        assert isinstance(m, MetricsCollector)


# ═══════════════════════════════════════════════════════════
#  HealthMonitor
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestHealthMonitor:
    def test_init(self):
        hm = HealthMonitor()
        assert hm.checks == []

    @patch("core.metrics.get_metrics")
    def test_check_data_freshness_no_data(self, mock_metrics):
        hm = HealthMonitor()
        mock_coll = MagicMock()
        mock_coll.get_recent_values.return_value = []
        mock_metrics.return_value = mock_coll
        hm._check_data_freshness()
        assert any(c.name == "data_freshness" for c in hm.checks)
        df_check = [c for c in hm.checks if c.name == "data_freshness"][0]
        assert df_check.status == HealthStatus.DEGRADED

    @patch("core.metrics.get_metrics")
    def test_check_portfolio_state_bankrupt(self, mock_metrics):
        hm = HealthMonitor()
        mock_coll = MagicMock()
        mock_coll.get_latest.side_effect = lambda k: -100.0 if k == "equity" else 0
        mock_metrics.return_value = mock_coll
        hm._check_portfolio_state()
        ps = [c for c in hm.checks if c.name == "portfolio_state"][0]
        assert ps.status == HealthStatus.UNHEALTHY

    @patch("core.metrics.get_metrics")
    def test_check_portfolio_state_healthy(self, mock_metrics):
        hm = HealthMonitor()
        mock_coll = MagicMock()
        mock_coll.get_latest.side_effect = lambda k: 15000.0 if k == "equity" else 3
        mock_metrics.return_value = mock_coll
        hm._check_portfolio_state()
        ps = [c for c in hm.checks if c.name == "portfolio_state"][0]
        assert ps.status == HealthStatus.HEALTHY

    @patch("core.metrics.get_metrics")
    def test_check_portfolio_state_excessive_positions(self, mock_metrics):
        hm = HealthMonitor()
        mock_coll = MagicMock()
        mock_coll.get_latest.side_effect = lambda k: 15000.0 if k == "equity" else 15
        mock_metrics.return_value = mock_coll
        hm._check_portfolio_state()
        ps = [c for c in hm.checks if c.name == "portfolio_state"][0]
        assert ps.status == HealthStatus.DEGRADED

    def test_get_report(self):
        with patch.object(HealthMonitor, "check_all", return_value=HealthStatus.HEALTHY):
            hm = HealthMonitor()
            report = hm.get_report()
            assert "overall_status" in report
            assert "checks" in report


# ═══════════════════════════════════════════════════════════
#  AlertManager
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAlertManager:
    def test_init(self):
        am = AlertManager()
        assert am.alerts == []

    def test_trigger_alert(self):
        am = AlertManager()
        am.trigger_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Something happened",
            component="test",
        )
        assert len(am.alerts) == 1
        assert am.alerts[0].severity == AlertSeverity.WARNING

    def test_deduplication(self):
        am = AlertManager()
        am.trigger_alert(AlertSeverity.WARNING, "Dup", "msg", "test")
        am.trigger_alert(AlertSeverity.WARNING, "Dup", "msg", "test")
        assert len(am.alerts) == 1  # second suppressed by cooldown

    def test_get_active_alerts(self):
        am = AlertManager()
        am.trigger_alert(AlertSeverity.WARNING, "A", "msg", "test")
        am.trigger_alert(AlertSeverity.CRITICAL, "B", "msg", "test2")
        active = am.get_active_alerts()
        assert len(active) == 2

    def test_get_active_alerts_by_severity(self):
        am = AlertManager()
        am.trigger_alert(AlertSeverity.WARNING, "A", "msg", "test")
        am.trigger_alert(AlertSeverity.CRITICAL, "B", "msg", "test2")
        critical = am.get_active_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical) == 1

    def test_acknowledge_alert(self):
        am = AlertManager()
        am.trigger_alert(AlertSeverity.WARNING, "A", "msg", "test")
        am.acknowledge_alert(am.alerts[0])
        assert am.alerts[0].acknowledged
        assert len(am.get_active_alerts()) == 0

    def test_clear_all(self):
        am = AlertManager()
        am.trigger_alert(AlertSeverity.WARNING, "A", "msg", "test")
        am.clear_all_alerts()
        assert len(am.alerts) == 0
        assert len(am.alert_history) == 0

    @patch("core.alerts.get_metrics")
    def test_check_drawdown_critical(self, mock_metrics):
        am = AlertManager()
        mock_coll = MagicMock()
        mock_coll.get_latest.return_value = 16.0
        mock_coll.get_recent_values.return_value = []
        mock_coll._get_success_rate.return_value = None
        mock_metrics.return_value = mock_coll
        am.check_all()
        assert any("Drawdown" in a.title for a in am.alerts)

    @patch("core.alerts.get_metrics")
    def test_check_drawdown_warning(self, mock_metrics):
        am = AlertManager()
        mock_coll = MagicMock()
        mock_coll.get_latest.return_value = 13.0
        mock_coll.get_recent_values.return_value = []
        mock_coll._get_success_rate.return_value = None
        mock_metrics.return_value = mock_coll
        am.check_all()
        assert any("Approaching" in a.title for a in am.alerts)

    @patch("core.alerts.get_metrics")
    def test_check_api_health_critical(self, mock_metrics):
        am = AlertManager()
        mock_coll = MagicMock()
        mock_coll.get_latest.return_value = None
        mock_coll.get_recent_values.return_value = []
        mock_coll._get_success_rate.return_value = 0.70
        mock_metrics.return_value = mock_coll
        am.check_all()
        assert any("API Failure" in a.title for a in am.alerts)

    @patch("core.alerts.get_metrics")
    def test_check_api_degradation(self, mock_metrics):
        am = AlertManager()
        mock_coll = MagicMock()
        mock_coll.get_latest.return_value = None
        mock_coll.get_recent_values.return_value = []
        mock_coll._get_success_rate.return_value = 0.90
        mock_metrics.return_value = mock_coll
        am.check_all()
        assert any("Degradation" in a.title for a in am.alerts)

    @patch("core.alerts.get_metrics")
    def test_check_data_quality_warning(self, mock_metrics):
        am = AlertManager()
        mock_coll = MagicMock()
        mock_coll.get_latest.return_value = None
        mock_coll._get_success_rate.return_value = None
        mock_val = MagicMock()
        mock_val.value = 8.0
        mock_coll.get_recent_values.return_value = [mock_val] * 10
        mock_metrics.return_value = mock_coll
        am.check_all()
        assert any("Data Quality" in a.title for a in am.alerts)

    @patch("core.alerts.get_metrics")
    def test_check_losing_streak(self, mock_metrics):
        am = AlertManager()
        mock_coll = MagicMock()
        mock_coll.get_latest.return_value = None
        mock_coll._get_success_rate.return_value = None
        # Create 12 consecutive losses
        mock_vals = [MagicMock(value=-100.0) for _ in range(12)]
        mock_coll.get_recent_values.return_value = mock_vals
        mock_metrics.return_value = mock_coll
        am.check_all()
        assert any("Losing Streak" in a.title for a in am.alerts)

    def test_send_email_no_creds(self):
        am = AlertManager(email_enabled=True)
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Test",
            message="msg",
            timestamp=datetime.now(UTC),
            component="test",
        )
        # Should not crash
        am._send_email(alert)
