"""
KA-MATS Cryptoz · Alerting System
Iknir Capital

Monitors metrics and triggers alerts for critical conditions.

Alert types:
- Drawdown breaches (15% max DD circuit breaker)
- API failures (exchange connectivity issues)
- Data quality degradation
- Stuck positions (open >72h with no movement)
- Performance anomalies (10+ losing trades in a row)
"""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import StrEnum

from loguru import logger

from core.metrics import get_metrics


class AlertSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Alert notification."""

    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    component: str
    acknowledged: bool = False


class AlertManager:
    """
    Monitors metrics and triggers alerts.

    Supports:
    - Email notifications
    - Webhook notifications (Slack, Discord, Telegram)
    - Alert deduplication (prevent spam)
    """

    def __init__(
        self,
        email_enabled: bool = False,
        webhook_url: str | None = None,
        check_interval_seconds: int = 60,
    ):
        self.email_enabled = email_enabled
        self.webhook_url = webhook_url
        self.check_interval = check_interval_seconds

        self.alerts: list[Alert] = []
        self.alert_history: dict[str, datetime] = {}  # For deduplication
        self.cooldown_minutes = 15  # Minimum time between duplicate alerts

        # Email configuration from environment
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.alert_email_to = os.getenv("ALERT_EMAIL_TO", "")

        logger.info("[AlertManager] Initialized")

    # ────────────────────────────────────────────────────────────
    #  ALERT TRIGGERS
    # ────────────────────────────────────────────────────────────

    def check_all(self):
        """Run all alert checks."""
        metrics = get_metrics()

        self._check_drawdown(metrics)
        self._check_api_health(metrics)
        self._check_data_quality(metrics)
        self._check_losing_streak(metrics)
        self._check_stuck_positions(metrics)

    def _check_drawdown(self, metrics):
        """Alert on excessive drawdown."""
        drawdown_pct = metrics.get_latest("drawdown_pct")

        if drawdown_pct is None:
            return

        if drawdown_pct > 15.0:
            self.trigger_alert(
                severity=AlertSeverity.CRITICAL,
                title="Circuit Breaker: Max Drawdown Exceeded",
                message=f"Portfolio drawdown: {drawdown_pct:.2f}% (limit: 15.0%)\n"
                f"Trading has been halted. Manual intervention required.",
                component="risk_manager",
            )
        elif drawdown_pct > 12.0:
            self.trigger_alert(
                severity=AlertSeverity.WARNING,
                title="Warning: Approaching Max Drawdown",
                message=f"Portfolio drawdown: {drawdown_pct:.2f}% (limit: 15.0%)\n"
                f"Consider reducing position sizes or halting new trades.",
                component="risk_manager",
            )

    def _check_api_health(self, metrics):
        """Alert on API failures."""
        success_rate = metrics._get_success_rate("api_calls")

        if success_rate is None:
            return

        if success_rate < 0.80:
            self.trigger_alert(
                severity=AlertSeverity.CRITICAL,
                title="API Failure: Exchange Connectivity Issues",
                message=f"API success rate: {success_rate:.1%} (threshold: 80%)\n"
                f"Check exchange status and network connectivity.",
                component="data_agent",
            )
        elif success_rate < 0.95:
            self.trigger_alert(
                severity=AlertSeverity.WARNING,
                title="API Degradation Detected",
                message=f"API success rate: {success_rate:.1%}\nMonitoring for further degradation.",
                component="data_agent",
            )

    def _check_data_quality(self, metrics):
        """Alert on poor data quality."""
        # Check recent data_missing_pct metrics
        missing_metrics = metrics.get_recent_values("data_missing_pct", last_n=10)

        if not missing_metrics:
            return

        avg_missing = sum(m.value for m in missing_metrics) / len(missing_metrics)

        if avg_missing > 5.0:
            self.trigger_alert(
                severity=AlertSeverity.WARNING,
                title="Data Quality Degradation",
                message=f"Average missing data: {avg_missing:.1f}% (threshold: 5%)\n"
                f"Strategies may underperform with incomplete data.",
                component="data_agent",
            )

    def _check_losing_streak(self, metrics):
        """Alert on extended losing streak."""
        # Get recent PnL values
        recent_pnls = metrics.get_recent_values("pnl", last_n=20)

        if len(recent_pnls) < 10:
            return

        # Count consecutive losses
        losing_streak = 0
        for pnl in reversed(recent_pnls):
            if pnl.value < 0:
                losing_streak += 1
            else:
                break

        if losing_streak >= 10:
            self.trigger_alert(
                severity=AlertSeverity.WARNING,
                title="Performance Alert: Extended Losing Streak",
                message=f"Consecutive losses: {losing_streak}\n"
                f"Consider reviewing strategy parameters or market conditions.",
                component="strategy_agent",
            )

    def _check_stuck_positions(self, metrics):
        """Alert on positions open too long without movement."""
        # This requires position tracking - placeholder for now
        # In production, would query executor.portfolio.positions
        # and check entry_time vs current time
        pass

    # ────────────────────────────────────────────────────────────
    #  ALERT DELIVERY
    # ────────────────────────────────────────────────────────────

    def trigger_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str,
    ):
        """Trigger an alert with deduplication."""
        # Deduplication key
        alert_key = f"{component}:{title}"

        # Check cooldown
        if alert_key in self.alert_history:
            last_sent = self.alert_history[alert_key]
            if datetime.now(UTC) - last_sent < timedelta(minutes=self.cooldown_minutes):
                logger.debug(f"[AlertManager] Suppressed duplicate alert: {title}")
                return

        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(UTC),
            component=component,
        )

        self.alerts.append(alert)
        self.alert_history[alert_key] = alert.timestamp

        # Log alert
        log_func = logger.critical if severity == AlertSeverity.CRITICAL else logger.warning
        log_func(f"[AlertManager] {severity.value}: {title}")

        # Send notifications
        if self.email_enabled and self.alert_email_to:
            self._send_email(alert)

        if self.webhook_url:
            self._send_webhook(alert)

    def _send_email(self, alert: Alert):
        """Send email notification."""
        if not self.smtp_user or not self.smtp_password:
            logger.warning("[AlertManager] Email credentials not configured")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = self.alert_email_to
            msg["Subject"] = f"[KA-MATS {alert.severity.value}] {alert.title}"

            body = f"""
KA-MATS Cryptoz Alert
Severity: {alert.severity.value}
Component: {alert.component}
Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

{alert.message}

---
This is an automated alert from KA-MATS Cryptoz.
"""

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"[AlertManager] Email sent: {alert.title}")

        except Exception as e:
            logger.error(f"[AlertManager] Failed to send email: {e}")

    def _send_webhook(self, alert: Alert):
        """Send webhook notification (Slack/Discord/Telegram)."""
        try:
            import requests

            # Generic webhook payload (works with most services)
            payload = {
                "text": f"**[{alert.severity.value}] {alert.title}**",
                "content": f"**Component:** {alert.component}\n**Time:** {alert.timestamp}\n\n{alert.message}",
            }

            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()

            logger.info(f"[AlertManager] Webhook sent: {alert.title}")

        except Exception as e:
            logger.error(f"[AlertManager] Failed to send webhook: {e}")

    # ────────────────────────────────────────────────────────────
    #  ALERT MANAGEMENT
    # ────────────────────────────────────────────────────────────

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get unacknowledged alerts."""
        alerts = [a for a in self.alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def acknowledge_alert(self, alert: Alert):
        """Mark alert as acknowledged."""
        alert.acknowledged = True
        logger.info(f"[AlertManager] Alert acknowledged: {alert.title}")

    def clear_all_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
        self.alert_history.clear()
        logger.info("[AlertManager] All alerts cleared")


# Global alert manager instance
_alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    return _alert_manager
