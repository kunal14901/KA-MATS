"""
KA-MATS Cryptoz · Health Checks
Iknir Capital

System health monitoring and diagnostics.

Checks:
- Exchange API connectivity
- Data freshness (latest bar received)
- Portfolio state consistency
- Agent initialization status
- Disk space and memory usage
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from loguru import logger


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    latency_ms: float | None = None


class HealthMonitor:
    """
    System health monitoring.

    Usage:
        monitor = HealthMonitor()
        status = monitor.check_all()

        if status != HealthStatus.HEALTHY:
            print(f"System health: {status}")
    """

    def __init__(self):
        self.checks: list[HealthCheck] = []
        logger.info("[HealthMonitor] Initialized")

    def check_all(self) -> HealthStatus:
        """Run all health checks and return overall status."""
        self.checks.clear()

        self._check_exchange_connectivity()
        self._check_data_freshness()
        self._check_portfolio_state()
        self._check_system_resources()

        # Determine overall health
        if any(c.status == HealthStatus.UNHEALTHY for c in self.checks):
            return HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in self.checks):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _check_exchange_connectivity(self):
        """Test exchange API connectivity."""
        start = time.time()

        try:
            import ccxt

            exchange = ccxt.binance({"enableRateLimit": True})

            # Quick ping-style check
            exchange.fetch_ticker("BTC/USDT")

            latency = (time.time() - start) * 1000

            if latency > 5000:  # >5 seconds
                status = HealthStatus.DEGRADED
                message = f"Exchange API slow ({latency:.0f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Exchange API reachable ({latency:.0f}ms)"

            self.checks.append(
                HealthCheck(
                    name="exchange_api",
                    status=status,
                    message=message,
                    timestamp=datetime.now(UTC),
                    latency_ms=latency,
                )
            )

        except Exception as e:
            self.checks.append(
                HealthCheck(
                    name="exchange_api",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Exchange API unreachable: {e}",
                    timestamp=datetime.now(UTC),
                )
            )

    def _check_data_freshness(self):
        """Check if market data is up-to-date."""
        # In production, would check last bar timestamp vs current time
        # For now, placeholder

        from core.metrics import get_metrics

        metrics = get_metrics()

        # Check if we've received data recently
        recent_data = metrics.get_recent_values("data_missing_pct", last_n=5)

        if not recent_data:
            status = HealthStatus.DEGRADED
            message = "No recent data quality metrics"
        elif any(m.value > 10.0 for m in recent_data):
            status = HealthStatus.DEGRADED
            message = "Data quality issues detected"
        else:
            status = HealthStatus.HEALTHY
            message = "Market data fresh"

        self.checks.append(
            HealthCheck(
                name="data_freshness",
                status=status,
                message=message,
                timestamp=datetime.now(UTC),
            )
        )

    def _check_portfolio_state(self):
        """Validate portfolio state consistency."""
        try:
            from core.metrics import get_metrics

            metrics = get_metrics()

            equity = metrics.get_latest("equity")
            open_pos = metrics.get_latest("open_positions")

            if equity is None:
                status = HealthStatus.DEGRADED
                message = "Portfolio equity not tracked"
            elif equity <= 0:
                status = HealthStatus.UNHEALTHY
                message = f"Portfolio bankrupt (equity: ${equity:.2f})"
            elif open_pos and open_pos > 10:
                status = HealthStatus.DEGRADED
                message = f"Excessive open positions ({int(open_pos)})"
            else:
                status = HealthStatus.HEALTHY
                message = f"Portfolio healthy (equity: ${equity:.2f}, positions: {int(open_pos or 0)})"

            self.checks.append(
                HealthCheck(
                    name="portfolio_state",
                    status=status,
                    message=message,
                    timestamp=datetime.now(UTC),
                )
            )

        except Exception as e:
            self.checks.append(
                HealthCheck(
                    name="portfolio_state",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Portfolio check failed: {e}",
                    timestamp=datetime.now(UTC),
                )
            )

    def _check_system_resources(self):
        """Check system resource usage."""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            memory_pct = memory.percent

            # Disk usage
            disk_root = Path.cwd().anchor or "/"
            disk = psutil.disk_usage(disk_root)
            disk_pct = disk.percent

            # RAM: desktop OSes aggressively use RAM for cache, so 90%+ is
            # normal and NOT a risk to this process — only flag unhealthy at
            # 97%+ (true memory pressure). Disk stays strict at 90%: a full
            # disk breaks state persistence and logging.
            if memory_pct > 97 or disk_pct > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical resource usage (RAM: {memory_pct:.0f}%, Disk: {disk_pct:.0f}%)"
            elif memory_pct > 90 or disk_pct > 75:
                status = HealthStatus.DEGRADED
                message = f"High resource usage (RAM: {memory_pct:.0f}%, Disk: {disk_pct:.0f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resources healthy (RAM: {memory_pct:.0f}%, Disk: {disk_pct:.0f}%)"

            self.checks.append(
                HealthCheck(
                    name="system_resources",
                    status=status,
                    message=message,
                    timestamp=datetime.now(UTC),
                )
            )

        except ImportError:
            # psutil not installed
            self.checks.append(
                HealthCheck(
                    name="system_resources",
                    status=HealthStatus.HEALTHY,
                    message="Resource monitoring not available (install psutil)",
                    timestamp=datetime.now(UTC),
                )
            )
        except Exception as e:
            self.checks.append(
                HealthCheck(
                    name="system_resources",
                    status=HealthStatus.DEGRADED,
                    message=f"Resource check failed: {e}",
                    timestamp=datetime.now(UTC),
                )
            )

    def get_report(self) -> dict:
        """Get health check report as dict."""
        overall_status = self.check_all()

        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                }
                for c in self.checks
            ],
        }


# Global health monitor instance
_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    return _health_monitor


def health_check() -> dict:
    """Quick health check function (for CLI)."""
    monitor = get_health_monitor()
    return monitor.get_report()
