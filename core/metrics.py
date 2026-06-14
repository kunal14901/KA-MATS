"""
KA-MATS Cryptoz · Metrics Collection
Iknir Capital

Prometheus-style metrics for monitoring system health and performance.

Metrics collected:
- Trade execution (count, PnL, win rate)
- Position management (open positions, exposure)
- System health (API latency, errors, data quality)
- Performance (equity, drawdown, Sharpe ratio)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from loguru import logger


@dataclass
class MetricValue:
    """Single metric observation."""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Lightweight metrics collection system.

    Stores recent metrics in memory for dashboard display and alerting.
    Can be extended to export to Prometheus, InfluxDB, etc.
    """

    def __init__(self, retention_seconds: int = 86400):
        """
        Args:
            retention_seconds: How long to keep metrics in memory (default: 24h)
        """
        self.retention_seconds = retention_seconds
        self._metrics: dict[str, deque[MetricValue]] = {}
        self._counters: dict[str, float] = {}
        self._start_time = time.time()

    # ────────────────────────────────────────────────────────────
    #  METRIC RECORDING
    # ────────────────────────────────────────────────────────────

    def record(self, metric_name: str, value: float, labels: dict[str, str] = None):
        """Record a metric value with optional labels."""
        if metric_name not in self._metrics:
            self._metrics[metric_name] = deque()

        metric = MetricValue(
            timestamp=datetime.now(UTC),
            value=value,
            labels=labels or {},
        )

        self._metrics[metric_name].append(metric)
        self._cleanup_old_metrics(metric_name)

    def increment(self, counter_name: str, amount: float = 1.0):
        """Increment a counter."""
        if counter_name not in self._counters:
            self._counters[counter_name] = 0.0
        self._counters[counter_name] += amount

    # ────────────────────────────────────────────────────────────
    #  TRADING METRICS
    # ────────────────────────────────────────────────────────────

    def record_trade_opened(self, symbol: str, strategy: str, regime: str, size: float):
        """Record trade entry."""
        self.increment("trades_total")
        self.increment("trades_opened")
        self.record("position_size", size, {"symbol": symbol, "strategy": strategy})
        logger.debug(f"[Metrics] Trade opened: {symbol} {strategy} {regime}")

    def record_trade_closed(self, symbol: str, strategy: str, pnl: float, exit_reason: str):
        """Record trade exit."""
        self.increment("trades_closed")

        if pnl > 0:
            self.increment("trades_won")
        else:
            self.increment("trades_lost")

        self.record("pnl", pnl, {"symbol": symbol, "strategy": strategy, "exit": exit_reason})
        logger.debug(f"[Metrics] Trade closed: {symbol} PnL=${pnl:.2f} ({exit_reason})")

    def record_equity(self, equity: float, drawdown_pct: float):
        """Record portfolio equity and drawdown."""
        self.record("equity", equity)
        self.record("drawdown_pct", drawdown_pct)

    def record_open_positions(self, count: int, total_exposure_pct: float):
        """Record current position count and exposure."""
        self.record("open_positions", count)
        self.record("portfolio_exposure_pct", total_exposure_pct)

    # ────────────────────────────────────────────────────────────
    #  SYSTEM HEALTH METRICS
    # ────────────────────────────────────────────────────────────

    def record_api_call(self, endpoint: str, latency_ms: float, success: bool):
        """Record API call performance."""
        self.increment("api_calls_total")
        if success:
            self.increment("api_calls_success")
        else:
            self.increment("api_calls_failed")

        self.record("api_latency_ms", latency_ms, {"endpoint": endpoint})

    def record_data_quality(self, symbol: str, bars_received: int, bars_expected: int):
        """Record data quality metrics."""
        missing_pct = (bars_expected - bars_received) / bars_expected * 100
        self.record("data_missing_pct", missing_pct, {"symbol": symbol})

        if missing_pct > 5.0:
            logger.warning(f"[Metrics] Data quality issue: {symbol} missing {missing_pct:.1f}%")

    def record_error(self, error_type: str, component: str):
        """Record system errors."""
        self.increment("errors_total")
        self.increment(f"errors_{error_type}")
        logger.warning(f"[Metrics] Error recorded: {error_type} in {component}")

    # ────────────────────────────────────────────────────────────
    #  METRIC RETRIEVAL
    # ────────────────────────────────────────────────────────────

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self._counters.get(name, 0.0)

    def get_recent_values(self, metric_name: str, last_n: int = 100) -> list[MetricValue]:
        """Get last N values for a metric."""
        if metric_name not in self._metrics:
            return []

        metrics = list(self._metrics[metric_name])
        return metrics[-last_n:]

    def get_latest(self, metric_name: str) -> float | None:
        """Get most recent value for a metric."""
        if metric_name not in self._metrics or not self._metrics[metric_name]:
            return None
        return self._metrics[metric_name][-1].value

    def get_win_rate(self) -> float | None:
        """Calculate current win rate."""
        won = self.get_counter("trades_won")
        lost = self.get_counter("trades_lost")
        total = won + lost

        if total == 0:
            return None
        return won / total

    def get_sharpe_ratio(self, recent_pnls: list[float]) -> float | None:
        """Calculate Sharpe ratio from recent PnL values."""
        if len(recent_pnls) < 10:
            return None

        import numpy as np

        returns = np.array(recent_pnls)
        if returns.std() == 0:
            return 0.0

        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        return float(sharpe)

    def get_summary(self) -> dict[str, any]:
        """Get comprehensive metrics summary."""
        equity = self.get_latest("equity")
        drawdown = self.get_latest("drawdown_pct")
        open_pos = self.get_latest("open_positions")

        return {
            "uptime_seconds": time.time() - self._start_time,
            "trades": {
                "total": self.get_counter("trades_total"),
                "won": self.get_counter("trades_won"),
                "lost": self.get_counter("trades_lost"),
                "win_rate": self.get_win_rate(),
            },
            "portfolio": {
                "equity": equity,
                "drawdown_pct": drawdown,
                "open_positions": int(open_pos) if open_pos else 0,
            },
            "system": {
                "api_calls": self.get_counter("api_calls_total"),
                "api_success_rate": self._get_success_rate("api_calls"),
                "errors": self.get_counter("errors_total"),
            },
        }

    # ────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ────────────────────────────────────────────────────────────

    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than retention period — always keep the latest value."""
        now = datetime.now(UTC)
        cutoff = now.timestamp() - self.retention_seconds

        while len(self._metrics[metric_name]) > 1:
            oldest = self._metrics[metric_name][0]
            if oldest.timestamp.timestamp() < cutoff:
                self._metrics[metric_name].popleft()
            else:
                break

    def _get_success_rate(self, prefix: str) -> float | None:
        """Calculate success rate for counters with _success and _failed suffixes."""
        success = self.get_counter(f"{prefix}_success")
        failed = self.get_counter(f"{prefix}_failed")
        total = success + failed

        if total == 0:
            return None
        return success / total


# Global metrics instance
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _metrics
