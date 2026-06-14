"""
KA-MATS Crypto · Raya — The CEO Agent
Iknir Capital

Raya is the autonomous CEO of KA-MATS Crypto.
She orchestrates all agents like a company:

    Morning Scrum (06:00 UTC)
        → Data Agent briefs on overnight price action
        → Market Analyst presents regime map
        → Knowledge Agent highlights relevant research
        → Thesis Agent presents macro conviction
        → Raya sets the daily trading plan

    Trading Session (06:30 – 17:30 UTC)
        → Strategy Agent generates signals
        → Adversarial Agent stress-tests them
        → Risk Manager sizes positions
        → Execution Agent places orders
        → Raya monitors; can override / halt

    Evening Review (18:00 UTC)
        → Reflection Agent analyzes closed trades
        → Adaptive Learner presents updated metrics
        → Raya writes daily performance report
        → Identifies logic improvements for next session

All conversations are logged to the dashboard.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger

# ─────────────────────────────────────────────────────────────
#  ENUMS & MODELS
# ─────────────────────────────────────────────────────────────


class CompanyPhase(StrEnum):
    PRE_MARKET = "pre_market"  # Before 06:00 UTC
    MORNING_SCRUM = "morning_scrum"  # 06:00 – 06:30
    TRADING = "trading"  # 06:30 – 17:30
    EVENING_REVIEW = "evening_review"  # 18:00 – 18:30
    AFTER_HOURS = "after_hours"  # After 18:30


class AgentRole(StrEnum):
    CEO = "Raya (CEO)"
    DATA = "Data Agent"
    MARKET_ANALYST = "Market Analyst"
    ALT_DATA = "Alt Data Agent"
    THESIS = "Thesis Agent"
    KNOWLEDGE = "Knowledge Agent"
    STRATEGY = "Strategy Agent"
    ADVERSARIAL = "Adversarial Agent"
    RISK_MANAGER = "Risk Manager"
    EXECUTION = "Execution Agent"
    REFLECTION = "Reflection Agent"
    LEARNER = "Adaptive Learner"


@dataclass
class Message:
    """Single message in the company communication log."""

    timestamp: datetime
    speaker: str
    phase: str
    content: str
    data: dict[str, Any] | None = None


@dataclass
class DailyPlan:
    """Raya's daily trading plan produced during morning scrum."""

    date: str
    market_regime: dict[str, str] = field(default_factory=dict)
    conviction: str = "NEUTRAL"
    conviction_strength: float = 0.0
    active_strategies: list[str] = field(default_factory=list)
    risk_stance: str = "normal"  # conservative | normal | aggressive
    watchlist: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    max_new_positions: int = 5


@dataclass
class DailyReport:
    """Evening review summary produced by Raya."""

    date: str
    equity: float = 0.0
    equity_change_pct: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0
    realized_pnl: float = 0.0
    win_rate: float | None = None
    regime_map: dict[str, str] = field(default_factory=dict)
    top_performers: list[str] = field(default_factory=list)
    worst_performers: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    risk_events: list[str] = field(default_factory=list)
    next_day_plan: str = ""


# ─────────────────────────────────────────────────────────────
#  RAYA CEO AGENT
# ─────────────────────────────────────────────────────────────


class RayaCEO:
    """
    Raya — Autonomous CEO of KA-MATS Crypto.

    Manages the daily rhythm of:
      Morning Scrum → Trading Session → Evening Review

    All agent interactions are logged as company communications.
    """

    # ── UTC schedule ──────────────────────────────────────────
    SCRUM_HOUR = 6  # 06:00 UTC
    TRADING_START = 6  # 06:30 UTC (starts after scrum)
    TRADING_END = 17  # 17:30 UTC
    REVIEW_HOUR = 18  # 18:00 UTC

    def __init__(self, orchestrator, log_dir: str = "logs/raya") -> None:
        self._orch = orchestrator
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Daily state
        self._today: str | None = None
        self._plan: DailyPlan | None = None
        self._report: DailyReport | None = None
        self._messages: list[Message] = []
        self._scrum_done: bool = False
        self._review_done: bool = False
        self._trades_at_day_start: int = 0
        self._equity_at_day_start: float = 0.0

        # Historical
        self._daily_reports: list[DailyReport] = []

        logger.info("[Raya] CEO Agent initialized — running the company")

    # ─────────────────────────────────────────────────────────
    #  PROPERTIES
    # ─────────────────────────────────────────────────────────

    @property
    def phase(self) -> CompanyPhase:
        """Current company phase based on UTC time."""
        hour = datetime.now(UTC).hour
        if hour < self.SCRUM_HOUR:
            return CompanyPhase.PRE_MARKET
        elif hour == self.SCRUM_HOUR:
            return CompanyPhase.MORNING_SCRUM
        elif hour < self.TRADING_END:
            return CompanyPhase.TRADING
        elif hour == self.REVIEW_HOUR:
            return CompanyPhase.EVENING_REVIEW
        else:
            return CompanyPhase.AFTER_HOURS

    @property
    def today_str(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    @property
    def messages(self) -> list[Message]:
        return self._messages

    @property
    def plan(self) -> DailyPlan | None:
        return self._plan

    @property
    def report(self) -> DailyReport | None:
        return self._report

    @property
    def daily_reports(self) -> list[DailyReport]:
        return self._daily_reports

    # ─────────────────────────────────────────────────────────
    #  MESSAGE BUS
    # ─────────────────────────────────────────────────────────

    def _say(self, speaker: str, content: str, data: dict = None) -> Message:
        """Log a company communication."""
        msg = Message(
            timestamp=datetime.now(UTC),
            speaker=speaker,
            phase=self.phase.value,
            content=content,
            data=data,
        )
        self._messages.append(msg)
        logger.info(f"[{speaker}] {content}")
        return msg

    # ─────────────────────────────────────────────────────────
    #  NEW DAY INITIALIZATION
    # ─────────────────────────────────────────────────────────

    def _new_day(self) -> None:
        """Reset daily state when date changes."""
        today = self.today_str
        if self._today == today:
            return

        # Save yesterday's messages
        if self._today and self._messages:
            self._save_daily_log(self._today)

        self._today = today
        self._messages = []
        self._scrum_done = False
        self._review_done = False
        self._plan = None
        self._report = None
        self._trades_at_day_start = len(self._orch.executor.portfolio.closed_trades)
        self._equity_at_day_start = float(self._orch.executor.portfolio.net_equity)

        self._say(AgentRole.CEO, f"Good morning. Starting operations for {today}.")

    # ─────────────────────────────────────────────────────────
    #  MORNING SCRUM
    # ─────────────────────────────────────────────────────────

    def run_morning_scrum(self) -> DailyPlan:
        """
        06:00 UTC — All-hands morning briefing.

        1. Data Agent: overnight price action summary
        2. Market Analyst: regime map for all symbols
        3. Alt Data: fear & greed, market sentiment
        4. Thesis Agent: macro conviction
        5. Knowledge Agent: relevant research for today's regime
        6. Raya: synthesize into daily trading plan
        """
        self._new_day()
        self._say(AgentRole.CEO, "Morning scrum starting. All agents, report in.")

        orch = self._orch
        datetime.now(UTC)

        # ── 1. Data Agent Brief ──────────────────────────────
        self._say(AgentRole.DATA, "Fetching overnight data for all symbols...")
        snapshots = {}
        price_changes = {}
        for sym in orch.symbols:
            try:
                df = orch.data_agent.fetch_ohlcv(sym, limit=300)
                if df is not None and len(df) >= 200:
                    df = orch.data_agent.compute_indicators(df)
                    from core.orchestrator import _build_snapshot

                    snap = _build_snapshot(sym, df)
                    if snap:
                        snapshots[sym] = snap
                        # 24h change
                        if len(df) >= 2:
                            prev_close = float(df.iloc[-2]["close"])
                            curr_close = float(df.iloc[-1]["close"])
                            if prev_close > 0:
                                price_changes[sym] = (curr_close - prev_close) / prev_close * 100
            except Exception as e:
                logger.debug(f"[Raya/Data] {sym} fetch failed: {e}")

        top_movers = sorted(price_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        movers_str = ", ".join(f"{s} ({c:+.1f}%)" for s, c in top_movers)
        self._say(
            AgentRole.DATA,
            f"Data loaded for {len(snapshots)}/{len(orch.symbols)} symbols. "
            f"Top movers: {movers_str or 'none'}",
            data={"price_changes": price_changes},
        )

        # ── 2. Market Analyst: Regime Map ─────────────────────
        self._say(AgentRole.MARKET_ANALYST, "Analyzing market regimes...")
        regime_map = {}
        regime_confidences = {}
        for sym, snap in snapshots.items():
            try:
                regime = orch.market_analyst.analyse(snap)
                regime_map[sym] = regime.regime.value
                regime_confidences[sym] = regime.confidence
            except Exception:
                regime_map[sym] = "unknown"
                regime_confidences[sym] = 0.0

        # Dominant regime
        from collections import Counter

        regime_counts = Counter(regime_map.values())
        dominant_regime = regime_counts.most_common(1)[0] if regime_counts else ("unknown", 0)
        self._say(
            AgentRole.MARKET_ANALYST,
            f"Regime map complete. Dominant: {dominant_regime[0]} "
            f"({dominant_regime[1]}/{len(regime_map)} symbols). "
            f"Avg confidence: {sum(regime_confidences.values()) / max(1, len(regime_confidences)):.0%}",
            data={"regimes": regime_map, "confidences": regime_confidences},
        )

        # ── 3. Alt Data: Sentiment ────────────────────────────
        self._say(AgentRole.ALT_DATA, "Checking market sentiment...")
        alt_data = None
        fear_greed = None
        try:
            ref_snap = next(iter(snapshots.values()), None)
            alt_data = orch.alt_data_agent.get_context(ref_snap)
            if alt_data:
                # Extract fear & greed from signals list
                for sig in alt_data.signals:
                    if "fear_greed" in sig.source:
                        fear_greed = sig.value
                        break
                self._say(
                    AgentRole.ALT_DATA,
                    f"Fear & Greed Index: {fear_greed or 'N/A'}. "
                    f"Data quality: {'OK' if alt_data.data_quality_ok else 'degraded'}",
                    data={"fear_greed": fear_greed},
                )
            else:
                self._say(AgentRole.ALT_DATA, "Alt data unavailable — proceeding without.")
        except Exception as e:
            self._say(AgentRole.ALT_DATA, f"Alt data fetch failed: {e}")

        # ── 4. Thesis Agent: Macro Conviction ─────────────────
        self._say(AgentRole.THESIS, "Evaluating macro thesis convictions...")
        thesis_summary = {"conviction": None, "strength": 0.0}
        try:
            ref_snap = next(iter(snapshots.values()), None)
            if ref_snap:
                ref_regime = orch.market_analyst.analyse(ref_snap)
                thesis = orch.thesis_agent.score(ref_snap, ref_regime)
                if thesis and thesis.dominant_conviction:
                    thesis_summary["conviction"] = thesis.dominant_conviction.value
                    thesis_summary["strength"] = thesis.overall_thesis_strength
                    self._say(
                        AgentRole.THESIS,
                        f"Dominant conviction: {thesis.dominant_conviction.value} "
                        f"(strength: {thesis.overall_thesis_strength:.0%}). "
                        f"Note: {thesis.advisory_note or 'none'}",
                        data=thesis_summary,
                    )
                else:
                    self._say(AgentRole.THESIS, "No dominant conviction today — neutral stance.")
        except Exception as e:
            self._say(AgentRole.THESIS, f"Thesis scoring error: {e}")

        # ── 5. Knowledge Agent: Research ──────────────────────
        self._say(AgentRole.KNOWLEDGE, "Checking research library for regime-relevant findings...")
        knowledge_notes = []
        try:
            ref_snap = next(iter(snapshots.values()), None)
            if ref_snap:
                ref_regime = orch.market_analyst.analyse(ref_snap)
                knowledge = orch.knowledge_agent.query(
                    regime=ref_regime,
                    signals=[],
                    snapshot=ref_snap,
                    alt_data=alt_data,
                )
                if knowledge and knowledge.suggested_constraints:
                    knowledge_notes = knowledge.suggested_constraints[:3]
                    self._say(
                        AgentRole.KNOWLEDGE,
                        f"Research constraints: {'; '.join(knowledge_notes)}. "
                        f"Confidence modifier: {knowledge.confidence_modifier:+.3f}",
                        data={"constraints": knowledge_notes, "modifier": knowledge.confidence_modifier},
                    )
                else:
                    self._say(AgentRole.KNOWLEDGE, "No specific research constraints for today's regime.")
        except Exception as e:
            self._say(AgentRole.KNOWLEDGE, f"Knowledge query error: {e}")

        # ── 6. Adaptive Learner: Performance Context ──────────
        self._say(AgentRole.LEARNER, "Presenting strategy performance data...")
        learner_summary = orch.learner.get_learning_summary()
        self._say(
            AgentRole.LEARNER,
            f"Total trades observed: {learner_summary['total_trades_observed']}. "
            f"Strategy insights: {len(learner_summary.get('strategy_insights', []))} entries. "
            f"Symbol insights: {len(learner_summary.get('symbol_insights', []))} entries.",
            data=learner_summary,
        )

        # Check recovery suppression
        suppressed = orch.learner.check_recovery_suppression()
        if suppressed:
            self._say(
                AgentRole.LEARNER,
                f"Warning: {len(suppressed)} strategy/regime pair(s) showing recovery suppression.",
                data={"suppressed": [s["message"] for s in suppressed]},
            )

        # ── 7. Raya Synthesizes the Plan ──────────────────────
        # Determine risk stance
        risk_stance = "normal"
        if fear_greed is not None:
            if fear_greed <= 20:
                risk_stance = "conservative"
            elif fear_greed >= 75:
                risk_stance = "aggressive"

        # Determine active strategies
        active_strategies = ["CryptoMomentumBreakout", "CryptoTrendPullback"]

        # Watchlist: symbols in trending_up with good confidence
        watchlist = [
            sym
            for sym, reg in regime_map.items()
            if reg == "trending_up" and regime_confidences.get(sym, 0) > 0.50
        ]

        # Build plan
        plan = DailyPlan(
            date=self.today_str,
            market_regime=regime_map,
            conviction=thesis_summary.get("conviction") or "NEUTRAL",
            conviction_strength=thesis_summary.get("strength", 0.0),
            active_strategies=active_strategies,
            risk_stance=risk_stance,
            watchlist=watchlist[:10],
            notes=[
                f"Dominant regime: {dominant_regime[0]} ({dominant_regime[1]} symbols)",
                f"Fear & Greed: {fear_greed or 'N/A'}",
                f"Risk stance: {risk_stance}",
            ]
            + knowledge_notes,
            max_new_positions=min(5, 9 - len(orch.executor.portfolio.positions)),
        )

        self._plan = plan
        self._scrum_done = True

        self._say(
            AgentRole.CEO,
            f"Daily plan set. Risk stance: {risk_stance}. "
            f"Watchlist: {len(watchlist)} symbols. "
            f"Max new positions: {plan.max_new_positions}. "
            f"Active strategies: {', '.join(active_strategies)}. "
            f"Let's trade.",
            data={
                "plan": {
                    "risk_stance": risk_stance,
                    "watchlist": watchlist,
                    "max_new_positions": plan.max_new_positions,
                    "conviction": plan.conviction,
                }
            },
        )

        self._save_plan(plan)
        return plan

    # ─────────────────────────────────────────────────────────
    #  TRADING SESSION
    # ─────────────────────────────────────────────────────────

    def run_trading_session(self) -> dict[str, Any]:
        """
        Execute the standard pipeline via CryptoOrchestrator.run_bar().

        Raya monitors and logs key decisions. The pipeline itself
        handles Data → Regime → Strategy → Adversarial → Risk → Execution.
        """
        if not self._scrum_done:
            self.run_morning_scrum()

        orch = self._orch

        prev_positions = set(orch.executor.portfolio.positions.keys())
        prev_trades = len(orch.executor.portfolio.closed_trades)
        prev_equity = float(orch.executor.portfolio.net_equity)

        self._say(AgentRole.CEO, "Trading session — running pipeline...")

        # Execute the pipeline
        orch.run_bar()

        # Analyze what happened
        curr_positions = set(orch.executor.portfolio.positions.keys())
        curr_trades = len(orch.executor.portfolio.closed_trades)
        curr_equity = float(orch.executor.portfolio.net_equity)

        new_opens = curr_positions - prev_positions
        new_closes = curr_trades - prev_trades
        equity_change = curr_equity - prev_equity

        # Report new opens
        if new_opens:
            for sym in new_opens:
                pos = orch.executor._positions.get(sym, {})
                self._say(
                    AgentRole.EXECUTION,
                    f"OPENED: {sym} {pos.get('direction', '?')} | "
                    f"strategy={pos.get('strategy_name', '?')} | "
                    f"entry={pos.get('entry_price', 0):.2f}",
                    data={"symbol": sym, "position": {k: str(v) for k, v in pos.items()}},
                )

        # Report new closes
        if new_closes > 0:
            for trade in orch.executor.portfolio.closed_trades[prev_trades:]:
                self._say(
                    AgentRole.EXECUTION,
                    f"CLOSED: {trade.symbol} | {trade.exit_reason} | PnL={trade.pnl:+.2f} USDT",
                    data={"symbol": trade.symbol, "pnl": trade.pnl, "exit_reason": trade.exit_reason},
                )

        # CEO commentary
        positions_count = len(curr_positions)
        self._say(
            AgentRole.CEO,
            f"Pipeline complete. Open: {positions_count} positions, "
            f"Equity: ${curr_equity:,.2f} ({equity_change:+.2f}), "
            f"New opens: {len(new_opens)}, Closes: {new_closes}",
            data={"equity": curr_equity, "positions": positions_count, "equity_change": equity_change},
        )

        return {
            "equity": curr_equity,
            "positions": positions_count,
            "new_opens": list(new_opens),
            "new_closes": new_closes,
            "equity_change": equity_change,
        }

    # ─────────────────────────────────────────────────────────
    #  EVENING REVIEW
    # ─────────────────────────────────────────────────────────

    def run_evening_review(self) -> DailyReport:
        """
        18:00 UTC — End-of-day review meeting.

        1. Reflection Agent: trade-by-trade analysis
        2. Adaptive Learner: updated metrics
        3. Raya: daily report + improvement suggestions
        """
        self._new_day()
        self._say(AgentRole.CEO, "Evening review starting. Let's assess today's performance.")

        orch = self._orch
        port = orch.executor.portfolio

        # Daily metrics
        equity = float(port.net_equity)
        equity_change_pct = 0.0
        if self._equity_at_day_start > 0:
            equity_change_pct = (equity - self._equity_at_day_start) / self._equity_at_day_start * 100

        all_trades = port.closed_trades
        today_trades = all_trades[self._trades_at_day_start :]
        today_wins = [t for t in today_trades if t.pnl > 0]
        today_pnl = sum(t.pnl for t in today_trades)
        win_rate = len(today_wins) / len(today_trades) if today_trades else None

        # ── 1. Reflection Agent: Trade Analysis ────────────────
        self._say(AgentRole.REFLECTION, "Analyzing today's closed trades...")
        lessons = []
        for trade in today_trades:
            lesson = (
                f"{trade.symbol} ({trade.strategy_name}): "
                f"{'WIN' if trade.pnl > 0 else 'LOSS'} {trade.pnl:+.2f} | "
                f"exit: {trade.exit_reason} | regime: {trade.regime}"
            )
            lessons.append(lesson)
            self._say(AgentRole.REFLECTION, lesson)

        if not today_trades:
            self._say(AgentRole.REFLECTION, "No trades closed today.")

        # ── 2. Adaptive Learner: Updated Metrics ──────────────
        self._say(AgentRole.LEARNER, "Presenting updated learning state...")
        summary = orch.learner.get_learning_summary()
        self._say(
            AgentRole.LEARNER,
            f"Lifetime trades: {summary['total_trades_observed']}. "
            f"Active strategy/regime pairs: {len(summary.get('strategy_insights', []))}.",
            data=summary,
        )

        # ── 3. Risk Manager: Portfolio Health ─────────────────
        self._say(AgentRole.RISK_MANAGER, "End-of-day portfolio health check...")
        positions = port.positions
        peak = float(port.peak_equity)
        drawdown_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
        risk_events = []
        if drawdown_pct > 10:
            risk_events.append(f"Drawdown alert: {drawdown_pct:.1f}% from peak")
        if len(positions) > 7:
            risk_events.append(f"High position count: {len(positions)}")

        self._say(
            AgentRole.RISK_MANAGER,
            f"Equity: ${equity:,.2f} | Drawdown: {drawdown_pct:.1f}% | "
            f"Open positions: {len(positions)} | Risk events: {len(risk_events)}",
            data={"drawdown_pct": drawdown_pct, "risk_events": risk_events},
        )

        # ── 4. Raya: Synthesize Daily Report ──────────────────
        # Identify improvements
        improvements = []
        if today_trades:
            stop_losses = [t for t in today_trades if t.exit_reason == "stop_loss"]
            if len(stop_losses) > len(today_trades) * 0.5:
                improvements.append("High stop-loss rate today — review entry timing")

            timeout_exits = [t for t in today_trades if "hold" in (t.exit_reason or "")]

            if timeout_exits:
                improvements.append(
                    f"{len(timeout_exits)} trades expired on max hold — "
                    "consider tighter targets or reviewing hold periods"
                )

        if drawdown_pct > 15:
            improvements.append("Portfolio drawdown exceeding 15% — reduce position sizing tomorrow")

        suppressed = orch.learner.check_recovery_suppression()
        if suppressed:
            improvements.append(
                f"Recovery suppression detected in {len(suppressed)} pair(s) — "
                "learner may be over-penalizing recovering strategies"
            )

        # Top/worst performers
        regime_map = {}
        for sym in orch.symbols:
            last_snap = orch._last_good_snapshot.get(sym)
            if last_snap:
                try:
                    r = orch.market_analyst.analyse(last_snap)
                    regime_map[sym] = r.regime.value
                except Exception:
                    regime_map[sym] = "unknown"

        # Today's open positions P&L
        top_performers = []
        worst_performers = []
        for sym, pos in positions.items():
            if isinstance(pos, dict):
                entry = pos.get("entry_price", 0)
                if entry > 0:
                    # Approximate current value
                    top_performers.append(sym)  # simplified

        report = DailyReport(
            date=self.today_str,
            equity=equity,
            equity_change_pct=equity_change_pct,
            trades_opened=len(positions)
            - max(0, len(positions) - len(orch.executor.portfolio.closed_trades)),
            trades_closed=len(today_trades),
            realized_pnl=today_pnl,
            win_rate=win_rate,
            regime_map=regime_map,
            top_performers=top_performers[:3],
            worst_performers=worst_performers[:3],
            lessons_learned=lessons[:10],
            improvements=improvements,
            risk_events=risk_events,
            next_day_plan=f"Continue {'conservative' if drawdown_pct > 10 else 'normal'} stance",
        )

        self._report = report
        self._review_done = True
        self._daily_reports.append(report)

        self._say(
            AgentRole.CEO,
            f"Daily report: equity ${equity:,.2f} ({equity_change_pct:+.1f}%), "
            f"PnL: {today_pnl:+.2f}, trades closed: {len(today_trades)}, "
            f"WR: {f'{win_rate:.0%}' if win_rate else 'N/A'}. "
            f"Improvements: {len(improvements)}. "
            f"Good work team, see you tomorrow.",
            data={
                "report": {
                    "equity": equity,
                    "pnl": today_pnl,
                    "trades_closed": len(today_trades),
                    "improvements": improvements,
                }
            },
        )

        self._save_report(report)
        return report

    # ─────────────────────────────────────────────────────────
    #  FULL DAY CYCLE
    # ─────────────────────────────────────────────────────────

    def run_full_day(self) -> DailyReport:
        """
        Execute a complete company day:
          1. Morning scrum
          2. Trading session (one bar)
          3. Evening review
        """
        self._new_day()
        self.run_morning_scrum()
        self.run_trading_session()
        return self.run_evening_review()

    # ─────────────────────────────────────────────────────────
    #  CONTINUOUS OPERATION
    # ─────────────────────────────────────────────────────────

    def run_continuous(self, poll_seconds: int = 14_400, max_days: int = None) -> None:
        """
        Run Raya in continuous mode. Each day:
          - Morning scrum at ~06:00 UTC
          - Multiple trading bars throughout the day
          - Evening review at ~18:00 UTC
        """
        logger.info(f"[Raya] Starting continuous operation (poll={poll_seconds}s)")
        day_count = 0
        try:
            while True:
                self._new_day()

                # Morning scrum (once per day)
                if not self._scrum_done:
                    self.run_morning_scrum()

                # Trading bar
                self.run_trading_session()

                # Evening review (once per day, after trading hours)
                now = datetime.now(UTC)
                if now.hour >= self.REVIEW_HOUR and not self._review_done:
                    self.run_evening_review()
                    day_count += 1
                    if max_days and day_count >= max_days:
                        logger.info(f"[Raya] Reached max_days={max_days}")
                        break

                logger.info(f"[Raya] Sleeping {poll_seconds}s until next bar...")
                time.sleep(poll_seconds)

        except KeyboardInterrupt:
            logger.warning("[Raya] Operations halted by user")
        finally:
            self._say(AgentRole.CEO, "Operations suspended. Saving state...")
            self._orch._shutdown()
            if self._messages:
                self._save_daily_log(self.today_str)

    # ─────────────────────────────────────────────────────────
    #  STATUS & LOGGING
    # ─────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Current company status for dashboard."""
        orch = self._orch
        port = orch.executor.portfolio
        equity = float(port.net_equity)
        peak = float(port.peak_equity) if port.peak_equity else equity
        dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0

        closed = port.closed_trades
        wins = sum(1 for t in closed if t.pnl > 0) if closed else 0
        wr = wins / len(closed) if closed else 0.0

        return {
            "phase": self.phase.value,
            "date": self.today_str,
            "equity": equity,
            "peak_equity": peak,
            "drawdown_pct": dd_pct,
            "open_positions": len(port.positions),
            "total_trades": len(closed),
            "win_rate": wr,
            "total_pnl": sum(t.pnl for t in closed) if closed else 0.0,
            "scrum_done": self._scrum_done,
            "review_done": self._review_done,
            "plan": {
                "risk_stance": self._plan.risk_stance if self._plan else "unknown",
                "watchlist": self._plan.watchlist if self._plan else [],
                "conviction": self._plan.conviction if self._plan else "NEUTRAL",
            }
            if self._plan
            else None,
            "positions": {
                sym: {
                    "direction": pos.get("direction", "?"),
                    "entry_price": pos.get("entry_price", 0),
                    "strategy": pos.get("strategy_name", "?"),
                    "bars_held": pos.get("bars_held", 0),
                }
                for sym, pos in port.positions.items()
                if isinstance(pos, dict)
            },
            "recent_messages": [
                {
                    "time": m.timestamp.strftime("%H:%M:%S"),
                    "speaker": m.speaker,
                    "content": m.content,
                }
                for m in self._messages[-20:]
            ],
        }

    def _save_daily_log(self, date_str: str) -> None:
        """Persist daily conversation log."""
        path = self._log_dir / f"{date_str}.jsonl"
        try:
            with open(path, "w", encoding="utf-8") as f:
                for msg in self._messages:
                    entry = {
                        "time": msg.timestamp.isoformat(),
                        "speaker": msg.speaker,
                        "phase": msg.phase,
                        "content": msg.content,
                    }
                    if msg.data:
                        entry["data"] = msg.data
                    f.write(json.dumps(entry, default=str) + "\n")
            logger.info(f"[Raya] Daily log saved: {path}")
        except Exception as e:
            logger.warning(f"[Raya] Failed to save daily log: {e}")

    def _save_plan(self, plan: DailyPlan) -> None:
        path = self._log_dir / f"{plan.date}_plan.json"
        try:
            import dataclasses

            path.write_text(json.dumps(dataclasses.asdict(plan), indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Raya] Failed to save plan: {e}")

    def _save_report(self, report: DailyReport) -> None:
        path = self._log_dir / f"{report.date}_report.json"
        try:
            import dataclasses

            path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Raya] Failed to save report: {e}")
