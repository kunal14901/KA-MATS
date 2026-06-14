"""
KA-MATS · Adaptive Learning Engine
Iknir Capital — Phase II (v17a)

v17a change: Regime-Aware Adaptive Learner
  - Replaces single EMA per strategy with regime-partitioned EMA matrix
  - strategy_regime_wr[strategy][family] where family ∈ {bull, bear, sideways}
  - Eliminates cross-regime contamination: bear losses no longer penalize bull decisions
  - decay_stale_records() decays per regime family independently (90-day threshold)
  - Backward compatible: loads v16 flat state and converts to nested

REGIME FAMILIES:
  trending_up    → bull
  trending_down  → bear
  volatile       → bear   (fear-driven, stop-heavy — same profile as trending_down)
  ranging        → sideways
  mean_reverting → sideways

WHY THIS MATTERS:
  v16 problem: CSM performed 59% WR in bull, 33% WR in volatile/bear.
  Single global EMA collapsed to ~45% → learner penalized CSM in bull regimes.
  v17a: bull-family EMA stays at 59%, bear-family at 33% — correctly isolated.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date as _date
from datetime import datetime
from datetime import timedelta as _td
from pathlib import Path

from loguru import logger

from config.settings import CONFIG

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

_STATE_FILE = Path("knowledge/.adaptive_state.json")

_EMA_ALPHA = 0.08
_MIN_TRADES = 3
_MAX_CONF_MOD = 0.25
_MAX_ATR_ADJ = 0.50
_MAX_FLAG_ADJ = 0.30

# Monitoring thresholds — log warnings when WR crosses these levels
_WARN_WR_LOW = 0.38  # EMA WR below this → meaningful penalty active
_WARN_WR_CRITICAL = 0.30  # Below this → strong suppression, recovery risk high
_RECOVERY_WINDOW = 5  # Recent trades to check for recovery signal

# v17a: Regime family mapping (mirrors bm25_memory.py)
REGIME_FAMILIES: dict[str, str] = {
    "trending_up": "bull",
    "trending_down": "bear",
    "volatile": "bear",
    "ranging": "sideways",
    "mean_reverting": "sideways",
}

_ALL_FAMILIES = ("bull", "bear", "sideways")


# ─────────────────────────────────────────────────────────────
#  LEARNING RECORDS (pure data — no logic)
# ─────────────────────────────────────────────────────────────


@dataclass
class SymbolRecord:
    """Rolling performance for a single symbol."""

    symbol: str
    total_trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    rolling_win_rate: float = 0.5
    rolling_pnl: float = 0.0
    stop_hits: int = 0
    take_profit_hits: int = 0
    last_updated: str = ""

    @property
    def stop_hit_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.stop_hits / self.total_trades

    @property
    def confidence_modifier(self) -> float:
        if self.total_trades < _MIN_TRADES:
            return 0.0
        deviation = self.rolling_win_rate - 0.5
        raw = deviation * 2.0 * _MAX_CONF_MOD
        return max(-_MAX_CONF_MOD, min(_MAX_CONF_MOD, raw))

    @property
    def atr_multiplier_adjustment(self) -> float:
        if self.total_trades < _MIN_TRADES:
            return 0.0
        rate = self.stop_hit_rate
        if rate > 0.55:
            excess = (rate - 0.55) / 0.45
            return min(_MAX_ATR_ADJ, excess * _MAX_ATR_ADJ)
        elif rate < 0.25 and self.total_trades >= 10:
            surplus = (0.25 - rate) / 0.25
            return max(-0.20, -surplus * 0.20)
        return 0.0


@dataclass
class FlagRecord:
    """Tracks whether adversarial FLAG verdicts were justified."""

    flag_type: str
    total_flagged: int = 0
    flag_led_to_loss: int = 0
    rolling_accuracy: float = 0.5

    @property
    def penalty_weight(self) -> float:
        if self.total_flagged < _MIN_TRADES:
            return 1.0
        raw = 0.5 + self.rolling_accuracy
        return max(0.5, min(1.5, raw))


@dataclass
class ConvictionRecord:
    """Tracks P&L when SA thesis conviction was aligned vs not aligned."""

    conviction: str
    aligned_trades: int = 0
    aligned_wins: int = 0
    aligned_pnl: float = 0.0
    rolling_win_rate: float = 0.5
    last_updated: str = ""

    @property
    def alignment_score_modifier(self) -> float:
        if self.aligned_trades < _MIN_TRADES:
            return 0.0
        deviation = self.rolling_win_rate - 0.5
        return max(-0.15, min(0.15, deviation * 0.6))


# ─────────────────────────────────────────────────────────────
#  ADAPTIVE LEARNER (v17a: regime-partitioned)
# ─────────────────────────────────────────────────────────────


class AdaptiveLearner:
    """
    Self-supervised feedback engine for KA-MATS agents.

    v17a: Regime-partitioned EMA matrix.
      strategy_regime_wr[strategy][family]  — separate EMA per regime family
      strategy_regime_count[strategy::family] — trade count per regime family

    strategy_modifier(strategy, regime) looks up the family-specific EMA,
    not a global average. Bear losses stay in bear bucket; bull stays clean.
    """

    def __init__(self, state_file: str = None) -> None:
        self._state_file = Path(state_file) if state_file else _STATE_FILE

        # v17a: nested regime-partitioned win-rate dict
        # {strategy: {family: ema_win_rate}}
        self._strategy_regime_wr: dict[str, dict[str, float]] = {}
        # {strategy::family: count}
        self._strategy_regime_count: dict[str, int] = {}
        # {strategy::family: last_trade_date_str}
        self._last_trade_date: dict[str, str] = {}

        self._symbols: dict[str, SymbolRecord] = {}
        self._flags: dict[str, FlagRecord] = {}
        self._convictions: dict[str, ConvictionRecord] = {}
        self._trade_count: int = 0
        self._total_pnl: float = 0.0

        # In-memory recent-trade ring buffer: {strategy::family -> [1/0, ...]}
        # Keeps last _RECOVERY_WINDOW outcomes for recovery suppression detection.
        # Not persisted — resets on process restart (intentional: fresh start = no penalty lag).
        self._recent_outcomes: dict[str, list[int]] = {}

        # Correlation tracker: pairwise co-loss detection for portfolio concentration
        self.correlation_tracker = CorrelationTracker()

    # ─────────────────────────────────────────────────────────
    #  REGIME FAMILY RESOLVER
    # ─────────────────────────────────────────────────────────

    def _regime_family(self, regime: str) -> str:
        return REGIME_FAMILIES.get(regime, "sideways")

    # ─────────────────────────────────────────────────────────
    #  CORE: RECORD OUTCOME
    # ─────────────────────────────────────────────────────────

    def decay_stale_records(self, as_of_date: str = "") -> None:
        """
        v17a: Nudge stale regime-family WRs toward 0.5 independently.
        Each (strategy, family) pair decays only if it hasn't seen a trade
        in > 90 days. Active regime families stay untouched.

        10% nudge per call: new = old + 0.10 * (0.50 - old)
        """
        if not as_of_date:
            return
        try:
            cutoff = _date.fromisoformat(as_of_date) - _td(days=90)
            cutoff_str = cutoff.isoformat()
        except Exception:
            return

        for strat, families in self._strategy_regime_wr.items():
            for family in list(families.keys()):
                key = f"{strat}::{family}"
                last = self._last_trade_date.get(key, "")
                if last and last[:10] >= cutoff_str:
                    continue  # recently active — leave it alone
                # Stale: nudge 10% toward neutral
                old = families[family]
                families[family] = old + 0.10 * (0.50 - old)
                logger.debug(f"[AdaptiveLearner] Decay {key}: {old:.3f} → {families[family]:.3f}")

        # Symbol records: original 18-month decay (unchanged)
        cutoff_sym = _date.fromisoformat(as_of_date) - _td(days=18 * 30)
        cutoff_sym_str = cutoff_sym.isoformat()
        for rec in self._symbols.values():
            if rec.last_updated and rec.last_updated[:10] < cutoff_sym_str:
                for _ in range(5):
                    rec.rolling_win_rate = _ema(rec.rolling_win_rate, 0.5)
                    rec.rolling_pnl = _ema(rec.rolling_pnl, 0.0)

    def record_outcome(
        self,
        symbol: str,
        strategy: str,
        regime: str,
        pnl: float,
        exit_reason: str = "",
        flag_types: list[str] | None = None,
        conviction: str | None = None,
        trade_date: str = "",
    ) -> None:
        """Update all learning records after a trade closes."""
        is_win = pnl > 0
        win_val = 1.0 if is_win else 0.0
        stop_hit = exit_reason == "stop_loss"
        now = trade_date or datetime.utcnow().isoformat()[:19]

        # ── v17a: Update strategy × regime-FAMILY stats ───────
        self._update_strategy_regime(strategy, regime, win_val, now)

        # ── Update correlation tracker ─────────────────────────
        self.correlation_tracker.record_bar_outcome(now[:10], symbol, is_win)

        # ── Update symbol stats ────────────────────────────────
        self._update_symbol(symbol, pnl, win_val, stop_hit, exit_reason, now)

        # ── Update adversarial flag accuracy ──────────────────
        if flag_types:
            for ftype in flag_types:
                self._update_flag(ftype, is_win)

        # ── Update SA conviction alignment stats ──────────────
        if conviction:
            self._update_conviction(conviction, pnl, win_val, now)

        self._trade_count += 1
        self._total_pnl += pnl

        logger.info(
            f"[AdaptiveLearner] Trade #{self._trade_count} recorded | "
            f"{symbol} {strategy}/{regime} | "
            f"P&L={pnl:+,.2f} | {'WIN' if is_win else 'LOSS'} | "
            f"exit={exit_reason} | "
            f"strat_mod={self.strategy_modifier(strategy, regime):+.3f} | "
            f"sym_mod={self.symbol_confidence(symbol):+.3f}"
        )

    # ─────────────────────────────────────────────────────────
    #  AGENT QUERY INTERFACE
    # ─────────────────────────────────────────────────────────

    def strategy_modifier(self, strategy: str, regime: str) -> float:
        """
        v17a: Looks up regime-family-specific EMA win rate.
        Bear losses only affect bear-family modifier; bull stays clean.

        modifier = (wr - 0.50) * (0.25 / 0.15), capped at ±0.25
        Returns 0.0 if fewer than 5 trades in this regime family.
        """
        family = self._regime_family(regime)

        if strategy not in self._strategy_regime_wr:
            return 0.0

        wr = self._strategy_regime_wr[strategy].get(family, 0.50)
        count_key = f"{strategy}::{family}"
        count = self._strategy_regime_count.get(count_key, 0)

        if count < 12:
            return 0.0  # cold-start protection — need 12 trades per family for reliable EMA

        modifier = (wr - 0.50) * (0.25 / 0.15)
        modifier = max(-0.25, min(0.25, modifier))

        if abs(modifier) > 0.02:
            logger.debug(
                f"[AdaptiveLearner] Strategy mod: {strategy}/{family} "
                f"wr={wr:.1%} ({count} trades) → {modifier:+.3f}"
            )
        return modifier

    def strategy_win_rate(self, strategy: str, regime: str) -> float | None:
        """Win rate for strategy in this regime's family. None if < 12 trades."""
        family = self._regime_family(regime)
        if strategy not in self._strategy_regime_wr:
            return None
        count_key = f"{strategy}::{family}"
        count = self._strategy_regime_count.get(count_key, 0)
        if count < 12:
            return None
        return self._strategy_regime_wr[strategy].get(family, 0.50)

    def strategy_participation_multiplier(
        self,
        strategy: str,
        regime: str,
        direction: str = "LONG",
    ) -> float:
        """
        Soft participation model for regime-aware sizing.

        Instead of fully blocking weak regime/strategy pairs, reduce size and keep
        a small amount of participation for learning and adaptability.
        """
        cfg = CONFIG.risk
        if not getattr(cfg, "regime_soft_participation_enabled", False):
            return 1.0

        family = self._regime_family(regime)
        mult = 1.0

        if family == "bear" and direction.upper() == "LONG":
            bear_mult = getattr(cfg, "regime_soft_bear_long_mult", 0.25)
            mult = min(mult, bear_mult)

        if strategy not in self._strategy_regime_wr:
            return max(0.0, min(1.0, mult))

        if not getattr(cfg, "regime_soft_wr_penalty_enabled", False):
            return max(0.0, min(1.0, mult))

        min_trades = getattr(cfg, "regime_soft_min_trades", 12)
        min_wr = getattr(cfg, "regime_soft_min_wr", 0.35)
        floor_mult = getattr(cfg, "regime_soft_floor_mult", 0.20)

        count_key = f"{strategy}::{family}"
        count = self._strategy_regime_count.get(count_key, 0)
        if count < min_trades:
            return max(0.0, min(1.0, mult))

        wr = self._strategy_regime_wr[strategy].get(family, 0.50)
        if wr < min_wr:
            severity = min(1.0, (min_wr - wr) / max(min_wr, 1e-9))
            wr_mult = 1.0 - severity * (1.0 - floor_mult)
            mult = min(mult, wr_mult)
            logger.debug(
                f"[RegimeParticipation] {strategy}/{family} {direction}: "
                f"WR={wr:.1%} ({count} trades) -> {mult:.2f}x"
            )

        return max(0.0, min(1.0, mult))

    def is_strategy_blocked(self, strategy: str, regime: str, direction: str = "LONG") -> bool:
        """
        Regime Kill Switch: returns True if strategy should be completely blocked.

        Two independent checks:
        1. Learner WR below threshold → block (negative expectancy)
        2. Bear-family + long direction → structural block (no longs in bear markets)
        """
        cfg = CONFIG.risk
        if not getattr(cfg, "regime_kill_switch_enabled", False):
            return False

        family = self._regime_family(regime)

        # Structural block: no longs in bear regime family
        if (
            getattr(cfg, "regime_kill_switch_bear_block_longs", True)
            and family == "bear"
            and direction.upper() == "LONG"
        ):
            logger.debug(f"[RegimeKillSwitch] BLOCK {strategy} LONG in {regime} (bear family)")
            return True

        # Data-driven block: WR below threshold
        min_trades = getattr(cfg, "regime_kill_switch_min_trades", 12)
        min_wr = getattr(cfg, "regime_kill_switch_min_wr", 0.35)

        if strategy not in self._strategy_regime_wr:
            return False  # no data → allow (benefit of the doubt)

        count_key = f"{strategy}::{family}"
        count = self._strategy_regime_count.get(count_key, 0)
        if count < min_trades:
            return False  # not enough data → allow

        wr = self._strategy_regime_wr[strategy].get(family, 0.50)
        if wr < min_wr:
            logger.debug(
                f"[RegimeKillSwitch] BLOCK {strategy}/{family}: WR={wr:.1%} < {min_wr:.0%} ({count} trades)"
            )
            return True

        return False

    def symbol_confidence(self, symbol: str) -> float:
        rec = self._symbols.get(symbol)
        return rec.confidence_modifier if rec else 0.0

    def stop_hit_rate(self, symbol: str) -> float:
        rec = self._symbols.get(symbol)
        if rec is None or rec.total_trades == 0:
            return 0.0
        return rec.stop_hit_rate

    def symbol_win_rate(self, symbol: str) -> float | None:
        rec = self._symbols.get(symbol)
        if rec is None or rec.total_trades < _MIN_TRADES:
            return None
        return rec.wins / rec.total_trades if rec.total_trades > 0 else 0.0

    def atr_multiplier_adj(self, symbol: str) -> float:
        rec = self._symbols.get(symbol)
        return rec.atr_multiplier_adjustment if rec else 0.0

    def flag_penalty_weight(self, flag_type: str) -> float:
        rec = self._flags.get(flag_type)
        return rec.penalty_weight if rec else 1.0

    def conviction_alignment_modifier(self, conviction: str) -> float:
        rec = self._convictions.get(conviction)
        return rec.alignment_score_modifier if rec else 0.0

    def get_learning_summary(self) -> dict:
        summary: dict = {
            "total_trades_observed": self._trade_count,
            "total_pnl_observed": round(self._total_pnl, 2),
            "strategy_insights": [],
            "symbol_insights": [],
            "flag_insights": [],
            "conviction_insights": [],
        }

        for strat, families in sorted(self._strategy_regime_wr.items()):
            for family, wr in families.items():
                count_key = f"{strat}::{family}"
                count = self._strategy_regime_count.get(count_key, 0)
                if count >= _MIN_TRADES:
                    mod = self.strategy_modifier(strat, family)
                    summary["strategy_insights"].append(
                        {
                            "key": f"{strat}::{family}",
                            "trades": count,
                            "win_rate": round(wr, 3),
                            "modifier": round(mod, 3),
                        }
                    )

        for sym, rec in sorted(self._symbols.items(), key=lambda x: x[1].rolling_win_rate):
            if rec.total_trades >= _MIN_TRADES:
                summary["symbol_insights"].append(
                    {
                        "symbol": sym,
                        "trades": rec.total_trades,
                        "win_rate": round(rec.rolling_win_rate, 3),
                        "stop_hit_rate": round(rec.stop_hit_rate, 3),
                        "conf_modifier": round(rec.confidence_modifier, 3),
                        "atr_adj": round(rec.atr_multiplier_adjustment, 3),
                        "total_pnl": round(rec.total_pnl, 2),
                    }
                )

        for ftype, rec in sorted(self._flags.items(), key=lambda x: -x[1].rolling_accuracy):
            if rec.total_flagged >= _MIN_TRADES:
                summary["flag_insights"].append(
                    {
                        "flag_type": ftype,
                        "flagged": rec.total_flagged,
                        "led_to_loss": rec.flag_led_to_loss,
                        "accuracy": round(rec.rolling_accuracy, 3),
                        "penalty_weight": round(rec.penalty_weight, 3),
                    }
                )

        for conv, rec in sorted(self._convictions.items(), key=lambda x: -x[1].rolling_win_rate):
            if rec.aligned_trades >= _MIN_TRADES:
                summary["conviction_insights"].append(
                    {
                        "conviction": conv,
                        "aligned_trades": rec.aligned_trades,
                        "win_rate": round(rec.rolling_win_rate, 3),
                        "modifier": round(rec.alignment_score_modifier, 3),
                        "total_pnl": round(rec.aligned_pnl, 2),
                    }
                )

        return summary

    def check_recovery_suppression(self) -> list[dict]:
        """
        Detect the 2023-H1 problem: EMA is still penalizing a strategy but
        recent trades suggest performance is already recovering.

        Returns a list of dicts for any (strategy, family) pairs where:
          - EMA WR < 0.45 (learner is applying a negative modifier)
          - Last N recent trades are >= 50% wins (actual performance improving)
          - Count >= 12 (past cold-start)

        The caller should log these as warnings so the operator knows the
        learner may be suppressing a strategy that has already turned around.
        """
        warnings = []
        for strat, families in self._strategy_regime_wr.items():
            for family, wr in families.items():
                count_key = f"{strat}::{family}"
                count = self._strategy_regime_count.get(count_key, 0)
                if count < 12:
                    continue
                if wr >= 0.45:
                    continue  # modifier is neutral/positive — no suppression

                buf = self._recent_outcomes.get(count_key, [])
                if len(buf) < 3:
                    continue  # not enough recent data

                recent_wr = sum(buf) / len(buf)
                modifier = max(-0.25, min(0.25, (wr - 0.50) * (0.25 / 0.15)))

                if recent_wr >= 0.50:
                    warnings.append(
                        {
                            "key": count_key,
                            "ema_wr": round(wr, 3),
                            "recent_wr": round(recent_wr, 3),
                            "recent_n": len(buf),
                            "modifier": round(modifier, 3),
                            "message": (
                                f"{count_key}: EMA WR={wr:.1%} (modifier={modifier:+.2f}) "
                                f"but last {len(buf)} trades = {recent_wr:.0%} wins. "
                                f"Learner may be suppressing a recovering strategy."
                            ),
                        }
                    )
        return warnings

    def log_monitoring_report(self, equity: float = 0.0) -> None:
        """
        Log a periodic adaptive learner status report at INFO level.
        Call from orchestrator every N bars (e.g., every 30 bars = ~1 month daily).

        Shows: current modifiers, suppressed strategies, recovery suppression risks.
        """
        logger.info("=" * 65)
        logger.info(
            f"[AdaptiveLearner] Monitoring Report | "
            f"Trades observed: {self._trade_count} | "
            f"Total PnL: ${self._total_pnl:+,.2f}" + (f" | Equity: ${equity:,.2f}" if equity else "")
        )

        if not self._strategy_regime_wr:
            logger.info("[AdaptiveLearner]   No strategy data yet (cold start)")
            logger.info("=" * 65)
            return

        # Per-strategy family modifier table
        for strat in sorted(self._strategy_regime_wr):
            for family in _ALL_FAMILIES:
                wr = self._strategy_regime_wr[strat].get(family, 0.50)
                count_key = f"{strat}::{family}"
                count = self._strategy_regime_count.get(count_key, 0)
                if count < 3:
                    continue
                modifier = max(-0.25, min(0.25, (wr - 0.50) * (0.25 / 0.15))) if count >= 12 else 0.0
                buf = self._recent_outcomes.get(count_key, [])
                recent_str = f"recent={sum(buf)}/{len(buf)}" if buf else "recent=n/a"
                flag = ""
                if modifier <= -0.15:
                    flag = " *** HEAVY PENALTY ***"
                elif modifier <= -0.08:
                    flag = " ** penalty active **"
                logger.info(
                    f"[AdaptiveLearner]   {count_key:<40} "
                    f"WR={wr:.1%} mod={modifier:+.3f} n={count} {recent_str}{flag}"
                )

        # Recovery suppression check
        suppressed = self.check_recovery_suppression()
        if suppressed:
            logger.warning(
                f"[AdaptiveLearner] RECOVERY SUPPRESSION RISK — "
                f"{len(suppressed)} strategy/family pair(s) may be over-penalized:"
            )
            for w in suppressed:
                logger.warning(f"[AdaptiveLearner]   {w['message']}")
        else:
            logger.info("[AdaptiveLearner]   No recovery suppression detected.")

        logger.info("=" * 65)

    def print_learning_report(self) -> None:
        s = self.get_learning_summary()
        print("+====================================================================+")
        print("  KA-MATS  Adaptive Learning Report (v17a — Regime-Partitioned)")
        print(f"  Observed: {s['total_trades_observed']} trades  |  P&L: ${s['total_pnl_observed']:+,.2f}")
        print("+--------------------------------------------------------------------+")
        if s["strategy_insights"]:
            print("  Strategy x Regime-Family  (win_rate -> modifier):")
            for r in s["strategy_insights"]:
                bar_fill = int(r["win_rate"] * 20)
                bar = "[" + "#" * bar_fill + " " * (20 - bar_fill) + "]"
                print(f"    {r['key']:<40} {bar}  {r['win_rate']:.0%}  mod={r['modifier']:+.3f}")
        print("+====================================================================+")

    # ─────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ─────────────────────────────────────────────────────────

    def save(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 3,  # v17a
            "saved_at": datetime.utcnow().isoformat()[:19],
            "trade_count": self._trade_count,
            "total_pnl": self._total_pnl,
            "strategy_regime_wr": self._strategy_regime_wr,
            "strategy_regime_count": self._strategy_regime_count,
            "last_trade_date": self._last_trade_date,
            "symbols": {k: asdict(v) for k, v in self._symbols.items()},
            "flags": {k: asdict(v) for k, v in self._flags.items()},
            "convictions": {k: asdict(v) for k, v in self._convictions.items()},
            "correlation": self.correlation_tracker.get_state(),
            "recent_outcomes": self._recent_outcomes,
        }
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.debug(
            f"[AdaptiveLearner] State saved (v17a) -> {self._state_file} ({self._trade_count} trades)"
        )

    def load(self) -> bool:
        if not self._state_file.exists():
            logger.info("[AdaptiveLearner] No prior state — starting fresh")
            return False
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            version = raw.get("version", 1)
            self._trade_count = raw.get("trade_count", 0)
            self._total_pnl = raw.get("total_pnl", 0.0)

            if version >= 3:
                # v17a native format
                self._strategy_regime_wr = raw.get("strategy_regime_wr", {})
                self._strategy_regime_count = raw.get("strategy_regime_count", {})
                self._last_trade_date = raw.get("last_trade_date", {})
            else:
                # v16 flat format: convert to nested
                # Old format had strategies dict with StrategyRecord-like dicts
                self._strategy_regime_wr = {}
                self._strategy_regime_count = {}
                self._last_trade_date = {}
                for key, rec in raw.get("strategies", {}).items():
                    # key was "strategy::regime" (e.g. "CrossSectionalMomentum::trending_up")
                    parts = key.split("::")
                    if len(parts) != 2:
                        continue
                    strat, old_regime = parts
                    family = REGIME_FAMILIES.get(old_regime, "sideways")
                    if strat not in self._strategy_regime_wr:
                        self._strategy_regime_wr[strat] = dict.fromkeys(_ALL_FAMILIES, 0.5)
                    # EMA-blend old WR into the family bucket
                    old_wr = rec.get("rolling_win_rate", 0.50)
                    old_count = rec.get("total_trades", 0)
                    ck = f"{strat}::{family}"
                    cur = self._strategy_regime_wr[strat][family]
                    # Weighted average into the family
                    self._strategy_regime_wr[strat][family] = (cur + old_wr) / 2.0
                    self._strategy_regime_count[ck] = self._strategy_regime_count.get(ck, 0) + old_count
                    if rec.get("last_updated"):
                        self._last_trade_date[ck] = rec["last_updated"]
                logger.info(
                    f"[AdaptiveLearner] Converted v{version} flat state → v17a nested "
                    f"({len(self._strategy_regime_wr)} strategies)"
                )

            self._symbols = {k: SymbolRecord(**v) for k, v in raw.get("symbols", {}).items()}
            self._flags = {k: FlagRecord(**v) for k, v in raw.get("flags", {}).items()}
            self._convictions = {k: ConvictionRecord(**v) for k, v in raw.get("convictions", {}).items()}

            # CorrelationTracker state (v17a+)
            corr_state = raw.get("correlation")
            if corr_state:
                self.correlation_tracker.load_state(corr_state)

            # Recent outcomes ring buffer (v17a+)
            saved_outcomes = raw.get("recent_outcomes")
            if saved_outcomes and isinstance(saved_outcomes, dict):
                self._recent_outcomes = {
                    k: v[-_RECOVERY_WINDOW:] for k, v in saved_outcomes.items() if isinstance(v, list)
                }
            logger.info(
                f"[AdaptiveLearner] State loaded (v{version}→v17a) | "
                f"{self._trade_count} prior trades | "
                f"{len(self._strategy_regime_wr)} strategy records"
            )
            return True
        except Exception as e:
            logger.warning(f"[AdaptiveLearner] Failed to load state: {e} — starting fresh")
            return False

    def reset(self) -> None:
        self._strategy_regime_wr.clear()
        self._strategy_regime_count.clear()
        self._last_trade_date.clear()
        self._symbols.clear()
        self._flags.clear()
        self._convictions.clear()
        self._trade_count = 0
        self._total_pnl = 0.0
        if self._state_file.exists():
            self._state_file.unlink()
        logger.warning("[AdaptiveLearner] State RESET — all learning cleared")

    # ─────────────────────────────────────────────────────────
    #  INTERNAL UPDATE HELPERS
    # ─────────────────────────────────────────────────────────

    def _update_strategy_regime(self, strategy: str, regime: str, win_val: float, now: str) -> None:
        family = self._regime_family(regime)

        if strategy not in self._strategy_regime_wr:
            self._strategy_regime_wr[strategy] = dict.fromkeys(_ALL_FAMILIES, 0.5)

        count_key = f"{strategy}::{family}"
        count = self._strategy_regime_count.get(count_key, 0) + 1

        old = self._strategy_regime_wr[strategy][family]
        alpha = _adaptive_alpha(count)
        new_wr = _ema(old, win_val, alpha=alpha)
        self._strategy_regime_wr[strategy][family] = new_wr

        self._strategy_regime_count[count_key] = count
        self._last_trade_date[count_key] = now[:10]

        # Update recent-outcome ring buffer
        buf = self._recent_outcomes.setdefault(count_key, [])
        buf.append(int(win_val))
        if len(buf) > _RECOVERY_WINDOW:
            buf.pop(0)

        # ── Monitoring: log threshold crossings ───────────────
        if count >= 12:  # only after cold-start period
            if new_wr < _WARN_WR_CRITICAL and old >= _WARN_WR_CRITICAL:
                logger.warning(
                    f"[AdaptiveLearner] CRITICAL: {count_key} WR dropped below "
                    f"{_WARN_WR_CRITICAL:.0%} → {new_wr:.1%}  "
                    f"(modifier ~{(new_wr - 0.5) * (0.25 / 0.15):+.2f}). "
                    f"Strategy will be heavily penalized. Recovery suppression risk HIGH."
                )
            elif new_wr < _WARN_WR_LOW and old >= _WARN_WR_LOW:
                logger.warning(
                    f"[AdaptiveLearner] WARN: {count_key} WR fell below "
                    f"{_WARN_WR_LOW:.0%} → {new_wr:.1%}  "
                    f"(modifier ~{(new_wr - 0.5) * (0.25 / 0.15):+.2f}). "
                    f"Sizing penalty now active. Monitor for recovery lag."
                )
            elif new_wr >= _WARN_WR_LOW and old < _WARN_WR_LOW:
                logger.info(
                    f"[AdaptiveLearner] RECOVERY: {count_key} WR recovered above "
                    f"{_WARN_WR_LOW:.0%} → {new_wr:.1%}. Penalty reducing."
                )

    def _update_symbol(
        self, symbol: str, pnl: float, win_val: float, stop_hit: bool, exit_reason: str, now: str
    ) -> None:
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolRecord(symbol=symbol)
        rec = self._symbols[symbol]
        rec.total_trades += 1
        rec.wins += int(win_val)
        rec.total_pnl += pnl
        rec.rolling_win_rate = _ema(rec.rolling_win_rate, win_val)
        rec.rolling_pnl = _ema(rec.rolling_pnl, pnl)
        if stop_hit:
            rec.stop_hits += 1
        if exit_reason == "take_profit":
            rec.take_profit_hits += 1
        rec.last_updated = now

    def _update_flag(self, flag_type: str, is_win: bool) -> None:
        if flag_type not in self._flags:
            self._flags[flag_type] = FlagRecord(flag_type=flag_type)
        rec = self._flags[flag_type]
        rec.total_flagged += 1
        was_accurate = 1.0 if not is_win else 0.0
        if not is_win:
            rec.flag_led_to_loss += 1
        rec.rolling_accuracy = _ema(rec.rolling_accuracy, was_accurate)

    def _update_conviction(self, conviction: str, pnl: float, win_val: float, now: str) -> None:
        if conviction not in self._convictions:
            self._convictions[conviction] = ConvictionRecord(conviction=conviction)
        rec = self._convictions[conviction]
        rec.aligned_trades += 1
        rec.aligned_wins += int(win_val)
        rec.aligned_pnl += pnl
        rec.rolling_win_rate = _ema(rec.rolling_win_rate, win_val)
        rec.last_updated = now


# ─────────────────────────────────────────────────────────────
#  CORRELATION TRACKER
# ─────────────────────────────────────────────────────────────


class CorrelationTracker:
    """
    Tracks pairwise same-bar win/loss co-occurrence across symbols.

    After each trade, call record_bar_outcome(bar_date, symbol, won).
    At each bar where trades close, call finalize_bar(bar_date) to compute
    pairwise co-occurrence and update rolling correlation estimates.

    Risk manager calls concentration_penalty(open_symbols) to get a [0,1]
    multiplier that reduces position size when too many correlated symbols
    are already open.

    Correlation is estimated via a simple co-loss EMA: for each pair (A, B),
    the tracker records whether both lost on the same bar. If the rolling
    co-loss rate exceeds a threshold, the pair is considered correlated.
    """

    _CO_LOSS_ALPHA = 0.25
    _CO_LOSS_THRESHOLD = 0.40  # above this → pair considered correlated
    _MAX_PENALTY = 0.40  # max sizing reduction (60% of normal)

    def __init__(self) -> None:
        # {bar_date_str: {symbol: bool(won)}}
        self._bar_buffer: dict[str, dict[str, bool]] = {}
        # {"SYM_A::SYM_B": co_loss_ema}  (keys sorted alphabetically)
        self._co_loss: dict[str, float] = {}
        self._pair_count: dict[str, int] = {}

    def record_bar_outcome(self, bar_date: str, symbol: str, won: bool) -> None:
        """Buffer a trade outcome for the given bar date."""
        day = bar_date[:10]
        if day not in self._bar_buffer:
            self._bar_buffer[day] = {}
        self._bar_buffer[day][symbol] = won

    def finalize_bar(self, bar_date: str) -> None:
        """Compute pairwise co-loss for all symbols that closed on this bar."""
        day = bar_date[:10]
        outcomes = self._bar_buffer.pop(day, {})
        if len(outcomes) < 2:
            return

        symbols = sorted(outcomes.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                key = f"{symbols[i]}::{symbols[j]}"
                both_lost = (not outcomes[symbols[i]]) and (not outcomes[symbols[j]])
                val = 1.0 if both_lost else 0.0
                old = self._co_loss.get(key, 0.0)
                self._co_loss[key] = self._CO_LOSS_ALPHA * val + (1.0 - self._CO_LOSS_ALPHA) * old
                self._pair_count[key] = self._pair_count.get(key, 0) + 1

    def concentration_penalty(self, open_symbols: list) -> float:
        """
        Returns a sizing multiplier in [1 - MAX_PENALTY, 1.0].
        If many open symbols are correlated (co-loss > threshold),
        reduce sizing for the next trade.
        """
        if len(open_symbols) < 2:
            return 1.0

        correlated_pairs = 0
        total_pairs = 0
        syms = sorted(open_symbols)
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                key = f"{syms[i]}::{syms[j]}"
                total_pairs += 1
                count = self._pair_count.get(key, 0)
                if count >= 3 and self._co_loss.get(key, 0.0) >= self._CO_LOSS_THRESHOLD:
                    correlated_pairs += 1

        if total_pairs == 0:
            return 1.0

        ratio = correlated_pairs / total_pairs
        penalty = ratio * self._MAX_PENALTY
        return max(1.0 - self._MAX_PENALTY, 1.0 - penalty)

    # ─────────────────────────────────────────────────────────
    #  v6 CROSS-ASSET CORRELATION LEARNING
    # ─────────────────────────────────────────────────────────

    _RETURN_CORR_ALPHA = 0.10  # EMA for return correlation
    _RETURN_CORR_WINDOW = 20  # bars to keep in rolling buffer
    _SECTOR_ROTATION_ALPHA = 0.12  # EMA for sector rotation tracker
    _HIGH_CORR_THRESHOLD = 0.70  # above this → pair returns highly correlated

    # Sector classification for crypto assets
    _SECTOR_MAP: dict[str, str] = {
        "BTC/USDT": "store_of_value",
        "ETH/USDT": "smart_contract_l1",
        "SOL/USDT": "smart_contract_l1",
        "BNB/USDT": "exchange",
        "AVAX/USDT": "smart_contract_l1",
        "LINK/USDT": "infrastructure",
        "DOT/USDT": "smart_contract_l1",
        "ADA/USDT": "smart_contract_l1",
        "DOGE/USDT": "meme",
        "UNI/USDT": "defi",
        "ATOM/USDT": "smart_contract_l1",
        "NEAR/USDT": "smart_contract_l1",
        "ARB/USDT": "l2",
        "OP/USDT": "l2",
        "POL/USDT": "l2",
    }

    def __init__(self) -> None:
        # Original co-loss tracking
        self._bar_buffer: dict[str, dict[str, bool]] = {}
        self._co_loss: dict[str, float] = {}
        self._pair_count: dict[str, int] = {}
        # v6: Return-based correlation
        self._return_buffer: dict[str, list[float]] = {}  # symbol → recent returns
        self._return_corr: dict[str, float] = {}  # "A::B" → rolling correlation EMA
        # v6: BTC correlation as macro proxy
        self._btc_corr: dict[str, float] = {}  # symbol → correlation with BTC
        # v6: Sector performance tracking
        self._sector_perf: dict[str, float] = {}  # sector → rolling avg return EMA
        self._sector_rank: list[str] = []  # sectors ranked by recent performance

    def record_bar_return(self, symbol: str, bar_return: float) -> None:
        """Record per-symbol return for cross-asset correlation computation."""
        buf = self._return_buffer.setdefault(symbol, [])
        buf.append(bar_return)
        if len(buf) > self._RETURN_CORR_WINDOW:
            buf.pop(0)

        # Update sector performance
        sector = self._SECTOR_MAP.get(symbol, "other")
        old_perf = self._sector_perf.get(sector, 0.0)
        self._sector_perf[sector] = (
            self._SECTOR_ROTATION_ALPHA * bar_return + (1.0 - self._SECTOR_ROTATION_ALPHA) * old_perf
        )

    def update_correlations(self) -> None:
        """
        Recompute pairwise return correlations and BTC beta for all symbols
        with sufficient history. Call once per bar after all returns recorded.
        """
        symbols = [s for s, buf in self._return_buffer.items() if len(buf) >= 10]
        if len(symbols) < 2:
            return

        # Pairwise return correlations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                key = f"{symbols[i]}::{symbols[j]}"
                corr = self._pearson(
                    self._return_buffer[symbols[i]],
                    self._return_buffer[symbols[j]],
                )
                if corr is not None:
                    old = self._return_corr.get(key, 0.0)
                    self._return_corr[key] = (
                        self._RETURN_CORR_ALPHA * corr + (1.0 - self._RETURN_CORR_ALPHA) * old
                    )

        # BTC correlation for each symbol (macro proxy)
        btc_returns = self._return_buffer.get("BTC/USDT")
        if btc_returns and len(btc_returns) >= 10:
            for sym in symbols:
                if sym == "BTC/USDT":
                    self._btc_corr[sym] = 1.0
                    continue
                corr = self._pearson(self._return_buffer[sym], btc_returns)
                if corr is not None:
                    old = self._btc_corr.get(sym, 0.5)
                    self._btc_corr[sym] = (
                        self._RETURN_CORR_ALPHA * corr + (1.0 - self._RETURN_CORR_ALPHA) * old
                    )

        # Update sector rankings
        self._sector_rank = sorted(
            self._sector_perf.keys(),
            key=lambda s: self._sector_perf[s],
            reverse=True,
        )

    def return_correlation_penalty(self, open_symbols: list, new_symbol: str) -> float:
        """
        Returns a sizing multiplier in [0.6, 1.0] based on return correlation
        between the proposed new_symbol and currently open positions.

        If new_symbol has high return correlation (>0.70) with many open symbols,
        adding it increases portfolio concentration risk → reduce sizing.
        """
        if not open_symbols or not new_symbol:
            return 1.0

        high_corr_count = 0
        for sym in open_symbols:
            key = "::".join(sorted([sym, new_symbol]))
            corr = self._return_corr.get(key, 0.0)
            if abs(corr) >= self._HIGH_CORR_THRESHOLD:
                high_corr_count += 1

        if high_corr_count == 0:
            return 1.0

        # Linearly penalize: 1 highly correlated → 0.90, 2 → 0.80, ..., 4+ → 0.60
        penalty = min(0.40, high_corr_count * 0.10)
        return max(0.60, 1.0 - penalty)

    def get_btc_correlation(self, symbol: str) -> float:
        """Get rolling BTC correlation for a symbol (macro sensitivity proxy)."""
        return self._btc_corr.get(symbol, 0.5)

    def get_sector_rotation_signal(self) -> dict[str, any]:
        """
        Return current sector rotation state.

        Returns dict with:
          - leading_sectors: top performing sectors
          - lagging_sectors: worst performing sectors
          - rotation_active: True if sector performance spread is wide
        """
        if len(self._sector_perf) < 3:
            return {"leading_sectors": [], "lagging_sectors": [], "rotation_active": False}

        sorted_sectors = self._sector_rank[:]
        perfs = [self._sector_perf[s] for s in sorted_sectors]

        spread = max(perfs) - min(perfs) if perfs else 0
        rotation_active = spread > 0.005  # 0.5% spread → rotation is active

        return {
            "leading_sectors": sorted_sectors[:2],
            "lagging_sectors": sorted_sectors[-2:],
            "rotation_active": rotation_active,
            "sector_performance": {s: round(self._sector_perf[s], 5) for s in sorted_sectors},
        }

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float | None:
        """Pearson correlation between two equal-length lists."""
        n = min(len(x), len(y))
        if n < 5:
            return None
        x, y = x[-n:], y[-n:]
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False)) / n
        sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
        sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
        if sx < 1e-12 or sy < 1e-12:
            return None
        return cov / (sx * sy)

    def get_state(self) -> dict:
        return {
            "co_loss": self._co_loss,
            "pair_count": self._pair_count,
            "return_corr": self._return_corr,
            "btc_corr": self._btc_corr,
            "sector_perf": self._sector_perf,
        }

    def load_state(self, state: dict) -> None:
        self._co_loss = state.get("co_loss", {})
        self._pair_count = state.get("pair_count", {})
        self._return_corr = state.get("return_corr", {})
        self._btc_corr = state.get("btc_corr", {})
        self._sector_perf = state.get("sector_perf", {})
        if self._sector_perf:
            self._sector_rank = sorted(
                self._sector_perf.keys(),
                key=lambda s: self._sector_perf[s],
                reverse=True,
            )


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────


def _ema(current: float, new_val: float, alpha: float = _EMA_ALPHA) -> float:
    return alpha * new_val + (1.0 - alpha) * current


def _adaptive_alpha(count: int) -> float:
    """Scale EMA alpha by trade count: smoother when cold, faster when warm.

    count < 12  → cold-start zone, keep base alpha (_EMA_ALPHA)
    12..50      → linearly ramp from _EMA_ALPHA toward _EMA_ALPHA + 0.08
    50+         → cap at _EMA_ALPHA + 0.08

    This means high-frequency strategy×regime pairs converge faster
    to their true WR, while low-frequency pairs remain conservatively smooth.
    """
    if count <= 12:
        return _EMA_ALPHA
    extra = min((count - 12) / 38.0, 1.0)  # 0→1 over trades 12→50
    return _EMA_ALPHA + extra * 0.08
