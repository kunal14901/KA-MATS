"""
KA-MATS Cryptoz · Strategy Personas
Iknir Capital — Phase II

Inspired by MiroFish's OasisProfileGenerator — each strategy gets an empirical
behavioral profile built from actual trade history.

Each persona captures:
  - win_rate per regime
  - avg holding period tendency (aggressive/patient)
  - risk tolerance (derived from stop-hit rate)
  - best and worst regimes
  - overall health score

The Strategy Agent uses personas to select the BEST-FIT strategy for today's regime
instead of running all strategies blindly and letting the suppression gate filter later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from core.adaptive_learner import AdaptiveLearner

_STRATEGY_NAMES = [
    "CryptoMeanReversion",
    "CryptoCSM",
    "CryptoTrendPullback",
    "BTCDominanceRotation",
    "CryptoVolatilityDip",
]


@dataclass
class StrategyPersona:
    """
    Empirical behavioral profile of a trading strategy.
    Built from adaptive learner trade history — not hardcoded.
    """

    name: str

    # Win rates per regime (key = regime string)
    win_rate_by_regime: dict[str, float] = field(default_factory=dict)

    # Overall stats
    overall_win_rate: float = 0.50
    overall_modifier: float = 0.0

    # Behavioral traits (derived empirically)
    risk_tolerance: float = 0.5  # 0=very conservative, 1=aggressive
    stop_sensitivity: float = 0.5  # how often stops are hit (high = bad)
    regime_selectivity: float = 0.5  # how much performance varies across regimes

    # Best / worst regimes
    best_regime: str | None = None
    worst_regime: str | None = None

    # Health score [0,1] — overall trustworthiness
    health_score: float = 0.5

    # Data quality
    total_trades: int = 0
    data_confidence: float = 0.0  # 0 = no data, 1 = ≥30 trades

    def summary(self) -> str:
        best = self.best_regime or "unknown"
        worst = self.worst_regime or "unknown"
        return (
            f"{self.name}: win={self.overall_win_rate:.0%} "
            f"health={self.health_score:.2f} "
            f"best={best}({self.win_rate_by_regime.get(best, 0):.0%}) "
            f"worst={worst}({self.win_rate_by_regime.get(worst, 0):.0%})"
        )

    def win_rate_for(self, regime: str) -> float:
        """Return win rate for a regime, falling back to overall win rate."""
        return self.win_rate_by_regime.get(regime, self.overall_win_rate)

    def is_healthy(self, min_health: float = 0.40) -> bool:
        """True if this strategy is trustworthy enough to fire in current conditions."""
        return self.health_score >= min_health and self.data_confidence > 0.0


class StrategyPersonaManager:
    """
    Builds and manages strategy personas from adaptive learner data.

    Usage:
        manager = StrategyPersonaManager()
        manager.build(learner)

        best = manager.best_for_regime("trending_up")
        # → ("TrendFollowing", 0.72)

        for name, persona in manager.personas.items():
            print(persona.summary())
    """

    def __init__(self) -> None:
        self.personas: dict[str, StrategyPersona] = {}
        self._built = False

    # ─────────────────────────────────────────────────────────
    #  BUILD
    # ─────────────────────────────────────────────────────────

    def build(self, learner: AdaptiveLearner) -> None:
        """Build personas from adaptive learner data."""
        summary = learner.get_learning_summary()
        strategy_insights = summary.get("strategy_insights", [])
        symbol_insights = summary.get("symbol_insights", [])

        # Initialize blank personas
        self.personas = {name: StrategyPersona(name=name) for name in _STRATEGY_NAMES}

        # ── Fill win rates per regime ─────────────────────────
        for row in strategy_insights:
            key = row["key"]
            if "::" not in key:
                continue
            strat, regime = key.split("::", 1)
            if strat not in self.personas:
                continue

            persona = self.personas[strat]
            persona.win_rate_by_regime[regime] = float(row["win_rate"])

        # ── Calculate overall stats + traits ──────────────────
        for name, persona in self.personas.items():
            # Overall win rate = average across regimes
            if persona.win_rate_by_regime:
                persona.overall_win_rate = sum(persona.win_rate_by_regime.values()) / len(
                    persona.win_rate_by_regime
                )

            # Find overall modifier from strategy insights
            for row in strategy_insights:
                if row["key"].startswith(f"{name}::"):
                    persona.overall_modifier = float(row.get("modifier", 0))
                    break

            # Best / worst regime
            if persona.win_rate_by_regime:
                persona.best_regime = max(persona.win_rate_by_regime, key=persona.win_rate_by_regime.get)
                persona.worst_regime = min(persona.win_rate_by_regime, key=persona.win_rate_by_regime.get)

            # Risk tolerance: inverse of how much win rates vary
            if len(persona.win_rate_by_regime) > 1:
                rates = list(persona.win_rate_by_regime.values())
                spread = max(rates) - min(rates)
                persona.regime_selectivity = min(1.0, spread * 2.5)
            else:
                persona.regime_selectivity = 0.3  # unknown

            # Stop sensitivity: from symbol stop-hit rates (proxy)
            avg_stop = sum(r["stop_hit_rate"] for r in symbol_insights) / max(len(symbol_insights), 1)
            persona.stop_sensitivity = float(avg_stop)

            # Risk tolerance: high win rate + low stop sensitivity = higher tolerance
            persona.risk_tolerance = max(
                0.0, min(1.0, persona.overall_win_rate * 0.6 + (1 - persona.stop_sensitivity) * 0.4)
            )

            # Data confidence: need ≥30 trades for full confidence
            total_rows = sum(1 for r in strategy_insights if r["key"].startswith(f"{name}::"))
            estimated_trades = total_rows * 10  # rough estimate
            persona.total_trades = estimated_trades
            persona.data_confidence = min(1.0, estimated_trades / 30)

            # Health score: blend of win rate + modifier direction + data confidence
            wr_score = (persona.overall_win_rate - 0.30) / 0.40  # normalize [30%,70%] → [0,1]
            mod_score = (persona.overall_modifier + 0.15) / 0.30  # normalize [-0.15,+0.15] → [0,1]
            persona.health_score = max(
                0.0, min(1.0, wr_score * 0.5 + mod_score * 0.3 + persona.data_confidence * 0.2)
            )

        self._built = True
        logger.info(f"[StrategyPersonas] Built {len(self.personas)} personas from learner data")
        for p in self.personas.values():
            logger.debug(f"[StrategyPersonas]   {p.summary()}")

    # ─────────────────────────────────────────────────────────
    #  QUERY
    # ─────────────────────────────────────────────────────────

    def best_for_regime(self, regime: str) -> tuple[str, float] | None:
        """
        Return (strategy_name, win_rate) for the best-fit healthy strategy
        for the given regime. Returns None if no healthy strategy found.
        """
        if not self._built:
            return None

        candidates = []
        for name, persona in self.personas.items():
            if not persona.is_healthy():
                continue
            wr = persona.win_rate_for(regime)
            candidates.append((name, wr))

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x[1])
        return best if best[1] >= 0.40 else None

    def rank_for_regime(self, regime: str) -> list[tuple[str, float, float]]:
        """
        Return [(strategy_name, win_rate, health_score)] ranked by win rate for regime.
        Includes all strategies regardless of health (for inspection).
        """
        if not self._built:
            return []

        result = []
        for name, persona in self.personas.items():
            wr = persona.win_rate_for(regime)
            result.append((name, wr, persona.health_score))

        return sorted(result, key=lambda x: x[1], reverse=True)

    def get_persona(self, strategy_name: str) -> StrategyPersona | None:
        return self.personas.get(strategy_name)

    def all_summaries(self) -> list[str]:
        return [p.summary() for p in self.personas.values()]

    def report(self, regime: str) -> str:
        """Human-readable report for a given regime."""
        lines = [f"Strategy Personas for regime: {regime}", "-" * 50]
        for name, wr, health in self.rank_for_regime(regime):
            p = self.personas[name]
            status = "✓ ACTIVE" if p.is_healthy() else "✗ SUPPRESSED"
            lines.append(f"  {name:<22} win={wr:.0%}  health={health:.2f}  {status}")
        return "\n".join(lines)
