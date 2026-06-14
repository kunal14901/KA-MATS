"""
KA-MATS Cryptoz · Strategy Ensemble & Genetic Selector
Iknir Capital — v6 Enhancement (Vertus-inspired)

Generates a population of strategy parameter variants, evaluates them
via walk-forward backtest, and uses a genetic algorithm to evolve the
best-performing subset per regime.

Vertus runs 1M+ subsystems with genetic selection. This is our scaled
version: 50-100 micro-variants, monthly evolution, top-K survivors trade.

Usage:
    ensemble = StrategyEnsemble(learner=learner)
    ensemble.evolve(trade_history, current_regime)
    active = ensemble.get_active_strategies(regime="bull")
    # Returns List[StrategyGenome] — the top performers for the regime

Architecture:
    StrategyGenome: parameter set for a strategy variant
    StrategyEnsemble: population manager with genetic operators
    GeneticSelector: fitness evaluation + crossover + mutation
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

# ─────────────────────────────────────────────────────────────
#  STRATEGY GENOME
# ─────────────────────────────────────────────────────────────


@dataclass
class StrategyGenome:
    """
    A single strategy parameter variant.

    Each genome represents a unique combination of entry/exit parameters
    that can be backtested and scored independently.
    """

    genome_id: str
    base_strategy: str  # "CryptoTrendPullback" or "CryptoMomentumBreakout"

    # Entry parameters
    rsi_low: float = 38.0
    rsi_high: float = 57.0
    ema_fast: int = 20
    ema_slow: int = 50
    adx_min: float = 15.0
    volume_ratio_min: float = 0.5
    cross_rank_min: float = 0.45

    # Exit parameters
    atr_stop_mult: float = 2.5
    atr_target_mult: float = 11.0
    trail_activate_atr: float = 2.0
    trail_distance_atr: float = 1.0

    # Fitness scores (updated by evaluator)
    fitness: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.5
    trade_count: int = 0
    regime_fitness: dict[str, float] = field(default_factory=dict)

    # Metadata
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    is_active: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
#  PARAMETER RANGES (for mutation/randomization)
# ─────────────────────────────────────────────────────────────

_PARAM_RANGES = {
    "CryptoTrendPullback": {
        "rsi_low": (30.0, 45.0),
        "rsi_high": (50.0, 62.0),
        "ema_fast": (10, 30),
        "ema_slow": (40, 60),
        "adx_min": (12.0, 25.0),
        "volume_ratio_min": (0.3, 1.5),
        "cross_rank_min": (0.30, 0.60),
        "atr_stop_mult": (1.5, 4.0),
        "atr_target_mult": (6.0, 15.0),
        "trail_activate_atr": (1.0, 3.5),
        "trail_distance_atr": (0.5, 2.0),
    },
    "CryptoMomentumBreakout": {
        "rsi_low": (50.0, 60.0),
        "rsi_high": (68.0, 78.0),
        "ema_fast": (10, 25),
        "ema_slow": (40, 60),
        "adx_min": (15.0, 30.0),
        "volume_ratio_min": (1.0, 3.0),
        "cross_rank_min": (0.35, 0.60),
        "atr_stop_mult": (1.5, 3.0),
        "atr_target_mult": (3.5, 8.0),
        "trail_activate_atr": (1.0, 3.0),
        "trail_distance_atr": (0.5, 2.0),
    },
}


# ─────────────────────────────────────────────────────────────
#  GENETIC SELECTOR
# ─────────────────────────────────────────────────────────────


class GeneticSelector:
    """
    Genetic algorithm for strategy variant selection.

    Operators:
      - Tournament selection (k=3)
      - Single-point crossover
      - Gaussian mutation (5% per gene)
      - Elitism (top 20% survive unchanged)
    """

    MUTATION_RATE = 0.05  # probability of mutating each gene
    MUTATION_SIGMA = 0.10  # std dev as fraction of range
    CROSSOVER_RATE = 0.70  # probability of crossover
    TOURNAMENT_K = 3  # tournament selection size
    ELITISM_FRAC = 0.20  # top 20% pass through unchanged

    @staticmethod
    def tournament_select(population: list[StrategyGenome], k: int = 3) -> StrategyGenome:
        """Select the fittest from k random individuals."""
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda g: g.fitness)

    @staticmethod
    def crossover(
        parent_a: StrategyGenome, parent_b: StrategyGenome, child_id: str, generation: int
    ) -> StrategyGenome:
        """Single-point crossover between two parents."""
        child = deepcopy(parent_a)
        child.genome_id = child_id
        child.generation = generation
        child.parent_ids = [parent_a.genome_id, parent_b.genome_id]
        child.fitness = 0.0
        child.trade_count = 0
        child.is_active = False

        # Crossover point: randomly swap a subset of parameters
        params = list(_PARAM_RANGES.get(parent_a.base_strategy, {}).keys())
        if not params:
            return child

        crossover_point = random.randint(1, len(params) - 1)
        for param in params[crossover_point:]:
            setattr(child, param, getattr(parent_b, param))

        return child

    @staticmethod
    def mutate(genome: StrategyGenome) -> StrategyGenome:
        """Apply Gaussian mutation to each gene with probability MUTATION_RATE."""
        ranges = _PARAM_RANGES.get(genome.base_strategy, {})
        for param, (lo, hi) in ranges.items():
            if random.random() > GeneticSelector.MUTATION_RATE:
                continue

            current = getattr(genome, param)
            span = hi - lo
            sigma = span * GeneticSelector.MUTATION_SIGMA

            if isinstance(lo, int):
                new_val = int(round(current + random.gauss(0, sigma)))
                new_val = max(lo, min(hi, new_val))
            else:
                new_val = current + random.gauss(0, sigma)
                new_val = max(lo, min(hi, new_val))
                new_val = round(new_val, 2)

            setattr(genome, param, new_val)

        return genome


# ─────────────────────────────────────────────────────────────
#  STRATEGY ENSEMBLE
# ─────────────────────────────────────────────────────────────


class StrategyEnsemble:
    """
    Manages a population of strategy variants across generations.

    The ensemble:
      1. Initializes with random variants + the known-good baseline
      2. Evaluates fitness via trade outcomes per regime
      3. Evolves the population using genetic operators
      4. Returns the top-K active strategies for the current regime

    Persistence: saves/loads population to knowledge/.ensemble_state.json
    """

    _STATE_FILE = Path("knowledge/.ensemble_state.json")
    _POPULATION_SIZE = 50
    _TOP_K = 5  # active strategies per regime
    _MIN_TRADES_FOR_FITNESS = 10
    _EVOLVE_EVERY_N_TRADES = 50  # evolve after every 50 trades

    def __init__(self) -> None:
        self._population: list[StrategyGenome] = []
        self._generation: int = 0
        self._trades_since_evolve: int = 0
        self._trade_log: list[dict] = []  # recent trades for fitness eval
        self._selector = GeneticSelector()

    def initialize(self) -> None:
        """Create initial population: baseline + random variants."""
        if self._population:
            return  # already initialized

        # Known-good baselines from v5
        baselines = [
            StrategyGenome(
                genome_id="baseline_pullback_v5",
                base_strategy="CryptoTrendPullback",
                rsi_low=38.0,
                rsi_high=57.0,
                ema_fast=20,
                ema_slow=50,
                adx_min=15.0,
                volume_ratio_min=0.5,
                cross_rank_min=0.45,
                atr_stop_mult=2.5,
                atr_target_mult=11.0,
                trail_activate_atr=2.0,
                trail_distance_atr=1.0,
                generation=0,
            ),
            StrategyGenome(
                genome_id="baseline_breakout_v5",
                base_strategy="CryptoMomentumBreakout",
                rsi_low=55.0,
                rsi_high=72.0,
                ema_fast=20,
                ema_slow=50,
                adx_min=18.0,
                volume_ratio_min=1.3,
                cross_rank_min=0.40,
                atr_stop_mult=2.0,
                atr_target_mult=5.0,
                trail_activate_atr=2.0,
                trail_distance_atr=1.0,
                generation=0,
            ),
        ]
        self._population.extend(baselines)

        # Generate random variants
        for i in range(self._POPULATION_SIZE - len(baselines)):
            base = random.choice(["CryptoTrendPullback", "CryptoMomentumBreakout"])
            genome = self._random_genome(f"gen0_var{i}", base)
            self._population.append(genome)

        logger.info(
            f"[StrategyEnsemble] Initialized with {len(self._population)} variants "
            f"(2 baselines + {len(self._population) - 2} random)"
        )

    def _random_genome(self, genome_id: str, base_strategy: str) -> StrategyGenome:
        """Generate a random genome within parameter ranges."""
        ranges = _PARAM_RANGES.get(base_strategy, {})
        genome = StrategyGenome(genome_id=genome_id, base_strategy=base_strategy)

        for param, (lo, hi) in ranges.items():
            if isinstance(lo, int):
                val = random.randint(lo, hi)
            else:
                val = round(random.uniform(lo, hi), 2)
            setattr(genome, param, val)

        return genome

    def record_trade(
        self,
        genome_id: str,
        pnl: float,
        regime: str,
        strategy_name: str,
    ) -> None:
        """Record a trade outcome for a specific genome variant."""
        self._trade_log.append(
            {
                "genome_id": genome_id,
                "pnl": pnl,
                "regime": regime,
                "strategy_name": strategy_name,
            }
        )

        # Update fitness for the genome
        for g in self._population:
            if g.genome_id == genome_id:
                g.trade_count += 1
                g.win_rate = (g.win_rate * (g.trade_count - 1) + (1.0 if pnl > 0 else 0.0)) / g.trade_count

                # Regime-specific fitness
                if regime not in g.regime_fitness:
                    g.regime_fitness[regime] = 0.0
                g.regime_fitness[regime] = 0.92 * g.regime_fitness[regime] + 0.08 * (1.0 if pnl > 0 else 0.0)

                # Overall fitness: Sharpe-like metric
                g.fitness = self._compute_fitness(g)
                break

        self._trades_since_evolve += 1
        if self._trades_since_evolve >= self._EVOLVE_EVERY_N_TRADES:
            self.evolve()
            self._trades_since_evolve = 0

    def _compute_fitness(self, genome: StrategyGenome) -> float:
        """
        Fitness function combining win rate, trade count, and regime consistency.

        fitness = wr_score × sqrt(trade_count) × regime_consistency
        Heavily penalizes genomes with < MIN_TRADES (cold-start penalty).
        """
        if genome.trade_count < self._MIN_TRADES_FOR_FITNESS:
            return genome.win_rate * 0.1  # cold-start penalty

        wr_score = max(0, genome.win_rate - 0.40)  # only reward above 40% WR

        # Bonus for consistent performance across regimes
        regime_scores = list(genome.regime_fitness.values())
        regime_consistency = 1.0
        if len(regime_scores) >= 2:
            avg = sum(regime_scores) / len(regime_scores)
            variance = sum((s - avg) ** 2 for s in regime_scores) / len(regime_scores)
            regime_consistency = max(0.3, 1.0 - math.sqrt(variance))

        return wr_score * math.sqrt(genome.trade_count) * regime_consistency

    def evolve(self) -> None:
        """Run one generation of genetic evolution."""
        if len(self._population) < 4:
            return

        self._generation += 1
        pop = self._population
        pop.sort(key=lambda g: g.fitness, reverse=True)

        # Elitism: top 20% pass through
        elite_n = max(2, int(len(pop) * GeneticSelector.ELITISM_FRAC))
        new_pop = [deepcopy(g) for g in pop[:elite_n]]

        # Fill the rest via tournament selection + crossover + mutation
        while len(new_pop) < self._POPULATION_SIZE:
            parent_a = GeneticSelector.tournament_select(pop)
            parent_b = GeneticSelector.tournament_select(pop)

            child_id = f"gen{self._generation}_var{len(new_pop)}"

            if random.random() < GeneticSelector.CROSSOVER_RATE:
                child = GeneticSelector.crossover(parent_a, parent_b, child_id, self._generation)
            else:
                child = deepcopy(parent_a)
                child.genome_id = child_id
                child.generation = self._generation
                child.fitness = 0.0
                child.trade_count = 0

            child = GeneticSelector.mutate(child)
            new_pop.append(child)

        self._population = new_pop[: self._POPULATION_SIZE]

        # Mark top-K as active
        self._update_active_strategies()

        logger.info(
            f"[StrategyEnsemble] Generation {self._generation} evolved | "
            f"Top fitness: {pop[0].fitness:.3f} ({pop[0].genome_id}) | "
            f"Active: {sum(1 for g in self._population if g.is_active)}"
        )

    def _update_active_strategies(self) -> None:
        """Mark top-K genomes per base strategy as active."""
        for g in self._population:
            g.is_active = False

        # Group by base strategy and pick top-K per group
        by_strategy: dict[str, list[StrategyGenome]] = {}
        for g in self._population:
            by_strategy.setdefault(g.base_strategy, []).append(g)

        for _strat, genomes in by_strategy.items():
            genomes.sort(key=lambda g: g.fitness, reverse=True)
            for g in genomes[: self._TOP_K]:
                g.is_active = True

    def get_active_strategies(self, regime: str = None) -> list[StrategyGenome]:
        """
        Return currently active strategy genomes.
        If regime is specified, sort by regime-specific fitness.
        """
        active = [g for g in self._population if g.is_active]
        if regime and active:
            active.sort(
                key=lambda g: g.regime_fitness.get(regime, g.fitness),
                reverse=True,
            )
        return active

    def get_genome_params(self, genome_id: str) -> StrategyGenome | None:
        """Get a specific genome by ID."""
        for g in self._population:
            if g.genome_id == genome_id:
                return g
        return None

    def get_best_params(self, base_strategy: str) -> StrategyGenome | None:
        """Get the best-performing genome for a given base strategy."""
        candidates = [
            g
            for g in self._population
            if g.base_strategy == base_strategy and g.trade_count >= self._MIN_TRADES_FOR_FITNESS
        ]
        if not candidates:
            # Fall back to baselines
            candidates = [g for g in self._population if g.base_strategy == base_strategy]
        if not candidates:
            return None
        return max(candidates, key=lambda g: g.fitness)

    # ─────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ─────────────────────────────────────────────────────────

    def save(self) -> None:
        self._STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "generation": self._generation,
            "trades_since_evolve": self._trades_since_evolve,
            "population": [g.to_dict() for g in self._population],
        }
        self._STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.debug(
            f"[StrategyEnsemble] Saved generation {self._generation} ({len(self._population)} genomes)"
        )

    def load(self) -> bool:
        if not self._STATE_FILE.exists():
            return False
        try:
            raw = json.loads(self._STATE_FILE.read_text(encoding="utf-8"))
            self._generation = raw.get("generation", 0)
            self._trades_since_evolve = raw.get("trades_since_evolve", 0)
            self._population = []
            for gd in raw.get("population", []):
                g = StrategyGenome(
                    **{k: v for k, v in gd.items() if k in StrategyGenome.__dataclass_fields__}
                )
                self._population.append(g)
            logger.info(
                f"[StrategyEnsemble] Loaded generation {self._generation} "
                f"with {len(self._population)} genomes"
            )
            return True
        except Exception as e:
            logger.warning(f"[StrategyEnsemble] Failed to load state: {e}")
            return False

    def get_summary(self) -> dict:
        """Summary for monitoring dashboard."""
        active = [g for g in self._population if g.is_active]
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "active_count": len(active),
            "top_genomes": [
                {
                    "id": g.genome_id,
                    "strategy": g.base_strategy,
                    "fitness": round(g.fitness, 3),
                    "win_rate": round(g.win_rate, 3),
                    "trades": g.trade_count,
                }
                for g in sorted(self._population, key=lambda x: x.fitness, reverse=True)[:10]
            ],
        }
