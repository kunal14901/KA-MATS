"""
KA-MATS · BM25 Experience Memory
Iknir Capital — Phase II (TradingAgents-inspired)

Lexical similarity memory store — no embeddings, no model load.
Stores (situation_text → outcome) pairs and retrieves the N most similar
past situations using BM25 scoring (Okapi BM25).

Inspired by TradingAgents' memory module: replay past market situations
to ground current decisions in empirical evidence rather than hardcoded rules.

Why BM25 instead of vector search?
  - Zero inference cost — no GPU / SentenceTransformer needed at every bar
  - Domain-specific tokens (regime names, strategy names, symbols) have exact
    lexical overlap — BM25 outperforms cosine for short structured texts
  - Works instantly with 0 trades; degrades gracefully with sparse history

MEMORY SCORING — REGIME-CONDITIONED EXPONENTIAL DECAY:
  TradingAgents uses BM25 alone (no time-awareness — stale lessons carry full weight).
  MiroFish uses a fixed recency_weight scalar (no regime-awareness).

  KA-MATS combines both insights:
    final_score = bm25_score × recency_weight × regime_relevance_weight

  recency_weight   = exp(-age_days / HALF_LIFE_DAYS)
    → Half-life of 180 days (6 months). A 2022 lesson at 3 years old has
      weight 0.018 — nearly zero in a neutral regime. But in a matching bear
      regime it's boosted back up via regime_relevance_weight.

  regime_relevance_weight: (MiroFish-inspired per-regime freshness multiplier)
    → SAME_REGIME     : 2.0  — a 2022 bear lesson is highly relevant in 2025 bear
    → RELATED_REGIME  : 1.0  — neutral, keep the lesson at face value
    → OPPOSITE_REGIME : 0.3  — 2022 bear lesson in 2025 bull run = mostly irrelevant

  This means:
    - No hard cutoff cliff edges (lessons never fully disappear)
    - Old lessons in MATCHING regimes stay influential (correct behaviour)
    - Old lessons in MISMATCHED regimes fade to near-zero (the core fix)
    - Recent lessons always carry full weight regardless of regime
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

from loguru import logger

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("[BM25Memory] rank_bm25 not installed — memory disabled. pip install rank-bm25")


@dataclass
class MemoryRecord:
    situation: str  # structured text describing the trading situation
    outcome: str  # natural-language summary of what happened
    pnl: float = 0.0  # numeric PnL (used for confidence weighting)
    regime: str = ""  # regime label at trade time
    strategy: str = ""  # which strategy fired
    trade_date: str = ""  # ISO date of trade exit (YYYY-MM-DD) — for decay calculation
    score: float = 0.0  # final composite score (query-time only, not persisted)


# ── Decay parameters ─────────────────────────────────────────────────────────
# Half-life: score halves every 180 days in a neutral/mismatched regime
HALF_LIFE_DAYS = 180

# Regime relevance multipliers (MiroFish-inspired per-context weighting)
# Regimes are grouped into families — bear, bull, sideways
_REGIME_FAMILIES: dict[str, str] = {
    "trending_up": "bull",
    "trending_down": "bear",
    "volatile": "bear",  # high vol correlates with bear stress
    "ranging": "sideways",
    "mean_reverting": "sideways",
}

# Score multiplier when current_regime family matches lesson's regime family
_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    # current → lesson
    "bull": {"bull": 2.0, "sideways": 1.0, "bear": 0.3},
    "bear": {"bear": 2.0, "sideways": 1.0, "bull": 0.3},
    "sideways": {"sideways": 2.0, "bull": 1.0, "bear": 1.0},
}


def _composite_weight(
    record: MemoryRecord,
    as_of_date: str = "",
    current_regime: str = "",
) -> float:
    """
    Compute the composite weight for a memory record:
        weight = recency_decay × regime_relevance

    recency_decay = exp(-age_days / HALF_LIFE_DAYS)
      → 0 days old  → 1.0
      → 180 days    → 0.5
      → 365 days    → 0.25
      → 730 days    → 0.063  (2 years, neutral regime: near zero)

    regime_relevance: 2.0 if lesson regime matches current, 0.3 if opposite.
      → A 3-year-old bear lesson in a current bear regime:
        0.063 × 2.0 = 0.126  (still meaningful)
      → A 3-year-old bear lesson in a current bull regime:
        0.063 × 0.3 = 0.019  (near zero — correctly ignored)
    """
    # Recency decay
    recency = 1.0
    if as_of_date and record.trade_date:
        try:
            age_days = (_date.fromisoformat(as_of_date) - _date.fromisoformat(record.trade_date)).days
            age_days = max(0, age_days)
            recency = math.exp(-age_days / HALF_LIFE_DAYS)
        except Exception:
            pass

    # Regime relevance
    regime_mult = 1.0
    if current_regime and record.regime:
        cur_family = _REGIME_FAMILIES.get(current_regime, "sideways")
        rec_family = _REGIME_FAMILIES.get(record.regime, "sideways")
        regime_mult = _REGIME_WEIGHTS.get(cur_family, {}).get(rec_family, 1.0)

    return recency * regime_mult


class BM25Memory:
    """
    BM25-based experience memory for trading situations.

    Stores past situation→outcome pairs and retrieves the N most similar
    past situations using Okapi BM25 lexical scoring.

    Usage:
        mem = BM25Memory(persist_path="knowledge/bm25_memory.json")
        mem.add("symbol SPY strategy DualMomentum regime trending_up", "WIN: ...", pnl=320.0)
        records = mem.query("symbol SPY regime trending_up strategy DualMomentum")
        mod, note = mem.confidence_modifier_from_memory("symbol SPY regime trending_up")
    """

    def __init__(
        self,
        persist_path: str | None = None,
        max_records: int = 2000,
    ) -> None:
        self._records: list[MemoryRecord] = []
        self._max_records = max_records
        self._persist_path = Path(persist_path) if persist_path else None
        self._bm25: BM25Okapi | None = None
        self._dirty = True  # index needs rebuild on next query

        if self._persist_path and self._persist_path.exists():
            self._load()

        logger.info(
            f"[BM25Memory] Initialized — {len(self._records)} record(s) loaded"
            + (f" from {self._persist_path}" if self._persist_path else " (in-memory only)")
        )

    # ─────────────────────────────────────────────────────────
    #  WRITE
    # ─────────────────────────────────────────────────────────

    def add(
        self,
        situation: str,
        outcome: str,
        pnl: float = 0.0,
        regime: str = "",
        strategy: str = "",
        trade_date: str = "",
    ) -> None:
        """Add a new experience record to memory."""
        record = MemoryRecord(
            situation=situation,
            outcome=outcome,
            pnl=pnl,
            regime=regime,
            strategy=strategy,
            trade_date=trade_date,
        )
        self._records.append(record)

        # Keep within limit — evict oldest records first
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

        self._dirty = True  # index will be rebuilt on next query

        if self._persist_path:
            self._save()

    # ─────────────────────────────────────────────────────────
    #  READ
    # ─────────────────────────────────────────────────────────

    def query(
        self,
        situation: str,
        n: int = 3,
        as_of_date: str = "",
        current_regime: str = "",
    ) -> list[MemoryRecord]:
        """
        Return the N most relevant past situations.

        Score = BM25_similarity × recency_decay × regime_relevance_weight

        No hard cutoffs — every record participates but old records in
        mismatched regimes score near-zero naturally. Old records in MATCHING
        regimes (e.g. 2022 bear lesson queried during 2025 bear) are boosted.

        as_of_date:     ISO date (YYYY-MM-DD) for recency decay calculation.
        current_regime: current regime label for regime-relevance weighting.
        """
        if not _BM25_AVAILABLE or not self._records:
            return []

        self._rebuild_if_dirty()
        if self._bm25 is None:
            return []

        tokens = self._tokenize(situation)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)

        # Apply composite weights: BM25 × recency_decay × regime_relevance
        composite: list[tuple[int, float]] = []
        for idx, bm25_score in enumerate(raw_scores):
            if bm25_score <= 0:
                continue  # no lexical overlap — skip entirely
            w = _composite_weight(self._records[idx], as_of_date, current_regime)
            composite.append((idx, bm25_score * w))

        if not composite:
            return []

        # Rank by composite score descending; take top-n
        composite.sort(key=lambda x: x[1], reverse=True)
        top_n = composite[:n]

        # Normalise so top hit = 1.0
        max_score = top_n[0][1] if top_n[0][1] > 0 else 1.0
        results: list[MemoryRecord] = []
        for idx, cscore in top_n:
            rec = self._records[idx]
            results.append(
                MemoryRecord(
                    situation=rec.situation,
                    outcome=rec.outcome,
                    pnl=rec.pnl,
                    regime=rec.regime,
                    strategy=rec.strategy,
                    trade_date=rec.trade_date,
                    score=round(cscore / max_score, 4),
                )
            )

        return results

    def confidence_modifier_from_memory(
        self,
        situation: str,
        n: int = 3,
        as_of_date: str = "",
        current_regime: str = "",
    ) -> tuple[float, str]:
        """
        Query memory and compute a confidence modifier from past outcomes.

        Returns (modifier, note) where modifier ∈ [-0.12, +0.12].

        Score weighting: BM25 × recency_decay × regime_relevance
        A 2022 bear loss in a 2025 bull market contributes near-zero weight.
        The same 2022 bear loss during a 2025 bear market contributes full weight.
        """
        records = self.query(situation, n=n, as_of_date=as_of_date, current_regime=current_regime)
        if not records:
            return 0.0, "No relevant memory found"

        total_weight = sum(r.score for r in records)
        if total_weight < 1e-9:
            return 0.0, "Memory scores too low — no modifier applied"

        weighted_pnl = sum(r.pnl * r.score for r in records) / total_weight
        wins = sum(1 for r in records if r.pnl > 0)
        win_rate = wins / len(records)

        # Map to ±0.12 modifier based on win rate deviation from 50%
        direction = 1.0 if weighted_pnl >= 0 else -1.0
        magnitude = min(0.12, abs(win_rate - 0.5) * 0.24)
        modifier = direction * magnitude

        note = (
            f"BM25Memory: {len(records)} similar situation(s) | "
            f"avg_pnl={weighted_pnl:+.1f} | WR={win_rate:.0%} -> mod={modifier:+.3f}"
        )
        logger.debug(f"[BM25Memory] {note}")
        return round(modifier, 3), note

    def __len__(self) -> int:
        return len(self._records)

    # ─────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────

    def _rebuild_if_dirty(self) -> None:
        if not self._dirty or not _BM25_AVAILABLE:
            return
        if not self._records:
            self._bm25 = None
            return
        corpus = [self._tokenize(r.situation) for r in self._records]
        self._bm25 = BM25Okapi(corpus)
        self._dirty = False
        logger.debug(f"[BM25Memory] BM25 index rebuilt ({len(self._records)} records)")

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def _save(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "situation": r.situation,
                    "outcome": r.outcome,
                    "pnl": r.pnl,
                    "regime": r.regime,
                    "strategy": r.strategy,
                    "trade_date": r.trade_date,
                }
                for r in self._records
            ]
            self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[BM25Memory] Failed to persist: {e}")

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._records = [
                MemoryRecord(
                    situation=d["situation"],
                    outcome=d.get("outcome", ""),
                    pnl=float(d.get("pnl", 0.0)),
                    regime=d.get("regime", ""),
                    strategy=d.get("strategy", ""),
                    trade_date=d.get("trade_date", ""),
                )
                for d in data
            ]
            self._dirty = True
            logger.info(f"[BM25Memory] Loaded {len(self._records)} records from {self._persist_path}")
        except Exception as e:
            logger.warning(f"[BM25Memory] Failed to load from {self._persist_path}: {e}")
            self._records = []
