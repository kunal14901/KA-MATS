"""
KA-MATS · Thesis Agent
Iknir Capital — Phase I (v2.0 Blueprint)

ROLE: Macro Hypothesis Generation — "Aschenbrenner in a box"

Tracks and scores five Situational Awareness (Aschenbrenner) tradeable convictions:
  1. COMPUTE_DEMAND      — AI compute scales 100,000x; GPU/semiconductor demand accelerates
  2. POWER_INFRASTRUCTURE — Power is the binding constraint; utility & energy infra
  3. AI_DISRUPTION       — Cognitive industries face disruption; AI-native cos win
  4. GEOPOLITICAL_DEFENSE — US-China AGI arms race; defense + onshoring plays
  5. GOVERNMENT_PROJECT  — Government AGI project by 2027/28; cleared infra

Phase I scoring uses three proxy signals (no external feeds required):
  - Symbol alignment: Is this ticker in the SA conviction instrument list?
  - Regime proxy: Does the detected regime support each conviction's thesis?
  - Knowledge base hits: Are SA-themed documents loaded? (tag frequency)

Phase II additions (Aug 2026+):
  - GPU shipment data (NVIDIA quarterly + analyst estimates)
  - Power ISO interconnection queue filings
  - TSMC wafer allocation data (proxy: TSMC earnings guidance)
  - AI lab hiring trends + research paper output velocity
  - Export control monitoring (Commerce Dept BIS)

Safety: Thesis scores are ADVISORY ONLY.
        They inform the Adversarial Agent and flag conviction alignment.
        They cannot directly approve or veto signals.
"""

from __future__ import annotations

from loguru import logger

from core.models import (
    ConvictionScore,
    MarketSnapshot,
    RegimeAnalysis,
    RegimeType,
    SAConviction,
    ThesisContext,
)

# ─────────────────────────────────────────────────────────────
#  SA CONVICTION MAPS
# ─────────────────────────────────────────────────────────────

# SA conviction → aligned instruments (Phase I proxy list)
# Includes both equity proxies and crypto symbols from the trading universe.
# Crypto mappings rationale:
#   COMPUTE_DEMAND      → BTC (PoW hash-rate = real compute demand),
#                          SOL (high-throughput validator compute scaling),
#                          BNB (exchange infra compute)
#   POWER_INFRASTRUCTURE → BTC (most energy-intensive crypto, energy price proxy),
#                          ETH (post-merge staking infra, data-center demand)
#   AI_DISRUPTION       → LINK (decentralised oracle = AI data feeds),
#                          NEAR (NEAR AI initiative — AI agent platform),
#                          ARB/OP (L2 scaling for AI dApp transactions)
#   GEOPOLITICAL_DEFENSE → BTC/ETH (censorship-resistant capital flight asset;
#                           historically rallies on geopolitical stress events)
#   GOVERNMENT_PROJECT  → no direct crypto mapping; CBDC projects are not tradeable here
SA_INSTRUMENT_MAP: dict[SAConviction, list[str]] = {
    SAConviction.COMPUTE_DEMAND: [
        # Equities
        "NVDA",
        "AMD",
        "ASML",
        "TSM",
        "AVGO",
        "SMCI",
        "AMAT",
        "LRCX",
        "SOXX",
        "SOXL",
        "MU",
        "INTC",
        # Crypto
        "BTC/USDT",  # hash-rate = real compute demand signal
        "SOL/USDT",  # validator compute scaling
        "BNB/USDT",  # exchange infrastructure
    ],
    SAConviction.POWER_INFRASTRUCTURE: [
        # Equities
        "VST",
        "CEG",
        "NRG",
        "NEE",
        "AES",
        "GEV",
        "IREN",
        "CORZ",
        "CIFR",
        "HUT",
        # Crypto
        "BTC/USDT",  # most energy-intensive PoW — energy price proxy
        "ETH/USDT",  # post-merge staking infra, data-center hosting demand
    ],
    SAConviction.AI_DISRUPTION: [
        # Equities
        "PLTR",
        "AI",
        "MSFT",
        "GOOGL",
        "META",
        "SNOW",
        "MDB",
        "DDOG",
        "PATH",
        "BBAI",
        # Crypto
        "LINK/USDT",  # decentralised oracle = AI data feeds backbone
        "NEAR/USDT",  # NEAR AI initiative — AI agent execution platform
        "ARB/USDT",  # L2 scaling for AI dApp transactions
        "OP/USDT",  # Optimism — scaling for AI-native dApps
    ],
    SAConviction.GEOPOLITICAL_DEFENSE: [
        # Equities
        "LMT",
        "RTX",
        "NOC",
        "GD",
        "AXON",
        "CACI",
        "SAIC",
        "BAH",
        "LEIDOS",
        "DRS",
        # Crypto — censorship-resistant capital flight during sanctions/controls
        "BTC/USDT",  # primary geopolitical stress hedge in crypto
        "ETH/USDT",  # programmable money resilient to capital controls
    ],
    SAConviction.GOVERNMENT_PROJECT: [
        # Equities only — no direct crypto equivalent (CBDCs not tradeable here)
        "PLTR",
        "BAH",
        "LEIDOS",
        "CACI",
        "DXC",
        "SAIC",
    ],
}

# SA conviction → regimes that support the thesis directionally
SA_REGIME_ALIGNMENT: dict[SAConviction, list[RegimeType]] = {
    SAConviction.COMPUTE_DEMAND: [RegimeType.TRENDING_UP],
    SAConviction.POWER_INFRASTRUCTURE: [RegimeType.TRENDING_UP, RegimeType.RANGING],
    SAConviction.AI_DISRUPTION: [RegimeType.TRENDING_UP, RegimeType.VOLATILE],
    SAConviction.GEOPOLITICAL_DEFENSE: [RegimeType.VOLATILE, RegimeType.TRENDING_UP],
    SAConviction.GOVERNMENT_PROJECT: [RegimeType.RANGING, RegimeType.TRENDING_UP],
}

# SA conviction → knowledge base tag keywords
# These match against tags ingested via knowledge_ingest.py
SA_KNOWLEDGE_TAGS: dict[SAConviction, list[str]] = {
    SAConviction.COMPUTE_DEMAND: [
        "compute",
        "gpu",
        "semiconductor",
        "nvidia",
        "scaling",
        "oom",
        "chip",
        "tsmc",
        "cluster",
    ],
    SAConviction.POWER_INFRASTRUCTURE: [
        "power",
        "energy",
        "electricity",
        "data_center",
        "grid",
        "utility",
        "infrastructure",
        "cooling",
    ],
    SAConviction.AI_DISRUPTION: [
        "disruption",
        "automation",
        "cognitive",
        "saas",
        "agi",
        "intelligence",
        "ai_native",
    ],
    SAConviction.GEOPOLITICAL_DEFENSE: [
        "defense",
        "geopolitical",
        "china",
        "export",
        "chips_act",
        "onshoring",
        "national_security",
        "semiconductor_supply",
    ],
    SAConviction.GOVERNMENT_PROJECT: [
        "government",
        "dod",
        "darpa",
        "national_security",
        "cleared",
        "procurement",
        "military",
    ],
}


# ─────────────────────────────────────────────────────────────
#  THESIS AGENT
# ─────────────────────────────────────────────────────────────


class ThesisAgent:
    """
    Tracks Situational Awareness macro conviction scores.

    Runs after the Market Analyst Agent. Produces ThesisContext that feeds:
      - Adversarial Agent: checks signal alignment with dominant conviction
      - Orchestrator logs: tracks which SA themes are active

    Phase I scoring weights:
      40% — Symbol in SA conviction instrument list
      35% — Regime supports conviction (scaled by regime confidence)
      25% — Knowledge base tag hits (capped)
    """

    def __init__(self) -> None:
        logger.info("[ThesisAgent] Initialized — SA conviction tracker active (Phase I)")

    # ─────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────

    def score(
        self,
        snapshot: MarketSnapshot,
        regime: RegimeAnalysis,
        knowledge_tag_counts: dict[str, int] | None = None,
    ) -> ThesisContext:
        """
        Score all five SA convictions for the current symbol and market state.

        Args:
            snapshot: MarketSnapshot from Data Agent
            regime: RegimeAnalysis from Market Analyst Agent
            knowledge_tag_counts: Optional {tag: count} from loaded knowledge chunks

        Returns:
            ThesisContext — conviction scores, dominant conviction, symbol alignment
        """
        logger.debug(f"[ThesisAgent] Scoring {snapshot.symbol} | regime={regime.regime.value}")

        tag_counts = knowledge_tag_counts or {}
        scores: list[ConvictionScore] = []
        symbol_alignment: SAConviction | None = None

        for conviction in SAConviction:
            score, evidence, rationale = self._score_conviction(conviction, snapshot, regime, tag_counts)
            aligned = SA_INSTRUMENT_MAP.get(conviction, [])

            # Check symbol alignment (case-insensitive)
            if snapshot.symbol.upper() in [s.upper() for s in aligned]:
                symbol_alignment = conviction

            scores.append(
                ConvictionScore(
                    conviction=conviction,
                    score=score,
                    evidence_count=evidence,
                    aligned_instruments=aligned,
                    rationale=rationale,
                )
            )

        dominant = max(scores, key=lambda c: c.score) if scores else None
        overall = sum(s.score for s in scores) / len(scores) if scores else 0.5
        note = self._build_note(dominant, symbol_alignment, snapshot.symbol)

        logger.debug(
            f"[ThesisAgent] {snapshot.symbol} | "
            f"dominant={dominant.conviction.value if dominant else 'none'} "
            f"(score={dominant.score:.2f}) | "
            f"aligned={symbol_alignment.value if symbol_alignment else 'none'}"
        )

        return ThesisContext(
            timestamp=snapshot.timestamp,
            conviction_scores=scores,
            dominant_conviction=dominant.conviction if dominant else None,
            overall_thesis_strength=round(overall, 3),
            symbol_conviction_alignment=symbol_alignment,
            advisory_note=note,
        )

    # ─────────────────────────────────────────────────────────
    #  SCORING LOGIC
    # ─────────────────────────────────────────────────────────

    def _score_conviction(
        self,
        conviction: SAConviction,
        snapshot: MarketSnapshot,
        regime: RegimeAnalysis,
        tag_counts: dict[str, int],
    ) -> tuple[float, int, str]:
        """Score a single conviction. Returns (score, evidence_count, rationale)."""
        score = 0.0
        evidence = 0
        rationale_parts: list[str] = []

        # ── Factor 1: Symbol alignment (40% weight) ───────
        aligned = SA_INSTRUMENT_MAP.get(conviction, [])
        if snapshot.symbol.upper() in [s.upper() for s in aligned]:
            score += 0.40
            evidence += 1
            rationale_parts.append(f"{snapshot.symbol} is SA-aligned [{conviction.value}]")

        # ── Factor 2: Regime alignment (35% weight) ───────
        valid_regimes = SA_REGIME_ALIGNMENT.get(conviction, [])
        if regime.regime in valid_regimes:
            regime_bonus = 0.35 * regime.confidence
            score += regime_bonus
            evidence += 1
            rationale_parts.append(
                f"regime={regime.regime.value} supports thesis "
                f"(conf={regime.confidence:.2f}, bonus={regime_bonus:.2f})"
            )

        # ── Factor 3: Knowledge base tag hits (25% weight) ─
        tag_keywords = SA_KNOWLEDGE_TAGS.get(conviction, [])
        tag_hits = sum(tag_counts.get(tag, 0) for tag in tag_keywords)
        if tag_hits > 0:
            tag_score = min(0.25, tag_hits * 0.05)
            score += tag_score
            evidence += tag_hits
            rationale_parts.append(f"{tag_hits} knowledge tag hit(s) (score_contribution={tag_score:.2f})")

        rationale = "; ".join(rationale_parts) if rationale_parts else "No supporting evidence detected"
        return round(min(1.0, score), 3), evidence, rationale

    def _build_note(
        self,
        dominant: ConvictionScore | None,
        symbol_alignment: SAConviction | None,
        symbol: str,
    ) -> str:
        if dominant is None:
            return "No SA conviction dominance detected."
        parts = [f"Dominant SA conviction: {dominant.conviction.value} (score={dominant.score:.2f})"]
        if symbol_alignment:
            parts.append(f"{symbol} is an SA-aligned instrument [{symbol_alignment.value}]")
        else:
            parts.append(f"{symbol} not found in any SA conviction instrument list")
        return ". ".join(parts) + "."
