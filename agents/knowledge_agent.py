"""
KA-MATS Cryptoz · Knowledge Agent
Iknir Capital — Phase I.2

Crypto-native RAG pipeline + hardcoded rules from research literature.

Architecture (TradingView-MCP-inspired multi-source synthesis):
  ┌─ Source 1: FAISS semantic RAG ────── literature chunks (regime-aware)
  ├─ Source 2: Funding rate advisory ─── Auer 2023 / Liu-Tsyvinski 2021
  ├─ Source 3: BTC dominance signal ──── Liu-Tsyvinski 2021 factor model
  ├─ Source 4: Basis/perp advisory ───── Avellaneda 2020
  ├─ Source 5: Momentum regime gate ──── Liu 2022 / Liu 2022b
  └─ Source 6: BM25 experience memory ── TradingAgents-inspired replay
  → Synthesised into KnowledgeContext (single advisory output)

  Mirrors TradingView-MCP's layered query pattern:
    quote_get → data_get_study_values → data_get_pine_lines → synthesis
  Our equivalent:
    snapshot  → regime               → alt_data            → KnowledgeContext

Grounded in crypto-specific research papers:
  [P11]  Liu, Tsyvinski, Wu (2022) — Momentum in Cryptocurrency Markets
  [P12]  Liu, Tsyvinski (2021)     — Crypto Carry (funding mechanics)
  [P14]  Liu, Tsyvinski (2021)     — Cross-Section of Crypto Returns
  [P15]  Auer et al. (2023)        — Funding Rates Predict Returns
  [P25]  Glassnode (2022)          — On-Chain Indicators as Trading Signals
  [P26]  Avellaneda, Stoikov (2020) — Perpetual Futures Basis Trading
  [P49]  Liu et al. (2022b)        — Cross-Sectional Momentum in Crypto

Safety constraint §9.1 — ENFORCED:
  Knowledge influence is ADVISORY ONLY — never authoritative.
  Trades CANNOT be initiated purely from retrieved text.
  knowledge_only_veto = True is ALWAYS set.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from loguru import logger

from core.bm25_memory import _REGIME_FAMILIES as REGIME_FAMILIES
from core.bm25_memory import BM25Memory
from core.models import (
    AltDataContext,
    CandidateSignal,
    KnowledgeChunk,
    KnowledgeContext,
    MarketSnapshot,
    RegimeAnalysis,
    RegimeType,
    SignalDirection,
)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

_MODEL_NAME = "all-MiniLM-L6-v2"
_TOP_K = 5

# Knowledge base lives in the shared KA-MATS knowledge folder.
# Resolved relative to this file: ../KA-MATS/knowledge/
_HERE = Path(__file__).resolve().parent  # agents/
_KNOWLEDGE_DIR = _HERE.parent.parent / "KA-MATS" / "knowledge"

# FAISS cache stored inside the crypto project's knowledge folder
_CACHE_DIR = _HERE.parent / "knowledge" / ".vector_cache"
_FAISS_INDEX_FILE = _CACHE_DIR / "faiss.index"
_CHUNKS_FILE = _CACHE_DIR / "chunks.pkl"

# Process-level model cache — loaded once, reused across bars/backtest periods
_SENTENCE_MODEL_CACHE = None


def _get_sentence_model(model_name: str):
    global _SENTENCE_MODEL_CACHE
    if _SENTENCE_MODEL_CACHE is None:
        import logging

        from sentence_transformers import SentenceTransformer

        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logger.info(f"[KnowledgeAgent] Loading model: {model_name} (once per process)")
        _SENTENCE_MODEL_CACHE = SentenceTransformer(model_name)
    return _SENTENCE_MODEL_CACHE


# ─────────────────────────────────────────────────────────────
#  FUNDING RATE ADVISORY TABLES  [Auer 2023, Liu-Tsyvinski 2021]
# ─────────────────────────────────────────────────────────────
# funding_rate is per 8h (0.01% = 0.0001)

_FUNDING_VETO_LONG_THRESHOLD = 0.0010  # > 0.10% per 8h → veto all longs
_FUNDING_REDUCE_LONG_60_THRESH = 0.0005  # > 0.05%        → reduce longs 60%
_FUNDING_REDUCE_LONG_30_THRESH = 0.0002  # > 0.02%        → reduce longs 30%
_FUNDING_VETO_SHORT_THRESHOLD = -0.0005  # < -0.05%       → veto all shorts

# BTC dominance thresholds  [Liu-Tsyvinski 2021 factor model]
_BTC_DOM_RISK_OFF = 55.0  # dominance >= 55% → prefer BTC over alts
_BTC_DOM_RISK_ON = 42.0  # dominance <= 42% → altcoin season


# ─────────────────────────────────────────────────────────────
#  CRYPTO KNOWLEDGE AGENT
# ─────────────────────────────────────────────────────────────


class CryptoKnowledgeAgent:
    """
    Crypto-native Knowledge Agent.

    Three-tier synthesis (TradingView-MCP-inspired layered sources):
      Tier 1 — DEEP:  FAISS semantic RAG over 100+ research papers
      Tier 2 — RULES: Crypto-specific hardcoded advisory rules from papers
      Tier 3 — QUICK: BM25 experience memory (past trade replay)

    Outputs KnowledgeContext — advisory, never authoritative.

    Usage:
        agent = CryptoKnowledgeAgent()
        ctx = agent.query(
            regime=regime_analysis,
            signals=candidate_signals,
            snapshot=market_snapshot,      # for funding rate
            alt_data=alt_data_context,     # for BTC dominance
        )
    """

    # ── v17c adaptive blend weights (same logic as equity) ───────────────────
    _MATURITY_THRESHOLDS = {"bull": 50, "bear": 30, "sideways": 40}
    _BM25_RELEVANCE_FLOOR = 0.10

    def __init__(
        self,
        bm25_memory: BM25Memory | None = None,
        knowledge_dir: str | None = None,
    ) -> None:
        self._loaded = False
        self._chunks: list[KnowledgeChunk] = []
        self._embeddings: np.ndarray | None = None
        self._faiss_index = None
        self._model = None

        self._bm25_memory: BM25Memory = bm25_memory or BM25Memory(
            persist_path="knowledge/bm25_memory.json",
            max_records=2000,
        )

        base = Path(knowledge_dir) if knowledge_dir else _KNOWLEDGE_DIR
        self._try_load(base)

    # ─────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────

    def query(
        self,
        regime: RegimeAnalysis,
        signals: list[CandidateSignal],
        snapshot: MarketSnapshot | None = None,
        alt_data: AltDataContext | None = None,
    ) -> KnowledgeContext:
        """
        Full three-tier query.

        Tier 1: FAISS RAG — literature-grounded regime advisory
        Tier 2: Crypto rules — funding, dominance, basis, momentum gate
        Tier 3: BM25 memory — past trade experience replay

        Returns synthesised KnowledgeContext (advisory only).
        """
        logger.debug(
            f"[KnowledgeAgent] Query regime={regime.regime.value} "
            f"signals={len(signals)}" + (f" symbol={snapshot.symbol}" if snapshot else "")
        )

        # ── Tier 1: RAG (DEEP) ────────────────────────────────────────────────
        if self._loaded:
            ctx = self._rag_query(regime, signals)
        else:
            ctx = self._neutral_context(regime)

        # ── Tier 2: Crypto-specific hardcoded rules ───────────────────────────
        ctx = self._apply_crypto_rules(ctx, regime, signals, snapshot, alt_data)

        # ── Tier 3: BM25 experience memory (QUICK) ────────────────────────────
        ctx = self._augment_with_bm25(ctx, regime, signals)

        return ctx

    def query_signal_quick(
        self,
        regime: RegimeAnalysis,
        signal: CandidateSignal,
    ) -> tuple[float, str]:
        """
        QUICK tier only — BM25 memory lookup per signal.
        No FAISS search, no model load. ~0ms.
        """
        situation = self._build_situation_string(regime, signal)
        as_of = str(regime.timestamp)[:10] if regime.timestamp else ""
        mod, note = self._bm25_memory.confidence_modifier_from_memory(
            situation, as_of_date=as_of, current_regime=regime.regime.value
        )
        return mod, note

    # ─────────────────────────────────────────────────────────
    #  TIER 1: NEUTRAL CONTEXT (no RAG)
    # ─────────────────────────────────────────────────────────

    def _neutral_context(self, regime: RegimeAnalysis) -> KnowledgeContext:
        """
        Rule-based advisory active even without RAG docs.

        Regime modifiers grounded in crypto literature:
          trending_up   → +0.07  [P11 P49] momentum strongest, Sharpe ~1.4
          trending_down → -0.09  [P14] BTC beta dominates in bear; longs fail
          volatile      → -0.12  [P15] funding spike + vol = liquidation cascade risk
          mean_reverting→ +0.05  [P25] SOPR dip-to-1.0 / oversold RSI = bounce
          ranging       → +0.02  [P14] attention factor works; momentum weak
        """
        _MODS = {
            RegimeType.TRENDING_UP: (
                +0.07,
                "[P11,P49] Crypto momentum strongest in bull trend — XS momentum Sharpe ~1.4. "
                "Volume-confirmed entries preferred (>150% 20d avg vol).",
            ),
            RegimeType.TRENDING_DOWN: (
                -0.09,
                "[P14] BTC beta dominates in bear — all alts correlated. "
                "Avoid new alt longs; BTCDominanceRotation is the active strategy.",
            ),
            RegimeType.VOLATILE: (
                -0.12,
                "[P15] High funding + vol = liquidation cascade risk (45% probability). "
                "CryptoVolatilityDip ONLY if RSI < 25. Tighten stops, reduce size 50%.",
            ),
            RegimeType.MEAN_REVERTING: (
                +0.05,
                "[P25] SOPR dip-to-1.0 = temporary profit-taking; BTC/ETH bounce likely. "
                "[P14] Attention factor predicts 1-3d reversal. Confirm: RSI < 32 + BB_lower.",
            ),
            RegimeType.RANGING: (
                +0.02,
                "[P14] Attention factor strongest in ranging; momentum weak (Sharpe ~0.6). "
                "CryptoMeanReversion preferred. Avoid new CSM entries.",
            ),
        }
        modifier, note = _MODS.get(regime.regime, (0.0, "Regime UNKNOWN — neutral."))
        return KnowledgeContext(
            query_regime=regime.regime,
            retrieved_chunks=[],
            strategy_bias=None,
            confidence_modifier=modifier,
            suggested_constraints=[],
            advisory_note=f"[CryptoLitAdvisory] {note}",
            knowledge_only_veto=True,
        )

    # ─────────────────────────────────────────────────────────
    #  TIER 2: CRYPTO-SPECIFIC HARDCODED RULES
    # ─────────────────────────────────────────────────────────

    def _apply_crypto_rules(
        self,
        ctx: KnowledgeContext,
        regime: RegimeAnalysis,
        signals: list[CandidateSignal],
        snapshot: MarketSnapshot | None,
        alt_data: AltDataContext | None,
    ) -> KnowledgeContext:
        """
        Apply crypto-native advisory rules from research papers.

        Sources consulted per rule:
          Rule A: Funding rate veto/reduction    [Auer 2023, Liu-Tsyvinski 2021]
          Rule B: BTC dominance regime signal    [Liu-Tsyvinski 2021 factor model]
          Rule C: Basis/premium advisory         [Avellaneda 2020]
          Rule D: Momentum regime gate           [Liu 2022, Liu 2022b]
          Rule E: Vol-scaled sizing advisory     [Liu 2022 vol-scaling section]

        TradingView-MCP pattern applied here:
          Each rule is an independent "source" (like tv-mcp's pine_lines, study_values etc.)
          Each contributes independently to the final modifier.
          Final synthesis = weighted sum, capped at ±0.20.
        """
        modifier_delta = 0.0
        constraints: list[str] = list(ctx.suggested_constraints)
        notes: list[str] = [ctx.advisory_note]
        bias = ctx.strategy_bias

        is_long_signal = any(s.direction == SignalDirection.BUY for s in signals)
        is_short_signal = any(s.direction == SignalDirection.SELL for s in signals)

        # ── Rule A: Funding Rate  [Auer 2023, Liu-Tsyvinski 2021] ────────────
        # Funding rate is the most important real-time risk signal in crypto.
        # Predictive horizon: strongest 1-24h, significant up to 3 days.
        funding_rate = None
        if snapshot and snapshot.funding_rate is not None:
            funding_rate = snapshot.funding_rate
        elif alt_data:
            # Extract from alt data signals if available
            for sig in alt_data.signals:
                if sig.signal_type == "funding_rate" and sig.value is not None:
                    funding_rate = float(sig.value) / 100.0  # convert % to decimal
                    break

        if funding_rate is not None:
            if funding_rate > _FUNDING_VETO_LONG_THRESHOLD:
                # VETO all longs — extreme overleveraging [Auer 2023 Table 1]
                modifier_delta -= 0.20
                constraints.append(
                    f"FUNDING_VETO: funding={funding_rate:.4f}/8h (>{_FUNDING_VETO_LONG_THRESHOLD:.4f}) — "
                    "VETO all longs. 32% probability of >3% drawdown in 24h [Auer 2023]."
                )
                notes.append(
                    f"[P15-CRITICAL] Funding {funding_rate:.4%}/8h EXTREME — "
                    "longs at severe liquidation cascade risk. Exit all longs."
                )
                if is_long_signal:
                    bias = SignalDirection.SELL
            elif funding_rate > _FUNDING_REDUCE_LONG_60_THRESH:
                modifier_delta -= 0.10
                constraints.append(
                    f"FUNDING_HIGH: funding={funding_rate:.4f}/8h — reduce long size 60% [Auer 2023]."
                )
                notes.append(f"[P15] Funding {funding_rate:.4%}/8h elevated — reduce long exposure 60%.")
            elif funding_rate > _FUNDING_REDUCE_LONG_30_THRESH:
                modifier_delta -= 0.05
                constraints.append(
                    f"FUNDING_MILD: funding={funding_rate:.4f}/8h — reduce long size 30% [Auer 2023]."
                )
                notes.append(f"[P15] Funding {funding_rate:.4%}/8h mildly high — reduce longs 30%.")
            elif funding_rate < _FUNDING_VETO_SHORT_THRESHOLD:
                # Negative funding = short squeeze environment [Liu-Tsyvinski 2021]
                modifier_delta += 0.08
                constraints.append(
                    f"FUNDING_NEGATIVE: funding={funding_rate:.4f}/8h — "
                    "short squeeze risk. Negative carry supports longs [Liu-Tsyvinski 2021]."
                )
                notes.append(
                    f"[P12] Funding {funding_rate:.4%}/8h negative — short squeeze risk, supports longs."
                )
                if is_long_signal:
                    bias = SignalDirection.BUY
            else:
                notes.append(f"[P15] Funding {funding_rate:.4%}/8h neutral — no funding override.")

        # ── Rule B: BTC Dominance  [Liu-Tsyvinski 2021 factor model] ──────────
        # Rising BTC dominance = risk-off, capital flees to BTC from alts.
        # 60-80% of altcoin return variance explained by BTC beta.
        btc_dominance: float | None = None
        if alt_data:
            for sig in alt_data.signals:
                if "btc_dominance" in sig.signal_type and sig.value is not None:
                    btc_dominance = float(sig.value)
                    break

        if btc_dominance is not None:
            symbol = snapshot.symbol if snapshot else ""
            is_alt = symbol not in {"BTC/USDT", "ETH/USDT", "BNB/USDT"}

            if btc_dominance >= _BTC_DOM_RISK_OFF and is_alt and is_long_signal:
                # Risk-off: penalise alt longs [Liu-Tsyvinski 2021]
                modifier_delta -= 0.07
                constraints.append(
                    f"BTC_DOM_RISKOFF: BTC dominance {btc_dominance:.1f}% (>={_BTC_DOM_RISK_OFF}%) — "
                    "capital concentrated in BTC; altcoin long signals penalised [P14]."
                )
                notes.append(
                    f"[P14] BTC dominance {btc_dominance:.1f}% risk-off — "
                    "prefer BTC/ETH over alts (BTCDominanceRotation strategy)."
                )
            elif btc_dominance <= _BTC_DOM_RISK_ON and is_alt and is_long_signal:
                # Altcoin season: boost alt longs [Liu-Tsyvinski 2021]
                modifier_delta += 0.06
                constraints.append(
                    f"BTC_DOM_ALTSEASON: BTC dominance {btc_dominance:.1f}% (<={_BTC_DOM_RISK_ON}%) — "
                    "altcoin season; CryptoCSM and TrendPullback signals boosted [P14]."
                )
                notes.append(
                    f"[P14] BTC dominance {btc_dominance:.1f}% — altcoin season. "
                    "Capital rotating to alts; CSM momentum more reliable."
                )

        # ── Rule C: Basis / Premium Advisory  [Avellaneda 2020] ──────────────
        # Extreme perp basis signals mean-reversion, NOT momentum continuation.
        # basis_zscore > 2.0 → suppress longs (basis likely to compress = price reversal)
        if snapshot and snapshot.mark_price is not None:
            close_price = snapshot.price.close
            if close_price > 0:
                basis_pct = (snapshot.mark_price - close_price) / close_price * 100
                # Simple threshold — no zscore since we don't carry history here
                if basis_pct > 0.5 and is_long_signal:
                    modifier_delta -= 0.06
                    constraints.append(
                        f"BASIS_EXTREME: perp basis={basis_pct:.2f}% (>0.5%) — "
                        "extreme premium; longs pay high carry. "
                        "Basis likely to compress = directional reversal risk [P26]."
                    )
                    notes.append(
                        f"[P26] Basis {basis_pct:.2f}% extreme premium — "
                        "cash-and-carry arb pressure will compress; suppress directional longs."
                    )
                elif basis_pct < -0.2 and is_short_signal:
                    modifier_delta -= 0.04
                    constraints.append(
                        f"BASIS_DISCOUNT: perp basis={basis_pct:.2f}% (<-0.2%) — "
                        "perp discount; shorts pay reverse carry [P26]."
                    )
                    notes.append(
                        f"[P26] Basis {basis_pct:.2f}% discount — shorts pay carry; reduce short exposure."
                    )

        # ── Rule D: Momentum Regime Gate  [Liu 2022, Liu 2022b] ──────────────
        # Cross-sectional momentum Sharpe by regime:
        #   trending_up: ~1.4 (full)   ranging: ~0.6 (weak)   trending_down: ~0.3 (very weak)
        # In ranging/volatile regime: XS momentum unreliable → flag CSM signals
        csm_signals = [s for s in signals if "CSM" in s.strategy_name]
        if csm_signals:
            if regime.regime in {RegimeType.RANGING}:
                modifier_delta -= 0.05
                constraints.append(
                    "CSM_REGIME_WEAK: CryptoCSM in ranging regime — XS momentum Sharpe ~0.6 "
                    "(vs 1.4 in trending). Prefer MeanReversion [Liu 2022b]."
                )
                notes.append("[P49] CSM in ranging regime — momentum signal unreliable. Reduce size.")
            elif regime.regime in {RegimeType.TRENDING_DOWN}:
                modifier_delta -= 0.08
                constraints.append(
                    "CSM_REGIME_VERY_WEAK: CryptoCSM in trending_down — XS momentum Sharpe ~0.3. "
                    "Long-side momentum fails in bear markets [Liu 2022, P11]."
                )
                notes.append("[P11,P49] CSM in bear regime — momentum long-side failure. High risk.")

        # ── Rule E: Volume-Confirmation Boost  [Liu 2022] ─────────────────────
        # Volume-confirmed momentum signals are MORE reliable.
        # vol_ratio > 1.5 + momentum signal = higher confidence [P11]
        if snapshot and snapshot.features and snapshot.features.volume_ratio is not None:
            vol_ratio = snapshot.features.volume_ratio
            if vol_ratio >= 1.5 and is_long_signal and regime.regime == RegimeType.TRENDING_UP:
                modifier_delta += 0.04
                notes.append(
                    f"[P11] Volume ratio {vol_ratio:.1f}x confirms momentum — "
                    "signal quality boosted (vol > 150% of 20d avg is strong confirmation)."
                )

        # ── Combine modifier ─────────────────────────────────────────────────
        new_modifier = ctx.confidence_modifier + modifier_delta
        new_modifier = max(-0.20, min(0.20, new_modifier))

        combined_note = " | ".join(n for n in notes if n)

        return KnowledgeContext(
            query_regime=ctx.query_regime,
            retrieved_chunks=ctx.retrieved_chunks,
            strategy_bias=bias,
            confidence_modifier=round(new_modifier, 3),
            suggested_constraints=constraints,
            advisory_note=combined_note,
            knowledge_only_veto=True,
        )

    # ─────────────────────────────────────────────────────────
    #  TIER 1: FULL RAG QUERY
    # ─────────────────────────────────────────────────────────

    def _rag_query(
        self,
        regime: RegimeAnalysis,
        signals: list[CandidateSignal],
    ) -> KnowledgeContext:
        """Semantic RAG: embed query → FAISS ANN → parse advisory."""
        query_text = self._build_query(regime, signals)
        chunks = self._retrieve_chunks(query_text, top_k=_TOP_K)
        modifier, constraints, bias, note = self._parse_advisory(chunks, regime)
        return KnowledgeContext(
            query_regime=regime.regime,
            retrieved_chunks=chunks,
            strategy_bias=bias,
            confidence_modifier=modifier,
            suggested_constraints=constraints,
            advisory_note=note,
            knowledge_only_veto=True,
        )

    def _build_query(self, regime: RegimeAnalysis, signals: list[CandidateSignal]) -> str:
        directions = list({s.direction.value for s in signals})
        strategies = list({s.strategy_name for s in signals})
        return (
            f"crypto {regime.regime.value} market "
            f"trading strategy {' '.join(directions)} {' '.join(strategies)} "
            f"momentum funding regime risk management position sizing"
        )

    def _retrieve_chunks(self, query: str, top_k: int = _TOP_K) -> list[KnowledgeChunk]:
        if self._faiss_index is not None:
            return self._faiss_retrieve(query, top_k)
        return self._token_overlap_retrieve(query, top_k)

    def _faiss_retrieve(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        try:
            model = self._get_model()
            q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
            k = min(top_k, len(self._chunks))
            distances, indices = self._faiss_index.search(q_emb, k)
            result = []
            for dist, idx in zip(distances[0], indices[0], strict=False):
                if idx < 0:
                    continue
                c = self._chunks[idx]
                result.append(
                    KnowledgeChunk(
                        text=c.text,
                        source=c.source,
                        relevance_score=float(min(1.0, max(0.0, (dist + 1.0) / 2.0))),
                        tags=c.tags,
                        context=c.context,
                    )
                )
            return result
        except Exception as e:
            logger.warning(f"[KnowledgeAgent] FAISS retrieve failed ({e}) — token fallback")
            return self._token_overlap_retrieve(query, top_k)

    def _token_overlap_retrieve(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        if not self._chunks:
            return []
        q_tokens = self._tokenize(query)
        scored = []
        for chunk in self._chunks:
            score = self._jaccard(q_tokens, self._tokenize(chunk.text))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            KnowledgeChunk(
                text=c.text,
                source=c.source,
                relevance_score=min(1.0, max(0.0, score)),
                tags=c.tags,
                context=c.context,
            )
            for score, c in scored[:top_k]
        ]

    def _parse_advisory(
        self,
        chunks: list[KnowledgeChunk],
        regime: RegimeAnalysis,
    ) -> tuple[float, list[str], SignalDirection | None, str]:
        """
        Parse advisory outputs from retrieved chunks.

        Crypto-specific keyword tables grounded in [P11, P12, P14, P15, P25, P26, P49].
        """
        if not chunks:
            return 0.0, [], None, "RAG: no relevant chunks."

        _constraint_rules = [
            (["funding", "carry"], "Monitor funding rate before new longs — check Auer 2023 thresholds."),
            (
                ["liquidation", "cascade"],
                "Liquidation cascade risk: funding + vol combined signal [Auer 2023].",
            ),
            (["volatility", "volatile"], "Reduce position size during volatility expansion [Liu 2022]."),
            (["momentum crash"], "Momentum crash risk: momentum strategies underperform post-crash."),
            (
                ["basis", "premium"],
                "Monitor perpetual futures basis — extreme premium suppresses longs [Avellaneda 2020].",
            ),
            (
                ["btc dominance", "dominance"],
                "Check BTC dominance: high dominance = risk-off, penalise alt longs [Liu-Tsyvinski 2021].",
            ),
            (["on-chain", "mvrv"], "On-chain MVRV > 3.5 = reduce all long sizing [Glassnode 2022]."),
            (
                ["mean reversion", "reversal"],
                "Mean reversion trades: confirm with RSI < 32 AND below BB lower band.",
            ),
        ]

        _pos_kws = {
            RegimeType.TRENDING_UP: ["momentum", "trend following", "bull", "cross-sectional", "uptrend"],
            RegimeType.TRENDING_DOWN: ["bear", "short", "dominance", "defensive", "btc dominance"],
            RegimeType.MEAN_REVERTING: ["mean reversion", "oversold", "bounce", "reversal", "contrarian"],
            RegimeType.RANGING: ["range", "mean reversion", "sideways", "carry", "funding"],
            RegimeType.VOLATILE: ["volatility", "risk management", "drawdown", "liquidation", "fear"],
        }
        _neg_kws = {
            RegimeType.TRENDING_UP: ["bear", "short selling", "reversion", "crash"],
            RegimeType.TRENDING_DOWN: ["bull", "trend following", "buy dip"],
            RegimeType.MEAN_REVERTING: ["trend following", "momentum", "breakout"],
            RegimeType.RANGING: ["trend following", "momentum crash", "breakout"],
            RegimeType.VOLATILE: ["buy and hold", "trend following", "momentum"],
        }

        pos_kws = _pos_kws.get(regime.regime, [])
        neg_kws = _neg_kws.get(regime.regime, [])
        n_pos = max(1, len(pos_kws))

        weighted_score = 0.0
        total_weight = 0.0
        seen_constraints: set = set()
        constraints: list[str] = []

        for chunk in chunks:
            text_lower = chunk.text.lower()
            w = chunk.relevance_score
            if w < 0.01:
                continue
            pos_hits = sum(1 for kw in pos_kws if kw in text_lower)
            neg_hits = sum(1 for kw in neg_kws if kw in text_lower)
            chunk_score = (pos_hits - neg_hits) / n_pos
            weighted_score += w * chunk_score
            total_weight += w
            for keywords, constraint_text in _constraint_rules:
                if constraint_text not in seen_constraints:
                    if any(kw in text_lower for kw in keywords):
                        constraints.append(constraint_text)
                        seen_constraints.add(constraint_text)

        if total_weight > 0:
            raw = weighted_score / total_weight
            modifier = max(-0.15, min(0.15, raw * 0.15))
        else:
            modifier = 0.0

        text_blob = " ".join(c.text.lower() for c in chunks)
        bias = None
        if "momentum" in text_blob and regime.regime == RegimeType.TRENDING_UP:
            bias = SignalDirection.BUY
        elif "bear" in text_blob and regime.regime == RegimeType.TRENDING_DOWN:
            bias = SignalDirection.SELL

        avg_rel = sum(c.relevance_score for c in chunks) / len(chunks)
        raw_sc = weighted_score / max(total_weight, 1e-9)
        note = (
            f"SemanticRAG: {len(chunks)} chunk(s) avg_rel={avg_rel:.3f} "
            f"regime_align={raw_sc:+.3f} → mod={modifier:+.3f}"
        )
        return modifier, constraints, bias, note

    # ─────────────────────────────────────────────────────────
    #  TIER 3: BM25 EXPERIENCE MEMORY  (TradingAgents-inspired)
    # ─────────────────────────────────────────────────────────

    def _augment_with_bm25(
        self,
        ctx: KnowledgeContext,
        regime: RegimeAnalysis,
        signals: list[CandidateSignal],
    ) -> KnowledgeContext:
        """
        v17c adaptive blend: scale BM25 weight with memory maturity.

        Empty memory  → 85% RAG / 15% BM25  (literature anchors early decisions)
        Mature memory → 40% RAG / 60% BM25  (experience takes over as data accumulates)
        """
        if len(self._bm25_memory) == 0:
            return ctx

        primary = signals[0] if signals else None
        situation = self._build_situation_string(regime, primary)
        as_of = str(regime.timestamp)[:10] if regime.timestamp else ""
        cur_reg = regime.regime.value

        bm25_results = self._bm25_memory.query(situation, n=5, as_of_date=as_of, current_regime=cur_reg)
        bm25_mod, bm25_note = self._bm25_memory.confidence_modifier_from_memory(
            situation, as_of_date=as_of, current_regime=cur_reg
        )
        if abs(bm25_mod) < 0.005:
            return ctx

        rag_w, bm25_w = self._blend_weights(cur_reg, bm25_results)
        blended = round(rag_w * ctx.confidence_modifier + bm25_w * bm25_mod, 3)
        blended = max(-0.20, min(0.20, blended))

        advisory = (
            f"{ctx.advisory_note} | "
            f"BM25Mem: {bm25_note} (blend={rag_w:.0%}RAG/{bm25_w:.0%}BM25 → {blended:+.3f})"
        )
        return KnowledgeContext(
            query_regime=ctx.query_regime,
            retrieved_chunks=ctx.retrieved_chunks,
            strategy_bias=ctx.strategy_bias,
            confidence_modifier=blended,
            suggested_constraints=ctx.suggested_constraints,
            advisory_note=advisory,
            knowledge_only_veto=True,
        )

    def _blend_weights(self, regime_str: str, bm25_results: list) -> tuple[float, float]:
        family = REGIME_FAMILIES.get(regime_str, "sideways")
        threshold = self._MATURITY_THRESHOLDS.get(family, 40)
        relevant = sum(1 for r in bm25_results if r.score >= self._BM25_RELEVANCE_FLOOR)
        maturity = min(1.0, relevant / threshold)
        bm25_w = 0.15 + 0.45 * maturity
        return 1.0 - bm25_w, bm25_w

    def _build_situation_string(
        self,
        regime: RegimeAnalysis,
        signal: CandidateSignal | None,
    ) -> str:
        parts = [f"regime {regime.regime.value}"]
        if signal:
            parts += [
                f"symbol {signal.symbol}",
                f"strategy {signal.strategy_name}",
                f"direction {signal.direction.value}",
            ]
        return " ".join(parts)

    # ─────────────────────────────────────────────────────────
    #  KNOWLEDGE BASE LOADING & FAISS CACHE
    # ─────────────────────────────────────────────────────────

    def _try_load(self, base: Path) -> None:
        try:
            if not base.exists():
                logger.warning(
                    f"[KnowledgeAgent] Knowledge dir not found: {base} — running in rule-only mode (no RAG)."
                )
                return

            for fp in sorted(base.rglob("*")):
                if fp.suffix.lower() in {".txt", ".md"}:
                    self._ingest_file(fp)

            if not self._chunks:
                logger.info(f"[KnowledgeAgent] No .txt/.md files in {base} — rule-only mode.")
                return

            self._loaded = True
            logger.info(f"[KnowledgeAgent] Loaded {len(self._chunks)} chunks from {base}")
            self._build_or_restore_faiss()

        except Exception as e:
            logger.error(f"[KnowledgeAgent] Load failed: {e}")
            self._loaded = len(self._chunks) > 0

    def _build_or_restore_faiss(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            logger.warning("[KnowledgeAgent] faiss-cpu not installed — token-overlap fallback.")
            return

        if self._try_restore_cache():
            return

        logger.info(
            f"[KnowledgeAgent] Building FAISS index for {len(self._chunks)} chunks "
            f"using {_MODEL_NAME} — runs once, cached to disk."
        )
        self._build_faiss_index()
        self._persist_cache()

    def _try_restore_cache(self) -> bool:
        try:
            if not _FAISS_INDEX_FILE.exists() or not _CHUNKS_FILE.exists():
                return False
            import faiss

            with open(_CHUNKS_FILE, "rb") as f:
                meta = pickle.load(f)
            if meta["chunk_count"] != len(self._chunks):
                logger.info("[KnowledgeAgent] Cache mismatch — rebuilding FAISS index.")
                return False
            self._faiss_index = faiss.read_index(str(_FAISS_INDEX_FILE))
            logger.info(f"[KnowledgeAgent] FAISS cache restored ({self._faiss_index.ntotal} vectors).")
            return True
        except Exception as e:
            logger.warning(f"[KnowledgeAgent] Cache restore failed ({e}) — rebuilding.")
            return False

    def _build_faiss_index(self) -> None:
        import faiss

        model = self._get_model()
        texts = [c.text for c in self._chunks]
        batch_size = 64
        all_embs = []
        for i in range(0, len(texts), batch_size):
            emb = model.encode(texts[i : i + batch_size], normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(emb.astype("float32"))
            logger.debug(f"[KnowledgeAgent] Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        self._embeddings = np.vstack(all_embs)
        dim = self._embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self._embeddings)
        self._faiss_index = index
        logger.info(f"[KnowledgeAgent] FAISS index built: {index.ntotal} vectors, dim={dim}")

    def _persist_cache(self) -> None:
        try:
            import faiss

            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._faiss_index, str(_FAISS_INDEX_FILE))
            with open(_CHUNKS_FILE, "wb") as f:
                pickle.dump({"chunk_count": len(self._chunks)}, f)
            logger.info(f"[KnowledgeAgent] FAISS cache persisted to {_CACHE_DIR}")
        except Exception as e:
            logger.warning(f"[KnowledgeAgent] Cache persist failed: {e}")

    def _get_model(self):
        if self._model is None:
            self._model = _get_sentence_model(_MODEL_NAME)
        return self._model

    # ─────────────────────────────────────────────────────────
    #  DOCUMENT INGESTION
    # ─────────────────────────────────────────────────────────

    def _ingest_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        parsed_tags, cleaned = self._parse_header_tags(text)
        tags = parsed_tags or (["crypto"] if "crypto" in path.name.lower() else ["trading"])
        for idx, chunk_text in enumerate(self._chunk_text(cleaned)):
            self._chunks.append(
                KnowledgeChunk(
                    text=chunk_text,
                    source=str(path),
                    relevance_score=0.0,
                    tags=tags,
                    context=f"chunk_{idx}",
                )
            )

    def _parse_header_tags(self, text: str) -> tuple[list[str], str]:
        lines = text.splitlines()
        tags: list[str] = []
        body_start = 0
        for i, line in enumerate(lines[:12]):
            normalized = line.strip()
            if not normalized:
                body_start = i + 1
                continue
            if normalized.upper().startswith("TAGS:"):
                raw = normalized.split(":", 1)[1].strip()
                tags = [
                    t.strip().lower() for t in (raw.split(",") if "," in raw else raw.split()) if t.strip()
                ]
                body_start = i + 1
        return tags, "\n".join(lines[body_start:]).strip() if body_start else text

    def _chunk_text(self, text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        start = 0
        n = len(cleaned)
        while start < n:
            end = min(n, start + chunk_size)
            chunks.append(cleaned[start:end])
            if end >= n:
                break
            start = max(end - overlap, start + 1)
        return chunks

    # ─────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> set:
        return {t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(t) > 2}

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0
