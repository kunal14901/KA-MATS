"""
KA-MATS Cryptoz · LLM Veto Validation Layer
Iknir Capital — v6 Enhancement (Vertus-inspired)

A veto-only validation layer powered by an LLM. It can REJECT signals
that pass quantitative filters but fail a qualitative sanity check.
It NEVER generates trade ideas — purely defensive.

Inspired by Vertus's proprietary financial LLM that validates
every decision. Our version uses local or API-based LLMs to catch
edge cases that rules miss: narrative inconsistency, macro misread,
regime-signal mismatch, known trap patterns.

Supported backends:
  - Ollama (local, zero cost) — default: llama3.2
  - OpenAI API (gpt-4o-mini for cost efficiency)
  - Anthropic API (Claude haiku)
  - Disabled (pass-through) for backtesting

Usage:
    validator = LLMValidator(backend="ollama")
    verdict = validator.validate(signal, market_context, regime)
    if verdict.vetoed:
        skip_trade(verdict.reason)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

from loguru import logger

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────


@dataclass
class LLMValidatorConfig:
    """Configuration for the LLM veto layer."""

    enabled: bool = False  # must be explicitly enabled
    backend: str = "ollama"  # "ollama", "openai", "anthropic", "disabled"
    model: str = "llama3.2"  # model name
    ollama_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    timeout_seconds: int = 15
    confidence_veto_threshold: float = 0.6  # LLM confidence > 0.6 to veto
    max_retries: int = 1
    cache_ttl_minutes: int = 10  # cache veto decisions


# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────


@dataclass
class VetoVerdict:
    """Result of LLM validation."""

    vetoed: bool = False
    confidence: float = 0.0  # 0-1, how confident the LLM is in its veto
    reason: str = ""
    analysis: str = ""
    latency_ms: int = 0
    backend_used: str = "disabled"
    error: str | None = None


# ─────────────────────────────────────────────────────────────
#  LLM VALIDATOR
# ─────────────────────────────────────────────────────────────


class LLMValidator:
    """
    LLM-based veto validation layer.

    Receives a structured signal context and asks the LLM whether
    the trade should proceed or be vetoed. The LLM must return a
    structured JSON response with vetoed (bool), confidence (float),
    and reason (string).
    """

    _SYSTEM_PROMPT = """You are a risk analyst for a crypto hedge fund.
Your job is to VETO trades that are likely to fail. You are NOT allowed
to suggest new trades or modify existing ones.

You will receive a trade signal with market context. Evaluate whether
this trade should be VETOED (rejected) or APPROVED.

VETO if any of these apply:
- The signal direction contradicts the market regime
- The signal is based on mean-reversion in a trending market (or vice versa)
- Risk/reward ratio is unfavorable given current volatility
- The symbol shows signs of a known trap pattern (bull trap, dead cat bounce)
- Macro conditions strongly oppose the trade direction
- On-chain flow data contradicts the signal direction

APPROVE if the trade is reasonable given the context.

Reply with ONLY valid JSON (no markdown, no code fences):
{"vetoed": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}"""

    def __init__(self, cfg: LLMValidatorConfig = None) -> None:
        self.cfg = cfg or LLMValidatorConfig()
        self._cache: dict[str, tuple] = {}  # hash → (timestamp, verdict)

        # Resolve API keys from env if not set
        if not self.cfg.openai_api_key:
            self.cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.cfg.anthropic_api_key:
            self.cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def validate(
        self,
        signal_direction: str,
        symbol: str,
        confidence: float,
        strategy_name: str,
        regime: str,
        regime_confidence: float,
        rsi: float,
        adx: float,
        volume_ratio: float,
        cross_rank: float,
        fear_greed: int | None = None,
        flow_bias: float | None = None,
        drawdown_pct: float = 0.0,
    ) -> VetoVerdict:
        """
        Validate a trade signal through the LLM.

        Returns VetoVerdict. If the LLM is unavailable/disabled,
        returns a non-vetoed verdict (fail-open).
        """
        if not self.cfg.enabled or self.cfg.backend == "disabled":
            return VetoVerdict(backend_used="disabled")

        # Build context for the LLM
        context = self._build_context(
            signal_direction,
            symbol,
            confidence,
            strategy_name,
            regime,
            regime_confidence,
            rsi,
            adx,
            volume_ratio,
            cross_rank,
            fear_greed,
            flow_bias,
            drawdown_pct,
        )

        # Check cache
        cache_key = f"{symbol}:{signal_direction}:{strategy_name}:{regime}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        # Call LLM
        t0 = time.time()
        verdict = self._call_llm(context)
        verdict.latency_ms = int((time.time() - t0) * 1000)

        # Apply veto threshold
        if verdict.vetoed and verdict.confidence < self.cfg.confidence_veto_threshold:
            logger.debug(
                f"[LLMValidator] Veto overridden: confidence {verdict.confidence:.2f} "
                f"< threshold {self.cfg.confidence_veto_threshold}"
            )
            verdict.vetoed = False
            verdict.reason = f"(low confidence veto dropped) {verdict.reason}"

        # Cache result
        self._cache[cache_key] = (time.time(), verdict)

        if verdict.vetoed:
            logger.warning(
                f"[LLMValidator] VETOED {signal_direction} {symbol} "
                f"(confidence={verdict.confidence:.2f}): {verdict.reason}"
            )
        else:
            logger.debug(f"[LLMValidator] Approved {signal_direction} {symbol} ({verdict.latency_ms}ms)")

        return verdict

    def _build_context(
        self,
        signal_direction: str,
        symbol: str,
        confidence: float,
        strategy_name: str,
        regime: str,
        regime_confidence: float,
        rsi: float,
        adx: float,
        volume_ratio: float,
        cross_rank: float,
        fear_greed: int | None,
        flow_bias: float | None,
        drawdown_pct: float,
    ) -> str:
        """Build a structured context string for the LLM."""
        ctx = {
            "trade_signal": {
                "direction": signal_direction,
                "symbol": symbol,
                "confidence": round(confidence, 3),
                "strategy": strategy_name,
            },
            "market_context": {
                "regime": regime,
                "regime_confidence": round(regime_confidence, 3),
                "rsi": round(rsi, 1),
                "adx": round(adx, 1),
                "volume_ratio": round(volume_ratio, 2),
                "cross_rank": round(cross_rank, 3),
            },
            "macro_context": {
                "fear_greed_index": fear_greed,
                "on_chain_flow_bias": round(flow_bias, 3) if flow_bias else None,
                "portfolio_drawdown": round(drawdown_pct, 4),
            },
        }
        return json.dumps(ctx, indent=2)

    def _call_llm(self, context: str) -> VetoVerdict:
        """Dispatch to the configured backend."""
        backend = self.cfg.backend.lower()
        try:
            if backend == "ollama":
                return self._call_ollama(context)
            elif backend == "openai":
                return self._call_openai(context)
            elif backend == "anthropic":
                return self._call_anthropic(context)
            else:
                return VetoVerdict(error=f"Unknown backend: {backend}")
        except Exception as e:
            logger.debug(f"[LLMValidator] Backend error ({backend}): {e}")
            return VetoVerdict(error=str(e), backend_used=backend)

    # ─────────────────────────────────────────────────────────
    #  OLLAMA (local)
    # ─────────────────────────────────────────────────────────

    def _call_ollama(self, context: str) -> VetoVerdict:
        """Call local Ollama instance."""
        payload = json.dumps(
            {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200},
            }
        ).encode("utf-8")

        req = Request(
            f"{self.cfg.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
            result = json.loads(resp.read().decode())

        content = result.get("message", {}).get("content", "")
        return self._parse_response(content, "ollama")

    # ─────────────────────────────────────────────────────────
    #  OPENAI
    # ─────────────────────────────────────────────────────────

    def _call_openai(self, context: str) -> VetoVerdict:
        """Call OpenAI API."""
        if not self.cfg.openai_api_key:
            return VetoVerdict(error="No OpenAI API key", backend_used="openai")

        payload = json.dumps(
            {
                "model": self.cfg.model or "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                "temperature": 0.1,
                "max_tokens": 200,
            }
        ).encode("utf-8")

        req = Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.openai_api_key}",
            },
        )

        with urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
            result = json.loads(resp.read().decode())

        content = result["choices"][0]["message"]["content"]
        return self._parse_response(content, "openai")

    # ─────────────────────────────────────────────────────────
    #  ANTHROPIC
    # ─────────────────────────────────────────────────────────

    def _call_anthropic(self, context: str) -> VetoVerdict:
        """Call Anthropic API."""
        if not self.cfg.anthropic_api_key:
            return VetoVerdict(error="No Anthropic API key", backend_used="anthropic")

        payload = json.dumps(
            {
                "model": self.cfg.model or "claude-3-haiku-20240307",
                "max_tokens": 200,
                "system": self._SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": context}],
            }
        ).encode("utf-8")

        req = Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.cfg.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        with urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
            result = json.loads(resp.read().decode())

        content = result["content"][0]["text"]
        return self._parse_response(content, "anthropic")

    # ─────────────────────────────────────────────────────────
    #  RESPONSE PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_response(self, raw: str, backend: str) -> VetoVerdict:
        """Parse the LLM JSON response into a VetoVerdict."""
        try:
            # Strip any markdown code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)
            return VetoVerdict(
                vetoed=bool(data.get("vetoed", False)),
                confidence=float(data.get("confidence", 0.0)),
                reason=str(data.get("reason", "")),
                analysis=raw,
                backend_used=backend,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"[LLMValidator] Failed to parse response: {e}")
            return VetoVerdict(
                error=f"Parse error: {e}",
                analysis=raw,
                backend_used=backend,
            )

    # ─────────────────────────────────────────────────────────
    #  CACHING
    # ─────────────────────────────────────────────────────────

    def _check_cache(self, key: str) -> VetoVerdict | None:
        if key in self._cache:
            ts, verdict = self._cache[key]
            if time.time() - ts < self.cfg.cache_ttl_minutes * 60:
                logger.debug(f"[LLMValidator] Cache hit: {key}")
                return verdict
            del self._cache[key]
        return None
