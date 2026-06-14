"""
KA-MATS Cryptoz · Alternative Data Agent
Iknir Capital

Crypto-native alternative data sources (all free, no API key required by default):

  1. Alternative.me Fear & Greed Index
     GET https://api.alternative.me/fng/?limit=2
     Returns 0-100 sentiment index for crypto market.
     Signal: Extreme Fear (0-25) → contrarian mean-reversion opportunity
             Extreme Greed (75-100) → momentum exhaustion caution

  2. CoinGecko Global Market Stats
     GET https://api.coingecko.com/api/v3/global
     Returns: BTC dominance %, total market cap, altcoin volumes
     Signal: Rising BTC dominance → BTCDominanceRotation boost
             Falling BTC dominance → altcoin season, boost CryptoCSM

  3. CoinGecko Trending Coins (free, no key)
     GET https://api.coingecko.com/api/v3/search/trending
     Trending coins = retail momentum proxy (crowd-following signal)
     Signal: If traded symbol is trending → mild momentum confirmation

All sources are cached for cfg.cache_ttl_minutes (default 60 min) to
respect free API rate limits (CoinGecko: 30 req/min on free tier).

Enable/disable via config (AltDataConfig in settings.py):
  fear_greed_enabled   = True   (default on)
  coingecko_enabled    = True   (default on)
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from urllib.error import URLError
from urllib.request import Request, urlopen

from loguru import logger

from config.settings import CONFIG, AltDataConfig
from core.models import AltDataContext, AltDataSignal, MarketSnapshot, SignalDirection

# ── Symbol → CoinGecko ID mapping (only universe symbols needed) ──────────────
# Used by the trending-coin check to match our symbols to CoinGecko IDs.
_SYMBOL_TO_COINGECKO: dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "SOL/USDT": "solana",
    "BNB/USDT": "binancecoin",
    "AVAX/USDT": "avalanche-2",
    "LINK/USDT": "chainlink",
    "DOT/USDT": "polkadot",
    "ADA/USDT": "cardano",
    "DOGE/USDT": "dogecoin",
    "POL/USDT": "matic-network",  # Polygon (MATIC rebranded to POL)
    "UNI/USDT": "uniswap",
    "ATOM/USDT": "cosmos",
    "NEAR/USDT": "near",
    "ARB/USDT": "arbitrum",
    "OP/USDT": "optimism",
}

# Fear & Greed Index thresholds
_FNG_EXTREME_FEAR = 25  # ≤ 25 → contrarian buy / mean reversion signal
_FNG_FEAR = 40  # ≤ 40 → mild caution for new longs
_FNG_GREED = 65  # ≥ 65 → mild caution for momentum entries
_FNG_EXTREME_GREED = 80  # ≥ 80 → exhaustion risk for CSM / TrendPullback

# BTC dominance thresholds
_BTC_DOM_HIGH = 55.0  # % — risk-off: capital in BTC, alts struggling
_BTC_DOM_LOW = 42.0  # % — risk-on: altcoin season, boost CryptoCSM


class AltDataAgent:
    """
    Crypto-native Alternative Data Agent.

    Pulls sentiment and market-structure signals that complement price/TA data.
    All sources are free and require no API keys.

    Results are cached for cache_ttl_minutes to avoid hammering free tier APIs.
    Each source is soft-skipped on failure — never blocks the main pipeline.
    """

    # Sources that MUST produce signals when healthy (used for data_quality_ok)
    _CRITICAL_SOURCES = {"fear_greed", "coingecko_global"}

    # How long to back-off a failed source before retrying (seconds)
    _FAIL_TTL = 300  # 5 minutes

    def __init__(self, cfg: AltDataConfig = None) -> None:
        self.cfg = cfg or CONFIG.alt_data
        self._cache: dict[str, tuple[float, list[AltDataSignal]]] = {}
        self._fail_cache: dict[str, float] = {}  # source → timestamp of last failure
        self._active_sources: list[str] = []

        if self.cfg.fear_greed_enabled:
            self._active_sources.append("fear_greed")
        if self.cfg.coingecko_enabled:
            self._active_sources.append("coingecko_global")
            self._active_sources.append("coingecko_trending")

        if self._active_sources:
            logger.info(f"[AltDataAgent] Active sources: {self._active_sources}")
        else:
            logger.info("[AltDataAgent] No alt data sources enabled.")

    # ─────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────

    def get_context(self, snapshot: MarketSnapshot | None = None) -> AltDataContext:
        """
        Fetch alt data signals for the current bar.
        Each source is independently cached and soft-skipped on failure.

        data_quality_ok is False when ALL critical sources (fear_greed +
        coingecko_global) fail to fetch — the orchestrator's fallback will
        then use the last known good context instead.
        """
        if not self._active_sources:
            return self._neutral_context()

        signals, failed_sources = self._aggregate_signals(snapshot)

        # Quality is degraded only when every critical source failed
        critical_failed = self._CRITICAL_SOURCES & failed_sources
        data_quality_ok = len(critical_failed) < len(self._CRITICAL_SOURCES)

        if failed_sources:
            logger.warning(
                f"[AltDataAgent] {len(failed_sources)} source(s) failed: "
                f"{failed_sources} | data_quality_ok={data_quality_ok}"
            )

        return AltDataContext(
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            signals=signals,
            data_quality_ok=data_quality_ok,
            sources_available=self._active_sources,
            advisory_note=(
                f"{len(signals)} alt data signal(s) from "
                f"{', '.join(s for s in self._active_sources if s not in failed_sources)}"
                if signals
                else "Alt data fetched — no actionable signals this bar"
            ),
        )

    # ─────────────────────────────────────────────────────────
    #  SIGNAL AGGREGATION
    # ─────────────────────────────────────────────────────────

    def _aggregate_signals(self, snapshot: MarketSnapshot | None) -> tuple[list[AltDataSignal], set]:
        """
        Returns (signals, failed_sources_set).
        failed_sources only contains sources that threw exceptions — a source
        that genuinely had no signal (e.g. trending) is NOT counted as failed.
        """
        signals: list[AltDataSignal] = []
        failed_sources: set = set()
        symbol = snapshot.symbol if snapshot else None

        for source in self._active_sources:
            try:
                if source == "fear_greed":
                    signals.extend(self._cached("fear_greed", self._fetch_fear_greed))
                elif source == "coingecko_global":
                    signals.extend(self._cached("coingecko_global", self._fetch_coingecko_global))
                elif source == "coingecko_trending" and symbol:
                    signals.extend(self._cached("coingecko_trending", self._fetch_coingecko_trending, symbol))
                # coingecko_trending with no symbol → skip silently (not a failure)
            except Exception as e:
                failed_sources.add(source)
                logger.warning(f"[AltDataAgent] Source {source} failed (skipping): {e}")

        return signals, failed_sources

    def _cached(self, key: str, fn, *args) -> list[AltDataSignal]:
        """
        Cache-with-failure-backoff:
        - Successful results cached for cache_ttl_minutes (default 60 min).
        - Failed fetches backed off for _FAIL_TTL (5 min) — NOT stored in the
          success cache, so a recovery within the backoff window is still tried.
        - Only non-empty successful results replace the success cache, so a
          transient empty response never overwrites valid cached data.
        """
        now = time.time()

        # Check failure backoff first — don't hammer a down API
        fail_ts = self._fail_cache.get(key)
        if fail_ts is not None and (now - fail_ts) < self._FAIL_TTL:
            logger.debug(f"[AltDataAgent] {key}: in failure backoff, skipping fetch")
            raise RuntimeError(f"{key} in failure backoff ({self._FAIL_TTL}s)")

        # Check success cache
        ttl = self.cfg.cache_ttl_minutes * 60
        if key in self._cache:
            ts, cached = self._cache[key]
            if now - ts < ttl:
                return cached

        # Fetch — let exceptions propagate to _aggregate_signals
        try:
            result = fn(*args)
        except Exception:
            self._fail_cache[key] = now  # record failure timestamp
            raise  # propagate — caller marks source failed

        # Only cache non-empty results; empty from a failed fetch must not
        # overwrite a previously good cached result
        if result:
            self._cache[key] = (now, result)
            self._fail_cache.pop(key, None)  # clear any prior failure on recovery
        elif key not in self._cache:
            # No prior cache and empty result — cache it anyway to avoid
            # hammering the API, but only for a short time
            self._cache[key] = (now, result)

        return result

    # ─────────────────────────────────────────────────────────
    #  SOURCE 1: ALTERNATIVE.ME FEAR & GREED INDEX
    # ─────────────────────────────────────────────────────────

    def _fetch_fear_greed(self) -> list[AltDataSignal]:
        """
        Alternative.me Crypto Fear & Greed Index.
        Free, no key. Returns composite 0-100 score + 1-day change.

        Raises on network/parse error — caller (_cached) handles failure tracking.
        """
        data = self._http_get("https://api.alternative.me/fng/?limit=2", timeout=8)

        entries = data.get("data", [])
        if not entries:
            raise RuntimeError("Fear & Greed API returned empty data list")

        current = int(entries[0]["value"])
        label = entries[0]["value_classification"]
        prev = int(entries[1]["value"]) if len(entries) > 1 else current
        change = current - prev

        if current <= _FNG_EXTREME_FEAR:
            direction = SignalDirection.BUY
            confidence = min(0.85, 0.60 + (_FNG_EXTREME_FEAR - current) / _FNG_EXTREME_FEAR * 0.25)
            desc = f"Extreme Fear ({current}/100) — contrarian mean-reversion opportunity"
        elif current <= _FNG_FEAR:
            direction = SignalDirection.BUY
            confidence = 0.55
            desc = f"Fear ({current}/100, '{label}') — mild buy bias"
        elif current >= _FNG_EXTREME_GREED:
            direction = SignalDirection.SELL
            confidence = min(0.75, 0.55 + (current - _FNG_EXTREME_GREED) / (100 - _FNG_EXTREME_GREED) * 0.20)
            desc = f"Extreme Greed ({current}/100) — momentum exhaustion risk, caution on new longs"
        elif current >= _FNG_GREED:
            direction = SignalDirection.SELL
            confidence = 0.50
            desc = f"Greed ({current}/100, '{label}') — mild momentum caution"
        else:
            direction = None
            confidence = 0.0
            desc = f"Neutral sentiment ({current}/100, '{label}')"

        logger.debug(
            f"[AltDataAgent] Fear & Greed: {current}/100 '{label}' "
            f"(Δ{change:+d} from yesterday) → {direction}"
        )

        return [
            AltDataSignal(
                source="fear_greed_index",
                signal_type="crypto_sentiment",
                value=float(current),
                direction=direction,
                confidence=confidence,
                description=desc,
                tags=["sentiment", "mean_reversion" if current <= _FNG_FEAR else "momentum_caution"],
            )
        ]

    # ─────────────────────────────────────────────────────────
    #  SOURCE 2: COINGECKO GLOBAL MARKET STATS
    # ─────────────────────────────────────────────────────────

    def _fetch_coingecko_global(self) -> list[AltDataSignal]:
        """
        CoinGecko /global endpoint — total market cap, BTC dominance, volumes.
        Free tier (no key). Rate limit: 30 req/min.

        Raises on network/parse error — caller (_cached) handles failure tracking.
        """
        data = self._http_get("https://api.coingecko.com/api/v3/global", timeout=10)

        gdata = data.get("data", {})
        if not gdata:
            raise RuntimeError("CoinGecko global API returned empty data")

        btc_dom = gdata.get("market_cap_percentage", {}).get("bitcoin", 0.0)
        mcap_change_24h = gdata.get("market_cap_change_percentage_24h_usd", 0.0)

        signals: list[AltDataSignal] = []

        # ── BTC Dominance signal ──────────────────────────────────────
        if btc_dom >= _BTC_DOM_HIGH:
            signals.append(
                AltDataSignal(
                    source="coingecko_global",
                    signal_type="btc_dominance_risk_off",
                    value=round(btc_dom, 2),
                    direction=None,
                    confidence=min(0.75, 0.50 + (btc_dom - _BTC_DOM_HIGH) / 10.0 * 0.25),
                    description=(
                        f"BTC dominance {btc_dom:.1f}% (high) — risk-off: "
                        f"capital concentrated in BTC/ETH; penalise altcoin momentum"
                    ),
                    tags=["btc_dominance", "risk_off", "btc_rotation"],
                )
            )
        elif btc_dom <= _BTC_DOM_LOW:
            signals.append(
                AltDataSignal(
                    source="coingecko_global",
                    signal_type="btc_dominance_altseason",
                    value=round(btc_dom, 2),
                    direction=SignalDirection.BUY,
                    confidence=min(0.70, 0.50 + (_BTC_DOM_LOW - btc_dom) / 10.0 * 0.20),
                    description=(
                        f"BTC dominance {btc_dom:.1f}% (low) — altcoin season: "
                        f"capital rotating to alts; boost CryptoCSM"
                    ),
                    tags=["btc_dominance", "altseason", "csm_boost"],
                )
            )
        else:
            signals.append(
                AltDataSignal(
                    source="coingecko_global",
                    signal_type="btc_dominance_neutral",
                    value=round(btc_dom, 2),
                    direction=None,
                    confidence=0.0,
                    description=f"BTC dominance {btc_dom:.1f}% — neutral range",
                    tags=["btc_dominance"],
                )
            )

        # ── Total market cap 24h change signal ───────────────────────
        if abs(mcap_change_24h) >= 3.0:
            direction = SignalDirection.BUY if mcap_change_24h > 0 else SignalDirection.SELL
            confidence = min(0.70, 0.50 + abs(mcap_change_24h) / 20.0 * 0.20)
            signals.append(
                AltDataSignal(
                    source="coingecko_global",
                    signal_type="market_cap_momentum",
                    value=round(mcap_change_24h, 2),
                    direction=direction,
                    confidence=confidence,
                    description=(
                        f"Total crypto market cap {mcap_change_24h:+.1f}% (24h) "
                        f"— {'bull' if mcap_change_24h > 0 else 'bear'} market momentum"
                    ),
                    tags=["market_momentum", "macro_crypto"],
                )
            )

        logger.debug(
            f"[AltDataAgent] CoinGecko global: BTC_dom={btc_dom:.1f}% mcap_chg={mcap_change_24h:+.1f}%"
        )
        return signals

    # ─────────────────────────────────────────────────────────
    #  SOURCE 3: COINGECKO TRENDING COINS
    # ─────────────────────────────────────────────────────────

    def _fetch_coingecko_trending(self, symbol: str) -> list[AltDataSignal]:
        """
        CoinGecko /search/trending — top-7 trending coins by search volume.
        Returning an empty list is VALID (symbol not trending) — not a failure.
        Raises only on network/parse error.
        """
        coingecko_id = _SYMBOL_TO_COINGECKO.get(symbol)
        if coingecko_id is None:
            return []

        data = self._http_get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=8,
        )

        trending_ids = [coin.get("item", {}).get("id", "") for coin in data.get("coins", [])]

        if coingecko_id not in trending_ids:
            return []  # legitimate empty — symbol not trending

        rank = trending_ids.index(coingecko_id) + 1
        confidence = 0.55 if rank <= 3 else 0.45
        logger.debug(f"[AltDataAgent] {symbol} trending #{rank} on CoinGecko")

        return [
            AltDataSignal(
                source="coingecko_trending",
                signal_type="trending_coin",
                value=float(rank),
                direction=SignalDirection.BUY,
                confidence=confidence,
                description=(
                    f"{symbol} is #{rank} trending on CoinGecko — "
                    f"retail interest elevated (momentum confirmation, watch for crowd top)"
                ),
                tags=["trending", "retail_momentum"],
            )
        ]

    # ─────────────────────────────────────────────────────────
    #  HTTP UTILITY
    # ─────────────────────────────────────────────────────────

    def _http_get(self, url: str, timeout: int = 10) -> dict:
        """
        Fetch JSON from URL. Raises on any network or parse error so that
        _cached() can record the failure and _aggregate_signals() can mark
        the source as failed (instead of silently caching an empty result).
        """
        req = Request(
            url,
            headers={
                "User-Agent": "KA-MATS-Cryptoz/2.0 AltDataAgent (research)",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except URLError as e:
            raise RuntimeError(f"HTTP GET failed ({url[:70]}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"HTTP parse error ({url[:70]}): {e}") from e

    def _neutral_context(self) -> AltDataContext:
        return AltDataContext(
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            signals=[],
            data_quality_ok=True,
            sources_available=[],
            advisory_note=(
                "AltDataAgent: no sources enabled. "
                "Set fear_greed_enabled=True or coingecko_enabled=True in AltDataConfig."
            ),
        )
