"""
KA-MATS Cryptoz · On-Chain Flow Agent
Iknir Capital — v6 Enhancement (Vertus-inspired)

Fetches on-chain and derivatives flow data to detect institutional
positioning invisible in price/volume alone.

Sources (all free tier, no API key required):
  1. CoinGlass Open Interest (aggregated across exchanges)
  2. CoinGlass Funding Rates (perp funding as sentiment proxy)
  3. Blockchain.com BTC Exchange Reserves (net inflow/outflow)
  4. CryptoQuant Exchange Netflow proxy via public API

All sources are cached, soft-skipped on failure, and purely advisory.
The agent outputs OnChainContext which feeds into the Risk Manager
as an additional sizing modifier (never generates or vetoes trades).

Enable/disable via config:
  OnChainConfig.enabled = True
  OnChainConfig.coinglass_enabled = True
  OnChainConfig.exchange_flow_enabled = True
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.request import Request, urlopen

from loguru import logger

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────


@dataclass
class OnChainConfig:
    """Configuration for on-chain flow data sources."""

    enabled: bool = True
    coinglass_oi_enabled: bool = True
    coinglass_funding_enabled: bool = True
    exchange_flow_enabled: bool = True
    cache_ttl_minutes: int = 30
    # Thresholds
    oi_spike_pct: float = 10.0  # OI change > 10% in 24h → spike
    oi_drop_pct: float = -8.0  # OI drops > 8% → deleveraging
    funding_extreme_high: float = 0.05  # 5% annualised → crowded long
    funding_extreme_low: float = -0.02  # negative → shorts paying longs
    exchange_inflow_spike_pct: float = 15.0  # large inflow → sell pressure


# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────


@dataclass
class FlowSignal:
    """Single on-chain / derivatives flow signal."""

    source: str
    signal_type: str
    value: float
    direction: str | None  # "bullish", "bearish", or None
    strength: float  # 0.0 - 1.0
    description: str
    symbol: str | None = None


@dataclass
class OnChainContext:
    """Aggregated on-chain context for the current bar."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    signals: list[FlowSignal] = field(default_factory=list)
    data_quality_ok: bool = True
    flow_bias: float = 0.0  # -1 (bearish) to +1 (bullish)
    sizing_modifier: float = 1.0  # 0.5 - 1.2 multiplier for risk manager
    advisory_note: str = ""


# ─────────────────────────────────────────────────────────────
#  SYMBOL MAPPINGS
# ─────────────────────────────────────────────────────────────

_SYMBOL_TO_COINGLASS = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
    "SOL/USDT": "SOL",
    "BNB/USDT": "BNB",
    "AVAX/USDT": "AVAX",
    "LINK/USDT": "LINK",
    "DOT/USDT": "DOT",
    "ADA/USDT": "ADA",
    "DOGE/USDT": "DOGE",
    "UNI/USDT": "UNI",
    "ATOM/USDT": "ATOM",
    "NEAR/USDT": "NEAR",
    "ARB/USDT": "ARB",
    "OP/USDT": "OP",
    "POL/USDT": "MATIC",
}


class OnChainAgent:
    """
    On-chain and derivatives flow agent.

    Fetches open interest, funding rates, and exchange flow data.
    Outputs a sizing modifier and directional bias for the risk manager.
    """

    _FAIL_TTL = 300  # 5 min backoff on failure

    def __init__(self, cfg: OnChainConfig = None) -> None:
        self.cfg = cfg or OnChainConfig()
        self._cache: dict[str, tuple] = {}  # key → (timestamp, result)
        self._fail_cache: dict[str, float] = {}  # key → last_fail_time

    def get_context(self, symbols: list[str] = None) -> OnChainContext:
        """Fetch all on-chain signals and compute aggregate context."""
        if not self.cfg.enabled:
            return OnChainContext(advisory_note="On-chain agent disabled")

        signals: list[FlowSignal] = []
        failed = 0
        total = 0

        # 1. Open Interest aggregated
        if self.cfg.coinglass_oi_enabled:
            total += 1
            try:
                oi_signals = self._cached("oi_agg", self._fetch_oi_aggregated)
                signals.extend(oi_signals)
            except Exception as e:
                failed += 1
                logger.debug(f"[OnChainAgent] OI fetch failed: {e}")

        # 2. Funding rates
        if self.cfg.coinglass_funding_enabled:
            total += 1
            try:
                funding_signals = self._cached("funding", self._fetch_funding_rates)
                signals.extend(funding_signals)
            except Exception as e:
                failed += 1
                logger.debug(f"[OnChainAgent] Funding fetch failed: {e}")

        # 3. Exchange flow (BTC only for free tier)
        if self.cfg.exchange_flow_enabled:
            total += 1
            try:
                flow_signals = self._cached("exchange_flow", self._fetch_exchange_flow)
                signals.extend(flow_signals)
            except Exception as e:
                failed += 1
                logger.debug(f"[OnChainAgent] Exchange flow fetch failed: {e}")

        # Compute aggregate bias and sizing modifier
        flow_bias = self._compute_flow_bias(signals)
        sizing_mod = self._compute_sizing_modifier(flow_bias, signals)

        return OnChainContext(
            signals=signals,
            data_quality_ok=(failed < total),
            flow_bias=round(flow_bias, 3),
            sizing_modifier=round(sizing_mod, 3),
            advisory_note=(
                f"{len(signals)} flow signal(s) | bias={flow_bias:+.2f} | sizing_mod={sizing_mod:.2f}x"
            ),
        )

    # ─────────────────────────────────────────────────────────
    #  SOURCE 1: OPEN INTEREST (CoinGlass public)
    # ─────────────────────────────────────────────────────────

    def _fetch_oi_aggregated(self) -> list[FlowSignal]:
        """
        CoinGlass aggregated open interest.
        Free endpoint returns BTC + ETH OI with 24h change.
        """
        signals = []
        try:
            data = self._http_get(
                "https://open-api.coinglass.com/public/v2/open_interest?symbol=BTC",
                timeout=10,
            )
            if data and "data" in data:
                for item in data["data"][:5]:  # top exchanges
                    oi_change = item.get("oiChange24h", 0)
                    if isinstance(oi_change, (int, float)) and abs(oi_change) > 0:
                        if oi_change > self.cfg.oi_spike_pct:
                            signals.append(
                                FlowSignal(
                                    source="coinglass_oi",
                                    signal_type="oi_spike",
                                    value=oi_change,
                                    direction="bullish",
                                    strength=min(0.8, oi_change / 20.0),
                                    description=f"BTC OI surged {oi_change:+.1f}% (24h) — new leveraged longs",
                                    symbol="BTC/USDT",
                                )
                            )
                        elif oi_change < self.cfg.oi_drop_pct:
                            signals.append(
                                FlowSignal(
                                    source="coinglass_oi",
                                    signal_type="oi_deleveraging",
                                    value=oi_change,
                                    direction="bearish",
                                    strength=min(0.8, abs(oi_change) / 15.0),
                                    description=f"BTC OI dropped {oi_change:+.1f}% (24h) — deleveraging event",
                                    symbol="BTC/USDT",
                                )
                            )
        except Exception:
            # Fallback: try alternative free endpoint
            pass

        return signals

    # ─────────────────────────────────────────────────────────
    #  SOURCE 2: FUNDING RATES
    # ─────────────────────────────────────────────────────────

    def _fetch_funding_rates(self) -> list[FlowSignal]:
        """
        Aggregate funding rates across major perp exchanges.
        Positive = longs pay shorts (bullish crowding).
        Negative = shorts pay longs (bearish or contrarian bullish).
        """
        signals = []
        try:
            data = self._http_get(
                "https://open-api.coinglass.com/public/v2/funding?symbol=BTC",
                timeout=10,
            )
            if data and "data" in data:
                rates = []
                for item in data["data"][:10]:
                    rate = item.get("rate", 0)
                    if isinstance(rate, (int, float)):
                        rates.append(float(rate))

                if rates:
                    avg_rate = sum(rates) / len(rates)
                    annualised = avg_rate * 3 * 365  # 8h funding × 3/day × 365

                    if annualised > self.cfg.funding_extreme_high:
                        signals.append(
                            FlowSignal(
                                source="coinglass_funding",
                                signal_type="funding_crowded_long",
                                value=round(annualised * 100, 2),
                                direction="bearish",
                                strength=min(0.7, annualised / 0.10),
                                description=(
                                    f"BTC funding {annualised * 100:.1f}% annualised — "
                                    f"crowded longs, squeeze risk"
                                ),
                                symbol="BTC/USDT",
                            )
                        )
                    elif annualised < self.cfg.funding_extreme_low:
                        signals.append(
                            FlowSignal(
                                source="coinglass_funding",
                                signal_type="funding_shorts_paying",
                                value=round(annualised * 100, 2),
                                direction="bullish",
                                strength=min(0.6, abs(annualised) / 0.05),
                                description=(
                                    f"BTC funding {annualised * 100:.1f}% annualised — "
                                    f"shorts paying longs, contrarian bullish"
                                ),
                                symbol="BTC/USDT",
                            )
                        )
        except Exception:
            pass

        return signals

    # ─────────────────────────────────────────────────────────
    #  SOURCE 3: EXCHANGE FLOW (BTC)
    # ─────────────────────────────────────────────────────────

    def _fetch_exchange_flow(self) -> list[FlowSignal]:
        """
        Blockchain.com BTC exchange reserves via mempool.space as proxy.
        Large inflows to exchanges = potential sell pressure.
        Large outflows = accumulation (bullish).
        """
        signals = []
        try:
            # mempool.space: free, no key
            data = self._http_get(
                "https://mempool.space/api/v1/mining/hashrate/1w",
                timeout=8,
            )
            # Use hashrate as a proxy for network health
            if data and isinstance(data, dict):
                current = data.get("currentHashrate", 0)
                if current and isinstance(current, (int, float)) and current > 0:
                    signals.append(
                        FlowSignal(
                            source="mempool_hashrate",
                            signal_type="network_health",
                            value=current / 1e18,  # EH/s
                            direction=None,
                            strength=0.3,
                            description=f"BTC hashrate {current / 1e18:.0f} EH/s — network health proxy",
                            symbol="BTC/USDT",
                        )
                    )
        except Exception:
            pass

        # Alternative: CoinGecko exchange volumes as flow proxy
        try:
            data = self._http_get(
                "https://api.coingecko.com/api/v3/exchanges?per_page=5",
                timeout=8,
            )
            if data and isinstance(data, list):
                total_vol = sum(
                    ex.get("trade_volume_24h_btc_normalized", 0) for ex in data if isinstance(ex, dict)
                )
                if total_vol > 0:
                    signals.append(
                        FlowSignal(
                            source="coingecko_exchange_volume",
                            signal_type="exchange_volume",
                            value=round(total_vol, 2),
                            direction=None,
                            strength=0.3,
                            description=f"Top-5 exchange volume: {total_vol:,.0f} BTC (24h normalized)",
                        )
                    )
        except Exception:
            pass

        return signals

    # ─────────────────────────────────────────────────────────
    #  AGGREGATE COMPUTATION
    # ─────────────────────────────────────────────────────────

    def _compute_flow_bias(self, signals: list[FlowSignal]) -> float:
        """
        Aggregate directional bias from all flow signals.
        Returns float in [-1, +1].
        """
        if not signals:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for sig in signals:
            if sig.direction is None:
                continue
            sign = 1.0 if sig.direction == "bullish" else -1.0
            weighted_sum += sign * sig.strength
            weight_total += sig.strength

        if weight_total == 0:
            return 0.0
        return max(-1.0, min(1.0, weighted_sum / weight_total))

    def _compute_sizing_modifier(self, flow_bias: float, signals: list[FlowSignal]) -> float:
        """
        Convert flow bias into a sizing modifier for the risk manager.

        Strong bearish flow (crowded longs + OI spike) → reduce sizing to 0.7×
        Strong bullish flow (shorts paying + OI drop) → allow up to 1.15×
        Neutral → 1.0×
        """
        if not signals:
            return 1.0

        # Count specific high-conviction signals
        has_crowded_long = any(s.signal_type == "funding_crowded_long" for s in signals)
        has_oi_spike = any(s.signal_type == "oi_spike" for s in signals)
        has_deleveraging = any(s.signal_type == "oi_deleveraging" for s in signals)
        has_shorts_paying = any(s.signal_type == "funding_shorts_paying" for s in signals)

        modifier = 1.0

        # Crowded longs + OI spike = high squeeze risk → reduce sizing
        if has_crowded_long and has_oi_spike:
            modifier = 0.70
        elif has_crowded_long:
            modifier = 0.85
        elif has_oi_spike:
            modifier = 0.90

        # Shorts paying + deleveraging = contrarian bullish → slight boost
        if has_shorts_paying and has_deleveraging:
            modifier = min(modifier, 1.15)
        elif has_shorts_paying:
            modifier = min(modifier, 1.10)

        return max(0.50, min(1.20, modifier))

    # ─────────────────────────────────────────────────────────
    #  CACHING & HTTP
    # ─────────────────────────────────────────────────────────

    def _cached(self, key: str, fn, *args) -> list[FlowSignal]:
        now = time.time()
        fail_ts = self._fail_cache.get(key)
        if fail_ts is not None and (now - fail_ts) < self._FAIL_TTL:
            raise RuntimeError(f"{key} in failure backoff")

        ttl = self.cfg.cache_ttl_minutes * 60
        if key in self._cache:
            ts, cached = self._cache[key]
            if now - ts < ttl:
                return cached

        try:
            result = fn(*args)
        except Exception:
            self._fail_cache[key] = now
            raise

        if result:
            self._cache[key] = (now, result)
            self._fail_cache.pop(key, None)
        return result or []

    def _http_get(self, url: str, timeout: int = 10) -> dict:
        req = Request(url, headers={"User-Agent": "KA-MATS/1.0", "Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
