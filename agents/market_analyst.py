"""
KA-MATS · Market Analyst Agent
Iknir Capital — Phase I Foundation

Responsibilities:
  - Detect market regimes: trending_up, trending_down, ranging, volatile, mean_reverting
  - Measure volatility structure
  - Provide contextual classification for strategy selection

Output: RegimeAnalysis {regime, confidence, trend_strength, volatility_pct, zscore}
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from loguru import logger

from config.settings import CONFIG, RegimeConfig
from core.models import MarketSnapshot, RegimeAnalysis, RegimeType


class MarketAnalystAgent:
    """
    Classifies the current market environment into one of four regimes:
      - TRENDING_UP / TRENDING_DOWN: Strong directional move (high ADX)
      - RANGING: Low trend strength, price oscillating around mean
      - VOLATILE: High ATR relative to historical, unclear direction
      - MEAN_REVERTING: Extended deviation from mean with low trend strength

    Regime classification gates strategy selection in the Strategy Agent.
    """

    def __init__(self, cfg: RegimeConfig = None) -> None:
        self.cfg = cfg or CONFIG.regime

    # ─────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────

    def analyse(self, snapshot: MarketSnapshot) -> RegimeAnalysis:
        """
        Classify the market regime from a MarketSnapshot.

        Args:
            snapshot: Validated MarketSnapshot from the Data Agent

        Returns:
            RegimeAnalysis with regime type, confidence, and supporting metrics
        """
        symbol = snapshot.symbol
        logger.debug(f"[MarketAnalyst] Analysing {symbol}")

        if not snapshot.data_quality_ok:
            return self._unknown_regime(symbol, snapshot.timestamp, "Data quality failed")

        ind = snapshot.indicators
        feat = snapshot.features

        # ── Extract key metrics ────────────────────────────
        adx = ind.adx_14
        plus_di = ind.plus_di
        minus_di = ind.minus_di
        atr = ind.atr_14
        close = snapshot.price.close
        zscore = feat.zscore_20
        vol_20d = feat.volatility_20d

        if adx is None or atr is None:
            return self._unknown_regime(symbol, snapshot.timestamp, "Insufficient indicator data")

        # ── Regime scoring ─────────────────────────────────
        regime, confidence, rationale = self._classify(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            atr=atr,
            close=close,
            zscore=zscore,
            vol_20d=vol_20d,
        )

        # ── Volatility percentile (simplified: compare ATR to 20-bar SMA of ATR)
        # We use vol_20d as a proxy since we don't carry full history here
        vol_pct = self._vol_percentile(vol_20d)

        zscore_text = f"{zscore:.2f}" if zscore is not None else "N/A"
        logger.info(
            f"[MarketAnalyst] {symbol} → {regime.value} "
            f"(conf={confidence:.2f}, ADX={adx:.1f}, z={zscore_text})"
        )

        return RegimeAnalysis(
            symbol=symbol,
            timestamp=snapshot.timestamp,
            regime=regime,
            confidence=confidence,
            trend_strength=adx,
            volatility_pct=vol_pct,
            zscore=zscore,
            rationale=rationale,
        )

    # ─────────────────────────────────────────────────────────
    #  REGIME CLASSIFICATION LOGIC
    # ─────────────────────────────────────────────────────────

    def _classify(
        self,
        adx: float,
        plus_di: float | None,
        minus_di: float | None,
        atr: float,
        close: float,
        zscore: float | None,
        vol_20d: float | None,
    ) -> tuple[RegimeType, float, str]:
        """
        Deterministic regime classification using ADX, DI, ATR, and z-score.

        Decision hierarchy:
          1. High ADX + DI difference → TRENDING (up or down)
          2. High volatility (ATR%) → VOLATILE
          3. Extended z-score + low ADX → MEAN_REVERTING
          4. Default → RANGING
        """
        cfg = self.cfg
        atr_pct = (atr / close * 100) if close > 0 else 0.0

        # ── Rule 1: Trending regime ────────────────────────
        if adx >= cfg.adx_trend_threshold:
            direction_known = (plus_di is not None) and (minus_di is not None)
            if direction_known:
                di_diff = plus_di - minus_di
                if di_diff > 5:
                    confidence = min(1.0, (adx - cfg.adx_trend_threshold) / 25.0 + 0.5)
                    return (
                        RegimeType.TRENDING_UP,
                        round(confidence, 2),
                        f"ADX={adx:.1f} > {cfg.adx_trend_threshold}, +DI > -DI by {di_diff:.1f}pts",
                    )
                elif di_diff < -5:
                    confidence = min(1.0, (adx - cfg.adx_trend_threshold) / 25.0 + 0.5)
                    return (
                        RegimeType.TRENDING_DOWN,
                        round(confidence, 2),
                        f"ADX={adx:.1f} > {cfg.adx_trend_threshold}, -DI > +DI by {abs(di_diff):.1f}pts",
                    )

        # ── Rule 2: Volatile regime ────────────────────────
        # High ATR as % of price — market is noisy with no direction
        high_vol_threshold = 2.5  # ATR > 2.5% of price
        if atr_pct > high_vol_threshold and adx < cfg.adx_trend_threshold:
            confidence = min(1.0, (atr_pct - high_vol_threshold) / 2.0 + 0.4)
            return (
                RegimeType.VOLATILE,
                round(confidence, 2),
                f"ATR={atr_pct:.2f}% of price (high), ADX={adx:.1f} (non-trending)",
            )

        # ── Rule 3: Mean-reverting regime ─────────────────
        # Extended z-score + low ADX = price stretched from mean, likely to snap back
        if zscore is not None:
            abs_z = abs(zscore)
            if abs_z >= cfg.mean_revert_zscore_threshold and adx < cfg.adx_trend_threshold:
                confidence = min(1.0, (abs_z - cfg.mean_revert_zscore_threshold) / 1.5 + 0.5)
                return (
                    RegimeType.MEAN_REVERTING,
                    round(confidence, 2),
                    f"|Z-score|={abs_z:.2f} >= {cfg.mean_revert_zscore_threshold}, ADX={adx:.1f} (weak trend)",
                )

        # ── Rule 4: Default → Ranging ─────────────────────
        confidence = 0.5 + max(0.0, (cfg.adx_trend_threshold - adx) / cfg.adx_trend_threshold) * 0.3
        return (
            RegimeType.RANGING,
            round(confidence, 2),
            f"ADX={adx:.1f} < {cfg.adx_trend_threshold} (weak trend), no vol/zscore extremes",
        )

    # ─────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────

    def _vol_percentile(self, vol_20d: float | None) -> float | None:
        """
        Convert annualised vol to a rough percentile (0-100).
        Typical equity vol ranges: 10% (low) → 80% (crisis).
        """
        if vol_20d is None:
            return None
        # Clamp and scale to 0-100
        vol_pct = np.clip(vol_20d * 100, 0, 100)
        # Approximate percentile: 15% vol ≈ 50th percentile for SPY
        percentile = np.clip((vol_pct - 5) / (80 - 5) * 100, 0, 100)
        return round(float(percentile), 1)

    def _unknown_regime(self, symbol: str, timestamp: datetime, reason: str) -> RegimeAnalysis:
        logger.warning(f"[MarketAnalyst] {symbol}: UNKNOWN regime — {reason}")
        return RegimeAnalysis(
            symbol=symbol,
            timestamp=timestamp,
            regime=RegimeType.UNKNOWN,
            confidence=0.0,
            rationale=reason,
        )
