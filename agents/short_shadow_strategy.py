"""
KA-MATS Crypto · 4h Short Shadow Strategy

Paper/shadow-only strategy generated from the 4h Sharpe research pass.
It emits SELL signals for weak coins in confirmed 4h downtrends.
"""

from __future__ import annotations

from config.settings import CONFIG, StrategyConfig
from core.models import (
    CandidateSignal,
    MarketSnapshot,
    RegimeAnalysis,
    RegimeType,
    SignalCondition,
    SignalDirection,
)


class ShortShadow4hStrategy:
    """Deployable paper-only wrapper for the best 4h short research candidate."""

    name = "ShortShadow4h"
    VALID_REGIMES = {RegimeType.TRENDING_DOWN}

    # Winning research_0028 parameters.
    adx_min = 20.0
    rank_max = 0.30
    rsi_min = 25.0
    rsi_max = 62.0
    vol_min = 0.60
    stop_atr = 2.5
    target_atr = 4.0
    max_hold_bars = 36

    def __init__(self, cfg: StrategyConfig = None, learner=None) -> None:
        self.cfg = cfg or CONFIG.strategy
        self._learner = learner
        self.last_reject_reason = "not_evaluated"

    def evaluate(self, snapshot: MarketSnapshot, regime: RegimeAnalysis) -> list[CandidateSignal]:
        self.last_reject_reason = "not_evaluated"
        if regime.regime not in self.VALID_REGIMES:
            return self._reject("regime_filter")

        ind = snapshot.indicators
        feat = snapshot.features
        ema20 = ind.ema_20
        ema50 = ind.ema_50
        ema200 = ind.ema_200
        rsi = ind.rsi_14
        atr = ind.atr_14
        adx = ind.adx_14
        close = snapshot.price.close
        cross_rank = feat.cross_rank if feat else None
        vol_ratio = feat.volume_ratio if feat else None

        if any(v is None for v in [ema20, ema50, rsi, atr, adx]):
            return self._reject("missing_indicators")
        if atr <= 0 or close <= 0:
            return self._reject("invalid_price_or_atr")
        if ema200 is not None and close >= ema200:
            return self._reject("macro_filter")
        if not (ema20 < ema50):
            return self._reject("ema_filter")
        if adx < self.adx_min:
            return self._reject("adx_filter")
        if cross_rank is not None and cross_rank > self.rank_max:
            return self._reject("rank_filter")
        if not (self.rsi_min <= rsi <= self.rsi_max):
            return self._reject("rsi_filter")
        if vol_ratio is not None and vol_ratio < self.vol_min:
            return self._reject("volume_filter")

        confidence = 0.62
        if cross_rank is not None:
            confidence += min(0.08, max(0.0, (self.rank_max - cross_rank) * 0.25))
        if self._learner is not None:
            confidence += self._learner.strategy_modifier(self.name, regime.regime.value)
            confidence += self._learner.symbol_confidence(snapshot.symbol)
        confidence = max(0.0, min(1.0, confidence))
        if confidence < self.cfg.min_signal_confidence:
            return self._reject("confidence_filter")

        stop = close + self.stop_atr * atr
        target = close - self.target_atr * atr
        if target <= 0:
            return self._reject("invalid_stop_target")

        self.last_reject_reason = ""
        return [
            CandidateSignal(
                symbol=snapshot.symbol,
                timestamp=snapshot.price.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SELL,
                confidence=round(confidence, 4),
                entry_price=close,
                stop_price=round(stop, 8),
                target_price=round(target, 8),
                dollar_volume_20d=(feat.dollar_volume_20d if feat else None),
                conditions=[
                    SignalCondition(name="regime_down", passed=True, description="regime=trending_down"),
                    SignalCondition(
                        name="ema_downtrend", passed=True, description=f"EMA20={ema20:.4f}<EMA50={ema50:.4f}"
                    ),
                    SignalCondition(name="weak_rank", passed=True, value=cross_rank, threshold=self.rank_max),
                    SignalCondition(name="rsi_window", passed=True, value=rsi, threshold=self.rsi_max),
                    SignalCondition(name="adx", passed=True, value=adx, threshold=self.adx_min),
                ],
            )
        ]

    def _reject(self, reason: str) -> list[CandidateSignal]:
        self.last_reject_reason = reason
        return []
