"""Tests for the paper-only 4h short shadow profile."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agents.short_shadow_strategy import ShortShadow4hStrategy
from core.models import (
    Features,
    Indicators,
    MarketSnapshot,
    PriceData,
    RegimeAnalysis,
    RegimeType,
    SignalDirection,
)


def _short_snapshot() -> MarketSnapshot:
    now = datetime.now(UTC)
    return MarketSnapshot(
        symbol="ARB/USDT",
        price=PriceData(
            symbol="ARB/USDT",
            timestamp=now,
            open=1.05,
            high=1.07,
            low=0.98,
            close=1.0,
            volume=2_000_000,
        ),
        indicators=Indicators(
            ema_20=0.98,
            ema_50=1.08,
            ema_200=1.35,
            rsi_14=45.0,
            atr_14=0.04,
            adx_14=32.0,
            plus_di=12.0,
            minus_di=34.0,
        ),
        features=Features(
            cross_rank=0.20,
            volume_ratio=1.1,
            dollar_volume_20d=50_000_000,
        ),
        timestamp=now,
        data_quality_ok=True,
    )


def test_short_shadow_emits_sell_signal():
    strategy = ShortShadow4hStrategy()
    snapshot = _short_snapshot()
    regime = RegimeAnalysis(
        symbol=snapshot.symbol,
        timestamp=snapshot.timestamp,
        regime=RegimeType.TRENDING_DOWN,
        confidence=0.9,
        trend_strength=32.0,
    )

    signals = strategy.evaluate(snapshot, regime)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.direction == SignalDirection.SELL
    assert signal.stop_price > snapshot.price.close
    assert signal.target_price < snapshot.price.close
    assert signal.strategy_name == "ShortShadow4h"
