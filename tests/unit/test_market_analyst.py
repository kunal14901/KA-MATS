"""Unit tests for Market Analyst Agent."""

from datetime import UTC, datetime, timezone

import pytest

from agents.market_analyst import MarketAnalystAgent
from core.models import (
    Features,
    Indicators,
    MarketSnapshot,
    PriceData,
    RegimeType,
)


@pytest.mark.unit
class TestMarketAnalyst:
    """Test regime classification logic."""

    def test_trending_up_detection(self):
        """Test detection of upward trending regime."""
        agent = MarketAnalystAgent()

        # Create bullish snapshot: high ADX, +DI > -DI
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=PriceData(
                symbol="BTC/USDT",
                timestamp=datetime.now(UTC),
                open=44000.0,
                high=44800.0,
                low=43900.0,
                close=44500.0,
                volume=2000.0,
            ),
            indicators=Indicators(
                adx_14=30.0,  # Above threshold (22)
                plus_di=35.0,
                minus_di=15.0,  # +DI > -DI by 20pts
                atr_14=850.0,
                ema_20=44500.0,
                ema_50=43800.0,
                ema_200=42000.0,
            ),
            features=Features(zscore_20=1.0),
            timestamp=datetime.now(UTC),
            data_quality_ok=True,
        )

        result = agent.analyse(snapshot)

        assert result.regime == RegimeType.TRENDING_UP
        assert result.confidence > 0.6
        assert result.trend_strength == 30.0

    def test_trending_down_detection(self):
        """Test detection of downward trending regime."""
        agent = MarketAnalystAgent()

        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=PriceData(
                symbol="BTC/USDT",
                timestamp=datetime.now(UTC),
                open=44000.0,
                high=44200.0,
                low=43500.0,
                close=43700.0,
                volume=2000.0,
            ),
            indicators=Indicators(
                adx_14=32.0,
                plus_di=12.0,
                minus_di=38.0,  # -DI > +DI significantly
                atr_14=900.0,
            ),
            features=Features(zscore_20=-1.2),
            timestamp=datetime.now(UTC),
            data_quality_ok=True,
        )

        result = agent.analyse(snapshot)

        assert result.regime == RegimeType.TRENDING_DOWN
        assert result.confidence > 0.6

    def test_ranging_detection(self):
        """Test detection of ranging (sideways) market."""
        agent = MarketAnalystAgent()

        snapshot = MarketSnapshot(
            symbol="ETH/USDT",
            price=PriceData(
                symbol="ETH/USDT",
                timestamp=datetime.now(UTC),
                open=2400.0,
                high=2420.0,
                low=2390.0,
                close=2410.0,
                volume=1500.0,
            ),
            indicators=Indicators(
                adx_14=15.0,  # Below trending threshold
                plus_di=22.0,
                minus_di=20.0,  # DI values close
                atr_14=45.0,
            ),
            features=Features(zscore_20=0.3),
            timestamp=datetime.now(UTC),
            data_quality_ok=True,
        )

        result = agent.analyse(snapshot)

        assert result.regime == RegimeType.RANGING
        assert result.confidence > 0.5

    def test_volatile_detection(self):
        """Test detection of volatile regime with high ATR."""
        agent = MarketAnalystAgent()

        snapshot = MarketSnapshot(
            symbol="SOL/USDT",
            price=PriceData(
                symbol="SOL/USDT",
                timestamp=datetime.now(UTC),
                open=100.0,
                high=105.0,
                low=97.0,
                close=102.0,
                volume=5000.0,
            ),
            indicators=Indicators(
                adx_14=18.0,  # Low ADX (not trending)
                atr_14=3.5,  # High ATR relative to price (3.5/102 = 3.4%)
            ),
            features=Features(zscore_20=0.5),
            timestamp=datetime.now(UTC),
            data_quality_ok=True,
        )

        result = agent.analyse(snapshot)

        assert result.regime == RegimeType.VOLATILE

    def test_mean_reverting_detection(self):
        """Test detection of mean-reverting regime."""
        agent = MarketAnalystAgent()

        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=PriceData(
                symbol="BTC/USDT",
                timestamp=datetime.now(UTC),
                open=44000.0,
                high=44500.0,
                low=43900.0,
                close=44200.0,
                volume=2000.0,
            ),
            indicators=Indicators(
                adx_14=16.0,  # Low trend strength
                atr_14=800.0,
            ),
            features=Features(zscore_20=2.5),  # Extended from mean
            timestamp=datetime.now(UTC),
            data_quality_ok=True,
        )

        result = agent.analyse(snapshot)

        assert result.regime == RegimeType.MEAN_REVERTING
        assert result.zscore == 2.5

    def test_handles_missing_data(self):
        """Test graceful handling of missing indicators."""
        agent = MarketAnalystAgent()

        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=PriceData(
                symbol="BTC/USDT",
                timestamp=datetime.now(UTC),
                open=44000.0,
                high=44500.0,
                low=43900.0,
                close=44200.0,
                volume=2000.0,
            ),
            indicators=Indicators(
                adx_14=None,  # Missing critical data
                atr_14=None,
            ),
            features=Features(),
            timestamp=datetime.now(UTC),
            data_quality_ok=False,
        )

        result = agent.analyse(snapshot)

        assert result.regime == RegimeType.UNKNOWN
        assert result.confidence == 0.0

    def test_confidence_scaling(self):
        """Test that confidence scales with signal strength."""
        agent = MarketAnalystAgent()

        # Strong trend (high ADX)
        strong_snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            price=PriceData(
                symbol="BTC/USDT",
                timestamp=datetime.now(UTC),
                open=44000.0,
                high=44800.0,
                low=43900.0,
                close=44500.0,
                volume=2000.0,
            ),
            indicators=Indicators(
                adx_14=45.0,  # Very high ADX
                plus_di=40.0,
                minus_di=10.0,
                atr_14=850.0,
            ),
            features=Features(zscore_20=1.5),
            timestamp=datetime.now(UTC),
            data_quality_ok=True,
        )

        # Weak trend (barely above threshold)
        weak_snapshot = strong_snapshot.model_copy(deep=True)
        weak_snapshot.indicators.adx_14 = 23.0
        weak_snapshot.indicators.plus_di = 27.0
        weak_snapshot.indicators.minus_di = 20.0

        strong_result = agent.analyse(strong_snapshot)
        weak_result = agent.analyse(weak_snapshot)

        assert strong_result.confidence > weak_result.confidence
        assert strong_result.regime == RegimeType.TRENDING_UP
        assert weak_result.regime == RegimeType.TRENDING_UP
