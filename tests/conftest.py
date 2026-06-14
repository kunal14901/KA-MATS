"""
KA-MATS Cryptoz · Pytest Configuration
Shared fixtures for all tests
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from core.models import (
    Features,
    Indicators,
    MarketSnapshot,
    PriceData,
    RegimeAnalysis,
    RegimeType,
)

# ─────────────────────────────────────────────────────────────
#  TEST ISOLATION
# ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_execution_state(tmp_path, monkeypatch):
    """
    Redirect execution state persistence to a per-test temp file.

    Without this, any test that constructs an execution agent and trades
    writes knowledge/.execution_state.json, which later tests (orchestrator
    integration, dashboard "not initialized" checks) then restore — causing
    order-dependent failures across the suite.
    """
    import agents.execution_agent as exec_mod

    monkeypatch.setattr(exec_mod, "_STATE_FILE", tmp_path / ".execution_state.json")
    try:
        import dashboard.server as server_mod

        monkeypatch.setattr(server_mod, "_STATE_FILE", tmp_path / ".execution_state.json")
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────
#  SAMPLE DATA FIXTURES
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate realistic OHLCV DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=300, freq="4h")

    # Generate synthetic price data with trend + noise
    np.random.seed(42)
    trend = np.linspace(40000, 45000, 300)
    noise = np.random.normal(0, 500, 300)
    close = trend + noise

    high = close * (1 + np.abs(np.random.normal(0, 0.01, 300)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, 300)))
    open_ = low + (high - low) * np.random.uniform(0.3, 0.7, 300)
    volume = np.random.uniform(1000, 5000, 300)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def sample_indicators() -> Indicators:
    """Sample indicators for a bullish trending market."""
    return Indicators(
        ema_20=44500.0,
        ema_50=43800.0,
        ema_200=42000.0,
        rsi_14=62.5,
        atr_14=850.0,
        bb_upper=46000.0,
        bb_middle=44500.0,
        bb_lower=43000.0,
        adx_14=28.5,
        plus_di=32.0,
        minus_di=18.0,
        macd=250.0,
        macd_signal=180.0,
    )


@pytest.fixture
def sample_features() -> Features:
    """Sample features for cross-rank and volume."""
    return Features(
        volume_ratio=1.35,
        dollar_volume_20d=120_000_000.0,
        zscore_20=1.2,
        cross_rank=0.75,
    )


@pytest.fixture
def sample_price_data() -> PriceData:
    """Sample price bar for BTC/USDT."""
    return PriceData(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=UTC),
        open=44200.0,
        high=44800.0,
        low=44000.0,
        close=44500.0,
        volume=2450.0,
    )


@pytest.fixture
def sample_snapshot(
    sample_price_data: PriceData,
    sample_indicators: Indicators,
    sample_features: Features,
) -> MarketSnapshot:
    """Complete market snapshot for testing."""
    return MarketSnapshot(
        symbol="BTC/USDT",
        price=sample_price_data,
        indicators=sample_indicators,
        features=sample_features,
        timestamp=sample_price_data.timestamp,
        data_quality_ok=True,
    )


@pytest.fixture
def sample_regime_trending_up() -> RegimeAnalysis:
    """Regime analysis for trending up market."""
    return RegimeAnalysis(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=UTC),
        regime=RegimeType.TRENDING_UP,
        confidence=0.85,
        trend_strength=28.5,
        volatility_pct=45.0,
        zscore=1.2,
        rationale="ADX=28.5 > 22.0, +DI > -DI by 14.0pts",
    )


@pytest.fixture
def sample_regime_ranging() -> RegimeAnalysis:
    """Regime analysis for ranging market."""
    return RegimeAnalysis(
        symbol="ETH/USDT",
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=UTC),
        regime=RegimeType.RANGING,
        confidence=0.65,
        trend_strength=18.2,
        volatility_pct=35.0,
        zscore=0.3,
        rationale="ADX=18.2 < 22.0 (weak trend), no vol/zscore extremes",
    )


@pytest.fixture
def multi_symbol_snapshots(sample_snapshot: MarketSnapshot) -> dict[str, MarketSnapshot]:
    """Multiple symbol snapshots for cross-rank testing."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    snapshots = {}

    for i, sym in enumerate(symbols):
        # Create variation of base snapshot
        snap = sample_snapshot.model_copy(deep=True)
        snap.symbol = sym
        snap.price.symbol = sym
        snap.price.close = 44500.0 * (1 + i * 0.1)  # Vary prices
        if snap.features:
            snap.features.cross_rank = i / (len(symbols) - 1)  # 0, 0.33, 0.67, 1.0
        snapshots[sym] = snap

    return snapshots


# ─────────────────────────────────────────────────────────────
#  MOCK FIXTURES
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_portfolio():
    """Mock portfolio for testing risk manager and execution."""
    from core.models import Portfolio

    return Portfolio(
        cash=10000.0,
        positions={},
        closed_trades=[],
        initial_capital=10000.0,
    )


@pytest.fixture
def temp_knowledge_dir(tmp_path):
    """Temporary knowledge directory for testing knowledge agent."""
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    papers_dir = kb_dir / "papers"
    papers_dir.mkdir()

    # Create sample paper
    paper = papers_dir / "momentum_crypto.txt"
    paper.write_text(
        "Cryptocurrency momentum strategies demonstrate strong persistence "
        "over 20-day periods. Cross-sectional momentum works best in trending "
        "markets with ADX > 25. Risk-adjusted returns improve with volatility "
        "scaling using ATR-based position sizing."
    )

    return kb_dir


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION FIXTURES
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def clean_config():
    """Reset CONFIG to defaults after test."""
    from config.settings import CONFIG

    CONFIG.model_copy(deep=True)
    yield CONFIG

    # Restore original (for isolation between tests)
    # Note: In-place mutation of global CONFIG


@pytest.fixture(autouse=True)
def disable_logging(caplog):
    """Suppress verbose logging during tests unless explicitly enabled."""
    import logging

    caplog.set_level(logging.WARNING)


# ─────────────────────────────────────────────────────────────
#  NETWORK MOCKING
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ccxt_exchange(monkeypatch):
    """Mock ccxt exchange for testing without network calls."""

    class MockExchange:
        def __init__(self, config=None):
            self.config = config or {}

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            """Return synthetic OHLCV data."""
            dates = pd.date_range(start="2024-01-01", periods=limit or 100, freq="4h")
            data = []
            for ts in dates:
                ms = int(ts.timestamp() * 1000)
                data.append(
                    [
                        ms,  # timestamp
                        44000 + np.random.uniform(-200, 200),  # open
                        44200 + np.random.uniform(-200, 200),  # high
                        43800 + np.random.uniform(-200, 200),  # low
                        44100 + np.random.uniform(-200, 200),  # close
                        np.random.uniform(1000, 3000),  # volume
                    ]
                )
            return data

    def mock_binance(*args, **kwargs):
        return MockExchange(kwargs)

    monkeypatch.setattr("ccxt.binance", mock_binance)
    return mock_binance
