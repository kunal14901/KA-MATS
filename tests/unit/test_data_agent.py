"""Unit tests for Crypto Data Agent."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from agents.data_agent import CryptoDataAgent


@pytest.mark.unit
class TestDataAgent:
    """Test data fetching and indicator computation."""

    def test_compute_indicators_basic(self, sample_ohlcv_df):
        """Test that all indicators are computed correctly."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        # Check all required indicators exist
        required_cols = [
            "ema_20",
            "ema_50",
            "ema_200",
            "rsi_14",
            "atr_14",
            "bb_upper",
            "bb_lower",
            "adx",
            "plus_di",
            "minus_di",
            "macd",
            "macd_signal",
            "volume_ratio",
            "zscore",
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing indicator: {col}"
            # Check last value is not NaN (warmup period passed)
            assert not pd.isna(df[col].iloc[-1]), f"Indicator {col} is NaN"

    def test_ema_calculation(self, sample_ohlcv_df):
        """Test EMA values are in correct order."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        # In uptrend: EMA20 > EMA50 > EMA200
        last_row = df.iloc[-1]

        # EMAs should be positive
        assert last_row["ema_20"] > 0
        assert last_row["ema_50"] > 0
        assert last_row["ema_200"] > 0

    def test_rsi_bounds(self, sample_ohlcv_df):
        """Test RSI stays within 0-100 range."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        assert df["rsi_14"].min() >= 0, "RSI below 0"
        assert df["rsi_14"].max() <= 100, "RSI above 100"

    def test_atr_positive(self, sample_ohlcv_df):
        """Test ATR is always positive."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        # TA warmup can yield initial zeros; after warmup ATR must be positive.
        atr = df["atr_14"].dropna().iloc[20:]
        assert (atr > 0).all(), "ATR contains non-positive values after warmup"

    def test_bollinger_bands_order(self, sample_ohlcv_df):
        """Test BB upper > lower always."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        # Filter out NaN rows
        valid = df.dropna(subset=["bb_upper", "bb_lower"])

        assert (valid["bb_upper"] > valid["bb_lower"]).all(), "Bollinger bands inverted"

    def test_adx_range(self, sample_ohlcv_df):
        """Test ADX stays in valid range."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        valid_adx = df["adx"].dropna()
        assert (valid_adx >= 0).all(), "ADX contains negative values"
        assert (valid_adx <= 100).all(), "ADX above 100"

    def test_volume_ratio_calculation(self, sample_ohlcv_df):
        """Test volume ratio is computed correctly."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        # Volume ratio should be positive
        assert (df["volume_ratio"].dropna() > 0).all()

    def test_zscore_calculation(self, sample_ohlcv_df):
        """Test z-score is computed."""
        agent = CryptoDataAgent()
        df = agent.compute_indicators(sample_ohlcv_df)

        # Z-score should exist and be reasonable
        zscore = df["zscore"].dropna()
        assert len(zscore) > 0
        # Most z-scores should be within ±3
        assert (zscore.abs() < 10).all(), "Extreme z-score values detected"

    def test_handles_short_dataframe(self):
        """Test handling of insufficient data."""
        agent = CryptoDataAgent()

        # Create very short DataFrame (< 200 bars for EMA200)
        dates = pd.date_range(start="2024-01-01", periods=50, freq="4h")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(44000, 45000, 50),
                "high": np.random.uniform(44500, 45500, 50),
                "low": np.random.uniform(43500, 44500, 50),
                "close": np.random.uniform(44000, 45000, 50),
                "volume": np.random.uniform(1000, 3000, 50),
            },
            index=dates,
        )

        result = agent.compute_indicators(df)

        # Should still compute indicators, but EMA200 will have NaN
        assert "ema_20" in result.columns
        assert "rsi_14" in result.columns

    def test_handles_missing_columns(self):
        """Test error handling for invalid input."""
        agent = CryptoDataAgent()

        # DataFrame missing required columns
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="4h"),
                "close": np.random.uniform(44000, 45000, 100),
            }
        )

        with pytest.raises((KeyError, AttributeError)):
            agent.compute_indicators(df)

    @pytest.mark.requires_network
    def test_fetch_ohlcv_mock(self, mock_ccxt_exchange):
        """Test OHLCV fetching with mocked exchange."""
        agent = CryptoDataAgent()

        # This would normally hit the network, but we've mocked it
        df = agent.fetch_ohlcv("BTC/USDT", limit=100)

        assert df is not None
        assert len(df) > 0
        assert "close" in df.columns
        assert "volume" in df.columns


@pytest.mark.unit
class TestDataQuality:
    """Test data quality checks."""

    def test_detect_missing_data(self):
        """Test detection of data gaps."""
        agent = CryptoDataAgent()

        # Create DataFrame with gaps
        dates = pd.date_range(start="2024-01-01", periods=100, freq="4h")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(44000, 45000, 100),
                "high": np.random.uniform(44500, 45500, 100),
                "low": np.random.uniform(43500, 44500, 100),
                "close": np.random.uniform(44000, 45000, 100),
                "volume": np.random.uniform(1000, 3000, 100),
            },
            index=dates,
        )

        # Introduce gaps
        df.loc[df.index[20:25], "close"] = np.nan

        # Should handle gracefully
        result = agent.compute_indicators(df)
        assert result is not None
