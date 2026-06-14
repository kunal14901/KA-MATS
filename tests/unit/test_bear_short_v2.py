"""
Unit tests for BearShort v2 — live parity with the validated backtest logic.

Covers:
  1. Fail-closed behavior: no shorts without BTC macro context
  2. The triple BTC gate (bear + rollover + active-decline band)
  3. Coin-level v2 gates (RSI band, close < EMA20)
  4. Risk manager half-size sizing for SHORT signals
"""

from datetime import UTC, datetime, timezone

import pytest

from agents.risk_manager import CryptoRiskManager
from agents.strategy_agent import CryptoBearShortStrategy, CryptoStrategyAgent
from core.adaptive_learner import AdaptiveLearner
from core.models import (
    CandidateSignal,
    Features,
    Indicators,
    MarketSnapshot,
    PortfolioState,
    PriceData,
    RegimeAnalysis,
    RegimeType,
    SignalDirection,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _bear_snapshot(
    close: float = 0.95,
    ema20: float = 1.00,
    ema50: float = 1.10,
    ema200: float = 1.40,
    rsi: float = 45.0,
    atr: float = 0.05,
    volume_ratio: float = 1.1,
    cross_rank: float = 0.20,
) -> MarketSnapshot:
    """Coin snapshot satisfying every coin-level BearShort v2 condition."""
    now = datetime.now(UTC)
    return MarketSnapshot(
        symbol="VET/USDT",
        price=PriceData(
            symbol="VET/USDT",
            timestamp=now,
            open=0.97,
            high=0.99,
            low=0.94,
            close=close,
            volume=2_000_000,
        ),
        indicators=Indicators(
            ema_20=ema20,
            ema_50=ema50,
            ema_200=ema200,
            rsi_14=rsi,
            atr_14=atr,
            adx_14=30.0,
        ),
        features=Features(
            cross_rank=cross_rank,
            volume_ratio=volume_ratio,
            dollar_volume_20d=50_000_000,
        ),
        timestamp=now,
        data_quality_ok=True,
    )


def _bear_regime() -> RegimeAnalysis:
    return RegimeAnalysis(
        symbol="VET/USDT",
        timestamp=datetime.now(UTC),
        regime=RegimeType.TRENDING_DOWN,
        confidence=0.85,
    )


def _evaluate(agent: CryptoStrategyAgent):
    return agent.evaluate(
        snapshot=_bear_snapshot(),
        regime=_bear_regime(),
        cross_rank=0.20,
    )


# ── Strategy gate tests ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestBearShortV2Gates:
    def test_fails_closed_without_macro_context(self):
        """No BTC context injected → no shorts, ever (fail-closed)."""
        agent = CryptoStrategyAgent()
        signals = _evaluate(agent)
        assert all(s.direction != SignalDirection.SELL for s in signals)

    def test_agent_blocks_bear_short_v13_policy(self):
        """v13: BearShort is PERMANENT_DISABLED at the agent level — even a
        perfect bear setup must produce no SELL through the dispatcher.
        (Negative expectancy under honest intrabar fills; see
        VALIDATION_METHODOLOGY.md §0.8.)"""
        agent = CryptoStrategyAgent()
        agent.set_macro_context(btc_rollover=True, btc_roc20=-0.12)
        assert "CryptoBearShort" in agent.PERMANENT_DISABLED
        signals = _evaluate(agent)
        assert all(s.direction != SignalDirection.SELL for s in signals)

    def test_strategy_fires_standalone_with_valid_context_and_setup(self):
        """The strategy class itself still works (research/backtest use):
        bear + rollover + active decline + valid coin setup → SELL signal."""
        strat = CryptoBearShortStrategy()
        strat.btc_rollover = True
        strat.btc_roc20 = -0.12
        signals = strat.evaluate(_bear_snapshot(), _bear_regime())
        sells = [s for s in signals if s.direction == SignalDirection.SELL]
        assert len(sells) == 1
        sig = sells[0]
        assert sig.strategy_name == "CryptoBearShort"
        assert sig.stop_price > sig.entry_price  # stop above entry (short)
        assert sig.target_price < sig.entry_price  # target below entry

    def test_blocked_when_btc_bouncing(self):
        """BTC EMA20 > EMA50 (squeeze rally underway) → no shorts."""
        agent = CryptoStrategyAgent()
        agent.set_macro_context(btc_rollover=False, btc_roc20=-0.12)
        signals = _evaluate(agent)
        assert all(s.direction != SignalDirection.SELL for s in signals)

    def test_blocked_post_capitulation(self):
        """BTC 20d return below -25% → crash exhausted, squeeze imminent → no shorts."""
        agent = CryptoStrategyAgent()
        agent.set_macro_context(btc_rollover=True, btc_roc20=-0.30)
        signals = _evaluate(agent)
        assert all(s.direction != SignalDirection.SELL for s in signals)

    def test_blocked_when_decline_not_active(self):
        """BTC 20d return above -5% (basing/chop) → no shorts."""
        agent = CryptoStrategyAgent()
        agent.set_macro_context(btc_rollover=True, btc_roc20=-0.02)
        signals = _evaluate(agent)
        assert all(s.direction != SignalDirection.SELL for s in signals)

    def test_blocked_when_coin_oversold(self):
        """Coin RSI below 38 → never short the bottom."""
        strat = CryptoBearShortStrategy()
        strat.btc_rollover = True
        strat.btc_roc20 = -0.12
        signals = strat.evaluate(_bear_snapshot(rsi=33.0), _bear_regime())
        assert signals == []

    def test_blocked_when_price_above_ema20(self):
        """Coin price above EMA20 → mid-bounce, rollover not confirmed."""
        strat = CryptoBearShortStrategy()
        strat.btc_rollover = True
        strat.btc_roc20 = -0.12
        signals = strat.evaluate(_bear_snapshot(close=1.02, ema20=1.00), _bear_regime())
        assert signals == []

    def test_blocked_when_coin_not_weak(self):
        """cross_rank above 0.30 (not a bottom-30% coin) → no short."""
        strat = CryptoBearShortStrategy()
        strat.btc_rollover = True
        strat.btc_roc20 = -0.12
        signals = strat.evaluate(_bear_snapshot(cross_rank=0.55), _bear_regime())
        assert signals == []


# ── Risk manager half-size tests ─────────────────────────────────────────────


def _portfolio() -> PortfolioState:
    return PortfolioState(
        initial_capital=10000.0,
        cash=10000.0,
        net_equity=10000.0,
        peak_equity=10000.0,
        positions={},
    )


def _signal(direction: SignalDirection) -> CandidateSignal:
    """BUY and SELL variants with identical stop distance and confidence.

    Stop distance is wide (20%) so risk-based sizing stays below the
    position-notional cap and the direction multiplier is observable.
    """
    entry = 100.0
    if direction == SignalDirection.BUY:
        stop, target = 80.0, 140.0
    else:
        stop, target = 120.0, 60.0
    return CandidateSignal(
        symbol="VET/USDT",
        timestamp=datetime.now(UTC),
        direction=direction,
        strategy_name="TestStrategy",
        confidence=0.75,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        dollar_volume_20d=120_000_000.0,
        conditions=[],
    )


def _regime(regime: RegimeType) -> RegimeAnalysis:
    return RegimeAnalysis(
        symbol="VET/USDT",
        timestamp=datetime.now(UTC),
        regime=regime,
        confidence=0.85,
    )


@pytest.mark.unit
class TestShortHalfSizing:
    def test_short_sized_at_half_of_long(self):
        """A SHORT with identical risk parameters gets ~0.5× the LONG size."""
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())

        buy = risk_mgr.evaluate(
            signal=_signal(SignalDirection.BUY),
            portfolio=_portfolio(),
            regime=_regime(RegimeType.TRENDING_UP),
        )
        sell = risk_mgr.evaluate(
            signal=_signal(SignalDirection.SELL),
            portfolio=_portfolio(),
            regime=_regime(RegimeType.TRENDING_UP),
        )

        assert buy.approved and sell.approved
        ratio = sell.position_size / buy.position_size
        assert ratio == pytest.approx(0.5, rel=0.10)
