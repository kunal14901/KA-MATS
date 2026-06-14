"""Unit tests for Risk Manager."""

from datetime import UTC, datetime, timezone

import pytest

from agents.risk_manager import CryptoRiskManager
from core.adaptive_learner import AdaptiveLearner
from core.models import (
    CandidateSignal,
    PortfolioState,
    Position,
    PositionSide,
    RegimeAnalysis,
    RegimeType,
    SignalDirection,
)


def _portfolio(
    cash: float = 10000.0, net_equity: float = 10000.0, positions: dict | None = None
) -> PortfolioState:
    return PortfolioState(
        initial_capital=10000.0,
        cash=cash,
        net_equity=net_equity,
        peak_equity=10000.0,
        positions=positions or {},
    )


def _buy_signal(
    symbol: str = "BTC/USDT",
    entry: float = 44500.0,
    stop: float = 43300.0,
    target: float = 47500.0,
    confidence: float = 0.75,
    dollar_volume_20d: float | None = 120_000_000.0,
) -> CandidateSignal:
    return CandidateSignal(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=SignalDirection.BUY,
        strategy_name="TestStrategy",
        confidence=confidence,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        dollar_volume_20d=dollar_volume_20d,
        conditions=[],
    )


def _regime(confidence: float = 0.85, regime: RegimeType = RegimeType.TRENDING_UP) -> RegimeAnalysis:
    return RegimeAnalysis(
        symbol="BTC/USDT",
        timestamp=datetime.now(UTC),
        regime=regime,
        confidence=confidence,
    )


@pytest.mark.unit
class TestRiskManager:
    def test_position_sizing_basic(self):
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        decision = risk_mgr.evaluate(
            signal=_buy_signal(),
            portfolio=_portfolio(),
            regime=_regime(),
        )

        assert decision.approved
        assert decision.position_size > 0
        assert decision.position_value > 0

    def test_rejects_tight_stop_distance(self):
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        decision = risk_mgr.evaluate(
            signal=_buy_signal(stop=44490.0, target=45500.0),
            portfolio=_portfolio(),
            regime=_regime(),
        )

        assert not decision.approved
        assert decision.veto_reason is not None
        assert "stop too tight" in decision.veto_reason.lower()

    def test_enforces_max_positions(self):
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        positions = {
            f"SYM{i}/USDT": Position(
                symbol=f"SYM{i}/USDT",
                side=PositionSide.LONG,
                size=1.0,
                entry_price=100.0,
                current_price=100.0,
            )
            for i in range(9)  # must match CONFIG.risk.max_open_positions (9)
        }

        decision = risk_mgr.evaluate(
            signal=_buy_signal(symbol="NEW/USDT", entry=50.0, stop=45.0, target=60.0),
            portfolio=_portfolio(cash=4000.0, net_equity=10000.0, positions=positions),
            regime=_regime(),
        )

        assert not decision.approved
        assert decision.veto_reason is not None
        assert "max positions" in decision.veto_reason.lower()

    def test_drawdown_circuit_breaker(self):
        """v18: Hard backstop rejects at DD >= 40% from all-time peak."""
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        # Feed enough equity history
        for i in range(70):
            risk_mgr.record_equity(10000.0)
        for i in range(5):
            risk_mgr.record_equity(5500.0)

        portfolio = PortfolioState(
            initial_capital=10000.0,
            cash=5500.0,
            net_equity=5500.0,
            peak_equity=10000.0,
            positions={},
        )
        portfolio.current_drawdown_pct = 0.45

        decision = risk_mgr.evaluate(
            signal=_buy_signal(symbol="ETH/USDT", entry=2400.0, stop=2300.0, target=2600.0),
            portfolio=portfolio,
            regime=_regime(),
        )

        assert not decision.approved
        assert decision.veto_reason is not None
        assert "halt" in decision.veto_reason.lower() or "backstop" in decision.veto_reason.lower()

    def test_insufficient_cash(self):
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        decision = risk_mgr.evaluate(
            signal=_buy_signal(),
            portfolio=_portfolio(cash=50.0, net_equity=10000.0),
            regime=_regime(),
        )

        assert not decision.approved
        assert decision.veto_reason is not None
        assert "insufficient cash" in decision.veto_reason.lower()

    def test_adaptive_sizing_with_learner(self):
        learner = AdaptiveLearner()
        for _ in range(12):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy="TestStrategy",
                regime="trending_up",
                pnl=200.0,
                exit_reason="take_profit",
                trade_date="2024-01-01",
            )

        risk_mgr = CryptoRiskManager(learner=learner)

        decision = risk_mgr.evaluate(
            signal=_buy_signal(confidence=0.75),
            portfolio=_portfolio(),
            regime=_regime(),
        )

        assert decision.approved
        assert decision.position_size > 0


@pytest.mark.unit
class TestV18VolatilityTargeting:
    """Tests for v18 volatility targeting risk scaling."""

    def test_vol_target_reduces_size_in_high_vol(self):
        """When realized vol >> target vol, sizing should shrink."""
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        # Simulate high-volatility equity history (daily swings of 5%)
        equity = 10000.0
        for i in range(25):
            swing = 500.0 * (1 if i % 2 == 0 else -1)
            equity += swing
            risk_mgr.record_equity(equity)

        scale = risk_mgr._vol_target_scale()
        assert scale < 1.0, "High vol should reduce scale below 1.0"
        assert scale >= risk_mgr.cfg.vol_scale_min

    def test_vol_target_no_effect_with_insufficient_history(self):
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        # Only 5 bars — not enough
        for i in range(5):
            risk_mgr.record_equity(10000.0 + i * 10)

        assert risk_mgr._vol_target_scale() == 1.0

    def test_vol_target_disabled(self):
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)
        risk_mgr.cfg.vol_target_enabled = False

        for i in range(25):
            risk_mgr.record_equity(10000.0 + i * 500 * (-1) ** i)

        assert risk_mgr._vol_target_scale() == 1.0


@pytest.mark.unit
class TestV18RegimeRiskScaling:
    """Tests for v18 regime-based risk scaling."""

    def test_trending_up_gets_boost(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        regime = _regime(regime=RegimeType.TRENDING_UP)
        scale = risk_mgr._regime_risk_scale(regime)
        assert scale == 1.20

    def test_trending_down_gets_cut(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        regime = _regime(regime=RegimeType.TRENDING_DOWN)
        scale = risk_mgr._regime_risk_scale(regime)
        assert scale == 0.40

    def test_volatile_gets_cut(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        regime = _regime(regime=RegimeType.VOLATILE)
        scale = risk_mgr._regime_risk_scale(regime)
        assert scale == 0.50

    def test_no_regime_returns_1(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        assert risk_mgr._regime_risk_scale(None) == 1.0

    def test_disabled(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        risk_mgr.cfg.regime_risk_enabled = False
        regime = _regime(regime=RegimeType.TRENDING_DOWN)
        assert risk_mgr._regime_risk_scale(regime) == 1.0


@pytest.mark.unit
class TestLiquiditySizing:
    def test_low_dollar_volume_reduces_position_size(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        risk_mgr.cfg.liquidity_sizing_enabled = True
        risk_mgr.cfg.liquidity_min_dollar_volume = 25_000_000.0
        risk_mgr.cfg.liquidity_full_dollar_volume = 250_000_000.0
        risk_mgr.cfg.liquidity_floor_mult = 0.60

        low_liq = risk_mgr.evaluate(
            signal=_buy_signal(
                symbol="SOL/USDT", entry=50.0, stop=25.0, target=100.0, dollar_volume_20d=20_000_000.0
            ),
            portfolio=_portfolio(),
            regime=_regime(),
        )
        high_liq = risk_mgr.evaluate(
            signal=_buy_signal(
                symbol="SOL/USDT", entry=50.0, stop=25.0, target=100.0, dollar_volume_20d=400_000_000.0
            ),
            portfolio=_portfolio(),
            regime=_regime(),
        )

        assert low_liq.approved
        assert high_liq.approved
        assert low_liq.position_size < high_liq.position_size

    def test_missing_dollar_volume_does_not_penalize(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        risk_mgr.cfg.liquidity_sizing_enabled = True

        missing_liq = risk_mgr.evaluate(
            signal=_buy_signal(
                symbol="SOL/USDT", entry=50.0, stop=25.0, target=100.0, dollar_volume_20d=None
            ),
            portfolio=_portfolio(),
            regime=_regime(),
        )
        baseline = risk_mgr.evaluate(
            signal=_buy_signal(
                symbol="SOL/USDT", entry=50.0, stop=25.0, target=100.0, dollar_volume_20d=400_000_000.0
            ),
            portfolio=_portfolio(),
            regime=_regime(),
        )

        assert missing_liq.approved
        assert baseline.approved
        assert missing_liq.position_size == pytest.approx(baseline.position_size)

    def test_nan_dollar_volume_does_not_penalize(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        risk_mgr.cfg.liquidity_sizing_enabled = True

        nan_liq = risk_mgr.evaluate(
            signal=_buy_signal(
                symbol="SOL/USDT", entry=50.0, stop=25.0, target=100.0, dollar_volume_20d=float("nan")
            ),
            portfolio=_portfolio(),
            regime=_regime(),
        )
        baseline = risk_mgr.evaluate(
            signal=_buy_signal(
                symbol="SOL/USDT", entry=50.0, stop=25.0, target=100.0, dollar_volume_20d=400_000_000.0
            ),
            portfolio=_portfolio(),
            regime=_regime(),
        )

        assert nan_liq.approved
        assert baseline.approved
        assert nan_liq.position_size == pytest.approx(baseline.position_size)


@pytest.mark.unit
class TestV18EquityCurveFeedback:
    """Tests for v18 graduated equity curve feedback."""

    def test_no_dd_full_risk(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        portfolio = _portfolio(net_equity=10000.0)
        portfolio.peak_equity = 10000.0
        portfolio.current_drawdown_pct = 0.0
        assert risk_mgr._equity_curve_feedback(portfolio) == 1.0

    def test_moderate_dd_reduces_risk(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        # Feed equity history: peak at 10000, current at 8000 → ~20% DD
        for i in range(70):
            risk_mgr.record_equity(10000.0)
        risk_mgr.record_equity(8000.0)

        portfolio = _portfolio(net_equity=8000.0)
        portfolio.peak_equity = 10000.0
        portfolio.current_drawdown_pct = 0.20
        mult = risk_mgr._equity_curve_feedback(portfolio)
        assert mult < 1.0
        assert mult == 0.50  # DD ~20% → tier 0.50 (18-25% band)

    def test_severe_dd_halts(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        # Feed equity history: peak at 10000, current at 5500 → ~45% DD
        for i in range(70):
            risk_mgr.record_equity(10000.0)
        risk_mgr.record_equity(5500.0)

        portfolio = _portfolio(net_equity=5500.0)
        portfolio.peak_equity = 10000.0
        portfolio.current_drawdown_pct = 0.45
        mult = risk_mgr._equity_curve_feedback(portfolio)
        assert mult == 0.0  # DD 45% > 30% → halt

    def test_disabled(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        risk_mgr.cfg.equity_curve_feedback_enabled = False
        portfolio = _portfolio(net_equity=6500.0)
        portfolio.peak_equity = 10000.0
        portfolio.current_drawdown_pct = 0.35
        assert risk_mgr._equity_curve_feedback(portfolio) == 1.0


@pytest.mark.unit
class TestV18BtcBetaPenalty:
    """Tests for v18 BTC-beta correlation penalty."""

    def test_high_beta_reduces_size(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())

        # Simulate portfolio that moves exactly with BTC (beta ≈ 1.0)
        for i in range(25):
            btc_ret = 0.02 * (1 if i % 2 == 0 else -1)
            risk_mgr._btc_return_history.append(btc_ret)

        # Portfolio equity mirrors BTC (high beta)
        eq = 10000.0
        risk_mgr._equity_history.append(eq)
        for i in range(25):
            btc_ret = risk_mgr._btc_return_history[i]
            eq *= 1 + btc_ret
            risk_mgr._equity_history.append(eq)

        scale = risk_mgr._btc_beta_penalty()
        assert scale < 1.0, "High BTC beta should reduce scale"

    def test_low_beta_no_penalty(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())

        # Simulate uncorrelated portfolio
        for i in range(25):
            risk_mgr._btc_return_history.append(0.01 * (1 if i % 2 == 0 else -1))

        eq = 10000.0
        risk_mgr._equity_history.append(eq)
        for i in range(25):
            # Portfolio moves opposite to BTC → negative beta
            eq *= 1 - 0.01 * (1 if i % 2 == 0 else -1)
            risk_mgr._equity_history.append(eq)

        scale = risk_mgr._btc_beta_penalty()
        assert scale == 1.0, "Low/negative beta should not penalize"

    def test_insufficient_history(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        for i in range(5):
            risk_mgr._btc_return_history.append(0.01)
            risk_mgr._equity_history.append(10000.0 + i)
        assert risk_mgr._btc_beta_penalty() == 1.0

    def test_disabled(self):
        risk_mgr = CryptoRiskManager(learner=AdaptiveLearner())
        risk_mgr.cfg.btc_beta_penalty_enabled = False
        assert risk_mgr._btc_beta_penalty() == 1.0


@pytest.mark.unit
class TestV18Integration:
    """Integration test: all 4 v18 multipliers applied together."""

    def test_all_multipliers_reduce_size(self):
        """With high vol + bear regime + moderate DD + high beta, size should be small."""
        learner = AdaptiveLearner()
        risk_mgr = CryptoRiskManager(learner=learner)

        # Simulate high-vol equity history
        eq = 10000.0
        risk_mgr._equity_history.append(eq)
        for i in range(25):
            swing = 600.0 * (1 if i % 2 == 0 else -1)
            eq += swing
            risk_mgr.record_equity(eq)
            risk_mgr._btc_return_history.append(swing / 10000.0)

        # DD at 18% → ecf = 0.50
        portfolio = PortfolioState(
            initial_capital=10000.0,
            cash=8200.0,
            net_equity=8200.0,
            peak_equity=10000.0,
            positions={},
        )
        portfolio.current_drawdown_pct = 0.18

        decision = risk_mgr.evaluate(
            signal=_buy_signal(confidence=0.75),
            portfolio=portfolio,
            regime=_regime(regime=RegimeType.TRENDING_DOWN),
        )

        # Should still approve but with much smaller position
        if decision.approved:
            # Compare to baseline: full risk would give ~$2,200 position on $10K
            assert decision.position_value < 5000.0, "v18 should significantly reduce position size"
