"""
KA-MATS Cryptoz · Risk Manager
Iknir Capital

Crypto-tuned risk management using the KA-MATS RiskDecision model.
Same 5-tier confidence sizing as equity, with wider crypto-appropriate limits.

v18 additions (fund-grade DD reduction):
  1. Volatility targeting — scale exposure to 20% annualized portfolio vol
  2. Regime-based risk scaling — multiply risk by regime factor
  3. Graduated equity curve feedback — DD-depth-based risk reduction
  4. BTC-beta correlation penalty — reduce sizing during high BTC correlation

v19 additions:
  5. Macro regime filter — halve sizing when 20-bar BTC cumulative return < -15%
     Targets the 2023-H1 and 2025-H1 recovery-floor / macro-shock periods where
     price-based regime filters fired "trending_up" but real trend hadn't resumed.

v20 additions (Vertus-inspired):
  6. Bayesian EV filter — compute expected value per signal using Bayesian
     posterior P(win | regime, strategy, vol_state). Only approve trades
     where EV > configurable threshold (transaction cost + slippage).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import UTC
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import CONFIG, RiskConfig
from core.models import (
    CandidateSignal,
    PortfolioState,
    RegimeAnalysis,
    RiskDecision,
    SignalDirection,
)

if TYPE_CHECKING:
    from core.adaptive_learner import AdaptiveLearner


class StrategyKellyTracker:
    """
    Per-strategy Half-Kelly position sizing from rolling closed-trade outcomes.

    Ported from the validated backtest engine (backtest/run_crypto_backtest.py)
    so live and backtest share the same sizing behaviour.

    Sizing formula (simplified Kelly):
      kelly_f = (WR * (avg_win/avg_loss + 1) - 1) / (avg_win/avg_loss)
      half_k  = kelly_f / 2
      mult    = clamp(half_k / BASELINE_HK, MIN_MULT, MAX_MULT)

    Returns 1.0 (no adjustment) until MIN_TRADES closed trades per strategy.
    """

    MIN_TRADES = 10
    WINDOW = 30
    BASELINE_HK = 0.15  # half-Kelly at 50% WR / 2.5:1 R/R → multiplier 1.0
    MIN_MULT = 0.50
    MAX_MULT = 1.50

    def __init__(self) -> None:
        self._trades: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.WINDOW))

    def record(self, strategy: str, pnl: float) -> None:
        self._trades[strategy].append(pnl)

    def get_mult(self, strategy: str) -> float:
        trades = list(self._trades[strategy])
        if len(trades) < self.MIN_TRADES:
            return 1.0
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        if not wins or not losses:
            return 1.0
        wr = len(wins) / len(trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        if avg_loss < 1e-9:
            return self.MAX_MULT
        b = avg_win / avg_loss
        kelly_f = (wr * (b + 1) - 1) / b
        half_k = max(0.0, kelly_f / 2)
        mult = half_k / self.BASELINE_HK
        return float(max(self.MIN_MULT, min(self.MAX_MULT, mult)))


class CryptoRiskManager:
    """Absolute veto authority. No trade executes without passing all risk checks."""

    def __init__(self, cfg: RiskConfig = None, learner: AdaptiveLearner = None) -> None:
        self.cfg = cfg or CONFIG.risk
        self._learner = learner
        # Daily loss tracking — reset each calendar day
        self._today_date: str = ""
        self._today_start_equity: float = 0.0
        # Ramp-up: track whether we've already logged the scale-up event
        self._ramp_scaled_up: bool = False
        # v18: Portfolio return history for volatility targeting
        self._equity_history: list[float] = []
        # v18: BTC return history for beta calculation
        self._btc_return_history: list[float] = []
        self._last_btc_price: float | None = None
        # v19: Decaying ATH tracking
        self._ath: float = 0.0
        self._bars_since_ath: int = 0
        # v21: Half-Kelly per-strategy sizing (parity with validated backtest)
        self.kelly = StrategyKellyTracker()

    def record_trade_outcome(self, strategy: str, pnl: float) -> None:
        """Feed closed-trade PnL into the Kelly tracker (called by orchestrator)."""
        self.kelly.record(strategy, pnl)

    def evaluate(
        self,
        signal: CandidateSignal,
        portfolio: PortfolioState,
        regime: RegimeAnalysis | None = None,
    ) -> RiskDecision:
        symbol = signal.symbol
        close = signal.entry_price
        # Use signal.timestamp (bar time) so the daily loss counter resets correctly
        # during backtesting. datetime.now() would always be "today", freezing the
        # counter at day 1 and permanently blocking all trades after the first bad day.
        now = signal.timestamp
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        def reject(reason: str) -> RiskDecision:
            logger.debug(f"[RiskManager] REJECT {symbol} — {reason}")
            return RiskDecision(
                signal_id=signal.signal_id,
                symbol=symbol,
                timestamp=now,
                approved=False,
                position_size=0.0,
                position_value=0.0,
                entry_price=close,
                stop_loss=signal.stop_price,
                take_profit=signal.target_price,
                risk_amount=0.0,
                veto_reason=reason,
                open_positions_count=len(portfolio.positions),
                strategy_name=signal.strategy_name,
            )

        # ── 1. Max open positions ──────────────────────────────
        if len(portfolio.positions) >= self.cfg.max_open_positions:
            return reject(f"Max positions ({self.cfg.max_open_positions})")

        # ── 2. Already holding ─────────────────────────────────
        if symbol in portfolio.positions:
            return reject(f"Already holding {symbol}")

        # ── 3. Graduated equity curve feedback (v18) ──────────────────────
        # Replaces binary DD breaker with smooth degradation.
        # At DD >= 30%, equity curve feedback returns 0.0 → full halt.
        dd = (portfolio.peak_equity - portfolio.net_equity) / max(portfolio.peak_equity, 1)
        ecf_mult = self._equity_curve_feedback(portfolio)
        if ecf_mult <= 0.0:
            return reject("Equity curve halt: rolling DD >= max tier")

        # Legacy hard limit kept as absolute backstop
        if dd >= 0.40:
            return reject(f"Hard DD backstop {dd:.1%} >= 40%")

        # ── 3b. Daily loss limit ───────────────────────────────
        # Track start-of-day equity; reject all new trades once daily loss cap hit.
        today_str = now.strftime("%Y-%m-%d")
        if today_str != self._today_date:
            # New calendar day — reset the daily baseline
            self._today_date = today_str
            self._today_start_equity = portfolio.net_equity
        elif self._today_start_equity > 0:
            daily_loss = (self._today_start_equity - portfolio.net_equity) / self._today_start_equity
            if daily_loss >= self.cfg.max_daily_loss_pct:
                return reject(f"Daily loss limit hit: {daily_loss:.1%} >= {self.cfg.max_daily_loss_pct:.0%}")

        # ── 4. Validate stop/target logic ──────────────────────
        if signal.direction == SignalDirection.BUY:
            if signal.stop_price >= close:
                return reject("Stop >= entry")
            if signal.target_price <= close:
                return reject("Target <= entry")
        else:
            if signal.stop_price <= close:
                return reject("Stop <= entry (short)")
            if signal.target_price >= close:
                return reject("Target >= entry (short)")

        # ── 4b. Minimum stop distance (prevents stop-hunting losses) ──
        # BTC/ETH/BNB: 4h candle noise can be 2-3%; require ≥ 2.5% stop room
        # Alts: higher volatility requires ≥ 3.5% minimum stop distance
        _LARGE_CAP = {"BTC/USDT", "ETH/USDT", "BNB/USDT"}
        min_stop_pct = 0.025 if symbol in _LARGE_CAP else 0.035
        stop_dist_pct = abs(close - signal.stop_price) / close
        if stop_dist_pct < min_stop_pct:
            return reject(
                f"Stop too tight ({stop_dist_pct:.1%} < min {min_stop_pct:.1%}) — "
                f"widens to ATR-based value by strategy, check ATR multipliers"
            )

        # ── 5. Confidence-based 5-tier sizing ─────────────────
        conf = signal.confidence

        # Adaptive modifier
        learner_mod = 0.0
        participation_mult = 1.0
        if self._learner and regime:
            participation_mult = self._learner.strategy_participation_multiplier(
                signal.strategy_name,
                regime.regime.value,
                signal.direction.value,
            )
            learner_mod = self._learner.strategy_modifier(signal.strategy_name, regime.regime.value)
            learner_mod += self._learner.symbol_confidence(symbol)
            conf += learner_mod

        if conf >= 0.75:
            size_mult = 1.00
        elif conf >= 0.65:
            size_mult = 0.75
        elif conf >= 0.55:
            size_mult = 0.50
        elif conf >= 0.45:
            size_mult = 0.30
        else:
            size_mult = 0.15

        # Regime confidence penalty
        if regime is not None and regime.confidence < 0.50:
            rc_mult = max(0.65, regime.confidence / 0.50)
            size_mult *= rc_mult

        # Soft regime participation penalty: preserve trading in weak regimes at reduced size.
        size_mult *= participation_mult

        # Correlation concentration penalty: reduce sizing when open positions are highly correlated
        if self._learner and portfolio.positions:
            open_syms = list(portfolio.positions.keys())
            corr_mult = self._learner.correlation_tracker.concentration_penalty(open_syms)
            size_mult *= corr_mult
            if corr_mult < 0.95:
                logger.debug(
                    f"[RiskManager] Correlation penalty: {corr_mult:.2f}× ({len(open_syms)} open positions)"
                )

        # ── v18 PORTFOLIO-LEVEL RISK MULTIPLIERS ──────────────
        # Equity curve feedback (already computed above; multiplied into size_mult)
        size_mult *= ecf_mult

        # Volatility targeting
        vol_scale = self._vol_target_scale()
        size_mult *= vol_scale

        # Regime-based risk scaling
        regime_scale = self._regime_risk_scale(regime)
        size_mult *= regime_scale

        # BTC-beta correlation penalty
        beta_scale = self._btc_beta_penalty()
        size_mult *= beta_scale

        # Macro regime filter (v19): reduce sizing during BTC macro shock/bear transition
        macro_scale = self._macro_filter_scale()
        size_mult *= macro_scale

        # Liquidity / tradeability penalty
        liquidity_scale = self._liquidity_scale(signal)
        size_mult *= liquidity_scale

        # v21: Half-Kelly per-strategy multiplier (parity with backtest engine)
        kelly_mult = self.kelly.get_mult(signal.strategy_name)
        size_mult *= kelly_mult

        # BearShort v2 parity: SHORT positions run half size — bear-market
        # squeezes are violent and short losses empirically run ~2× long
        # losses per unit of risk (validated in the 2020-2026 backtest).
        short_mult = 1.0
        if signal.direction.value.upper() in ("SELL", "SHORT"):
            short_mult = 0.5
            size_mult *= short_mult

        if size_mult < 0.05:
            return reject(
                f"Combined risk multipliers too low ({size_mult:.3f}): "
                f"participation={participation_mult:.2f} ecf={ecf_mult:.2f} "
                f"vol={vol_scale:.2f} regime={regime_scale:.2f} beta={beta_scale:.2f} "
                f"macro={macro_scale:.2f} liq={liquidity_scale:.2f} kelly={kelly_mult:.2f}"
            )

        # Risk-based sizing — uses ramp-up rate until 50 live trades, then full rate
        risk_capital = portfolio.net_equity * self._effective_risk_pct(portfolio) * size_mult
        stop_dist = abs(close - signal.stop_price)
        if stop_dist <= 1e-12:
            return reject("Zero stop distance")

        shares = risk_capital / stop_dist

        # Cap by max position %
        max_by_pct = (portfolio.net_equity * self.cfg.max_position_pct) / close
        shares = min(shares, max_by_pct)

        # Cap by remaining portfolio exposure
        current_exp = sum(
            p.get("value", p.get("entry_price", 0) * p.get("shares", 0))
            for p in portfolio.positions.values()
            if isinstance(p, dict)
        )
        remaining_exp = portfolio.net_equity * self.cfg.max_portfolio_exposure_pct - current_exp
        if remaining_exp <= 0:
            return reject("Max portfolio exposure reached")
        shares = min(shares, remaining_exp / close)

        # Must have enough cash
        cost = shares * close
        if cost > portfolio.cash * 0.99 or shares < 1e-9:
            return reject(f"Insufficient cash (need ${cost:.2f}, have ${portfolio.cash:.2f})")

        effective_pct = self._effective_risk_pct(portfolio)
        logger.info(
            f"[RiskManager] APPROVE {symbol} {signal.direction.value} | "
            f"conf={conf:.3f} tier={size_mult:.2f}× | "
            f"risk_pct={effective_pct:.0%} | "
            f"part={participation_mult:.2f} vol={vol_scale:.2f} regime={regime_scale:.2f} "
            f"ecf={ecf_mult:.2f} beta={beta_scale:.2f} macro={macro_scale:.2f} "
            f"liq={liquidity_scale:.2f} kelly={kelly_mult:.2f} short={short_mult:.2f} | "
            f"shares={shares:.6f} | cost=${cost:.2f}"
        )

        return RiskDecision(
            signal_id=signal.signal_id,
            symbol=symbol,
            timestamp=now,
            approved=True,
            position_size=shares,
            position_value=cost,
            entry_price=close,
            stop_loss=signal.stop_price,
            take_profit=signal.target_price,
            risk_amount=risk_capital,
            veto_reason="",
            open_positions_count=len(portfolio.positions),
            strategy_name=signal.strategy_name,
        )

    def _effective_risk_pct(self, portfolio: PortfolioState) -> float:
        """
        Returns the risk-per-trade fraction to use for this trade.

        During the ramp-up period (first ramp_target_trades live trades):
          → ramp_initial_risk_pct (default 4%, see config.settings.RiskConfig)
        After ramp-up:
          → risk_per_trade_pct    (default 6%, Half-Kelly calibrated)

        Ramp-up is disabled if ramp_enabled=False (paper trading, backtesting).
        The scale-up event is logged exactly once so it's clearly visible in logs.
        """
        if not self.cfg.ramp_enabled:
            return self.cfg.risk_per_trade_pct

        n_closed = len(portfolio.closed_trades) if hasattr(portfolio, "closed_trades") else 0

        if n_closed >= self.cfg.ramp_target_trades:
            if not self._ramp_scaled_up:
                self._ramp_scaled_up = True
                logger.warning(
                    f"[RiskManager] RAMP-UP COMPLETE: {n_closed} live trades recorded. "
                    f"Scaling risk from {self.cfg.ramp_initial_risk_pct:.0%} → "
                    f"{self.cfg.risk_per_trade_pct:.0%} (full validated size). "
                    f"Monitor closely for the next 10 trades."
                )
            return self.cfg.risk_per_trade_pct

        remaining = self.cfg.ramp_target_trades - n_closed
        logger.debug(
            f"[RiskManager] Ramp-up active: {n_closed}/{self.cfg.ramp_target_trades} trades "
            f"({remaining} remaining). Using {self.cfg.ramp_initial_risk_pct:.0%} risk."
        )
        return self.cfg.ramp_initial_risk_pct

    # ── v18 FUND-GRADE PORTFOLIO-LEVEL RISK METHODS ───────────────────────

    def record_equity(self, equity: float) -> None:
        """Call once per bar with current portfolio equity for vol targeting."""
        self._equity_history.append(equity)
        # v19: Update ATH and bars since ATH for decaying peak
        if equity >= self._ath:
            self._ath = equity
            self._bars_since_ath = 0
        else:
            self._bars_since_ath += 1

    def record_btc_price(self, price: float) -> None:
        """Call once per bar with current BTC price for beta calculation."""
        if self._last_btc_price is not None and self._last_btc_price > 0:
            ret = (price - self._last_btc_price) / self._last_btc_price
            self._btc_return_history.append(ret)
        self._last_btc_price = price

    def _vol_target_scale(self) -> float:
        """
        Volatility targeting: scale position size so portfolio vol ≈ target.

        Computes realized portfolio vol from daily returns over a lookback window,
        annualizes it (×√365 for crypto), and returns target_vol / realized_vol
        clamped to [vol_scale_min, vol_scale_max].
        """
        if not self.cfg.vol_target_enabled:
            return 1.0

        n = self.cfg.vol_lookback_bars
        if len(self._equity_history) < n + 1:
            return 1.0  # not enough history yet

        # Daily portfolio returns over the lookback window
        recent = self._equity_history[-(n + 1) :]
        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                returns.append((recent[i] - recent[i - 1]) / recent[i - 1])

        if len(returns) < 5:
            return 1.0

        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        daily_vol = math.sqrt(var)
        annual_vol = daily_vol * math.sqrt(365)  # crypto = 365 trading days

        if annual_vol < 1e-6:
            return self.cfg.vol_scale_max  # near-zero vol → full scale

        scale = self.cfg.vol_target_annual_pct / annual_vol
        clamped = max(self.cfg.vol_scale_min, min(self.cfg.vol_scale_max, scale))
        logger.debug(
            f"[VolTarget] realized={annual_vol:.1%} target={self.cfg.vol_target_annual_pct:.1%} "
            f"raw_scale={scale:.2f} clamped={clamped:.2f}"
        )
        return clamped

    def _regime_risk_scale(self, regime: RegimeAnalysis | None) -> float:
        """
        Regime-based risk scaling: multiply risk by a pre-defined factor
        depending on the current market regime.
        """
        if not self.cfg.regime_risk_enabled:
            return 1.0
        if regime is None:
            return 1.0

        regime_name = regime.regime.value if hasattr(regime.regime, "value") else str(regime.regime)
        factor = self.cfg.regime_risk_factors.get(regime_name, 1.0)
        logger.debug(f"[RegimeRisk] regime={regime_name} factor={factor:.2f}")
        return factor

    def _equity_curve_feedback(self, portfolio: PortfolioState) -> float:
        """
        Graduated equity curve feedback: DD-depth-based risk multiplier.

        v19: Supports two modes:
        - Decaying ATH (ecf_use_decaying_ath=True): Reference peak decays slowly
          from all-time high, preventing permanent halt after bull mania peaks
          while still detecting slow multi-period drawdowns.
        - Rolling peak (legacy): Uses last N bars as lookback window.

        Returns 0.0 when DD exceeds the deepest tier (halts new entries completely).
        """
        if not self.cfg.equity_curve_feedback_enabled:
            return 1.0

        use_decay = getattr(self.cfg, "ecf_use_decaying_ath", False)

        if use_decay and self._ath > 0 and len(self._equity_history) >= 1:
            # Decaying ATH: reference peak decays toward current equity
            decay_rate = getattr(self.cfg, "ecf_decay_rate", 0.002)
            decay_floor = getattr(self.cfg, "ecf_decay_floor", 0.50)
            decay_factor = max(decay_floor, 1.0 - decay_rate * self._bars_since_ath)
            ref_peak = self._ath * decay_factor
            current_eq = self._equity_history[-1]
            # If equity exceeds decayed peak, no drawdown
            ref_peak = max(ref_peak, current_eq)
            dd = (ref_peak - current_eq) / ref_peak if ref_peak > 0 else 0.0
        elif len(self._equity_history) >= 2:
            # Legacy rolling peak mode
            peak_bars = getattr(self.cfg, "equity_curve_peak_bars", 120)
            lookback = (
                self._equity_history[-peak_bars:]
                if len(self._equity_history) > peak_bars
                else self._equity_history
            )
            rolling_peak = max(lookback)
            current_eq = self._equity_history[-1]
            dd = (rolling_peak - current_eq) / rolling_peak if rolling_peak > 0 else 0.0
        else:
            dd = getattr(portfolio, "current_drawdown_pct", 0.0) or 0.0
        dd = abs(dd)  # ensure positive

        mult = 1.0
        for threshold, factor in sorted(self.cfg.equity_curve_tiers, key=lambda t: t[0]):
            if dd < threshold:
                break
            mult = factor

        if mult < 1.0:
            logger.debug(f"[EquityCurve] DD={dd:.1%} → risk_mult={mult:.2f}")
        return mult

    def _btc_beta_penalty(self) -> float:
        """
        BTC-beta correlation penalty: when portfolio returns are highly
        correlated with BTC (beta > threshold), reduce sizing.

        Beta = cov(portfolio, BTC) / var(BTC) over a rolling window.
        """
        if not self.cfg.btc_beta_penalty_enabled:
            return 1.0

        n = self.cfg.btc_beta_lookback_bars
        if len(self._btc_return_history) < n or len(self._equity_history) < n + 1:
            return 1.0

        # Portfolio returns aligned with BTC returns
        btc_rets = self._btc_return_history[-n:]
        eq = self._equity_history[-(n + 1) :]
        port_rets = []
        for i in range(1, len(eq)):
            if eq[i - 1] > 0:
                port_rets.append((eq[i] - eq[i - 1]) / eq[i - 1])
            else:
                port_rets.append(0.0)
        port_rets = port_rets[-n:]

        if len(port_rets) != len(btc_rets):
            return 1.0

        mean_p = sum(port_rets) / len(port_rets)
        mean_b = sum(btc_rets) / len(btc_rets)
        var_b = sum((b - mean_b) ** 2 for b in btc_rets) / len(btc_rets)
        cov_pb = sum((p - mean_p) * (b - mean_b) for p, b in zip(port_rets, btc_rets, strict=False)) / len(
            btc_rets
        )

        if var_b < 1e-12:
            return 1.0

        beta = cov_pb / var_b
        if beta <= self.cfg.btc_beta_high_threshold:
            return 1.0

        # Linear penalty: beta 0.80 → 1.0× ... beta 1.0+ → (1 - max_penalty)
        excess = beta - self.cfg.btc_beta_high_threshold
        range_size = max(0.20, 1.0 - self.cfg.btc_beta_high_threshold)
        penalty_frac = min(1.0, excess / range_size)
        mult = 1.0 - penalty_frac * self.cfg.btc_beta_penalty_max
        logger.debug(f"[BtcBeta] beta={beta:.2f} penalty_mult={mult:.2f}")
        return max(1.0 - self.cfg.btc_beta_penalty_max, mult)

    def _macro_filter_scale(self) -> float:
        """
        Macro regime filter (v19): detect BTC macro shock / bear-transition periods.

        Computes the 20-bar cumulative BTC return from stored bar-by-bar returns.
        When this return falls below the configured threshold (default -15%), the market
        is likely in a macro-driven sell-off that price-based regime filters miss
        (they can show 'trending_up' off a bounce while the structural trend is still down).

        Returns macro_filter_size_mult (default 0.50) during stress, 1.0 otherwise.
        """
        if not getattr(self.cfg, "macro_filter_enabled", True):
            return 1.0

        n = 20
        if len(self._btc_return_history) < n:
            return 1.0

        # Compound the last 20 individual bar returns into a cumulative return
        recent = self._btc_return_history[-n:]
        cum_return = 1.0
        for r in recent:
            cum_return *= 1.0 + r
        cum_return -= 1.0

        threshold = getattr(self.cfg, "macro_filter_btc_return_threshold", -0.15)
        if cum_return < threshold:
            mult = getattr(self.cfg, "macro_filter_size_mult", 0.50)
            logger.debug(
                f"[MacroFilter] 20-bar BTC return={cum_return:.1%} < threshold={threshold:.0%} "
                f"→ applying {mult:.0%} sizing reduction"
            )
            return mult

        return 1.0

    def _liquidity_scale(self, signal: CandidateSignal) -> float:
        """Softly reduce size when rolling dollar volume sits near the tradeability floor."""
        if not self.cfg.liquidity_sizing_enabled:
            return 1.0

        dollar_volume = signal.dollar_volume_20d
        if dollar_volume is None or not math.isfinite(dollar_volume) or dollar_volume <= 0:
            return 1.0

        min_dv = self.cfg.liquidity_min_dollar_volume
        full_dv = self.cfg.liquidity_full_dollar_volume
        floor = self.cfg.liquidity_floor_mult

        if full_dv <= min_dv:
            return 1.0
        if dollar_volume <= min_dv:
            return floor
        if dollar_volume >= full_dv:
            return 1.0

        progress = (dollar_volume - min_dv) / (full_dv - min_dv)
        return floor + progress * (1.0 - floor)


# ─────────────────────────────────────────────────────────────
#  v20: BAYESIAN EXPECTED VALUE FILTER (Vertus-inspired)
# ─────────────────────────────────────────────────────────────


class BayesianEVFilter:
    """
    Computes expected value per trade signal using Bayesian posteriors.

    Tracks P(win | regime, strategy) and average win/loss magnitudes.
    Only approves trades where:
        EV = P(win) × avg_win - P(loss) × avg_loss > threshold

    The threshold should be set to cover transaction costs + slippage
    (typically 0.15-0.25% per round trip for crypto).

    Priors:
      P(win) starts at 0.50 (uninformative) and updates via EMA.
      avg_win/avg_loss start at equal values and update via EMA.

    State is keyed by (strategy, regime_family) for regime awareness.
    """

    _EMA_ALPHA = 0.10  # posterior update speed
    _MIN_OBSERVATIONS = 8  # need at least 8 trades before filtering
    _DEFAULT_EV_THRESHOLD = 0.002  # 0.2% — covers typical crypto round-trip costs

    def __init__(self, ev_threshold: float = None) -> None:
        self._ev_threshold = ev_threshold or self._DEFAULT_EV_THRESHOLD
        # {(strategy, regime_family): {p_win, avg_win, avg_loss, count}}
        self._posteriors: dict[str, dict[str, float]] = {}
        # Global fallback posterior
        self._global_p_win: float = 0.50
        self._global_avg_win: float = 0.02  # 2% avg win
        self._global_avg_loss: float = 0.015  # 1.5% avg loss
        self._global_count: int = 0

    def record_trade(
        self,
        strategy_name: str,
        regime: str,
        pnl_pct: float,
        won: bool,
    ) -> None:
        """Update Bayesian posteriors after a trade closes."""
        from core.adaptive_learner import REGIME_FAMILIES

        family = REGIME_FAMILIES.get(regime, "sideways")
        key = f"{strategy_name}::{family}"

        if key not in self._posteriors:
            self._posteriors[key] = {
                "p_win": 0.50,
                "avg_win": 0.02,
                "avg_loss": 0.015,
                "count": 0,
            }

        post = self._posteriors[key]
        post["count"] += 1

        win_val = 1.0 if won else 0.0
        post["p_win"] = self._EMA_ALPHA * win_val + (1.0 - self._EMA_ALPHA) * post["p_win"]

        if won and pnl_pct > 0:
            post["avg_win"] = self._EMA_ALPHA * abs(pnl_pct) + (1.0 - self._EMA_ALPHA) * post["avg_win"]
        elif not won and pnl_pct < 0:
            post["avg_loss"] = self._EMA_ALPHA * abs(pnl_pct) + (1.0 - self._EMA_ALPHA) * post["avg_loss"]

        # Update global too
        self._global_count += 1
        self._global_p_win = self._EMA_ALPHA * win_val + (1.0 - self._EMA_ALPHA) * self._global_p_win
        if won and pnl_pct > 0:
            self._global_avg_win = (
                self._EMA_ALPHA * abs(pnl_pct) + (1.0 - self._EMA_ALPHA) * self._global_avg_win
            )
        elif not won:
            self._global_avg_loss = (
                self._EMA_ALPHA * abs(pnl_pct) + (1.0 - self._EMA_ALPHA) * self._global_avg_loss
            )

    def compute_ev(
        self,
        strategy_name: str,
        regime: str,
    ) -> dict[str, float]:
        """
        Compute expected value for a (strategy, regime) combination.

        Returns dict with:
          - ev: expected value per trade (as fraction)
          - p_win: posterior win probability
          - avg_win: rolling average win magnitude
          - avg_loss: rolling average loss magnitude
          - count: observation count
          - sufficient_data: whether we have enough to filter
        """
        from core.adaptive_learner import REGIME_FAMILIES

        family = REGIME_FAMILIES.get(regime, "sideways")
        key = f"{strategy_name}::{family}"

        post = self._posteriors.get(key)

        if post and post["count"] >= self._MIN_OBSERVATIONS:
            p_win = post["p_win"]
            avg_win = post["avg_win"]
            avg_loss = post["avg_loss"]
            count = post["count"]
            sufficient = True
        elif self._global_count >= self._MIN_OBSERVATIONS:
            p_win = self._global_p_win
            avg_win = self._global_avg_win
            avg_loss = self._global_avg_loss
            count = self._global_count
            sufficient = True
        else:
            return {
                "ev": 0.01,  # optimistic default
                "p_win": 0.50,
                "avg_win": 0.02,
                "avg_loss": 0.015,
                "count": 0,
                "sufficient_data": False,
            }

        ev = p_win * avg_win - (1.0 - p_win) * avg_loss

        return {
            "ev": round(ev, 6),
            "p_win": round(p_win, 4),
            "avg_win": round(avg_win, 5),
            "avg_loss": round(avg_loss, 5),
            "count": count,
            "sufficient_data": sufficient,
        }

    def should_take_trade(
        self,
        strategy_name: str,
        regime: str,
    ) -> tuple:
        """
        Returns (approved: bool, ev_info: dict).

        Trade is approved if:
          - insufficient data (fail-open) OR
          - EV > threshold
        """
        ev_info = self.compute_ev(strategy_name, regime)

        if not ev_info["sufficient_data"]:
            return True, ev_info

        if ev_info["ev"] > self._ev_threshold:
            return True, ev_info

        logger.debug(
            f"[BayesianEV] FILTER {strategy_name}@{regime}: "
            f"EV={ev_info['ev']:.4f} < threshold={self._ev_threshold:.4f} | "
            f"P(win)={ev_info['p_win']:.2f} "
            f"avg_win={ev_info['avg_win']:.3f} avg_loss={ev_info['avg_loss']:.3f}"
        )
        return False, ev_info

    def get_state(self) -> dict:
        return {
            "posteriors": self._posteriors,
            "global_p_win": self._global_p_win,
            "global_avg_win": self._global_avg_win,
            "global_avg_loss": self._global_avg_loss,
            "global_count": self._global_count,
        }

    def load_state(self, state: dict) -> None:
        self._posteriors = state.get("posteriors", {})
        self._global_p_win = state.get("global_p_win", 0.50)
        self._global_avg_win = state.get("global_avg_win", 0.02)
        self._global_avg_loss = state.get("global_avg_loss", 0.015)
        self._global_count = state.get("global_count", 0)
