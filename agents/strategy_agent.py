"""
KA-MATS Crypto · Strategy Agent
Iknir Capital

Four deterministic strategies validated by the +1529% backtest (2022-2024, daily bars).
Parameters are identical to run_crypto_backtest.py — no divergence between backtest and live.

Strategy Map:
  trending_up    → CryptoTrendPullback + CryptoMomentumBreakout
  ranging        → CryptoTrendPullback + CryptoMomentumBreakout + CryptoRangeCapture
  trending_down  → CryptoBearShort (weak coins only, cross_rank ≤ 0.30)

Validated parameters (do NOT tune without re-running the backtest):
  TrendPullback:    stop=2.5×ATR, target=11×ATR, RSI [38,57], rank≥0.45
  MomentumBreakout: stop=2.0×ATR, target=5×ATR,  RSI [55,72] trending / [62,75] ranging
  BearShort v2:     ENABLED — bear mode only, triple BTC gate (bear + rollover +
                    active decline -5%..-25% / 20d), RSI [38,60], half-size.
                    v1 (BTC<EMA200 alone) was a 36% WR drag and stays retired.
  RangeCapture:     DISABLED (rare setup, net negative PnL in extended backtest)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import CONFIG, StrategyConfig
from core.models import (
    CandidateSignal,
    MarketSnapshot,
    RegimeAnalysis,
    RegimeType,
    SignalCondition,
    SignalDirection,
)

if TYPE_CHECKING:
    from core.adaptive_learner import AdaptiveLearner


# ─────────────────────────────────────────────────────────────
#  WR GATE CONSTANTS
# ─────────────────────────────────────────────────────────────

_WR_WINDOW = 20
_SUSPEND_GATE = 0.28
_RESTORE_GATE = 0.40
_MIN_TRADES_GATE = 15


# ─────────────────────────────────────────────────────────────
#  BASE STRATEGY
# ─────────────────────────────────────────────────────────────


class BaseStrategy(ABC):
    name: str = "BaseStrategy"

    def __init__(self, cfg: StrategyConfig = None, learner: AdaptiveLearner = None) -> None:
        self.cfg = cfg or CONFIG.strategy
        self._learner: AdaptiveLearner | None = learner

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot, regime: RegimeAnalysis) -> list[CandidateSignal]: ...

    def _cond(self, name, passed, value=None, threshold=None, description="") -> SignalCondition:
        return SignalCondition(
            name=name,
            passed=passed,
            value=value,
            threshold=threshold,
            description=description,
        )

    def _build_signal(
        self, snapshot, direction, conditions, confidence, stop, target, regime: RegimeAnalysis = None
    ):
        # ── Adaptive learner modifiers ────────────────────────
        if self._learner and regime is not None:
            strat_mod = self._learner.strategy_modifier(self.name, regime.regime.value)
            sym_mod = self._learner.symbol_confidence(snapshot.symbol)
            raw_conf = confidence
            confidence = confidence + strat_mod + sym_mod
            confidence = max(0.0, min(1.0, confidence))
            if abs(strat_mod) + abs(sym_mod) > 0.01:
                logger.debug(
                    f"[{self.name}] Learner adj: {raw_conf:.3f} → {confidence:.3f} "
                    f"(strat={strat_mod:+.3f} sym={sym_mod:+.3f})"
                )

        if confidence < self.cfg.min_signal_confidence:
            return None
        if stop <= 0 or target <= 0:
            return None
        return CandidateSignal(
            symbol=snapshot.symbol,
            timestamp=snapshot.price.timestamp,
            strategy_name=self.name,
            direction=direction,
            confidence=round(confidence, 4),
            entry_price=snapshot.price.close,
            stop_price=round(stop, 8),
            target_price=round(target, 8),
            dollar_volume_20d=(snapshot.features.dollar_volume_20d if snapshot.features else None),
            conditions=conditions,
        )


# ─────────────────────────────────────────────────────────────
#  STRATEGY 1: CRYPTO TREND PULLBACK
# ─────────────────────────────────────────────────────────────


class CryptoTrendPullbackStrategy(BaseStrategy):
    """
    Buy RSI dips in established uptrends. Core insight: crypto trends are persistent —
    pullbacks to RSI 40-54 while EMA20 > EMA50 are high R/R entries.

    Filters tightened from original [38,57] / rank≥0.45 to reduce low-quality setups
    that caused OOS degradation (78% of trades at below-average expectancy).

    Entry (LONG):
      trending_up or ranging + EMA20 > EMA50 + RSI [40,54]
      + close > EMA50×0.96 + volume_ratio < 2.0 + cross_rank ≥ 0.55
      + ADX > 15

    Stop:   2.5× ATR below entry
    Target: 11× ATR above entry  (trailing stop captures extended runs)
    """

    name = "CryptoTrendPullback"
    VALID_REGIMES = {RegimeType.TRENDING_UP, RegimeType.RANGING}

    def evaluate(self, snapshot: MarketSnapshot, regime: RegimeAnalysis) -> list[CandidateSignal]:
        if regime.regime not in self.VALID_REGIMES:
            return []

        ind = snapshot.indicators
        feat = snapshot.features
        ema20 = ind.ema_20
        ema50 = ind.ema_50
        rsi = ind.rsi_14
        atr = ind.atr_14
        close = snapshot.price.close
        adx = getattr(ind, "adx", None)
        cross_rank = feat.cross_rank if feat else None
        vol_ratio = feat.volume_ratio if feat else None

        if any(v is None for v in [ema20, ema50, rsi, atr]):
            return []
        if atr <= 0 or close <= 0:
            return []

        if not (ema20 > ema50):
            return []
        if not (40.0 <= rsi <= 54.0):
            return []
        if close < ema50 * 0.96:
            return []
        if vol_ratio is not None and vol_ratio >= 2.0:
            return []
        if cross_rank is not None and cross_rank < 0.55:
            return []
        if adx is not None and adx < 15.0:
            return []

        rsi_sweet = 47.0
        rsi_bonus = min(0.08, max(0.0, (7.0 - abs(rsi - rsi_sweet)) / 100.0))
        rank_bonus = min(0.07, (cross_rank - 0.45) * 0.15) if cross_rank is not None else 0.0
        ema200 = ind.ema_200
        stack_bon = 0.06 if (ema200 is not None and ema50 > ema200) else 0.0
        confidence = 0.57 + rsi_bonus + rank_bonus + stack_bon

        stop = close - 2.5 * atr
        target = close + 11.0 * atr

        conds = [
            self._cond("regime", True, description=f"regime={regime.regime.value}"),
            self._cond("ema_trend", ema20 > ema50, description=f"EMA20={ema20:.4f}>EMA50={ema50:.4f}"),
            self._cond("rsi_dip", 40.0 <= rsi <= 54.0, rsi, 54.0, f"RSI={rsi:.1f}"),
            self._cond(
                "rank",
                cross_rank is None or cross_rank >= 0.55,
                cross_rank,
                0.55,
                f"rank={cross_rank:.2f}" if cross_rank else "",
            ),
        ]
        sig = self._build_signal(snapshot, SignalDirection.BUY, conds, confidence, stop, target, regime)
        return [sig] if sig else []


# ─────────────────────────────────────────────────────────────
#  STRATEGY 2: CRYPTO MOMENTUM BREAKOUT
# ─────────────────────────────────────────────────────────────


class CryptoMomentumBreakoutStrategy(BaseStrategy):
    """
    Buy near 20-bar highs on volume surge in uptrends or consolidation breakouts.
    Core insight: crypto breakouts run far — top coins breaking highs on 1.3×+ volume
    are in the strongest phase of a trend move.

    Validated: 40-42 trades, 50-52% WR, avg_win $4,057-5,849, PnL $73K-96K (2022-2024).
    THE primary profit engine.

    Entry (LONG, trending_up):
      EMA20 > EMA50 + RSI [55,72] + volume_ratio > 1.3 + close near 20-bar high
      + cross_rank ≥ 0.40 + ADX > 18

    Entry (LONG, ranging — consolidation breakout):
      EMA20 > EMA50 + RSI [62,75] + volume_ratio > 2.0 + close near 20-bar high
      + cross_rank ≥ 0.50 + ADX > 18

    Stop:   2.0× ATR below entry
    Target: 5.0× ATR above entry  (trailing stop captures extended runs beyond TP)
    """

    name = "CryptoMomentumBreakout"
    VALID_REGIMES = {RegimeType.TRENDING_UP, RegimeType.RANGING}

    def evaluate(self, snapshot: MarketSnapshot, regime: RegimeAnalysis) -> list[CandidateSignal]:
        if regime.regime not in self.VALID_REGIMES:
            return []

        ind = snapshot.indicators
        feat = snapshot.features
        ema20 = ind.ema_20
        ema50 = ind.ema_50
        rsi = ind.rsi_14
        atr = ind.atr_14
        close = snapshot.price.close
        adx = getattr(ind, "adx", None)
        high_20 = getattr(ind, "high_20", None)
        cross_rank = feat.cross_rank if feat else None
        vol_ratio = feat.volume_ratio if feat else None

        if any(v is None for v in [ema20, ema50, rsi, atr]):
            return []
        if atr <= 0 or close <= 0:
            return []
        if not (ema20 > ema50):
            return []
        if adx is not None and adx < 18.0:
            return []

        # Near 20-bar high check (breakout proximity)
        if high_20 is not None and close < high_20 * 0.995:
            return []

        # Regime-specific thresholds
        if regime.regime == RegimeType.RANGING:
            if not (62.0 <= rsi <= 75.0):
                return []
            if vol_ratio is not None and vol_ratio <= 2.0:
                return []
            rank_min = 0.50
            range_pen = -0.02
        else:  # trending_up
            if not (55.0 <= rsi <= 72.0):
                return []
            if vol_ratio is not None and vol_ratio <= 1.3:
                return []
            rank_min = 0.40
            range_pen = 0.0

        if cross_rank is not None and cross_rank < rank_min:
            return []

        vol_bonus = min(0.06, ((vol_ratio or 1.3) - 1.3) * 0.06)
        rank_bonus = min(0.06, (cross_rank - 0.40) * 0.12) if cross_rank is not None else 0.0
        confidence = 0.58 + vol_bonus + rank_bonus + range_pen

        stop = close - 2.0 * atr
        target = close + 5.0 * atr

        conds = [
            self._cond("regime", True, description=f"regime={regime.regime.value}"),
            self._cond("ema_trend", ema20 > ema50, description=f"EMA20={ema20:.4f}>EMA50={ema50:.4f}"),
            self._cond("rsi_range", True, rsi, None, f"RSI={rsi:.1f}"),
            self._cond(
                "vol_surge", True, vol_ratio, rank_min, f"vol_ratio={vol_ratio:.2f}" if vol_ratio else ""
            ),
            self._cond(
                "rank",
                cross_rank is None or cross_rank >= rank_min,
                cross_rank,
                rank_min,
                f"rank={cross_rank:.2f}" if cross_rank else "",
            ),
        ]
        sig = self._build_signal(snapshot, SignalDirection.BUY, conds, confidence, stop, target, regime)
        return [sig] if sig else []


# ─────────────────────────────────────────────────────────────
#  STRATEGY 3: CRYPTO BEAR SHORT
# ─────────────────────────────────────────────────────────────


class CryptoBearShortStrategy(BaseStrategy):
    """
    BearShort v2 — short the weakest coins during ACTIVE bear-market declines.

    v1 post-mortem (2020-2026 backtest): firing on "BTC < EMA200" alone lost
    money in squeeze rallies (Jul 2022) and V-recoveries (2023-H1). v2 requires
    three independent BTC-level conditions before any coin-level check:
      1. Structural bear  — coin below EMA200 + macro bear regime
      2. BTC rollover     — BTC EMA20 < EMA50 (bear bounce is over)
      3. Active decline   — BTC 20-day return in (-25%, -5%]: falling, but NOT
                            post-capitulation (below -25% the squeeze is next)

    Backtest-validated (v2, half-size): 62 trades, 58% WR, +14.9% in 2022-H1
    while BTC fell 57%. BTC context is injected each bar by the orchestrator
    via set_macro_context(); without it the strategy FAILS CLOSED (no shorts).

    Entry (SELL):
      regime = trending_down + close < ema200
      + EMA20 < EMA50 (coin local downtrend) + cross_rank ≤ 0.30 (weakest coins)
      + RSI [38,60] (rolled over, NOT oversold — never short the bottom)
      + close < EMA20 (rollover confirmed, not knife-catching mid-bounce)
      + volume_ratio > 0.5

    Stop:   2.5× ATR above entry
    Target: 6.0× ATR below entry
    """

    name = "CryptoBearShort"
    VALID_REGIMES = {RegimeType.TRENDING_DOWN}

    # BTC active-decline band: 20-day return must be inside (LO, HI]
    BTC_ROC20_LO = -0.25
    BTC_ROC20_HI = -0.05

    def __init__(self, cfg=None, learner=None) -> None:
        super().__init__(cfg, learner)
        # Injected each bar by CryptoStrategyAgent.set_macro_context().
        # None = unknown → fail closed (no shorts without confirmed BTC context).
        self.btc_rollover: bool | None = None
        self.btc_roc20: float | None = None

    def evaluate(self, snapshot: MarketSnapshot, regime: RegimeAnalysis) -> list[CandidateSignal]:
        if regime.regime not in self.VALID_REGIMES:
            return []

        # ── v2 BTC-level gates (fail closed when context missing) ──
        if self.btc_rollover is not True:
            return []  # BTC bouncing or unknown — never short into a squeeze
        if self.btc_roc20 is None or not (self.BTC_ROC20_LO < self.btc_roc20 <= self.BTC_ROC20_HI):
            return []  # decline not active, or post-capitulation velocity

        ind = snapshot.indicators
        feat = snapshot.features
        ema20 = ind.ema_20
        ema50 = ind.ema_50
        ema200 = ind.ema_200
        rsi = ind.rsi_14
        atr = ind.atr_14
        close = snapshot.price.close
        cross_rank = feat.cross_rank if feat else None
        vol_ratio = feat.volume_ratio if feat else None

        if any(v is None for v in [ema20, ema50, rsi, atr]):
            return []
        if atr <= 0 or close <= 0:
            return []

        # Structural bear: price below EMA200
        if ema200 is not None and close >= ema200:
            return []

        if not (ema20 < ema50):
            return []
        if not (38.0 <= rsi <= 60.0):  # v2: never short oversold capitulation
            return []
        if close >= ema20:  # v2: rollover confirmed — under short-term mean
            return []
        if cross_rank is not None and cross_rank > 0.30:
            return []
        if vol_ratio is not None and vol_ratio <= 0.5:
            return []

        rank_bonus = min(0.08, (0.30 - (cross_rank or 0.10)) * 0.25)
        confidence = 0.56 + rank_bonus

        stop = close + 2.5 * atr
        target = close - 6.0 * atr

        conds = [
            self._cond("regime_bear", True, description="regime=trending_down"),
            self._cond("btc_rollover", True, description="BTC EMA20<EMA50"),
            self._cond("btc_active_fall", True, description=f"BTC roc20={self.btc_roc20:.1%}"),
            self._cond("ema_downtrend", ema20 < ema50, description=f"EMA20={ema20:.4f}<EMA50={ema50:.4f}"),
            self._cond("rsi_window", 38.0 <= rsi <= 60.0, rsi, 60.0, f"RSI={rsi:.1f}"),
            self._cond(
                "weak_rank",
                cross_rank is None or cross_rank <= 0.30,
                cross_rank,
                0.30,
                f"rank={cross_rank:.2f}" if cross_rank else "",
            ),
        ]
        sig = self._build_signal(snapshot, SignalDirection.SELL, conds, confidence, stop, target, regime)
        return [sig] if sig else []


# ─────────────────────────────────────────────────────────────
#  STRATEGY 4: CRYPTO RANGE CAPTURE
# ─────────────────────────────────────────────────────────────


class CryptoRangeCaptureStrategy(BaseStrategy):
    """
    Buy extreme oversold dips at Bollinger Band lower in ranging bull markets.
    Core insight: coins in ranging regimes above EMA200 bounce sharply from RSI extremes.

    Validated: low trade count (~4 trades/3yr) — rare setup, used as supplemental.

    Entry (LONG):
      ranging + close > EMA200 (macro bull) + RSI < 36
      + close ≤ BB_lower × 1.02 + cross_rank ≥ 0.30

    Stop:   2.0× ATR below entry
    Target: 4.0× ATR above entry
    """

    name = "CryptoRangeCapture"
    VALID_REGIMES = {RegimeType.RANGING}

    def evaluate(self, snapshot: MarketSnapshot, regime: RegimeAnalysis) -> list[CandidateSignal]:
        if regime.regime not in self.VALID_REGIMES:
            return []

        ind = snapshot.indicators
        feat = snapshot.features
        rsi = ind.rsi_14
        atr = ind.atr_14
        bb_lower = ind.bb_lower
        ema200 = ind.ema_200
        close = snapshot.price.close
        cross_rank = feat.cross_rank if feat else None

        if any(v is None for v in [rsi, atr, bb_lower]):
            return []
        if atr <= 0 or close <= 0 or bb_lower <= 0:
            return []

        # Macro bull filter: skip if below EMA200 (knife-catch in structural downtrend)
        if ema200 is not None and close < ema200:
            return []

        if rsi >= 36.0:
            return []
        if close >= bb_lower * 1.02:
            return []
        if cross_rank is not None and cross_rank < 0.30:
            return []

        rsi_bonus = min(0.08, max(0.0, (36.0 - rsi) / 100.0))
        rank_bonus = min(0.05, (cross_rank - 0.30) * 0.10) if cross_rank is not None else 0.0
        confidence = 0.56 + rsi_bonus + rank_bonus

        stop = close - 2.0 * atr
        target = close + 4.0 * atr

        conds = [
            self._cond("regime_ranging", True, description="regime=ranging"),
            self._cond("rsi_oversold", rsi < 36.0, rsi, 36.0, f"RSI={rsi:.1f}"),
            self._cond(
                "bb_lower_touch",
                close <= bb_lower * 1.02,
                close,
                bb_lower,
                f"close={close:.4f}≤BB_lower×1.02",
            ),
            self._cond("macro_bull", ema200 is None or close >= ema200, description="close≥EMA200"),
        ]
        sig = self._build_signal(snapshot, SignalDirection.BUY, conds, confidence, stop, target, regime)
        return [sig] if sig else []


# ─────────────────────────────────────────────────────────────
#  STRATEGY AGENT (DISPATCHER)
# ─────────────────────────────────────────────────────────────


class CryptoStrategyAgent:
    """
    Dispatches the 4 validated daily-bar strategies.

    Regime coverage:
      trending_up   → TrendPullback + MomentumBreakout
      ranging       → TrendPullback + MomentumBreakout + RangeCapture
      trending_down → BearShort

    WR Gate: suspend strategy when 20-trade WR < 28%, restore at 40%.
    """

    # v13 (June 2026): BearShort disabled — under honest intrabar fills it has
    # negative expectancy (-$7.2k over 2020-25; 56% WR but avg win $843 vs avg
    # loss $1,320). Cutting it: Sharpe 1.084 → 1.171, MaxDD 44% → 37%.
    # See VALIDATION_METHODOLOGY.md §0.8. Do not re-enable without the sleeve
    # passing the intrabar engine on its own.
    PERMANENT_DISABLED: set = {"CryptoBearShort"}

    def __init__(self, cfg: StrategyConfig = None, learner: AdaptiveLearner = None) -> None:
        self.cfg = cfg or CONFIG.strategy
        self._learner = learner
        self._strategies: list[BaseStrategy] = [
            CryptoTrendPullbackStrategy(cfg, learner),
            CryptoMomentumBreakoutStrategy(cfg, learner),
            CryptoBearShortStrategy(cfg, learner),
        ]
        self.suspended_strategies: set = set()
        self._wr_window: dict[str, deque[bool]] = {s.name: deque(maxlen=_WR_WINDOW) for s in self._strategies}

    def set_macro_context(
        self,
        btc_rollover: bool | None = None,
        btc_roc20: float | None = None,
    ) -> None:
        """
        Inject per-bar BTC macro context (computed by the orchestrator) into
        strategies that need it. BearShort v2 fails closed without it.

        btc_rollover : True when BTC EMA20 < EMA50 (bear bounce rolled over)
        btc_roc20    : BTC 20-bar return as a fraction (e.g. -0.12 = -12%)
        """
        for strat in self._strategies:
            if isinstance(strat, CryptoBearShortStrategy):
                strat.btc_rollover = btc_rollover
                strat.btc_roc20 = btc_roc20

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        regime: RegimeAnalysis,
        cross_rank: float = None,
    ) -> list[CandidateSignal]:
        if not snapshot.data_quality_ok:
            return []
        if regime.regime == RegimeType.UNKNOWN:
            return []

        if cross_rank is not None and snapshot.features:
            snapshot.features.cross_rank = cross_rank

        all_signals: list[CandidateSignal] = []
        for strat in self._strategies:
            if strat.name in self.PERMANENT_DISABLED:
                continue
            if strat.name in self.suspended_strategies:
                continue
            try:
                sigs = strat.evaluate(snapshot, regime)
                all_signals.extend(sigs)
            except Exception as e:
                logger.warning(f"[StrategyAgent] {strat.name} error on {snapshot.symbol}: {e}")

        if all_signals:
            logger.debug(
                f"[StrategyAgent] {snapshot.symbol}: "
                f"{len(all_signals)} signal(s) → "
                f"{[s.strategy_name for s in all_signals]}"
            )
        return all_signals

    def record_trade_outcome(self, strategy_name: str, won: bool) -> None:
        if strategy_name not in self._wr_window:
            self._wr_window[strategy_name] = deque(maxlen=_WR_WINDOW)
        self._wr_window[strategy_name].append(won)
        self._check_gate(strategy_name)

    def _check_gate(self, strategy_name: str) -> None:
        window = self._wr_window.get(strategy_name)
        if window is None or len(window) < _MIN_TRADES_GATE:
            return

        wr = sum(window) / len(window)

        if strategy_name in self.suspended_strategies:
            if wr >= _RESTORE_GATE:
                self.suspended_strategies.discard(strategy_name)
                logger.warning(
                    f"[StrategyAgent] {strategy_name} RESTORED (WR={wr:.1%} over {len(window)} trades)"
                )
        else:
            if wr < _SUSPEND_GATE and strategy_name not in self.PERMANENT_DISABLED:
                self.suspended_strategies.add(strategy_name)
                logger.warning(
                    f"[StrategyAgent] {strategy_name} SUSPENDED (WR={wr:.1%} over {len(window)} trades)"
                )

    def get_strategy_wr(self, strategy_name: str) -> float | None:
        window = self._wr_window.get(strategy_name)
        if window is None or len(window) < _MIN_TRADES_GATE:
            return None
        return sum(window) / len(window)
