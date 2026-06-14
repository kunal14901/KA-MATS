"""
KA-MATS Crypto · Unified Strategy — "KA-MATS Alpha"
Iknir Capital

Merges TrendPullback + MomentumBreakout into a SINGLE strategy for paper trading.

CANONICAL BASELINE (v15, June 2026 — see VALIDATION_METHODOLOGY.md §0.11):
  Full-sample Sharpe 1.389, MaxDD 29.9%, OOS (2023+) Sharpe 0.961, with the
  partial-fill maker cost model and 45% vol targeting. Older docstring claims
  (+4296%) came from the pre-Phase-1 close-only engine and are SUPERSEDED.

Two entry modes (mutually exclusive per bar):
  Mode A · PULLBACK: RSI dip [38-57] in established uptrend (EMA20 > EMA50)
  Mode B · BREAKOUT: Near 20-bar high + volume surge in uptrend

Priority: Breakout wins when both fire (stronger momentum signal).

PARITY WARNING: this class is a live-side re-implementation of
backtest/run_crypto_backtest.py strategy logic. Any change to entry/exit
parameters in EITHER file must be mirrored in the other and re-validated by
re-running the backtest. The paper-phase shadow log (logs/shadow) is the
parity check: live decisions should match what the backtest engine would
have produced on the same daily bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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


@dataclass(frozen=True)
class _AlphaProfile:
    pullback_rsi_min: float
    pullback_rsi_max: float
    pullback_close_ema50_mult: float
    pullback_vol_max: float
    pullback_rank_min: float
    pullback_adx_min: float
    pullback_base_conf: float
    pullback_stop_atr: float
    pullback_target_atr: float
    breakout_trend_rsi_min: float
    breakout_trend_rsi_max: float
    breakout_range_rsi_min: float
    breakout_range_rsi_max: float
    breakout_trend_vol_min: float
    breakout_range_vol_min: float
    breakout_trend_rank_min: float
    breakout_range_rank_min: float
    breakout_adx_min: float
    breakout_high_mult: float
    breakout_base_conf: float
    breakout_stop_atr: float
    breakout_target_atr: float


_PROFILES = {
    "1d": _AlphaProfile(
        pullback_rsi_min=40.0,
        pullback_rsi_max=54.0,
        pullback_close_ema50_mult=0.96,
        pullback_vol_max=2.0,
        pullback_rank_min=0.55,
        pullback_adx_min=15.0,
        pullback_base_conf=0.57,
        pullback_stop_atr=2.5,
        pullback_target_atr=11.0,
        breakout_trend_rsi_min=55.0,
        breakout_trend_rsi_max=72.0,
        breakout_range_rsi_min=62.0,
        breakout_range_rsi_max=75.0,
        breakout_trend_vol_min=1.3,
        breakout_range_vol_min=2.0,
        breakout_trend_rank_min=0.40,
        breakout_range_rank_min=0.50,
        breakout_adx_min=18.0,
        breakout_high_mult=0.995,
        breakout_base_conf=0.58,
        breakout_stop_atr=2.0,
        breakout_target_atr=5.0,
    ),
    # More bars means more chances to over-trade. Start stricter than daily and
    # smaller in risk until the dedicated 4h validation clears the profile.
    "4h": _AlphaProfile(
        pullback_rsi_min=42.0,
        pullback_rsi_max=55.0,
        pullback_close_ema50_mult=0.98,
        pullback_vol_max=1.8,
        pullback_rank_min=0.60,
        pullback_adx_min=18.0,
        pullback_base_conf=0.60,
        pullback_stop_atr=3.0,
        pullback_target_atr=7.0,
        breakout_trend_rsi_min=56.0,
        breakout_trend_rsi_max=70.0,
        breakout_range_rsi_min=63.0,
        breakout_range_rsi_max=73.0,
        breakout_trend_vol_min=1.5,
        breakout_range_vol_min=2.2,
        breakout_trend_rank_min=0.50,
        breakout_range_rank_min=0.60,
        breakout_adx_min=20.0,
        breakout_high_mult=0.997,
        breakout_base_conf=0.61,
        breakout_stop_atr=2.5,
        breakout_target_atr=5.5,
    ),
}


class UnifiedAlphaStrategy:
    """
    Single strategy combining TrendPullback + MomentumBreakout.

    Backtest baseline:
      TrendPullback:    293 trades, 53.2% WR, PnL $237K
      MomentumBreakout:  84 trades, 56.0% WR, PnL $182K
      Combined:         377 trades, 53.8% WR, +4296% return, Sharpe 1.54

    Entry modes:
      A) PULLBACK (trending_up or ranging):
         EMA20 > EMA50 + RSI [40,54] + close > EMA50×0.96
         + volume_ratio < 2.0 + cross_rank ≥ 0.55 + ADX > 15
         → Stop 2.5×ATR, Target 11×ATR

      B) BREAKOUT (trending_up or ranging):
         EMA20 > EMA50 + near 20-bar high + volume surge
         + trending: RSI [55,72], vol > 1.3×, rank ≥ 0.40
         + ranging:  RSI [62,75], vol > 2.0×, rank ≥ 0.50
         + ADX > 18
         → Stop 2.0×ATR, Target 5.0×ATR
    """

    name = "KA_MATS_Alpha"
    VALID_REGIMES = {RegimeType.TRENDING_UP, RegimeType.RANGING}

    def __init__(
        self,
        cfg: StrategyConfig = None,
        learner: AdaptiveLearner = None,
        profile: str = "1d",
    ) -> None:
        self.cfg = cfg or CONFIG.strategy
        self._learner: AdaptiveLearner | None = learner
        self.profile_name = profile if profile in _PROFILES else "1d"
        self.profile = _PROFILES[self.profile_name]
        self.last_reject_reason: str = "not_evaluated"

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
        close = snapshot.price.close
        adx = getattr(ind, "adx_14", None) or getattr(ind, "adx", None)
        high_20 = getattr(ind, "high_20", None)
        cross_rank = feat.cross_rank if feat else None
        vol_ratio = feat.volume_ratio if feat else None

        if any(v is None for v in [ema20, ema50, rsi, atr]):
            return self._reject("missing_indicators")
        if atr <= 0 or close <= 0:
            return self._reject("invalid_price_or_atr")
        if not (ema20 > ema50):
            return self._reject("ema_filter")

        # ── Try BREAKOUT first (higher priority) ─────────────
        breakout_sig = self._eval_breakout(
            snapshot,
            regime,
            close,
            ema20,
            ema50,
            ema200,
            rsi,
            atr,
            adx,
            high_20,
            cross_rank,
            vol_ratio,
        )
        if breakout_sig is not None:
            self.last_reject_reason = ""
            return [breakout_sig]
        breakout_reason = self.last_reject_reason

        # ── Then try PULLBACK ─────────────────────────────────
        pullback_sig = self._eval_pullback(
            snapshot,
            regime,
            close,
            ema20,
            ema50,
            ema200,
            rsi,
            atr,
            adx,
            cross_rank,
            vol_ratio,
        )
        if pullback_sig is not None:
            self.last_reject_reason = ""
            return [pullback_sig]

        if self.last_reject_reason == "rsi_filter" and breakout_reason != "rsi_filter":
            self.last_reject_reason = breakout_reason
        return []

    # ── MODE A: PULLBACK ─────────────────────────────────────

    def _eval_pullback(
        self,
        snapshot,
        regime,
        close,
        ema20,
        ema50,
        ema200,
        rsi,
        atr,
        adx,
        cross_rank,
        vol_ratio,
    ) -> CandidateSignal | None:
        p = self.profile
        if not (p.pullback_rsi_min <= rsi <= p.pullback_rsi_max):
            self.last_reject_reason = "rsi_filter"
            return None
        if close < ema50 * p.pullback_close_ema50_mult:
            self.last_reject_reason = "price_filter"
            return None
        if vol_ratio is not None and vol_ratio >= p.pullback_vol_max:
            self.last_reject_reason = "volume_filter"
            return None
        if cross_rank is not None and cross_rank < p.pullback_rank_min:
            self.last_reject_reason = "rank_filter"
            return None
        if adx is not None and adx < p.pullback_adx_min:
            self.last_reject_reason = "adx_filter"
            return None

        rsi_sweet = 47.0
        rsi_bonus = min(0.08, max(0.0, (7.0 - abs(rsi - rsi_sweet)) / 100.0))
        rank_bonus = min(0.07, (cross_rank - 0.45) * 0.15) if cross_rank is not None else 0.0
        stack_bon = 0.06 if (ema200 is not None and ema50 > ema200) else 0.0
        confidence = p.pullback_base_conf + rsi_bonus + rank_bonus + stack_bon

        stop = close - p.pullback_stop_atr * atr
        target = close + p.pullback_target_atr * atr

        conds = [
            self._cond("mode", True, description="PULLBACK"),
            self._cond("ema_trend", True, description=f"EMA20={ema20:.2f}>EMA50={ema50:.2f}"),
            self._cond(
                "rsi_dip",
                True,
                rsi,
                p.pullback_rsi_max,
                f"RSI={rsi:.1f} in [{p.pullback_rsi_min:.0f},{p.pullback_rsi_max:.0f}]",
            ),
            self._cond(
                "rank",
                True,
                cross_rank,
                p.pullback_rank_min,
                f"rank={cross_rank:.2f}" if cross_rank else "n/a",
            ),
        ]
        return self._build_signal(
            snapshot,
            SignalDirection.BUY,
            conds,
            confidence,
            stop,
            target,
            regime,
            entry_mode="pullback",
        )

    # ── MODE B: BREAKOUT ─────────────────────────────────────

    def _eval_breakout(
        self,
        snapshot,
        regime,
        close,
        ema20,
        ema50,
        ema200,
        rsi,
        atr,
        adx,
        high_20,
        cross_rank,
        vol_ratio,
    ) -> CandidateSignal | None:
        p = self.profile
        if adx is not None and adx < p.breakout_adx_min:
            self.last_reject_reason = "adx_filter"
            return None
        if high_20 is not None and close < high_20 * p.breakout_high_mult:
            self.last_reject_reason = "high_filter"
            return None

        if regime.regime == RegimeType.RANGING:
            if not (p.breakout_range_rsi_min <= rsi <= p.breakout_range_rsi_max):
                self.last_reject_reason = "rsi_filter"
                return None
            if vol_ratio is not None and vol_ratio <= p.breakout_range_vol_min:
                self.last_reject_reason = "volume_filter"
                return None
            rank_min = p.breakout_range_rank_min
            range_pen = -0.02
        else:  # trending_up
            if not (p.breakout_trend_rsi_min <= rsi <= p.breakout_trend_rsi_max):
                self.last_reject_reason = "rsi_filter"
                return None
            if vol_ratio is not None and vol_ratio <= p.breakout_trend_vol_min:
                self.last_reject_reason = "volume_filter"
                return None
            rank_min = p.breakout_trend_rank_min
            range_pen = 0.0

        if cross_rank is not None and cross_rank < rank_min:
            self.last_reject_reason = "rank_filter"
            return None

        vol_bonus = min(0.06, ((vol_ratio or p.breakout_trend_vol_min) - p.breakout_trend_vol_min) * 0.06)
        rank_bonus = min(0.06, (cross_rank - 0.40) * 0.12) if cross_rank is not None else 0.0
        confidence = p.breakout_base_conf + vol_bonus + rank_bonus + range_pen

        stop = close - p.breakout_stop_atr * atr
        target = close + p.breakout_target_atr * atr

        conds = [
            self._cond("mode", True, description="BREAKOUT"),
            self._cond("ema_trend", True, description=f"EMA20={ema20:.2f}>EMA50={ema50:.2f}"),
            self._cond("rsi_range", True, rsi, None, f"RSI={rsi:.1f}"),
            self._cond(
                "vol_surge", True, vol_ratio, rank_min, f"vol_ratio={vol_ratio:.2f}" if vol_ratio else "n/a"
            ),
            self._cond("rank", True, cross_rank, rank_min, f"rank={cross_rank:.2f}" if cross_rank else "n/a"),
        ]
        return self._build_signal(
            snapshot,
            SignalDirection.BUY,
            conds,
            confidence,
            stop,
            target,
            regime,
            entry_mode="breakout",
        )

    # ── Helpers ───────────────────────────────────────────────

    def _cond(self, name, passed, value=None, threshold=None, description="") -> SignalCondition:
        return SignalCondition(
            name=name,
            passed=passed,
            value=value,
            threshold=threshold,
            description=description,
        )

    def _reject(self, reason: str) -> list[CandidateSignal]:
        self.last_reject_reason = reason
        return []

    def _build_signal(
        self,
        snapshot,
        direction,
        conditions,
        confidence,
        stop,
        target,
        regime: RegimeAnalysis = None,
        entry_mode: str = "",
    ):
        # Adaptive learner modifiers
        if self._learner and regime is not None:
            strat_mod = self._learner.strategy_modifier(self.name, regime.regime.value)
            sym_mod = self._learner.symbol_confidence(snapshot.symbol)
            confidence = confidence + strat_mod + sym_mod
            confidence = max(0.0, min(1.0, confidence))

        if confidence < self.cfg.min_signal_confidence:
            self.last_reject_reason = "confidence_filter"
            return None
        if stop <= 0 or target <= 0:
            self.last_reject_reason = "invalid_stop_target"
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
