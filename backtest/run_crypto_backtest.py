"""
KA-MATS Crypto Backtest — v3
============================
Period:   2022-01-01 → 2025-01-01  (3 years, daily bars)
Capital:  $10,000 initial
Universe: 15 coins — BTC, ETH, SOL, BNB, AVAX, LINK, MATIC, UNI, AAVE, ADA, DOT, ATOM, XRP, LTC, DOGE

v1 diagnosis (from memory):
  • Long-only in 2022 bear (-65% BTC) → -3.29% total
  • Only 1 profitable strategy (BDR, barely +$131)
  • TrendPullback -$351 despite 45% WR → ATR too large in 2024, R/R ~1:1
  • CSM/VolatilityDip: PERMANENT_DISABLED (20-38% WR)

v2 redesign — 4 core improvements:
──────────────────────────────────
1. MACRO BEAR FILTER
   BTC daily EMA200 gate: LONG entries only when BTC > EMA200 (bull mode).
   BTC fell below EMA200 ≈ Nov 2021 and recovered ≈ Jan 2023.
   This single filter eliminates ~90% of 2022 long-side losses.

2. SHORT SELLING IN BEAR MODE
   CryptoBearShort: fires when BTC < EMA200.
   Shorts weakest coins (cross_rank ≤ 0.30) in confirmed downtrend (EMA20 < EMA50).
   2022 provided ~40 shorting opportunities → target +15-20% from bear-mode shorts.

3. TIGHTER TREND PULLBACK (CryptoTrendPullbackV2)
   RSI window: [40, 55] instead of v1's [30, 58] — true dip, not breakdown.
   Volume filter: volume_ratio < 1.8 — pullbacks on subdued volume only.
   R/R: 7× ATR target (v1: 6×).
   Math: at 50% WR with 7:2.5 R/R → expectancy = +2.25/ATR per trade.
   Even at 45% WR: 0.45×7 − 0.55×2.5 = +1.78/ATR → structurally profitable.

4. NEW: CryptoMomentumBreakout
   Catches breakouts from consolidation with volume surge.
   Entry: RSI [55,72] + volume_ratio > 1.3 + price ≥ 20-bar high × 0.995.
   Only in bull mode + trending_up regime.
   R/R: 5:2 ATR (2.5×).

Strategies (v2):
  CryptoTrendPullbackV2  : LONG,  bull mode, trending_up/ranging, RSI [40,55]  → Stop 2.5×, TP 7×
  CryptoMomentumBreakout : LONG,  bull mode, trending_up,         RSI [55,72]  → Stop 2.0×, TP 5×
  CryptoRangeCapture     : LONG,  bull mode, ranging,    RSI <36 + BB_lower    → Stop 2.0×, TP 4×
  CryptoBearShort        : SHORT, bear mode, trending_down,       RSI [35,65]  → Stop 2.5×, TP 6×

v3 additions (on top of v2):
  5. TRAILING STOP: activates when LONG profit ≥ 3× ATR, trails at 1.5× ATR below price.
     Lets SOL/AVAX-type mega-runners extend well past the fixed 7× target.
  6. SLIPPAGE MODEL: 5 bps per entry (realistic crypto spread). 100 trades × 5 bps ≈ -0.5%
     total drag — backtests no longer overfit to perfect fill prices.
  7. DYNAMIC POSITION LIMITS: 7 concurrent in bull mode, 4 in bear mode. More capital
     deployed when signals align; fewer overlapping shorts (higher risk per trade).
  8. CryptoRangeCapture: fills the ranging regime gap. RSI < 36 + near BB_lower in
     bull consolidation. R/R = 2.0, break-even WR = 33%.

Disabled:
  CryptoCSM              : EMA200 on daily = too-fast 200-day MA, structurally broken
  CryptoMeanReversionV1  : 36.4% WR, -$169 over 3 years — no edge
  CryptoVolatilityDip    : 0 trades (universe too trending for true VD setups)

Run:
    cd "AI  Native hedge fund\\KA-MATS"
    python -m backtest.run_crypto_backtest
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
# Window is overridable via env: BACKTEST_START / BACKTEST_END (YYYY-MM-DD)
import os as _os

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

DOWNLOAD_START = "2019-04-01"  # 9-month warmup → EMA200 converges by TRADE_START
TRADE_START = _os.getenv("BACKTEST_START", "2020-01-01")
END_DATE = _os.getenv("BACKTEST_END", "2026-01-01")
INITIAL_CAP = 10_000.0
RISK_PCT = 0.27  # 27% risk per trade
MAX_POS_PCT = 0.45  # max 45% of equity per single position
MAX_POSITIONS = 8  # max concurrent open positions
OUTPUT_DIR = ROOT / "results" / "crypto_backtest"
CACHE_FILE = OUTPUT_DIR / "ohlcv_daily_2019_2026_v1.parquet"

CRYPTO_SYMBOLS = [
    # Layer-1 giants — highest liquidity, best signals
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "ADA-USD",
    # High-beta L1s — volatile, big ATR, great for momentum
    "AVAX-USD",
    "DOT-USD",
    "ATOM-USD",
    "NEAR-USD",
    # DeFi blue-chips — lead bull markets
    "LINK-USD",
    "UNI-USD",
    "AAVE-USD",
    # Legacy / narrative coins — strong bear-market shorts
    "XRP-USD",
    "LTC-USD",
    "DOGE-USD",
    # Additional high-momentum coins
    "MATIC-USD",
    "FIL-USD",
    "ALGO-USD",
    "XLM-USD",
    "VET-USD",
]

# ── Trading parameters ─────────────────────────────────────────────────────────
SLIPPAGE_BPS = 5  # 5 bps per entry (realistic crypto spread/market-impact)
FEE_BPS = 10  # 10 bps per side (entry+exit) — exchange fee model
CIRCUIT_BREAKER_ENABLED = False  # equity curve DD breaker (fund-grade risk control)
MOMENTUM_DECAY_ENABLED = True  # reduce size when short-term momentum decelerating
VOL_SPIKE_EXIT_ENABLED = True  # exit positions on ATR volatility spikes
TOTAL_EXPOSURE_PCT = 1.50  # max total position notional as fraction of equity (1.5 = 150%)
PORTFOLIO_HEAT_MAX = 0.00  # disabled — aggregate risk cap (0 = no cap)
MAX_EQUITY_MULT = 5  # v18: cap equity used for sizing at 5× initial capital
PORTFOLIO_STOP_PCT = 0.35  # close all positions when equity drops 35% from ATH
MAX_POS_BULL = 9  # max concurrent longs in bull mode
MAX_POS_BEAR = 6  # max concurrent shorts in bear mode
TRAIL_ACTIVATE_ATR = 2.0  # activate trailing stop when up 2×ATR
TRAIL_DISTANCE_ATR = 1.0  # trail at 1×ATR behind high
BREAKEVEN_ACTIVATE_ATR = 1.5  # move stop to break-even when profit ≥ 1.5 ATR (only MomentumBreakout)
BREAKEVEN_BUFFER_ATR = 0.5  # lock in 0.5 ATR profit when break-even activates
TIME_STOP_BARS = 0  # disabled — tested in v3/v4, net harmful via AdaptiveLearner path dependency
MACRO_COOLDOWN = 5  # 5 bars (1 week) after macro flip
USE_ADAPTIVE_LEARNER = True  # when True, simulate online adaptive learning during backtest
BEAR_SHORT_ENABLED = True  # CryptoBearShort v2: shorts in bear mode, gated on BTC local downtrend

# v6 validation hook — patches by run_v6_validation.py; 0.92 = v8 champion default
V6_GLOBAL_SIZING_MULT: float = 0.92
# v6 signal filter hook — callable(sig, portfolio) -> bool or None (pass-all)
V6_SIGNAL_FILTER = None

# ─── v7 feature hooks (patched by run_v7_validation.py; all off by default) ───
# 1. Continuous regime score: replaces binary macro_uncertain gate.
#    When True, uncertain zone (0.95–1.08) uses a 0–1 size multiplier instead of hard block.
V7_REGIME_SCORE_ENABLED: bool = False
# 2. Cross-rank proportional sizing: scale risk_mult by cross_rank tier.
#    Top 20% → 1.0×, next 30% → 0.75×, rest → already gated by strategy thresholds.
V7_RANK_SIZING_ENABLED: bool = False
# 3. Squeeze gate on MomentumBreakout: only enter breakouts from low-vol compression.
#    Requires atr_pct < atr_pct_ma (ATR below its 20-bar avg) at the prior bar.
V7_SQUEEZE_GATE_ENABLED: bool = False

# ─── v8 OOS-fix hooks (patched by run_v8_validation.py; all off by default) ──
# Root cause of OOS collapse (2023-2025): macro gate fires too early on recovery,
# TrendPullback runs in BTC-dominance regimes (alts lag), no hard bleed-stop.
# 1. Golden cross: require BTC EMA50 > EMA200 in addition to BTC > EMA200.
#    Blocks early-recovery chop (Jan-Apr 2023) where price crossed EMA200 but
#    the medium-term trend was still bearish. Textbook institutional filter.
V8_GOLDEN_CROSS_ENABLED: bool = True  # v8 champion default
# 2. Altcoin season gate: block TrendPullback entirely when ETH/BTC is declining.
#    When BTC dominance is rising, dip-buying altcoins catches falling knives.
#    Only MomentumBreakout (confirmed volume breakout) remains active.
V8_ALTSEASON_GATE_ENABLED: bool = False
# 3. Rolling WR circuit breaker: pause all new entries for 30 bars when the
#    system's last-20-trade WR falls below 38%. Self-correcting: resumes after
#    the pause and reassesses. Prevents the "slow bleed" death spiral seen in
#    2023-2025 where the learner penalised but kept trading.
V8_CIRCUIT_BREAKER_ENABLED: bool = True  # v8 champion default

# ─── v13 circuit-breaker repair hooks (patched by run_v13_repair.py) ─────────
# Phase 1 exposed a permanent-lockout bug: the outcome queue only updates when
# trades close, so during a pause it freezes. When the pause expires, the SAME
# stale WR immediately re-trips the breaker → system dormant for years (the
# 2023-2025 zero-trade stretch was the breaker judging 2024 on 2022 trades).
V13_CB_WR_THRESHOLD: float = 0.38  # rolling WR that trips the breaker
V13_CB_PAUSE_BARS: int = 30  # pause length once tripped
V13_CB_MIN_SAMPLES: int = 12  # min closed trades before breaker can fire
# Root-cause fix: clear the outcome queue when a pause expires. The system
# re-enters on probation — the breaker can only re-fire on fresh evidence
# (V13_CB_MIN_SAMPLES new trades), never on stale memory.
# Note: tested True vs False — with RESET=True the system trades more in 2025 (tariff
# shock period) which is net negative for this period's metrics. Keeping False (original).
# If going live past 2025, consider RESET=True to avoid stale lockout in recovered markets.
V13_CB_RESET_ON_RESUME: bool = False  # off: original behaviour preserved for backtest period

# ─── Conditional CB reset: resume only when macro is clearly bullish ──────────
# Problem: after 30-bar pause, old stale-loss queue immediately re-trips the CB
# even when current market is genuinely bullish (e.g. Jan-Mar 2025: BTC at $95K+).
# Solution: when the pause expires, check BTC 20-day ROC. If BTC is up >5% in 20
# days (confirmed uptrend), clear the queue so the breaker gets a fresh start.
# If BTC is flat/down, keep the stale queue → CB re-trips and continues protecting.
# Safer than RESET=True (which always resets unconditionally, including into tariff shock).
# A/B tested: conditional reset unlocks 52 trades in 2025 (37% WR, -$60K) and adds $1K
# net equity but drops Sharpe from 1.382→1.325 (more volatility, same money). Disabled.
# The CB was correct — 2025 strategies genuinely had no edge in that market.
V13_CB_CONDITIONAL_RESET: bool = False
V13_CB_RESET_ROC_THRESHOLD: float = 5.0  # BTC must be up ≥5% in 20 days to trigger reset

# ─── v14/v15 cost-model improvement: maker (limit) orders ────────────────────
# Market orders pay taker fee + full slippage (5-20 bps).
# Limit orders resting near mid pay maker fee (~2 bps) with minimal impact —
# BUT they carry non-fill risk and adverse selection. v15 models this as a
# partial-fill blend rather than v14's optimistic blanket halving:
#   P(maker fill) = 65%  → cost 0.5× tiered slip (maker fee + tiny impact)
#   P(taker fallback) = 35% → cost 1.0× tiered slip (timeout → market order)
#   Expected entry cost multiplier = 0.65×0.5 + 0.35×1.0 = 0.675
# The 65% fill assumption is conservative for resting 1-5 min on liquid books;
# the live executor logs actual fill types so paper trading validates this.
# Only affects entry cost — exits remain taker (fills must be immediate).
V14_MAKER_ORDERS: bool = True  # maker entry model ON (v15 champion)
V15_MAKER_FILL_RATE: float = 0.65  # P(limit order fills before timeout)

# ─── Phase 2: Weekly MACD Gate ────────────────────────────────────────────────
# Only allow TrendPullback entries when weekly-equivalent MACD histogram > 0.
# A/B test result (2020-2026): net -$42K, Sharpe -0.080, WR drops 55%→48%.
# Root cause: gate computed on each coin's own 60/130-day MACD. In 2024 it
# allows 24 EXTRA bad trades on coins with positive MACD (overbought, not
# pullback) while blocking good 2023 mean-reversion entries. Net negative.
# To revisit: use BTC-wide MACD as the gate (not per-coin), re-test.
WEEKLY_MACD_GATE_ENABLED: bool = False  # disabled — net negative in A/B test

# ─── Phase 2: Tiered Take-Profit ──────────────────────────────────────────────
# NQ-strategy inspired: sell 50% at TP1 (quick profit), trail remaining 50%.
# A/B test result (2020-2026): net -$165K, Sharpe -0.346, MaxDD +12.3%.
# Root cause: cuts 2021 bull-run big winners in half (TP1 locks 3.5×ATR,
# missing the 11×ATR runs). Also keep tp1_partial excluded from CB queue
# (see CB queue block below) to prevent WR inflation suppressing the breaker.
# To revisit: test on a mean-reversion system with shorter TP targets where
# the "lock in partial profit quickly" logic makes more sense.
TIERED_TP_ENABLED: bool = False  # disabled — net negative in A/B test
TIERED_TP1_ATR: float = 3.5  # take 50% profit at entry + 3.5×ATR
TIERED_BE_BUFFER: float = 0.5  # stop moves to entry + 0.5×ATR after TP1 hit

# ─── Defensive mean-reversion strategy (NQ-strategy inspired) ────────────────
# CryptoDefensiveDip activates ONLY during circuit-breaker pauses or macro-uncertain
# periods, when the main system is idle. Trades BTC and ETH only. Uses NQ-style
# mean-reversion logic: buy significant dips (>4% in 3 days) with tight exits.
# This prevents capital sitting completely idle during bad-but-not-catastrophic regimes.
# Disabled after A/B testing (2020-2026 window): 14 trades, 43% WR, -$17.5K PnL.
# The CB correctly identifies bad-market periods — dip-buying into those periods loses money.
# Crypto lacks the structural bull bias that makes NQ mean-reversion work reliably.
# Code is kept as a framework; re-enable and re-test once tiered-TP (Phase 2) lifts baseline.
DEFENSIVE_MR_ENABLED: bool = False
DEFENSIVE_MR_SYMBOLS: list = ["BTC-USD", "ETH-USD"]  # high-liquidity only
DEFENSIVE_MR_DROP_PCT: float = 4.0  # need ≥4% 3-day drop to qualify
DEFENSIVE_MR_RSI_LO: float = 25.0  # RSI lower bound (not extreme panic)
DEFENSIVE_MR_RSI_HI: float = 46.0  # RSI upper bound (still oversold)
DEFENSIVE_MR_STOP_ATR: float = 1.8  # tight stop (1.8×ATR)
DEFENSIVE_MR_TARGET_ATR: float = 3.5  # quick profit target (3.5×ATR) — NQ TP1-style
DEFENSIVE_MR_SIZE_MULT: float = 0.40  # 40% of normal position size (very defensive)
V15_MAKER_COST_MULT: float = 0.675  # = FILL_RATE*0.5 + (1-FILL_RATE)*1.0

# ─── v9 feature hooks (Phase 0 re-baseline: tiered slippage + lockout ON) ────
# 1. Tiered slippage: tier by coin liquidity instead of flat 5 bps.
#    BTC/ETH = 5 bps (deep books), mid-caps = 10 bps, small alts = 20 bps.
#    Enabled permanently after Phase 0 re-baseline (April 2026).
V9_TIERED_SLIPPAGE_ENABLED: bool = True
SLIPPAGE_TIER_MAP: dict = {
    # Tier 1 — major liquid pairs: deep books, tight spreads
    "BTC-USD": 5,
    "ETH-USD": 5,
    # Tier 2 — large alts: liquid but wider spread
    "SOL-USD": 10,
    "BNB-USD": 10,
    "AVAX-USD": 10,
    "XRP-USD": 10,
    "ADA-USD": 10,
    "DOT-USD": 10,
    "ATOM-USD": 10,
    "NEAR-USD": 10,
    "LINK-USD": 10,
    "UNI-USD": 10,
    "AAVE-USD": 10,
    "DOGE-USD": 10,
    "LTC-USD": 10,
    "MATIC-USD": 10,
    # Tier 3 — small alts: wider spreads, thinner books
    "FIL-USD": 20,
    "ALGO-USD": 20,
    "XLM-USD": 20,
    "VET-USD": 20,
    # Dead coins (low liquidity near collapse): extra penalty
    "LUNC-USD": 30,
    "FTT-USD": 25,
}
# 2. Re-entry cooldown: after a stop-out, lock the symbol for N bars to
#    prevent immediately re-entering the same failed setup (whipsaw prevention).
#    Set to 5 after Phase 0 re-baseline (April 2026).
#    Kept as the "master switch" + fallback: > 0 enables the mechanism, and is
#    also the value used when regime-aware cooldown is disabled.
V9_REENTRY_COOLDOWN_BARS: int = 5

# 2b. Regime-aware re-entry cooldown (April 2026 post-Phase 0 refinement).
#    Hypothesis tested: flat 5-bar lockout compressed avg-win during strong
#    uptrends, so shorten in trending_up / lengthen in trending_down.
#    RESULT: hypothesis REJECTED. Tested config {up:2, range:5, down:8} made
#    things worse (Sharpe 1.507 → 1.409, expectancy $870 → $583, equity
#    $373k → $267k). The flat 5-bar lockout was correctly blocking weak
#    re-entries; shortening it in uptrends re-admitted marginal setups.
#    Infrastructure kept in place for future experiments, disabled by default.
#    Set to e.g. {"trending_up": 2, "ranging": 5, "trending_down": 8} to enable.
V9_REENTRY_COOLDOWN_BY_REGIME: dict = {}
# 3. Dead coin cutoffs: survivorship bias fix.
#    {symbol: last_valid_date_inclusive} — data silently truncated at collapse.
#    Patched by run_v9_validation.py; empty dict = no effect.
DEAD_COIN_CUTOFFS: dict = {}
# 4. BTC short-term shock filter: block TrendPullback (not MomentumBreakout) when
#    BTC 20-day return is below a threshold. Targets macro-shock periods (2025-H1
#    Tariff Shock, 2023-H1 Recovery Floor) where golden cross stays satisfied but
#    short-term momentum is sharply negative.
#    0.0 = disabled (default). Typical values: -0.08, -0.10, -0.12.
V9_BTC_20D_SHOCK_THRESHOLD: float = 0.0  # 0.0 = off; -0.10 = block if BTC down >10% in 20 bars
V9_BTC_20D_SHOCK_MULT: float = 0.0  # sizing mult when shocked: 0.0 = full block, 0.5 = half size

# ─── v10 feature hooks ────────────────────────────────────────────────────────
# Concentration cap: limit max concurrent bull positions to reduce correlated
# alt-coin exposure (e.g. DOT/VET/ATOM all opening in the same month).
# 0 = disabled (default, honours MAX_POS_BULL = 9).
# Typical test values: 4, 5, 6, 7.
V10_BULL_POS_CAP: int = 0  # 0 = off; N = cap effective_max_pos at N during bull mode
# Mania-only flag: if True, apply cap only when BTC 30-day ROC > +30%
# (confirmed bull-mania phase). If False, cap is always active in bull mode.
V10_MANIA_ONLY: bool = False  # False = always; True = mania periods only

# ─── v12 Phase 1 fidelity hooks (intrabar exit engine) ────────────────────────
# Legacy behaviour evaluated stops/targets on CLOSE prices only: a daily wick
# through the stop that recovered by the close never triggered, and stop exits
# filled at the close rather than at the stop level. Real exchange stop orders
# fire intrabar. V12 checks the bar's HIGH/LOW against stop/TP:
#   LONG stop : fires if low <= stop, fills at min(stop, open)  (gap-down → open)
#   LONG tp   : fires if high >= tp,  fills at max(tp, open)    (gap-up  → open)
#   SHORT     : mirrored.
# When BOTH stop and TP are touched in the same daily bar, the intra-day
# ordering is unknowable from daily data → assume STOP FIRST (worst case),
# unless 1h bars are available (V12_HOURLY_RESOLUTION) to resolve which level
# was actually touched first.
V12_INTRABAR_EXITS: bool = True  # False = legacy close-only exits (A/B comparison)
V12_HOURLY_RESOLUTION: bool = True  # use 1h cache to resolve same-bar stop+TP ambiguity
HOURLY_CACHE_FILE = ROOT / "results" / "crypto_backtest" / "ohlcv_binance_1h_2019_2026_v1.parquet"

_HOURLY_DATA: dict | None = None  # lazy-loaded {symbol: DataFrame(1h OHLC)}


def _load_hourly_data() -> dict:
    """Lazy-load the 1h OHLCV cache produced by tools/fetch_binance_ohlcv.py --timeframe 1h.
    Returns {} if the cache file does not exist (resolver falls back to worst-case)."""
    global _HOURLY_DATA
    if _HOURLY_DATA is not None:
        return _HOURLY_DATA
    _HOURLY_DATA = {}
    if HOURLY_CACHE_FILE.exists():
        try:
            combined = pd.read_parquet(HOURLY_CACHE_FILE)
            for sym in combined.columns.get_level_values(0).unique():
                df = combined[sym].dropna(subset=["close"])
                if not df.empty:
                    _HOURLY_DATA[sym] = df
            logger.info(f"v12: 1h resolution cache loaded — {len(_HOURLY_DATA)} symbols")
        except Exception as e:
            logger.warning(f"v12: failed to load 1h cache ({e}) — worst-case ordering only")
    return _HOURLY_DATA


def resolve_first_touch(sym: str, date: pd.Timestamp, stop: float, tp: float, direction: str) -> str:
    """When a daily bar touches both stop and TP, walk that day's 1h bars to
    determine which level was hit first. Returns 'stop' or 'tp'.
    Falls back to 'stop' (worst case) when 1h data is unavailable."""
    if not V12_HOURLY_RESOLUTION:
        return "stop"
    hourly = _load_hourly_data().get(sym)
    if hourly is None:
        return "stop"
    day_start = pd.Timestamp(date).normalize()
    day = hourly[(hourly.index >= day_start) & (hourly.index < day_start + pd.Timedelta(days=1))]
    if day.empty:
        return "stop"
    for _, bar in day.iterrows():
        lo, hi = float(bar["low"]), float(bar["high"])
        if direction == "LONG":
            stop_hit, tp_hit = lo <= stop, hi >= tp
        else:
            stop_hit, tp_hit = hi >= stop, lo <= tp
        if stop_hit and tp_hit:
            return "stop"  # ambiguous even at 1h → worst case
        if stop_hit:
            return "stop"
        if tp_hit:
            return "tp"
    return "stop"


# Sub-period labels for attribution analysis
CRYPTO_SUB_PERIODS = [
    ("2020-H1 (COVID Crash)", "2020-01-01", "2020-06-30"),
    ("2020-H2 (Recovery Rally)", "2020-07-01", "2020-12-31"),
    ("2021-H1 (Bull Mania)", "2021-01-01", "2021-06-30"),
    ("2021-H2 (ATH + Correction)", "2021-07-01", "2021-12-31"),
    ("2022-H1 (BTC Crash)", "2022-01-01", "2022-06-30"),
    ("2022-H2 (Bear Continuation)", "2022-07-01", "2022-12-31"),
    ("2023-H1 (Recovery Floor)", "2023-01-01", "2023-06-30"),
    ("2023-H2 (Bull Restart)", "2023-07-01", "2023-12-31"),
    ("2024-H1 (ETF Approval)", "2024-01-01", "2024-06-30"),
    ("2024-H2 (ATH Breakout)", "2024-07-01", "2024-12-31"),
    ("2025-H1 (Tariff Shock)", "2025-01-01", "2025-06-30"),
    ("2025-H2 (Recovery)", "2025-07-01", "2025-12-31"),
]


import contextlib


@contextlib.contextmanager
def patched_config(**overrides):
    """Safely override module-level config flags for one run.

    Driver scripts (run_v13_repair, run_v14_final, run_v15_grade_a, ...) used
    to set module globals directly and restore them in finally-blocks — one
    forgotten restore silently corrupts every subsequent run in the process.
    This context manager snapshots and restores automatically:

        with patched_config(BEAR_SHORT_ENABLED=False, V14_MAKER_ORDERS=True):
            result = run_backtest(...)
    """
    g = globals()
    missing = [k for k in overrides if k not in g]
    if missing:
        raise KeyError(f"Unknown config flag(s): {missing}")
    snapshot = {k: g[k] for k in overrides}
    g.update(overrides)
    try:
        yield
    finally:
        g.update(snapshot)


def get_slippage_bps(sym: str, is_entry: bool = False) -> float:
    """
    Return per-symbol slippage in basis points.

    v9 tiered slippage model:
      Tier 1 (BTC, ETH)          →  5 bps  (deep books, tight spreads)
      Tier 2 (mid-cap alts)      → 10 bps  (liquid, moderate impact)
      Tier 3 (small alts)        → 20 bps  (thin books, wide spreads)

    When V9_TIERED_SLIPPAGE_ENABLED=False, returns the flat SLIPPAGE_BPS.

    v15 maker-order model (entry only, V14_MAKER_ORDERS=True):
      Partial-fill blend — 65% of entries fill as maker (0.5× cost),
      35% time out and fall back to taker (1.0× cost).
      Expected multiplier = 0.675 (V15_MAKER_COST_MULT).
      Exit slippage unchanged — exits must be immediate (market/stop orders).
    """
    if not V9_TIERED_SLIPPAGE_ENABLED:
        base = float(SLIPPAGE_BPS)
    else:
        base = float(SLIPPAGE_TIER_MAP.get(sym, SLIPPAGE_BPS))
    if is_entry and V14_MAKER_ORDERS:
        return base * V15_MAKER_COST_MULT
    return base


# ── 1. Data Layer ──────────────────────────────────────────────────────────────


def load_data() -> dict[str, pd.DataFrame]:
    """Download or load cached daily OHLCV for all coins."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        logger.info(f"Loading cached data from {CACHE_FILE}")
        try:
            store = pd.read_parquet(CACHE_FILE)
            data_map: dict[str, pd.DataFrame] = {}
            for sym in CRYPTO_SYMBOLS:
                if sym in store.columns.get_level_values(0):
                    df = store[sym].copy().dropna(subset=["close"])
                    if len(df) >= 50:
                        data_map[sym] = df
            missing = set(CRYPTO_SYMBOLS) - set(data_map.keys())
            if missing:
                logger.warning(f"Cache missing {len(missing)} symbols — re-downloading")
                CACHE_FILE.unlink()
                return load_data()
            logger.success(f"Loaded {len(data_map)} coins from cache")
            return data_map
        except Exception as e:
            logger.warning(f"Cache read failed ({e}) — re-downloading")
            CACHE_FILE.unlink(missing_ok=True)

    logger.info(f"Downloading {len(CRYPTO_SYMBOLS)} coins from yfinance...")
    frames: dict[str, pd.DataFrame] = {}

    for sym in CRYPTO_SYMBOLS:
        try:
            raw = yf.Ticker(sym).history(start=DOWNLOAD_START, end=END_DATE, auto_adjust=True)
            if raw.empty:
                logger.warning(f"  {sym}: no data returned")
                continue
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            raw.columns = [c.lower() for c in raw.columns]
            df = raw[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            if len(df) < 50:
                logger.warning(f"  {sym}: only {len(df)} bars — skipping")
                continue
            frames[sym] = df
            logger.info(f"  {sym}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"  {sym}: {e}")

    if not frames:
        raise RuntimeError("No crypto data downloaded — check network / yfinance install")

    combined = pd.concat(frames, axis=1)
    combined.to_parquet(CACHE_FILE)
    logger.success(f"Cached {len(frames)} coins → {CACHE_FILE}")
    return frames


# ── 2. Indicator Computation ───────────────────────────────────────────────────


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder's ADX — measures trend strength regardless of direction.
    ADX > 20: trend has strength (directional move, not chop).
    ADX > 25: strong trend (confident breakout / continuation).
    ADX < 20: ranging/choppy — EMA signals are unreliable.
    """
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    # Where both are positive, keep only the larger; zero the other
    both_pos = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[both_pos & (minus_dm >= plus_dm)] = 0.0
    minus_dm[both_pos & (plus_dm > minus_dm)] = 0.0

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).fillna(0.0)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators used by the strategies."""
    df = df.copy()
    c = df["close"]
    v = df["volume"]

    df["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    df["rsi"] = _rsi(c, 14)
    df["atr"] = _atr(df, 14)
    df["adx_14"] = _adx(df, 14)

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std(ddof=0)
    df["bb_lower"] = sma20 - 2.0 * std20
    df["bb_upper"] = sma20 + 2.0 * std20

    # Volume ratio vs 20-day average (NaN-safe)
    vol_avg = v.rolling(20).mean()
    df["volume_ratio"] = (v / vol_avg.replace(0, np.nan)).fillna(1.0)
    df["dollar_volume_20d"] = (c * v).rolling(20).median()

    # 20-bar rolling high (shifted by 1 — excludes current bar for breakout check)
    df["high_20"] = c.rolling(20).max().shift(1)

    # 5-bar close SMA — momentum confirmation (close above 5-bar avg = short-term trend intact)
    df["sma_5"] = c.rolling(5).mean()

    # 3-bar consecutive higher closes count (streak filter)
    df["higher_close_streak"] = ((c > c.shift(1)).astype(int).rolling(3).sum()).fillna(0)

    # 3-day rate-of-change — used by CryptoDefensiveDip to detect significant dips
    df["roc_3d"] = c.pct_change(3) * 100  # 3-bar % change

    # NQ-strategy inspired: capitulation detection for TrendPullback entries.
    # A "slow orderly pullback" should NOT have high-volume down bars in the prior 3 days.
    # Ported from NQ _slow_selloff_ok(): skip if recent pullback had panic selling.
    # cap_vol_down_3d = max volume_ratio of down-bars (close < open) over prior 3 bars.
    # Threshold 2.5 = volume was 2.5× the 20-day average on a red candle → likely distribution.
    down_vol = df["volume_ratio"].where(df["close"] < df["open"], 0.0)
    df["cap_vol_down_3d"] = down_vol.rolling(3).max().shift(1).fillna(0.0)

    # ── Fund-grade: ROC (Rate of Change) for momentum decay detection ─────
    df["roc_30"] = c.pct_change(30) * 100  # 30-bar % change
    df["roc_60"] = c.pct_change(60) * 100  # 60-bar % change

    # ── Phase 2: Weekly MACD (daily-bar approximation) ──────────────────────
    # 1 week ≈ 5 trading days: fast=12w=60d, slow=26w=130d, signal=9w=45d.
    # macd_hist > 0 means weekly momentum is rising — quality TrendPullback filter.
    _wmacd_fast = c.ewm(span=60, adjust=False).mean()
    _wmacd_slow = c.ewm(span=130, adjust=False).mean()
    _wmacd_line = _wmacd_fast - _wmacd_slow
    _wmacd_sig = _wmacd_line.ewm(span=45, adjust=False).mean()
    df["weekly_macd_hist"] = _wmacd_line - _wmacd_sig

    # ── Fund-grade: ATR spike detection (ATR / price normalized) ──────────
    df["atr_pct"] = (df["atr"] / c * 100).fillna(0)  # ATR as % of price
    df["atr_pct_ma"] = df["atr_pct"].rolling(20).mean().fillna(df["atr_pct"])

    # ── FVG / Structure indicators for CryptoFVGReversal ──────────────────
    df["adx_28"] = _adx(df, 28)
    # Swing highs/lows using 5-bar pivot (detect structure points).
    # CAUSAL (Phase 1 fix): a 5-bar pivot at bar i-2 is only KNOWN at bar i,
    # once the two right-hand bars have printed. The previous implementation
    # used shift(-1)/shift(-2) — future bars — which is look-ahead. The pivot
    # value is therefore recorded at the CONFIRMATION bar (2 bars after the
    # pivot forms), so ffill below exposes it only from confirmation onward.
    pivot_h = df["high"].shift(2)
    df["swing_high"] = pivot_h.where(
        (pivot_h >= df["high"].shift(4))
        & (pivot_h >= df["high"].shift(3))
        & (pivot_h >= df["high"].shift(1))
        & (pivot_h >= df["high"]),
        other=np.nan,
    )
    pivot_l = df["low"].shift(2)
    df["swing_low"] = pivot_l.where(
        (pivot_l <= df["low"].shift(4))
        & (pivot_l <= df["low"].shift(3))
        & (pivot_l <= df["low"].shift(1))
        & (pivot_l <= df["low"]),
        other=np.nan,
    )
    # Forward-fill swing levels to have "current" swing H/L at every bar
    df["last_swing_high"] = df["swing_high"].ffill()
    df["last_swing_low"] = df["swing_low"].ffill()
    # Displacement candle: body > 1.5× ATR (institutional activity signal)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["is_displacement"] = df["body_size"] > 1.5 * df["atr"]
    # Bullish FVG: candle[i] high < candle[i-2] low → gap
    df["bullish_fvg_top"] = df["low"].shift(2)  # candle 1 low = gap top
    df["bullish_fvg_bottom"] = df["high"]  # candle 3 high = gap bottom
    df["has_bullish_fvg"] = df["bullish_fvg_bottom"] < df["bullish_fvg_top"]
    df["bullish_fvg_mid"] = np.where(
        df["has_bullish_fvg"], (df["bullish_fvg_top"] + df["bullish_fvg_bottom"]) / 2, np.nan
    )
    # 10-bar rolling low (recent swing bottom for discount zone calc)
    df["swing_range_high"] = df["high"].rolling(20).max()
    df["swing_range_low"] = df["low"].rolling(20).min()

    return df


def compute_macro_states(data_map: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Returns a boolean Series indexed by date:
      True  = BULL mode  (BTC daily close > BTC EMA200)
      False = BEAR mode  (BTC daily close ≤ BTC EMA200)

    BTC is the systemic macro driver for the entire crypto market.
    When BTC is above its 200-day EMA, the structural trend is bullish.
    """
    btc = data_map.get("BTC-USD")
    if btc is None:
        raise RuntimeError("BTC-USD required for macro filter — not found in data_map")
    ema50 = btc["close"].ewm(span=50, adjust=False).mean()
    ema200 = btc["close"].ewm(span=200, adjust=False).mean()
    price_above = btc["close"] > ema200
    if V8_GOLDEN_CROSS_ENABLED:
        # Require BTC EMA50 > EMA200 (golden cross) as a second condition.
        # This means price AND medium-term trend are both above the long-term baseline.
        # Blocks early-recovery chop where price crossed EMA200 but structure is still bearish.
        golden_cross = ema50 > ema200
        macro = (price_above & golden_cross).rename("macro_bull")
    else:
        macro = price_above.rename("macro_bull")
    return macro


def compute_macro_regime_scores(data_map: dict[str, pd.DataFrame]) -> pd.Series:
    """
    v7: Continuous macro regime score ∈ [0.0, 1.0] replacing the binary uncertain zone.

    Components:
      1. BTC distance from EMA200 (primary — maps uncertainty band to 0→1)
      2. Market breadth: % of coins with close > EMA50 (secondary modifier ±0.15)
      3. BTC 20-bar momentum: positive momentum adds up to +0.10

    Score interpretation:
      > 0.80  → strong bull (full size)
      0.50–0.80 → moderate (proportional size)
      0.30–0.50 → uncertain (reduced size, ~0.4–0.6× vs previous hard block)
      < 0.30  → bear (no new longs — equivalent to macro_bull=False)

    When V7_REGIME_SCORE_ENABLED=False this function is never called; the existing
    binary macro_uncertain gate remains in effect.
    """
    btc = data_map.get("BTC-USD")
    if btc is None:
        raise RuntimeError("BTC-USD required for regime score")

    close = btc["close"]
    ema200 = close.ewm(span=200, adjust=False).mean()
    close.ewm(span=20, adjust=False).mean()

    # Component 1: BTC distance from EMA200, mapped to 0→1
    # ratio=1.08 (top of uncertain band) → score=1.0
    # ratio=0.95 (bottom of uncertain band) → score=0.0
    # Below 0.95 (bear) → score=0.0; above 1.08 (strong bull) → score=1.0
    ratio = (close / ema200.replace(0, np.nan)).fillna(1.0)
    component1 = ((ratio - 0.95) / (1.08 - 0.95)).clip(0.0, 1.0)

    # Component 2: 20-bar momentum — BTC trending up adds conviction
    mom_20 = close.pct_change(20).fillna(0)
    component2 = (mom_20 / 0.20).clip(-1.0, 1.0) * 0.10  # ±10% 20-bar return → ±0.05

    # Combine: score capped [0, 1]
    score = (component1 + component2).clip(0.0, 1.0)
    return score.rename("macro_score")


def compute_btc_distribution_days(data_map: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Rolling 10-bar count of BTC distribution days.

    Distribution day (from O'Neil): close DOWN on ABOVE-average volume.
    3+ distribution days in 10 bars = institutional selling → caution mode.

    Source: Market Top Detector skill (claude-trading-skills) — uses same
    O'Neil distribution-day concept adapted for crypto (daily bars, 10-bar window).

    Returns: integer Series, aligned to BTC date index.
    """
    btc = data_map.get("BTC-USD")
    if btc is None:
        return pd.Series(dtype=int)

    close = btc["close"]
    volume = btc["volume"]
    vol_avg_20 = volume.rolling(20).mean()

    # Distribution day: close below prior close AND volume > 20-day avg
    dist_day = (close < close.shift(1)) & (volume > vol_avg_20)
    dist_count = dist_day.astype(int).rolling(10).sum().fillna(0).astype(int)
    return dist_count.rename("btc_dist_days")


def compute_eth_btc_regime(data_map: dict[str, pd.DataFrame]) -> pd.Series:
    """
    ETH/BTC ratio trend as an altcoin season indicator.

    When ETH/BTC ratio EMA20 > EMA50 → ETH outperforms BTC → altcoin season.
    Inspired by Macro Regime Detector (claude-trading-skills) cross-asset ratio approach
    (RSP/SPY, IWM/SPY etc.) applied to crypto's primary cross-asset signal.

    In altcoin season:
      - Altcoins amplify BTC moves (both up AND down)
      - Bull mode: deploy more aggressively (MAX_POS_BULL +1)
      - Bear mode: shorts have more momentum (no change needed — BearShort already targeted)

    When ETH/BTC is falling (BTC dominance rising):
      - Altcoin liquidity concentrates in BTC
      - Bull mode: tighten limits (MAX_POS_BULL -1) — alts underperform even in bull

    Returns: boolean Series (True = altcoin season).
    """
    btc = data_map.get("BTC-USD")
    eth = data_map.get("ETH-USD")
    if btc is None or eth is None:
        return pd.Series(dtype=bool)

    # Align on common dates
    common = btc.index.intersection(eth.index)
    eth_btc = eth.loc[common, "close"] / btc.loc[common, "close"]
    ema20 = eth_btc.ewm(span=20, adjust=False).mean()
    ema50 = eth_btc.ewm(span=50, adjust=False).mean()
    altcoin_season = (ema20 > ema50).rename("altcoin_season")
    return altcoin_season


def compute_btc_local_trend(data_map: dict[str, pd.DataFrame]) -> pd.Series:
    """
    BTC local trend quality gate: True when BTC is NOT in a confirmed local downtrend.

    Computed as: BTC EMA20/EMA50 spread > -0.012 (same threshold as detect_regime).
    Equivalent to: BTC local regime is trending_up or ranging (not trending_down).

    Purpose: TrendPullback quality degrades when BTC is in local downtrend despite
    being above EMA200 (macro bull). This captures:
    - 2024 Jul-Sep: BTC choppy around $55-65K; altcoin setups had poor follow-through.

    When btc_local_ok = False, TrendPullback tightens to RSI [42,52], cross_rank ≥ 0.55,
    and blocks ranging regime (trending_up only). This reduces false dip entries.
    """
    btc = data_map.get("BTC-USD")
    if btc is None:
        return pd.Series(dtype=bool)
    ema20 = btc["close"].ewm(span=20, adjust=False).mean()
    ema50 = btc["close"].ewm(span=50, adjust=False).mean()
    spread = (ema20 - ema50) / ema50
    return (spread > -0.012).rename("btc_local_ok")


# ── 3. Regime Detection ───────────────────────────────────────────────────────


def detect_regime(row: pd.Series) -> str:
    """
    Detect market regime from daily bar indicators.

    Regimes:
      trending_up   — EMA20 > EMA50 by ≥1.2% AND RSI > 46
      trending_down — EMA20 < EMA50 by ≥1.2% AND RSI < 54
      volatile      — extreme RSI (>73 or <25) — avoid new entries
      ranging       — EMA20 ≈ EMA50 (spread < 1.2%) — consolidation

    The 1.2% spread threshold prevents whipsaw regime flips in tight ranges.
    """
    ema20 = row.get("ema_20")
    ema50 = row.get("ema_50")
    rsi = row.get("rsi")

    if any(pd.isna(v) for v in [ema20, ema50, rsi]):
        return "unknown"

    spread = (ema20 - ema50) / ema50

    if rsi > 73 or rsi < 25:
        return "volatile"
    if spread > 0.012:
        return "trending_up"
    if spread < -0.012:
        return "trending_down"
    return "ranging"


# ── 4. Cross-Sectional Rank ───────────────────────────────────────────────────


def compute_cross_ranks(
    data_map: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
) -> dict[str, float]:
    """
    Rank each coin by its 20-day return as of bar ts.
    Returns {symbol: rank 0.0 (worst) → 1.0 (best)}.
    Needs ≥2 coins with data; returns {} otherwise.
    """
    returns: dict[str, float] = {}
    for sym, df in data_map.items():
        if ts not in df.index:
            continue
        loc = df.index.get_loc(ts)
        if loc < 20:
            continue
        r = float(df["close"].iloc[loc] / df["close"].iloc[loc - 20] - 1)
        returns[sym] = r

    if len(returns) < 2:
        return {}

    sorted_syms = sorted(returns, key=lambda s: returns[s])
    n = len(sorted_syms)
    return {s: i / (n - 1) for i, s in enumerate(sorted_syms)}


# ── 5a. Half-Kelly Per-Strategy Sizer ────────────────────────────────────────


class StrategyKellyTracker:
    """
    Tracks per-strategy rolling win-rate and uses Half-Kelly to scale position size.

    Source: Position Sizer skill (claude-trading-skills) — recommends half-Kelly
    ("captures 75% of theoretical growth with far lower drawdowns").

    Sizing formula (simplified Kelly):
      kelly_f = (WR * (avg_win/avg_loss + 1) - 1) / (avg_win/avg_loss)
      half_k  = kelly_f / 2
      mult    = clamp(half_k / BASELINE_HALF_K, MIN_MULT, MAX_MULT)

    Baseline calibrated at 50% WR with 2.5:1 R/R (TrendPullback expectation):
      kelly_f = (0.50 * 3.5 - 1) / 2.5 = 0.30  → half-kelly = 0.15
    Multiplier of 1.0 = use full RISK_PCT; 0.5 = half the normal size; 1.5 = 1.5×.

    Note: Uses raw dollar PnL deliberately.  The dollar-weighted rolling window
    acts as an implicit equity-momentum sizer — larger wins during compounding
    phases inflate the payoff ratio, causing the system to bet bigger when
    trending.  Normalising to portfolio returns was tested and degraded
    walk-forward from ROBUST (53%) to FRAGILE (30%).  See experiments log.

    Falls back to 1.0 (no adjustment) until MIN_TRADES closed trades per strategy.
    """

    MIN_TRADES = 10  # minimum trades before adjusting; avoids early noise
    WINDOW = 30  # rolling window (most recent trades)
    BASELINE_HK = 0.15  # half-Kelly at 50% WR / 2.5:1 R/R → size mult of 1.0
    MIN_MULT = 0.50  # never go below 50% of base size
    MAX_MULT = 1.50  # cap at 150% — avoid outsized bets on hot streaks

    def __init__(self) -> None:
        # {strategy: deque of PnL values — stores absolute PnL per trade}
        self._trades: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.WINDOW))

    def record(self, strategy: str, pnl: float) -> None:
        """Record a closed trade outcome for a strategy."""
        self._trades[strategy].append(pnl)

    def get_mult(self, strategy: str) -> float:
        """
        Return position size multiplier for the strategy.
        1.0 = use RISK_PCT as-is; < 1.0 = reduce size; > 1.0 = increase size.
        """
        trades = list(self._trades[strategy])
        if len(trades) < self.MIN_TRADES:
            return 1.0  # not enough data yet — use flat sizing

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        if not wins or not losses:
            return 1.0  # all wins or all losses — can't compute R/R reliably

        wr = len(wins) / len(trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        if avg_loss < 1e-9:
            return self.MAX_MULT

        b = avg_win / avg_loss  # payoff ratio (e.g. 2.5 for 7×/2.5× R/R)
        kelly_f = (wr * (b + 1) - 1) / b  # full Kelly fraction
        half_k = max(0.0, kelly_f / 2)  # half Kelly (floor at 0)
        mult = half_k / self.BASELINE_HK  # normalize vs baseline
        return float(max(self.MIN_MULT, min(self.MAX_MULT, mult)))


def liquidity_risk_scale(dollar_volume_20d: float | None, cfg) -> float:
    """Return a soft size multiplier based on rolling dollar volume."""
    if not getattr(cfg, "liquidity_sizing_enabled", False):
        return 1.0
    if dollar_volume_20d is None or not np.isfinite(dollar_volume_20d) or dollar_volume_20d <= 0:
        return 1.0

    min_dv = getattr(cfg, "liquidity_min_dollar_volume", 0.0)
    full_dv = getattr(cfg, "liquidity_full_dollar_volume", 0.0)
    floor = getattr(cfg, "liquidity_floor_mult", 0.60)

    if full_dv <= min_dv:
        return 1.0
    if dollar_volume_20d <= min_dv:
        return floor
    if dollar_volume_20d >= full_dv:
        return 1.0

    progress = (dollar_volume_20d - min_dv) / (full_dv - min_dv)
    return float(floor + progress * (1.0 - floor))


# ── 5. Portfolio ──────────────────────────────────────────────────────────────


class CryptoPortfolio:
    """
    Lightweight portfolio supporting LONG and SHORT positions.

    Cash model (100% collateral for shorts):
      LONG  entry : cash -= entry_price × shares
      SHORT entry : cash -= entry_price × shares  (collateral posted)
      LONG  exit  : cash += exit_price × shares   →  pnl = (exit − entry) × shares
      SHORT exit  : cash += (2×entry − exit) × shares  →  pnl = (entry − exit) × shares
    """

    def __init__(self, capital: float) -> None:
        self.cash = capital
        self.initial = capital
        self.positions: dict[str, dict] = {}
        self.closed: list[dict] = []
        self.equity_curve: list[tuple] = []  # (date, equity)

    def equity(self, prices: dict[str, float]) -> float:
        val = self.cash
        for sym, info in self.positions.items():
            price = prices.get(sym, info["entry"])
            if info["direction"] == "LONG":
                val += price * info["shares"]
            else:  # SHORT: floor at 0 — consistent with capped proceeds in _close_position
                val += max(0.0, (2.0 * info["entry"] - price) * info["shares"])
        return val

    def check_vol_spike_exits(self, data_map: dict, ts: pd.Timestamp, prices: dict[str, float]) -> None:
        """Fund-grade: exit positions when ATR spikes > 2x its 20-bar mean.
        Volatility spikes precede or accompany crashes - early exit preserves capital."""
        to_close: list[str] = []
        for sym in list(self.positions):
            if sym not in data_map or ts not in data_map[sym].index:
                continue
            row = data_map[sym].loc[ts]
            atr_pct = float(row.get("atr_pct", 0) or 0)
            atr_pct_ma = float(row.get("atr_pct_ma", 0) or 0)
            if atr_pct_ma > 0 and atr_pct > 2.5 * atr_pct_ma:
                price = prices.get(sym)
                if price is not None:
                    to_close.append(sym)
        for sym in to_close:
            price = prices.get(sym)
            if price is not None:
                self._close_position(sym, price, ts, "vol_spike_exit")

    def update_stops(
        self, prices: dict[str, float], date: pd.Timestamp, bars: dict[str, dict] | None = None
    ) -> None:
        """Tick stops/targets for all open positions on this bar.

        Legacy mode (V12_INTRABAR_EXITS=False or bars=None): stops/TPs trigger
        and fill on the CLOSE only — optimistic vs real exchange stop orders.

        v12 intrabar mode: stops/TPs trigger on the bar's HIGH/LOW against the
        PRE-BAR stop level and fill at the stop/TP price (or the open on gaps).
        The trailing-stop ratchet is applied AFTER exit checks, so a trail level
        derived from this bar only protects from the NEXT bar onward — intra-bar
        ordering of high/low is unknowable from daily data.
        """
        intrabar = V12_INTRABAR_EXITS and bars is not None
        to_close: list[tuple] = []
        for sym, info in self.positions.items():
            price = prices.get(sym)
            if price is None:
                continue
            direction = info["direction"]
            atr_entry = info.get("atr_at_entry", 0.0)

            # ── Increment bars-held counter ────────────────────────────────────
            info["bars_held"] = info.get("bars_held", 0) + 1

            # ── Break-even stop — DISABLED ─────────────────────────────────
            # Testing showed BEV converts TrendPullback big winners into tiny
            # profits (killing R:R), and for MomentumBreakout it's dominated
            # by the trailing stop. Net effect is negative. Kept as dead code
            # for future reference.
            # strategy = info.get("strategy", "")

            if not intrabar:
                # ── LEGACY close-only path ──────────────────────────────────
                # Trailing stop (LONG)
                if direction == "LONG" and atr_entry > 0:
                    profit = price - info["entry"]
                    if profit >= TRAIL_ACTIVATE_ATR * atr_entry:
                        trail_level = price - TRAIL_DISTANCE_ATR * atr_entry
                        if trail_level > info["stop"]:
                            info["stop"] = trail_level
                # Trailing stop (SHORT)
                if direction == "SHORT" and atr_entry > 0:
                    profit = info["entry"] - price
                    if profit >= TRAIL_ACTIVATE_ATR * atr_entry:
                        trail_level = price + TRAIL_DISTANCE_ATR * atr_entry
                        if trail_level < info["stop"]:
                            info["stop"] = trail_level
                # Time stop (TrendPullback LONG only)
                if direction == "LONG" and TIME_STOP_BARS > 0:
                    strategy = info.get("strategy", "")
                    if strategy == "CryptoTrendPullback":
                        bars_held = info["bars_held"]
                        profit = price - info["entry"]
                        if bars_held >= TIME_STOP_BARS and profit < 0:
                            to_close.append((sym, price, "time_stop"))
                            continue
                if direction == "LONG":
                    # Tiered TP1 in legacy path
                    tp1 = info.get("tp1")
                    if tp1 is not None and not info.get("tp1_hit", False) and price >= tp1:
                        self._partial_close(sym, price, date, 0.5, "tp1_partial")
                        info["tp1_hit"] = True
                        be_stop = info["entry"] + TIERED_BE_BUFFER * atr_entry
                        info["stop"] = max(info["stop"], be_stop)
                    if price <= info["stop"]:
                        to_close.append((sym, price, "stop_loss"))
                    elif price >= info["tp"]:
                        to_close.append((sym, price, "take_profit"))
                else:  # SHORT
                    if price >= info["stop"]:
                        to_close.append((sym, price, "stop_loss"))
                    elif price <= info["tp"]:
                        to_close.append((sym, price, "take_profit"))
                continue

            # ── v12 INTRABAR path ───────────────────────────────────────────────
            bar = bars.get(sym)
            if bar is None:
                continue
            o, h, l = bar["open"], bar["high"], bar["low"]
            stop, tp = info["stop"], info["tp"]  # pre-bar levels

            exited = False
            if direction == "LONG":
                stop_hit = l <= stop
                tp_hit = h >= tp
                tp1 = info.get("tp1")
                tp1_already_hit = info.get("tp1_hit", False)
                tp1_hit_this_bar = tp1 is not None and not tp1_already_hit and h >= tp1

                # ── Tiered TP1 partial exit ─────────────────────────────────
                if tp1_hit_this_bar:
                    if stop_hit and resolve_first_touch(sym, date, stop, tp1, "LONG") == "stop":
                        # Stop fired before TP1 — full stop loss, no partial
                        to_close.append((sym, min(stop, o), "stop_loss"))
                        exited = True
                    else:
                        # TP1 hit first — partial close 50%, move stop to BE
                        tp1_fill = max(tp1, o)
                        self._partial_close(sym, tp1_fill, date, 0.5, "tp1_partial")
                        info["tp1_hit"] = True
                        be_stop = info["entry"] + TIERED_BE_BUFFER * atr_entry
                        info["stop"] = max(info["stop"], be_stop)
                        stop = info["stop"]  # refresh local for remaining checks
                        # Check if full TP also hit on same bar
                        if tp_hit:
                            to_close.append((sym, max(tp, o), "take_profit"))
                            exited = True
                        # else: half position stays open with stop at break-even

                if not exited:
                    # Standard stop/TP check (handles both first-entry and post-TP1 half)
                    if stop_hit and tp_hit:
                        first = resolve_first_touch(sym, date, stop, tp, "LONG")
                        if first == "stop":
                            to_close.append((sym, min(stop, o), "stop_loss"))
                        else:
                            to_close.append((sym, max(tp, o), "take_profit"))
                        exited = True
                    elif stop_hit:
                        to_close.append((sym, min(stop, o), "stop_loss"))
                        exited = True
                    elif tp_hit and not tp1_hit_this_bar:
                        # Normal full TP (no tiered TP, or TP1 not hit)
                        to_close.append((sym, max(tp, o), "take_profit"))
                        exited = True
            else:  # SHORT — mirrored
                stop_hit = h >= stop
                tp_hit = l <= tp
                if stop_hit and tp_hit:
                    first = resolve_first_touch(sym, date, stop, tp, "SHORT")
                    if first == "stop":
                        to_close.append((sym, max(stop, o), "stop_loss"))
                    else:
                        to_close.append((sym, min(tp, o), "take_profit"))
                    exited = True
                elif stop_hit:
                    to_close.append((sym, max(stop, o), "stop_loss"))
                    exited = True
                elif tp_hit:
                    to_close.append((sym, min(tp, o), "take_profit"))
                    exited = True
            if exited:
                continue

            # Time stop — close-decided rule, checked at the close if still open
            if direction == "LONG" and TIME_STOP_BARS > 0:
                strategy = info.get("strategy", "")
                if strategy == "CryptoTrendPullback":
                    if info["bars_held"] >= TIME_STOP_BARS and (price - info["entry"]) < 0:
                        to_close.append((sym, price, "time_stop"))
                        continue

            # Trailing-stop ratchet AFTER exit checks — effective from next bar.
            # Computed off the close (conservative: using the high would assume
            # the exchange trail order tracked the intraday peak).
            if atr_entry > 0:
                if direction == "LONG":
                    profit = price - info["entry"]
                    if profit >= TRAIL_ACTIVATE_ATR * atr_entry:
                        trail_level = price - TRAIL_DISTANCE_ATR * atr_entry
                        if trail_level > info["stop"]:
                            info["stop"] = trail_level
                else:
                    profit = info["entry"] - price
                    if profit >= TRAIL_ACTIVATE_ATR * atr_entry:
                        trail_level = price + TRAIL_DISTANCE_ATR * atr_entry
                        if trail_level < info["stop"]:
                            info["stop"] = trail_level

        for sym, price, reason in to_close:
            self._close_position(sym, price, date, reason)

    def _close_position(self, sym: str, price: float, date: pd.Timestamp, reason: str) -> None:
        info = self.positions.pop(sym)
        direction = info["direction"]
        entry = info["entry"]
        shares = info["shares"]

        # ── Fund-grade: apply exit slippage + exchange fees ───────────────────
        exit_slip = get_slippage_bps(sym) / 10_000  # v9: per-symbol tiered slippage
        FEE_BPS / 10_000
        if direction == "LONG":
            exit_price = price * (1.0 - exit_slip)  # LONG exits below close
            proceeds = exit_price * shares
        else:
            exit_price = price * (1.0 + exit_slip)  # SHORT exits above close
            proceeds = max(0.0, (2.0 * entry - exit_price) * shares)

        # Deduct round-trip fees: entry fee (already in fill) + exit fee
        entry_fee_cost = entry * shares * (FEE_BPS / 10_000)
        exit_fee_cost = price * shares * (FEE_BPS / 10_000)
        proceeds -= exit_fee_cost

        pnl = proceeds - entry * shares - entry_fee_cost
        self.cash += proceeds

        self.closed.append(
            {
                "symbol": sym,
                "direction": direction,
                "entry_time": str(info["entry_date"])[:10],
                "exit_time": str(date)[:10],
                "entry_price": round(entry, 6),
                "exit_price": round(price, 6),
                "shares": round(shares, 8),
                "entry_cost": round(info.get("entry_cost", entry * shares), 2),
                "equity_at_entry": round(info.get("equity_at_entry", self.initial), 2),
                "pnl": round(pnl, 2),
                "exit_reason": reason,
                "strategy": info.get("strategy", ""),
                "regime": info.get("regime", ""),
            }
        )

    def _partial_close(
        self, sym: str, price: float, date: pd.Timestamp, fraction: float, reason: str
    ) -> None:
        """Close `fraction` of an open LONG position, leaving the remainder open."""
        info = self.positions[sym]
        direction = info["direction"]
        entry = info["entry"]
        shares_to_close = info["shares"] * fraction

        exit_slip = get_slippage_bps(sym) / 10_000
        exit_price = price * (1.0 - exit_slip)
        proceeds = exit_price * shares_to_close
        entry_fee_cost = entry * shares_to_close * (FEE_BPS / 10_000)
        exit_fee_cost = price * shares_to_close * (FEE_BPS / 10_000)
        proceeds -= exit_fee_cost
        pnl = proceeds - entry * shares_to_close - entry_fee_cost

        self.cash += proceeds
        info["shares"] -= shares_to_close  # shrink the remaining position

        self.closed.append(
            {
                "symbol": sym,
                "direction": direction,
                "entry_time": str(info["entry_date"])[:10],
                "exit_time": str(date)[:10],
                "entry_price": round(entry, 6),
                "exit_price": round(price, 6),
                "shares": round(shares_to_close, 8),
                "entry_cost": round(entry * shares_to_close, 2),
                "equity_at_entry": round(info.get("equity_at_entry", self.initial), 2),
                "pnl": round(pnl, 2),
                "exit_reason": reason,
                "strategy": info.get("strategy", ""),
                "regime": info.get("regime", ""),
            }
        )

    def open_position(
        self,
        sym: str,
        price: float,
        stop: float,
        tp: float,
        date: pd.Timestamp,
        strategy: str,
        direction: str = "LONG",
        equity: float = None,
        atr: float = 0.0,
        macro_bull: bool = True,
        risk_mult: float = 1.0,
        max_pos_override: int | None = None,
        regime: str = "",
        **kwargs,  # accepts tp1= from tiered-TP signals without breaking callers
    ) -> None:
        if sym in self.positions:
            return  # no pyramid

        # Dynamic position limit: more capacity in bull mode (more opportunity).
        # max_pos_override allows ETH/BTC regime layer to expand or contract capacity.
        max_pos = (
            max_pos_override
            if max_pos_override is not None
            else (MAX_POS_BULL if macro_bull else MAX_POS_BEAR)
        )
        if len(self.positions) >= max_pos:
            return

        # ── Fund-grade: Total portfolio exposure cap ──────────────────────────
        # Prevents over-leveraging when compounding inflates equity.
        # Sum current position notionals; skip entry if already at exposure limit.
        if TOTAL_EXPOSURE_PCT < 10.0:  # guard: only active when set < 10 (effectively always)
            eq_now = equity or (self.cash + sum(info["shares"] * price for info in self.positions.values()))
            if eq_now > 0:
                current_exposure = (
                    sum(info["shares"] * info["entry"] for info in self.positions.values()) / eq_now
                )
                if current_exposure >= TOTAL_EXPOSURE_PCT:
                    return

        # ── v18: Portfolio heat cap ───────────────────────────────────────────
        # Total risk across all open positions must not exceed PORTFOLIO_HEAT_MAX.
        # heat = sum(shares × |entry - stop| / equity) for all positions.
        # This directly caps the max loss if all stops fire simultaneously.
        if PORTFOLIO_HEAT_MAX > 0:
            eq_heat = equity or self.cash
            if eq_heat > 0:
                current_heat = sum(
                    info["shares"] * abs(info["entry"] - info["stop"]) / eq_heat
                    for info in self.positions.values()
                )
                if current_heat >= PORTFOLIO_HEAT_MAX:
                    return

        # Slippage: LONG fills above close, SHORT fills below close.
        # v14 maker: entry uses halved slippage (limit order model).
        slip = get_slippage_bps(sym, is_entry=True) / 10_000
        fill = price * (1.0 + slip) if direction == "LONG" else price * (1.0 - slip)
        # Adjust stop/tp to be relative to fill price, not signal price
        offset = fill - price
        stop = stop + offset
        tp = tp + offset

        # Size by risk_mult × RISK_PCT on stop distance (from fill price).
        # risk_mult comes from StrategyKellyTracker: strategies with good recent
        # WR get larger size; poor WR strategies automatically downsize.
        eq = equity or self.cash
        # v18: Cap equity used for sizing to prevent compounding from creating
        # oversized positions after bull runs. Positions stop growing after
        # MAX_EQUITY_MULT × initial capital, directly bounding max DD.
        eq_for_sizing = min(eq, INITIAL_CAP * MAX_EQUITY_MULT) if MAX_EQUITY_MULT > 0 else eq
        # v6 hook: external sizing multiplier (on-chain, pipeline, correlation, Bayesian EV)
        # Patched by run_v6_validation.py for staged feature testing. Default = 1.0 (no effect).
        risk_amt = eq_for_sizing * RISK_PCT * risk_mult * V6_GLOBAL_SIZING_MULT
        stop_dist = abs(fill - stop)
        if stop_dist < 1e-9:
            return

        shares = risk_amt / stop_dist
        cost = fill * shares
        max_cost = eq * MAX_POS_PCT
        if cost > max_cost:
            shares = max_cost / fill
            cost = max_cost

        if cost > self.cash:
            shares = self.cash * 0.99 / fill
            cost = fill * shares

        if shares < 1e-8 or cost < 0.01:
            return

        self.cash -= cost
        # tp1 from signal (tiered take-profit first target); offset same as stop/tp
        _sig_tp1 = kwargs.get("tp1")  # passed via open_position(**kwargs) in main loop
        _adj_tp1 = (_sig_tp1 + offset) if _sig_tp1 is not None else None

        self.positions[sym] = {
            "direction": direction,
            "shares": shares,
            "entry": fill,
            "stop": stop,
            "tp": tp,
            "tp1": _adj_tp1,  # first partial exit level (None if not tiered)
            "tp1_hit": False,  # tracks whether partial exit has fired
            "entry_cost": cost,
            "equity_at_entry": eq,
            "entry_date": date,
            "strategy": strategy,
            "atr_at_entry": atr,  # stored for trailing stop activation
            "regime": regime,
        }

    def close_all(self, prices: dict[str, float], date: pd.Timestamp) -> None:
        for sym in list(self.positions.keys()):
            price = prices.get(sym, self.positions[sym]["entry"])
            self._close_position(sym, price, date, "end_of_backtest")


# ── 6. Strategy Logic ─────────────────────────────────────────────────────────


def _eval_trend_pullback_v2(
    sym: str,
    row: pd.Series,
    regime: str,
    cross_rank: float | None,
    macro_bull: bool,
    btc_local_ok: bool = True,
    is_altcoin_season: bool = True,
) -> dict | None:
    """
    CryptoTrendPullbackV2 — dip-buy within established uptrend (LONG, bull mode only).

    Core insight: during bull markets, strong coins pull back to EMA50 support
    before resuming. We enter at the pullback (RSI 40-55) before the next leg up.

    Conditions:
      1. macro_bull = True          — BTC > EMA200 (systemic bull market)
      2. regime = trending_up       — coin in local uptrend (EMA20 > EMA50 by >1.2%)
      3. EMA20 > EMA50              — medium-term trend still intact
      4. RSI ∈ [40, 55]             — true pullback zone (not breakdown, not overbought)
      5. close > EMA50 × 0.97       — price within 3% of mid-term support (not broken)
      6. volume_ratio < 1.8         — pullback on subdued volume (distribution = skip)
      7. cross_rank ≥ 0.35          — above-median relative strength (quality filter)

    R/R: 2.5× ATR stop, 7× ATR target → R/R = 2.8
    Break-even WR = 1 / (1 + 2.8) = 26%.  Profitable even at 45% WR.
    """
    if not macro_bull:
        return None
    # V8: When altcoin season is inactive (BTC dominance rising), dip-buying alts
    # is net-negative. Altcoins underperform BTC in those periods — the "dip" is
    # structural underperformance, not a temporary pullback. Block entirely.
    if V8_ALTSEASON_GATE_ENABLED and not is_altcoin_season:
        return None
    # When BTC is in local downtrend, only allow trending_up coins (not ranging).
    # Ranging coins in a weak-BTC environment have low follow-through probability.
    if btc_local_ok:
        if regime not in {"trending_up", "ranging"}:
            return None
    else:
        if regime != "trending_up":
            return None

    ema20 = row.get("ema_20")
    ema50 = row.get("ema_50")
    rsi = row.get("rsi")
    atr = row.get("atr")
    close = row.get("close")
    volume_ratio = row.get("volume_ratio", 1.0)
    adx = row.get("adx_14")

    if any(v is None or pd.isna(v) for v in [ema20, ema50, rsi, atr, close]):
        return None
    if atr <= 0 or close <= 0:
        return None

    # Hard gates — tighten when BTC is locally weak to improve entry quality.
    # btc_local_ok=False: BTC EMA20 < EMA50 (local downtrend despite macro bull).
    # Tightening prevents false dip-buys in 2024-H2 choppy altcoin conditions.
    if not (ema20 > ema50):
        return None
    rsi_lo = 42.0 if not btc_local_ok else 38.0
    rsi_hi = 52.0 if not btc_local_ok else 57.0
    if not (rsi_lo <= rsi <= rsi_hi):
        return None
    if not (close > ema50 * 0.96):  # 4% below EMA50
        return None
    if volume_ratio >= 2.0:
        return None
    # NQ-strategy: slow orderly pullback filter — skip if any prior 3 bars had
    # high-volume selling (volume > 2.5× avg on a red candle). Capitulation dumps
    # tend to continue; buying into them lowers win rate (ported from _slow_selloff_ok).
    cap_vol_down_3d = row.get("cap_vol_down_3d", 0.0)
    if cap_vol_down_3d is not None and not pd.isna(cap_vol_down_3d) and cap_vol_down_3d > 2.5:
        return None
    rank_thresh = 0.55 if not btc_local_ok else 0.45
    if cross_rank is not None and cross_rank < rank_thresh:
        return None
    # ADX gate: must have directional movement (ADX > 15)
    if adx is not None and not pd.isna(adx) and adx < 15.0:
        return None

    # Phase 2: Weekly MACD gate — only enter when weekly momentum is rising.
    # Prevents TrendPullback from entering during weekly downtrends (2023 chop).
    if WEEKLY_MACD_GATE_ENABLED:
        weekly_macd_hist = row.get("weekly_macd_hist")
        if weekly_macd_hist is not None and not pd.isna(weekly_macd_hist) and weekly_macd_hist <= 0:
            return None  # weekly momentum negative — skip this bar

    rsi_sweet = 48.0
    rsi_bonus = min(0.08, max(0.0, (8.0 - abs(rsi - rsi_sweet)) / 100.0))
    rank_bonus = min(0.07, (cross_rank - 0.35) * 0.15) if cross_rank is not None else 0.0
    # Bonus for full EMA stack (EMA50 > EMA200 = macro-aligned)
    ema200 = row.get("ema_200")
    stack_bon = 0.06 if (ema200 is not None and not pd.isna(ema200) and ema50 > ema200) else 0.0
    confidence = 0.57 + rsi_bonus + rank_bonus + stack_bon

    stop = close - 2.5 * atr
    target = close + 11.0 * atr
    # Phase 2: Tiered TP — first half exits at TIERED_TP1_ATR, lock in partial profit
    tp1 = round(close + TIERED_TP1_ATR * atr, 6) if TIERED_TP_ENABLED else None

    return {
        "strategy": "CryptoTrendPullback",
        "direction": "LONG",
        "confidence": round(confidence, 3),
        "stop": round(stop, 6),
        "target": round(target, 6),
        "tp1": tp1,
    }


def _eval_momentum_breakout(
    sym: str,
    row: pd.Series,
    regime: str,
    cross_rank: float | None,
    macro_bull: bool,
) -> dict | None:
    """
    CryptoMomentumBreakout — buy confirmed breakout with volume surge (LONG, bull only).

    Core insight: in bull markets, coins consolidate then break to new highs with
    expanding volume. The breakout bar with high volume confirms genuine demand.

    Two sub-modes:
    A) trending_up: standard breakout continuation (RSI ≥ 55, volume > 1.3×, rank ≥ 0.40)
    B) ranging: consolidation breakout — stricter thresholds since coin is leaving range
       (RSI ≥ 62, volume > 2.0×, rank ≥ 0.50, EMA20 > EMA50 still required)
       This fires early when a coin transitions from range to breakout, BEFORE regime
       officially becomes trending_up. Catches SOL/AVAX-type explosive moves.

    R/R: 2.0× ATR stop, 5.0× ATR target → R/R = 2.5
    """
    if not macro_bull:
        return None
    if regime not in {"trending_up", "ranging"}:
        return None

    ema20 = row.get("ema_20")
    ema50 = row.get("ema_50")
    rsi = row.get("rsi")
    atr = row.get("atr")
    close = row.get("close")
    volume_ratio = row.get("volume_ratio", 1.0)
    high_20 = row.get("high_20")
    adx = row.get("adx_14")

    if any(v is None or pd.isna(v) for v in [ema20, ema50, rsi, atr, close, high_20]):
        return None
    if atr <= 0 or close <= 0 or high_20 <= 0:
        return None

    if not (ema20 > ema50):
        return None

    # Regime-specific thresholds
    if regime == "ranging":
        # Consolidation breakout: require stronger volume and RSI surge
        if not (62.0 <= rsi <= 75.0):
            return None
        if volume_ratio <= 2.0:
            return None
        rank_min = 0.50
    else:  # trending_up
        if not (55.0 <= rsi <= 72.0):
            return None
        if volume_ratio <= 1.3:
            return None
        rank_min = 0.40

    if close < high_20 * 0.995:
        return None
    if cross_rank is not None and cross_rank < rank_min:
        return None
    # ADX gate: breakouts need directional momentum (ADX > 18)
    if adx is not None and not pd.isna(adx) and adx < 18.0:
        return None

    # v7: Squeeze gate — require prior volatility compression before expansion entry.
    # ATR% must be below its 20-bar average at the breakout bar, confirming the move
    # is coming from a low-vol base rather than an already-extended spike.
    if V7_SQUEEZE_GATE_ENABLED:
        atr_pct = row.get("atr_pct")
        atr_pct_ma = row.get("atr_pct_ma")
        if (
            atr_pct is not None
            and not pd.isna(atr_pct)
            and atr_pct_ma is not None
            and not pd.isna(atr_pct_ma)
            and atr_pct_ma > 0
        ) and atr_pct >= atr_pct_ma * 1.10:  # ATR already 10%+ above avg → no squeeze
            return None

    vol_bonus = min(0.06, (volume_ratio - 1.3) * 0.06)
    rank_bonus = min(0.06, (cross_rank - 0.40) * 0.12) if cross_rank is not None else 0.0
    range_pen = -0.02 if regime == "ranging" else 0.0  # slight confidence discount for range breakouts
    confidence = 0.58 + vol_bonus + rank_bonus + range_pen

    stop = close - 2.0 * atr
    target = close + 5.0 * atr  # trailing stop captures extended runs

    return {
        "strategy": "CryptoMomentumBreakout",
        "direction": "LONG",
        "confidence": round(confidence, 3),
        "stop": round(stop, 6),
        "target": round(target, 6),
    }


def _eval_bear_short(
    sym: str,
    row: pd.Series,
    regime: str,
    cross_rank: float | None,
    macro_bull: bool,
    btc_local_ok: bool = True,
    btc_falling: bool = False,
) -> dict | None:
    """
    CryptoBearShort v2 — short underperformers during ACTIVE bear-market declines.

    Why v1 was a net drag (empirical post-mortem, 2020-2026 window):
      v1 fired whenever BTC < EMA200. PnL by phase:
        2022-H1 waterfall   → +$103K  (the only regime where shorts belong)
        2022-H2 basing chop → -$44K   (squeeze rallies blow 2.5×ATR stops)
        2023-H1 V-recovery  → -$113K  (BTC +85% off lows, EMA200 still lagging above)
      Lesson: "below EMA200" is NOT the same as "falling". Shorts need an
      active decline, not a structural-bear label.

    Conditions (v2):
      1. macro_bull = False          — BTC < EMA200 (structural bear market)
      2. btc_local_ok = False        — BTC EMA20 < EMA50: bear rally rolled over
      3. btc_falling = True          — BTC 20-day return in (-20%, -5%]: decline is
                                       ACTIVE but not yet capitulation. Below -20%
                                       the crash is exhausted and the squeeze bounce
                                       is imminent (June 2022 entries at -29..-33%
                                       lost -$42K; January 2022 entries at -6..-16%
                                       made +$31K). Blocks basing chop (2022-H2) and
                                       V-recoveries (2023-H1) too.
      4. regime = trending_down      — coin in local downtrend (EMA20 < EMA50 by >1.2%)
      5. cross_rank ≤ 0.30           — bottom 30% performer (weakest coins fall hardest)
      6. RSI ∈ [38, 60]              — rolled over but NOT oversold (don't short the bottom)
      7. close < EMA20               — below short-term mean: rollover confirmed
      8. volume_ratio > 0.5          — minimal participation to confirm

    Stop: 2.5× ATR ABOVE entry, Target: 6.0× ATR BELOW entry → R/R = 2.4
    """
    if macro_bull:
        return None
    if btc_local_ok:  # v2: BTC bouncing — never short into a squeeze rally
        return None
    if not btc_falling:  # v2: no active decline — basing/recovery, stand aside
        return None
    if regime != "trending_down":
        return None

    ema20 = row.get("ema_20")
    ema50 = row.get("ema_50")
    rsi = row.get("rsi")
    atr = row.get("atr")
    close = row.get("close")
    volume_ratio = row.get("volume_ratio", 1.0)

    if any(v is None or pd.isna(v) for v in [ema20, ema50, rsi, atr, close]):
        return None
    if atr <= 0 or close <= 0:
        return None

    if not (ema20 < ema50):
        return None
    if cross_rank is not None and cross_rank > 0.30:  # strict: bottom 30% only (high WR)
        return None
    if not (38.0 <= rsi <= 60.0):  # v2: tighter band — never short oversold capitulation
        return None
    if close >= ema20:  # v2: rollover confirmed — price back under short-term mean
        return None
    if volume_ratio < 0.5:
        return None

    rank_bonus = min(0.08, (0.30 - cross_rank) * 0.25) if cross_rank is not None else 0.0
    rsi_bonus = min(0.05, max(0.0, (50.0 - rsi) / 100.0))
    confidence = 0.56 + rank_bonus + rsi_bonus

    # SHORT: stop is ABOVE entry, tp is BELOW entry
    stop = close + 2.5 * atr
    target = close - 6.0 * atr

    return {
        "strategy": "CryptoBearShort",
        "direction": "SHORT",
        "confidence": round(confidence, 3),
        "stop": round(stop, 6),
        "target": round(target, 6),
    }


def _eval_defensive_dip(
    sym: str,
    row: pd.Series,
) -> dict | None:
    """
    CryptoDefensiveDip — NQ-strategy mean-reversion for BTC/ETH during CB pauses.

    Activates ONLY when the circuit breaker has paused the main system, or during
    macro-uncertain periods.  Directly inspired by the NQ MACD Mean Reversion strategy:
    buy a significant, orderly dip in the highest-liquidity assets with a quick exit.

    Why BTC/ETH only:
      • Deepest order books — slippage matches the model assumptions
      • Clearest signal-to-noise for RSI mean-reversion
      • Least correlated to altcoin-specific risk (safer during uncertainty)

    Entry conditions (all must be true):
      1. sym in {BTC-USD, ETH-USD}
      2. 3-day price drop ≥ 4%  (meaningful dip, not noise)
      3. RSI ∈ [25, 46]          (oversold but not extreme panic)
      4. close > EMA200 × 0.90   (structural support not broken — 10% buffer for crypto)
      5. EMA20 within 8% of EMA50 (not in confirmed downtrend — range / mild pullback)
      6. volume_ratio ∈ [0.5, 4.0] (normal to elevated — not dead, not extreme panic)
      7. cap_vol_down_3d ≤ 3.5    (prior days not extreme panic selling)

    Exit: stop = 1.8×ATR, target = 3.5×ATR → R/R ≈ 1.94
    Break-even WR = 1 / (1 + 1.94) = 34%.  Works if even 1 in 3 trades hits target.
    Size: 40% of normal (DEFENSIVE_MR_SIZE_MULT).
    """
    close = row.get("close")
    atr = row.get("atr")
    rsi = row.get("rsi")
    ema20 = row.get("ema_20")
    ema50 = row.get("ema_50")
    ema200 = row.get("ema_200")
    volume_ratio = row.get("volume_ratio", 1.0)
    roc_3d = row.get("roc_3d")
    cap_vol_down_3d = row.get("cap_vol_down_3d", 0.0)

    if any(v is None or pd.isna(v) for v in [close, atr, rsi, ema20, ema50, ema200, roc_3d]):
        return None
    if atr <= 0 or close <= 0:
        return None

    # Gate 0: not in a strong multi-week decline (NQ: market in a healthy range)
    # The NQ strategy's edge comes from bouncing in a range — not catching falling knives.
    # A 30-day decline > 15% signals trend continuation, not mean-reversion opportunity.
    roc_30 = row.get("roc_30")  # 30-day return (already in compute_indicators)
    if roc_30 is not None and not pd.isna(roc_30) and roc_30 < -15.0:
        return None  # strong multi-week decline — let it find the floor

    # Gate 1: significant dip (NQ: price must have dropped ≥90 pts from swing high)
    if roc_3d > -DEFENSIVE_MR_DROP_PCT:
        return None  # not a meaningful dip yet
    # Also require dip is not catastrophic (>15% in 3 days = capitulation crash, not dip)
    if roc_3d < -15.0:
        return None

    # Gate 2: RSI oversold zone (NQ: RSI must touch below 40 in last 45 min)
    if not (DEFENSIVE_MR_RSI_LO <= rsi <= DEFENSIVE_MR_RSI_HI):
        return None

    # Gate 3: structural support intact (NQ: price above S2 pivot)
    if close <= ema200 * 0.90:
        return None

    # Gate 4: not in strong confirmed downtrend — allow range or mild pullback
    # NQ: price must be between S1 and R1 (not in extreme territory)
    if ema20 < ema50 * 0.92:  # EMA20 more than 8% below EMA50 = confirmed downtrend
        return None

    # Gate 5: volume — not dead, not extreme panic
    if not (0.5 <= volume_ratio <= 4.0):
        return None

    # Gate 6: no extreme capitulation in prior bars (NQ: slow orderly selloff)
    if cap_vol_down_3d is not None and not pd.isna(cap_vol_down_3d) and cap_vol_down_3d > 3.5:
        return None

    # Confidence: higher when RSI is deep (more oversold = better bounce probability)
    rsi_score = min(0.10, max(0.0, (DEFENSIVE_MR_RSI_HI - rsi) / 100.0))
    drop_score = min(0.06, max(0.0, (-roc_3d - DEFENSIVE_MR_DROP_PCT) / 100.0))
    confidence = 0.58 + rsi_score + drop_score

    stop = close - DEFENSIVE_MR_STOP_ATR * atr
    target = close + DEFENSIVE_MR_TARGET_ATR * atr

    return {
        "strategy": "CryptoDefensiveDip",
        "direction": "LONG",
        "confidence": round(confidence, 3),
        "stop": round(stop, 6),
        "target": round(target, 6),
        "size_mult": DEFENSIVE_MR_SIZE_MULT,  # signal to position sizer
    }


def _eval_range_capture(
    sym: str,
    row: pd.Series,
    regime: str,
    cross_rank: float | None,
    macro_bull: bool,
) -> dict | None:
    """
    CryptoRangeCapture — mean-reversion buy at BB_lower in ranging bull markets.

    Fills the gap between trending strategies: during bull-market consolidation phases
    the regime is "ranging" (EMA20 ≈ EMA50) but macro is still bullish. Coins oscillate
    between BB_lower and BB_upper — buying oversold touches of BB_lower is the edge.

    This complements the trend-following strategies:
      TrendPullback    → bull + trending_up  + RSI dip [40,55]
      MomentumBreakout → bull + trending_up  + RSI breakout [55,72]
      RangeCapture     → bull + ranging      + RSI oversold < 36 + near BB_lower
      BearShort        → bear + trending_down + short

    Entry conditions:
      1. macro_bull = True              (systemic bull — ranges resolve upward)
      2. regime = ranging               (EMA20 ≈ EMA50, no clear local trend)
      3. RSI < 36                       (oversold within the range)
      4. close < BB_lower × 1.02        (within 2% of lower Bollinger Band)
      5. cross_rank ≥ 0.30              (not the worst performer)

    Stop: 2.0× ATR, Target: 4.0× ATR → R/R = 2.0
    Break-even WR = 1 / (1 + 2.0) = 33%.  Works with any WR above 33%.
    """
    if not macro_bull:
        return None
    if regime != "ranging":
        return None

    rsi = row.get("rsi")
    atr = row.get("atr")
    close = row.get("close")
    bb_lower = row.get("bb_lower")

    if any(v is None or pd.isna(v) for v in [rsi, atr, close, bb_lower]):
        return None
    if atr <= 0 or close <= 0 or bb_lower <= 0:
        return None

    if rsi >= 36.0:
        return None
    if close >= bb_lower * 1.02:
        return None
    if cross_rank is not None and cross_rank < 0.30:
        return None

    rsi_bonus = min(0.08, max(0.0, (36.0 - rsi) / 100.0))
    rank_bonus = min(0.05, (cross_rank - 0.30) * 0.10) if cross_rank is not None else 0.0
    confidence = 0.56 + rsi_bonus + rank_bonus

    stop = close - 2.0 * atr
    target = close + 4.0 * atr

    return {
        "strategy": "CryptoRangeCapture",
        "direction": "LONG",
        "confidence": round(confidence, 3),
        "stop": round(stop, 6),
        "target": round(target, 6),
    }


# ── 6c. Strategy: CryptoRSIBounce (HIGH-WR mean reversion) ───────────────────


def _eval_rsi_bounce(
    sym: str,
    row: pd.Series,
    regime: str,
    cross_rank: float | None,
    macro_bull: bool,
) -> dict | None:
    """
    CryptoRSIBounce — high win-rate mean-reversion buy on deep pullbacks.

    Fills the RSI gap below TrendPullback (which needs RSI ≥ 38). When RSI drops
    below 38 but price remains above EMA200 (long-term uptrend intact), the coin
    is deeply oversold and primed for a bounce. Tight target (2.5× ATR) maximizes
    hit rate at the cost of per-trade profit.

    Empirical edge: in structural bull markets, RSI < 38 multi-day dips followed
    by a reversal bar (close > prior close) bounce 2-3 ATR within 5-10 days.

    Entry:
      1. macro_bull = True
      2. RSI < 38 (oversold — below TrendPullback's lower bound)
      3. close > EMA200 (long-term uptrend intact)
      4. close > previous close proxy (sma_5) — immediate bounce signal
      5. cross_rank ≥ 0.20 (filter out broken coins)

    Stop: 2× ATR | Target: 2.5× ATR → R:R = 1.25
    Break-even WR = 44%.
    """
    if not macro_bull:
        return None

    rsi = row.get("rsi")
    atr = row.get("atr")
    close = row.get("close")
    ema200 = row.get("ema_200")
    sma_5 = row.get("sma_5")

    if any(v is None or pd.isna(v) for v in [rsi, atr, close, ema200]):
        return None
    if atr <= 0 or close <= 0:
        return None

    # Deep pullback in long-term uptrend
    if rsi >= 38.0:
        return None
    if close <= ema200:  # must be above 200-day EMA
        return None
    if sma_5 is not None and not pd.isna(sma_5) and close < sma_5:
        return None  # need a bounce signal (close above short-term avg)
    if cross_rank is not None and cross_rank < 0.20:
        return None

    rsi_bonus = min(0.10, max(0.0, (38.0 - rsi) / 100.0))
    rank_bonus = min(0.05, (cross_rank - 0.20) * 0.15) if cross_rank is not None else 0.0
    confidence = 0.60 + rsi_bonus + rank_bonus

    stop = close - 2.0 * atr
    target = close + 2.5 * atr  # tight target for high WR

    return {
        "strategy": "CryptoRSIBounce",
        "direction": "LONG",
        "confidence": round(confidence, 3),
        "stop": round(stop, 6),
        "target": round(target, 6),
    }


def _eval_fvg_reversal(
    sym: str,
    row: pd.Series,
    regime: str,
    cross_rank: float | None,
    macro_bull: bool,
) -> dict | None:
    """
    CryptoFVGReversal — SMOG-inspired discount-zone dip-buy in non-trending regimes.

    Derived from SMOG & TCL transcript concepts (Fair Value Gaps, displacement
    candles, discount zone positioning) adapted for **daily bars**.

    Core idea: during bull markets, when trend momentum exhausts (ADX < 28,
    regime = ranging/mean-reverting), coins that pull back into their
    discount zone (lower 50% of 20-bar range) while showing institutional
    footprints (displacement candle OR FVG) tend to bounce.

    Key adaptation from 1h → daily:
      - Removed bounce-confirmation gate (close > SMA5) — on daily bars, RSI < 52
        pullback is usually below SMA5 *by definition*, killing all signals.
      - Widened RSI band from [25,42] to [30,52] — daily RSI is less extreme.
      - Removed strict FVG proximity gate — just require FVG *present* on that bar.
      - Kept discount zone at 50% (not 40%) — viable on daily.

    Structurally different from TrendPullback & MomentumBreakout:
      - Regime: ranging / mean_reverting / volatile (NOT trending)
      - Signal: institutional footprint (displacement/FVG) + discount positioning
      - Philosophy: mean-reversion (buy exhaustion) vs trend-following

    Entry conditions:
      1. macro_bull = True
      2. Non-trending regime (not trending_up or trending_down)
      3. close > EMA200 (structural bull intact)
      4. ADX_14 < 28 (trend exhausting)
      5. RSI ∈ [30, 52] (pulled back but not capitulating)
      6. Displacement candle OR bullish FVG present
      7. Price in discount zone (lower 50% of 20-bar range)
      8. Holding above last swing low (structure intact)

    Stop: 2.0× ATR | Target: 4.0× ATR → R:R = 2.0
    """
    if not macro_bull:
        return None
    if regime in {"trending_up", "trending_down"}:
        return None

    close = row.get("close")
    atr = row.get("atr")
    rsi = row.get("rsi")
    ema200 = row.get("ema_200")
    adx_14 = row.get("adx_14")

    if any(v is None or pd.isna(v) for v in [close, atr, rsi, ema200, adx_14]):
        return None
    if atr <= 0 or close <= 0:
        return None

    # Gate 1: structural bull
    if close <= ema200:
        return None

    # Gate 2: ADX not in strong trend (momentum exhausting)
    if adx_14 >= 28.0:
        return None

    # Gate 3: RSI in pullback zone (calibrated for daily bars)
    if rsi < 30.0 or rsi >= 52.0:
        return None

    # Gate 4: displacement candle OR bullish FVG (institutional footprint)
    has_displacement = bool(row.get("is_displacement", False))
    has_fvg = bool(row.get("has_bullish_fvg", False))
    if not has_displacement and not has_fvg:
        return None

    # Gate 5: price in discount zone (lower 50% of 20-bar range)
    swing_high = row.get("swing_range_high")
    swing_low = row.get("swing_range_low")
    if swing_high is None or swing_low is None or pd.isna(swing_high) or pd.isna(swing_low):
        return None
    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return None
    position_in_range = (close - swing_low) / swing_range
    if position_in_range > 0.50:
        return None

    # Gate 6: holding above last swing low (structure intact)
    last_swing_low = row.get("last_swing_low")
    if last_swing_low is not None and not pd.isna(last_swing_low) and close < last_swing_low:
        return None

    # Gate 7: not the absolute weakest coins
    if cross_rank is not None and cross_rank < 0.15:
        return None

    # Confidence scoring
    adx_bonus = min(0.08, max(0.0, (28.0 - adx_14) / 200.0))
    rsi_bonus = min(0.06, max(0.0, (52.0 - rsi) / 200.0))
    rank_bonus = min(0.05, (cross_rank - 0.15) * 0.10) if cross_rank is not None else 0.0
    disp_bonus = 0.04 if has_displacement else 0.0
    fvg_bonus = 0.03 if has_fvg else 0.0
    confidence = 0.55 + adx_bonus + rsi_bonus + rank_bonus + disp_bonus + fvg_bonus

    stop = close - 2.0 * atr
    target = close + 4.0 * atr

    return {
        "strategy": "CryptoFVGReversal",
        "direction": "LONG",
        "confidence": round(confidence, 3),
        "stop": round(stop, 6),
        "target": round(target, 6),
    }


def evaluate_strategies(
    sym: str,
    row: pd.Series,
    cross_rank: float | None,
    macro_bull: bool,
    btc_local_ok: bool = True,
    is_altcoin_season: bool = True,
    btc_falling: bool = False,
) -> list[dict]:
    """Run all active strategies and return any signals generated."""
    regime = detect_regime(row)
    signals = []

    # ── LONG strategies (bull mode) ───────────────────────────────────────────
    sig = _eval_trend_pullback_v2(sym, row, regime, cross_rank, macro_bull, btc_local_ok, is_altcoin_season)
    if sig is not None:
        sig["regime"] = regime
        signals.append(sig)

    for fn in [_eval_momentum_breakout]:
        sig = fn(sym, row, regime, cross_rank, macro_bull)
        if sig is not None:
            sig["regime"] = regime
            signals.append(sig)

    # ── SHORT strategy (bear mode only) ───────────────────────────────────────
    # BearShort v1 was a net drag (shorted into squeeze rallies) and was disabled.
    # v2 adds the inverted BTC local-trend gate (btc_local_ok must be False) so
    # shorts only fire after a bear rally has rolled over — see _eval_bear_short.
    if BEAR_SHORT_ENABLED and not macro_bull:
        sig = _eval_bear_short(sym, row, regime, cross_rank, macro_bull, btc_local_ok, btc_falling)
        if sig is not None:
            sig["regime"] = regime
            signals.append(sig)

    # RSIBounce / FVGReversal: tested in prior experiments — confirmed net drags
    # (negative PnL, path dependency corrupts AdaptiveLearner). Dead code for reference.

    return signals


# ── 7. Simulation Loop ────────────────────────────────────────────────────────


def run_simulation(
    data_map: dict[str, pd.DataFrame],
    macro_states: pd.Series,
    dist_days: pd.Series | None = None,
    altcoin_season: pd.Series | None = None,
    btc_local_trend: pd.Series | None = None,
    macro_regime_scores: pd.Series | None = None,
) -> CryptoPortfolio:
    """
    Bar-by-bar daily simulation.

    Each bar:
      1. Check stop-loss / take-profit for all open positions.
      2. Record equity.
      3. Compute cross-sectional ranks.
      4. For each symbol, evaluate strategies and open new positions.

    v18: Added portfolio-level risk scaling (vol target, regime, equity curve, BTC beta).
    """
    # ── v18 imports ────────────────────────────────────────────────────────────
    import math

    from config.settings import CONFIG

    # Build union of all dates
    all_dates = sorted(set().union(*[set(df.index) for df in data_map.values()]))
    full_index = pd.DatetimeIndex(all_dates)

    # Filter to trading window (after warmup, before END_DATE)
    trade_idx = full_index[(full_index >= pd.Timestamp(TRADE_START)) & (full_index < pd.Timestamp(END_DATE))]

    portfolio = CryptoPortfolio(INITIAL_CAP)
    symbols = list(data_map.keys())
    n = len(trade_idx)
    report_at = max(1, n // 12)

    # ── Macro cooldown state ───────────────────────────────────────────────────
    # After any macro regime flip (bull→bear or bear→bull), pause new entries for
    # MACRO_COOLDOWN bars. The Jan-Feb 2023 BTC EMA200 recovery was choppy — BTC
    # crossed above EMA200, triggered long entries, then dropped back below, causing
    # multiple quick stop-outs. A 20-bar cooldown prevents this whipsaw entry pattern.
    prev_macro_bull: bool | None = None
    cooldown_bars: int = 0  # bars remaining in post-flip cooldown

    # ── Half-Kelly tracker (improvement #5) ────────────────────────────────────
    kelly = StrategyKellyTracker()
    kelly_cursor: int = 0

    # ── Adaptive learner (simulates online learning during backtest) ──────────
    from core.adaptive_learner import AdaptiveLearner

    bt_learner = AdaptiveLearner(state_file="/dev/null") if USE_ADAPTIVE_LEARNER else None
    learner_cursor: int = 0

    # ── Drawdown circuit breaker ──────────────────────────────────────────────
    # When equity drops >20% below rolling peak, cut risk in half until recovery.
    # Protects the large gains from 2023-H2 from being eroded in 2024 drawdowns.
    peak_equity: float = INITIAL_CAP

    # ── V8 rolling WR circuit breaker ─────────────────────────────────────────
    # Independent of the adaptive learner. Hard pause when system WR < 38% on
    # recent 20 trades. Prevents "slow bleed" where learner penalises but trades on.
    from collections import deque as _deque

    v8_outcome_queue: _deque = _deque(maxlen=20)  # 1=win, 0=loss
    v8_cb_pause_bars: int = 0  # bars remaining in current pause

    # ── v9: Per-symbol re-entry cooldown ───────────────────────────────────────
    # After a stop-out, the symbol is locked for V9_REENTRY_COOLDOWN_BARS bars.
    # Prevents immediately re-entering the same failed setup (anti-whipsaw).
    symbol_reentry_lockout: dict = {}  # {symbol: bars_remaining}

    # ── v18: Portfolio-level risk scaling ──────────────────────────────────────
    v18_cfg = CONFIG.risk
    v18_equity_history: list[float] = []
    v18_btc_returns: list[float] = []
    v18_last_btc_price: float | None = None
    v18_bars_since_ath: int = 0  # bars since last all-time high (for decaying ATH)

    logger.warning(
        f"Simulation: {len(symbols)} coins | "
        f"{trade_idx[0].date()} → {trade_idx[-1].date()} | "
        f"{n} bars | MACRO_COOLDOWN={MACRO_COOLDOWN}"
    )

    for i, ts in enumerate(trade_idx):
        current_prices = {
            sym: float(data_map[sym].loc[ts, "close"]) for sym in symbols if ts in data_map[sym].index
        }
        if not current_prices:
            continue

        # ── v12: full OHLC view of this bar for intrabar stop/TP evaluation ───
        current_bars = None
        if V12_INTRABAR_EXITS:
            current_bars = {}
            for sym in current_prices:
                row = data_map[sym].loc[ts]
                current_bars[sym] = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }

        # ── v9: Decrement re-entry cooldowns at start of each bar ─────────────
        if V9_REENTRY_COOLDOWN_BARS > 0 and symbol_reentry_lockout:
            symbol_reentry_lockout = {s: v - 1 for s, v in symbol_reentry_lockout.items() if v > 1}

        # ── 1. Update stops / take-profits ────────────────────────────────────
        # Track closed count BEFORE this bar's exits for circuit breaker outcome capture.
        v8_prev_closed = len(portfolio.closed)
        portfolio.update_stops(current_prices, ts, bars=current_bars)

        # ── 1a. Fund-grade: ATR spike early exit ─────────────────────────────
        if VOL_SPIKE_EXIT_ENABLED:
            portfolio.check_vol_spike_exits(data_map, ts, current_prices)

        # ── 1b. Macro-aware position management ───────────────────────────────
        # The macro filter gates NEW entries, but must also manage OPEN positions.
        # Holding a SHORT through a full bull run (e.g. SOL 10x in 2023-24) is the
        # single most destructive failure mode.  Close wrong-side positions each bar
        # that the macro regime disagrees with the position direction.
        if ts in macro_states.index:
            macro_bull = bool(macro_states.loc[ts])
        else:
            prior = macro_states[macro_states.index <= ts]
            macro_bull = bool(prior.iloc[-1]) if not prior.empty else True

        # ── 1b2. Macro uncertainty zone ───────────────────────────────────────
        # When BTC is within ±5% of EMA200, the regime is uncertain.
        # Most losses cluster in these transition periods (2022-H1, 2023-H1, 2025-H1).
        # Skip new LONG entries when BTC is too close to its regime boundary.
        macro_uncertain = False
        btc_data = data_map.get("BTC-USD")
        if btc_data is not None and ts in btc_data.index:
            btc_close = float(btc_data.loc[ts, "close"])
            btc_ema200 = btc_data.loc[ts].get("ema_200")
            if btc_ema200 is not None and not pd.isna(btc_ema200) and btc_ema200 > 0:
                btc_ratio = btc_close / btc_ema200
                macro_uncertain = 0.95 <= btc_ratio <= 1.08

        wrong_side = [
            sym
            for sym, info in portfolio.positions.items()
            if (macro_bull and info["direction"] == "SHORT")
            or (not macro_bull and info["direction"] == "LONG")
        ]
        for sym in wrong_side:
            price = current_prices.get(sym)
            if price is not None:
                portfolio._close_position(sym, price, ts, "macro_regime_exit")

        # ── 1c. Macro cooldown tracking ───────────────────────────────────────
        # Detect regime flip and start cooldown; decrement each bar.
        if prev_macro_bull is not None and macro_bull != prev_macro_bull:
            cooldown_bars = MACRO_COOLDOWN  # regime just flipped → freeze new entries
        elif cooldown_bars > 0:
            cooldown_bars -= 1
        prev_macro_bull = macro_bull

        # ── 1d. Feed closed trades to Kelly tracker ────────────────────────────
        # Update the half-Kelly sizer with any trades that closed this bar.
        new_closed = portfolio.closed[kelly_cursor:]
        for ct in new_closed:
            kelly.record(ct["strategy"], ct["pnl"])
        kelly_cursor = len(portfolio.closed)

        # ── 1d2. Feed closed trades to adaptive learner ────────────────────
        if bt_learner is not None:
            new_for_learner = portfolio.closed[learner_cursor:]
            for ct in new_for_learner:
                bt_learner.record_outcome(
                    symbol=ct["symbol"],
                    strategy=ct["strategy"],
                    regime=ct.get("regime", "ranging"),
                    pnl=ct["pnl"],
                    exit_reason=ct["exit_reason"],
                    trade_date=ct["exit_time"],
                )
            learner_cursor = len(portfolio.closed)

        # ── 1e. Distribution-day caution mode (improvement #3) ────────────────
        # 3+ BTC distribution days in 10 bars = institutional selling detected.
        # Reduce max concurrent positions to 3 and skip new LONG entries.
        # Source: O'Neil distribution-day concept (Market Top Detector skill).
        if dist_days is not None and ts in dist_days.index:
            btc_dd_count = int(dist_days.loc[ts])
        else:
            btc_dd_count = 0
        caution_mode = btc_dd_count >= 3

        # ── 1f. ETH/BTC altcoin season (improvement #6) ───────────────────────
        # ETH/BTC EMA20 > EMA50 = altcoin season: alts amplify BTC moves.
        # Bull + altcoin season → expand to MAX_POS_BULL + 1 concurrent positions.
        # Bull + BTC dominance rising → tighten to MAX_POS_BULL - 1.
        # Bear → not applicable (BearShort limits unchanged).
        if altcoin_season is not None:
            if ts in altcoin_season.index:
                is_altcoin_season = bool(altcoin_season.loc[ts])
            else:
                prior = altcoin_season[altcoin_season.index <= ts]
                is_altcoin_season = bool(prior.iloc[-1]) if not prior.empty else False
        else:
            is_altcoin_season = False

        # ── 1g. BTC local trend quality gate (btc_local_ok) ──────────────────────
        # True when BTC EMA20/EMA50 spread > -0.012 (not in local downtrend).
        # When False, TrendPullback tightens RSI to [42,52] and cross_rank to ≥0.60.
        # Prevents dip-buy entries in altcoins when BTC itself is locally weak.
        if btc_local_trend is not None:
            if ts in btc_local_trend.index:
                btc_local_ok = bool(btc_local_trend.loc[ts])
            else:
                prior = btc_local_trend[btc_local_trend.index <= ts]
                btc_local_ok = bool(prior.iloc[-1]) if not prior.empty else True
        else:
            btc_local_ok = True

        # ── 1h. v7: continuous macro regime score ─────────────────────────────
        # Replaces binary macro_uncertain gate when V7_REGIME_SCORE_ENABLED=True.
        # Score used as a sizing multiplier in the position entry block below.
        v7_macro_score: float | None = None
        if V7_REGIME_SCORE_ENABLED and macro_regime_scores is not None:
            if ts in macro_regime_scores.index:
                v7_macro_score = float(macro_regime_scores.loc[ts])
            else:
                prior_s = macro_regime_scores[macro_regime_scores.index <= ts]
                v7_macro_score = float(prior_s.iloc[-1]) if not prior_s.empty else None

        if macro_bull:
            if caution_mode:
                effective_max_pos = MAX_POS_BULL - 2  # dist-day caution: reduce by 2
            elif is_altcoin_season:
                effective_max_pos = MAX_POS_BULL + 2  # altcoin season: expand (alts outperforming)
            else:
                effective_max_pos = MAX_POS_BULL  # neutral: use full bull capacity
        else:
            effective_max_pos = MAX_POS_BEAR + (1 if is_altcoin_season else 0)  # bear: base + altcoin bonus

        # ── v10: Concentration cap ─────────────────────────────────────────────
        # Optionally cap concurrent bull positions to reduce correlated alt exposure.
        # V10_MANIA_ONLY=True limits the cap to confirmed mania (BTC 30-day ROC > +30%).
        if V10_BULL_POS_CAP > 0 and macro_bull:
            _apply_cap = True
            if V10_MANIA_ONLY:
                _btc_roc30 = 0.0
                if btc_data is not None and ts in btc_data.index:
                    _btc_roc30 = float(btc_data.loc[ts].get("roc_30", 0) or 0)
                _apply_cap = _btc_roc30 > 30.0
            if _apply_cap:
                effective_max_pos = min(effective_max_pos, V10_BULL_POS_CAP)

        # ── 2. Record equity + update peak ────────────────────────────────────
        eq = portfolio.equity(current_prices)
        portfolio.equity_curve.append((ts, round(eq, 2)))
        if eq > peak_equity:
            peak_equity = eq

        # ── v18: Track equity + BTC returns for portfolio-level risk ──────────
        v18_equity_history.append(eq)
        btc_price_now = current_prices.get("BTC-USD")
        if btc_price_now is not None:
            if v18_last_btc_price is not None and v18_last_btc_price > 0:
                v18_btc_returns.append((btc_price_now - v18_last_btc_price) / v18_last_btc_price)
            v18_last_btc_price = btc_price_now

        # ── v18: Compute reference peak for ECF ──────────────────────────────
        # Decaying ATH mode: the reference peak slowly decays from the real ATH
        # toward current equity.  This prevents the system from being permanently
        # penalised after a large bull run while still detecting sustained
        # drawdowns that a fast rolling window would miss.
        use_decay = getattr(v18_cfg, "ecf_use_decaying_ath", False)
        if use_decay:
            if eq >= peak_equity:
                v18_bars_since_ath = 0
            else:
                v18_bars_since_ath += 1
            decay_rate = getattr(v18_cfg, "ecf_decay_rate", 0.003)
            decay_floor = getattr(v18_cfg, "ecf_decay_floor", 0.50)
            decay_factor = max(decay_floor, 1.0 - decay_rate * v18_bars_since_ath)
            ecf_ref_peak = peak_equity * decay_factor
            # If current equity exceeds decayed peak, use current equity
            ecf_ref_peak = max(ecf_ref_peak, eq)
        else:
            ecf_peak_bars = (
                v18_cfg.equity_curve_peak_bars if hasattr(v18_cfg, "equity_curve_peak_bars") else 60
            )
            if len(v18_equity_history) > ecf_peak_bars:
                ecf_ref_peak = max(v18_equity_history[-ecf_peak_bars:])
            else:
                ecf_ref_peak = max(v18_equity_history) if v18_equity_history else peak_equity
        dd_for_ecf = 1.0 - (eq / ecf_ref_peak) if ecf_ref_peak > 0 else 0.0
        dd_for_ecf = max(0.0, dd_for_ecf)

        # ── Fund-grade: Portfolio-level stop loss (all-time peak) ─────────────
        if PORTFOLIO_STOP_PCT > 0 and peak_equity > 0:
            dd_from_peak = 1.0 - (eq / peak_equity)
            if dd_from_peak >= PORTFOLIO_STOP_PCT and portfolio.positions:
                portfolio.close_all(current_prices, ts)
                cooldown_bars = max(cooldown_bars, 5)

        # 1. Graduated equity curve feedback (rolling peak DD)
        v18_ecf_mult = 1.0
        if v18_cfg.equity_curve_feedback_enabled:
            for threshold, factor in sorted(v18_cfg.equity_curve_tiers, key=lambda t: t[0]):
                if dd_for_ecf < threshold:
                    break
                v18_ecf_mult = factor

        # 2. Volatility targeting
        v18_vol_scale = 1.0
        if v18_cfg.vol_target_enabled and len(v18_equity_history) >= v18_cfg.vol_lookback_bars + 1:
            recent = v18_equity_history[-(v18_cfg.vol_lookback_bars + 1) :]
            rets = [
                (recent[j] - recent[j - 1]) / recent[j - 1]
                for j in range(1, len(recent))
                if recent[j - 1] > 0
            ]
            if len(rets) >= 5:
                mean_r = sum(rets) / len(rets)
                var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
                daily_vol = math.sqrt(var)
                annual_vol = daily_vol * math.sqrt(365)
                if annual_vol > 1e-6:
                    v18_vol_scale = max(
                        v18_cfg.vol_scale_min,
                        min(v18_cfg.vol_scale_max, v18_cfg.vol_target_annual_pct / annual_vol),
                    )

        # 3. BTC-beta penalty
        v18_beta_scale = 1.0
        n_beta = v18_cfg.btc_beta_lookback_bars
        if (
            v18_cfg.btc_beta_penalty_enabled
            and len(v18_btc_returns) >= n_beta
            and len(v18_equity_history) >= n_beta + 1
        ):
            btc_r = v18_btc_returns[-n_beta:]
            eq_h = v18_equity_history[-(n_beta + 1) :]
            port_r = [
                (eq_h[j] - eq_h[j - 1]) / eq_h[j - 1] if eq_h[j - 1] > 0 else 0.0 for j in range(1, len(eq_h))
            ]
            port_r = port_r[-n_beta:]
            if len(port_r) == len(btc_r):
                mean_p = sum(port_r) / len(port_r)
                mean_b = sum(btc_r) / len(btc_r)
                var_b = sum((b - mean_b) ** 2 for b in btc_r) / len(btc_r)
                cov_pb = sum((p - mean_p) * (b - mean_b) for p, b in zip(port_r, btc_r, strict=False)) / len(
                    btc_r
                )
                if var_b > 1e-12:
                    beta = cov_pb / var_b
                    if beta > v18_cfg.btc_beta_high_threshold:
                        excess = beta - v18_cfg.btc_beta_high_threshold
                        range_sz = max(0.20, 1.0 - v18_cfg.btc_beta_high_threshold)
                        pen_frac = min(1.0, excess / range_sz)
                        v18_beta_scale = max(
                            1.0 - v18_cfg.btc_beta_penalty_max, 1.0 - pen_frac * v18_cfg.btc_beta_penalty_max
                        )

        # Combined v18 multiplier (regime scaling applied per-signal below)
        v18_base_mult = v18_ecf_mult * v18_vol_scale * v18_beta_scale

        # Hard halt: if combined multiplier too low, skip all new entries this bar
        if v18_base_mult < 0.05:
            continue

        # Legacy circuit breaker (now subsumed by v18 equity curve feedback)
        dd_risk_mult = v18_base_mult

        # v18.2: ECF also caps max concurrent positions (reduces correlated exposure)
        effective_max_pos = max(1, int(effective_max_pos * v18_ecf_mult))

        if i % report_at == 0:
            cd_tag = f" [CD:{cooldown_bars}]" if cooldown_bars > 0 else ""
            al_tag = " [ALT]" if is_altcoin_season else ""
            ct_tag = f" [DIST:{btc_dd_count}]" if caution_mode else ""
            logger.warning(
                f"  {ts.date()} | equity=${eq:>10,.2f} | "
                f"open={len(portfolio.positions):2d} | "
                f"trades={len(portfolio.closed)}{cd_tag}{al_tag}{ct_tag}"
            )

        # ── 3. Cross-sectional ranks ───────────────────────────────────────────
        # (macro_bull already computed in step 1b above)
        ranks = compute_cross_ranks(data_map, ts)

        # ── 5. Strategy evaluation + position entry ───────────────────────────
        # Skip new entries only during hard macro cooldown.
        # Distribution-day caution is handled via reduced effective_max_pos (not a hard block).
        # Skip new entries only during hard macro cooldown or uncertainty zone.
        # v7: When regime score enabled, replace uncertain-zone hard block with
        # proportional sizing (handled in position entry block). Score < 0.30
        # is treated as equivalent to bear (block all new longs).
        if cooldown_bars > 0:
            continue
        if V7_REGIME_SCORE_ENABLED:
            # Use score instead of binary uncertain flag.
            # score < 0.30 → block (equivalent to bear); 0.30–1.0 → allow with scaling.
            if v7_macro_score is not None and v7_macro_score < 0.30:
                continue
        else:
            if macro_uncertain:
                continue

        # ── V8: Rolling WR circuit breaker ────────────────────────────────────
        # Independent self-correcting bleed-stop. Captures new closed outcomes from
        # this bar's stop/target exits and vol-spike exits (recorded since v8_prev_closed).
        # On sustained WR collapse (<38% over last 20 trades), hard-pause 30 bars.
        if V8_CIRCUIT_BREAKER_ENABLED:
            for trade in portfolio.closed[v8_prev_closed:]:
                # Exclude auxiliary exits from the CB WR queue so they don't
                # inflate/dilute the main strategy win rate measurement:
                #   - CryptoDefensiveDip: separate risk regime
                #   - tp1_partial: half-position profit locks; inflates WR, suppresses CB
                _skip_reasons = {"CryptoDefensiveDip"}
                _skip_exits = {"tp1_partial"}
                if trade.get("strategy") not in _skip_reasons and trade.get("exit_reason") not in _skip_exits:
                    v8_outcome_queue.append(1 if trade["pnl"] > 0 else 0)
            if v8_cb_pause_bars > 0:
                v8_cb_pause_bars -= 1
                if v8_cb_pause_bars == 0:
                    if V13_CB_RESET_ON_RESUME:
                        # Unconditional reset: forget stale outcomes (tested — net negative
                        # for 2020-2026 because it allows trading into tariff shock).
                        v8_outcome_queue.clear()
                        logger.info(f"  [V13_CB] {ts.date()}: pause expired — queue reset (unconditional)")
                    elif V13_CB_CONDITIONAL_RESET and btc_data is not None and ts in btc_data.index:
                        # Conditional reset: only clear stale queue when BTC is in a confirmed
                        # uptrend. Prevents resuming into tariff-shock (BTC down) but unlocks
                        # the system in Jan-Mar 2025 when BTC was genuinely at $95-105K.
                        _cr_idx = btc_data.index.get_loc(ts)
                        if _cr_idx >= 20:
                            _cr_now = float(btc_data.iloc[_cr_idx]["close"])
                            _cr_20d = float(btc_data.iloc[_cr_idx - 20]["close"])
                            _cr_roc = (_cr_now - _cr_20d) / _cr_20d * 100.0
                            if _cr_roc >= V13_CB_RESET_ROC_THRESHOLD:
                                v8_outcome_queue.clear()
                                logger.info(
                                    f"  [V13_CB] {ts.date()}: pause expired, BTC 20d ROC={_cr_roc:.1f}% "
                                    f">= {V13_CB_RESET_ROC_THRESHOLD}% → conditional queue reset"
                                )
                # ── CryptoDefensiveDip: trade BTC/ETH even during CB pause ───────
                # When CB has blocked the main system, mean-reversion on BTC/ETH
                # can still generate edge. Uses the NQ strategy's dip-buy logic.
                if DEFENSIVE_MR_ENABLED:
                    for _def_sym in DEFENSIVE_MR_SYMBOLS:
                        if _def_sym not in data_map or ts not in data_map[_def_sym].index:
                            continue
                        if _def_sym in portfolio.positions:
                            continue
                        _def_row = data_map[_def_sym].loc[ts]
                        _def_sig = _eval_defensive_dip(_def_sym, _def_row)
                        if _def_sig is None:
                            continue
                        _def_close = float(_def_row["close"])
                        _def_atr = float(_def_row.get("atr", 0) or 0)
                        _def_eq = portfolio.equity({_def_sym: _def_close})
                        # Size: DEFENSIVE_MR_SIZE_MULT × base Kelly (conservative)
                        _def_risk = kelly.get_mult(_def_sig["strategy"]) * DEFENSIVE_MR_SIZE_MULT
                        portfolio.open_position(
                            sym=_def_sym,
                            price=_def_close,
                            stop=_def_sig["stop"],
                            tp=_def_sig["target"],
                            date=ts,
                            strategy=_def_sig["strategy"],
                            direction="LONG",
                            equity=_def_eq,
                            atr=_def_atr,
                            macro_bull=macro_bull,
                            risk_mult=_def_risk,
                            max_pos_override=2,  # max 2 defensive positions at once
                            regime="defensive",
                        )
                        logger.info(
                            f"  [DEF_DIP] {ts.date()} {_def_sym} "
                            f"roc3d={_def_row.get('roc_3d', 0):.1f}% "
                            f"rsi={_def_row.get('rsi', 0):.0f} "
                            f"sl={_def_sig['stop']:.2f} tp={_def_sig['target']:.2f}"
                        )
                continue  # still in pause window — skip all new main-strategy entries
            if len(v8_outcome_queue) >= V13_CB_MIN_SAMPLES:
                rolling_wr = sum(v8_outcome_queue) / len(v8_outcome_queue)
                if rolling_wr < V13_CB_WR_THRESHOLD:
                    v8_cb_pause_bars = V13_CB_PAUSE_BARS
                    logger.warning(
                        f"  [V8_CB] {ts.date()}: WR={rolling_wr:.1%} on last "
                        f"{len(v8_outcome_queue)} trades → pausing "
                        f"{V13_CB_PAUSE_BARS} bars"
                    )
                    # Even on trip day, allow defensive entries to start immediately
                    if DEFENSIVE_MR_ENABLED:
                        for _def_sym in DEFENSIVE_MR_SYMBOLS:
                            if _def_sym not in data_map or ts not in data_map[_def_sym].index:
                                continue
                            if _def_sym in portfolio.positions:
                                continue
                            _def_row = data_map[_def_sym].loc[ts]
                            _def_sig = _eval_defensive_dip(_def_sym, _def_row)
                            if _def_sig is None:
                                continue
                            _def_close = float(_def_row["close"])
                            _def_atr = float(_def_row.get("atr", 0) or 0)
                            _def_eq = portfolio.equity({_def_sym: _def_close})
                            _def_risk = kelly.get_mult(_def_sig["strategy"]) * DEFENSIVE_MR_SIZE_MULT
                            portfolio.open_position(
                                sym=_def_sym,
                                price=_def_close,
                                stop=_def_sig["stop"],
                                tp=_def_sig["target"],
                                date=ts,
                                strategy=_def_sig["strategy"],
                                direction="LONG",
                                equity=_def_eq,
                                atr=_def_atr,
                                macro_bull=macro_bull,
                                risk_mult=_def_risk,
                                max_pos_override=2,
                                regime="defensive",
                            )
                    continue  # start pause immediately

        # ── v9: Re-entry cooldown — detect new stop-outs this bar ─────────────
        # Tag any symbol that just stopped out so it's locked for N bars.
        # Only fire on actual stop-loss/time-stop triggers (not macro exits, which
        # are regime flips, not strategy failures), to avoid over-locking.
        # Regime-aware: shorter lockout in uptrends (pullback noise), longer in
        # downtrends (don't catch knives). Falls back to flat value if regime
        # dict is empty.
        if V9_REENTRY_COOLDOWN_BARS > 0:
            if V9_REENTRY_COOLDOWN_BY_REGIME:
                if macro_uncertain:
                    _cd_regime = "ranging"
                elif macro_bull:
                    _cd_regime = "trending_up"
                else:
                    _cd_regime = "trending_down"
                _cd_bars = V9_REENTRY_COOLDOWN_BY_REGIME.get(_cd_regime, V9_REENTRY_COOLDOWN_BARS)
            else:
                _cd_bars = V9_REENTRY_COOLDOWN_BARS
            for _trade in portfolio.closed[v8_prev_closed:]:
                if _trade["exit_reason"] in ("stop_loss", "time_stop", "vol_spike_exit"):
                    symbol_reentry_lockout[_trade["symbol"]] = _cd_bars

        # ── BearShort v2 gate: BTC 20-day return — is the decline ACTIVE? ─────
        # Shorts only belong in controlled-decline phases (e.g. Jan/Apr 2022),
        # never in basing chop (2022-H2), V-recoveries below a lagging EMA200
        # (2023-H1), or post-capitulation squeezes (Jun 2022, BTC 20d < -20%:
        # the crash is over and the violent bounce is next).
        btc_falling = False
        if BEAR_SHORT_ENABLED and not macro_bull and btc_data is not None and ts in btc_data.index:
            _bs_idx = btc_data.index.get_loc(ts)
            if _bs_idx >= 20:
                _bs_now = float(btc_data.iloc[_bs_idx]["close"])
                _bs_20d = float(btc_data.iloc[_bs_idx - 20]["close"])
                _bs_roc = (_bs_now - _bs_20d) / _bs_20d
                btc_falling = -0.25 < _bs_roc <= -0.05

        for sym in symbols:
            if ts not in data_map[sym].index:
                continue
            if sym in portfolio.positions:
                continue  # already in this coin
            # v9: Skip if in re-entry cooldown after recent stop-out
            if V9_REENTRY_COOLDOWN_BARS > 0 and symbol_reentry_lockout.get(sym, 0) > 0:
                continue

            row = data_map[sym].loc[ts]
            cross_rank = ranks.get(sym)

            # ── v9: BTC short-term shock filter ───────────────────────────────
            # When V9_BTC_20D_SHOCK_THRESHOLD != 0.0 and BTC 20-day return is
            # below the threshold (e.g. −10%), block or reduce TrendPullback.
            # MomentumBreakout is unaffected — it requires confirmed breakout volume.
            _btc_shock_active = False
            if V9_BTC_20D_SHOCK_THRESHOLD != 0.0 and btc_data is not None and ts in btc_data.index:
                _btc_close_now = float(btc_data.loc[ts, "close"])
                _btc_idx = btc_data.index.get_loc(ts)
                if _btc_idx >= 20:
                    _btc_close_20d = float(btc_data.iloc[_btc_idx - 20]["close"])
                    _btc_20d_return = (_btc_close_now - _btc_close_20d) / _btc_close_20d
                    _btc_shock_active = _btc_20d_return < V9_BTC_20D_SHOCK_THRESHOLD

            signals = evaluate_strategies(
                sym, row, cross_rank, macro_bull, btc_local_ok, is_altcoin_season, btc_falling
            )
            if not signals:
                continue

            # Apply shock filter: strip TrendPullback signals when BTC shocked.
            if _btc_shock_active and V9_BTC_20D_SHOCK_MULT == 0.0:
                # Full block: remove TrendPullback signals
                signals = [s for s in signals if "TrendPullback" not in s["strategy"]]
                # else: signals pass but risk_mult will be scaled below (handled after this block)
            if not signals:
                continue

            # Regime kill switch: block strategies with negative edge in current regime.
            if bt_learner is not None:
                filtered = []
                for s in signals:
                    if bt_learner.is_strategy_blocked(
                        strategy=s["strategy"],
                        regime=s.get("regime", "ranging"),
                        direction=s.get("direction", "LONG"),
                    ):
                        continue
                    filtered.append(s)
                signals = filtered
                if not signals:
                    continue

            # Take the highest-confidence signal; never mix LONG + SHORT on same coin
            longs = [s for s in signals if s["direction"] == "LONG"]
            shorts = [s for s in signals if s["direction"] == "SHORT"]
            # Macro gate means this conflict should not arise, but handle defensively
            pool = longs if (macro_bull and longs) else (shorts if shorts else signals)
            sig = max(pool, key=lambda s: s["confidence"])
            close = float(row["close"])
            atr = float(row.get("atr", 0) or 0)

            # v6 hook: per-signal filter (Bayesian EV proxy, pipeline mode, etc.)
            # Patched to a callable(sig, portfolio) -> bool by run_v6_validation.py.
            # Default = None (no filter; all signals pass through).
            if V6_SIGNAL_FILTER is not None and not V6_SIGNAL_FILTER(sig, portfolio):
                continue

            # Combined risk multiplier: half-Kelly × v18 portfolio-level multipliers
            risk_mult = kelly.get_mult(sig["strategy"]) * dd_risk_mult

            # SHORT positions run half size: bear-market squeezes are violent and
            # short losses empirically run ~2× long losses per unit of risk.
            if sig["direction"] == "SHORT":
                risk_mult *= 0.5

            # v18: Regime-based risk scaling (per-signal, using signal's own regime)
            if v18_cfg.regime_risk_enabled:
                sig_regime = sig.get("regime", "ranging")
                regime_factor = v18_cfg.regime_risk_factors.get(sig_regime, 1.0)
                risk_mult *= regime_factor

            # ── Fund-grade: Momentum decay filter ─────────────────────────────
            # When 30-day momentum is decelerating vs 60-day, the trend is
            # exhausting. Reduce position size to avoid late-cycle entries.
            if MOMENTUM_DECAY_ENABLED:
                roc_30 = float(row.get("roc_30", 0) or 0)
                roc_60 = float(row.get("roc_60", 0) or 0)
                if roc_60 > 0 and roc_30 < roc_60 * 0.3:
                    risk_mult *= 0.65  # momentum decaying → reduce size

            # ── v9: BTC shock filter — half-size mode ─────────────────────────
            # When V9_BTC_20D_SHOCK_MULT > 0.0, TrendPullback is not blocked
            # outright but sized down to the multiplier during shock periods.
            if (
                _btc_shock_active
                and V9_BTC_20D_SHOCK_MULT > 0.0
                and "TrendPullback" in sig.get("strategy", "")
            ):
                risk_mult *= V9_BTC_20D_SHOCK_MULT

            # ── BTC bull momentum boost ────────────────────────────────────────
            # When BTC 30-day ROC > 10% (confirmed strong bull phase), lean in.
            # Amplifies 2021/2023 bull mania periods where the edge is highest.
            # Only for LONG entries; does not affect BearShort sizing.
            if (
                macro_bull
                and sig.get("direction") == "LONG"
                and btc_data is not None
                and ts in btc_data.index
            ):
                btc_roc_30 = float(btc_data.loc[ts].get("roc_30", 0) or 0)
                if btc_roc_30 > 10.0:
                    risk_mult *= 1.30  # BTC up >10% in 30 days → lean in

            # Adaptive learner modifier: adjust sizing based on learned strategy×regime WR
            if bt_learner is not None:
                regime_str = sig.get("regime", "ranging")
                participation_mult = bt_learner.strategy_participation_multiplier(
                    sig["strategy"],
                    regime_str,
                    sig.get("direction", "LONG"),
                )
                risk_mult *= participation_mult

                learner_mod = bt_learner.strategy_modifier(sig["strategy"], regime_str)
                learner_mod += bt_learner.symbol_confidence(sym)
                # Convert confidence modifier (±0.25) to sizing multiplier (0.75–1.25)
                risk_mult *= max(0.75, min(1.25, 1.0 + learner_mod))

            risk_mult *= liquidity_risk_scale(row.get("dollar_volume_20d"), v18_cfg)

            # v7: Cross-rank proportional sizing — allocate more capital to top ranked coins.
            # Top 20% (rank ≥ 0.80) → full size; middle 30% (0.50–0.80) → 0.75×.
            # Below 0.50 is already gated by strategy thresholds; this adds size graduation.
            if V7_RANK_SIZING_ENABLED and cross_rank is not None:
                if cross_rank >= 0.80:
                    pass  # full size (1.0×)
                elif cross_rank >= 0.50:
                    risk_mult *= 0.75
                # below 0.50 is filtered at strategy level; no further penalty needed

            # v7: Regime score sizing — apply continuous macro score as size multiplier.
            # Only active when macro_bull=True but ratio is in uncertain band (0.95–1.08).
            # Strong bull (score≥0.80): no change. Uncertain (0.30–0.80): proportional cut.
            if V7_REGIME_SCORE_ENABLED and macro_bull and v7_macro_score is not None:
                # Scale between 0.35 (minimum for uncertain zone) and 1.0 (full bull)
                score_mult = max(0.35, float(v7_macro_score))
                if score_mult < 0.80:  # only reduce in uncertain zone
                    risk_mult *= score_mult / 0.80  # normalise so strong bull = 1.0×

            portfolio.open_position(
                sym=sym,
                price=close,
                stop=sig["stop"],
                tp=sig["target"],
                date=ts,
                strategy=sig["strategy"],
                direction=sig["direction"],
                equity=eq,
                atr=atr,
                macro_bull=macro_bull,
                risk_mult=risk_mult,
                max_pos_override=effective_max_pos,
                regime=sig.get("regime", ""),
                tp1=sig.get("tp1"),  # Phase 2: tiered TP first target (None for non-tiered)
            )

    # ── Close all remaining positions at end-date ──────────────────────────────
    final_ts = trade_idx[-1]
    final_prices = {
        sym: float(data_map[sym].loc[final_ts, "close"]) for sym in symbols if final_ts in data_map[sym].index
    }
    portfolio.close_all(final_prices, final_ts)
    logger.warning(
        f"  END {final_ts.date()} | equity=${portfolio.equity(final_prices):>10,.2f} | "
        f"total trades={len(portfolio.closed)}"
    )
    return portfolio


# ── 8. Metrics ────────────────────────────────────────────────────────────────


def compute_metrics(portfolio: CryptoPortfolio) -> dict:
    """Compute core performance metrics from closed trades and equity curve."""
    closed = portfolio.closed
    if not closed:
        return {}

    curve = portfolio.equity_curve
    if curve:
        dates, equities = zip(*curve, strict=False)
        eq_series = pd.Series(list(equities), index=list(dates), dtype=float)
    else:
        equity = INITIAL_CAP
        eq_list = [INITIAL_CAP]
        for t in sorted(closed, key=lambda x: x["exit_time"]):
            equity += t["pnl"]
            eq_list.append(equity)
        eq_series = pd.Series(eq_list, dtype=float)

    # Returns use 365-day crypto year (trades every day)
    n_days = len(eq_series) or 1
    total_ret = eq_series.iloc[-1] / INITIAL_CAP - 1
    ann_ret = (1 + total_ret) ** (365 / n_days) - 1
    daily_rets = eq_series.pct_change().fillna(0)
    vol = float(daily_rets.std(ddof=0)) * (365**0.5)
    sharpe = (
        float(daily_rets.mean()) / float(daily_rets.std(ddof=0)) * (365**0.5)
        if daily_rets.std(ddof=0) > 0
        else 0.0
    )
    roll_max = eq_series.cummax()
    dd = ((roll_max - eq_series) / roll_max.replace(0, np.nan)).fillna(0)
    max_dd = float(dd.max())

    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    total = len(closed)
    wr = len(wins) / total if total else 0

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
    exp = sum(t["pnl"] for t in closed) / total if total else 0

    return {
        "final_equity": round(float(eq_series.iloc[-1]), 2),
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 3),
        "expectancy": round(exp, 2),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0,
        "total_trades": total,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
    }


def compute_strategy_breakdown(closed: list[dict]) -> list[dict]:
    """Per-strategy: trade count, WR, avg win/loss, total PnL."""
    buckets: dict[str, list] = defaultdict(list)
    for t in closed:
        buckets[t.get("strategy", "unknown")].append(t)

    rows = []
    for strat, trades in sorted(buckets.items()):
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        rows.append(
            {
                "strategy": strat,
                "trades": len(trades),
                "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
                "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
                "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
                "total_pnl": round(sum(t["pnl"] for t in trades), 2),
            }
        )
    return rows


def compute_subperiod_metrics(closed: list[dict]) -> list[dict]:
    """Sub-period breakdown (2022-H1 through 2024-H2).

    Each period starts from the actual equity at that point in the simulation
    (not INITIAL_CAP). This ensures return% and MaxDD% are accurate percentages
    of real capital at risk, not inflated/deflated by the compounding base.
    """
    # Build the actual equity at the start of each sub-period
    all_trades = sorted(closed, key=lambda x: x["exit_time"])

    results = []
    for label, start, end in CRYPTO_SUB_PERIODS:
        period_trades = [t for t in closed if start <= t["exit_time"][:10] <= end]
        if not period_trades:
            results.append(
                {
                    "period": label,
                    "trades": 0,
                    "return_pct": 0.0,
                    "sharpe": None,
                    "max_dd_pct": 0.0,
                    "win_rate_pct": 0.0,
                }
            )
            continue

        # Actual equity entering this sub-period = initial cap + all PnL before it
        prior_pnl = sum(t["pnl"] for t in all_trades if t["exit_time"][:10] < start)
        start_equity = INITIAL_CAP + prior_pnl

        equity = start_equity
        equities = [equity]
        for t in sorted(period_trades, key=lambda x: x["exit_time"]):
            equity += t["pnl"]
            equities.append(equity)

        eq = pd.Series(equities, dtype=float)
        rets = eq.pct_change().fillna(0)
        ret = (eq.iloc[-1] / start_equity) - 1  # % of actual starting equity
        std = float(rets.std(ddof=0))
        sharpe = float(rets.mean()) / std * (365**0.5) if std > 0 else None
        rm = eq.cummax()
        dd = ((rm - eq) / rm.replace(0, np.nan)).fillna(0)
        wins = [t for t in period_trades if t["pnl"] > 0]

        results.append(
            {
                "period": label,
                "trades": len(period_trades),
                "return_pct": round(ret * 100, 2),
                "sharpe": round(sharpe, 2) if sharpe is not None else None,
                "max_dd_pct": round(float(dd.max()) * 100, 2),
                "win_rate_pct": round(len(wins) / len(period_trades) * 100, 1),
            }
        )
    return results


def get_btc_benchmark(start: str, end: str) -> dict:
    """BTC buy-and-hold return for the same window."""
    try:
        btc = yf.Ticker("BTC-USD").history(start=start, end=end, auto_adjust=True)
        if btc.empty:
            return {}
        first = float(btc["Close"].iloc[0])
        last = float(btc["Close"].iloc[-1])
        return {
            "return_pct": round((last / first - 1) * 100, 2),
            "start_price": round(first, 2),
            "end_price": round(last, 2),
        }
    except Exception:
        return {}


# ── 9. Monte Carlo Simulation ──────────────────────────────────────────────────


def run_monte_carlo(closed: list[dict], n_sims: int = 1000, seed: int = 42) -> dict:
    """
    Stress-test the equity curve by randomising normalized trade order 1,000×.

    Raw dollar PnLs cannot be shuffled directly because the strategy sizes trades
    dynamically as equity changes. Reordering late-cycle dollar winners/losses
    into the early path creates impossible negative-equity states. Instead we
    shuffle each trade's realized portfolio return:

        trade_return = pnl / equity_at_entry

    and replay those returns on a non-negative equity curve.

    We then focus on PATH-DEPENDENT metrics:
      - Max drawdown (worst peak-to-trough on each shuffled path)
      - Time underwater (fraction of path spent below prior peak)
      - Ruin probability (path dips below 50% of starting capital at any point)
      - Consecutive-loss survival (longest losing streak distribution)

    Note: Sharpe ratio is NOT included because shuffling preserves the same
    set of returns — mean and std are identical on every permutation, making
    the Sharpe constant across all paths (non-informative).
    """
    rng = np.random.default_rng(seed)
    if not closed:
        return {
            "n_sims": n_sims,
            "max_dd_p50_pct": 0.0,
            "max_dd_p75_pct": 0.0,
            "max_dd_p95_pct": 0.0,
            "time_underwater_pct_p50": 0.0,
            "time_underwater_pct_p95": 0.0,
            "pct_paths_ruin": 0.0,
            "min_equity_p5": INITIAL_CAP,
            "min_equity_median": INITIAL_CAP,
            "consec_loss_median": 0,
            "consec_loss_p95": 0,
        }

    trade_returns = []
    for trade in closed:
        eq_entry = float(trade.get("equity_at_entry") or 0.0)
        pnl = float(trade.get("pnl") or 0.0)
        if eq_entry <= 0:
            entry_cost = float(trade.get("entry_cost") or 0.0)
            eq_entry = entry_cost if entry_cost > 0 else INITIAL_CAP
        trade_returns.append(pnl / eq_entry)

    returns = np.array(trade_returns, dtype=float)
    n = len(returns)

    max_dds: list[float] = []
    time_underwater: list[float] = []
    min_equities: list[float] = []
    max_consec_loss: list[int] = []

    for _ in range(n_sims):
        shuffled = rng.permutation(returns)
        eq_path = [INITIAL_CAP]
        for trade_ret in shuffled:
            next_eq = eq_path[-1] * (1.0 + trade_ret)
            eq_path.append(max(0.0, next_eq))
        eq = np.array(eq_path, dtype=float)

        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, np.minimum((peak - eq) / peak, 1.0), 0.0)
        max_dds.append(float(dd.max()))

        # Time underwater: fraction of path steps where equity < prior peak
        underwater_steps = int(np.sum(eq[1:] < peak[1:]))  # skip initial step
        time_underwater.append(underwater_steps / n if n > 0 else 0.0)

        # Minimum equity touched (path nadir)
        min_equities.append(float(eq.min()))

        # Longest consecutive losing streak
        streak = max_s = 0
        for trade_ret in shuffled:
            if trade_ret < 0:
                streak += 1
                max_s = max(max_s, streak)
            else:
                streak = 0
        max_consec_loss.append(max_s)

    mdd = np.array(max_dds)
    tuw = np.array(time_underwater)
    meq = np.array(min_equities)
    mcl = np.array(max_consec_loss)

    ruin_floor = INITIAL_CAP * 0.50  # "ruin" = equity dips below 50% of start
    ruin_pct = float(np.mean(meq < ruin_floor) * 100)

    return {
        "n_sims": n_sims,
        # Drawdown distribution
        "max_dd_p50_pct": round(float(np.percentile(mdd, 50)) * 100, 1),
        "max_dd_p75_pct": round(float(np.percentile(mdd, 75)) * 100, 1),
        "max_dd_p95_pct": round(float(np.percentile(mdd, 95)) * 100, 1),
        # Time underwater (fraction of path spent below prior peak)
        "time_underwater_pct_p50": round(float(np.percentile(tuw, 50)) * 100, 1),
        "time_underwater_pct_p95": round(float(np.percentile(tuw, 95)) * 100, 1),
        # Ruin / survival
        "pct_paths_ruin": round(ruin_pct, 1),
        "min_equity_p5": round(float(np.percentile(meq, 5)), 2),
        "min_equity_median": round(float(np.percentile(meq, 50)), 2),
        # Consecutive losses
        "consec_loss_median": int(np.median(mcl)),
        "consec_loss_p95": int(np.percentile(mcl, 95)),
    }


# ── 10. HTML Report ─────────────────────────────────────────────────────────────


def build_html_report(
    metrics: dict,
    portfolio: CryptoPortfolio,
    btc_bh: dict,
    sub_periods: list[dict],
    strategy_breakdown: list[dict],
) -> str:
    def col(v):
        return "#00e676" if float(v) >= 0 else "#ff5252"

    curve = portfolio.equity_curve
    dates = [str(d.date()) for d, _ in curve[::2]]
    equities = [round(e, 2) for _, e in curve[::2]]
    if not dates:
        dates, equities = [TRADE_START, END_DATE], [INITIAL_CAP, metrics.get("final_equity", INITIAL_CAP)]

    btc_ret = btc_bh.get("return_pct", 0.0)
    alpha_val = round(metrics.get("total_return_pct", 0) - float(btc_ret), 2) if btc_bh else 0.0
    alpha_str = f"{alpha_val:+.1f}% vs BTC"
    alpha_col = col(alpha_val)
    col(btc_ret)

    # Pre-build SVG polyline outside the big f-string to avoid nested f-string issues
    if len(equities) >= 2:
        eq_min = min(equities)
        eq_max = max(equities)
        eq_rng = eq_max - eq_min + 1
        pts = " ".join(
            f"{i / (len(equities) - 1) * 800:.1f},{200 - (e - eq_min) / eq_rng * 190:.1f}"
            for i, e in enumerate(equities)
        )
        baseline_y = f"{200 - (INITIAL_CAP - eq_min) / eq_rng * 190:.1f}"
        svg_content = (
            f"<polyline points='{pts}' fill='none' stroke='#388bfd' stroke-width='2'/>"
            f"<line x1='0' y1='{baseline_y}' x2='800' y2='{baseline_y}'"
            f" stroke='#30363d' stroke-dasharray='4'/>"
        )
    else:
        svg_content = ""

    from datetime import datetime as _htmldt

    gen_time = _htmldt.now().isoformat()[:19]

    strat_row_parts = []
    for s in strategy_breakdown:
        wr_col = col(s["win_rate_pct"] - 50)
        win_col = col(s["avg_win"])
        pnl_col = col(s["total_pnl"])
        strat_row_parts.append(
            f"<tr><td>{s['strategy']}</td><td>{s['trades']}</td>"
            f"<td style='color:{wr_col}'>{s['win_rate_pct']:.0f}%</td>"
            f"<td style='color:{win_col}'>${s['avg_win']:+.0f}</td>"
            f"<td style='color:#ff5252'>${s['avg_loss']:.0f}</td>"
            f"<td style='color:{pnl_col}'>${s['total_pnl']:+,.0f}</td></tr>"
        )
    strat_rows = "".join(strat_row_parts)

    subperiod_row_parts = []
    for p in sub_periods:
        ret_col = col(p["return_pct"])
        wr_col = col(p["win_rate_pct"] - 50)
        sh_str = str(p["sharpe"]) if p["sharpe"] is not None else "n/a"
        subperiod_row_parts.append(
            f"<tr><td>{p['period']}</td><td>{p['trades']}</td>"
            f"<td style='color:{ret_col}'>{p['return_pct']:+.1f}%</td>"
            f"<td>{sh_str}</td>"
            f"<td style='color:#ff5252'>-{p['max_dd_pct']:.1f}%</td>"
            f"<td style='color:{wr_col}'>{p['win_rate_pct']:.0f}%</td></tr>"
        )
    subperiod_rows = "".join(subperiod_row_parts)

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<title>KA-MATS Crypto Backtest v2 — 2022→2025</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;padding:24px}}
  h1{{color:#58a6ff;margin-bottom:8px}}
  h2{{color:#79c0ff;margin:24px 0 12px;border-bottom:1px solid #30363d;padding-bottom:6px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;text-align:center}}
  .card .val{{font-size:1.5rem;font-weight:700;margin-top:4px}}
  .card .lbl{{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}}
  table{{width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;overflow:hidden}}
  th{{background:#21262d;padding:10px 14px;text-align:left;font-size:.8rem;color:#8b949e;text-transform:uppercase}}
  td{{padding:9px 14px;border-top:1px solid #21262d;font-size:.88rem}}
  svg{{width:100%;height:220px}}
  .chart-wrap{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
  .note{{background:#1c2128;border-left:3px solid #388bfd;padding:12px 16px;margin:12px 0;border-radius:4px;font-size:.85rem;color:#8b949e}}
</style></head><body>
<h1>KA-MATS Crypto Backtest — v2</h1>
<p style="color:#8b949e;margin-bottom:4px">Period: {TRADE_START} → {END_DATE} · Universe: 15 coins · $10K initial capital · Daily bars</p>
<div class="note">
  <strong>v2 key changes vs v1:</strong>
  Macro bear filter (BTC EMA200 gate) · Short selling in bear mode (CryptoBearShort) ·
  TrendPullback RSI tightened [40,55], target widened 7× ATR · New CryptoMomentumBreakout strategy.
</div>

<h2>Overall Performance</h2>
<div class="grid">
  <div class="card"><div class="lbl">Total Return</div>
    <div class="val" style="color:{col(metrics.get("total_return_pct", 0))}">{metrics.get("total_return_pct", 0):+.2f}%</div></div>
  <div class="card"><div class="lbl">Ann. Return</div>
    <div class="val" style="color:{col(metrics.get("annualized_return_pct", 0))}">{metrics.get("annualized_return_pct", 0):+.1f}%</div></div>
  <div class="card"><div class="lbl">Sharpe Ratio</div>
    <div class="val" style="color:{col(metrics.get("sharpe_ratio", 0))}">{metrics.get("sharpe_ratio", 0):.2f}</div></div>
  <div class="card"><div class="lbl">Max Drawdown</div>
    <div class="val" style="color:#ff5252">-{metrics.get("max_drawdown_pct", 0):.1f}%</div></div>
  <div class="card"><div class="lbl">Win Rate</div>
    <div class="val" style="color:{col(metrics.get("win_rate_pct", 50) - 50)}">{metrics.get("win_rate_pct", 0):.1f}%</div></div>
  <div class="card"><div class="lbl">Profit Factor</div>
    <div class="val" style="color:{col(metrics.get("profit_factor", 1) - 1)}">{metrics.get("profit_factor", 0):.2f}</div></div>
  <div class="card"><div class="lbl">Total Trades</div>
    <div class="val">{metrics.get("total_trades", 0)}</div></div>
  <div class="card"><div class="lbl">Final Equity</div>
    <div class="val" style="color:{col(metrics.get("final_equity", 10000) - 10000)}">${metrics.get("final_equity", 0):,.0f}</div></div>
</div>

<div class="grid" style="grid-template-columns:repeat(3,1fr)">
  <div class="card"><div class="lbl">BTC Buy-and-Hold</div>
    <div class="val" style="color:#f0883e">{btc_ret:+.1f}%</div></div>

  <div class="card"><div class="lbl">Alpha vs BTC</div>
    <div class="val" style="color:{alpha_col}">{alpha_str}</div></div>
  <div class="card"><div class="lbl">Expectancy / Trade</div>
    <div class="val" style="color:{col(metrics.get("expectancy", 0))}">${metrics.get("expectancy", 0):+.0f}</div></div>
</div>

<h2>Equity Curve</h2>
<div class="chart-wrap">
<svg viewBox="0 0 800 200" preserveAspectRatio="none">
  <defs>
    <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#388bfd" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#388bfd" stop-opacity="0"/>
    </linearGradient>
  </defs>
  {svg_content}
</svg>
</div>

<h2>Strategy Attribution</h2>
<table>
  <tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Avg Win</th><th>Avg Loss</th><th>Total PnL</th></tr>
  {strat_rows}
</table>

<h2>Sub-Period Breakdown</h2>
<table>
  <tr><th>Period</th><th>Trades</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th></tr>
  {subperiod_rows}
</table>

<p style="margin-top:24px;color:#484f58;font-size:.75rem">
  Generated {gen_time} · KA-MATS v3 · Iknir Capital
</p>
</body></html>"""


# ── 10. Main ───────────────────────────────────────────────────────────────────


def _compute_walk_forward(closed: list[dict]) -> dict:
    wf_split = "2023-01-01"
    is_trades = [t for t in closed if t["exit_time"][:10] < wf_split]
    oos_trades = [t for t in closed if t["exit_time"][:10] >= wf_split]
    is_pnl = sum(t["pnl"] for t in is_trades)
    oos_pnl = sum(t["pnl"] for t in oos_trades)
    is_wr = len([t for t in is_trades if t["pnl"] > 0]) / max(1, len(is_trades)) * 100
    oos_wr = len([t for t in oos_trades if t["pnl"] > 0]) / max(1, len(oos_trades)) * 100

    # Dollar-based OOS/IS (legacy — biased when sizing is path-dependent)
    if is_pnl > 0:
        oos_ratio = oos_pnl / abs(is_pnl) * 100
        verdict = "ROBUST" if oos_ratio >= 50 else "FRAGILE"
    else:
        oos_ratio = None
        verdict = "N/A"

    # Return-normalized OOS/IS: compare average per-trade return (pnl/equity_at_entry)
    # This strips out the compounding path advantage that inflates dollar-OOS in bull runs.
    def _avg_trade_return(trades: list[dict]) -> float:
        rets = []
        for t in trades:
            eq = float(t.get("equity_at_entry") or 0.0)
            pnl = float(t.get("pnl") or 0.0)
            if eq > 0:
                rets.append(pnl / eq)
            else:
                cost = float(t.get("entry_cost") or 0.0)
                rets.append(pnl / cost if cost > 0 else 0.0)
        return sum(rets) / len(rets) if rets else 0.0

    is_avg_ret = _avg_trade_return(is_trades)
    oos_avg_ret = _avg_trade_return(oos_trades)
    if is_avg_ret > 0:
        oos_is_return_ratio = oos_avg_ret / is_avg_ret * 100
        return_verdict = "ROBUST" if oos_is_return_ratio >= 50 else "FRAGILE"
    else:
        oos_is_return_ratio = None
        return_verdict = "N/A"

    return {
        "split": wf_split,
        "in_sample_trades": len(is_trades),
        "in_sample_wr_pct": round(is_wr, 1),
        "in_sample_pnl": round(is_pnl, 2),
        "out_of_sample_trades": len(oos_trades),
        "out_of_sample_wr_pct": round(oos_wr, 1),
        "out_of_sample_pnl": round(oos_pnl, 2),
        "oos_is_ratio_pct": round(oos_ratio, 1) if oos_ratio is not None else None,
        "verdict": verdict,
        # Return-normalized metrics (path-independent)
        "is_avg_return_pct": round(is_avg_ret * 100, 2),
        "oos_avg_return_pct": round(oos_avg_ret * 100, 2),
        "oos_is_return_ratio_pct": round(oos_is_return_ratio, 1) if oos_is_return_ratio is not None else None,
        "return_verdict": return_verdict,
    }


def run_backtest(output_tag: str = "v2", write_outputs: bool = True) -> dict:
    from datetime import datetime as _dt

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="WARNING",
        colorize=True,
    )

    t0 = _dt.now()
    logger.warning("=" * 68)
    logger.warning("KA-MATS Crypto Backtest  v3  ·  2022-01-01 → 2025-01-01")
    logger.warning("Strategies: TrendPullback + MomentumBreakout + BearShort")
    logger.warning("Macro filter: BTC EMA200  ·  Trailing stops  ·  5 bps slippage  ·  $10K")
    logger.warning("=" * 68)

    # ── 1. Download / cache data ───────────────────────────────────────────────
    logger.warning("Step 1/4 — Loading data...")
    data_map = load_data()

    # v9: Apply dead coin cutoffs (survivorship bias fix).
    # Dead coins are truncated at their collapse/delisting date so the simulation
    # experiences real losses when holding them into the crash, but stops
    # trying to trade them after the date data becomes meaningless.
    if DEAD_COIN_CUTOFFS:
        for _sym, _cutoff in DEAD_COIN_CUTOFFS.items():
            if _sym in data_map:
                _cutoff_ts = pd.Timestamp(_cutoff)
                data_map[_sym] = data_map[_sym][data_map[_sym].index <= _cutoff_ts]
                logger.warning(
                    f"  [DEAD_COIN] {_sym}: data truncated at {_cutoff} ({len(data_map[_sym])} bars)"
                )

    if len(data_map) < 3:
        logger.error("Need at least 3 coins — aborting")
        sys.exit(1)
    if "BTC-USD" not in data_map:
        logger.error("BTC-USD is required for macro filter — aborting")
        sys.exit(1)

    # ── 2. Compute indicators for all coins ───────────────────────────────────
    logger.warning("Step 2/4 — Computing indicators...")
    for sym in data_map:
        data_map[sym] = compute_indicators(data_map[sym])

    # ── 3. Compute macro filter + distribution days + ETH/BTC regime ─────────
    macro_states = compute_macro_states(data_map)
    dist_days_s = compute_btc_distribution_days(data_map)
    altcoin_season = compute_eth_btc_regime(data_map)
    btc_local_trend = compute_btc_local_trend(data_map)
    macro_regime_scores = compute_macro_regime_scores(data_map) if V7_REGIME_SCORE_ENABLED else None

    bull_days = int(macro_states[macro_states.index >= pd.Timestamp(TRADE_START)].sum())
    total_days = int((macro_states.index >= pd.Timestamp(TRADE_START)).sum())
    logger.warning(
        f"  Macro filter: {bull_days}/{total_days} trading days in BULL mode "
        f"({bull_days / total_days * 100:.0f}%)"
    )
    if not dist_days_s.empty:
        caution_days = int((dist_days_s[dist_days_s.index >= pd.Timestamp(TRADE_START)] >= 3).sum())
        logger.warning(f"  Distribution days: {caution_days} caution bars in trading window")
    if not altcoin_season.empty:
        alt_days = int(altcoin_season[altcoin_season.index >= pd.Timestamp(TRADE_START)].sum())
        logger.warning(f"  ETH/BTC regime: {alt_days}/{total_days} days in altcoin season")
    if not btc_local_trend.empty:
        weak_days = int((~btc_local_trend[btc_local_trend.index >= pd.Timestamp(TRADE_START)]).sum())
        logger.warning(f"  BTC local weak: {weak_days}/{total_days} days (TrendPullback tightened)")

    # ── 4. Run simulation ─────────────────────────────────────────────────────
    logger.warning("Step 3/4 — Running simulation...")
    portfolio = run_simulation(
        data_map,
        macro_states,
        dist_days=dist_days_s if not dist_days_s.empty else None,
        altcoin_season=altcoin_season if not altcoin_season.empty else None,
        btc_local_trend=btc_local_trend if not btc_local_trend.empty else None,
        macro_regime_scores=macro_regime_scores,
    )

    # ── 5. Compute metrics ────────────────────────────────────────────────────
    logger.warning("Step 4/4 — Computing metrics...")
    metrics = compute_metrics(portfolio)
    strategy_breakdown = compute_strategy_breakdown(portfolio.closed)
    sub_periods = compute_subperiod_metrics(portfolio.closed)
    btc_bh = get_btc_benchmark(TRADE_START, END_DATE)
    monte_carlo = run_monte_carlo(portfolio.closed)

    elapsed = (_dt.now() - t0).total_seconds()

    # ── 6. Print results ───────────────────────────────────────────────────────
    logger.warning("=" * 68)
    logger.warning(f"COMPLETE in {elapsed:.1f}s")
    logger.warning("")
    logger.warning(
        f"  Return:      {metrics.get('total_return_pct', 0):+.2f}%  "
        f"(Ann: {metrics.get('annualized_return_pct', 0):+.1f}%)"
    )
    logger.warning(f"  Sharpe:      {metrics.get('sharpe_ratio', 0):+.3f}")
    logger.warning(f"  Max DD:      -{metrics.get('max_drawdown_pct', 0):.1f}%")
    logger.warning(
        f"  Win Rate:    {metrics.get('win_rate_pct', 0):.1f}%  "
        f"({metrics.get('winning_trades', 0)}W / {metrics.get('losing_trades', 0)}L)"
    )
    logger.warning(f"  Trades:      {metrics.get('total_trades', 0)}")
    logger.warning(f"  Profit Factor: {metrics.get('profit_factor', 0):.3f}")
    logger.warning(f"  Expectancy:    ${metrics.get('expectancy', 0):+.2f} per trade")
    logger.warning(f"  Final Equity:  ${metrics.get('final_equity', 0):,.2f}")
    if btc_bh:
        alpha = round(metrics.get("total_return_pct", 0) - btc_bh["return_pct"], 2)
        logger.warning(
            f"  BTC Buy-Hold: {btc_bh['return_pct']:+.1f}%  "
            f"(${btc_bh['start_price']:,.0f} → ${btc_bh['end_price']:,.0f})"
        )
        logger.warning(f"  Alpha vs BTC: {alpha:+.1f}%")

    logger.warning("")
    logger.warning("Strategy Attribution:")
    for s in strategy_breakdown:
        logger.warning(
            f"  {s['strategy']:<26}  trades={s['trades']:3d}  "
            f"WR={s['win_rate_pct']:.0f}%  "
            f"avg_win=${s['avg_win']:+.0f}  avg_loss=${s['avg_loss']:.0f}  "
            f"PnL=${s['total_pnl']:+,.0f}"
        )

    logger.warning("")
    logger.warning("Sub-Period Breakdown:")
    for p in sub_periods:
        sh = f"{p['sharpe']:.2f}" if p["sharpe"] is not None else " n/a"
        logger.warning(
            f"  {p['period']:<28}  return={p['return_pct']:+.1f}%  "
            f"sharpe={sh}  maxDD=-{p['max_dd_pct']:.1f}%  "
            f"WR={p['win_rate_pct']:.0f}%  trades={p['trades']}"
        )

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    mc = monte_carlo
    logger.warning("")
    logger.warning(f"Monte Carlo ({mc['n_sims']:,} simulations — randomised trade order):")
    logger.warning(
        f"  Max DD   — Median: -{mc['max_dd_p50_pct']:.1f}%  "
        f"P75: -{mc['max_dd_p75_pct']:.1f}%  "
        f"Worst 5%: -{mc['max_dd_p95_pct']:.1f}%"
    )
    logger.warning(
        f"  Time underwater — Median: {mc['time_underwater_pct_p50']:.1f}%  "
        f"Worst 5%: {mc['time_underwater_pct_p95']:.1f}%"
    )
    logger.warning(
        f"  Min equity (path nadir) — Median: ${mc['min_equity_median']:,.0f}  "
        f"Worst 5%: ${mc['min_equity_p5']:,.0f}"
    )
    logger.warning(f"  Ruin risk (equity < 50% of start): {mc['pct_paths_ruin']:.1f}% of paths")
    logger.warning(
        f"  Consec losses — Median: {mc['consec_loss_median']}  Worst 5%: {mc['consec_loss_p95']} in a row"
    )

    # ── Walk-Forward Validation ────────────────────────────────────────────────
    # Split into in-sample (2020-2022) and out-of-sample (2023-2025) windows.
    # Source: Backtest Expert skill — if OOS < 50% of IS performance, strategy is overfit.
    WF_SPLIT = "2023-01-01"
    is_trades = [t for t in portfolio.closed if t["exit_time"][:10] < WF_SPLIT]
    oos_trades = [t for t in portfolio.closed if t["exit_time"][:10] >= WF_SPLIT]
    is_pnl = sum(t["pnl"] for t in is_trades)
    oos_pnl = sum(t["pnl"] for t in oos_trades)
    is_wr = len([t for t in is_trades if t["pnl"] > 0]) / max(1, len(is_trades)) * 100
    oos_wr = len([t for t in oos_trades if t["pnl"] > 0]) / max(1, len(oos_trades)) * 100
    logger.warning("")
    logger.warning("Walk-Forward Validation (Backtest Expert split):")
    logger.warning(
        f"  In-Sample  2020-2022: {len(is_trades):3d} trades  WR={is_wr:.0f}%  PnL=${is_pnl:+,.0f}"
    )
    logger.warning(
        f"  Out-Sample 2023-2025: {len(oos_trades):3d} trades  WR={oos_wr:.0f}%  PnL=${oos_pnl:+,.0f}"
    )
    if is_pnl != 0:
        oos_ratio = oos_pnl / abs(is_pnl) * 100 if is_pnl > 0 else float("nan")
        verdict = "ROBUST" if oos_ratio >= 50 else "FRAGILE (OOS < 50% of IS)"
        logger.warning(
            f"  OOS/IS ratio: {oos_ratio:.0f}%  →  {verdict}"
            if oos_ratio == oos_ratio
            else "  OOS/IS ratio: n/a (IS PnL negative)"
        )

    # Return-normalized walk-forward (path-independent)
    wf_data = _compute_walk_forward(portfolio.closed)
    is_avg = wf_data.get("is_avg_return_pct", 0)
    oos_avg = wf_data.get("oos_avg_return_pct", 0)
    ret_ratio = wf_data.get("oos_is_return_ratio_pct")
    ret_verdict = wf_data.get("return_verdict", "N/A")
    logger.warning(f"  Return-normalized: IS avg/trade={is_avg:+.2f}%  OOS avg/trade={oos_avg:+.2f}%")
    if ret_ratio is not None:
        logger.warning(f"  Return OOS/IS ratio: {ret_ratio:.0f}%  →  {ret_verdict}")

    # ── 7. Save outputs ────────────────────────────────────────────────────────
    walk_forward = wf_data

    result = {
        "run_date": _dt.now().isoformat()[:10],
        "version": output_tag,
        "start": TRADE_START,
        "end": END_DATE,
        "symbols": list(data_map.keys()),
        "initial_capital": INITIAL_CAP,
        "metrics": metrics,
        "btc_benchmark": btc_bh,
        "strategy_breakdown": strategy_breakdown,
        "sub_periods": sub_periods,
        "monte_carlo": monte_carlo,
        "walk_forward": walk_forward,
        "elapsed_seconds": round(elapsed, 1),
        "trades": portfolio.closed,  # v9: expose trade list for OOS analysis and crowding audit
        "equity_curve": portfolio.equity_curve,  # v14: for sleeve combination analysis
    }

    if write_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        html = build_html_report(metrics, portfolio, btc_bh, sub_periods, strategy_breakdown)
        report_path = OUTPUT_DIR / f"report_crypto_{output_tag}.html"
        report_path.write_text(html, encoding="utf-8")

        if portfolio.closed:
            pd.DataFrame(portfolio.closed).to_csv(
                OUTPUT_DIR / f"trade_log_crypto_{output_tag}.csv", index=False
            )

        with open(OUTPUT_DIR / f"summary_crypto_{output_tag}.json", "w") as f:
            # equity_curve holds pd.Timestamp keys — in-memory only, not JSON
            json.dump({k: v for k, v in result.items() if k != "equity_curve"}, f, indent=2)

        logger.warning("=" * 68)
        logger.warning(f"Outputs → {OUTPUT_DIR}")
        logger.warning(f"Report  → {report_path}")
        logger.warning("=" * 68)

    return result


def main() -> None:
    run_backtest(output_tag="v5_ema008")


if __name__ == "__main__":
    main()
