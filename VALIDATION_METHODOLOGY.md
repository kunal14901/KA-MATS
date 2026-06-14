# KA-MATS Crypto — Validation Methodology

**Version:** Phase 1 Honest Baseline (fidelity + statistical hygiene)
**Last validated:** June 2026
**Backtest window:** 2020-01-01 → 2026-01-01 (6 years, daily bars, 22 symbols)

---

## 0. Phase 1 — Backtest Fidelity + Statistical Hygiene (June 2026)

Phase 1 changed nothing about the strategy. It changed how honestly the
strategy is measured: causal indicators, intrabar exit fills, a null-benchmark
comparison, bootstrap confidence intervals, and a deflated Sharpe ratio.
Scripts: `backtest/run_phase1_intrabar.py`, `backtest/run_null_benchmark.py`,
`backtest/run_stat_hygiene.py`, `backtest/_phase1_repro_check.py`.

### 0.1 Canonical numbers (Phase 1 honest baseline — use these externally)

Configuration: 22-coin universe (incl. LUNC/FTT dead coins), tiered slippage,
5-bar re-entry lockout, **intrabar exits with 1h ordering resolution**, fresh
yfinance data (June 2026 download).

| Metric | Value |
|---|---|
| Total return | **+862.4%** |
| Annualized return | **+45.8%** |
| Sharpe ratio | **1.084** |
| Max drawdown | 44.3% |
| Win rate | 48.8% |
| Total trades | 402 |
| Final equity (from $10k) | $96,236 |
| OOS (2023+) expectancy / trade | **+0.322% — 95% CI [−0.198%, +0.904%] — includes zero** |
| Deflated Sharpe (31 trials) | 0.9992 (PASS) — full-sample only |

3-way walk-forward on this baseline (per-trade return normalized):

| Split | IS n / WR / avg | OOS n / WR / avg | OOS PnL |
|---|---|---|---|
| A: IS 2020-21, OOS 2022 | 209 / 47.4% / +1.379% | 51 / 64.7% / +0.248% (bear shorts) | +$5,780 |
| B: IS 2020-22, OOS 2023-24 | 260 / 50.8% / +1.157% | 142 / 45.1% / +0.322% | +$29,002 |
| C: IS 2020-23, OOS 2024-25 | 355 / 49.0% / +0.929% | 47 / 46.8% / +0.351% | +$10,698 |

### 0.2 Finding 1 — The documented numbers do not reproduce on fresh data

Running the exact documented v8 champion configuration (flat 5 bps slippage,
no lockout, no dead coins, legacy close-only exits) on freshly downloaded
yfinance data:

| Metric | Documented (old cache) | Fresh yfinance (June 2026) |
|---|---|---|
| Total return | +4,678% | **+2,269%** |
| Sharpe | 1.593 | **1.330** |
| Max drawdown | −36.4% | **−46.6%** |
| Trades | 313 | **493** |

yfinance is not a stable data source: retroactive revisions, changed listing
dates (UNI-USD now truncates at 2025-04-17), and bar differences alter the
trade sequence enough to change the trade count by 57%. **All Phase 0 numbers
quoted below are artifacts of a specific data snapshot.** Action: pin the
canonical dataset to the committed Binance parquet caches
(`tools/fetch_binance_ohlcv.py`) and never re-quote numbers across data
snapshots. A related consequence on fresh data: 2022 is no longer a zero-trade
year — `CryptoBearShort` (enabled engine default) fired 51 OOS trades in 2022
(WR 64.7%, +$5,780), where the old data produced none.

### 0.3 Finding 2 — Exit fill mechanics (intrabar A/B/C)

Legacy engine evaluated stops/TPs on daily closes only: wicks through the stop
never triggered, and stop exits filled at the close (often far below the
stop). The v12 intrabar engine triggers on high/low, fills at the stop/TP
level (or the open on gaps), assumes stop-first when both are touched in one
daily bar, and resolves that ambiguity with Binance 1h bars when available.
All three runs below share identical data and Phase 0 corrections:

| Metric | A. Legacy (close-only) | B. Intrabar (worst-case) | C. Intrabar + 1h |
|---|---|---|---|
| Total return | +637.2% | +603.9% | **+862.4%** |
| Sharpe | 0.935 | 0.993 | **1.084** |
| Max drawdown | −44.2% | −36.5% | −44.3% |
| Win rate | 47.9% | 50.6% | 48.8% |
| Trades | 384 | 261 | 402 |

Counter-intuitively, honest fills *improve* the result: filling stop exits at
the stop level (instead of the day's close, which on crash days is far below
the stop) and capturing intraday TP touches outweigh the wick-through-stop
losses. **Interpretation for live deployment: place real stop/TP orders on
the exchange.** This is now implemented — see Section 0.7 item 2. The live
executor previously used daily-close local bookkeeping, which replicated the
*worse* legacy mechanics (Sharpe 0.935 vs 1.084).
Note: the `end_of_backtest` exit label includes 35%-drawdown portfolio-stop
close-alls, not only final mark-to-market (pre-existing engine labeling).

### 0.4 Finding 3 — Null benchmarks (does the stack beat a 2-line timer?)

Same fee/slippage model, full window 2020–2026, fresh data:

| Strategy | Return | Sharpe | MaxDD |
|---|---|---|---|
| BTC buy-and-hold | +1,115% | 0.993 | −76.6% |
| BTC golden-cross timer | +421% | 0.842 | −62.2% |
| Equal-weight 20-coin GX timer | +1,420% | 1.005 | −67.0% |
| **KA-MATS (Phase 1 honest)** | **+862%** | **1.084** | **−44.3%** |

The system's edge over the null is **risk-shaping, not return**: it beats the
equal-weight timer on Sharpe (+0.08) and cuts max drawdown by a third, but
returns less in absolute terms. In the OOS window (2023+) BTC buy-and-hold
returned +426% at Sharpe 1.417 — the system did not beat its own beta
out-of-sample. Quote the system as a drawdown-managed crypto exposure, not as
an alpha engine, until live data shows otherwise.

### 0.5 Finding 4 — Bootstrap confidence intervals and deflated Sharpe

From the Phase 1 honest trade log (n=402; OOS 2023+ n=142), 10,000 bootstrap
resamples:

| Window | Expectancy/trade | 95% CI | Win rate | PSR vs 0 |
|---|---|---|---|---|
| Full sample | +0.862% | [+0.303%, +1.504%] | 48.8% [43.8%, 53.7%] | 1.000 |
| OOS 2023+ | +0.322% | **[−0.198%, +0.904%]** | 45.1% [36.6%, 52.8%] | 0.900 |

**The OOS expectancy CI includes zero.** The out-of-sample edge cannot be
statistically distinguished from noise at the 95% level with the available
sample. PSR(OOS) = 0.90 < 0.95. The full-sample result is significant but is
dominated by the 2020–21 in-sample bull regime.

Deflated Sharpe across the 31 documented experiments (v2→v11): expected max
Sharpe of skill-less trials = 0.303 annualized; champion = 1.327; DSR = 0.9992
(PASS). The full-sample result survives selection-bias correction; the OOS
result does not have enough sample to pass on its own.

### 0.6 Look-ahead fix (causal swing pivots)

`compute_indicators` built swing highs/lows with `shift(-1)`/`shift(-2)` —
future bars. The pivots fed only the permanently disabled `CryptoFVGReversal`
strategy, so no published number was contaminated, but the columns were a
loaded gun in a shared function. Pivots are now recorded at their
confirmation bar (2 bars after forming). Verified: zero impact on active
signals.

### 0.7 What Phase 1 changes for deployment

1. **Sizing must be based on the OOS bootstrap CI, not the point estimate.**
   With a CI spanning zero, deployment sizing stays at the Phase 2 ramp floor
   (4.0%) until live trades move the CI off zero.
2. **Place exchange-side stop/TP orders** — worth ~0.15 Sharpe vs daily-close
   local bookkeeping (Finding 2). **IMPLEMENTED (June 2026):** `LiveExecution`
   now places resting protective orders on entry (Binance spot OCO; futures
   reduce-only stop-market + take-profit-market pair), reconciles intrabar
   fills back into local state at the real fill price, cancels/re-places when
   the trailing stop ratchets, and cancels before locally-decided exits
   (max-hold, macro). Poll-based exit checks remain as a backstop for
   positions whose protective placement failed (retried each bar).
3. **Pin the canonical dataset** to committed Binance parquet caches. Any
   number quoted externally must name its data snapshot.
4. The 90-day testnet remains the only true holdout. Nothing in Phase 1
   touched strategy parameters, so the holdout is intact.

### 0.8 Phase 2 — Structural Repair Grid (v13, June 2026)

Five variants run on the honest engine, selected on **OOS walk-forward
avg/trade** (recent-weighted), not full-period return (`run_v13_repair.py`):

| Variant | Return | Sharpe | MaxDD | OOS avg/t (wtd) | OOS PnL |
|---|---|---|---|---|---|
| V0 baseline (Phase 1) | +862% | 1.084 | 44.3% | +0.319% | +$45,479 |
| V1 CB-reset (lockout fix) | +736% | 1.001 | 44.3% | +0.200% | +$35,677 |
| **V2 no-BearShort ← champion** | **+1,237%** | **1.171** | **36.6%** | +0.223% | +$36,230 |
| V3 CB-reset + no-BearShort | +1,027% | 1.066 | 40.4% | +0.137% | +$27,685 |
| V4 = V3 + risk 15% | +778% | 0.984 | 37.6% | +0.055% | -$3,509 |

**Champion: V2 (BearShort disabled, legacy CB behaviour).** Stat hygiene on
the v13 champion: full-sample expectancy +1.217%/t [CI +0.499..+2.082,
excludes zero], DSR 0.9999 (PASS vs 31 trials). OOS (2023+) expectancy
+0.249%/t [CI -0.384..+0.915] — **still includes zero**; OOS PSR 0.78.

Two findings worth more than the champion itself:

1. **The circuit-breaker lockout bug was accidentally protective.** Phase 1
   found the CB outcome queue freezes during a pause, so the stale WR
   re-trips it forever — the system was dormant for most of 2023-25. The
   root-cause fix exists (`V13_CB_RESET_ON_RESUME`), but enabling it makes
   things WORSE OOS (V1 split C: 161 trades, 37.9% WR, -$3,846). The extra
   2024-25 trades the bug was blocking have ~zero expectancy. The fix stays
   in the codebase, **disabled**: the honest conclusion is the system has no
   edge in the 2024-25 regime, and that must be solved with new alpha (e.g.
   cross-sectional breadth), not by unlocking more trades from old alpha.
2. **BearShort death by honest fills.** 56% WR but 0.64 win/loss ratio —
   the sleeve only ever looked viable under close-only exit optimism.
   Live deployment must not enable bear-mode shorts until a sleeve passes
   the intrabar engine on its own.

Risk reduction (27% → 15%, V4) was rejected: Sharpe fell and OOS PnL went
negative — drawdown in this system is regime-driven, not size-driven, and
smaller size just slows compounding (and partially mechanically deflates
avg/trade, which we accounted for; Sharpe falling is the disqualifier).

### 0.9 Phase 3 — Cross-Sectional Momentum Sleeve (June 2026)

The new-alpha door. A standalone weekly-rebalanced XS momentum sleeve
(`run_xs_momentum.py`): 46-coin universe, momentum = LB-day return skipping
the last 7 days, long top-5 inverse-vol weighted, 20 bps/side costs, 1-day
implementation lag. Grid of 6 configs fully disclosed (3 lookbacks × gate
on/off, added to the DSR trial ledger), selected on OOS 2023+ Sharpe.

**Sleeve champion: LB90, ungated.** OOS (2023+) Sharpe **0.987**, ann +55.1%
— it trades and earns in exactly the regime where the time-series system is
dormant. OOS daily-mean 95% CI [-0.022%, +0.394%] — still includes zero, but
narrowly. Sleeve/system daily correlation: **+0.41**.

Blend frontier (daily-rebalanced, v13 system + sleeve):

| Sleeve weight | Return | Sharpe | MaxDD | OOS Sharpe |
|---|---|---|---|---|
| 0% (v13 alone) | +1,237% | 1.171 | 36.6% | 0.386 |
| 10% | +1,450% | 1.231 | 37.3% | 0.564 |
| **20% ← v14 recommended** | **+1,628%** | **1.252** | 43.2% | **0.705** |
| 30% | +1,754% | 1.238 | 49.1% | 0.805 |
| 50% | +1,798% | 1.147 | 61.8% | 0.917 |

**Recommendation: v14 = 80% v13 system + 20% XS sleeve** — the Sharpe-optimal
point (1.252). OOS Sharpe nearly doubles vs the system alone. Caveats stated
plainly: (a) the combined OOS CI still includes zero — only live/testnet time
or more breadth closes that; (b) the 24 added universe coins are selected
from current listings (survivorship; only LUNC/FTT cover delistings);
(c) sleeve drawdown alone is deep (87%) — it must never run unblended.

---

### 0.10 Phase 4 — Three Structural Improvements (June 2026)

Script: `backtest/run_v14_final.py`

Three targeted improvements layered onto the v13+XS baseline:

**1. Maker-order cost model** (`V14_MAKER_ORDERS = True`)  
Entry slippage halved (10 bps → 5 bps BTC/ETH, 10 bps large alts, etc.) to
model limit-order fills resting 0.05-0.10% from mid. Exit costs unchanged
(stops/TPs must fill immediately as market/taker). Result: system alone
improved from Sharpe 1.171 → **1.364**, MaxDD 36.6% → **36.4%**, +292 trades.

**2. Breadth cash filter on XS sleeve** (`V14_BREADTH_FILTER = True`)  
XS sleeve goes to cash whenever < 40% of the eligible universe has positive
90-day momentum. Prevents holding "least-bad" coins in broad bear markets.
Sleeve MaxDD improved 87% → 80% (modest; survivorship and correlations limit
the gain further).

**3. Volatility targeting** (`vol_target()` post-combination)  
Scaled daily exposure to maintain 20% annualised target vol, capped at 1×.
**This did not help for crypto.** Crypto's structural portfolio vol is 60-80%
ann, so the scaler averages ~0.25× — the portfolio is 75% cash most of the
time, crushing absolute returns and Sharpe alike. 20% is an equity target;
an appropriate crypto target would be 50-60%.

**Phase 4 blend frontier (after maker orders, before vol-targeting):**

| Sleeve weight | Return | Sharpe | MaxDD | OOS Sharpe |
|---|---|---|---|---|
| **0% ← champion** | **+1,950%** | **1.643** | **36.4%** | 0.423 |
| 10% | +1,774% | 1.620 | 40.4% | 0.442 |
| 20% | +1,566% | 1.558 | 44.3% | **0.449** |
| 30% | +1,340% | 1.462 | 48.2% | 0.444 |

**Key finding:** With maker orders, the pure system (w=0%) dominates on Sharpe
(1.643). Adding the XS sleeve marginally improves OOS Sharpe (0.423 → 0.449)
at the cost of lower full-period Sharpe and higher MaxDD.

**Recommended final deployment: pure v13 system + maker orders**  
Sharpe **1.643**, MaxDD **36.4%**, Win Rate 57.2%, 292 trades, Return +1,950%.
OOS (2023+) Sharpe 0.423. Vol-targeting shelved — not suitable for 60-80% vol
assets at a 20% target; re-evaluate with 50% target in a future phase.

> **Superseded by Section 0.11 (v15).** The Phase 4 champion was selected on
> full-sample Sharpe (re-weighting toward the 2020-21 bull) and used an
> optimistic blanket-halving maker model. Both issues are fixed in v15.

---

### 0.11 Phase 5 — v15 Grade-A Run (June 2026) ← CANONICAL

Script: `backtest/run_v15_grade_a.py`. Fixes the three findings from the
senior-desk review of Phase 4:

**1. Defensible maker model.** v14 halved entry slippage outright — ignoring
non-fill risk and adverse selection. v15 models a partial-fill blend:
65% of entries fill as maker (0.5× cost), 35% time out and pay taker (1.0×).
Expected entry multiplier **0.675×** (`V15_MAKER_COST_MULT`). The live
executor (`agents/live_execution.py`) now actually places limit entries with
a 90s timeout → market fallback, and logs every fill type to
`logs/fill_stats.jsonl` — the paper phase **measures** the 65% assumption.

**2. OOS-first champion selection.** Phase 4 maximised full-sample Sharpe,
which re-weighted toward 2020-21. v15 selects on **OOS (2023+) Sharpe**
subject to guardrails: full Sharpe ≥ 1.20 and MaxDD ≤ 40%. All 24 grid
configs (6 sleeve weights × 4 vol targets) disclosed and added to the DSR
trial ledger.

**3. Crypto-appropriate vol targeting.** Grid {none, 45%, 55%, 65%} with
scale capped at 1.0× (no leverage), 20d lookback, 1-day lag.

**Champion: pure system + 45% vol target** (sleeve weight 0%):

| Metric | No targeting | **v15 champion (45% target)** |
|---|---|---|
| Total return | +1,494% | +595% |
| Sharpe (full) | 1.445 | **1.389** |
| Max drawdown | 41.3% | **29.9%** |
| OOS (2023+) Sharpe | 0.886 | **0.961** |
| OOS max drawdown | 25.9% | **22.6%** |

Walk-forward (trade-level): OOS/IS PnL ratio **62% → ROBUST** (Phase 4: 19%
FRAGILE). Return-normalized per-trade ratio is still 19% — the per-trade
edge decays OOS even though total PnL holds up on higher OOS trade count.

OOS bootstrap (champion daily returns, 2023+): mean +0.048%/day
(≈ +17.7%/yr), 95% CI [−0.020%, +0.118%] — **still includes zero**, but the
interval is now centred well above it. Only live sample size closes this.

The XS momentum sleeve (with breadth filter, v15 costs) degraded to OOS
Sharpe 0.326 standalone and reduced champion OOS Sharpe at every weight —
it stays in the research book, NOT in deployment.

**Deployment config synced** (`config/settings.py`): `vol_target_enabled=True`,
`vol_target_annual_pct=0.45`, `vol_scale_max=1.0`, maker entries ON in
`LiveExecution` with fill-type logging.

---

## 1. Phase 0 Canonical Numbers (SUPERSEDED — kept for history)

> **Superseded by Section 0.** These numbers came from a data snapshot that
> no longer reproduces (Section 0.2) and a close-only exit engine
> (Section 0.3). Do not quote them externally.

| Metric | Value |
|---|---|
| Total return | **+3,634.6%** |
| Annualized return | **82.7%** |
| Sharpe ratio | **1.507** |
| Max drawdown | 45.8% |
| Win rate | 51.2% |
| Profit factor | 1.587 |
| Expectancy / trade | $870 |
| Total trades | 402 |
| Final equity (from $10k) | $373,455 |
| OOS edge / trade | 0.388% |
| OOS Half-Kelly | 5.95% → **deployed at 5.5%** |

These were the numbers quoted to investors, used for live sizing, and put on pitch materials. All earlier internal numbers (v4, v5, v6…) are research artifacts, not performance claims.

---

## 2. The v5 → Phase 0 Haircut

| Metric | v5 (inflated) | Phase 0 (honest) | Δ |
|---|---|---|---|
| Total return | +4,678.4% | +3,634.6% | −1,043.8 pp |
| Sharpe | 1.593 | 1.507 | −0.086 |
| Win rate | 53.0% | 51.2% | −1.8 pp |
| Expectancy | $1,468 | $870 | −$598 |
| Trades | 313 | 402 | +89 |
| Final equity | $477,844 | $373,455 | −$104,389 |

**The $104k gap is not performance lost — it is performance that was never real.** Attribution of the +89 trade delta and the equity gap:

| Source | Trades | PnL impact | Nature |
|---|---|---|---|
| Extended data window (Jun 2025 → Oct 2025) | +45 | +$34k | New observations |
| Dead coins (LUNC, FTT) added to universe | +14 | +$64k | Survivorship correction |
| Tiered slippage (5/10/20 bps by tier) | 0 | −$80k est. | Honest execution cost |
| 5-bar re-entry lockout | −12 net | −$30k est. | Whipsaw protection |
| 2025-H1/H2 open positions (mark-to-market at `end_of_backtest`) | +43 | −$22k | Reporting artifact |
| Slippage-driven timestamp drift on shared trades | ±120 | ±$10k | Reshuffle, not new alpha |

The haircut is dominated by survivorship-bias correction and tiered slippage — both one-directional adjustments that Phase 0 was explicitly designed to reveal.

---

## 3. Out-of-Sample Validation

**3-split non-overlapping walk-forward:**

| Split | IS window | OOS window | OOS n | OOS WR | OOS avg/trade | Verdict |
|---|---|---|---|---|---|---|
| A | 2020–2021 | 2022 (bear) | 0 | — | — | NO TRADES (macro filter held) |
| B | 2020–2022 | 2023–2024 | 149 | 47.0% | +0.47% | PASS |
| C | 2020–2023 | 2024–2025 | 74 | 47.3% | +0.24% | PASS |

**OOS aggregate:** 159 trades, 46.0% WR, avg win $1,901, avg loss $1,224, W/L 1.553.

**Kill criteria both passed:**
- Sharpe 1.507 > 0.9 threshold ✅
- OOS edge 0.388% > 0.2% threshold ✅

### 3.1 Two OOS numbers exist — use the conservative one externally

| Split method | OOS trades | OOS WR | OOS avg/trade | Source |
|---|---|---|---|---|
| Single split @ 2023-01-01 | 245 | 42.9% | **0.16%** | `summary_crypto_phase0_full.json` walk_forward |
| 3-split non-overlapping (A/B/C) | 159 | 46.0% | **0.388%** | Section 3 above |

The single-split number (0.16%) is **more pessimistic** because it includes 2023-H1 chop and 2025-H1 Tariff Shock in one unsegmented block. The 3-split number is more granular but also gentler on bad periods. **For external framing and sizing decisions, quote 0.16%.** Internal sizing was calibrated off the 3-split Kelly (Half-Kelly = 5.95%, deployed at 5.5%), which remains conservative even against the 0.16% number (0.16% × OOS WR 42.9% implies half-Kelly ≈ 3–4%, so 5.5% is at upper bound — this is why Phase 2 starts at 4% and ramps).

---

## 4. Kelly Calibration

$$\text{Kelly} = \text{WR} - \frac{1 - \text{WR}}{W/L} = 0.46 - \frac{0.54}{1.585} = 0.119$$

$$\text{Half-Kelly} = 5.95\%$$

**Deployed at 5.5%** — deliberately below Half-Kelly to account for:
- Single OOS sample (not infinite-horizon assumption behind Kelly formula)
- Unknown parameter drift in live regimes not seen in backtest
- Asymmetric cost of over-sizing (permanent impairment) vs. under-sizing (opportunity cost)

**Phase 2 ramp schedule:** 4.0% initial → 5.5% after 50 clean live trades.

---

## 5. Rejected Hypotheses (tested, failed walk-forward stability)

Recorded here so future maintainers don't retry them blindly.

### 5.1 Regime-aware re-entry lockout
- **Hypothesis:** Flat 5-bar lockout compresses avg-win in strong uptrends (observed 2024-Q4: $6,326 → $4,593 on same WR). Proposed `{trending_up: 2, ranging: 5, trending_down: 8}`.
- **Result:** Sharpe 1.507 → 1.409, expectancy $870 → $583, equity $373k → $267k. Shorter uptrend lockout re-admitted marginal setups that the flat 5-bar was correctly blocking.
- **Lesson:** The avg-win compression is from tiered slippage on mid-cap high-vol breakouts, not the lockout. Fix belongs at the slippage-tier level, not here. Phase 0's premise is to accept those costs.
- **Infrastructure retained** (`V9_REENTRY_COOLDOWN_BY_REGIME` dict, empty by default) for future experiments without code changes.

### 5.2 BTC 20-day shock filter
- **Hypothesis:** Block TrendPullback entries when BTC 20-day return < −10% to avoid 2025-H1 Tariff Shock losses.
- **Result:** Filter also blocked the 2023-H2 Recovery entries that produced +$130k. Net negative.
- **Verdict:** Macro shocks are not cleanly separable from recoveries with this feature.

### 5.3 Concentration cap (v10 `BULL_POS_CAP`)
- **Hypothesis:** Cap concurrent bull positions at 5–7 to reduce correlated alt-coin clustering.
- **Result:** Truncated the strongest periods (2021-Q1, 2023-Q4, 2024-Q1) where 8-9 concurrent positions were the actual alpha. PnL dropped meaningfully without a Sharpe improvement.
- **Verdict:** Correlation risk is real but already managed by macro-regime gating; adding a second cap double-counts.

**The system is near-optimal within its strategy family. Further gains require changing the family (multi-timeframe, alt-universe rotation, different strategy class) rather than tuning the existing one.**

---

## 6. What's Frozen Before Phase 1 Testnet

- Macro filter (BTC vs EMA200, ±5% uncertainty zone)
- Golden cross gate (BTC 50/200 SMA)
- Tiered slippage table (5/10/20/25/30 bps)
- 5-bar flat re-entry lockout
- Dead-coin cutoffs (LUNC 2022-05-12, FTT 2022-11-09)
- Half-Kelly sizing infrastructure
- 22-symbol universe

## 7. What Changes For Live Deployment

- `risk_per_trade_pct`: 15% (backtest default) → **5.5%** (OOS-calibrated)
- Testnet runs at **5.5%**, not 15% — validation must match production config
- Phase 2 (₹25–40k micro-capital): ramp starts at **4.0%**, graduates to 5.5% after 50 clean trades

---

## 8. Headline Framing (memorized pitch)

> "KA-MATS is a 9-agent crypto momentum system, daily bars, 20-coin universe. Validated on 6 years 2020–2026 with a rigorous walk-forward split. Honest numbers after survivorship-bias correction, tiered slippage, and re-entry lockout: 3,635% total return, Sharpe 1.51, OOS edge of 0.39% per trade across two independent test windows, max drawdown 46%. Kelly-calibrated to 5.5% risk per trade for live deployment. Two proposed improvements were tested and rigorously rejected for failing walk-forward stability — current system is near-optimal within its strategy family."

## 9. Reviewer-grade framing (use this when depth matters)

> "KA-MATS is a trend-convex crypto momentum system with 402 trades across a 6-year backtest (2020–2026). Full-sample Sharpe is 1.51 and total return is 3,635%, but that top-line number is dominated by 2020–2021 regime fit — **82% of profit comes from 157 of the 402 trades (2020–2022 in-sample window)**. The honest metric is out-of-sample (2023–2026): expectancy compresses to 0.16% per trade with win-rate 42.9% on the single-split walk-forward, still positive but fragile. Position sizing is therefore calibrated off OOS Kelly rather than IS (Half-Kelly ≈ 5.95% on the 3-split test, **deployed at 5.5% with a Phase 2 ramp starting at 4.0%**). Monte Carlo across 1,000 paths shows **zero probability of ruin** and a 95th-percentile drawdown of 34.4%. The system deliberately stayed flat in 2022 (macro-regime filter working as designed) but struggles in sideways chop regimes (2023-H1, 2025-H1). A **90-day Binance-testnet validation is currently in progress** before any real-capital deployment; graduation to live is contingent on OOS expectancy holding positive through this fresh out-of-sample window. Known weakness: sideways-regime detection is incomplete. Planned work: post-testnet, add volatility-regime gating, validated on a NEW 2026–2027 OOS window rather than re-fit on existing data."

## 10. Known weaknesses (disclose upfront)

| Weakness | Evidence | Mitigation (current) | Mitigation (planned, post-testnet) |
|---|---|---|---|
| Chop-regime vulnerability | 2023-H1 (-19.5%, WR 21%), 2025-H1 (-15.4%, WR 20%) | Macro filter + re-entry lockout | Volatility-regime gate (ADX/Hurst/slope stability) — **only after 90-day testnet**, validated on fresh 2026–27 OOS window |
| IS/OOS edge compression | 3.04% → 0.16% per trade | Sized off OOS, not IS (5.5% risk vs 11% full-Kelly) | Per-strategy OOS recalibration at Day 90 checkpoint |
| Bull-era dependence | 82% of profit from 2020–21 | Accept as structural — trend systems rely on trend regimes | Multi-strategy allocation (momentum + mean-reversion) — research, not Phase 1 |
| Small-N period Sharpes are noisy | 2023-H1 Sharpe = -11.5 on n=24 trades | Quote year-level and full-sample Sharpe, not monthly | N/A — statistical reality |

## 11. What NOT to do during the 90-day testnet

1. ❌ Do not change code, parameters, symbol list, or risk sizing
2. ❌ Do not add regime filters because a reviewer critiqued chop vulnerability — that's textbook overfitting-to-critique
3. ❌ Do not lead conversations with "+3,634%" — lead with OOS 0.16%/trade and honest disclosure
4. ❌ Do not call the system "production ready" — call it "in 90-day live validation"
5. ❌ Do not restart the run, modify `.env`, or tinker with the server without a documented reason

**The 90-day testnet is a true holdout. Tampering with it voids the validation.**

---

*This document is the canonical reference for KA-MATS validation. Any modification that beats Phase 0 must reproduce these walk-forward results plus demonstrate OOS improvement on an additional held-out window before it becomes the new baseline.*
