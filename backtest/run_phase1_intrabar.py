"""
backtest/run_phase1_intrabar.py
===============================
Phase 1 — Backtest Fidelity Re-Baseline (intrabar exit engine A/B).

Legacy engine behaviour evaluated stops/TPs on daily CLOSES only:
  * a wick through the stop that recovered by the close never stopped out
  * stop exits filled at the close, not at the stop level
Real exchange stop orders fire intrabar. This script quantifies the gap.

Runs (all with Phase 0 corrections: tiered slippage, 5-bar lockout, dead
coins when the expanded cache is available):

  RUN A — LEGACY     close-only exits (reproduces the Phase 0 baseline)
  RUN B — INTRABAR   high/low triggers, stop-price fills, gap handling,
                     worst-case ordering when stop+TP share a bar
  RUN C — INTRABAR+1H same as B, but same-bar stop/TP ambiguity resolved
                     by walking that day's 1h Binance bars

Then: comparison table, exit-reason attribution, and 3-way walk-forward
on the honest run. RUN C (or B if no 1h cache) becomes the new canonical
"Phase 1 honest baseline".

Usage:
    python -m backtest.run_phase1_intrabar
"""

from __future__ import annotations

import json
import sys
from datetime import datetime as _dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

import backtest.run_crypto_backtest as bt

OUTPUT_DIR = ROOT / "results" / "crypto_backtest"

DEAD_COIN_CUTOFFS = {
    "LUNC-USD": "2022-05-12",
    "FTT-USD": "2022-11-09",
}
DEAD_CACHE = OUTPUT_DIR / "ohlcv_daily_2019_2026_v9_dead.parquet"

METRIC_KEYS = [
    ("Total Return %", "total_return_pct", ".1f"),
    ("Annualized %", "annualized_return_pct", ".1f"),
    ("Sharpe Ratio", "sharpe_ratio", ".3f"),
    ("Max Drawdown %", "max_drawdown_pct", ".1f"),
    ("Win Rate %", "win_rate_pct", ".1f"),
    ("Total Trades", "total_trades", ".0f"),
    ("Profit Factor", "profit_factor", ".3f"),
    ("Expectancy $/t", "expectancy", ".2f"),
    ("Final Equity $", "final_equity", ",.0f"),
]


def _banner(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def _phase0_config() -> None:
    """Phase 0 honest-baseline configuration (v8 champion + corrections)."""
    bt.V6_GLOBAL_SIZING_MULT = 0.92
    bt.V8_GOLDEN_CROSS_ENABLED = True
    bt.V8_CIRCUIT_BREAKER_ENABLED = True
    bt.V8_ALTSEASON_GATE_ENABLED = False
    bt.V9_TIERED_SLIPPAGE_ENABLED = True
    bt.V9_REENTRY_COOLDOWN_BARS = 5
    bt.V9_BTC_20D_SHOCK_THRESHOLD = 0.0
    bt.V9_BTC_20D_SHOCK_MULT = 0.0
    bt.V10_BULL_POS_CAP = 0
    bt.V10_MANIA_ONLY = False


_dead_universe_ready: bool | None = None


def _try_enable_dead_coins() -> bool:
    """Point the engine at the expanded (dead-coin) universe if data can be
    sourced. Falls back to the standard 20-coin universe with a warning —
    the A/B comparison stays internally consistent either way."""
    global _dead_universe_ready
    if _dead_universe_ready is None:
        if not DEAD_CACHE.exists():
            # Build the expanded cache via yfinance once
            orig_syms, orig_cache = bt.CRYPTO_SYMBOLS[:], bt.CACHE_FILE
            bt.CRYPTO_SYMBOLS = orig_syms + [s for s in DEAD_COIN_CUTOFFS if s not in orig_syms]
            bt.CACHE_FILE = DEAD_CACHE
            try:
                frames = bt.load_data()
                _dead_universe_ready = all(s in frames for s in DEAD_COIN_CUTOFFS)
            except Exception as e:
                logger.warning(f"Dead-coin cache build failed: {e}")
                _dead_universe_ready = False
            finally:
                bt.CRYPTO_SYMBOLS, bt.CACHE_FILE = orig_syms, orig_cache
            if not _dead_universe_ready and DEAD_CACHE.exists():
                # partial cache would trigger delete/re-download loops — remove
                DEAD_CACHE.unlink(missing_ok=True)
        else:
            _dead_universe_ready = True
        if not _dead_universe_ready:
            logger.warning(
                "Dead coins (LUNC/FTT) unavailable — running 20-coin universe. "
                "Numbers will be slightly optimistic vs the Phase 0 22-coin baseline."
            )

    if _dead_universe_ready:
        bt.CRYPTO_SYMBOLS = bt.CRYPTO_SYMBOLS + [s for s in DEAD_COIN_CUTOFFS if s not in bt.CRYPTO_SYMBOLS]
        bt.CACHE_FILE = DEAD_CACHE
        bt.DEAD_COIN_CUTOFFS = DEAD_COIN_CUTOFFS.copy()
    return bool(_dead_universe_ready)


def _run(label: str, tag: str, intrabar: bool, hourly: bool, write_outputs: bool = False) -> dict:
    _banner(f"RUN — {label}")
    orig_syms, orig_cache = bt.CRYPTO_SYMBOLS[:], bt.CACHE_FILE
    _phase0_config()
    _try_enable_dead_coins()
    bt.V12_INTRABAR_EXITS = intrabar
    bt.V12_HOURLY_RESOLUTION = hourly
    try:
        result = bt.run_backtest(output_tag=tag, write_outputs=write_outputs)
    finally:
        bt.CRYPTO_SYMBOLS, bt.CACHE_FILE = orig_syms, orig_cache
        bt.DEAD_COIN_CUTOFFS = {}
        bt.V12_INTRABAR_EXITS, bt.V12_HOURLY_RESOLUTION = True, True
    m = result["metrics"]
    print(f"\n  {label}:")
    print(
        f"    Return: {m.get('total_return_pct', 0):+.1f}%   "
        f"Sharpe: {m.get('sharpe_ratio', 0):.3f}   "
        f"MaxDD: -{m.get('max_drawdown_pct', 0):.1f}%   "
        f"WR: {m.get('win_rate_pct', 0):.1f}%   "
        f"Trades: {m.get('total_trades', 0)}"
    )
    return result


def _exit_reason_table(runs: list[tuple[str, dict]]) -> dict:
    out = {}
    print(f"\n  {'Exit reason':<22}", end="")
    for name, _ in runs:
        print(f"  {name:>18}", end="")
    print()
    reasons = sorted({t["exit_reason"] for _, r in runs for t in r.get("trades", [])})
    for reason in reasons:
        print(f"  {reason:<22}", end="")
        for name, r in runs:
            trades = [t for t in r.get("trades", []) if t["exit_reason"] == reason]
            n = len(trades)
            wr = (sum(1 for t in trades if t["pnl"] > 0) / n * 100) if n else 0.0
            print(f"  {n:>7d} ({wr:>4.0f}%WR)", end="")
            out.setdefault(reason, {})[name] = {"n": n, "wr_pct": round(wr, 1)}
        print()
    return out


def _wf_3split(trades: list[dict]) -> list[dict]:
    splits_def = [
        ("Split A: IS 2020-2021 | OOS 2022", "2022-01-01", "2022-01-01", "2023-01-01"),
        ("Split B: IS 2020-2022 | OOS 2023-24", "2023-01-01", "2023-01-01", "2025-01-01"),
        ("Split C: IS 2020-2023 | OOS 2024-25", "2024-01-01", "2024-01-01", "2026-01-01"),
    ]
    results = []
    for label, is_end, oos_start, oos_end in splits_def:
        is_t = [t for t in trades if pd.Timestamp(t["exit_time"]) < pd.Timestamp(is_end)]
        oos_t = [
            t
            for t in trades
            if pd.Timestamp(oos_start) <= pd.Timestamp(t["exit_time"]) < pd.Timestamp(oos_end)
        ]

        def _stats(ts):
            if not ts:
                return {"n": 0, "wr": 0.0, "pnl": 0.0, "avg_ret": 0.0}
            rets = [t["pnl"] / t["equity_at_entry"] for t in ts if t.get("equity_at_entry", 0) > 0]
            return {
                "n": len(ts),
                "wr": sum(1 for t in ts if t["pnl"] > 0) / len(ts),
                "pnl": sum(t["pnl"] for t in ts),
                "avg_ret": float(np.mean(rets)) if rets else 0.0,
            }

        s_is, s_oos = _stats(is_t), _stats(oos_t)
        print(f"\n  {label}")
        print(
            f"    IS:  n={s_is['n']:3d}  WR={s_is['wr']:.1%}  "
            f"PnL=${s_is['pnl']:>+10,.0f}  avg/trade={s_is['avg_ret']:+.3%}"
        )
        if s_oos["n"] == 0:
            print("    OOS: NO TRADES (macro/GX filter held)")
        else:
            print(
                f"    OOS: n={s_oos['n']:3d}  WR={s_oos['wr']:.1%}  "
                f"PnL=${s_oos['pnl']:>+10,.0f}  avg/trade={s_oos['avg_ret']:+.3%}"
            )
        results.append({"label": label, "is": s_is, "oos": s_oos})
    return results


def main() -> None:
    t0 = _dt.now()
    _banner("PHASE 1 — INTRABAR EXIT FIDELITY RE-BASELINE")

    hourly_available = bt.HOURLY_CACHE_FILE.exists()
    if not hourly_available:
        logger.warning(
            f"1h cache not found ({bt.HOURLY_CACHE_FILE.name}) — "
            "RUN C skipped; worst-case ordering (RUN B) is the honest baseline. "
            "Build it with: python tools/fetch_binance_ohlcv.py --timeframe 1h"
        )

    run_a = _run("A. LEGACY (close-only exits)", "phase1_legacy", intrabar=False, hourly=False)
    run_b = _run("B. INTRABAR (worst-case ordering)", "phase1_intrabar_wc", intrabar=True, hourly=False)
    run_c = None
    if hourly_available:
        run_c = _run(
            "C. INTRABAR + 1h resolution",
            "phase1_intrabar_1h",
            intrabar=True,
            hourly=True,
            write_outputs=True,
        )

    honest = run_c or run_b
    honest_name = "C. INTRABAR+1h" if run_c else "B. INTRABAR (worst-case)"

    # ── Comparison table ───────────────────────────────────────────────────────
    _banner("COMPARISON — Legacy vs Intrabar")
    runs = [("A. LEGACY", run_a), ("B. INTRABAR-WC", run_b)]
    if run_c:
        runs.append(("C. INTRABAR-1H", run_c))

    print(f"  {'Metric':<18}", end="")
    for name, _ in runs:
        print(f"  {name:>16}", end="")
    print(f"  {'Delta (honest-A)':>18}")
    for label, key, fmt in METRIC_KEYS:
        print(f"  {label:<18}", end="")
        vals = []
        for _, r in runs:
            v = r["metrics"].get(key, 0) or 0
            vals.append(v)
            print(f"  {format(v, fmt):>16}", end="")
        delta = (honest["metrics"].get(key, 0) or 0) - vals[0]
        print(f"  {format(delta, '+' + fmt):>18}")

    # ── Exit reason attribution ────────────────────────────────────────────────
    _banner("EXIT REASON ATTRIBUTION (where the wicks bite)")
    exit_table = _exit_reason_table(runs)

    # ── 3-way walk-forward on the honest run ───────────────────────────────────
    _banner(f"3-WAY WALK-FORWARD — {honest_name}")
    wf = _wf_3split(honest.get("trades", []))

    # ── Save ───────────────────────────────────────────────────────────────────
    elapsed = (_dt.now() - t0).total_seconds()
    summary = {
        "run_date": _dt.now().isoformat()[:19],
        "elapsed_seconds": round(elapsed, 1),
        "hourly_resolution_used": bool(run_c),
        "honest_baseline": honest_name,
        "runs": {name: {k: r["metrics"].get(k) for _, k, _f in METRIC_KEYS} for name, r in runs},
        "exit_reasons": exit_table,
        "walk_forward_3split": [
            {
                "label": s["label"],
                "is_n": s["is"]["n"],
                "is_wr": round(s["is"]["wr"], 3),
                "is_avg_ret": round(s["is"]["avg_ret"], 5),
                "oos_n": s["oos"]["n"],
                "oos_wr": round(s["oos"]["wr"], 3),
                "oos_pnl": round(s["oos"]["pnl"], 2),
                "oos_avg_ret": round(s["oos"]["avg_ret"], 5),
            }
            for s in wf
        ],
    }

    class _NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    out_path = OUTPUT_DIR / "phase1_intrabar_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_NpEnc)

    _banner("PHASE 1 RE-BASELINE COMPLETE")
    a_m, h_m = run_a["metrics"], honest["metrics"]
    print(f"""
  Elapsed: {elapsed:.0f}s

  LEGACY (close-only) numbers:
    Return: {a_m.get("total_return_pct", 0):+.1f}%  |  Sharpe: {a_m.get("sharpe_ratio", 0):.3f}  |  WR: {a_m.get("win_rate_pct", 0):.1f}%  |  Trades: {a_m.get("total_trades", 0)}

  HONEST ({honest_name}) numbers:
    Return: {h_m.get("total_return_pct", 0):+.1f}%  |  Sharpe: {h_m.get("sharpe_ratio", 0):.3f}  |  WR: {h_m.get("win_rate_pct", 0):.1f}%  |  Trades: {h_m.get("total_trades", 0)}

  The gap between the two is fill-mechanics optimism that live trading
  will not give back. The honest number is the new canonical baseline.

  Summary saved: {out_path}
""")


if __name__ == "__main__":
    main()
