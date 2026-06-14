"""
backtest/run_stat_hygiene.py
============================
Phase 1 statistical hygiene for the KA-MATS backtest.

Three questions a reviewer will ask, answered with real statistics instead
of point estimates:

1. BOOTSTRAP CONFIDENCE INTERVALS
   The OOS edge is quoted as a point estimate (e.g. +0.16%/trade). With only
   a few hundred trades, the sampling error is large. We bootstrap (10,000
   resamples with replacement) the per-trade return and win rate, full-sample
   and OOS (2023+), and report 95% CIs. If the OOS CI includes zero, the edge
   is not statistically distinguishable from noise.

2. PROBABILISTIC SHARPE RATIO (PSR, Bailey & Lopez de Prado 2012)
   P(true Sharpe > 0) given the observed Sharpe, sample size, skew and
   kurtosis of returns. Fat-tailed, skewed crypto returns inflate the
   variance of the Sharpe estimator — PSR corrects for this.

3. DEFLATED SHARPE RATIO (DSR, Bailey & Lopez de Prado 2014)
   The v2->v11 research history ran ~25+ documented experiments against the
   same 6-year window. Under the null of zero true skill, the BEST of N
   trials has a positive expected maximum Sharpe (selection bias). DSR is
   the PSR measured against that expected-max threshold: P(true Sharpe of
   the selected champion > expected max Sharpe of N skill-less trials).

Usage:
    python -m backtest.run_stat_hygiene                       # newest trade log
    python -m backtest.run_stat_hygiene --trade-log results/crypto_backtest/trade_log_crypto_<tag>.csv

Output: console report + results/crypto_backtest/stat_hygiene.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

OUTPUT_DIR = ROOT / "results" / "crypto_backtest"

# ── Documented experiment history (trial set for DSR) ─────────────────────────
# Annualized Sharpes of every distinct configuration recorded in
# Full_Explanation_KA-MATS_Crypto.md sections 14-16 (v2 -> v11 + sensitivity
# scans). These are the trials from which the champion was SELECTED — the
# multiple-comparison set that DSR must deflate against.
DOCUMENTED_TRIAL_SHARPES = [
    1.540,  # baseline (2 strategies + macro filter)
    1.535,  # v2 verify
    1.531,  # v5 (EMA alpha 0.08)
    1.479,  # alpha 0.05
    1.532,  # alpha 0.10
    1.532,  # alpha 0.12
    1.562,  # learner OFF
    1.582,  # v6 stage 1 (onchain sizing)
    1.547,  # v6 stage 2
    1.533,  # v6 stage 3+
    1.249,  # v7a regime score
    1.522,  # v7b rank sizing
    1.284,  # v7c squeeze gate
    1.288,  # v7 full
    1.569,  # v8a golden cross
    1.139,  # v8b altseason gate
    1.573,  # v8c circuit breaker
    1.593,  # v8a+c champion
    1.301,  # v9a tiered slippage
    1.218,  # v9b re-entry cooldown
    1.581,  # v9c quarter-kelly
    1.469,  # v9d dead coins
    1.235,  # v9e slip+cooldown
    1.507,  # v9 full (Phase 0 honest baseline)
    1.662,  # v10 shock -8%
    1.394,  # v10 shock -10%
    1.588,  # v10 shock -12%
    1.346,  # v11 cap=4
    1.434,  # v11 cap=5
    1.199,  # v11 cap=6
    1.308,  # v11 cap=7
]

EULER_GAMMA = 0.5772156649015329


# ── Math helpers ───────────────────────────────────────────────────────────────


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |err| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def probabilistic_sharpe(sr_hat: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """PSR = P(true SR > sr_benchmark | observed sr_hat over n periods).
    sr_hat / sr_benchmark are PER-PERIOD (not annualized). kurt is
    non-excess kurtosis (normal = 3)."""
    if n <= 1:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2))
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / denom
    return _norm_cdf(z)


def expected_max_sharpe(trial_sharpes: list[float]) -> float:
    """Expected maximum Sharpe of N skill-less trials (Bailey & LdP 2014):
        E[max SR] = sqrt(V[SR_trials]) * ((1-g)*Z^-1(1-1/N) + g*Z^-1(1-1/(N*e)))
    where the variance is estimated from the empirical spread of the trials."""
    n_trials = len(trial_sharpes)
    var_trials = float(np.var(trial_sharpes, ddof=1))
    return math.sqrt(var_trials) * (
        (1 - EULER_GAMMA) * _norm_ppf(1 - 1 / n_trials) + EULER_GAMMA * _norm_ppf(1 - 1 / (n_trials * math.e))
    )


# ── Bootstrap ──────────────────────────────────────────────────────────────────


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, stat=np.mean, seed: int = 42) -> dict:
    """Percentile bootstrap 95% CI for a statistic of `values`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = stat(values[idx], axis=1)
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return {
        "point": round(float(stat(values)), 6),
        "ci95_low": round(float(lo), 6),
        "ci95_high": round(float(hi), 6),
        "n": n,
        "includes_zero": bool(lo <= 0.0 <= hi),
    }


# ── Trade-log analysis ─────────────────────────────────────────────────────────


def load_trade_log(path: Path | None) -> pd.DataFrame:
    if path is None:
        candidates = sorted(
            OUTPUT_DIR.glob("trade_log_crypto_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No trade_log_crypto_*.csv in {OUTPUT_DIR} — run the backtest first")
        path = candidates[0]
    logger.info(f"Trade log: {path}")
    df = pd.read_csv(path)
    eq = df["equity_at_entry"].replace(0, np.nan).fillna(df["entry_cost"])
    df["trade_return"] = df["pnl"] / eq
    return df


def analyse_window(df: pd.DataFrame, label: str, n_boot: int) -> dict:
    rets = df["trade_return"].to_numpy(dtype=float)
    wins = (df["pnl"] > 0).to_numpy(dtype=float)
    if len(rets) < 10:
        logger.warning(f"{label}: only {len(rets)} trades — skipping")
        return {"label": label, "n_trades": len(rets), "insufficient": True}

    exp_ci = bootstrap_ci(rets, n_boot=n_boot)
    wr_ci = bootstrap_ci(wins, n_boot=n_boot)

    # Per-trade Sharpe + PSR (annualized using observed trade frequency)
    sr_trade = float(np.mean(rets) / np.std(rets, ddof=0)) if np.std(rets) > 0 else 0.0
    days = (pd.to_datetime(df["exit_time"]).max() - pd.to_datetime(df["entry_time"]).min()).days or 1
    trades_per_year = len(rets) * 365.25 / days
    skew = float(pd.Series(rets).skew())
    kurt = float(pd.Series(rets).kurt()) + 3.0  # pandas gives excess kurtosis
    psr0 = probabilistic_sharpe(sr_trade, 0.0, len(rets), skew, kurt)

    return {
        "label": label,
        "n_trades": len(rets),
        "expectancy_per_trade": exp_ci,
        "win_rate": wr_ci,
        "per_trade_sharpe": round(sr_trade, 4),
        "annualized_sharpe_est": round(sr_trade * math.sqrt(trades_per_year), 3),
        "skew": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "psr_vs_zero": round(psr0, 4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def run(trade_log: Path | None = None, oos_start: str = "2023-01-01", n_boot: int = 10_000) -> dict:
    df = load_trade_log(trade_log)
    oos = df[df["exit_time"] >= oos_start]

    results = {
        "full_sample": analyse_window(df, "Full sample", n_boot),
        "oos": analyse_window(oos, f"OOS ({oos_start}+)", n_boot),
    }

    # ── Deflated Sharpe of the champion ────────────────────────────────────────
    full = results["full_sample"]
    champion_sr_ann = full.get("annualized_sharpe_est", 0.0)
    exp_max = expected_max_sharpe(DOCUMENTED_TRIAL_SHARPES)
    # Deflate at TRADE frequency: convert annualized expected-max to per-trade
    rets = df["trade_return"].to_numpy(dtype=float)
    days = (pd.to_datetime(df["exit_time"]).max() - pd.to_datetime(df["entry_time"]).min()).days or 1
    trades_per_year = len(rets) * 365.25 / days
    sr_trade = full.get("per_trade_sharpe", 0.0)
    exp_max_trade = exp_max / math.sqrt(trades_per_year)
    dsr = probabilistic_sharpe(
        sr_trade,
        exp_max_trade,
        len(rets),
        full.get("skew", 0.0),
        full.get("kurtosis", 3.0),
    )
    results["deflated_sharpe"] = {
        "n_trials": len(DOCUMENTED_TRIAL_SHARPES),
        "expected_max_sharpe_annualized": round(exp_max, 3),
        "champion_sharpe_annualized": round(champion_sr_ann, 3),
        "dsr": round(dsr, 4),
        "interpretation": (
            "DSR is P(true Sharpe > expected max of N skill-less trials). "
            "Below 0.95 the champion's edge cannot be distinguished from "
            "selection bias across the documented experiment history."
        ),
    }

    # ── Print report ───────────────────────────────────────────────────────────
    logger.warning("")
    logger.warning("=" * 72)
    logger.warning("STATISTICAL HYGIENE REPORT")
    logger.warning("=" * 72)
    for key in ("full_sample", "oos"):
        r = results[key]
        if r.get("insufficient"):
            logger.warning(f"\n{r['label']}: insufficient trades (n={r['n_trades']})")
            continue
        e, w = r["expectancy_per_trade"], r["win_rate"]
        logger.warning("")
        logger.warning(f"── {r['label']} (n={r['n_trades']}) " + "─" * 30)
        logger.warning(
            f"  Expectancy/trade : {e['point'] * 100:+.3f}%  "
            f"[95% CI {e['ci95_low'] * 100:+.3f}% .. {e['ci95_high'] * 100:+.3f}%]"
            f"{'  ⚠ CI INCLUDES ZERO' if e['includes_zero'] else ''}"
        )
        logger.warning(
            f"  Win rate         : {w['point'] * 100:.1f}%   "
            f"[95% CI {w['ci95_low'] * 100:.1f}% .. {w['ci95_high'] * 100:.1f}%]"
        )
        logger.warning(
            f"  Per-trade Sharpe : {r['per_trade_sharpe']:.4f}  "
            f"(annualized est. {r['annualized_sharpe_est']:.3f})"
        )
        logger.warning(f"  Skew / Kurtosis  : {r['skew']:+.2f} / {r['kurtosis']:.2f}")
        logger.warning(f"  PSR vs 0         : {r['psr_vs_zero']:.4f}  (P(true Sharpe > 0); want > 0.95)")

    d = results["deflated_sharpe"]
    logger.warning("")
    logger.warning("── Deflated Sharpe (selection-bias correction) " + "─" * 18)
    logger.warning(f"  Documented trials       : {d['n_trials']}")
    logger.warning(f"  E[max SR] of null trials: {d['expected_max_sharpe_annualized']:.3f} (annualized)")
    logger.warning(f"  Champion SR (annualized): {d['champion_sharpe_annualized']:.3f}")
    logger.warning(
        f"  DSR                     : {d['dsr']:.4f}  "
        f"({'PASS' if d['dsr'] >= 0.95 else 'FAIL'} at 0.95 threshold)"
    )
    logger.warning("")

    out_path = OUTPUT_DIR / "stat_hygiene.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Saved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CIs + deflated Sharpe")
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--oos-start", default="2023-01-01")
    parser.add_argument("--boot", type=int, default=10_000)
    args = parser.parse_args()
    run(args.trade_log, args.oos_start, args.boot)
