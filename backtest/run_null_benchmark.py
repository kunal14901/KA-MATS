"""
backtest/run_null_benchmark.py
==============================
Phase 1 statistical hygiene: compare the full 9-agent system against the
"null" strategies it must beat to justify its complexity.

The system's macro filter (BTC > EMA200 + golden cross) does most of the
heavy lifting — it produced zero trades in 2022. The honest question is:
does the rest of the stack (regime detection, strategies, adaptive learner,
risk layers) add anything over simply HOLDING the market whenever that same
filter is bullish?

Benchmarks (same fee/slippage model as the backtest engine):
  A. BTC buy-and-hold              — the market beta everyone quotes
  B. BTC golden-cross timer        — hold BTC when macro_bull, else cash
  C. Equal-weight GX timer         — hold all 20 coins equal-weight when
                                     macro_bull, else cash

Usage:
    python -m backtest.run_null_benchmark
    python -m backtest.run_null_benchmark --summary results/crypto_backtest/summary_crypto_<tag>.json

Output: console table + results/crypto_backtest/null_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from backtest.run_crypto_backtest import (
    CRYPTO_SYMBOLS,
    END_DATE,
    FEE_BPS,
    INITIAL_CAP,
    OUTPUT_DIR,
    TRADE_START,
    compute_macro_states,
    get_slippage_bps,
    load_data,
)

# ── Metrics ────────────────────────────────────────────────────────────────────


def equity_metrics(eq: pd.Series, label: str) -> dict:
    """Standard metrics from a daily equity series. Sharpe uses the same
    convention as the backtest engine: daily mean/std * sqrt(365)."""
    eq = eq.dropna()
    rets = eq.pct_change().fillna(0.0)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    ann_ret = (1.0 + total_ret) ** (1.0 / years) - 1.0
    std = float(rets.std(ddof=0))
    sharpe = float(rets.mean()) / std * (365**0.5) if std > 0 else 0.0
    roll_max = eq.cummax()
    max_dd = float(((roll_max - eq) / roll_max).max())
    return {
        "label": label,
        "total_return_pct": round(total_ret * 100, 1),
        "annualized_return_pct": round(ann_ret * 100, 1),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "final_equity": round(float(eq.iloc[-1]), 2),
    }


# ── Benchmark simulators ───────────────────────────────────────────────────────


def _round_trip_cost(sym: str) -> float:
    """One-side cost (slippage + fee) as a fraction, per transition."""
    return (get_slippage_bps(sym) + FEE_BPS) / 10_000


def benchmark_buy_hold(closes: pd.Series, sym: str = "BTC-USD") -> pd.Series:
    """Buy at the first bar (paying entry costs), hold to the end."""
    cost = _round_trip_cost(sym)
    units = INITIAL_CAP * (1.0 - cost) / closes.iloc[0]
    return units * closes


def benchmark_timer(closes: pd.Series, macro: pd.Series, sym: str = "BTC-USD") -> pd.Series:
    """Hold the asset when macro_bull is True at the prior close, else cash.
    The signal at close t puts the position on from close t (same-bar close
    fill — identical convention to the system's entries). Each transition
    pays slippage + fee on the traded side."""
    cost = _round_trip_cost(sym)
    macro_aligned = macro.reindex(closes.index).ffill().fillna(False)

    equity = []
    cash = INITIAL_CAP
    units = 0.0
    holding = False
    for ts, px in closes.items():
        want = bool(macro_aligned.loc[ts])
        if want and not holding:
            units = cash * (1.0 - cost) / px
            cash, holding = 0.0, True
        elif not want and holding:
            cash = units * px * (1.0 - cost)
            units, holding = 0.0, False
        equity.append(cash + units * px)
    return pd.Series(equity, index=closes.index)


def benchmark_equal_weight_timer(
    data_map: dict[str, pd.DataFrame], macro: pd.Series, trade_idx: pd.DatetimeIndex
) -> pd.Series:
    """Equal-weight portfolio of all available coins when macro_bull, else cash.
    Weights are set at each cash->invested flip across coins with data on that
    date; positions are then held untouched until the flip back to cash
    (no daily rebalancing — keeps transaction costs honest and comparable)."""
    macro_aligned = macro.reindex(trade_idx).ffill().fillna(False)

    cash = INITIAL_CAP
    holdings: dict[str, float] = {}  # sym -> units
    invested = False
    equity = []

    for ts in trade_idx:
        prices = {
            sym: float(df.loc[ts, "close"])
            for sym, df in data_map.items()
            if ts in df.index and np.isfinite(df.loc[ts, "close"])
        }
        want = bool(macro_aligned.loc[ts])

        if want and not invested and prices:
            alloc = cash / len(prices)
            holdings = {sym: alloc * (1.0 - _round_trip_cost(sym)) / px for sym, px in prices.items()}
            cash, invested = 0.0, True
        elif not want and invested:
            for sym, units in holdings.items():
                if sym in prices:
                    cash += units * prices[sym] * (1.0 - _round_trip_cost(sym))
                # symbol with no price today (delisted/dead): mark worthless
            holdings, invested = {}, False

        mark = cash + sum(units * prices.get(sym, 0.0) for sym, units in holdings.items())
        equity.append(mark)

    return pd.Series(equity, index=trade_idx)


# ── System results loader ──────────────────────────────────────────────────────


def load_system_metrics(summary_path: Path | None) -> dict | None:
    """Load the system's summary JSON (explicit path or newest in OUTPUT_DIR)."""
    if summary_path is None:
        candidates = sorted(
            OUTPUT_DIR.glob("summary_crypto_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        summary_path = candidates[0]
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics", data)
        return {
            "label": f"KA-MATS system ({summary_path.name})",
            "total_return_pct": m.get("total_return_pct"),
            "annualized_return_pct": m.get("annualized_return_pct"),
            "sharpe": m.get("sharpe_ratio"),
            "max_drawdown_pct": m.get("max_drawdown_pct"),
            "final_equity": m.get("final_equity"),
        }
    except Exception as e:
        logger.warning(f"Could not load system summary {summary_path}: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────


def run(summary_path: Path | None = None) -> dict:
    logger.info("Loading daily data (cache or yfinance download)...")
    data_map = load_data()
    macro = compute_macro_states(data_map)

    btc = data_map["BTC-USD"]
    full_idx = btc.index[(btc.index >= TRADE_START) & (btc.index < END_DATE)]
    btc_closes = btc.loc[full_idx, "close"].astype(float)

    windows = {
        "full_2020_2026": full_idx,
        "oos_2023_plus": full_idx[full_idx >= "2023-01-01"],
    }

    results: dict[str, list[dict]] = {}
    for win_name, idx in windows.items():
        closes = btc_closes.loc[idx]
        rows = [
            equity_metrics(benchmark_buy_hold(closes), "A. BTC buy-and-hold"),
            equity_metrics(benchmark_timer(closes, macro), "B. BTC golden-cross timer"),
            equity_metrics(
                benchmark_equal_weight_timer(data_map, macro, idx),
                "C. Equal-weight 20-coin GX timer",
            ),
        ]
        results[win_name] = rows

    system = load_system_metrics(summary_path)

    # ── Print ──────────────────────────────────────────────────────────────────
    for win_name, rows in results.items():
        logger.warning("")
        logger.warning(f"── Null benchmarks · {win_name} " + "─" * 30)
        header = f"{'Strategy':<36} {'Return':>10} {'Ann.':>8} {'Sharpe':>8} {'MaxDD':>8}"
        logger.warning(header)
        all_rows = rows + ([system] if (system and win_name == "full_2020_2026") else [])
        for r in all_rows:
            if r is None:
                continue
            logger.warning(
                f"{r['label']:<36} {r['total_return_pct']:>+9.1f}% "
                f"{r['annualized_return_pct']:>+7.1f}% {r['sharpe']:>8.3f} "
                f"{-abs(r['max_drawdown_pct']):>7.1f}%"
            )

    logger.warning("")
    logger.warning("Verdict guide: the system must beat benchmark B/C on Sharpe")
    logger.warning("(not just return) to justify the 9-agent stack over a 2-line timer.")

    out = {"benchmarks": results, "system": system}
    out_path = OUTPUT_DIR / "null_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.success(f"Saved -> {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Null benchmark comparison")
    parser.add_argument(
        "--summary", type=Path, default=None, help="path to a system summary_crypto_*.json for comparison"
    )
    args = parser.parse_args()
    run(args.summary)
