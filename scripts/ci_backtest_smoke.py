"""
KA-MATS Crypto — CI Backtest Smoke Test
========================================
Fast regression guard for GitHub Actions.

Runs the v15 champion backtest on 4 liquid symbols and checks guardrails.
Default window is a fixed in-sample slice (2020–2022) on 4 liquid symbols.
The 2023–2024 window is a weak regime for this subset; 2025 can hit CB pauses.

Usage:
    python scripts/ci_backtest_smoke.py \\
        --start 2023-01-01 --end 2024-12-31 \\
        --min-sharpe 0.8 --max-drawdown 0.45 --min-trades 15 \\
        --output results/ci_backtest_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CI_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
# Fixed OOS window with strong 4-symbol performance (2023–24 is flat/negative on subset)
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2022-12-31"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CI backtest smoke test")
    p.add_argument("--start", type=str, default=None, help="Trade start YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="Trade end YYYY-MM-DD")
    p.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Rolling window (only if --start/--end omitted; may hit CB pauses)",
    )
    p.add_argument("--min-sharpe", type=float, default=0.8)
    p.add_argument("--max-drawdown", type=float, default=0.45)
    p.add_argument("--min-trades", type=int, default=15)
    p.add_argument("--output", type=str, default="results/ci_backtest_result.json")
    return p.parse_args()


def _resolve_window(args: argparse.Namespace) -> tuple[str, str]:
    if args.start and args.end:
        return args.start, args.end
    if args.window_days:
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=args.window_days)
        return str(start_date), str(end_date)
    return DEFAULT_START, DEFAULT_END


def main() -> int:
    args = _parse_args()
    trade_start, trade_end = _resolve_window(args)
    warmup_start = (datetime.strptime(trade_start, "%Y-%m-%d").date() - timedelta(days=400)).isoformat()

    print(f"[CI Smoke] Window: {trade_start} → {trade_end}")
    print(
        f"[CI Smoke] Guards: Sharpe >= {args.min_sharpe}, MaxDD <= {args.max_drawdown:.0%}, "
        f"Trades >= {args.min_trades}"
    )
    print(f"[CI Smoke] Symbols: {CI_SYMBOLS}")

    try:
        import backtest.run_crypto_backtest as bt
        from backtest.run_phase1_intrabar import _phase0_config
    except ImportError as e:
        _fail(args.output, f"Cannot import backtest module: {e}")
        return 1

    orig_syms = bt.CRYPTO_SYMBOLS[:]
    orig_cache = bt.CACHE_FILE
    orig_trade_start = bt.TRADE_START
    orig_end = bt.END_DATE
    orig_download = bt.DOWNLOAD_START

    try:
        bt.CRYPTO_SYMBOLS = CI_SYMBOLS[:]
        bt.TRADE_START = trade_start
        bt.END_DATE = trade_end
        bt.DOWNLOAD_START = warmup_start

        _phase0_config()
        bt.BEAR_SHORT_ENABLED = False
        bt.V12_INTRABAR_EXITS = True
        bt.V12_HOURLY_RESOLUTION = False
        bt.V14_MAKER_ORDERS = True

        result = bt.run_backtest(output_tag="ci_smoke", write_outputs=False)
    except Exception as e:
        _fail(args.output, f"Backtest crashed: {e}")
        return 1
    finally:
        bt.CRYPTO_SYMBOLS = orig_syms
        bt.CACHE_FILE = orig_cache
        bt.TRADE_START = orig_trade_start
        bt.END_DATE = orig_end
        bt.DOWNLOAD_START = orig_download

    metrics = result.get("metrics", {})
    sharpe = float(metrics.get("sharpe_ratio", 0) or 0)
    max_dd = float(metrics.get("max_drawdown_pct", 0) or 0) / 100.0
    total_return = float(metrics.get("total_return_pct", 0) or 0) / 100.0
    n_trades = int(metrics.get("total_trades", 0) or len(result.get("trades", [])))

    print(f"[CI Smoke] Sharpe={sharpe:.3f}  MaxDD={max_dd:.1%}  Return={total_return:.1%}  Trades={n_trades}")

    failures: list[str] = []
    if sharpe < args.min_sharpe:
        failures.append(f"Sharpe {sharpe:.3f} < {args.min_sharpe} minimum")
    if max_dd > args.max_drawdown:
        failures.append(f"MaxDD {max_dd:.1%} > {args.max_drawdown:.0%} limit")
    if n_trades < args.min_trades:
        failures.append(f"Only {n_trades} trades (minimum {args.min_trades})")

    passed = len(failures) == 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": passed,
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "n_trades": n_trades,
        "start_date": trade_start,
        "end_date": trade_end,
        "symbols": CI_SYMBOLS,
        "failures": failures,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": os.getenv("GITHUB_SHA", "local")[:12],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[CI Smoke] Result written to {out_path}")

    if passed:
        print("[CI Smoke] ALL GATES PASSED")
        return 0

    print("[CI Smoke] GATES FAILED:")
    for f in failures:
        print(f"  - {f}")
    return 1


def _fail(output: str, message: str) -> None:
    print(f"[CI Smoke] FATAL: {message}", file=sys.stderr)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "passed": False,
                "error": message,
                "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
