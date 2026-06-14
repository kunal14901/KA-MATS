"""
KA-MATS Cryptoz · Shadow Log Analyzer
Iknir Capital

Reads all shadow JSONL files and answers the key question:
  "Is the Adversarial/Knowledge layer destroying valid signals?"

Usage:
    python tools/analyze_shadow_log.py
    python tools/analyze_shadow_log.py --log-dir logs/shadow --min-bars 20

Output:
  1. Overall filter rate: what % of Strategy Agent signals get rejected
  2. Per-strategy breakdown: which strategies get blocked most
  3. Confidence shift: does the Knowledge modifier help or hurt
  4. Rejection reason summary: top adversarial_notes
  5. Verdict: "destroying value" / "neutral" / "adding value"
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter
from pathlib import Path


def load_logs(log_dir: str) -> list[dict]:
    p = Path(log_dir)
    entries = []
    for f in sorted(p.glob("shadow_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        entries.append(json.loads(line))
    return entries


def analyze(entries: list[dict], min_bars: int) -> None:
    if not entries:
        print("No shadow log entries found.")
        return

    total = len(entries)
    survived = sum(1 for e in entries if e["survived"])
    rejected = total - survived
    filter_rate = rejected / total * 100 if total > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  KA-MATS Shadow Log Analysis — {total} signals")
    print(f"{'=' * 60}")
    print(f"\n  Survived : {survived:>5}  ({100 - filter_rate:.1f}%)")
    print(f"  Rejected : {rejected:>5}  ({filter_rate:.1f}%)")

    if total < min_bars:
        print(f"\n  [!] Only {total} entries — need {min_bars}+ for reliable stats.")
        print("      Check back after more paper trading bars.\n")
        return

    # ── Per-strategy breakdown ──────────────────────────────
    strat_total = Counter()
    strat_survive = Counter()
    for e in entries:
        s = e["strategy"]
        strat_total[s] += 1
        if e["survived"]:
            strat_survive[s] += 1

    print(f"\n  {'Strategy':<35} {'Signals':>7} {'Survived':>9} {'FilterRate':>11}")
    print(f"  {'-' * 64}")
    for s in sorted(strat_total, key=lambda x: -strat_total[x]):
        tot = strat_total[s]
        sur = strat_survive[s]
        fr = (tot - sur) / tot * 100
        print(f"  {s:<35} {tot:>7} {sur:>9} {fr:>10.1f}%")

    # ── Verdict distribution ────────────────────────────────
    verdicts = Counter(e["adversarial_verdict"] for e in entries)
    print("\n  Adversarial verdicts:")
    for v, cnt in verdicts.most_common():
        print(f"    {v:<20} {cnt:>5}  ({cnt / total * 100:.1f}%)")

    # ── Knowledge modifier effect ───────────────────────────
    mods = [e["knowledge_modifier"] for e in entries if e["knowledge_modifier"] != 0.0]
    if mods:
        avg_mod = sum(mods) / len(mods)
        pos = sum(1 for m in mods if m > 0)
        neg = sum(1 for m in mods if m < 0)
        print(f"\n  Knowledge modifier: {len(mods)} signals modified")
        print(f"    avg={avg_mod:+.4f}  positive={pos}  negative={neg}")
    else:
        print("\n  Knowledge modifier: 0.0 on all signals (no effect)")

    # ── Top rejection reasons ───────────────────────────────
    fail_notes = [e["adversarial_note"] for e in entries if not e["survived"] and e["adversarial_note"]]
    if fail_notes:
        note_counts = Counter(fail_notes)
        print("\n  Top rejection reasons:")
        for note, cnt in note_counts.most_common(10):
            print(f"    [{cnt:>3}] {note[:80]}")

    # ── Confidence shift analysis ───────────────────────────
    conf_diffs = [e["final_confidence"] - e["raw_confidence"] for e in entries if e["survived"]]
    if conf_diffs:
        avg_shift = sum(conf_diffs) / len(conf_diffs)
        print("\n  Confidence shift (survived signals only):")
        print(f"    avg delta = {avg_shift:+.4f}")
        improved = sum(1 for d in conf_diffs if d > 0.005)
        degraded_n = sum(1 for d in conf_diffs if d < -0.005)
        unchanged = len(conf_diffs) - improved - degraded_n
        print(f"    improved={improved}  degraded={degraded_n}  unchanged={unchanged}")

    # ── Verdict ─────────────────────────────────────────────
    print(f"\n  {'=' * 60}")
    if filter_rate > 25:
        verdict = (
            "CAUTION — layer is rejecting >25% of signals. Compare PnL of rejected vs survived to confirm."
        )
    elif filter_rate > 10:
        verdict = (
            "MONITOR — layer is active but moderate. Watch if rejected signals would have been profitable."
        )
    else:
        verdict = "NEUTRAL — layer is passing most signals through. Minimal interference with backtest logic."

    print(f"  VERDICT: {verdict}")
    print(f"  {'=' * 60}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/shadow")
    ap.add_argument(
        "--min-bars", type=int, default=20, help="Minimum entries before printing reliability warning"
    )
    args = ap.parse_args()

    entries = load_logs(args.log_dir)
    analyze(entries, args.min_bars)


if __name__ == "__main__":
    main()
