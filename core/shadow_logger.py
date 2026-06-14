"""
KA-MATS Cryptoz · Shadow Logger
Iknir Capital

Logs every Strategy Agent signal at two checkpoints:
  1. Raw   — immediately after Strategy Agent generates the signal
  2. Final — after Adversarial/Knowledge filtering

Purpose: audit whether the Adversarial/Knowledge layer is destroying
valid signals. Compare raw vs final sets over 2-3 months to measure
their net effect on the validated backtest logic.

Output: logs/shadow/shadow_YYYYMMDD.jsonl  (one JSON object per line)

Each entry has:
  bar_time            UTC timestamp of the bar
  symbol              e.g. "BTC/USDT"
  signal_id           UUID from CandidateSignal
  strategy            strategy_name
  direction           BUY or SELL
  regime              regime label from MarketAnalyst
  raw_confidence      confidence straight out of Strategy Agent
  knowledge_modifier  modifier applied by KnowledgeAgent (0.0 if none)
  post_knowledge_conf confidence after knowledge modifier, before adversarial
  adversarial_verdict pass | flag | fail | error | no_assessment
  adversarial_note    reason text from AdversarialAgent (empty if pass)
  conf_adjustment     confidence delta applied by a "flag" verdict
  final_confidence    confidence after all modifications
  survived            true  = passed to Risk Manager
                      false = filtered out before Risk Manager
  stop_price          stop-loss price from signal
  target_price        take-profit price from signal
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger


class ShadowLogger:
    """
    Lightweight JSONL logger for shadow-mode signal auditing.

    Usage in orchestrator:
        self.shadow = ShadowLogger()
        # After strategy_agent.evaluate():
        raw_state = self.shadow.snapshot_raw(sym, signals, regime, knowledge_modifier=0.0)
        # After adversarial filtering:
        self.shadow.write_outcomes(raw_state, surviving_signal_ids, assessments, bar_time)
    """

    def __init__(self, log_dir: str = "logs/shadow") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._today: str = ""
        self._fh = None
        logger.info(f"[ShadowLogger] Writing to {self._log_dir.resolve()}")

    def _ensure_file(self, bar_time: datetime) -> None:
        """Rotate to a new file each UTC day."""
        day_str = bar_time.strftime("%Y%m%d")
        if day_str != self._today:
            if self._fh:
                self._fh.close()
            path = self._log_dir / f"shadow_{day_str}.jsonl"
            self._fh = open(path, "a", encoding="utf-8")
            self._today = day_str

    def snapshot_raw(
        self,
        symbol: str,
        signals: list,
        regime_label: str,
        knowledge_modifier: float,
    ) -> list[dict]:
        """
        Capture signal state after knowledge modifier is applied, before adversarial.
        Returns a list of raw-state dicts keyed by signal_id for later outcome writing.
        """
        states = []
        for sig in signals:
            raw_conf = getattr(sig, "_raw_confidence", sig.confidence)
            states.append(
                {
                    "symbol": symbol,
                    "signal_id": str(sig.signal_id),
                    "strategy": sig.strategy_name,
                    "direction": str(sig.direction.value)
                    if hasattr(sig.direction, "value")
                    else str(sig.direction),
                    "regime": regime_label,
                    "raw_confidence": round(raw_conf, 4),
                    "knowledge_modifier": round(knowledge_modifier, 4),
                    "post_knowledge_conf": round(sig.confidence, 4),
                    "stop_price": round(sig.stop_price, 4) if sig.stop_price else None,
                    "target_price": round(sig.target_price, 4) if sig.target_price else None,
                }
            )
        return states

    def write_outcomes(
        self,
        raw_states: list[dict],
        surviving_ids: set,
        assessments: list,
        bar_time: datetime,
    ) -> None:
        """
        Write one JSONL entry per signal with verdict + survived flag.
        Call this after adversarial filtering is complete.
        """
        if not raw_states:
            return

        self._ensure_file(bar_time)

        assessment_map: dict[str, object] = {}
        for a in assessments or []:
            assessment_map[str(a.signal_id)] = a

        for state in raw_states:
            sig_id = state["signal_id"]
            assessment = assessment_map.get(sig_id)

            if assessment is None:
                verdict = "no_assessment"
                note = ""
                conf_adj = 0.0
            else:
                verdict = (
                    assessment.verdict.value
                    if hasattr(assessment.verdict, "value")
                    else str(assessment.verdict)
                )
                note = getattr(assessment, "adversarial_note", "") or ""
                conf_adj = float(getattr(assessment, "confidence_adjustment", 0.0) or 0.0)

            survived = sig_id in surviving_ids

            # Compute final confidence: post_knowledge + conf_adj (only for "flag" verdicts)
            final_conf = state["post_knowledge_conf"]
            if verdict == "flag":
                final_conf = max(0.0, min(1.0, final_conf + conf_adj))

            entry = {
                "bar_time": bar_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                **state,
                "adversarial_verdict": verdict,
                "adversarial_note": note[:200],  # truncate long notes
                "conf_adjustment": round(conf_adj, 4),
                "final_confidence": round(final_conf, 4),
                "survived": survived,
            }
            self._fh.write(json.dumps(entry) + "\n")

        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __del__(self) -> None:
        self.close()
