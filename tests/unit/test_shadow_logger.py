"""Unit tests for ShadowLogger — 36% → ~90%."""

import json
import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.shadow_logger import ShadowLogger


def _fake_signal(symbol="BTC/USDT", confidence=0.70, strategy="TestStrat", direction="BUY"):
    sig = MagicMock()
    sig.signal_id = uuid4()
    sig.symbol = symbol
    sig.confidence = confidence
    sig._raw_confidence = confidence
    sig.strategy_name = strategy
    sig.direction = MagicMock()
    sig.direction.value = direction
    sig.stop_price = 43000.0
    sig.target_price = 47000.0
    return sig


def _fake_assessment(signal_id, verdict="pass", note="ok", conf_adj=0.0):
    a = MagicMock()
    a.signal_id = signal_id
    a.verdict = MagicMock()
    a.verdict.value = verdict
    a.adversarial_note = note
    a.confidence_adjustment = conf_adj
    return a


@pytest.mark.unit
class TestShadowLogger:
    def test_init_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "shadow_test"
            ShadowLogger(log_dir=str(log_dir))
            assert log_dir.exists()

    def test_snapshot_raw_captures_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sig = _fake_signal()
            states = sl.snapshot_raw("BTC/USDT", [sig], "trending_up", 0.05)
            assert len(states) == 1
            assert states[0]["symbol"] == "BTC/USDT"
            assert states[0]["strategy"] == "TestStrat"
            assert states[0]["regime"] == "trending_up"
            assert states[0]["knowledge_modifier"] == 0.05

    def test_snapshot_raw_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sigs = [_fake_signal(symbol=f"SYM{i}") for i in range(3)]
            states = sl.snapshot_raw("X", sigs, "ranging", 0.0)
            assert len(states) == 3

    def test_write_outcomes_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sig = _fake_signal()
            states = sl.snapshot_raw("BTC/USDT", [sig], "trending_up", 0.0)
            bar_time = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
            assessment = _fake_assessment(sig.signal_id, "pass", "All OK")
            sl.write_outcomes(states, {str(sig.signal_id)}, [assessment], bar_time)
            sl.close()

            # Check file was written
            files = list((Path(tmp) / "s").glob("shadow_*.jsonl"))
            assert len(files) == 1
            content = files[0].read_text(encoding="utf-8").strip()
            entry = json.loads(content)
            assert entry["survived"] is True
            assert entry["adversarial_verdict"] == "pass"

    def test_write_outcomes_failed_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sig = _fake_signal()
            states = sl.snapshot_raw("BTC/USDT", [sig], "volatile", 0.0)
            bar_time = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
            assessment = _fake_assessment(sig.signal_id, "fail", "Macro filter")
            sl.write_outcomes(states, set(), [assessment], bar_time)  # no survivors
            sl.close()

            files = list((Path(tmp) / "s").glob("*.jsonl"))
            entry = json.loads(files[0].read_text(encoding="utf-8").strip())
            assert entry["survived"] is False
            assert entry["adversarial_verdict"] == "fail"

    def test_write_outcomes_flag_adjusts_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sig = _fake_signal(confidence=0.60)
            states = sl.snapshot_raw("BTC/USDT", [sig], "volatile", 0.0)
            bar_time = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
            assessment = _fake_assessment(sig.signal_id, "flag", "RSI extended", conf_adj=-0.05)
            sl.write_outcomes(states, {str(sig.signal_id)}, [assessment], bar_time)
            sl.close()

            files = list((Path(tmp) / "s").glob("*.jsonl"))
            entry = json.loads(files[0].read_text(encoding="utf-8").strip())
            assert entry["conf_adjustment"] == -0.05
            assert entry["final_confidence"] == pytest.approx(0.55)

    def test_write_outcomes_no_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sig = _fake_signal()
            states = sl.snapshot_raw("BTC/USDT", [sig], "ranging", 0.0)
            bar_time = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
            sl.write_outcomes(states, set(), [], bar_time)
            sl.close()

            files = list((Path(tmp) / "s").glob("*.jsonl"))
            entry = json.loads(files[0].read_text(encoding="utf-8").strip())
            assert entry["adversarial_verdict"] == "no_assessment"

    def test_write_outcomes_empty_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            bar_time = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
            sl.write_outcomes([], set(), [], bar_time)
            sl.close()
            # Should not create any file
            files = list((Path(tmp) / "s").glob("*.jsonl"))
            assert len(files) == 0

    def test_file_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sig = _fake_signal()
            states = sl.snapshot_raw("X", [sig], "r", 0.0)

            day1 = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
            day2 = datetime(2024, 6, 16, 12, 0, tzinfo=UTC)

            a = _fake_assessment(sig.signal_id)
            sl.write_outcomes(states, {str(sig.signal_id)}, [a], day1)
            sl.write_outcomes(states, {str(sig.signal_id)}, [a], day2)
            sl.close()

            files = list((Path(tmp) / "s").glob("*.jsonl"))
            assert len(files) == 2

    def test_close_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl = ShadowLogger(log_dir=str(Path(tmp) / "s"))
            sl.close()
            sl.close()  # should not error
