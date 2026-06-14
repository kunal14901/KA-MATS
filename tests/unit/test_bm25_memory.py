"""Unit tests for BM25Memory — 36% → ~85%."""

import json
import tempfile
from pathlib import Path

import pytest

from core.bm25_memory import (
    HALF_LIFE_DAYS,
    BM25Memory,
    MemoryRecord,
    _composite_weight,
)


@pytest.mark.unit
class TestMemoryRecord:
    def test_defaults(self):
        r = MemoryRecord(situation="test", outcome="ok")
        assert r.pnl == 0.0
        assert r.regime == ""
        assert r.score == 0.0


@pytest.mark.unit
class TestCompositeWeight:
    def test_no_date_returns_one(self):
        r = MemoryRecord(situation="x", outcome="y", regime="trending_up")
        w = _composite_weight(r, as_of_date="", current_regime="")
        assert w == pytest.approx(1.0)

    def test_same_day_returns_regime_mult(self):
        r = MemoryRecord(situation="x", outcome="y", regime="trending_up", trade_date="2024-06-15")
        w = _composite_weight(r, as_of_date="2024-06-15", current_regime="trending_up")
        # recency=1.0, same family (bull::bull) = 2.0
        assert w == pytest.approx(2.0)

    def test_old_same_regime_still_meaningful(self):
        r = MemoryRecord(situation="x", outcome="y", regime="volatile", trade_date="2022-01-01")
        w = _composite_weight(r, as_of_date="2025-01-01", current_regime="volatile")
        # ~3 years old but same bear family → still boosted above zero
        assert w > 0.001

    def test_old_opposite_regime_near_zero(self):
        r = MemoryRecord(situation="x", outcome="y", regime="trending_up", trade_date="2022-01-01")
        w = _composite_weight(r, as_of_date="2025-01-01", current_regime="trending_down")
        # ~3 years old + opposite (bull vs bear) → near zero
        assert w < 0.05

    def test_invalid_dates_graceful(self):
        r = MemoryRecord(situation="x", outcome="y", trade_date="bad-date")
        w = _composite_weight(r, as_of_date="also-bad", current_regime="")
        assert w == pytest.approx(1.0)

    def test_half_life_decay(self):
        r = MemoryRecord(situation="x", outcome="y", trade_date="2024-01-01")
        w = _composite_weight(r, as_of_date="2024-06-29", current_regime="")
        # ~180 days → significant decay
        assert 0.2 < w < 0.6


@pytest.mark.unit
class TestBM25Memory:
    def test_init_empty(self):
        mem = BM25Memory()
        assert len(mem) == 0

    def test_add_record(self):
        mem = BM25Memory()
        mem.add("symbol BTC regime bull", "WIN", pnl=100.0, regime="trending_up")
        assert len(mem) == 1

    def test_query_empty(self):
        mem = BM25Memory()
        result = mem.query("symbol BTC")
        assert result == []

    def test_add_and_query(self):
        mem = BM25Memory()
        mem.add(
            "symbol BTC strategy Momentum regime trending_up", "WIN: profit", pnl=200.0, regime="trending_up"
        )
        mem.add("symbol ETH strategy MeanRev regime ranging", "LOSS: stopped", pnl=-50.0, regime="ranging")
        mem.add(
            "symbol BTC strategy Momentum regime trending_up direction BUY",
            "WIN",
            pnl=300.0,
            regime="trending_up",
        )

        results = mem.query("symbol BTC strategy Momentum regime trending_up", n=2)
        assert len(results) <= 2
        # Top result should be BTC related
        if results:
            assert results[0].score == pytest.approx(1.0)  # normalised to 1.0

    def test_query_with_regime_weighting(self):
        mem = BM25Memory()
        mem.add(
            "symbol BTC strategy X regime trending_up",
            "WIN",
            pnl=100.0,
            regime="trending_up",
            trade_date="2024-01-01",
        )
        mem.add(
            "symbol BTC strategy X regime volatile",
            "LOSS",
            pnl=-50.0,
            regime="volatile",
            trade_date="2024-01-01",
        )

        # Query in a bull regime — trending_up lesson should rank higher
        results = mem.query(
            "symbol BTC strategy X",
            n=2,
            as_of_date="2024-06-01",
            current_regime="trending_up",
        )
        if len(results) == 2:
            # The trending_up (bull) lesson should be boosted
            bull_rec = [r for r in results if r.regime == "trending_up"]
            bear_rec = [r for r in results if r.regime == "volatile"]
            if bull_rec and bear_rec:
                assert bull_rec[0].score >= bear_rec[0].score

    def test_confidence_modifier_no_records(self):
        mem = BM25Memory()
        mod, note = mem.confidence_modifier_from_memory("symbol BTC")
        assert mod == 0.0
        assert "No relevant memory" in note

    def test_confidence_modifier_all_wins(self):
        mem = BM25Memory()
        for i in range(5):
            mem.add(f"symbol BTC strategy X regime bull trade_{i}", "WIN", pnl=100.0)
        mod, note = mem.confidence_modifier_from_memory("symbol BTC strategy X regime bull")
        assert mod >= 0.0

    def test_confidence_modifier_all_losses(self):
        mem = BM25Memory()
        for i in range(5):
            mem.add(f"symbol BTC strategy X regime bear trade_{i}", "LOSS", pnl=-80.0)
        mod, note = mem.confidence_modifier_from_memory("symbol BTC strategy X regime bear")
        assert mod <= 0.0

    def test_max_records_eviction(self):
        mem = BM25Memory(max_records=5)
        for i in range(10):
            mem.add(f"record {i}", f"outcome {i}")
        assert len(mem) == 5

    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bm25_test.json")
            mem = BM25Memory(persist_path=path)
            mem.add("symbol BTC regime bull", "WIN", pnl=100.0, regime="trending_up", strategy="X")
            mem.add("symbol ETH regime bear", "LOSS", pnl=-50.0, regime="volatile", strategy="Y")

            # Reload
            mem2 = BM25Memory(persist_path=path)
            assert len(mem2) == 2

    def test_persist_bad_path_no_crash(self):
        # Non-writable path shouldn't crash
        mem = BM25Memory(persist_path="/nonexistent_dir_xyz_unique/bm25_test.json")
        initial_len = len(mem)
        mem.add("test", "outcome")  # _save will fail gracefully
        assert len(mem) == initial_len + 1

    def test_load_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("NOT JSON", encoding="utf-8")
            mem = BM25Memory(persist_path=str(path))
            assert len(mem) == 0  # graceful fallback

    def test_tokenize(self):
        mem = BM25Memory()
        tokens = mem._tokenize("symbol BTC/USDT strategy CryptoMomentumBreakout regime_label trending_up")
        assert "btc" in tokens
        assert "usdt" in tokens
        assert "cryptomomentumbreakout" in tokens

    def test_confidence_modifier_bounded(self):
        mem = BM25Memory()
        for i in range(10):
            mem.add(f"symbol BTC test {i}", "WIN", pnl=10000.0)
        mod, _ = mem.confidence_modifier_from_memory("symbol BTC test")
        assert -0.12 <= mod <= 0.12
