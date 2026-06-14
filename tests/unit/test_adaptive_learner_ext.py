"""Extended tests for AdaptiveLearner — boost from 51% to ~80%."""

import json
import tempfile
from pathlib import Path

import pytest

from config.settings import CONFIG
from core.adaptive_learner import (
    REGIME_FAMILIES,
    AdaptiveLearner,
    ConvictionRecord,
    CorrelationTracker,
    FlagRecord,
    SymbolRecord,
    _adaptive_alpha,
    _ema,
)


@pytest.mark.unit
class TestHelperFunctions:
    def test_ema_basic(self):
        # Uses the production default _EMA_ALPHA (0.08):
        # 0.08 * 1.0 + 0.92 * 0.5 = 0.54
        result = _ema(0.5, 1.0)
        assert result == pytest.approx(0.54)

    def test_ema_custom_alpha(self):
        result = _ema(0.0, 1.0, alpha=0.50)
        assert result == pytest.approx(0.50)

    def test_adaptive_alpha_cold(self):
        from core.adaptive_learner import _EMA_ALPHA

        assert _adaptive_alpha(5) == pytest.approx(_EMA_ALPHA)

    def test_adaptive_alpha_warm(self):
        from core.adaptive_learner import _EMA_ALPHA

        alpha = _adaptive_alpha(50)
        assert alpha == pytest.approx(_EMA_ALPHA + 0.08)

    def test_adaptive_alpha_mid(self):
        from core.adaptive_learner import _EMA_ALPHA

        alpha = _adaptive_alpha(31)
        assert _EMA_ALPHA < alpha < _EMA_ALPHA + 0.08


@pytest.mark.unit
class TestSymbolRecord:
    def test_stop_hit_rate_no_trades(self):
        r = SymbolRecord(symbol="BTC/USDT")
        assert r.stop_hit_rate == 0.0

    def test_stop_hit_rate(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=10, stop_hits=4)
        assert r.stop_hit_rate == pytest.approx(0.4)

    def test_confidence_modifier_insufficient_trades(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=2)
        assert r.confidence_modifier == 0.0

    def test_confidence_modifier_positive(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=10, rolling_win_rate=0.70)
        assert r.confidence_modifier > 0.0

    def test_confidence_modifier_negative(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=10, rolling_win_rate=0.30)
        assert r.confidence_modifier < 0.0

    def test_confidence_modifier_bounded(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=10, rolling_win_rate=1.0)
        assert r.confidence_modifier <= 0.25

    def test_atr_multiplier_adjustment_insufficient(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=2)
        assert r.atr_multiplier_adjustment == 0.0

    def test_atr_multiplier_adjustment_high_stop_rate(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=10, stop_hits=7)
        assert r.atr_multiplier_adjustment > 0.0  # widen ATR

    def test_atr_multiplier_adjustment_low_stop_rate(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=15, stop_hits=2)
        assert r.atr_multiplier_adjustment < 0.0  # tighten ATR

    def test_atr_multiplier_adjustment_normal(self):
        r = SymbolRecord(symbol="BTC/USDT", total_trades=10, stop_hits=4)
        assert r.atr_multiplier_adjustment == 0.0


@pytest.mark.unit
class TestFlagRecord:
    def test_penalty_weight_insufficient(self):
        r = FlagRecord(flag_type="crowding")
        assert r.penalty_weight == 1.0

    def test_penalty_weight_accurate(self):
        r = FlagRecord(flag_type="crowding", total_flagged=10, rolling_accuracy=0.80)
        pw = r.penalty_weight
        assert pw > 1.0  # high accuracy → stronger penalty


@pytest.mark.unit
class TestConvictionRecord:
    def test_alignment_score_modifier_insufficient(self):
        r = ConvictionRecord(conviction="BULLISH")
        assert r.alignment_score_modifier == 0.0

    def test_alignment_score_modifier_positive(self):
        r = ConvictionRecord(conviction="BULLISH", aligned_trades=10, rolling_win_rate=0.70)
        assert r.alignment_score_modifier > 0.0


@pytest.mark.unit
class TestAdaptiveLearnerExtended:
    def _record_trades(self, learner, strategy, regime, n_wins, n_losses):
        for _ in range(n_wins):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy=strategy,
                regime=regime,
                pnl=100.0,
                exit_reason="take_profit",
                trade_date="2024-06-01",
            )
        for _ in range(n_losses):
            learner.record_outcome(
                symbol="BTC/USDT",
                strategy=strategy,
                regime=regime,
                pnl=-50.0,
                exit_reason="stop_loss",
                trade_date="2024-06-01",
            )

    def test_regime_family_mapping(self):
        learner = AdaptiveLearner()
        assert learner._regime_family("trending_up") == "bull"
        assert learner._regime_family("trending_down") == "bear"
        assert learner._regime_family("volatile") == "bear"
        assert learner._regime_family("ranging") == "sideways"
        assert learner._regime_family("mean_reverting") == "sideways"
        assert learner._regime_family("unknown") == "sideways"

    def test_symbol_confidence_unknown(self):
        learner = AdaptiveLearner()
        assert learner.symbol_confidence("UNKNOWN/USDT") == 0.0

    def test_stop_hit_rate_unknown(self):
        learner = AdaptiveLearner()
        assert learner.stop_hit_rate("UNKNOWN/USDT") == 0.0

    def test_symbol_win_rate_insufficient(self):
        learner = AdaptiveLearner()
        assert learner.symbol_win_rate("BTC/USDT") is None

    def test_symbol_win_rate_after_trades(self):
        learner = AdaptiveLearner()
        for i in range(5):
            learner.record_outcome("BTC/USDT", "X", "trending_up", 100.0 if i < 3 else -50.0)
        wr = learner.symbol_win_rate("BTC/USDT")
        assert wr is not None
        assert 0.5 <= wr <= 0.7

    def test_atr_multiplier_adj_unknown(self):
        learner = AdaptiveLearner()
        assert learner.atr_multiplier_adj("UNKNOWN/USDT") == 0.0

    def test_flag_penalty_weight_unknown(self):
        learner = AdaptiveLearner()
        assert learner.flag_penalty_weight("unknown_flag") == 1.0

    def test_conviction_alignment_modifier_unknown(self):
        learner = AdaptiveLearner()
        assert learner.conviction_alignment_modifier("BULLISH") == 0.0

    def test_record_with_flags(self):
        learner = AdaptiveLearner()
        learner.record_outcome(
            "BTC/USDT",
            "Strat",
            "trending_up",
            -50.0,
            exit_reason="stop_loss",
            flag_types=["crowding_check", "volatility_regime"],
        )
        assert learner._flags.get("crowding_check") is not None
        assert learner._flags["crowding_check"].total_flagged == 1

    def test_record_with_conviction(self):
        learner = AdaptiveLearner()
        learner.record_outcome(
            "BTC/USDT",
            "Strat",
            "trending_up",
            100.0,
            conviction="BULLISH",
            trade_date="2024-06-01",
        )
        assert "BULLISH" in learner._convictions
        assert learner._convictions["BULLISH"].aligned_trades == 1

    def test_strategy_modifier_cold_start(self):
        learner = AdaptiveLearner()
        self._record_trades(learner, "Strat", "trending_up", 5, 2)
        # Only 7 trades — below 12 minimum
        assert learner.strategy_modifier("Strat", "trending_up") == 0.0

    def test_strategy_modifier_warm(self):
        learner = AdaptiveLearner()
        self._record_trades(learner, "Strat", "trending_up", 10, 5)
        # 15 trades in bull family → should have a modifier
        mod = learner.strategy_modifier("Strat", "trending_up")
        assert isinstance(mod, float)

    def test_regime_isolation(self):
        """Bear losses should NOT affect bull modifier."""
        learner = AdaptiveLearner()
        self._record_trades(learner, "Strat", "trending_up", 12, 3)
        self._record_trades(learner, "Strat", "volatile", 2, 13)

        bull_mod = learner.strategy_modifier("Strat", "trending_up")
        bear_mod = learner.strategy_modifier("Strat", "volatile")

        assert bull_mod > bear_mod

    def test_decay_stale_records(self):
        learner = AdaptiveLearner()
        self._record_trades(learner, "Strat", "trending_up", 10, 5)
        # Manually set last trade to > 90 days ago
        learner._last_trade_date["Strat::bull"] = "2023-01-01"
        old_wr = learner._strategy_regime_wr["Strat"]["bull"]
        learner.decay_stale_records(as_of_date="2024-06-01")
        new_wr = learner._strategy_regime_wr["Strat"]["bull"]
        # Should nudge toward 0.50
        assert abs(new_wr - 0.50) < abs(old_wr - 0.50)

    def test_decay_no_date(self):
        learner = AdaptiveLearner()
        learner.decay_stale_records()  # no date → no-op
        learner.decay_stale_records(as_of_date="")  # empty → no-op

    def test_check_recovery_suppression_empty(self):
        learner = AdaptiveLearner()
        assert learner.check_recovery_suppression() == []

    def test_check_recovery_suppression_detected(self):
        learner = AdaptiveLearner()
        # Simulate low EMA but recent wins
        learner._strategy_regime_wr["Strat"] = {"bull": 0.35, "bear": 0.50, "sideways": 0.50}
        learner._strategy_regime_count["Strat::bull"] = 20
        learner._recent_outcomes["Strat::bull"] = [1, 1, 1, 0, 1]
        warnings = learner.check_recovery_suppression()
        assert len(warnings) == 1
        assert "recovering" in warnings[0]["message"].lower()

    def test_get_learning_summary(self):
        learner = AdaptiveLearner()
        self._record_trades(learner, "Strat", "trending_up", 10, 5)
        s = learner.get_learning_summary()
        assert s["total_trades_observed"] == 15
        assert isinstance(s["strategy_insights"], list)
        assert isinstance(s["symbol_insights"], list)

    def test_log_monitoring_report(self):
        learner = AdaptiveLearner()
        learner.log_monitoring_report()  # empty — should not crash
        self._record_trades(learner, "Strat", "trending_up", 15, 5)
        learner.log_monitoring_report(equity=20000.0)

    def test_print_learning_report(self):
        learner = AdaptiveLearner()
        self._record_trades(learner, "Strat", "trending_up", 5, 5)
        learner.print_learning_report()  # should not crash

    def test_save_and_load_v17a(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "state.json")
            learner = AdaptiveLearner(state_file=path)
            self._record_trades(learner, "Strat", "trending_up", 10, 5)
            learner.record_outcome(
                "ETH/USDT",
                "Strat",
                "volatile",
                -50.0,
                flag_types=["crowding_check"],
                conviction="BEARISH",
            )
            learner.save()

            l2 = AdaptiveLearner(state_file=path)
            loaded = l2.load()
            assert loaded
            assert l2._trade_count == 16
            assert "Strat" in l2._strategy_regime_wr
            assert "bull" in l2._strategy_regime_wr["Strat"]

    def test_load_nonexistent(self):
        learner = AdaptiveLearner(state_file="/nonexistent/state.json")
        assert learner.load() is False

    def test_load_corrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("NOT JSON", encoding="utf-8")
            learner = AdaptiveLearner(state_file=str(path))
            assert learner.load() is False

    def test_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "state.json")
            learner = AdaptiveLearner(state_file=path)
            self._record_trades(learner, "Strat", "trending_up", 5, 5)
            learner.save()
            learner.reset()
            assert learner._trade_count == 0
            assert len(learner._strategy_regime_wr) == 0

    def test_recent_outcomes_ring_buffer(self):
        learner = AdaptiveLearner()
        # Record many trades to test ring buffer
        for i in range(10):
            learner.record_outcome("BTC/USDT", "Strat", "trending_up", 100.0 if i % 2 == 0 else -50.0)
        buf = learner._recent_outcomes.get("Strat::bull", [])
        assert len(buf) <= 5  # _RECOVERY_WINDOW = 5

    def test_regime_kill_switch_blocks_bear_longs(self, monkeypatch):
        learner = AdaptiveLearner()

        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_enabled", True)
        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_bear_block_longs", True)

        blocked = learner.is_strategy_blocked(
            strategy="Strat",
            regime="trending_down",
            direction="LONG",
        )
        assert blocked is True

    def test_regime_kill_switch_blocks_low_wr(self, monkeypatch):
        learner = AdaptiveLearner()

        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_enabled", True)
        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_bear_block_longs", False)
        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_min_trades", 12)
        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_min_wr", 0.35)

        # 3W / 9L in bull family -> WR ~ 0.31 with enough trades.
        self._record_trades(learner, "Strat", "trending_up", 3, 9)

        blocked = learner.is_strategy_blocked(
            strategy="Strat",
            regime="trending_up",
            direction="LONG",
        )
        assert blocked is True

    def test_regime_kill_switch_allows_cold_start(self, monkeypatch):
        learner = AdaptiveLearner()

        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_enabled", True)
        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_bear_block_longs", False)
        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_min_trades", 12)
        monkeypatch.setattr(CONFIG.risk, "regime_kill_switch_min_wr", 0.35)

        # Low WR but only 8 trades: should not block yet.
        self._record_trades(learner, "Strat", "trending_up", 2, 6)

        blocked = learner.is_strategy_blocked(
            strategy="Strat",
            regime="trending_up",
            direction="LONG",
        )
        assert blocked is False

    def test_regime_participation_penalizes_bear_longs(self, monkeypatch):
        learner = AdaptiveLearner()

        monkeypatch.setattr(CONFIG.risk, "regime_soft_participation_enabled", True)
        monkeypatch.setattr(CONFIG.risk, "regime_soft_bear_long_mult", 0.25)

        mult = learner.strategy_participation_multiplier(
            strategy="Strat",
            regime="trending_down",
            direction="LONG",
        )
        assert mult == pytest.approx(0.25)

    def test_regime_participation_penalizes_low_wr(self, monkeypatch):
        learner = AdaptiveLearner()

        monkeypatch.setattr(CONFIG.risk, "regime_soft_participation_enabled", True)
        monkeypatch.setattr(CONFIG.risk, "regime_soft_wr_penalty_enabled", True)
        monkeypatch.setattr(CONFIG.risk, "regime_soft_min_trades", 12)
        monkeypatch.setattr(CONFIG.risk, "regime_soft_min_wr", 0.35)
        monkeypatch.setattr(CONFIG.risk, "regime_soft_floor_mult", 0.20)

        self._record_trades(learner, "Strat", "trending_up", 3, 9)
        mult = learner.strategy_participation_multiplier(
            strategy="Strat",
            regime="trending_up",
            direction="LONG",
        )

        assert 0.20 <= mult < 1.0

    def test_regime_participation_allows_cold_start_full_size(self, monkeypatch):
        learner = AdaptiveLearner()

        monkeypatch.setattr(CONFIG.risk, "regime_soft_participation_enabled", True)
        monkeypatch.setattr(CONFIG.risk, "regime_soft_min_trades", 12)

        self._record_trades(learner, "Strat", "trending_up", 2, 6)
        mult = learner.strategy_participation_multiplier(
            strategy="Strat",
            regime="trending_up",
            direction="LONG",
        )

        assert mult == pytest.approx(1.0)


@pytest.mark.unit
class TestCorrelationTracker:
    def test_init(self):
        ct = CorrelationTracker()
        assert ct.concentration_penalty([]) == 1.0

    def test_single_symbol(self):
        ct = CorrelationTracker()
        assert ct.concentration_penalty(["BTC/USDT"]) == 1.0

    def test_record_and_finalize(self):
        ct = CorrelationTracker()
        ct.record_bar_outcome("2024-06-01", "BTC/USDT", False)
        ct.record_bar_outcome("2024-06-01", "ETH/USDT", False)
        ct.finalize_bar("2024-06-01")

        key = "BTC/USDT::ETH/USDT"
        assert key in ct._co_loss
        assert ct._co_loss[key] > 0.0

    def test_co_loss_ema_decays(self):
        ct = CorrelationTracker()
        # First bar: both lose
        ct.record_bar_outcome("2024-06-01", "BTC/USDT", False)
        ct.record_bar_outcome("2024-06-01", "ETH/USDT", False)
        ct.finalize_bar("2024-06-01")
        val1 = ct._co_loss["BTC/USDT::ETH/USDT"]

        # Second bar: both win (no co-loss)
        ct.record_bar_outcome("2024-06-02", "BTC/USDT", True)
        ct.record_bar_outcome("2024-06-02", "ETH/USDT", True)
        ct.finalize_bar("2024-06-02")
        val2 = ct._co_loss["BTC/USDT::ETH/USDT"]

        assert val2 < val1  # EMA decayed

    def test_concentration_penalty_correlated(self):
        ct = CorrelationTracker()
        # Create highly correlated pair
        for d in range(5):
            ct.record_bar_outcome(f"2024-06-{d + 1:02d}", "BTC/USDT", False)
            ct.record_bar_outcome(f"2024-06-{d + 1:02d}", "ETH/USDT", False)
            ct.finalize_bar(f"2024-06-{d + 1:02d}")

        penalty = ct.concentration_penalty(["BTC/USDT", "ETH/USDT"])
        assert penalty < 1.0  # reduced sizing

    def test_concentration_penalty_uncorrelated(self):
        ct = CorrelationTracker()
        # Alternating wins/losses → low co-loss
        for d in range(5):
            ct.record_bar_outcome(f"2024-06-{d + 1:02d}", "BTC/USDT", d % 2 == 0)
            ct.record_bar_outcome(f"2024-06-{d + 1:02d}", "ETH/USDT", d % 2 != 0)
            ct.finalize_bar(f"2024-06-{d + 1:02d}")

        penalty = ct.concentration_penalty(["BTC/USDT", "ETH/USDT"])
        assert penalty == 1.0  # no penalty

    def test_finalize_bar_single_symbol(self):
        ct = CorrelationTracker()
        ct.record_bar_outcome("2024-06-01", "BTC/USDT", False)
        ct.finalize_bar("2024-06-01")
        # No pairs to compute → no co_loss entries
        assert len(ct._co_loss) == 0

    def test_get_and_load_state(self):
        ct = CorrelationTracker()
        ct._co_loss = {"A::B": 0.5}
        ct._pair_count = {"A::B": 10}

        state = ct.get_state()
        ct2 = CorrelationTracker()
        ct2.load_state(state)
        assert ct2._co_loss["A::B"] == 0.5
        assert ct2._pair_count["A::B"] == 10
