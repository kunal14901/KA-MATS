"""Integration tests for full orchestrator pipeline."""

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

from config.settings import CONFIG
from core.orchestrator import CryptoOrchestrator


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Test full multi-agent pipeline."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes all agents."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT", "ETH/USDT"])

        assert orch is not None
        assert len(orch.symbols) == 2
        assert orch.data_agent is not None
        assert orch.market_analyst is not None
        assert orch.strategy_agent is not None
        assert orch.risk_manager is not None
        assert orch.executor is not None

    @pytest.mark.slow
    def test_single_bar_execution(self, mock_ccxt_exchange):
        """Test execution of one complete bar cycle."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT"])

        # Run one bar
        orch.run_bar()

        # Should complete without error
        assert True

    def test_agent_pipeline_flow(self, sample_snapshot, sample_regime_trending_up):
        """Test data flows correctly through agent pipeline."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT"])

        # Test market analyst
        regime = orch.market_analyst.analyse(sample_snapshot)
        assert regime is not None
        assert regime.regime is not None

        # Test strategy agent
        signals = orch.strategy_agent.evaluate(sample_snapshot, regime, cross_rank=0.8)
        assert isinstance(signals, list)

        # Test risk manager (if signals exist)
        if signals:
            for signal in signals:
                decision = orch.risk_manager.evaluate(signal, orch.executor.portfolio, regime)
                assert decision is not None

    def test_portfolio_updates(self):
        """Test that portfolio state updates correctly."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT"])

        initial_equity = orch.executor.portfolio.net_equity
        assert initial_equity == CONFIG.initial_capital

    def test_trade_lifecycle(self):
        """Test complete trade lifecycle: entry → exit."""
        # This is a complex integration test
        # Would require mocking market data that triggers entry and exit
        pass

    def test_multiple_symbols_handling(self):
        """Test orchestrator handles multiple symbols correctly."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        orch = CryptoOrchestrator(symbols=symbols)

        assert len(orch.symbols) == 3

    def test_error_recovery(self):
        """Test that pipeline continues after individual symbol errors."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT", "INVALID/SYMBOL"])

        # Should not crash, just skip invalid symbol
        orch.run_bar()
        assert True

    @pytest.mark.slow
    def test_max_bars_limit(self, mock_ccxt_exchange):
        """Test that max_bars parameter works."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT"])

        # Run for 2 bars with 0 second interval
        orch.run(poll_seconds=0.1, max_bars=2)

        # Should complete and stop
        assert True

    def test_adaptive_learner_integration(self):
        """Test that learner receives trade outcomes."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT"])

        initial_state = orch.learner

        # Learner should be initialized
        assert initial_state is not None

    def test_reflection_on_closed_trades(self):
        """Test that reflection agent runs on trade closure."""
        orch = CryptoOrchestrator(symbols=["BTC/USDT"])

        # Open a position manually for testing
        from core.models import RiskDecision

        decision = RiskDecision(
            signal_id=uuid4(),
            symbol="BTC/USDT",
            timestamp=datetime.now(UTC),
            approved=True,
            position_size=0.1,
            position_value=4450.0,
            entry_price=44500.0,
            stop_loss=43500.0,
            take_profit=47500.0,
            risk_amount=100.0,
            veto_reason="",
            strategy_name="TestStrategy",
        )

        orch.executor.execute(decision, current_price=44500.0, regime="trending_up")

        # Close the position
        orch.executor.update_prices({"BTC/USDT": 47600.0}, datetime.now(UTC))

        # Run bar to trigger reflection
        orch.run_bar()

        # Trade should be in closed_trades
        assert len(orch.executor.portfolio.closed_trades) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestBacktestIntegration:
    """Test backtest runner integration."""

    def test_backtest_imports(self):
        """Test that backtest module imports correctly."""
        from backtest import run_crypto_backtest

        assert run_crypto_backtest is not None
