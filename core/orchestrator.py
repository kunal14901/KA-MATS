"""
KA-MATS Cryptoz · Orchestrator
Iknir Capital

Wires all 9 agents into a single pipeline for live paper trading.

Decision flow (per symbol, per bar):
  Data → AltData → MarketAnalyst → Thesis → Knowledge → Strategy
       → Adversarial → Risk → Execution

Run:
    python main.py

Architecture notes:
  - Deterministic core: Strategy Agent uses pure numerical rules, no LLM
  - Knowledge modifies confidence, never generates trades
  - Risk Manager has absolute veto — no trade skips it
  - Adaptive Learner accumulates regime-partitioned win-rate data in real-time
  - BM25 experience memory writes reflections after each closed trade

Agents that are optional in live mode:
  - ThesisAgent: uses rule-based SA conviction scoring (no external data needed)
  - AdversarialAgent: pure rule-based; slightly reduces false positives
  - KnowledgeAgent: FAISS + BM25; requires knowledge/ directory with papers
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from loguru import logger

from agents.adversarial_agent import AdversarialAgent
from agents.alt_data_agent import AltDataAgent
from agents.data_agent import CryptoDataAgent
from agents.execution_agent import CryptoPaperExecution
from agents.knowledge_agent import CryptoKnowledgeAgent
from agents.live_execution import LiveExecution
from agents.llm_validator import LLMValidator
from agents.market_analyst import MarketAnalystAgent
from agents.onchain_agent import OnChainAgent, OnChainContext
from agents.risk_manager import BayesianEVFilter, CryptoRiskManager
from agents.strategy_agent import CryptoStrategyAgent
from agents.thesis_agent import ThesisAgent
from config.settings import CONFIG, CRYPTO_SYMBOLS
from core.adaptive_learner import AdaptiveLearner
from core.alerts import get_alert_manager
from core.bm25_memory import BM25Memory
from core.health import HealthStatus, get_health_monitor
from core.metrics import get_metrics
from core.models import (
    AltDataContext,
    CandidateSignal,
    Features,
    Indicators,
    KnowledgeContext,
    MarketSnapshot,
    PriceData,
    RegimeAnalysis,
    SignalAssessment,
    ThesisContext,
)
from core.pipeline_router import PipelineRouter
from core.reflection_agent import ReflectionAgent
from core.shadow_logger import ShadowLogger
from core.strategy_ensemble import StrategyEnsemble
from core.swarm_voter import SwarmVoter
from exchanges import create_exchange

# ── Snapshot builder ─────────────────────────────────────────


def _build_snapshot(sym: str, df: pd.DataFrame) -> MarketSnapshot | None:
    """Build a MarketSnapshot from the most-recent bar of a computed DataFrame."""
    if df is None or df.empty:
        return None

    row = df.iloc[-1]
    ts = df.index[-1]
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime().replace(tzinfo=UTC)

    def _f(col: str) -> float | None:
        v = row.get(col, float("nan"))
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    price = PriceData(
        symbol=sym,
        timestamp=ts,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )
    ind = Indicators(
        ema_20=_f("ema_20"),
        ema_50=_f("ema_50"),
        ema_200=_f("ema_200"),
        rsi_14=_f("rsi_14"),
        atr_14=_f("atr_14"),
        bb_upper=_f("bb_upper"),
        bb_lower=_f("bb_lower"),
        adx_14=_f("adx"),
        plus_di=_f("plus_di"),
        minus_di=_f("minus_di"),
        high_20=_f("high_20"),
        macd=_f("macd"),
        macd_signal=_f("macd_signal"),
    )
    # 20-bar return — required by BearShort v2's BTC active-decline gate
    returns_20d: float | None = None
    if len(df) >= 21:
        prev = float(df["close"].iloc[-21])
        if prev > 0:
            returns_20d = float(row["close"]) / prev - 1.0

    feat = Features(
        volume_ratio=_f("volume_ratio"),
        dollar_volume_20d=_f("dollar_volume_20d"),
        zscore_20=_f("zscore"),
        returns_20d=returns_20d,
        cross_rank=None,  # injected by orchestrator after cross-rank calculation
    )
    return MarketSnapshot(
        symbol=sym,
        price=price,
        indicators=ind,
        features=feat,
        timestamp=ts,
        data_quality_ok=True,
    )


# ── Cross-rank helper ─────────────────────────────────────────


def _cross_ranks(snapshots: dict[str, MarketSnapshot], lookback_bars: int = 20) -> dict[str, float]:
    """
    Compute cross-sectional momentum rank across all symbols from the
    most recent `lookback_bars` bar returns embedded in each snapshot.

    With live data we only have the latest bar; approximate 20-bar return
    from the snapshot's features.zscore_20 as a rank proxy until a proper
    rolling window is maintained between bars.

    For a proper implementation the orchestrator should maintain a short
    price history deque per symbol and call this with actual returns.
    """
    approx_returns: dict[str, float] = {}
    for sym, snap in snapshots.items():
        feat = snap.features
        if feat and feat.cross_rank is not None:
            approx_returns[sym] = feat.cross_rank
        elif feat and feat.zscore_20 is not None:
            approx_returns[sym] = float(feat.zscore_20)

    if len(approx_returns) < 2:
        return {}
    sorted_syms = sorted(approx_returns, key=lambda s: approx_returns[s])
    n = len(sorted_syms)
    return {s: i / (n - 1) for i, s in enumerate(sorted_syms)}


# ─────────────────────────────────────────────────────────────
#  CRYPTO ORCHESTRATOR
# ─────────────────────────────────────────────────────────────


class CryptoOrchestrator:
    """
    Full 9-agent pipeline for live paper trading.

    Usage:
        orch = CryptoOrchestrator()
        orch.run(symbols=CRYPTO_SYMBOLS, poll_seconds=14_400)  # 4h bars
    """

    def __init__(self, symbols: list[str] = None) -> None:
        self.symbols = symbols or CRYPTO_SYMBOLS
        cfg = CONFIG

        # ── Agents ────────────────────────────────────────────
        self.data_agent = CryptoDataAgent()

        # Optional WebSocket data layer (push-based, ~5ms vs ~200ms REST poll).
        # Enabled via USE_WEBSOCKET_DATA=1; REST stays as automatic fallback.
        self.ws_bridge = None
        if cfg.data.use_websocket:
            try:
                from agents.ws_data_agent import WSDataBridge

                self.ws_bridge = WSDataBridge(symbols=self.symbols)
                self.ws_bridge.start(warmup=True)
                logger.info("[Orchestrator] WebSocket data layer ACTIVE (ccxt.pro)")
            except Exception as e:
                logger.warning(f"[Orchestrator] WebSocket init failed, using REST: {e}")
                self.ws_bridge = None
        self.market_analyst = MarketAnalystAgent()
        self.alt_data_agent = AltDataAgent()
        self.thesis_agent = ThesisAgent()
        self.adversarial_agent = AdversarialAgent()

        # Adaptive learning (persisted across sessions) — init BEFORE agents that use it
        self.learner = AdaptiveLearner()
        self.learner.load()

        # Strategy agent (learner wired in for signal-level confidence adjustment)
        self.strategy_agent = CryptoStrategyAgent(learner=self.learner)

        # Knowledge agent (FAISS + BM25)
        self.knowledge_agent = CryptoKnowledgeAgent()

        # BM25 experience memory + post-trade reflection
        self.bm25_memory = BM25Memory()
        self.reflection_agent = ReflectionAgent(memory=self.bm25_memory)

        # Risk + Execution (learner wired into both for adaptive sizing + ATR adjustment)
        self.risk_manager = CryptoRiskManager(learner=self.learner)

        # Execution mode: paper (local) or live (exchange-connected)
        exec_mode = cfg.execution.mode
        if exec_mode == "paper":
            self.executor = CryptoPaperExecution(learner=self.learner)
        else:
            exchange = create_exchange(
                mode=exec_mode,
                api_key=cfg.execution.api_key,
                api_secret=cfg.execution.api_secret,
            )
            self.executor = LiveExecution(
                exchange=exchange,
                learner=self.learner,
            )
            logger.info(f"[Orchestrator] Live execution mode: {exec_mode}")

        recover_state = os.getenv("EXECUTION_RECOVER_STATE", "0") == "1"
        if recover_state:
            self.executor.load_state()

        # Monitoring and alerting
        self.metrics = get_metrics()
        self.alerts = get_alert_manager()
        self.health = get_health_monitor()

        # Shadow mode: log raw vs filtered signals for Adversarial/Knowledge audit
        self.shadow = ShadowLogger(log_dir="logs/shadow")

        # ── v6 Vertus-inspired enhancements ──────────────────
        # On-chain flow agent
        self.onchain_agent = OnChainAgent(cfg=None)  # uses defaults from OnChainConfig
        self._last_onchain_context: OnChainContext | None = None

        # LLM veto validator
        llm_cfg_data = cfg.llm_veto
        from agents.llm_validator import LLMValidatorConfig as LLMCfg

        llm_cfg = LLMCfg(
            enabled=llm_cfg_data.enabled,
            backend=llm_cfg_data.backend,
            model=llm_cfg_data.model,
            ollama_url=llm_cfg_data.ollama_url,
            openai_api_key=llm_cfg_data.openai_api_key,
            anthropic_api_key=llm_cfg_data.anthropic_api_key,
            timeout_seconds=llm_cfg_data.timeout_seconds,
            confidence_veto_threshold=llm_cfg_data.confidence_veto_threshold,
        )
        self.llm_validator = LLMValidator(cfg=llm_cfg)

        # Dynamic pipeline router
        self.pipeline_router = PipelineRouter()
        self._pipeline_enabled = cfg.pipeline.enabled

        # Strategy ensemble (genetic selection)
        self.ensemble = StrategyEnsemble()
        if cfg.ensemble.enabled and not self.ensemble.load():
            self.ensemble.initialize()

        # Bayesian EV filter
        self.bayesian_ev = BayesianEVFilter(
            ev_threshold=cfg.bayesian_ev.ev_threshold if cfg.bayesian_ev.enabled else 0.0
        )
        self._bayesian_ev_enabled = cfg.bayesian_ev.enabled

        # Swarm Voter — multi-agent consensus gate (AutoHedge-inspired quorum voting)
        # Aggregates: AdversarialAgent + LLMValidator + BayesianEV + RegimeAlignment + ConfidenceGate
        # Requires 3/5 weighted votes to approve a trade entry.
        self.swarm_voter = SwarmVoter()

        # Track closed trades for reflection + WR gate
        self._prev_closed: int = len(self.executor.portfolio.closed_trades)

        # Graceful degradation state
        self._symbol_failures: dict[str, int] = {}
        self._symbol_cooldown_until: dict[str, datetime] = {}
        self._last_good_snapshot: dict[str, MarketSnapshot] = {}
        self._last_good_alt_data: AltDataContext | None = None
        self._last_good_alt_data_at: datetime | None = None
        self._last_good_thesis: dict[str, ThesisContext] = {}
        self._last_good_thesis_at: dict[str, datetime] = {}
        self._last_good_knowledge: dict[str, KnowledgeContext] = {}
        self._last_good_knowledge_at: dict[str, datetime] = {}

        # Adaptive Learner monitoring: report every 30 bars (~1 month on daily)
        self._bar_count: int = 0
        self._LEARNER_REPORT_INTERVAL: int = 30

        logger.info(f"[Orchestrator] Initialized | {len(self.symbols)} symbols | mode={cfg.execution.mode}")

    # ─────────────────────────────────────────────────────────
    #  SINGLE BAR CYCLE
    # ─────────────────────────────────────────────────────────

    def run_bar(self) -> None:
        """
        Execute one full pipeline cycle — equivalent to processing one 4h bar.

        Called automatically by run() on each poll interval.
        """
        now = datetime.now(UTC)
        logger.info(f"[Orchestrator] ── Bar {now.strftime('%Y-%m-%d %H:%M')} UTC ──")
        skip_stats: Counter[str] = Counter()

        # ── 1. Data Agent: fetch latest bars for each symbol ──
        # Fetch 300 bars for full indicator warmup (EMA200 needs 200+ bars).
        snapshots: dict[str, MarketSnapshot] = {}
        degraded_symbols: set[str] = set()
        for sym in self.symbols:
            if self._in_symbol_cooldown(sym, now):
                stale = self._get_stale_snapshot(sym, now)
                if stale is not None:
                    snapshots[sym] = stale
                    degraded_symbols.add(sym)
                    logger.warning(
                        f"[Orchestrator] {sym}: using stale snapshot during cooldown "
                        f"until {self._symbol_cooldown_until[sym].strftime('%H:%M:%S')}"
                    )
                continue

            fetch_start = time.time()
            try:
                df = None
                if self.ws_bridge is not None:
                    # Push-updated buffer — already includes indicators
                    df = self.ws_bridge.get_dataframe(sym)
                if df is None:
                    df = self.data_agent.fetch_ohlcv(sym, limit=300)
                self.metrics.record_api_call(
                    "fetch_ohlcv", (time.time() - fetch_start) * 1000, success=df is not None
                )
                if df is not None and len(df) >= CONFIG.data.warmup_bars:
                    self.metrics.record_data_quality(sym, len(df), CONFIG.data.warmup_bars)
                    df = self.data_agent.compute_indicators(df)
                    snap = _build_snapshot(sym, df)
                    if snap:
                        self._symbol_failures[sym] = 0
                        self._symbol_cooldown_until.pop(sym, None)
                        self._last_good_snapshot[sym] = snap.model_copy(deep=True)
                        snapshots[sym] = snap
                else:
                    self._handle_symbol_fetch_failure(sym, now, "missing/insufficient bars")
                    stale = self._get_stale_snapshot(sym, now)
                    if stale is not None:
                        snapshots[sym] = stale
                        degraded_symbols.add(sym)
            except Exception as e:
                logger.warning(f"[Orchestrator] Data fetch failed for {sym}: {e}")
                self.metrics.record_api_call("fetch_ohlcv", (time.time() - fetch_start) * 1000, success=False)
                self.metrics.record_error("data_fetch", "data_agent")
                self._handle_symbol_fetch_failure(sym, now, str(e))
                stale = self._get_stale_snapshot(sym, now)
                if stale is not None:
                    snapshots[sym] = stale
                    degraded_symbols.add(sym)

        if not snapshots:
            skip_stats["no_data"] = len(self.symbols)
            logger.warning("[Orchestrator] No data received — skipping bar")
            logger.info(f"[Orchestrator] Skip summary: {self._format_skip_summary(skip_stats)}")
            return

        # ── 2. Alt Data Agent: market-wide context ───────────
        # Pass a representative snapshot (first available symbol) so that the
        # per-symbol trending signal is included in the global context.
        # The trending check is per-symbol but the global context is reused for
        # all symbols; trending signals for other symbols are advisory-only anyway.
        try:
            _ref_snapshot = next(iter(snapshots.values()), None)
            alt_data: AltDataContext | None = self.alt_data_agent.get_context(_ref_snapshot)
            if alt_data is not None and alt_data.data_quality_ok:
                self._last_good_alt_data = alt_data
                self._last_good_alt_data_at = now
            else:
                # API(s) failed — use last known good context as fallback
                fallback = self._get_fallback_alt_data(now)
                alt_data = fallback if fallback is not None else alt_data
        except Exception as e:
            logger.debug(f"[Orchestrator] AltData unavailable: {e}")
            alt_data = self._get_fallback_alt_data(now)

        # ── 2b. On-chain flow agent (v6) ─────────────────────
        onchain_ctx: OnChainContext | None = None
        try:
            if CONFIG.onchain.enabled:
                onchain_ctx = self.onchain_agent.get_context(list(snapshots.keys()))
                self._last_onchain_context = onchain_ctx
                if onchain_ctx and onchain_ctx.signals:
                    logger.info(f"[Orchestrator] On-chain: {onchain_ctx.advisory_note}")
        except Exception as e:
            logger.debug(f"[Orchestrator] OnChain unavailable: {e}")
            onchain_ctx = self._last_onchain_context

        # ── 3. Cross-sectional ranks (injected into features) ─
        ranks = _cross_ranks(snapshots)
        for sym, snap in snapshots.items():
            if sym in ranks and snap.features:
                snap.features.cross_rank = ranks[sym]

        # ── 4. Update stops/targets for open positions ────────
        prices = {sym: snap.price.close for sym, snap in snapshots.items()}
        self.executor.update_prices(prices, now)

        # ── 4b. v18: Feed equity + BTC price to risk manager for portfolio-level scaling
        self.executor.portfolio.compute_equity()
        self.risk_manager.record_equity(self.executor.portfolio.net_equity)
        btc_syms = [s for s in prices if "BTC" in s.upper()]
        if btc_syms:
            self.risk_manager.record_btc_price(prices[btc_syms[0]])

        # ── 4c. v6: Cross-asset return correlation tracking ───
        for sym, snap in snapshots.items():
            if snap.price and snap.price.close and hasattr(snap.price, "open") and snap.price.open > 0:
                bar_ret = (snap.price.close - snap.price.open) / snap.price.open
                self.learner.correlation_tracker.record_bar_return(sym, bar_ret)
        self.learner.correlation_tracker.update_correlations()

        # ── 5. Post-trade reflection for newly-closed trades ──
        closed = self.executor.portfolio.closed_trades
        new_trades_this_bar = len(closed) > self._prev_closed
        if new_trades_this_bar:
            for trade in closed[self._prev_closed :]:
                # WR gate
                self.strategy_agent.record_trade_outcome(trade.strategy_name, trade.pnl > 0)
                # v21: Half-Kelly sizing tracker (live parity with backtest)
                self.risk_manager.record_trade_outcome(trade.strategy_name, trade.pnl)
                # Adaptive learner
                self.learner.record_outcome(
                    symbol=trade.symbol,
                    strategy=trade.strategy_name,
                    regime=trade.regime,
                    pnl=trade.pnl,
                    exit_reason=trade.exit_reason,
                    trade_date=str(trade.exit_time)[:10],
                )
                # BM25 reflection
                try:
                    self.reflection_agent.reflect(trade)
                except Exception as e:
                    logger.debug(f"[Orchestrator] Reflection error: {e}")
                self.metrics.record_trade_closed(
                    symbol=trade.symbol,
                    strategy=trade.strategy_name,
                    pnl=trade.pnl,
                    exit_reason=trade.exit_reason,
                )
                # v6: Bayesian EV posterior update
                if self._bayesian_ev_enabled and trade.regime:
                    entry_price = getattr(trade, "entry_price", 0)
                    pnl_pct = trade.pnl / entry_price if entry_price > 0 else 0
                    self.bayesian_ev.record_trade(
                        strategy_name=trade.strategy_name,
                        regime=trade.regime,
                        pnl_pct=pnl_pct,
                        won=trade.pnl > 0,
                    )
                # v6: Ensemble trade recording
                if CONFIG.ensemble.enabled:
                    self.ensemble.record_trade(
                        genome_id=f"baseline_{trade.strategy_name.lower().replace(' ', '_')}",
                        pnl=trade.pnl,
                        regime=trade.regime or "unknown",
                        strategy_name=trade.strategy_name,
                    )
            # Finalize correlation tracker for all trades closed on this bar
            self.learner.correlation_tracker.finalize_bar(str(now)[:10])
            self._prev_closed = len(closed)

            # After each trade closes: run recovery suppression check immediately
            suppressed = self.learner.check_recovery_suppression()
            if suppressed:
                logger.warning(
                    f"[Orchestrator] Adaptive Learner recovery suppression check: "
                    f"{len(suppressed)} potential issue(s)"
                )
                for w in suppressed:
                    logger.warning(f"[Orchestrator]   {w['message']}")

        system_health = HealthStatus.HEALTHY
        try:
            system_health = self.health.check_all()
            if system_health != HealthStatus.HEALTHY:
                logger.warning(f"[Orchestrator] System health is {system_health.value}")
        except Exception as e:
            logger.debug(f"[Orchestrator] Health check error: {e}")

        if system_health == HealthStatus.UNHEALTHY:
            skip_stats["system_unhealthy"] = len(snapshots)
            logger.warning("[Orchestrator] New entries paused due to unhealthy status")
            logger.info(f"[Orchestrator] Skip summary: {self._format_skip_summary(skip_stats)}")
            self._record_portfolio_metrics()
            self._run_alert_checks()
            return

        # ── 5c. v8 BTC Golden Cross gate (Phase 0b fidelity fix) ──────────────
        # Require BTC EMA50 > EMA200 before generating any new long entries.
        # Mirrors the backtest engine's V8_GOLDEN_CROSS_ENABLED=True (the single
        # biggest OOS improvement — blocked early-recovery chop in 2023-H1).
        # When BTC macro conditions are not met, block trades for all symbols.
        btc_golden_cross: bool = True  # default open; becomes False when BTC EMA50 < EMA200
        btc_rollover: bool | None = None  # BearShort v2: BTC EMA20 < EMA50
        btc_roc20: float | None = None  # BearShort v2: BTC 20-bar return
        btc_snap = snapshots.get("BTC/USDT") or snapshots.get("BTC-USD")
        if btc_snap and btc_snap.indicators:
            btc_ema20 = btc_snap.indicators.ema_20
            btc_ema50 = btc_snap.indicators.ema_50
            btc_ema200 = btc_snap.indicators.ema_200
            if btc_ema50 is not None and btc_ema200 is not None and btc_ema200 > 0:
                btc_golden_cross = btc_ema50 > btc_ema200
                if not btc_golden_cross:
                    logger.warning(
                        f"[Orchestrator] v8 BTC Golden Cross INACTIVE — "
                        f"EMA50={btc_ema50:,.0f} < EMA200={btc_ema200:,.0f}. "
                        f"New LONG entries blocked (all symbols); SHORT candidates still evaluated."
                    )
            if btc_ema20 is not None and btc_ema50 is not None:
                btc_rollover = btc_ema20 < btc_ema50
            if btc_snap.features:
                btc_roc20 = btc_snap.features.returns_20d

        # BearShort v2 macro context — injected each bar; strategy fails closed
        # without it, so shorts can never fire on stale/missing BTC data.
        self.strategy_agent.set_macro_context(btc_rollover=btc_rollover, btc_roc20=btc_roc20)

        # ── 5b. v6 Dynamic Pipeline Routing ─────────────────
        pipeline_config = None
        if self._pipeline_enabled:
            drawdown_pct = (self.executor.portfolio.peak_equity - self.executor.portfolio.net_equity) / max(
                self.executor.portfolio.peak_equity, 1.0
            )
            avg_flow_bias = onchain_ctx.flow_bias if onchain_ctx else 0.0
            pipeline_config = self.pipeline_router.route(
                regime="unknown",
                drawdown_pct=drawdown_pct,
                flow_bias=avg_flow_bias,
                volatility_pct=0.50,
                macro_mode="bull",
                open_positions=len(self.executor.portfolio.positions),
            )

        # ── 6. Per-symbol signal pipeline ─────────────────────
        for sym, snapshot in snapshots.items():
            if sym in self.executor.portfolio.positions:
                skip_stats["already_holding"] += 1
                continue  # already holding this symbol

            try:
                self._process_symbol(
                    sym,
                    snapshot,
                    alt_data,
                    degraded=(sym in degraded_symbols),
                    bar_time=now,
                    onchain_ctx=onchain_ctx,
                    pipeline_config=pipeline_config,
                    btc_golden_cross=btc_golden_cross,
                    skip_stats=skip_stats,
                )
            except Exception as e:
                skip_stats["process_error"] += 1
                logger.warning(f"[Orchestrator] Error processing {sym}: {e}")

        # ── 7. Portfolio status ────────────────────────────────
        eq = self.executor.portfolio.net_equity
        n_pos = len(self.executor.portfolio.positions)
        n_trades = len(self.executor.portfolio.closed_trades)
        logger.info(f"[Orchestrator] Equity=${eq:,.2f} | Positions={n_pos} | ClosedTrades={n_trades}")
        logger.info(f"[Orchestrator] Skip summary: {self._format_skip_summary(skip_stats)}")
        self._record_portfolio_metrics()
        self._run_alert_checks()

        # ── 8. Periodic Adaptive Learner monitoring report ────
        self._bar_count += 1
        if self._bar_count % self._LEARNER_REPORT_INTERVAL == 0:
            self.learner.log_monitoring_report(equity=float(eq))

    def _process_symbol(
        self,
        sym: str,
        snapshot: MarketSnapshot,
        alt_data: AltDataContext | None,
        degraded: bool = False,
        bar_time: datetime | None = None,
        onchain_ctx: OnChainContext | None = None,
        pipeline_config=None,
        btc_golden_cross: bool = True,
        skip_stats: Counter[str] | None = None,
    ) -> None:
        """Run the full agent pipeline for a single symbol (v6 enhanced)."""
        now = bar_time or datetime.now(UTC)

        def record(reason: str) -> None:
            if skip_stats is not None:
                skip_stats[reason] += 1

        # ── 3a. Market Analyst: regime detection ─────────────
        regime: RegimeAnalysis = self.market_analyst.analyse(snapshot)

        # ── Dynamic pipeline: check if agents should be skipped ──
        skip_thesis = pipeline_config and not self.pipeline_router.should_run_agent(
            "thesis_agent", pipeline_config
        )
        skip_knowledge = pipeline_config and not self.pipeline_router.should_run_agent(
            "knowledge_agent", pipeline_config
        )

        # ── 4. Thesis Agent: SA conviction scoring ────────────
        thesis = None
        if not skip_thesis:
            try:
                thesis = self.thesis_agent.score(snapshot, regime)
                if thesis is not None:
                    self._last_good_thesis[sym] = thesis
                    self._last_good_thesis_at[sym] = datetime.now(UTC)
            except Exception:
                thesis = self._get_fallback_thesis(sym)
        else:
            thesis = self._get_fallback_thesis(sym)

        # ── 5. Strategy Agent: generate candidate signals ─────
        # Runs BEFORE Knowledge so the RAG/BM25 query receives real signal
        # context (direction, strategy name) for better retrieval relevance.
        #
        # v8 Golden Cross gate: if BTC EMA50 < EMA200 (early recovery / bear market),
        # block new LONG entries for every symbol. This is the single biggest OOS
        # improvement from the v8 backtest (see Section 14.3 in the doc).
        #
        # IMPORTANT: the gate is direction-aware. Shorts (CryptoBearShort) are
        # designed to fire precisely when BTC macro trend is broken — blocking
        # them here was the root cause of 0 short trades in every bear market.
        cross_rank = snapshot.features.cross_rank if snapshot.features else None
        signals: list[CandidateSignal] = self.strategy_agent.evaluate(snapshot, regime, cross_rank)

        if not btc_golden_cross:
            short_signals = [s for s in signals if s.direction.value.upper() in ("SELL", "SHORT")]
            if not short_signals:
                record("btc_golden_cross")
                return  # longs blocked, no short candidates this bar
            signals = short_signals
            logger.info(
                f"[Orchestrator] {sym}: golden cross inactive — "
                f"{len(signals)} SHORT candidate(s) pass the directional gate"
            )

        if not signals:
            record(self._strategy_reject_reason())
            return

        # Regime kill switch: block strategies with negative edge in current regime.
        # Applied before adversarial/risk so blocked signals do not consume pipeline budget.
        learner = getattr(self, "learner", None)
        if learner is not None:
            filtered_signals: list[CandidateSignal] = []
            for sig in signals:
                if learner.is_strategy_blocked(
                    strategy=sig.strategy_name,
                    regime=regime.regime.value,
                    direction=sig.direction.value,
                ):
                    logger.debug(
                        f"[Orchestrator] Regime kill switch blocked {sym} {sig.strategy_name} "
                        f"{sig.direction.value} in {regime.regime.value}"
                    )
                    continue
                filtered_signals.append(sig)
            signals = filtered_signals
            if not signals:
                record("regime_kill_switch")
                return

        # ── 6. Knowledge Agent: research-based advisory ───────
        # Queried AFTER strategy so RAG/BM25 retrieval has real signal direction
        # and strategy name as context — materially improves retrieval quality.
        knowledge = None
        if not skip_knowledge:
            try:
                knowledge = self.knowledge_agent.query(
                    regime=regime,
                    signals=signals,
                    snapshot=snapshot,
                    alt_data=alt_data,
                )
                if knowledge is not None:
                    self._last_good_knowledge[sym] = knowledge
                    self._last_good_knowledge_at[sym] = datetime.now(UTC)
            except Exception as e:
                logger.debug(f"[Orchestrator] KnowledgeAgent error for {sym}: {e}")
                knowledge = self._get_fallback_knowledge(sym)
        else:
            knowledge = self._get_fallback_knowledge(sym)

        if degraded:
            penalty = CONFIG.data.degraded_confidence_penalty
            for sig in signals:
                sig.confidence = max(0.0, sig.confidence - penalty)
            logger.warning(f"[Orchestrator] {sym}: degraded mode active, confidence penalty {penalty:.2f}")

        # v6: On-chain flow modifier
        if onchain_ctx is not None and onchain_ctx.signals:
            flow_bias = onchain_ctx.flow_bias
            for sig in signals:
                if sig.direction.value == "buy" and flow_bias < -0.25:
                    sig.confidence = max(0.0, sig.confidence - min(0.10, abs(flow_bias) * 0.10))
                elif sig.direction.value == "buy" and flow_bias > 0.25:
                    sig.confidence = min(1.0, sig.confidence + min(0.05, flow_bias * 0.05))

        # v6: Pipeline confidence floor
        if pipeline_config is not None:
            signals = [s for s in signals if s.confidence >= pipeline_config.confidence_floor]
            if not signals:
                record("pipeline_confidence")
                return

        # Shadow: stamp raw confidence on each signal before any modification
        for sig in signals:
            sig._raw_confidence = sig.confidence  # type: ignore[attr-defined]

        # Apply knowledge modifier to signal confidence
        knowledge_modifier = knowledge.confidence_modifier if knowledge else 0.0
        if knowledge_modifier != 0.0:
            for sig in signals:
                sig.confidence = max(0.0, min(1.0, sig.confidence + knowledge_modifier))
                logger.debug(
                    f"[Orchestrator] Knowledge mod {knowledge_modifier:+.3f} "
                    f"applied to {sym} {sig.strategy_name}"
                )

        # Shadow: snapshot raw state after knowledge modifier, before adversarial
        raw_shadow = self.shadow.snapshot_raw(
            symbol=sym,
            signals=signals,
            regime_label=regime.regime.value,
            knowledge_modifier=knowledge_modifier,
        )

        # ── 7. Adversarial Agent: stress-test signals ─────────
        try:
            assessments: list[SignalAssessment] = self.adversarial_agent.assess(
                signals=signals,
                snapshot=snapshot,
                regime=regime,
                thesis=thesis,
                knowledge=knowledge,
                alt_data=alt_data,
            )
        except Exception as e:
            logger.debug(f"[Orchestrator] AdversarialAgent error: {e}")
            # If adversarial fails, pass signals through unchecked
            assessments = []

        # Map assessment results back to signals
        if assessments:
            assessment_map = {a.signal_id: a for a in assessments}
            surviving_signals = []
            for sig in signals:
                assessment = assessment_map.get(sig.signal_id)
                if assessment is None:
                    surviving_signals.append(sig)
                    continue
                if assessment.verdict.value == "fail":
                    logger.debug(
                        f"[Orchestrator] {sym} {sig.strategy_name} FAILED adversarial: "
                        f"{assessment.adversarial_note}"
                    )
                    continue
                if assessment.verdict.value == "flag":
                    sig.confidence = max(0.0, sig.confidence + assessment.confidence_adjustment)
                surviving_signals.append(sig)
            signals = surviving_signals

        # Shadow: write outcomes — one entry per raw signal with verdict + survived flag
        surviving_ids = {str(sig.signal_id) for sig in signals}
        self.shadow.write_outcomes(
            raw_states=raw_shadow,
            surviving_ids=surviving_ids,
            assessments=assessments,
            bar_time=now,
        )

        if not signals:
            record("adversarial_reject")
            return

        # ── 8. Risk Manager: sizing + veto ────────────────────
        for signal in signals:
            # v6: LLM veto validation layer (fail-open if unavailable)
            if self.llm_validator and CONFIG.llm_veto.enabled:
                fear_greed = None
                try:
                    if alt_data and getattr(alt_data, "signals", None):
                        for ads in alt_data.signals:
                            if getattr(ads, "signal_type", "") == "fear_greed_extreme":
                                fear_greed = int(getattr(ads, "value", 0))
                                break
                except Exception:
                    fear_greed = None

                veto = self.llm_validator.validate(
                    signal_direction=signal.direction.value,
                    symbol=signal.symbol,
                    confidence=signal.confidence,
                    strategy_name=signal.strategy_name,
                    regime=regime.regime.value,
                    regime_confidence=regime.confidence,
                    rsi=snapshot.indicators.rsi_14 or 50.0,
                    adx=snapshot.indicators.adx_14 or 0.0,
                    volume_ratio=snapshot.features.volume_ratio or 1.0,
                    cross_rank=snapshot.features.cross_rank or 0.5,
                    fear_greed=fear_greed,
                    flow_bias=(onchain_ctx.flow_bias if onchain_ctx else 0.0),
                    drawdown_pct=(self.executor.portfolio.peak_equity - self.executor.portfolio.net_equity)
                    / max(self.executor.portfolio.peak_equity, 1.0),
                )
                if veto.vetoed:
                    record("llm_veto")
                    logger.info(f"[Orchestrator] LLM vetoed {sym} {signal.strategy_name}: {veto.reason}")
                    continue

            # v6: Bayesian EV filter
            if self._bayesian_ev_enabled:
                approved, ev_info = self.bayesian_ev.should_take_trade(
                    strategy_name=signal.strategy_name,
                    regime=regime.regime.value,
                )
                if not approved:
                    record("bayesian_ev_reject")
                    logger.info(
                        f"[Orchestrator] Bayesian EV rejected {sym} {signal.strategy_name}: "
                        f"EV={ev_info['ev']:.4f}"
                    )
                    continue

            # ── Swarm consensus vote ───────────────────────────────────
            # Aggregate all prior agent verdicts into a quorum decision before
            # committing risk capital. Fail-open if swarm voting is disabled.
            _adv_verdict = assessment.verdict if assessment else "pass"
            _llm_vetoed = (
                veto.vetoed if (self.llm_validator and CONFIG.llm_veto.enabled and "veto" in dir()) else False
            )
            _llm_enabled = bool(self.llm_validator and CONFIG.llm_veto.enabled)
            _bayes_ok, _bayes_ev = (
                (True, {})
                if not self._bayesian_ev_enabled
                else self.bayesian_ev.should_take_trade(
                    strategy_name=signal.strategy_name,
                    regime=regime.regime.value,
                )
            )
            swarm = self.swarm_voter.vote(
                adversarial_verdict=_adv_verdict,
                llm_vetoed=_llm_vetoed,
                llm_enabled=_llm_enabled,
                bayes_approved=_bayes_ok,
                bayes_has_data=_bayes_ev.get("has_data", False),
                regime=regime.regime.value,
                signal_direction=signal.direction.value,
                confidence=signal.confidence,
            )
            if not swarm.approved:
                record("swarm_reject")
                logger.info(f"[SwarmVoter] REJECT {sym} {signal.strategy_name} | {swarm.summary}")
                continue
            logger.debug(f"[SwarmVoter] APPROVE {sym} | {swarm.summary}")

            decision = self.risk_manager.evaluate(signal, self.executor.portfolio, regime)
            if not decision.approved:
                record("risk_reject")
                continue

            # v6: Dynamic pipeline global sizing multiplier
            if pipeline_config is not None and decision.approved:
                decision.position_size *= pipeline_config.sizing_multiplier
                decision.position_value *= pipeline_config.sizing_multiplier
                decision.risk_amount *= pipeline_config.sizing_multiplier

            # ── 9. Execution Agent ────────────────────────────
            executed = self.executor.execute(
                decision,
                snapshot.price.close,
                regime=regime.regime.value,
                atr_value=snapshot.indicators.atr_14 or 0.0,
            )
            if executed:
                record("opened")
                self.metrics.record_trade_opened(
                    symbol=sym,
                    strategy=signal.strategy_name,
                    regime=regime.regime.value,
                    size=decision.position_size,
                )
                logger.info(
                    f"[Orchestrator] TRADE OPENED: {sym} | "
                    f"{signal.strategy_name} | regime={regime.regime.value} | "
                    f"conf={signal.confidence:.3f}"
                )
                break  # one trade per symbol per bar
            record("execution_reject")

    def _strategy_reject_reason(self) -> str:
        """Return the latest strategy-level rejection reason for diagnostics."""
        reasons: list[str] = []
        for strat in getattr(self.strategy_agent, "_strategies", []):
            reason = getattr(strat, "last_reject_reason", "")
            if reason:
                reasons.append(str(reason))
        if not reasons:
            return "no_signal"

        preferred = [
            "regime_filter",
            "ema_filter",
            "rsi_filter",
            "high_filter",
            "volume_filter",
            "rank_filter",
            "adx_filter",
            "price_filter",
            "confidence_filter",
        ]
        for reason in preferred:
            if reason in reasons:
                return reason
        return reasons[-1]

    @staticmethod
    def _format_skip_summary(skip_stats: Counter[str]) -> str:
        if not skip_stats:
            return "no_skips=0"
        parts = [f"{reason}={count}" for reason, count in skip_stats.most_common()]
        return " | ".join(parts)

    def _record_portfolio_metrics(self) -> None:
        """Capture portfolio metrics for monitoring and alerting."""
        port = self.executor.portfolio
        eq = float(port.net_equity)
        peak = float(port.peak_equity) if port.peak_equity else eq
        dd_pct = 0.0
        if peak > 0:
            dd_pct = max(0.0, (peak - eq) / peak * 100.0)
        self.metrics.record_equity(eq, dd_pct)

        exposure_pct = 0.0
        if eq > 0:
            exposure_pct = (
                sum(abs(pos.get("value", 0.0)) for pos in port.positions.values() if isinstance(pos, dict))
                / eq
                * 100.0
            )
        self.metrics.record_open_positions(len(port.positions), exposure_pct)

    def _run_alert_checks(self) -> None:
        try:
            self.alerts.check_all()
        except Exception as e:
            logger.debug(f"[Orchestrator] Alert check error: {e}")

    def _in_symbol_cooldown(self, symbol: str, now: datetime) -> bool:
        until = self._symbol_cooldown_until.get(symbol)
        return until is not None and now < until

    def _handle_symbol_fetch_failure(self, symbol: str, now: datetime, reason: str) -> None:
        """Track failures and apply exponential cooldown after repeated misses."""
        fails = self._symbol_failures.get(symbol, 0) + 1
        self._symbol_failures[symbol] = fails

        if fails < CONFIG.data.fetch_failures_before_cooldown:
            logger.debug(f"[Orchestrator] {symbol}: fetch failure #{fails} ({reason})")
            return

        exp_step = fails - CONFIG.data.fetch_failures_before_cooldown
        cooldown = CONFIG.data.fetch_cooldown_base_seconds * (2 ** max(0, exp_step))
        cooldown = min(cooldown, CONFIG.data.fetch_cooldown_max_seconds)
        self._symbol_cooldown_until[symbol] = now + pd.Timedelta(seconds=cooldown)
        logger.warning(f"[Orchestrator] {symbol}: cooldown {cooldown}s after {fails} failures ({reason})")

    def _get_stale_snapshot(self, symbol: str, now: datetime) -> MarketSnapshot | None:
        snap = self._last_good_snapshot.get(symbol)
        if snap is None:
            return None
        age = (now - snap.timestamp).total_seconds()
        if age > CONFIG.data.stale_snapshot_max_age_seconds:
            return None
        return snap.model_copy(deep=True)

    def _get_fallback_alt_data(self, now: datetime) -> AltDataContext | None:
        if self._last_good_alt_data is None or self._last_good_alt_data_at is None:
            return None
        age = (now - self._last_good_alt_data_at).total_seconds()
        if age > CONFIG.data.context_fallback_max_age_seconds:
            return None
        logger.warning("[Orchestrator] Using fallback alt-data context")
        return self._last_good_alt_data.model_copy(deep=True)

    def _get_fallback_thesis(self, symbol: str) -> ThesisContext | None:
        ctx = self._last_good_thesis.get(symbol)
        ts = self._last_good_thesis_at.get(symbol)
        if ctx is None or ts is None:
            return None
        if (datetime.now(UTC) - ts).total_seconds() > CONFIG.data.context_fallback_max_age_seconds:
            return None
        logger.warning(f"[Orchestrator] {symbol}: using fallback thesis context")
        return ctx.model_copy(deep=True)

    def _get_fallback_knowledge(self, symbol: str) -> KnowledgeContext | None:
        ctx = self._last_good_knowledge.get(symbol)
        ts = self._last_good_knowledge_at.get(symbol)
        if ctx is None or ts is None:
            return None
        if (datetime.now(UTC) - ts).total_seconds() > CONFIG.data.context_fallback_max_age_seconds:
            return None
        logger.warning(f"[Orchestrator] {symbol}: using fallback knowledge context")
        return ctx.model_copy(deep=True)

    # ─────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ─────────────────────────────────────────────────────────

    def run(
        self,
        poll_seconds: int = 14_400,  # 4 hours = one 4h bar
        max_bars: int | None = None,
    ) -> None:
        """
        Start the live paper trading loop.

        Args:
            poll_seconds: How often to run a bar cycle (default: 4h).
            max_bars:     Stop after this many bars (None = run forever).
        """
        logger.info(
            f"[Orchestrator] Starting live loop | poll={poll_seconds}s "
            f"({'%.1f' % (poll_seconds / 3600)}h per bar)"
        )
        bar_count = 0
        try:
            while True:
                self.run_bar()
                bar_count += 1
                if max_bars and bar_count >= max_bars:
                    logger.info(f"[Orchestrator] Reached max_bars={max_bars}, stopping.")
                    break
                logger.info(f"[Orchestrator] Sleeping {poll_seconds}s until next bar...")
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            logger.warning("[Orchestrator] Interrupted by user")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Save adaptive learner state on exit."""
        self.shadow.close()
        if self.ws_bridge is not None:
            with contextlib.suppress(Exception):
                self.ws_bridge.stop()
        try:
            self.learner.save()
            logger.info("[Orchestrator] Adaptive learner state saved")
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to save learner state: {e}")

        try:
            self.executor.save_state()
            logger.info("[Orchestrator] Execution state saved")
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to save execution state: {e}")

        # Print final portfolio summary
        port = self.executor.portfolio
        eq = port.net_equity
        trades = port.closed_trades
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            total_pnl = sum(t.pnl for t in trades)
            logger.info(
                f"[Orchestrator] FINAL: equity=${eq:,.2f} | "
                f"trades={len(trades)} | WR={wins / len(trades):.0%} | "
                f"PnL={total_pnl:+,.2f}"
            )
