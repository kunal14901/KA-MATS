"""
KA-MATS Cryptoz · Configuration
Iknir Capital — Crypto Paper Trading Engine

Crypto differences from equity KA-MATS:
  - 24/7 markets (no open/close)
  - 5-10× higher ATR volatility → wider stops/targets
  - Leverage available (default 1× for safety, up to 3×)
  - Cross-sectional momentum is even stronger in crypto
  - Mean reversion on BTC/ETH is reliable at RSI extremes
  - No defensive rotation (crypto all correlates in crashes)
  - VolatilityDip is highly effective (panic dips recover fast)
"""

import os
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
#  CRYPTO UNIVERSE
# ─────────────────────────────────────────────────────────────

# Primary trading universe — all vs USDT on Binance
# 15 liquid pairs: large caps + high-momentum mid caps
CRYPTO_SYMBOLS = [
    "BTC/USDT",  # Digital gold — lowest volatility, most liquid
    "ETH/USDT",  # Smart contract platform — strong trends
    "SOL/USDT",  # High-speed chain — extreme momentum
    "BNB/USDT",  # Exchange token — steady
    "AVAX/USDT",  # DeFi platform — strong momentum
    "LINK/USDT",  # Oracle — AI narrative driven
    "DOT/USDT",  # Parachain ecosystem
    "ADA/USDT",  # Large cap alt
    "DOGE/USDT",  # Meme + momentum
    "POL/USDT",  # Polygon (formerly MATIC — rebranded 2024)
    "UNI/USDT",  # DeFi blue chip
    "ATOM/USDT",  # Cosmos ecosystem
    "NEAR/USDT",  # Layer 1
    "ARB/USDT",  # Arbitrum — L2 momentum
    "OP/USDT",  # Optimism — L2 momentum
]

# Momentum universe (for cross-rank — all symbols compete)
MOMENTUM_SYMBOLS = CRYPTO_SYMBOLS  # All crypto is momentum-grade

# Safe haven in crypto context: BTC/ETH (flight-to-quality during altcoin panic)
CRYPTO_SAFE_HAVENS = {"BTC/USDT", "ETH/USDT"}

# Timeframe
TIMEFRAME = "1d"  # Daily bars — matches the validated +1529% backtest exactly


# ─────────────────────────────────────────────────────────────
#  DATA CONFIG
# ─────────────────────────────────────────────────────────────


@dataclass
class DataConfig:
    exchange: str = "binance"  # Exchange for data + paper trading
    timeframe: str = TIMEFRAME
    warmup_bars: int = 250  # Bars needed for indicators (EMA200 needs 200)
    data_quality_min_bars: int = 60
    max_missing_pct: float = 0.02

    # Graceful degradation controls
    fetch_failures_before_cooldown: int = 2
    fetch_cooldown_base_seconds: int = 60
    fetch_cooldown_max_seconds: int = 900
    stale_snapshot_max_age_seconds: int = 86_400  # 24h fallback window for daily bars
    context_fallback_max_age_seconds: int = 43_200  # 12h fallback for alt/thesis/knowledge
    degraded_confidence_penalty: float = 0.05  # subtract from signal confidence on stale data

    # WebSocket streaming (ccxt.pro) — push-based data instead of REST polling.
    # Falls back to REST automatically if the stream drops. Set USE_WEBSOCKET_DATA=1.
    use_websocket: bool = os.getenv("USE_WEBSOCKET_DATA", "0") == "1"


# ─────────────────────────────────────────────────────────────
#  REGIME CONFIG  (same detection, crypto-tuned thresholds)
# ─────────────────────────────────────────────────────────────


@dataclass
class RegimeConfig:
    adx_period: int = 14
    adx_trend_threshold: float = 22.0  # Lower than equity (crypto trends earlier)
    adx_strong_threshold: float = 35.0
    atr_period: int = 14
    volatility_lookback: int = 20
    volatility_high_pct: float = 75.0
    mean_revert_zscore_period: int = 20
    mean_revert_zscore_threshold: float = 1.8  # Slightly wider for crypto


# ─────────────────────────────────────────────────────────────
#  STRATEGY CONFIG
# ─────────────────────────────────────────────────────────────


@dataclass
class StrategyConfig:
    rsi_period: int = 14
    rsi_oversold: float = 32.0  # Slightly more generous than equity
    rsi_overbought: float = 72.0
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    bb_period: int = 20
    bb_std: float = 2.0
    min_signal_confidence: float = 0.50

    # Crypto ATR multipliers — wider than equity (higher daily volatility)
    # Stop unchanged; targets extended for 3:1 R/R minimum
    atr_stop_trending: float = 2.5  # equity: 2.0
    atr_target_trending: float = 7.5  # equity: 6.0 → 3:1 R/R
    atr_stop_ranging: float = 3.0  # equity: 2.5
    atr_target_ranging: float = 7.5  # equity: 6.0
    atr_stop_volatile: float = 3.5  # equity: 3.0
    atr_target_volatile: float = 8.0  # equity: 6.0


# ─────────────────────────────────────────────────────────────
#  RISK CONFIG
# ─────────────────────────────────────────────────────────────


@dataclass
class RiskConfig:
    # Position sizing — calibrated from Phase 0 OOS Kelly analysis (April 2026).
    # OOS win rate 46%, win/loss ratio 1.585 → Full Kelly 11.87%, Half-Kelly 5.93%.
    # Previous 15% was 2.5× Kelly-optimal → unnecessary ruin risk.
    risk_per_trade_pct: float = 0.06  # 6% risk per trade (Half-Kelly from OOS data)
    max_position_pct: float = 0.35  # 35% max per position
    max_portfolio_exposure_pct: float = 1.0  # fully invested in bull mode

    # Crypto leverage (set 1.0 for spot-only — safe for paper trading start)
    leverage: float = 1.0  # 1× = no leverage (spot equivalent)

    # Position limits
    max_open_positions: int = 9  # MAX_POS_BULL from backtest

    # Drawdown controls — tighter for crypto (can drop 40-50% fast)
    max_drawdown_pct: float = 0.15  # 15% max drawdown circuit breaker
    max_daily_loss_pct: float = 0.04  # 4% daily loss limit

    # Stop/target (overridden by strategy's ATR multipliers)
    stop_loss_atr_multiplier: float = 2.5
    take_profit_atr_multiplier: float = 7.5

    # ── Portfolio-level risk controls (v18 — fund-grade DD reduction) ─────
    #
    # 1. Volatility targeting: scale total exposure so portfolio vol ≈ target.
    # v15 CHAMPION SETTING: 45% annualised target — crypto-appropriate, unlike
    # the 20% equity-style target Phase 4 tested (which deleveraged to ~25%
    # exposure and killed returns). The 45% target was grid-validated in
    # backtest/run_v15_grade_a.py: MaxDD 41% -> 30%, OOS Sharpe 0.89 -> 0.96,
    # full-sample Sharpe held at 1.39. Scale capped at 1.0 (no leverage).
    vol_target_enabled: bool = True
    vol_target_annual_pct: float = 0.45  # v15 champion (grid: 45/55/65)
    vol_lookback_bars: int = 20  # realized vol estimation window
    vol_scale_min: float = 0.10  # matches backtest (no floor in model)
    vol_scale_max: float = 1.00  # no leverage — matches backtest cap

    # 2. Regime-based risk scaling: multiply risk by regime factor
    regime_risk_enabled: bool = True
    regime_risk_factors: dict = field(
        default_factory=lambda: {
            "trending_up": 1.20,  # strong trend → lean in
            "trending_down": 0.40,  # bear → minimal new longs
            "volatile": 0.50,  # high noise → cut size
            "ranging": 0.80,  # consolidation → moderate
            "mean_reverting": 0.60,  # stretched → cautious
        }
    )

    # 3. Graduated equity curve feedback: scale risk based on DD depth
    #    Uses a *decaying all-time high* as reference peak.  The reference
    #    decays toward current equity so after a long consolidation the system
    #    resumes normal sizing instead of being permanently penalised post-bull.
    equity_curve_feedback_enabled: bool = True
    equity_curve_tiers: list = field(
        default_factory=lambda: [
            # (dd_threshold, risk_multiplier) — tiered sizing vs decayed-ATH DD.
            (0.10, 1.00),  # DD <10%: full risk
            (0.18, 0.50),  # DD 10-18%: 50% sizing
            (0.25, 0.20),  # DD 18-25%: 20% sizing
            (0.30, 0.00),  # DD ≥30%: halt new entries
        ]
    )
    equity_curve_peak_bars: int = 60  # (legacy — ignored when decay mode active)
    ecf_decay_rate: float = 0.002  # decayed-ATH forgiveness rate per bar
    ecf_decay_floor: float = 0.50  # decayed ATH never drops below 50% of real ATH
    ecf_use_decaying_ath: bool = True  # True = decaying ATH; False = rolling peak

    # 4. BTC-beta correlation penalty: penalize alt exposure during BTC stress
    btc_beta_penalty_enabled: bool = True
    btc_beta_lookback_bars: int = 20
    btc_beta_high_threshold: float = 0.80  # portfolio beta > 0.80 → penalize
    btc_beta_penalty_max: float = 0.40  # max 40% reduction

    # 5. Liquidity / tradeability sizing overlay
    # Keep disabled by default until a dedicated backtest clears it.
    liquidity_sizing_enabled: bool = False
    liquidity_lookback_bars: int = 20
    liquidity_min_dollar_volume: float = 25_000_000.0
    liquidity_full_dollar_volume: float = 250_000_000.0
    liquidity_floor_mult: float = 0.60

    # 6. Regime participation controls
    # Soft participation is the default: weak regime edge cuts size instead of fully blocking.
    regime_soft_participation_enabled: bool = True
    regime_soft_wr_penalty_enabled: bool = False  # learner modifier already penalizes weak WR by default
    regime_soft_min_wr: float = 0.35  # begin penalizing below 35% WR when enabled
    regime_soft_min_trades: int = 12  # need at least 12 trades for reliable signal
    regime_soft_floor_mult: float = 0.20  # deepest WR penalty still keeps 20% size alive
    regime_soft_bear_long_mult: float = 0.25  # structural downweight for longs in bear family

    # 7. Optional hard kill switch (disabled by default)
    # Keep as an emergency brake for research/live incidents, not the baseline model.
    regime_kill_switch_enabled: bool = False
    regime_kill_switch_min_wr: float = 0.35
    regime_kill_switch_min_trades: int = 12
    regime_kill_switch_bear_block_longs: bool = True

    # ── Live capital ramp-up ──────────────────────────────────
    # Start at 4% for first 50 live trades, then scale to full 6% (Half-Kelly).
    # 4% = ~2/3 of Half-Kelly — conservative training wheels for Phase 2.
    #
    # How it works:
    #   trades 0–49   → ramp_initial_risk_pct (4%)
    #   trade 50+     → risk_per_trade_pct    (6%)
    #
    # Set ramp_enabled=False to skip ramp-up (paper trading / backtesting).
    # Set via env var:  RISK_RAMP_ENABLED=0 python main.py   (disables ramp)
    ramp_enabled: bool = field(default_factory=lambda: os.environ.get("RISK_RAMP_ENABLED", "1") != "0")
    ramp_initial_risk_pct: float = 0.04  # 4% during ramp-up (Phase 2 training wheels)
    ramp_target_trades: int = 50  # switch to full risk after this many live trades

    # ── Macro regime filter ───────────────────────────────────
    # When BTC's 20-bar cumulative return falls below this threshold (e.g. -15%),
    # the system is likely in a macro shock / bear transition. Halve new position
    # sizing until BTC recovers. Addresses 2023-H1 and 2025-H1 drawdown periods.
    macro_filter_enabled: bool = True
    macro_filter_btc_return_threshold: float = (
        -0.20
    )  # -20% 20-bar BTC return triggers filter (relaxed from -15%)
    macro_filter_size_mult: float = 0.50  # halve sizing during macro stress


# ─────────────────────────────────────────────────────────────
#  ALT DATA CONFIG — Crypto-native alternative data feeds
# ─────────────────────────────────────────────────────────────


@dataclass
class AltDataConfig:
    # Alternative.me Fear & Greed Index (free, no key required)
    fear_greed_enabled: bool = True

    # CoinGecko global market stats (free tier, no key required)
    coingecko_enabled: bool = True

    # On-chain metrics — disabled by default (requires Glassnode/Dune key)
    onchain_enabled: bool = False

    # Cache TTL to respect free API rate limits
    cache_ttl_minutes: int = 60

    # Optional GitHub token for higher rate limits (on AI lab commit velocity)
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))


# ─────────────────────────────────────────────────────────────
#  ON-CHAIN FLOW CONFIG (v6 — Vertus-inspired)
# ─────────────────────────────────────────────────────────────


@dataclass
class OnChainConfig:
    # v6 STAGE 1: On-chain flow signals (no backtest impact — advisory/sizing only)
    # Enable now: free APIs, fail-open on error, sizing modifier is mild (0.7-1.15x)
    # Disable if API errors appear in logs: set enabled=False
    enabled: bool = True  # on-chain flow agent active
    coinglass_oi_enabled: bool = True  # CoinGlass open interest
    coinglass_funding_enabled: bool = True  # CoinGlass funding rates
    exchange_flow_enabled: bool = True  # exchange flow proxy
    cache_ttl_minutes: int = 30
    oi_spike_pct: float = 10.0
    oi_drop_pct: float = -8.0
    funding_extreme_high: float = 0.05
    funding_extreme_low: float = -0.02
    exchange_inflow_spike_pct: float = 15.0


# ─────────────────────────────────────────────────────────────
#  LLM VALIDATOR CONFIG (v6 — Vertus-inspired)
# ─────────────────────────────────────────────────────────────


@dataclass
class LLMValidatorConfig:
    # v6 STAGE 4: LLM veto layer — KEEP DISABLED until Stage 3 shows no regression.
    # Enable only after ≥30 paper-traded days confirm baseline is maintained.
    # Steps to enable:
    #   1. Install Ollama + pull llama3.2: `ollama pull llama3.2`
    #   2. Set enabled=True, backend="ollama"
    #   3. Watch logs for VETOED events; adjust confidence_veto_threshold if too aggressive
    #   4. Alternatively set backend="openai" with OPENAI_API_KEY env var
    enabled: bool = False  # STAGE 4 — keep False until stage 3 cleared
    backend: str = "ollama"  # "ollama", "openai", "anthropic", "disabled"
    model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    timeout_seconds: int = 15
    confidence_veto_threshold: float = 0.65  # high bar — only high-confidence vetoes
    max_retries: int = 1
    cache_ttl_minutes: int = 10


# ─────────────────────────────────────────────────────────────
#  STRATEGY ENSEMBLE CONFIG (v6 — Vertus-inspired)
# ─────────────────────────────────────────────────────────────


@dataclass
class StrategyEnsembleConfig:
    # v6 STAGE 5: Genetic strategy ensemble — KEEP DISABLED initially.
    # The ensemble needs ≥10 trades per variant to build reliable fitness scores.
    # After 50 live trades: review ensemble.get_summary() and enable if top variants
    # outperform the v5 baselines in regime-specific win rate.
    enabled: bool = False  # STAGE 5 — enable after 50+ live trades
    population_size: int = 50
    top_k_per_strategy: int = 5
    min_trades_for_fitness: int = 10
    evolve_every_n_trades: int = 50
    mutation_rate: float = 0.05
    crossover_rate: float = 0.70


# ─────────────────────────────────────────────────────────────
#  DYNAMIC PIPELINE CONFIG (v6 — Vertus-inspired)
# ─────────────────────────────────────────────────────────────


@dataclass
class DynamicPipelineConfig:
    # v6 STAGE 2: Dynamic pipeline routing — safe to enable immediately.
    # FAST/NORMAL/DEEP modes only affect agent execution order, NOT sizing.
    # CRISIS mode applies 0.5x sizing only at DD>=20% (same as existing ECF tier).
    # No net change to baseline when drawdown < 10% (NORMAL mode = existing behaviour).
    enabled: bool = False  # STAGE 2 — disabled: reduces return vs Stage 1 alone
    dd_crisis_threshold: float = 0.20
    dd_deep_threshold: float = 0.10
    vol_high_percentile: float = 0.75
    flow_extreme_threshold: float = 0.60
    crisis_sizing_mult: float = 0.50
    deep_sizing_mult: float = 0.80


# ─────────────────────────────────────────────────────────────
#  BAYESIAN EV FILTER CONFIG (v6 — Vertus-inspired)
# ─────────────────────────────────────────────────────────────


@dataclass
class BayesianEVConfig:
    # v6 STAGE 3: Bayesian EV filter — enabled but conservative.
    # Safe because:
    #   a) min_observations=15 means it's silent for first 15 trades (fail-open)
    #   b) ev_threshold=0.001 (0.1%) is below typical round-trip costs — allows
    #      most trades through, only blocks strategies with clearly negative EV
    # After 50 trades: review EV estimates in logs and tighten ev_threshold to 0.002
    enabled: bool = False  # STAGE 3 — disabled: no incremental gain over Stage 1
    ev_threshold: float = 0.001  # 0.1% minimum EV (loose — tighten after 50 trades)
    min_observations: int = 15  # trades needed before filter activates
    ema_alpha: float = 0.10  # posterior update speed


# ─────────────────────────────────────────────────────────────
#  EXECUTION CONFIG — Binance Testnet Paper Trading
# ─────────────────────────────────────────────────────────────


@dataclass
class ExecutionConfig:
    mode: str = os.environ.get("EXECUTION_MODE", "paper")
    # paper           = local simulation (no exchange, fastest)
    # binance_testnet = Binance testnet orders (needs testnet API keys)
    # binance_live    = Binance real money
    # bybit_testnet   = Bybit testnet orders
    # bybit_live      = Bybit real money

    # Exchange credentials — load from environment or .env file
    # Unified env vars (work for any exchange):
    api_key: str = os.environ.get("EXCHANGE_API_KEY", "") or os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret: str = os.environ.get("EXCHANGE_API_SECRET", "") or os.environ.get(
        "BINANCE_TESTNET_SECRET", ""
    )

    # Simulated costs (applied in paper mode; live mode uses real exchange fees)
    # Phase 4 champion uses maker (limit) orders for entry → reduced entry cost.
    # Entry: maker fee (0.05%) + minimal impact (~0.05%) = 0.1% total per entry side.
    # Exit: taker fee (0.1%) + stop slippage (~0.1%) = 0.2% total per exit side.
    taker_fee_pct: float = 0.001  # 0.1% taker fee (exits, stop orders)
    maker_fee_pct: float = 0.0005  # 0.05% maker fee (entry limit orders)
    slippage_pct: float = 0.0005  # 0.05% entry slippage (limit order, Phase 4 model)

    # Paper capital
    initial_capital_usdt: float = 10_000.0  # $10K starting capital


# ─────────────────────────────────────────────────────────────
#  SYSTEM CONFIG
# ─────────────────────────────────────────────────────────────


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    alt_data: AltDataConfig = field(default_factory=AltDataConfig)

    # v6 Vertus-inspired enhancements
    onchain: OnChainConfig = field(default_factory=OnChainConfig)
    llm_veto: LLMValidatorConfig = field(default_factory=LLMValidatorConfig)
    ensemble: StrategyEnsembleConfig = field(default_factory=StrategyEnsembleConfig)
    pipeline: DynamicPipelineConfig = field(default_factory=DynamicPipelineConfig)
    bayesian_ev: BayesianEVConfig = field(default_factory=BayesianEVConfig)

    symbols: list[str] = field(default_factory=lambda: CRYPTO_SYMBOLS)
    initial_capital: float = 10_000.0  # USDT

    log_level: str = "INFO"
    log_file: str = "logs/ka_mats_crypto.log"


CONFIG = SystemConfig()
