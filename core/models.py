"""
KA-MATS · Core Data Models
Iknir Capital — Phase I Foundation

All inter-agent communication uses these schema-constrained Pydantic models.
Principle: Structured Outputs Only — agents produce validated, typed outputs.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────────────────────


class RegimeType(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    MEAN_REVERTING = "mean_reverting"
    UNKNOWN = "unknown"


class SignalDirection(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class InstrumentType(StrEnum):
    SPOT = "SPOT"
    PERP = "PERP"
    FUTURES = "FUTURES"


# ─────────────────────────────────────────────────────────────
#  DATA AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class PriceData(BaseModel):
    """Latest OHLCV bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Indicators(BaseModel):
    """Computed technical indicators from the Data Agent."""

    # Trend
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None

    # Momentum
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None  # MACD - signal line

    # Volatility
    atr_14: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_pct: float | None = None  # (close - lower) / (upper - lower)

    # Trend strength
    adx_14: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    high_20: float | None = None

    # VWAP (daily-anchored, resets at UTC midnight)
    vwap: float | None = None

    # Keltner Channels (EMA20 ± 1.5×ATR20)
    kc_upper: float | None = None
    kc_lower: float | None = None

    # Squeeze Momentum (LazyBear style)
    squeeze_on: bool | None = None  # True when BB inside KC (volatility coiling)
    squeeze_mom: float | None = None  # momentum oscillator value (+ = bullish, − = bearish)

    # Heikin Ashi fields
    ha_open: float | None = None
    ha_close: float | None = None
    ha_bullish: bool | None = None  # True when HA close > HA open
    ha_no_upper_wick: bool | None = None  # True when bull bar has no upper wick (full body)


class Features(BaseModel):
    """Derived statistical features from the Data Agent."""

    returns_1d: float | None = None
    returns_5d: float | None = None
    returns_20d: float | None = None
    volatility_20d: float | None = None  # Annualised rolling vol
    zscore_20: float | None = None  # (close - SMA20) / rolling_std
    volume_ratio: float | None = None  # volume / avg_volume_20
    dollar_volume_20d: float | None = None  # rolling 20-bar dollar volume proxy
    price_vs_ema200: float | None = None  # (close - EMA200) / EMA200
    spread_bps: float | None = None
    orderbook_depth_usd_1pct: float | None = None
    cross_rank: float | None = None  # 0-1 percentile rank by 20d return among peers (1=best)


class OrderBookSnapshot(BaseModel):
    best_bid: float | None = None
    best_ask: float | None = None
    spread_bps: float | None = None
    depth_usd_1pct: float | None = None


class MarketSnapshot(BaseModel):
    """
    Output of the Data Agent.
    Represents the complete numerical view of a symbol at a point in time.
    """

    symbol: str
    timestamp: datetime
    price: PriceData
    indicators: Indicators
    features: Features
    bars_available: int = 0
    data_quality_ok: bool = True
    quality_notes: list[str] = Field(default_factory=list)
    exchange: str | None = None
    instrument_type: InstrumentType = InstrumentType.SPOT
    mark_price: float | None = None
    funding_rate: float | None = None
    orderbook: OrderBookSnapshot | None = None


# ─────────────────────────────────────────────────────────────
#  MARKET ANALYST AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class RegimeAnalysis(BaseModel):
    """
    Output of the Market Analyst Agent.
    Classifies the current market environment.
    """

    symbol: str
    timestamp: datetime
    regime: RegimeType
    confidence: float = Field(ge=0.0, le=1.0)

    # Supporting metrics
    trend_strength: float | None = None  # ADX value
    volatility_pct: float | None = None  # Normalised vol percentile (0-100)
    zscore: float | None = None  # Price z-score

    # Descriptive explanation
    rationale: str = ""


# ─────────────────────────────────────────────────────────────
#  STRATEGY AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class SignalCondition(BaseModel):
    """A single evaluated rule condition."""

    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    description: str = ""


class CandidateSignal(BaseModel):
    """
    Output of the Strategy Agent.
    A candidate trade signal backed by deterministic rule logic.
    Principle: Signals must originate from measurable numerical conditions.
    """

    signal_id: UUID = Field(default_factory=uuid4)
    symbol: str
    timestamp: datetime
    direction: SignalDirection
    strategy_name: str
    conditions: list[SignalCondition] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    # Execution parameters (set by Strategy Agent, validated by Risk Manager)
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    dollar_volume_20d: float | None = None

    @model_validator(mode="after")
    def validate_all_conditions_passed(self) -> CandidateSignal:
        if self.direction != SignalDirection.HOLD:
            failed = [c.name for c in self.conditions if not c.passed]
            if failed:
                raise ValueError(f"Signal {self.strategy_name} has unmet conditions: {failed}")
        return self


# ─────────────────────────────────────────────────────────────
#  KNOWLEDGE AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class KnowledgeChunk(BaseModel):
    """A retrieved document chunk from the vector store."""

    text: str
    source: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    context: str = ""


class KnowledgeContext(BaseModel):
    """
    Output of the Knowledge Agent.
    Advisory only — never authoritative. Cannot initiate trades.
    Provides regime-aware constraints and strategy biases from research literature.
    """

    query_regime: RegimeType
    retrieved_chunks: list[KnowledgeChunk] = Field(default_factory=list)

    # Advisory outputs — the Strategy Agent may consult these
    strategy_bias: SignalDirection | None = None  # Advisory bias, not a signal
    confidence_modifier: float = Field(default=0.0, ge=-0.3, le=0.3)  # ±30% max
    suggested_constraints: list[str] = Field(default_factory=list)
    advisory_note: str = ""

    # Safety flag: knowledge alone cannot approve a trade
    knowledge_only_veto: bool = True  # Always True — enforces safety constraint §9.1


# ─────────────────────────────────────────────────────────────
#  RISK MANAGER AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class RiskDecision(BaseModel):
    """
    Output of the Risk Manager Agent.
    Has absolute veto authority over all signals.
    No trade proceeds without risk approval.
    """

    decision_id: UUID = Field(default_factory=uuid4)
    signal_id: UUID
    symbol: str
    timestamp: datetime
    approved: bool

    # If approved: execution parameters
    position_size: float = 0.0  # Number of shares / units
    position_value: float = 0.0  # Dollar value of position
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_amount: float = 0.0  # Dollar amount at risk

    # If vetoed: reason
    veto_reason: str | None = None

    # Risk metrics at time of decision
    portfolio_exposure_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    open_positions_count: int = 0

    # Crypto/perp extensions
    exchange: str | None = None
    instrument_type: InstrumentType = InstrumentType.SPOT
    leverage: float = 1.0
    mark_price: float | None = None
    funding_rate: float | None = None
    liquidation_price: float | None = None
    estimated_slippage_bps: float | None = None
    estimated_spread_bps: float | None = None
    # Trailing stop: if set, execution agent ratchets stop_loss toward price each bar
    trail_distance: float | None = None
    # Strategy name forwarded from signal — used by execution agent for signal-based exits
    strategy_name: str | None = None


# ─────────────────────────────────────────────────────────────
#  EXECUTION AGENT — ORDERS & FILLS
# ─────────────────────────────────────────────────────────────


class Order(BaseModel):
    """An order submitted to the Execution Agent."""

    order_id: UUID = Field(default_factory=uuid4)
    risk_decision_id: UUID
    symbol: str
    direction: SignalDirection
    order_type: OrderType = OrderType.MARKET
    size: float  # Units to buy/sell
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING


class Fill(BaseModel):
    """A confirmed execution fill from the Execution Agent."""

    fill_id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    symbol: str
    direction: SignalDirection
    size: float
    fill_price: float
    slippage: float = 0.0  # Dollar slippage vs expected
    commission: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mode: str = "paper"  # "paper" | "live"


class ClosedTrade(BaseModel):
    """A completed trade round-trip recorded by the Execution Agent."""

    trade_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    exit_reason: str = "manual"
    # Metadata set by Orchestrator after trade closes (used by AdaptiveLearner)
    strategy_name: str = ""
    regime: str = ""


# ─────────────────────────────────────────────────────────────
#  PORTFOLIO STATE
# ─────────────────────────────────────────────────────────────


class Position(BaseModel):
    """An open position tracked by the Execution Agent."""

    position_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    exchange: str | None = None
    instrument_type: InstrumentType = InstrumentType.SPOT
    leverage: float = 1.0
    mark_price: float | None = None
    funding_paid: float = 0.0
    # If set, stop_loss is ratcheted toward price each bar (trailing stop)
    trail_distance: float | None = None
    # Strategy that opened this position — used for signal-based exits
    strategy_name: str | None = None

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.size


class PortfolioState(BaseModel):
    """Complete portfolio state maintained by the Execution Agent."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    initial_capital: float
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)

    # Computed metrics
    gross_exposure: float = 0.0
    net_equity: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_pnl: float = 0.0
    exchange_exposure: dict[str, float] = Field(default_factory=dict)
    venue_status: dict[str, bool] = Field(default_factory=dict)
    stablecoin_basis_bps: float = 0.0

    # Execution history
    fills: list[Fill] = Field(default_factory=list)
    closed_trades: list[ClosedTrade] = Field(default_factory=list)
    total_trades: int = 0
    total_commission: float = 0.0

    def compute_equity(self) -> float:
        unrealized = 0.0
        market_value = 0.0
        gross = 0.0

        for p in self.positions.values():
            if isinstance(p, dict):
                # Dict-based position stored by CryptoPaperExecution.
                # current_price is not updated in the dict between bars, so we
                # fall back to entry_price (gives 0 unrealized PnL, no crash).
                direction = p.get("direction", "BUY")
                shares = float(p.get("shares", 0.0))
                cur_price = float(p.get("current_price") or p.get("entry_price", 0.0))
                entry = float(p.get("entry_price", 0.0))
                if direction == "BUY":
                    unrealized += (cur_price - entry) * shares
                    market_value += cur_price * shares
                else:
                    unrealized += (entry - cur_price) * shares
                    market_value -= cur_price * shares
                gross += abs(cur_price * shares)
            else:
                # Pydantic Position model
                unrealized += p.unrealized_pnl
                if p.side == PositionSide.LONG:
                    market_value += p.size * p.current_price
                else:
                    market_value -= p.size * p.current_price
                gross += abs(p.size * p.current_price)

        self.total_unrealized_pnl = unrealized
        self.net_equity = self.cash + market_value
        if self.net_equity > self.peak_equity:
            self.peak_equity = self.net_equity
        if self.peak_equity > 0:
            self.current_drawdown_pct = (self.peak_equity - self.net_equity) / self.peak_equity
        self.gross_exposure = gross
        return self.net_equity


# ─────────────────────────────────────────────────────────────
#  THESIS AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class SAConviction(StrEnum):
    """Situational Awareness (Aschenbrenner) tradeable convictions."""

    COMPUTE_DEMAND = "compute_demand"  # Ch I: Counting OOMs — GPU/semiconductor
    POWER_INFRASTRUCTURE = "power_infrastructure"  # Ch IIIa: Trillion-$ Cluster — energy/utility
    AI_DISRUPTION = "ai_disruption"  # Ch II: Intelligence Explosion — AI-native cos
    GEOPOLITICAL_DEFENSE = "geopolitical_defense"  # Ch IIId: Free World — defense/onshoring
    GOVERNMENT_PROJECT = "government_project"  # Ch IV: The Project — cleared infra/defense


class ConvictionScore(BaseModel):
    """Score for a single SA tradeable conviction."""

    conviction: SAConviction
    score: float = Field(ge=0.0, le=1.0)
    evidence_count: int = 0
    aligned_instruments: list[str] = Field(default_factory=list)
    rationale: str = ""


class ThesisContext(BaseModel):
    """
    Output of the Thesis Agent.
    Tracks conviction scores across Situational Awareness macro themes.
    Advisory only — informs Adversarial Agent and Risk Manager.
    """

    timestamp: datetime
    conviction_scores: list[ConvictionScore] = Field(default_factory=list)
    dominant_conviction: SAConviction | None = None
    overall_thesis_strength: float = Field(ge=0.0, le=1.0, default=0.5)
    symbol_conviction_alignment: SAConviction | None = None
    advisory_note: str = ""


# ─────────────────────────────────────────────────────────────
#  ALTERNATIVE DATA AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class AltDataSignal(BaseModel):
    """A single signal from an alternative data source."""

    source: str  # e.g. "sec_edgar", "job_postings"
    signal_type: str  # e.g. "filing", "hiring_trend"
    value: float | None = None
    direction: SignalDirection | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class AltDataContext(BaseModel):
    """
    Output of the Alternative Data Agent.
    Phase I: stub returning neutral context.
    Phase II: SEC EDGAR, job postings, patent data, satellite imagery.
    """

    timestamp: datetime
    signals: list[AltDataSignal] = Field(default_factory=list)
    data_quality_ok: bool = True
    sources_available: list[str] = Field(default_factory=list)
    advisory_note: str = ""


# ─────────────────────────────────────────────────────────────
#  ADVERSARIAL AGENT OUTPUT
# ─────────────────────────────────────────────────────────────


class AdversarialVerdict(StrEnum):
    PASS = "pass"  # Signal survives all checks — proceeds to Risk Manager
    FLAG = "flag"  # Concerns found — proceeds with confidence penalty
    FAIL = "fail"  # Fatal issue — dropped before Risk Manager


class AdversarialCheck(BaseModel):
    """Result of a single adversarial stress-test check."""

    name: str
    passed: bool
    description: str = ""
    severity: str = "low"  # "low" | "medium" | "high"


class SignalAssessment(BaseModel):
    """
    Output of the Adversarial Agent per signal.
    Every trade must survive adversarial scrutiny before reaching the Risk Manager.
    """

    assessment_id: UUID = Field(default_factory=uuid4)
    signal_id: UUID
    symbol: str
    timestamp: datetime
    verdict: AdversarialVerdict
    checks: list[AdversarialCheck] = Field(default_factory=list)
    confidence_adjustment: float = Field(ge=-0.3, le=0.3, default=0.0)
    adversarial_note: str = ""


# ─────────────────────────────────────────────────────────────
#  PIPELINE RESULT — full cycle record
# ─────────────────────────────────────────────────────────────


class PipelineResult(BaseModel):
    """Complete record of one pipeline cycle for a symbol."""

    run_id: UUID = Field(default_factory=uuid4)
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    snapshot: MarketSnapshot | None = None
    alt_data: AltDataContext | None = None
    regime: RegimeAnalysis | None = None
    thesis: ThesisContext | None = None
    knowledge: KnowledgeContext | None = None
    signals: list[CandidateSignal] = Field(default_factory=list)
    signal_assessments: list[SignalAssessment] = Field(default_factory=list)
    risk_decisions: list[RiskDecision] = Field(default_factory=list)
    fills: list[Fill] = Field(default_factory=list)
    portfolio: PortfolioState | None = None

    # Diagnostics
    errors: list[str] = Field(default_factory=list)
    pipeline_ms: float = 0.0
