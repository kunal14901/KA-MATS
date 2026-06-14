"""
KA-MATS Cryptoz · Dynamic Pipeline Router
Iknir Capital — v6 Enhancement (Vertus-inspired)

Inspired by Vertus's "Dynamic Neural Topology" — adaptive complexity
that scales with market conditions. Simple markets get a fast path,
crisis markets get deeper analysis.

The router evaluates current market state and returns a PipelineConfig
that tells the orchestrator which agents to run and at what depth.

Pipeline Modes:
  FAST   — Calm bull market. Skip Thesis + Knowledge, minimal Adversarial.
  NORMAL — Standard 9-agent pipeline. Default for most conditions.
  DEEP   — Regime transition or high volatility. Full pipeline + extra checks.
  CRISIS — Extreme stress (DD > 20%, macro bear). Emergency deleveraging mode.

Usage:
    router = PipelineRouter()
    config = router.route(regime, portfolio, flow_context)
    # config.skip_agents → set of agent names to skip
    # config.extra_checks → additional risk checks to enable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger


class PipelineMode(StrEnum):
    FAST = "fast"
    NORMAL = "normal"
    DEEP = "deep"
    CRISIS = "crisis"


@dataclass
class PipelineConfig:
    """Configuration for a single bar's pipeline execution."""

    mode: PipelineMode = PipelineMode.NORMAL
    skip_agents: set[str] = field(default_factory=set)
    extra_checks: set[str] = field(default_factory=set)
    confidence_floor: float = 0.50  # min confidence to proceed
    max_new_entries: int = 3  # max new positions this bar
    sizing_multiplier: float = 1.0  # global sizing factor
    reason: str = ""


class PipelineRouter:
    """
    Routes the pipeline based on current market conditions.

    Inputs:
      - regime: current market regime (from Market Analyst)
      - drawdown_pct: current portfolio drawdown
      - flow_bias: on-chain flow directional bias (-1 to +1)
      - volatility_pct: current portfolio volatility percentile
      - macro_mode: "bull", "uncertain", or "bear"

    Output:
      - PipelineConfig with skip/add instructions for the orchestrator
    """

    # Thresholds
    _DD_CRISIS = 0.20  # 20% DD → crisis mode
    _DD_DEEP = 0.10  # 10% DD → deep mode
    _VOL_HIGH = 0.75  # top 25% vol → deep mode
    _FLOW_EXTREME = 0.6  # |flow_bias| > 0.6 → deep mode

    def route(
        self,
        regime: str = "unknown",
        drawdown_pct: float = 0.0,
        flow_bias: float = 0.0,
        volatility_pct: float = 50.0,
        macro_mode: str = "bull",
        open_positions: int = 0,
    ) -> PipelineConfig:
        """Determine pipeline configuration for this bar."""

        # 1. Crisis detection (highest priority)
        if drawdown_pct >= self._DD_CRISIS or macro_mode == "bear":
            config = PipelineConfig(
                mode=PipelineMode.CRISIS,
                skip_agents=set(),  # run everything in crisis
                extra_checks={"emergency_deleverage", "correlation_check", "max_dd_recheck"},
                confidence_floor=0.65,  # higher bar for new entries
                max_new_entries=1,  # max 1 new entry in crisis
                sizing_multiplier=0.50,  # half sizing
                reason=f"Crisis: DD={drawdown_pct:.1%}, macro={macro_mode}",
            )
            logger.warning(f"[PipelineRouter] CRISIS mode: {config.reason}")
            return config

        # 2. Deep analysis (regime transition, high vol, extreme flow)
        is_transition = regime in ("ranging", "mean_reverting", "volatile")
        is_high_vol = volatility_pct >= self._VOL_HIGH
        is_extreme_flow = abs(flow_bias) >= self._FLOW_EXTREME
        is_dd_elevated = drawdown_pct >= self._DD_DEEP

        if is_transition or is_high_vol or is_extreme_flow or is_dd_elevated:
            extra = set()
            if is_extreme_flow:
                extra.add("flow_confirmation")
            if is_high_vol:
                extra.add("vol_regime_recheck")
            if is_dd_elevated:
                extra.add("equity_curve_recheck")

            config = PipelineConfig(
                mode=PipelineMode.DEEP,
                skip_agents=set(),  # full pipeline
                extra_checks=extra,
                confidence_floor=0.55,
                max_new_entries=2,
                sizing_multiplier=0.80,
                reason=(
                    f"Deep: regime={regime}, vol_pct={volatility_pct:.0f}, "
                    f"flow={flow_bias:+.2f}, DD={drawdown_pct:.1%}"
                ),
            )
            logger.debug(f"[PipelineRouter] DEEP mode: {config.reason}")
            return config

        # 3. Fast path (calm bull, low DD, neutral flow)
        is_calm_bull = (
            regime == "trending_up"
            and macro_mode == "bull"
            and drawdown_pct < 0.05
            and abs(flow_bias) < 0.3
            and open_positions < 6
        )

        if is_calm_bull:
            config = PipelineConfig(
                mode=PipelineMode.FAST,
                skip_agents={"thesis_agent", "knowledge_agent"},
                extra_checks=set(),
                confidence_floor=0.50,
                max_new_entries=3,
                sizing_multiplier=1.0,
                reason=f"Fast: calm bull, DD={drawdown_pct:.1%}",
            )
            logger.debug(f"[PipelineRouter] FAST mode: {config.reason}")
            return config

        # 4. Normal (default)
        return PipelineConfig(
            mode=PipelineMode.NORMAL,
            reason=f"Normal: regime={regime}, DD={drawdown_pct:.1%}",
        )

    def should_run_agent(self, agent_name: str, config: PipelineConfig) -> bool:
        """Check if a specific agent should run given the pipeline config."""
        return agent_name not in config.skip_agents
