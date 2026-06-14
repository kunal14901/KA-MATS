"""
KA-MATS · Reflection Agent
Iknir Capital — Phase II (TradingAgents-inspired)

Post-trade learning loop. After each closed trade, the Reflection Agent:
  1. Builds a structured situation string from trade attributes
  2. Composes an outcome + lesson summary
  3. Writes the pair to BM25Memory for future retrieval

Inspired by TradingAgents' recursive reflection mechanism:
  "What situation was this? What happened? What should I remember next time?"

The situation string is structured to maximise token overlap with future
BM25 queries, which are also built from trade attributes (symbol, strategy,
regime, direction).

Architecture note:
  Reflection is fire-and-forget — it never blocks the pipeline.
  All errors are caught and logged; a failed reflection never prevents
  a trade from executing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from core.bm25_memory import BM25Memory


class ReflectionAgent:
    """
    Post-trade reflection loop — extracts lessons and writes to BM25 memory.

    Called once per newly closed trade in the orchestrator pipeline.
    Designed to be stateless per call — all state lives in BM25Memory.
    """

    def __init__(self, memory: BM25Memory) -> None:
        self._memory = memory
        self._reflect_count = 0
        logger.info("[ReflectionAgent] Initialized — post-trade learning active")

    # ─────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────

    def reflect(self, trade) -> None:
        """
        Build a situation→outcome pair from a closed trade and store in BM25 memory.

        Args:
            trade: ClosedTrade object (or any object with the relevant attributes:
                   .symbol, .strategy_name, .regime, .pnl, .exit_reason, .direction)
        """
        try:
            situation = self._build_situation(trade)
            outcome = self._build_outcome(trade)
            pnl = float(getattr(trade, "pnl", 0.0))
            regime = str(getattr(trade, "regime", ""))
            strategy = str(getattr(trade, "strategy_name", ""))

            # Use exit_time as trade_date for rolling window filtering
            exit_time = getattr(trade, "exit_time", None)
            trade_date = str(exit_time)[:10] if exit_time else ""

            self._memory.add(
                situation=situation,
                outcome=outcome,
                pnl=pnl,
                regime=regime,
                strategy=strategy,
                trade_date=trade_date,
            )
            self._reflect_count += 1
            logger.debug(
                f"[ReflectionAgent] Reflected: {getattr(trade, 'symbol', '?')} "
                f"({strategy}, {regime}) | PnL={pnl:+.2f} | total={self._reflect_count}"
            )
        except Exception as e:
            logger.warning(f"[ReflectionAgent] Reflection failed: {e}")

    # ─────────────────────────────────────────────────────────
    #  SITUATION BUILDER
    # ─────────────────────────────────────────────────────────

    def _build_situation(self, trade) -> str:
        """
        Build a rich, structured situation string from trade attributes.

        Token structure is designed for BM25 overlap with future queries that
        are built from the same attribute set (symbol, strategy, regime, direction).
        """
        symbol = str(getattr(trade, "symbol", ""))
        strategy = str(getattr(trade, "strategy_name", ""))
        regime = str(getattr(trade, "regime", ""))
        direction = str(getattr(trade, "direction", ""))
        exit_reason = str(getattr(trade, "exit_reason", ""))
        hold_days = getattr(trade, "hold_days", None)

        parts = [
            f"symbol {symbol}",
            f"strategy {strategy}",
            f"regime {regime}",
            f"direction {direction}",
        ]

        if hold_days is not None:
            try:
                days = float(hold_days)
                tier = "short" if days <= 2 else ("medium" if days <= 7 else "long")
                parts.append(f"holding {tier}")
            except (TypeError, ValueError):
                pass

        if "stop_loss" in exit_reason.lower() or "stop" in exit_reason.lower():
            parts.append("stop_hit true")
        elif "take_profit" in exit_reason.lower() or "target" in exit_reason.lower():
            parts.append("target_hit true")
        elif "signal" in exit_reason.lower():
            parts.append("signal_exit true")
        elif "timeout" in exit_reason.lower() or "time" in exit_reason.lower():
            parts.append("timeout_exit true")

        return " ".join(parts)

    # ─────────────────────────────────────────────────────────
    #  OUTCOME BUILDER
    # ─────────────────────────────────────────────────────────

    def _build_outcome(self, trade) -> str:
        """Build a natural-language outcome summary with an embedded lesson."""
        symbol = str(getattr(trade, "symbol", "?"))
        strategy = str(getattr(trade, "strategy_name", "?"))
        regime = str(getattr(trade, "regime", "?"))
        pnl = float(getattr(trade, "pnl", 0.0))
        exit_reason = str(getattr(trade, "exit_reason", "unknown"))
        direction = str(getattr(trade, "direction", ""))

        result = "WIN" if pnl > 0 else "LOSS"
        lesson = self._derive_lesson(pnl, regime, strategy, exit_reason)

        return (
            f"{result}: {symbol} {direction} via {strategy} in {regime} regime. "
            f"PnL={pnl:+.2f}. Exit={exit_reason}. {lesson}"
        )

    def _derive_lesson(
        self,
        pnl: float,
        regime: str,
        strategy: str,
        exit_reason: str,
    ) -> str:
        """Derive a short lesson phrase to help future query matching."""
        stop_hit = "stop_loss" in exit_reason.lower() or "stop" in exit_reason.lower()
        take_profit = "take_profit" in exit_reason.lower() or "target" in exit_reason.lower()

        if pnl > 0 and take_profit:
            return f"Lesson: {strategy} hit target cleanly in {regime} — valid setup."
        elif pnl < 0 and stop_hit:
            return f"Lesson: {strategy} stopped out in {regime} — reconsider entry filter."
        elif pnl > 0:
            return f"Lesson: {strategy} was profitable in {regime} regime."
        else:
            return f"Lesson: {strategy} underperformed in {regime} — review conditions."
