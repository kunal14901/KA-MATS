# KA-MATS Crypto: Multi-Agent Autonomous Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Production-grade autonomous crypto trading engine.** 9 specialised agents collaborate through a **swarm consensus vote** before any capital is deployed — combining deterministic strategy signals with LLM validation, adversarial stress-testing, Bayesian EV filtering, and on-chain intelligence from Solana/Jupiter.

---

## Verified Backtest Performance (v15 Champion)

> Walk-forward validated on 22 crypto assets, Jan 2020 – Jan 2026. Intrabar fills using 1h resolution. Tiered slippage + maker-order cost model. **No parameter fitting on out-of-sample data.**

| Metric | KA-MATS v15 | BTC Buy & Hold |
|---|---|---|
| **Total Return** | **+2,475%** | +487% |
| **Annualised Return** | **71.8%** | 30.1% |
| **Sharpe Ratio** | **1.382** | 0.73 |
| **Max Drawdown** | 40.0% | 76.8% |
| **Win Rate** | **55.0%** (177W/145L) | — |
| **Profit Factor** | **1.75** | — |
| **Calmar Ratio** | **1.79** | 0.39 |
| **Total Trades** | 322 | — |
| **Final Equity** | **$257,498** | $58,700 |
| Starting Capital | $10,000 | $10,000 |

**Year-by-year breakdown:**

| Year | Trades | Win Rate | PnL | Regime |
|---|---|---|---|---|
| 2020 | 85 | 62% | +$20,029 | Bull recovery |
| 2021 | 98 | 59% | +$133,124 | Full bull run |
| 2022 | 0 | — | $0 | Bear — CB protecting capital |
| 2023 | 99 | 46% | +$17,220 | Choppy recovery |
| 2024 | 40 | 50% | +$70,924 | Late bull |
| 2025 | 0 | — | $0 | Tariff shock — CB protecting |

The circuit breaker **correctly avoids** 2022 and 2025. Every attempt to force trading in those years (BearShort, conditional CB reset, defensive dip strategy) was A/B tested and proved net-negative — the system protects capital by design.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              KA-MATS v15 Pipeline                │
                    └─────────────────────────────────────────────────┘

  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────────┐
  │  Data    │──▶│ Alt Data │──▶│ Market       │──▶│ Thesis Agent  │
  │  Agent   │   │  Agent   │   │ Analyst      │   │ (SA + context)│
  │(OHLCV+   │   │(Fear/Greed│   │(Regime:      │   │               │
  │indicators)│   │sentiment) │   │trend/range)  │   │               │
  └──────────┘   └──────────┘   └──────────────┘   └───────────────┘
                                                            │
  ┌──────────────────────────────────────────────────────────▼──────┐
  │                    Strategy Agent                                │
  │   CryptoTrendPullback · CryptoMomentumBreakout · CryptoBearShort│
  │   Deterministic rules only — no LLM inventing trade ideas       │
  └─────────────────────────────────────┬───────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │         SWARM CONSENSUS VOTE          │
                    │  ┌───────────────────────────────┐    │
                    │  │ 1. AdversarialAgent   [0-1.0] │    │
                    │  │ 2. LLM Validator      [0-1.0] │    │
                    │  │ 3. Bayesian EV Filter [0-1.0] │    │
                    │  │ 4. Regime Alignment   [0-1.0] │    │
                    │  │ 5. Confidence Gate    [0-1.0] │    │
                    │  │          Quorum: 3.0 / 5.0    │    │
                    │  └───────────────────────────────┘    │
                    │     APPROVE ▶─────────┐               │
                    │     REJECT  ▶ (logged │ dropped)      │
                    └───────────────────────┼───────────────┘
                                            │
                    ┌───────────────────────▼───────────────┐
                    │          Risk Manager                  │
                    │  Half-Kelly sizing · Portfolio heat    │
                    │  Daily loss limit · Hard DD backstop   │
                    └───────────────────────┬───────────────┘
                                            │
                    ┌───────────────────────▼───────────────┐
                    │      Execution Agent                   │
                    │  Maker limit (65% fill) → market fbk  │
                    │  Exchange-side OCO stop/TP             │
                    │  Paper / Testnet / Live (CCXT)         │
                    │  Solana DEX via Jupiter v6             │
                    └───────────────────────────────────────┘

  Supporting agents (always active):
  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Adaptive     │  │ BM25 Memory │  │ Reflection   │  │ On-Chain     │
  │ Learner      │  │(RAG trades) │  │ Agent        │  │ Agent +      │
  │(regime WR    │  │             │  │(post-trade   │  │ Solana/Jup.  │
  │ sizing)      │  │             │  │ narrative)   │  │              │
  └──────────────┘  └─────────────┘  └──────────────┘  └──────────────┘
```

## Quick Start

### Paper trading (no API keys needed, 100% offline)

```bash
git clone https://github.com/kunal14901/KA-MATS
cd KA-MATS
pip install -r requirements.txt
```

### Run the backtest

```bash
# v15 champion backtest (Jan 2020 - Jan 2026)
python -m backtest.run_phase1_intrabar

# Full statistical validation suite
python -m backtest.run_null_benchmark    # vs BTC hold / golden-cross
python -m backtest.run_stat_hygiene      # bootstrap CIs + deflated Sharpe
python -m backtest.run_v15_grade_a       # 24-config OOS grid search
```

### Solana DEX (on-chain trading)

```bash
# Add to .env:
# JUPITER_API_KEY=your_key  (portal.jup.ag)
# HELIUS_API_KEY=your_key   (helius.dev)
# WALLET_PRIVATE_KEY=...    (live mode only)

python -c "
import asyncio
from agents.solana_agent import SolanaAgent
agent = SolanaAgent(paper_mode=True)
metrics = asyncio.run(agent.get_token_metrics('SOL'))
print(f'SOL flow_bias: {metrics.flow_bias}  accumulating: {metrics.is_accumulating}')
"
```

---

## The 9-Agent Pipeline

Each bar, every symbol flows through this fixed sequence:

| # | Agent | Role | Output |
|---|---|---|---|
| 1 | **Data Agent** | OHLCV + EMA/RSI/ATR/ADX/BB | `MarketSnapshot` |
| 2 | **Alt Data Agent** | Fear-greed, sentiment, macro context | `AltDataContext` |
| 3 | **Market Analyst** | Regime: `trending_up / ranging / volatile` | `RegimeAnalysis` |
| 4 | **Thesis Agent** | SA conviction + situational scoring | `ThesisContext` |
| 5 | **Knowledge Agent** | RAG from 14 research papers (confidence mod) | `KnowledgeContext` |
| 6 | **Strategy Agent** | Deterministic signal generation | `list[CandidateSignal]` |
| 7 | **Adversarial Agent** | Stress-test: 6 checks → PASS/FLAG/FAIL | `SignalAssessment` |
| 8 | **Risk Manager** | Sizing + portfolio veto (absolute authority) | `RiskDecision` |
| 9 | **Execution Agent** | Paper/testnet/live orders via CCXT + Jupiter | `Fill` |

**Swarm Voter** sits between steps 7 and 8, aggregating all prior agent verdicts into a 5-vote quorum.

**Supporting agents:** Adaptive Learner · BM25 Memory · Reflection Agent · Bayesian EV Filter · LLM Validator · On-Chain Agent · Solana Agent

---

## Swarm Consensus Voting

KA-MATS implements multi-agent quorum voting (inspired by [AutoHedge](https://github.com/The-Swarm-Corporation/AutoHedge), but with deterministic + auditable votes rather than pure LLM opinions):

```python
# core/swarm_voter.py
swarm = voter.vote(
    adversarial_verdict="pass",   # AdversarialAgent
    llm_vetoed=False,             # LLM Validator
    bayes_approved=True,          # Bayesian EV Filter
    regime="trending_up",         # RegimeAlignment
    confidence=0.71,              # ConfidenceGate
)
# SwarmVoter [4.35/3.0] APPROVE
#   [█████] 1.00  AdversarialAgent       PASS
#   [████░] 0.75  LLMValidator           DISABLED (fail-open)
#   [████░] 0.75  BayesianEVFilter       INSUFFICIENT_DATA (fail-open)
#   [█████] 1.00  RegimeAlignment        trending_up+BUY ALIGNED
#   [████░] 0.85  ConfidenceGate         conf=0.710 (OK)
```

A trade only reaches the Risk Manager if **≥ 3.0 / 5.0 weighted votes** approve it.

---

## Solana & Jupiter DEX Integration

KA-MATS extends beyond CEX trading with native Solana on-chain capabilities:

```python
from agents.solana_agent import SolanaAgent

agent = SolanaAgent(paper_mode=True)

# On-chain metrics (Birdeye + Helius)
metrics = await agent.get_token_metrics("SOL")
print(metrics.flow_bias)      # -1.0 (selling) to +1.0 (buying)
print(metrics.is_accumulating)

# Jupiter swap quote (best DEX route)
quote = await agent.get_swap_quote("USDC", "SOL", amount_usd=1000)
print(f"Price impact: {quote.price_impact_pct:.3f}%")

# Execute swap (paper mode: logs only)
result = await agent.execute_swap(quote)
print(result.tx_signature)
```

**Supported:** `SOL, BTC, ETH, JUP, WIF, BONK, USDC, USDT`
**APIs:** Jupiter v6 · Birdeye · Helius · Solana mainnet RPC

---

## Configuration (v15 Champion)

| Setting | Value | Rationale |
|---|---|---|
| Strategy | TrendPullback + MomentumBreakout | Tested — BearShort net negative for 2020-2026 |
| Entry orders | **Maker limit** (90s timeout) | 32.5% lower entry cost vs market orders |
| Maker fill rate | 65% (35% taker fallback) | Validated on paper trade logs |
| Volatility targeting | **45% annual** | Cuts MaxDD 41%→30% in isolation |
| Risk per trade | 4% → 6% (after 50 trades) | Adaptive ramp-up as edge is confirmed |
| Max positions | 9 | Diversification without over-correlation |
| Timeframe | 1d (24h bars) | Matches backtest resolution |
| Circuit breaker | WR < 38% over 20 trades → pause 30 bars | Prevents bleed in adverse regimes |

---

## Project Structure

```
New_Trading_Architecture/
├── agents/                  # All trading agents
│   ├── data_agent.py        # OHLCV + indicators
│   ├── alt_data_agent.py    # Fear-greed, sentiment
│   ├── market_analyst.py    # Regime detection
│   ├── thesis_agent.py      # SA conviction scoring
│   ├── knowledge_agent.py   # RAG from research papers
│   ├── strategy_agent.py    # Deterministic signals
│   ├── adversarial_agent.py # Devil's advocate filter
│   ├── risk_manager.py      # Position sizing + veto
│   ├── execution_agent.py   # Paper / testnet orders
│   ├── live_execution.py    # Real exchange (CCXT + OCO)
│   ├── onchain_agent.py     # Crypto on-chain data
│   └── solana_agent.py      # Solana DEX (Jupiter v6)  ← NEW
│
├── core/                    # Engine
│   ├── orchestrator.py      # Main pipeline wiring
│   ├── swarm_voter.py       # Multi-agent quorum gate ← NEW
│   ├── adaptive_learner.py  # Regime WR sizing
│   ├── bm25_memory.py       # Trade experience RAG
│   ├── reflection_agent.py  # Post-trade narrative
│   ├── strategy_ensemble.py # Genetic selection
│   ├── models.py            # Pydantic schemas
│   ├── metrics.py           # Runtime metrics
│   └── pipeline_router.py   # Dynamic routing
│
├── backtest/                # 15+ backtest scripts
│   └── run_crypto_backtest.py   # v15 primary engine
│
├── config/settings.py       # Central configuration
├── knowledge/papers/        # 14 research papers (RAG source)
├── tools/                   # Analysis utilities
└── tests/                   # Unit + integration tests (504, 62% coverage)
```

---

## Research Foundation

The strategy logic is grounded in 14 academic papers stored in `knowledge/papers/` and used by the Knowledge Agent (RAG):

- Cross-Sectional Momentum in Cryptocurrency Markets
- Trend Following in Crypto Assets
- Market Microstructure and Crypto Liquidity
- Volatility Targeting for Systematic Strategies
- On-Chain Flow Analysis and Price Discovery
- *(+ 9 more — see `knowledge/README.md`)*

---

## Setup

```bash
# Python 3.10+
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
cp .env.example .env
# Edit .env — API keys optional for paper mode
```

**Optional for full features:**
```bash
# Solana DEX
pip install aiohttp solders solana base58

# LLM validation (local)
# Install Ollama: https://ollama.ai
ollama pull llama3.2

# LLM validation (cloud)
# Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env
```

---

## Run Commands

```bash
# v15 champion backtest (Jan 2020 – Jan 2026)
python -m backtest.run_phase1_intrabar

# Statistical validation suite
python -m backtest.run_null_benchmark    # vs BTC buy-hold / golden-cross
python -m backtest.run_stat_hygiene      # bootstrap CIs + deflated Sharpe
python -m backtest.run_v15_grade_a       # 24-config OOS grid search

# Run the full test suite
pytest

# Analyse adversarial signal filtering
python tools/analyze_shadow_log.py
```

---

## Key Design Principles

1. **Deterministic core** — Strategy Agent uses pure numerical rules. LLM is veto-only, never invents trades.
2. **Swarm consensus** — 5 agents must agree (quorum ≥ 3/5) before any capital is deployed.
3. **Risk veto is absolute** — nothing reaches execution without the Risk Manager's approval.
4. **Walk-forward validated** — all backtest results are tested on held-out data, not in-sample fits.
5. **Fail-open design** — if any non-critical agent fails (LLM timeout, Bayesian insufficient data), the pipeline continues.
6. **Capital preservation first** — circuit breaker pauses all trading when WR < 38%; every attempted "fix" for 2022/2025 was A/B tested and found to make things worse.

---

## Disclaimer

Research and education only. Not financial advice. Crypto is extremely volatile. Always use paper or testnet trading first. Past backtest performance does not guarantee future results.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Run tests: `pytest && python -m backtest.run_stat_hygiene`
4. Open a PR with A/B backtest results attached

---

*Built on the principle that the best trading system is one that knows when NOT to trade.*
