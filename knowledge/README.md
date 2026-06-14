# KA-MATS Cryptoz · Knowledge Base
## Curated Crypto Trading Research

This directory contains research papers and articles that inform the trading strategies in KA-MATS_Crypto.

### Structure
```
knowledge/
├── papers/          # Research papers and articles (txt format for easy indexing)
├── Strategies/      # Video-derived transcript texts for strategy research
├── indexes/         # FAISS vector indexes
├── index.json       # Paper manifest with metadata
└── README.md        # This file
```

### Paper Categories

**Momentum & Trend Following**
- Momentum strategies in cryptocurrency markets
- Cross-sectional momentum effectiveness
- Trend persistence in digital assets

**Mean Reversion**
- Short-term mean reversion in crypto
- RSI effectiveness for oversold/overbought detection
- Bollinger Band strategies

**Risk Management**
- Volatility scaling and position sizing
- ATR-based stops in high-volatility assets
- Drawdown control mechanisms

**Market Microstructure**
- 24/7 market dynamics
- Crypto-specific volatility patterns
- Flight-to-quality (BTC/ETH safe haven behavior)

**Alternative Data**
- Fear & Greed Index correlation with returns
- On-chain metrics (active addresses, transaction volume)
- GitHub activity as AI/DeFi project signal

### Citations
All papers include proper attribution. See `index.json` for full metadata.

### Usage
The Knowledge Agent (agents/knowledge_agent.py) uses:
1. **BM25** for keyword-based retrieval
2. **FAISS + Sentence Transformers** for semantic search
3. **Hybrid ranking** to return most relevant papers for current market regime

### Adding New Papers
1. Convert paper to plain text (.txt)
2. Save to `papers/`
3. Update `index.json` with metadata
4. Rebuild FAISS index: `python -m agents.knowledge_agent --rebuild-index`
