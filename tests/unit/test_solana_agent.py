"""Unit tests for agents/solana_agent.py — SolanaAgent on-chain integration."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.solana_agent import (
    TOKEN_MINTS,
    OnChainTokenMetrics,
    SolanaAgent,
    SwapQuote,
    SwapResult,
)


# Helper so tests work on Python 3.14 (asyncio.run() fails when a loop is
# already active in the process; explicit new_event_loop is always safe).
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def agent():
    return SolanaAgent(paper_mode=True)


# ── TOKEN_MINTS sanity checks ─────────────────────────────────────────────────


class TestTokenMints:
    def test_known_symbols_present(self):
        for sym in ("SOL", "USDC", "USDT", "BTC", "ETH", "JUP", "WIF", "BONK"):
            assert sym in TOKEN_MINTS

    def test_mint_addresses_non_empty(self):
        for sym, mint in TOKEN_MINTS.items():
            assert len(mint) > 20, f"{sym} mint too short"


# ── SolanaAgent init ──────────────────────────────────────────────────────────


class TestSolanaAgentInit:
    def test_paper_mode_default(self):
        a = SolanaAgent()
        assert a.paper_mode is True

    def test_live_mode_flag(self):
        a = SolanaAgent(paper_mode=False)
        assert a.paper_mode is False

    def test_cache_initially_empty(self):
        a = SolanaAgent()
        assert a._cache == {}

    def test_rpc_url_default(self):
        a = SolanaAgent()
        assert "mainnet-beta" in a.rpc_url or "solana" in a.rpc_url.lower()


# ── OnChainTokenMetrics properties ───────────────────────────────────────────


class TestOnChainTokenMetrics:
    def _make(self, flow_bias=0.0, whale_buy=1000.0, whale_sell=500.0):
        return OnChainTokenMetrics(
            symbol="SOL",
            price_usd=100.0,
            price_change_24h=2.5,
            volume_24h_usd=1_000_000.0,
            liquidity_usd=5_000_000.0,
            unique_wallets_24h=1200,
            whale_buy_volume=whale_buy,
            whale_sell_volume=whale_sell,
            flow_bias=flow_bias,
            jupiter_swap_count_1h=50,
            holder_count=80_000,
        )

    def test_is_accumulating_true(self):
        m = self._make(flow_bias=0.5, whale_buy=2000.0, whale_sell=500.0)
        assert m.is_accumulating is True

    def test_is_accumulating_false_low_flow(self):
        m = self._make(flow_bias=0.1, whale_buy=2000.0, whale_sell=500.0)
        assert m.is_accumulating is False  # flow_bias <= 0.2

    def test_is_distributing_negative_flow(self):
        m = self._make(flow_bias=-0.5, whale_buy=500.0, whale_sell=2000.0)
        assert m.is_distributing is True

    def test_is_distributing_whale_ratio(self):
        m = self._make(flow_bias=0.0, whale_buy=100.0, whale_sell=200.0)
        # 200 > 100 * 1.5 = 150 → distributing
        assert m.is_distributing is True

    def test_is_not_accumulating_not_distributing_neutral(self):
        m = self._make(flow_bias=0.0, whale_buy=500.0, whale_sell=500.0)
        assert m.is_accumulating is False
        assert m.is_distributing is False

    def test_fetched_at_set(self):
        m = self._make()
        assert m.fetched_at <= time.time()


# ── SwapQuote dataclass ───────────────────────────────────────────────────────


class TestSwapQuote:
    def test_fields(self):
        q = SwapQuote(
            input_mint="MINT_A",
            output_mint="MINT_B",
            in_amount=1_000_000_000,
            out_amount=150_000_000,
            price_impact_pct=0.05,
            route_plan=[{"swapInfo": {}}],
        )
        assert q.input_mint == "MINT_A"
        assert q.price_impact_pct == 0.05
        assert len(q.route_plan) == 1


# ── SwapResult dataclass ──────────────────────────────────────────────────────


class TestSwapResult:
    def test_successful_result(self):
        r = SwapResult(
            success=True,
            tx_signature="ABCDEF",
            in_amount=10.0,
            out_amount=9.97,
            fee_usd=0.003,
            paper_mode=True,
        )
        assert r.success is True
        assert r.error is None

    def test_failed_result_with_error(self):
        r = SwapResult(
            success=False,
            tx_signature=None,
            in_amount=10.0,
            out_amount=0.0,
            fee_usd=0.0,
            paper_mode=False,
            error="network_timeout",
        )
        assert r.success is False
        assert r.error == "network_timeout"


# ── Cache helpers ─────────────────────────────────────────────────────────────


class TestCacheHelpers:
    def test_cache_miss_returns_none(self, agent):
        assert agent._get_cache("missing_key") is None

    def test_cache_set_and_get(self, agent):
        agent._set_cache("k", 42)
        assert agent._get_cache("k") == 42

    def test_cache_expiry(self, agent):
        agent._cache_ttl = 0.01  # 10 ms
        agent._set_cache("k", 99)
        time.sleep(0.05)
        assert agent._get_cache("k") is None  # expired

    def test_cache_not_expired_within_ttl(self, agent):
        agent._cache_ttl = 60.0
        agent._set_cache("x", "value")
        assert agent._get_cache("x") == "value"


# ── _get / _post (no aiohttp) ─────────────────────────────────────────────────


class TestHTTPHelpers:
    def test_get_returns_none_without_aiohttp(self, agent):
        import agents.solana_agent as mod

        original = mod._AIOHTTP_AVAILABLE
        mod._AIOHTTP_AVAILABLE = False
        try:
            result = _run(agent._get("https://example.com"))
            assert result is None
        finally:
            mod._AIOHTTP_AVAILABLE = original

    def test_post_returns_none_without_aiohttp(self, agent):
        import agents.solana_agent as mod

        original = mod._AIOHTTP_AVAILABLE
        mod._AIOHTTP_AVAILABLE = False
        try:
            result = _run(agent._post("https://example.com", json={}))
            assert result is None
        finally:
            mod._AIOHTTP_AVAILABLE = original


# ── execute_swap — paper mode ─────────────────────────────────────────────────


class TestExecuteSwapPaperMode:
    def _make_quote(self):
        return SwapQuote(
            input_mint=TOKEN_MINTS["SOL"],
            output_mint=TOKEN_MINTS["USDC"],
            in_amount=1_000_000_000,  # 1 SOL in lamports
            out_amount=100_000_000,  # 100 USDC in micro-units
            price_impact_pct=0.01,
            route_plan=[],
            raw={"quoteResponse": "mock"},
        )

    def test_paper_swap_succeeds(self, agent):
        quote = self._make_quote()
        result = _run(agent.execute_swap(quote))
        assert result.success is True
        assert result.paper_mode is True
        assert result.tx_signature is not None
        assert "PAPER_MODE_SIM_" in result.tx_signature

    def test_paper_swap_fee_approx(self, agent):
        quote = self._make_quote()
        result = _run(agent.execute_swap(quote))
        assert result.fee_usd > 0.0

    def test_live_mode_no_solders_returns_error(self):
        live_agent = SolanaAgent(paper_mode=False)
        quote = self._make_quote()
        result = _run(live_agent.execute_swap(quote))
        assert result.success is False
        assert result.error is not None


# ── get_token_metrics ─────────────────────────────────────────────────────────


class TestGetTokenMetrics:
    def test_unknown_symbol_returns_none(self, agent):
        result = _run(agent.get_token_metrics("UNKNOWN_TOKEN_XYZ"))
        assert result is None

    def test_known_symbol_no_network_returns_none(self, agent):
        # Without aiohttp or network, should return None gracefully
        import agents.solana_agent as mod

        original = mod._AIOHTTP_AVAILABLE
        mod._AIOHTTP_AVAILABLE = False
        try:
            result = _run(agent.get_token_metrics("SOL"))
            assert result is None
        finally:
            mod._AIOHTTP_AVAILABLE = original

    def test_cache_hit_returns_cached(self, agent):
        fake_metrics = OnChainTokenMetrics(
            symbol="SOL",
            price_usd=100.0,
            price_change_24h=1.0,
            volume_24h_usd=1e6,
            liquidity_usd=5e6,
            unique_wallets_24h=0,
            whale_buy_volume=500.0,
            whale_sell_volume=200.0,
            flow_bias=0.3,
            jupiter_swap_count_1h=0,
            holder_count=5000,
        )
        agent._set_cache("metrics:SOL", fake_metrics)
        result = _run(agent.get_token_metrics("SOL"))
        assert result is fake_metrics


# ── get_swap_quote ─────────────────────────────────────────────────────────────


class TestGetSwapQuote:
    def test_unknown_input_symbol_returns_none(self, agent):
        result = _run(agent.get_swap_quote("UNKNOWN", "USDC", 100.0))
        assert result is None

    def test_unknown_output_symbol_returns_none(self, agent):
        result = _run(agent.get_swap_quote("SOL", "UNKNOWN", 100.0))
        assert result is None

    def test_no_aiohttp_returns_none(self, agent):
        import agents.solana_agent as mod

        original = mod._AIOHTTP_AVAILABLE
        mod._AIOHTTP_AVAILABLE = False
        try:
            result = _run(agent.get_swap_quote("SOL", "USDC", 100.0))
            assert result is None
        finally:
            mod._AIOHTTP_AVAILABLE = original


# ── get_flow_bias (sync wrapper) ──────────────────────────────────────────────


class TestGetFlowBias:
    def test_returns_zero_on_exception(self, agent):
        # Patching get_token_metrics to raise
        async def _raise(*a, **kw):
            raise RuntimeError("network_error")

        agent.get_token_metrics = _raise
        bias = agent.get_flow_bias("SOL")
        assert bias == 0.0

    def test_returns_cached_bias(self, agent):
        agent._set_cache("flow:SOL", 0.42)
        assert agent.get_flow_bias("SOL") == pytest.approx(0.42)

    def test_returns_zero_for_none_metrics(self, agent):
        async def _none(*a, **kw):
            return None

        agent.get_token_metrics = _none
        bias = agent.get_flow_bias("BTC")
        assert bias == 0.0

    def test_returns_flow_bias_from_metrics(self, agent):
        fake = OnChainTokenMetrics(
            symbol="ETH",
            price_usd=3000.0,
            price_change_24h=1.5,
            volume_24h_usd=5e6,
            liquidity_usd=1e7,
            unique_wallets_24h=0,
            whale_buy_volume=800.0,
            whale_sell_volume=200.0,
            flow_bias=0.6,
            jupiter_swap_count_1h=0,
            holder_count=10000,
        )

        async def _fake(*a, **kw):
            return fake

        agent.get_token_metrics = _fake
        bias = agent.get_flow_bias("ETH")
        assert bias == pytest.approx(0.6)


# ── _fetch_birdeye_overview ───────────────────────────────────────────────────


class TestFetchBirdeye:
    def test_no_api_key_returns_none(self, agent):
        agent.birdeye_api_key = ""
        result = _run(agent._fetch_birdeye_overview("some_mint"))
        assert result is None
