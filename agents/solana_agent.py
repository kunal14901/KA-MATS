"""
SolanaAgent — on-chain intelligence and DEX execution for KA-MATS.

Integrates with:
  • Jupiter Aggregator v6  — best-price DEX swaps + price quotes (Solana)
  • Helius RPC             — real-time transaction monitoring, wallet tracking
  • Birdeye API            — token analytics, holder distribution, volume trends

Why Solana?
  On-chain DEX tokens often show measurable on-chain momentum before they appear
  on Binance/Coinbase. Tracking Jupiter swap volume and wallet accumulation gives
  KA-MATS a lead-time signal advantage over CEX-only systems.

Usage:
    agent = SolanaAgent()                          # reads env vars
    ctx   = await agent.get_context("SOL/USDT")    # on-chain flow context
    quote = await agent.get_swap_quote(...)         # Jupiter swap quote
    tx    = await agent.execute_swap(...)           # live swap (paper mode: logs only)

Environment variables:
    JUPITER_API_KEY        — portal.jup.ag API key (optional for price, required for swaps)
    HELIUS_API_KEY         — helius.dev API key (on-chain tracking)
    BIRDEYE_API_KEY        — birdeye.so API key (token analytics)
    WALLET_PRIVATE_KEY     — Solana wallet private key (live mode only)
    SOLANA_RPC_URL         — custom RPC endpoint (default: mainnet-beta)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import aiohttp

    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

JUPITER_PRICE_URL = "https://price.jup.ag/v6/price"
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
HELIUS_BASE_URL = "https://api.helius.xyz/v0"

# Well-known Solana token mints
TOKEN_MINTS: dict[str, str] = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",  # Wrapped BTC on Solana
    "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # Wrapped ETH on Solana
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
}


@dataclass
class OnChainTokenMetrics:
    symbol: str
    price_usd: float
    price_change_24h: float  # %
    volume_24h_usd: float
    liquidity_usd: float
    unique_wallets_24h: int
    whale_buy_volume: float  # USD volume from wallets > $100K
    whale_sell_volume: float
    flow_bias: float  # -1.0 (heavy selling) to +1.0 (heavy buying)
    jupiter_swap_count_1h: int  # swap transactions in last hour
    holder_count: int
    fetched_at: float = field(default_factory=time.time)

    @property
    def is_accumulating(self) -> bool:
        """Whales net buying and positive flow bias."""
        return self.flow_bias > 0.2 and self.whale_buy_volume > self.whale_sell_volume

    @property
    def is_distributing(self) -> bool:
        """Whales net selling or negative flow bias."""
        return self.flow_bias < -0.2 or self.whale_sell_volume > self.whale_buy_volume * 1.5


@dataclass
class SwapQuote:
    input_mint: str
    output_mint: str
    in_amount: int  # lamports / smallest unit
    out_amount: int
    price_impact_pct: float
    route_plan: list[dict]
    raw: dict = field(default=None, repr=False)


@dataclass
class SwapResult:
    success: bool
    tx_signature: str | None
    in_amount: float
    out_amount: float
    fee_usd: float
    paper_mode: bool
    error: str | None = None


class SolanaAgent:
    """
    Provides on-chain context and DEX execution for KA-MATS.

    Paper mode (default): all swap calls are logged but not sent to chain.
    Live mode: requires WALLET_PRIVATE_KEY and solders/solana-py.
    """

    def __init__(self, paper_mode: bool = True) -> None:
        self.paper_mode = paper_mode
        self.jupiter_api_key = os.getenv("JUPITER_API_KEY", "")
        self.helius_api_key = os.getenv("HELIUS_API_KEY", "")
        self.birdeye_api_key = os.getenv("BIRDEYE_API_KEY", "")
        self.rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = 60.0  # seconds

    # ── Public API ────────────────────────────────────────────────────

    async def get_token_metrics(self, symbol: str) -> OnChainTokenMetrics | None:
        """Fetch comprehensive on-chain metrics for a token (Birdeye + Helius)."""
        cached = self._get_cache(f"metrics:{symbol}")
        if cached:
            return cached

        mint = TOKEN_MINTS.get(symbol.upper())
        if not mint:
            return None

        price_data = await self._fetch_jupiter_price(mint)
        bird_data = await self._fetch_birdeye_overview(mint)

        if not price_data:
            return None

        price_usd = price_data.get("price", 0.0)
        vol_24h = bird_data.get("v24hUSD", 0.0) if bird_data else 0.0
        liquidity = bird_data.get("liquidity", 0.0) if bird_data else 0.0
        holder_cnt = bird_data.get("holder", 0) if bird_data else 0
        buy_vol = bird_data.get("buy24h", 0.0) if bird_data else 0.0
        sell_vol = bird_data.get("sell24h", 0.0) if bird_data else 0.0
        pct_change = bird_data.get("priceChange24hPercent", 0.0) if bird_data else 0.0

        total_vol = buy_vol + sell_vol
        flow_bias = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

        metrics = OnChainTokenMetrics(
            symbol=symbol,
            price_usd=price_usd,
            price_change_24h=pct_change,
            volume_24h_usd=vol_24h,
            liquidity_usd=liquidity,
            unique_wallets_24h=0,  # requires Helius paid tier
            whale_buy_volume=buy_vol,
            whale_sell_volume=sell_vol,
            flow_bias=round(flow_bias, 3),
            jupiter_swap_count_1h=0,  # requires Helius indexer
            holder_count=holder_cnt,
        )

        self._set_cache(f"metrics:{symbol}", metrics)
        return metrics

    async def get_swap_quote(
        self,
        input_symbol: str,
        output_symbol: str,
        amount_usd: float,
        slippage_bps: int = 50,
    ) -> SwapQuote | None:
        """Get best-route swap quote from Jupiter aggregator."""
        input_mint = TOKEN_MINTS.get(input_symbol.upper())
        output_mint = TOKEN_MINTS.get(output_symbol.upper())
        if not input_mint or not output_mint:
            return None

        # Convert USD to lamports (approximate — needs price for exact conversion)
        price_data = await self._fetch_jupiter_price(input_mint)
        if not price_data:
            return None
        price = price_data.get("price", 1.0)
        # USDC has 6 decimals, SOL has 9
        decimals = 6 if input_symbol.upper() in ("USDC", "USDT") else 9
        amount_raw = int((amount_usd / price) * (10**decimals))

        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount_raw,
            "slippageBps": slippage_bps,
            "onlyDirectRoutes": False,
        }
        data = await self._get(JUPITER_QUOTE_URL, params=params)
        if not data:
            return None

        return SwapQuote(
            input_mint=input_mint,
            output_mint=output_mint,
            in_amount=int(data.get("inAmount", 0)),
            out_amount=int(data.get("outAmount", 0)),
            price_impact_pct=float(data.get("priceImpactPct", 0)),
            route_plan=data.get("routePlan", []),
            raw=data,
        )

    async def execute_swap(
        self,
        quote: SwapQuote,
        wallet_address: str | None = None,
    ) -> SwapResult:
        """Execute a Jupiter swap. Paper mode: returns a simulated result."""
        in_usd = quote.in_amount / 1e9  # rough approximation
        out_usd = quote.out_amount / 1e6

        if self.paper_mode:
            return SwapResult(
                success=True,
                tx_signature="PAPER_MODE_SIM_" + str(int(time.time())),
                in_amount=in_usd,
                out_amount=out_usd,
                fee_usd=in_usd * 0.0003,  # ~3bps Jupiter fee
                paper_mode=True,
            )

        # Live execution — requires solders + solana-py
        try:
            import base58
            from solana.rpc.async_api import AsyncClient  # type: ignore
            from solders.keypair import Keypair  # type: ignore

            pk = os.getenv("WALLET_PRIVATE_KEY", "")
            keypair = Keypair.from_bytes(base58.b58decode(pk))

            payload = {
                "quoteResponse": quote.raw,
                "userPublicKey": str(keypair.pubkey()),
                "wrapAndUnwrapSol": True,
            }
            swap_data = await self._post(JUPITER_SWAP_URL, json=payload)
            if not swap_data:
                return SwapResult(
                    success=False,
                    tx_signature=None,
                    in_amount=in_usd,
                    out_amount=0.0,
                    fee_usd=0.0,
                    paper_mode=False,
                    error="swap_data_empty",
                )

            # Sign and send transaction
            async with AsyncClient(self.rpc_url) as client:
                raw_tx = bytes(swap_data["swapTransaction"], "utf-8")
                import base64

                from solders.transaction import VersionedTransaction  # type: ignore

                tx = VersionedTransaction.from_bytes(base64.b64decode(raw_tx))
                tx.sign([keypair])
                result = await client.send_raw_transaction(bytes(tx))
                sig = str(result.value)

            return SwapResult(
                success=True,
                tx_signature=sig,
                in_amount=in_usd,
                out_amount=out_usd,
                fee_usd=in_usd * 0.0003,
                paper_mode=False,
            )

        except ImportError:
            return SwapResult(
                success=False,
                tx_signature=None,
                in_amount=in_usd,
                out_amount=0.0,
                fee_usd=0.0,
                paper_mode=False,
                error="solders/solana-py not installed. pip install solders solana base58",
            )
        except Exception as e:
            return SwapResult(
                success=False,
                tx_signature=None,
                in_amount=in_usd,
                out_amount=0.0,
                fee_usd=0.0,
                paper_mode=False,
                error=str(e),
            )

    # ── Sync wrapper for orchestrator compatibility ───────────────────

    def get_flow_bias(self, symbol: str) -> float:
        """Blocking wrapper — returns on-chain flow bias (-1 to +1). Cached."""
        cached = self._get_cache(f"flow:{symbol}")
        if cached is not None:
            return cached
        try:
            loop = asyncio.new_event_loop()
            try:
                metrics = loop.run_until_complete(self.get_token_metrics(symbol))
            finally:
                loop.close()
            bias = metrics.flow_bias if metrics else 0.0
            self._set_cache(f"flow:{symbol}", bias)
            return bias
        except Exception:
            return 0.0

    # ── Internal HTTP helpers ─────────────────────────────────────────

    async def _get(self, url: str, params: dict | None = None) -> dict | None:
        if not _AIOHTTP_AVAILABLE:
            return None
        try:
            headers = {}
            if self.jupiter_api_key and "jup.ag" in url:
                headers["Authorization"] = f"Bearer {self.jupiter_api_key}"
            if self.birdeye_api_key and "birdeye" in url:
                headers["X-API-KEY"] = self.birdeye_api_key
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=8)
                ) as r:
                    if r.status == 200:
                        return await r.json()
        except Exception:
            pass
        return None

    async def _post(self, url: str, json: dict | None = None) -> dict | None:
        if not _AIOHTTP_AVAILABLE:
            return None
        try:
            headers = {"Content-Type": "application/json"}
            if self.jupiter_api_key:
                headers["Authorization"] = f"Bearer {self.jupiter_api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=json, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    if r.status == 200:
                        return await r.json()
        except Exception:
            pass
        return None

    async def _fetch_jupiter_price(self, mint: str) -> dict | None:
        data = await self._get(JUPITER_PRICE_URL, params={"ids": mint})
        if data and "data" in data:
            return data["data"].get(mint)
        return None

    async def _fetch_birdeye_overview(self, mint: str) -> dict | None:
        if not self.birdeye_api_key:
            return None
        data = await self._get(f"{BIRDEYE_BASE_URL}/defi/token_overview", params={"address": mint})
        if data and "data" in data:
            return data["data"]
        return None

    # ── Cache helpers ─────────────────────────────────────────────────

    def _get_cache(self, key: str) -> Any | None:
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return val
        return None

    def _set_cache(self, key: str, val: Any) -> None:
        self._cache[key] = (time.time(), val)
