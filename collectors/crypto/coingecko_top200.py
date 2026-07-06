"""
Cryptocurrency market cap history via CoinGecko API (free tier).
Fetches top 200 coins by market cap: historical price, market cap, volume.
Free tier limits: 10-50 calls/min, 500/day — batch carefully.
Output: data/crypto/coingecko_<SYMBOL>_1d.parquet
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

CG_API = "https://api.coingecko.com/api/v3"
CG_MARKET_CHART = CG_API + "/coins/{id}/market_chart"
CG_LIST = CG_API + "/coins/list"

# Top 200 by market cap (static list — refresh quarterly)
# Manual curation from CoinGecko top 200 as of 2026-07
TOP_200_IDS = [
    "bitcoin", "ethereum", "tether", "binancecoin", "solana", "usd-coin", "ripple", "cardano",
    "avalanche-2", "dogecoin", "polkadot", "tron", "polygon", "chainlink", "litecoin",
    "shiba-inu", "bitcoin-cash", "stellar", "uniswap", "cosmos", "ethereum-classic", "monero",
    "algorand", "vechain", "filecoin", "hedera-hashgraph", "internet-computer", "aptos",
    "near", "maker", "the-graph", "quant-network", "lido-dao", "aave", "cronos", "flow",
    "elrond-erd-2", "eos", "theta-token", "axie-infinity", "tezos", "pancakeswap-token",
    "decentraland", "the-sandbox", "kucoin-shares", "neo", "fantom", "render-token",
    "gala", "curve-dao-token", "synthetix-network-token", "zilliqa", "enjincoin", "mina-protocol",
    "chiliz", "basic-attention-token", "1inch", "loopring", "sushi", "compound-governance-token",
    "yearn-finance", "kusama", "thorchain", "helium", "terra-luna-2", "osmosis", "celo",
    "arweave", "nexo", "frax", "frax-share", "rocket-pool", "gmx", "kava", "iotex", "bitcoin-cash-sv",
    "conflux-token", "casper-network", "audius", "convex-finance", "aurora-near", "ecash",
    "ravencoin", "nervos-network", "ontology", "ocean-protocol", "mask-network", "energy-web-token",
    "telcoin", "ankr", "livepeer", "immutable-x", "singularitynet", "fetch-ai", "ren", "balancer",
    "stacks", "umee", "radix", "polymesh", "woo-network", "my-neighbor-alice", "origin-protocol",
    "reserve-rights-token", "tribe-2", "status", "civic", "raydium", "biconomy", "parsiq",
    "band-protocol", "aragon", "request-network", "storj", "numeraire", "cartesi", "illuvium",
    "holotoken", "liquity-usd", "multichain", "spell-token", "perpetual-protocol", "magic",
    "ampleforth", "alien-worlds", "tribe", "oasis-network", "marlin", "dydx", "polyswarm",
    "pundi-x-2", "boba-network", "orchid-protocol", "gods-unchained", "golem", "chromia",
    "strike", "skale", "metacraft", "akash-network", "dent", "api3", "serum", "smooth-love-potion",
    "verge", "republic-protocol", "rally-2", "nuls", "horizen", "metal", "uma", "orion-protocol",
    "matic-network", "keep-network", "celsius-degree-token", "velas", "gyen", "gas", "steem",
    "elastos", "super-zero", "power-ledger", "wazirx", "aergo", "ardor", "contentos", "dodo",
    "nimiq-2", "cortex", "qtum", "wink", "lisk", "iost", "moviebloc", "steem-dollars", "wanchain",
    "kin", "nano", "voyager-token", "reddcoin", "mobox", "stratis", "district0x", "augur",
    "crypterium", "tenset", "dock", "selfkey", "dusk-network", "ergo", "ocean-protocol",
]


def fetch_coingecko_history(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical price/market cap/volume for a coin."""
    try:
        url = CG_MARKET_CHART.format(id=coin_id)
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        resp = fetch(url, params=params, retries=3)
        data = resp.json()
        
        if "prices" not in data or not data["prices"]:
            return pd.DataFrame()
        
        # Convert lists to DataFrame
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        market_caps = pd.DataFrame(data.get("market_caps", []), columns=["timestamp", "market_cap"])
        volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])
        
        # Merge on timestamp
        df = prices.merge(market_caps, on="timestamp", how="left").merge(volumes, on="timestamp", how="left")
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp"]).set_index("date").sort_index()
        
        return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    
    except Exception as exc:
        print(f"  WARNING: {coin_id} — {exc}")
        return pd.DataFrame()


def main() -> None:
    print(f"=== CoinGecko Top 200 Market Cap History ===")
    print(f"Total: {len(TOP_200_IDS)} coins")
    
    ok = 0
    for i, coin_id in enumerate(TOP_200_IDS, 1):
        # Fetch 1 year history (max for free tier without pagination)
        df = fetch_coingecko_history(coin_id, days=365)
        
        if not df.empty:
            # Use coin_id as filename (e.g., bitcoin, ethereum)
            symbol = coin_id.upper().replace("-", "_")
            save(df, "crypto", f"CG_{symbol}_1d.parquet")
            ok += 1
        
        if i % 10 == 0:
            print(f"  [{i}/{len(TOP_200_IDS)}] processed")
        
        # Rate limit: 10-50 calls/min free tier → 1 call per 2s is safe
        time.sleep(2)
    
    print(f"\n✓ {ok}/{len(TOP_200_IDS)} coins saved")


if __name__ == "__main__":
    main()
