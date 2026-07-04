"""
Ethereum on-chain metrics collector.
Source: Etherscan (free, no API key for basic stats) + blockchain.com ETH endpoints.
Also uses: CoinGecko free API for ETH market cap history.
Output: data/onchain/ETH_<metric>_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# CoinGecko free API — ETH on-chain proxy metrics via market data
COINGECKO_HISTORY = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

# Blockchain.info ETH stats (free)
BLOCKCHAIN_ETH = "https://api.blockchain.info/charts/{chart}?timespan=all&format=json&sampled=true"

BLOCKCHAIN_CHARTS = {
    "ETH_TxCount_1d": "n-transactions",       # not ETH-specific but illustrative
}

# CoinGecko provides: price, market_cap, total_volume for ETH
CG_COINS = {
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "MATIC": "matic-network",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
}

# Glassnode free tier metrics via their public API (no key needed for basic)
GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"

# Open blockchain stats — Etherscan public stats (no key for aggregates)
ETHERSCAN_STATS = "https://api.etherscan.io/api"


def fetch_coingecko_history(coin_id: str, days: int = 1825) -> dict[str, pd.DataFrame]:
    """Fetch price, market_cap, volume history from CoinGecko."""
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    resp = fetch(COINGECKO_HISTORY.format(coin_id=coin_id), params=params)
    data = resp.json()

    results = {}
    for key in ["prices", "market_caps", "total_volumes"]:
        if key not in data:
            continue
        records = [{"date": pd.Timestamp(ts, unit="ms", tz="UTC"), "value": val}
                   for ts, val in data[key] if val is not None]
        if records:
            df = pd.DataFrame(records).set_index("date").sort_index()
            results[key] = df
    return results


def main() -> None:
    for symbol, coin_id in CG_COINS.items():
        print(f"Fetching {symbol} ({coin_id}) market metrics ...")
        try:
            metrics = fetch_coingecko_history(coin_id)
            name_map = {
                "prices": f"{symbol}_Price_1d",
                "market_caps": f"{symbol}_MarketCap_1d",
                "total_volumes": f"{symbol}_Volume_1d",
            }
            for key, fname in name_map.items():
                if key in metrics:
                    save(metrics[key], "onchain", f"{fname}.parquet")
            time.sleep(1.2)  # CoinGecko rate limit: ~50 req/min free
        except Exception as exc:
            print(f"  WARNING: {symbol} — {exc}")
            time.sleep(5)

    # ETH-specific: gas price history via Etherscan (no key needed for some endpoints)
    print("\nFetching ETH gas stats ...")
    try:
        resp = fetch(
            ETHERSCAN_STATS,
            params={"module": "stats", "action": "ethsupply", "apikey": "YourApiKeyToken"},
            timeout=15,
        )
        # Supply is a single value — wrap as single-row
        data = resp.json()
        if data.get("status") == "1":
            supply = float(data["result"]) / 1e18  # Wei to ETH
            df = pd.DataFrame([{"value": supply}], index=[pd.Timestamp.utcnow().floor("D")])
            df.index = pd.DatetimeIndex(df.index, tz="UTC")
            save(df, "onchain", "ETH_Supply_snapshot.parquet")
    except Exception as exc:
        print(f"  WARNING: ETH supply — {exc}")


if __name__ == "__main__":
    main()
