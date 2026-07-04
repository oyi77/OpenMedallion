"""
Solana on-chain metrics collector.
Source: Helius public API (free tier, no key for basic stats) +
        Solana Beach public API + CoinGecko SOL data.
Output: data/onchain/SOL_<metric>_1d.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

COINGECKO_SOL = "https://api.coingecko.com/api/v3/coins/solana/market_chart"
SOLANA_BEACH = "https://api.solanabeach.io/v1"

# Solana validator stats (public endpoint)
SOLANA_RPC = "https://api.mainnet-beta.solana.com"

# CoinGecko free — SOL + related ecosystem tokens
SOL_ECOSYSTEM = {
    "SOL": "solana",
    "RAY": "raydium",
    "JUP": "jupiter-exchange-solana",
    "BONK": "bonk",
    "WIF": "dogwifcoin",
    "PYTH": "pyth-network",
    "JTO": "jito-governance-token",
    "MSOLANA": "msol",
}


def fetch_coingecko_market(coin_id: str, days: int = 365) -> dict[str, pd.DataFrame]:
    """Fetch price, market_cap, volume from CoinGecko."""
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    resp = fetch(COINGECKO_SOL.replace("solana", coin_id).replace("/solana/", f"/{coin_id}/"), params=params)
    # Rebuild URL properly
    resp = fetch(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params=params,
    )
    data = resp.json()
    results = {}
    for key in ["prices", "market_caps", "total_volumes"]:
        if key not in data:
            continue
        records = [
            {"date": pd.Timestamp(ts, unit="ms", tz="UTC"), "value": val}
            for ts, val in data[key] if val is not None
        ]
        if records:
            df = pd.DataFrame(records).set_index("date").sort_index()
            results[key] = df
    return results


def fetch_solana_staking() -> pd.DataFrame:
    """Get current epoch staking stats via Solana JSON-RPC."""
    try:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getEpochInfo"}
        import requests
        resp = requests.post(SOLANA_RPC, json=payload, timeout=10)
        info = resp.json().get("result", {})
        if not info:
            return pd.DataFrame()
        df = pd.DataFrame(
            [{"value": info.get("slotsInEpoch"), "metric": "slots_in_epoch",
              "absolute_slot": info.get("absoluteSlot"),
              "block_height": info.get("blockHeight")}],
            index=[pd.Timestamp.utcnow().floor("D")],
        )
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        return df
    except Exception:
        return pd.DataFrame()


def main() -> None:
    for symbol, coin_id in SOL_ECOSYSTEM.items():
        print(f"Fetching {symbol} ({coin_id}) ...")
        try:
            metrics = fetch_coingecko_market(coin_id)
            name_map = {
                "prices": f"{symbol}_Price_1d",
                "market_caps": f"{symbol}_MarketCap_1d",
                "total_volumes": f"{symbol}_Volume_1d",
            }
            for key, fname in name_map.items():
                if key in metrics:
                    save(metrics[key], "onchain", f"{fname}.parquet")
            time.sleep(1.5)
        except Exception as exc:
            print(f"  WARNING: {symbol} — {exc}")
            time.sleep(5)

    # Solana epoch/staking snapshot
    print("Fetching Solana epoch info ...")
    df_epoch = fetch_solana_staking()
    if not df_epoch.empty:
        save(df_epoch, "onchain", "SOL_Epoch_snapshot.parquet")


if __name__ == "__main__":
    main()
