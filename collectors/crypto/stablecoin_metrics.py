"""
Stablecoin metrics snapshot via DeFiLlama Stablecoins API (free, no key).

Fetches current snapshot data for the six major USD stablecoins:
  USDT (id=1), USDC (id=2), DAI (id=5), BUSD (id=4), TUSD (id=7), USDP (id=11)

Uses the stablecoins list endpoint:
  https://stablecoins.llama.fi/stablecoins?includePrices=true

Columns: date, stablecoin, peg, circulating, chain, price, price_change_1d
  - date:             UTC date of collection
  - stablecoin:       ticker symbol (USDT, USDC, ...)
  - peg:              peg type from API (e.g. peggedUSD)
  - circulating:      total circulating supply in USD
  - chain:            top 3 chains by adoption (comma-separated)
  - price:            current market price
  - price_change_1d:  daily supply change pct vs previous day

Output: data/crypto/stablecoin_metrics_1d.parquet
Index:  stablecoin (each row = one stablecoin snapshot)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins"

# DeFiLlama IDs for the six target stablecoins (string, matching API format)
TARGET_IDS: dict[str, str] = {
    "USDT": "1",
    "USDC": "2",
    "DAI":  "5",
    "BUSD": "4",
    "TUSD": "7",
    "USDP": "11",
}


def collect_stablecoin_metrics() -> None:
    """Fetch stablecoin list with prices and build metrics table."""
    import requests

    resp = requests.get(STABLECOINS_URL, params={"includePrices": True}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    assets: list[dict] = payload.get("peggedAssets", [])
    if not assets:
        print("  WARNING: no stablecoin assets in API response")
        return

    id_to_asset = {str(a["id"]): a for a in assets}
    records: list[dict] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for symbol, lid in TARGET_IDS.items():
        asset = id_to_asset.get(lid)
        if asset is None:
            print(f"  WARNING: id={lid} ({symbol}) not found in API response")
            continue

        circulating_raw = asset.get("circulating", {})
        circulating: float = circulating_raw.get("peggedUSD", 0.0) if isinstance(circulating_raw, dict) else 0.0

        prev_day_raw = asset.get("circulatingPrevDay", {})
        prev_day: float = prev_day_raw.get("peggedUSD", 0.0) if isinstance(prev_day_raw, dict) else 0.0

        price: float = asset.get("price", 0.0) or 0.0

        # price_change_1d: daily supply change % (consistent with snapshot semantics)
        supply_change: float = 0.0
        if prev_day > 0:
            supply_change = round((circulating / prev_day - 1.0) * 100.0, 4)

        # Top 3 chains
        chains_raw = asset.get("chains", [])
        top_chains = chains_raw[:3] if chains_raw else []
        chain_str = ", ".join(top_chains) if top_chains else "N/A"

        records.append({
            "date": today,
            "stablecoin": symbol,
            "peg": asset.get("pegType", "peggedUSD"),
            "circulating": circulating,
            "chain": chain_str,
            "price": price,
            "price_change_1d": supply_change,
        })

    if not records:
        print("  WARNING: no stablecoin metrics collected")
        return

    df = pd.DataFrame(records)
    df = to_datetime_index(df)
    df.index.name = "date"

    save(df, "crypto", "stablecoin_metrics_1d.parquet")


def main() -> None:
    print("Fetching: Stablecoin Metrics (DeFiLlama)")
    try:
        collect_stablecoin_metrics()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
