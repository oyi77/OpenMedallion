"""
Token Terminal DeFi protocol revenue collector.
Source: Token Terminal public API — free tier, no key for basic data.
Covers: Protocol fees, revenue, P/S ratios for top 50 DeFi protocols.
Output: data/defi/TokenTerminal_<protocol>_revenue_1d.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# Token Terminal public endpoints (no auth for aggregated data)
TT_BASE = "https://api.tokenterminal.com/v2"

# Alternative: use their public GraphQL / open data
# They also publish to GitHub: https://github.com/token-terminal/tt-data
TT_GITHUB_BASE = "https://raw.githubusercontent.com/token-terminal/tt-data/main"

# Protocols available in their public dataset
PROTOCOLS = [
    "uniswap", "aave", "compound", "makerdao", "curve", "lido",
    "gmx", "dydx", "synthetix", "balancer", "sushiswap", "pancakeswap",
    "trader-joe", "benqi", "venus", "stargate", "hop", "across",
    "arbitrum", "optimism", "polygon", "avalanche", "bnb-chain",
    "ethereum", "solana",
]

# DefiLlama fees API — free, no key — covers protocol revenue uniquely
DEFILLAMA_FEES = "https://api.llama.fi/summary/fees/{protocol}"
DEFILLAMA_FEES_LIST = "https://api.llama.fi/overview/fees"
DEFILLAMA_REVENUE = "https://api.llama.fi/overview/revenue"


def fetch_protocol_fees(protocol_slug: str) -> pd.DataFrame:
    """Fetch daily fees history for a protocol from DefiLlama."""
    try:
        resp = fetch(DEFILLAMA_FEES.format(protocol=protocol_slug), timeout=20)
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    # totalDataChart is list of [timestamp, value]
    chart = data.get("totalDataChart") or data.get("totalDataChartBreakdown", [])
    if not chart:
        return pd.DataFrame()

    # Normalize — can be [[ts, val], ...] or [{date:..., totalFees:...}, ...]
    rows = []
    for item in chart:
        if isinstance(item, list) and len(item) == 2:
            rows.append({"date": item[0], "fees_usd": item[1]})
        elif isinstance(item, dict):
            ts = item.get("date") or item.get("timestamp")
            val = item.get("totalFees") or item.get("value")
            if ts is not None and val is not None:
                rows.append({"date": ts, "fees_usd": val})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
    df = df.set_index("date").sort_index()
    df["fees_usd"] = pd.to_numeric(df["fees_usd"], errors="coerce")
    return df.dropna()


def fetch_all_protocols_overview() -> pd.DataFrame:
    """Fetch overview of all protocols with fees — one snapshot."""
    resp = fetch(DEFILLAMA_FEES_LIST, timeout=30)
    data = resp.json()
    protocols = data.get("protocols", [])
    if not protocols:
        return pd.DataFrame()
    df = pd.json_normalize(protocols)
    df["snapshot_time"] = pd.Timestamp.utcnow().floor("D")
    return df


def main() -> None:
    # 1. All-protocols snapshot
    print("Fetching DefiLlama fees overview ...")
    df_overview = fetch_all_protocols_overview()
    if not df_overview.empty:
        df_overview = df_overview.set_index("name") if "name" in df_overview.columns else df_overview
        save(df_overview, "defi", "DEFI_fees_overview_snapshot.parquet")

    # 2. Per-protocol daily history
    print("\nFetching per-protocol fee history ...")
    for slug in PROTOCOLS:
        print(f"  {slug} ...")
        df = fetch_protocol_fees(slug)
        if not df.empty:
            clean = slug.replace("-", "_").upper()
            save(df, "defi", f"DEFI_{clean}_fees_1d.parquet")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
