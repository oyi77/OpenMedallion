"""
Bitcoin network metrics collector (combined long-format output).
Source: blockchain.com charts API (free, no key)
Output: data/onchain/bitcoin_network_1d.parquet
Columns: date (DatetimeIndex), metric, value
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

BLOCKCHAIN_BASE = "https://api.blockchain.info/charts/{chart}"

# Network-focused chart slugs from blockchain.com
NETWORK_CHARTS: dict[str, str] = {
    "hash-rate": "hash-rate",
    "difficulty": "difficulty",
    "n-transactions": "n-transactions",
    "mempool-size": "mempool-size",
    "avg-block-size": "avg-block-size",
    "miners-revenue": "miners-revenue",
}


def fetch_chart(chart_name: str) -> pd.DataFrame | None:
    """Fetch a single blockchain.com chart and return a DataFrame with metric + value."""
    url = BLOCKCHAIN_BASE.format(chart=chart_name)
    resp = fetch(url, params={"timespan": "all", "format": "json", "sampled": "true"}, timeout=30)
    data = resp.json()
    values = data.get("values", [])
    if not values:
        print(f"  SKIP {chart_name} — no data")
        return None

    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df[["date", "y"]].rename(columns={"y": "value"})
    df["metric"] = chart_name
    df = df.set_index("date")[["metric", "value"]].sort_index()
    print(f"  OK {chart_name} — {len(df)} rows")
    return df


def main() -> None:
    print("=== Bitcoin network metrics (blockchain.com) ===")
    frames: list[pd.DataFrame] = []

    for chart_name in NETWORK_CHARTS:
        result = fetch_chart(chart_name)
        if result is not None:
            frames.append(result)
        time.sleep(0.4)

    if not frames:
        print("  No data collected — skipping save")
        return

    combined = pd.concat(frames).sort_index()
    save(combined, "onchain", "bitcoin_network_1d.parquet")


if __name__ == "__main__":
    main()
