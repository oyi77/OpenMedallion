"""
DEX trading volume snapshot from DeFiLlama (free, no API key).

Source:  https://api.llama.fi/overview/dexs
Grabs the top 20 DEXes by 24-hour volume, broken down per chain.

Output:  data/crypto/dex_volume_1d.parquet
Columns: date (UTC DatetimeIndex), dex_name, chain, volume_24h,
         total_24h, change_1d
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

LLAMA_DEX_OVERVIEW = "https://api.llama.fi/overview/dexs"
TOP_N = 20


def _fetch_dex_overview() -> list[dict]:
    """Fetch the full DEX overview from DeFiLlama."""
    resp = fetch(LLAMA_DEX_OVERVIEW)
    data = resp.json()
    return data.get("protocols", [])


def _build_rows(protocols: list[dict]) -> list[dict]:
    """Extract per-chain volume rows for the top N DEXes by total 24h volume."""
    # Filter out entries without valid volume
    valid = [p for p in protocols if isinstance(p.get("total24h"), (int, float)) and p["total24h"] > 0]
    valid.sort(key=lambda p: p["total24h"], reverse=True)
    top = valid[:TOP_N]

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows: list[dict] = []

    for protocol in top:
        name = protocol["name"]
        total_24h = protocol["total24h"]
        change_1d = protocol.get("change_1d")

        breakdown = protocol.get("breakdown24h")
        if isinstance(breakdown, dict) and breakdown:
            # One row per chain that has volume for this DEX
            for chain, dex_volumes in breakdown.items():
                if not isinstance(dex_volumes, dict):
                    continue
                volume = dex_volumes.get(name, 0)
                if volume and volume > 0:
                    rows.append({
                        "date": now,
                        "dex_name": name,
                        "chain": chain,
                        "volume_24h": volume,
                        "total_24h": total_24h,
                        "change_1d": change_1d,
                    })
        else:
            # No chain breakdown — single aggregate row
            rows.append({
                "date": now,
                "dex_name": name,
                "chain": "all",
                "volume_24h": total_24h,
                "total_24h": total_24h,
                "change_1d": change_1d,
            })

    return rows


def collect_dex_volume() -> None:
    """Fetch top-20 DEX volumes and persist to parquet."""
    protocols = _fetch_dex_overview()
    print(f"  Fetched {len(protocols)} DEX protocols from DeFiLlama")

    rows = _build_rows(protocols)
    if not rows:
        print("  WARNING: No DEX volume rows extracted")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    save(df, "crypto", "dex_volume_1d.parquet")
    print(f"  Top DEXes: {', '.join(df['dex_name'].unique()[:5])} ...")


def main() -> None:
    print("=== DEX Volume Snapshot (DeFiLlama) ===")
    try:
        collect_dex_volume()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
