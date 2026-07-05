"""
Stablecoin supply history via DeFiLlama Stablecoins API (free, no key).

Fetches daily circulating supply (USD) for:
  USDT (id=1), USDC (id=2), DAI (id=5), BUSD (id=4), FRAX (id=6), TUSD (id=7)

Uses the per-stablecoin chart endpoint:
  https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=<id>

Output: data/crypto/stablecoin_supply_1d.parquet
Columns: usdt_supply, usdc_supply, dai_supply, busd_supply, frax_supply,
         tusd_supply, total_stablecoin_supply
Index:   date (UTC DatetimeIndex)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

LLAMA_BASE = "https://stablecoins.llama.fi/stablecoincharts/all"

# (llama_id, column_name)
STABLECOINS: list[tuple[int, str]] = [
    (1, "usdt_supply"),
    (2, "usdc_supply"),
    (5, "dai_supply"),
    (4, "busd_supply"),
    (6, "frax_supply"),
    (7, "tusd_supply"),
]


def _fetch_supply_series(llama_id: int) -> pd.Series:
    """Return a daily Series of USD circulating supply for a stablecoin."""
    resp = fetch(LLAMA_BASE, params={"stablecoin": llama_id})
    rows: list[dict] = resp.json()

    if not rows:
        return pd.Series(dtype=float)

    records: list[tuple[int, float]] = []
    for row in rows:
        ts: int = int(row["date"])
        # totalCirculatingUSD.peggedUSD gives the USD-denominated total
        usd_val: float = (
            row.get("totalCirculatingUSD", {}).get("peggedUSD", float("nan"))
        )
        records.append((ts, usd_val))

    df = pd.DataFrame(records, columns=["ts", "supply"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.normalize()
    df = df.drop_duplicates("date").set_index("date")["supply"]
    df = df.dropna()
    return df


def collect_stablecoin_supply() -> None:
    """Fetch per-stablecoin histories and merge into a wide supply table."""
    series_map: dict[str, pd.Series] = {}

    for llama_id, col_name in STABLECOINS:
        print(f"  Fetching stablecoin supply: {col_name} (id={llama_id})")
        try:
            s = _fetch_supply_series(llama_id)
            if s.empty:
                print(f"    WARNING: no data for id={llama_id}")
            else:
                print(f"    {len(s):,} daily rows")
                series_map[col_name] = s
        except Exception as exc:
            print(f"    WARNING: id={llama_id} — {exc}")
        time.sleep(0.5)

    if not series_map:
        print("  WARNING: no stablecoin data collected")
        return

    combined = pd.DataFrame(series_map)
    combined.index.name = "date"
    combined = combined.sort_index()
    supply_cols = list(series_map.keys())
    combined["total_stablecoin_supply"] = combined[supply_cols].sum(axis=1, min_count=1)

    save(combined, "crypto", "stablecoin_supply_1d.parquet")


def main() -> None:
    print("Fetching: Stablecoin Supply (DeFiLlama)")
    try:
        collect_stablecoin_supply()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
