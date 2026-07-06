"""
EIA Crude Oil Inventories — Weekly (v2 API).
Source: EIA Open Data API v2 (free key).
Set env var EIA_API_KEY or pass --api-key. Get free key at https://www.eia.gov/opendata/register.php
Covers: US weekly crude oil commercial stocks (WCESTUS1).
Output: data/energy/crude_oil_inventories_1w.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

EIA_V2_BASE = "https://api.eia.gov/v2"
ROUTE = "petroleum/stoc/wstoc/data/"
PAGE_SIZE = 5000


def fetch_crude_inventories(api_key: str) -> pd.DataFrame:
    """Fetch US crude oil commercial stocks (weekly) via EIA v2 API.

    Pagination handles datasets larger than PAGE_SIZE rows.
    Returns DataFrame indexed by UTC date with columns:
      - inventory_barrels (thousand barrels)
      - change_barrels   (week-over-week change, thousand barrels)
    """
    url = f"{EIA_V2_BASE}/{ROUTE}"
    all_records: list[dict] = []
    offset = 0

    while True:
        params = {
            "api_key": api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[product][]": "EPM0",
            "facets[du][]": "NUS",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": offset,
            "length": PAGE_SIZE,
        }
        resp = fetch(url, params=params)
        body = resp.json()

        response = body.get("response", {})
        rows = response.get("data", [])
        if not rows:
            break

        all_records.extend(rows)
        total = response.get("total", 0)
        offset += PAGE_SIZE
        if offset >= total:
            break

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Build clean DataFrame: date → inventory_barrels
    df["date"] = pd.to_datetime(df["period"], utc=True, errors="coerce")
    df["inventory_barrels"] = pd.to_numeric(df["value"], errors="coerce")
    df = (
        df[["date", "inventory_barrels"]]
        .dropna()
        .sort_values("date")
        .set_index("date")
    )

    # Week-over-week change
    df["change_barrels"] = df["inventory_barrels"].diff()

    return df


def main() -> None:
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        print("WARNING: No EIA API key. Set EIA_API_KEY env var.")
        print("         Get free key at https://www.eia.gov/opendata/register.php")
        return

    print("--- EIA Crude Oil Inventories (v2 API) ---")
    df = fetch_crude_inventories(api_key)
    save(df, "energy", "crude_oil_inventories_1w.parquet")


if __name__ == "__main__":
    main()
