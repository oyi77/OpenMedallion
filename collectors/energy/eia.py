"""
EIA (US Energy Information Administration) collector.
Source: EIA Open Data API v2 — free, requires free API key.
Set env var EIA_API_KEY or pass --api-key. Get free key at https://www.eia.gov/opendata/
Covers: WTI/Brent spot prices, natural gas storage, petroleum inventory,
        electricity prices, US energy consumption.
Output: data/energy/EIA_<series>_1w.parquet or _1d.parquet
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

EIA_BASE = "https://api.eia.gov/v2"

# Series IDs -> output name, frequency
SERIES = {
    # Petroleum
    "petroleum/pri/spt": ("EIA_WTI_Crude_Spot_1d", "D", {"product": "EPCWTI", "duoarea": "NUS", "series-description": "WTI"}),
    # Natural Gas
    "natural-gas/sum/snd": ("EIA_NatGas_Storage_1w", "W", {}),
    # Electricity
    "electricity/retail-sales": ("EIA_Electricity_RetailSales_1m", "M", {}),
}

# Simpler direct series approach using v2 route
EIA_SERIES_V2 = {
    "EIA_WTI_Crude_1d": {
        "route": "petroleum/pri/spt",
        "params": {"frequency": "daily", "data[0]": "value", "facets[product][]": "EPCWTI", "facets[duoarea][]": "NUS", "length": 2000},
    },
    "EIA_Brent_Crude_1d": {
        "route": "petroleum/pri/spt",
        "params": {"frequency": "daily", "data[0]": "value", "facets[product][]": "EPCOIL", "facets[duoarea][]": "NUS", "length": 2000},
    },
    "EIA_NatGas_HenryHub_1d": {
        "route": "natural-gas/pri/fut",
        "params": {"frequency": "daily", "data[0]": "value", "facets[series][]": "RNGC1", "length": 2000},
    },
    "EIA_NatGas_Storage_1w": {
        "route": "natural-gas/sum/snd",
        "params": {"frequency": "weekly", "data[0]": "value", "length": 500},
    },
    "EIA_US_Crude_Inventory_1w": {
        "route": "petroleum/sum/crdsnd",
        "params": {"frequency": "weekly", "data[0]": "value", "facets[duoarea][]": "NUS", "length": 500},
    },
    "EIA_US_Gasoline_Price_1w": {
        "route": "petroleum/pri/gnd",
        "params": {"frequency": "weekly", "data[0]": "value", "facets[duoarea][]": "NUS", "facets[series][]": "EMM_EPM0_PTE_NUS_DPG", "length": 500},
    },
}


def fetch_eia_series(route: str, params: dict, api_key: str) -> pd.DataFrame:
    url = f"{EIA_BASE}/{route}/data/"
    all_params = {"api_key": api_key, "offset": 0, **params}
    resp = fetch(url, params=all_params)
    data = resp.json()

    if "response" not in data or "data" not in data["response"]:
        return pd.DataFrame()

    records = data["response"]["data"]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Find period/date column
    date_col = next((c for c in df.columns if c in ["period", "date", "reportDate"]), None)
    if not date_col:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Find value column
    val_col = next((c for c in df.columns if c == "value"), None)
    if val_col:
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        return df[[val_col]].dropna()

    # Numeric columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return df[num_cols].dropna(how="all") if num_cols else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("EIA_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("WARNING: No EIA API key. Set EIA_API_KEY env var or pass --api-key.")
        print("         Get free key at https://www.eia.gov/opendata/register.php")
        return

    for name, cfg in EIA_SERIES_V2.items():
        print(f"Fetching {name} ...")
        try:
            df = fetch_eia_series(cfg["route"], cfg["params"], args.api_key)
            save(df, "energy", f"{name}.parquet")
        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
