"""
UN Comtrade international trade flows collector.
Source: UN Comtrade API v2 — free tier (500 req/hour), requires free API key.
Set env var COMTRADE_API_KEY. Get free at https://comtradeplus.un.org/
Covers: Bilateral trade flows for top commodities and major country pairs.
Output: data/trade/Comtrade_<reporter>_<commodity>_1y.parquet
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

COMTRADE_API = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

# Top commodity codes (HS 2-digit level)
COMMODITIES = {
    "27": "Mineral fuels (oil, gas, coal)",
    "85": "Electrical machinery",
    "84": "Machinery",
    "72": "Iron and steel",
    "39": "Plastics",
    "87": "Vehicles",
    "71": "Precious metals/gems",
    "90": "Optical/medical instruments",
    "30": "Pharmaceuticals",
    "10": "Cereals (wheat, corn, rice)",
    "12": "Oil seeds (soybeans)",
    "15": "Animal/vegetable fats (palm oil)",
    "09": "Coffee, tea, spices",
    "26": "Ores, slag, ash (metals)",
    "74": "Copper",
}

# Major reporter countries (ISO3)
REPORTERS = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IDN", "IND", "BRA", "KOR"]


def fetch_trade_data(reporter: str, cmd_code: str, year: str, api_key: str) -> pd.DataFrame:
    params = {
        "reporterCode": reporter,
        "period": year,
        "cmdCode": cmd_code,
        "flowCode": "X,M",  # exports and imports
        "partnerCode": "0",  # world (aggregated)
        "subscription-key": api_key,
    }
    resp = fetch(COMTRADE_API, params=params, timeout=30)
    data = resp.json()
    datasets = data.get("data", [])
    if not datasets:
        return pd.DataFrame()

    df = pd.DataFrame(datasets)
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"].astype(str), format="%Y", utc=True, errors="coerce")
        df = df.set_index("period").sort_index()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("COMTRADE_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("WARNING: No Comtrade API key. Set COMTRADE_API_KEY env var.")
        print("         Get free key at https://comtradeplus.un.org/")
        return

    import datetime
    years = [str(y) for y in range(datetime.date.today().year - 9, datetime.date.today().year)]

    for reporter in REPORTERS:
        for cmd_code, cmd_name in COMMODITIES.items():
            print(f"Fetching trade: {reporter} / {cmd_name} ...")
            all_rows = []
            for year in years:
                try:
                    df = fetch_trade_data(reporter, cmd_code, year, args.api_key)
                    if not df.empty:
                        all_rows.append(df)
                except Exception as exc:
                    print(f"  WARNING: {reporter}/{cmd_code}/{year} — {exc}")

            if all_rows:
                combined = pd.concat(all_rows).sort_index()
                safe_name = cmd_name.split("(")[0].strip().replace(" ", "_").replace("/", "_")[:30]
                save(combined, "trade", f"Comtrade_{reporter}_{safe_name}_1y.parquet")


if __name__ == "__main__":
    main()
