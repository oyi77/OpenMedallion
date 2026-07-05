"""
World Bank global indicators collector.
Source: World Bank API v2 — free, no API key.
Fetches all countries, specific development indicators.
Output: data/macro/worldbank_global_indicators_1y.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

WB_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"

# Indicators: WB code -> human label
INDICATORS: dict[str, str] = {
    "NY.GDP.MKTP.CD": "GDP_USD",
    "SP.POP.TOTL": "Population",
    "NY.GNP.PCAP.CD": "GNI_PerCapita_USD",
    "IT.NET.USER.ZS": "Internet_Users_Pct",
    "EG.USE.ELEC.KH.PC": "Electric_Power_kWh_PerCapita",
}


def fetch_global_indicator(indicator_code: str, indicator_label: str) -> pd.DataFrame:
    """Fetch all countries for one indicator, most recent 30 values."""
    url = WB_API.format(indicator=indicator_code)
    params: dict[str, str | int] = {
        "format": "json",
        "per_page": 1000,
        "mrv": 30,
    }
    resp = fetch(url, params=params, timeout=60)
    payload = resp.json()

    # World Bank returns [metadata, data_array]
    if not payload or len(payload) < 2 or not payload[1]:
        return pd.DataFrame()

    rows: list[dict] = []
    for record in payload[1]:
        if record.get("value") is None:
            continue
        country_code = record.get("countryiso3code") or record.get("country", {}).get("id", "")
        if not country_code:
            continue
        try:
            rows.append(
                {
                    "date": pd.Timestamp(f"{record['date']}-01-01", tz="UTC"),
                    "country": country_code,
                    "indicator": indicator_label,
                    "value": float(record["value"]),
                }
            )
        except (ValueError, TypeError, KeyError):
            continue

    return pd.DataFrame(rows)


def main() -> None:
    all_frames: list[pd.DataFrame] = []

    for code, label in INDICATORS.items():
        print(f"Fetching World Bank {label} ({code}) ...")
        try:
            df = fetch_global_indicator(code, label)
            if df.empty:
                print(f"  WARNING: no data for {code}")
                continue
            all_frames.append(df)
            print(f"  {len(df):,} rows for {label}")
        except Exception as exc:
            print(f"  WARNING: {code} — {exc}")

    if not all_frames:
        print("  WARNING: all World Bank indicators failed — skipping save")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    # Set date as index
    combined = combined.sort_values(["indicator", "country", "date"])
    combined = combined.set_index("date").sort_index()
    save(combined, "macro", "worldbank_global_indicators_1y.parquet")


if __name__ == "__main__":
    main()
