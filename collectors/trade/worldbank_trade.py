"""
World Bank trade data collector — no API key required.
Source: World Bank Open Data API (free, no key)
Covers: Trade as % of GDP, merchandise exports/imports, top-10 trading economies.
Output: data/trade/WB_trade_<indicator>_1y.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

WB_API = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

# Indicators: code → label
INDICATORS = {
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "TX.VAL.MRCH.CD.WT": "merchandise_exports_usd",
    "TM.VAL.MRCH.CD.WT": "merchandise_imports_usd",
    "BX.GSR.TOTL.CD": "service_exports_usd",
    "BM.GSR.TOTL.CD": "service_imports_usd",
    "TX.VAL.MANF.ZS.UN": "manufactures_exports_pct",
    "TM.VAL.MANF.ZS.UN": "manufactures_imports_pct",
    "TX.VAL.AGRI.ZS.UN": "agri_exports_pct",
    "TM.VAL.AGRI.ZS.UN": "agri_imports_pct",
    "TX.VAL.FUEL.ZS.UN": "fuel_exports_pct",
    "TM.VAL.FUEL.ZS.UN": "fuel_imports_pct",
}

# Major economies + Indonesia
COUNTRIES = ["US", "CN", "DE", "JP", "GB", "FR", "ID", "IN", "BR", "KR", "SG", "AU", "CA", "MX", "ZA"]


def fetch_wb(country: str, indicator: str) -> pd.DataFrame:
    url = WB_API.format(country=country, indicator=indicator)
    resp = fetch(url, params={"format": "json", "per_page": "60", "mrv": "30"}, timeout=30)
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        return pd.DataFrame()
    records = payload[1]
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        val = r.get("value")
        date_str = r.get("date", "")
        if val is None or not date_str:
            continue
        try:
            year = int(date_str[:4])
        except ValueError:
            continue
        rows.append({"date": pd.Timestamp(f"{year}-01-01", tz="UTC"), "value": float(val), "country": country, "indicator": indicator})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def main() -> None:
    for indicator, label in INDICATORS.items():
        print(f"Fetching WB trade indicator: {label} ...")
        all_parts: list[pd.DataFrame] = []
        for country in COUNTRIES:
            try:
                df = fetch_wb(country, indicator)
                if not df.empty:
                    all_parts.append(df)
            except Exception as exc:
                print(f"  WARNING: {country}/{label} — {exc}")

        if all_parts:
            combined = pd.concat(all_parts).sort_index()
            save(combined, "trade", f"WB_{label}_1y.parquet")
            print(f"  Saved {len(combined)} rows for {label}")
        else:
            print(f"  No data for {label}")


if __name__ == "__main__":
    main()
