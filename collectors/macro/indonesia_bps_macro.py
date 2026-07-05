"""
Indonesia macro data from World Bank Open Data API.
Source: https://api.worldbank.org/v2/country/ID/indicator/{indicator}
No API key required.
Output: data/macro/indonesia_worldbank_macro_1y.parquet — date, indicator, value
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

WB_API = "https://api.worldbank.org/v2/country/ID/indicator/{indicator}"

INDICATORS: dict[str, str] = {
    "NY.GDP.MKTP.CD":   "GDP_USD",
    "FP.CPI.TOTL.ZG":  "CPI_Inflation",
    "SL.UEM.TOTL.ZS":  "Unemployment_Rate",
    "BX.KLT.DINV.CD.WD": "FDI_Inflows_USD",
    "PA.NUS.FCRF":     "FX_IDR_USD",
}


def _fetch_indicator(code: str) -> pd.DataFrame:
    url = WB_API.format(indicator=code)
    params = {"format": "json", "per_page": 500}
    resp = fetch(url, params=params)
    data = resp.json()
    if not data or len(data) < 2 or not data[1]:
        return pd.DataFrame()
    records = [
        {"date": r["date"], "value": r["value"]}
        for r in data[1]
        if r.get("value") is not None
    ]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date").sort_index()


def collect_indonesia_macro() -> None:
    frames: list[pd.DataFrame] = []

    for code, name in INDICATORS.items():
        print(f"  Fetching WB {name} ({code}) for Indonesia ...")
        try:
            df = _fetch_indicator(code)
            if df.empty:
                print(f"    No data returned for {code}")
                continue
            df = df.rename(columns={"value": name})
            # Add long-format columns for stacking
            df_long = df.copy()
            df_long.columns = ["value"]
            df_long["indicator"] = name
            frames.append(df_long)
            print(f"    {len(df):,} rows")
        except Exception as exc:
            print(f"    WARNING: {code} — {exc}")

    if not frames:
        print("  WARNING: No Indonesia macro data retrieved")
        return

    combined = pd.concat(frames).sort_index()
    # Reset so date + indicator are both columns for long format
    combined = combined.reset_index()
    combined = combined.rename(columns={"date": "date"})
    combined = combined.set_index("date").sort_index()

    save(combined, "macro", "indonesia_worldbank_macro_1y.parquet")


def main() -> None:
    print("Fetching: Indonesia macro (World Bank)")
    try:
        collect_indonesia_macro()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
