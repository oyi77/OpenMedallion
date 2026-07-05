"""
Bank Indonesia benchmark interest rate collector.
Source: World Bank Open Data API — free, no key required.
Indicators used (annual, country ID = Indonesia):
  - FR.INR.DPST  : deposit interest rate (%)
  - FR.INR.LEND  : lending interest rate (%)
  - FR.INR.RINR  : real interest rate (%)
Output: data/macro/indonesia_bi_rate_1m.parquet — date, deposit_rate, lending_rate, real_rate
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

WB_API = "https://api.worldbank.org/v2/country/ID/indicator/{indicator}"

RATE_INDICATORS: dict[str, str] = {
    "FR.INR.DPST": "deposit_rate",
    "FR.INR.LEND": "lending_rate",
    "FR.INR.RINR": "real_rate",
}


def _fetch_wb_indicator(code: str) -> pd.DataFrame:
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


def collect_bi_rates() -> None:
    frames: dict[str, pd.DataFrame] = {}

    for code, col in RATE_INDICATORS.items():
        print(f"  Fetching WB {code} → {col} ...")
        try:
            df = _fetch_wb_indicator(code)
            if df.empty:
                print(f"    No data for {code}")
                continue
            df = df.rename(columns={"value": col})
            frames[col] = df
            print(f"    {len(df):,} rows")
        except Exception as exc:
            print(f"    WARNING: {code} — {exc}")

    if not frames:
        print("  WARNING: No BI rate data retrieved")
        return

    combined = pd.concat(list(frames.values()), axis=1, join="outer").sort_index()
    save(combined, "macro", "indonesia_bi_rate_1m.parquet")


def main() -> None:
    print("Fetching: Bank Indonesia interest rates (World Bank)")
    try:
        collect_bi_rates()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
