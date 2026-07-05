"""
Indonesian government bond yield proxies from World Bank Open Data.
Source: World Bank API v2 — free, no key required.
FRED requires a real API key; we use World Bank rate indicators instead:
  - FR.INR.LEND  : lending rate as long-term yield proxy
  - FR.INR.DPST  : deposit rate as short-term yield proxy
  - FR.INR.RINR  : real interest rate
Output: data/macro/indonesia_bond_yields_1m.parquet — date, yield_long, yield_short, real_rate
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

WB_API = "https://api.worldbank.org/v2/country/ID/indicator/{indicator}"

# (indicator_code, output_column)
SERIES: list[tuple[str, str]] = [
    ("FR.INR.LEND", "yield_long"),   # lending rate ≈ long-term yield proxy
    ("FR.INR.DPST", "yield_short"),  # deposit rate ≈ short-term yield proxy
    ("FR.INR.RINR", "real_rate"),
]


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


def collect_bond_yields() -> None:
    frames: dict[str, pd.DataFrame] = {}

    for code, col in SERIES:
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
        print("  WARNING: No bond yield data retrieved")
        return

    combined = pd.concat(list(frames.values()), axis=1, join="outer").sort_index()
    save(combined, "macro", "indonesia_bond_yields_1m.parquet")


def main() -> None:
    print("Fetching: Indonesia bond yield proxies (World Bank)")
    try:
        collect_bond_yields()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
