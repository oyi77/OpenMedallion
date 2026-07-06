"""
FRED trade & current account data collector — no API key required.
Source: FRED (St. Louis Fed) CSV endpoint — free, no key.
Covers: US trade balance, goods/services exports/imports, current account.
Output: data/trade/FRED_<series>_1m.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import save

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Series: FRED ID → output label
SERIES = {
    "BOPGSTB":  "TradeBalance_GoodsServices_1m",      # Trade balance total
    "BOPGTB":   "TradeBalance_Goods_1m",              # Goods trade balance
    "BOPSTB":   "TradeBalance_Services_1m",           # Services trade balance
    "EXPGS":    "Exports_GoodsServices_1q",           # Total exports (quarterly, billions)
    "IMPGS":    "Imports_GoodsServices_1q",           # Total imports (quarterly)
    "IEAXGS":   "Exports_Goods_1m",                   # Goods exports monthly
    "IEAMGS":   "Imports_Goods_1m",                   # Goods imports monthly
    "NETEXP":   "NetExports_1q",                      # Net exports GDP component
    "BOPBCA":   "CurrentAccount_1q",                  # Current account balance
    "XTEXVA01USM664S": "Exports_Value_1m",            # Export value index
    "XTIMVA01USM664S": "Imports_Value_1m",            # Import value index
    "DGSITEAM": "Imports_China_1m",                   # US imports from China
    "DGSEXAM":  "Exports_China_1m",                   # US exports to China
}


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    import requests
    url = FRED_CSV.format(series=series_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        return pd.DataFrame()

    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str, val_str = parts[0].strip(), parts[1].strip()
        if val_str in (".", "", "NA"):
            continue
        try:
            dt = pd.Timestamp(date_str, tz="UTC")
            val = float(val_str)
            rows.append({"date": dt, "value": val})
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.index.name = "date"
    return df


def main() -> None:
    for series_id, label in SERIES.items():
        print(f"Fetching FRED: {series_id} ({label}) ...")
        try:
            df = fetch_fred_series(series_id)
            if df.empty:
                print(f"  No data for {series_id}")
                continue
            save(df, "trade", f"FRED_{label}.parquet")
            print(f"  Saved {len(df)} rows → data/trade/FRED_{label}.parquet")
        except Exception as exc:
            print(f"  WARNING: {series_id} failed — {exc}")


if __name__ == "__main__":
    main()
