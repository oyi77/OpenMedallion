"""
Key US macroeconomic indicators via FRED CSV export.
Source: FRED (St. Louis Fed) — no API key, public CSV endpoint.
Series: UNRATE (Unemployment), GDP, CPIAUCSL (CPI), FEDFUNDS (Fed Funds Rate),
        T10Y2Y (Yield Curve spread).
Output: data/macro/fred_macro_1d.parquet
Columns: date (DatetimeIndex), indicator, value
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SLEEP = 0.4

# (series_id, label)
FRED_SERIES: list[tuple[str, str]] = [
    ("UNRATE",   "Unemployment"),
    ("GDP",      "GDP"),
    ("CPIAUCSL", "CPI"),
    ("FEDFUNDS", "FedFundsRate"),
    ("T10Y2Y",   "YieldCurve"),
]


def fetch_fred_csv(series_id: str) -> pd.DataFrame | None:
    """Fetch one FRED series as a DataFrame with columns: date, value."""
    try:
        url = f"{FRED_BASE}?id={series_id}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(
            pd.io.common.BytesIO(resp.content),
            parse_dates=["observation_date"],
            index_col="observation_date",
            na_values=[".", ""],
        )
        df.columns = ["value"]
        df = df.dropna()
        return to_datetime_index(df)
    except Exception:
        return None


def main() -> None:
    print("Fetching: FRED macro indicators")
    frames: list[pd.DataFrame] = []
    ok = 0
    for series_id, label in FRED_SERIES:
        df = fetch_fred_csv(series_id)
        if df is not None and not df.empty:
            df["indicator"] = label
            frames.append(df)
            ok += 1
            print(f"  ✓ {label} ({len(df)} rows)")
        else:
            print(f"  ✗ {label}")
        time.sleep(SLEEP)

    if not frames:
        print("No data fetched — nothing to save")
        return

    combined = pd.concat(frames)[["indicator", "value"]].sort_index()
    combined.index.name = "date"
    save(combined, "macro", "fred_macro_1d.parquet")
    print(f"\n✓ {ok}/{len(FRED_SERIES)} FRED macro series saved")


if __name__ == "__main__":
    main()
