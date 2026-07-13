"""
S&P 500 fundamentals collector.
Source: FRED (St. Louis Fed) — no API key required, uses CSV export.
Covers: S&P 500 earnings, dividends, P/E ratio, earnings yield, price level.
Output: data/macro/sp_fundamentals_1d.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SLEEP = 0.5

# (series_id, label) — all public CSV, no API key
FRED_SERIES: list[tuple[str, str]] = [
    ("SPASTT01USM661N",                        "SP500_PE_RATIO"),
    ("MULT",                                   "SP500_PE_SHILLER"),
    ("SPDIV",                                  "SP500_DIVIDEND"),
    ("BOGZ1FA103064103Q",                      "SP500_EARNINGS_GROWTH"),
    ("SPEARN",                                 "SP500_EARNINGS"),
    ("DIVIDEND",                               "US_DIVIDEND_INDEX"),
    ("CDA0960A-1861-4C31-A5FB-0E2B1DEC354B",  "SP500_SALES_PER_SHARE"),
    ("SP500",                                  "SP500_PRICE_LEVEL"),
]


def fetch_fred_csv(series_id: str, label: str) -> pd.DataFrame | None:
    """Fetch one FRED series via CSV export — no API key."""
    try:
        url = f"{FRED_BASE}?id={series_id}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  x {label}: HTTP {resp.status_code}")
            return None

        df = pd.read_csv(
            pd.io.common.BytesIO(resp.content),
            parse_dates=["observation_date"],
            index_col="observation_date",
            na_values=[".", ""],
        )
        df = df.dropna()
        if df.empty:
            print(f"  x {label}: empty after dropna")
            return None

        df.index.name = "date"
        col_name = df.columns[0]
        df = df.rename(columns={col_name: label})
        return df[[label]]
    except Exception as e:
        print(f"  x {label}: {e}")
        return None


def main() -> None:
    print("Fetching: S&P 500 fundamentals")
    frames: list[pd.DataFrame] = []
    ok = 0

    for series_id, label in FRED_SERIES:
        df = fetch_fred_csv(series_id, label)
        if df is not None and not df.empty:
            frames.append(df)
            ok += 1
            print(f"  v {label} ({len(df)} rows)")
        time.sleep(SLEEP)

    if not frames:
        print("x No S&P 500 fundamental data collected")
        return

    # Outer-join on date index — each series becomes its own column
    combined = frames[0]
    for df in frames[1:]:
        combined = combined.join(df, how="outer")
    combined = combined.sort_index()

    save(combined, "macro", "sp_fundamentals_1d.parquet")
    print(f"\nv {ok}/{len(FRED_SERIES)} S&P 500 fundamental series saved to data/macro/sp_fundamentals_1d.parquet")


if __name__ == "__main__":
    main()
