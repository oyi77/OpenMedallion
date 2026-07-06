"""
FRED forex rates collector.
Source: FRED (St. Louis Fed) — no API key required, uses CSV export.
Covers: Major currency pairs, trade-weighted USD indices, real effective exchange rates.
Output: data/forex/fred_forex_1d.parquet (consolidated — date, pair, rate)
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
    # Major currency pairs (vs USD)
    ("DEXUSEU",    "EUR/USD"),
    ("DEXJPUS",    "USD/JPY"),
    ("DEXUSUK",    "GBP/USD"),
    ("DEXUSAL",    "AUD/USD"),
    ("DEXCAUS",    "USD/CAD"),
    ("DEXCHUS",    "CNY/USD"),
    ("DEXINUS",    "INR/USD"),
    ("DEXKOUS",    "KRW/USD"),
    ("DEXBZUS",    "BRL/USD"),
    ("DEXMXUS",    "MXN/USD"),
    ("DEXSIUS",    "SGD/USD"),
    ("DEXHKUS",    "HKD/USD"),
    ("DEXSFUS",    "CHF/USD"),
    ("DEXSDUS",    "SEK/USD"),
    ("DEXNOUS",    "NOK/USD"),
    ("DEXDNUS",    "DKK/USD"),

    # Trade-weighted USD indices
    ("DTWEXBGS",   "USD_TradeWeighted_Broad"),
    ("DTWEXAFEGS", "USD_TradeWeighted_AdvancedEconomies"),
    ("DTWEXEMEGS", "USD_TradeWeighted_EmergingMarkets"),

    # Real effective exchange rates (BIS)
    ("RBUSBIS",    "USD_RealEffective"),
    ("RBGBBIS",    "GBP_RealEffective"),
    ("RBJPBIS",    "JPY_RealEffective"),
    ("RBCABIS",    "CAD_RealEffective"),
]


def fetch_fred_csv(series_id: str, label: str) -> pd.DataFrame | None:
    """Fetch one FRED series via CSV export — no API key."""
    try:
        url = f"{FRED_BASE}?id={series_id}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  ✗ {label}: HTTP {resp.status_code}")
            return None

        df = pd.read_csv(
            pd.io.common.BytesIO(resp.content),
            parse_dates=["observation_date"],
            index_col="observation_date",
            na_values=[".", ""],
        )
        df = df.dropna()
        if df.empty:
            print(f"  ✗ {label}: empty after dropna")
            return None

        df.index.name = "date"
        df.columns = ["rate"]
        df["pair"] = label
        return df[["pair", "rate"]]
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return None


def main() -> None:
    print("Fetching: FRED forex rates")
    frames: list[pd.DataFrame] = []
    ok = 0

    for series_id, label in FRED_SERIES:
        df = fetch_fred_csv(series_id, label)
        if df is not None and not df.empty:
            frames.append(df)
            ok += 1
            print(f"  ✓ {label} ({len(df)} rows)")
        time.sleep(SLEEP)

    if not frames:
        print("✗ No FRED forex data collected")
        return

    combined = pd.concat(frames).sort_index()
    save(combined, "forex", "fred_forex_1d.parquet")
    print(f"\n✓ {ok}/{len(FRED_SERIES)} FRED forex series saved to data/forex/fred_forex_1d.parquet")


if __name__ == "__main__":
    main()
