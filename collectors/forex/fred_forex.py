"""
FRED forex rates collector.
Source: FRED (St. Louis Fed) — no API key required, uses CSV export.
Covers: Major currency pairs, trade-weighted USD indices, real effective exchange rates.
Output: data/forex/FRED_<series>_1d.parquet
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
SLEEP = 0.5

# FRED forex series — all public, CSV endpoint requires no API key
FRED_SERIES: dict[str, str] = {
    # Major currency pairs (vs USD)
    "DEXUSEU": "EUR/USD",
    "DEXJPUS": "USD/JPY",
    "DEXUSUK": "GBP/USD",
    "DEXUSAL": "AUD/USD",
    "DEXCAUS": "CAD/USD",
    "DEXCHUS": "CNY/USD",
    "DEXINUS": "INR/USD",
    "DEXKOUS": "KRW/USD",
    "DEXBZUS": "BRL/USD",
    "DEXMXUS": "MXN/USD",
    "DEXSIUS": "SGD/USD",
    "DEXHKUS": "HKD/USD",
    "DEXSFUS": "CHF/USD",
    "DEXSDUS": "SEK/USD",
    "DEXNOUS": "NOK/USD",
    "DEXDNUS": "DKK/USD",
    
    # Trade-weighted USD indices
    "DTWEXBGS": "USD_TradeWeighted_Broad",
    "DTWEXAFEGS": "USD_TradeWeighted_AdvancedEconomies",
    "DTWEXEMEGS": "USD_TradeWeighted_EmergingMarkets",
    
    # Real effective exchange rates
    "RBUSBIS": "USD_RealEffective",
    "RBGBBIS": "GBP_RealEffective",
    "RBJPBIS": "JPY_RealEffective",
    "RBCABIS": "CAD_RealEffective",
    "RBEUBIS": "EUR_RealEffective",
}


def fetch_fred_csv(series_id: str, label: str) -> pd.DataFrame | None:
    """FRED CSV export — no API key."""
    try:
        url = f"{FRED_BASE}?id={series_id}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  ✗ {label}: HTTP {resp.status_code}")
            return None
        
        df = pd.read_csv(
            pd.io.common.BytesIO(resp.content),
            parse_dates=["DATE"],
            index_col="DATE"
        )
        df.columns = ["value"]
        df = to_datetime_index(df)
        return df
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return None


def main() -> None:
    print("Fetching: FRED forex rates")
    ok = 0
    for series_id, label in FRED_SERIES.items():
        df = fetch_fred_csv(series_id, label)
        if df is not None and not df.empty:
            save(df, "forex", f"FRED_{label}_1d.parquet")
            ok += 1
        time.sleep(SLEEP)
    
    print(f"✓ {ok}/{len(FRED_SERIES)} FRED forex series saved")


if __name__ == "__main__":
    main()
