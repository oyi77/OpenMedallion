"""
US major stock indices via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/equities/sp_index_1d.parquet
Covers: S&P 500, Nasdaq Composite, Dow Jones, Russell 2000, CBOE Volatility Index
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

INDICES: list[tuple[str, str]] = [
    ("^GSPC", "SP500_INDEX"),
    ("^IXIC", "NASDAQ_INDEX"),
    ("^DJI",  "DOW_INDEX"),
    ("^RUT",  "RUSSELL2000_INDEX"),
    ("^VIX",  "VIX_INDEX"),
]

_START = "1920-01-01"
_SLEEP = 0.3


def _fetch(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=_START, progress=False, auto_adjust=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        # suffix each column with the label
        df = df.add_suffix(f"_{label}")
        return df
    except Exception as exc:
        print(f"  WARNING {label} ({ticker}): {exc}")
        return None


def collect() -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    for ticker, label in INDICES:
        df = _fetch(ticker, label)
        if df is not None and not df.empty:
            print(f"  {label:20s} {df.index[0].date()} -> {df.index[-1].date()}  ({len(df):,} rows)")
            if combined is None:
                combined = df
            else:
                combined = combined.join(df, how="outer")
        else:
            print(f"  {label:20s} no data")
        time.sleep(_SLEEP)
    if combined is None:
        return pd.DataFrame()
    combined = combined.sort_index()
    return combined


def main() -> None:
    print(f"Fetching {len(INDICES)} US indices via yfinance")
    df = collect()
    if not df.empty:
        save(df, "equities", "sp_index_1d.parquet")
    print("Done")


if __name__ == "__main__":
    main()
