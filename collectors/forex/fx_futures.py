"""
Major FX futures via Yahoo Finance (yfinance).
Source: Yahoo Finance — free, no API key required.
Output: data/forex/fx_futures_1d.parquet  (OHLCV daily, all contracts stacked)
Covers: EUR, GBP, JPY, AUD, CAD, CHF, NZD, MXN futures + USD index.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

_START = "1990-01-01"
_SLEEP = 0.3

# (yfinance_ticker, label)
FUTURES: list[tuple[str, str]] = [
    ("6E=F", "EUR_FUTURES"),
    ("6B=F", "GBP_FUTURES"),
    ("6J=F", "JPY_FUTURES"),
    ("6A=F", "AUD_FUTURES"),
    ("6C=F", "CAD_FUTURES"),
    ("6S=F", "CHF_FUTURES"),
    ("6N=F", "NZD_FUTURES"),
    ("6M=F", "MXN_FUTURES"),
    ("DX-Y.NYB", "USD_INDEX"),
]


def _fetch(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=_START, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["ticker"] = label
        return df
    except Exception as exc:
        print(f"  WARNING {label} ({ticker}): {exc}")
        return None


def collect_futures() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker, label in FUTURES:
        df = _fetch(ticker, label)
        if df is not None:
            frames.append(df)
            print(f"  {label}: {len(df)} rows")
        time.sleep(_SLEEP)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def main() -> None:
    print(f"Fetching {len(FUTURES)} FX futures (daily OHLCV) via yfinance")
    df = collect_futures()
    if not df.empty:
        save(df, "forex", "fx_futures_1d.parquet")


if __name__ == "__main__":
    main()
