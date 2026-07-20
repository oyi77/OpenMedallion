"""
Treasury futures via Yahoo Finance (yfinance).
Source: Yahoo Finance — free, no API key required.
Output: data/derivatives/<LABEL>_1d.parquet  (OHLCV daily)
Covers: 2Y, 5Y, 10Y, Ultra 10Y, 30Y, Ultra T-Bond futures.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

from collectors.base import HISTORY_START
_START = HISTORY_START or "1977-01-01"
_SLEEP = 0.3

# (yfinance_ticker, label)
FUTURES: list[tuple[str, str]] = [
    ("ZN=F", "TNOTE_10Y_FUTURES"),
    ("ZB=F", "TBOND_30Y_FUTURES"),
    ("ZF=F", "TNOTE_5Y_FUTURES"),
    ("ZT=F", "TNOTE_2Y_FUTURES"),
    ("UB=F", "ULTRABOND_FUTURES"),
    ("TN=F", "TNOTE_10Y_ULTRA_FUTURES"),
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
        return df
    except Exception as exc:
        print(f"  WARNING {label} ({ticker}): {exc}")
        return None


def collect_futures() -> None:
    for ticker, label in FUTURES:
        df = _fetch(ticker, label)
        if df is not None:
            save(df, "derivatives", f"{label}_1d.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    print(f"Fetching {len(FUTURES)} treasury futures (daily OHLCV) via yfinance")
    collect_futures()


if __name__ == "__main__":
    main()
