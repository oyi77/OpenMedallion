"""
Major equity index futures + VIX volatility index via Yahoo Finance (yfinance).
Source: Yahoo Finance — free, no API key required.
Output: data/derivatives/equity_index_futures_1d.parquet  (OHLCV daily)
Covers:
  - ES=F  — S&P 500 E-mini futures
  - NQ=F  — Nasdaq 100 E-mini futures
  - RTY=F — Russell 2000 futures
  - YM=F  — Dow Jones Mini futures
  - ^VIX  — VIX volatility index spot
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
    ("ES=F",  "SPX_EMINI_FUTURES"),
    ("NQ=F",  "NDX100_EMINI_FUTURES"),
    ("RTY=F", "RUSSELL2000_FUTURES"),
    ("YM=F",  "DOW_JONES_FUTURES"),
    ("^VIX",  "VIX_SPOT"),
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
        # Prefix columns with the label to keep them unique on concat
        df = df.add_prefix(f"{label.lower()}_")
        return df
    except Exception as exc:
        print(f"  WARNING {label} ({ticker}): {exc}")
        return None


def collect() -> None:
    """Fetch all equity index futures & VIX, combine into a single wide parquet."""
    frames: list[pd.DataFrame] = []
    for ticker, label in FUTURES:
        df = _fetch(ticker, label)
        if df is not None:
            frames.append(df)
        time.sleep(_SLEEP)

    if not frames:
        print("  WARNING: no equity index futures data collected")
        return

    combined = pd.concat(frames, axis=1)
    save(combined, "derivatives", "equity_index_futures_1d.parquet")


def main() -> None:
    print("Fetching 5 equity index futures + VIX (daily OHLCV) via yfinance")
    collect()


if __name__ == "__main__":
    main()
