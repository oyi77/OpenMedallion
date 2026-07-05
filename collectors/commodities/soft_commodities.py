"""
Soft commodities futures price collector via Yahoo Finance (yfinance).
Source: Yahoo Finance — free, no API key required.
Covers: Cocoa, Coffee, Cotton, Sugar, Orange Juice, Lumber,
        Corn, Wheat, Soybeans, Soybean Oil futures.
Output: data/commodities/soft_commodities_1d.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

# Yahoo Finance continuous futures tickers -> column name
TICKERS: dict[str, str] = {
    "CC=F": "cocoa",
    "KC=F": "coffee",
    "CT=F": "cotton",
    "SB=F": "sugar",
    "OJ=F": "orange_juice",
    "LB=F": "lumber",
    "ZC=F": "corn",
    "ZW=F": "wheat",
    "ZS=F": "soybeans",
    "ZL=F": "soybean_oil",
}

START_DATE = "2000-01-01"


def collect_soft_commodities() -> None:
    """Download daily close prices for all soft commodity futures."""
    try:
        import yfinance as yf
    except ImportError as exc:
        print(f"  ERROR: yfinance not installed — {exc}")
        return

    print(f"  Downloading {len(TICKERS)} tickers from {START_DATE} ...")
    symbols = list(TICKERS.keys())

    try:
        raw = yf.download(
            symbols,
            start=START_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        print(f"  ERROR: yfinance.download failed — {exc}")
        return

    # yfinance returns a MultiIndex DataFrame; extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "close" in raw.columns.get_level_values(0):
            close = raw["close"]
        else:
            print(f"  ERROR: unexpected columns: {raw.columns[:8]}")
            return
    else:
        # Single ticker fallback
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})

    # Rename ticker columns to friendly names
    close = close.rename(columns=TICKERS)
    close.index = pd.to_datetime(close.index, utc=True)
    close.index.name = "date"
    close = close.sort_index().dropna(how="all")

    available = [c for c in close.columns if not close[c].isna().all()]
    print(f"  Available columns: {available}")
    print(f"  Total rows: {len(close)}")

    if close.empty:
        print("  WARNING: no data returned — skipping save")
        return

    save(close, "commodities", "soft_commodities_1d.parquet")


def main() -> None:
    print("Fetching soft commodities futures via Yahoo Finance ...")
    collect_soft_commodities()


if __name__ == "__main__":
    main()
