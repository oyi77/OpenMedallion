"""
REIT ETF and individual REIT prices via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/real_estate/REIT_<TICKER>_1d.parquet (OHLCV, daily)
Covers:
  - REIT ETFs: VNQ, IYR, SCHH, RWR, ICF, VNQI
  - Large REITs: PLD, AMT, EQIX, WELL, SPG, PSA, O, DLR, SBAC
  - Residential: AVB, EQR, UDR, CPT, ESS
  - Commercial/Specialty: BXP, VTR, ARE, AIR, KIM
  - Net Lease: WPC, NNN, STOR, VICI, MGM
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

_SLEEP = 0.3

REITS: list[str] = [
    # REIT ETFs
    "VNQ", "IYR", "SCHH", "RWR", "ICF", "VNQI",
    # Large REITs
    "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "DLR", "SBAC",
    # Residential
    "AVB", "EQR", "UDR", "CPT", "ESS",
    # Commercial / Specialty
    "BXP", "VTR", "ARE", "AIR", "KIM",
    # Net Lease
    "WPC", "NNN", "STOR", "VICI", "MGM",
]


def _fetch(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="10y", auto_adjust=False, progress=False)
        if df.empty:
            print(f"  WARNING {ticker}: empty response")
            return None
        # Flatten MultiIndex columns (yfinance returns MultiIndex for single ticker too)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["ticker"] = ticker
        return df
    except Exception as exc:
        print(f"  WARNING {ticker}: {exc}")
        return None


def collect_reits() -> None:
    ok = 0
    for ticker in REITS:
        df = _fetch(ticker)
        if df is not None and not df.empty:
            save(df, "real_estate", f"REIT_{ticker}_1d.parquet")
            ok += 1
        time.sleep(_SLEEP)
    print(f"\nDone: {ok}/{len(REITS)} REIT files saved")


def main() -> None:
    print(f"Fetching {len(REITS)} REIT tickers (10y daily OHLCV) via yfinance")
    collect_reits()


if __name__ == "__main__":
    main()
