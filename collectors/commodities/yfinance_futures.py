"""
Broad commodity futures via Yahoo Finance (yfinance).
Source: Yahoo Finance — free, no API key required.
Output: data/commodities/<LABEL>_1d.parquet  (OHLCV daily)
Covers: metals, energy, agricultural, livestock futures.
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
_START = HISTORY_START or "1990-01-01"
_SLEEP = 0.3

# (yfinance_ticker, label)
FUTURES: list[tuple[str, str]] = [
    # --- Precious Metals ---
    ("GC=F",  "COM_GOLD_FUTURES"),
    ("SI=F",  "COM_SILVER_FUTURES"),
    ("PL=F",  "COM_PLATINUM_FUTURES"),
    ("PA=F",  "COM_PALLADIUM_FUTURES"),
    # --- Base Metals ---
    ("HG=F",  "COM_COPPER_FUTURES"),
    ("ALI=F", "COM_ALUMINUM_FUTURES"),
    ("ZNC=F", "COM_ZINC_FUTURES"),
    # --- Energy ---
    ("CL=F",  "COM_CRUDE_WTI_FUTURES"),
    ("BZ=F",  "COM_CRUDE_BRENT_FUTURES"),
    ("RB=F",  "COM_RBOB_GAS_FUTURES"),
    ("HO=F",  "COM_HEATING_OIL_FUTURES"),
    ("NG=F",  "COM_NATGAS_FUTURES"),
    ("QM=F",  "COM_MINI_CRUDE_FUTURES"),
    # --- Agricultural / Grains ---
    ("ZC=F",  "COM_CORN_FUTURES"),
    ("ZW=F",  "COM_WHEAT_FUTURES"),
    ("ZS=F",  "COM_SOYBEAN_FUTURES"),
    ("ZL=F",  "COM_SOYOIL_FUTURES"),
    ("ZM=F",  "COM_SOYMEAL_FUTURES"),
    ("KE=F",  "COM_KC_WHEAT_FUTURES"),
    ("MWE=F", "COM_MINN_WHEAT_FUTURES"),
    ("ZR=F",  "COM_ROUGH_RICE_FUTURES"),
    ("ZO=F",  "COM_OATS_FUTURES"),
    # --- Soft Commodities ---
    ("CC=F",  "COM_COCOA_FUTURES"),
    ("KC=F",  "COM_COFFEE_FUTURES"),
    ("CT=F",  "COM_COTTON_FUTURES"),
    ("SB=F",  "COM_SUGAR_FUTURES"),
    ("OJ=F",  "COM_ORANGEJUICE_FUTURES"),
    ("LBS=F", "COM_LUMBER_FUTURES"),
    # --- Livestock ---
    ("LE=F",  "COM_LIVE_CATTLE_FUTURES"),
    ("GF=F",  "COM_FEEDER_CATTLE_FUTURES"),
    ("HE=F",  "COM_LEAN_HOGS_FUTURES"),
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
            save(df, "commodities", f"{label}_1d.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    print(f"Fetching {len(FUTURES)} commodity futures (daily OHLCV) via yfinance")
    collect_futures()


if __name__ == "__main__":
    main()
