"""
Major global stock indices via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/indices/yf/<SYMBOL>_1d.parquet  (OHLCV, daily)
        data/indices/yf/<SYMBOL>_1h.parquet  (OHLCV, hourly, last 730d)
Covers: US, Europe, Asia-Pacific, EM, sector indices
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

# (symbol, label) — yfinance tickers
INDICES: list[tuple[str, str]] = [
    # US
    ("^GSPC",  "SP500"),
    ("^DJI",   "Dow30"),
    ("^IXIC",  "Nasdaq"),
    ("^RUT",   "Russell2000"),
    ("^VIX",   "VIX"),
    # Europe
    ("^FTSE",  "FTSE100"),
    ("^GDAXI", "DAX"),
    ("^FCHI",  "CAC40"),
    ("^STOXX50E", "EuroStoxx50"),
    ("^AEX",   "AEX"),
    ("^IBEX",  "IBEX35"),
    ("^FTMIB", "FTSE_MIB"),
    # Asia
    ("^N225",  "Nikkei225"),
    ("^HSI",   "HangSeng"),
    ("000001.SS", "SSE_Composite"),
    ("^KS11",  "KOSPI"),
    ("^TWII",  "TAIEX"),
    ("^STI",   "STI_Singapore"),
    ("^BSESN", "BSE_Sensex"),
    ("^NSEI",  "Nifty50"),
    # Australia / NZ
    ("^AXJO",  "ASX200"),
    # EM / Latin America
    ("^BVSP",  "BOVESPA"),
    ("^MXX",   "IPC_Mexico"),
    ("^IPSA",  "IPSA_Chile"),
    # Indonesia
    ("^JKSE",  "IDX_Composite"),
    # Sector (S&P)
    ("^SP500-20", "SP500_Comm"),
    ("^SP500-25", "SP500_ConsDisc"),
    ("^SP500-30", "SP500_ConsSt"),
    ("^SP500-35", "SP500_Health"),
    ("^SP500-40", "SP500_Financials"),
    ("^SP500-45", "SP500_IT"),
    ("^SP500-50", "SP500_Energy"),
    ("^SP500-55", "SP500_Utilities"),
    ("^SP500-60", "SP500_RealEstate"),
]

_START_DAILY = "2000-01-01"
_SLEEP = 0.3  # be polite to Yahoo


def _fetch_daily(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=_START_DAILY, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # flatten MultiIndex columns if present (yf 1.x behaviour)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as exc:
        print(f"  WARNING {label}: {exc}")
        return None


def _fetch_hourly(ticker: str, label: str) -> pd.DataFrame | None:
    """yfinance hourly is limited to ~730 days."""
    try:
        df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as exc:
        print(f"  WARNING {label} 1h: {exc}")
        return None


def collect_indices() -> None:
    for ticker, label in INDICES:
        # daily
        df = _fetch_daily(ticker, label)
        if df is not None and not df.empty:
            save(df, "indices/yf", f"{label}_1d.parquet")
        time.sleep(_SLEEP)

        # hourly
        df_h = _fetch_hourly(ticker, label)
        if df_h is not None and not df_h.empty:
            save(df_h, "indices/yf", f"{label}_1h.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    print(f"Fetching: {len(INDICES)} global indices (daily + hourly) via yfinance")
    collect_indices()


if __name__ == "__main__":
    main()
