"""
Forex major & cross hourly via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/forex/yahoo/<LABEL>_1h.parquet
Covers: 27 major and cross forex pairs, hourly.
Note: yfinance hourly limited to ~730 days of history.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, HISTORY_START

_START = HISTORY_START or "2023-01-01"  # hourly limited ~730d
_SLEEP = 0.35

# (yfinance_ticker, label_prefix, description)
PAIRS: list[tuple[str, str, str]] = [
    # Majors
    ("EURUSD=X", "EURUSD", "EUR/USD"),
    ("USDJPY=X", "USDJPY", "USD/JPY"),
    ("GBPUSD=X", "GBPUSD", "GBP/USD"),
    ("USDCHF=X", "USDCHF", "USD/CHF"),
    ("AUDUSD=X", "AUDUSD", "AUD/USD"),
    ("USDCAD=X", "USDCAD", "USD/CAD"),
    ("NZDUSD=X", "NZDUSD", "NZD/USD"),
    # Crosses
    ("EURJPY=X", "EURJPY", "EUR/JPY"),
    ("EURGBP=X", "EURGBP", "EUR/GBP"),
    ("EURAUD=X", "EURAUD", "EUR/AUD"),
    ("EURCHF=X", "EURCHF", "EUR/CHF"),
    ("EURCAD=X", "EURCAD", "EUR/CAD"),
    ("EURNZD=X", "EURNZD", "EUR/NZD"),
    ("GBPJPY=X", "GBPJPY", "GBP/JPY"),
    ("GBPCHF=X", "GBPCHF", "GBP/CHF"),
    ("GBPAUD=X", "GBPAUD", "GBP/AUD"),
    ("GBPCAD=X", "GBPCAD", "GBP/CAD"),
    ("GBPNZD=X", "GBPNZD", "GBP/NZD"),
    ("AUDJPY=X", "AUDJPY", "AUD/JPY"),
    ("AUDCHF=X", "AUDCHF", "AUD/CHF"),
    ("AUDCAD=X", "AUDCAD", "AUD/CAD"),
    ("AUDNZD=X", "AUDNZD", "AUD/NZD"),
    ("CADJPY=X", "CADJPY", "CAD/JPY"),
    ("CHFJPY=X", "CHFJPY", "CHF/JPY"),
    ("NZDJPY=X", "NZDJPY", "NZD/JPY"),
    ("NZDCAD=X", "NZDCAD", "NZD/CAD"),
    ("NZDCHF=X", "NZDCHF", "NZD/CHF"),
]


def _fetch_hourly(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, interval="1h", period="730d", progress=False, auto_adjust=True)
        if df is None or df.empty:
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


def collect_forex_hourly() -> None:
    for ticker, label, desc in PAIRS:
        print(f"  {desc} ({label}) 1h ...")
        df = _fetch_hourly(ticker, label)
        if df is not None:
            save(df, "forex/yahoo", f"{label}_1h.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    print(f"Fetching: {len(PAIRS)} forex pairs hourly OHLCV via yfinance")
    collect_forex_hourly()


if __name__ == "__main__":
    main()
