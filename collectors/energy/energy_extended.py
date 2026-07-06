"""
Extended energy data collector.
Sources:
  - yfinance: EU ETS carbon proxies, LNG futures, renewable ETFs
  - FRED CSV: Henry Hub monthly spot, WTI crude daily, Brent crude daily
Output: data/energy/<NAME>_1d.parquet
"""
from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_SLEEP = 0.3


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def _fetch_fred(series_id: str) -> pd.DataFrame:
    """Download a FRED series; returns a DatetimeIndex DataFrame with column 'value'."""
    resp = fetch(FRED_BASE, params={"id": series_id}, timeout=30)
    df = pd.read_csv(StringIO(resp.text), na_values=".")
    df.columns = ["date", "value"]
    df = to_datetime_index(df, col="date")
    return df.dropna()


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _fetch_yf(ticker: str) -> pd.DataFrame | None:
    """Download full price history for ticker; returns Close as single-column DataFrame."""
    try:
        raw = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if raw.empty:
            print(f"  WARN {ticker} — empty response from yfinance")
            return None
        close = raw[["Close"]].copy()
        close.columns = ["close"]
        close.index = pd.to_datetime(close.index, utc=True)
        close = close.sort_index().dropna()
        return close
    except Exception as exc:
        print(f"  WARN {ticker} — {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. EU ETS carbon allowance price proxies
# ---------------------------------------------------------------------------

EU_ETS_TICKERS: list[tuple[str, str]] = [
    ("CO2.F",  "EU_ETS_carbon_price"),  # EUA futures on ICE (via Yahoo)
    ("CARP.L", "EU_ETS_carbon_CARP"),   # WisdomTree Carbon ETP
]


def collect_eu_ets() -> None:
    print("--- EU ETS carbon proxies (yfinance) ---")
    for ticker, label in EU_ETS_TICKERS:
        df = _fetch_yf(ticker)
        if df is not None:
            save(df, "energy", f"{label}_1d.parquet")
        time.sleep(_SLEEP)


# ---------------------------------------------------------------------------
# 2. LNG / Natural Gas
# ---------------------------------------------------------------------------

LNG_YF_TICKERS: list[tuple[str, str]] = [
    ("NG=F",  "LNG_spot_henry_hub_futures"),  # Henry Hub front-month futures
    ("TTF=F", "LNG_spot_TTF_futures"),        # TTF European gas futures
]

LNG_FRED_SERIES: list[tuple[str, str]] = [
    ("MHHNGSP", "LNG_spot_henry_hub_monthly"),  # Henry Hub monthly spot (FRED)
]


def collect_lng() -> None:
    print("--- LNG / Natural Gas (yfinance) ---")
    for ticker, label in LNG_YF_TICKERS:
        df = _fetch_yf(ticker)
        if df is not None:
            save(df, "energy", f"{label}_1d.parquet")
        time.sleep(_SLEEP)

    print("--- LNG / Natural Gas (FRED) ---")
    for series_id, label in LNG_FRED_SERIES:
        try:
            df = _fetch_fred(series_id)
            save(df, "energy", f"{label}_1d.parquet")
        except Exception as exc:
            print(f"  WARN FRED {series_id} — {exc}")
        time.sleep(_SLEEP)


# ---------------------------------------------------------------------------
# 3. Renewable energy capacity proxies (ETFs)
# ---------------------------------------------------------------------------

RENEWABLE_ETF_TICKERS: list[tuple[str, str]] = [
    ("ICLN",   "ICLN"),   # iShares Global Clean Energy ETF
    ("QCLN",   "QCLN"),   # First Trust NASDAQ Clean Edge ETF
    ("INRG.L", "INRG_L"), # iShares Global Clean Energy UCITS
]


def collect_renewable_etfs() -> None:
    print("--- Renewable energy ETFs (yfinance) ---")
    for ticker, label in RENEWABLE_ETF_TICKERS:
        df = _fetch_yf(ticker)
        if df is not None:
            save(df, "energy", f"renewable_etf_{label}_1d.parquet")
        time.sleep(_SLEEP)


# ---------------------------------------------------------------------------
# 4. Oil & energy via FRED
# ---------------------------------------------------------------------------

FRED_OIL_SERIES: list[tuple[str, str]] = [
    ("DCOILWTICO",   "FRED_WTI_crude_daily"),   # WTI crude daily
    ("DCOILBRENTEU", "FRED_Brent_crude_daily"), # Brent crude daily
]


def collect_fred_oil() -> None:
    print("--- Oil & energy (FRED) ---")
    for series_id, label in FRED_OIL_SERIES:
        try:
            df = _fetch_fred(series_id)
            save(df, "energy", f"{label}_1d.parquet")
        except Exception as exc:
            print(f"  WARN FRED {series_id} — {exc}")
        time.sleep(_SLEEP)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Extended energy data collector ===")
    collect_eu_ets()
    collect_lng()
    collect_renewable_etfs()
    collect_fred_oil()
    print("=== Done ===")


if __name__ == "__main__":
    main()
