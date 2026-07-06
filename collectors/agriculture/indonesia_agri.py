"""
Indonesian agricultural commodity data collector.

Sources:
  1. World Bank Pink Sheet via FRED CSV (monthly) — palm oil, coffee, cocoa,
     rubber, rice; no API key required.
  2. ETF proxies (daily) via yfinance — agribusiness and soft-commodity ETFs
     that capture Indonesian exposure.
  3. Palm oil ETC proxies via yfinance — PALM.L and 3PAL.L.

Output layout:
  data/agriculture/indonesia/WB_<LABEL>_1m.parquet   (one file per FRED series)
  data/agriculture/indonesia/ETF_<TICKER>_1d.parquet  (one file per ETF)
  data/agriculture/indonesia/palm_oil_etf_1d.parquet  (best available palm oil proxy)
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

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
FRED_SLEEP = 0.4  # seconds between requests — polite crawl rate

# World Bank Pink Sheet series available on FRED (no API key required)
# series_id -> (output_label, column_name)
WB_SERIES: dict[str, tuple[str, str]] = {
    "PPOILINDEXM": ("palm_oil_index",  "palm_oil_price_index"),
    "PCOFFOTM":    ("coffee",          "coffee_usd_per_kg"),
    "PCOCOA":      ("cocoa",           "cocoa_usd_per_kg"),
    "PRUBB":       ("rubber",          "rubber_usd_per_kg"),
    "PRICENPETM":  ("rice",            "rice_usd_per_mt"),
}

# ETF proxies — agribusiness & soft commodity exposure
ETF_TICKERS: list[str] = [
    "MOO",   # VanEck Agribusiness ETF
    "WEAT",  # Teucrium Wheat ETF
    "SOYB",  # Teucrium Soybean ETF
    "CORN",  # Teucrium Corn ETF
    "BAL",   # iPath Series B Bloomberg Cotton Subindex Total Return ETN
    "NIB",   # iPath Series B Bloomberg Cocoa Subindex Total Return ETN
    "JO",    # iPath Series B Bloomberg Coffee Subindex Total Return ETN
]

# Palm oil ETF/ETC candidates — try in order, use first that returns data
PALM_OIL_CANDIDATES: list[str] = [
    "PALM.L",  # Invesco Bloomberg Commodity ex-Energy UCITS ETF
    "3PAL.L",  # WisdomTree Palm Oil ETC
]


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str, col: str) -> pd.DataFrame | None:
    """Fetch a single FRED CSV series; return a single-column DataFrame or None."""
    url = FRED_CSV.format(id=series_id)
    try:
        resp = fetch(url, timeout=30)
    except Exception as exc:
        print(f"    ERROR fetching {series_id}: {exc}")
        return None

    df = pd.read_csv(StringIO(resp.text))
    if df.shape[1] < 2:
        print(f"    WARNING: unexpected CSV shape for {series_id}")
        return None

    df.columns = ["date", col]
    # FRED uses "." as missing value placeholder
    df = df[df[col] != "."].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])

    if df.empty:
        print(f"    WARNING: all values missing for {series_id}")
        return None

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    return df


def collect_wb_series() -> None:
    """Fetch each World Bank commodity series from FRED; save individually."""
    for series_id, (label, col) in WB_SERIES.items():
        print(f"  Fetching FRED {series_id} ({label}) ...")
        df = _fetch_fred_series(series_id, col)
        if df is not None and not df.empty:
            filename = f"WB_{label}_1m.parquet"
            save(df, "agriculture/indonesia", filename)
            print(f"    {len(df)} observations")
        time.sleep(FRED_SLEEP)


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _fetch_yf_ticker(ticker: str) -> pd.DataFrame | None:
    """Download max-period daily close for a single yfinance ticker."""
    try:
        raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            # multi-ticker download returns a DataFrame; flatten
            close = close.iloc[:, 0]
        df = close.to_frame(name="close")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index().dropna()
        if df.empty:
            return None
        return df
    except Exception as exc:
        print(f"    ERROR fetching {ticker}: {exc}")
        return None


def collect_etf_proxies() -> None:
    """Fetch daily close for each agri ETF; save individually."""
    for ticker in ETF_TICKERS:
        print(f"  Fetching ETF {ticker} ...")
        df = _fetch_yf_ticker(ticker)
        if df is not None and not df.empty:
            filename = f"ETF_{ticker}_1d.parquet"
            save(df, "agriculture/indonesia", filename)
            print(f"    {len(df)} trading days")
        else:
            print(f"    WARNING: no data for {ticker}")


def collect_palm_oil_etf() -> None:
    """Try palm oil ETC/ETF candidates in order; save first successful result."""
    for ticker in PALM_OIL_CANDIDATES:
        print(f"  Trying palm oil proxy {ticker} ...")
        df = _fetch_yf_ticker(ticker)
        if df is not None and not df.empty:
            save(df, "agriculture/indonesia", "palm_oil_etf_1d.parquet")
            print(f"    Saved {ticker} with {len(df)} trading days")
            return
        print(f"    No data for {ticker}, trying next ...")

    print("  WARNING: no palm oil proxy returned data — skipping palm_oil_etf_1d.parquet")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Indonesia Agricultural Commodity Collector ===")

    print("\n[1/3] World Bank commodity prices via FRED (monthly) ...")
    collect_wb_series()

    print("\n[2/3] ETF proxies (daily) ...")
    collect_etf_proxies()

    print("\n[3/3] Palm oil ETF/ETC proxy (daily) ...")
    collect_palm_oil_etf()

    print("\nDone.")


if __name__ == "__main__":
    main()
