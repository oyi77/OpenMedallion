"""
Volatility surface / term structure collector.

Sources:
  - FRED CSV (no key): VIX, VXV, VXD, VXN, OVX, GVZ, EVZ — daily closing levels
  - yfinance: ^VIX, ^VIX3M, ^VVIX, ^SKEW, UVXY, SVXY, VXX, VIXY — adjusted close
  - Contango proxy: VXX/VXZ close ratio (positive = contango, negative = backwardation)

Output files (data/volatility/):
  FRED_<label>_1d.parquet   — one parquet per FRED series
  YF_<label>_1d.parquet     — one parquet per yfinance ticker
  YF_CONTANGO_PROXY_1d.parquet — VXX/VXZ ratio
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# (fred_series_id, output_label)
FRED_SERIES: list[tuple[str, str]] = [
    ("VIXCLS",  "VIX"),
    ("VXVCLS",  "VXV"),
    ("VXDCLS",  "VXD"),
    ("VXNCLS",  "VXN"),
    ("OVXCLS",  "OVX"),
    ("GVZCLS",  "GVZ"),
    ("EVZCLS",  "EVZ"),
]

# (yfinance_ticker, output_label)
YF_TICKERS: list[tuple[str, str]] = [
    ("^VIX",   "VIX"),
    ("^VIX3M", "VIX3M"),
    ("^VVIX",  "VVIX"),
    ("^SKEW",  "SKEW"),
    ("UVXY",   "UVXY"),
    ("SVXY",   "SVXY"),
    ("VXX",    "VXX"),
    ("VIXY",   "VIXY"),
]

# Tickers needed for the contango/backwardation proxy ratio (VXX = 1M, VXZ = 5M)
CONTANGO_TICKERS = ("VXX", "VXZ")

SLEEP_BETWEEN = 0.3  # seconds — polite pacing


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Download a FRED series CSV and return a single-column DataFrame.

    Returns a DataFrame with DatetimeIndex and a column named after series_id.
    """
    resp = fetch(FRED_BASE, params={"id": series_id})
    # FRED CSV format: DATE,<SERIES_ID>
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), na_values=".")
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.dropna()


def collect_fred_series() -> None:
    """Fetch each FRED vol series and save to its own parquet."""
    for series_id, label in FRED_SERIES:
        print(f"  FRED {series_id} ({label})...")
        try:
            df = _fetch_fred_series(series_id)
            df = to_datetime_index(df)
            save(df, "volatility", f"FRED_{label}_1d.parquet")
        except Exception as exc:
            print(f"  WARNING: FRED {series_id} failed — {exc}")
        time.sleep(SLEEP_BETWEEN)


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _fetch_yf_ticker(ticker: str) -> pd.DataFrame:
    """Download max history for a yfinance ticker; return single-column DataFrame."""
    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")
    # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].rename(ticker)
    df = close.to_frame()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index().dropna()
    return df


def collect_yf_tickers() -> None:
    """Fetch each yfinance ticker and save to its own parquet."""
    for ticker, label in YF_TICKERS:
        print(f"  yfinance {ticker} ({label})...")
        try:
            df = _fetch_yf_ticker(ticker)
            df = to_datetime_index(df)
            save(df, "volatility", f"YF_{label}_1d.parquet")
        except Exception as exc:
            print(f"  WARNING: yfinance {ticker} failed — {exc}")
        time.sleep(SLEEP_BETWEEN)


# ---------------------------------------------------------------------------
# Contango / backwardation proxy
# ---------------------------------------------------------------------------

def collect_contango_proxy() -> None:
    """Compute VXX/VXZ ratio as a contango (>1) / backwardation (<1) proxy.

    VXX tracks ~1-month VIX futures; VXZ tracks ~5-month futures.
    Ratio > 1 → term structure in contango (front month elevated).
    Ratio < 1 → backwardation (spot vol elevated vs. deferred).
    """
    print("  Contango proxy (VXX / VXZ)...")
    try:
        vxx = _fetch_yf_ticker("VXX").rename(columns={"VXX": "vxx"})
        time.sleep(SLEEP_BETWEEN)
        vxz = _fetch_yf_ticker("VXZ").rename(columns={"VXZ": "vxz"})

        merged = vxx.join(vxz, how="inner")
        merged["vxx_vxz_ratio"] = merged["vxx"] / merged["vxz"]
        merged = merged.dropna(subset=["vxx_vxz_ratio"])

        merged = to_datetime_index(merged)
        save(merged, "volatility", "YF_CONTANGO_PROXY_1d.parquet")
    except Exception as exc:
        print(f"  WARNING: contango proxy failed — {exc}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Volatility surface / term structure collector ===")

    print("\n[1/3] FRED volatility indices...")
    collect_fred_series()

    print("\n[2/3] yfinance vol tickers...")
    collect_yf_tickers()

    print("\n[3/3] Contango / backwardation proxy...")
    collect_contango_proxy()

    print("\nDone.")


if __name__ == "__main__":
    main()
