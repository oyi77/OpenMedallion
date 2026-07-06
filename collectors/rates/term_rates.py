"""
Short-term rate benchmarks and term structure data.
Sources: FRED CSV (no API key) + yfinance fallback
Output: data/rates/FRED_<label>_<freq>.parquet

Covers:
  - SOFR (overnight + 30/90/180-day averages + index)
  - IORB, DFF, OBFR (Fed policy / funding rates)
  - ECB €STR and EURIBOR proxies
  - US T-Bill term structure (3M, 6M, 1Y, 2Y, 5Y, 10Y)
"""
from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
_SLEEP = 0.4

# ---------------------------------------------------------------------------
# FRED series: (series_id, label, frequency)
# ---------------------------------------------------------------------------
_FRED_DAILY: list[tuple[str, str]] = [
    # SOFR family
    ("SOFR",          "SOFR_overnight"),
    ("SOFR30DAYAVG",  "SOFR_30d_avg"),
    ("SOFR90DAYAVG",  "SOFR_90d_avg"),
    ("SOFR180DAYAVG", "SOFR_180d_avg"),
    ("SOFRINDEX",     "SOFR_index"),
    # Fed / US funding rates
    ("IORB",  "IORB"),
    ("DFF",   "Fed_Funds_Effective"),
    ("OBFR",  "Overnight_Bank_Funding"),
    # ECB €STR (daily)
    ("ECBESTRVOLWGTTRR", "ESTR"),
    # US T-Bill term structure
    ("DTB3",   "US_TBill_3M"),
    ("DTB6",   "US_TBill_6M"),
    ("DTB1YR", "US_TBill_1Y"),
    ("DGS2",   "US_Treasury_2Y"),
    ("DGS5",   "US_Treasury_5Y"),
    ("DGS10",  "US_Treasury_10Y"),
]

_FRED_MONTHLY: list[tuple[str, str]] = [
    # EURIBOR proxies via FRED (monthly)
    ("IR3TIB01EZM156N", "EURIBOR_3M"),
    ("IR3TBE01EZM156N", "EURIBOR_BE_3M_proxy"),
]

# ---------------------------------------------------------------------------
# yfinance fallback tickers: (ticker, label, frequency)
# Used for SONIA proxy (no FRED series) and cross-check
# ---------------------------------------------------------------------------
_YF_TICKERS: list[tuple[str, str]] = [
    # Eurodollar futures as SOFR/short-rate proxy
    ("IR=F", "Eurodollar_Front"),
    # 13-week T-Bill ETF as 3M rate proxy
    ("BIL",  "BIL_TBill_ETF"),
    # Short-term rate ETF
    ("SHV",  "SHV_ShortTreasury_ETF"),
]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_fred_csv(series_id: str, label: str) -> pd.DataFrame | None:
    """Download a FRED series via the free CSV endpoint (no API key needed)."""
    url = f"{_FRED_BASE}{series_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        date_col = df.columns[0]
        val_col = df.columns[1]
        df = df.rename(columns={date_col: "date", val_col: "value"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        # FRED uses "." for missing observations
        df = df[df["value"] != "."].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.rename(columns={"value": label.lower()})
        df = to_datetime_index(df, col="date")
        return df
    except Exception as exc:
        print(f"  FAIL FRED {series_id} ({label}): {exc}")
        return None


def _fetch_yf(ticker: str, label: str) -> pd.DataFrame | None:
    """Download closing price history from Yahoo Finance."""
    try:
        raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  FAIL yfinance {ticker}: empty response")
            return None
        # yfinance may return MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"][ticker]
        else:
            close = raw["Close"]
        df = close.to_frame(name=label.lower())
        df = to_datetime_index(df)
        return df
    except Exception as exc:
        print(f"  FAIL yfinance {ticker} ({label}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Collection routines
# ---------------------------------------------------------------------------

def collect_fred_daily() -> tuple[int, int]:
    """Collect all daily FRED rate series."""
    ok = fail = 0
    print(f"\n[FRED daily — {len(_FRED_DAILY)} series]")
    for series_id, label in _FRED_DAILY:
        df = _fetch_fred_csv(series_id, label)
        if df is not None and not df.empty:
            save(df, "rates", f"FRED_{label}_1d.parquet")
            ok += 1
        else:
            fail += 1
        time.sleep(_SLEEP)
    return ok, fail


def collect_fred_monthly() -> tuple[int, int]:
    """Collect monthly FRED rate series (EURIBOR proxies)."""
    ok = fail = 0
    print(f"\n[FRED monthly — {len(_FRED_MONTHLY)} series]")
    for series_id, label in _FRED_MONTHLY:
        df = _fetch_fred_csv(series_id, label)
        if df is not None and not df.empty:
            save(df, "rates", f"FRED_{label}_1mo.parquet")
            ok += 1
        else:
            fail += 1
        time.sleep(_SLEEP)
    return ok, fail


def collect_yfinance() -> tuple[int, int]:
    """Collect yfinance short-rate proxies (SONIA proxy, T-Bill ETFs)."""
    ok = fail = 0
    print(f"\n[yfinance — {len(_YF_TICKERS)} tickers]")
    for ticker, label in _YF_TICKERS:
        df = _fetch_yf(ticker, label)
        if df is not None and not df.empty:
            save(df, "rates", f"YF_{label}_1d.parquet")
            ok += 1
        else:
            fail += 1
        time.sleep(_SLEEP)
    return ok, fail


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching: Short-term rate benchmarks and term structure")

    total_ok = total_fail = 0

    ok, fail = collect_fred_daily()
    total_ok += ok
    total_fail += fail

    ok, fail = collect_fred_monthly()
    total_ok += ok
    total_fail += fail

    ok, fail = collect_yfinance()
    total_ok += ok
    total_fail += fail

    total = total_ok + total_fail
    print(f"\nDone — {total_ok}/{total} ok, {total_fail}/{total} failed")


if __name__ == "__main__":
    main()
