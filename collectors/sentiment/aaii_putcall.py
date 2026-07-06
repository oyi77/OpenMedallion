"""
Sentiment indicators collector.
Sources:
  - CBOE VIX family via FRED CSV (VIXCLS, OVXCLS, GVZCLS, EVZCLS) — daily
  - AAII-proxy via yfinance (VIX, SKEW, equity proxies) — weekly resampled
  - CNN Fear & Greed proxies via yfinance (VIX, SKEW, HYGH, JNK, GLD, TLT, SHY) — daily
  - Market breadth proxy via yfinance (advance/decline ETF proxies) — daily
Output:
  data/sentiment/CBOE_<label>_1d.parquet
  data/sentiment/AAII_proxy_<label>_1w.parquet
  data/sentiment/FearGreed_proxy_<label>_1d.parquet
  data/sentiment/market_breadth_1d.parquet
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

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SLEEP = 0.3

# ---------------------------------------------------------------------------
# CBOE VIX family via FRED CSV (daily)
# ---------------------------------------------------------------------------
_CBOE_FRED: dict[str, str] = {
    "VIXCLS": "VIX",
    "OVXCLS": "OVX",
    "GVZCLS": "GVZ",
    "EVZCLS": "EVZ",
}


def fetch_cboe_fred() -> None:
    """CBOE volatility indices via FRED CSV — daily."""
    for sid, label in _CBOE_FRED.items():
        try:
            resp = fetch(FRED_CSV, params={"id": sid}, timeout=20)
            df = pd.read_csv(StringIO(resp.text), na_values=".")
            df.columns = ["date", "value"]
            df = to_datetime_index(df)
            df = df[["value"]].dropna()
            df.columns = [label]
            save(df, "sentiment", f"CBOE_{label}_1d.parquet")
        except Exception as exc:
            print(f"  WARN CBOE {label} ({sid}) — {exc}")
        time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# AAII proxy via yfinance (weekly)
# AAII survey data is not on FRED; use VIX + equity index weekly resampled
# as a fear/greed proxy for weekly sentiment cadence.
# ---------------------------------------------------------------------------
_AAII_PROXY: dict[str, str] = {
    "VIX_weekly":    "^VIX",
    "SPY_weekly":    "SPY",
    "QQQ_weekly":    "QQQ",
    "SKEW_weekly":   "^SKEW",
}


def fetch_aaii_proxy() -> None:
    """AAII-cadence weekly sentiment proxies via yfinance."""
    for label, ticker in _AAII_PROXY.items():
        try:
            raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  SKIP AAII proxy {label} — no data")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = "close"
            # Resample to weekly (Friday close)
            weekly = close.resample("W-FRI").last().dropna()
            if isinstance(weekly.index, pd.DatetimeIndex):
                weekly.index = weekly.index.tz_localize("UTC") if weekly.index.tz is None else weekly.index.tz_convert("UTC")
            df = weekly.to_frame()
            save(df, "sentiment", f"AAII_proxy_{label}_1w.parquet")
        except Exception as exc:
            print(f"  WARN AAII proxy {label} — {exc}")
        time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# CNN Fear & Greed component proxies via yfinance (daily)
# Components: momentum (VIX), safe haven (GLD/TLT/SHY),
#             junk bonds (HYGH, JNK), put/call proxy (SKEW).
# ---------------------------------------------------------------------------
_FEAR_GREED: dict[str, str] = {
    "VIX":  "^VIX",
    "SKEW": "^SKEW",
    "HYGH": "HYGH",
    "JNK":  "JNK",
    "GLD":  "GLD",
    "TLT":  "TLT",
    "SHY":  "SHY",
}


def fetch_fear_greed_proxy() -> None:
    """CNN Fear & Greed component proxies via yfinance — daily close."""
    for label, ticker in _FEAR_GREED.items():
        try:
            raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  SKIP FearGreed proxy {label} — no data")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = "close"
            close = close.dropna()
            if isinstance(close.index, pd.DatetimeIndex):
                close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index.tz_convert("UTC")
            df = close.to_frame()
            save(df, "sentiment", f"FearGreed_proxy_{label}_1d.parquet")
        except Exception as exc:
            print(f"  WARN FearGreed proxy {label} — {exc}")
        time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# Market breadth via yfinance
# ^SPXADP is not available on Yahoo Finance; use RSP (equal-weight S&P 500)
# vs SPY spread as a breadth proxy, plus MMTW (% above 200-day MA ETF).
# ---------------------------------------------------------------------------
_BREADTH_TICKERS: dict[str, str] = {
    "RSP":   "RSP",    # S&P 500 equal-weight — breadth proxy
    "SPY":   "SPY",    # S&P 500 cap-weight
    "IWM":   "IWM",    # Russell 2000 — small-cap breadth
    "VTI":   "VTI",    # Total market
}


def fetch_market_breadth() -> None:
    """Market breadth proxies — daily close prices and RSP/SPY spread."""
    frames: list[pd.Series] = []
    for label, ticker in _BREADTH_TICKERS.items():
        try:
            raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  SKIP breadth {label} — no data")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = label
            if isinstance(close.index, pd.DatetimeIndex):
                close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index.tz_convert("UTC")
            frames.append(close.dropna())
        except Exception as exc:
            print(f"  WARN breadth {label} — {exc}")
        time.sleep(SLEEP)

    if not frames:
        print("  SKIP market_breadth — all tickers failed")
        return

    df = pd.concat(frames, axis=1).sort_index()

    # Derived breadth spread: RSP vs SPY normalised ratio (higher = broader rally)
    if "RSP" in df.columns and "SPY" in df.columns:
        df["RSP_SPY_ratio"] = df["RSP"] / df["SPY"]

    save(df, "sentiment", "market_breadth_1d.parquet")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== CBOE VIX family (FRED CSV) ===")
    fetch_cboe_fred()

    print("\n=== AAII-cadence weekly proxies (yfinance) ===")
    fetch_aaii_proxy()

    print("\n=== CNN Fear & Greed component proxies (yfinance) ===")
    fetch_fear_greed_proxy()

    print("\n=== Market breadth proxies (yfinance) ===")
    fetch_market_breadth()


if __name__ == "__main__":
    main()
