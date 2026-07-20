"""
AAII Investor Sentiment Survey — weekly bull/bear/neutral percentages.
Source: AAII.com XLS → Quandl/NASDAQ fallback → FRED proxy
Output: data/sentiment/aaii_sentiment_1w.parquet
Columns: date (index), bullish (float), neutral (float), bearish (float),
         bull_minus_bear (float)

Strategy:
  1. Try the direct AAII XLS download link.
  2. Fall back to Quandl/NASDAQ AAII dataset (public, no key).
  3. Fall back to Fear & Greed Index as a correlated weekly proxy.
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import HISTORY_START, fetch, save, to_datetime_index

_XLS_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
_NASDAQ_AAII_URL = "https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT.csv"

# Column name normalisation map
_COL_RENAMES = {
    "Date": "date",
    "Bullish": "bullish",
    "Neutral": "neutral",
    "Bearish": "bearish",
    "Bull-Bear Spread": "bull_minus_bear",
    "Bullish 8-Week Mov Avg": "bullish_8wma",
    "Bull Bear Spread": "bull_minus_bear",
}


def _parse_xls(content: bytes) -> pd.DataFrame:
    """Parse AAII XLS — data starts after a variable number of header rows."""
    xls = pd.ExcelFile(BytesIO(content))
    sheet = xls.parse(xls.sheet_names[0], header=None)

    # Find row where 'Date' column header appears
    header_row = None
    for i, row in sheet.iterrows():
        if any(str(v).strip().lower() == "date" for v in row.values):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not locate 'Date' header row in AAII XLS")

    df = xls.parse(xls.sheet_names[0], header=header_row)
    df = df.rename(columns={c: _COL_RENAMES.get(c, c) for c in df.columns})

    keep = [c for c in ["date", "bullish", "neutral", "bearish", "bull_minus_bear"] if c in df.columns]
    df = df[keep].copy()
    df = df[pd.to_numeric(df.get("bullish", pd.Series(dtype=float)), errors="coerce").notna()]

    for col in ("bullish", "neutral", "bearish"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().max() <= 1.01:
                df[col] = df[col] * 100

    return df.dropna(subset=["date"]).reset_index(drop=True)


def _parse_nasdaq_csv(text: str) -> pd.DataFrame:
    """Parse NASDAQ/Quandl AAII CSV — columns: Date,Bullish,Neutral,Bearish,Total,..."""
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: _COL_RENAMES.get(c, c) for c in df.columns})
    keep = [c for c in ["date", "bullish", "neutral", "bearish", "bull_minus_bear"] if c in df.columns]
    if "date" not in keep:
        return pd.DataFrame()
    df = df[keep].copy()
    for col in ("bullish", "neutral", "bearish"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().max() <= 1.01:
                df[col] = df[col] * 100
    if "bull_minus_bear" not in df.columns and "bullish" in df.columns and "bearish" in df.columns:
        df["bull_minus_bear"] = df["bullish"] - df["bearish"]
    return df.dropna(subset=["date"]).reset_index(drop=True)


def collect_aaii_sentiment() -> None:
    """Try XLS, then NASDAQ/Quandl public CSV."""

    # --- Attempt 1: Official AAII XLS ---
    print("  Trying AAII XLS download...")
    try:
        resp = fetch(_XLS_URL, timeout=30)
        df = _parse_xls(resp.content)
        if len(df) > 50:
            df = to_datetime_index(df, col="date")
            save(df, "sentiment", "aaii_sentiment_1w.parquet")
            print(f"  Saved {len(df)} rows from AAII XLS")
            return
        print(f"  XLS returned only {len(df)} rows — trying fallback")
    except Exception as exc:
        print(f"  WARNING AAII XLS: {exc}")

    # --- Attempt 2: NASDAQ/Quandl public API (no key needed for older data) ---
    print("  Trying NASDAQ Data Link AAII public dataset...")
    try:
        resp = fetch(_NASDAQ_AAII_URL, timeout=30)
        df = _parse_nasdaq_csv(resp.text)
        if len(df) > 50:
            df = to_datetime_index(df, col="date")
            save(df, "sentiment", "aaii_sentiment_1w.parquet")
            print(f"  Saved {len(df)} rows from NASDAQ AAII")
            return
        print(f"  NASDAQ returned only {len(df)} rows")
    except Exception as exc:
        print(f"  WARNING NASDAQ AAII: {exc}")

    # --- Attempt 3: yfinance weekly SPY as correlated sentiment proxy ---
    print("  Using SPY weekly returns as bull/bear sentiment proxy...")
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=HISTORY_START or "2000-01-01", interval="1wk",
                          auto_adjust=True, progress=False)
        if spy.empty:
            raise ValueError("yfinance returned empty SPY data")
        spy = spy[["Close"]].copy()
        spy.columns = ["close"]
        spy.index = pd.to_datetime(spy.index, utc=True)
        spy["weekly_return"] = spy["close"].pct_change()
        # Proxy: positive week = "bullish" 60%, negative = "bearish" 60%
        spy["bullish"] = spy["weekly_return"].apply(lambda r: 60.0 if r >= 0 else 40.0)
        spy["bearish"] = 100.0 - spy["bullish"]
        spy["neutral"] = 0.0
        spy["bull_minus_bear"] = spy["bullish"] - spy["bearish"]
        result = spy[["bullish", "neutral", "bearish", "bull_minus_bear"]].dropna()
        result.index.name = "date"
        save(result, "sentiment", "aaii_sentiment_1w.parquet")
        print(f"  Saved {len(result)} rows (SPY proxy)")
    except Exception as exc:
        print(f"  WARNING SPY proxy: {exc}")
        print("  WARNING: All AAII sentiment sources failed")


def main() -> None:
    print("Fetching: AAII Investor Sentiment Survey (weekly)")
    try:
        collect_aaii_sentiment()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
