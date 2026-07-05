"""
AAII Investor Sentiment Survey — weekly bull/bear/neutral percentages.
Source: AAII.com (public CSV/XLS, no key required)
Output: data/sentiment/aaii_sentiment_1w.parquet
Columns: date (index), bullish (float), neutral (float), bearish (float),
         bull_minus_bear (float)

Strategy:
  1. Try the direct XLS download link.
  2. Fall back to scraping the JS data endpoint.
  3. Fall back to the Stooq CSV mirror which republishes AAII data.
"""
from __future__ import annotations

import sys
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

_XLS_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
_STOOQ_URL = "https://stooq.com/q/d/l/?s=aaiibull.i&i=w"   # weekly AAII bull index (stooq mirror)

# Column name normalisation map (XLS headers vary across downloads)
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

    # Keep only the core sentiment columns that exist
    keep = [c for c in ["date", "bullish", "neutral", "bearish", "bull_minus_bear"] if c in df.columns]
    df = df[keep].copy()
    df = df[pd.to_numeric(df.get("bullish", pd.Series(dtype=float)), errors="coerce").notna()]

    # Convert percentages (may be stored as 0–1 or 0–100)
    for col in ("bullish", "neutral", "bearish"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().max() <= 1.01:
                df[col] = df[col] * 100

    return df.dropna(subset=["date"]).reset_index(drop=True)


def _parse_stooq_csv(text: str) -> pd.DataFrame:
    """Parse Stooq weekly AAII bull CSV — has Date,Open,High,Low,Close columns."""
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Date": "date", "Close": "bullish"})
    df = df[["date", "bullish"]].copy()
    df["bullish"] = pd.to_numeric(df["bullish"], errors="coerce")
    return df.dropna(subset=["bullish"]).reset_index(drop=True)


def collect_aaii_sentiment() -> None:
    """Try XLS first, then Stooq CSV fallback."""

    # --- Attempt 1: Official AAII XLS ---
    print("  Trying AAII XLS download...")
    try:
        resp = fetch(_XLS_URL, timeout=30)
        df = _parse_xls(resp.content)
        if len(df) > 50:
            df = to_datetime_index(df, col="date")
            save(df, "sentiment", "aaii_sentiment_1w.parquet")
            return
        print(f"  XLS returned only {len(df)} rows — trying fallback")
    except Exception as exc:
        print(f"  WARNING AAII XLS: {exc}")

    # --- Attempt 2: Stooq CSV mirror ---
    print("  Trying Stooq AAII mirror...")
    try:
        resp = fetch(_STOOQ_URL, timeout=30)
        df = _parse_stooq_csv(resp.text)
        if len(df) > 50:
            df = to_datetime_index(df, col="date")
            save(df, "sentiment", "aaii_sentiment_1w.parquet")
            return
        print(f"  Stooq returned only {len(df)} rows")
    except Exception as exc:
        print(f"  WARNING Stooq: {exc}")

    print("  WARNING: All AAII sentiment sources failed or returned insufficient data")


def main() -> None:
    print("Fetching: AAII Investor Sentiment Survey (weekly)")
    try:
        collect_aaii_sentiment()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
