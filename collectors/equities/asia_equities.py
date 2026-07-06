"""
Asia equities collector — India NSE (30), Korea KRX (20), Taiwan TWSE (20).

Fetches max available daily OHLCV history via yfinance and saves each ticker
to data/equities/asia/<TICKER>_1d.parquet.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index  # noqa: E402

# ---------------------------------------------------------------------------
# Ticker lists
# ---------------------------------------------------------------------------

INDIA_NSE: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "NESTLEIND.NS",
    "WIPRO.NS", "ULTRACEMCO.NS", "ONGC.NS", "TITAN.NS", "POWERGRID.NS",
    "NTPC.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "HCLTECH.NS",
    "TECHM.NS", "DRREDDY.NS", "DIVISLAB.NS", "COALINDIA.NS", "ADANIPORTS.NS",
]

KOREA_KRX: list[str] = [
    "005930.KS", "000660.KS", "035420.KS", "005380.KS", "051910.KS",
    "035720.KS", "006400.KS", "000270.KS", "105560.KS", "096770.KS",
    "003550.KS", "032830.KS", "055550.KS", "012330.KS", "018880.KS",
    "010130.KS", "316140.KS", "028260.KS", "009150.KS", "011200.KS",
]

TAIWAN_TWSE: list[str] = [
    "2330.TW", "2317.TW", "2454.TW", "2412.TW", "2308.TW",
    "2303.TW", "2881.TW", "2882.TW", "2891.TW", "1301.TW",
    "1303.TW", "2886.TW", "2884.TW", "2885.TW", "2002.TW",
    "2207.TW", "3711.TW", "2357.TW", "4904.TW", "2395.TW",
]

ALL_TICKERS: list[str] = INDIA_NSE + KOREA_KRX + TAIWAN_TWSE

BATCH_SIZE = 5
BATCH_SLEEP = 0.2  # seconds between batches


# ---------------------------------------------------------------------------
# Fetch logic
# ---------------------------------------------------------------------------

def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
    """Download max daily history for a single ticker; returns (ticker, df)."""
    try:
        raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if raw.empty:
            return ticker, pd.DataFrame()

        # yf.download may return MultiIndex columns when a single ticker is
        # passed as a string — flatten if needed.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
        df = raw[cols].copy()
        df.columns = df.columns.str.lower()
        df = to_datetime_index(df)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return ticker, df
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN {ticker}: {exc}")
        return ticker, pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ok = 0
    fail = 0
    total = len(ALL_TICKERS)

    # Process in batches to respect rate limits
    batches = [ALL_TICKERS[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in batch}
            for future in as_completed(futures):
                ticker, df = future.result()
                if df.empty:
                    print(f"  FAIL {ticker}")
                    fail += 1
                else:
                    save(df, "equities/asia", f"{ticker}_1d.parquet")
                    ok += 1

        if batch_idx < len(batches) - 1:
            time.sleep(BATCH_SLEEP)

    print(f"\nDone — {ok}/{total} ok, {fail}/{total} failed")


if __name__ == "__main__":
    main()
