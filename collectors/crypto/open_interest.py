"""
Open interest history for major perpetual futures via Bybit API.

Bybit's cursor-based pagination returns 1d OI going back to ~2020-08-05 for BTC.
Fetches BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT.

Output: data/crypto/crypto_open_interest_1d.parquet
Columns: symbol, open_interest_usd, open_interest_coins
Index:   date (UTC DatetimeIndex)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

BYBIT_OI_URL = "https://api.bybit.com/v5/market/open-interest"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
LIMIT = 200  # max per Bybit request


def _fetch_symbol_oi(symbol: str) -> pd.DataFrame:
    """Paginate via Bybit cursor to fetch full daily OI history."""
    all_rows: list[dict] = []
    cursor: str | None = None

    while True:
        params: dict = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": "1d",
            "limit": LIMIT,
        }
        if cursor:
            params["cursor"] = cursor

        resp = fetch(BYBIT_OI_URL, params=params)
        result = resp.json().get("result", {})
        rows: list[dict] = result.get("list", [])

        if not rows:
            break

        all_rows.extend(rows)
        cursor = result.get("nextPageCursor", "")

        if not cursor or len(rows) < LIMIT:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True).dt.normalize()
    df["open_interest_coins"] = pd.to_numeric(df["openInterest"], errors="coerce")
    # Bybit does not give USD value directly; use singleOpenInterest × 2 as notional proxy
    # openInterest = total open interest in base coin; mark price not included here
    # We save coins; USD can be derived later with price data
    df["open_interest_usd"] = float("nan")

    return df[["date", "symbol", "open_interest_usd", "open_interest_coins"]]


def collect_open_interest() -> None:
    """Fetch and persist daily OI history across symbols."""
    frames: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        print(f"  Fetching open interest: {symbol}")
        try:
            df = _fetch_symbol_oi(symbol)
            if not df.empty:
                print(f"    {len(df):,} rows for {symbol}")
                frames.append(df)
        except Exception as exc:
            print(f"  WARNING: {symbol} — {exc}")

    if not frames:
        print("  WARNING: no open-interest data collected")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index("date").sort_index()
    save(combined, "crypto", "crypto_open_interest_1d.parquet")


def main() -> None:
    print("Fetching: Open Interest (Bybit)")
    try:
        collect_open_interest()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
