"""
Funding rates history for major perpetual futures (Binance FAPI).

Fetches 8-hourly funding rates for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
going back to launch via paginated requests (max 1000 rows per call, forward walk).

Output: data/crypto/crypto_funding_rates_8h.parquet
Columns: symbol, funding_rate, mark_price
Index:   date (UTC DatetimeIndex)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

FAPI_BASE = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
LIMIT = 1000  # max per request

# Approx launch timestamps (ms) for each symbol's perpetual contract
SYMBOL_START_MS: dict[str, int] = {
    "BTCUSDT": 1567296000000,   # 2019-09-01
    "ETHUSDT": 1577836800000,   # 2020-01-01
    "SOLUSDT": 1609459200000,   # 2021-01-01
    "BNBUSDT": 1609459200000,   # 2021-01-01
    "XRPUSDT": 1609459200000,   # 2021-01-01
}


def _fetch_symbol_history(symbol: str) -> pd.DataFrame:
    """Paginate forward from launch to fetch full funding rate history."""
    all_rows: list[dict] = []
    start_time: int = SYMBOL_START_MS.get(symbol, 1577836800000)

    while True:
        params: dict = {"symbol": symbol, "limit": LIMIT, "startTime": start_time}
        resp = fetch(FAPI_BASE, params=params)
        rows = resp.json()

        if not rows:
            break

        all_rows.extend(rows)
        newest_ts: int = rows[-1]["fundingTime"]
        start_time = newest_ts + 1

        if len(rows) < LIMIT:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={
        "fundingTime": "date",
        "fundingRate": "funding_rate",
        "markPrice": "mark_price",
    })
    df["symbol"] = symbol
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    return df[["date", "symbol", "funding_rate", "mark_price"]]


def collect_funding_rates() -> None:
    """Fetch and persist full funding-rate history across symbols."""
    frames: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        print(f"  Fetching funding rates: {symbol}")
        try:
            df = _fetch_symbol_history(symbol)
            if not df.empty:
                print(f"    {len(df):,} rows for {symbol}")
                frames.append(df)
        except Exception as exc:
            print(f"  WARNING: {symbol} — {exc}")

    if not frames:
        print("  WARNING: no funding-rate data collected")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index("date").sort_index()
    save(combined, "crypto", "crypto_funding_rates_8h.parquet")


def main() -> None:
    print("Fetching: Funding Rates (Binance FAPI)")
    try:
        collect_funding_rates()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
