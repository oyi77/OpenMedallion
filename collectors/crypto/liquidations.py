"""
Liquidations history for major perpetual futures.

Primary source: Binance FAPI allForceOrders endpoint — returns forced liquidation
orders (market buy/sell orders triggered by liquidation engine).

Aggregates per-symbol daily: long_liq_usd, short_liq_usd, total_liq_usd.

Output: data/crypto/crypto_liquidations_1d.parquet
Columns: symbol, long_liq_usd, short_liq_usd, total_liq_usd
Index:   date (UTC DatetimeIndex)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

FORCE_ORDERS_URL = "https://fapi.binance.com/fapi/v1/allForceOrders"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
LIMIT = 1000  # max per request


def _fetch_symbol_liquidations(symbol: str) -> pd.DataFrame:
    """Paginate backwards across allForceOrders to build liquidation history."""
    all_rows: list[dict] = []
    end_time: int | None = None

    while True:
        params: dict = {"symbol": symbol, "limit": LIMIT}
        if end_time is not None:
            params["endTime"] = end_time

        resp = fetch(FORCE_ORDERS_URL, params=params)
        rows = resp.json()

        if not rows:
            break

        all_rows.extend(rows)

        oldest_ts = rows[0]["time"]
        end_time = oldest_ts - 1

        if len(rows) < LIMIT:
            break

        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.normalize()
    df["symbol"] = symbol
    df["qty"] = pd.to_numeric(df["origQty"], errors="coerce")
    df["price"] = pd.to_numeric(df["averagePrice"], errors="coerce")
    df["notional"] = df["qty"] * df["price"]

    # "SELL" side = long liquidation (forced close of a long position)
    # "BUY"  side = short liquidation (forced close of a short position)
    longs = (
        df[df["side"] == "SELL"]
        .groupby("date")["notional"]
        .sum()
        .rename("long_liq_usd")
    )
    shorts = (
        df[df["side"] == "BUY"]
        .groupby("date")["notional"]
        .sum()
        .rename("short_liq_usd")
    )

    daily = pd.concat([longs, shorts], axis=1).fillna(0.0)
    daily["total_liq_usd"] = daily["long_liq_usd"] + daily["short_liq_usd"]
    daily.index = pd.to_datetime(daily.index, utc=True)
    daily.index.name = "date"
    daily["symbol"] = symbol
    return daily.reset_index()[["date", "symbol", "long_liq_usd", "short_liq_usd", "total_liq_usd"]]


def collect_liquidations() -> None:
    """Fetch and persist daily liquidation history across symbols."""
    frames: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        print(f"  Fetching liquidations: {symbol}")
        try:
            df = _fetch_symbol_liquidations(symbol)
            if not df.empty:
                print(f"    {len(df):,} daily rows for {symbol}")
                frames.append(df)
        except Exception as exc:
            print(f"  WARNING: {symbol} — {exc}")

    if not frames:
        print("  WARNING: no liquidation data collected")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index("date").sort_index()
    save(combined, "crypto", "crypto_liquidations_1d.parquet")


def main() -> None:
    print("Fetching: Liquidations (Binance FAPI allForceOrders)")
    try:
        collect_liquidations()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
