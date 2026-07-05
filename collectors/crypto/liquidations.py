"""
Liquidations history proxy for major perpetual futures.

Since Binance deprecated the allForceOrders public endpoint, this collector
builds a liquidation pressure proxy from two free sources:

1. Binance FAPI 1d klines → taker-sell volume (aggressive market sells, the
   dominant signature of long liquidations) and taker-buy volume (short liq proxy).
   taker_sell_usd  = quote_volume - taker_buy_quote
   taker_buy_usd   = taker_buy_quote
   These are correlated with, but larger than, pure liquidations.

2. Bybit account-ratio → long/short position ratio as a sentiment overlay.

Both are combined into a single wide parquet.

Output: data/crypto/crypto_liquidations_1d.parquet
Columns: symbol, taker_sell_usd (long_liq_proxy), taker_buy_usd (short_liq_proxy),
         total_taker_usd, long_short_ratio (Bybit)
Index:   date (UTC DatetimeIndex)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BYBIT_RATIO = "https://api.bybit.com/v5/market/account-ratio"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
LIMIT = 1000

# Approximate perp launch dates (ms) per symbol
SYMBOL_START_MS: dict[str, int] = {
    "BTCUSDT": 1568073600000,   # 2019-09-10 (Binance USDT perp launch)
    "ETHUSDT": 1577836800000,   # 2020-01-01
    "SOLUSDT": 1609459200000,   # 2021-01-01
    "BNBUSDT": 1609459200000,   # 2021-01-01
    "XRPUSDT": 1609459200000,   # 2021-01-01
}


def _fetch_fapi_klines(symbol: str) -> pd.DataFrame:
    """Paginate Binance FAPI 1d klines forward from launch to present."""
    all_rows: list[list] = []
    start_ms: int = SYMBOL_START_MS.get(symbol, 1577836800000)

    while True:
        resp = fetch(FAPI_KLINES, params={
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ms,
            "limit": LIMIT,
        })
        rows: list[list] = resp.json()

        if not rows:
            break

        all_rows.extend(rows)
        start_ms = rows[-1][0] + 86_400_000

        if len(rows) < LIMIT:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["date"] = (
        pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        .dt.normalize()
    )
    df["taker_buy_usd"] = pd.to_numeric(df["taker_buy_quote"], errors="coerce")
    df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    df["taker_sell_usd"] = df["quote_volume"] - df["taker_buy_usd"]
    df["total_taker_usd"] = df["quote_volume"]
    df["symbol"] = symbol

    return (
        df[["date", "symbol", "taker_sell_usd", "taker_buy_usd", "total_taker_usd"]]
        .drop_duplicates("date")
    )


def _fetch_bybit_ls_ratio(symbol: str) -> pd.DataFrame:
    """Paginate Bybit daily long/short account ratio for one symbol."""
    all_rows: list[dict] = []
    cursor: str | None = None

    while True:
        params: dict = {
            "category": "linear",
            "symbol": symbol,
            "period": "1d",
            "limit": 500,
        }
        if cursor:
            params["cursor"] = cursor

        resp = fetch(BYBIT_RATIO, params=params)
        result = resp.json().get("result", {})
        rows: list[dict] = result.get("list", [])

        if not rows:
            break

        all_rows.extend(rows)
        cursor = result.get("nextPageCursor", "")

        if not cursor or len(rows) < 500:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = (
        pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        .dt.normalize()
    )
    df["long_short_ratio"] = pd.to_numeric(df["buyRatio"], errors="coerce") / pd.to_numeric(df["sellRatio"], errors="coerce")
    df["symbol"] = symbol
    return df[["date", "symbol", "long_short_ratio"]].drop_duplicates("date")


def collect_liquidations() -> None:
    """Fetch taker-volume proxy and L/S ratio; merge and persist."""
    frames: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        print(f"  Fetching taker volume (liquidation proxy): {symbol}")
        try:
            df_klines = _fetch_fapi_klines(symbol)
            if df_klines.empty:
                print(f"    WARNING: no kline data for {symbol}")
                continue
            print(f"    {len(df_klines):,} kline rows")

            df_ratio = _fetch_bybit_ls_ratio(symbol)
            if not df_ratio.empty:
                print(f"    {len(df_ratio):,} L/S ratio rows")
                df_klines = df_klines.merge(
                    df_ratio[["date", "long_short_ratio"]],
                    on="date", how="left",
                )
            else:
                df_klines["long_short_ratio"] = float("nan")

            frames.append(df_klines)
        except Exception as exc:
            print(f"    WARNING: {symbol} — {exc}")

        time.sleep(0.2)

    if not frames:
        print("  WARNING: no liquidation proxy data collected")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index("date").sort_index()
    save(combined, "crypto", "crypto_liquidations_1d.parquet")


def main() -> None:
    print("Fetching: Liquidations / Taker Volume Proxy (Binance FAPI + Bybit)")
    try:
        collect_liquidations()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
