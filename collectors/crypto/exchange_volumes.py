"""
Cross-exchange volume aggregation and BTC market dominance history.

Sources (free, no API key required):
  1. Binance SPOT klines (BTCUSDT 1d) → BTC volume and price history
  2. DeFiLlama stablecoin charts → total stablecoin market cap as a crypto
     market size proxy
  3. DeFiLlama TVL charts → DeFi TVL for broader market context

Output: data/crypto/exchange_volume_dominance_1d.parquet
Columns: btc_volume_usd, btc_price_close, btc_market_cap_approx,
         total_stablecoin_supply, defi_tvl_usd
Index:   date (UTC DatetimeIndex)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
LLAMA_STABLECOINS = "https://stablecoins.llama.fi/stablecoincharts/all"
LLAMA_TVL = "https://api.llama.fi/charts"

# BTC circulating supply approximation (used only for rough market cap estimate)
BTC_APPROX_SUPPLY = 19_700_000


def _fetch_btc_daily_klines() -> pd.DataFrame:
    """Paginate Binance 1d klines for BTCUSDT from 2017 to present."""
    all_rows: list[list] = []
    # BTC/USDT listing on Binance ~2017-08-22
    start_ms: int = 1503360000000
    limit = 1000

    while True:
        resp = fetch(BINANCE_KLINES, params={
            "symbol": "BTCUSDT",
            "interval": "1d",
            "startTime": start_ms,
            "limit": limit,
        })
        rows: list[list] = resp.json()

        if not rows:
            break

        all_rows.extend(rows)
        newest_ts: int = rows[-1][0]
        start_ms = newest_ts + 86_400_000  # advance one day

        if len(rows) < limit:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True).dt.normalize()
    df["btc_price_close"] = pd.to_numeric(df["close"], errors="coerce")
    df["btc_volume_usd"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    df["btc_market_cap_approx"] = df["btc_price_close"] * BTC_APPROX_SUPPLY

    return (
        df[["date", "btc_volume_usd", "btc_price_close", "btc_market_cap_approx"]]
        .drop_duplicates("date")
        .set_index("date")
    )


def _fetch_stablecoin_total() -> pd.Series:
    """Total stablecoin USD circulation from DeFiLlama."""
    resp = fetch(LLAMA_STABLECOINS)
    rows: list[dict] = resp.json()
    records = [
        (int(r["date"]), r.get("totalCirculatingUSD", {}).get("peggedUSD", float("nan")))
        for r in rows
    ]
    df = pd.DataFrame(records, columns=["ts", "total_stablecoin_supply"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.normalize()
    return df.drop_duplicates("date").set_index("date")["total_stablecoin_supply"]


def _fetch_defi_tvl() -> pd.Series:
    """DeFi TVL in USD from DeFiLlama."""
    resp = fetch(LLAMA_TVL)
    rows: list[dict] = resp.json()
    records = [
        (int(r["date"]), float(r.get("totalLiquidityUSD", float("nan"))))
        for r in rows
    ]
    df = pd.DataFrame(records, columns=["ts", "defi_tvl_usd"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.normalize()
    return df.drop_duplicates("date").set_index("date")["defi_tvl_usd"]


def collect_exchange_volumes() -> None:
    """Merge BTC volume, stablecoin supply, and DeFi TVL into one wide table."""
    print("  Fetching BTC daily klines (Binance)")
    df_btc = _fetch_btc_daily_klines()
    print(f"    {len(df_btc):,} rows")

    time.sleep(1)

    print("  Fetching total stablecoin supply (DeFiLlama)")
    try:
        s_stable = _fetch_stablecoin_total()
        print(f"    {len(s_stable):,} rows")
    except Exception as exc:
        print(f"    WARNING: {exc}")
        s_stable = pd.Series(dtype=float)

    time.sleep(0.5)

    print("  Fetching DeFi TVL (DeFiLlama)")
    try:
        s_tvl = _fetch_defi_tvl()
        print(f"    {len(s_tvl):,} rows")
    except Exception as exc:
        print(f"    WARNING: {exc}")
        s_tvl = pd.Series(dtype=float)

    combined = df_btc.copy()
    if not s_stable.empty:
        combined = combined.join(s_stable, how="outer")
    if not s_tvl.empty:
        combined = combined.join(s_tvl, how="outer")

    combined.index.name = "date"
    combined = combined.sort_index()

    save(combined, "crypto", "exchange_volume_dominance_1d.parquet")


def main() -> None:
    print("Fetching: Exchange Volume & Dominance (Binance + DeFiLlama)")
    try:
        collect_exchange_volumes()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
