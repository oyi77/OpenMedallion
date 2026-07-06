"""
Top-200 crypto snapshot via CoinGecko /coins/markets (free, no key).

Single call returns top 200 coins ranked by market cap with current price,
market cap, 24h volume, and 24h price change.

Source:  https://api.coingecko.com/api/v3/coins/markets
Params:  vs_currency=usd, per_page=200, order=market_cap_desc
Limits:  ~30 calls/min on free tier — this collector uses 1 call.

Output:  data/crypto/coingecko_top200_1d.parquet
Columns: date (UTC DatetimeIndex), coin, symbol, price, market_cap,
         volume_24h, price_change_24h
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
TOP_N = 200

MARKETS_PARAMS: dict = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": TOP_N,
    "page": 1,
    "sparkline": "false",
}


def _fetch_top200() -> list[dict]:
    """Return the raw JSON list from /coins/markets."""
    resp = fetch(MARKETS_URL, params=MARKETS_PARAMS, retries=3)
    data = resp.json()
    if not isinstance(data, list):
        print(f"  WARNING: unexpected response type {type(data).__name__}")
        return []
    return data


def _build_dataframe(coins: list[dict]) -> pd.DataFrame:
    """Convert the API list into a tidy DataFrame with DatetimeIndex."""
    now = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    rows: list[dict] = []
    for coin in coins:
        price = coin.get("current_price")
        if price is None:
            continue
        rows.append({
            "date": now,
            "coin": coin.get("name", ""),
            "symbol": coin.get("symbol", "").upper(),
            "price": price,
            "market_cap": coin.get("market_cap"),
            "volume_24h": coin.get("total_volume"),
            "price_change_24h": coin.get("price_change_percentage_24h"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_values("market_cap", ascending=False)
    return df


def collect_coingecko_top200() -> None:
    """Fetch top-200 snapshot and persist to parquet."""
    coins = _fetch_top200()
    df = _build_dataframe(coins)
    save(df, "crypto", "coingecko_top200_1d.parquet")
    if not df.empty:
        print(f"  Top 5: {', '.join(df['symbol'].head(5).tolist())}")


def main() -> None:
    print("=== CoinGecko Top 200 Snapshot ===")
    try:
        collect_coingecko_top200()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
