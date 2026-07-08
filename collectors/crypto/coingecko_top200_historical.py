"""
Top-200 crypto historical OHLCV via CoinGecko (free, no key).

Fetches full price history from each coin's launch date to today.
Unlike coingecko_top200.py (daily snapshot), this builds a complete
time series dataset.

Source:  https://api.coingecko.com/api/v3/coins/{id}/market_chart
Params:  vs_currency=usd, days=max (gets full history)
Limits:  ~30 calls/min on free tier — this collector makes 200 calls.
         Rate limiting: 0.5s sleep between calls = 100 seconds total.

Output:  data/crypto/COIN_{symbol}_1d.parquet (one file per coin)
Columns: date (UTC DatetimeIndex), open, high, low, close, volume, market_cap
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
TOP_N = 200

MARKETS_PARAMS: dict = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": TOP_N,
    "page": 1,
    "sparkline": "false",
}

CHART_PARAMS: dict = {
    "vs_currency": "usd",
    "days": "max",  # Get full history since launch
    "interval": "daily",
}


def _fetch_top200() -> list[dict]:
    """Return the raw JSON list from /coins/markets."""
    resp = fetch(MARKETS_URL, params=MARKETS_PARAMS, retries=3)
    data = resp.json()
    if not isinstance(data, list):
        print(f"  WARNING: unexpected response type {type(data).__name__}")
        return []
    return data


def _fetch_coin_history(coin_id: str, symbol: str) -> pd.DataFrame | None:
    """Fetch full price history for a single coin."""
    url = CHART_URL.format(coin_id=coin_id)
    try:
        resp = fetch(url, params=CHART_PARAMS, retries=3)
        data = resp.json()
        
        if not isinstance(data, dict):
            print(f"  WARNING: {symbol} unexpected response type")
            return None
        
        prices = data.get("prices", [])
        market_caps = data.get("market_caps", [])
        volumes = data.get("total_volumes", [])
        
        if not prices:
            print(f"  WARNING: {symbol} no price data")
            return None
        
        # Convert to DataFrame
        df_price = pd.DataFrame(prices, columns=["timestamp", "close"])
        df_mcap = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
        df_vol = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        
        # Merge on timestamp
        df = df_price.merge(df_mcap, on="timestamp", how="left")
        df = df.merge(df_vol, on="timestamp", how="left")
        
        # Convert timestamp (milliseconds) to datetime
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop("timestamp", axis=1)
        
        # CoinGecko daily data only has close prices, not OHLC
        # We'll use close for all OHLC to maintain schema consistency
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        
        # Reorder columns
        df = df[["date", "open", "high", "low", "close", "volume", "market_cap"]]
        df = df.set_index("date").sort_index()
        
        return df
    
    except Exception as exc:
        print(f"  WARNING: {symbol} failed: {exc}")
        return None


def collect_coingecko_top200_historical() -> None:
    """Fetch top-200 historical OHLCV and persist per-coin parquet files."""
    print("Fetching top 200 coins list...")
    coins = _fetch_top200()
    
    if not coins:
        print("  No coins returned from markets API")
        return
    
    print(f"Fetching historical data for {len(coins)} coins...")
    success_count = 0
    
    for i, coin in enumerate(coins, 1):
        coin_id = coin.get("id")
        symbol = coin.get("symbol", "").upper()
        name = coin.get("name", "")
        
        if not coin_id or not symbol:
            print(f"  [{i}/{len(coins)}] Skipping invalid entry")
            continue
        
        print(f"  [{i}/{len(coins)}] {symbol:8s} ({name})")
        
        df = _fetch_coin_history(coin_id, symbol)
        if df is not None and not df.empty:
            filename = f"COIN_{symbol}_1d.parquet"
            save(df, "crypto", filename)
            
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")
            years = (df.index.max() - df.index.min()).days / 365.25
            
            print(f"      ✓ {len(df)} days ({start_date} → {end_date}, {years:.1f} years)")
            success_count += 1
        
        # Rate limiting: 0.5s between calls
        if i < len(coins):
            time.sleep(0.5)
    
    print(f"\n✓ Successfully collected {success_count}/{len(coins)} coins")


def main() -> None:
    print("=== CoinGecko Top 200 Historical OHLCV ===")
    try:
        collect_coingecko_top200_historical()
    except Exception as exc:
        print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
