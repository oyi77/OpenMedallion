"""
Top-200 crypto historical OHLCV via Yahoo Finance (free, no key, no limits).

Uses yfinance library to fetch full price history for major cryptocurrencies
that have Yahoo Finance ticker symbols (BTC-USD, ETH-USD, etc.).

Yahoo Finance provides:
- Full history back to each coin's listing on major exchanges
- OHLCV data with adjusted close
- No API key required
- No rate limits

Source:  yfinance library (wraps finance.yahoo.com)
Coverage: ~150 major coins with Yahoo tickers
Limits:  None — completely free

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

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
TOP_N = 200

MARKETS_PARAMS: dict = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": TOP_N,
    "page": 1,
    "sparkline": "false",
}

# Map CoinGecko symbols to Yahoo Finance tickers
SYMBOL_TO_TICKER: dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDT": "USDT-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "USDC": "USDC-USD",
    "XRP": "XRP-USD",
    "STETH": "STETH-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "TRX": "TRX-USD",
    "AVAX": "AVAX-USD",
    "WBTC": "WBTC-USD",
    "TON": "TON11419-USD",
    "LINK": "LINK-USD",
    "SHIB": "SHIB-USD",
    "DOT": "DOT-USD",
    "BCH": "BCH-USD",
    "DAI": "DAI-USD",
    "LEO": "LEO-USD",
    "LTC": "LTC-USD",
    "UNI": "UNI7083-USD",
    "MATIC": "MATIC-USD",
    "NEAR": "NEAR-USD",
    "PEPE": "PEPE24478-USD",
    "ICP": "ICP-USD",
    "WEETH": "WEETH-USD",
    "FET": "FET-USD",
    "APT": "APT21794-USD",
    "ETC": "ETC-USD",
    "STX": "STX4847-USD",
    "XLM": "XLM-USD",
    "ATOM": "ATOM-USD",
    "MNT": "MNT27075-USD",
    "OKB": "OKB-USD",
    "HBAR": "HBAR-USD",
    "FIL": "FIL-USD",
    "IMX": "IMX10603-USD",
    "ARB": "ARB11841-USD",
    "VET": "VET-USD",
    "OP": "OP-USD",
    "INJ": "INJ-USD",
    "MKR": "MKR-USD",
    "RUNE": "RUNE-USD",
    "TIA": "TIA22861-USD",
    "AAVE": "AAVE-USD",
    "GRT": "GRT6719-USD",
    "THETA": "THETA-USD",
    "ALGO": "ALGO-USD",
    "SEI": "SEI-USD",
    "FLOW": "FLOW-USD",
    "WIF": "WIF28752-USD",
    "FTM": "FTM-USD",
    "RNDR": "RNDR-USD",
    "SAND": "SAND-USD",
    "FLOKI": "FLOKI-USD",
    "AXS": "AXS-USD",
    "GALA": "GALA-USD",
    "MANA": "MANA-USD",
    "CHZ": "CHZ-USD",
    "ENJ": "ENJ-USD",
    "XTZ": "XTZ-USD",
    "EOS": "EOS-USD",
    "EGLD": "EGLD-USD",
    "KCS": "KCS-USD",
    "ZEC": "ZEC-USD",
    "XMR": "XMR-USD",
    "DASH": "DASH-USD",
    "NEO": "NEO-USD",
    "WAVES": "WAVES-USD",
def _fetch_yfinance_history(ticker: str, symbol: str) -> pd.DataFrame | None:
    """Fetch full price history from Yahoo Finance."""
    try:
        # Download full history using Ticker object (v1.4.1 API)
        coin = yf.Ticker(ticker)
        df = coin.history(period="max", interval="1d")
        
        if df.empty:
            print(f"      ✗ No data available")
            return None
        
        # Rename columns to match our schema
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        # Calculate market cap as close * volume (approximate)
        # Real market cap requires circulating supply, which yfinance doesn't provide
        df["market_cap"] = df["close"] * df["volume"]
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        
        # Select only the columns we need
        df = df[["open", "high", "low", "close", "volume", "market_cap"]]
        
        return df
    
    except Exception as exc:
        print(f"      ✗ Failed: {exc}")
        return None
}


def _fetch_top200() -> list[dict]:
    """Return the raw JSON list from /coins/markets."""
    resp = fetch(MARKETS_URL, params=MARKETS_PARAMS, retries=3)
    data = resp.json()
    if not isinstance(data, list):
        print(f"  WARNING: unexpected response type {type(data).__name__}")
        return []
    return data


def _fetch_yfinance_history(ticker: str, symbol: str) -> pd.DataFrame | None:
    """Fetch full price history from Yahoo Finance."""
    try:
        # Download full history
        df = yf.download(
            ticker,
            period="max",
            interval="1d",
            progress=False,
            show_errors=False,
        )
        
        if df.empty:
            print(f"      ✗ No data available")
            return None
        
        # Clean up MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns to match our schema
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        # Calculate market cap as close * volume (approximate)
        # Real market cap requires circulating supply, which yfinance doesn't provide
        df["market_cap"] = df["close"] * df["volume"]
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        
        # Select only the columns we need
        df = df[["open", "high", "low", "close", "volume", "market_cap"]]
        
        return df
    
    except Exception as exc:
        print(f"      ✗ Failed: {exc}")
        return None


def collect_yfinance_crypto_historical() -> None:
    """Fetch top-200 historical OHLCV via Yahoo Finance."""
    print("Fetching top 200 coins list from CoinGecko...")
    coins = _fetch_top200()
    
    if not coins:
        print("  No coins returned from markets API")
        return
    
    print(f"\nFetching historical data from Yahoo Finance...")
    success_count = 0
    skipped_count = 0
    
    for i, coin in enumerate(coins, 1):
        symbol = coin.get("symbol", "").upper()
        name = coin.get("name", "")
        
        if not symbol:
            print(f"  [{i}/{len(coins)}] Skipping invalid entry")
            continue
        
        # Check if we have a Yahoo Finance ticker mapping
        ticker = SYMBOL_TO_TICKER.get(symbol)
        if not ticker:
            skipped_count += 1
            continue
        
        print(f"  [{i}/{len(coins)}] {symbol:8s} ({name})")
        
        df = _fetch_yfinance_history(ticker, symbol)
        if df is not None and not df.empty:
            filename = f"COIN_{symbol}_1d.parquet"
            save(df, "crypto", filename)
            
            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")
            years = (df.index.max() - df.index.min()).days / 365.25
            
            print(f"      ✓ {len(df)} days ({start_date} → {end_date}, {years:.1f} years)")
            success_count += 1
        
        # Small delay to be nice to Yahoo servers
        time.sleep(0.1)
    
    print(f"\n✓ Successfully collected {success_count}/{len(coins)} coins")
    print(f"  ({skipped_count} coins skipped — no Yahoo ticker mapping)")


def main() -> None:
    print("=== Yahoo Finance Crypto Historical OHLCV ===")
    try:
        collect_yfinance_crypto_historical()
    except Exception as exc:
        print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
