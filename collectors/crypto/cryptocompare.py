"""
CryptoCompare social stats + extended OHLCV collector.
Source: CryptoCompare API — free tier (100k calls/month), no key for basic.
Covers: Social stats (Reddit, Twitter, Facebook followers), 2000+ coin OHLCV hourly.
Output: data/crypto/CC_<SYMBOL>_social_1d.parquet
         data/crypto/CC_<SYMBOL>_1h.parquet
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

CC_BASE = "https://min-api.cryptocompare.com/data"
CC_KEY = os.environ.get("CRYPTOCOMPARE_API_KEY", "")  # Optional: higher rate limits

# Coins beyond what Binance covers — smaller/mid caps
EXTRA_COINS = [
    "XLM", "ALGO", "VET", "IOTA", "XTZ", "EOS", "DASH", "ZEC", "XMR",
    "THETA", "FIL", "ETC", "BTT", "ONE", "HBAR", "EGLD", "FLOW",
    "MINA", "ROSE", "CFX", "KSM", "ZIL", "ENJ", "CHZ", "GRT",
    "1INCH", "COMP", "BAL", "SUSHI", "YFI", "ALPHA", "PERP",
    # Indonesian-listed tokens
    "REEF", "ORBS", "CELR", "DENT", "HOT", "WIN",
]

SOCIAL_COINS = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOGE", "DOT", "LINK", "BNB", "MATIC",
                "AVAX", "UNI", "LTC", "BCH", "XLM", "ALGO", "VET", "FIL"]


def headers() -> dict:
    if CC_KEY:
        return {"authorization": f"Apikey {CC_KEY}"}
    return {}


def fetch_ohlcv_hourly(symbol: str, limit: int = 2000) -> pd.DataFrame:
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": limit,
        "aggregate": 1,
    }
    resp = fetch(f"{CC_BASE}/v2/histohour", params=params)
    data = resp.json()
    if data.get("Response") != "Success":
        return pd.DataFrame()
    rows = data.get("Data", {}).get("Data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volumefrom"]].rename(
        columns={"volumefrom": "volume"}
    ).apply(pd.to_numeric, errors="coerce")


def fetch_social_stats(symbol: str) -> pd.DataFrame:
    """Fetch current social stats snapshot."""
    params = {"fsym": symbol, "extraParams": "OpenMedallion"}
    resp = fetch(f"{CC_BASE}/social/coin/latest", params=params)
    data = resp.json()
    if data.get("Response") != "Success":
        return pd.DataFrame()

    stats = data.get("Data", {})
    reddit = stats.get("Reddit", {})
    twitter = stats.get("Twitter", {})
    fb = stats.get("Facebook", {})

    row = {
        "reddit_subscribers": reddit.get("subscribers"),
        "reddit_active_users": reddit.get("active_users"),
        "twitter_followers": twitter.get("followers"),
        "twitter_statuses": twitter.get("statuses"),
        "facebook_likes": fb.get("likes"),
        "facebook_talking_about": fb.get("talking_about"),
    }
    df = pd.DataFrame([row], index=[pd.Timestamp.utcnow().floor("D")])
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def fetch_social_history(symbol: str, limit: int = 365) -> pd.DataFrame:
    """Fetch historical daily social stats."""
    params = {"fsym": symbol, "limit": limit}
    resp = fetch(f"{CC_BASE}/social/coin/histo/day", params=params)
    data = resp.json()
    if data.get("Response") != "Success":
        return pd.DataFrame()
    rows = data.get("Data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def main() -> None:
    # Extra coin OHLCV
    for symbol in EXTRA_COINS:
        print(f"Fetching CC OHLCV {symbol} ...")
        try:
            df = fetch_ohlcv_hourly(symbol)
            if not df.empty:
                save(df, "crypto", f"CC_{symbol}_1h.parquet")
            time.sleep(0.5)
        except Exception as exc:
            print(f"  WARNING: {symbol} OHLCV — {exc}")

    # Social stats history
    print("\nFetching social stats history ...")
    for symbol in SOCIAL_COINS:
        try:
            df_social = fetch_social_history(symbol)
            if not df_social.empty:
                save(df_social, "crypto", f"CC_{symbol}_social_1d.parquet")
            time.sleep(0.5)
        except Exception as exc:
            print(f"  WARNING: {symbol} social — {exc}")


if __name__ == "__main__":
    main()
