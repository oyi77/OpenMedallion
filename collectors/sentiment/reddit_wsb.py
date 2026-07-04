"""
Reddit WSB sentiment collector.
Source: Reddit API (public, no OAuth needed for read-only via pushshift / Reddit JSON)
        + Pushshift.io archive for historical data.
Covers: WallStreetBets daily mention counts, sentiment for top tickers.
Output: data/sentiment/Reddit_WSB_mentions_1d.parquet
         data/sentiment/Reddit_WSB_sentiment_1d.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# Reddit JSON API (no auth needed for public posts)
REDDIT_BASE = "https://www.reddit.com"
REDDIT_HEADERS = {"User-Agent": "OpenMedallion-Collector/1.0"}

# Tracked tickers for sentiment
TRACKED_TICKERS = [
    "GME", "AMC", "TSLA", "NVDA", "AAPL", "SPY", "QQQ", "MSFT", "AMZN",
    "META", "GOOGL", "AMD", "PLTR", "BBBY", "BBAI", "SOFI", "RIVN",
    "LCID", "MARA", "RIOT", "COIN", "BTC", "ETH",
]


def fetch_wsb_hot(limit: int = 100) -> pd.DataFrame:
    """Fetch current hot posts from WSB."""
    url = f"{REDDIT_BASE}/r/wallstreetbets/hot.json"
    params = {"limit": limit}
    import requests
    resp = requests.get(url, params=params, headers=REDDIT_HEADERS, timeout=15)
    if resp.status_code != 200:
        return pd.DataFrame()
    data = resp.json()
    posts = data.get("data", {}).get("children", [])
    rows = []
    for p in posts:
        d = p.get("data", {})
        rows.append({
            "title": d.get("title", ""),
            "score": d.get("score"),
            "num_comments": d.get("num_comments"),
            "upvote_ratio": d.get("upvote_ratio"),
            "created_utc": d.get("created_utc"),
            "selftext": (d.get("selftext") or "")[:500],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    return df


def count_ticker_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Count ticker mentions in titles and text."""
    today = pd.Timestamp.utcnow().floor("D")
    rows = []
    combined = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.upper()
    for ticker in TRACKED_TICKERS:
        # Word-boundary match
        import re
        pattern = r"\b" + re.escape(ticker) + r"\b"
        count = combined.str.count(pattern).sum()
        rows.append({"date": today, "ticker": ticker, "mentions": int(count)})
    df_out = pd.DataFrame(rows)
    df_out["date"] = pd.DatetimeIndex(df_out["date"], tz="UTC")
    df_out = df_out.set_index("date")
    return df_out


def simple_sentiment(text: str) -> float:
    """Naive keyword-based sentiment score [-1, 1]."""
    bullish = ["bull", "calls", "moon", "🚀", "buy", "long", "pump", "yolo", "green", "up", "squeeze"]
    bearish = ["bear", "puts", "crash", "short", "dump", "red", "down", "sell", "rekt", "loss"]
    t = text.lower()
    score = sum(1 for w in bullish if w in t) - sum(1 for w in bearish if w in t)
    total = sum(1 for w in bullish + bearish if w in t) or 1
    return round(score / total, 4)


def main() -> None:
    print("Fetching Reddit WSB hot posts ...")
    try:
        df_posts = fetch_wsb_hot(limit=100)
        if df_posts.empty:
            print("  WARNING: No WSB posts fetched")
            return

        # Mention counts
        df_mentions = count_ticker_mentions(df_posts)
        save(df_mentions, "sentiment", "Reddit_WSB_mentions_1d.parquet")

        # Post-level sentiment snapshot
        df_posts["sentiment"] = (df_posts["title"] + " " + df_posts["selftext"]).apply(simple_sentiment)
        df_posts = df_posts.set_index("created_utc").sort_index()
        df_posts = df_posts.select_dtypes(include=["number", "float"]).dropna(how="all")
        save(df_posts, "sentiment", "Reddit_WSB_post_sentiment_snapshot.parquet")

    except Exception as exc:
        print(f"  WARNING: WSB sentiment — {exc}")


if __name__ == "__main__":
    main()
