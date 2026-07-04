"""
Wikipedia pageviews financial collector.
Source: Wikimedia Pageviews API — free, no key required.
Covers: Daily pageviews for major company/asset articles as investor attention proxy.
Output: data/alternative/Wiki_Pageviews_<article>_1d.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
HEADERS = {"User-Agent": "OpenMedallion-Collector/1.0 (research; https://github.com/oyi77/OpenMedallion)"}

# Articles to track as attention proxy
ARTICLES = {
    # Companies
    "Apple_Inc": "AAPL",
    "Microsoft": "MSFT",
    "Tesla,_Inc": "TSLA",
    "NVIDIA": "NVDA",
    "Amazon_(company)": "AMZN",
    "Alphabet_Inc": "GOOGL",
    "Meta_Platforms": "META",
    "JPMorgan_Chase": "JPM",
    "Berkshire_Hathaway": "BRK",
    # Crypto
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Dogecoin": "DOGE",
    "Solana_(blockchain_platform)": "SOL",
    # Financial concepts (narrative tracking)
    "Recession": "RECESSION",
    "Inflation": "INFLATION",
    "Federal_Reserve": "FED",
    "Stock_market_crash": "CRASH",
    "GameStop_short_squeeze": "GME_SQUEEZE",
    # Indonesian assets
    "Bank_Central_Asia": "BBCA",
    "Bank_Rakyat_Indonesia": "BBRI",
    "Telkom_Indonesia": "TLKM",
}


def fetch_pageviews(article: str, start: str = "20200101", end: str | None = None) -> pd.DataFrame:
    import datetime
    if end is None:
        end = datetime.date.today().strftime("%Y%m%d")

    url = f"{WIKI_API}/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
    import requests
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame()

    items = resp.json().get("items", [])
    if not items:
        return pd.DataFrame()

    rows = [{"date": item["timestamp"][:8], "views": item["views"]} for item in items]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", utc=True)
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    return df.set_index("date").sort_index().dropna()


def main() -> None:
    for article, ticker in ARTICLES.items():
        print(f"Fetching Wikipedia pageviews: {article} ({ticker}) ...")
        try:
            df = fetch_pageviews(article)
            if not df.empty:
                safe = article.replace(",", "").replace("(", "").replace(")", "").replace(" ", "_")[:50]
                save(df, "alternative", f"Wiki_{safe}_pageviews_1d.parquet")
            time.sleep(0.3)
        except Exception as exc:
            print(f"  WARNING: {article} — {exc}")


if __name__ == "__main__":
    main()
