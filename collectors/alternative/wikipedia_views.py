"""
Wikipedia page views for market-relevant finance topics.
Source: Wikimedia Pageviews REST API — free, no key required.
Covers monthly views for key financial articles as a retail attention/narrative proxy.
Output: data/alternative/wikipedia_views_finance_1m.parquet — date, article, views
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
HEADERS = {
    "User-Agent": "OpenMedallion-Collector/1.0 (research; https://github.com/oyi77/OpenMedallion)"
}

# Article slug -> human-readable label
ARTICLES: dict[str, str] = {
    "Bitcoin": "Bitcoin",
    "Gold": "Gold",
    "Stock_market": "Stock_market",
    "Recession": "Recession",
    "Inflation": "Inflation",
    "Federal_Reserve": "Federal_Reserve",
    "Price_of_oil": "Oil_price",
    "S%26P_500": "S&P_500",
    "Cryptocurrency": "Cryptocurrency",
    "Interest_rate": "Interest_rate",
}

START = "20150101"
END = "20241201"


def _fetch_monthly(article: str) -> pd.DataFrame:
    """Fetch monthly pageviews for a single article."""
    url = f"{WIKI_API}/en.wikipedia/all-access/all-agents/{article}/monthly/{START}/{END}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame()
    items = resp.json().get("items", [])
    if not items:
        return pd.DataFrame()
    rows = [
        {"date": item["timestamp"][:8], "views": item["views"]}
        for item in items
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", utc=True)
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    return df.dropna(subset=["views"])


def collect_wikipedia_views() -> None:
    """Collect and stack monthly pageviews for all tracked articles."""
    frames: list[pd.DataFrame] = []

    for slug, label in ARTICLES.items():
        print(f"  Fetching: {label}")
        try:
            df = _fetch_monthly(slug)
            if df.empty:
                print(f"    WARNING: No data for {label}")
                continue
            df["article"] = label
            frames.append(df)
            print(f"    Got {len(df):,} monthly rows")
        except Exception as exc:
            print(f"    WARNING: {label} — {exc}")
        time.sleep(0.3)

    if not frames:
        print("  WARNING: No Wikipedia data collected — skipping")
        return

    combined = pd.concat(frames, ignore_index=True)
    # Index on date; keep article as column
    combined = combined.set_index("date").sort_index()
    save(combined, "alternative", "wikipedia_views_finance_1m.parquet")


def main() -> None:
    print("Fetching: Wikipedia finance page views")
    try:
        collect_wikipedia_views()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
