"""
Crypto Fear & Greed Index — full history backfill.
Source: alternative.me public API (no key required)
Output: data/sentiment/fear_greed_history.parquet
Columns: date (index), value (float), rating (str)

Note: This is the crypto-focused Fear & Greed Index (Bitcoin/crypto market sentiment).
It covers 2018-present with daily granularity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

# limit=0 returns all available history
_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def collect_fear_greed_history() -> None:
    """Fetch full crypto Fear & Greed history and save to parquet."""
    resp = fetch(_URL, timeout=30)
    payload = resp.json()

    raw: list[dict] = payload["data"]
    records = [
        {
            "date": pd.to_datetime(int(item["timestamp"]), unit="s", utc=True),
            "value": float(item["value"]),
            "rating": item["value_classification"],
        }
        for item in raw
    ]

    df = pd.DataFrame(records)
    df = to_datetime_index(df, col="date")
    save(df, "sentiment", "fear_greed_history.parquet")


def main() -> None:
    print("Fetching: Crypto Fear & Greed full history (alternative.me)")
    try:
        collect_fear_greed_history()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
