"""
Google Trends — finance keyword interest over time.
Source: pytrends (unofficial Google Trends API, no key required)
Output: data/sentiment/google_trends_finance.parquet
Columns: date (index), one column per keyword (interest 0–100)

Note: pytrends is rate-limited — 10s sleep between keyword batches.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

_KEYWORDS = [
    "stock market",
    "bitcoin",
    "gold",
    "recession",
    "inflation",
    "interest rates",
    "oil price",
    "dollar",
    "crypto",
    "S&P 500",
]

_SLEEP_BETWEEN = 12  # seconds — stays under Google's soft rate limit
_BATCH_SIZE = 5      # pytrends accepts up to 5 keywords per request


def _fetch_trends_batch(kws: list[str], start_year: int = 2010) -> pd.DataFrame:
    """Return weekly interest-over-time DataFrame for up to 5 keywords."""
    from pytrends.request import TrendReq  # type: ignore[import]

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=3, backoff_factor=1.5)
    timeframe = f"{start_year}-01-01 {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    pytrends.build_payload(kws, cat=0, timeframe=timeframe, geo="", gprop="")
    df = pytrends.interest_over_time()
    if df.empty:
        return df
    # Drop the "isPartial" column added by pytrends
    df = df.drop(columns=["isPartial"], errors="ignore")
    return df


def collect_google_trends() -> None:
    """Fetch all finance keywords in batches and save combined parquet."""
    batches: list[pd.DataFrame] = []
    keywords_chunked = [_KEYWORDS[i : i + _BATCH_SIZE] for i in range(0, len(_KEYWORDS), _BATCH_SIZE)]

    for idx, batch in enumerate(keywords_chunked):
        print(f"  Fetching batch {idx + 1}/{len(keywords_chunked)}: {batch}")
        try:
            df_batch = _fetch_trends_batch(batch)
            if not df_batch.empty:
                batches.append(df_batch)
        except Exception as exc:
            print(f"  WARNING batch {batch}: {exc}")
        if idx < len(keywords_chunked) - 1:
            time.sleep(_SLEEP_BETWEEN)

    if not batches:
        print("  WARNING: No trends data retrieved — all batches failed")
        return

    # Merge all batches on the shared date index
    combined = batches[0]
    for extra in batches[1:]:
        combined = combined.join(extra, how="outer")

    combined.index.name = "date"
    combined = to_datetime_index(combined)
    save(combined, "sentiment", "google_trends_finance.parquet")


def main() -> None:
    print("Fetching: Google Trends finance keywords (weekly, 2010–now)")
    try:
        collect_google_trends()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
