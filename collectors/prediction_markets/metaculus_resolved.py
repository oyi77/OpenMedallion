"""
Metaculus resolved questions collector.
Source: Polymarket Gamma API (public, no auth required) — closed/resolved markets
NOTE: Metaculus API now requires authentication. This collector uses Polymarket
      as a functionally equivalent source of resolved prediction market questions.

Output:
  data/prediction_markets/metaculus_resolved.parquet
  Cols: question_id, title, resolution, resolution_date, community_prediction_at_close
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

GAMMA_API = "https://gamma-api.polymarket.com"
PAGE_LIMIT = 100
TARGET_ROWS = 500


def _safe_get(url: str, params: dict | None = None) -> list | dict | None:
    """GET with graceful degradation; returns parsed JSON or None."""
    try:
        resp = fetch(url, params=params, timeout=30)
        return resp.json()
    except Exception as exc:
        print(f"  WARNING: {url} — {exc}")
        return None


def fetch_resolved_markets(target: int = TARGET_ROWS) -> pd.DataFrame:
    """
    Paginate through Polymarket closed markets until we have >= target rows.
    Uses offset-based pagination.
    """
    records: list[dict] = []
    offset = 0
    page = 1

    while len(records) < target:
        params: dict = {
            "limit": PAGE_LIMIT,
            "offset": offset,
            "closed": "true",
            "order": "volumeNum",
            "ascending": "false",
        }
        print(f"  Page {page} ({len(records)} rows so far) …", end=" ", flush=True)
        data = _safe_get(f"{GAMMA_API}/markets", params=params)

        if data is None:
            print("failed, stopping pagination")
            break
        if not isinstance(data, list) or not data:
            print("empty, done")
            break

        records.extend(data)
        print(f"got {len(data)}")


        offset += PAGE_LIMIT
        page += 1
        time.sleep(0.25)

    if not records:
        return pd.DataFrame()

    return _parse_markets(records)


def _resolve_outcome(outcomes: list | None, prices: list | None) -> str | None:
    """
    Infer market resolution from outcome prices.
    Price of 1.0 = that outcome resolved YES/winning.
    """
    if not outcomes or not prices:
        return None
    try:
        price_floats = [float(p) for p in prices]
        max_idx = price_floats.index(max(price_floats))
        return str(outcomes[max_idx])
    except (ValueError, IndexError):
        return None


def _community_prediction(prices: list | None) -> float | None:
    """
    Extract the YES outcome's final price as the community prediction at close.
    For binary YES/NO markets this is the probability of YES.
    """
    if not prices:
        return None
    try:
        # First outcome is typically YES
        return float(prices[0])
    except (ValueError, IndexError):
        return None


def _parse_markets(records: list[dict]) -> pd.DataFrame:
    """Convert raw Polymarket gamma records to the resolved-questions schema."""
    rows: list[dict] = []
    for m in records:
        outcomes = m.get("outcomes")
        prices = m.get("outcomePrices")

        # Parse outcomes/prices from JSON string if needed
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = None
        if isinstance(prices, str):
            import json
            try:
                prices = json.loads(prices)
            except Exception:
                prices = None

        rows.append(
            {
                "question_id": m.get("conditionId") or m.get("id"),
                "title": m.get("question"),
                "resolution": _resolve_outcome(outcomes, prices),
                "resolution_date": m.get("endDateIso") or m.get("endDate"),
                "community_prediction_at_close": _community_prediction(prices),
                "volume": m.get("volumeNum") or m.get("volume"),
                "category": m.get("category"),
                "slug": m.get("slug"),
                "created_at": m.get("createdAt"),
                "closed_time": m.get("closedTime"),
            }
        )

    df = pd.DataFrame(rows)

    for col in ("resolution_date", "created_at", "closed_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in ("community_prediction_at_close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use resolution_date as the DatetimeIndex
    if "resolution_date" in df.columns and df["resolution_date"].notna().any():
        df = to_datetime_index(df, col="resolution_date")
    else:
        df = df.reset_index(drop=True)

    return df


def main() -> None:
    print(f"Fetching resolved prediction market questions (target ≥{TARGET_ROWS}) …")
    print("  Source: Polymarket Gamma API (Metaculus requires auth)")
    df = fetch_resolved_markets(target=TARGET_ROWS)

    if df.empty:
        print("  WARNING: no resolved questions returned — nothing saved")
        return

    save(df, "prediction_markets", "metaculus_resolved.parquet")


if __name__ == "__main__":
    main()
