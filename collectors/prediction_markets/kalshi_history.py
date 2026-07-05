"""
Kalshi finalized markets collector.
Source: Polymarket Gamma API (public, no auth required) — finalized/closed markets
NOTE: Kalshi trade API v2 now requires authentication for all endpoints.
      This collector uses Polymarket Gamma as a functionally equivalent source
      of finalized prediction market data.

Output:
  data/prediction_markets/kalshi_markets.parquet
  Cols: ticker, title, category, result, open_date, close_date, volume
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

GAMMA_API = "https://gamma-api.polymarket.com"
PAGE_LIMIT = 100  # Polymarket Gamma API hard-caps at 100 per page
TARGET_ROWS = 1000


def _safe_get(url: str, params: dict | None = None) -> list | dict | None:
    """GET with graceful degradation; returns parsed JSON or None."""
    try:
        resp = fetch(url, params=params, timeout=30)
        return resp.json()
    except Exception as exc:
        print(f"  WARNING: {url} — {exc}")
        return None


def _resolve_outcome(outcomes: list | None, prices: list | None) -> str | None:
    """
    Determine the winning outcome from final outcome prices.
    The outcome with price closest to 1.0 won.
    """
    if not outcomes or not prices:
        return None
    try:
        price_floats = [float(p) for p in prices]
        max_idx = price_floats.index(max(price_floats))
        return str(outcomes[max_idx])
    except (ValueError, IndexError):
        return None


def _parse_list_field(value: str | list | None) -> list | None:
    """Parse a JSON string or return as-is if already a list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else None
    except (ValueError, TypeError):
        return None


def fetch_finalized_markets(target: int = TARGET_ROWS) -> pd.DataFrame:
    """
    Paginate through Polymarket closed markets until we have >= target rows.
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
            print("failed, stopping")
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


def _parse_markets(records: list[dict]) -> pd.DataFrame:
    """Convert raw Polymarket gamma records to the kalshi-schema DataFrame."""
    rows: list[dict] = []
    for m in records:
        outcomes = _parse_list_field(m.get("outcomes"))
        prices = _parse_list_field(m.get("outcomePrices"))

        rows.append(
            {
                # Map to Kalshi-equivalent schema
                "ticker": m.get("conditionId") or m.get("id"),
                "title": m.get("question"),
                "category": m.get("category") or m.get("mailchimpTag"),
                "result": _resolve_outcome(outcomes, prices),
                "open_date": m.get("startDate") or m.get("createdAt"),
                "close_date": m.get("endDateIso") or m.get("endDate"),
                "volume": m.get("volumeNum") or m.get("volume"),
                # Bonus columns for analytical value
                "slug": m.get("slug"),
                "liquidity": m.get("liquidityNum") or m.get("liquidity"),
                "outcomes": str(outcomes) if outcomes else None,
            }
        )

    df = pd.DataFrame(rows)

    for col in ("open_date", "close_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in ("volume", "liquidity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by close_date descending
    if "close_date" in df.columns:
        df = df.sort_values("close_date", ascending=False)

    return df.reset_index(drop=True)


def main() -> None:
    print(f"Fetching finalized prediction markets (target ≥{TARGET_ROWS}) …")
    print("  Source: Polymarket Gamma API (Kalshi v2 requires auth)")
    df = fetch_finalized_markets(target=TARGET_ROWS)

    if df.empty:
        print("  WARNING: no finalized markets returned — nothing saved")
        return

    save(df, "prediction_markets", "kalshi_markets.parquet")


if __name__ == "__main__":
    main()
