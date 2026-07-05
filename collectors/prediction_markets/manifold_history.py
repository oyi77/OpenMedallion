"""
Manifold Markets history collector.
Source: https://api.manifold.markets/v0 (public, no auth required)

Covers:
  1. Top 1000 markets by last-bet-time — metadata snapshot
  2. Daily probability snapshots for top 200 resolved markets (via /bets endpoint)

Output:
  data/prediction_markets/manifold_markets_metadata.parquet
  data/prediction_markets/manifold_prob_history.parquet
  Cols (history): market_id, question, date, probability
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

MANIFOLD_API = "https://api.manifold.markets/v0"

# Valid sort values: created-time | updated-time | last-bet-time | last-comment-time
_SORT = "last-bet-time"

METADATA_COLS = [
    "id",
    "question",
    "createdTime",
    "closeTime",
    "resolution",
    "probability",
    "volume",
    "totalLiquidity",
    "isResolved",
    "outcomeType",
]


def _ms_to_utc(series: pd.Series) -> pd.Series:
    """Convert millisecond-epoch int series to UTC datetime."""
    return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")


def fetch_top_markets(limit: int = 1000) -> pd.DataFrame:
    """Fetch markets from Manifold API sorted by last-bet-time."""
    try:
        resp = fetch(
            f"{MANIFOLD_API}/markets",
            params={"limit": limit, "sort": _SORT},
            timeout=30,
        )
        data = resp.json()
    except Exception as exc:
        print(f"  WARNING: failed to fetch markets list — {exc}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.json_normalize(data)
    present = [c for c in METADATA_COLS if c in df.columns]
    df = df[present].copy()

    for col in ("createdTime", "closeTime"):
        if col in df.columns:
            df[col] = _ms_to_utc(df[col])

    for col in ("probability", "volume", "totalLiquidity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_prob_history_via_bets(market_id: str) -> pd.DataFrame:
    """
    Fetch daily probability snapshots for a market via the /bets endpoint.
    Returns tidy DataFrame with cols: market_id, question, date, probability.
    Empty DataFrame on any failure.
    """
    try:
        resp = fetch(
            f"{MANIFOLD_API}/bets",
            params={"contractId": market_id, "limit": 1000},
            timeout=30,
        )
        bets = resp.json()
    except Exception as exc:
        print(f"  WARNING: bets fetch for {market_id} — {exc}")
        return pd.DataFrame()

    if not bets or not isinstance(bets, list):
        return pd.DataFrame()

    df = pd.json_normalize(bets)

    # probAfter = probability after the bet; fall back to probBefore
    prob_col = next((c for c in ("probAfter", "probBefore") if c in df.columns), None)
    if prob_col is None or "createdTime" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["createdTime"], unit="ms", utc=True, errors="coerce").dt.normalize()
    df["probability"] = pd.to_numeric(df[prob_col], errors="coerce")
    df = df.dropna(subset=["probability", "date"])

    if df.empty:
        return pd.DataFrame()

    # Collapse to daily: take the last (close) probability per day
    daily = df.groupby("date")["probability"].last().reset_index()
    return daily


def collect_metadata(markets_df: pd.DataFrame) -> None:
    """Save market metadata snapshot."""
    if markets_df.empty:
        print("  WARNING: no markets data — skipping metadata save")
        return
    save(markets_df, "prediction_markets", "manifold_markets_metadata.parquet")


def collect_prob_history(markets_df: pd.DataFrame, top_n: int = 200) -> None:
    """
    Fetch daily probability history for top N resolved markets and save as a
    single tidy parquet: market_id, question, date, probability.
    """
    if markets_df.empty:
        print("  WARNING: no markets data — skipping history collection")
        return

    resolved_mask = markets_df.get("isResolved", pd.Series(dtype=object)) == True  # noqa: E712
    resolved = markets_df[resolved_mask].copy()

    if resolved.empty:
        print("  WARNING: no resolved markets found in fetched batch — skipping history")
        return

    if "volume" in resolved.columns:
        resolved["volume"] = pd.to_numeric(resolved["volume"], errors="coerce")
        top = resolved.nlargest(top_n, "volume")
    else:
        top = resolved.head(top_n)

    print(f"Fetching probability history for {len(top)} resolved markets …")

    all_rows: list[pd.DataFrame] = []
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        market_id = str(row["id"])
        question = str(row.get("question", market_id))
        print(f"  [{i}/{len(top)}] {market_id}: {question[:55]}", end=" … ", flush=True)

        daily = fetch_prob_history_via_bets(market_id)
        if daily.empty:
            print("no data, skipping")
            time.sleep(0.1)
            continue

        daily.insert(0, "market_id", market_id)
        daily.insert(1, "question", question)
        all_rows.append(daily)
        print(f"{len(daily)} daily rows")
        time.sleep(0.15)

    if not all_rows:
        print("  WARNING: no probability history rows collected — skipping save")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined = to_datetime_index(combined, col="date")
    save(combined, "prediction_markets", "manifold_prob_history.parquet")


def main() -> None:
    # ── 1. Market metadata snapshot ───────────────────────────────────────────
    print(f"Fetching Manifold Markets top-1000 (sort={_SORT}) …")
    markets_df = fetch_top_markets(limit=1000)
    collect_metadata(markets_df)

    # ── 2. Daily probability history for top 200 resolved markets ─────────────
    collect_prob_history(markets_df, top_n=200)


if __name__ == "__main__":
    main()
