"""
Manifold Markets collector.
Source: https://api.manifold.markets/v0 (public, no auth required)
Covers:
  1. Top 500 markets sorted by liquidity (fetched then sorted in-memory)
  2. Daily OHLC probability for top 30 resolved markets by volume
Output:
  data/prediction_markets/Manifold_markets_snapshot.parquet
  data/prediction_markets/Manifold_{safe_id}_prob_1d.parquet
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save, to_datetime_index

MANIFOLD_API = "https://api.manifold.markets/v0"

MARKET_KEEP_COLS = [
    "id",
    "question",
    "createdTime",
    "closeTime",
    "probability",
    "volume",
    "totalLiquidity",
    "isResolved",
    "resolution",
]


def _ms_to_utc(series: pd.Series) -> pd.Series:
    """Convert millisecond-epoch int series to UTC datetime, coercing out-of-range values to NaT."""
    return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")


def _safe_id(market_id: str) -> str:
    """Sanitise market ID for use in a filename."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", market_id)[:80]


def fetch_markets(limit: int = 500) -> pd.DataFrame:
    """
    Fetch markets from Manifold API.
    The API only supports time-based sort values; we fetch by last-bet-time
    (most recently active markets) and then sort by totalLiquidity in-memory.
    """
    params = {"limit": limit, "sort": "last-bet-time"}
    resp = fetch(f"{MANIFOLD_API}/markets", params=params)
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.json_normalize(data)

    # Keep only columns present in the response
    present = [c for c in MARKET_KEEP_COLS if c in df.columns]
    df = df[present].copy()

    for col in ("createdTime", "closeTime"):
        if col in df.columns:
            df[col] = _ms_to_utc(df[col])

    if "totalLiquidity" in df.columns:
        df["totalLiquidity"] = pd.to_numeric(df["totalLiquidity"], errors="coerce")
        df = df.sort_values("totalLiquidity", ascending=False)

    return df


def fetch_bets_ohlc(market_id: str) -> pd.DataFrame:
    """
    Fetch up to 1 000 bets for a market and pivot to daily OHLC of probability.
    Returns empty DataFrame on any error.
    """
    try:
        resp = fetch(
            f"{MANIFOLD_API}/bets",
            params={"contractId": market_id, "limit": 1000},
            timeout=30,
        )
        bets = resp.json()
        if not bets or not isinstance(bets, list):
            return pd.DataFrame()

        df = pd.json_normalize(bets)

        # probAfter is the probability after each bet; fall back to probBefore
        if "probAfter" in df.columns:
            prob_col = "probAfter"
        elif "probBefore" in df.columns:
            prob_col = "probBefore"
        else:
            return pd.DataFrame()

        if "createdTime" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["createdTime"], unit="ms", utc=True, errors="coerce").dt.normalize()
        df["prob"] = pd.to_numeric(df[prob_col], errors="coerce")
        df = df.dropna(subset=["prob"])

        if df.empty:
            return pd.DataFrame()

        ohlc = (
            df.groupby("date")["prob"]
            .agg(open="first", high="max", low="min", close="last")
            .reset_index()
        )
        ohlc = to_datetime_index(ohlc, col="date")
        return ohlc

    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: bets fetch for {market_id} failed — {exc}")
        return pd.DataFrame()


def main() -> None:
    # ── Step 1: market snapshot ───────────────────────────────────────────────
    print("Fetching Manifold Markets snapshot (top 500 by liquidity) …")
    try:
        markets_df = fetch_markets(limit=500)
    except Exception as exc:
        print(f"  WARNING: failed to fetch markets — {exc}")
        markets_df = pd.DataFrame()

    if markets_df.empty:
        print("  WARNING: no markets returned — skipping snapshot")
    else:
        save(markets_df, "prediction_markets", "Manifold_markets_snapshot.parquet")

    # ── Step 2: daily OHLC for top 30 resolved markets by volume ─────────────
    if markets_df.empty:
        return

    resolved_mask = markets_df.get("isResolved", pd.Series(dtype=bool)) == True  # noqa: E712
    resolved_df = markets_df[resolved_mask].copy()

    if "volume" in resolved_df.columns:
        resolved_df["volume"] = pd.to_numeric(resolved_df["volume"], errors="coerce")
        top30 = resolved_df.nlargest(30, "volume")
    else:
        top30 = resolved_df.head(30)

    print(f"Fetching daily probability OHLC for {len(top30)} resolved markets …")
    for _, row in top30.iterrows():
        market_id = str(row["id"])
        question_preview = str(row.get("question", market_id))[:60]
        print(f"  {market_id}: {question_preview}")

        ohlc_df = fetch_bets_ohlc(market_id)
        if ohlc_df.empty:
            print(f"  WARNING: no bet data for {market_id} — skipping")
            continue

        filename = f"Manifold_{_safe_id(market_id)}_prob_1d.parquet"
        save(ohlc_df, "prediction_markets", filename)


if __name__ == "__main__":
    main()
