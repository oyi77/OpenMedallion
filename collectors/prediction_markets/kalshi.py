"""
Kalshi prediction markets collector.
Source: Kalshi public trade API v2 (no auth required for market data).
Base: https://trading-api.kalshi.com/trade-api/v2

Endpoints used:
  GET /markets?limit=200&status=closed  — closed markets snapshot
  GET /markets/{ticker}/candlesticks?period_interval=60 — hourly OHLCV

Output:
  data/prediction_markets/Kalshi_markets_snapshot.parquet
  data/prediction_markets/Kalshi_{ticker}_1h.parquet  (top 50 by volume)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"

# Columns to keep from the markets list; degrade if any are missing.
MARKET_COLS = [
    "ticker",
    "title",
    "category",
    "close_time",
    "result",
    "yes_bid",
    "no_bid",
    "volume",
    "open_interest",
]

# How many top markets (by volume) to fetch candlesticks for.
TOP_N_CANDLESTICKS = 50


def _safe_get(url: str, params: dict | None = None) -> dict[str, Any] | None:
    """
    GET with graceful degradation:
    - 401/403 → print WARNING, return None immediately (no retries).
    - 429     → sleep 5s, retry up to 3 times.
    - Other errors → propagate to caller.
    Returns parsed JSON dict or None.
    """
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"  WARNING: {url} — {exc}, skipping endpoint.")
            return None

        if resp.status_code in (401, 403):
            print(f"  WARNING: {url} — HTTP {resp.status_code} (no auth), skipping endpoint.")
            return None

        if resp.status_code == 429:
            if attempt == 3:
                print(f"  WARNING: {url} — HTTP 429 after {attempt} attempts, skipping endpoint.")
                return None
            print(f"  WARNING: {url} — HTTP 429, sleeping 5s (attempt {attempt}/3) ...")
            time.sleep(5)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"  WARNING: {url} — {exc}, skipping endpoint.")
            return None

        return resp.json()

    return None


def fetch_closed_markets(limit: int = 200) -> pd.DataFrame:
    """
    Fetch closed markets snapshot from Kalshi.
    Returns whatever columns are available; degrades gracefully.
    """
    url = f"{KALSHI_BASE}/markets"
    params = {"limit": limit, "status": "closed"}
    payload = _safe_get(url, params=params)
    if payload is None:
        return pd.DataFrame()

    # Kalshi wraps results under "markets" key
    records: list[dict] = payload.get("markets", [])
    if not records:
        print("  WARNING: Kalshi /markets returned 0 records.")
        return pd.DataFrame()

    df = pd.json_normalize(records)

    # Keep only available columns from desired set
    available = [c for c in MARKET_COLS if c in df.columns]
    df = df[available].copy()

    # Coerce numeric columns
    for col in ("yes_bid", "no_bid", "volume", "open_interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce datetime
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")

    return df


def fetch_candlesticks(ticker: str, period_interval: int = 60) -> pd.DataFrame:
    """
    Fetch hourly OHLCV candlesticks for a single market ticker.
    Returns empty DataFrame on any failure.
    """
    url = f"{KALSHI_BASE}/markets/{ticker}/candlesticks"
    params = {"period_interval": period_interval}
    payload = _safe_get(url, params=params)
    if payload is None:
        return pd.DataFrame()

    candles: list[dict] = payload.get("candlesticks", [])
    if not candles:
        return pd.DataFrame()

    df = pd.json_normalize(candles)

    # Normalise timestamp column — Kalshi may use "end_period_ts" or "ts"
    ts_col = next((c for c in ("end_period_ts", "ts", "timestamp") if c in df.columns), None)
    if ts_col:
        df["datetime"] = pd.to_datetime(df[ts_col], unit="s", utc=True, errors="coerce")
        df = df.drop(columns=[ts_col])

    # Coerce numeric OHLCV columns if present
    for col in ("open", "high", "low", "close", "volume", "open_interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    # ── 1. Closed markets snapshot ────────────────────────────────────────────
    print("Fetching Kalshi closed markets snapshot ...")
    df_markets = fetch_closed_markets(limit=200)
    save(df_markets, "prediction_markets", "Kalshi_markets_snapshot.parquet")

    if df_markets.empty:
        print("  No markets data — skipping candlestick collection.")
        return

    # ── 2. Top N markets by volume → hourly candlesticks ─────────────────────
    if "volume" not in df_markets.columns or "ticker" not in df_markets.columns:
        print("  WARNING: volume/ticker columns missing — skipping candlesticks.")
        return

    top = (
        df_markets.dropna(subset=["volume", "ticker"])
        .nlargest(TOP_N_CANDLESTICKS, "volume")["ticker"]
        .tolist()
    )
    print(f"Fetching hourly candlesticks for top {len(top)} markets by volume ...")

    for ticker in top:
        # Sanitise ticker for use as a filename component
        safe_ticker = ticker.replace("/", "_").replace(" ", "_")
        print(f"  {ticker} ...", end=" ", flush=True)
        try:
            df_candles = fetch_candlesticks(ticker)
            if df_candles.empty:
                print("0 rows, skipping.")
                continue
            save(df_candles, "prediction_markets", f"Kalshi_{safe_ticker}_1h.parquet")
        except Exception as exc:
            print(f"\n  WARNING: {ticker} — {exc}")
        # Polite pacing: avoid hammering the API
        time.sleep(0.2)


if __name__ == "__main__":
    main()
