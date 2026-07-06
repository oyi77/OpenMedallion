"""
Per-coin OHLCV history for top-50 coins via CoinGecko free API (no key).

Steps per coin:
  1. /coins/{id}/ohlc?vs_currency=usd&days=365  -> [[ts_ms, o, h, l, c], ...]
  2. /coins/{id}/market_chart?vs_currency=usd&days=365&interval=daily
     -> total_volumes: [[ts_ms, vol], ...]  merged by date

Output (one file per coin):
  data/crypto/COIN_{coin_id}_1d.parquet
  Columns: open, high, low, close, volume  (DatetimeIndex, UTC)

Rate limit: 30 req/min on free tier — 2.5s sleep between coins keeps well under limit.
"""
from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

LOG = logging.getLogger(__name__)

MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
OHLC_URL    = "https://api.coingecko.com/api/v3/coins/{id}/ohlc"
CHART_URL   = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

SLEEP_BETWEEN_COINS = 2.5   # seconds; 2 calls/coin → ~24 calls/min, safely under 30
SLEEP_BETWEEN_CALLS = 0.5   # between the OHLC and market_chart calls inside each coin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_top50() -> list[dict]:
    """Return top-50 coins by market cap from /coins/markets."""
    resp = fetch(MARKETS_URL, params={
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 50,
        "page": 1,
        "sparkline": "false",
    }, retries=3)
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected /coins/markets response: {type(data)}")
    return data


def _fetch_ohlc(coin_id: str) -> pd.DataFrame:
    """Fetch 365-day daily OHLC candles for *coin_id*.

    CoinGecko OHLC endpoint returns [[ts_ms, open, high, low, close], ...].
    For days=365, granularity is 4-hour candles; we resample to daily OHLC.
    """
    resp = fetch(
        OHLC_URL.format(id=coin_id),
        params={"vs_currency": "usd", "days": 365},
        retries=3,
    )
    rows = resp.json()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close"])
    df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df.index.name = "date"
    df = df.drop(columns="ts_ms")

    # Resample 4h -> 1D: open=first, high=max, low=min, close=last
    daily = df.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna(how="all")
    return daily


def _fetch_volume(coin_id: str) -> pd.Series:
    """Fetch daily volume series for *coin_id* from market_chart endpoint.

    Returns a Series indexed by UTC date (floored to day).
    """
    resp = fetch(
        CHART_URL.format(id=coin_id),
        params={"vs_currency": "usd", "days": 365, "interval": "daily"},
        retries=3,
    )
    data = resp.json()
    volumes = data.get("total_volumes", [])
    if not volumes:
        return pd.Series(dtype=float, name="volume")

    df_vol = pd.DataFrame(volumes, columns=["ts_ms", "volume"])
    df_vol.index = pd.to_datetime(df_vol["ts_ms"], unit="ms", utc=True).dt.floor("D")
    df_vol.index.name = "date"
    return df_vol["volume"].groupby(level=0).last()


def _build_coin_df(coin_id: str) -> pd.DataFrame:
    """Fetch OHLC + volume and merge into a single daily DataFrame."""
    ohlc = _fetch_ohlc(coin_id)
    time.sleep(SLEEP_BETWEEN_CALLS)
    vol = _fetch_volume(coin_id)

    if ohlc.empty:
        return pd.DataFrame()

    # Align index: floor OHLC index to day for clean merge
    ohlc.index = ohlc.index.floor("D")
    ohlc.index.name = "date"
    ohlc = ohlc.groupby(level=0).last()  # deduplicate same-day entries

    df = ohlc.copy()
    df["volume"] = vol.reindex(df.index)
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def collect_coin_ohlcv(skip_existing: bool = True) -> None:
    """Fetch OHLCV for top-50 coins and save one parquet per coin.

    *skip_existing* — when True (default) coins already saved are skipped so
    a re-run after a timeout only fetches the remaining coins.
    """
    from collectors.base import DATA_ROOT  # noqa: PLC0415

    print("Fetching top-50 coin list…")
    coins = _fetch_top50()
    print(f"  Got {len(coins)} coins")

    saved = 0
    skipped = 0
    failed: list[str] = []

    for i, coin in enumerate(coins, start=1):
        coin_id = coin.get("id", "")
        symbol  = coin.get("symbol", "").upper()
        if not coin_id:
            continue

        out_path = DATA_ROOT / "crypto" / f"COIN_{coin_id}_1d.parquet"
        if skip_existing and out_path.exists():
            print(f"  [{i:02d}/{len(coins)}] {coin_id} ({symbol})… SKIP (already exists)")
            skipped += 1
            continue

        print(f"  [{i:02d}/{len(coins)}] {coin_id} ({symbol})…", end=" ", flush=True)
        try:
            df = _build_coin_df(coin_id)
            if df.empty:
                print("EMPTY — skipped")
                failed.append(coin_id)
            else:
                save(df, "crypto", f"COIN_{coin_id}_1d.parquet")
                saved += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            LOG.warning("Failed %s: %s", coin_id, exc)
            failed.append(coin_id)

        # Rate-limit gap between coins (already slept SLEEP_BETWEEN_CALLS inside)
        remaining = SLEEP_BETWEEN_COINS - SLEEP_BETWEEN_CALLS
        if remaining > 0:
            time.sleep(remaining)

    total_on_disk = len(list((DATA_ROOT / "crypto").glob("COIN_*_1d.parquet")))
    print(f"\nDone. Saved {saved} new files this run, skipped {skipped} existing.")
    print(f"Total COIN_*.parquet on disk: {total_on_disk}")
    if failed:
        print(f"Skipped/failed ({len(failed)}): {', '.join(failed)}")


def main() -> None:
    print("=== CoinGecko Per-Coin OHLCV (top 50, 365d) ===")
    collect_coin_ohlcv()


if __name__ == "__main__":
    main()
