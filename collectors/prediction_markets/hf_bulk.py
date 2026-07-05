"""
Prediction market historical bulk collector — pulls from public HuggingFace datasets.

Sources (all free, no API key required):
  1. manja316/polymarket-historical-prices  — 15-min price snapshots, thousands of markets
  2. SII-WANGZJ/Polymarket_data             — 1.9B records on-chain trades (markets + trades subset)
  3. TimeSeventeen/Polymarket-v1            — normalized contract lifecycle 2022–2026

Strategy:
  - Download only the smallest useful subset of each large dataset (markets metadata + price history)
  - Skip if local file is already up-to-date (mtime < 24h)
  - Normalize to a common schema: market_id, question, category, resolution, yes_price, volume, ts

Output:
  data/prediction_markets/HF_Polymarket_prices_15m.parquet
  data/prediction_markets/HF_Polymarket_markets_meta.parquet
  data/prediction_markets/HF_Polymarket_trades_sample.parquet
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import save

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# HuggingFace dataset IDs
HF_PRICES = "manja316/polymarket-historical-prices"
HF_FULL = "SII-WANGZJ/Polymarket_data"
HF_V1 = "TimeSeventeen/Polymarket-v1"

CATEGORY = "prediction_markets"

# Max rows to pull from large datasets (avoid OOM on huge files)
MAX_ROWS_PRICES = 5_000_000
MAX_ROWS_TRADES = 500_000


def _hf_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        LOG.warning("huggingface_hub not installed — run: pip install huggingface_hub datasets")
        return False


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        LOG.warning("datasets not installed — run: pip install datasets")
        return False


def _stale(filename: str, max_age_hours: int = 24) -> bool:
    """Return True if file doesn't exist or is older than max_age_hours."""
    p = Path(__file__).resolve().parents[2] / "data" / CATEGORY / filename
    if not p.exists():
        return True
    age = time.time() - p.stat().st_mtime
    return age > max_age_hours * 3600


# ---------------------------------------------------------------------------
# Source 1: manja316/polymarket-historical-prices
# Small enough to download config-by-config; contains 15-min YES price history
# ---------------------------------------------------------------------------

def fetch_hf_prices() -> None:
    """Pull 15-min price snapshots from manja316/polymarket-historical-prices."""
    fname = "HF_Polymarket_prices_15m.parquet"
    if not _stale(fname):
        LOG.info("HF prices up-to-date, skipping")
        return

    if not _datasets_available():
        return

    try:
        from datasets import load_dataset
        LOG.info("Loading %s …", HF_PRICES)
        # streaming=True to avoid downloading full dataset first
        ds = load_dataset(HF_PRICES, split="train", streaming=True)

        rows = []
        for i, row in enumerate(ds):
            rows.append(row)
            if i >= MAX_ROWS_PRICES - 1:
                LOG.info("Reached MAX_ROWS_PRICES=%d — stopping stream", MAX_ROWS_PRICES)
                break
            if i % 500_000 == 0 and i > 0:
                LOG.info("  streamed %d rows …", i)

        if not rows:
            LOG.warning("No rows from %s", HF_PRICES)
            return

        df = pd.DataFrame(rows)
        # Normalize timestamp column
        ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower() or "ts" in c.lower()), None)
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.set_index(ts_col).sort_index()

        save(df, CATEGORY, fname)

    except Exception as exc:
        LOG.warning("fetch_hf_prices failed: %s", exc)


# ---------------------------------------------------------------------------
# Source 2: SII-WANGZJ/Polymarket_data — markets metadata subset
# Full dataset is enormous; we pull only the 'markets' config (metadata, not trades)
# ---------------------------------------------------------------------------

def fetch_hf_markets_meta() -> None:
    """Pull markets metadata from SII-WANGZJ/Polymarket_data (markets config only)."""
    fname = "HF_Polymarket_markets_meta.parquet"
    if not _stale(fname, max_age_hours=168):  # weekly — metadata doesn't change fast
        LOG.info("HF markets meta up-to-date, skipping")
        return

    if not _datasets_available():
        return

    try:
        from datasets import load_dataset, get_dataset_config_names

        # Discover available configs
        try:
            configs = get_dataset_config_names(HF_FULL)
            LOG.info("%s configs: %s", HF_FULL, configs)
        except Exception:
            configs = ["default"]

        # Prefer 'markets' or 'market_info' config; fall back to default
        target = next(
            (c for c in configs if "market" in c.lower() and "trade" not in c.lower()),
            configs[0] if configs else "default"
        )
        LOG.info("Loading config=%s from %s …", target, HF_FULL)

        ds = load_dataset(HF_FULL, target, split="train", streaming=True)

        rows = []
        for i, row in enumerate(ds):
            rows.append(row)
            if i >= 200_000:  # markets meta is manageable at 200K rows
                break

        if not rows:
            LOG.warning("No rows from %s config=%s", HF_FULL, target)
            return

        df = pd.DataFrame(rows)
        save(df, CATEGORY, fname)

    except Exception as exc:
        LOG.warning("fetch_hf_markets_meta failed: %s", exc)


# ---------------------------------------------------------------------------
# Source 3: TimeSeventeen/Polymarket-v1 — normalized lifecycle 2022–2026
# ---------------------------------------------------------------------------

def fetch_hf_v1() -> None:
    """Pull normalized Polymarket data from TimeSeventeen/Polymarket-v1."""
    fname = "HF_Polymarket_v1_lifecycle.parquet"
    if not _stale(fname, max_age_hours=168):
        LOG.info("HF v1 up-to-date, skipping")
        return

    if not _datasets_available():
        return

    try:
        from datasets import load_dataset
        LOG.info("Loading %s …", HF_V1)

        ds = load_dataset(HF_V1, split="train", streaming=True)

        rows = []
        for i, row in enumerate(ds):
            rows.append(row)
            if i >= 500_000:
                LOG.info("Reached 500K rows from %s — stopping", HF_V1)
                break
            if i % 100_000 == 0 and i > 0:
                LOG.info("  streamed %d rows …", i)

        if not rows:
            LOG.warning("No rows from %s", HF_V1)
            return

        df = pd.DataFrame(rows)

        # Normalize date/time columns
        for col in df.columns:
            if any(k in col.lower() for k in ("time", "date", "created", "closed", "resolved")):
                try:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                except Exception:
                    pass

        save(df, CATEGORY, fname)

    except Exception as exc:
        LOG.warning("fetch_hf_v1 failed: %s", exc)


# ---------------------------------------------------------------------------
# Bonus: Kalshi community dump via analisto/kalshi_com GitHub releases
# They publish CSV snapshots; we pull the latest available
# ---------------------------------------------------------------------------

def fetch_kalshi_community_dump() -> None:
    """Pull Kalshi historical snapshot from analisto/kalshi_com GitHub releases."""
    fname = "Kalshi_community_markets_snapshot.parquet"
    if not _stale(fname, max_age_hours=168):
        LOG.info("Kalshi community dump up-to-date, skipping")
        return

    import requests

    # Check latest release from the GitHub repo
    api_url = "https://api.github.com/repos/analisto/kalshi_com/releases/latest"
    try:
        resp = requests.get(api_url, timeout=15, headers={"Accept": "application/vnd.github+json"})
        if resp.status_code == 404:
            LOG.warning("analisto/kalshi_com has no releases — trying raw CSV fallback")
            _fetch_kalshi_csv_fallback(fname)
            return
        resp.raise_for_status()
        release = resp.json()
        assets = release.get("assets", [])
        csv_assets = [a for a in assets if a["name"].endswith(".csv")]
        if not csv_assets:
            LOG.warning("No CSV assets in latest release — skipping Kalshi community dump")
            return

        # Download the first (or largest) CSV
        asset = max(csv_assets, key=lambda a: a.get("size", 0))
        LOG.info("Downloading %s (%d bytes) …", asset["name"], asset.get("size", 0))
        data_resp = requests.get(asset["browser_download_url"], timeout=120)
        data_resp.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(data_resp.text))
        save(df, CATEGORY, fname)

    except Exception as exc:
        LOG.warning("fetch_kalshi_community_dump failed: %s", exc)


def _fetch_kalshi_csv_fallback(fname: str) -> None:
    """Try direct Kalshi API historical endpoint as fallback."""
    import requests
    # Kalshi public API — no auth needed for market listing
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    params = {"limit": 1000, "status": "settled"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("markets", [])
        if not markets:
            LOG.warning("Kalshi API returned 0 settled markets")
            return
        df = pd.json_normalize(markets)
        save(df, CATEGORY, fname)
    except Exception as exc:
        LOG.warning("Kalshi CSV fallback also failed: %s", exc)


# ---------------------------------------------------------------------------

def main() -> None:
    if not _hf_available():
        LOG.error("Install huggingface_hub: pip install huggingface_hub datasets")
        return

    print("=== HF Bulk Prediction Market Historical Collector ===")
    print("Source 1: manja316/polymarket-historical-prices (15-min prices)")
    fetch_hf_prices()

    print("Source 2: SII-WANGZJ/Polymarket_data (markets metadata)")
    fetch_hf_markets_meta()

    print("Source 3: TimeSeventeen/Polymarket-v1 (lifecycle 2022-2026)")
    fetch_hf_v1()

    print("Source 4: analisto/kalshi_com (Kalshi community dump)")
    fetch_kalshi_community_dump()

    print("Done.")


if __name__ == "__main__":
    main()
