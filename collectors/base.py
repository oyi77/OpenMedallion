"""Shared base utilities for all OpenMedallion collectors."""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Repo root is two levels up from this file (collectors/base.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"


def repo_path(category: str, filename: str) -> Path:
    """Return absolute output path, creating parent dirs as needed."""
    p = DATA_ROOT / category / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch(url: str, params: dict | None = None, retries: int = 3, timeout: int = 30) -> requests.Response:
    """GET with exponential backoff. Raises on final failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            LOG.warning("Attempt %d/%d failed (%s) — retrying in %ds", attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def save(df: pd.DataFrame, category: str, filename: str) -> None:
    """Save DataFrame to parquet. Warns and skips if empty."""
    if df.empty:
        LOG.warning("0 rows returned for %s/%s — skipping", category, filename)
        return
    out = repo_path(category, filename)
    df.to_parquet(out, index=True)
    rel = out.relative_to(REPO_ROOT)
    print(f"Saved {len(df):,} rows to data/{rel.as_posix().removeprefix('data/')}")


def to_datetime_index(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure col is a proper DatetimeIndex."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True)
        df = df.set_index(col).sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
    return df
