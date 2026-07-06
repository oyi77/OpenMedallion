"""
FRED agricultural commodity price series — one parquet file per series.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import io
import time
import urllib.request

import pandas as pd

from collectors.base import save

# FRED series ID → filename label
_SERIES = {
    "PRICENPQUSDM":   "rice_price",
    "PCOCOAUSDM":     "cocoa_price",
    "POILAPSPUSDM":   "olive_oil_price",
    "PPOILUSDM":      "pork_price",
    "PPORKUSDM":      "pork_futures",
    "PLOGSKUSDM":     "logs_price",
    "POILWTIUSDM":    "wti_oil_price",
    "PTEAUSDM":       "tea_price",
}

# Fallback IDs tried when the primary returns no data
_FALLBACKS = {
    "PCOCOAUSDM":  "PCOCOUSDM",
}


def _fetch_series(series_id: str, label: str) -> pd.DataFrame | None:
    """Fetch one FRED CSV series; return a single-column DataFrame or None."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
    except Exception as exc:
        print(f"    WARN [{series_id}]: {exc}")
        return None

    df = pd.read_csv(io.BytesIO(raw))
    if df.empty or df.shape[1] < 2:
        return None

    df.columns = ["date", label]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[label] = pd.to_numeric(df[label], errors="coerce")
    df = df.dropna(subset=["date", label]).set_index("date")
    df.index.name = "date"
    return df if not df.empty else None


def collect():
    """Fetch each FRED agricultural series and save as its own parquet file."""
    created = 0

    for series_id, label in _SERIES.items():
        print(f"  {label} ({series_id})...")
        df = _fetch_series(series_id, label)

        # Try fallback ID if primary returned nothing
        if df is None and series_id in _FALLBACKS:
            fallback_id = _FALLBACKS[series_id]
            print(f"    retrying with fallback {fallback_id}...")
            df = _fetch_series(fallback_id, label)

        if df is None:
            print(f"    SKIP: no data for {label}")
        else:
            filename = f"AGRI_{label}_1m.parquet"
            out = save(df, "agriculture", filename)
            print(f"    saved {filename}: {len(df)} rows")
            created += 1

        time.sleep(0.3)

    print(f"\nCreated {created} parquet files")


if __name__ == "__main__":
    collect()
