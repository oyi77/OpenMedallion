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
    "PWHEAMTUSDM":    "wheat_price",
    "PMAIZEUSDM":     "corn_price",
    "PRICENPQUSDM":   "rice_price",
    "PSOYBUSDM":      "soybean_price",
    "PSUNOUSDM":      "sunflower_price",
    "PPALMOLUSDM":    "palm_oil_price",
    "PCOCOAUSDM":     "cocoa_price",
    "PCOFFOTMUSDM":   "coffee_price",
    "PSUGAISAUSDM":   "sugar_price",
    "PBANSOPUSDM":    "banana_price",
    "PCOTTINDUSDM":   "cotton_price",
    "PRUBBINDM":      "rubber_price",
    "PBEEFUSDM":      "beef_price",
    "PPOULTUSDM":     "chicken_price",
    "PSALMUSDM":      "salmon_price",
    "PCOALAUUSDM":    "coal_price",
    "PFOODINDEXM":    "food_index",
    "POILAPSPUSDM":   "olive_oil_price",
    "WPU01":          "ppi_grains",
    "WPU013":         "ppi_livestock",
    "WPU015":         "ppi_dairy",
    "WPU017":         "ppi_oilseeds",
    "WPU021":         "ppi_grain_mill",
    "WPU022":         "ppi_processed_foods",
    "PPOILUSDM":      "pork_price",
    "PPORKUSDM":      "pork_futures",
    "PLOGSKUSDM":     "logs_price",
    "POILWTIUSDM":    "wti_oil_price",
    "PTEAUSDM":       "tea_price",
}

# Fallback IDs tried when the primary returns no data
_FALLBACKS = {
    "PCOCOAUSDM":  "PCOCOUSDM",
    "PRUBBINDM":   "PRUBBUSDM",
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
