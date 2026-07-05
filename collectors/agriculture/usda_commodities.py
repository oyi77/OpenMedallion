"""
USDA agricultural commodity price collector via FRED and IMF series.
Source: FRED API (DEMO_KEY, no registration required).
Covers: wheat flour, corn PPI, corn IMF, wheat IMF, soybeans, cotton,
        sugar, coffee Arabica, cocoa.
Output: data/agriculture/usda_agri_prices_1m.parquet
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&api_key=DEMO_KEY"

# FRED series_id -> column name in output
SERIES: dict[str, str] = {
    "APU0000711211": "wheat_flour_retail_usd_per_lb",
    "PCU11121112":   "corn_ppi",
    "PMAIZMTUSDM":   "corn_imf_usd_per_mt",
    "PWHEAMTUSDM":   "wheat_imf_usd_per_mt",
    "PSOYBUSDQ":     "soybean_usd_per_mt",
    "PCOTTINDUSDM":  "cotton_usd_per_kg",
    "PSUGAISAUSDM":  "sugar_usd_per_kg",
    "PCOFFOTMUSDM":  "coffee_arabica_usd_per_kg",
    "PCOCOISAUSDM":  "cocoa_usd_per_kg",
}


def _fetch_series(series_id: str, col: str) -> pd.Series | None:
    """Fetch a single FRED series; return a named Series indexed by date."""
    url = FRED_CSV.format(id=series_id)
    resp = fetch(url, timeout=30)
    df = pd.read_csv(StringIO(resp.text))
    if df.shape[1] < 2:
        return None
    df.columns = ["date", col]
    df = df[df[col] != "."].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")[col].sort_index()


def collect_usda_agri_prices() -> None:
    """Fetch all FRED agricultural series and merge into one monthly file."""
    frames: list[pd.Series] = []
    for series_id, col in SERIES.items():
        print(f"  Fetching FRED {series_id} -> {col}")
        try:
            s = _fetch_series(series_id, col)
            if s is not None and not s.empty:
                frames.append(s)
                print(f"    {len(s)} observations")
            else:
                print(f"    WARNING: empty response")
        except Exception as exc:
            print(f"    WARNING: {series_id} — {exc}")

    if not frames:
        print("  ERROR: no series collected — skipping save")
        return

    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    save(df, "agriculture", "usda_agri_prices_1m.parquet")


def main() -> None:
    print("Fetching USDA agricultural commodity prices via FRED ...")
    collect_usda_agri_prices()


if __name__ == "__main__":
    main()
