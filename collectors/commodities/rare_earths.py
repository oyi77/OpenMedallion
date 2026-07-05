"""
Critical minerals and rare earth price collector via FRED.
Source: FRED API (DEMO_KEY, no registration required).
Covers: cobalt, nickel, aluminum (from IMF commodity price series via FRED),
        lithium if available.
Output: data/commodities/critical_minerals_1m.parquet
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&api_key=DEMO_KEY"

# FRED series_id -> output column name
# All are monthly IMF commodity prices unless noted
SERIES: dict[str, str] = {
    "PCOBAUSDM":  "cobalt_usd_per_mt",
    "PNICKUSDM":  "nickel_usd_per_mt",
    "PALUMUSDM":  "aluminum_usd_per_mt",
    # Lithium carbonate — World Bank / IMF via FRED (added ~2019)
    "PLITHIUSDM": "lithium_usd_per_mt",
    # Extra critical minerals for completeness
    "PCOALUSDQ":  "coal_usd_per_mt",       # thermal coal quarterly
    "PIORECRUSDM": "iron_ore_usd_per_dmt",
    "PCOPPERUSDM": "copper_usd_per_mt",
    "PTINUSDM":   "tin_usd_per_mt",
    "PZINCUSDM":  "zinc_usd_per_mt",
    "PLEADUSDM":  "lead_usd_per_mt",
}


def _fetch_fred_series(series_id: str, col: str) -> pd.Series | None:
    """Fetch one FRED series; return a named Series indexed by UTC date."""
    url = FRED_CSV.format(id=series_id)
    try:
        resp = fetch(url, timeout=30)
    except Exception as exc:
        print(f"    WARNING: HTTP error for {series_id} — {exc}")
        return None
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


def collect_critical_minerals() -> None:
    """Fetch all critical mineral series from FRED and merge into one file."""
    frames: list[pd.Series] = []
    for series_id, col in SERIES.items():
        print(f"  Fetching FRED {series_id} -> {col}")
        try:
            s = _fetch_fred_series(series_id, col)
            if s is not None and not s.empty:
                frames.append(s)
                print(f"    {len(s)} observations")
            else:
                print(f"    WARNING: empty or unavailable — skipping {series_id}")
        except Exception as exc:
            print(f"    WARNING: {series_id} — {exc}")

    if not frames:
        print("  ERROR: no series collected — skipping save")
        return

    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    available = df.columns.tolist()
    print(f"  Columns saved: {available}")
    save(df, "commodities", "critical_minerals_1m.parquet")


def main() -> None:
    print("Fetching critical minerals / rare earth prices via FRED ...")
    collect_critical_minerals()


if __name__ == "__main__":
    main()
