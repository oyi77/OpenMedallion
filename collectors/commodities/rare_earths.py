"""
Critical minerals and industrial metals price collector via FRED.
Source: FRED CSV endpoint (no API key required).
Covers: nickel, aluminum, copper, tin, zinc, lead, iron ore, coal
        (IMF commodity price series published monthly on FRED).
        Cobalt and lithium are not available on FRED CSV; skipped gracefully.
Output: data/commodities/critical_minerals_1m.parquet
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

# FRED series_id -> output column name
# Only series confirmed to return HTTP 200 from the no-key CSV endpoint
SERIES: dict[str, str] = {
    "PNICKUSDM":   "nickel_usd_per_mt",        # IMF nickel monthly
    "PALUMUSDM":   "aluminum_usd_per_mt",       # IMF aluminum monthly
    "PCOPPUSDM":   "copper_usd_per_mt",         # IMF copper monthly
    "PTINUSDM":    "tin_usd_per_mt",            # IMF tin monthly
    "PZINCUSDM":   "zinc_usd_per_mt",           # IMF zinc monthly
    "PLEADUSDM":   "lead_usd_per_mt",           # IMF lead monthly
    "PIORECRUSDM": "iron_ore_usd_per_dmt",      # IMF iron ore monthly
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
