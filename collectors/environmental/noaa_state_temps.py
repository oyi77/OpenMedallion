"""
NOAA NCEI Climate at a Glance — statewide annual average temperature.

Fetches 30 US states, saves each as a separate parquet file.
URL: https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/{state_id}/tavg/12/12/1895-2025.csv
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from collectors.base import fetch, save

# State IDs 1-50; subset of 30 spread across regions
# NOAA state IDs: https://www.ncei.noaa.gov/access/monitoring/reference-maps/us-regions
STATES: list[tuple[int, str]] = [
    (1,  "alabama"),
    (2,  "arizona"),
    (3,  "arkansas"),
    (4,  "california"),
    (5,  "colorado"),
    (6,  "connecticut"),
    (7,  "delaware"),
    (8,  "florida"),
    (9,  "georgia"),
    (10, "idaho"),
    (11, "illinois"),
    (12, "indiana"),
    (13, "iowa"),
    (14, "kansas"),
    (15, "kentucky"),
    (16, "louisiana"),
    (17, "maine"),
    (18, "maryland"),
    (19, "massachusetts"),
    (20, "michigan"),
    (21, "minnesota"),
    (22, "mississippi"),
    (23, "missouri"),
    (24, "montana"),
    (25, "nebraska"),
    (26, "nevada"),
    (27, "new_hampshire"),
    (28, "new_jersey"),
    (29, "new_mexico"),
    (30, "new_york"),
]

_BASE = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/{sid}/tavg/12/12/1895-2025.csv"


def _parse_noaa_csv(raw: str, state_name: str) -> pd.DataFrame:
    """Parse NOAA Climate at a Glance CSV (header lines start with #)."""
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        raise ValueError("No data rows")
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    # Columns: Date (YYYYMM), Value
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df = df.rename(columns={"Value": "tavg"})
    df["tavg"] = pd.to_numeric(df["tavg"], errors="coerce")
    df = df.dropna(subset=["date", "tavg"]).set_index("date")[["tavg"]]
    df.index.name = "date"
    return df.sort_index()


def collect() -> None:
    saved = 0
    failed: list[str] = []

    for state_id, state_name in STATES:
        url = _BASE.format(sid=state_id)
        filename = f"NOAA_state{state_id:02d}_{state_name}_tavg_1m.parquet"
        print(f"  Fetching state {state_id:2d} ({state_name})...")
        try:
            resp = fetch(url, timeout=30)
            df = _parse_noaa_csv(resp.text, state_name)
            save(df, "environmental", filename)
            saved += 1
        except Exception as exc:
            print(f"    WARN: state {state_id} ({state_name}) failed — {exc}")
            failed.append(state_name)
        time.sleep(0.4)

    print(f"\nDone: {saved}/{len(STATES)} states saved.")
    if failed:
        print(f"  Failed: {', '.join(failed)}")


if __name__ == "__main__":
    collect()
