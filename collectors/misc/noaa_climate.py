"""
NOAA global land+ocean temperature anomaly collector.
Source: NOAA NCEI Climate at a Glance — free, no API key required.
Output: data/misc/noaa_global_temperature.parquet
Columns: date (DatetimeIndex), anomaly_celsius
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

NOAA_URL = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
    "global/time-series/globe/land_ocean/12/0/1850-2025.json"
)


def main() -> None:
    """Fetch monthly global temperature anomalies and save as parquet."""
    resp = fetch(NOAA_URL)
    payload = resp.json()

    data = payload.get("data")
    if not data:
        print("WARNING: NOAA climate — empty response, skipping")
        return

    # Build rows: keys are YYYYMM, values have 'departure' in degrees C
    rows = []
    for yyyymm, entry in data.items():
        departure = entry.get("departure")
        if departure is None:
            continue
        rows.append({"date": yyyymm, "anomaly_celsius": departure})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m", utc=True)
    df = df.set_index("date").sort_index()
    df["anomaly_celsius"] = pd.to_numeric(df["anomaly_celsius"], errors="coerce")
    df = df.dropna()

    save(df, "misc", "noaa_global_temperature.parquet")


if __name__ == "__main__":
    main()
