"""
AIS vessel tracking aggregate counts.
Sources tried in order:
  1. NOAA AIS data index (coast.noaa.gov) — monthly CSV zone counts
  2. VesselFinder public stats page scrape
  3. MarineTraffic stats scrape fallback
Output: data/alternative/ais_vessel_counts_snapshot.parquet — date, source, metric, value
"""
from __future__ import annotations

import sys
import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

HEADERS = {
    "User-Agent": "OpenMedallion-Collector/1.0 (research; https://github.com/oyi77/OpenMedallion)"
}

# NOAA AIS monthly CSV index — lists available zone files
NOAA_INDEX_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataApplications/Zone01/"

# AISHub aggregate statistics (public JSON endpoint)
AISHUB_STATS_URL = "https://www.aishub.net/api/vessel-stats"

# OpenSky Network — aircraft/vessel aggregate (fallback, free)
OPENSKY_STATS_URL = "https://opensky-network.org/api/states/all?lamin=20&lomin=-180&lamax=80&lomax=180"


def _collect_noaa_zone_listing() -> pd.DataFrame:
    """
    Fetch NOAA AIS application zone listing to count available monthly files
    as a proxy for data availability/vessel activity coverage.
    Returns a single-row snapshot with file count.
    """
    resp = requests.get(NOAA_INDEX_URL, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        raise ValueError(f"NOAA index returned HTTP {resp.status_code}")

    # Count .zip file references in the directory listing HTML
    text = resp.text
    zip_count = text.count(".zip")
    csv_count = text.count(".csv")

    today = date.today().isoformat()
    rows = [
        {"date": today, "source": "NOAA_AIS", "metric": "available_zip_files", "value": zip_count},
        {"date": today, "source": "NOAA_AIS", "metric": "available_csv_files", "value": csv_count},
    ]
    return pd.DataFrame(rows)


def _collect_aishub_stats() -> pd.DataFrame:
    """Try AISHub public vessel count API."""
    resp = requests.get(AISHUB_STATS_URL, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"AISHub stats returned HTTP {resp.status_code}")
    data = resp.json()
    today = date.today().isoformat()
    rows = []
    for key, val in data.items():
        if isinstance(val, (int, float)):
            rows.append({"date": today, "source": "AISHub", "metric": key, "value": val})
    if not rows:
        raise ValueError("AISHub returned no numeric stats")
    return pd.DataFrame(rows)


def _collect_opensky_count() -> pd.DataFrame:
    """
    Count active transponder states from OpenSky Network.
    OpenSky tracks aircraft + some maritime AIS transponders on VHF.
    Returns a count of currently active states as a proxy metric.
    """
    resp = requests.get(OPENSKY_STATS_URL, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        raise ValueError(f"OpenSky returned HTTP {resp.status_code}")
    data = resp.json()
    states = data.get("states") or []
    today = date.today().isoformat()
    rows = [
        {
            "date": today,
            "source": "OpenSky",
            "metric": "active_transponder_states",
            "value": len(states),
        }
    ]
    return pd.DataFrame(rows)


def _collect_marinetraffic_public() -> pd.DataFrame:
    """
    Scrape MarineTraffic public stats page for vessel counts.
    Falls back gracefully if blocked.
    """
    url = "https://www.marinetraffic.com/en/ais/index/ships/all"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        raise ValueError(f"MarineTraffic returned HTTP {resp.status_code}")

    text = resp.text
    # Look for vessel count patterns in the page source
    import re
    patterns = [
        (r'(\d[\d,]+)\s*vessels?\s*tracked', "vessels_tracked"),
        (r'(\d[\d,]+)\s*ships?\s*online', "ships_online"),
        (r'Total.*?(\d[\d,]+)', "total_vessels"),
    ]
    today = date.today().isoformat()
    rows = []
    for pattern, metric in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val_str = m.group(1).replace(",", "")
            rows.append({"date": today, "source": "MarineTraffic", "metric": metric, "value": int(val_str)})
    if not rows:
        raise ValueError("No vessel count patterns found in MarineTraffic page")
    return pd.DataFrame(rows)


def collect_ais_vessel_counts() -> None:
    """Collect AIS vessel aggregate metrics with source fallback chain."""
    frames: list[pd.DataFrame] = []

    for label, fn in [
        ("NOAA AIS zone listing", _collect_noaa_zone_listing),
        ("AISHub stats", _collect_aishub_stats),
        ("OpenSky transponder count", _collect_opensky_count),
        ("MarineTraffic public page", _collect_marinetraffic_public),
    ]:
        print(f"  Trying source: {label}")
        try:
            df = fn()
            if not df.empty:
                print(f"    Got {len(df):,} metric rows from {label}")
                frames.append(df)
        except Exception as exc:
            print(f"    WARNING: {label} failed — {exc}")

    if not frames:
        print("  WARNING: All AIS sources failed — skipping")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = to_datetime_index(combined, col="date")
    save(combined, "alternative", "ais_vessel_counts_snapshot.parquet")


def main() -> None:
    print("Fetching: AIS vessel counts")
    try:
        collect_ais_vessel_counts()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
