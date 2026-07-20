"""
OECD macro data collector.
Source: OECD SDMX REST API v2 — free, no API key required.
Covers: GDP growth (quarterly), CPI inflation (monthly).
Output: data/macro/OECD_<indicator>_<country>.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save, HISTORY_START

OECD_API = "https://sdmx.oecd.org/public/rest/data"

# GDP Growth — quarterly, percent change vs same quarter prior year
# Dataflow: OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1
# Key dims (13): FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION
#                .INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
GDP_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1"
GDP_COUNTRIES = (
    "AUS+AUT+BEL+CAN+CHE+CHL+CZE+DEU+DNK+ESP+EST+FIN+FRA+GBR"
    "+GRC+HUN+IRL+ISL+ISR+ITA+JPN+KOR+LTU+LUX+LVA+MEX+NLD+NOR"
    "+NZL+POL+PRT+SVK+SVN+SWE+TUR+USA+COL+CRI+IDN"
)
# Q.Y.<COUNTRIES>.S1.S1.B1GQ._Z._Z._Z.PC.L.GY.T0102
GDP_KEY = f"Q.Y.{GDP_COUNTRIES}.S1.S1.B1GQ._Z._Z._Z.PC.L.GY.T0102"

# CPI Inflation — monthly, percent change vs same month prior year
# Dataflow: OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0
# Key dims (8): REF_AREA.FREQ.METHODOLOGY.MEASURE.UNIT_MEASURE.EXPENDITURE.ADJUSTMENT.TRANSFORMATION
CPI_FLOW = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0"
CPI_COUNTRIES = (
    "AUS+AUT+BEL+CAN+CHE+CHL+CZE+DEU+DNK+ESP+EST+FIN+FRA+GBR"
    "+GRC+HUN+IRL+ISL+ISR+ITA+JPN+KOR+LTU+LUX+LVA+MEX+NLD+NOR"
    "+NZL+POL+PRT+SVK+SVN+SWE+TUR+USA+COL+CRI+IDN"
)
# <COUNTRIES>.M.N.CPI.PA._T.N.GY
CPI_KEY = f"{CPI_COUNTRIES}.M.N.CPI.PA._T.N.GY"


def _parse_sdmx_json(data: dict) -> pd.DataFrame:
    """
    Parse OECD SDMX-JSON response (dimensionAtObservation=AllDimensions).
    Returns DataFrame with columns: date, country, value.
    """
    structures = data.get("data", {}).get("structures") or data.get("structures", [])
    if not structures:
        return pd.DataFrame()
    st = structures[0]

    obs_dims = st["dimensions"].get("observation", [])
    if not obs_dims:
        return pd.DataFrame()

    # Build index: position -> list of value ids
    dim_values: list[list[str]] = [
        [v["id"] for v in dim.get("values", [])] for dim in obs_dims
    ]

    # Find REF_AREA and TIME_PERIOD positions
    ref_pos = next((d["keyPosition"] for d in obs_dims if d["id"] == "REF_AREA"), None)
    time_pos = next((d["keyPosition"] for d in obs_dims if d["id"] == "TIME_PERIOD"), None)
    if ref_pos is None or time_pos is None:
        return pd.DataFrame()

    ds = data["data"]["dataSets"][0]
    observations = ds.get("observations", {})

    rows = []
    for key_str, obs_val in observations.items():
        val = obs_val[0] if obs_val else None
        if val is None:
            continue
        parts = key_str.split(":")
        try:
            country_idx = int(parts[ref_pos])
            time_idx = int(parts[time_pos])
            country = dim_values[ref_pos][country_idx]
            date_str = dim_values[time_pos][time_idx]
            rows.append({"date": date_str, "country": country, "value": float(val)})
        except (IndexError, ValueError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.dropna(subset=["date"])


def fetch_oecd(flow: str, key: str, start: str = "") -> pd.DataFrame:
    """Fetch OECD data via SDMX REST API."""
    start = start or HISTORY_START or "1960-01-01"
    url = f"{OECD_API}/{flow}/{key}"
    params = {
        "startPeriod": start,
        "format": "jsondata",
        "dimensionAtObservation": "AllDimensions",
    }
    resp = fetch(url, params=params, timeout=120)
    return _parse_sdmx_json(resp.json())


def main() -> None:
    datasets = {
        "GDP_Growth_1q": (GDP_FLOW, GDP_KEY, ""),
        "CPI_Inflation_1m": (CPI_FLOW, CPI_KEY, ""),
    }

    for name, (flow, key, start) in datasets.items():
        print(f"Fetching OECD {name} ...")
        try:
            df = fetch_oecd(flow, key, start)
            if df.empty:
                print(f"  WARNING: no data for {name}")
                continue
            for country, grp in df.groupby("country"):
                grp = grp.set_index("date").sort_index()[["value"]]
                save(grp, "macro", f"OECD_{name}_{country}.parquet")
        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
