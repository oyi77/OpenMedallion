"""
OECD Composite Leading Indicators (CLI) collector.
Source: OECD SDMX REST API v2 — free, no key required.
Covers: CLI amplitude-adjusted (ADJUSTMENT=AA, MEASURE=LI) for OECD members, monthly.
Output: data/macro/oecd_cli_1m.parquet

DF_CLI,4.0 key dimensions (10 total, pos 0-9):
  0  REF_AREA      — country ISO3
  1  FREQ          — M (monthly)
  2  MEASURE       — LI (leading indicator)
  3  UNIT_MEASURE  — IX (index)
  4  ACTIVITY      — _T (total)
  5  ADJUSTMENT    — AA (amplitude-adjusted)
  6  TRANSFORMATION— IX (index level)
  7  TIME_HORIZ    — _Z (not applicable)
  8  METHODOLOGY   — N (normalised)
  9  TIME_PERIOD   — auto
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

OECD_API = "https://sdmx.oecd.org/public/rest/data"
CLI_FLOW = "OECD.SDD.STES,DSD_STES@DF_CLI,4.0"

# OECD members + key non-members tracked in CLI
CLI_COUNTRIES = (
    "AUS+AUT+BEL+BRA+CAN+CHE+CHL+CHN+COL+CRI+CZE+DEU+DNK+"
    "ESP+EST+FIN+FRA+GBR+GRC+HUN+IDN+IND+IRL+ISL+ISR+ITA+"
    "JPN+KOR+LTU+LUX+LVA+MEX+NLD+NOR+NZL+POL+PRT+RUS+SVK+"
    "SVN+SWE+TUR+USA+ZAF+EA19+G20+OECD"
)

# Key: REF_AREA.FREQ.MEASURE.UNIT_MEASURE.ACTIVITY.ADJUSTMENT.TRANSFORMATION.TIME_HORIZ.METHODOLOGY
# Amplitude-adjusted CLI — ACTIVITY=_Z, METHODOLOGY=H (confirmed from live data)
CLI_KEY = f"{CLI_COUNTRIES}.M.LI.IX._Z.AA.IX._Z.H"


def _parse_sdmx_json(data: dict) -> pd.DataFrame:
    """Parse OECD SDMX-JSON (dimensionAtObservation=AllDimensions)."""
    structures = data.get("data", {}).get("structures") or data.get("structures", [])
    if not structures:
        return pd.DataFrame()
    st = structures[0]

    obs_dims = st["dimensions"].get("observation", [])
    if not obs_dims:
        return pd.DataFrame()

    dim_values: list[list[str]] = [
        [v["id"] for v in dim.get("values", [])] for dim in obs_dims
    ]

    ref_pos = next(
        (d["keyPosition"] for d in obs_dims if d["id"] == "REF_AREA"),
        None,
    )
    time_pos = next(
        (d["keyPosition"] for d in obs_dims if d["id"] == "TIME_PERIOD"),
        None,
    )
    if ref_pos is None or time_pos is None:
        return pd.DataFrame()

    ds = data["data"]["dataSets"][0]
    observations = ds.get("observations", {})

    rows: list[dict] = []
    for key_str, obs_val in observations.items():
        val = obs_val[0] if obs_val else None
        if val is None:
            continue
        parts = key_str.split(":")
        try:
            country = dim_values[ref_pos][int(parts[ref_pos])]
            date_str = dim_values[time_pos][int(parts[time_pos])]
            rows.append({"date": date_str, "country": country, "cli_value": float(val)})
        except (IndexError, ValueError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.dropna(subset=["date"])


def fetch_cli() -> pd.DataFrame:
    """Fetch CLI amplitude-adjusted via SDMX 2.1 endpoint."""
    url = f"{OECD_API}/{CLI_FLOW}/{CLI_KEY}"
    params = {
        "startPeriod": "2000-01",
        "format": "jsondata",
        "dimensionAtObservation": "AllDimensions",
    }
    resp = fetch(url, params=params, timeout=120)
    return _parse_sdmx_json(resp.json())


def main() -> None:
    print("Fetching OECD Composite Leading Indicators ...")
    try:
        df = fetch_cli()
    except Exception as exc:
        print(f"  WARNING: OECD CLI fetch failed — {exc}")
        return

    if df.empty:
        print("  WARNING: no data returned — skipping save")
        return

    print(f"  {len(df):,} rows")
    pivot = (
        df.pivot_table(index="date", columns="country", values="cli_value", aggfunc="first")
        .sort_index()
    )
    save(pivot, "macro", "oecd_cli_1m.parquet")


if __name__ == "__main__":
    main()
