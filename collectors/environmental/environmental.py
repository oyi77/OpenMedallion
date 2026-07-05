"""
Environmental / climate data collector — no API keys required.

Sources:
1. NOAA ESRL CO2 Mauna Loa weekly      → environmental/NOAA_CO2_MaunaLoa_1w.parquet
2. NOAA Global Temperature Anomaly 1m  → environmental/NOAA_GlobalTemp_Anomaly_1m.parquet
3. NASA GISS Surface Temperature annual → environmental/NASA_GISTEMP_GlobalTemp_1y.parquet
4. OWID CO2 Emissions annual           → environmental/OWID_CO2_Emissions_1y.parquet
5. NSIDC Arctic Sea Ice Extent monthly → environmental/NSIDC_ArcticSeaIce_1m.parquet
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

# NaN sentinels used across these datasets
_NAN_VALUES = [-999, -9999, -999.0, -9999.0, "-999", "-9999", "NaN", "nan", "NA", "N/A"]


def _clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common sentinel values with proper NaN."""
    return df.replace(_NAN_VALUES, pd.NA)


# ---------------------------------------------------------------------------
# 1. NOAA Mauna Loa weekly CO2
# ---------------------------------------------------------------------------

def collect_noaa_co2_mauna_loa() -> None:
    """Weekly CO2 concentrations from NOAA ESRL Mauna Loa Observatory."""
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.csv"
    resp = fetch(url)

    # File has comment lines starting with '#'; skip them
    lines = [l for l in resp.text.splitlines() if not l.startswith("#")]
    text = "\n".join(lines)

    df = pd.read_csv(
        io.StringIO(text),
        names=["year", "month", "day", "decimal_year", "co2_ppm",
               "days_in_average", "1_year_ago", "10_years_ago", "since_1800"],
        skipinitialspace=True,
    )
    df = _clean_sentinels(df)

    # Build date column from year/month/day
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].rename(
            columns={"year": "year", "month": "month", "day": "day"}
        ),
        errors="coerce",
        utc=True,
    )
    df = df.drop(columns=["year", "month", "day", "decimal_year"])
    df = df.dropna(subset=["date"])
    df = to_datetime_index(df, col="date")

    # co2_ppm should be numeric
    df["co2_ppm"] = pd.to_numeric(df["co2_ppm"], errors="coerce")
    df = df.dropna(subset=["co2_ppm"])

    save(df, "environmental", "NOAA_CO2_MaunaLoa_1w.parquet")


# ---------------------------------------------------------------------------
# 2. NOAA Global Temperature Anomaly monthly
# ---------------------------------------------------------------------------

def collect_noaa_global_temp_anomaly() -> None:
    """Annual global land+ocean temperature anomaly from NOAA NCEI."""
    url = (
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
        "global/time-series/globe/land_ocean/1/1/1850-2024.csv"
    )
    resp = fetch(url)

    # File has comment lines starting with '#'; header is 'Year,Departure from Average'
    lines = [l for l in resp.text.splitlines() if not l.startswith("#")]
    header_idx = next(
        (i for i, l in enumerate(lines) if l.strip().lower().startswith("year")),
        None,
    )
    if header_idx is None:
        print("  WARNING: NOAA Global Temp — could not locate header row, skipping")
        return

    text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    # Normalise: 'Year' -> 'year', 'Departure from Average' -> 'temp_anomaly_c'
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "year":
            col_map[c] = "year"
        elif "departure" in lc or "value" in lc or "anomaly" in lc:
            col_map[c] = "temp_anomaly_c"
    df = df.rename(columns=col_map)
    df = _clean_sentinels(df)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str), format="%Y", errors="coerce", utc=True)
    df = df.drop(columns=["year"])
    df = df.dropna(subset=["date"])
    df["temp_anomaly_c"] = pd.to_numeric(df["temp_anomaly_c"], errors="coerce")
    df = df.dropna(subset=["temp_anomaly_c"])
    df = to_datetime_index(df, col="date")

    save(df, "environmental", "NOAA_GlobalTemp_Anomaly_1m.parquet")


# ---------------------------------------------------------------------------
# 3. NASA GISS Surface Temperature (GISTEMP) annual
# ---------------------------------------------------------------------------

def collect_nasa_gistemp() -> None:
    """Annual global surface temperature anomaly from NASA GISS (GISTEMP v4)."""
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    # NASA blocks default Python user-agent; send a browser-like header
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARNING: NASA GISTEMP — {exc}")
        return

    lines = resp.text.splitlines()
    header_idx = next(
        (i for i, l in enumerate(lines) if "Year" in l and "Jan" in l),
        None,
    )
    if header_idx is None:
        print("  WARNING: NASA GISTEMP — could not locate header row, skipping")
        return

    text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = _clean_sentinels(df)

    year_col = "Year"
    seasonal_cols = [c for c in ["J-D", "DJF", "MAM", "JJA", "SON"] if c in df.columns]
    cols_to_keep = [c for c in [year_col] + seasonal_cols if c in df.columns]
    df = df[cols_to_keep].copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col])
    df["date"] = pd.to_datetime(df[year_col].astype(int).astype(str), format="%Y", errors="coerce", utc=True)
    df = df.drop(columns=[year_col])
    df = df.dropna(subset=["date"])

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = to_datetime_index(df, col="date")
    save(df, "environmental", "NASA_GISTEMP_GlobalTemp_1y.parquet")


# ---------------------------------------------------------------------------
# 4. Our World in Data — CO2 Emissions annual
# ---------------------------------------------------------------------------

_OWID_COLS = ["country", "year", "co2", "co2_per_capita", "cumulative_co2", "share_global_co2"]


def collect_owid_co2() -> None:
    """Annual CO2 emissions by country from Our World in Data (OWID GitHub)."""
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    resp = fetch(url)

    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    df = _clean_sentinels(df)

    available = [c for c in _OWID_COLS if c in df.columns]
    df = df[available].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str), format="%Y", errors="coerce", utc=True)
    df = df.drop(columns=["year"])
    df = df.dropna(subset=["date"])

    for col in df.columns:
        if col not in ("date", "country"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("date").sort_index()

    save(df, "environmental", "OWID_CO2_Emissions_1y.parquet")


# ---------------------------------------------------------------------------
# 5. NSIDC Arctic Sea Ice Extent monthly (September minimum)
# ---------------------------------------------------------------------------

def collect_nsidc_arctic_sea_ice() -> None:
    """Monthly Arctic sea ice extent from NSIDC (September series)."""
    url = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/N_09_extent_v4.0.csv"
    resp = fetch(url)

    lines = resp.text.splitlines()
    # Skip comment/header lines until we find the data header
    header_idx = next(
        (i for i, l in enumerate(lines) if "Year" in l or "year" in l),
        None,
    )
    if header_idx is None:
        # Try parsing as plain CSV with first row as header
        header_idx = 0

    text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = _clean_sentinels(df)

    # Expect columns: year, mo (month), data-type, region, extent, area
    year_col = next((c for c in df.columns if "year" in c), None)
    month_col = next((c for c in df.columns if c in ("mo", "month", "mon")), None)
    extent_col = next((c for c in df.columns if "extent" in c), None)

    if year_col is None or extent_col is None:
        print(f"  WARNING: NSIDC Arctic Sea Ice — unexpected columns {list(df.columns)}, skipping")
        return

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[extent_col] = pd.to_numeric(df[extent_col], errors="coerce")

    if month_col:
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce").fillna(9).astype(int)
        df["date"] = pd.to_datetime(
            df[year_col].astype(int).astype(str) + "-" + df[month_col].astype(str).str.zfill(2) + "-01",
            format="%Y-%m-%d",
            errors="coerce",
            utc=True,
        )
    else:
        # September-only series; month = 9
        df["date"] = pd.to_datetime(
            df[year_col].astype(int).astype(str) + "-09-01",
            format="%Y-%m-%d",
            errors="coerce",
            utc=True,
        )

    df = df.dropna(subset=["date", extent_col])
    keep = ["date", extent_col] + [
        c for c in df.columns if c not in ("date", year_col, month_col, extent_col, "data-type", "region")
        and c is not None
    ]
    df = df[[c for c in keep if c in df.columns]]
    df = to_datetime_index(df, col="date")

    save(df, "environmental", "NSIDC_ArcticSeaIce_1m.parquet")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COLLECTORS: list[tuple[str, object]] = [
    ("NOAA CO2 Mauna Loa weekly", collect_noaa_co2_mauna_loa),
    ("NOAA Global Temp Anomaly monthly", collect_noaa_global_temp_anomaly),
    ("NASA GISTEMP annual", collect_nasa_gistemp),
    ("OWID CO2 Emissions annual", collect_owid_co2),
    ("NSIDC Arctic Sea Ice monthly", collect_nsidc_arctic_sea_ice),
]


def main() -> None:
    for name, collector in _COLLECTORS:
        print(f"Fetching: {name}")
        try:
            collector()
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
