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
# 6. SIDC Solar Sunspot Number monthly (1749–present)
# ---------------------------------------------------------------------------

SIDC_SUNSPOT_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"

def collect_sidc_sunspots() -> None:
    """Monthly international sunspot number from SIDC/SILSO (1749–present)."""
    resp = fetch(SIDC_SUNSPOT_URL)
    # Fixed-width: year month decimal_year sn_mean sn_std n_obs provisional
    rows = []
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year, month = int(parts[0]), int(parts[1])
            sn = float(parts[3])
            rows.append({"date": f"{year:04d}-{month:02d}-01", "sunspot_number": sn if sn >= 0 else pd.NA})
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = to_datetime_index(df, col="date")
    save(df, "environmental", "SIDC_Sunspots_1m.parquet")


# ---------------------------------------------------------------------------
# 7. NOAA Palmer Drought Severity Index (PDSI) — US climate divisions, monthly
# ---------------------------------------------------------------------------

NOAA_PDSI_BASE = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"
NOAA_PDSI_PATTERN = "climdiv-pdsidv-v1.0.0-"

def _find_pdsi_url() -> str:
    """Resolve the latest PDSI file (filename includes datestamp)."""
    resp = fetch(NOAA_PDSI_BASE)
    import re
    names = re.findall(r"climdiv-pdsidv-v1\.0\.0-\d{8}", resp.text)
    if not names:
        raise RuntimeError("Could not find PDSI filename on NCEI index page")
    latest = sorted(names)[-1]
    return NOAA_PDSI_BASE + latest

def collect_noaa_pdsi() -> None:
    """Monthly Palmer Drought Severity Index for US climate divisions (NOAA NCEI)."""
    url = _find_pdsi_url()
    resp = fetch(url)
    # Format: SSDDDDYYYY followed by 12 monthly values (fixed-width space-separated)
    # State(2) + Div(2) + Element(2) + Year(4) then 12 monthly values
    rows = []
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            code = parts[0]           # e.g. "0101051895"
            year = int(code[-4:])
            state_div = code[:4]
            monthly = [float(v) for v in parts[1:13]]
        except (ValueError, IndexError):
            continue
        for m_idx, val in enumerate(monthly, start=1):
            if val in (-99.99, -9.99):
                val = float("nan")
            rows.append({"date": f"{year:04d}-{m_idx:02d}-01", "state_div": state_div, "pdsi": val})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Aggregate to national monthly mean for a single time-series
    nat = df.groupby("date", as_index=False)["pdsi"].mean().rename(columns={"pdsi": "pdsi_national_mean"})
    nat = to_datetime_index(nat, col="date")
    save(nat, "environmental", "NOAA_PDSI_National_1m.parquet")


# ---------------------------------------------------------------------------
# 8. US Wildfire statistics — NIFC annual (1983–present)
# ---------------------------------------------------------------------------

EPA_WILDFIRE_URL = "https://www.epa.gov/sites/default/files/2021-04/wildfires_fig-1.csv"

def collect_us_wildfires() -> None:
    """Annual US wildfire count and acres burned from EPA/NIFC (1983–2020)."""
    resp = fetch(EPA_WILDFIRE_URL)
    lines = resp.text.splitlines()
    # Skip preamble rows until we hit the "Year," header
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("Year,")), None)
    if start is None:
        raise RuntimeError("Wildfire CSV header 'Year,' not found")
    import io as _io
    text = "\n".join(lines[start:])
    df = pd.read_csv(_io.StringIO(text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Rename columns for clarity
    col_map = {c: c for c in df.columns}
    if "national_interagency_fire_center" in df.columns:
        col_map["national_interagency_fire_center"] = "fires_count"
    if "forest_service_wildfire_statistics" in df.columns:
        col_map["forest_service_wildfire_statistics"] = "acres_burned"
    df = df.rename(columns=col_map)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01", utc=True, errors="coerce")
    df = df.drop(columns=["year"]).sort_values("date").reset_index(drop=True)
    df = to_datetime_index(df, col="date")
    save(df, "environmental", "US_Wildfires_Annual_1y.parquet")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COLLECTORS: list[tuple[str, object]] = [
    ("NOAA CO2 Mauna Loa weekly", collect_noaa_co2_mauna_loa),
    ("NOAA Global Temp Anomaly monthly", collect_noaa_global_temp_anomaly),
    ("NASA GISTEMP annual", collect_nasa_gistemp),
    ("OWID CO2 Emissions annual", collect_owid_co2),
    ("NSIDC Arctic Sea Ice monthly", collect_nsidc_arctic_sea_ice),
    ("SIDC Sunspots monthly", collect_sidc_sunspots),
    ("NOAA PDSI Drought monthly", collect_noaa_pdsi),
    ("US Wildfires annual", collect_us_wildfires),
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
