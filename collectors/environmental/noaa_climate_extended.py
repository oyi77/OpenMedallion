"""
NOAA extended climate data — precipitation, drought, temperature.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save

_SERIES = {
    "CPIAUCSL": "cpi_all",
    "PPIACO": "ppi_all",
    "UNRATE": "unemployment",
    # NOAA/Climate proxy series through FRED
    "USACPALTTRI": "palmer_drought",
    "CLVMNACSCAB1GQIT": "euro_gdp",
}


def collect():
    """Fetch extended environmental & climate proxy data."""
    # Focus on actual climate data: Palmer Drought Severity Index, precipitation
    import urllib.request, io, time

    # NOAA NCEI Climate at a Glance - free CSV
    datasets = {
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/pcp/12/12/1895-2025.csv": "us_precip",
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tavg/12/12/1895-2025.csv": "us_temp",
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tmax/12/12/1895-2025.csv": "us_tmax",
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tmin/12/12/1895-2025.csv": "us_tmin",
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/pdsi/12/12/1895-2025.csv": "us_pdsi",
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/phdi/12/12/1895-2025.csv": "us_phdi",
        "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/zndx/12/12/1895-2025.csv": "us_zndx",
    }

    rows = []
    for url, label in datasets.items():
        print(f"  {label}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8")
            # Skip header lines starting with #
            lines = [l for l in raw.split("\n") if l.strip() and not l.startswith("#")]
            if len(lines) < 2:
                continue
            df = pd.read_csv(io.StringIO("\n".join(lines)))
            if "Date" in df.columns:
                date_col = "Date"
            elif "Year" in df.columns and "Month" in df.columns:
                df["date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
                date_col = "date"
            elif "Year" in df.columns:
                df["date"] = pd.to_datetime(df["Year"], format="%Y")
                date_col = "date"
            else:
                date_col = df.columns[0]
            df[label] = pd.to_numeric(df.iloc[:, -1], errors="coerce")
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["date", label]).set_index("date")[[label]]
            rows.append(df)
        except Exception as e:
            print(f"    WARN: {label}: {e}")
        time.sleep(0.5)

    if not rows:
        print("  No environmental data fetched")
        return
    combined = pd.concat(rows, axis=1).sort_index()
    combined.index.name = "date"
    out = save(combined, "environmental", "noaa_climate_extended_1m.parquet")
    print(f"Saved {out}: {len(combined)} rows, {len(combined.columns)} series")


if __name__ == "__main__":
    collect()
