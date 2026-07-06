"""
UCDP Georeferenced Event Dataset — conflict fatalities.
Free CSV download, no API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save


_URL = "https://ucdp.uu.se/downloads/ged/ged241-csv.zip"


def collect():
    """Fetch UCDP conflict events, aggregate monthly."""
    import urllib.request, io, zipfile
    print(f"Downloading UCDP conflict data (~20MB)...")
    try:
        with urllib.request.urlopen(_URL, timeout=120) as r:
            raw = r.read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(csv_name), low_memory=False)
    except Exception as e:
        print(f"  WARN: {e}")
        return

    print(f"  Parsed {len(df):,} events")
    df["date"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()

    # Aggregations
    monthly = df.resample("ME").agg({"best": "sum", "low": "sum", "high": "sum", "id": "count"}).rename(
        columns={"best": "fatalities_best", "low": "fatalities_low", "high": "fatalities_high", "id": "event_count"}
    )
    monthly["fatalities_best"] = pd.to_numeric(monthly["fatalities_best"], errors="coerce")
    monthly["fatalities_high"] = pd.to_numeric(monthly["fatalities_high"], errors="coerce")
    monthly["fatalities_low"] = pd.to_numeric(monthly["fatalities_low"], errors="coerce")
    monthly.index.name = "date"

    out = save(monthly, "geopolitical", "ucdp_conflict_1m.parquet")
    print(f"Saved {out}: {len(monthly):,} months")

    # Also save raw events by region
    for region in df["region"].dropna().unique() if "region" in df.columns else []:
        rdf = df[df["region"] == region].resample("ME").agg(
            {"best": "sum", "low": "sum", "high": "sum", "id": "count"}
        ).rename(
            columns={"best": "fatalities_best", "low": "fatalities_low", "high": "fatalities_high", "id": "event_count"}
        )
        rdf.index.name = "date"
        safe_name = region.lower().replace(" ", "_").replace("/", "_")[:20]
        save(rdf, "geopolitical", f"ucdp_{safe_name}_1m.parquet")


if __name__ == "__main__":
    collect()
