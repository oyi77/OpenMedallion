"""
USGS significant earthquakes — last 5 years, M4.5+.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save


_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&minmagnitude=4.5&orderby=magnitude"


def collect():
    """Fetch significant earthquakes from USGS."""
    import urllib.request, json
    import datetime

    # Fetch in yearly chunks
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=5)
    url = f"{_URL}&starttime={start:%Y-%m-%d}&endtime={end:%Y-%m-%d}"

    print(f"Fetching USGS earthquakes...")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  WARN: {e}")
        return

    features = data.get("features", [])
    if not features:
        print("  No earthquakes found")
        return

    rows = []
    for f in features:
        props = f.get("properties", {})
        geo = f.get("geometry", {})
        coords = geo.get("coordinates", [None, None, None])
        rows.append({
            "time": pd.to_datetime(props.get("time"), unit="ms"),
            "mag": props.get("mag"),
            "depth_km": coords[2],
            "lat": coords[1],
            "lon": coords[0],
            "place": props.get("place", ""),
            "magType": props.get("magType"),
            "tsunami": 1 if props.get("tsunami") else 0,
            "sig": props.get("sig"),
        })

    df = pd.DataFrame(rows).dropna(subset=["time"]).set_index("time").sort_index()
    for c in ["mag", "depth_km", "lat", "lon", "sig"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = save(df, "environmental", "usgs_earthquakes_1d.parquet")
    print(f"Saved {out}: {len(df)} quakes (M4.5+, 5 years)")


if __name__ == "__main__":
    collect()
