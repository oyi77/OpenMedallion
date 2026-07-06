"""
FRED agricultural price indexes — PPI sub-indexes for farming & food.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save

_SERIES = {
    "WPU01": "ppi_grains",
    "WPU012": "ppi_cereal_grains",
    "WPU013": "ppi_livestock",
    "WPU0131": "ppi_slaughter_cattle",
    "WPU0132": "ppi_slaughter_hogs",
    "WPU0133": "ppi_lambs_sheep",
    "WPU014": "ppi_poultry",
    "WPU0141": "ppi_chickens",
    "WPU0142": "ppi_turkeys",
    "WPU015": "ppi_dairy",
    "WPU0151": "ppi_raw_milk",
    "WPU016": "ppi_plant_fiber",
    "WPU017": "ppi_oilseeds",
    "WPU018": "ppi_tobacco",
    "WPU0191": "ppi_feed",
    "WPU021": "ppi_grain_mill",
    "WPU022": "ppi_processed_foods",
    "WPU024": "ppi_vegetable_oils",
    "WPU025": "ppi_sugar_confectionery",
    "WPU026": "ppi_beverages",
}


def collect():
    """Fetch all FRED agricultural PPI series."""
    import urllib.request, io, time
    rows = []
    for sid, label in _SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        print(f"  {label}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            df.columns = ["date", label]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[label] = pd.to_numeric(df[label], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
            rows.append(df)
        except Exception as e:
            print(f"    WARN: {label}: {e}")
        time.sleep(0.3)

    if not rows:
        return
    combined = pd.concat(rows, axis=1).sort_index()
    combined.index.name = "date"
    out = save(combined, "agriculture", "fred_agri_ppi_1m.parquet")
    print(f"Saved {out}: {len(combined)} rows, {len(combined.columns)} series")


if __name__ == "__main__":
    collect()
