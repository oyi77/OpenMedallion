"""
FRED real estate & housing series — mortgage rates, ownership, construction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save

_SERIES = {
    "MORTGAGE30US": "mortgage_30yr",
    "MORTGAGE15US": "mortgage_15yr",
    "MORTGAGE5US": "mortgage_5_1_arm",
    "FHFHPINUSA": "fhfa_hpi_national",
    "HOSAVACUSQ176N": "homeownership_rate",
    "TLPPUCON": "construction_spending",
    "ASPS": "avg_sales_price",
    "OBMMIC30YF": "mortgage_rate_optimal_blue",
    "RRSFPSHSORE": "rental_vacancy_rate",
    "EVACANTUSQ176N": "housing_vacancy_rate",
    "RRVRUSQ156N": "rental_residential_vacancy",
    "BUSAPPWNSAUS": "building_permits_apps",
}


def collect():
    """Fetch all FRED real estate series."""
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
    out = save(combined, "real_estate", "fred_housing_1m.parquet")
    print(f"Saved {out}: {len(combined)} rows, {len(combined.columns)} series")


if __name__ == "__main__":
    collect()
