"""
Commodity prices from FRED — wheat, corn, coal, gold, silver, etc.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save

_SERIES = {
    "PCOALAUUSDM": "coal_usd_mt",
    "PCOALBBAUSDM": "coal_bb_usd_mt",
    "PFOODINDEXM": "food_price_index",
    "PCEREALSINDEXM": "cereals_index",
    "PMETALINDEXM": "metals_index",
    "PRAWMINDEXM": "raw_materials_index",
    "PWHEATUSDM": "wheat_usd_mt",
    "PMAIZEUSDM": "maize_usd_mt",
    "PRICEQ": "rice_usd_mt",
    "PSOYBUSDM": "soybean_usd_mt",
    "PSUNOUSDM": "sunflower_usd_mt",
    "PPALMOLUSDM": "palm_oil_usd_mt",
    "PBEEFUSDM": "beef_usd_kg",
    "PPOULTUSDM": "chicken_usd_kg",
    "PSALMUSDM": "salmon_usd_kg",
    "PCOCOUSDM": "cocoa_usd_mt",
    "PRUBBINDM": "rubber_usd_kg",
    "PBANSOPUSDM": "banana_usd_mt",
}


def collect():
    import urllib.request, io, time
    print("Fetching FRED commodity series...")
    rows = []
    for sid, label in _SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            df.columns = ["date", label]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[label] = pd.to_numeric(df[label], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
            rows.append(df)
        except Exception as e:
            print(f"  WARN {label}: {e}")
        time.sleep(0.3)

    combined = pd.concat(rows, axis=1).sort_index()
    out = save(combined, "agriculture", "fred_commodities_1m.parquet")
    print(f"Saved {out}: {len(combined)} rows, {len(combined.columns)} series")


if __name__ == "__main__":
    collect()
