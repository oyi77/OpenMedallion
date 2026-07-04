"""
Shipping / Freight Rate collector.
Sources: FRED (free, no key needed) — shipping and freight proxy series
  - PPIACO   : PPI Commodities (monthly) — commodity freight cost proxy
  - TRUCKD11 : Truck Tonnage Index (monthly)
  - RAILFRTINTERMODAL: Rail Freight Intermodal (weekly)
  - PPIITM   : PPI Intermediate Materials (monthly)
  - PPIFES   : PPI Finished Energy Goods (monthly)
  - PPILFE   : PPI Less Food and Energy (monthly)
  - INDPRO   : Industrial Production (monthly) — demand-side shipping proxy
Output: data/shipping/SHIPPING_<series>.parquet
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

FRED_SERIES = {
    "SHIPPING_PPI_Commodities_1m": "PPIACO",
    "SHIPPING_TruckTonnage_1m": "TRUCKD11",
    "SHIPPING_RailFreight_1w": "RAILFRTINTERMODAL",
    "SHIPPING_PPI_IntermediateMaterials_1m": "PPIITM",
    "SHIPPING_PPI_FinishedEnergy_1m": "PPIFES",
    "SHIPPING_IndustrialProduction_1m": "INDPRO",
}


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = FRED_CSV.format(id=series_id)
    resp = fetch(url, timeout=60)
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    for name, series_id in FRED_SERIES.items():
        print(f"Fetching {name} ({series_id}) ...")
        try:
            df = fetch_fred_series(series_id)
            save(df, "shipping", f"{name}.parquet")
        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
