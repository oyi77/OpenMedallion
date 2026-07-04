"""
Supply chain pressure and industrial indicators collector.
Source: FRED API (free, no key required)
Covers: NY Fed GSCPI proxy, ISM PMI, inventories, new orders, backlogs
Output: data/supply_chain/SC_<series>_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from base import save, fetch

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {
    "SC_Manufacturing_Employment_1m": "MANEMP",
    "SC_Business_Inventories_1m": "ISRATIO",
    "SC_New_Orders_Manufactured_1m": "AMTMNO",
    "SC_Durable_Goods_Orders_1m": "DGORDER",
    "SC_Core_Capital_Goods_Orders_1m": "ACOGNO",
    "SC_Unfilled_Orders_1m": "AMDMUO",
    "SC_Truck_Tonnage_Index_1m": "TRUCKD11",
    "SC_Rail_Traffic_1w": "RAILFRTINTERMODAL",
    "SC_Industrial_Production_1m": "INDPRO",
    "SC_Capacity_Utilization_1m": "TCU",
    "SC_Manufacturing_Output_1m": "IPMAN",
    "SC_PPI_Commodities_1m": "PPIACO",
    "SC_PPI_Intermediate_1m": "PPIIDC",
    "SC_PPI_Finished_Goods_1m": "WPSFD4131",
}


def fetch_fred(series_id: str, value_col: str) -> pd.DataFrame:
    url = FRED_CSV.format(series_id)
    resp = fetch(url, timeout=30)
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", value_col]
    df = df[df[value_col] != "."].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=[value_col]).reset_index(drop=True)


def main() -> None:
    print("Collecting supply chain and industrial data...")
    ok = 0
    for filename, series_id in SERIES.items():
        try:
            df = fetch_fred(series_id, series_id)
            if not df.empty:
                save(df, "supply_chain", f"{filename}.parquet")
                print(f"  {series_id}: {len(df)} rows")
                ok += 1
            else:
                print(f"  {series_id}: empty")
        except Exception as exc:
            print(f"  WARNING: {series_id} - {exc}")
    print(f"\nDone: {ok}/{len(SERIES)} series collected")


if __name__ == "__main__":
    main()
