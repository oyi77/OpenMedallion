"""
EIA natural gas & crude oil data collector.
Sources:
  - EIA Open Data API v2 (no key for public series)
  - FRED via base fetcher fallback
Output: data/energy/EIA_<metric>_1d.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

EIA_BASE = "https://api.eia.gov/v2"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    resp = fetch(FRED_BASE, params={"id": series_id}, timeout=20)
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), na_values=".")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")[["value"]].dropna().sort_index()


# FRED series covering EIA-published data
FRED_ENERGY: dict[str, str] = {
    "EIA_CrudeOil_WTI_Spot_1d": "DCOILWTICO",
    "EIA_CrudeOil_Brent_Spot_1d": "DCOILBRENTEU",
    "EIA_Gasoline_Regular_1w": "GASREGCOVW",
    "EIA_Diesel_Retail_1w": "GASDESW",
    "EIA_NatGas_HenryHub_Spot_1d": "DHHNGSP",
    "EIA_CrudeOil_Inventories_1w": "WCESTUS1",
    "EIA_Cushing_Inventories_1w": "WCSSTUS1",
    "EIA_Gasoline_Inventories_1w": "WGTSTUS1",
    "EIA_Distillate_Inventories_1w": "WDISTUS1",
    "EIA_NatGas_Storage_1w": "WNGSA",
    "EIA_Crude_Imports_1w": "WCRIMUS2",
    "EIA_Crude_Production_1w": "WCRFPUS2",
    "EIA_Refinery_Utilization_1w": "WPULEUS3",
    "EIA_Jet_Fuel_1w": "WJFUPUS2",
    "EIA_Propane_Inventories_1w": "WPRSTUS1",
}


def main() -> None:
    print("=== EIA / FRED energy series ===")
    for name, sid in FRED_ENERGY.items():
        try:
            df = fetch_fred_series(sid)
            save(df, "energy", f"{name}.parquet")
        except Exception as exc:
            print(f"  WARN {name} ({sid}) — {exc}")


if __name__ == "__main__":
    main()
