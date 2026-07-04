"""
US Real Estate data collector.
Sources: FRED (free, no key needed)
  - CSUSHPISA : S&P/Case-Shiller US National Home Price Index (monthly)
  - HOUST      : Housing Starts: Total New Privately Owned (monthly, thousands)
  - RHVRUSQ156N: Rental Vacancy Rate (quarterly, %)
  - MSPUS      : Median Sales Price of Houses Sold (quarterly, USD)
  - PERMIT     : New Private Housing Units Authorized by Building Permits
  - EXHOSLUSM495S: Existing Home Sales (monthly)
Output: data/real_estate/REALESTATE_<series>_1m.parquet
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
    "REALESTATE_CaseShiller_National_1m": "CSUSHPISA",
    "REALESTATE_HousingStarts_1m": "HOUST",
    "REALESTATE_RentalVacancyRate_1q": "RHVRUSQ156N",
    "REALESTATE_MedianSalesPrice_1q": "MSPUS",
    "REALESTATE_BuildingPermits_1m": "PERMIT",
    "REALESTATE_ExistingHomeSales_1m": "EXHOSLUSM495S",
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
            save(df, "real_estate", f"{name}.parquet")
        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
