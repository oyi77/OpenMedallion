"""
Zillow / housing market data via FRED CSV API.
Source: FRED (St. Louis Fed) — no API key required
Output: data/real_estate/ZILLOW_<label>_1m.parquet (monthly)

Series included:
  - MSPUS       : Median Sales Price of Houses Sold (US)
  - ASPUS       : Average Sales Price of Houses Sold (US)
  - MEDLISPRI   : Median Listing Price (US)
  - MEDLISPPERSQUFEE : Median Listing Price per Sq Ft (US)
  - ACTLISCOUUS : Active Listing Count (US)
  - NEWLISCOUUS : New Listing Count (US)
  - DAYSONDOUS  : Median Days on Market (US)
  - HOUST        : Housing Starts (total)
  - PERMIT       : Building Permits (total)
  - HSOLD        : Houses Sold (annual rate)
  - HSODINUSM052N: Houses for Sale
  - EVACANTUSQ176N: Vacant housing units
  - RHVRUSQ156N  : Rental vacancy rate
  - HVACUSQ176N  : Homeowner vacancy rate
  - CSUSHPINSA   : Case-Shiller HPI (national, NSA)
  - CSUSHPISA    : Case-Shiller HPI (national, SA)
  - USSTHPI      : FHFA HPI (all transactions)
  - HPIPONM226S  : FHFA HPI (purchase only)
  - MORTGAGE30US : 30-Year Fixed Rate Mortgage Average
  - MORTGAGE15US : 15-Year Fixed Rate Mortgage Average
  - MSPNHSUS     : Median Sales Price New Houses
  - MSPUS        : Median Sales Price All Houses  (already listed, kept for alias)
"""
from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

_SLEEP = 0.4
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# (series_id, label)  — label used in filename
SERIES: list[tuple[str, str]] = [
    ("MSPUS",           "MedianSalesPrice_US"),
    ("ASPUS",           "AvgSalesPrice_US"),
    ("MEDLISPRI",       "MedianListPrice_US"),
    ("MEDLISPPERSQUFEE","MedianListPricePerSqFt_US"),
    ("ACTLISCOUUS",     "ActiveListings_US"),
    ("NEWLISCOUUS",     "NewListings_US"),
    ("DAYSONDOUS",      "DaysOnMarket_US"),
    ("HOUST",           "HousingStarts_US"),
    ("PERMIT",          "BuildingPermits_US"),
    ("HSOLD",           "HousesSold_US"),
    ("HSODINUSM052N",   "HousesForSale_US"),
    ("EVACANTUSQ176N",  "VacantHousingUnits_US"),
    ("RHVRUSQ156N",     "RentalVacancyRate_US"),
    ("HVACUSQ176N",     "HomeownerVacancyRate_US"),
    ("CSUSHPINSA",      "CaseShillerHPI_NSA"),
    ("CSUSHPISA",       "CaseShillerHPI_SA"),
    ("USSTHPI",         "FHFAHPI_AllTransactions"),
    ("HPIPONM226S",     "FHFAHPI_PurchaseOnly"),
    ("MORTGAGE30US",    "Mortgage30YFixed"),
    ("MORTGAGE15US",    "Mortgage15YFixed"),
    ("MSPNHSUS",        "MedianSalesPriceNewHomes_US"),
]


def _fetch_fred(series_id: str, label: str) -> pd.DataFrame | None:
    url = _FRED_CSV.format(series_id=series_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if df.empty or len(df.columns) < 2:
            print(f"  WARNING {series_id}: empty or malformed response")
            return None
        # FRED CSV always has DATE + value columns
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        # Drop rows where value is '.', which FRED uses for missing
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            print(f"  WARNING {series_id}: all values missing after coerce")
            return None
        df["series_id"] = series_id
        df["label"] = label
        return df
    except Exception as exc:
        print(f"  WARNING {series_id} ({label}): {exc}")
        return None


def collect_zillow() -> None:
    ok = 0
    for series_id, label in SERIES:
        df = _fetch_fred(series_id, label)
        if df is not None and not df.empty:
            save(df, "real_estate", f"ZILLOW_{label}_1m.parquet")
            ok += 1
        time.sleep(_SLEEP)
    print(f"\nDone: {ok}/{len(SERIES)} Zillow/housing files saved")


def main() -> None:
    print(f"Fetching {len(SERIES)} housing/FRED series (monthly)")
    collect_zillow()


if __name__ == "__main__":
    main()
