"""
US macro indicators via FRED CSV export.
Source: FRED (St. Louis Fed) — no API key, public CSV endpoint.
Covers: GDP, CPI, PCE, industrial production, retail sales, housing,
        unemployment, money supply, consumer sentiment.
Output: data/macro/FRED_<series>_<freq>.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SLEEP = 0.4

# (series_id, label, freq)
FRED_SERIES: list[tuple[str, str, str]] = [
    # GDP & Growth
    ("GDP",      "GDP_Nominal",           "1q"),
    ("GDPC1",    "GDP_Real",              "1q"),
    ("GDPPOT",   "GDP_Potential",         "1q"),
    ("PCEC96",   "PCE_Real",              "1q"),

    # Inflation
    ("CPIAUCSL", "CPI_AllItems",          "1m"),
    ("CPILFESL", "CPI_Core",              "1m"),
    ("PCEPI",    "PCE_Deflator",          "1m"),
    ("PCEPILFE", "PCE_Core",              "1m"),
    ("PPIFIS",   "PPI_FinalDemand",       "1m"),

    # Labor market
    ("UNRATE",   "Unemployment_Rate",     "1m"),
    ("PAYEMS",   "Nonfarm_Payrolls",      "1m"),
    ("CIVPART",  "Labor_Participation",   "1m"),
    ("AWHMAN",   "AvgWeeklyHours_Mfg",   "1m"),
    ("ICSA",     "InitialJoblessClaims",  "1w"),
    ("CCSA",     "ContJoblessClaims",     "1w"),
    ("JTSJOL",   "JOLTS_OpenJobs",        "1m"),

    # Industrial & manufacturing
    ("INDPRO",   "IndustrialProduction",  "1m"),
    ("TCU",      "CapacityUtilization",   "1m"),
    ("UMCSENT",  "ConsumerSentiment",     "1m"),
    ("RSAFS",    "RetailSales",           "1m"),

    # Housing
    ("HOUST",    "HousingStarts",         "1m"),
    ("PERMIT",   "BuildingPermits",       "1m"),
    ("EXHOSLUSM495S", "ExistingHomeSales","1m"),
    ("MSPUS",    "MedianHomeSalePrice",   "1q"),

    # Money supply & Fed
    ("M2SL",     "M2_MoneySupply",        "1m"),
    ("M1SL",     "M1_MoneySupply",        "1m"),
    ("FEDFUNDS", "FedFunds_Rate",         "1m"),
    ("DFF",      "FedFunds_Daily",        "1d"),
    ("BOGMBASE", "MonetaryBase",          "1m"),

    # Trade & external
    ("BOPGSTB",  "TradeBalance",          "1m"),
    ("NETEXP",   "NetExports",            "1q"),

    # Credit & financial conditions
    ("DRCCLACBS", "CreditCard_DeliqRate", "1q"),
    ("DRSBLACBS", "CommercialRE_DelinqRate","1q"),
    ("DRSFRMACBS","Mortgage_DelinqRate",  "1q"),

    # Confidence & expectations
    ("USEPUINDXD", "EconPolicyUncertainty","1d"),
]


def fetch_fred_csv(series_id: str) -> pd.DataFrame | None:
    try:
        url = f"{FRED_BASE}?id={series_id}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(
            pd.io.common.BytesIO(resp.content),
            parse_dates=["DATE"],
            index_col="DATE",
            na_values=[".", ""],
        )
        df.columns = ["value"]
        df = df.dropna()
        return to_datetime_index(df)
    except Exception:
        return None


def main() -> None:
    print("Fetching: FRED macro indicators")
    ok = 0
    for series_id, label, freq in FRED_SERIES:
        df = fetch_fred_csv(series_id)
        if df is not None and not df.empty:
            save(df, "macro", f"FRED_{label}_{freq}.parquet")
            ok += 1
            print(f"  ✓ {label} ({len(df)} rows)")
        else:
            print(f"  ✗ {label}")
        time.sleep(SLEEP)

    print(f"\n✓ {ok}/{len(FRED_SERIES)} FRED macro series saved")


if __name__ == "__main__":
    main()
