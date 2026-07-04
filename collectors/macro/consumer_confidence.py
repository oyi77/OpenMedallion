"""
Consumer confidence, retail sales, and consumer health indicators.
Source: FRED API (free, no key required)
Output: data/macro/Consumer_<series>_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from base import save, fetch

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {
    "Consumer_Confidence_CB_1m": "CSCICP03USM665S",
    "Consumer_Confidence_Michigan_1m": "UMCSENT",
    "Consumer_Confidence_Michigan_Exp_1m": "MICH",
    "Consumer_Sentiment_Current_1m": "UMCSENT",
    "Consumer_Retail_Sales_1m": "RSAFS",
    "Consumer_Retail_Sales_ExAuto_1m": "RSFSXMV",
    "Consumer_Retail_Sales_Online_1m": "ECOMNSA",
    "Consumer_Personal_Income_1m": "PI",
    "Consumer_Personal_Spending_1m": "PCE",
    "Consumer_Savings_Rate_1m": "PSAVERT",
    "Consumer_Credit_Growth_1m": "TOTCI",
    "Consumer_Credit_Revolving_1m": "REVOLSL",
    "Consumer_Credit_Nonrevolving_1m": "NONREVSL",
    "Consumer_CPI_All_1m": "CPIAUCSL",
    "Consumer_CPI_Core_1m": "CPILFESL",
    "Consumer_CPI_Energy_1m": "CPIENGSL",
    "Consumer_CPI_Food_1m": "CPIUFDSL",
    "Consumer_CPI_Services_1m": "CPIUFDSL",
    "Consumer_PCE_Core_1m": "PCEPILFE",
    "Consumer_Gas_Price_1w": "GASREGCOVM",
    "Consumer_Delinquency_CreditCard_1q": "DRCCLACBS",
    "Consumer_Delinquency_Auto_1q": "DRAUTOACBS",
    "Consumer_Bankruptcy_1q": "QUSAUS",
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
    print("Collecting consumer confidence and retail data...")
    ok = 0
    for filename, series_id in SERIES.items():
        try:
            df = fetch_fred(series_id, series_id)
            if not df.empty:
                save(df, "macro", f"FRED_{filename}_1d.parquet")
                print(f"  {series_id}: {len(df)} rows")
                ok += 1
            else:
                print(f"  {series_id}: empty")
        except Exception as exc:
            print(f"  WARNING: {series_id} - {exc}")
    print(f"\nDone: {ok}/{len(SERIES)} series collected")


if __name__ == "__main__":
    main()
