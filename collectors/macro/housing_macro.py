"""
Extended macro indicators collector — 60+ additional FRED series not yet in dataset.
Source: FRED API (free, no key required)
Covers: GDP components, trade balance, current account, productivity, fiscal
Output: data/macro/FRED_<series>_1d.parquet  (matches existing naming)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from base import save, fetch

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# All series saved as FRED_<name>_1d.parquet matching existing convention
SERIES = {
    "GDP_Real_1q": "GDPC1",
    "GDP_Nominal_1q": "GDP",
    "GDP_Growth_Rate_1q": "A191RL1Q225SBEA",
    "GDP_Per_Capita_1q": "A939RX0Q048SBEA",
    "GNP_1q": "GNP",
    "GDI_1q": "GDI",
    "Net_Exports_1q": "NETEXP",
    "Trade_Balance_1m": "BOPGSTB",
    "Current_Account_1q": "NETFI",
    "Capital_Account_1q": "NCBCMDPMVCE",
    "Federal_Debt_Total_1q": "GFDEBTN",
    "Federal_Debt_Pct_GDP_1q": "GFDEGDQ188S",
    "Federal_Deficit_1m": "MTSDS133FMS",
    "Federal_Receipts_1m": "MTSR133FMS",
    "Federal_Outlays_1m": "MTSO133FMS",
    "State_Local_Gov_Spending_1q": "SLEXPND",
    "Business_Fixed_Investment_1q": "PNFI",
    "Residential_Investment_1q": "PRFI",
    "Inventory_Change_1q": "CBI",
    "Gov_Consumption_1q": "GCE",
    "Real_Disposable_Income_1m": "DSPIC96",
    "Nonfarm_Business_Output_1q": "OUTNFB",
    "Manufacturing_Output_1m": "IPMAN",
    "Services_Output_1q": "B535RC1Q027SBEA",
    "Construction_Output_1m": "TTLCONS",
    "Mining_Output_1m": "IPG211S",
    "Utilities_Output_1m": "IPG2211S",
    "Agricultural_Output_1q": "A01000NA",
    "Export_Price_Index_1m": "IQ",
    "Import_Price_Index_1m": "IR",
    "Terms_of_Trade_1m": "DCOILBRENTEU",
    "Dollar_Index_Major_1d": "DTWEXBGS",
    "Dollar_Index_Broad_1d": "DTWEXAFEGS",
    "Euro_Dollar_1d": "DEXUSEU",
    "Yen_Dollar_1d": "DEXJPUS",
    "GBP_Dollar_1d": "DEXUSUK",
    "Yuan_Dollar_1d": "DEXCHUS",
    "Won_Dollar_1d": "DEXKOUS",
    "Real_Effective_Exchange_Rate_1m": "RBUSBIS",
    "Corporate_Profits_1q": "CP",
    "Profit_Margin_1q": "A466RD3Q052SBEA",
    "S_and_P_Earnings_1q": "SPASTT01USQ657N",
    "NIPA_PCE_Deflator_1q": "DPCERD3Q086SBEA",
    "GDP_Deflator_1q": "GDPDEF",
    "PPI_All_1m": "PPIACO",
    "PPI_Core_1m": "PPICOR",
    "Import_Prices_1m": "IR",
    "Export_Prices_1m": "IQ",
    "Breakeven_5y5y_1d": "T5YIFR",
    "Real_Interest_Rate_10yr_1d": "REAINTRATREARAT10Y",
    "Real_Interest_Rate_1yr_1d": "REAINTRATREARAT1YE",
    "Natl_Financial_Conditions_1w": "NFCI",
    "Adj_Financial_Conditions_1w": "ANFCI",
    "Credit_Conditions_1q": "DRBLACBS",
    "Senior_Loan_Officer_1q": "DRSDCILM",
    "TFP_Business_1q": "TFPNFBBS",
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
    print("Collecting extended macro FRED data...")
    ok = 0
    for name, series_id in SERIES.items():
        try:
            df = fetch_fred(series_id, series_id)
            if not df.empty:
                save(df, "macro", f"FRED_{name}_1d.parquet")
                print(f"  {series_id}: {len(df)} rows")
                ok += 1
            else:
                print(f"  {series_id}: empty")
        except Exception as exc:
            print(f"  WARNING: {series_id} - {exc}")
    print(f"\nDone: {ok}/{len(SERIES)} series collected")


if __name__ == "__main__":
    main()
