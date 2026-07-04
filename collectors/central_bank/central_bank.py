"""
Central bank policy rates and communication indices collector.
Source: FRED API (free, no key required)
Covers: Fed, ECB, BOJ, BOE, SNB, RBA, PBOC proxy rates; Fed communication metrics
Output: data/central_bank/CB_<series>_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from base import save, fetch

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {
    "CB_Fed_Funds_Rate_1d": "DFF",
    "CB_Fed_Funds_Target_Upper_1d": "DFEDTARU",
    "CB_Fed_Funds_Target_Lower_1d": "DFEDTARL",
    "CB_Fed_Balance_Sheet_1w": "WALCL",
    "CB_Fed_Reserves_1w": "WRESBAL",
    "CB_Fed_RRP_1d": "RRPONTSYD",
    "CB_Fed_TGA_1w": "WTREGEN",
    "CB_ECB_Rate_1d": "ECBDFR",
    "CB_ECB_Balance_Sheet_1w": "ECBASSETS",
    "CB_BOJ_Rate_1d": "IRSTCI01JPM156N",
    "CB_BOE_Rate_1d": "IUDSOIA",
    "CB_SNB_Rate_1d": "IRSTCI01CHM156N",
    "CB_RBA_Rate_1d": "IRSTCI01AUM156N",
    "CB_BOC_Rate_1d": "IRSTCI01CAM156N",
    "CB_RBNZ_Rate_1d": "IRSTCI01NZM156N",
    "CB_Riksbank_Rate_1d": "IRSTCI01SEM156N",
    "CB_Norges_Rate_1d": "IRSTCI01NOM156N",
    "CB_Fed_1yr_Inflation_Exp_1m": "EXPINF1YR",
    "CB_Fed_5yr_Inflation_Exp_1m": "EXPINF5YR",
    "CB_Fed_10yr_Inflation_Exp_1m": "EXPINF10YR",
    "CB_Breakeven_2yr_1d": "T5YIE",
    "CB_Breakeven_5yr_1d": "T5YIE",
    "CB_Breakeven_10yr_1d": "T10YIE",
    "CB_QT_Treasury_Holdings_1w": "TREAST",
    "CB_QT_MBS_Holdings_1w": "MBST",
    "CB_Reverse_Repo_Total_1d": "RRPONTSYD",
    "CB_M2_Money_Supply_1w": "M2SL",
    "CB_M1_Money_Supply_1w": "M1SL",
    "CB_MZM_Money_Supply_1w": "MZMSL",
    "CB_Money_Velocity_M2_1q": "M2V",
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
    print("Collecting central bank policy data...")
    ok = 0
    for filename, series_id in SERIES.items():
        try:
            df = fetch_fred(series_id, series_id)
            if not df.empty:
                save(df, "central_bank", f"{filename}.parquet")
                print(f"  {series_id}: {len(df)} rows")
                ok += 1
            else:
                print(f"  {series_id}: empty")
        except Exception as exc:
            print(f"  WARNING: {series_id} - {exc}")
    print(f"\nDone: {ok}/{len(SERIES)} series collected")


if __name__ == "__main__":
    main()
