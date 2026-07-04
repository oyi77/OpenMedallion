"""
Labor market data collector.
Source: FRED API (free, no key required)
Covers: Unemployment, NFP, JOLTS, wages, participation, initial claims, ADP
Output: data/labor/Labor_<series>_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from base import save, fetch

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {
    "Labor_Unemployment_Rate_1m": "UNRATE",
    "Labor_Unemployment_Rate_U6_1m": "U6RATE",
    "Labor_Nonfarm_Payrolls_1m": "PAYEMS",
    "Labor_Private_Payrolls_1m": "USPRIV",
    "Labor_JOLTS_Openings_1m": "JTSJOL",
    "Labor_JOLTS_Hires_1m": "JTSHIL",
    "Labor_JOLTS_Quits_1m": "JTSQUL",
    "Labor_JOLTS_Layoffs_1m": "JTSLAL",
    "Labor_Participation_Rate_1m": "CIVPART",
    "Labor_Prime_Age_Participation_1m": "LNS11300060",
    "Labor_Avg_Hourly_Earnings_1m": "CES0500000003",
    "Labor_Avg_Weekly_Hours_1m": "AWHAETP",
    "Labor_Initial_Claims_1w": "ICSA",
    "Labor_Continuing_Claims_1w": "CCSA",
    "Labor_4wk_Avg_Claims_1w": "IC4WSA",
    "Labor_Employment_Cost_Index_1q": "ECIALLCIV",
    "Labor_Productivity_1q": "OPHNFB",
    "Labor_Unit_Labor_Cost_1q": "ULCNFB",
    "Labor_Wage_Growth_Tracker_1m": "AHETOTNSAUS",
    "Labor_Long_Term_Unemployed_1m": "UEMPLT5",
    "Labor_Discouraged_Workers_1m": "LNS15026645",
    "Labor_Part_Time_Economic_1m": "LNS12032194",
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
    print("Collecting labor market data...")
    ok = 0
    for filename, series_id in SERIES.items():
        try:
            df = fetch_fred(series_id, series_id)
            if not df.empty:
                save(df, "labor", f"{filename}.parquet")
                print(f"  {series_id}: {len(df)} rows")
                ok += 1
            else:
                print(f"  {series_id}: empty")
        except Exception as exc:
            print(f"  WARNING: {series_id} - {exc}")
    print(f"\nDone: {ok}/{len(SERIES)} series collected")


if __name__ == "__main__":
    main()
