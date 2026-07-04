"""
Credit markets data collector.
Source: FRED API (free, no key required)
Covers: Corporate spreads, HY/IG OAS, TED spread, SOFR, fed funds, CDS proxies
Output: data/credit/Credit_<series>_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from base import save, fetch

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES = {
    "Credit_HY_OAS_1d": "BAMLH0A0HYM2",
    "Credit_IG_OAS_1d": "BAMLC0A0CM",
    "Credit_IG_AAA_OAS_1d": "BAMLC0A1CAAAEY",
    "Credit_IG_BBB_OAS_1d": "BAMLC0A4CBBBEY",
    "Credit_HY_BB_OAS_1d": "BAMLH0A1HYBBEY",
    "Credit_HY_B_OAS_1d": "BAMLH0A2HYBEY",
    "Credit_HY_CCC_OAS_1d": "BAMLH0A3HYCEY",
    "Credit_EM_Sovereign_OAS_1d": "BAMLEMRECRPIEMSTRR",
    "Credit_TED_Spread_1d": "TEDRATE",
    "Credit_SOFR_1d": "SOFR",
    "Credit_Fed_Funds_Rate_1d": "DFF",
    "Credit_3m_Tbill_1d": "TB3MS",
    "Credit_2yr_Treasury_1d": "DGS2",
    "Credit_10yr_Treasury_1d": "DGS10",
    "Credit_30yr_Treasury_1d": "DGS30",
    "Credit_Yield_Curve_10y2y_1d": "T10Y2Y",
    "Credit_Yield_Curve_10y3m_1d": "T10Y3M",
    "Credit_LIBOR_3m_1d": "USD3MTD156N",
    "Credit_LIBOR_OIS_Spread_1d": "LLRI3M",
    "Credit_Commercial_Paper_3m_1d": "DCPF3M",
    "Credit_Mortgage_Backed_OAS_1d": "BAMLMBS0A0CMB",
    "Credit_Muni_OAS_1d": "BAMLM0A0CMU",
    "Credit_CMB_OAS_1d": "BAMLMBS0A0CMB",
    "Credit_IG_Euro_OAS_1d": "BAMLHE00EHY0EY",
    "Credit_Investment_Grade_Issuance_1m": "NCBCMDPMVCE",
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
    print("Collecting credit market data...")
    ok = 0
    for filename, series_id in SERIES.items():
        try:
            df = fetch_fred(series_id, series_id)
            if not df.empty:
                save(df, "credit", f"{filename}.parquet")
                print(f"  {series_id}: {len(df)} rows")
                ok += 1
            else:
                print(f"  {series_id}: empty")
        except Exception as exc:
            print(f"  WARNING: {series_id} - {exc}")
    print(f"\nDone: {ok}/{len(SERIES)} series collected")


if __name__ == "__main__":
    main()
