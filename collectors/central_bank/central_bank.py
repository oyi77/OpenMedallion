"""
Central bank policy rates and communication indices collector.
Source: FRED API (free, no key required)
Covers: Fed, ECB, BOJ, BOE, SNB, RBA, PBOC proxy rates; Fed communication metrics
Output: data/central_bank/CB_<series>_1d.parquet
        data/central_bank/central_bank_rates_1d.parquet (consolidated policy rates)
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from base import fetch, save, to_datetime_index

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

# ---------------------------------------------------------------------------
# Consolidated policy-rate series → single parquet
# FRED series IDs for the four major central-bank policy rates
# ---------------------------------------------------------------------------
_RATE_SERIES: dict[str, str] = {
    "FEDFUNDS": "Fed",
    "ECBDFR": "ECB",           # ECB Deposit Facility Rate
    "IUDSOIA": "BOE",          # Sterling Overnight Index Avg (BOE proxy)
    "IRSTCI01JPM156N": "BOJ",  # Japan Policy Rate
}


def fetch_fred(series_id: str, value_col: str) -> pd.DataFrame:
    url = FRED_CSV.format(series_id)
    resp = fetch(url, timeout=30)
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", value_col]
    df = df[df[value_col] != "."].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=[value_col]).reset_index(drop=True)


def _fetch_single_rate(series_id: str, label: str) -> pd.DataFrame | None:
    """Download one FRED series and return with central_bank + rate_pct columns."""
    try:
        url = FRED_CSV.format(series_id)
        resp = fetch(url, timeout=30)
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "rate_pct"]
        df = df[df["rate_pct"] != "."].copy()
        df["rate_pct"] = pd.to_numeric(df["rate_pct"], errors="coerce")
        df = df.dropna(subset=["rate_pct"])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["central_bank"] = label
        return df[["date", "central_bank", "rate_pct"]]
    except Exception as exc:
        print(f"  FAIL {series_id} ({label}): {exc}")
        return None


def collect_rates_consolidated() -> None:
    """Fetch major CB policy rates into central_bank_rates_1d.parquet."""
    frames: list[pd.DataFrame] = []
    for series_id, label in _RATE_SERIES.items():
        print(f"  Fetching {label} ({series_id})...", end=" ")
        part = _fetch_single_rate(series_id, label)
        if part is not None and not part.empty:
            frames.append(part)
            print(f"{len(part)} rows")
        else:
            print("empty/failed")
    if not frames:
        print("  WARNING: no rate data collected")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "central_bank"]).reset_index(drop=True)
    df = to_datetime_index(df, col="date")
    save(df, "central_bank", "central_bank_rates_1d.parquet")
    print(f"  Consolidated rates: {len(df)} rows ({len(_RATE_SERIES)} banks)")


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
    print("\nCollecting consolidated policy rates...")
    collect_rates_consolidated()


if __name__ == "__main__":
    main()
