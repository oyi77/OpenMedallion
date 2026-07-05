"""
ECB key interest rates and M3 money supply collector.
Source: ECB Data Portal SDMX REST API — free, no key required.
Covers: Deposit facility rate, main refinancing rate, marginal lending rate, M3.
Output: data/macro/ecb_rates_1m.parquet
        data/macro/ecb_m3_1m.parquet
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

ECB_API = "https://data-api.ecb.europa.eu/service/data"

# ECB key interest rates (FM dataflow)
# B.U2.EUR.4F.KR.{rate_code}.LEV
ECB_RATES: dict[str, str] = {
    "DFR": "ecb_deposit_facility_rate",   # Deposit facility rate
    "MRR_FR": "ecb_main_refinancing_rate", # Main refinancing operations (fixed)
    "MLFR": "ecb_marginal_lending_rate",   # Marginal lending facility rate
}

# ECB M3 money supply (BSI dataflow)
# M.U2.Y.V.M30.X.1.U2.2300.Z01.E
ECB_M3_SERIES = "M.U2.Y.V.M30.X.1.U2.2300.Z01.E"


def parse_ecb_csv(content: bytes) -> pd.DataFrame:
    """Parse ECB CSV response (key;date;value format or standard SDMX-CSV)."""
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Skip comment lines starting with '#'
    data_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    if not data_lines:
        return pd.DataFrame()

    block = "\n".join(data_lines)
    # ECB uses semicolon-delimited SDMX-CSV
    sep = ";" if ";" in data_lines[0] else ","
    df = pd.read_csv(io.StringIO(block), sep=sep, low_memory=False)

    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]

    time_col = next(
        (c for c in df.columns if "TIME_PERIOD" in c or c == "TIME"),
        None,
    )
    val_col = next(
        (c for c in df.columns if "OBS_VALUE" in c or c == "VALUE"),
        None,
    )

    if time_col is None or val_col is None:
        return pd.DataFrame()

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[time_col, val_col])

    return (
        df[[time_col, val_col]]
        .rename(columns={time_col: "date", val_col: "value"})
        .set_index("date")
        .sort_index()
    )


def fetch_ecb_rate(rate_code: str) -> pd.DataFrame:
    """Fetch one ECB interest rate series as CSV."""
    series_key = f"B.U2.EUR.4F.KR.{rate_code}.LEV"
    url = f"{ECB_API}/FM/{series_key}"
    params = {"format": "csvdata", "startPeriod": "1999-01-01"}
    resp = fetch(url, params=params, timeout=60)
    return parse_ecb_csv(resp.content)


def fetch_ecb_m3() -> pd.DataFrame:
    """Fetch ECB M3 money supply series as CSV."""
    url = f"{ECB_API}/BSI/{ECB_M3_SERIES}"
    params = {"format": "csvdata", "startPeriod": "1999-01-01"}
    resp = fetch(url, params=params, timeout=60)
    return parse_ecb_csv(resp.content)


def main() -> None:
    # --- Key interest rates ---
    rate_frames: dict[str, pd.DataFrame] = {}
    for rate_code, label in ECB_RATES.items():
        print(f"Fetching ECB rate: {label} ({rate_code}) ...")
        try:
            df = fetch_ecb_rate(rate_code)
            if df.empty:
                print(f"  WARNING: no data for {rate_code}")
                continue
            rate_frames[label] = df.rename(columns={"value": label})
            print(f"  {len(df):,} rows")
        except Exception as exc:
            print(f"  WARNING: {rate_code} — {exc}")

    if rate_frames:
        combined = pd.concat(list(rate_frames.values()), axis=1).sort_index()
        save(combined, "macro", "ecb_rates_1m.parquet")
    else:
        print("  WARNING: all ECB rate series failed — skipping ecb_rates_1m.parquet")

    # --- M3 money supply ---
    print("Fetching ECB M3 money supply ...")
    try:
        df_m3 = fetch_ecb_m3()
        if df_m3.empty:
            print("  WARNING: no M3 data returned")
        else:
            print(f"  {len(df_m3):,} rows")
            save(df_m3, "macro", "ecb_m3_1m.parquet")
    except Exception as exc:
        print(f"  WARNING: M3 — {exc}")


if __name__ == "__main__":
    main()
