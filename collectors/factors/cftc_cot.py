"""
CFTC Commitments of Traders (COT) collector.
Source: CFTC.gov — free weekly CSV downloads, no API key needed.
Covers: Legacy COT for futures-only: Gold, Silver, Oil (WTI), S&P500, Nasdaq,
        10Y Treasury, EUR, JPY, GBP, AUD, BTC.
Output: data/factors/COT_<asset>_1w.parquet
"""
from __future__ import annotations
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# CFTC annual zip files — fetch last 5 years
COT_BASE = "https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip"

# Market names to filter (as they appear in CFTC data)
MARKETS_OF_INTEREST = {
    "GOLD": "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER": "SILVER - COMMODITY EXCHANGE INC.",
    "CRUDE_OIL_WTI": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    "SP500": "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "NASDAQ": "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE",
    "TREASURY_10Y": "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "GBP": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "BTC": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
    "NATURAL_GAS": "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
    "COPPER": "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
    "VIX": "CBOE VOLATILITY INDEX - CHICAGO FUTURES EXCHANGE",
}

KEEP_COLS = [
    "Market_and_Exchange_Names",
    "As_of_Date_In_Form_YYMMDD",
    "NonComm_Positions_Long_All",
    "NonComm_Positions_Short_All",
    "Comm_Positions_Long_All",
    "Comm_Positions_Short_All",
    "Open_Interest_All",
    "NonComm_Net",  # derived
]


def fetch_year(year: int) -> pd.DataFrame | None:
    url = COT_BASE.format(year=year)
    print(f"  Fetching COT {year} ...")
    try:
        resp = fetch(url, timeout=60)
    except Exception as exc:
        print(f"    Skipping {year}: {exc}")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next((n for n in zf.namelist() if n.endswith(".csv") or n.endswith(".txt")), None)
        if not csv_name:
            csv_name = zf.namelist()[0]
        raw = zf.read(csv_name)

    df = pd.read_excel(io.BytesIO(raw), engine="xlrd")
    return df


def main() -> None:
    import datetime
    current_year = datetime.date.today().year
    years = range(current_year - 9, current_year + 1)

    all_frames: list[pd.DataFrame] = []
    for year in years:
        df = fetch_year(year)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        print("WARNING: No COT data fetched")
        return

    full = pd.concat(all_frames, ignore_index=True)

    # Normalize column names (vary slightly across years)
    full.columns = [c.strip().replace(" ", "_") for c in full.columns]

    # Date column
    date_col = next((c for c in full.columns if "date" in c.lower() or "as_of" in c.lower()), None)
    if date_col is None:
        print("WARNING: Cannot find date column")
        return

    full[date_col] = pd.to_datetime(full[date_col], errors="coerce")
    full = full.dropna(subset=[date_col])

    market_col = next((c for c in full.columns if "market" in c.lower() and "exchange" in c.lower()), None)
    if market_col is None:
        print("WARNING: Cannot find market column")
        return

    # Extract and save per market
    for asset_name, market_pattern in MARKETS_OF_INTEREST.items():
        mask = full[market_col].str.upper().str.contains(market_pattern.split(" - ")[0], na=False)
        sub = full[mask].copy()
        if sub.empty:
            print(f"  No data for {asset_name}")
            continue

        sub = sub.set_index(date_col).sort_index()
        sub.index = sub.index.tz_localize("UTC")

        # Keep numeric cols only + derive net positioning
        num_cols = [c for c in sub.columns if sub[c].dtype in ["float64", "int64"] or
                    pd.to_numeric(sub[c], errors="coerce").notna().sum() > len(sub) * 0.5]
        sub = sub[num_cols].apply(pd.to_numeric, errors="coerce")

        # Add net non-commercial (speculators) as key signal
        long_col = next((c for c in sub.columns if "noncomm" in c.lower() and "long" in c.lower()), None)
        short_col = next((c for c in sub.columns if "noncomm" in c.lower() and "short" in c.lower()), None)
        if long_col and short_col:
            sub["noncomm_net"] = sub[long_col] - sub[short_col]

        sub = sub.dropna(how="all")
        save(sub, "factors", f"COT_{asset_name}_1w.parquet")


if __name__ == "__main__":
    main()
