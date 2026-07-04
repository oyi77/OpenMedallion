"""
USDA agricultural commodity data collector.
Source: USDA ERS & NASS — free, no API key for bulk CSV downloads.
Covers: Feed grain prices, food price index, crop production estimates.
Output: data/agriculture/USDA_<series>_1m.parquet
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# USDA ERS direct CSV downloads (no API key)
USDA_ERS_DATASETS = {
    "USDA_FoodPriceOutlook_1m": "https://www.ers.usda.gov/webdocs/DataFiles/50048/CPIforecast.xlsx",
    "USDA_FoodCPI_History_1m": "https://www.ers.usda.gov/webdocs/DataFiles/50048/CPIforecast.xlsx",
}

# USDA NASS QuickStats API — free, requires API key
NASS_API = "https://quickstats.nass.usda.gov/api/api_GET/"

# USDA PSD (Production, Supply, Distribution) — free bulk download
USDA_PSD_URL = "https://apps.fas.usda.gov/psdonline/downloads/psd_alldata_csv.zip"

# World commodity prices from World Bank Pinks Sheet (free)
PINK_SHEET_URL = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"


def fetch_world_bank_commodities() -> pd.DataFrame:
    """Parse World Bank Pink Sheet commodity prices."""
    print("  Downloading World Bank commodity pink sheet ...")
    resp = fetch(PINK_SHEET_URL, timeout=120)
    xls = pd.ExcelFile(io.BytesIO(resp.content))

    # Use 'Monthly Prices' sheet
    sheet = next((s for s in xls.sheet_names if "Monthly" in s or "Price" in s), xls.sheet_names[0])
    df_raw = pd.read_excel(io.BytesIO(resp.content), sheet_name=sheet, header=None)

    # Find header row
    header_row = 0
    for i, row in df_raw.iterrows():
        if str(row.iloc[0]).strip().upper() in ["DATE", "MONTH", "PERIOD"] or \
           any(str(v).strip() in ["Jan", "Jan-", "1960"] for v in row):
            header_row = i
            break

    df = pd.read_excel(io.BytesIO(resp.content), sheet_name=sheet, header=header_row, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    df = df[df.index.notna()].sort_index()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def fetch_usda_psd_zip() -> dict[str, pd.DataFrame]:
    """Fetch USDA PSD bulk data — crop production/supply/distribution."""
    print("  Downloading USDA PSD bulk data (~30MB) ...")
    resp = fetch(USDA_PSD_URL, timeout=180)
    import zipfile
    results = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                raw = zf.read(name)
                try:
                    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
                    # Keep key crops
                    if "commodity_description" in df.columns:
                        key_crops = ["Corn", "Wheat", "Soybeans", "Rice", "Cotton", "Sugar", "Palm Oil", "Coffee"]
                        for crop in key_crops:
                            sub = df[df["commodity_description"].str.contains(crop, na=False, case=False)]
                            if not sub.empty:
                                # Pivot to wide format
                                if "marketing_year" in sub.columns and "value" in sub.columns:
                                    sub["marketing_year"] = pd.to_datetime(
                                        sub["marketing_year"].astype(str), format="%Y", errors="coerce", utc=True
                                    )
                                    sub = sub.dropna(subset=["marketing_year"])
                                    sub = sub.set_index("marketing_year").sort_index()
                                    results[f"USDA_PSD_{crop.replace(' ', '_')}_1y"] = sub
                except Exception:
                    pass
    return results


def main() -> None:
    # World Bank commodity prices (most comprehensive free source)
    print("Fetching World Bank commodity pink sheet ...")
    try:
        df_pink = fetch_world_bank_commodities()
        if not df_pink.empty:
            save(df_pink, "agriculture", "WorldBank_Commodity_Prices_1m.parquet")
    except Exception as exc:
        print(f"  WARNING: Pink sheet — {exc}")

    # USDA PSD crop data
    print("\nFetching USDA PSD crop production data ...")
    try:
        psd_data = fetch_usda_psd_zip()
        for name, df in psd_data.items():
            save(df, "agriculture", f"{name}.parquet")
    except Exception as exc:
        print(f"  WARNING: USDA PSD — {exc}")


if __name__ == "__main__":
    main()
