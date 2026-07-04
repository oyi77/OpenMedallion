"""
IMF World Economic Outlook (WEO) collector.
Source: IMF Data API — free, no key required.
Also fetches IFS (International Financial Statistics).
Covers: GDP, inflation forecasts, current account, debt for 190+ countries.
Output: data/macro/IMF_WEO_<indicator>_1y.parquet
"""
from __future__ import annotations
import sys
import io
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# IMF WEO is published as a bulk download twice per year
IMF_WEO_URL = "https://www.imf.org/en/Publications/WEO/weo-database/2024/April/download-entire-database"
# Direct download of WEO data (tab-separated)
IMF_WEO_TSV = "https://www.imf.org/external/pubs/ft/weo/2024/01/weodata/WEOApr2024all.ashx"

# IMF JSON API for real-time data
IMF_API = "https://www.imf.org/external/datamapper/api/v1"

# Key indicators available in IMF DataMapper
IMF_INDICATORS = {
    "NGDP_RPCH": "GDP_Growth_Real",          # Real GDP growth
    "PCPIPCH": "CPI_Inflation",               # CPI % change
    "LUR": "Unemployment_Rate",               # Unemployment
    "BCA_NGDPDP": "CurrentAccount_PctGDP",   # Current account
    "GGXWDG_NGDP": "Govt_Debt_PctGDP",       # Gross debt
    "NGDPDPC": "GDP_PerCapita_USD",           # GDP per capita
    "PPPGDP": "GDP_PPP_USD",                  # GDP PPP
}


def fetch_imf_indicator(indicator_code: str, indicator_name: str) -> pd.DataFrame:
    """Fetch indicator for all countries from IMF DataMapper API."""
    url = f"{IMF_API}/indicators/{indicator_code}"
    resp = fetch(url, timeout=30)
    meta = resp.json()

    # Fetch actual data
    url_data = f"{IMF_API}/{indicator_code}"
    resp_data = fetch(url_data, timeout=60)
    data = resp_data.json()

    values = data.get("values", {}).get(indicator_code, {})
    if not values:
        return pd.DataFrame()

    rows = []
    for country_code, year_vals in values.items():
        for year, val in year_vals.items():
            if val is not None:
                rows.append({
                    "date": year,
                    "country": country_code,
                    "value": float(val),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y", utc=True)
    return df.dropna(subset=["date"])


def main() -> None:
    for code, name in IMF_INDICATORS.items():
        print(f"Fetching IMF {name} ({code}) ...")
        try:
            df = fetch_imf_indicator(code, name)
            if df.empty:
                print(f"  WARNING: no data for {name}")
                continue

            # Save combined (all countries in one file — more efficient for WEO)
            df_pivot = df.pivot(index="date", columns="country", values="value").sort_index()
            save(df_pivot, "macro", f"IMF_WEO_{name}_1y.parquet")

        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
