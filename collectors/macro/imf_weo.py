"""
IMF World Economic Outlook (WEO) collector — G20 + ASEAN focus.
Source: IMF DataMapper API — free, no key required.
Covers: GDP growth, CPI inflation, unemployment, current account for major economies.
Output: data/macro/imf_weo_gdp_growth_1y.parquet
        data/macro/imf_weo_inflation_1y.parquet
        data/macro/imf_weo_unemployment_1y.parquet
        data/macro/imf_weo_current_account_1y.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

IMF_API = "https://www.imf.org/external/datamapper/api/v1"

# Indicators: IMF code -> output filename stem
INDICATORS: dict[str, str] = {
    "NGDP_RPCH": "imf_weo_gdp_growth_1y",
    "PCPIPCH": "imf_weo_inflation_1y",
    "LUR": "imf_weo_unemployment_1y",
    "BCA_NGDPD": "imf_weo_current_account_1y",
}

# G20 economies + ASEAN members (ISO2 as used by IMF DataMapper)
TARGET_COUNTRIES: list[str] = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "BRA",
    "KOR", "IDN", "MYS", "THA", "SGP", "PHL", "VNM",
    "CAN", "AUS", "MEX", "SAU", "TUR", "ZAF", "ARG", "ITA",
    "RUS", "NLD", "ESP",
]


def fetch_indicator(indicator_code: str) -> pd.DataFrame:
    """Fetch one IMF indicator (all countries), filter to TARGET_COUNTRIES."""
    url = f"{IMF_API}/{indicator_code}"
    resp = fetch(url, timeout=60)
    data = resp.json()

    values: dict[str, dict[str, float | None]] = (
        data.get("values", {}).get(indicator_code, {})
    )
    if not values:
        return pd.DataFrame()

    rows: list[dict] = []
    for country_code, year_vals in values.items():
        if country_code not in TARGET_COUNTRIES:
            continue
        for year_str, val in year_vals.items():
            if val is None:
                continue
            try:
                rows.append(
                    {
                        "date": pd.Timestamp(f"{year_str}-01-01", tz="UTC"),
                        "country": country_code,
                        "value": float(val),
                    }
                )
            except (ValueError, TypeError):
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.dropna(subset=["date"])


def main() -> None:
    for code, filename in INDICATORS.items():
        print(f"Fetching IMF WEO {code} ...")
        try:
            df = fetch_indicator(code)
            if df.empty:
                print(f"  WARNING: no data returned for {code}")
                continue
            # Pivot: index=date, columns=country, values=value
            pivot = (
                df.pivot(index="date", columns="country", values="value")
                .sort_index()
            )
            save(pivot, "macro", f"{filename}.parquet")
        except Exception as exc:
            print(f"  WARNING: {code} — {exc}")


if __name__ == "__main__":
    main()
