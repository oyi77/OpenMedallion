"""
ASEAN macro data collector via World Bank API (no key required).
Covers: Indonesia, Thailand, Vietnam, Malaysia, Philippines, Singapore
Indicators: GDP, CPI, unemployment, exports, imports, current account, FX rate
Output: data/macro/asean/<COUNTRY>_<indicator_slug>_1y.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

SLEEP = 0.3

# World Bank API — no key, public JSON endpoint
WB_API = (
    "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    "?format=json&mrv=50&per_page=50"
)

# ASEAN-6 ISO2 codes
COUNTRIES: dict[str, str] = {
    "ID": "Indonesia",
    "TH": "Thailand",
    "VN": "Vietnam",
    "MY": "Malaysia",
    "PH": "Philippines",
    "SG": "Singapore",
}

# (WB indicator code, output slug)
INDICATORS: list[tuple[str, str]] = [
    ("NY.GDP.MKTP.CD",  "gdp_current_usd"),
    ("FP.CPI.TOTL.ZG",  "cpi_inflation_pct"),
    ("SL.UEM.TOTL.ZS",  "unemployment_pct"),
    ("NE.EXP.GNFS.ZS",  "exports_pct_gdp"),
    ("NE.IMP.GNFS.ZS",  "imports_pct_gdp"),
    ("BX.CAB.XOKA.CD",  "current_account_usd"),
    ("PA.NUS.FCRF",     "fx_rate_lcu_per_usd"),
]


def fetch_wb_indicator(country_code: str, indicator: str) -> pd.DataFrame:
    """Fetch one World Bank indicator for one country.

    Returns a DataFrame with DatetimeIndex and a single 'value' column,
    or an empty DataFrame on error / no data.
    """
    url = WB_API.format(country=country_code, indicator=indicator)
    resp = fetch(url)
    payload = resp.json()

    # World Bank returns [metadata_dict, data_list]
    if not payload or len(payload) < 2 or not payload[1]:
        return pd.DataFrame()

    rows: list[dict] = []
    for record in payload[1]:
        raw_value = record.get("value")
        if raw_value is None:
            continue
        try:
            rows.append(
                {
                    "date": pd.Timestamp(f"{record['date']}-01-01", tz="UTC"),
                    "value": float(raw_value),
                }
            )
        except (ValueError, TypeError, KeyError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def main() -> None:
    total_ok = 0
    total_attempted = 0

    for iso2, country_name in COUNTRIES.items():
        print(f"\n=== {country_name} ({iso2}) ===")
        ok = 0

        for wb_code, slug in INDICATORS:
            total_attempted += 1
            try:
                df = fetch_wb_indicator(iso2, wb_code)
                if df.empty:
                    print(f"  SKIP {slug} — no data returned")
                else:
                    filename = f"{country_name.upper()}_{slug}_1y.parquet"
                    save(df, "macro/asean", filename)
                    ok += 1
                    total_ok += 1
            except Exception as exc:
                print(f"  WARNING {slug} — {exc}")

            time.sleep(SLEEP)

        print(f"  ✓ {ok}/{len(INDICATORS)} indicators saved")

    print(f"\nDone. {total_ok}/{total_attempted} series saved to data/macro/asean/")


if __name__ == "__main__":
    main()
