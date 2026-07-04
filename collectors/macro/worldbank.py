"""
World Bank Open Data collector.
Source: World Bank API v2 — free, no API key.
Covers: GDP, inflation, population, unemployment, trade, FDI, debt for 50+ countries.
Output: data/macro/WB_<indicator>_<iso3>_1y.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

WB_API = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

# Key indicators: code -> human name
INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP_USD",
    "NY.GDP.MKTP.KD.ZG": "GDP_Growth",
    "FP.CPI.TOTL.ZG": "CPI_Inflation",
    "SL.UEM.TOTL.ZS": "Unemployment_Rate",
    "NE.TRD.GNFS.ZS": "Trade_PctGDP",
    "BX.KLT.DINV.CD.WD": "FDI_Inflows_USD",
    "GC.DOD.TOTL.GD.ZS": "Govt_Debt_PctGDP",
    "NY.GNP.PCAP.CD": "GNI_PerCapita",
    "SP.POP.TOTL": "Population",
    "FM.LBL.BMNY.GD.ZS": "M2_PctGDP",
    "FR.INR.RINR": "Real_Interest_Rate",
    "PA.NUS.FCRF": "FX_OfficialRate_USD",
    "BN.CAB.XOKA.GD.ZS": "CurrentAccount_PctGDP",
}

# Focus countries — major economies + Indonesia (your focus)
COUNTRIES = [
    "US", "CN", "JP", "DE", "GB", "FR", "IN", "BR", "CA", "KR",
    "AU", "MX", "ID", "SA", "TR", "NL", "CH", "AR", "SE", "PL",
    "TH", "MY", "PH", "VN", "SG", "ZA", "NG", "EG", "PK", "BD",
    "RU", "IT", "ES", "IA", "HK",
]


def fetch_indicator(indicator_code: str, country: str) -> pd.DataFrame:
    url = WB_API.format(country=country, indicator=indicator_code)
    params = {"format": "json", "per_page": 100, "mrv": 60}
    resp = fetch(url, params=params)
    data = resp.json()
    if not data or len(data) < 2 or not data[1]:
        return pd.DataFrame()
    records = [
        {"date": r["date"], "value": r["value"]}
        for r in data[1]
        if r["value"] is not None
    ]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y", utc=True)
    df = df.set_index("date").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def main() -> None:
    for code, name in INDICATORS.items():
        print(f"\nFetching World Bank {name} ({code}) for {len(COUNTRIES)} countries ...")
        for country in COUNTRIES:
            try:
                df = fetch_indicator(code, country)
                if not df.empty:
                    save(df, "macro", f"WB_{name}_{country}_1y.parquet")
            except Exception as exc:
                print(f"  WARNING: {country}/{name} — {exc}")


if __name__ == "__main__":
    main()
