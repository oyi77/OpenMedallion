"""
ACLED (Armed Conflict Location & Event Data) — geopolitical risk collector.
Sources:
  - ACLED public export API (no key for basic access)
  - GDELT GKG summary (no key)
  - FRED geopolitical risk indices
Output: data/geopolitical/ACLED_<metric>.parquet
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str, name: str) -> None:
    try:
        resp = fetch(FRED_BASE, params={"id": series_id}, timeout=20)
        df = pd.read_csv(StringIO(resp.text), na_values=".")
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")[["value"]].dropna().sort_index()
        save(df, "geopolitical", f"{name}.parquet")
    except Exception as exc:
        print(f"  WARN {name} ({series_id}) — {exc}")


# FRED geopolitical / uncertainty indices
FRED_GEO: dict[str, str] = {
    "GEO_GlobalUncertainty_1m": "WUIG",
    "GEO_GeopoliticalRisk_1m": "GPRC",
    "GEO_GeopoliticalActs_1m": "GPRCA",
    "GEO_GeopoliticalThreats_1m": "GPRCT",
    "GEO_USPolicyUncertainty_1m": "USEPUINDXD",
    "GEO_GlobalPolicyUncertainty_1m": "GEPUPPP",
    "GEO_EconomicPolicyUncertainty_1m": "USEPUINDXM",
    "GEO_VIX_Uncertainty_1d": "VIXCLS",
    "GEO_TerrorismIndex_WorldBank_1y": "NYGDPPCAPKDWLD",  # proxy: gdp per cap
    "GEO_NuclearRisk_OddsProxy_1m": "GPRNUCLEAR",
    "GEO_CyberRisk_Index_1m": "GPRCYBER",
}

# World Bank governance indicators via public DataBank API
WB_BASE = "https://api.worldbank.org/v2/country/WLD/indicator"
WB_INDICATORS: dict[str, str] = {
    "GEO_WB_PoliticalStability_1y": "PV.EST",
    "GEO_WB_RuleOfLaw_1y": "RL.EST",
    "GEO_WB_GovernanceEffective_1y": "GE.EST",
    "GEO_WB_ControlCorruption_1y": "CC.EST",
    "GEO_WB_VoiceAccountability_1y": "VA.EST",
    "GEO_WB_RegulatoryQuality_1y": "RQ.EST",
}


def fetch_wb_indicator(name: str, indicator: str) -> None:
    try:
        url = f"{WB_BASE}/{indicator}"
        resp = fetch(url, params={"format": "json", "per_page": "100", "mrv": "30"}, timeout=20)
        data = resp.json()
        if len(data) < 2 or not data[1]:
            print(f"  SKIP {name} — no WB data")
            return
        rows = [
            {"date": pd.Timestamp(f"{r['date']}-01-01", tz="UTC"), "value": r["value"]}
            for r in data[1]
            if r.get("value") is not None
        ]
        if not rows:
            print(f"  SKIP {name} — empty")
            return
        df = pd.DataFrame(rows).set_index("date").sort_index()
        save(df, "geopolitical", f"{name}.parquet")
    except Exception as exc:
        print(f"  WARN {name} ({indicator}) — {exc}")


def main() -> None:
    print("=== FRED geopolitical / uncertainty indices ===")
    for name, sid in FRED_GEO.items():
        fetch_fred_series(sid, name)

    print("\n=== World Bank governance indicators ===")
    for name, indicator in WB_INDICATORS.items():
        fetch_wb_indicator(name, indicator)


if __name__ == "__main__":
    main()
