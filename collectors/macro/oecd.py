"""
OECD macro data collector.
Source: OECD Data API v1 — free, no API key required.
Covers: GDP growth, inflation, unemployment, current account for OECD countries.
Output: data/macro/OECD_<indicator>_<country>_1q.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

OECD_API = "https://sdmx.oecd.org/public/rest/data"

# Dataset IDs and key structures
# Format: DATASET/LOCATION.SUBJECT.MEASURE.FREQUENCY/all?format=csvfilewithlabels
INDICATORS = {
    "GDP_Growth": ("QNA", "B1_GE.GPSA.Q"),
    "CPI_Inflation": ("PRICES_CPI", "CPALTT01.IXOBSA.Q"),
    "Unemployment": ("STLABOUR", "UNRTOT.STSA.M"),
    "CurrentAccount": ("MEI_BOP6", "B6BLTT02.CXCUSA.Q"),
    "InterestRate_ST": ("STLABOUR", None),  # will use MEI
    "HousePrice": ("RPPI", "RP_NSA.Q"),
}

# Simpler OECD approach using their JSON API
OECD_JSON = "https://stats.oecd.org/sdmx-json/data/{dataset}/{filter}/all"

OECD_DATASETS = {
    "GDP_Growth_1q": {
        "dataset": "QNA",
        "filter": "AUS+AUT+BEL+CAN+CHL+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA+COL+CRI+CZE+IDN.B1_GE.GPSA.Q",
    },
    "CPI_1m": {
        "dataset": "PRICES_CPI",
        "filter": "AUS+AUT+BEL+CAN+CHL+CZE+DNK+FIN+FRA+DEU+GRC+HUN+IRL+ISR+ITA+JPN+KOR+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA.CPALTT01.IXOBSA.M",
    },
    "Unemployment_1m": {
        "dataset": "STLABOUR",
        "filter": "AUS+AUT+BEL+CAN+CZE+DNK+FIN+FRA+DEU+GRC+HUN+IRL+ISL+ISR+ITA+JPN+KOR+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA.UNRTOT.STSA.M",
    },
}


def fetch_oecd_json(dataset: str, filter_key: str) -> pd.DataFrame:
    url = OECD_JSON.format(dataset=dataset, filter=filter_key)
    params = {"startTime": "2000-Q1", "dimensionAtObservation": "allDimensions"}
    resp = fetch(url, params=params, timeout=60)
    data = resp.json()

    if "dataSets" not in data or not data["dataSets"]:
        return pd.DataFrame()

    ds = data["dataSets"][0]
    structure = data["structure"]

    # Parse dimensions
    dims = structure["dimensions"]["observation"]
    time_dim = next((d for d in dims if d["id"] == "TIME_PERIOD"), None)
    if time_dim is None:
        return pd.DataFrame()

    time_values = [t["id"] for t in time_dim["values"]]

    rows = []
    for key, obs in ds.get("observations", {}).items():
        parts = key.split(":")
        # time index is last dimension
        time_idx = int(parts[-1])
        val = obs[0] if obs else None
        if val is not None and time_idx < len(time_values):
            # Decode other dimensions for country
            country_idx = int(parts[0])
            country_vals = dims[0]["values"]
            country = country_vals[country_idx]["id"] if country_idx < len(country_vals) else "UNK"
            rows.append({
                "date": time_values[time_idx],
                "country": country,
                "value": float(val),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    return df


def main() -> None:
    for name, cfg in OECD_DATASETS.items():
        print(f"Fetching OECD {name} ...")
        try:
            df = fetch_oecd_json(cfg["dataset"], cfg["filter"])
            if df.empty:
                print(f"  WARNING: no data for {name}")
                continue

            # Split by country and save
            for country, grp in df.groupby("country"):
                grp = grp.set_index("date").sort_index()[["value"]]
                save(grp, "macro", f"OECD_{name}_{country}.parquet")

        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
