"""
F1 race results per season via Ergast/jolpi mirror (no API key).
Source: https://api.jolpi.ca/ergast/f1/{year}/results/
Output: data/sports/F1_{year}_results_1d.parquet  (one file per season)
Seasons: 2018–2024
Columns: date, race_name, driver, constructor, position, points, laps
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

_BASE = "https://api.jolpi.ca/ergast/f1/{year}/results/"
_SEASONS = list(range(2018, 2025))   # 2018-2024 inclusive
_PAGE_SIZE = 100
_SLEEP = 0.4


def _fetch_season_races(year: int) -> list[dict]:
    """
    Return all Race dicts for a season, paginating through the API.

    Ergast returns races (each with a Results list).  We walk pages until
    all races are fetched.
    """
    url = _BASE.format(year=year)
    races: list[dict] = []
    offset = 0

    while True:
        resp = fetch(url, params={"format": "json", "limit": _PAGE_SIZE, "offset": offset})
        data = resp.json()["MRData"]
        page_races = data["RaceTable"]["Races"]
        races.extend(page_races)

        total = int(data["total"])
        offset += _PAGE_SIZE
        if offset >= total:
            break
        time.sleep(_SLEEP)

    return races


def _races_to_df(races: list[dict], year: int) -> pd.DataFrame:
    """Flatten race+result dicts into a tidy DataFrame."""
    rows: list[dict] = []
    for race in races:
        race_date = race.get("date", "")
        race_name = race.get("raceName", "")
        for res in race.get("Results", []):
            driver = res["Driver"]
            rows.append({
                "date": race_date,
                "race_name": race_name,
                "driver": f"{driver['givenName']} {driver['familyName']}",
                "constructor": res["Constructor"]["name"],
                "position": int(res["position"]),
                "points": float(res["points"]),
                "laps": int(res["laps"]),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.dropna(subset=["date"]).sort_values(["date", "position"]).set_index("date")


def collect_f1_results() -> None:
    """Fetch F1 results for each season and save one parquet per year."""
    print(f"F1 seasons: {_SEASONS[0]}–{_SEASONS[-1]}")
    saved = 0

    for year in _SEASONS:
        print(f"  Fetching {year} …")
        try:
            races = _fetch_season_races(year)
            print(f"    {len(races)} races")
            df = _races_to_df(races, year)
            if df.empty:
                print(f"  SKIP {year}: no rows")
                continue
            save(df, "sports", f"F1_{year}_results_1d.parquet")
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING {year}: {exc}")
        time.sleep(_SLEEP)

    print(f"\nSaved {saved} F1 season parquet files")


def main() -> None:
    print("Fetching: F1 race results (Ergast/jolpi, 2018-2024)")
    collect_f1_results()


if __name__ == "__main__":
    main()
