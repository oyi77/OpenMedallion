"""
NFL scores and game results collector.
Sources:
  - ESPN public API (no key) — concurrent fetch via ThreadPoolExecutor
Output: data/sports/NFL_Scores_<years>.parquet
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# (seasontype, max_weeks): 2=regular, 3=playoffs
SEASON_TYPES = [(2, 18), (3, 5)]

CURRENT_YEAR = pd.Timestamp.now("UTC").year
SEASONS = list(range(CURRENT_YEAR - 9, CURRENT_YEAR + 1))


def fetch_week(season: int, seasontype: int, week: int) -> list[dict]:
    try:
        resp = fetch(
            ESPN_BASE,
            params={"seasontype": seasontype, "week": week, "limit": 20, "dates": season},
            timeout=10,
        )
        events = resp.json().get("events", [])
    except Exception:
        return []

    rows = []
    for evt in events:
        comps = evt.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])
        if len(competitors) < 2:
            continue
        status = evt.get("status", {}).get("type", {}).get("name", "")
        if "FINAL" not in status:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        date_str = evt.get("date")
        rows.append({
            "date": pd.Timestamp(date_str, tz="UTC") if date_str else pd.NaT,
            "season": season,
            "seasontype": seasontype,
            "week": week,
            "home_team": home.get("team", {}).get("abbreviation", ""),
            "away_team": away.get("team", {}).get("abbreviation", ""),
            "home_score": int(home.get("score", 0) or 0),
            "away_score": int(away.get("score", 0) or 0),
            "attendance": comps.get("attendance"),
            "venue": comps.get("venue", {}).get("fullName", ""),
        })
    return rows


def main() -> None:
    print(f"=== NFL scores (ESPN) — seasons {SEASONS[0]}-{SEASONS[-1]} ===")

    tasks = [
        (season, st, week)
        for season in SEASONS
        for st, max_week in SEASON_TYPES
        for week in range(1, max_week + 1)
    ]
    print(f"  Fetching {len(tasks)} week-slots concurrently …")

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(fetch_week, s, st, w): (s, st, w) for s, st, w in tasks}
        for done in as_completed(futures):
            all_rows.extend(done.result())

    if not all_rows:
        print("  No completed games found")
        return

    df = pd.DataFrame(all_rows).dropna(subset=["date"])
    df = df.set_index("date").sort_index()
    label = f"{SEASONS[0]}_{SEASONS[-1]}"
    save(df, "sports", f"NFL_Scores_{label}.parquet")
    print(f"  Saved {len(df)} rows to data/sports/NFL_Scores_{label}.parquet")


if __name__ == "__main__":
    main()
