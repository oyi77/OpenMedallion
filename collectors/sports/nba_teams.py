"""
NBA per-team game logs via nba_api (official NBA stats, no API key).
Source: https://www.nba.com/stats (via nba_api wrapper)
Output: data/sports/NBA_{ABBREV}_1d.parquet  (one file per team, 30 files)
Seasons: 2018-19 through 2023-24
Columns: date, home_team, away_team, home_score, away_score, status
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

# nba_api installed via: pip3 install nba_api --break-system-packages
from nba_api.stats.endpoints import leaguegamefinder  # type: ignore[import]
from nba_api.stats.static import teams as nba_teams_static  # type: ignore[import]

_SEASONS = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24",
]
_SLEEP = 0.5


def _fetch_season(season: str) -> pd.DataFrame:
    """Return all regular season + playoff game rows for one season."""
    gf = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        timeout=60,
    )
    return gf.get_data_frames()[0]


def _build_team_df(df_season: pd.DataFrame, team_abbrev: str) -> pd.DataFrame:
    """
    Extract game-level rows for one team.

    nba_api returns one row per team per game.  We keep only the home-team
    row (MATCHUP contains "vs.") so each game appears exactly once.
    """
    team_rows = df_season[df_season["TEAM_ABBREVIATION"] == team_abbrev].copy()
    # Cast to plain str to avoid ArrowStringArray regex quirks
    team_rows["MATCHUP"] = team_rows["MATCHUP"].astype(str)
    # home rows: "BOS vs. DAL" (away rows use "@")
    home = team_rows[team_rows["MATCHUP"].str.contains("vs.", na=False, regex=False)].copy()
    home["home_team"] = team_abbrev
    home["away_team"] = home["MATCHUP"].str.extract(r"vs\.\s+(\w+)")
    home["home_score"] = pd.to_numeric(home["PTS"], errors="coerce")

    # Get away team score by joining on GAME_ID from the away team's row
    away_rows = df_season[
        df_season["GAME_ID"].isin(home["GAME_ID"])
        & (df_season["TEAM_ABBREVIATION"] != team_abbrev)
    ][["GAME_ID", "PTS"]].rename(columns={"PTS": "away_score_raw"})

    home = home.merge(away_rows, on="GAME_ID", how="left")
    home["away_score"] = pd.to_numeric(home["away_score_raw"], errors="coerce")
    home["status"] = "Final"
    home["date"] = pd.to_datetime(home["GAME_DATE"], errors="coerce", utc=True)

    result = home[["date", "home_team", "away_team", "home_score", "away_score", "status"]].copy()
    return result.dropna(subset=["date"]).sort_values("date").set_index("date")


def collect_nba_teams() -> None:
    """Fetch all seasons, split by team, save one parquet per team."""
    all_teams = nba_teams_static.get_teams()
    abbrevs = [t["abbreviation"] for t in all_teams]
    print(f"NBA teams: {len(abbrevs)}  seasons: {len(_SEASONS)}")

    # Accumulate all seasons into one DataFrame
    season_frames: list[pd.DataFrame] = []
    for season in _SEASONS:
        print(f"  Fetching season {season} …")
        try:
            df = _fetch_season(season)
            season_frames.append(df)
            print(f"    {len(df):,} rows")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING {season}: {exc}")
        time.sleep(_SLEEP)

    if not season_frames:
        print("ERROR: no data fetched")
        return

    combined = pd.concat(season_frames, ignore_index=True)
    # Deduplicate — same game may appear in multiple season queries if seasons overlap
    combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ABBREVIATION"])

    saved = 0
    for abbrev in abbrevs:
        try:
            team_df = _build_team_df(combined, abbrev)
            if team_df.empty:
                print(f"  SKIP {abbrev}: no rows")
                continue
            save(team_df, "sports", f"NBA_{abbrev}_1d.parquet")
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING {abbrev}: {exc}")

    print(f"\nSaved {saved} NBA team parquet files")


def main() -> None:
    print("Fetching: NBA per-team game logs (nba_api, 2018-2024)")
    collect_nba_teams()


if __name__ == "__main__":
    main()
