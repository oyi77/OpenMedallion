"""
NBA historical game results + ELO ratings (FiveThirtyEight dataset).

Source: https://github.com/fivethirtyeight/data/tree/master/nba-elo
Fields: date, home/away team, scores, ELO before/after, win probability forecast.
Coverage: 1947–present (NBA + ABA).
No API key required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

NBA_ELO_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"

# Columns we keep (drop duplicate mirror rows and low-value fields)
_KEEP = [
    "date_game", "year_id", "is_playoffs",
    "team_id", "fran_id", "pts",
    "opp_id", "opp_fran", "opp_pts",
    "game_location", "game_result",
    "elo_i", "elo_n", "opp_elo_i", "opp_elo_n",
    "forecast", "win_equiv",
]


def collect_nba_elo() -> str:
    """Download FiveThirtyEight NBA ELO dataset and save as parquet."""
    out_path = Path("data/sports/NBA_ELO_1d.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = fetch(NBA_ELO_URL)
    df = pd.read_csv(pd.io.common.StringIO(resp.text), low_memory=False)

    # Keep only home-team rows to avoid duplicate game entries
    df = df[df["_iscopy"] == 0].copy()

    keep = [c for c in _KEEP if c in df.columns]
    df = df[keep].rename(columns={"date_game": "date"})

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ("pts", "opp_pts", "elo_i", "elo_n", "opp_elo_i", "opp_elo_n", "forecast", "win_equiv"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = to_datetime_index(df, col="date")
    save(df, "sports", "NBA_ELO_1d.parquet")
    return f"OK NBA_ELO_1d.parquet ({len(df)} rows)"


def main() -> None:
    print("Fetching: FiveThirtyEight NBA ELO")
    try:
        result = collect_nba_elo()
        print(f"  {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: NBA ELO — {exc}")


if __name__ == "__main__":
    main()
