"""
NHL scores & game data from public NHL API (no key needed).
Paginates through full seasons via nextStartDate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save
import urllib.request, json, time


def _fetch_season(season_start: int) -> pd.DataFrame:
    """Fetch all regular-season games by following nextStartDate links."""
    rows = []
    date = f"{season_start}-10-01"
    end_date = f"{season_start + 1}-07-01"
    seen = set()

    while date and date < end_date:
        if date in seen:
            break
        seen.add(date)

        url = f"https://api-web.nhle.com/v1/schedule/{date}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  WARN {date}: {e}")
            break

        for gw in data.get("gameWeek", []):
            for g in gw.get("games", []):
                rows.append({
                    "date": pd.to_datetime(gw["date"]),
                    "season": f"{season_start}-{season_start+1}",
                    "home_team": g["homeTeam"]["abbrev"],
                    "away_team": g["awayTeam"]["abbrev"],
                    "home_score": g["homeTeam"].get("score"),
                    "away_score": g["awayTeam"].get("score"),
                    "game_type": g.get("gameType"),
                    "game_state": g.get("gameState"),
                })

        date = data.get("nextStartDate")
        time.sleep(0.3)

    return pd.DataFrame(rows)


def collect():
    seasons = list(range(2019, 2025))  # 2019-20 through 2024-25
    print(f"Fetching NHL scores for {len(seasons)} seasons...")

    all_rows = []
    for s in seasons:
        df = _fetch_season(s)
        n = len(df)
        print(f"  {s}-{s+1}: {n} games")
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("  WARN: no NHL data")
        return

    combined = pd.concat(all_rows).dropna(subset=["date"]).set_index("date").sort_index()
    for c in ["home_score", "away_score"]:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")

    out = save(combined, "sports", "nhl_scores_1d.parquet")
    print(f"Saved {out}: {len(combined)} games")


if __name__ == "__main__":
    collect()
