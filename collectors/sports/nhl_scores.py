"""
NHL scores & game data from public NHL API (no key needed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save
from concurrent.futures import ThreadPoolExecutor, as_completed


def _fetch_season(season: str) -> pd.DataFrame:
    """Fetch all games for a season."""
    import urllib.request, json
    url = f"https://api-web.nhle.com/v1/schedule/{season}-10-01"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
    except Exception:
        return pd.DataFrame()

    rows = []
    for gw in data.get("gameWeek", []):
        for g in gw.get("games", []):
            rows.append({
                "date": pd.to_datetime(gw["date"]),
                "season": season,
                "home_team": g["homeTeam"]["abbrev"],
                "away_team": g["awayTeam"]["abbrev"],
                "home_score": g["homeTeam"].get("score"),
                "away_score": g["awayTeam"].get("score"),
                "game_state": g.get("gameState"),
                "period": g.get("period"),
            })
    return pd.DataFrame(rows)


def collect():
    """Fetch NHL games for last 5 seasons."""
    import time
    seasons = [str(y) for y in range(2021, 2027)]
    print(f"Fetching NHL scores for {len(seasons)} seasons...")

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_season, s): s for s in seasons}
        for f in as_completed(futures):
            df = f.result()
            if not df.empty:
                rows.append(df)
            time.sleep(0.3)

    if not rows:
        print("  WARN: no NHL data")
        return

    df = pd.concat(rows).dropna(subset=["date"]).set_index("date").sort_index()
    for c in ["home_score", "away_score", "period"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = save(df, "sports", "nhl_scores_1d.parquet")
    print(f"Saved {out}: {len(df)} games")


if __name__ == "__main__":
    collect()
