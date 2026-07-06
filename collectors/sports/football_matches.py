"""
Free football/soccer data from openfootball CSV repo (no API key).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save
from concurrent.futures import ThreadPoolExecutor, as_completed

_LEAGUES = {
    "https://raw.githubusercontent.com/openfootball/euro-cup/main/2024/euro.csv": "euro_2024",
    "https://raw.githubusercontent.com/openfootball/world-cup/main/2022/worldcup.csv": "worldcup_2022",
    "https://raw.githubusercontent.com/openfootball/eng-england/master/2024-25/premierleague.csv": "epl_2024_25",
    "https://raw.githubusercontent.com/openfootball/de-deutschland/master/2024-25/bundesliga.csv": "bundesliga_2024_25",
    "https://raw.githubusercontent.com/openfootball/es-espana/master/2024-25/laliga.csv": "laliga_2024_25",
    "https://raw.githubusercontent.com/openfootball/it-italy/master/2024-25/seriea.csv": "seriea_2024_25",
    "https://raw.githubusercontent.com/openfootball/fr-france/master/2024-25/ligue1.csv": "ligue1_2024_25",
}


def _fetch_league(url: str, label: str) -> pd.DataFrame | None:
    import urllib.request, io, time
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8")
            lines = [l for l in raw.split("\n") if l.strip() and not l.startswith("#")]
            if not lines:
                return None
            df = pd.read_csv(io.StringIO("\n".join(lines)), header=None)
            # Standardize columns
            if len(df.columns) >= 6:
                df = df.iloc[:, :6]
                df.columns = ["round", "date", "home", "away", "score_h", "score_a"]
                df["league"] = label
                return df
        except Exception as e:
            print(f"    attempt {attempt+1}: {e}")
            time.sleep(1)
    return None


def collect():
    """Fetch football match results from openfootball."""
    print(f"Fetching {len(_LEAGUES)} football leagues...")
    import time

    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_fetch_league, url, label): label for url, label in _LEAGUES.items()}
        for f in as_completed(futures):
            df = f.result()
            if df is not None and not df.empty:
                rows.append(df)
            time.sleep(0.3)

    if not rows:
        print("  WARN: no football data")
        return

    df = pd.concat(rows).dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    for c in ["score_h", "score_a"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = save(df, "sports", "football_matches_1d.parquet")
    print(f"Saved {out}: {len(df)} matches")


if __name__ == "__main__":
    collect()
