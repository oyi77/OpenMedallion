#!/usr/bin/env python3
"""
Sports betting historical odds collector.

Sources:
- football-data.co.uk: 18 European soccer leagues, 1993-present, 6+ bookmakers
- NFL spreadspoke (Kaggle-hosted GitHub CSV): scores + odds since 1966/1979

Output (parquet, one file per league/sport):
  data/betting/soccer_{league_code}_odds_1d.parquet
  data/betting/nfl_odds_1d.parquet

Columns (soccer):
  date, home_team, away_team, league, season,
  fthg, ftag, ftr,           # full-time score + result
  hthg, htag, htr,           # half-time score + result
  b365h, b365d, b365a,       # Bet365 H/D/A odds
  bwh, bwd, bwa,             # Betway
  psh, psd, psa,             # Pinnacle (via PS)
  maxh, maxd, maxa,          # market max
  avgh, avgd, avga,          # market average
  b365_o25, b365_u25,        # Bet365 over/under 2.5
  avg_o25, avg_u25           # market avg over/under

Columns (NFL):
  date, season, week, home_team, away_team,
  score_home, score_away, result,
  spread_favorite, spread_home,
  over_under_line, over_under_result,
  favorite_team, underdog_team
"""
from __future__ import annotations

import io
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "betting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STALE_HOURS = 24

# ── football-data.co.uk ───────────────────────────────────────────────────────

FDCO_BASE = "https://www.football-data.co.uk/mmz4281"

FDCO_LEAGUES: dict[str, str] = {
    "E0":  "EPL",
    "E1":  "Championship",
    "SP1": "LaLiga",
    "SP2": "LaLiga2",
    "D1":  "Bundesliga",
    "D2":  "Bundesliga2",
    "I1":  "SerieA",
    "I2":  "SerieB",
    "F1":  "Ligue1",
    "F2":  "Ligue2",
    "N1":  "Eredivisie",
    "B1":  "JupilerPro",
    "P1":  "PrimeiraLiga",
    "T1":  "SuperLig",
    "G1":  "SuperLeague",
    "SC0": "ScottishPrem",
}

# columns we want to keep (subset of the 100+ available)
FDCO_KEEP = [
    # match info
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    # full-time
    "FTHG", "FTAG", "FTR",
    # half-time
    "HTHG", "HTAG", "HTR",
    # Bet365 1x2
    "B365H", "B365D", "B365A",
    # Betway 1x2
    "BWH", "BWD", "BWA",
    # Pinnacle 1x2
    "PSH", "PSD", "PSA",
    # market max / avg 1x2
    "MaxH", "MaxD", "MaxA",
    "AvgH", "AvgD", "AvgA",
    # over/under 2.5
    "B365>2.5", "B365<2.5",
    "Max>2.5",  "Max<2.5",
    "Avg>2.5",  "Avg<2.5",
]

FDCO_RENAME = {
    "Div": "league_code", "Date": "date", "Time": "time",
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr",
    "HTHG": "hthg", "HTAG": "htag", "HTR": "htr",
    "B365H": "b365h", "B365D": "b365d", "B365A": "b365a",
    "BWH": "bwh", "BWD": "bwd", "BWA": "bwa",
    "PSH": "psh", "PSD": "psd", "PSA": "psa",
    "MaxH": "maxh", "MaxD": "maxd", "MaxA": "maxa",
    "AvgH": "avgh", "AvgD": "avgd", "AvgA": "avga",
    "B365>2.5": "b365_o25", "B365<2.5": "b365_u25",
    "Max>2.5": "max_o25", "Max<2.5": "max_u25",
    "Avg>2.5": "avg_o25", "Avg<2.5": "avg_u25",
}


def _gen_seasons() -> list[str]:
    """Return all season codes from 9394 through current."""
    now = datetime.now()
    cur_year = now.year % 100
    # football season starts Aug; if before Aug we're still in prior season
    end_year = cur_year if now.month >= 8 else cur_year - 1
    seasons: list[str] = []
    for y in range(93, 100):
        seasons.append(f"{y:02d}{(y+1)%100:02d}")
    for y in range(0, end_year + 1):
        seasons.append(f"{y:02d}{(y+1)%100:02d}")
    return seasons


def _fetch_fdco_csv(season: str, code: str) -> pd.DataFrame | None:
    url = f"{FDCO_BASE}/{season}/{code}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OpenMedallion/1.0"})
        raw = urllib.request.urlopen(req, timeout=15).read()
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip", encoding="latin-1")
        if df.empty or "Date" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _parse_fdco_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    return df


def collect_soccer(code: str, league_name: str) -> str:
    out_path = OUT_DIR / f"soccer_{code.lower()}_odds_1d.parquet"

    # stale check
    if out_path.exists():
        age_h = (datetime.now(timezone.utc).timestamp() - out_path.stat().st_mtime) / 3600
        if age_h < STALE_HOURS:
            return f"SKIP {out_path.name} ({age_h:.1f}h old)"

    seasons = _gen_seasons()
    frames: list[pd.DataFrame] = []

    for season in seasons:
        df = _fetch_fdco_csv(season, code)
        if df is None:
            continue
        # keep only columns that exist in this season's CSV
        keep = [c for c in FDCO_KEEP if c in df.columns]
        df = df[keep].copy()
        df = df.rename(columns={c: FDCO_RENAME[c] for c in keep if c in FDCO_RENAME})
        df["season"] = season
        df["league"] = league_name
        frames.append(df)

    if not frames:
        return f"EMPTY {code} - no data found"

    merged = pd.concat(frames, ignore_index=True)
    merged = _parse_fdco_date(merged)
    merged = merged.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    merged.to_parquet(out_path, index=False)
    return f"OK {out_path.name} ({len(merged)} rows, {merged['date'].min().date()} – {merged['date'].max().date()})"


# ── NFL (leesharpe/nfldata) ───────────────────────────────────────────────────

NFL_GAMES_URL = (
    "https://raw.githubusercontent.com/leesharpe/nfldata/master/data/games.csv"
)
NFL_LINES_URL = (
    "https://raw.githubusercontent.com/leesharpe/nfldata/master/data/closing_lines.csv"
)

# games.csv columns we want
NFL_KEEP = [
    "game_id", "season", "game_type", "week", "gameday",
    "away_team", "home_team",
    "away_score", "home_score", "result", "total", "overtime",
    "away_moneyline", "home_moneyline",
    "spread_line", "away_spread_odds", "home_spread_odds",
    "total_line", "under_odds", "over_odds",
    "div_game", "roof", "surface",
    "temp", "wind",
    "away_qb_name", "home_qb_name",
    "away_coach", "home_coach",
    "stadium_id",
]

NFL_RENAME = {
    "gameday": "date",
    "game_type": "game_type",
    "away_score": "score_away",
    "home_score": "score_home",
    "total": "total_score",
    "total_line": "over_under_line",
    "temp": "temp_f",
    "wind": "wind_mph",
}


def collect_nfl() -> str:
    out_path = OUT_DIR / "nfl_odds_1d.parquet"

    if out_path.exists():
        age_h = (datetime.now(timezone.utc).timestamp() - out_path.stat().st_mtime) / 3600
        if age_h < STALE_HOURS:
            return f"SKIP {out_path.name} ({age_h:.1f}h old)"

    try:
        req = urllib.request.Request(NFL_GAMES_URL, headers={"User-Agent": "OpenMedallion/1.0"})
        raw = urllib.request.urlopen(req, timeout=20).read()
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip")
    except Exception as exc:
        return f"ERR nfl_odds_1d.parquet - {exc}"

    if df.empty:
        return "EMPTY NFL - no rows from games.csv"

    keep = [c for c in NFL_KEEP if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={c: NFL_RENAME[c] for c in keep if c in NFL_RENAME})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # numeric coercion for score columns
    for col in ("score_home", "score_away"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(out_path, index=False)
    return (
        f"OK {out_path.name} ({len(df)} rows, "
        f"{df['date'].min().date()} – {df['date'].max().date()})"
    )


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    results: list[str] = []

    # soccer leagues
    for code, name in FDCO_LEAGUES.items():
        r = collect_soccer(code, name)
        print(r)
        results.append(r)

    # NFL
    r = collect_nfl()
    print(r)
    results.append(r)

    errors = [r for r in results if r.startswith("ERR") or r.startswith("EMPTY")]
    if errors:
        print(f"\n{len(errors)} errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
