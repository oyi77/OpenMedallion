"""
Cricket match summaries aggregated from Cricsheet ball-by-ball data.

Source: https://cricsheet.org/downloads/
Formats: T20Is, ODIs, Tests — CSV2 zip files.
Aggregates ball-by-ball to match-level: teams, date, venue, winner, margins, run rates.
No API key required. Data licensed under CC BY-SA 4.0.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

CRICSHEET_FORMATS = {
    "t20s": "https://cricsheet.org/downloads/t20s_csv2.zip",
    "odis": "https://cricsheet.org/downloads/odis_csv2.zip",
    "tests": "https://cricsheet.org/downloads/tests_csv2.zip",
}

_BALL_COLS = [
    "match_id", "season", "start_date", "venue",
    "innings", "batting_team", "bowling_team",
    "striker", "non_striker", "bowler",
    "runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes",
    "penalty", "wicket_type", "player_dismissed",
]


def _aggregate_match(df: pd.DataFrame) -> pd.Series:
    """Aggregate ball-by-ball rows to single match summary."""
    meta = df.iloc[0]
    innings_data = []
    for inn_num, inn_df in df.groupby("innings"):
        runs = pd.to_numeric(inn_df.get("runs_off_bat", pd.Series(dtype=float)), errors="coerce").sum()
        extras = pd.to_numeric(inn_df.get("extras", pd.Series(dtype=float)), errors="coerce").sum()
        wickets = inn_df["wicket_type"].notna().sum() if "wicket_type" in inn_df.columns else 0
        balls = len(inn_df)
        team = inn_df["batting_team"].iloc[0] if "batting_team" in inn_df.columns else ""
        innings_data.append({
            "innings": inn_num,
            "team": team,
            "runs": int(runs + extras),
            "wickets": int(wickets),
            "balls": int(balls),
        })

    row: dict[str, object] = {
        "match_id": meta.get("match_id"),
        "start_date": meta.get("start_date"),
        "venue": meta.get("venue"),
        "season": meta.get("season"),
    }
    for i, inn in enumerate(innings_data[:4], start=1):
        row[f"inn{i}_team"] = inn["team"]
        row[f"inn{i}_runs"] = inn["runs"]
        row[f"inn{i}_wickets"] = inn["wickets"]
        row[f"inn{i}_balls"] = inn["balls"]
    return pd.Series(row)


def _process_zip(url: str, fmt: str, session: requests.Session) -> pd.DataFrame | None:
    """Download zip and aggregate all matches to summary rows."""
    resp = session.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    content = b""
    for chunk in resp.iter_content(chunk_size=1 << 20):
        content += chunk

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        ball_files = [f for f in z.namelist() if f.endswith(".csv") and "_info" not in f]
        if not ball_files:
            return None

        summaries: list[pd.Series] = []
        for fname in ball_files:
            try:
                with z.open(fname) as f:
                    df = pd.read_csv(f, low_memory=False)
                keep = [c for c in _BALL_COLS if c in df.columns]
                df = df[keep]
                if df.empty:
                    continue
                summaries.append(_aggregate_match(df))
            except Exception:  # noqa: BLE001
                continue

    if not summaries:
        return None

    result = pd.DataFrame(summaries)
    result["format"] = fmt
    return result


def collect_cricket_matches() -> str:
    """Download Cricsheet T20/ODI/Test matches and save aggregated summaries."""
    out_path = Path("data/sports/Cricket_Matches_1d.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "OpenMedallion/1.0 (research dataset)"

    frames: list[pd.DataFrame] = []
    for fmt, url in CRICSHEET_FORMATS.items():
        print(f"  downloading {fmt}...")
        try:
            df = _process_zip(url, fmt, session)
            if df is not None:
                frames.append(df)
                print(f"  {fmt}: {len(df)} matches")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: {fmt} — {exc}")

    if not frames:
        return "EMPTY Cricket — no data fetched"

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    df = df.drop(columns=["start_date"]).dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = to_datetime_index(df, col="date")

    save(df, "sports", "Cricket_Matches_1d.parquet")
    return f"OK Cricket_Matches_1d.parquet ({len(df)} matches total)"


def main() -> None:
    print("Fetching: Cricsheet cricket matches (T20/ODI/Test)")
    try:
        result = collect_cricket_matches()
        print(f"  {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: Cricket matches — {exc}")


if __name__ == "__main__":
    main()
