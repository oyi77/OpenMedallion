"""
MLB historical game logs from Retrosheet (1871–present).

Source: https://www.retrosheet.org/gamelogs/
Format: Fixed 161-field CSV, one row per game.
No API key required. Data is free for non-commercial use.
"""
from __future__ import annotations

import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index, HISTORY_START

RETROSHEET_BASE = "https://www.retrosheet.org/gamelogs/"

# Field indices (0-based). See https://www.retrosheet.org/gamelogs/glfields.txt
_FIELD_MAP = {
    0:  "date",           # yyyymmdd
    3:  "visitor_team",
    4:  "visitor_league",
    6:  "home_team",
    7:  "home_league",
    9:  "visitor_score",
    10: "home_score",
    11: "game_length_outs",
    12: "day_night",
    17: "attendance",
    18: "game_duration_min",
}

import datetime
_START_YEAR = int(HISTORY_START[:4]) if HISTORY_START else 1880
_END_YEAR   = datetime.date.today().year

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_year(year: int) -> pd.DataFrame | None:
    """Download and parse one year of Retrosheet game logs."""
    url = f"{RETROSHEET_BASE}gl{year}.zip"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": _UA})
        resp.raise_for_status()
    except requests.HTTPError:
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        txt_files = [f for f in z.namelist() if f.lower().endswith(".txt")]
        if not txt_files:
            return None
        with z.open(txt_files[0]) as f:
            raw = f.read().decode("latin-1", errors="replace")

    rows = []
    for line in raw.splitlines():
        parts = [p.strip('"') for p in line.split(",")]
        if len(parts) < 12:
            continue
        row: dict[str, object] = {}
        for idx, name in _FIELD_MAP.items():
            row[name] = parts[idx] if idx < len(parts) else None
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def collect_mlb_gamelogs() -> str:
    """Download Retrosheet MLB game logs 1990–2024 and save as parquet."""
    years = list(range(_START_YEAR, _END_YEAR + 1))
    frames: list[pd.DataFrame] = []
    missing: list[int] = []

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_year, y): y for y in years}
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                df_year = fut.result()
            except Exception:  # noqa: BLE001
                df_year = None
            if df_year is not None:
                frames.append(df_year)
            else:
                missing.append(year)

    if not frames:
        return "EMPTY MLB — no data fetched"

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce", utc=True)
    df = df.dropna(subset=["date"])

    for col in ("visitor_score", "home_score", "game_length_outs", "attendance", "game_duration_min"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    df = to_datetime_index(df, col="date")

    save(df, "sports", "MLB_Gamelogs_1d.parquet")
    note = f" (missing: {sorted(missing)})" if missing else ""
    return f"OK MLB_Gamelogs_1d.parquet ({len(df)} rows, {_START_YEAR}–{_END_YEAR}){note}"


def main() -> None:
    print("Fetching: Retrosheet MLB game logs")
    try:
        result = collect_mlb_gamelogs()
        print(f"  {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: MLB gamelogs — {exc}")


if __name__ == "__main__":
    main()
