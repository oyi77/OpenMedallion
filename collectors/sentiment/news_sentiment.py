"""
GDELT GKG — daily media tone / news sentiment from raw CSV files.
Source: GDELT 2.0 GKG bulk CSVs (public, no key required)
Output: data/sentiment/gdelt_media_tone_1d.parquet
Columns: date (index, UTC), avg_tone (float), num_articles (int)

Strategy:
  Fetch ~30 recent GKG 15-min snapshots from the GDELT lastupdate list,
  extract the tone field (column 15, first sub-field), aggregate to daily.
  Falls back to the GDELT 1.0 daily CSV export if GKGv2 snapshots fail.
"""
from __future__ import annotations

import io
import sys
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

# GKGv2 15-min snapshot index (gives the 3 most recent files)
_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GDELT 1.0 daily export — has GLOBALEVENTID ... AvgTone column
# Each file is one day, available back to ~2013.
# We'll fetch the last 90 days to build a recent history.
_GDELT1_BASE = "http://data.gdeltproject.org/events/"

# GKG tone is field index 15 (0-based), sub-field 0 = average tone
_GKG_TONE_FIELD = 15
_GKG_DATE_FIELD = 1   # YYYYMMDDHHMMSS


def _parse_gkg_zip(content: bytes, source_date: str) -> list[dict]:
    """Extract (date_str, avg_tone) rows from one GKG zip file."""
    rows: list[dict] = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    for raw_line in f:
                        try:
                            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                            fields = line.split("\t")
                            if len(fields) <= _GKG_TONE_FIELD:
                                continue
                            date_ts = fields[_GKG_DATE_FIELD][:8]  # YYYYMMDD
                            tone_str = fields[_GKG_TONE_FIELD].split(",")[0]
                            tone = float(tone_str)
                            rows.append({"date": date_ts, "avg_tone": tone})
                        except (ValueError, IndexError):
                            continue
    except Exception:
        pass
    return rows


def _collect_gkgv2_recent(num_snapshots: int = 30) -> pd.DataFrame:
    """Download recent GKGv2 snapshots and aggregate to daily tone."""
    resp = fetch(_LASTUPDATE_URL, timeout=20)
    lines = [ln.strip() for ln in resp.text.splitlines() if "gkg.csv.zip" in ln]
    if not lines:
        return pd.DataFrame()

    # Extract GKG zip URLs from lastupdate.txt (format: size hash url)
    urls = []
    for line in lines[:num_snapshots]:
        parts = line.split()
        if len(parts) >= 3:
            urls.append(parts[2])

    all_rows: list[dict] = []
    for url in urls:
        try:
            print(f"  Fetching {url.split('/')[-1]}")
            r = fetch(url, timeout=60)
            rows = _parse_gkg_zip(r.content, url)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  WARNING {url}: {exc}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["avg_tone"] = pd.to_numeric(df["avg_tone"], errors="coerce")
    df = df.dropna(subset=["avg_tone"])
    daily = df.groupby("date").agg(avg_tone=("avg_tone", "mean"), num_articles=("avg_tone", "count")).reset_index()
    daily["date"] = pd.to_datetime(daily["date"], format="%Y%m%d", utc=True)
    return daily


def _collect_gdelt1_daily(days: int = 90) -> pd.DataFrame:
    """
    Fallback: fetch GDELT 1.0 daily export CSVs.
    Each has AvgTone at column index 34 (0-based).
    Files are named YYYYMMDD.export.CSV.zip.
    """
    today = datetime.now(tz=timezone.utc).date()
    all_rows: list[dict] = []
    for delta in range(days):
        d = today - timedelta(days=delta + 1)
        fname = d.strftime("%Y%m%d") + ".export.CSV.zip"
        url = _GDELT1_BASE + fname
        try:
            r = fetch(url, timeout=30)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for name in z.namelist():
                    with z.open(name) as f:
                        df_day = pd.read_csv(f, sep="\t", header=None, usecols=[34], dtype={34: float}, on_bad_lines="skip")
                        tones = df_day.iloc[:, 0].dropna()
                        if len(tones) > 0:
                            all_rows.append({"date": d.strftime("%Y%m%d"), "avg_tone": float(tones.mean()), "num_articles": len(tones)})
            print(f"  GDELT1 {d}: {len(tones)} articles")
        except Exception as exc:
            print(f"  WARNING GDELT1 {d}: {exc}")

    if not all_rows:
        return pd.DataFrame()

    daily = pd.DataFrame(all_rows)
    daily["date"] = pd.to_datetime(daily["date"], format="%Y%m%d", utc=True)
    return daily


def collect_gdelt_tone() -> None:
    """Try GKGv2 snapshots first, then GDELT 1.0 daily fallback."""

    print("  Fetching GKGv2 snapshots (recent)...")
    df = _collect_gkgv2_recent(num_snapshots=30)

    if df.empty or len(df) < 2:
        print("  GKGv2 insufficient — trying GDELT 1.0 daily exports (60 days)...")
        df = _collect_gdelt1_daily(days=60)

    if df.empty:
        print("  WARNING: All GDELT sources failed or returned empty data")
        return

    df = to_datetime_index(df, col="date")
    save(df, "sentiment", "gdelt_media_tone_1d.parquet")


def main() -> None:
    print("Fetching: GDELT GKG media tone / news sentiment")
    try:
        collect_gdelt_tone()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
