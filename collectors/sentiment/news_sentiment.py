"""
GDELT GKG — daily media tone / news sentiment from raw CSV files.
Source: GDELT 2.0 GKG bulk CSVs (public, no key required)
Output: data/sentiment/gdelt_media_tone_1d.parquet
Columns: date (index, UTC), avg_tone (float), num_articles (int)

Strategy:
  Fetch recent GKGv2 15-min snapshots via the master file list.
  Each snapshot has tone at tab-field 15 (CSV sub-field 0).
  Aggregates rows to daily averages.
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

# GKGv2 master file list — one line per 15-min file since 2015
_MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
# Single latest snapshot fallback
_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

_GKG_TONE_FIELD = 15   # tab-separated, tone is field index 15
_GKG_DATE_FIELD = 1    # YYYYMMDDHHMMSS


def _parse_gkg_zip(content: bytes) -> list[dict]:
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
                            date_str = fields[_GKG_DATE_FIELD][:8]  # YYYYMMDD
                            tone_str = fields[_GKG_TONE_FIELD].split(",")[0]
                            tone = float(tone_str)
                            rows.append({"date": date_str, "avg_tone": tone})
                        except (ValueError, IndexError):
                            continue
    except Exception:
        pass
    return rows


def _get_recent_gkg_urls(days: int = 7) -> list[str]:
    """Return GKGv2 snapshot URLs covering the last `days` days via masterfilelist."""
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    try:
        resp = fetch(_MASTERFILELIST_URL, timeout=60)
        urls = []
        for line in resp.text.splitlines():
            if "gkg.csv.zip" not in line:
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            url = parts[2]
            # Extract date from URL filename e.g. 20260706011500
            fname = url.split("/")[-1]
            if len(fname) >= 8 and fname[:8] >= cutoff:
                urls.append(url)
        return urls
    except Exception as exc:
        print(f"  WARNING masterfilelist: {exc}")

    # Fallback: use lastupdate.txt (only last 3 files, but recent)
    try:
        resp = fetch(_LASTUPDATE_URL, timeout=20)
        urls = []
        for line in resp.text.splitlines():
            if "gkg.csv.zip" in line:
                parts = line.strip().split()
                if len(parts) >= 3:
                    urls.append(parts[2])
        return urls
    except Exception as exc:
        print(f"  WARNING lastupdate: {exc}")
        return []


def collect_gdelt_tone(days: int = 7) -> None:
    """Fetch GKGv2 snapshots for the last `days` days and save daily tone."""
    print(f"  Fetching GKGv2 snapshot URLs (last {days} days)...")
    urls = _get_recent_gkg_urls(days=days)
    if not urls:
        print("  WARNING: No GKGv2 URLs found")
        return

    print(f"  Found {len(urls)} snapshot(s) — downloading...")
    all_rows: list[dict] = []
    # Sample up to 96 snapshots (4/hr × 24hr = 96 per day; cap at 7 days = 672)
    sample = urls[-min(len(urls), 96 * days):]
    for i, url in enumerate(sample):
        try:
            fname = url.split("/")[-1]
            print(f"  [{i+1}/{len(sample)}] {fname}")
            r = fetch(url, timeout=90)
            all_rows.extend(_parse_gkg_zip(r.content))
        except Exception as exc:
            print(f"  WARNING {url}: {exc}")

    if not all_rows:
        print("  WARNING: No data parsed from GKG snapshots")
        return

    df = pd.DataFrame(all_rows)
    df["avg_tone"] = pd.to_numeric(df["avg_tone"], errors="coerce")
    df = df.dropna(subset=["avg_tone"])
    daily = (
        df.groupby("date")
        .agg(avg_tone=("avg_tone", "mean"), num_articles=("avg_tone", "count"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"], format="%Y%m%d", utc=True)
    daily = to_datetime_index(daily, col="date")
    save(daily, "sentiment", "gdelt_media_tone_1d.parquet")


def main() -> None:
    print("Fetching: GDELT GKG media tone / news sentiment")
    try:
        collect_gdelt_tone(days=7)
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
