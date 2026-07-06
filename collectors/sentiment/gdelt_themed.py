"""
GDELT GKGv2 — per-theme and per-country sentiment from 15-min snapshots.
Source: GDELT 2.0 GKG bulk CSVs (public, no key required)

Outputs:
  data/sentiment/gdelt_theme_finance_1d.parquet    — ECON_ themes
  data/sentiment/gdelt_theme_energy_1d.parquet     — ENV_ / ENERGY themes
  data/sentiment/gdelt_theme_geopolitical_1d.parquet — CRISISLEX_/TERROR/WB_/TAX_FNCACT_/GEOPOLITICS
  data/sentiment/gdelt_country_top20_1d.parquet    — top-20 countries, long format

Strategy:
  Sample 4 evenly-spaced snapshots per day (0000, 0600, 1200, 1800 UTC)
  over the last 7 days — 28 snapshots total — to keep runtime reasonable.
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

_GKG_DATE_FIELD = 1       # YYYYMMDDHHMMSS
_GKG_COUNTRY_FIELD = 3    # 2-letter source country
_GKG_THEMES_FIELD = 7     # semicolon-separated theme strings
_GKG_TONE_FIELD = 15      # CSV; first sub-field is avg_tone

# Four UTC hours sampled per day
_SAMPLE_HOURS = {"00", "06", "12", "18"}

# Theme-group membership tests (prefix / substring checks)
_FINANCE_PREFIXES = ("ECON_", "ECON")
_ENERGY_SUBSTRINGS = ("ENV_", "ENERGY")
_GEO_SUBSTRINGS = ("CRISISLEX_", "TERROR", "WB_", "TAX_FNCACT_", "GEOPOLITICS")


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _sample_gkg_urls(days: int = 7) -> list[str]:
    """Return GKGv2 snapshot URLs for the last `days` days, sampled at 4 UTC hours/day."""
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    try:
        resp = fetch(_MASTERFILELIST_URL, timeout=60)
        urls: list[str] = []
        for line in resp.text.splitlines():
            if "gkg.csv.zip" not in line:
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            url = parts[2]
            fname = url.split("/")[-1]   # e.g. 20260706061500.gkg.csv.zip
            if len(fname) < 12:
                continue
            date_part = fname[:8]         # YYYYMMDD
            hour_part = fname[8:10]       # HH
            if date_part >= cutoff and hour_part in _SAMPLE_HOURS:
                urls.append(url)
        return urls
    except Exception as exc:
        print(f"  WARNING masterfilelist: {exc}")

    # Fallback: lastupdate.txt (only last few files)
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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_gkg_zip(content: bytes) -> list[dict]:
    """
    Parse one GKG zip and return a list of row dicts.
    Each dict has: date (YYYYMMDD str), country (str), themes (list[str]), avg_tone (float).
    """
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
                            date_str = fields[_GKG_DATE_FIELD][:8]   # YYYYMMDD
                            country = fields[_GKG_COUNTRY_FIELD].strip()
                            raw_themes = fields[_GKG_THEMES_FIELD].strip()
                            themes = [t for t in raw_themes.split(";") if t]
                            tone_str = fields[_GKG_TONE_FIELD].split(",")[0]
                            avg_tone = float(tone_str)
                            rows.append({
                                "date": date_str,
                                "country": country,
                                "themes": themes,
                                "avg_tone": avg_tone,
                            })
                        except (ValueError, IndexError):
                            continue
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Theme membership helpers
# ---------------------------------------------------------------------------

def _has_finance_theme(themes: list[str]) -> bool:
    return any(t.startswith(_FINANCE_PREFIXES) for t in themes)


def _has_energy_theme(themes: list[str]) -> bool:
    return any(any(sub in t for sub in _ENERGY_SUBSTRINGS) for t in themes)


def _has_geo_theme(themes: list[str]) -> bool:
    return any(any(sub in t for sub in _GEO_SUBSTRINGS) for t in themes)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a filtered df to daily avg_tone + num_articles with DatetimeIndex."""
    daily = (
        df.groupby("date")
        .agg(avg_tone=("avg_tone", "mean"), num_articles=("avg_tone", "count"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"], format="%Y%m%d", utc=True)
    return to_datetime_index(daily, col="date")


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def collect_gdelt_themed(days: int = 7) -> None:
    """Download sampled GKGv2 snapshots and save four themed/country parquet files."""
    print(f"  Fetching sampled GKGv2 URLs (last {days} days, 4 snapshots/day)...")
    urls = _sample_gkg_urls(days=days)
    if not urls:
        print("  WARNING: No GKGv2 URLs found — aborting")
        return

    print(f"  Found {len(urls)} snapshot(s) — downloading...")
    all_rows: list[dict] = []

    for i, url in enumerate(urls):
        try:
            fname = url.split("/")[-1]
            print(f"  [{i + 1}/{len(urls)}] {fname}")
            r = fetch(url, timeout=90)
            all_rows.extend(_parse_gkg_zip(r.content))
        except Exception as exc:
            print(f"  WARNING {url}: {exc}")
        time.sleep(0.3)

    if not all_rows:
        print("  WARNING: No data parsed from GKGv2 snapshots")
        return

    df = pd.DataFrame(all_rows)
    df["avg_tone"] = pd.to_numeric(df["avg_tone"], errors="coerce")
    df = df.dropna(subset=["avg_tone"])

    # -----------------------------------------------------------------------
    # 1. Finance themes (ECON_)
    # -----------------------------------------------------------------------
    try:
        mask_fin = df["themes"].apply(_has_finance_theme)
        df_fin = df[mask_fin]
        daily_fin = _daily_agg(df_fin)
        save(daily_fin, "sentiment", "gdelt_theme_finance_1d.parquet")
    except Exception as exc:
        print(f"  WARNING finance theme save: {exc}")

    # -----------------------------------------------------------------------
    # 2. Energy / environment themes (ENV_ / ENERGY)
    # -----------------------------------------------------------------------
    try:
        mask_eng = df["themes"].apply(_has_energy_theme)
        df_eng = df[mask_eng]
        daily_eng = _daily_agg(df_eng)
        save(daily_eng, "sentiment", "gdelt_theme_energy_1d.parquet")
    except Exception as exc:
        print(f"  WARNING energy theme save: {exc}")

    # -----------------------------------------------------------------------
    # 3. Geopolitical themes
    # -----------------------------------------------------------------------
    try:
        mask_geo = df["themes"].apply(_has_geo_theme)
        df_geo = df[mask_geo]
        daily_geo = _daily_agg(df_geo)
        save(daily_geo, "sentiment", "gdelt_theme_geopolitical_1d.parquet")
    except Exception as exc:
        print(f"  WARNING geopolitical theme save: {exc}")

    # -----------------------------------------------------------------------
    # 4. Top-20 countries — long format, date as datetime column (not index)
    # -----------------------------------------------------------------------
    try:
        # Compute total article count per country to find top 20
        country_counts = df.groupby("country")["avg_tone"].count()
        top20 = country_counts.nlargest(20).index.tolist()

        df_c = df[df["country"].isin(top20)].copy()
        country_daily = (
            df_c.groupby(["date", "country"])
            .agg(avg_tone=("avg_tone", "mean"), num_articles=("avg_tone", "count"))
            .reset_index()
        )
        country_daily["date"] = pd.to_datetime(
            country_daily["date"], format="%Y%m%d", utc=True
        )
        # Long format: keep date as a regular column, do NOT set as index
        # (to_datetime_index requires a single date col; not applicable here)
        save(country_daily, "sentiment", "gdelt_country_top20_1d.parquet")
    except Exception as exc:
        print(f"  WARNING country top20 save: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching: GDELT GKGv2 themed sentiment (finance / energy / geopolitical / country)")
    try:
        collect_gdelt_themed(days=7)
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
