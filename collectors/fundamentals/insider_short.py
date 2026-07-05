"""
Insider trading (SEC EDGAR Form 4 bulk index) and FINRA short interest collector.

Sources:
  - SEC EDGAR full-index Form 4: https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx
  - FINRA REGSHO daily short volume: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

Outputs:
  data/fundamentals/SEC_Form4_Index_bulk.parquet
  data/fundamentals/FINRA_ShortInterest_daily.parquet
"""
from __future__ import annotations

import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save, to_datetime_index

# SEC requires a descriptive User-Agent; anonymous requests are rejected.
EDGAR_HEADERS = {"User-Agent": "OpenMedallion research@openmedallion.org"}
# FINRA CDN requires a browser-like User-Agent; plain requests get 403.
FINRA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; OpenMedallion/1.0; research@openmedallion.org)"
    )
}

EDGAR_IDX_URL = (
    "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx"
)
FINRA_SHORT_URL = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
)


def _last_four_quarters() -> list[tuple[int, int]]:
    """Return [(year, quarter), ...] for the 4 most recent completed quarters."""
    today = date.today()
    current_q = (today.month - 1) // 3 + 1
    current_year = today.year
    quarters: list[tuple[int, int]] = []
    q, y = current_q, current_year
    # step back one quarter to get the most recent *completed* quarter
    q -= 1
    if q == 0:
        q = 4
        y -= 1
    for _ in range(4):
        quarters.append((y, q))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return quarters


def fetch_form4_bulk_index() -> pd.DataFrame:
    """
    Download SEC EDGAR full-index form.idx for the last 4 completed quarters,
    filter to Form 4 filings, and return a combined DataFrame.
    """
    quarters = _last_four_quarters()
    frames: list[pd.DataFrame] = []

    for year, q in quarters:
        url = EDGAR_IDX_URL.format(year=year, q=q)
        print(f"  Fetching SEC Form 4 index: {year} Q{q} …")
        try:
            # base.fetch doesn't accept custom headers; use requests directly
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  WARNING: failed to fetch {url}: {exc}")
            continue

        lines = resp.text.splitlines()

        # form.idx fixed-width format (stable since EDGAR launch):
        # Col start positions derived from the header line:
        # "Form Type   Company Name  ...  CIK         Date Filed  File Name"
        #  0           12                 74          86          98
        C_FORM = 0
        C_NAME = 12
        C_CIK  = 74
        C_DATE = 86
        C_FILE = 98

        # Find the separator line so we skip header rows
        sep_idx = None
        for i, line in enumerate(lines):
            if line.startswith("---"):
                sep_idx = i
                break

        if sep_idx is None:
            print(f"  WARNING: unexpected format for {year} Q{q}, skipping.")
            continue

        records: list[dict] = []
        for line in lines[sep_idx + 1:]:
            if not line.strip():
                continue
            try:
                form_type = line[C_FORM:C_NAME].strip()
                if form_type not in ("4", "4/A"):
                    continue
                company_name = line[C_NAME:C_CIK].strip()
                cik           = line[C_CIK:C_DATE].strip()
                date_filed    = line[C_DATE:C_FILE].strip()
                filename      = line[C_FILE:].strip()
                records.append(
                    {
                        "date": date_filed,
                        "form_type": form_type,
                        "company_name": company_name,
                        "cik": cik,
                        "filename": filename,
                        "year": year,
                        "quarter": q,
                    }
                )
            except IndexError:
                continue

        if records:
            df_q = pd.DataFrame(records)
            frames.append(df_q)
            print(f"    → {len(records):,} Form 4 filings for {year} Q{q}")
        else:
            print(f"  WARNING: 0 Form 4 records parsed for {year} Q{q}")

        time.sleep(0.5)  # be polite to EDGAR

    if not frames:
        print("  WARNING: no Form 4 data collected.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = to_datetime_index(combined, col="date")
    return combined


def _trading_day_candidates(n: int = 20) -> list[date]:
    """Return up to n weekday dates going backward from yesterday."""
    days: list[date] = []
    d = date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:  # Mon–Fri
            days.append(d)
        d -= timedelta(days=1)
    return days


def fetch_finra_short_interest(target_days: int = 10, max_attempts: int = 20) -> pd.DataFrame:
    """Download FINRA REGSHO daily short volume files for the last `target_days`
    available trading days. Tries up to `max_attempts` candidate dates,
    skipping 404/403s gracefully (dates not yet published return one or the other).
    """
    candidates = _trading_day_candidates(n=max_attempts)
    frames: list[pd.DataFrame] = []
    found = 0

    for d in candidates:
        if found >= target_days:
            break
        date_str = d.strftime("%Y%m%d")
        url = FINRA_SHORT_URL.format(date=date_str)
        print(f"  Fetching FINRA short interest: {date_str} …")
        try:
            resp = requests.get(url, headers=FINRA_HEADERS, timeout=30)
            if resp.status_code in (403, 404):
                print(f"    → {resp.status_code}, skipping {date_str}")
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  WARNING: failed to fetch {url}: {exc}")
            continue

        try:
            # Pipe-delimited: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
            df_day = pd.read_csv(
                io.StringIO(resp.text),
                sep="|",
                dtype=str,
            )
            # Drop blank / summary rows FINRA appends at the bottom
            if "Symbol" in df_day.columns:
                df_day = df_day[
                    df_day["Symbol"].notna() & (df_day["Symbol"].str.strip() != "")
                ]

            # Ensure a Date column exists
            if "Date" not in df_day.columns:
                df_day.insert(0, "Date", date_str)
            else:
                df_day["Date"] = df_day["Date"].fillna(date_str)

            # Coerce numeric columns
            for col in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
                if col in df_day.columns:
                    df_day[col] = pd.to_numeric(df_day[col], errors="coerce")

            frames.append(df_day)
            found += 1
            print(f"    → {len(df_day):,} rows")
        except Exception as exc:
            print(f"  WARNING: failed to parse FINRA data for {date_str}: {exc}")
            continue

    if not frames:
        print("  WARNING: no FINRA short interest data collected.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = to_datetime_index(combined, col="Date")
    return combined


def main() -> None:
    print("=== SEC EDGAR Form 4 Bulk Index ===")
    df_form4 = fetch_form4_bulk_index()
    save(df_form4, "fundamentals", "SEC_Form4_Index_bulk.parquet")

    print("\n=== FINRA Short Interest (daily) ===")
    df_short = fetch_finra_short_interest(target_days=10, max_attempts=20)
    save(df_short, "fundamentals", "FINRA_ShortInterest_daily.parquet")

    print("\nDone.")


if __name__ == "__main__":
    main()
