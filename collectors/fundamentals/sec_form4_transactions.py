"""SEC EDGAR Form 4 transaction collector.

Fetches the last 30 days of Form 4 XML filings via the EDGAR full-index
quarterly files and extracts non-derivative insider transaction rows
(shares, price, buy/sell direction).

Output: data/fundamentals/SEC_Form4_Transactions_30d.parquet
Columns: filing_date (DatetimeIndex), ticker, issuer_name, owner_name,
         transaction_date, shares, price_per_share, acquired_disposed (A/D),
         transaction_value (shares * price)
"""

from __future__ import annotations

import re
import sys
import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from collectors.base import fetch, save, to_datetime_index

LOG = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent (robots.txt policy)
_HEADERS = {"User-Agent": "OpenMedallion research@openmedallion.org"}

# 10 req/s is the SEC rate limit; 0.1 s keeps us safely under it
_SLEEP = 0.1

MAX_FILINGS = 500


# ---------------------------------------------------------------------------
# Step 1 — collect Form 4 file paths from quarterly full-index files
# ---------------------------------------------------------------------------

def _quarters_for_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Return list of (year, quarter) tuples covering [start, end]."""
    seen: list[tuple[int, int]] = []
    cur = start.replace(day=1)
    while cur <= end:
        q = (cur.month - 1) // 3 + 1
        entry = (cur.year, q)
        if entry not in seen:
            seen.append(entry)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return seen


def _collect_via_full_index(start: datetime, end: datetime) -> list[dict]:
    """Yield up to MAX_FILINGS Form 4 filing metadata dicts from EDGAR full-index."""
    filings: list[dict] = []

    for year, qtr in _quarters_for_range(start, end):
        idx_url = (
            f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"
        )
        LOG.info("Fetching index: %s", idx_url)
        try:
            resp = fetch(idx_url, timeout=60)
            resp.request.headers.update(_HEADERS)  # best-effort; fetch() doesn't expose headers
        except Exception as exc:
            LOG.warning("Could not fetch %s: %s", idx_url, exc)
            time.sleep(_SLEEP)
            continue
        time.sleep(_SLEEP)

        for line in resp.text.splitlines():
            if not line.startswith("4 "):
                continue
            # Fixed-width columns split by 2+ spaces:
            # Form Type | Company Name | CIK | Date Filed | Filename
            cols = re.split(r"  +", line.strip())
            if len(cols) < 5:
                continue
            if cols[0].strip() != "4":
                continue

            date_filed = cols[3].strip()
            try:
                filed_dt = datetime.strptime(date_filed, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

            if filed_dt < start or filed_dt > end:
                continue

            filings.append(
                {
                    "company": cols[1].strip(),
                    "date_filed": date_filed,
                    "filename": cols[4].strip(),  # e.g. edgar/data/<cik>/<acc>/<file>.txt
                }
            )
            if len(filings) >= MAX_FILINGS:
                break

        if len(filings) >= MAX_FILINGS:
            break

    LOG.info("Discovered %d Form 4 index entries", len(filings))
    return filings


# ---------------------------------------------------------------------------
# Step 2 — resolve .txt wrapper path → primary XML content
# ---------------------------------------------------------------------------

def _fetch_xml_from_filing(filename: str) -> str | None:
    """Return raw XML text for a Form 4 filing.

    filename is a path like 'edgar/data/<cik>/<acc_nodash>/<file>.txt'
    from the quarterly index.
    """
    base_archives = "https://www.sec.gov/Archives/"

    # Build the folder URL (drop the filename component)
    parts = filename.split("/")
    if len(parts) < 2:
        return None
    folder_path = "/".join(parts[:-1]) + "/"
    folder_url = base_archives + folder_path

    # Try fetching the folder listing to find the .xml document
    xml_content = _xml_content_from_folder(folder_url)
    if xml_content:
        return xml_content

    # Fallback: fetch the .txt submission wrapper and extract embedded XML
    txt_url = base_archives + filename
    try:
        resp = fetch(txt_url, timeout=20)
        text = resp.text
        # SGML wrapper may contain <?xml ...> or <ownershipDocument>
        start_tag = text.find("<?xml")
        if start_tag != -1:
            return text[start_tag:]
        start_tag = text.find("<ownershipDocument>")
        end_tag = text.find("</ownershipDocument>")
        if start_tag != -1 and end_tag != -1:
            return text[start_tag : end_tag + len("</ownershipDocument>")]
    except Exception as exc:
        LOG.debug("TXT fetch failed for %s: %s", txt_url, exc)

    return None


def _xml_content_from_folder(folder_url: str) -> str | None:
    """Fetch folder index page, find the primary .xml href, return its text."""
    try:
        resp = fetch(folder_url, timeout=20)
        text = resp.text
    except Exception as exc:
        LOG.debug("Folder fetch failed (%s): %s", folder_url, exc)
        return None

    matches = re.findall(r'href="([^"]+\.xml)"', text, re.IGNORECASE)
    if not matches:
        return None

    # Prefer the shortest name — usually the primary doc, not an attachment
    matches.sort(key=len)
    xml_href = matches[0]
    if xml_href.startswith("http"):
        xml_url = xml_href
    elif xml_href.startswith("/"):
        xml_url = f"https://www.sec.gov{xml_href}"
    else:
        xml_url = folder_url + xml_href

    try:
        r2 = fetch(xml_url, timeout=20)
        return r2.text
    except Exception as exc:
        LOG.debug("XML fetch failed (%s): %s", xml_url, exc)
        return None


# ---------------------------------------------------------------------------
# Step 3 — parse Form 4 XML into transaction rows
# ---------------------------------------------------------------------------

def _text(el: ET.Element | None, path: str) -> str:
    """Return stripped text of a descendent element, or empty string."""
    if el is None:
        return ""
    child = el.find(path)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _parse_form4_xml(xml_text: str, file_date: str) -> list[dict]:
    """Parse Form 4 XML and return one dict per non-derivative transaction."""
    rows: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        LOG.debug("XML parse error: %s", exc)
        return rows

    issuer = root.find(".//issuer")
    issuer_name = _text(issuer, "issuerName")
    ticker = _text(issuer, "issuerTradingSymbol")

    owner_el = root.find(".//reportingOwner")
    owner_name = (
        _text(owner_el, "reportingOwnerId/rptOwnerName") if owner_el is not None else ""
    )

    for txn in root.findall(".//nonDerivativeTransaction"):
        txn_date = _text(txn, "transactionDate/value")
        shares_txt = _text(txn, "transactionShares/value")
        price_txt = _text(txn, "transactionPricePerShare/value")
        ad_code = _text(txn, "transactionAcquiredDisposedCode/value")  # A or D

        try:
            shares = float(shares_txt) if shares_txt else None
        except ValueError:
            shares = None

        try:
            price = float(price_txt) if price_txt else None
        except ValueError:
            price = None

        txn_value = (
            shares * price
            if shares is not None and price is not None
            else None
        )

        rows.append(
            {
                "filing_date": file_date,
                "ticker": ticker,
                "issuer_name": issuer_name,
                "owner_name": owner_name,
                "transaction_date": txn_date or file_date,
                "shares": shares,
                "price_per_share": price,
                "acquired_disposed": ad_code,
                "transaction_value": txn_value,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect() -> None:
    today = datetime.now(tz=timezone.utc)
    start = today - timedelta(days=30)

    LOG.info(
        "Collecting Form 4 filings %s → %s (max %d)",
        start.strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
        MAX_FILINGS,
    )

    index_entries = _collect_via_full_index(start, today)

    all_rows: list[dict] = []

    for filing in index_entries:
        try:
            xml_content = _fetch_xml_from_filing(filing["filename"])
            if xml_content:
                rows = _parse_form4_xml(xml_content, filing["date_filed"])
                all_rows.extend(rows)
        except Exception as exc:
            LOG.debug("Skipping %s: %s", filing.get("filename", "?"), exc)
        time.sleep(_SLEEP)

    _EMPTY_COLS = [
        "filing_date",
        "ticker",
        "issuer_name",
        "owner_name",
        "transaction_date",
        "shares",
        "price_per_share",
        "acquired_disposed",
        "transaction_value",
    ]

    if not all_rows:
        LOG.warning("No Form 4 rows extracted — saving empty frame")
        df: pd.DataFrame = pd.DataFrame(columns=_EMPTY_COLS)
        df["filing_date"] = pd.to_datetime(df["filing_date"], utc=True)
        df = df.set_index("filing_date")
        save(df, "fundamentals", "SEC_Form4_Transactions_30d.parquet")
        return

    df = pd.DataFrame(all_rows)

    for col in ("shares", "price_per_share", "transaction_value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename filing_date → date so to_datetime_index uses it as the index
    df = df.rename(columns={"filing_date": "date"})
    df = to_datetime_index(df, col="date")

    save(df, "fundamentals", "SEC_Form4_Transactions_30d.parquet")


if __name__ == "__main__":
    collect()
