"""
SEC EDGAR earnings and insider trades collector.
Source: SEC EDGAR full-text search + EDGAR APIs — free, no key required.
Covers: Insider trades (Form 4), company facts (XBRL), earnings dates.
Output: data/fundamentals/SEC_<ticker>_form4_insider.parquet
         data/fundamentals/SEC_<ticker>_facts_1q.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

SEC_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {"User-Agent": "OpenMedallion research@example.com"}  # EDGAR requires User-Agent

# Top 30 S&P500 tickers to collect
TOP_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "UNH", "LLY", "V", "XOM", "MA", "AVGO", "PG", "HD",
    "JNJ", "MRK", "COST", "ABBV", "BAC", "KO", "NFLX", "CVX",
    "AMD", "WMT", "PEP", "TMO", "ORCL",
]


def get_cik_map() -> dict[str, str]:
    """Fetch ticker -> CIK mapping from EDGAR."""
    import requests
    resp = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in data.values()
    }


def fetch_company_facts(cik: str, ticker: str) -> pd.DataFrame:
    """Fetch XBRL company facts for key financial metrics."""
    import requests
    url = SEC_COMPANY_FACTS.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()

    data = resp.json()
    facts = data.get("facts", {})

    rows = []
    # Extract key US-GAAP metrics
    gaap = facts.get("us-gaap", {})
    metrics_wanted = {
        "Revenues": "revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_v2",
        "NetIncomeLoss": "net_income",
        "EarningsPerShareBasic": "eps_basic",
        "EarningsPerShareDiluted": "eps_diluted",
        "GrossProfit": "gross_profit",
        "OperatingIncomeLoss": "operating_income",
        "CashAndCashEquivalentsAtCarryingValue": "cash",
        "LongTermDebt": "long_term_debt",
        "CommonStockSharesOutstanding": "shares_outstanding",
    }

    for gaap_name, col_name in metrics_wanted.items():
        if gaap_name not in gaap:
            continue
        units = gaap[gaap_name].get("units", {})
        # Use USD or shares
        for unit_type, entries in units.items():
            for entry in entries:
                if entry.get("form") in ("10-K", "10-Q", "20-F"):
                    rows.append({
                        "date": entry.get("end"),
                        "metric": col_name,
                        "value": entry.get("val"),
                        "form": entry.get("form"),
                        "period": entry.get("fp"),
                        "unit": unit_type,
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date").sort_index()


def fetch_insider_trades(cik: str) -> pd.DataFrame:
    """Fetch Form 4 insider trade filings from EDGAR submissions."""
    import requests
    url = SEC_SUBMISSIONS.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()

    data = resp.json()
    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        return pd.DataFrame()

    form_col = filings.get("form", [])
    date_col = filings.get("filingDate", [])
    accession_col = filings.get("accessionNumber", [])

    rows = [
        {"date": d, "form": f, "accession": a}
        for f, d, a in zip(form_col, date_col, accession_col)
        if f == "4"  # Form 4 = insider trades
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.set_index("date").sort_index()


def main() -> None:
    print("Fetching SEC EDGAR ticker->CIK map ...")
    try:
        cik_map = get_cik_map()
    except Exception as exc:
        print(f"  ERROR: Cannot fetch CIK map — {exc}")
        return

    for ticker in TOP_TICKERS:
        clean = ticker.replace("-", "_")
        cik = cik_map.get(ticker.upper()) or cik_map.get(ticker.replace("-", ".").upper())
        if not cik:
            print(f"  WARNING: CIK not found for {ticker}")
            continue

        print(f"Fetching SEC EDGAR {ticker} (CIK {cik}) ...")
        try:
            df_facts = fetch_company_facts(cik, ticker)
            if not df_facts.empty:
                save(df_facts, "fundamentals", f"SEC_{clean}_facts_1q.parquet")
            time.sleep(0.15)  # EDGAR rate limit: 10 req/s

            df_insiders = fetch_insider_trades(cik)
            if not df_insiders.empty:
                save(df_insiders, "fundamentals", f"SEC_{clean}_form4_insider.parquet")
            time.sleep(0.15)

        except Exception as exc:
            print(f"  WARNING: {ticker} — {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main()
