"""
Frankfurter ECB exchange rates collector.
Source: frankfurter.app — free, no API key, based on official ECB data.
Covers: EUR base rates for 32 currencies since 1999.
These are OFFICIAL ECB rates, distinct from market rates in forex/.
Output: data/forex/ECB_EUR<XXX>_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

FRANKFURTER = "https://api.frankfurter.dev/v1"


def fetch_all_currencies() -> list[str]:
    resp = fetch(f"{FRANKFURTER}/currencies")
    return list(resp.json().keys())


def fetch_history(to_currency: str, start: str = "1999-01-04") -> pd.DataFrame:
    """Fetch full EUR/XXX history from ECB via Frankfurter."""
    resp = fetch(f"{FRANKFURTER}/{start}..", params={"from": "EUR", "to": to_currency}, timeout=30)
    data = resp.json()
    rates = data.get("rates", {})
    if not rates:
        return pd.DataFrame()

    rows = [{"date": d, "value": v.get(to_currency)} for d, v in rates.items() if v.get(to_currency)]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").sort_index().dropna()
    return df


def main() -> None:
    currencies = fetch_all_currencies()
    # Remove EUR itself
    currencies = [c for c in currencies if c != "EUR"]
    print(f"Found {len(currencies)} currencies: {currencies}")

    for ccy in currencies:
        print(f"  Fetching ECB EUR/{ccy} ...")
        try:
            df = fetch_history(ccy)
            save(df, "forex", f"ECB_EUR{ccy}_1d.parquet")
        except Exception as exc:
            print(f"  WARNING: EUR/{ccy} — {exc}")


if __name__ == "__main__":
    main()
