"""
Geopolitical Risk Index collector.
Sources: FRED (free, no key needed) — proxy series for geopolitical risk
  - VIXCLS       : CBOE Volatility Index (VIX) — daily risk sentiment proxy
  - STLFSI       : St. Louis Fed Financial Stress Index (weekly)
  - BAMLH0A0HYM2 : ICE BofA HY Spread — geopolitical stress proxy (daily)
  - UMCSENT      : U. of Michigan Consumer Sentiment (monthly)
  - DPSACBW027SBOG: Deposits All Commercial Banks (weekly, systemic risk proxy)
Output: data/geopolitical/GEOPOLITICAL_<series>.parquet
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

FRED_SERIES = {
    "GEOPOLITICAL_VIX_1d": "VIXCLS",
    "GEOPOLITICAL_FinStressIndex_1w": "STLFSI",
    "GEOPOLITICAL_HYSpread_RiskProxy_1d": "BAMLH0A0HYM2",
    "GEOPOLITICAL_ConsumerSentiment_1m": "UMCSENT",
}


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = FRED_CSV.format(id=series_id)
    resp = fetch(url, timeout=60)
    df = pd.read_csv(io.StringIO(resp.text))
    # FRED CSV: columns are DATE and the series_id
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    for name, series_id in FRED_SERIES.items():
        print(f"Fetching {name} ({series_id}) ...")
        try:
            df = fetch_fred_series(series_id)
            save(df, "geopolitical", f"{name}.parquet")
        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
