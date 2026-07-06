"""
Collect US population estimates from FRED (POPTHM).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save


def collect():
    """Collect FRED POPTHM — US population (monthly)."""
    import urllib.request, io
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=POPTHM"
    print(f"Fetching US population from FRED: {url}")
    with urllib.request.urlopen(url, timeout=30) as r:
        df = pd.read_csv(io.BytesIO(r.read()))

    df.columns = ["date", "pop_thousands"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df["pop_thousands"] = pd.to_numeric(df["pop_thousands"], errors="coerce")

    out = save(df, "macro", "us_population.parquet")
    print(f"Saved {out}: {len(df)} rows")
    return out


if __name__ == "__main__":
    collect()
