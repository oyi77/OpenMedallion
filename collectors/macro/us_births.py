"""
Collect US births data from CDC WONDER.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save, to_datetime_index


def collect():
    """Collect CDC birth data via FRED (monthly births)."""
    import urllib.request, io
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRTH"
    print(f"Fetching US births from FRED: {url}")
    with urllib.request.urlopen(url, timeout=30) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    
    df.columns = ["date", "births_thousands"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df["births_thousands"] = pd.to_numeric(df["births_thousands"], errors="coerce")
    
    out = save(df, "macro", "us_births.parquet")
    print(f"Saved {out}: {len(df)} rows")
    return out


if __name__ == "__main__":
    collect()
