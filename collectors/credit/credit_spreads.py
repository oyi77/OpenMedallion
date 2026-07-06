"""
Collect credit spreads from FRED.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save

_SERIES = {
    "BAMLC0A0CM": "ig_oas",
    "BAMLH0A0HYM2": "hy_oas",
    "T10Y2Y": "yield_curve_10y2y",
    "T10Y3M": "yield_curve_10y3m",
    "DPRIME": "fed_prime_rate",
}


def collect():
    """Collect credit spread indicators."""
    import urllib.request, io, time
    frames = {}
    for sid, label in _SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        print(f"  Fetching {label} ({sid})...")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            df.columns = ["date", label]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[label] = pd.to_numeric(df[label], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
            frames[label] = df[label]
        except Exception as e:
            print(f"    WARN: {label}: {e}")
        time.sleep(0.5)

    combined = pd.concat(frames.values(), axis=1).sort_index()
    out = save(combined, "credit", "credit_spreads.parquet")
    print(f"Saved {out}: {len(combined)} rows")
    return out


if __name__ == "__main__":
    collect()
