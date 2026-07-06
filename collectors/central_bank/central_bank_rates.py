"""
Collect central bank policy rates from FRED.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save, to_datetime_index

_RATES = {
    "DFEDTARU": "us_upper",
    "DFEDTARL": "us_lower",
    "IRSTCB01EZM156N": "ecb",
    "IRSTCB01JPM156N": "japan",
    "IRSTCB01GBM156N": "uk",
}


def collect():
    """Collect central bank policy rates."""
    import urllib.request, io, time
    rows = []
    for sid, label in _RATES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        print(f"  Fetching {label} ({sid})...")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            df.columns = ["date", "rate"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
            df = df.dropna(subset=["date", "rate"])
            df["central_bank"] = label
            rows.append(df)
        except Exception as e:
            print(f"    WARN: {label}: {e}")
        time.sleep(0.5)

    combined = pd.concat(rows).set_index("date").sort_index()
    out = save(combined, "central_bank", "central_bank_rates.parquet")
    print(f"Saved {out}: {len(combined)} rows")
    return out


if __name__ == "__main__":
    collect()
