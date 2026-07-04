"""
Fama-French factor data collector.
Source: Kenneth French Data Library (free, no API key)
Covers: 3-factor (MKT-RF, SMB, HML), 5-factor (+RMW, CMA), Momentum (MOM)
Output: data/factors/FF3_1d.parquet, FF5_1d.parquet, MOM_1d.parquet
"""
from __future__ import annotations
import io
import zipfile
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save, to_datetime_index

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

DATASETS = {
    "FF3_1d": "F-F_Research_Data_Factors_daily_CSV.zip",
    "FF5_1d": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "MOM_1d": "F-F_Momentum_Factor_daily_CSV.zip",
}


def parse_ff_zip(zipped_bytes: bytes, filename_hint: str) -> pd.DataFrame:
    """Extract and parse the CSV inside a French data zip."""
    with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
        raw = zf.read(csv_name).decode("latin-1")

    lines = raw.splitlines()

    # Find header row: starts with "," (blank first field = date index) and has factor names
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(",") and "," in stripped[1:]:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Cannot find header row in {filename_hint}")

    # Data rows immediately follow header; stop at first blank line (annual section)
    end = len(lines)
    for i in range(header_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "":
            end = i
            break

    block = "\n".join(lines[header_idx:end])
    df = pd.read_csv(io.StringIO(block), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d", utc=True, errors="coerce")
    df.index.name = "date"
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df[df.index.notna()].dropna(how="all").sort_index()


def main() -> None:
    for out_name, zip_file in DATASETS.items():
        url = f"{BASE_URL}/{zip_file}"
        print(f"Fetching {out_name} from {url} ...")
        resp = fetch(url)
        df = parse_ff_zip(resp.content, zip_file)
        save(df, "factors", f"{out_name}.parquet")


if __name__ == "__main__":
    main()
