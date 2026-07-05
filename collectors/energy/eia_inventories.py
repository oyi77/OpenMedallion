"""
EIA petroleum and natural gas data collector.
Sources (in priority order):
  1. FRED CSV (confirmed accessible) — EIA-sourced price and production series
  2. yfinance — energy futures as inventory-adjacent proxies (CL=F, NG=F, HO=F, RB=F)

Note: EIA direct API (api.eia.gov) and EIA XLS bulk files are network-blocked
      in this environment; FRED serves the same underlying series for most
      weekly/monthly EIA publications.

Output:
  data/energy/eia_petroleum_inventories_1w.parquet  — petroleum price/flow series
  data/energy/eia_natgas_storage_1w.parquet         — natural gas price + futures series
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

# EIA petroleum-related FRED series (confirmed 200 via CSV endpoint)
# These are weekly or daily EIA-sourced series published on FRED
PETROLEUM_SERIES: dict[str, str] = {
    "DCOILWTICO":  "wti_crude_usd_per_bbl",       # WTI crude spot price (daily → resample)
    "DCOILBRENTEU": "brent_crude_usd_per_bbl",    # Brent crude spot price
    "GASREGCOVW":  "regular_gasoline_usd_per_gal", # Weekly US regular gasoline price
    "GASALLCOVW":  "all_gasoline_usd_per_gal",     # Weekly US all grades gasoline
    "GASPRMCOVW":  "premium_gasoline_usd_per_gal", # Weekly US premium gasoline
    "DGASNYH":     "heating_oil_nyh_usd_per_gal",  # NY Harbor heating oil (daily)
}

# EIA natural gas-related FRED series
NATGAS_SERIES: dict[str, str] = {
    "DHHNGSP":     "henry_hub_spot_usd_per_mmbtu", # Henry Hub daily spot price
    "MHHNGSP":     "henry_hub_monthly_usd_per_mmbtu",
    "PNGASEUUSDM": "natgas_eu_imf_usd_per_mmbtu",  # European natural gas (IMF/FRED monthly)
    "PNRGINDEXM":  "energy_price_index_imf",        # IMF energy price index
}

# yfinance energy futures (fallback / supplement)
ENERGY_FUTURES: dict[str, str] = {
    "CL=F": "wti_crude_futures",
    "BZ=F": "brent_crude_futures",
    "NG=F": "natgas_futures",
    "HO=F": "heating_oil_futures",
    "RB=F": "gasoline_futures",
}


def _fetch_fred_series(series_id: str, col: str) -> pd.Series | None:
    """Fetch one FRED series; return a named Series indexed by UTC date."""
    url = FRED_CSV.format(id=series_id)
    try:
        resp = fetch(url, timeout=30)
    except Exception as exc:
        print(f"    WARNING: {series_id} — {exc}")
        return None
    df = pd.read_csv(StringIO(resp.text))
    if df.shape[1] < 2:
        return None
    df.columns = ["date", col]
    df = df[df[col] != "."].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")[col].sort_index()


def _fetch_yfinance_energy() -> pd.DataFrame | None:
    """Download daily close prices for energy futures via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        print(f"    WARNING: yfinance not installed — {exc}")
        return None
    try:
        raw = yf.download(
            list(ENERGY_FUTURES.keys()),
            start="1990-01-01",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        print(f"    WARNING: yfinance.download failed — {exc}")
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
    close = close.rename(columns=ENERGY_FUTURES)
    close.index = pd.to_datetime(close.index, utc=True)
    close.index.name = "date"
    return close.sort_index().dropna(how="all")


def collect_petroleum_inventories() -> None:
    """Merge petroleum FRED series + yfinance energy futures into one file."""
    frames: list[pd.Series] = []

    print("  Fetching FRED petroleum series ...")
    for series_id, col in PETROLEUM_SERIES.items():
        print(f"    {series_id} -> {col}")
        try:
            s = _fetch_fred_series(series_id, col)
            if s is not None and not s.empty:
                frames.append(s)
                print(f"      {len(s)} observations")
            else:
                print(f"      WARNING: empty")
        except Exception as exc:
            print(f"      WARNING: {series_id} — {exc}")

    # Supplement with yfinance energy futures
    print("  Fetching yfinance energy futures ...")
    try:
        yf_df = _fetch_yfinance_energy()
        if yf_df is not None and not yf_df.empty:
            for col in yf_df.columns:
                if not yf_df[col].isna().all():
                    frames.append(yf_df[col].dropna())
                    print(f"    {col}: {yf_df[col].notna().sum()} observations")
    except Exception as exc:
        print(f"    WARNING: yfinance energy — {exc}")

    if not frames:
        print("  ERROR: no data collected — skipping save")
        return

    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    print(f"  Columns: {df.columns.tolist()}")
    save(df, "energy", "eia_petroleum_inventories_1w.parquet")


def collect_natgas_storage() -> None:
    """Fetch natural gas price series from FRED; supplement with yfinance futures."""
    frames: list[pd.Series] = []

    print("  Fetching FRED natural gas series ...")
    for series_id, col in NATGAS_SERIES.items():
        print(f"    {series_id} -> {col}")
        try:
            s = _fetch_fred_series(series_id, col)
            if s is not None and not s.empty:
                frames.append(s)
                print(f"      {len(s)} observations")
            else:
                print(f"      WARNING: empty")
        except Exception as exc:
            print(f"      WARNING: {series_id} — {exc}")

    # Add NG=F futures from yfinance if not already in petroleum file
    print("  Fetching yfinance natgas futures (NG=F) ...")
    try:
        import yfinance as yf
        raw = yf.download(["NG=F"], start="1990-01-01", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]["NG=F"]
        else:
            close = raw["Close"]
        close.index = pd.to_datetime(close.index, utc=True)
        close.index.name = "date"
        close = close.dropna().rename("natgas_futures_usd_per_mmbtu")
        frames.append(close)
        print(f"    NG=F: {len(close)} observations")
    except Exception as exc:
        print(f"    WARNING: NG=F yfinance — {exc}")

    if not frames:
        print("  ERROR: no data collected — skipping save")
        return

    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    print(f"  Columns: {df.columns.tolist()}")
    save(df, "energy", "eia_natgas_storage_1w.parquet")


def main() -> None:
    print("Fetching EIA petroleum series (FRED + yfinance) ...")
    collect_petroleum_inventories()

    print("\nFetching EIA natural gas series (FRED + yfinance) ...")
    collect_natgas_storage()


if __name__ == "__main__":
    main()
