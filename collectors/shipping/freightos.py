"""
Container shipping / freight rate collector.

Sources:
  1. World Bank Commodity Prices (monthly Excel) — fuel oil, coal, natural gas
     → data/shipping/WB_ShippingCommodities_1m.parquet

  2. FRED freight-proxy series (no API key required):
       CORESTICKM159SFRBATL  — Atlanta Fed Sticky CPI (freight cost proxy)
       PCU4841484148          — PPI deep-sea freight
       PCU4842484248          — PPI inland-water freight
     → data/shipping/FRED_Shipping_{name}_1m.parquet

  3. UNCTAD Liner Shipping Connectivity Index (annual CSV)
     → data/shipping/UNCTAD_LSCI_1y.parquet
     Skipped silently with a WARNING if the endpoint is unavailable.
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save  # noqa: E402

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WB_COMMODITIES_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "18675f1d1639c7a34d463f59259e97d4-0050012023/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)

# Substrings to match column headers in the WB sheet (case-insensitive).
WB_COMMODITY_KEYWORDS = ("fuel oil", "coal", "natural gas")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

# name → FRED series ID
FRED_SERIES: dict[str, str] = {
    "StickyCPI": "CORESTICKM159SFRBATL",
    "PPI_WaterTransportation": "PCU483111483111",
    "PPI_InlandWaterFreight": "PCU484110484110",
}

UNCTAD_LSCI_URL = (
    "https://unctadstat.unctad.org/datacentre/dataDownload.aspx"
    "?y=2024&Reporter=0&tableId=3&dimensionBreakdown=Reporter"
)


# ---------------------------------------------------------------------------
# Source 1 — World Bank commodity prices
# ---------------------------------------------------------------------------

def _match_wb_columns(columns: list[str]) -> list[str]:
    """Return column labels that contain any commodity keyword."""
    matched: list[str] = []
    for col in columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in WB_COMMODITY_KEYWORDS):
            matched.append(col)
    return matched


def fetch_wb_commodities() -> pd.DataFrame:
    """Download WB CMO monthly Excel and return shipping-related commodity rows."""
    resp = fetch(WB_COMMODITIES_URL, timeout=120)
    xl = pd.ExcelFile(io.BytesIO(resp.content))

    # Sheet name may vary slightly; fall back to first sheet containing 'Monthly'
    sheet = next(
        (s for s in xl.sheet_names if "Monthly" in s),
        xl.sheet_names[0],
    )

    # Data rows start around row 7 (0-indexed row 6); header is one row above.
    raw = xl.parse(sheet, header=6)

    # First column is typically the date/period label.
    date_col = raw.columns[0]
    commodity_cols = _match_wb_columns(raw.columns.tolist()[1:])

    if not commodity_cols:
        raise ValueError(
            f"No commodity columns matched {WB_COMMODITY_KEYWORDS} in sheet '{sheet}'. "
            f"Available columns (first 20): {raw.columns.tolist()[:20]}"
        )

    keep = [date_col, *commodity_cols]
    df = raw[keep].copy()
    df.rename(columns={date_col: "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Coerce all commodity value columns to numeric
    for col in commodity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Source 2 — FRED series
# ---------------------------------------------------------------------------

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Fetch a single FRED CSV series and return a tidy two-column DataFrame."""
    url = FRED_CSV_URL.format(id=series_id)
    resp = fetch(url, timeout=60)
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Source 3 — UNCTAD LSCI
# ---------------------------------------------------------------------------

def fetch_unctad_lsci() -> pd.DataFrame:
    """
    Attempt to download the UNCTAD Liner Shipping Connectivity Index CSV.
    Returns an empty DataFrame (with a WARNING) if the endpoint is unavailable
    or does not return parseable CSV content.
    """
    try:
        resp = fetch(UNCTAD_LSCI_URL, timeout=60)
    except Exception as exc:
        LOG.warning("UNCTAD LSCI fetch failed — skipping: %s", exc)
        return pd.DataFrame()

    content_type = resp.headers.get("Content-Type", "")
    if "text/csv" not in content_type and "text/plain" not in content_type:
        # Try to parse anyway; if it fails, warn and skip.
        try:
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            LOG.warning(
                "UNCTAD LSCI response is not valid CSV (Content-Type: %s) — skipping: %s",
                content_type,
                exc,
            )
            return pd.DataFrame()
    else:
        df = pd.read_csv(io.StringIO(resp.text))

    if df.empty:
        LOG.warning("UNCTAD LSCI returned 0 rows — skipping")
        return pd.DataFrame()

    # Normalise a year/date column if present
    date_candidates = [c for c in df.columns if str(c).lower() in ("year", "date", "period")]
    if date_candidates:
        df.rename(columns={date_candidates[0]: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).reset_index(drop=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # --- World Bank commodities ---
    print("Fetching World Bank shipping-related commodities ...")
    try:
        df_wb = fetch_wb_commodities()
        save(df_wb, "shipping", "WB_ShippingCommodities_1m.parquet")
    except Exception as exc:
        LOG.warning("WB commodities failed — skipping: %s", exc)

    # --- FRED proxy series ---
    for name, series_id in FRED_SERIES.items():
        print(f"Fetching FRED {name} ({series_id}) ...")
        try:
            df_fred = fetch_fred_series(series_id)
            save(df_fred, "shipping", f"FRED_Shipping_{name}_1m.parquet")
        except Exception as exc:
            LOG.warning("FRED %s (%s) failed — skipping: %s", name, series_id, exc)

    # --- UNCTAD LSCI ---
    print("Fetching UNCTAD Liner Shipping Connectivity Index ...")
    df_unctad = fetch_unctad_lsci()
    if not df_unctad.empty:
        save(df_unctad, "shipping", "UNCTAD_LSCI_1y.parquet")
    else:
        print("  WARNING: UNCTAD LSCI skipped (no data).")


if __name__ == "__main__":
    main()
