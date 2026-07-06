"""
Port congestion and shipping delay proxy collector.

Outputs
-------
data/shipping/WB_LPI_overall.parquet
    World Bank Logistics Performance Index — overall score, all countries,
    latest available observation per country from the JSON API.

data/shipping/shipping_equity_proxies_1d.parquet
    Daily OHLCV (long format, 'ticker' column) for a basket of container /
    dry-bulk / tanker operators whose stock prices correlate with freight
    rates and port congestion.

data/shipping/FRED_container_trade_proxies_1d.parquet
    FRED demand-side shipping proxies: consumer sentiment index, durable
    goods orders, and manufacturing IP — all in tidy long format with a
    'series' column.
"""
from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save, to_datetime_index  # noqa: E402

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WB_LPI_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/LP.LPI.OVRL.XQ"
    "?format=json&per_page=300&mrv=1"
)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

# Demand-side proxies: consumer sentiment, durable goods orders, mfg IP
FRED_SERIES: dict[str, str] = {
    "consumer_sentiment": "CSCICP03USM665S",
    "durable_goods_orders": "DGORDER",
    "mfg_ip_index": "IPG3361T3S",
}

# Container / dry-bulk / tanker basket
SHIPPING_TICKERS: list[str] = [
    "ZIM",   # ZIM Integrated Shipping
    "MATX",  # Matson (container, Pacific)
    "SBLK",  # Star Bulk (dry bulk)
    "GOGL",  # Golden Ocean (dry bulk)
    "NMM",   # Navios Maritime Partners
    "CMRE",  # Costamare (container)
    "DAC",   # Danaos (container)
    "STNG",  # Scorpio Tankers
    "INSW",  # International Seaways (tanker)
    "TNK",   # Teekay Tankers
]

_SLEEP = 0.3  # seconds between network calls


# ---------------------------------------------------------------------------
# Source 1 — World Bank LPI
# ---------------------------------------------------------------------------

def fetch_wb_lpi() -> pd.DataFrame:
    """
    Download WB LPI overall score from the JSON API.

    The API paginates via 'pages'; we follow all pages and keep the most
    recent non-null value per country.
    """
    rows: list[dict] = []
    page = 1

    while True:
        url = f"{WB_LPI_URL}&page={page}"
        try:
            resp = fetch(url)
        except Exception as exc:
            LOG.error("WB LPI page %d failed: %s", page, exc)
            break

        payload = resp.json()
        if len(payload) < 2 or not payload[1]:
            break

        meta = payload[0]
        data = payload[1]

        for item in data:
            if item.get("value") is None:
                continue
            rows.append(
                {
                    "country_code": item.get("countryiso3code") or item["country"]["id"],
                    "country": item["country"]["value"],
                    "year": int(item["date"]),
                    "lpi_overall": float(item["value"]),
                }
            )

        total_pages = meta.get("pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(_SLEEP)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Keep most recent year per country
    df = (
        df.sort_values("year", ascending=False)
        .drop_duplicates(subset=["country_code"])
        .reset_index(drop=True)
    )
    # Use year-end date as the time index
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31", utc=True)
    df = df.drop(columns=["year"])
    return to_datetime_index(df)


# ---------------------------------------------------------------------------
# Source 2 — Shipping equity proxies via yfinance
# ---------------------------------------------------------------------------

def fetch_shipping_equities(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV for the shipping basket and return long-format DataFrame.
    One ticker at a time to respect rate limits.
    """
    frames: list[pd.DataFrame] = []

    for ticker in SHIPPING_TICKERS:
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                LOG.warning("yfinance: no data for %s", ticker)
                time.sleep(_SLEEP)
                continue

            # yfinance may return MultiIndex columns when single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            raw["ticker"] = ticker
            raw.index = pd.to_datetime(raw.index, utc=True)
            frames.append(raw)
        except Exception as exc:
            LOG.error("yfinance error for %s: %s", ticker, exc)

        time.sleep(_SLEEP)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    return df


# ---------------------------------------------------------------------------
# Source 3 — FRED demand-side proxies
# ---------------------------------------------------------------------------

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Fetch a single FRED CSV series; return tidy two-column DataFrame."""
    url = FRED_CSV_URL.format(id=series_id)
    resp = fetch(url)
    df = pd.read_csv(io.StringIO(resp.text), na_values=[".", ""])
    df.columns = ["date", "value"]
    df = df.dropna(subset=["value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def fetch_fred_proxies() -> pd.DataFrame:
    """
    Fetch all FRED series; return long-format DataFrame with a 'series' column.
    """
    frames: list[pd.DataFrame] = []

    for name, series_id in FRED_SERIES.items():
        try:
            df = fetch_fred_series(series_id)
            df["series"] = name
            df["fred_id"] = series_id
            frames.append(df)
        except Exception as exc:
            LOG.error("FRED series %s (%s) failed: %s", name, series_id, exc)

        time.sleep(_SLEEP)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return to_datetime_index(combined)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # --- World Bank LPI ---
    print("Fetching World Bank LPI overall scores ...")
    try:
        df_lpi = fetch_wb_lpi()
        save(df_lpi, "shipping", "WB_LPI_overall.parquet")
    except Exception as exc:
        LOG.error("WB LPI collection failed: %s", exc)

    # --- Shipping equity proxies ---
    print("Fetching shipping equity proxies via yfinance ...")
    try:
        df_eq = fetch_shipping_equities()
        save(df_eq, "shipping", "shipping_equity_proxies_1d.parquet")
    except Exception as exc:
        LOG.error("Shipping equity collection failed: %s", exc)

    # --- FRED demand proxies ---
    print("Fetching FRED container trade proxies ...")
    try:
        df_fred = fetch_fred_proxies()
        save(df_fred, "shipping", "FRED_container_trade_proxies_1d.parquet")
    except Exception as exc:
        LOG.error("FRED proxy collection failed: %s", exc)


if __name__ == "__main__":
    main()
