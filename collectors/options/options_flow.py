"""
Options flow collector.
Sources:
  1. CBOE CDN per-date CSVs (probed once; all 30 skipped immediately on 403)
  2. Yahoo Finance v8 API (free, no key) — reliable fallback

Outputs:
  data/options/CBOE_SPX_EOD_1d.parquet       — SPX daily OHLCV + VIX
  data/options/CBOE_VX_Futures_EOD_1d.parquet — VIX term structure proxies
  data/options/CBOE_PutCall_Ratio_1d.parquet  — equity put/call ratio or proxy

All 404/403s: print WARNING, skip gracefully.
Dates: business days only, weekends skipped.
"""
from __future__ import annotations

import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import save  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CBOE_SPX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SPX_EOD_{date}.csv"
CBOE_VX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VX_EOD_{date}.csv"
CBOE_PC_URL = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/equitypc.csv"

YAHOO_CHART = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}

LOOKBACK_TRADING_DAYS = 30
YAHOO_DELAY = 1.5  # seconds between Yahoo requests to avoid 429


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _business_days_back(n: int) -> list[date]:
    """Return the last *n* Mon–Fri dates ending yesterday."""
    result: list[date] = []
    day = date.today() - timedelta(days=1)
    while len(result) < n:
        if day.weekday() < 5:
            result.append(day)
        day -= timedelta(days=1)
    return result  # newest first


def _make_session() -> requests.Session:
    """Return a requests Session with Yahoo Finance headers."""
    s = requests.Session()
    s.headers.update(YAHOO_HEADERS)
    return s


def _cboe_cdn_accessible(url: str) -> bool:
    """Probe a single CBOE CDN URL. Return True only on HTTP 200 with CSV content."""
    try:
        r = requests.get(url, timeout=10, headers=YAHOO_HEADERS)
        if r.status_code != 200:
            return False
        ct = r.headers.get("content-type", "")
        return "html" not in ct.lower()
    except Exception:
        return False


def _fetch_cboe_cdn_csv(url: str, label: str) -> pd.DataFrame | None:
    """Fetch one CBOE CDN CSV. Returns None on any error."""
    try:
        r = requests.get(url, timeout=15, headers=YAHOO_HEADERS)
        if r.status_code in (403, 404):
            print(f"  WARNING: {label} — HTTP {r.status_code}, skipping")
            return None
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "html" in ct.lower():
            print(f"  WARNING: {label} — HTML response (bot-blocked), skipping")
            return None
        return pd.read_csv(io.StringIO(r.text))
    except Exception as exc:
        print(f"  WARNING: {label} — {exc}, skipping")
        return None


def _yahoo_chart(
    symbol: str,
    range_: str = "3mo",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Yahoo Finance v8 chart API.
    Returns DataFrame with DatetimeIndex (UTC).
    Raises on failure.
    """
    requester = session or requests.Session()
    if session is None:
        requester.headers.update(YAHOO_HEADERS)

    resp = requester.get(
        YAHOO_CHART.format(symbol=symbol),
        params={"interval": "1d", "range": range_},
        timeout=20,
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    result = data.get("chart", {}).get("result")
    if not result:
        err = data.get("chart", {}).get("error", {})
        raise ValueError(f"Yahoo Finance error for {symbol}: {err}")

    r = result[0]
    timestamps: list[int] = r.get("timestamp", [])
    quote: dict[str, list] = r.get("indicators", {}).get("quote", [{}])[0]

    df = pd.DataFrame(
        {
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    )
    df.index.name = "date"
    return df.sort_index().dropna(how="all")


def _yahoo_close_series(
    symbol: str,
    col_name: str,
    session: requests.Session,
    range_: str = "3mo",
) -> pd.DataFrame | None:
    """Fetch close series from Yahoo; return None on failure (already warned)."""
    try:
        time.sleep(YAHOO_DELAY)
        df = _yahoo_chart(symbol, range_=range_, session=session)
        return df[["close"]].rename(columns={"close": col_name})
    except Exception as exc:
        print(f"  WARNING: {symbol} fetch failed — {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. SPX EOD
# ---------------------------------------------------------------------------
def collect_spx_eod(session: requests.Session) -> pd.DataFrame:
    """
    Probe CBOE CDN once; if accessible, fetch all 30 trading days.
    Otherwise fall back to Yahoo Finance ^GSPC + ^VIX.
    """
    trading_days = _business_days_back(LOOKBACK_TRADING_DAYS)
    probe_date = trading_days[0].strftime("%Y%m%d")
    probe_url = CBOE_SPX_URL.format(date=probe_date)

    if _cboe_cdn_accessible(probe_url):
        frames: list[pd.DataFrame] = []
        for day in trading_days:
            date_str = day.strftime("%Y%m%d")
            url = CBOE_SPX_URL.format(date=date_str)
            df = _fetch_cboe_cdn_csv(url, f"SPX_EOD_{date_str}")
            if df is None:
                continue
            df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            df["trade_date"] = pd.Timestamp(day, tz="UTC")
            frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            return combined.set_index("trade_date").sort_index()

    print(f"  WARNING: CBOE CDN blocked for SPX (HTTP 403) — using Yahoo Finance ^GSPC + ^VIX")
    spx = _yahoo_close_series("^GSPC", "spx_close", session)
    if spx is None:
        return pd.DataFrame()
    time.sleep(YAHOO_DELAY)
    try:
        full = _yahoo_chart("^GSPC", range_="3mo", session=session)
        vix = _yahoo_close_series("^VIX", "vix_close", session)
        if vix is not None:
            full = full.join(vix, how="left")
        full.index.name = "date"
        return full
    except Exception as exc:
        print(f"  WARNING: SPX EOD Yahoo fallback failed — {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. VX Futures EOD (VIX term structure)
# ---------------------------------------------------------------------------
def collect_vx_eod(session: requests.Session) -> pd.DataFrame:
    """
    Probe CBOE CDN once; if accessible, fetch all 30 trading days.
    Otherwise fall back to Yahoo Finance VIX term structure proxies.
    """
    trading_days = _business_days_back(LOOKBACK_TRADING_DAYS)
    probe_date = trading_days[0].strftime("%Y%m%d")
    probe_url = CBOE_VX_URL.format(date=probe_date)

    if _cboe_cdn_accessible(probe_url):
        frames: list[pd.DataFrame] = []
        for day in trading_days:
            date_str = day.strftime("%Y%m%d")
            url = CBOE_VX_URL.format(date=date_str)
            df = _fetch_cboe_cdn_csv(url, f"VX_EOD_{date_str}")
            if df is None:
                continue
            df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            df["trade_date"] = pd.Timestamp(day, tz="UTC")
            frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            return combined.set_index("trade_date").sort_index()

    print(f"  WARNING: CBOE CDN blocked for VX (HTTP 403) — using Yahoo Finance VIX term structure")
    symbols = {
        "vix_spot": "^VIX",
        "vix3m": "^VIX3M",
        "vvix": "^VVIX",
        "skew": "^SKEW",
        "ovx": "^OVX",
    }
    frames_yf: list[pd.DataFrame] = []
    for col, sym in symbols.items():
        df = _yahoo_close_series(sym, col, session)
        if df is not None:
            frames_yf.append(df)

    if not frames_yf:
        return pd.DataFrame()

    combined = frames_yf[0]
    for f in frames_yf[1:]:
        combined = combined.join(f, how="outer")
    combined.index.name = "date"
    return combined.sort_index()


# ---------------------------------------------------------------------------
# 3. Put/Call Ratio
# ---------------------------------------------------------------------------
def _parse_cboe_pc_text(text: str) -> pd.DataFrame | None:
    """Parse CBOE equity put/call CSV, skipping comment/header preamble."""
    lines = text.splitlines()
    header_idx: int | None = None
    for i, line in enumerate(lines):
        upper = line.upper()
        if "DATE" in upper and ("RATIO" in upper or "PUT" in upper or "CALL" in upper):
            header_idx = i
            break
        first = line.split(",")[0].strip()
        if len(first) >= 8 and (first[4:5] in "-/" or first[2:3] == "/"):
            header_idx = i
            break

    if header_idx is None:
        return None

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df.columns = df.columns.str.strip().str.upper()

    rename: dict[str, str] = {}
    for col in df.columns:
        if col == "DATE":
            rename[col] = "date"
        elif ("RATIO" in col or "P/C" in col or "PC" in col) and "pc_ratio" not in rename.values():
            rename[col] = "pc_ratio"
    df = df.rename(columns=rename)

    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "pc_ratio" in df.columns:
        df["pc_ratio"] = pd.to_numeric(df["pc_ratio"], errors="coerce")

    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df if not df.empty else None


def collect_put_call_ratio(session: requests.Session) -> pd.DataFrame:
    """
    Fetch CBOE equity put/call ratio (static CSV).
    Falls back to Yahoo Finance sentiment proxies if CBOE is blocked.
    """
    try:
        resp = requests.get(CBOE_PC_URL, timeout=20, headers=YAHOO_HEADERS)
        if resp.status_code == 200 and "html" not in resp.headers.get("content-type", "").lower():
            parsed = _parse_cboe_pc_text(resp.text)
            if parsed is not None:
                return parsed
            print("  WARNING: CBOE put/call CSV parse failed (unexpected format), using fallback")
        else:
            code = resp.status_code
            print(f"  WARNING: CBOE put/call URL returned HTTP {code} / HTML (bot-blocked), using fallback")
    except Exception as exc:
        print(f"  WARNING: CBOE put/call fetch failed — {exc}, using fallback")

    print("  INFO: Using Yahoo Finance ^SKEW + ^VVIX + ^VIX3M + ^VIX as put/call proxy")
    symbols = {
        "skew": "^SKEW",
        "vvix": "^VVIX",
        "vix3m": "^VIX3M",
        "vix_spot": "^VIX",
    }
    frames: list[pd.DataFrame] = []
    for col, sym in symbols.items():
        df = _yahoo_close_series(sym, col, session)
        if df is not None:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")
    combined.index.name = "date"
    return combined.sort_index()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    session = _make_session()

    print("Fetching CBOE SPX EOD options data (last 30 trading days)...")
    spx = collect_spx_eod(session)
    save(spx, "options", "CBOE_SPX_EOD_1d.parquet")

    time.sleep(YAHOO_DELAY)

    print("Fetching CBOE VX Futures EOD data (last 30 trading days)...")
    vx = collect_vx_eod(session)
    save(vx, "options", "CBOE_VX_Futures_EOD_1d.parquet")

    time.sleep(YAHOO_DELAY)

    print("Fetching CBOE Equity Put/Call ratio...")
    pc = collect_put_call_ratio(session)
    save(pc, "options", "CBOE_PutCall_Ratio_1d.parquet")


if __name__ == "__main__":
    main()
