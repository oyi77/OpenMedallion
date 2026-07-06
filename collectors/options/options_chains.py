"""
Options chain snapshots — open interest, IV by strike/expiry.
Source: yfinance (free, no key required)
Tickers: SPY, QQQ, IWM, GLD, TLT + AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, JPM, BRK-B
Strategy: next 4 expiry dates per ticker to stay fast; one file per ticker + summary.

Outputs:
  data/options/chain_{TICKER}_snapshot.parquet  — per-ticker chain (strike, OI, IV, …)
  data/options/chains_iv_surface_summary.parquet — aggregated IV surface summary
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index  # noqa: E402

TICKERS = [
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AVGO", "JPM", "BRK-B",
]

MAX_EXPIRIES = 4   # next N expiry dates per ticker
SLEEP = 0.5        # seconds between tickers


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns (yfinance >= 0.2.x)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def _collect_ticker(symbol: str, snapshot_date: str) -> pd.DataFrame | None:
    """Fetch options chain for next MAX_EXPIRIES expiries. Returns combined df or None."""
    try:
        tk = yf.Ticker(symbol)
        expiries = tk.options
    except Exception as exc:
        print(f"  WARNING {symbol}: could not get expiry list — {exc}")
        return None

    if not expiries:
        print(f"  WARNING {symbol}: no expiries available")
        return None

    frames: list[pd.DataFrame] = []
    for expiry in expiries[:MAX_EXPIRIES]:
        try:
            chain = tk.option_chain(expiry)
            for opt_type, raw in (("call", chain.calls), ("put", chain.puts)):
                df = _flatten_columns(raw.copy())
                df["ticker"] = symbol
                df["expiry"] = expiry
                df["option_type"] = opt_type
                df["snapshot_date"] = snapshot_date
                frames.append(df)
            time.sleep(0.2)
        except Exception as exc:
            print(f"  WARNING {symbol} expiry {expiry}: {exc}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Keep only useful columns; others vary by yfinance version
    keep = [
        "ticker", "expiry", "option_type", "snapshot_date",
        "strike", "lastPrice", "bid", "ask",
        "volume", "openInterest", "impliedVolatility", "inTheMoney",
    ]
    existing = [c for c in keep if c in combined.columns]
    combined = combined[existing]

    # Coerce numerics
    for col in ("strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"])
    return combined


def _build_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-(ticker, expiry, option_type): median IV, total OI, total volume."""
    rows: list[dict] = []
    for symbol, df in frames.items():
        if df is None or df.empty:
            continue
        for (expiry, opt_type), grp in df.groupby(["expiry", "option_type"]):
            rows.append({
                "ticker": symbol,
                "expiry": expiry,
                "option_type": opt_type,
                "median_iv": grp["impliedVolatility"].median() if "impliedVolatility" in grp else None,
                "total_oi": grp["openInterest"].sum() if "openInterest" in grp else None,
                "total_volume": grp["volume"].sum() if "volume" in grp else None,
                "snapshot_date": grp["snapshot_date"].iloc[0] if "snapshot_date" in grp else None,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> None:
    today = date.today().isoformat()
    print(f"Fetching: options chain snapshots for {len(TICKERS)} tickers (next {MAX_EXPIRIES} expiries each)")

    chain_frames: dict[str, pd.DataFrame | None] = {}
    for symbol in TICKERS:
        print(f"  {symbol}...")
        df = _collect_ticker(symbol, today)
        chain_frames[symbol] = df
        if df is not None:
            safe_sym = symbol.replace("-", "_")
            try:
                df_save = df.copy()
                df_save = to_datetime_index(df_save, col="snapshot_date")
                save(df_save, "options", f"chain_{safe_sym}_snapshot.parquet")
                print(f"    -> {len(df_save)} rows saved")
            except Exception as exc:
                print(f"    WARNING save {symbol}: {exc}")
        time.sleep(SLEEP)

    # Build and save summary
    try:
        summary = _build_summary(chain_frames)
        if not summary.empty:
            summary["snapshot_date"] = pd.to_datetime(summary["snapshot_date"])
            summary = to_datetime_index(summary, col="snapshot_date")
            save(summary, "options", "chains_iv_surface_summary.parquet")
            print(f"  -> IV surface summary: {len(summary)} rows saved")
    except Exception as exc:
        print(f"  WARNING summary: {exc}")


if __name__ == "__main__":
    main()
