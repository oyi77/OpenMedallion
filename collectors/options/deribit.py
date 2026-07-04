"""
Deribit options IV surface collector.
Source: Deribit public REST API — free, no API key required.
Covers: BTC and ETH options — IV surface snapshot, term structure, skew.
Output:
  data/options/Deribit_BTC_options_snapshot.parquet
  data/options/Deribit_ETH_options_snapshot.parquet
  data/options/Deribit_BTC_iv_term_structure_1d.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

DERIBIT = "https://www.deribit.com/api/v2/public"


def get_instruments(currency: str, kind: str = "option") -> list[dict]:
    resp = fetch(f"{DERIBIT}/get_instruments", params={"currency": currency, "kind": kind, "expired": "false"})
    return resp.json().get("result", [])


def get_ticker(instrument_name: str) -> dict:
    resp = fetch(f"{DERIBIT}/ticker", params={"instrument_name": instrument_name})
    return resp.json().get("result", {})


def get_iv_surface(currency: str) -> pd.DataFrame:
    """Fetch all live options and extract IV surface."""
    instruments = get_instruments(currency)
    if not instruments:
        return pd.DataFrame()

    rows = []
    for inst in instruments[:150]:  # cap to avoid rate limiting
        name = inst["instrument_name"]
        try:
            t = get_ticker(name)
            if not t or t.get("mark_iv") is None:
                continue
            rows.append({
                "instrument": name,
                "expiry": inst.get("expiration_timestamp"),
                "strike": inst.get("strike"),
                "option_type": inst.get("option_type"),
                "mark_iv": t.get("mark_iv"),
                "bid_iv": t.get("bid_iv"),
                "ask_iv": t.get("ask_iv"),
                "mark_price": t.get("mark_price"),
                "underlying_price": t.get("underlying_price"),
                "open_interest": t.get("open_interest"),
                "volume": t.get("stats", {}).get("volume"),
                "delta": t.get("greeks", {}).get("delta"),
                "gamma": t.get("greeks", {}).get("gamma"),
                "theta": t.get("greeks", {}).get("theta"),
                "vega": t.get("greeks", {}).get("vega"),
            })
            time.sleep(0.05)  # ~20 req/s free tier
        except Exception as exc:
            print(f"    skip {name}: {exc}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"], unit="ms", utc=True)
    df["snapshot_time"] = pd.Timestamp.utcnow().floor("min")
    return df


def get_historical_volatility(currency: str) -> pd.DataFrame:
    """Fetch historical 30d realized vol from Deribit."""
    resp = fetch(
        f"{DERIBIT}/get_historical_volatility",
        params={"currency": currency},
    )
    data = resp.json().get("result", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def main() -> None:
    for currency in ["BTC", "ETH"]:
        print(f"Fetching Deribit {currency} options IV surface ...")
        df_surf = get_iv_surface(currency)
        if not df_surf.empty:
            df_surf = df_surf.set_index("instrument")
            save(df_surf, "options", f"Deribit_{currency}_options_snapshot.parquet")

        print(f"Fetching Deribit {currency} historical volatility ...")
        df_hvol = get_historical_volatility(currency)
        if not df_hvol.empty:
            save(df_hvol, "options", f"Deribit_{currency}_realized_vol_1d.parquet")

        time.sleep(1)


if __name__ == "__main__":
    main()
