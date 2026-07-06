"""
Indonesia IDX equity collector via yfinance.
Covers: LQ45, IDX30 (subset of LQ45), and additional IDX80 components.
All tickers use the .JK suffix for the Jakarta Stock Exchange.
Output: data/equities/indonesia/<TICKER>_1d.parquet
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

# LQ45 — 45 most liquid IDX stocks (IDX30 is a subset; all covered here)
LQ45 = [
    "ACES.JK", "ADRO.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK",
    "ASII.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK",
    "BMRI.JK", "BRIS.JK", "BRPT.JK", "BUKA.JK", "CPIN.JK",
    "EMTK.JK", "ERAA.JK", "ESSA.JK", "EXCL.JK", "GOTO.JK",
    "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INTP.JK",
    "ISAT.JK", "ITMG.JK", "JPFA.JK", "KLBF.JK", "MAPI.JK",
    "MBMA.JK", "MDKA.JK", "MEDC.JK", "MTEL.JK", "PGEO.JK",
    "PGAS.JK", "PTBA.JK", "PTPP.JK", "SMGR.JK", "SRTG.JK",
    "TINS.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
]

# Additional IDX80 tickers not already in LQ45
IDX80_EXTRA = [
    "ADMR.JK", "AGII.JK", "AKRA.JK", "ALDO.JK", "AMAG.JK",
    "ANJT.JK", "APEX.JK", "APIC.JK", "ARNA.JK", "ASGR.JK",
    "AUTO.JK", "BBYB.JK", "BCAP.JK", "BFIN.JK", "BGTG.JK",
    "BJBR.JK", "BJTM.JK", "BNGA.JK", "BNII.JK", "BPAM.JK",
    "BRMS.JK", "BSDE.JK", "BTPS.JK", "CASS.JK", "CEKA.JK",
    "CLEO.JK", "CMRY.JK", "CSAP.JK", "DEWA.JK", "DILD.JK",
    "DMAS.JK", "DSNG.JK", "EKAD.JK", "ELSA.JK",
]

# Deduplicated master list (LQ45 first, then IDX80 extras)
ALL_TICKERS: list[str] = list(dict.fromkeys(LQ45 + IDX80_EXTRA))


def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
    """Fetch max daily history for a single .JK ticker. Returns (ticker, df)."""
    try:
        raw = yf.download(
            ticker,
            period="max",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            return ticker, pd.DataFrame()

        # yfinance multi-level columns when single ticker — flatten if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
        df = raw[cols].copy()
        df.columns = df.columns.str.lower()
        df = to_datetime_index(df)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return ticker, df
    except Exception as exc:
        print(f"  WARNING: {ticker} — {exc}")
        return ticker, pd.DataFrame()


def main() -> None:
    print(f"=== Indonesia IDX ({len(ALL_TICKERS)} tickers) ===")
    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in ALL_TICKERS}
        for future in as_completed(futures):
            ticker, df = future.result()
            base_name = ticker.split(".")[0]
            if not df.empty:
                save(df, "equities/indonesia", f"{base_name}_1d.parquet")
                ok += 1
            else:
                print(f"  SKIP: {ticker} — no data")
                fail += 1
            time.sleep(0.1)  # gentle inter-result pacing

    print(f"\n✓ {ok}/{len(ALL_TICKERS)} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
