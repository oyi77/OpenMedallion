"""
Major global ETFs via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/etfs/<SYMBOL>_1d.parquet  (OHLCV, daily)
Covers: equity, bond, commodity, currency, volatility, sector, country ETFs
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

_START = "2000-01-01"
_SLEEP = 0.3

# (symbol, label) pairs — publicly-traded ETFs, no key required
ETFS: list[tuple[str, str]] = [
    # --- Broad US Equity ---
    ("SPY",   "SPY"),
    ("IVV",   "IVV"),
    ("VOO",   "VOO"),
    ("QQQ",   "QQQ"),
    ("IWM",   "IWM"),
    ("DIA",   "DIA"),
    ("VTI",   "VTI"),
    ("ITOT",  "ITOT"),
    # --- International Equity ---
    ("EFA",   "EFA"),
    ("VEA",   "VEA"),
    ("IEFA",  "IEFA"),
    ("EEM",   "EEM"),
    ("VWO",   "VWO"),
    ("IEMG",  "IEMG"),
    ("ACWI",  "ACWI"),
    ("VT",    "VT"),
    # --- Country ETFs ---
    ("EWJ",   "EWJ_Japan"),
    ("EWZ",   "EWZ_Brazil"),
    ("EWY",   "EWY_Korea"),
    ("EWT",   "EWT_Taiwan"),
    ("FXI",   "FXI_China"),
    ("MCHI",  "MCHI_China"),
    ("EWA",   "EWA_Australia"),
    ("EWC",   "EWC_Canada"),
    ("EWG",   "EWG_Germany"),
    ("EWU",   "EWU_UK"),
    ("EWI",   "EWI_Italy"),
    ("EWQ",   "EWQ_France"),
    ("EWS",   "EWS_Singapore"),
    ("EIDO",  "EIDO_Indonesia"),
    ("EPHE",  "EPHE_Philippines"),
    ("THD",   "THD_Thailand"),
    ("VNM",   "VNM_Vietnam"),
    ("INDA",  "INDA_India"),
    ("RSX",   "RSX_Russia"),  # may be delisted/halted
    ("EZA",   "EZA_SouthAfrica"),
    ("ECH",   "ECH_Chile"),
    ("EWW",   "EWW_Mexico"),
    ("EPU",   "EPU_Peru"),
    ("EWD",   "EWD_Sweden"),
    ("EWN",   "EWN_Netherlands"),
    ("EWP",   "EWP_Spain"),
    # --- US Sector ETFs ---
    ("XLK",   "XLK_Tech"),
    ("XLF",   "XLF_Financials"),
    ("XLE",   "XLE_Energy"),
    ("XLV",   "XLV_Health"),
    ("XLI",   "XLI_Industrial"),
    ("XLY",   "XLY_ConsDisc"),
    ("XLP",   "XLP_ConsSt"),
    ("XLU",   "XLU_Utilities"),
    ("XLRE",  "XLRE_RealEst"),
    ("XLB",   "XLB_Materials"),
    ("XLC",   "XLC_Comms"),
    # --- Fixed Income ---
    ("AGG",   "AGG"),
    ("BND",   "BND"),
    ("TLT",   "TLT"),
    ("IEF",   "IEF"),
    ("SHY",   "SHY"),
    ("LQD",   "LQD"),
    ("HYG",   "HYG"),
    ("JNK",   "JNK"),
    ("EMB",   "EMB"),
    ("VCSH",  "VCSH"),
    ("VGLT",  "VGLT"),
    ("TIP",   "TIP"),
    ("BNDX",  "BNDX"),
    # --- Commodities ---
    ("GLD",   "GLD_Gold"),
    ("SLV",   "SLV_Silver"),
    ("IAU",   "IAU_Gold"),
    ("USO",   "USO_Oil"),
    ("UNG",   "UNG_Gas"),
    ("DBC",   "DBC_Commodity"),
    ("PDBC",  "PDBC_Commodity"),
    ("GSG",   "GSG_Commodity"),
    ("CPER",  "CPER_Copper"),
    ("WEAT",  "WEAT_Wheat"),
    ("CORN",  "CORN_Corn"),
    ("SOYB",  "SOYB_Soy"),
    # --- Real Estate ---
    ("VNQ",   "VNQ_REIT"),
    ("IYR",   "IYR_REIT"),
    ("SCHH",  "SCHH_REIT"),
    # --- Volatility ---
    ("UVXY",  "UVXY"),
    ("SVXY",  "SVXY"),
    ("VXX",   "VXX"),
    # --- Currency ---
    ("UUP",   "UUP_USD"),
    ("FXE",   "FXE_EUR"),
    ("FXY",   "FXY_JPY"),
    ("FXB",   "FXB_GBP"),
    ("FXA",   "FXA_AUD"),
    ("FXC",   "FXC_CAD"),
    ("CYB",   "CYB_CNY"),
    # --- Crypto ---
    ("BITO",  "BITO_BTC"),
    ("GBTC",  "GBTC_BTC"),
    ("IBIT",  "IBIT_BTC"),
    ("ETHA",  "ETHA_ETH"),
    # --- Thematic ---
    ("ARKK",  "ARKK"),
    ("ARKG",  "ARKG"),
    ("ARKW",  "ARKW"),
    ("BOTZ",  "BOTZ_Robotics"),
    ("WCLD",  "WCLD_Cloud"),
    ("HACK",  "HACK_Cyber"),
    ("CLOU",  "CLOU_Cloud"),
    ("ICLN",  "ICLN_CleanEnergy"),
    ("TAN",   "TAN_Solar"),
    ("KARS",  "KARS_EV"),
    ("LIT",   "LIT_Lithium"),
    ("AIQ",   "AIQ_AI"),
    ("ROBO",  "ROBO_Robotics"),
    # --- Leverage / Inverse ---
    ("TQQQ",  "TQQQ"),
    ("SQQQ",  "SQQQ"),
    ("SPXL",  "SPXL"),
    ("SPXS",  "SPXS"),
    ("UPRO",  "UPRO"),
    ("SH",    "SH_Short_SP500"),
    ("PSQ",   "PSQ_Short_QQQ"),
]


def _fetch(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=_START, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as exc:
        print(f"  WARNING {label}: {exc}")
        return None


def collect_etfs() -> None:
    for ticker, label in ETFS:
        df = _fetch(ticker, label)
        if df is not None and not df.empty:
            save(df, "etfs", f"{label}_1d.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    print(f"Fetching: {len(ETFS)} ETFs (daily OHLCV) via yfinance")
    collect_etfs()


if __name__ == "__main__":
    main()
