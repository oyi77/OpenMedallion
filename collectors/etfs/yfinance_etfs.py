"""
Major global ETFs via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/etfs/yfinance_etfs_1d.parquet  (OHLCV + adj_close, daily)
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
    # --- Fixed Income (extended) ---
    ("VCIT",  "VCIT_IG_Corp"),
    ("VCLT",  "VCLT_LT_Corp"),
    ("BSV",   "BSV_ST_Bond"),
    ("BIV",   "BIV_IT_Bond"),
    ("BLV",   "BLV_LT_Bond"),
    ("GOVT",  "GOVT_Treasury"),
    ("VGSH",  "VGSH_ST_Treasury"),
    ("VGIT",  "VGIT_IT_Treasury"),
    ("IAGG",  "IAGG_Intl_Bond"),
    ("PCY",   "PCY_EM_Sovereign"),
    ("ANGL",  "ANGL_Fallen_Angel"),
    ("HYEM",  "HYEM_EM_HY"),
    ("FLOT",  "FLOT_Float_Rate"),
    ("FTSL",  "FTSL_Senior_Loan"),
    # --- Country ETFs (extended) ---
    ("EWK",   "EWK_Belgium"),
    ("EWL",   "EWL_Switzerland"),
    ("EWO",   "EWO_Austria"),
    ("NORW",  "NORW_Norway"),
    ("EDEN",  "EDEN_Denmark"),
    ("EFNL",  "EFNL_Finland"),
    ("EWH",   "EWH_HongKong"),
    ("EIS",   "EIS_Israel"),
    ("TUR",   "TUR_Turkey"),
    ("EWM",   "EWM_Malaysia"),
    ("EPOL",  "EPOL_Poland"),
    ("GXG",   "GXG_Colombia"),
    ("EWX",   "EWX_SmallCap_EM"),
    # --- Sector ETFs (extended) ---
    ("SOXX",  "SOXX_Semis"),
    ("SMH",   "SMH_Semis"),
    ("IBB",   "IBB_Biotech"),
    ("XBI",   "XBI_Biotech"),
    ("KBE",   "KBE_Banks"),
    ("KRE",   "KRE_RegBanks"),
    ("IAT",   "IAT_RegBanks"),
    ("KBWB",  "KBWB_Banks"),
    ("XRT",   "XRT_Retail"),
    ("ITB",   "ITB_Homebuilders"),
    ("XHB",   "XHB_Homebuilders"),
    ("XME",   "XME_Metals_Mining"),
    ("MOO",   "MOO_Agriculture"),
    ("JETS",  "JETS_Airlines"),
    ("ITA",   "ITA_Aerospace"),
    ("PBJ",   "PBJ_Food_Bev"),
    ("XPH",   "XPH_Pharma"),
    ("IHI",   "IHI_MedDevices"),
    ("KIE",   "KIE_Insurance"),
    # --- Commodities (extended) ---
    ("PPLT",  "PPLT_Platinum"),
    ("PALL",  "PALL_Palladium"),
    ("SGOL",  "SGOL_Gold"),
    ("SIVR",  "SIVR_Silver"),
    ("REMX",  "REMX_RareEarth"),
    ("COPX",  "COPX_CopperMiners"),
    ("GDX",   "GDX_GoldMiners"),
    ("GDXJ",  "GDXJ_JrGoldMiners"),
    ("SIL",   "SIL_SilverMiners"),
    ("FCG",   "FCG_NatGas"),
    ("DRIP",  "DRIP_OilBear"),
    ("GUSH",  "GUSH_OilBull"),
    # --- Factor / Dividend ETFs ---
    ("MTUM",  "MTUM_Momentum"),
    ("VLUE",  "VLUE_Value"),
    ("SIZE",  "SIZE_SmallSize"),
    ("QUAL",  "QUAL_Quality"),
    ("USMV",  "USMV_MinVol"),
    ("HDV",   "HDV_Dividend"),
    ("DVY",   "DVY_Dividend"),
    ("VIG",   "VIG_DivGrowth"),
    ("NOBL",  "NOBL_DivAristocrat"),
    ("SDY",   "SDY_DivAristocrat"),
    ("SCHD",  "SCHD_Dividend"),
    ("SPHD",  "SPHD_HiDiv"),
    ("COWZ",  "COWZ_CashFlow"),
    ("QQQM",  "QQQM_Nasdaq"),
    ("QQEW",  "QQEW_EqWt"),
    ("RSP",   "RSP_EqWt_SP500"),
    ("IWB",   "IWB_LargeCap"),
    ("IWD",   "IWD_LargeValue"),
    ("IWF",   "IWF_LargeGrowth"),
    ("IWS",   "IWS_MidValue"),
    ("IWP",   "IWP_MidGrowth"),
    ("IWN",   "IWN_SmallValue"),
    ("IWO",   "IWO_SmallGrowth"),
    # --- Leverage / Inverse (extended) ---
    ("SSO",   "SSO_2x_SP500"),
    ("SDS",   "SDS_2x_Short_SP500"),
    ("QLD",   "QLD_2x_QQQ"),
    ("QID",   "QID_2x_Short_QQQ"),
    ("TNA",   "TNA_3x_SmallCap"),
    ("TZA",   "TZA_3x_Short_SC"),
    ("LABU",  "LABU_3x_Biotech"),
    ("LABD",  "LABD_3x_Short_Bio"),
    ("DUST",  "DUST_3x_Short_Miners"),
    ("NUGT",  "NUGT_3x_Miners"),
    ("UVIX",  "UVIX_2x_VIX"),
]


def _fetch(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=_START, progress=False, auto_adjust=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["ticker"] = label
        # reorder to: open, high, low, close, volume, adj_close
        cols = ["open", "high", "low", "close", "volume", "adj_close"]
        existing = [c for c in cols if c in df.columns]
        df = df[existing + ["ticker"]]
        return df
    except Exception as exc:
        print(f"  WARNING {label}: {exc}")
        return None


def collect_etfs() -> None:
    frames: list[pd.DataFrame] = []
    for ticker, label in ETFS:
        df = _fetch(ticker, label)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(_SLEEP)
    if not frames:
        print("  No ETF data fetched.")
        return
    combined = pd.concat(frames)
    combined = combined.sort_index()
    save(combined, "etfs", "yfinance_etfs_1d.parquet")


def main() -> None:
    print(f"Fetching: {len(ETFS)} ETFs (daily OHLCV) via yfinance")
    collect_etfs()


if __name__ == "__main__":
    main()
