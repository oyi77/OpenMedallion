"""
Major global equities via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/equities/yf/<LABEL>_1d.parquet  (OHLCV, daily)
Covers:
  - US: S&P 500 blue chips, Nasdaq 100 top names
  - Indonesia: LQ45 + IDX30 stocks
  - Europe: blue chips (UK, DE, FR, NL, CH)
  - Asia: JP, HK, SG, AU, KR, TW top names
  - EM: BR, IN top names
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save

_START = "2005-01-01"
_SLEEP = 0.25

# (yfinance_ticker, label)
EQUITIES: list[tuple[str, str]] = [
    # ── US Mega Cap ──
    ("AAPL",  "AAPL_US"),
    ("MSFT",  "MSFT_US"),
    ("NVDA",  "NVDA_US"),
    ("GOOGL", "GOOGL_US"),
    ("AMZN",  "AMZN_US"),
    ("META",  "META_US"),
    ("BRK-B", "BRKB_US"),
    ("TSLA",  "TSLA_US"),
    ("JPM",   "JPM_US"),
    ("V",     "V_US"),
    ("UNH",   "UNH_US"),
    ("XOM",   "XOM_US"),
    ("JNJ",   "JNJ_US"),
    ("WMT",   "WMT_US"),
    ("MA",    "MA_US"),
    ("PG",    "PG_US"),
    ("HD",    "HD_US"),
    ("AVGO",  "AVGO_US"),
    ("LLY",   "LLY_US"),
    ("COST",  "COST_US"),
    ("AMD",   "AMD_US"),
    ("INTC",  "INTC_US"),
    ("NFLX",  "NFLX_US"),
    ("CRM",   "CRM_US"),
    ("ADBE",  "ADBE_US"),
    ("PYPL",  "PYPL_US"),
    ("BAC",   "BAC_US"),
    ("WFC",   "WFC_US"),
    ("GS",    "GS_US"),
    ("MS",    "MS_US"),
    ("C",     "C_US"),
    ("CVX",   "CVX_US"),
    ("COP",   "COP_US"),
    ("PFE",   "PFE_US"),
    ("MRNA",  "MRNA_US"),
    ("ABBV",  "ABBV_US"),
    ("T",     "T_US"),
    ("VZ",    "VZ_US"),
    ("DIS",   "DIS_US"),
    ("SBUX",  "SBUX_US"),
    ("MCD",   "MCD_US"),
    ("KO",    "KO_US"),
    ("PEP",   "PEP_US"),
    ("CAT",   "CAT_US"),
    ("BA",    "BA_US"),
    ("GE",    "GE_US"),
    ("MMM",   "MMM_US"),
    ("UPS",   "UPS_US"),
    ("FDX",   "FDX_US"),
    ("UBER",  "UBER_US"),
    ("LYFT",  "LYFT_US"),
    ("SQ",    "SQ_US"),
    ("SHOP",  "SHOP_US"),
    ("SNOW",  "SNOW_US"),
    ("PLTR",  "PLTR_US"),
    ("COIN",  "COIN_US"),
    # ── Indonesia (IDX) ──
    ("BBCA.JK", "BBCA_ID"),
    ("BBRI.JK", "BBRI_ID"),
    ("BMRI.JK", "BMRI_ID"),
    ("BBNI.JK", "BBNI_ID"),
    ("TLKM.JK", "TLKM_ID"),
    ("ASII.JK", "ASII_ID"),
    ("UNVR.JK", "UNVR_ID"),
    ("GGRM.JK", "GGRM_ID"),
    ("HMSP.JK", "HMSP_ID"),
    ("ICBP.JK", "ICBP_ID"),
    ("INDF.JK", "INDF_ID"),
    ("KLBF.JK", "KLBF_ID"),
    ("UNTR.JK", "UNTR_ID"),
    ("PGAS.JK", "PGAS_ID"),
    ("JSMR.JK", "JSMR_ID"),
    ("ADRO.JK", "ADRO_ID"),
    ("PTBA.JK", "PTBA_ID"),
    ("MDKA.JK", "MDKA_ID"),
    ("ANTM.JK", "ANTM_ID"),
    ("INCO.JK", "INCO_ID"),
    ("GOTO.JK", "GOTO_ID"),
    ("BUKA.JK", "BUKA_ID"),
    ("EXCL.JK", "EXCL_ID"),
    ("ISAT.JK", "ISAT_ID"),
    # ── Japan ──
    ("7203.T", "Toyota_JP"),
    ("6758.T", "Sony_JP"),
    ("9984.T", "Softbank_JP"),
    ("8306.T", "MUFG_JP"),
    ("6861.T", "Keyence_JP"),
    ("6501.T", "Hitachi_JP"),
    ("9432.T", "NTT_JP"),
    ("4519.T", "Chugai_JP"),
    # ── Hong Kong ──
    ("0700.HK", "Tencent_HK"),
    ("9988.HK", "Alibaba_HK"),
    ("1299.HK", "AIA_HK"),
    ("0005.HK", "HSBC_HK"),
    ("0941.HK", "ChinaMobile_HK"),
    ("2318.HK", "PingAn_HK"),
    # ── Singapore ──
    ("D05.SI",  "DBS_SG"),
    ("O39.SI",  "OCBC_SG"),
    ("U11.SI",  "UOB_SG"),
    ("Z74.SI",  "SingTel_SG"),
    ("C6L.SI",  "SIA_SG"),
    # ── Australia ──
    ("BHP.AX",  "BHP_AU"),
    ("CBA.AX",  "CBA_AU"),
    ("CSL.AX",  "CSL_AU"),
    ("NAB.AX",  "NAB_AU"),
    ("WBC.AX",  "WBC_AU"),
    # ── Korea ──
    ("005930.KS", "Samsung_KR"),
    ("000660.KS", "SKHynix_KR"),
    ("005380.KS", "Hyundai_KR"),
    # ── Taiwan ──
    ("2330.TW", "TSMC_TW"),
    ("2317.TW", "Foxconn_TW"),
    # ── Europe ──
    ("ASML.AS", "ASML_NL"),
    ("SAP.DE",  "SAP_DE"),
    ("SIE.DE",  "Siemens_DE"),
    ("ALV.DE",  "Allianz_DE"),
    ("DTE.DE",  "Deutsche_DE"),
    ("ADS.DE",  "Adidas_DE"),
    ("VOW3.DE", "VW_DE"),
    ("HSBA.L",  "HSBC_UK"),
    ("BP.L",    "BP_UK"),
    ("GSK.L",   "GSK_UK"),
    ("ULVR.L",  "Unilever_UK"),
    ("RIO.L",   "RIO_UK"),
    ("SHEL.L",  "Shell_UK"),
    ("AZN.L",   "AZN_UK"),
    ("OR.PA",   "LOreal_FR"),
    ("MC.PA",   "LVMH_FR"),
    ("SAN.PA",  "Sanofi_FR"),
    ("NOVN.SW", "Novartis_CH"),
    ("NESN.SW", "Nestle_CH"),
    ("ROG.SW",  "Roche_CH"),
    ("UBSG.SW", "UBS_CH"),
    # ── Brazil ──
    ("VALE3.SA", "Vale_BR"),
    ("PETR4.SA", "Petrobras_BR"),
    ("ITUB4.SA", "Itau_BR"),
    # ── India ──
    ("RELIANCE.NS", "Reliance_IN"),
    ("TCS.NS",      "TCS_IN"),
    ("INFY.NS",     "Infosys_IN"),
    ("HDFCBANK.NS", "HDFC_IN"),
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
        print(f"  WARNING {label} ({ticker}): {exc}")
        return None


def collect_equities() -> None:
    for ticker, label in EQUITIES:
        df = _fetch(ticker, label)
        if df is not None and not df.empty:
            save(df, "equities/yf", f"{label}_1d.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    print(f"Fetching: {len(EQUITIES)} global equities (daily OHLCV) via yfinance")
    collect_equities()


if __name__ == "__main__":
    main()
