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
    # ── More US (S&P 500 top 100 coverage) ──
    ("ORCL",  "ORCL_US"),
    ("ACN",   "ACN_US"),
    ("QCOM",  "QCOM_US"),
    ("TXN",   "TXN_US"),
    ("IBM",   "IBM_US"),
    ("NOW",   "NOW_US"),
    ("INTU",  "INTU_US"),
    ("AMAT",  "AMAT_US"),
    ("MU",    "MU_US"),
    ("LRCX",  "LRCX_US"),
    ("KLAC",  "KLAC_US"),
    ("MRVL",  "MRVL_US"),
    ("PANW",  "PANW_US"),
    ("CRWD",  "CRWD_US"),
    ("NET",   "NET_US"),
    ("ZS",    "ZS_US"),
    ("DDOG",  "DDOG_US"),
    ("ABNB",  "ABNB_US"),
    ("BKNG",  "BKNG_US"),
    ("MAR",   "MAR_US"),
    ("HLT",   "HLT_US"),
    ("LOW",   "LOW_US"),
    ("NKE",   "NKE_US"),
    ("TGT",   "TGT_US"),
    ("AMGN",  "AMGN_US"),
    ("GILD",  "GILD_US"),
    ("REGN",  "REGN_US"),
    ("VRTX",  "VRTX_US"),
    ("BMY",   "BMY_US"),
    ("MDT",   "MDT_US"),
    ("TMO",   "TMO_US"),
    ("DHR",   "DHR_US"),
    ("ABT",   "ABT_US"),
    ("SYK",   "SYK_US"),
    ("ISRG",  "ISRG_US"),
    ("NEE",   "NEE_US"),
    ("SO",    "SO_US"),
    ("DUK",   "DUK_US"),
    ("D",     "D_US"),
    ("AEP",   "AEP_US"),
    ("PLD",   "PLD_US"),
    ("AMT",   "AMT_US"),
    ("CCI",   "CCI_US"),
    ("EQIX",  "EQIX_US"),
    ("WM",    "WM_US"),
    ("RSG",   "RSG_US"),
    ("ECL",   "ECL_US"),
    ("SHW",   "SHW_US"),
    ("FCX",   "FCX_US"),
    ("NEM",   "NEM_US"),
    ("BX",    "BX_US"),
    ("KKR",   "KKR_US"),
    ("APO",   "APO_US"),
    ("SPGI",  "SPGI_US"),
    ("MCO",   "MCO_US"),
    ("ICE",   "ICE_US"),
    ("CME",   "CME_US"),
    ("CB",    "CB_US"),
    ("PGR",   "PGR_US"),
    ("MET",   "MET_US"),
    ("AFL",   "AFL_US"),
    # ── More Indonesia (IDX composite) ──
    ("SMGR.JK", "SMGR_ID"),
    ("TOWR.JK", "TOWR_ID"),
    ("BSDE.JK", "BSDE_ID"),
    ("CPIN.JK", "CPIN_ID"),
    ("JPFA.JK", "JPFA_ID"),
    ("MYOR.JK", "MYOR_ID"),
    ("HRUM.JK", "HRUM_ID"),
    ("ITMG.JK", "ITMG_ID"),
    ("INTP.JK", "INTP_ID"),
    ("WIKA.JK", "WIKA_ID"),
    ("WSKT.JK", "WSKT_ID"),
    ("SCMA.JK", "SCMA_ID"),
    ("MNCN.JK", "MNCN_ID"),
    ("EMTK.JK", "EMTK_ID"),
    ("BTPS.JK", "BTPS_ID"),
    ("BJTM.JK", "BJTM_ID"),
    ("BJBR.JK", "BJBR_ID"),
    ("PWON.JK", "PWON_ID"),
    ("CTRA.JK", "CTRA_ID"),
    ("LPKR.JK", "LPKR_ID"),
    # ── More Japan ──
    ("6098.T", "RecruitHD_JP"),
    ("4543.T", "Terumo_JP"),
    ("2802.T", "Ajinomoto_JP"),
    ("4063.T", "ShinEtsu_JP"),
    ("4901.T", "Fujifilm_JP"),
    ("8035.T", "TokyoElec_JP"),
    ("6902.T", "Denso_JP"),
    ("7267.T", "Honda_JP"),
    ("7751.T", "Canon_JP"),
    ("4568.T", "DaiichiSankyo_JP"),
    # ── More Hong Kong / China ──
    ("3690.HK", "Meituan_HK"),
    ("1211.HK", "BYD_HK"),
    ("2269.HK", "WuXiBio_HK"),
    ("9999.HK", "NetEase_HK"),
    ("9618.HK", "JD_HK"),
    ("1024.HK", "Kuaishou_HK"),
    ("2382.HK", "SunshineInsur_HK"),
    # ── More Korea ──
    ("035420.KS", "Naver_KR"),
    ("051910.KS", "LGChem_KR"),
    ("006400.KS", "SamsungSDI_KR"),
    ("003550.KS", "LGElec_KR"),
    ("090430.KS", "Amorepacific_KR"),
    ("207940.KS", "SamsungBio_KR"),
    # ── More Taiwan ──
    ("2412.TW", "Chunghwa_TW"),
    ("2308.TW", "Delta_TW"),
    ("2454.TW", "MediaTek_TW"),
    ("2382.TW", "Quanta_TW"),
    ("2357.TW", "ASUS_TW"),
    # ── More India ──
    ("WIPRO.NS",      "Wipro_IN"),
    ("TATAMOTORS.NS", "Tata_IN"),
    ("BAJFINANCE.NS", "BajajFin_IN"),
    ("HINDUNILVR.NS", "HUL_IN"),
    ("ITC.NS",        "ITC_IN"),
    ("ASIANPAINT.NS", "AsianPaint_IN"),
    ("MARUTI.NS",     "Maruti_IN"),
    ("SUNPHARMA.NS",  "SunPharma_IN"),
    ("ULTRACEMCO.NS", "UltraCem_IN"),
    ("HCLTECH.NS",    "HCLTech_IN"),
    # ── More Europe ──
    ("AIR.PA",   "Airbus_FR"),
    ("BNP.PA",   "BNP_FR"),
    ("CS.PA",    "AXA_FR"),
    ("SU.PA",    "Schneider_FR"),
    ("DG.PA",    "Vinci_FR"),
    ("BAS.DE",   "BASF_DE"),
    ("MBG.DE",   "Mercedes_DE"),
    ("BMW.DE",   "BMW_DE"),
    ("MUV2.DE",  "MunichRe_DE"),
    ("DBK.DE",   "DeutscheBank_DE"),
    ("IFX.DE",   "Infineon_DE"),
    ("RWE.DE",   "RWE_DE"),
    ("ENI.MI",   "ENI_IT"),
    ("ENEL.MI",  "Enel_IT"),
    ("ISP.MI",   "IntesaSanPaolo_IT"),
    ("LIN.DE",   "Linde_DE"),
    ("BAB.L",    "Babcock_UK"),
    ("RDSA.AS",  "Shell_NL"),
    ("PHIA.AS",  "Philips_NL"),
    ("UNA.AS",   "Unilever_NL"),
    ("ABN.AS",   "ABN_NL"),
    ("INGA.AS",  "ING_NL"),
    ("ABBN.SW",  "ABB_CH"),
    ("SREN.SW",  "SwissRe_CH"),
    ("ZURN.SW",  "Zurich_CH"),
    # ── LatAm additions ──
    ("WEGE3.SA", "WEG_BR"),
    ("RENT3.SA", "Localiza_BR"),
    ("ABEV3.SA", "Ambev_BR"),
    ("B3SA3.SA", "B3_BR"),
    ("EQTL3.SA", "Equatorial_BR"),
    # ── More Australia ──
    ("ANZ.AX",  "ANZ_AU"),
    ("WES.AX",  "Wesfarmers_AU"),
    ("MQG.AX",  "Macquarie_AU"),
    ("RIO.AX",  "RIO_AU"),
    ("WOW.AX",  "Woolworths_AU"),
    # ── More Singapore ──
    ("J36.SI",  "Jardine_SG"),
    ("C38U.SI", "CapitaLandIT_SG"),
    ("A17U.SI", "CapitaMall_SG"),
    # ── More US Financials / Industrials ──
    ("AXP",  "AXP_US"),
    ("BLK",  "BLK_US"),
    ("SCHW", "SCHW_US"),
    ("USB",  "USB_US"),
    ("PNC",  "PNC_US"),
    ("MMC",  "MMC_US"),
    ("RTX",  "RTX_US"),
    ("LMT",  "LMT_US"),
    ("NOC",  "NOC_US"),
    ("GD",   "GD_US"),
    ("HON",  "HON_US"),
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
