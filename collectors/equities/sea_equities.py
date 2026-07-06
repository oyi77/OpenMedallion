"""
SEA + Asia equity markets collector via yfinance.
Covers: Vietnam (.VN), Thailand (.BK), India (.NS), Korea (.KS), Taiwan (.TW)
Note: Malaysia (.KL) and Philippines (.PS) have poor yfinance coverage - skipped.
Output: data/equities/country_stocks/<TICKER>_1d.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

# Top 30 stocks by market cap for each country
# Vietnam: Vinamilk, Vietcombank, FPT, VietinBank, Masan, Hoa Phat, Vingroup, etc.
VIETNAM_VN = [
    "VNM.VN", "VCB.VN", "FPT.VN", "CTG.VN", "MSN.VN", "HPG.VN", "VIC.VN",
    "VHM.VN", "GAS.VN", "MBB.VN", "BID.VN", "TCB.VN", "SSI.VN", "VPB.VN",
    "SAB.VN", "PLX.VN", "POW.VN", "VRE.VN", "HDB.VN", "STB.VN",
    "GEX.VN", "ACB.VN", "VJC.VN", "NVL.VN", "TPB.VN", "MWG.VN",
    "VCI.VN", "PDR.VN", "VIB.VN", "REE.VN",
]

# Thailand: PTT, Kasikornbank, AOT, SCB, CP All, Advanced Info Service, etc.
THAILAND_BK = [
    "PTT.BK", "KBANK.BK", "AOT.BK", "SCB.BK", "CPALL.BK", "ADVANC.BK",
    "BBL.BK", "TRUE.BK", "GULF.BK", "TOP.BK", "PTTEP.BK", "PTTGC.BK",
    "SCC.BK", "DELTA.BK", "BEM.BK", "BCP.BK", "CPN.BK", "INTUCH.BK",
    "AWC.BK", "IVL.BK", "BH.BK", "COM7.BK", "EGCO.BK", "BANPU.BK",
    "OSP.BK", "MINT.BK", "RATCH.BK", "TMB.BK", "GPSC.BK", "AP.BK",
]

# India NSE: Reliance, TCS, Infosys, HDFC Bank, ITC, Bharti Airtel, etc.
INDIA_NS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "BHARTIARTL.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS", "M&M.NS", "POWERGRID.NS",
    "TATASTEEL.NS", "NTPC.NS", "ONGC.NS", "JSWSTEEL.NS", "ADANIENT.NS", "HINDALCO.NS",
    "COALINDIA.NS", "GRASIM.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "DIVISLAB.NS",
    "HEROMOTOCO.NS", "DRREDDY.NS", "EICHERMOT.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "TATAMOTORS.NS", "BRITANNIA.NS", "SHREECEM.NS", "BPCL.NS", "IOC.NS",
    "UPL.NS", "ADANIPORTS.NS", "VEDL.NS", "HINDZINC.NS", "TATACONSUM.NS",
]

# Korea: Samsung, SK Hynix, NAVER, Kakao, LG Chem, Hyundai Motor, etc.
KOREA_KS = [
    "005930.KS", "000660.KS", "035420.KS", "035720.KS", "051910.KS", "005380.KS",
    "006400.KS", "068270.KS", "005490.KS", "012330.KS", "028260.KS", "055550.KS",
    "034730.KS", "017670.KS", "036570.KS", "015760.KS", "009150.KS", "033780.KS",
    "003550.KS", "011200.KS", "032830.KS", "010130.KS", "066570.KS", "000270.KS",
    "018260.KS", "042700.KS", "326030.KS", "003670.KS", "096770.KS", "000810.KS",
]

# Taiwan: TSMC, Hon Hai, MediaTek, UMC, Delta Electronics, etc.
TAIWAN_TW = [
    "2330.TW", "2317.TW", "2454.TW", "2303.TW", "2308.TW", "1301.TW",
    "2382.TW", "2412.TW", "1303.TW", "2881.TW", "2886.TW", "2002.TW",
    "1326.TW", "6505.TW", "2891.TW", "2882.TW", "2357.TW", "2912.TW",
    "3711.TW", "5880.TW", "2892.TW", "2395.TW", "2884.TW", "2885.TW",
    "1216.TW", "2207.TW", "2409.TW", "3008.TW", "2301.TW", "2603.TW",
]

ALL_TICKERS = {
    "Vietnam": VIETNAM_VN,
    "Thailand": THAILAND_BK,
    "India": INDIA_NS,
    "Korea": KOREA_KS,
    "Taiwan": TAIWAN_TW,
}


def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    """Fetch max available daily history for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", interval="1d")
        if hist.empty:
            return pd.DataFrame()
        
        hist = hist[["Open", "High", "Low", "Close", "Volume"]]
        hist.columns = hist.columns.str.lower()
        hist.index = pd.to_datetime(hist.index, utc=True)
        hist = hist.sort_index()
        return hist.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    except Exception as exc:
        print(f"  WARNING: {ticker} — {exc}")
        return pd.DataFrame()


def main() -> None:
    for country, tickers in ALL_TICKERS.items():
        print(f"\n=== {country} ({len(tickers)} tickers) ===")
        ok = 0
        for ticker in tickers:
            df = fetch_ticker_history(ticker)
            if not df.empty:
                clean_ticker = ticker.split(".")[0]
                save(df, "equities/country_stocks", f"{clean_ticker}_{country[:2].upper()}_1d.parquet")
                ok += 1
            time.sleep(0.3)  # Rate limit safety
        print(f"  ✓ {ok}/{len(tickers)} succeeded")


if __name__ == "__main__":
    main()
