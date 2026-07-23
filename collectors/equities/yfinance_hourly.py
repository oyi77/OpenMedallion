"""
Major US equities hourly OHLCV via yfinance.
Source: Yahoo Finance (no API key required)
Output: data/equities/yahoo/<LABEL>_1h.parquet
Covers: ~150 US stocks (S&P 500 and Nasdaq top names)
Note: yfinance hourly limited to ~730 days of history.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, HISTORY_START

_START = HISTORY_START or "2023-01-01"  # hourly limited ~730d
_SLEEP = 0.35

# (yfinance_ticker, human_label)
EQUITIES: list[tuple[str, str]] = [
    # ── Mega Cap ──
    ("AAPL",  "Apple"),
    ("MSFT",  "Microsoft"),
    ("GOOGL", "Alphabet"),
    ("AMZN",  "Amazon"),
    ("NVDA",  "Nvidia"),
    ("META",  "Meta"),
    ("TSLA",  "Tesla"),
    ("BRK-B", "BerkshireHathaway"),
    # ── Financials ──
    ("JPM",   "JPMorgan"),
    ("BAC",   "BankOfAmerica"),
    ("WFC",   "WellsFargo"),
    ("C",     "Citigroup"),
    ("GS",    "GoldmanSachs"),
    ("MS",    "MorganStanley"),
    ("BLK",   "BlackRock"),
    ("SCHW",  "CharlesSchwab"),
    ("AXP",   "AmericanExpress"),
    ("PYPL",  "PayPal"),
    ("BX",    "Blackstone"),
    ("KKR",   "KKR"),
    ("APO",   "ApolloGlobal"),
    ("SPGI",  "SPGlobal"),
    ("MCO",   "Moodys"),
    ("ICE",   "Intercontinental"),
    ("CME",   "CMEGroup"),
    ("CB",    "Chubb"),
    ("PGR",   "Progressive"),
    ("MET",   "MetLife"),
    ("AFL",   "Aflac"),
    # ── Tech ──
    ("AVGO",  "Broadcom"),
    ("ORCL",  "Oracle"),
    ("AMD",   "AMD"),
    ("NFLX",  "Netflix"),
    ("CRM",   "Salesforce"),
    ("ADBE",  "Adobe"),
    ("INTC",  "Intel"),
    ("TXN",   "TexasInstruments"),
    ("QCOM",  "Qualcomm"),
    ("IBM",   "IBM"),
    ("AMAT",  "AppliedMaterials"),
    ("MU",    "Micron"),
    ("NOW",   "ServiceNow"),
    ("INTU",  "Intuit"),
    ("LRCX",  "LamResearch"),
    ("KLAC",  "KLA"),
    ("MRVL",  "Marvell"),
    ("PANW",  "PaloAlto"),
    ("CRWD",  "CrowdStrike"),
    ("SNOW",  "Snowflake"),
    ("PLTR",  "Palantir"),
    ("DDOG",  "Datadog"),
    ("UBER",  "Uber"),
    ("SQ",    "Block"),
    ("SHOP",  "Shopify"),
    ("COIN",  "Coinbase"),
    ("MSTR",  "MicroStrategy"),
    ("SNAP",  "Snap"),
    ("DASH",  "DoorDash"),
    ("HOOD",  "Robinhood"),
    ("SOFI",  "SoFi"),
    ("AFRM",  "Affirm"),
    ("ZS",    "Zscaler"),
    ("NET",   "Cloudflare"),
    ("OKTA",  "Okta"),
    ("MDB",   "MongoDB"),
    ("DOCU",  "DocuSign"),
    ("TWLO",  "Twilio"),
    ("ZM",    "Zoom"),
    ("RBLX",  "Roblox"),
    ("ROKU",  "Roku"),
    ("PINS",  "Pinterest"),
    ("ETSY",  "Etsy"),
    ("CSCO",  "Cisco"),
    ("HPQ",   "HP"),
    ("DELL",  "Dell"),
    ("HPE",   "HPE"),
    ("WBD",   "WarnerBros"),
    ("ET",    "EnergyTransfer"),
    # ── Consumer / Retail ──
    ("WMT",   "Walmart"),
    ("COST",  "Costco"),
    ("PG",    "ProcterGamble"),
    ("HD",    "HomeDepot"),
    ("LOW",   "Lowes"),
    ("TGT",   "Target"),
    ("MCD",   "McDonalds"),
    ("SBUX",  "Starbucks"),
    ("NKE",   "Nike"),
    ("ABNB",  "Airbnb"),
    ("BKNG",  "Booking"),
    ("AMZN",  "Amazon"),
    ("DIS",   "Disney"),
    ("CMCSA", "Comcast"),
    ("VZ",    "Verizon"),
    ("T",     "ATT"),
    ("TMUS",  "TMobile"),
    ("CHWY",  "Chewy"),
    ("CROX",  "Crocs"),
    ("GME",   "GameStop"),
    ("AMC",   "AMC"),
    # ── Healthcare ──
    ("UNH",   "UnitedHealth"),
    ("JNJ",   "JohnsonJohnson"),
    ("LLY",   "EliLilly"),
    ("ABBV",  "AbbVie"),
    ("PFE",   "Pfizer"),
    ("ABT",   "Abbott"),
    ("MDT",   "Medtronic"),
    ("SYK",   "Stryker"),
    ("ISRG",  "IntuitiveSurgical"),
    ("TMO",   "ThermoFisher"),
    ("DHR",   "Danaher"),
    ("AMGN",  "Amgen"),
    ("GILD",  "Gilead"),
    ("REGN",  "Regeneron"),
    ("VRTX",  "Vertex"),
    ("BMY",   "BristolMyers"),
    ("MRNA",  "Moderna"),
    ("SIRI",  "SiriusXM"),
    # ── Energy ──
    ("XOM",   "ExxonMobil"),
    ("CVX",   "Chevron"),
    ("COP",   "ConocoPhillips"),
    ("SLB",   "Schlumberger"),
    ("EOG",   "EOGResources"),
    ("PSX",   "Phillips66"),
    ("VLO",   "Valero"),
    ("MPC",   "MarathonPetroleum"),
    # ── Industrials ──
    ("CAT",   "Caterpillar"),
    ("BA",    "Boeing"),
    ("GE",    "GeneralElectric"),
    ("HON",   "Honeywell"),
    ("UPS",   "UPS"),
    ("FDX",   "FedEx"),
    ("RTX",   "Raytheon"),
    ("LMT",   "LockheedMartin"),
    ("NOC",   "NorthropGrumman"),
    ("GD",    "GeneralDynamics"),
    ("DE",    "Deere"),
    ("MMM",   "3M"),
    ("WM",    "WasteManagement"),
    ("RSG",   "RepublicServices"),
    ("ECL",   "Ecolab"),
    ("SHW",   "SherwinWilliams"),
    # ── Materials / Mining ──
    ("FCX",   "FreeportMcMoRan"),
    ("NEM",   "Newmont"),
    ("RIO",   "RioTinto"),
    ("BHP",   "BHP"),
    # ── Utilities ──
    ("NEE",   "NextEra"),
    ("DUK",   "Duke"),
    ("SO",    "SouthernCo"),
    ("AEP",   "AmericanElectric"),
    # ── REITs ──
    ("PLD",   "Prologis"),
    ("AMT",   "AmericanTower"),
    ("CCI",   "CrownCastle"),
    ("EQIX",  "Equinix"),
    # ── Payments / Services ──
    ("V",     "Visa"),
    ("MA",    "Mastercard"),
    # ── Major ETFs ──
    ("SPY",   "SPY"),
    ("QQQ",   "QQQ"),
    ("IWM",   "IWM"),
    ("DIA",   "DIA"),
    ("VTI",   "VTI"),
    ("EFA",   "EFA"),
    ("EEM",   "EEM"),
    ("TLT",   "TLT"),
    ("IEF",   "IEF"),
    ("GLD",   "GLD"),
    ("SLV",   "SLV"),
    ("USO",   "USO"),
    ("UNG",   "UNG"),
    ("XLF",   "XLF"),
    ("XLK",   "XLK"),
    ("XLE",   "XLE"),
    ("XLV",   "XLV"),
    ("XLI",   "XLI"),
]


def _fetch_hourly(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, interval="1h", period="730d", progress=False, auto_adjust=True)
        if df is None or df.empty:
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


def collect_equity_hourly() -> None:
    for ticker, label in EQUITIES:
        print(f"  {label} ({ticker}) 1h ...")
        df = _fetch_hourly(ticker, label)
        if df is not None:
            save(df, "equities/yahoo", f"{label}_1h.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    print(f"Fetching: {len(EQUITIES)} US equities hourly OHLCV via yfinance")
    collect_equity_hourly()


if __name__ == "__main__":
    main()
