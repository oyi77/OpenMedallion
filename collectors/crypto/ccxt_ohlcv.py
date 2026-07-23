"""
Multi-exchange crypto intraday OHLCV via CCXT.
Source: Binance (primary), Kraken (secondary), Bybit (derivatives).
Free, no API key required for public OHLCV data.
Output: data/crypto/<EXCHANGE>_<SYMBOL>_<INTERVAL>.parquet
Covers: top 100 pairs by volume, 1m/5m/15m/30m/1h/4h intervals.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, HISTORY_START

# ── Configuration ──────────────────────────────────────────────────
_DEFAULT_START = "2020-01-01"
_START = HISTORY_START or _DEFAULT_START
_SLEEP = 0.25  # be polite between symbols
_EXCHANGES = {
    "binance": {
        "rateLimit": 1200,
        "options": {"defaultType": "spot"},
    },
    "kraken": {
        "rateLimit": 3000,
    },
}

INTERVALS = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "4h":  14400,
}

# Top ~100 Binance USDT pairs (by approximate volume rank).
# Stored as {exchange: [(symbol, label), ...]}.
SYMBOLS: dict[str, list[tuple[str, str]]] = {
    "binance": [
        ("BTC/USDT",  "BTCUSDT"),
        ("ETH/USDT",  "ETHUSDT"),
        ("BNB/USDT",  "BNBUSDT"),
        ("SOL/USDT",  "SOLUSDT"),
        ("XRP/USDT",  "XRPUSDT"),
        ("DOGE/USDT", "DOGEUSDT"),
        ("ADA/USDT",  "ADAUSDT"),
        ("AVAX/USDT", "AVAXUSDT"),
        ("DOT/USDT",  "DOTUSDT"),
        ("LINK/USDT", "LINKUSDT"),
        ("MATIC/USDT","MATICUSDT"),
        ("UNI/USDT",  "UNIUSDT"),
        ("ATOM/USDT", "ATOMUSDT"),
        ("SHIB/USDT", "SHIBUSDT"),
        ("LTC/USDT",  "LTCUSDT"),
        ("TRX/USDT",  "TRXUSDT"),
        ("ETC/USDT",  "ETCUSDT"),
        ("FIL/USDT",  "FILUSDT"),
        ("XLM/USDT",  "XLMUSDT"),
        ("APT/USDT",  "APTUSDT"),
        ("ARB/USDT",  "ARBUSDT"),
        ("NEAR/USDT", "NEARUSDT"),
        ("OP/USDT",   "OPUSDT"),
        ("TIA/USDT",  "TIAUSDT"),
        ("SEI/USDT",  "SEIUSDT"),
        ("INJ/USDT",  "INJUSDT"),
        ("AAVE/USDT", "AAVEUSDT"),
        ("CRV/USDT",  "CRVUSDT"),
        ("ICP/USDT",  "ICPUSDT"),
        ("RUNE/USDT", "RUNEUSDT"),
        ("ALGO/USDT", "ALGOUSDT"),
        ("EGLD/USDT", "EGLDUSDT"),
        ("SAND/USDT", "SANDUSDT"),
        ("MANA/USDT", "MANAUSDT"),
        ("ENJ/USDT",  "ENJUSDT"),
        ("VET/USDT",  "VETUSDT"),
        ("THETA/USDT","THETAUSDT"),
        ("FTM/USDT",  "FTMUSDT"),
        ("AXS/USDT",  "AXSUSDT"),
        ("CHZ/USDT",  "CHZUSDT"),
        ("KAVA/USDT", "KAVAUSDT"),
        ("GALA/USDT", "GALAUSDT"),
        ("APE/USDT",  "APEUSDT"),
        ("XEC/USDT",  "XECUSDT"),
        ("DYDX/USDT", "DYDXUSDT"),
        ("1INCH/USDT","1INCHUSDT"),
        ("GRT/USDT",  "GRTUSDT"),
        ("STX/USDT",  "STXUSDT"),
        ("FET/USDT",  "FETUSDT"),
        ("AGIX/USDT", "AGIXUSDT"),
        ("OCEAN/USDT","OCEANUSDT"),
        ("ANKR/USDT", "ANKRUSDT"),
        ("IOST/USDT", "IOSTUSDT"),
        ("IOTX/USDT", "IOTXUSDT"),
        ("ZIL/USDT",  "ZILUSDT"),
        ("HOT/USDT",  "HOTUSDT"),
        ("TFUEL/USDT","TFUELUSDT"),
        ("ONE/USDT",  "ONEUSDT"),
        ("HBAR/USDT", "HBARUSDT"),
        ("OM/USDT",   "OMUSDT"),
        ("WOO/USDT",  "WOOUSDT"),
        ("GNO/USDT",  "GNOUSDT"),
        ("BAT/USDT",  "BATUSDT"),
        ("ZRX/USDT",  "ZRXUSDT"),
        ("LRC/USDT",  "LRCUSDT"),
        ("SKL/USDT",  "SKLUSDT"),
        ("CVC/USDT",  "CVCUSDT"),
        ("POLYX/USDT","POLYXUSDT"),
        ("CTSI/USDT", "CTSIUSDT"),
        ("RLC/USDT",  "RLCUSDT"),
        ("BAND/USDT", "BANDUSDT"),
        ("NMR/USDT",  "NMRUSDT"),
        ("TRB/USDT",  "TRBUSDT"),
        ("UMA/USDT",  "UMAUSDT"),
        ("LPT/USDT",  "LPTUSDT"),
        ("REN/USDT",  "RENUSDT"),
        ("SNX/USDT",  "SNXUSDT"),
        ("C98/USDT",  "C98USDT"),
        ("ALPHA/USDT","ALPHAUSDT"),
        ("SUSHI/USDT","SUSHIUSDT"),
        ("CAKE/USDT", "CAKEUSDT"),
        ("PENDLE/USDT","PENDLEUSDT"),
        ("COMP/USDT", "COMPUSDT"),
        ("MKR/USDT",  "MKRUSDT"),
        ("YFI/USDT",  "YFIUSDT"),
        ("CVX/USDT",  "CVXUSDT"),
        ("FXS/USDT",  "FXSUSDT"),
        ("WLD/USDT",  "WLDUSDT"),
        ("BLUR/USDT", "BLURUSDT"),
        ("PYTH/USDT", "PYTHUSDT"),
        ("JUP/USDT",  "JUPUSDT"),
        ("STRK/USDT", "STRKUSDT"),
        ("ENA/USDT",  "ENAUSDT"),
        ("WIF/USDT",  "WIFUSDT"),
        ("PEPE/USDT", "PEPEUSDT"),
        ("BONK/USDT", "BONKUSDT"),
        ("FLOKI/USDT","FLOKIUSDT"),
    ],
}


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    interval: str,
    since_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch all OHLCV candles for one symbol/interval via pagination."""
    import ccxt

    ex = getattr(ccxt, exchange_id)({
        "rateLimit": _EXCHANGES[exchange_id]["rateLimit"],
        "enableRateLimit": True,
    })
    timeframe_sec = INTERVALS[interval]
    all_candles: list = []

    while since_ms < int(time.time() * 1000):
        try:
            candles = ex.fetch_ohlcv(symbol, interval, since=since_ms, limit=limit)
        except Exception as exc:
            print(f"    WARNING {symbol} {interval}: {exc}")
            break
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + (timeframe_sec * 1000)
        # rate limit courtesy
        time.sleep(ex.rateLimit / 1000 * 1.5)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index.name = "date"
    return df


def collect_crypto_intraday() -> None:
    """Fetch intraday OHLCV for all exchanges, symbols, and intervals."""
    for exchange_id, symbols in SYMBOLS.items():
        ex_label = exchange_id.title()
        for ccxt_symbol, label in symbols:
            for interval in INTERVALS:
                filename = f"{ex_label}_{label}_{interval}.parquet"
                print(f"  {filename} ...")
                since = pd.Timestamp(_START, tz="UTC").value // 1_000_000
                df = fetch_ohlcv(exchange_id, ccxt_symbol, interval, since)
                save(df, "crypto", filename)
                time.sleep(_SLEEP)


def main() -> None:
    total = len(SYMBOLS["binance"]) * len(INTERVALS)
    print(f"Fetching {total} symbol×interval combos (crypto intraday) via CCXT")
    collect_crypto_intraday()


if __name__ == "__main__":
    main()
