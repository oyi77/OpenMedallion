"""
Feature engineering for FinTS.

All transformations are stateless functions operating on a single-asset DataFrame.
No global state, no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core return transformation
# ---------------------------------------------------------------------------


def log_returns(prices: pd.Series) -> pd.Series:
    """Log return: ln(P_t / P_{t-1}). First row is NaN."""
    return np.log(prices / prices.shift(1))


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling std of log-returns (sqrt(252) scaling)."""
    return returns.rolling(window).std() * np.sqrt(252)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100]."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=prices.index,
    )


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of volume."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0, np.nan)


def lag_returns(returns: pd.Series, lags: list[int] = (1, 5, 20)) -> pd.DataFrame:
    """Lagged log-returns as features."""
    return pd.DataFrame(
        {f"ret_lag_{lag}": returns.shift(lag) for lag in lags},
        index=returns.index,
    )


# ---------------------------------------------------------------------------
# Full feature matrix
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    lookback_window: int = 20,
    lag_periods: list[int] = (1, 5, 20),
) -> pd.DataFrame:
    """
    Build feature matrix from an OHLCV DataFrame.

    Returns a DataFrame aligned to the original index, containing:
    - log_ret            : log return of close
    - roll_vol_{w}       : rolling volatility
    - rsi_14             : RSI(14)
    - macd, macd_signal, macd_hist
    - vol_zscore         : volume z-score (if volume available)
    - ret_lag_{1,5,20}   : lagged returns

    All rows with any NaN are dropped.
    """
    close = df["close"]
    ret = log_returns(close).rename("log_ret")
    vol = rolling_volatility(ret, lookback_window).rename(
        f"roll_vol_{lookback_window}"
    )
    rsi_feat = rsi(close).rename("rsi_14")
    macd_feats = macd(close)
    lag_feats = lag_returns(ret, list(lag_periods))

    parts = [ret, vol, rsi_feat, macd_feats, lag_feats]

    if "volume" in df.columns and df["volume"].notna().any():
        parts.append(volume_zscore(df["volume"]).rename("vol_zscore"))

    feat = pd.concat(parts, axis=1)
    return feat.dropna()


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def make_target(
    returns: pd.Series,
    horizon: int = 1,
) -> pd.Series:
    """
    Forward log-return over `horizon` steps.
    Shifted so feat[t] predicts target[t].
    """
    fwd = returns.rolling(horizon).sum().shift(-horizon)
    return fwd.rename(f"fwd_ret_{horizon}")


def make_direction_target(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary: 1 if forward return > 0, else 0."""
    fwd = make_target(returns, horizon)
    return (fwd > 0).astype(int).rename(f"direction_{horizon}")
