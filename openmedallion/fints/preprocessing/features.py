"""
Feature engineering for time-series forecasting.
Technical indicators + lag features.
"""
import pandas as pd
import numpy as np


def log_returns(df: pd.DataFrame, col: str = 'close') -> pd.Series:
    """Log returns: log(price_t / price_{t-1})"""
    return np.log(df[col] / df[col].shift(1))


def rolling_volatility(df: pd.DataFrame, window: int = 20, col: str = 'close') -> pd.Series:
    """Rolling standard deviation of log returns"""
    returns = log_returns(df, col)
    return returns.rolling(window).std()


def rsi(df: pd.DataFrame, window: int = 14, col: str = 'close') -> pd.Series:
    """Relative Strength Index"""
    delta = df[col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = 'close') -> pd.DataFrame:
    """MACD indicator: returns DataFrame with macd, signal, histogram"""
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    })


def volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume z-score relative to rolling window"""
    if 'volume' not in df.columns:
        return pd.Series(0, index=df.index)
    mean = df['volume'].rolling(window).mean()
    std = df['volume'].rolling(window).std()
    return (df['volume'] - mean) / (std + 1e-8)


def lag_returns(df: pd.DataFrame, lags: list[int], col: str = 'close') -> pd.DataFrame:
    """Lagged returns for multiple periods"""
    returns = log_returns(df, col)
    return pd.DataFrame({
        f'return_lag_{lag}': returns.shift(lag)
        for lag in lags
    })


def build_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Build complete feature set from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        lookback: Window size for rolling features (default: 20)
    
    Returns:
        DataFrame with all engineered features
    """
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['log_return'] = log_returns(df)
    features['volatility_20'] = rolling_volatility(df, window=lookback)
    features['rsi_14'] = rsi(df, window=min(14, lookback))
    
    # MACD
    macd_df = macd(df)
    features = pd.concat([features, macd_df], axis=1)
    
    # Volume
    features['volume_zscore'] = volume_zscore(df)
    
    # Lagged returns
    lag_df = lag_returns(df, lags=[1, 2, 3, 5, 10])
    features = pd.concat([features, lag_df], axis=1)
    
    # Price momentum
    features['momentum_5'] = df['close'].pct_change(min(5, lookback))
    features['momentum_20'] = df['close'].pct_change(lookback)
    
    return features


def make_target(df: pd.DataFrame, horizon: int = 1, col: str = 'close') -> pd.Series:
    """
    Create regression target: future log return.
    
    Args:
        df: OHLCV dataframe
        horizon: Prediction horizon in periods
        col: Column to predict
    
    Returns:
        Series of future log returns
    """
    return np.log(df[col].shift(-horizon) / df[col])


def make_direction_target(df: pd.DataFrame, horizon: int = 1, col: str = 'close') -> pd.Series:
    """
    Create classification target: future direction (up=1, down=0).
    
    Args:
        df: OHLCV dataframe
        horizon: Prediction horizon in periods
        col: Column to predict
    
    Returns:
        Series of binary direction labels
    """
    future_return = make_target(df, horizon, col)
    return (future_return > 0).astype(int)
