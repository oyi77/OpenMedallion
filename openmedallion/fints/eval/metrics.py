"""
Backtest evaluation metrics for time-series forecasting models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoid division by zero)"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy of predicting direction (sign) of returns.
    Critical metric for trading systems.
    """
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    return np.mean(true_direction == pred_direction)


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio of strategy returns.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        
    Returns:
        Annualized Sharpe ratio (assumes daily returns)
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Maximum drawdown from peak.
    
    Args:
        cumulative_returns: Cumulative return series (1 + return compounded)
        
    Returns:
        Maximum drawdown as negative percentage
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return np.min(drawdown) * 100  # As percentage


def calmar_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calmar ratio: annualized return / max drawdown.
    Higher is better.
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    mdd = abs(max_drawdown(cumulative))
    
    if mdd == 0:
        return 0.0
    
    annualized_return = (cumulative[-1] ** (252 / len(returns)) - 1) * 100
    return annualized_return / mdd


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of predictions where actual return has same sign as predicted.
    Same as direction_accuracy but returns percentage.
    """
    return direction_accuracy(y_true, y_pred) * 100


def profit_factor(returns: np.ndarray) -> float:
    """
    Profit factor: sum of gains / sum of losses.
    Values > 1 indicate profitability.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    
    return gains / losses


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Sortino ratio: like Sharpe but only penalizes downside volatility.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns)
    return np.mean(excess_returns) / downside_std * np.sqrt(252)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all forecast evaluation metrics.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        prices: Optional price series for calculating strategy returns
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'direction_accuracy': direction_accuracy(y_true, y_pred),
        'hit_rate': hit_rate(y_true, y_pred)
    }
    
    # MAPE only if no zeros in y_true
    if not np.any(y_true == 0):
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    
    # Strategy metrics (assume trade on predicted direction)
    strategy_returns = y_true * np.sign(y_pred)  # Go long/short based on prediction
    
    metrics['sharpe_ratio'] = sharpe_ratio(strategy_returns)
    metrics['sortino_ratio'] = sortino_ratio(strategy_returns)
    metrics['profit_factor'] = profit_factor(strategy_returns)
    
    cumulative = (1 + strategy_returns).cumprod()
    metrics['max_drawdown'] = max_drawdown(cumulative)
    metrics['calmar_ratio'] = calmar_ratio(strategy_returns)
    
    # Total return
    metrics['total_return'] = (cumulative[-1] - 1) * 100  # As percentage
    
    return metrics


def print_metrics_report(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty-print metrics report.
    
    Args:
        metrics: Dictionary from calculate_all_metrics()
        title: Report title
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")
    
    print("Forecast Accuracy:")
    print(f"  MAE:                  {metrics.get('mae', 0):.6f}")
    print(f"  RMSE:                 {metrics.get('rmse', 0):.6f}")
    if 'mape' in metrics:
        print(f"  MAPE:                 {metrics['mape']:.2f}%")
    print(f"  Direction Accuracy:   {metrics.get('direction_accuracy', 0):.2f}")
    print(f"  Hit Rate:             {metrics.get('hit_rate', 0):.2f}%")
    
    print("\nStrategy Performance:")
    print(f"  Total Return:         {metrics.get('total_return', 0):.2f}%")
    print(f"  Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Sortino Ratio:        {metrics.get('sortino_ratio', 0):.4f}")
    print(f"  Profit Factor:        {metrics.get('profit_factor', 0):.4f}")
    print(f"  Max Drawdown:         {metrics.get('max_drawdown', 0):.2f}%")
    print(f"  Calmar Ratio:         {metrics.get('calmar_ratio', 0):.4f}")
    
    print(f"\n{'='*60}\n")
