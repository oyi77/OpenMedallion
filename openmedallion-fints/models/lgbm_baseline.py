"""
LightGBM baseline for time-series forecasting.
Simple gradient boosting baseline to benchmark against PatchTST.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Optional


class LGBMForecaster:
    """
    LightGBM baseline for next-period return prediction.
    
    Features: technical indicators (RSI, MACD, volatility, lags)
    Target: next-period log return or direction
    """
    
    def __init__(
        self,
        task: str = 'regression',
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        num_leaves: int = 31,
        random_state: int = 42
    ):
        """
        Args:
            task: 'regression' or 'classification'
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            num_leaves: Maximum leaves per tree
            random_state: Random seed
        """
        self.task = task
        self.random_state = random_state
        
        if task == 'regression':
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'num_leaves': num_leaves,
                'random_state': random_state,
                'verbose': -1
            }
            self.model = lgb.LGBMRegressor(**self.params)
            
        elif task == 'classification':
            self.params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'num_leaves': num_leaves,
                'random_state': random_state,
                'verbose': -1
            }
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            raise ValueError(f"task must be 'regression' or 'classification', got {task}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 10
    ):
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            early_stopping_rounds: Stop if val metric doesn't improve
        """
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict next-period return or direction.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (regression: returns, classification: probabilities)
        """
        if self.task == 'classification':
            # Return probability of positive class
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.Series:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'split' or 'gain'
            
        Returns:
            Series of feature importances sorted descending
        """
        importance = self.model.booster_.feature_importance(importance_type=importance_type)
        feature_names = self.model.booster_.feature_name()
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
    
    def save(self, path: str):
        """Save model to disk"""
        self.model.booster_.save_model(path)
    
    def load(self, path: str):
        """Load model from disk"""
        self.model = lgb.Booster(model_file=path)
        return self
