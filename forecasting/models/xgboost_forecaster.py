"""
===============================================================================
Module : XGBoost / GBDT Forecaster
File   : forecasting/models/xgboost_forecaster.py
Description : Baseline Machine Learning basée sur XGBoost / Gradient Boosting.
              Inclut un fallback NumPy robuste si XGBoost / Scikit-Learn est absent.
===============================================================================
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from forecasting.base_forecaster import BaseForecaster


class NumPyRidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_offset = np.mean(X, axis=0)
        y_offset = np.mean(y, axis=0)

        X_c = X - X_offset
        y_c = y - y_offset

        # Closed-form solution: W = (X^T X + alpha * I)^-1 X^T y
        reg = self.alpha * np.eye(X_c.shape[1])
        self.weights = np.linalg.solve(X_c.T @ X_c + reg, X_c.T @ y_c)
        self.intercept = y_offset - X_offset @ self.weights
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.intercept


class XGBoostForecaster(BaseForecaster):
    """
    Forecaster basé sur XGBoost / Gradient Boosting Regressor.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="XGBoost", config=config)
        self.n_estimators = self.config.get("n_estimators", 100)
        self.max_depth = self.config.get("max_depth", 6)
        self.lr = self.config.get("learning_rate", 0.05)
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "XGBoostForecaster":
        t0 = time.time()
        n_samples, context_len, n_features = X_train.shape
        n_samples_y, pred_len, n_targets = y_train.shape

        X_flat = X_train.reshape(n_samples, -1)
        y_flat = y_train.reshape(n_samples, -1)

        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.lr,
                random_state=self.seed
            )
            self.model.fit(X_flat, y_flat)
        except ImportError:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                from sklearn.multioutput import MultiOutputRegressor
                base_gbdt = HistGradientBoostingRegressor(max_iter=self.n_estimators, random_state=self.seed)
                self.model = MultiOutputRegressor(base_gbdt)
                self.model.fit(X_flat, y_flat)
            except ImportError:
                # Fallback NumPy Ridge Regressor
                self.model = NumPyRidgeRegressor(alpha=1.0)
                self.model.fit(X_flat, y_flat)

        self.train_time = time.time() - t0
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Le modèle XGBoost n'a pas encore été entraîné.")

        n_samples, context_len, n_features = X.shape
        pred_len = self.config.get("prediction_length", 24)
        n_targets = self.config.get("n_targets", 1)

        X_flat = X.reshape(n_samples, -1)
        preds_flat = self.model.predict(X_flat)

        return preds_flat.reshape(n_samples, pred_len, -1)
