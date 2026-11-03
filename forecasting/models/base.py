"""
===============================================================================
Module : Base Forecaster
File   : forecasting/models/base.py
Description : Classe abstraite de référence pour tous les modèles de prévision.
===============================================================================
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple


class BaseForecaster(ABC):
    """
    Interface unifiée pour les modèles de prévision de séries temporelles multivariées.
    """
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.is_trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        """Entraîne le modèle sur la matrice X [T, D] et la cible y [T, H]."""
        pass

    @abstractmethod
    def predict(self, X_history: np.ndarray, horizon: int = 24) -> np.ndarray:
        """Prédit la série temporelle sur l'horizon H."""
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcule les métriques d'erreur standard : RMSE, MAE, R²."""
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        }
