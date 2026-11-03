"""
===============================================================================
Module : Base Forecaster Protocol & Abstract Interface
File   : forecasting/base_forecaster.py
Description : Interface abstraite commune pour tous les forecasters du pipeline.
===============================================================================
"""

from abc import ABC, abstractmethod
import time
import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques scientifiques standardisées : MAE, RMSE, sMAPE, MAPE.
    S'assure de l'alignement des formes et prévient toute division par zéro.
    """
    y_true_f = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred_f = np.asarray(y_pred, dtype=np.float64).flatten()

    mae = float(np.mean(np.abs(y_true_f - y_pred_f)))
    rmse = float(np.sqrt(np.mean((y_true_f - y_pred_f) ** 2)))

    # sMAPE (%) : Symmetric Mean Absolute Percentage Error
    denom = (np.abs(y_true_f) + np.abs(y_pred_f)) / 2.0
    smape = float(np.mean(np.where(denom > 1e-8, np.abs(y_pred_f - y_true_f) / denom, 0.0)) * 100.0)

    # MAPE (%) sécurisé contre zéro
    mape = float(np.mean(np.where(np.abs(y_true_f) > 1e-8, np.abs((y_true_f - y_pred_f) / y_true_f), 0.0)) * 100.0)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "sMAPE": round(smape, 4),
        "MAPE": round(mape, 4)
    }


class BaseForecaster(ABC):
    """
    Classe abstraite commune imposée pour tous les modèles de forecasting.
    """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config if config is not None else {}
        self.seed = self.config.get("seed", 42)
        self.is_fitted = False
        
        # Attributs de résultat obligatoires
        self.last_y_pred: Optional[np.ndarray] = None
        self.last_metrics: Dict[str, float] = {}
        self.train_time: float = 0.0
        self.inference_time: float = 0.0

        # Fixer les graines de reproductibilité
        self._set_seed(self.seed)

    def _set_seed(self, seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "BaseForecaster":
        """
        Entraîne le modèle de prévision.
        X_train: shape (n_samples, context_length, n_features)
        y_train: shape (n_samples, prediction_length, n_targets)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Génère les prédictions sur X.
        Entrée X : shape (n_samples, context_length, n_features)
        Sortie y_pred : shape (n_samples, prediction_length, n_targets)
        """
        pass

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Évalue le modèle et renvoie MAE, RMSE, sMAPE, MAPE.
        """
        t0 = time.time()
        y_pred = self.predict(X)
        self.inference_time = time.time() - t0
        self.last_y_pred = y_pred

        metrics = compute_metrics(y_true, y_pred)
        metrics["train_time_sec"] = round(self.train_time, 4)
        metrics["inference_time_sec"] = round(self.inference_time, 4)
        metrics["seed"] = self.seed

        self.last_metrics = metrics
        return metrics

    def save(self, path: str):
        """Sauvegarde l'instance du modèle et sa configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseForecaster":
        """Charge une instance de modèle depuis le disque."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
