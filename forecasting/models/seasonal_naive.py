"""
===============================================================================
Module : Seasonal Naive Forecaster Baseline
File   : forecasting/models/seasonal_naive.py
Description : Baseline obligatoire prédisant la valeur observée une période auparavant.
===============================================================================
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from forecasting.base_forecaster import BaseForecaster


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Forecaster naïf saisonnier de référence.
    Prédit la valeur observée il y a S pas de temps (saisonnalité par défaut : 24 pas).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="SeasonalNaive", config=config)
        self.seasonality_period = self.config.get("seasonality_period", 24)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "SeasonalNaiveForecaster":
        t0 = time.time()
        # Modèle sans paramètres à apprendre, fit instantané
        self.train_time = time.time() - t0
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (n_samples, context_length, n_features)
        Sortie y_pred shape: (n_samples, prediction_length, n_targets)
        """
        n_samples, context_len, n_features = X.shape
        prediction_length = self.config.get("prediction_length", 24)

        # Extraire la cible (première colonne ou cible spécifiée)
        # Prendre les derniers pas de la période de saisonnalité
        y_pred_list = []
        for i in range(n_samples):
            history = X[i, :, 0] # cible principale
            period = min(self.seasonality_period, context_len)
            recent_season = history[-period:]
            
            # Répéter la saison pour couvrir prediction_length
            repeats = (prediction_length // len(recent_season)) + 1
            pred_full = np.tile(recent_season, repeats)[:prediction_length]
            y_pred_list.append(np.expand_dims(pred_full, axis=-1))

        return np.array(y_pred_list, dtype=np.float32)
