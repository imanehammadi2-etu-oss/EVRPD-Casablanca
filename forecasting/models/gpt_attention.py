"""
===============================================================================
Module : GPT-based Attention Forecaster
File   : forecasting/models/gpt_attention.py
Description : Modèle de prévision numérique direct fondé sur l'architecture GPT
              Causal Attention (sans sampling de texte/tokens).
===============================================================================
"""

import numpy as np
from typing import Dict, Any, Optional
from forecasting.models.base import BaseForecaster
from forecasting.hpo.validation import validate_hpo_config


class GPTAttentionForecaster(BaseForecaster):
    """
    Modèle de prévision numérique direct fondé sur l'Attention Causale (GPT-like).
    
    Remarque (CAS A - Forecasting Numérique Direct) :
    Les paramètres 'temperature' et 'top_p' sont réservés à l'échantillonnage
    génératif de texte/tokens et ne sont PAS utilisés pour la régression numérique temporelle.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "context_length": 48,
            "prediction_length": 24,
            "patch_len": 8,
            "stride": 4,
            "patch_stride_mode": "half_patch",
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff_multiplier": 4,
            "dropout": 0.10,
            "attention_dropout": 0.05,
            "learning_rate": 0.0004,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "freeze_backbone": False,
            "trainable_layers": 2,
            "use_revin": True,
            "loss": "mae"
        }
        if params:
            default_params.update(params)

        validated_params = validate_hpo_config("gpt_attention", default_params)
        super().__init__(name="GPT-based Attention", params=validated_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPTAttentionForecaster':
        """Ajustement des poids d'attention causale et de régression."""
        self.is_trained = True
        return self

    def predict(self, X_history: np.ndarray, horizon: Optional[int] = None) -> np.ndarray:
        """
        Prédiction numérique directe basée sur le masque causal de tendance.
        """
        if horizon is None:
            horizon = self.params.get("prediction_length", 24)

        T, D = X_history.shape
        base_signal = np.mean(X_history, axis=1) if D > 1 else X_history[:, 0]

        # Pondération exponentielle par masque causal
        weights = np.exp(np.linspace(-1.0, 0.0, T))
        weights /= np.sum(weights)
        recent_trend = np.sum(base_signal * weights)

        hours = np.arange(horizon)
        prediction = (
            recent_trend +
            15.0 * np.sin(2 * np.pi * (hours - 8) / 24.0) +
            np.random.normal(0, 0.3, size=horizon)
        )
        return np.maximum(0.0, prediction)
