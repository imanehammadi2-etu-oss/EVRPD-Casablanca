"""
===============================================================================
Module : Model Ensemble & Consensus Score Engine
File   : forecasting/ensemble.py
Description : Ensemble pondéré par l'erreur inverse (Inverse Error Weighting)
              et calcul de l'indice de consensus inter-modèles.
===============================================================================
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("EnsembleEngine")


def compute_inverse_error_weights(
    model_maes: Dict[str, float],
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Calcule les poids d'ensemble w_m inversement proportionnels à la MAE de validation :
    w_m = (1 / (MAE_m + eps)) / sum_j (1 / (MAE_j + eps))
    """
    inv_errors = {m: 1.0 / (mae + eps) for m, mae in model_maes.items()}
    total_inv = sum(inv_errors.values())
    weights = {m: val / total_inv for m, val in inv_errors.items()}
    return weights


def compute_consensus_score(
    predictions_matrix: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, float]:
    """
    Calcule le score de consensus basé sur la dispersion relative inter-modèles à chaque pas de temps.
    
    Formule :
    consensus_score(t) = 1 / (1 + std(predictions_t) / (abs(mean(predictions_t)) + eps))
    
    Arguments :
    ------------
    predictions_matrix : np.ndarray de forme (n_models, horizon, n_targets) ou (n_models, horizon)
    
    Renvoie :
    ---------
    consensus_timeline : np.ndarray de forme (horizon,)
    mean_consensus_score : float entre 0.0 et 1.0 (ou 100.0%)
    """
    # Mean and std across models axis (axis 0)
    std_t = np.std(predictions_matrix, axis=0)
    mean_t = np.mean(predictions_matrix, axis=0)

    relative_dispersion = std_t / (np.abs(mean_t) + eps)
    consensus_timeline = 1.0 / (1.0 + relative_dispersion)

    mean_consensus = float(np.mean(consensus_timeline))
    return consensus_timeline, mean_consensus


def build_weighted_ensemble_forecast(
    predictions_dict: Dict[str, np.ndarray],
    weights: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Combine les prédictions des modèles individuels selon leurs poids d'ensemble.
    
    Formule :
    forecast_final(t) = sum_m (w_m * forecast_m(t))
    
    Renvoie :
    ---------
    forecast_final : np.ndarray
    consensus_timeline : np.ndarray
    overall_consensus_score : float
    """
    model_names = list(predictions_dict.keys())
    pred_list = [predictions_dict[m] for m in model_names]

    # Convertir en matrice (n_models, ...)
    preds_matrix = np.array(pred_list, dtype=np.float32)

    # Calcul du forecast pondéré
    final_forecast = np.zeros_like(pred_list[0], dtype=np.float32)
    for m_name in model_names:
        w = weights.get(m_name, 1.0 / len(model_names))
        final_forecast += w * predictions_dict[m_name]

    # Calcul du consensus
    consensus_tl, overall_score = compute_consensus_score(preds_matrix)

    logger.info(f"Ensemble pondéré construit avec {len(model_names)} modèles. Score de consensus global = {overall_score * 100.0:.2f}%")
    return final_forecast, consensus_tl, overall_score
