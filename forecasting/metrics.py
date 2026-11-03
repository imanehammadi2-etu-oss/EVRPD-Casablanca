"""
===============================================================================
Module : Scientific Performance Metrics & Evaluation Breakdown
File   : forecasting/metrics.py
Description : Calcul unifié des métriques MAE, RMSE, sMAPE, MASE, MAPE avec décomposition
              par horizon temporelle, par zone, heures de pointe et baselines.
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    seasonality: int = 24
) -> float:
    """
    Calcule l'erreur scalaire absolue moyenne (MASE - Mean Absolute Scaled Error).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    mae_pred = np.mean(np.abs(y_t - y_p))

    if y_train_history is not None and len(y_train_history) > seasonality:
        y_hist = np.asarray(y_train_history, dtype=np.float64).flatten()
        mae_naive = np.mean(np.abs(y_hist[seasonality:] - y_hist[:-seasonality]))
    else:
        if len(y_t) > 1:
            mae_naive = np.mean(np.abs(np.diff(y_t)))
        else:
            mae_naive = 1.0

    if mae_naive < 1e-8:
        mae_naive = 1e-8

    return float(mae_pred / mae_naive)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    train_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Calcule l'ensemble des métriques globales (MAE, RMSE, sMAPE, MASE, MAPE).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    smape = float(np.mean(np.where(denom > 1e-8, np.abs(y_p - y_t) / denom, 0.0)) * 100.0)
    mape = float(np.mean(np.where(np.abs(y_t) > 1e-8, np.abs((y_t - y_p) / y_t), 0.0)) * 100.0)
    mase = compute_mase(y_t, y_p, y_train_history=y_train_history)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "sMAPE": round(smape, 4),
        "MASE": round(mase, 4),
        "MAPE": round(mape, 4),
        "train_time_sec": round(train_time_sec, 4),
        "inference_time_sec": round(inference_time_sec, 4),
        "seed": seed
    }


def compute_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calcule les métriques séparément pour chaque pas de temps d'horizon h in [1..H]
    ou pour une liste spécifique (ex: h=1, h=6, h=12, h=24).
    """
    if y_true.ndim == 2:
        y_true = np.expand_dims(y_true, axis=-1)
    if y_pred.ndim == 2:
        y_pred = np.expand_dims(y_pred, axis=-1)

    n_samples, horizon, n_targets = y_true.shape
    records = []

    steps_to_eval = [h - 1 for h in target_horizons if 1 <= h <= horizon] if target_horizons else list(range(horizon))

    for h_idx in steps_to_eval:
        t_h = y_true[:, h_idx, :].flatten()
        p_h = y_pred[:, h_idx, :].flatten()

        m = compute_all_metrics(t_h, p_h)
        m["horizon_step"] = h_idx + 1
        records.append(m)

    return pd.DataFrame(records)


def compute_peak_hour_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours_array: np.ndarray,
    peak_hours: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Évalue la performance de prédiction spécifiquement pendant les heures de pointe.
    """
    if peak_hours is None:
        peak_hours = [8, 9, 10, 15, 16, 17]

    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    hours = np.asarray(hours_array, dtype=np.int32).flatten()[:len(y_t)]

    mask = np.isin(hours, peak_hours)
    if not np.any(mask):
        return compute_all_metrics(y_t, y_p)

    return compute_all_metrics(y_t[mask], y_p[mask])


def evaluate_baselines(
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    target_idx: int = 0
) -> Dict[str, Dict[str, float]]:
    """
    Évalue les modèles de baseline obligatoires :
    1. Last Value (Naive)
    2. Seasonal Naive 24h
    3. Seasonal Naive 168h
    """
    # X_windows shape: (n_samples, context_length, n_features)
    # y_windows shape: (n_samples, prediction_length, n_targets)
    n_samples, context_len, n_features = X_windows.shape
    _, pred_len, n_targets = y_windows.shape

    y_true = y_windows[:, :, target_idx]

    # 1. Naive (Last Value)
    last_val = X_windows[:, -1, target_idx]  # shape: (n_samples,)
    pred_naive = np.repeat(last_val[:, np.newaxis], pred_len, axis=1)

    # 2. Seasonal Naive 24h
    if context_len >= 24:
        s24 = X_windows[:, -24:, target_idx]
        pred_s24 = np.tile(s24, (1, int(np.ceil(pred_len / 24.0))))[:, :pred_len]
    else:
        pred_s24 = pred_naive

    # 3. Seasonal Naive 168h
    if context_len >= 168:
        s168 = X_windows[:, -168:, target_idx]
        pred_s168 = np.tile(s168, (1, int(np.ceil(pred_len / 168.0))))[:, :pred_len]
    else:
        pred_s168 = pred_s24

    return {
        "Naive_LastValue": compute_all_metrics(y_true, pred_naive),
        "Seasonal_Naive_24h": compute_all_metrics(y_true, pred_s24),
        "Seasonal_Naive_168h": compute_all_metrics(y_true, pred_s168),
    }
