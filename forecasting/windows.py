"""
===============================================================================
Module : Supervised Window Generation
File   : forecasting/windows.py
Description : Découpage en fenêtres supervisées pour l'apprentissage séquentiel.
              Strict respect des frontières sans fuite temporelle ni mélange inter-zones.
===============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union

logger = logging.getLogger("SupervisedWindows")


def create_windows(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    context_length: int = 168,
    prediction_length: int = 24,
    stride: int = 1,
    timestamp_col: Optional[Union[str, pd.Series, np.ndarray, List]] = None,
    id_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Génère des fenêtres supervisées à partir de DataFrames X et y.
    
    Formes retournées :
    - X_windows : np.ndarray de forme (n_samples, context_length, n_features)
    - y_windows : np.ndarray de forme (n_samples, prediction_length, n_targets)
    - metadata  : liste de dictionnaires contenant les timestamps de contexte et d'horizon.
    """
    X_vals = X_df.values
    y_vals = y_df.values

    n_rows, n_features = X_vals.shape
    _, n_targets = y_vals.shape

    total_window_len = context_length + prediction_length

    if n_rows < total_window_len:
        logger.warning(
            f"Taille insuffisante ({n_rows} lignes) pour générer des fenêtres "
            f"de taille totale {total_window_len} (contexte={context_length} + pred={prediction_length})."
        )
        return (
            np.empty((0, context_length, n_features), dtype=np.float32),
            np.empty((0, prediction_length, n_targets), dtype=np.float32),
            []
        )

    if timestamp_col is None:
        timestamps = X_df.index
    elif isinstance(timestamp_col, (pd.Series, np.ndarray, list)):
        timestamps = np.asarray(timestamp_col)
    elif hasattr(X_df, "columns") and isinstance(timestamp_col, str) and timestamp_col in X_df.columns:
        timestamps = X_df[timestamp_col].values
    else:
        timestamps = X_df.index

    X_list = []
    y_list = []
    metadata = []

    for i in range(0, n_rows - total_window_len + 1, stride):
        context_start = i
        context_end = i + context_length
        
        pred_start = context_end
        pred_end = pred_start + prediction_length

        x_win = X_vals[context_start:context_end, :]
        y_win = y_vals[pred_start:pred_end, :]

        X_list.append(x_win)
        y_list.append(y_win)

        meta = {
            "window_index": len(X_list) - 1,
            "context_timestamps": timestamps[context_start:context_end] if len(timestamps) > 0 else None,
            "prediction_timestamps": timestamps[pred_start:pred_end] if len(timestamps) > 0 else None,
        }
        metadata.append(meta)

    X_windows = np.array(X_list, dtype=np.float32)
    y_windows = np.array(y_list, dtype=np.float32)

    logger.info(
        f"Génération réussie de {len(X_windows)} fenêtres supervisées "
        f"(X: {X_windows.shape}, y: {y_windows.shape})."
    )
    return X_windows, y_windows, metadata


def create_forecasting_windows(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    context_length: int = 168,
    prediction_length: int = 24,
    window_stride: int = 1,
    group_column: str = "zone_id",
    timestamp_column: str = "timestamp"
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Crée des fenêtres supervisées séparément par groupe/zone sans mélanger les observations.

    Shapes attendues :
    - X.shape = (n_samples, context_length, n_features)
    - y.shape = (n_samples, prediction_length, n_targets)
    """
    logger.info(f"Génération des fenêtres supervisées par zone ({group_column}) avec context={context_length}, pred={prediction_length}...")
    
    all_X = []
    all_y = []
    all_metadata = []

    if group_column not in data.columns:
        X_df = data[feature_columns]
        y_df = data[target_columns]
        ts_vals = data[timestamp_column].values if timestamp_column in data.columns else None
        return create_windows(
            X_df, y_df,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=window_stride,
            timestamp_col=ts_vals
        )

    grouped = data.groupby(group_column)

    for zone_id, group_df in grouped:
        group_df_sorted = group_df.sort_values(by=timestamp_column if timestamp_column in group_df.columns else group_df.index)
        
        X_df = group_df_sorted[feature_columns]
        y_df = group_df_sorted[target_columns]
        ts_vals = group_df_sorted[timestamp_column].values if timestamp_column in group_df_sorted.columns else None

        X_win, y_win, meta = create_windows(
            X_df, y_df,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=window_stride,
            timestamp_col=ts_vals,
            id_col=group_column
        )

        for m in meta:
            m["zone_id"] = zone_id

        if len(X_win) > 0:
            all_X.append(X_win)
            all_y.append(y_win)
            all_metadata.extend(meta)

    if len(all_X) == 0:
        logger.warning("Aucune fenêtre supervisée n'a pu être générée (données insuffisantes).")
        return (
            np.empty((0, context_length, len(feature_columns)), dtype=np.float32),
            np.empty((0, prediction_length, len(target_columns)), dtype=np.float32),
            []
        )

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)

    logger.info(f"Fenêtrage multi-zones réussi : X {X_final.shape}, y {y_final.shape}, total samples = {len(X_final)}")
    return X_final, y_final, all_metadata
