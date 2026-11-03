"""
===============================================================================
Module : Chronological Model Selection & Data Splitting
File   : forecasting/model_selection.py
Description : Split chronologique strict (Train < Validation < Test) et
              validation croisée à origine roulante (Rolling-Origin Cross Validation).
===============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Generator

logger = logging.getLogger("ModelSelection")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exécute un split chronologique STRICT sans aucun mélange aléatoire (No Random Split).
    Garantie physique : Train < Validation < Test.
    """
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-4:
        train_ratio /= total
        validation_ratio /= total
        test_ratio /= total

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * validation_ratio)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    logger.info(
        f"Split chronologique strict effectué : "
        f"Train={len(df_train)} (0:{train_end}), "
        f"Val={len(df_val)} ({train_end}:{val_end}), "
        f"Test={len(df_test)} ({val_end}:{n_rows})."
    )
    return df_train, df_val, df_test


def rolling_origin_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    min_train_size: float = 0.50,
    horizon: int = 24
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Génère des plis de validation croisée temporelle à origine roulante (Expanding Window / Rolling Origin).
    À chaque pli, la taille du train augmente et la validation reste chronologiquement supérieure.
    """
    n_rows = len(df)
    min_train_len = int(n_rows * min_train_size)
    available_len = n_rows - min_train_len - horizon

    if available_len <= 0 or n_splits <= 0:
        yield df.iloc[:min_train_len], df.iloc[min_train_len:]
        return

    step_size = max(1, available_len // n_splits)

    for i in range(n_splits):
        train_end = min_train_len + i * step_size
        val_end = train_end + horizon
        if val_end > n_rows:
            break
        
        train_fold = df.iloc[:train_end]
        val_fold = df.iloc[train_end:val_end]
        yield train_fold, val_fold
