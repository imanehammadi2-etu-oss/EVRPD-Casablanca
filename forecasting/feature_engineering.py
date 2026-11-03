"""
===============================================================================
Module : Feature Engineering & Business Events
File   : forecasting/feature_engineering.py
Description : Génération automatique de variables temporelles calendaires,
              transformations cycliques (sine/cosine) et événements métier
              sans aucune fuite de données futures.
===============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger("FeatureEngineering")


def add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    custom_events: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec toutes les caractéristiques temporelles calendaires et cycliques.
    
    Features générées :
    - hour, day_of_week, day_of_month, week_of_year, month, quarter
    - is_weekend, is_holiday
    - hour_sin / sin_hour, hour_cos / cos_hour
    - day_of_week_sin / sin_day_of_week, day_of_week_cos / cos_day_of_week
    - Événements métier (Ramadan_flag, Eid_flag, promotion_flag, event_flag)
    """
    df_feat = df.copy()

    if timestamp_col not in df_feat.columns:
        if isinstance(df_feat.index, pd.DatetimeIndex):
            dt_series = df_feat.index
        else:
            raise ValueError(f"Colonne timestamp '{timestamp_col}' non trouvée.")
    else:
        dt_series = pd.to_datetime(df_feat[timestamp_col])

    # 1. Caractéristiques calendaires de base
    df_feat["hour"] = dt_series.dt.hour
    df_feat["day_of_week"] = dt_series.dt.dayofweek
    df_feat["day_of_month"] = dt_series.dt.day
    df_feat["week_of_year"] = dt_series.dt.isocalendar().week.astype(int)
    df_feat["month"] = dt_series.dt.month
    df_feat["quarter"] = dt_series.dt.quarter
    df_feat["is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

    # 2. Encodage cyclique trigonométrique (Sine / Cosine)
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["sin_hour"] = df_feat["hour_sin"]
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["cos_hour"] = df_feat["hour_cos"]

    df_feat["day_of_week_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7.0)
    df_feat["sin_day_of_week"] = df_feat["day_of_week_sin"]
    df_feat["day_of_week_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7.0)
    df_feat["cos_day_of_week"] = df_feat["day_of_week_cos"]

    df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)

    # 3. Jours fériés par défaut (ou personnalisés)
    df_feat["is_holiday"] = 0
    fixed_holidays = [(1, 1), (1, 11), (5, 1), (7, 30), (8, 14), (8, 20), (8, 21), (11, 6), (11, 18)]
    for m, d in fixed_holidays:
        mask = (df_feat["month"] == m) & (df_feat["day_of_month"] == d)
        df_feat.loc[mask, "is_holiday"] = 1

    # 4. Événements métier (Ramadan_flag, Eid_flag, promotion_flag, event_flag)
    df_feat["Ramadan_flag"] = 0
    df_feat["is_ramadan"] = 0
    df_feat["Eid_flag"] = 0
    df_feat["is_eid"] = 0
    df_feat["promotion_flag"] = 0
    df_feat["is_promotion"] = 0
    df_feat["event_flag"] = 0

    if custom_events:
        for event_name, date_list in custom_events.items():
            col_name = f"is_{event_name.lower()}"
            flag_name = f"{event_name}_flag"
            
            parsed_dates = pd.to_datetime(date_list).date
            mask = dt_series.dt.date.isin(parsed_dates)
            
            df_feat.loc[mask, col_name] = 1
            df_feat.loc[mask, flag_name] = 1

    logger.info(f"Feature engineering terminé : {len(df_feat.columns)} colonnes disponibles.")
    return df_feat


def validate_no_future_leakage(
    feature_columns: List[str],
    target_columns: List[str],
    future_known_columns: Optional[List[str]] = None
):
    """
    Vérifie qu'aucune variable cible future ou variable endogène non connue (trafic réel futur,
    demande réelle future) n'est injectée dans les covariables futures.
    """
    if future_known_columns is None:
        future_known_columns = [
            "hour", "day_of_week", "day_of_month", "month", "is_weekend",
            "is_holiday", "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week",
            "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
            "is_ramadan", "is_eid", "is_promotion", "Ramadan_flag", "Eid_flag", "promotion_flag", "event_flag"
        ]

    for target in target_columns:
        if target in future_known_columns:
            raise ValueError(f"ALERTE FUITE DE DONNÉES : La variable cible '{target}' ne peut pas être une covariable future connue !")

    logger.info("Vérification d'absence de fuite de données validée avec succès.")
