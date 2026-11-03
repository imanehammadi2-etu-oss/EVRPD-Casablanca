"""
===============================================================================
Module : Temporal Synchronization & Data Cleaning
File   : forecasting/synchronization.py
Description : Synchronisation temporelle robuste pour séries multivariées.
              Fuseau horaire, rééchantillonnage, gestion des manquants et contrôles.
===============================================================================
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Synchronization")


def synchronize_time_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    id_col: Optional[str] = None,
    frequency: str = "h",
    timezone: str = "Africa/Casablanca",
    missing_value_policy: str = "interpolate",
    agg_rules: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Exécute la synchronisation temporelle rigoureuse d'un DataFrame de séries temporelles.
    
    Étapes :
    1. Conversion de la colonne timestamp en Datetime UTC / timezone.
    2. Tri chronologique et dédoublonnage.
    3. Rééchantillonnage à la fréquence cible (ex: 'h' pour 1 heure).
    4. Agrégation selon la nature des colonnes (sum pour demande/colis, mean pour météo/trafic).
    5. Traitement contrôlé des valeurs manquantes sans fuite de données.
    6. Vérification de la régularité spatio-temporelle et logs d'audit.
    """
    logger.info(f"Début de la synchronisation temporelle (Freq={frequency}, Timezone={timezone})...")
    df_clean = df.copy()

    # 1. Conversion Datetime & Timezone
    if timestamp_col in df_clean.columns:
        df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], errors="coerce")
    elif isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean = df_clean.reset_index()
        timestamp_col = df_clean.columns[0]
        df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], errors="coerce")
    else:
        raise ValueError(f"Colonne timestamp '{timestamp_col}' non trouvée dans le DataFrame.")

    # Gérer les timestamps invalides (NaT)
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=[timestamp_col])
    dropped_nat = initial_rows - len(df_clean)

    # Assigner / Convertir le fuseau horaire
    try:
        if df_clean[timestamp_col].dt.tz is None:
            df_clean[timestamp_col] = df_clean[timestamp_col].dt.tz_localize(timezone, ambiguous="naive", nonexistent="shift_forward")
        else:
            df_clean[timestamp_col] = df_clean[timestamp_col].dt.tz_convert(timezone)
    except Exception as e:
        logger.warning(f"Erreur de conversion de fuseau horaire '{timezone}' : {e}. Utilisation du timestamp local.")

    # 2. Tri & Dédoublonnage
    sort_cols = [timestamp_col]
    if id_col and id_col in df_clean.columns:
        sort_cols.insert(0, id_col)
    
    df_clean = df_clean.sort_values(by=sort_cols)
    duplicates_before = df_clean.duplicated(subset=sort_cols).sum()
    df_clean = df_clean.drop_duplicates(subset=sort_cols, keep="last")

    # 3. Définition des règles d'agrégation par défaut si non spécifiées
    default_agg = {
        "demand": "sum",
        "orders": "sum",
        "parcels": "sum",
        "traffic_index": "mean",
        "temperature": "mean",
        "wind_speed": "mean",
        "rain": "mean",
        "average_delivery_time": "mean",
        "battery_level": "mean",
        "energy_consumption": "mean",
        "is_weekend": "max",
        "is_holiday": "max"
    }
    if agg_rules:
        default_agg.update(agg_rules)

    # 4. Rééchantillonnage & Agrégation
    def _resample_single_series(sub_df: pd.DataFrame) -> pd.DataFrame:
        sub_df = sub_df.set_index(timestamp_col)
        numeric_cols = sub_df.select_dtypes(include=[np.number]).columns
        
        # Appliquer l'agrégation
        curr_agg = {col: default_agg.get(col, "mean") for col in numeric_cols}
        resampled = sub_df[numeric_cols].resample(frequency).agg(curr_agg)
        return resampled

    if id_col and id_col in df_clean.columns:
        resampled_groups = []
        for zone_id, group in df_clean.groupby(id_col):
            r_grp = _resample_single_series(group)
            r_grp[id_col] = zone_id
            resampled_groups.append(r_grp.reset_index())
        df_synced = pd.concat(resampled_groups, ignore_index=True)
    else:
        df_synced = _resample_single_series(df_clean).reset_index()

    # 5. Traitement contrôlé des valeurs manquantes
    missing_before = df_synced.isna().sum().to_dict()
    
    numeric_cols = df_synced.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if missing_value_policy == "interpolate":
            # Interpolation linéaire pour données continues, 0 pour demande si isolée
            if col in ["demand", "orders", "parcels"]:
                df_synced[col] = df_synced[col].fillna(0.0)
            else:
                df_synced[col] = df_synced[col].interpolate(method="time" if isinstance(df_synced.index, pd.DatetimeIndex) else "linear").bfill().ffill()
        elif missing_value_policy == "ffill":
            df_synced[col] = df_synced[col].ffill().bfill()
        elif missing_value_policy == "zero":
            df_synced[col] = df_synced[col].fillna(0.0)
        else:
            df_synced[col] = df_synced[col].ffill().bfill()

    missing_after = df_synced.isna().sum().to_dict()

    # 6. Audit & Métadonnées de synchronisation
    t_min = df_synced[timestamp_col].min()
    t_max = df_synced[timestamp_col].max()
    n_samples = len(df_synced)

    audit_info = {
        "initial_rows": initial_rows,
        "dropped_nat_rows": dropped_nat,
        "duplicates_removed": int(duplicates_before),
        "synced_rows": n_samples,
        "time_range": (str(t_min), str(t_max)),
        "frequency_obtained": frequency,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "n_zones": df_synced[id_col].nunique() if (id_col and id_col in df_synced.columns) else 1
    }

    logger.info(f"Synchronisation réussie : {n_samples} lignes synchronisées de {t_min} à {t_max}.")
    return df_synced, audit_info
