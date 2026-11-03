"""
===============================================================================
Module : End-to-End Forecasting Data Pipeline
File   : forecasting/data_pipeline.py
Description : Orchestrateur complet de préparation des données :
              Synchronisation, Feature Engineering, Split Chronologique,
              Normalisation sans fuite et Fenêtrage Supervisé.
===============================================================================
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from forecasting.synchronization import synchronize_time_series
from forecasting.feature_engineering import add_calendar_features, validate_no_future_leakage
from forecasting.model_selection import chronological_split
from forecasting.preprocessing import TimeSeriesScalerManager
from forecasting.windows import create_windows

logger = logging.getLogger("DataPipeline")


class ForecastingDataPipeline:
    """
    Pipeline complet de préparation des données temporelles multivariées pour les forecasters.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.split_cfg = config.get("split", {})
        self.fc_cfg = config.get("forecasting", {})

        self.timestamp_col = self.data_cfg.get("timestamp_column", "timestamp")
        self.id_col = self.data_cfg.get("id_column", "unique_id")
        self.target_cols = self.data_cfg.get("target_columns", ["demand"])
        self.feature_cols = self.data_cfg.get("feature_columns", ["demand", "traffic_index", "temperature"])

        self.frequency = self.data_cfg.get("frequency", "h")
        self.timezone = self.data_cfg.get("timezone", "Africa/Casablanca")
        self.missing_policy = self.data_cfg.get("missing_value_policy", "interpolate")
        self.scaling_method = self.data_cfg.get("scaling_method", "standard")

        self.context_length = self.fc_cfg.get("context_length", 168)
        self.prediction_length = self.fc_cfg.get("prediction_length", 24)
        self.stride = self.fc_cfg.get("window_stride", 1)

        self.scaler_manager = TimeSeriesScalerManager(method=self.scaling_method)
        self.audit_info: Dict[str, Any] = {}

    def prepare_pipeline_data(
        self,
        raw_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Exécute la préparation complète de A à Z :
        1. Synchronisation temporelle & dédoublonnage.
        2. Feature engineering calendaire & cyclique (sine/cos).
        3. Split chronologique strict Train / Validation / Test.
        4. Normalisation fittée uniquement sur Train.
        5. Génération des fenêtres supervisées 3D pour chaque partition.
        """
        # 1. Synchronisation Temporelle
        df_synced, audit = synchronize_time_series(
            raw_df,
            timestamp_col=self.timestamp_col,
            id_col=self.id_col,
            frequency=self.frequency,
            timezone=self.timezone,
            missing_value_policy=self.missing_policy
        )
        self.audit_info = audit

        # 2. Feature Engineering
        df_feat = add_calendar_features(df_synced, timestamp_col=self.timestamp_col)
        
        # S'assurer que toutes les feature_cols existent
        actual_feature_cols = [c for c in self.feature_cols if c in df_feat.columns]
        if not actual_feature_cols:
            actual_feature_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns]

        validate_no_future_leakage(actual_feature_cols, self.target_cols)

        # 3. Split Chronologique Strict
        df_train, df_val, df_test = chronological_split(
            df_feat,
            train_ratio=self.split_cfg.get("train_ratio", 0.70),
            validation_ratio=self.split_cfg.get("validation_ratio", 0.15),
            test_ratio=self.split_cfg.get("test_ratio", 0.15)
        )

        # 4. Normalisation sans fuite de données
        df_feat_train_sc, df_targ_train_sc = self.scaler_manager.fit_transform_train(
            df_train, actual_feature_cols, self.target_cols
        )
        df_feat_val_sc, df_targ_val_sc = self.scaler_manager.transform(
            df_val, actual_feature_cols, self.target_cols
        )
        df_feat_test_sc, df_targ_test_sc = self.scaler_manager.transform(
            df_test, actual_feature_cols, self.target_cols
        )

        # 5. Fenêtrage Supervisé
        X_train, y_train, meta_train = create_windows(
            df_feat_train_sc, df_targ_train_sc,
            self.context_length, self.prediction_length, self.stride
        )
        X_val, y_val, meta_val = create_windows(
            df_feat_val_sc, df_targ_val_sc,
            self.context_length, self.prediction_length, self.stride
        )
        X_test, y_test, meta_test = create_windows(
            df_feat_test_sc, df_targ_test_sc,
            self.context_length, self.prediction_length, self.stride
        )

        return {
            "df_synced": df_synced,
            "df_feat": df_feat,
            "feature_cols": actual_feature_cols,
            "target_cols": self.target_cols,
            "scaler_manager": self.scaler_manager,
            "X_train": X_train, "y_train": y_train, "meta_train": meta_train,
            "X_val": X_val, "y_val": y_val, "meta_val": meta_val,
            "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
            "audit_info": audit
        }
