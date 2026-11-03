"""
===============================================================================
Module : Preprocessing & Normalization without Data Leakage
File   : forecasting/preprocessing.py
Description : Normalisation scientifique (StandardScaler, RobustScaler, MinMaxScaler, RevIN).
              Fit uniquement sur Train, transform sur Validation et Test.
              Inverse transform séparé pour les cibles et features.
===============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union

try:
    from sklearn.preprocessing import StandardScaler as SKStandardScaler
    from sklearn.preprocessing import RobustScaler as SKRobustScaler
    from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger("Preprocessing")


class NumPyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.scale_

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr * self.scale_ + self.mean_


class NumPyMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.data_min_ = np.min(X_arr, axis=0)
        self.data_max_ = np.max(X_arr, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ < 1e-8] = 1.0
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        norm = (X_arr - self.data_min_) / self.data_range_
        return norm * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        norm = (X_arr - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return norm * self.data_range_ + self.data_min_


class RevIN:
    """
    Reversible Instance Normalization (RevIN) pour séries temporelles (Kim et al., ICLR 2022).
    Ajustement réversible par instance de séquence temporelle.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.last_mean = None
        self.last_stdev = None

    def _get_stats(self, x: np.ndarray):
        if x.ndim == 2:
            mean = np.mean(x, axis=0, keepdims=True)
            stdev = np.sqrt(np.var(x, axis=0, keepdims=True) + self.eps)
        else:
            mean = np.mean(x, axis=1, keepdims=True)
            stdev = np.sqrt(np.var(x, axis=1, keepdims=True) + self.eps)
        return mean, stdev

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.last_mean, self.last_stdev = self._get_stats(x)
        return (x - self.last_mean) / self.last_stdev

    def transform(self, x: np.ndarray) -> np.ndarray:
        mean, stdev = self._get_stats(x)
        self.last_mean, self.last_stdev = mean, stdev
        return (x - mean) / stdev

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.last_mean is None or self.last_stdev is None:
            return x

        mean = self.last_mean
        stdev = self.last_stdev

        if x.ndim == 3 and mean.ndim == 3:
            if mean.shape[0] != x.shape[0]:
                mean = np.mean(mean, axis=0, keepdims=True)
                stdev = np.mean(stdev, axis=0, keepdims=True)
            mean = mean[:, :, :x.shape[2]]
            stdev = stdev[:, :, :x.shape[2]]
        elif x.ndim == 2 and mean.ndim == 2:
            mean = mean[:, :x.shape[1]]
            stdev = stdev[:, :x.shape[1]]

        return x * stdev + mean


class TimeSeriesScalerManager:
    """
    Gestionnaire unifié de normalisation sans fuite de données (Data Leakage).
    Fitte uniquement sur les données d'entraînement (Train) et transforme Validation / Test.
    """
    def __init__(self, method: str = "standard"):
        self.method = method.lower()
        self.feature_scaler = None
        self.target_scaler = None
        self.is_fitted = False

    def _create_scaler(self):
        if self.method == "standard":
            return SKStandardScaler() if HAS_SKLEARN else NumPyStandardScaler()
        elif self.method == "robust":
            return SKRobustScaler() if HAS_SKLEARN else NumPyStandardScaler()
        elif self.method == "minmax":
            return SKMinMaxScaler(feature_range=(0, 1)) if HAS_SKLEARN else NumPyMinMaxScaler(feature_range=(0, 1))
        elif self.method == "revin":
            return None
        else:
            return SKStandardScaler() if HAS_SKLEARN else NumPyStandardScaler()

    def fit_transform_train(
        self,
        df_train: pd.DataFrame,
        feature_cols: list,
        target_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Normalisation des données Train avec la méthode '{self.method}'...")
        
        df_feat_train = df_train[feature_cols].copy()
        df_targ_train = df_train[target_cols].copy()

        if self.method == "revin":
            self.is_fitted = True
            return df_feat_train, df_targ_train

        self.feature_scaler = self._create_scaler()
        self.target_scaler = self._create_scaler()

        scaled_feat = self.feature_scaler.fit_transform(df_feat_train)
        scaled_targ = self.target_scaler.fit_transform(df_targ_train)

        df_feat_scaled = pd.DataFrame(scaled_feat, columns=feature_cols, index=df_train.index)
        df_targ_scaled = pd.DataFrame(scaled_targ, columns=target_cols, index=df_train.index)

        self.is_fitted = True
        return df_feat_scaled, df_targ_scaled

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.is_fitted and self.method != "revin":
            raise ValueError("Le scaler n'a pas encore été fitted sur le Train !")

        df_feat = df[feature_cols].copy()
        df_targ = df[target_cols].copy()

        if self.method == "revin":
            return df_feat, df_targ

        scaled_feat = self.feature_scaler.transform(df_feat)
        scaled_targ = self.target_scaler.transform(df_targ)

        df_feat_scaled = pd.DataFrame(scaled_feat, columns=feature_cols, index=df.index)
        df_targ_scaled = pd.DataFrame(scaled_targ, columns=target_cols, index=df.index)

        return df_feat_scaled, df_targ_scaled

    def inverse_transform_targets(self, y_pred: np.ndarray) -> np.ndarray:
        if self.method == "revin" or self.target_scaler is None:
            return y_pred

        orig_shape = y_pred.shape
        if y_pred.ndim == 3:
            n_samples, pred_len, n_targets = orig_shape
            y_flat = y_pred.reshape(-1, n_targets)
            y_inv_flat = self.target_scaler.inverse_transform(y_flat)
            return y_inv_flat.reshape(n_samples, pred_len, n_targets)
        elif y_pred.ndim == 2:
            return self.target_scaler.inverse_transform(y_pred)
        else:
            return y_pred
