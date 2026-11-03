"""
===============================================================================
Module : Forecasting Data Ingestion, Decomposition & Aggregation
File   : forecasting/decomposition.py
Description : Ingestion des séries multivariées, agrégation de commandes (Cas A & B),
              décomposition STL optionnelle et utilitaires de conversion.
===============================================================================
"""

import os
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger("Decomposition")


def aggregate_orders_to_demand(
    df: pd.DataFrame,
    timestamp_column: str = "created_at",
    zone_column: str = "zone_id",
    frequency: str = "1h",
    aggregation_method: str = "count",
    quantity_column: Optional[str] = None,
    order_id_column: Optional[str] = "order_id",
    timezone: str = "Africa/Casablanca",
    fill_zero: bool = True,
    output_path: Optional[str] = "data/processed/demand_by_zone_hour.csv"
) -> pd.DataFrame:
    """
    Agrége un fichier de commandes individuelles (Cas A) ou vérifie/normalise
    une demande déjà agrégée (Cas B) en une grille temporelle complète par zone.

    Format de sortie requis :
    timestamp, zone_id, demand
    """
    logger.info(f"Agrégation des données (Source shape: {df.shape}, timezone={timezone}, freq={frequency})...")
    data = df.copy()

    # Si c'est déjà agrégé (Cas B)
    if "demand" in data.columns and (timestamp_column not in data.columns or timestamp_column == "timestamp"):
        ts_col = "timestamp" if "timestamp" in data.columns else timestamp_column
        z_col = "zone_id" if "zone_id" in data.columns else zone_column

        data[ts_col] = pd.to_datetime(data[ts_col])
        if data[ts_col].dt.tz is None:
            data[ts_col] = data[ts_col].dt.tz_localize(timezone, ambiguous="naive", nonexistent="shift_forward")
        else:
            data[ts_col] = data[ts_col].dt.tz_convert(timezone)

        result_df = data[[ts_col, z_col, "demand"]].rename(columns={ts_col: "timestamp", z_col: "zone_id"})
        result_df = result_df.sort_values(by=["zone_id", "timestamp"]).reset_index(drop=True)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Demande agrégée (Cas B) sauvegardée sous : {output_path}")

        return result_df

    # Cas A : Commandes individuelles
    if timestamp_column not in data.columns:
        raise KeyError(f"Colonne timestamp '{timestamp_column}' introuvable dans les données.")

    if zone_column not in data.columns:
        # Fallback si zone_column absent : zone par défaut
        data[zone_column] = "zone_default"

    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    if data[timestamp_column].dt.tz is None:
        data[timestamp_column] = data[timestamp_column].dt.tz_localize(timezone, ambiguous="naive", nonexistent="shift_forward")
    else:
        data[timestamp_column] = data[timestamp_column].dt.tz_convert(timezone)

    # Floor selon la fréquence (ex: 1h)
    data["time_bucket"] = data[timestamp_column].dt.floor(frequency)

    # Règle d'agrégation : sum(quantity) ou count(order_id)
    if quantity_column and quantity_column in data.columns:
        agg_fn = {quantity_column: "sum"}
        val_col = quantity_column
    elif order_id_column and order_id_column in data.columns:
        agg_fn = {order_id_column: "count"}
        val_col = order_id_column
    else:
        # Compter n'importe quelle colonne non nulle
        data["_cnt"] = 1
        agg_fn = {"_cnt": "sum"}
        val_col = "_cnt"

    grouped = data.groupby([zone_column, "time_bucket"]).agg(agg_fn).reset_index()
    grouped = grouped.rename(columns={"time_bucket": "timestamp", val_col: "demand"})

    if fill_zero:
        # Grille complète : chaque zone x chaque heure
        all_zones = grouped[zone_column].unique()
        min_ts = grouped["timestamp"].min()
        max_ts = grouped["timestamp"].max()

        if pd.notnull(min_ts) and pd.notnull(max_ts):
            full_ts = pd.date_range(start=min_ts, end=max_ts, freq=frequency, tz=timezone)
            idx = pd.MultiIndex.from_product([all_zones, full_ts], names=[zone_column, "timestamp"])
            full_grid = pd.DataFrame(index=idx).reset_index()

            merged = pd.merge(full_grid, grouped, on=[zone_column, "timestamp"], how="left")
            merged["demand"] = merged["demand"].fillna(0).astype(float)
            result_df = merged.sort_values(by=[zone_column, "timestamp"]).reset_index(drop=True)
        else:
            result_df = grouped.sort_values(by=[zone_column, "timestamp"]).reset_index(drop=True)
    else:
        result_df = grouped.sort_values(by=[zone_column, "timestamp"]).reset_index(drop=True)

    result_df = result_df.rename(columns={zone_column: "zone_id"})

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"Agrégation des commandes (Cas A) réussie et sauvegardée sous : {output_path}")

    return result_df


def parse_date_features(date_str: str) -> Dict[str, Any]:
    """
    Extrait les caractéristiques temporelles et calendaires à partir d'une date (ex: '25/10/2025').
    """
    try:
        if "/" in date_str:
            parts = [int(p) for p in date_str.strip().split("/")]
            if len(parts) == 3:
                dt = date(parts[2], parts[1], parts[0])
            else:
                dt = date.today()
        elif "-" in date_str:
            parts = [int(p) for p in date_str.strip().split("-")]
            if len(parts) == 3:
                dt = date(parts[0], parts[1], parts[2])
            else:
                dt = date.today()
        else:
            dt = date.today()
    except Exception:
        dt = date.today()

    day_of_week = dt.weekday()
    day_name_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"][day_of_week]
    is_weekend = day_of_week >= 5
    day_of_year = dt.timetuple().tm_yday
    month = dt.month

    if day_of_week == 5:
        dow_factor = 0.85
        peak1_hour, peak2_hour = 11, 16
    elif day_of_week == 6:
        dow_factor = 0.60
        peak1_hour, peak2_hour = 12, 17
    else:
        dow_factor = 1.00
        peak1_hour, peak2_hour = 9, 15

    monthly_clim = {
        1: {"temp": 14.0, "wind": 16.0, "rain": 0.3},
        2: {"temp": 15.0, "wind": 17.0, "rain": 0.3},
        3: {"temp": 17.0, "wind": 16.0, "rain": 0.2},
        4: {"temp": 18.0, "wind": 15.0, "rain": 0.15},
        5: {"temp": 20.0, "wind": 14.0, "rain": 0.05},
        6: {"temp": 23.0, "wind": 13.0, "rain": 0.0},
        7: {"temp": 26.0, "wind": 15.0, "rain": 0.0},
        8: {"temp": 27.0, "wind": 16.0, "rain": 0.0},
        9: {"temp": 25.0, "wind": 14.0, "rain": 0.05},
        10: {"temp": 22.0, "wind": 14.5, "rain": 0.10},
        11: {"temp": 18.0, "wind": 15.5, "rain": 0.25},
        12: {"temp": 15.0, "wind": 16.0, "rain": 0.35},
    }
    clim = monthly_clim.get(month, {"temp": 21.0, "wind": 14.0, "rain": 0.1})

    return {
        "date_str": dt.strftime("%d/%m/%Y"),
        "date_obj": dt,
        "day_name_fr": day_name_fr,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "dow_factor": dow_factor,
        "peak1_hour": peak1_hour,
        "peak2_hour": peak2_hour,
        "month": month,
        "day_of_year": day_of_year,
        "temp_base": clim["temp"],
        "wind_base": clim["wind"],
        "rain_base": clim["rain"],
    }


def get_active_clients_info(date_str: str, total_clients: int) -> Dict[str, Any]:
    """
    Calcule la prévision du nombre total de commandes quotidiennes de manière illimitée.
    """
    date_info = parse_date_features(date_str)
    dow = date_info["day_of_week"]

    day_multipliers = [1.4, 1.2, 1.3, 1.5, 1.8, 1.1, 0.8]
    mult = day_multipliers[dow]

    month = date_info["month"]
    if month in [10, 11, 12]:
        mult *= 1.25
    elif month in [7, 8]:
        mult *= 0.85

    if date_info["rain_base"] > 0.2:
        mult *= 1.10

    h_val = int(hashlib.md5(date_str.encode('utf-8')).hexdigest(), 16) % 1000
    daily_delta = 1.0 + ((h_val / 1000.0) - 0.5) * 0.12

    base_nodes = max(50, total_clients)
    n_active = max(10, int(round(base_nodes * mult * daily_delta)))

    all_indices = list(range(1, max(base_nodes, n_active) + 1))
    rng = np.random.RandomState(h_val)
    if n_active <= len(all_indices):
        active_indices = sorted(rng.choice(all_indices, size=n_active, replace=False).tolist())
    else:
        active_indices = list(range(1, n_active + 1))

    active_demands = {0: 0}
    active_orders = {}
    active_parcels = {}

    for cid in range(1, max(base_nodes, n_active) + 1):
        if cid in active_indices:
            n_parcels = int(rng.randint(1, 13))
            ptype = 1 if (rng.rand() > 0.35) else 2
            urgency = 3 if (rng.rand() > 0.8) else 1
            tw_start = int(rng.choice([8, 9, 10, 11, 14]))
            tw_end = tw_start + int(rng.choice([3, 4, 5, 6]))

            active_demands[cid] = n_parcels
            active_parcels[cid] = n_parcels
            active_orders[cid] = {
                "client_id": cid,
                "parcels": n_parcels,
                "parcel_type": ptype,
                "urgency": urgency,
                "tw_start": tw_start,
                "tw_end": tw_end
            }
        else:
            active_demands[cid] = 0

    return {
        "total_clients": base_nodes,
        "active_count": n_active,
        "inactive_count": max(0, base_nodes - n_active),
        "active_indices": active_indices,
        "active_demands": active_demands,
        "active_orders": active_orders,
        "active_parcels": active_parcels,
        "activity_rate_pct": (n_active / base_nodes) * 100.0 if base_nodes > 0 else 0.0
    }


def run_stl_decomposition(
    series: Union[pd.Series, np.ndarray],
    period: int = 24,
    robust: bool = True,
    target_column: str = "demand",
    save_figures: bool = False,
    output_dir: str = "outputs/figures"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Exécute la décomposition STL (Seasonal and Trend decomposition using Loess).
    Optionnelle et dédiée à l'analyse exploratoire EDA et feature engineering ciblé.
    
    Renvoie :
    - trend    : composante de tendance [T]
    - seasonal : composante de saisonnalité [T]
    - residual : composante résiduelle [T]
    - stats    : métadonnées et statistiques de décomposition
    """
    logger.info(f"Exécution de la décomposition STL sur '{target_column}' (Période={period}, Robust={robust})...")
    s_arr = np.asarray(series, dtype=np.float64).flatten()
    T = len(s_arr)

    try:
        from statsmodels.tsa.seasonal import STL
        s_series = pd.Series(s_arr)
        stl_model = STL(s_series, period=period, robust=robust)
        res = stl_model.fit()
        trend = res.trend.values
        seasonal = res.seasonal.values
        residual = res.resid.values
        method_used = "statsmodels_STL"
    except Exception as e:
        logger.warning(f"Statsmodels STL indisponible ({e}). Utilisation du fallback lissage par moyenne glissante.")
        s_series = pd.Series(s_arr)
        trend = s_series.rolling(window=period, min_periods=1, center=True).mean().values
        detrended = s_arr - trend
        p_len = min(period, T)
        seasonal_pattern = np.mean(detrended[: (T // p_len) * p_len].reshape(-1, p_len), axis=0) if T >= p_len else np.zeros(p_len)
        seasonal = np.tile(seasonal_pattern, T // p_len + 1)[:T]
        residual = s_arr - trend - seasonal
        method_used = "moving_average_fallback"

    stats = {
        "target_column": target_column,
        "period": period,
        "robust": robust,
        "method_used": method_used,
        "trend_mean": float(np.mean(trend)),
        "seasonal_std": float(np.std(seasonal)),
        "residual_std": float(np.std(residual)),
        "residual_skew": float(pd.Series(residual).skew())
    }

    if save_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(output_dir, exist_ok=True)
            
            # Figure 1 : STL standard
            fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"STL Decomposition — {target_column} (Period={period})", fontsize=12, fontweight="bold")

            axes[0].plot(s_arr, label="Original", color="#38BDF8")
            axes[0].legend(loc="upper right")
            axes[1].plot(trend, label="Trend", color="#8B5CF6")
            axes[1].legend(loc="upper right")
            axes[2].plot(seasonal, label="Seasonal", color="#10B981")
            axes[2].legend(loc="upper right")
            axes[3].plot(residual, label="Residual", color="#F59E0B")
            axes[3].legend(loc="upper right")

            plt.tight_layout()
            suffix = "daily" if period == 24 else ("weekly" if period == 168 else f"p{period}")
            fig_path = os.path.join(output_dir, f"stl_decomposition_{suffix}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
            stats["fig_path"] = fig_path

            # Figure 2 : Série historique globale
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            ax_hist.plot(s_arr, color="#38BDF8", linewidth=1.5)
            ax_hist.set_title(f"Historique de Demande — {target_column}", fontsize=11, fontweight="bold")
            ax_hist.set_xlabel("Pas de temps (heures)")
            ax_hist.set_ylabel("Demande (colis)")
            plt.tight_layout()
            hist_path = os.path.join(output_dir, "demand_history.png")
            plt.savefig(hist_path, dpi=150)
            plt.close(fig_hist)

            # Figure 3 : Demande par heure (0-23)
            if T >= 24:
                hours_arr = np.tile(np.arange(24), T // 24 + 1)[:T]
                df_h = pd.DataFrame({"hour": hours_arr, "demand": s_arr})
                mean_by_h = df_h.groupby("hour")["demand"].mean()

                fig_h, ax_h = plt.subplots(figsize=(8, 4))
                ax_h.bar(mean_by_h.index, mean_by_h.values, color="#8B5CF6", alpha=0.85)
                ax_h.set_title("Demande Moyenne par Heure (00h - 23h)", fontsize=11, fontweight="bold")
                ax_h.set_xlabel("Heure de la journée")
                ax_h.set_ylabel("Demande Moyenne (colis)")
                plt.tight_layout()
                h_path = os.path.join(output_dir, "demand_by_hour.png")
                plt.savefig(h_path, dpi=150)
                plt.close(fig_h)

            logger.info(f"Figures STL sauvegardées sous : {output_dir}")
        except Exception as err:
            logger.warning(f"Impossible de sauvegarder les figures STL : {err}")

    return trend, seasonal, residual, stats


class MultivariateTimeSeriesIngestor:
    """
    Gestionnaire d'ingestion et de prétraitement pour séries temporelles multivariées.
    """
    @staticmethod
    def create_patches(X: np.ndarray, patch_len: int = 12, stride: int = 6, patch_length: Optional[int] = None, patch_stride: Optional[int] = None):
        from forecasting.patching import create_patches as _cp
        p_len = patch_length if patch_length is not None else patch_len
        p_stride = patch_stride if patch_stride is not None else stride
        return _cp(X, patch_length=p_len, patch_stride=p_stride)

    def __init__(self, time_steps: int = 24):
        self.time_steps = time_steps
        self.raw_df: Optional[pd.DataFrame] = None
        self.decomposed_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.target_date_info: Optional[Dict[str, Any]] = None

    def build_multivariate_matrix(
        self,
        clients_coords: List[Tuple[float, float]],
        orders: Dict[int, Any],
        weather_info: Dict[str, Any],
        traffic_penalties: Optional[List[float]] = None,
        target_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Construit la matrice temporelle multivariée X conditionnée par la date cible.
        """
        if not target_date:
            target_date = date.today().strftime("%d/%m/%Y")

        date_info = parse_date_features(target_date)
        self.target_date_info = date_info

        hours = np.arange(self.time_steps)
        n_clients = max(1, len(clients_coords) - 1)
        
        total_parcels = sum(orders.get(i, {}).get("parcels", 1) for i in range(1, n_clients + 1))
        dow_factor = date_info["dow_factor"]
        p1, p2 = date_info["peak1_hour"], date_info["peak2_hour"]

        base_demand = dow_factor * (
            total_parcels * 0.4 * np.exp(-((hours - p1) ** 2) / 8.0) +
            total_parcels * 0.5 * np.exp(-((hours - p2) ** 2) / 10.0) +
            np.random.normal(0, 1.0, size=self.time_steps)
        )
        demand = np.clip(base_demand, 0.0, None)

        base_wind = float(weather_info.get("wind_speed", date_info["wind_base"]))
        wind_profile = base_wind + 3.5 * np.sin(2 * np.pi * hours / 24.0) + np.random.normal(0, 0.5, size=self.time_steps)
        wind_profile = np.clip(wind_profile, 0.0, 50.0)

        base_rain = float(weather_info.get("rain", date_info["rain_base"]))
        rain_profile = np.maximum(0.0, base_rain + np.random.normal(0, 0.15, size=self.time_steps))

        mean_tp = float(np.mean(traffic_penalties)) if traffic_penalties else 1.2
        if date_info["is_weekend"]:
            traffic_profile = (
                mean_tp * 0.85 +
                0.3 * np.exp(-((hours - 12) ** 2) / 6.0) +
                0.4 * np.exp(-((hours - 18) ** 2) / 8.0) +
                np.random.normal(0, 0.03, size=self.time_steps)
            )
        else:
            traffic_profile = (
                mean_tp +
                0.5 * np.exp(-((hours - 8) ** 2) / 4.0) +
                0.6 * np.exp(-((hours - 17) ** 2) / 6.0) +
                np.random.normal(0, 0.05, size=self.time_steps)
            )
        traffic_profile = np.clip(traffic_profile, 1.0, 3.0)

        base_temp = float(weather_info.get("temp", date_info["temp_base"]))
        temp_profile = base_temp + 4.5 * np.sin(2 * np.pi * (hours - 6) / 24.0)

        start_date = pd.Timestamp(date_info["date_obj"])
        timestamps = pd.date_range(start=start_date, periods=self.time_steps, freq="h")

        df = pd.DataFrame({
            "timestamp": timestamps,
            "hour": hours,
            "demand": demand,
            "wind_speed": wind_profile,
            "rain": rain_profile,
            "traffic_index": traffic_profile,
            "temperature": temp_profile
        })

        self.raw_df = df
        return df

    def decompose_series(self, df: Optional[pd.DataFrame] = None, window_size: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Décompose chaque variable numérique de la série en : Tendance, Saisonnalité, Résidu.
        """
        if df is None:
            df = self.raw_df
        if df is None:
            raise ValueError("Aucune donnée multivariée disponible.")

        feature_cols = ["demand", "wind_speed", "rain", "traffic_index", "temperature"]
        decomposed = {}

        for col in feature_cols:
            if col not in df.columns:
                continue
            series = df[col].values
            trend, seasonal, residual, _ = run_stl_decomposition(series, period=min(12, len(series)), target_column=col)
            decomposed[col] = {
                "original": series,
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual
            }

        self.decomposed_data = decomposed
        return decomposed
