import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import skew, kurtosis
from typing import Dict, Any

logger = logging.getLogger("TimeSeriesStudy")

def analyze_time_series(df: pd.DataFrame, target_col: str = "demand", zone_col: str = "zone_id", timestamp_col: str = "timestamp"):
    """
    Perform a complete time series study phase before model training.
    """
    out_fig_dir = "outputs/time_series_analysis/figures"
    out_tab_dir = "outputs/time_series_analysis/tables"
    os.makedirs(out_fig_dir, exist_ok=True)
    os.makedirs(out_tab_dir, exist_ok=True)
    os.makedirs(f"{out_fig_dir}/reports", exist_ok=True)

    # Process time index
    if timestamp_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df = df.sort_values([zone_col, timestamp_col])
        df.set_index(timestamp_col, inplace=True)
    
    # Global Stats
    global_series = df.groupby(level=0)[target_col].sum()
    
    stats = {
        "mean": global_series.mean(),
        "median": global_series.median(),
        "std": global_series.std(),
        "variance": global_series.var(),
        "min": global_series.min(),
        "max": global_series.max(),
        "q1": global_series.quantile(0.25),
        "q3": global_series.quantile(0.75),
        "cv": global_series.std() / (global_series.mean() + 1e-9),
        "skewness": skew(global_series.dropna()),
        "kurtosis": kurtosis(global_series.dropna()),
        "zero_rate": (global_series == 0).mean()
    }
    
    pd.DataFrame([stats]).to_csv(f"{out_tab_dir}/global_stats.csv", index=False)
    
    # Zone Stats
    zone_stats = df.groupby(zone_col)[target_col].describe()
    zone_stats["zero_rate"] = df.groupby(zone_col)[target_col].apply(lambda x: (x == 0).mean())
    zone_stats.to_csv(f"{out_tab_dir}/zone_stats.csv")

    # Time stats
    if hasattr(global_series.index, "hour"):
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        hour_stats = df.groupby("hour")[target_col].describe()
        hour_stats.to_csv(f"{out_tab_dir}/hour_stats.csv")
        
        day_stats = df.groupby("day_of_week")[target_col].describe()
        day_stats.to_csv(f"{out_tab_dir}/day_stats.csv")
        
        weekend_stats = df.groupby("is_weekend")[target_col].describe()
        weekend_stats.to_csv(f"{out_tab_dir}/weekend_stats.csv")

    # STL Decomposition
    try:
        stl_daily = STL(global_series, period=24, robust=True).fit()
        plt.figure(figsize=(12, 8))
        stl_daily.plot()
        plt.title("STL Decomposition - Daily (24h)")
        plt.savefig(f"{out_fig_dir}/stl_global_daily.png")
        plt.close()

        stl_weekly = STL(global_series, period=168, robust=True).fit()
        plt.figure(figsize=(12, 8))
        stl_weekly.plot()
        plt.title("STL Decomposition - Weekly (168h)")
        plt.savefig(f"{out_fig_dir}/stl_global_weekly.png")
        plt.close()

        stl_components = pd.DataFrame({
            "trend_daily": stl_daily.trend,
            "seasonal_daily": stl_daily.seasonal,
            "resid_daily": stl_daily.resid,
            "trend_weekly": stl_weekly.trend,
            "seasonal_weekly": stl_weekly.seasonal,
            "resid_weekly": stl_weekly.resid
        })
        stl_components.to_csv(f"{out_tab_dir}/stl_components.csv")
    except Exception as e:
        logger.warning(f"STL decomposition failed: {e}")

    # ACF / PACF
    try:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(global_series.dropna(), lags=168, ax=ax[0])
        plot_pacf(global_series.dropna(), lags=168, ax=ax[1])
        plt.tight_layout()
        plt.savefig(f"{out_fig_dir}/acf_pacf.png")
        plt.close()
    except Exception as e:
        logger.warning(f"ACF/PACF plot failed: {e}")

    # Stationarity
    try:
        adf_res = adfuller(global_series.dropna())
        kpss_res = kpss(global_series.dropna(), nlags="auto")
        stationarity_stats = pd.DataFrame([
            {"Test": "ADF", "Stat": adf_res[0], "p-value": adf_res[1], "Stationary": adf_res[1] < 0.05},
            {"Test": "KPSS", "Stat": kpss_res[0], "p-value": kpss_res[1], "Stationary": kpss_res[1] > 0.05}
        ])
        stationarity_stats.to_csv(f"{out_tab_dir}/stationarity.csv", index=False)
    except Exception as e:
        logger.warning(f"Stationarity test failed: {e}")

    logger.info("Time series study completed and outputs saved.")
