"""
===============================================================================
Module : Synthetic Multi-Zone Time Series Generator
File   : forecasting/generator.py
Description : Générateur stochastique réaliste de séries temporelles multivariées
              (Demande par zone, Météo Vent/Pluie, Congestion Trafic OSRM) sur n'importe
              quel horizon (ex. 1 An / 8760h, 1 Mois / 720h, 1 Semaine / 168h).
===============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger("ForecastingGenerator")


def generate_synthetic_demand_dataset(
    duration_hours: int = 8760,
    zones: Optional[List[str]] = None,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = "2026-07-01 00:00:00",
    timezone: str = "Africa/Casablanca",
    output_path: Optional[str] = "data/processed/demand_by_zone_hour.csv"
) -> pd.DataFrame:
    """
    Génère une série temporelle multivariée synthétique d'HISTORIQUE PASSÉ (dans le passé jusqu'à end_date_str).
    
    Paramètres:
    - duration_hours : Nombre total d'heures (ex: 26280h = 3 Ans, 8760h = 1 An, 720h = 1 Mois, 168h = 1 Semaine)
    - zones : Liste des identifiants de zones (par défaut ['zone_A', 'zone_B', 'zone_C'])
    - start_date_str : Date de début optionnelle (si non spécifié, calculé dans le passé depuis end_date_str)
    - end_date_str : Date de fin / point présent (par défaut '2026-07-01 00:00:00')
    - timezone : Fuseau horaire (Africa/Casablanca)
    - output_path : Chemin de sauvegarde CSV optionnel
    
    Retourne:
    - DataFrame pandas avec les colonnes : timestamp, zone_id, demand, wind_speed, rain, traffic_index, temperature
    """
    if zones is None:
        zones = ["zone_A", "zone_B", "zone_C"]

    # Si end_date_str est fourni sans start_date_str, l'historique part du passé pour aboutir au présent
    if end_date_str is not None and start_date_str is None:
        end_ts = pd.to_datetime(end_date_str)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(timezone)
        else:
            end_ts = end_ts.tz_convert(timezone)
        start_ts = end_ts - pd.Timedelta(hours=duration_hours)
    else:
        if start_date_str is None:
            start_date_str = "2023-07-01 00:00:00"
        start_ts = pd.to_datetime(start_date_str)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(timezone)
        else:
            start_ts = start_ts.tz_convert(timezone)

    logger.info(f"Génération automatique d'historique passé ({duration_hours}h, zones={zones}, du {start_ts} au {start_ts + pd.Timedelta(hours=duration_hours)})...")
    timestamps = pd.date_range(start=start_ts, periods=duration_hours, freq="h")

    # Climatologie mensuelle pour Casablanca
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

    records = []

    for z_idx, zone_id in enumerate(zones):
        zone_mult = 1.2 if z_idx == 0 else (0.9 if z_idx == 1 else 1.0)
        rng = np.random.RandomState(42 + z_idx)

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.weekday()
            month = ts.month
            day_of_year = ts.dayofyear

            # Multiplicateur quotidien (Weekend vs Semaine)
            dow_mult = 1.0
            if dow == 5: # Samedi
                dow_mult = 0.85
            elif dow == 6: # Dimanche
                dow_mult = 0.55
            elif dow == 4: # Vendredi pic de livraison
                dow_mult = 1.20

            # Motif journalier bimodal (pics 10h et 16h)
            p1_demand = 18.0 * np.exp(-((hour - 10) ** 2) / 6.0)
            p2_demand = 22.0 * np.exp(-((hour - 16) ** 2) / 8.0)
            base_hourly = (5.0 + p1_demand + p2_demand) * dow_mult * zone_mult

            # Tendance annuelle (légère croissance progressive + effet Q4 / Black Friday / Ramadan)
            annual_trend = 1.0 + 0.15 * (i / max(1, duration_hours))
            if month in [11, 12]: # Pic de fin d'année
                annual_trend *= 1.30
            elif month in [3, 4]: # Effet Ramadan (simulation)
                annual_trend *= 1.15

            # Bruit stochastique
            noise = rng.normal(0, 1.8)
            demand_val = max(0.0, (base_hourly * annual_trend) + noise)

            # Météo
            clim = monthly_clim.get(month, {"temp": 20.0, "wind": 15.0, "rain": 0.1})
            temp_val = clim["temp"] + 4.0 * np.sin(2 * np.pi * (hour - 6) / 24.0) + rng.normal(0, 0.5)
            wind_val = max(0.0, clim["wind"] + 3.0 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 1.0))
            rain_val = max(0.0, clim["rain"] * (1.5 if rng.rand() < 0.1 else 0.0) + rng.normal(0, 0.05))

            # Trafic OSRM Congestion
            if dow >= 5:
                traffic_val = 1.0 + 0.3 * np.exp(-((hour - 14) ** 2) / 10.0) + rng.normal(0, 0.03)
            else:
                traffic_val = 1.1 + 0.6 * np.exp(-((hour - 8) ** 2) / 4.0) + 0.7 * np.exp(-((hour - 17) ** 2) / 5.0) + rng.normal(0, 0.05)
            traffic_val = float(np.clip(traffic_val, 1.0, 3.0))

            # Nombre de colis strictement entier (processus stochastique de comptage discret)
            discrete_demand = int(max(0, round(demand_val)))
            if discrete_demand == 0:
                clients_count = 0
            else:
                clients_count = max(1, int(round(discrete_demand / rng.uniform(1.8, 2.4))))

            records.append({
                "timestamp": ts,
                "hour": hour,
                "zone_id": zone_id,
                "clients_count": clients_count,
                "demand": discrete_demand,
                "wind_speed": round(float(wind_val), 1),
                "rain": round(float(rain_val), 2),
                "traffic_index": round(traffic_val, 2),
                "temperature": round(float(temp_val), 1)
            })

    df_gen = pd.DataFrame(records)
    # Tri temporel puis spatial pour alterner naturellement les zones (zone_A, zone_B, zone_C)
    df_gen = df_gen.sort_values(by=["timestamp", "zone_id"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_gen.to_csv(output_path, index=False)
        logger.info(f"Série temporelle synthétique ({len(df_gen)} lignes) sauvegardée dans {output_path}")

    return df_gen


def generate_client_orders_from_demand(df_hourly: pd.DataFrame, max_orders: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Génère le journal individuel des commandes clients :
    Pour chaque client : ID Client, Horodatage exact (heure, minute, seconde), Zone, Colis commandés, Vecteur (VE/Drone), Urgence, Créneau.
    """
    rng = np.random.RandomState(seed)
    order_records = []
    client_counter = 1

    active_rows = df_hourly[df_hourly["demand"] > 0]
    if len(active_rows) > max_orders // 2:
        step = max(1, len(active_rows) // (max_orders // 2))
        rows_to_process = active_rows.iloc[::step]
    else:
        rows_to_process = active_rows

    for _, row in rows_to_process.iterrows():
        base_ts = row["timestamp"]
        zone_id = row["zone_id"]
        n_clients = max(1, int(row.get("clients_count", 1)))
        total_parcels = int(row["demand"])

        remaining_parcels = total_parcels
        for c_idx in range(n_clients):
            if c_idx == n_clients - 1:
                c_parcels = max(1, remaining_parcels)
            else:
                c_parcels = max(1, min(remaining_parcels - (n_clients - c_idx - 1), rng.randint(1, 4)))
                remaining_parcels -= c_parcels

            minute = rng.randint(0, 60)
            second = rng.randint(0, 60)
            try:
                exact_ts = base_ts.replace(minute=minute, second=second)
            except Exception:
                exact_ts = base_ts

            is_urgent = bool(rng.rand() < 0.22)
            if is_urgent and c_parcels <= 2 and rng.rand() < 0.65:
                vector = "🛸 Aérien (Drone)"
                urg_str = "🔥 Haute (Urgent)"
                tw_start = exact_ts.hour
                tw_end = min(23, tw_start + rng.randint(1, 3))
            else:
                vector = "⚡ Terrestre (VE)"
                urg_str = "📦 Standard (Normal)" if not is_urgent else "🔥 Haute (Urgent)"
                tw_start = exact_ts.hour
                tw_end = min(23, tw_start + rng.randint(3, 6))

            tw_str = f"{tw_start:02d}:00 - {tw_end:02d}:00"

            order_records.append({
                "timestamp": exact_ts,
                "order_id": f"CMD-{exact_ts.strftime('%Y%m%d')}-{client_counter:04d}",
                "client_id": f"Client #{client_counter}",
                "zone_id": zone_id,
                "parcels": c_parcels,
                "delivery_vector": vector,
                "urgency": urg_str,
                "time_window": tw_str
            })
            client_counter += 1

    df_orders = pd.DataFrame(order_records)
    if len(df_orders) > 0:
        df_orders = df_orders.sort_values(by=["timestamp"]).reset_index(drop=True)
    return df_orders
