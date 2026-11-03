import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import datetime

def delta_meteo(weather_val: float) -> float:
    """
    Multiplicateur lié à la météo.
    Exemple simple : une valeur de météo élevée (ex: pluie) augmente légèrement la demande (livraisons).
    """
    # Si weather_val représente la pluie en mm
    return 1.0 + 0.05 * weather_val

def phi_orsm(orsm_val: float) -> float:
    """
    Multiplicateur lié à l'indice ORSM (qualité de l'air ou trafic).
    Exemple : un mauvais indice (congestion ou pollution élevée) peut encourager les gens à commander.
    """
    # Si orsm_val = 1.0 (fluide), phi = 1.0. Si orsm_val = 1.8 (bouchon), phi augmente.
    return 1.0 + 0.1 * max(0.0, orsm_val - 1.0)

def compute_lambda(z: str, t: pd.Timestamp, config: Dict[str, Any], weather: float, orsm: float) -> float:
    """
    Calcule l'intensité de demande lambda_{z,t} pour un Processus de Poisson Non Homogène (NHPP).
    
    Formule: lambda_{z,t} = lambda_0 * alpha_z * beta_h(t) * gamma_d(t) * delta_meteo * phi_orsm
    
    Pourquoi un NHPP ?
    La modélisation par NHPP est le standard pour les systèmes d'arrivées dynamiques (VRP Stochastique/Dynamique).
    Elle modélise parfaitement la nature discrète (de comptage) des commandes tout en capturant les 
    variations temporelles continues de l'intensité d'arrivée.
    """
    lambda_0 = config.get('lambda_0', 10.0)
    
    # Récupération des poids
    alpha_z = config.get('spatial_weights', {}).get(z, 1.0)
    beta_h = config.get('hourly_weights', {}).get(t.hour, 1.0)
    gamma_d = config.get('daily_weights', {}).get(t.weekday(), 1.0)
    
    d_meteo = delta_meteo(weather)
    p_orsm = phi_orsm(orsm)
    
    lmbda = lambda_0 * alpha_z * beta_h * gamma_d * d_meteo * p_orsm
    return lmbda

def generate_demand(z: str, t: pd.Timestamp, config: Dict[str, Any], weather: float, orsm: float, distribution: str = "poisson") -> int:
    """
    Génère un entier N_{z,t} représentant le nombre de commandes pour une zone et un temps donnés.
    """
    lmbda = compute_lambda(z, t, config, weather, orsm)
    
    if distribution == "poisson" or "Poisson" in distribution:
        return np.random.poisson(lmbda)
    elif distribution == "negative_binomial" or "Binomiale Négative" in distribution or "Negative Binomial" in distribution:
        # N_{z,t} ~ NB(mu, k) où mu = lmbda et k est la dispersion
        k = config.get('negative_binomial', {}).get('k_dispersion', 2.0)
        if lmbda <= 0:
            return 0
        # numpy's negative binomial: n=k, p=k/(k+mu)
        p = k / (k + lmbda)
        return np.random.negative_binomial(k, p)
    else:
        raise ValueError(f"Distribution non supportée: {distribution}")

def generate_order_attributes(z: str, t: pd.Timestamp, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère les attributs détaillés pour une commande individuelle.
    """
    attr_cfg = config.get('order_attributes', {})
    
    # 1. request_id
    req_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"
    
    # 2. timestamp intra-intervalle (répartition uniforme dans l'heure)
    minutes_offset = np.random.randint(0, 60)
    seconds_offset = np.random.randint(0, 60)
    exact_time = t.replace(minute=minutes_offset, second=seconds_offset)
    
    # 3. coords (Génération simplifiée: centré autour d'un point arbitraire de la zone)
    # En réalité, on tirerait dans un polygone GeoJSON
    x = np.random.uniform(33.5, 33.6)
    y = np.random.uniform(-7.6, -7.5)
    
    # 4. quantity (Poisson)
    qty_lambda = attr_cfg.get('quantity', {}).get('lambda_qty', 2.5)
    qty = max(1, np.random.poisson(qty_lambda))
    
    # 5. weight & volume (Lognormale)
    w_mean = attr_cfg.get('weight', {}).get('mean', 1.5)
    w_sig = attr_cfg.get('weight', {}).get('sigma', 0.5)
    weight = np.random.lognormal(mean=np.log(w_mean), sigma=w_sig)
    
    v_mean = attr_cfg.get('volume', {}).get('mean', 0.05)
    v_sig = attr_cfg.get('volume', {}).get('sigma', 0.01)
    volume = np.random.lognormal(mean=np.log(v_mean), sigma=v_sig)
    
    # 6. perishable & time_window & priority
    p_perish = attr_cfg.get('perishable_probability', 0.15)
    is_perishable = bool(np.random.rand() < p_perish)
    
    std_window = attr_cfg.get('time_windows', {}).get('standard_duration_hours', 4.0)
    perish_window = attr_cfg.get('time_windows', {}).get('perishable_duration_hours', 1.0)
    
    window_duration = perish_window if is_perishable else std_window
    window_end = exact_time + datetime.timedelta(hours=window_duration)
    
    priority = 1 if is_perishable else 0
    
    return {
        "request_id": req_id,
        "timestamp": exact_time,
        "zone_id": z,
        "coords": (x, y),
        "quantity": qty,
        "weight": round(weight, 2),
        "volume": round(volume, 3),
        "is_perishable": is_perishable,
        "time_window_start": exact_time,
        "time_window_end": window_end,
        "priority": priority
    }

def generate_daily_demand(zones: List[str], time_slots: pd.DatetimeIndex, config: Dict[str, Any], weather_series: Dict[pd.Timestamp, float], orsm_series: Dict[pd.Timestamp, float], distribution: str = "poisson") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parcourt toutes les zones et intervalles pour générer la demande globale et les commandes détaillées.
    
    Retourne:
        - demand_df : DataFrame agrégé (zone, timestamp, demand)
        - orders_df : DataFrame détaillé des commandes individuelles
    """
    demand_records = []
    order_records = []
    
    for t in time_slots:
        w_val = weather_series.get(t, 0.0)
        o_val = orsm_series.get(t, 1.0)
        
        for z in zones:
            n_orders = generate_demand(z, t, config, w_val, o_val, distribution=distribution)
            
            demand_records.append({
                "zone_id": z,
                "timestamp": t,
                "demand": n_orders
            })
            
            # Générer les détails pour chaque commande
            for _ in range(n_orders):
                order_details = generate_order_attributes(z, t, config)
                order_records.append(order_details)
                
    demand_df = pd.DataFrame(demand_records)
    orders_df = pd.DataFrame(order_records)
    
    return demand_df, orders_df
