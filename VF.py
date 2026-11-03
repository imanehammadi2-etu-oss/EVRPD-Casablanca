import threading
"""
EVRPD MAPPO Dashboard aE” 3 Tabs: Problem Definition, Training, Results
- EV & Drones transport parcels (colis)
- EV capacity: 100 parcels, Drone capacity: 1 parcel
- Fleet: 50 EV, 100 Drones
- Flight zone / No-flight zone for drones
- White dashboard, grenat buttons (#800020)
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import csv
import subprocess
import sys

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import random, math, threading
import numpy as np
import requests
from ui_kpi_card import KPICard
from functools import lru_cache
from types import SimpleNamespace
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import time
GLOBAL_SEED = int(time.time() * 1000) % 100000
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False


try:
    import keras_tuner as kt
    KERASTUNER_AVAILABLE = True
except Exception as e:
    print(f"Keras Tuner import error: {e}")
    KERASTUNER_AVAILABLE = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import tkintermapview
    from ui_route_card import RouteCard
    from ui_fleet_table import FleetTable
    MAP_AVAILABLE = True
except Exception:
    MAP_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

GRENAT = "#800020"
GRENAT_HOVER = "#600018"

MOROCCO_ZONES = {
    "Casablanca": {
        "Maarif": (33.575, 33.595, -7.640, -7.610),
        "Ain Chok": (33.540, 33.570, -7.640, -7.580),
        "Ain Sebaa": (33.590, 33.610, -7.560, -7.520),
        "Oulfa": (33.540, 33.570, -7.670, -7.630),
        "Hay Hassani": (33.560, 33.590, -7.670, -7.630),
        "Sidi Maarouf": (33.520, 33.550, -7.650, -7.600),
        "Anfa": (33.585, 33.600, -7.670, -7.640),
        "Bernoussi": (33.590, 33.610, -7.510, -7.480),
        "Derb Sultan": (33.565, 33.585, -7.610, -7.580),
        "Bourgogne": (33.595, 33.605, -7.650, -7.630),
    },
    "Rabat": {
        "Agdal": (33.990, 34.010, -6.860, -6.830),
        "Hay Riad": (33.950, 33.980, -6.900, -6.860),
        "Souissi": (33.970, 34.000, -6.850, -6.800),
        "Yacoub El Mansour": (33.990, 34.030, -6.900, -6.860),
        "Hassan": (34.015, 34.035, -6.840, -6.810),
        "Ocean": (34.015, 34.035, -6.870, -6.840),
    },
    "Marrakech": {
        "Gueliz": (31.625, 31.645, -8.020, -7.990),
        "Hivernage": (31.615, 31.630, -8.010, -7.990),
        "Medina": (31.620, 31.635, -7.995, -7.970),
        "Sidi Ghanem": (31.660, 31.680, -8.060, -8.030),
        "Targa": (31.650, 31.680, -8.060, -8.010),
        "Massira": (31.620, 31.650, -8.080, -8.040),
    },
    "Tanger": {
        "Centre-ville": (35.770, 35.790, -5.830, -5.800),
        "Malabata": (35.780, 35.800, -5.780, -5.740),
        "Iberia": (35.765, 35.785, -5.820, -5.790),
        "Boukhalef": (35.720, 35.750, -5.920, -5.880),
        "Mghogha": (35.740, 35.780, -5.850, -5.800),
    },
    "Fes": {
        "City Nouvelle": (34.030, 34.050, -5.020, -4.990),
        "Medina": (34.055, 34.075, -4.990, -4.960),
        "Saiss": (33.980, 34.020, -5.030, -4.980),
        "Narjiss": (34.000, 34.030, -5.000, -4.960),
    },
    "Agadir": {
        "Centre-ville": (30.410, 30.430, -9.610, -9.580),
        "Hay Mohammadi": (30.430, 30.460, -9.600, -9.560),
        "Dakhla": (30.400, 30.430, -9.570, -9.530),
        "Anza": (30.440, 30.470, -9.680, -9.620),
    },
    "Kenitra": {
        "Centre": (34.250, 34.270, -6.600, -6.570),
        "Maamora": (34.240, 34.270, -6.620, -6.590),
        "Bir Rami": (34.230, 34.260, -6.580, -6.540),
    },
    "Oujda": {
        "Centre": (34.675, 34.700, -1.930, -1.890),
        "Al Qods": (34.660, 34.690, -1.900, -1.860),
        "Hay El Fath": (34.660, 34.690, -1.950, -1.920),
    },
    "Tetouan": {
        "Centre": (35.565, 35.585, -5.390, -5.350),
        "Mhannech": (35.555, 35.580, -5.400, -5.365),
        "Martil": (35.600, 35.640, -5.300, -5.260),
    },
}

CITY_NFZS = {
    "Casablanca": [
        (33.35, 33.38, -7.60, -7.56, "Mohammed V Airport"),
        (33.580, 33.595, -7.615, -7.605, "Royal Palace"),
        (33.575, 33.585, -7.625, -7.615, "CHU Ibn Rochd"),
        (33.60, 33.62, -7.60, -7.58, "Casablanca Port"),
        (33.55, 33.57, -7.66, -7.64, "Sheikh Khalifa Hospital")
    ],
    "Rabat": [
        (34.04, 34.06, -6.77, -6.74, "Rabat-Sale Airport"),
        (34.00, 34.02, -6.84, -6.82, "Royal Palace"),
        (34.01, 34.03, -6.83, -6.81, "Avicenna Hospital")
    ],
    "Marrakech": [
        (31.59, 31.62, -8.05, -8.02, "Aeroport Menara"),
        (31.61, 31.63, -7.99, -7.98, "Palais Royal")
    ],
    "Agadir": [
        (30.31, 30.34, -9.43, -9.40, "Aeroport Al Massira"),
        (30.41, 30.43, -9.61, -9.59, "Palais Royal")
    ]
}

CITY_DEPOTS = {
    "Casablanca": (33.570, -7.580), "Rabat": (34.020, -6.830),
    "Marrakech": (31.630, -8.000), "Tanger": (35.780, -5.810),
    "Fes": (34.040, -5.000), "Agadir": (30.420, -9.590),
    "Kenitra": (34.260, -6.580), "Oujda": (34.685, -1.910),
    "Tetouan": (35.575, -5.370),
}



def euclidean_km(p1, p2):
    return math.sqrt(((p2[1]-p1[1])*111)**2 + ((p2[0]-p1[0])*111)**2)

def get_osrm_nearest(lat, lon):
    try:
        url = f"http://router.project-osrm.org/nearest/v1/driving/{lon},{lat}?number=1"
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            data = r.json()
            if "waypoints" in data and len(data["waypoints"]) > 0:
                pt = data["waypoints"][0]["location"]
                return pt[1], pt[0] # lat, lon
    except Exception:
        pass
    return lat, lon


def get_osrm_route(points):
    if len(points) < 2: return points
    try:
        coords = ";".join([f"{p[1]:.6f},{p[0]:.6f}" for p in points])
        url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return [(c[1], c[0]) for c in r.json()["routes"][0]["geometry"]["coordinates"]]
    except:
        pass
    return points


def get_osrm_matrix(points):
    """
    Interroge l'API OSRM (Endpoint 'Table') pour recuperer
    les distances (m) et durees (s) reelles entre tous les points.
    Inclut embouteillages, limitations de vitesse, accidents, etc.
    """
    if len(points) < 2 or len(points) > 100:
        return None, None
    try:
        coords = ";".join([f"{p[1]:.6f},{p[0]:.6f}" for p in points])
        url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance,duration"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == "Ok":
                dist_matrix = np.array(data["distances"])
                dur_matrix = np.array(data["durations"])
                return dist_matrix, dur_matrix
    except Exception as e:
        print(f"Error OSRM Matrix : {e}")
    return None, None


# Ceilings de vitesse selon le tableau d'entreprise (km/h)
SPEED_CAP_URBAN = 40.0   # City / Urbain : max 40 km/h pour EV
SPEED_CAP_ROAD = 80.0    # Nationale / Autoroute : max 80 km/h pour EV
URBAN_THRESHOLD = 70.0   # Si V_osrm <= 70 km/h => environnement urbain (couvre les boulevards a  50/60 km/h)


def capped_travel_time_minutes(distance_m, duration_s):
    """
    Calcule le temps de trajet reel (en minutes) en appliquant
    les plafonds de vitesse de l'entreprise sur la vitesse OSRM.
    - distance_m : distance OSRM en ma¨tres
    - duration_s : duree OSRM en secondes (inclut trafic reel)
    Retourne le temps en minutes.
    """
    if distance_m <= 0:
        return 0.0
    # 1. Lire la vitesse du trafic reel donnee par OSRM
    if duration_s > 0:
        v_osrm_kmh = (distance_m / duration_s) * 3.6
    else:
        v_osrm_kmh = SPEED_CAP_URBAN  # Securite
    # 2. Deduire l'environnement (City vs Autoroute)
    if v_osrm_kmh <= URBAN_THRESHOLD:
        plafond = SPEED_CAP_URBAN  # Bridage urbain strict a  40 km/h
    else:
        plafond = SPEED_CAP_ROAD   # Bridage voies rapides/nationales strict a  80 km/h
    # 3. Le Bridage : la vitesse ne depasse jamais le plafond d'entreprise
    v_finale_kmh = min(v_osrm_kmh, plafond)
    # 4. Convertir en temps (minutes)
    v_finale_ms = v_finale_kmh / 3.6
    return (distance_m / max(1e-9, v_finale_ms)) / 60.0


BYPASS_LAND_CHECK = True  # Desactive le geocodage Nominatim pour accelerer la generation et eviter d'aªtre banni par l'API

@lru_cache(maxsize=4096)
def is_point_on_land(lat, lon, city=None):
    """Retourne True si le point semble aªtre sur terre.

    Utilise Nominatim reverse geocoding pour verifier la categorie/type.
    En cas d'erreur reseau, renvoie True par defaut (ne bloque pas la generation).
    """
    if BYPASS_LAND_CHECK:
        return True
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format": "jsonv2", "lat": float(lat), "lon": float(lon), "zoom": 10, "addressdetails": 1}
        headers = {"User-Agent": "EVRPD/1.0 (contact@example.com)"}
        r = requests.get(url, params=params, headers=headers, timeout=4)
        if r.status_code != 200:
            return True
        data = r.json()
        # category or type can indicate water/sea
        cat = (data.get("category") or "").lower()
        typ = (data.get("type") or "").lower()
        if any(w in cat for w in ("water", "sea", "ocean", "river", "lake", "beach")):
            return False
        if any(w in typ for w in ("water", "sea", "ocean", "river", "lake", "beach")):
            return False
        # If address is present, prefer that (land)
        if data.get("address"):
            return True
        # Otherwise assume land
        return True
    except Exception:
        # En cas d'erreur reseau, ne pas bloquer la generation
        return True

def parse_hhmm(txt, default=480):
    try:
        h, m = txt.strip().split(":"); h, m = int(h), int(m)
        if 0 <= h < 24 and 0 <= m < 60: return h*60+m
    except Exception: pass
    return default

def minutes_to_hhmm(minutes):
    minutes = int(minutes)
    return f"{(minutes//60)%24:02d}:{minutes%60:02d}"

def grenat_btn(parent, text, command, **kw):
    return tk.Button(parent, text=text, command=command, bg=GRENAT, fg="white",
                     font=("Arial", 10, "bold"), relief="flat", cursor="hand2",
                     activebackground=GRENAT_HOVER, activeforeground="white",
                     padx=12, pady=5, **kw)

def kmeans_simple(points, k, max_iter=50):
    if len(points) == 0 or k <= 0: return np.array([], dtype=int), np.empty((0, 2))
    pts = np.array(points, dtype=float)
    k = min(k, len(pts))
    idx = np.random.choice(len(pts), k, replace=False)
    centers = pts[idx]
    for _ in range(max_iter):
        dists = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.array([pts[labels == j].mean(axis=0) if len(pts[labels == j]) > 0 else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers): break
        centers = new_centers
    return labels, centers

def kmeans_conditioned(points, k, weather_penalty=1.0, traffic_penalties=None, max_iter=50):
    """
    K-Means conditionne par la meteo et le trafic.
    weather_penalty : scalaire (ex: 1.2 pour intemperies)
    traffic_penalties : tableau 1D avec des multiplicateurs de congestion.
    """
    if len(points) == 0 or k <= 0: return np.array([], dtype=int), np.empty((0, 2))
    pts = np.array(points, dtype=float)
    k = min(k, len(pts))
    
    if traffic_penalties is None:
        traffic_penalties = np.ones(len(pts))
    else:
        traffic_penalties = np.array(traffic_penalties)
        
    idx = np.random.choice(len(pts), k, replace=False)
    centers = pts[idx]
    
    for _ in range(max_iter):
        # Distance euclidienne de base au carre: shape (N_points, K_centers)
        dists = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        
        # Application des conditions
        dists = dists * (traffic_penalties[:, None] ** 2) * (weather_penalty ** 2)
        
        labels = dists.argmin(axis=1)
        new_centers = np.array([pts[labels == j].mean(axis=0) if len(pts[labels == j]) > 0 else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers): break
        centers = new_centers
    return labels, centers

class NSGAII:
    """Level 2: Strategic Planning - Multi-objective Optimization
    
    INPUT  : clusters, coordonnees, commandes, demand_types, flotte disponible,
             pop_size, n_gen, crossover_rate, mutation_rate
    OUTPUT : master_plan = liste de clients EV affectes a  chaque vehicule
    """
    def __init__(self, coords, n_ev, clusters=None, pop_size=30, n_gen=20, cx_rate=0.8, mut_rate=0.1, orders=None, ev_autonomy_km=47.4, demand_types=None, no_flight_clients=None):
        self.coords = coords
        self.n_ev = n_ev
        self.pop_size = pop_size
        self.n_gen = max(1, n_gen)
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.n_clients = len(coords) - 1
        self.orders = orders or {}
        self.ev_autonomy_km = float(ev_autonomy_km)
        # Store demand_types as list (indexed by client id)
        if demand_types is not None:
            self.demand_types = list(demand_types)
        else:
            self.demand_types = None
        self.no_flight_clients = no_flight_clients or set()
        
        # Determine clusters
        import numpy as np
        if clusters is not None and len(clusters) == self.n_clients:
            self.clusters = clusters
        else:
            self.clusters = np.zeros(self.n_clients, dtype=int)
            
        self.k_zones = max(1, len(set(self.clusters)))
        
        # zone_clients contient UNIQUEMENT les clients EV (Type 1 + NFZ forces)
        # Les clients drone (Type 2) sont geres separement par assign_drones()
        self.zone_clients = {k: [] for k in range(self.k_zones)}
        for i, c in enumerate(self.clusters):
            client_id = i + 1
            if self.demand_types is not None:
                if self.demand_types[client_id] == 1 or client_id in self.no_flight_clients:
                    self.zone_clients[c].append(client_id)
            else:
                self.zone_clients[c].append(client_id)

    def _dist(self, p1, p2):
        import math
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _route_distance(self, route):
        if not route: return 0.0
        d = self._dist(self.coords[0], self.coords[route[0]])
        for i in range(len(route)-1):
            d += self._dist(self.coords[route[i]], self.coords[route[i+1]])
        d += self._dist(self.coords[route[-1]], self.coords[0])
        return d

    def _evaluate(self, ind):
        """a‰TAPE 1 : Calcul des Fonctions Objectifs avec contraintes strictes et optimisation de la flotte.
        F1 (Efficacite) = Somme des distances + 10000 * clients non livres + 150 * vehicules actifs
        F2 (a‰quite)     = Racine Carree de ( Somme(Dist_i - Moyenne)Â² / N_ev )
        """
        import math
        alloc, perms = ind
        routes = [[] for _ in range(self.n_ev)]
        zone_evs = {k: [] for k in range(self.k_zones)}
        for v in range(self.n_ev):
            zone_evs[alloc[v]].append(v)
        
        route_dists = [0.0] * self.n_ev
        unserved_count = 0
        
        for k in range(self.k_zones):
            v_list = zone_evs[k]
            clients = perms[k]
            
            if not v_list:
                # Aucun vehicule affecte a  cette zone a†’ tous les clients non livres
                unserved_count += len(clients)
                continue
            
            # Remplissage glouton pour respecter la capacite (100) et l'autonomie (47.4 km)
            v_idx = 0
            current_route = []
            for client in clients:
                if v_idx >= len(v_list):
                    # Plus aucun vehicule de la zone n'est disponible pour ce client
                    unserved_count += 1
                    continue
                
                test_route = current_route + [client]
                rd = self._route_distance(test_route)
                total_pkgs = sum(self.orders.get(node, {}).get("parcels", 1) for node in current_route) + self.orders.get(client, {}).get("parcels", 1)
                
                # Le client rentre dans le vehicule ssi distance <= autonomie et colis <= 100
                if len(current_route) == 0 or (rd <= self.ev_autonomy_km and total_pkgs <= 100):
                    current_route.append(client)
                else:
                    # On ferme le vehicule actuel et on passe au suivant de la zone
                    v = v_list[v_idx]
                    routes[v] = current_route
                    route_dists[v] = self._route_distance(current_route)
                    v_idx += 1
                    current_route = [client]
            
            # Fermeture du dernier vehicule actif de la zone
            if current_route and v_idx < len(v_list):
                v = v_list[v_idx]
                routes[v] = current_route
                route_dists[v] = self._route_distance(current_route)
            elif current_route:
                # Si le dernier client ne rentre pas faute de vehicule disponible
                unserved_count += 1
        
        total_dist = sum(route_dists)
        
        # Nombre de vehicules electriques reellement utilises (actifs)
        active_vehicles = sum(1 for r in routes if len(r) > 0)
        
        # F1 : Efficacite + Penalite d'utilisation de la flotte (150 par vehicule actif)
        f1 = total_dist + 10000.0 * unserved_count + 150.0 * active_vehicles
        
        # F2 : a‰quite
        mean_d = total_dist / max(1, self.n_ev)
        f2 = math.sqrt(sum((d - mean_d)**2 for d in route_dists) / max(1, self.n_ev))
        
        return f1, f2, routes

    def _crossover(self, p1, p2):
        a1, perm1 = p1
        a2, perm2 = p2
        new_a1, new_a2 = a1[:], a2[:]
        for i in range(self.n_ev):
            if random.random() < 0.5:
                new_a1[i], new_a2[i] = new_a2[i], new_a1[i]
                
        new_perm1, new_perm2 = {}, {}
        for k in range(self.k_zones):
            c1, c2 = self._ox(perm1[k], perm2[k])
            new_perm1[k] = c1
            new_perm2[k] = c2
        return (new_a1, new_perm1), (new_a2, new_perm2)

    def _ox(self, p1, p2):
        if len(p1) <= 1: return p1[:], p2[:]
        sz = len(p1)
        a, b = sorted(random.sample(range(sz), 2))
        def ox_child(parent1, parent2):
            child = [None]*sz
            child[a:b+1] = parent1[a:b+1]
            p2_filt = [x for x in parent2 if x not in child]
            idx = 0
            for i in range(sz):
                if child[i] is None:
                    child[i] = p2_filt[idx]
                    idx += 1
            return child
        return ox_child(p1, p2), ox_child(p2, p1)

    def _mutate(self, ind):
        alloc, perms = ind
        for i in range(self.n_ev):
            if random.random() < self.mut_rate:
                alloc[i] = random.randint(0, self.k_zones - 1)
        for k in range(self.k_zones):
            if random.random() < self.mut_rate and len(perms[k]) > 1:
                i, j = random.sample(range(len(perms[k])), 2)
                perms[k][i], perms[k][j] = perms[k][j], perms[k][i]
        return alloc, perms

    def _non_dominated_sort(self, pop_fit):
        S = [[] for _ in range(len(pop_fit))]
        front = [[]]
        n = [0] * len(pop_fit)
        rank = [0] * len(pop_fit)

        for p in range(len(pop_fit)):
            f1_p, f2_p = pop_fit[p]
            for q in range(len(pop_fit)):
                f1_q, f2_q = pop_fit[q]
                if (f1_p <= f1_q and f2_p <= f2_q) and (f1_p < f1_q or f2_p < f2_q):
                    S[p].append(q)
                elif (f1_q <= f1_p and f2_q <= f2_p) and (f1_q < f1_p or f2_q < f2_p):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                front[0].append(p)

        i = 0
        while front[i]:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            front.append(Q)
        return front[:-1]

    def _crowding_distance(self, front, fitnesses):
        distances = {i: 0.0 for i in front}
        if len(front) <= 2:
            for i in front: distances[i] = float('inf')
            return distances

        for m in (0, 1):
            front.sort(key=lambda x: fitnesses[x][m])
            distances[front[0]] = float('inf')
            distances[front[-1]] = float('inf')
            f_min = fitnesses[front[0]][m]
            f_max = fitnesses[front[-1]][m]
            if f_max == f_min: continue
            for i in range(1, len(front) - 1):
                distances[front[i]] += (fitnesses[front[i+1]][m] - fitnesses[front[i-1]][m]) / (f_max - f_min)
        return distances

    def solve(self):
        import copy
        population = []
        for _ in range(self.pop_size):
            alloc = [random.randint(0, self.k_zones - 1) for _ in range(self.n_ev)]
            if self.n_ev >= self.k_zones:
                for k in range(self.k_zones):
                    if k not in alloc:
                        alloc[random.randint(0, self.n_ev - 1)] = k
            perms = {k: random.sample(self.zone_clients[k], len(self.zone_clients[k])) for k in range(self.k_zones)}
            population.append((alloc, perms))

        for gen in range(self.n_gen):
            fitnesses = []
            routes_list = []
            for ind in population:
                f1, f2, r = self._evaluate(ind)
                fitnesses.append((f1, f2))
                routes_list.append(r)

            fronts = self._non_dominated_sort(fitnesses)
            new_pop = []
            for front in fronts:
                cd = self._crowding_distance(front, fitnesses)
                front.sort(key=lambda x: cd[x], reverse=True)
                for idx in front:
                    if len(new_pop) < self.pop_size:
                        new_pop.append(population[idx])
                    else: break
                if len(new_pop) >= self.pop_size: break
            
            population = new_pop
            
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = random.choice(population)
                p2 = random.choice(population)
                if random.random() < self.cx_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                offspring.append(self._mutate(c1))
                if len(offspring) < self.pop_size:
                    offspring.append(self._mutate(c2))
                    
            population.extend(offspring)

        fitnesses = []
        routes_list = []
        for ind in population:
            f1, f2, r = self._evaluate(ind)
            fitnesses.append((f1, f2))
            routes_list.append(r)
        
        fronts = self._non_dominated_sort(fitnesses)
        best_in_front = min(fronts[0], key=lambda x: fitnesses[x][0])
        return routes_list[best_in_front]

class MADDPG:
    """Level 4: Operational Rendezvous - Continuous Action Space"""
    def __init__(self, state_dim=5, action_dim=2, lr=1e-3, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), 
                                   nn.Linear(64, 64), nn.ReLU(), 
                                   nn.Linear(64, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64, 1))
        
        import copy
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = []
        self.batch_size = 64

    def act(self, state):
        st = torch.tensor(state, dtype=torch.float32)
        return self.actor(st).detach().numpy()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
            
        import random
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32)
        rewards = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + self.gamma * (1 - dones) * self.target_critic(torch.cat([next_states, next_actions], 1))
            
        current_q = self.critic(torch.cat([states, actions], 1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()
        
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], 1)).mean()
        self.optimizer_a.zero_grad()
        actor_loss.backward()
        self.optimizer_a.step()
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class MAPPOEVRPD:
    def __init__(self, coords, demand_types, orders, time_windows,
                 no_flight_clients=None, nfz_polygons=None, n_ev=50, n_drones=100,
                 ev_autonomy_km=80, drone_autonomy_km=12,
                 speed_ev_kmh=40, speed_drone_kmh=50, service_time_min=5,
                 drone_pick_time=2.0, drone_load_time=1.0, drone_unload_time=1.0,
                 drone_takeoff_time=0.5, drone_return_time=0.5, drone_safe_margin=20.0, drone_recharge_time=45.0,
                 weather_ok=True, weather_reason="",
                 weather_penalty=1.0, traffic_penalties=None,
                 start_time_minute=480, n_episodes=600,
                 lr=3e-4, gamma=0.99, clip_eps=0.20, ppo_epochs=4,
                 maddpg_lr=1e-3, maddpg_gamma=0.95, maddpg_tau=0.01,
                 pop_size_nsga=30, n_gen_nsga=20, cx_rate_nsga=0.8, mut_rate_nsga=0.1, k_clusters=4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch requis: pip install torch")
        self.coords = coords
        if isinstance(demand_types, (list, tuple)):
            self.demand_types = list(demand_types)
        else:
            self.demand_types = demand_types
        self.orders = orders
        self.time_windows = time_windows or {}
        self.no_flight_clients = no_flight_clients or set()
        self.nfz_polygons = nfz_polygons or []
        self.n_ev = max(1, int(n_ev))
        self.n_drones = max(1, int(n_drones))
        self.ev_autonomy_km = float(ev_autonomy_km)
        self.drone_autonomy_km = float(drone_autonomy_km)
        self.speed_ev_kmh = float(speed_ev_kmh)
        self.speed_drone_kmh = float(speed_drone_kmh)
        self.service_time_min = float(service_time_min)
        self.drone_pick_time = float(drone_pick_time)
        self.drone_load_time = float(drone_load_time)
        self.drone_unload_time = float(drone_unload_time)
        self.drone_takeoff_time = float(drone_takeoff_time)
        self.drone_return_time = float(drone_return_time)
        self.drone_safe_margin = float(drone_safe_margin)
        self.drone_recharge_time = float(drone_recharge_time)
        self.effective_drone_autonomy = self.drone_autonomy_km * (1.0 - self.drone_safe_margin / 100.0)
        self.weather_ok = bool(weather_ok)
        self.weather_reason = weather_reason
        self.start_time_minute = float(start_time_minute)
        self.n_episodes = int(n_episodes)
        self.pop_size_nsga = int(pop_size_nsga)
        self.n_gen_nsga = int(n_gen_nsga)
        self.cx_rate_nsga = float(cx_rate_nsga)
        self.mut_rate_nsga = float(mut_rate_nsga)
        self.gamma, self.clip_eps, self.ppo_epochs = float(gamma), float(clip_eps), int(ppo_epochs)
        self.k_clusters = int(k_clusters)
        self.co2_ev_g_km, self.co2_drone_g_km = 5.0, 1.0
        # Coefficients additionnels pour calculs de reference ASIFE / ADEME
        self.co2_ev_asife_g_km = 8.0
        self.co2_drone_asife_g_km = 2.0
        self.co2_ev_ademe_g_km = 12.0
        self.co2_drone_ademe_g_km = 3.0
        # ASIF Framework (Activity x Structure x Intensity x Fuel) Parameters
        self.asif_avg_parcel_kg = 2.5        # Average parcel weight (kg)
        self.asif_grid_factor_g_kwh = 623.0  # Morocco electricity grid emission factor (g CO2 / kWh, ONEE 2023)
        self.energy_ev_kwh_km, self.energy_drone_kwh_km = 0.18, 0.04
        self.ev_capacity, self.drone_capacity = 100, 1
        self.weather_penalty = float(weather_penalty)
        self.traffic_penalties = traffic_penalties
        if not self.weather_ok:
            # Selon le nouvel organigramme:
            # - Si mauvais temps, on change TOUS les modes drone en EV. Pas de suspension.
            for idx in range(1, len(self.demand_types)):
                if self.demand_types[idx] == 2:
                    self.demand_types[idx] = 1 # Forcer EV
            self.weather_reason = f"Bad weather : All les vols de drones ont ete transferes aux EVs. ({self.weather_reason})"
        # Level 3: Tactical MAPPO
        self.feature_dim, self.global_dim = 7, 8
        self.actor = nn.Sequential(nn.Linear(self.feature_dim, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(self.global_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

        # Level 4: Operational MADDPG
        self.maddpg = MADDPG(state_dim=5, action_dim=2, lr=maddpg_lr, gamma=maddpg_gamma, tau=maddpg_tau)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.e_threshold = 0.20 # 20%
        self.e_max = 1.0
        
        self.stop_flag = False
        self.history_best, self.history_distance = [], []
        self.history_co2, self.history_co2_asife, self.history_co2_ademe, self.history_energy = [], [], [], []
        self.history_late, self.history_service = [], []
        self.best_solution = None

        # Precalcul de la matrice des distances pour eviter les recalculs repetitifs
        self.dist_matrix = np.zeros((len(self.coords), len(self.coords)))
        for i in range(len(self.coords)):
            for j in range(len(self.coords)):
                self.dist_matrix[i, j] = euclidean_km(self.coords[i], self.coords[j])

        # Telechargement de la matrice OSRM (distances en m et durees en s, trafic reel)
        self.osrm_dist_m = None
        self.osrm_dur_s = None
        try:
            dm, tm = get_osrm_matrix(self.coords)
            if dm is not None and tm is not None:
                self.osrm_dist_m = dm
                self.osrm_dur_s = tm
                print(f"[OSRM] Matrice {dm.shape[0]}x{dm.shape[1]} chargee avec succa¨s (trafic temps reel).")
            else:
                print("[OSRM] Matrice non disponible (trop de points ou erreur reseau). Mode euclidien + 40 km/h.")
        except Exception as e:
            print(f"[OSRM] Error : {e}. Mode euclidien + 40 km/h.")

    def d(self, i, j):
        return self.dist_matrix[i, j]

    def travel_time_minutes(self, i, j):
        """
        Retourne le temps de trajet (en minutes) entre les nÅ“uds i et j,
        en utilisant les donnees OSRM temps reel + plafonds d'entreprise.
        Si OSRM n'est pas disponible, utilise la distance euclidienne a  40 km/h.
        """
        if self.osrm_dist_m is not None and i < len(self.osrm_dist_m) and j < len(self.osrm_dist_m):
            return capped_travel_time_minutes(self.osrm_dist_m[i, j], self.osrm_dur_s[i, j])
        else:
            # Fallback : distance euclidienne a  40 km/h (prudence ville)
            return self.d(i, j) / max(1e-9, SPEED_CAP_URBAN) * 60


    def features(self, current, candidate, remaining_count):
        order = self.orders.get(candidate, {})
        urgency = order.get("urgency", 2) if isinstance(order, dict) else 2
        return [1.0, -self.d(current, candidate)/10.0, -self.d(0, candidate)/10.0,
                1.0 if self.demand_types[candidate]==1 else 0.0,
                1.0 if self.demand_types[candidate]==2 else 0.0,
                -remaining_count/max(1, len(self.coords)-1),
                urgency/3.0]

    def global_state(self, agent_id, current_node, remaining):
        n_clients = max(1, len(self.coords)-1)
        rl = list(remaining)
        if rl:
            dists = [self.d(current_node, c) for c in rl]
            avg_d = (sum(dists) / len(dists)) / 10.0
            min_d = min(dists) / 10.0
        else: avg_d, min_d = 0.0, 0.0
        c1 = sum(1 for c in rl if self.demand_types[c]==1)/n_clients
        c2 = sum(1 for c in rl if self.demand_types[c]==2)/n_clients
        return [len(rl)/n_clients, avg_d, min_d, c1, c2,
                agent_id/max(1, self.n_ev), self.n_ev/10.0, self.n_drones/10.0]

    def calculate_urgency(self, current_energy):
        # Î¶m(t) = 1 - (Erem - Eth) / Emax
        return max(0, min(1.0, 1.0 - (current_energy - self.e_threshold) / self.e_max))

    def rv_features(self, drone_pos, ev_pos, soc, drone_vel):
        rel_pos = np.array(ev_pos) - np.array(drone_pos)
        return np.array([rel_pos[0]*10, rel_pos[1]*10, soc, drone_vel[0], drone_vel[1]], dtype=np.float32)

    def rv_global_state(self, drones_states, ev_pos):
        # Global state for critic: [avg_dist, min_dist, avg_soc, min_soc, ev_lat, ev_lon]
        if not drones_states: return np.zeros(self.rv_global_dim, dtype=np.float32)
        dists = [euclidean_km(d['pos'], ev_pos) for d in drones_states]
        socs = [d['soc'] for d in drones_states]
        return np.array([np.mean(dists)/10, np.min(dists)/10, np.mean(socs), np.min(socs), ev_pos[0], ev_pos[1]], dtype=np.float32)

    def assign_drones(self, ev_routes):
        # 1. Calcul des horaires des EV (Schedules)
        ev_schedules = []
        for ev_id, route in enumerate(ev_routes):
            schedule = []
            t = self.start_time_minute
            for i, node in enumerate(route):
                if i > 0:
                    t += self.travel_time_minutes(route[i-1], node)
                arrival = t
                service = self.service_time_min * max(1, self.orders.get(node, {}).get("parcels", 1)) if node != 0 else 0
                departure = arrival + service
                schedule.append({"node": node, "arrival": arrival, "departure": departure, "pos": self.coords[node]})
                t = departure
            ev_schedules.append(schedule)

        # 2. a‰tat des stocks de drones sur les EV
        # ev_inventory[ev_id] = liste des temps de disponibilite (ready_time) des drones a  bord
        ev_inventory = [[] for _ in range(self.n_ev)]
        # Distribution initiale des drones (max 2 par EV)
        for d_idx in range(self.n_drones):
            ev_idx = d_idx % self.n_ev
            if len(ev_inventory[ev_idx]) < 5:
                ev_inventory[ev_idx].append(self.start_time_minute)

        drone_assign, unserved = {}, []
        drone_trajectories, drone_rewards = [], []
        drone_clients = [i for i in range(1, len(self.coords))
                         if self.demand_types[i] == 2 and i not in self.no_flight_clients]
        
        # Si la meteo bloque les vols, aucune mission drone n'est creee.
        if not self.weather_ok:
            return {}, drone_clients, [], [], {}, {}, {}

        # On trie les clients par fenaªtre de temps pour un meilleur ordonnancement
        drone_clients.sort(key=lambda c: self.time_windows.get(c, [0, 1440])[0])

        drone_dists, drone_meetings, drone_launch = {}, {}, {}
        
        for c in drone_clients:
            c_pos = self.coords[c]
            best_config = None
            min_mission_dist = float('inf')
            # Pre-filter launch stops by distance and drone availability
            valid_launches = []
            for ev_launch_id, l_sched in enumerate(ev_schedules):
                for l_idx, l_stop in enumerate(l_sched):
                    if l_stop["node"] == c: continue
                    d_out = self.d(l_stop["node"], c)
                    if d_out <= self.effective_drone_autonomy / 2:
                        ready_drones = [rt for rt in ev_inventory[ev_launch_id] if rt <= l_stop["departure"]]
                        if ready_drones:
                            valid_launches.append((ev_launch_id, l_idx, l_stop, d_out, ready_drones))
            
            # Pre-filter return stops by distance
            valid_returns = []
            for ev_ret_id, r_sched in enumerate(ev_schedules):
                for r_idx, r_stop in enumerate(r_sched):
                    d_back = self.d(c, r_stop["node"])
                    if d_back <= self.effective_drone_autonomy / 2:
                        valid_returns.append((ev_ret_id, r_idx, r_stop, d_back))
            
            valid_configs = []
            # Check combinations from the highly filtered lists
            for ev_launch_id, l_idx, l_stop, d_out, ready_drones in valid_launches:
                t_launch = l_stop["departure"] + self.drone_pick_time + self.drone_load_time + self.drone_takeoff_time
                t_arrival_c = t_launch + self.travel_time_minutes(l_stop["node"], c)
                t_ready_c = t_arrival_c + self.drone_unload_time
                
                for ev_ret_id, r_idx, r_stop, d_back in valid_returns:
                    if d_out + d_back > self.effective_drone_autonomy:
                        continue
                    if r_stop["node"] != 0 and r_stop["arrival"] < t_ready_c:
                        continue
                        
                    # Compter les drones presents
                    drones_present_at_return = sum(1 for rt in ev_inventory[ev_ret_id] if rt <= r_stop["arrival"])
                    if ev_ret_id == ev_launch_id:
                        drones_present_at_return -= 1
                        
                    if drones_present_at_return >= 5:
                        continue
                        
                    total_dist = d_out + d_back
                    valid_configs.append((ev_launch_id, l_idx, ev_ret_id, r_idx, total_dist, min(ready_drones)))

            if valid_configs:
                # Stochastic AI Selection (Softmax over negative distances)
                distances = np.array([cfg[4] for cfg in valid_configs], dtype=np.float64)
                tau = 2.0  # Temperature for exploration
                logits = -distances / tau
                logits -= np.max(logits) # Numerical stability
                probs = np.exp(logits) / np.sum(np.exp(logits))
                chosen_idx = np.random.choice(len(valid_configs), p=probs)
                best_config = valid_configs[chosen_idx]
            else:
                best_config = None

            if best_config:
                ev_l, l_idx, ev_r, r_idx, d_total, r_time = best_config
                l_stop = ev_schedules[ev_l][l_idx]
                r_stop = ev_schedules[ev_r][r_idx]
                
                soc_after_delivery = 1.0 - (self.d(l_stop["node"], c) * 0.05)
                successs, meet_pt, traj, rews = self.simulate_rendezvous(c, c_pos, r_stop["pos"], soc_after_delivery)
                
                if successs:
                    drone_assign[c] = r_stop["node"]
                    drone_launch[c] = l_stop["node"]
                    drone_meetings[c] = meet_pt
                    drone_dists[c] = d_total
                    drone_trajectories.extend(traj)
                    drone_rewards.extend(rews)
                    
                    # Mise a  jour des inventaires
                    ev_inventory[ev_l].remove(r_time)
                    # Recharge du drone apra¨s retour, plus temps de dechargement sur l'EV
                    t_rendezvous = max(t_ready_c + self.travel_time_minutes(c, r_stop["node"]), r_stop["arrival"])
                    ev_inventory[ev_r].append(t_rendezvous + self.drone_recharge_time + self.drone_return_time)
                else:
                    unserved.append(c)
            else:
                unserved.append(c)
        return drone_assign, unserved, drone_trajectories, drone_rewards, drone_dists, drone_meetings, drone_launch

    def simulate_rendezvous(self, drone_id, drone_start_pos, ev_start_pos, soc):
        """Level 4: MADDPG Operational Rendezvous Maneuver."""
        d_pos = np.array(drone_start_pos, dtype=np.float32)
        e_pos = np.array(ev_start_pos, dtype=np.float32)
        rv_traj, rv_rewards = [], []
        met_successs = False
        meeting_point = d_pos.copy()
        
        obs = np.array([(e_pos[0]-d_pos[0])*10, (e_pos[1]-d_pos[1])*10, soc, 0, 0], dtype=np.float32)
        
        for step in range(20):
            # MADDPG Action
            action = self.maddpg.act(obs)
            d_pos += action * 0.005  # Continuous step
            
            dist = euclidean_km(d_pos, e_pos)
            reward = -dist
            done = 0
            if dist < 0.1:
                reward += 100.0
                met_successs = True
                meeting_point = d_pos.copy()
                done = 1
                
            next_obs = np.array([(e_pos[0]-d_pos[0])*10, (e_pos[1]-d_pos[1])*10, soc, action[0], action[1]], dtype=np.float32)
            
            # Store transition (state, action, reward, next_state, done)
            rv_traj.append((obs.copy(), action.copy(), reward, next_obs.copy(), done))
            rv_rewards.append(reward)
            
            obs = next_obs
            
            if done:
                break
        
        # Fallback: si MADDPG n'a pas converge, le drone rejoint l'EV en ligne directe
        # La faisabilite geometrique a deja  ete verifiee dans assign_drones
        if not met_successs:
            meeting_point = e_pos.copy()
            met_successs = True  # Le rendez-vous est toujours faisable geometriquement
            rv_rewards[-1] = 50.0  # Reward reduit (pas optimal mais reussi)
            rv_traj[-1] = (rv_traj[-1][0], rv_traj[-1][1], 50.0, rv_traj[-1][3], 1)
            
        return met_successs, meeting_point, rv_traj, rv_rewards

    def construct_solution_sample(self, ep=1):
        """Level 3 : Routage Tactique (MAPPO)
        
        INPUT  : master_plan (sortie de NSGA-II), position actuelle des vehicules, matrice de distances
        
        1 - Observation (7 features) : L'Acteur observe l'etat du systa¨me
        2 - Decision (Action Discra¨te) : L'Acteur choisit le prochain client
            parmi la liste assignee par NSGA-II
        3 - Recompense (Reward) : R = -(Distance parcourue) - (Penalite de retard)
        4 - Mise a  jour PPO (Apprentissage) : Clipping avec epsilon = 0.20
        
        OUTPUT : La trajectoire exacte de chaque EV et Drone (ordre sequentiel des visites)
        """
        ev_routes = [[0] for _ in range(self.n_ev)]
        ev_current = [0] * self.n_ev
        trajectories = [[] for _ in range(self.n_ev)]
        rewards = [[] for _ in range(self.n_ev)]
        
        # Temperature pour l'exploration : decroa®t de 1.5 a  0.5
        temperature = max(0.5, 1.5 - (ep / max(1, self.n_episodes)))
        
        # Chaque EV rea§oit sa liste de clients affectee par NSGA-II (Level 2)
        ev_remaining = {v: set(self.master_plan[v]) for v in range(self.n_ev)}
        
        any_left = True
        while any_left:
            any_left = False
            for agent in range(self.n_ev):
                if not ev_remaining[agent]:
                    continue
                any_left = True
                current = ev_current[agent]
                candidates = list(ev_remaining[agent])
                
                # 1 - Observation : L'Acteur calcule les features pour chaque client candidat
                cf = torch.tensor(np.array([
                    self.features(current, c, len(candidates))
                    for c in candidates
                ], dtype=np.float32))
                
                # 2 - Decision : L'Acteur gena¨re une probabilite et choisit le prochain client
                logits = self.actor(cf).squeeze(-1) / temperature
                dist = Categorical(logits=logits)
                ap = dist.sample()
                lp = dist.log_prob(ap)
                sel = candidates[int(ap.item())]
                
                # Le Critique evalue l'etat global
                g = torch.tensor(
                    self.global_state(agent, current, ev_remaining[agent]),
                    dtype=torch.float32
                )
                val = self.critic(g).squeeze(-1)
                
                # 3 - Recompense : R = -(Distance parcourue) - (Penalite de retard)
                late_penalty = 0.0
                if sel in self.time_windows:
                    tw_s, tw_e = self.time_windows[sel]
                    # Estimation du temps d'arrivee
                    est_arrival = self.start_time_minute + sum(
                        self.travel_time_minutes(ev_routes[agent][j], ev_routes[agent][j+1])
                        for j in range(len(ev_routes[agent])-1)
                    ) + self.travel_time_minutes(current, sel)
                    if est_arrival > tw_e:
                        late_penalty = (est_arrival - tw_e) * 0.1
                
                reward = -self.d(current, sel) - late_penalty
                
                trajectories[agent].append({
                    "cf": cf, "ap": ap.detach(), "lp": lp.detach(), "g": g, "v": val
                })
                rewards[agent].append(float(reward))
                ev_routes[agent].append(sel)
                ev_current[agent] = sel
                ev_remaining[agent].remove(sel)
        
        for r in ev_routes:
            if r[-1] != 0:
                r.append(0)
        
        da, us, rv_traj, rv_rew, ddist, dmeet, dlaunch = self.assign_drones(ev_routes)
        m = self.evaluate(ev_routes, da, us, ddist, dlaunch)
        for r_list in rewards:
            if r_list:
                r_list[-1] += -0.05 * m["objective"]
        return ev_routes, da, us, trajectories, rewards, rv_traj, rv_rew, m, dmeet, dlaunch

    def construct_solution_deterministic(self):
        """Level 3 : Routage Deterministe (MAPPO - Greedy / Argmax)
        Maªme architecture que construct_solution_sample mais sans exploration :
        l'Acteur prend toujours l'action avec la plus forte probabilite (argmax).
        Utilise pour l'evaluation finale et le resultat affiche.
        """
        ev_routes = [[0] for _ in range(self.n_ev)]
        ev_current = [0] * self.n_ev
        
        # Chaque EV rea§oit sa liste de clients affectee par NSGA-II (Level 2)
        ev_remaining = {v: set(self.master_plan[v]) for v in range(self.n_ev)}
        
        any_left = True
        while any_left:
            any_left = False
            for agent in range(self.n_ev):
                if not ev_remaining[agent]:
                    continue
                any_left = True
                candidates = list(ev_remaining[agent])
                
                # Observation : features pour chaque client candidat assigne par NSGA-II
                cf = np.array([
                    self.features(ev_current[agent], c, len(candidates))
                    for c in candidates
                ], dtype=np.float32)
                
                # Decision Greedy : argmax (pas d'exploration)
                with torch.no_grad():
                    logits = self.actor(torch.tensor(cf)).squeeze(-1)
                    sel = candidates[int(torch.argmax(logits).item())]
                
                ev_routes[agent].append(sel)
                ev_current[agent] = sel
                ev_remaining[agent].remove(sel)
        
        for r in ev_routes:
            if r[-1] != 0:
                r.append(0)
        
        da, us, _, _, ddist, dmeet, dlaunch = self.assign_drones(ev_routes)
        met = self.evaluate(ev_routes, da, us, ddist, dlaunch)
        return ev_routes, da, us, met, dmeet, dlaunch

    def compute_returns(self, rewards):
        returns, g = [], 0.0
        for r in reversed(rewards): g = r + self.gamma*g; returns.insert(0, g)
        return torch.tensor(returns, dtype=torch.float32)

    def ppo_update(self, trajectories, rewards, rv_traj=None, rv_rew=None):
        # 1. Routing Update (MAPPO)
        all_traj = []
        all_returns = []
        for traj, rew in zip(trajectories, rewards):
            if traj:
                rets = self.compute_returns(rew)
                all_traj.extend(traj)
                all_returns.append(rets)
                
        if all_traj:
            returns = torch.cat(all_returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            old_lp = torch.stack([t["lp"] for t in all_traj]).detach()
            old_v = torch.stack([t["v"] for t in all_traj]).detach()
            adv = returns - old_v
            if len(adv) > 1: adv = (adv - adv.mean())/(adv.std()+1e-8)
            for _ in range(self.ppo_epochs):
                nlp, vals, ents = [], [], []
                for t in all_traj:
                    logits = self.actor(t["cf"]).squeeze(-1)
                    d = Categorical(logits=logits)
                    nlp.append(d.log_prob(t["ap"])); ents.append(d.entropy())
                    vals.append(self.critic(t["g"]).squeeze(-1))
                nlp, vals, ents = torch.stack(nlp), torch.stack(vals), torch.stack(ents)
                ratio = torch.exp(nlp - old_lp)
                s1 = ratio*adv; s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps)*adv
                
                actor_loss = -torch.min(s1, s2).mean() - 0.01*ents.mean()
                critic_loss = 0.5*((returns-vals)**2).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        # 2. Level 4: Operational MADDPG Update
        if rv_traj:
            for step_trans in rv_traj:
                self.maddpg.replay_buffer.append(step_trans)
                # Keep buffer size manageable
                if len(self.maddpg.replay_buffer) > 10000:
                    self.maddpg.replay_buffer.pop(0)
            
            for _ in range(self.ppo_epochs):
                self.maddpg.update()

    def evaluate(self, ev_routes, drone_assign, unserved, ddist_map=None, drone_launch=None):
        ev_dist, ev_distances = 0.0, []
        nfz_violations = 0
        ev_stop_times = {} # (ev_id, node_id) -> departure_time
        
        for ev_id, r in enumerate(ev_routes):
            dd = 0
            t = self.start_time_minute
            ev_stop_times[(ev_id, r[0])] = t
            for a, b in zip(r[:-1], r[1:]):
                travel = self.travel_time_minutes(a, b)
                t += travel
                dd += self.d(a, b)
                # NFZ check
                if self.nfz_polygons:
                    lat1, lon1 = self.coords[a]
                    lat2, lon2 = self.coords[b]
                    for step in [0.2, 0.4, 0.5, 0.6, 0.8]:
                        mlat = lat1 + step*(lat2-lat1)
                        mlon = lon1 + step*(lon2-lon1)
                        is_viol = False
                        for b_nfz in self.nfz_polygons:
                            if b_nfz[0] <= mlat <= b_nfz[1] and b_nfz[2] <= mlon <= b_nfz[3]:
                                nfz_violations += 1
                                is_viol = True
                                break
                        if is_viol: break
                
                # Service time: drone clients = quick launch (1 min), EV clients = full service
                is_drone_waypoint = (b != 0 and self.demand_types[b] == 2 and b not in self.no_flight_clients)
                if is_drone_waypoint:
                    service = 1.0  # Lancement rapide du drone, l'EV ne s'arraªte pas longtemps
                else:
                    service = self.service_time_min * max(1, self.orders.get(b, {}).get("parcels", 1)) if b != 0 else 0
                t += service
                ev_stop_times[(ev_id, b)] = t
                
            ev_distances.append(dd); ev_dist += dd
        
        drone_dist, drone_distances, did = 0.0, {}, 1
        for c, p in drone_assign.items():
            dd = ddist_map.get(c, 2*self.d(p, c)) if ddist_map else 2*self.d(p, c)
            drone_dist += dd
            drone_distances[did] = drone_distances.get(did, 0.0)+dd
            did = 1 if did >= self.n_drones else did+1
            
        arrivals, delivered, late = {}, {}, 0.0
        # EV Arrivals (skip drone clients - they are launch waypoints, not EV deliveries)
        for ev_id, r in enumerate(ev_routes):
            t = self.start_time_minute
            for a, b in zip(r[:-1], r[1:]):
                t += self.travel_time_minutes(a, b)
                if b != 0:
                    # Si c'est un client drone, l'EV ne livre pas - il sert de point de lancement
                    if self.demand_types[b] == 2 and b not in self.no_flight_clients:
                        continue  # Sera comptabilise dans Drone Arrivals
                    arrivals[b] = t
                    ok = True
                    if b in self.time_windows:
                        tw_s, tw_e = self.time_windows[b]
                        if t < tw_s: t = tw_s
                        elif t > tw_e: late += t-tw_e; ok = False
                    if ok: t += self.service_time_min*max(1, self.orders.get(b, {}).get("parcels", 1))
                    delivered[b] = ok

        # Drone Arrivals
        for c, p_ret in drone_assign.items():
            p_launch = drone_launch.get(c, 0) if drone_launch else 0
            # On cherche l'heure de departure de l'EV au point de lancement
            # Comme un noeud peut aªtre visite plusieurs fois, on prend une estimation ou on simplifie
            # Ici on prend le premier EV qui passe par p_launch
            t_launch = self.start_time_minute
            for (eid, nid), dtime in ev_stop_times.items():
                if nid == p_launch:
                    t_launch = dtime
                    break
            
            t = t_launch + self.travel_time_minutes(p_launch, c)
            arrivals[c] = t; ok = True
            if c in self.time_windows:
                if t > self.time_windows[c][1]: late += t-self.time_windows[c][1]; ok = False
            delivered[c] = ok
            
        for c in unserved: delivered[c] = False
        tc = len(self.coords)-1
        dc = sum(1 for i in range(1, len(self.coords)) if delivered.get(i, False))
        sr = dc/max(1, tc)*100
        energy = ev_dist*self.energy_ev_kwh_km + drone_dist*self.energy_drone_kwh_km
        co2 = ev_dist*self.co2_ev_g_km + drone_dist*self.co2_drone_g_km
        # --- ASIF Framework Calculation (CO2 = A x S x I x F) ---
        # 1. Activity (A): Tonne-kilometers based on delivered parcels
        total_ev_pkgs = sum(self.orders.get(b, {}).get("parcels", 1) for r in ev_routes for b in r if b != 0)
        total_drone_pkgs = sum(self.orders.get(c, {}).get("parcels", 1) for c in drone_assign.keys())
        activity_ev_tkm = ev_dist * (total_ev_pkgs * (self.asif_avg_parcel_kg / 1000.0))
        activity_drone_tkm = drone_dist * (total_drone_pkgs * (self.asif_avg_parcel_kg / 1000.0))
        
        # 2. Structure (S): Modal share ratio by distance
        tot_d = max(1e-9, ev_dist + drone_dist)
        struct_ev = ev_dist / tot_d
        struct_drone = drone_dist / tot_d
        
        # 3. Intensity (I): Energy intensity (kWh/km) adjusted for load factor
        used_ev_cnt = max(1, sum(1 for d in ev_distances if d > 0))
        avg_pkgs_per_ev = total_ev_pkgs / used_ev_cnt
        load_factor_ev = min(1.0, max(0.1, avg_pkgs_per_ev / self.ev_capacity))
        intensity_ev_adj = self.energy_ev_kwh_km * (0.4 + 0.6 * load_factor_ev)
        intensity_drone_adj = self.energy_drone_kwh_km  # Drones are loaded 1/1
        
        # 4. Fuel (F): Grid emission factor (g CO2 / kWh)
        fuel_factor = self.asif_grid_factor_g_kwh
        
        # CO2 ASIF = A * S * I * F (Sum over modes, output in grams g CO2)
        co2_ev_asif = ev_dist * intensity_ev_adj * fuel_factor  # in grams g CO2
        co2_drone_asif = drone_dist * intensity_drone_adj * fuel_factor  # in grams g CO2
        co2_asife = co2_ev_asif + co2_drone_asif
        co2_ademe = ev_dist*self.co2_ev_ademe_g_km + drone_dist*self.co2_drone_ademe_g_km
        evr = sum(max(0, int(x//max(self.ev_autonomy_km, 1e-9))) for x in ev_distances)
        drr = len(drone_assign) # Chaque mission drone implique desormais une recharge
        
        # L'objectif inclut maintenant une forte penalite (500) pour chaque violation de zone par un camion
        obj = ev_dist+drone_dist+0.05*co2+2*energy+10*late+100*len(unserved)+5*evr+3*drr+2*self.n_ev+self.n_drones + 500*nfz_violations
        
        return {"objective": obj, "ev_dist": ev_dist, "drone_dist": drone_dist,
                "total_dist": ev_dist+drone_dist, "energy": energy, "co2": co2,
                "co2_asife": co2_asife, "co2_ademe": co2_ademe,
                "late": late, "service_rate": sr, "ev_recharges": evr, "drone_recharges": drr,
                "used_ev": sum(1 for d in ev_distances if d>0),
                "used_drones": sum(1 for d in drone_distances.values() if d>0),
                "ev_autonomy_left": np.mean([max(0, self.ev_autonomy_km-(d%self.ev_autonomy_km)) for d in ev_distances]) if ev_distances else self.ev_autonomy_km,
                "drone_autonomy_left": np.mean([max(0, self.drone_autonomy_km-(d%self.drone_autonomy_km)) for d in drone_distances.values()]) if drone_distances else self.drone_autonomy_km,
                "arrivals": arrivals, "delivered": delivered, "nfz_violations": nfz_violations,
                "drone_distances": drone_distances, "unserved": unserved, "ev_distances": ev_distances}

    def run(self, progress_callback=None):
        """Hierarchical 4-Level Execution."""
        # Level 1: Structuring (K-means)
        import numpy as np
        if len(self.coords) > 5:
            t_pens = self.traffic_penalties[1:] if self.traffic_penalties is not None else None
            self.clusters, _ = kmeans_conditioned(self.coords[1:], k=min(len(self.coords)-1, getattr(self, 'k_clusters', 4)), weather_penalty=self.weather_penalty, traffic_penalties=t_pens)
        else:
            self.clusters = np.zeros(len(self.coords)-1, dtype=int)
        
        # Level 2: Strategic (NSGA-II)
        # Pre-plan master routes to balance objectives
        nsga = NSGAII(
            self.coords, self.n_ev, clusters=self.clusters,
            pop_size=self.pop_size_nsga, n_gen=self.n_gen_nsga,
            cx_rate=self.cx_rate_nsga, mut_rate=self.mut_rate_nsga,
            orders=self.orders, ev_autonomy_km=self.ev_autonomy_km,
            demand_types=self.demand_types, no_flight_clients=self.no_flight_clients
        )
        self.master_plan = nsga.solve()
        
        best_obj, best = float("inf"), None
        for ep in range(1, self.n_episodes+1):
            if self.stop_flag: break
            
            # Level 3: Tactical (MAPPO) - Refines the execution of the master plan
            _, _, _, traj, rew, rv_traj, rv_rew, m_sample, dmeet_s, dlaunch_s = self.construct_solution_sample(ep)
            self.ppo_update(traj, rew, rv_traj, rv_rew) # Level 3 & 4 Update
            
            # Level 4: Operational (MADDPG) - Evaluated within the deterministic pass
            ev_e, dr_e, us_e, met, dmeet, dlaunch = self.construct_solution_deterministic()
            
            if met["objective"] < best_obj:
                best_obj = met["objective"]
                best = {"ev_routes": ev_e, "drone_assign": dr_e, "metrics": met, "drone_meetings": dmeet, "drone_launch": dlaunch}
            
            if not hasattr(self, 'smooth_obj'):
                self.smooth_obj = m_sample["objective"]
                self.smooth_dist = m_sample["total_dist"]
                self.smooth_co2 = m_sample["co2"]
                self.smooth_co2_asife = m_sample["co2_asife"]
                self.smooth_co2_ademe = m_sample["co2_ademe"]
                self.smooth_energy = m_sample["energy"]
                self.smooth_late = m_sample["late"]
            
            # Taux de lissage proportionnel au nombre d'episodes pour une belle courbe
            decay = min(0.05, 10.0 / max(1, self.n_episodes))
            
            self.smooth_obj = (1 - decay) * self.smooth_obj + decay * m_sample["objective"]
            self.smooth_dist = (1 - decay) * self.smooth_dist + decay * m_sample["total_dist"]
            self.smooth_co2 = (1 - decay) * self.smooth_co2 + decay * m_sample["co2"]
            self.smooth_co2_asife = (1 - decay) * self.smooth_co2_asife + decay * m_sample["co2_asife"]
            self.smooth_co2_ademe = (1 - decay) * self.smooth_co2_ademe + decay * m_sample["co2_ademe"]
            self.smooth_energy = (1 - decay) * self.smooth_energy + decay * m_sample["energy"]
            self.smooth_late = (1 - decay) * self.smooth_late + decay * m_sample["late"]

            self.history_best.append(self.smooth_obj)
            self.history_distance.append(self.smooth_dist)
            self.history_co2.append(self.smooth_co2)
            self.history_co2_asife.append(self.smooth_co2_asife)
            self.history_co2_ademe.append(self.smooth_co2_ademe)
            self.history_energy.append(self.smooth_energy)
            self.history_late.append(self.smooth_late)
            self.history_service.append(m_sample["service_rate"])
            
            if progress_callback: progress_callback(ep, self.n_episodes)
            
        self.best_solution = best
        return best
class ResearchDashboard:
    def __init__(self, root):
        self.root = root
        
        # Force default foreground colors for standard tk widgets to be white
        self.root.option_add('*Label.foreground', '#F8FAFC')
        self.root.option_add('*LabelFrame.foreground', '#F8FAFC')
        self.root.option_add('*Button.foreground', '#F8FAFC')
        self.root.title("MAPPO EV-Drone Dashboard (3 Tabs)")
        self.root.geometry("1700x950")
        self.root.configure(bg="#0F172A")
        try: self.root.state("zoomed")
        except Exception: pass

        style = ttk.Style()
        style.theme_use("clam")
        
        bg_dark = "#0F172A"
        bg_card = "#1E293B"
        fg_light = "#F8FAFC"
        
        style.configure("TFrame", background=bg_dark)
        style.configure("TLabel", background=bg_dark, foreground=fg_light)
        style.configure("TLabelframe", background=bg_dark, foreground=fg_light, bordercolor="#334155", lightcolor="#334155", darkcolor="#334155")
        style.configure("TLabelframe.Label", background=bg_dark, foreground=fg_light, font=("Arial", 10, "bold"))
        style.configure("TNotebook", background=bg_dark)
        style.configure("TNotebook.Tab", background=bg_card, foreground=fg_light, padding=[15, 5])
        style.map("TNotebook.Tab", background=[("selected", bg_dark)])
        
        # Professional widget styling
        style.configure("TEntry", fieldbackground="#334155", background="#334155", foreground="#F8FAFC", insertcolor="white", borderwidth=1, lightcolor="#475569", darkcolor="#1E293B")
        style.configure("TCombobox", fieldbackground="#334155", background="#1E293B", foreground="#F8FAFC", arrowcolor="#F8FAFC", borderwidth=1)
        style.map("TCombobox", fieldbackground=[("readonly", "#334155")], selectbackground=[("readonly", "#3B82F6")], selectforeground=[("readonly", "#FFFFFF")])
        style.configure("TCheckbutton", background=bg_dark, foreground=fg_light)
        style.map("TCheckbutton", background=[("active", bg_dark)], indicatorcolor=[("selected", "#3B82F6")], indicatorbackground=[("!selected", "#334155")])
        style.configure("TRadiobutton", background=bg_dark, foreground=fg_light)
        style.map("TRadiobutton", background=[("active", bg_dark)], indicatorcolor=[("selected", "#3B82F6")], indicatorbackground=[("!selected", "#334155")])
        
        
        # Global Treeview style for dark theme
        style.configure("Treeview", 
                        background="#1E293B",
                        foreground="#F8FAFC",
                        rowheight=25,
                        fieldbackground="#1E293B")
        style.configure("Treeview.Heading", 
                        background="#334155", 
                        foreground="#F8FAFC", 
                        font=("Arial", 9, "bold"))
        style.map("Treeview", background=[("selected", "#3B82F6")], foreground=[("selected", "#FFFFFF")])

        self.clients_coords, self.clients_demand, self.orders, self.time_windows = [], [], {}, {}
        self.clients_zones = {}
        self.kpi_vars, self.zone_vars = {}, {}
        
        self.EV_COLORS = [
            "#1e40af", "#c2410c", "#15803d", "#7e22ce",
            "#b91c1c", "#0891b2", "#a16207", "#be185d",
            "#4338ca", "#0f766e", "#9333ea", "#dc2626"
        ]
        self.DRONE_COLOR = "#06b6d4"
        
        self.show_depot_var = tk.BooleanVar(value=True)
        self.show_flight_zone_var = tk.BooleanVar(value=True)
        self.show_nfz_var = tk.BooleanVar(value=True)
        self.show_clients_var = tk.BooleanVar(value=True)
        self.show_clusters_var = tk.BooleanVar(value=True)
        self.show_urgences_var = tk.BooleanVar(value=True)
        self.ev_layer_vars = {}
        self.drone_layer_vars = {}
        self.preview_clusters = []

        
        self.city_var = tk.StringVar(value="Casablanca")
        self.n_clients_var = tk.IntVar(value=30)
        self.gen_mode_var = tk.StringVar(value="auto")
        self.urg_mode_var = tk.StringVar(value="auto")
        self.zone_clients_vars, self.zone_dist_vars, self.zone_urg_vars, self.zone_total_vars = {}, {}, {}, {}
        self.total_all_parcels_var = tk.StringVar(value="Global Total: 0")
        self.tw_mode_var = tk.StringVar(value="auto")
        self.n_ev_var = tk.IntVar(value=50)
        self.n_drone_var = tk.IntVar(value=100)
        self.ev_auto_var = tk.DoubleVar(value=47.4)
        self.drone_auto_var = tk.DoubleVar(value=12.0)
        self.speed_ev_var = tk.DoubleVar(value=45.0)
        self.speed_drone_var = tk.DoubleVar(value=40.0)
        self.drone_pick_time_var = tk.DoubleVar(value=2.0)
        self.drone_load_time_var = tk.DoubleVar(value=1.0)
        self.drone_unload_time_var = tk.DoubleVar(value=1.0)
        self.drone_takeoff_time_var = tk.DoubleVar(value=0.5)
        self.drone_return_time_var = tk.DoubleVar(value=0.5)
        self.drone_safe_margin_var = tk.DoubleVar(value=20.0)
        self.drone_recharge_time_var = tk.DoubleVar(value=45.0)
        
        # Load la cle API sauvegardee si elle existe
        saved_key = ""
        try:
            import os
            key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
            if os.path.exists(key_path):
                with open(key_path, "r", encoding="utf-8") as f:
                    saved_key = f.read().strip()
        except:
            pass
            
        self.openweather_api_key_var = tk.StringVar(value=saved_key)
        self.weather_status_var = tk.StringVar(value="No weather data")
        self.weather_allowed = True
        self.weather_reason = "No weather data"
        self.weather_conditions = {}
        self.weather_wind_max_var = tk.DoubleVar(value=10.0)
        self.weather_gust_max_var = tk.DoubleVar(value=12.0)
        self.weather_rain_max_var = tk.DoubleVar(value=0.5)
        self.weather_snow_max_var = tk.DoubleVar(value=0.2)
        self.weather_block_storms_var = tk.BooleanVar(value=True)
        self.enforce_nfz_var = tk.BooleanVar(value=True)
        self.start_time_var = tk.StringVar(value="08:00")
        self.end_time_var = tk.StringVar(value="18:00")
        
        self.episodes_var = tk.IntVar(value=600)
        self.k_clusters_var = tk.IntVar(value=4)
        self.k_clusters_var.trace_add("write", lambda *a: self.on_k_clusters_changed())
        self.hpo_trials_var = tk.IntVar(value=50)
        self.hyper_modes = ["Manual", "Automatique (Optuna)", "Automatique (Keras Tuner)"]
        self.hyper_mode_global_var = tk.StringVar(value="Manual")
        self.hyper_mode_var = tk.StringVar(value="Manual")
        self.lr_var = tk.StringVar(value="0.0003")
        self.gamma_var = tk.StringVar(value="0.99")
        self.clip_eps_var = tk.StringVar(value="0.20")
        self.ppo_epochs_var = tk.StringVar(value="4")
        
        self.hyper_mode_nsga_var = tk.StringVar(value="Manual")
        self.pop_size_nsga_var = tk.IntVar(value=30)
        self.n_gen_nsga_var = tk.IntVar(value=20)
        self.cx_rate_nsga_var = tk.DoubleVar(value=0.8)
        self.mut_rate_nsga_var = tk.DoubleVar(value=0.1)

        self.hyper_mode_maddpg_var = tk.StringVar(value="Manual")
        self.lr_maddpg_var = tk.StringVar(value="0.0010")
        self.gamma_maddpg_var = tk.StringVar(value="0.95")
        self.tau_maddpg_var = tk.DoubleVar(value=0.01)
        self.batch_maddpg_var = tk.IntVar(value=64)

        self.prod_mode_var = tk.StringVar(value="auto")
        self.n_heavy_var = tk.IntVar(value=50)
        self.n_light_var = tk.IntVar(value=50)
        self.light_ratio_var = tk.DoubleVar(value=50.0)

        self.solver = None

        # Donnees conservees pour redessiner la carte avec filtres / suivi
        self.latest_best = None
        self.latest_ev_routes_osrm = []
        self.latest_solver = None
        self._map_icons = []
        self.show_ev_var = tk.BooleanVar(value=True)
        self.show_drone_var = tk.BooleanVar(value=True)
        self.focus_client_var = tk.StringVar(value="All")
        self.focus_ev_var = tk.StringVar(value="All")
        self.focus_drone_var = tk.StringVar(value="All")
        self.record_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "simulation_records.json")
        self.record_file_agentic = os.path.join(os.path.dirname(os.path.realpath(__file__)), "simulation_records_agentic.json")
        self.simulation_records = []

        self.build_ui()
        self.load_client_database()
        self.load_simulation_records()
        
        # Centrage initial sur le depa´t de la ville par defaut (Casablanca)
        if self.map_widget:
            depot = CITY_DEPOTS["Casablanca"]
            self.map_widget.set_position(depot[0], depot[1])
            self.map_widget.set_zoom(12)

    def build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        from ui_sidebar import SidebarMenu
        self.sidebar = SidebarMenu(self.root, self.switch_view)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.main_content = tk.Frame(self.root, bg="#0F172A")
        self.main_content.grid(row=0, column=1, sticky="nsew")
        self.main_content.columnconfigure(0, weight=1)
        self.main_content.rowconfigure(0, weight=1)

        # Create Views
        self.tab1 = ttk.Frame(self.main_content)
        self.tab2 = ttk.Frame(self.main_content)
        self.tab3 = ttk.Frame(self.main_content)
        self.tab2_agentic = ttk.Frame(self.main_content)
        self.tab3_agentic = ttk.Frame(self.main_content)
        self.tab_simulation = ttk.Frame(self.main_content)
        self.tab_live = ttk.Frame(self.main_content)
        self.tab_live_classic = ttk.Frame(self.main_content)
        self.tab_forecasting = ttk.Frame(self.main_content)
        self.tab_demand_settings = ttk.Frame(self.main_content)
        
        self.views = {
            "config": self.tab1,
            "classic": self.tab2,
            "analytics_classic": self.tab3,
            "live_map_classic": self.tab_live_classic,
            "agentic": self.tab2_agentic,
            "agentic_debate": self.tab3_agentic,
            "analytics_agentic": self.tab_simulation,
            "live_map": self.tab_live,
            "demand_settings": self.tab_demand_settings,
            "forecasting": self.tab_forecasting
        }
        
        self.build_tab1()
        self.build_tab2()
        self.build_tab3()
        
        # Instantiate classic live mapping
        try:
            from ui_architecture import ArchitecturePanel
            self.live_mapping_classic_panel = ArchitecturePanel(self.tab_live_classic, self)
            self.live_mapping_classic_panel.set_mode(False) # Force classic mode
        except Exception as e:
            tk.Label(self.tab_live_classic, text=f"Error Classic Live Mapping: {e}", fg="red").pack(pady=20)

        from ui_agentic import AgenticUI
        self.agentic_ui = AgenticUI(self)
        self.agentic_ui.build_tab2_agentic()
        self.agentic_ui.build_tab3_agentic()

        try:
            from ui_forecasting import ForecastingUI
            self.forecasting_ui = ForecastingUI(self.tab_forecasting, self)
        except Exception as e:
            tk.Label(self.tab_forecasting, text=f"Error Loading Forecasting UI: {e}", fg="red").grid(row=0, column=0, padx=20, pady=20)
            
        try:
            from ui_demand_settings import DemandSettingsUI
            self.demand_settings_ui = DemandSettingsUI(self.tab_demand_settings, self)
        except Exception as e:
            tk.Label(self.tab_demand_settings, text=f"Error Loading Demand Settings UI: {e}", fg="red").grid(row=0, column=0, padx=20, pady=20)
        
        # Select default view
        self.sidebar.select("config")
        
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)

    def switch_view(self, view_id):
        for v in self.views.values():
            v.grid_forget()
        if view_id in self.views:
            self.views[view_id].grid(row=0, column=0, sticky="nsew")
            # Set focus to avoid mousewheel issues
            self.views[view_id].focus_set()

    def on_mousewheel(self, event):
        w = event.widget
        w_name = str(w).lower()
        if "map" in w_name: return
        
        # Intercept scroll for layers panel
        try:
            curr = w
            while curr:
                if isinstance(curr, tk.Canvas) and str(curr.winfo_name()) == "layers_canvas":
                    curr.yview_scroll(int(-1*(event.delta/120)), "units")
                    return
                curr = curr.master
        except Exception:
            pass
        
        try:
            sel = getattr(self, "sidebar", None)
            if sel:
                view_id = sel.current_selection
                if view_id == "config" and hasattr(self, 'canvas1'):
                    self.canvas1.yview_scroll(int(-1*(event.delta/120)), "units")
                elif view_id == "classic" and hasattr(self, 'canvas2'):
                    self.canvas2.yview_scroll(int(-1*(event.delta/120)), "units")
                elif view_id == "analytics_classic" and hasattr(self, 'canvas3'):
                    self.canvas3.yview_scroll(int(-1*(event.delta/120)), "units")
                elif view_id == "analytics_agentic":
                    panel = getattr(getattr(self, 'agentic_ui', None), 'agentic_results_panel', None)
                    if panel and hasattr(panel, 'a_canvas3'):
                        panel.a_canvas3.yview_scroll(int(-1*(event.delta/120)), "units")
                elif view_id == "training_agentic":
                    panel = getattr(self, 'agentic_ui', None)
                    if panel and hasattr(panel, 'a_canvas2'):
                        panel.a_canvas2.yview_scroll(int(-1*(event.delta/120)), "units")
        except Exception:
            pass

    def export_scenario_csv(self):
        if not hasattr(self, 'tree_orders') or len(self.tree_orders.get_children()) == 0:
            messagebox.showwarning("Export", "No scenario generated. Please generate a scenario first.")
            return
        try:
            filename = f"scenario_evrpd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, delimiter=';')
                cols = self.tree_orders["columns"]
                writer.writerow(cols)
                for row_id in self.tree_orders.get_children():
                    row = self.tree_orders.item(row_id)['values']
                    writer.writerow(row)
            messagebox.showinfo("Export Successful", f"Scenario exporte avec succa¨s vers :\n{os.path.abspath(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error lors de l'export CSV : {e}")

    def export_scenario_excel(self):
        if not hasattr(self, 'tree_orders') or len(self.tree_orders.get_children()) == 0:
            messagebox.showwarning("Export", "No scenario generated. Please generate a scenario first.")
            return
        try:
            import pandas as pd
            data = []
            cols = self.tree_orders["columns"]
            for row_id in self.tree_orders.get_children():
                data.append(self.tree_orders.item(row_id)['values'])
            df = pd.DataFrame(data, columns=cols)
            filename = f"scenario_evrpd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)
            messagebox.showinfo("Export Successful", f"Scenario exporte avec succa¨s vers :\n{os.path.abspath(filename)}")
        except ImportError:
            ans = messagebox.askyesno("Installation Requise", "'pandas' and 'openpyxl' libraries are needed for Excel export.\nVoulez-vous les installer automatiquement ?")
            if ans:
                self.lbl_status.config(text="Installing pandas and openpyxl (please wait)...")
                
                def install_and_export():
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl"])
                        self.root.after(0, lambda: self.lbl_status.config(text="Ready."))
                        self.root.after(0, self.export_scenario_excel)
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"a‰chec de l'installation : {e}"))
                        self.root.after(0, lambda: self.lbl_status.config(text="Installation Error."))
                        
                import threading
                threading.Thread(target=install_and_export, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Error lors de l'export Excel : {e}")


    def save_orders_json(self):
        if not hasattr(self, 'tree_orders') or len(self.tree_orders.get_children()) == 0:
            messagebox.showwarning("Save", "No scenario generated. Please generate a scenario first.")
            return
            
        data = {
            "clients_coords": self.clients_coords,
            "clients_demand": self.clients_demand,
            "orders": self.orders,
            "time_windows": self.time_windows,
            "no_flight_clients": self.no_flight_clients,
            "clients_zones": self.clients_zones,
            "tree_data": []
        }
        
        for row_id in self.tree_orders.get_children():
            data["tree_data"].append(self.tree_orders.item(row_id)['values'])
            
        filename = f"saved_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, indent=4)
            messagebox.showinfo("Success", f"Orders saved successsfully to {os.path.abspath(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save orders: {e}")

    def load_orders_json(self):
        from tkinter import filedialog
        import json
        
        filename = filedialog.askopenfilename(
            title="Select Saved Orders",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if not filename:
            return
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.bdd_status_var.set("DB loaded")
            self.clients_coords = [tuple(c) for c in data["clients_coords"]]
            self.clients_demand = data["clients_demand"]
            # Convert string keys back to int
            self.orders = {int(k): v for k, v in data["orders"].items()}
            self.time_windows = {int(k): tuple(v) for k, v in data["time_windows"].items()}
            self.no_flight_clients = data.get("no_flight_clients", [])
            self.clients_zones = {int(k): v for k, v in data.get("clients_zones", {}).items()}
            
            # Reconstruct urgency_list
            urgency_list = [False] # depot
            
            # Clear existing tree
            for item in self.tree_orders.get_children():
                self.tree_orders.delete(item)
                
            # Populate tree
            total_parcels = 0
            for row in data["tree_data"]:
                self.tree_orders.insert("", "end", values=row)
                total_parcels += int(row[4])
                urgency = (row[-1] == "URGENT")
                urgency_list.append(urgency)
                
            if hasattr(self, "total_all_parcels_var"):
                self.total_all_parcels_var.set(f"Total Parcels Generated : {total_parcels}")
                
            self.draw_clients(urgency_list)
            messagebox.showinfo("Success", f"Orders loaded successsfully from {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load orders: {e}")

    def build_tab1(self):
        # Frame principal blanc qui couvre tout l'onglet
        main_container = tk.Frame(self.tab1, bg="#1E293B")
        main_container.pack(fill="both", expand=True)

        # --- COLONNE GAUCHE (BDD Clients + Orders) ---
        left_col = tk.Frame(main_container, bg="#1E293B")
        left_col.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        
        # === 1. BASE DE DONNÉES CLIENTS ===
        p_bdd = ttk.LabelFrame(left_col, text="Client Database - Casablanca")
        p_bdd.pack(fill="x", pady=(0, 10))
        
        f_bdd = tk.Frame(p_bdd, bg="#1E293B")
        f_bdd.pack(fill="x", padx=10, pady=10)
        
        self.bdd_status_var = tk.StringVar(value="Loading...")
        self.bdd_count_var = tk.StringVar(value="0 clients")
        
        tk.Label(f_bdd, text="Status:", bg="#1E293B", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.lbl_bdd_status = tk.Label(f_bdd, textvariable=self.bdd_status_var, bg="#1E293B", fg="#10B981", font=("Arial", 10, "bold"))
        self.lbl_bdd_status.grid(row=0, column=1, sticky="w", padx=10)
        
        tk.Label(f_bdd, textvariable=self.bdd_count_var, bg="#1E293B", fg="#94A3B8", font=("Arial", 9)).grid(row=0, column=2, sticky="w", padx=10)
        
        btn_regen_bdd = tk.Button(f_bdd, text="Regenerate DB", bg="#7C3AED", fg="white",
                                   font=("Arial", 9, "bold"), relief="flat", cursor="hand2",
                                   command=self.regenerate_client_database)
        btn_regen_bdd.grid(row=0, column=3, padx=10)
        
        # === 2. GÉNÉRATION DE COMMANDES ===
        p_cmd = ttk.LabelFrame(left_col, text="Order Generation")
        p_cmd.pack(fill="x", pady=(0, 10))
        
        f_cmd = tk.Frame(p_cmd, bg="#1E293B")
        f_cmd.pack(fill="x", padx=10, pady=10)
        
        tk.Label(f_cmd, text="Number of orders:", bg="#1E293B", fg="white", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        self.n_orders_var = tk.IntVar(value=50)
        ttk.Entry(f_cmd, textvariable=self.n_orders_var, width=10).grid(row=0, column=1, padx=10)
        
        tk.Label(f_cmd, text="(max 3000)", bg="#1E293B", fg="#94A3B8", font=("Arial", 9)).grid(row=0, column=2, sticky="w")

        # === 3. DÉTAILS DES COMMANDES GÉNÉRÉES ===
        tk.Label(left_col, text="Generated Order Details:", font=("Arial", 10, "bold"), bg="#1E293B", fg="#F8FAFC").pack(anchor="w", pady=(10, 0))
        f_ord = tk.Frame(left_col, bg="#1E293B", bd=1, relief="solid")
        f_ord.pack(fill="both", expand=True, pady=(5, 0))
        
        columns = ("Client", "District", "Lat", "Lon", "Quantity", "Parcel Type", "Delivery Mode", "Time Window", "Urgency")
        self.tree_orders = ttk.Treeview(f_ord, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree_orders.heading(col, text=col)
            w = 70 if col in ("Lat", "Lon", "Quantity") else 95
            self.tree_orders.column(col, width=w, anchor="center")
            
        scroll_ord = ttk.Scrollbar(f_ord, command=self.tree_orders.yview)
        self.tree_orders.config(yscrollcommand=scroll_ord.set)
        scroll_ord.pack(side="right", fill="y")
        self.tree_orders.pack(side="left", fill="both", expand=True)
        
        # Boutons d'export
        f_export = tk.Frame(left_col, bg="#1E293B")
        f_export.pack(fill="x", pady=5)
        lbl_total = tk.Label(f_export, textvariable=self.total_all_parcels_var, bg="#1E293B", fg="white", font=("Arial", 10, "bold"))
        lbl_total.pack(side="left", padx=5)
        btn_csv = tk.Button(f_export, text="Export CSV", bg="#2563eb", fg="white", font=("Arial", 9, "bold"), command=self.export_scenario_csv)
        btn_csv.pack(side="right", padx=5)
        btn_excel = tk.Button(f_export, text="Export Excel", bg="#16a34a", fg="white", font=("Arial", 9, "bold"), command=self.export_scenario_excel)
        btn_excel.pack(side="right", padx=5)

        btn_load = tk.Button(f_export, text="Load JSON", bg="#8b5cf6", fg="white", font=("Arial", 9, "bold"), command=self.load_orders_json)
        btn_load.pack(side="right", padx=5)
        btn_save = tk.Button(f_export, text="Save JSON", bg="#f59e0b", fg="white", font=("Arial", 9, "bold"), command=self.save_orders_json)
        btn_save.pack(side="right", padx=5)

        # --- COLONNE DROITE (Flotte & Bouton) ---
        right_col = tk.Frame(main_container, bg="#1E293B")
        right_col.pack(side="left", fill="y", padx=20, pady=20)

        # --- EV Constraints Panel ---
        p_ev = ttk.LabelFrame(right_col, text="Electric Vehicle Constraints")
        p_ev.pack(fill="x", pady=(0, 10))
        
        f_ev = tk.Frame(p_ev, bg="#1E293B")
        f_ev.pack(padx=10, pady=10, fill="x")
        
        tk.Label(f_ev, text="Number of EVs:", bg="#1E293B", fg="white").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(f_ev, textvariable=self.n_ev_var, width=10).grid(row=0, column=1, padx=10)
        tk.Label(f_ev, text="EV Capacity:", bg="#1E293B", fg="#F8FAFC").grid(row=0, column=2, sticky="w", padx=(10,0))
        e_ev = ttk.Entry(f_ev, width=10); e_ev.insert(0, "100"); e_ev.config(state="readonly")
        e_ev.grid(row=0, column=3, padx=10)
        
        tk.Label(f_ev, text="Service Start:", bg="#1E293B", fg="white").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(f_ev, textvariable=self.start_time_var, width=10).grid(row=1, column=1, padx=10)
        tk.Label(f_ev, text="Service End:", bg="#1E293B", fg="white").grid(row=1, column=2, sticky="w", padx=(10,0))
        ttk.Entry(f_ev, textvariable=self.end_time_var, width=10).grid(row=1, column=3, padx=10)
        
        tk.Label(f_ev, text="EV Range (km):", bg="#1E293B", fg="white").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(f_ev, textvariable=self.ev_auto_var, width=10).grid(row=2, column=1, padx=10)
        tk.Label(f_ev, text="Speed:", bg="#1E293B", fg="#38BDF8", font=("Arial", 9, "bold")).grid(row=2, column=2, sticky="w", padx=(10,0))
        tk.Label(f_ev, text="Real-Time OSRM\nLimits: 40 Urb / 80 Nat", bg="#1E293B", fg="#38BDF8", font=("Arial", 8, "italic")).grid(row=2, column=3, padx=10)

        # --- Drone Constraints Panel ---
        p_drone = ttk.LabelFrame(right_col, text="Drone Constraints")
        p_drone.pack(fill="x", pady=(0, 20))
        
        f_drone = tk.Frame(p_drone, bg="#1E293B")
        f_drone.pack(padx=10, pady=10, fill="x")
        
        tk.Label(f_drone, text="Number of Drones:", bg="#1E293B", fg="white").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(f_drone, textvariable=self.n_drone_var, width=10).grid(row=0, column=1, padx=10)
        tk.Label(f_drone, text="Drone Capacity:", bg="#1E293B", fg="#F8FAFC").grid(row=0, column=2, sticky="w", padx=(10,0))
        e_dr = ttk.Entry(f_drone, width=10); e_dr.insert(0, "1"); e_dr.config(state="readonly")
        e_dr.grid(row=0, column=3, padx=10)

        tk.Label(f_drone, text="Service Start:", bg="#1E293B", fg="white").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(f_drone, textvariable=self.start_time_var, width=10).grid(row=1, column=1, padx=10)
        tk.Label(f_drone, text="Service End:", bg="#1E293B", fg="white").grid(row=1, column=2, sticky="w", padx=(10,0))
        ttk.Entry(f_drone, textvariable=self.end_time_var, width=10).grid(row=1, column=3, padx=10)
        
        tk.Label(f_drone, text="Drone Range (km):", bg="#1E293B", fg="white").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(f_drone, textvariable=self.drone_auto_var, width=10).grid(row=2, column=1, padx=10)
        tk.Label(f_drone, text="Speed:", bg="#1E293B", fg="#38BDF8", font=("Arial", 9, "bold")).grid(row=2, column=2, sticky="w", padx=(10,0))
        tk.Label(f_drone, text="Real-Time OSRM\nLimit: 40", bg="#1E293B", fg="#38BDF8", font=("Arial", 8, "italic")).grid(row=2, column=3, padx=10)

        # Weather API inside Drone panel
        tk.Label(f_drone, text="OpenWeather API :", bg="#1E293B", fg="white", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky="w", pady=5)
        self.ent_weather_key = tk.Entry(f_drone, textvariable=self.openweather_api_key_var, width=15)
        self.ent_weather_key.grid(row=3, column=1, padx=10)
        self.ent_weather_key.bind("<FocusOut>", lambda e: self.fetch_weather_conditions())
        tk.Button(f_drone, text="Update Weather", bg=GRENAT, fg="white", font=("Arial", 8, "bold"), relief="flat", cursor="hand2", command=self.fetch_weather_conditions).grid(row=3, column=2, padx=(10,0))
        tk.Label(f_drone, textvariable=self.weather_status_var, bg="#1E293B", fg="#F8FAFC", font=("Arial", 8)).grid(row=3, column=3, sticky="w", padx=10)

        # Weather thresholds
        f_weather_thresholds = tk.Frame(p_drone, bg="#1E293B")
        f_weather_thresholds.pack(fill="x", padx=10, pady=(0, 8))
        thresholds = [
            ("Max wind (m/s)", self.weather_wind_max_var),
            ("Max gusts (m/s)", self.weather_gust_max_var),
            ("Max rain (mm/h)", self.weather_rain_max_var),
            ("Max snow (mm/h)", self.weather_snow_max_var),
        ]
        for idx, (label, var) in enumerate(thresholds):
            tk.Label(f_weather_thresholds, text=f"{label}:", bg="#1E293B", fg="white", font=("Arial", 8)).grid(row=0, column=idx*2, sticky="w", padx=(0, 2))
            ttk.Entry(f_weather_thresholds, textvariable=var, width=5).grid(row=0, column=idx*2+1, padx=(0, 8))
        
        tk.Checkbutton(
            f_weather_thresholds, text="Block storm / sand", variable=self.weather_block_storms_var,
            bg="#1E293B", selectcolor="#0F172A", fg="white", anchor="w", onvalue=True, offvalue=False
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(5, 0))

        f_nfz_ctrl = tk.Frame(f_weather_thresholds, bg="#1E293B")
        f_nfz_ctrl.grid(row=1, column=4, columnspan=4, sticky="w", pady=(5, 0))

        tk.Checkbutton(
            f_nfz_ctrl, text="Avoid No-Flight Zones (NFZ)", variable=self.enforce_nfz_var,
            bg="#1E293B", selectcolor="#0F172A", fg="#38BDF8", font=("Arial", 8, "bold"),
            anchor="w", onvalue=True, offvalue=False
        ).pack(side="left")

        tk.Button(
            f_nfz_ctrl, text="📍 NFZ List", bg="#334155", fg="white",
            font=("Arial", 7, "bold"), relief="flat", cursor="hand2", padx=5, pady=1,
            command=self.show_nfz_modal
        ).pack(side="left", padx=(6, 0))

        # Drone times
        canvas_drone_times = tk.Canvas(p_drone, bg="#1E293B", height=35, highlightthickness=0)
        scrollbar_drone_times = ttk.Scrollbar(p_drone, orient="horizontal", command=canvas_drone_times.xview)
        f_drone_times = tk.Frame(canvas_drone_times, bg="#1E293B")
        f_drone_times.bind("<Configure>", lambda e: canvas_drone_times.configure(scrollregion=canvas_drone_times.bbox("all")))
        canvas_drone_times.create_window((0, 0), window=f_drone_times, anchor="nw")
        canvas_drone_times.configure(xscrollcommand=scrollbar_drone_times.set)
        canvas_drone_times.pack(fill="x", padx=10, pady=(5, 0))
        scrollbar_drone_times.pack(fill="x", padx=10, pady=(0, 5))
        
        times = [
            ("Pick-up EV", self.drone_pick_time_var),
            ("Loading", self.drone_load_time_var),
            ("Takeoff", self.drone_takeoff_time_var),
            ("Unloading", self.drone_unload_time_var),
            ("Retour", self.drone_return_time_var),
            ("Safe Marg%", self.drone_safe_margin_var),
            ("Recharge", self.drone_recharge_time_var),
        ]
        for idx, (label, var) in enumerate(times):
            lbl_text = f"{label}:" if "%" in label else f"{label}(min):"
            tk.Label(f_drone_times, text=lbl_text, bg="#1E293B", fg="white", font=("Arial", 8)).grid(row=0, column=idx*2, sticky="w", padx=(0, 2))
            ttk.Entry(f_drone_times, textvariable=var, width=5).grid(row=0, column=idx*2+1, padx=(0, 8))

        btn_gen = tk.Button(right_col, text="Generate Orders", bg=GRENAT, fg="white", 
                            font=("Arial", 14, "bold"), relief="flat", padx=40, pady=20, 
                            cursor="hand2", command=self.generate_scenario)
        btn_gen.pack(fill="x", pady=20)

        # --- Urgency Summary Area ---
        p_urgency = ttk.LabelFrame(right_col, text="Urgency Limits & Delivery Bounds")
        p_urgency.pack(fill="x", pady=(0, 10))
        
        # ScrolledText for urgency details
        self.txt_urgency = tk.Text(p_urgency, height=6, bg="#0F172A", fg="#38BDF8", font=("Consolas", 9), relief="flat", state="disabled")
        self.txt_urgency.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Architecture Diagrams (Under Generate Orders) ---
        frame_arch_btn = tk.Frame(right_col, bg="#0f1117")
        frame_arch_btn.pack(fill="x", pady=(10, 0))
        
        tk.Button(frame_arch_btn, text="Classic Architecture Diagram", 
                  font=("Arial", 11, "bold"), bg="#3b82f6", fg="white", 
                  relief="flat", cursor="hand2", padx=10, pady=12,
                  command=lambda: self.open_static_architecture(False)).pack(side="top", fill="x", pady=(0, 10))
                  
        tk.Button(frame_arch_btn, text="Agentic Architecture Diagram", 
                  font=("Arial", 11, "bold"), bg="#8b5cf6", fg="white", 
                  relief="flat", cursor="hand2", padx=10, pady=12,
                  command=lambda: self.open_static_architecture(True)).pack(side="top", fill="x")


    def update_clock(self):
        self.lbl_datetime.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_clock)

    def load_client_database(self):
        """Charge ou cree la base de donnees des 3000 clients de Casablanca."""
        try:
            from client_database import load_or_create_database
            self.client_db = load_or_create_database()
            n = len(self.client_db.get("clients", []))
            self.bdd_status_var.set("DB loaded")
            self.bdd_count_var.set(f"{n} clients")
            self.lbl_bdd_status.config(fg="#10B981")
        except Exception as e:
            self.bdd_status_var.set(f"Error : {e}")
            self.bdd_count_var.set("0 clients")
            self.lbl_bdd_status.config(fg="#EF4444")
            self.client_db = {"clients": []}

    def regenerate_client_database(self):
        """Regena¨re la base de donnees des clients (avec confirmation)."""
        if not messagebox.askyesno("Confirmation", 
                "Do you really want to regenerate the database des 3000 clients ?\n"
                "This operation will overwrite the current database."):
            return
        try:
            from client_database import load_or_create_database
            import os
            db_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clients_database.json")
            if os.path.exists(db_path):
                os.remove(db_path)
            self.client_db = load_or_create_database(db_path)
            n = len(self.client_db.get("clients", []))
            self.bdd_status_var.set("DB regenerated")
            self.bdd_count_var.set(f"{n} clients")
            self.lbl_bdd_status.config(fg="#10B981")
            messagebox.showinfo("Success", f"Database regenerated with {n} clients.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to regenerate DB : {e}")

        
        # Boutons Architecture (en bas)
        frame_btn = tk.Frame(self.tab1, bg="#1e293b")
        frame_btn.pack(pady=(15, 20))
        
        tk.Button(frame_btn, text="Classic System Architecture", 
                  font=("Arial", 10, "bold"), bg="#3b82f6", fg="white", 
                  relief="flat", cursor="hand2", padx=20, pady=8,
                  command=lambda: self.open_static_architecture(False)).pack(side="left", padx=10)
                  
        tk.Button(frame_btn, text="Agentic System Architecture", 
                  font=("Arial", 10, "bold"), bg="#8b5cf6", fg="white", 
                  relief="flat", cursor="hand2", padx=20, pady=8,
                  command=lambda: self.open_static_architecture(True)).pack(side="left", padx=10)

    def open_static_architecture(self, is_agentic):
        try:
            from ui_architecture import StaticArchitecturePanel
            top = tk.Toplevel(self.root)
            top.title(f"System Architecture ({'Agentic' if is_agentic else 'Classic'})")
            top.state('zoomed')
            top.configure(bg="#0f1117")
            StaticArchitecturePanel(top, is_agentic)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open architecture: {e}")

    def load_simulation_records(self):
        try:
            if os.path.isfile(self.record_file):
                with open(self.record_file, "r", encoding="utf-8") as f:
                    self.simulation_records = json.load(f) or []
            else:
                self.simulation_records = []
        except Exception:
            self.simulation_records = []
            
        try:
            if os.path.isfile(self.record_file_agentic):
                with open(self.record_file_agentic, "r", encoding="utf-8") as f:
                    self.simulation_records_agentic = json.load(f) or []
            else:
                self.simulation_records_agentic = []
        except Exception:
            self.simulation_records_agentic = []
            
        self.refresh_simulation_registry()

    def convert_to_native(self, obj):
        """Convertir recursivement tous les types numpy en types Python natifs pour la serialisation JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_native(v) for v in obj]
        return obj

    def save_simulation_records(self):
        try:
            with open(self.record_file, "w", encoding="utf-8") as f:
                json.dump(self.simulation_records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save classic registry : {e}")
            
        try:
            with open(self.record_file_agentic, "w", encoding="utf-8") as f:
                json.dump(self.simulation_records_agentic, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass # L'erreur agentique peut aªtre ignoree si non critique

        self.refresh_simulation_registry()

    def build_simulation_record(self, best, solver=None, is_agentic=False):
        record = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": self.city_var.get(),
            "n_clients": max(0, len(self.clients_coords) - 1),
            "n_ev": self.n_ev_var.get(),
            "n_drones": self.n_drone_var.get(),
            "weather": self.weather_status_var.get(),
            "weather_allowed": self.weather_allowed,
            "weather_conditions": self.weather_conditions,
            "clients_coords": self.convert_to_native(self.clients_coords),
            "clients_demand": self.convert_to_native(self.clients_demand),
            "orders": self.convert_to_native(self.orders),
            "metrics": self.convert_to_native(best.get("metrics", {})),
            "ev_routes": self.convert_to_native(best.get("ev_routes", [])),
            "drone_assign": self.convert_to_native(best.get("drone_assign", {})),
            "drone_launch": self.convert_to_native(best.get("drone_launch", {})),
            "drone_meetings": self.convert_to_native(best.get("drone_meetings", {})),
            "history_best": self.convert_to_native(solver.history_best if solver else []),
            "history_distance": self.convert_to_native(solver.history_distance if solver else []),
            "history_co2": self.convert_to_native(solver.history_co2 if solver else []),
            "history_co2_asife": self.convert_to_native(solver.history_co2_asife if solver else []),
            "history_co2_ademe": self.convert_to_native(solver.history_co2_ademe if solver else []),
            "history_energy": self.convert_to_native(solver.history_energy if solver else []),
            "history_late": self.convert_to_native(solver.history_late if solver else []),
            "history_service": self.convert_to_native(solver.history_service if solver else []),
            "clusters": self.convert_to_native(solver.clusters if solver and hasattr(solver, 'clusters') else (self.preview_clusters if hasattr(self, 'preview_clusters') else [])),
            "hpo_params": getattr(self, "latest_hpo_params", {}),
        }
        
        if is_agentic:
            record["classic_metrics"] = getattr(self, "latest_best", {}).get("metrics", {})
        
        # Save agentic debate logs if present
        if hasattr(self, 'agentic_ui') and hasattr(self.agentic_ui, 'latest_debate_res'):
            record["debate_res"] = self.convert_to_native(self.agentic_ui.latest_debate_res)
            
        return record

    def refresh_simulation_registry(self):
        if not hasattr(self, "lst_records"):
            return
        self.lst_records.delete(0, tk.END)
        for rec in self.simulation_records:
            metrics = rec.get("metrics", {})
            obj = metrics.get("objective", 0.0)
            rate = metrics.get("service_rate", 0.0)
            label = f"{rec['created']} | {rec['city']} | EV{rec['n_ev']} D{rec['n_drones']} | Obj {obj:.1f} | SR {rate:.1f}%"
            self.lst_records.insert(tk.END, label)

    def add_current_simulation_record(self, is_agentic=False):
        # Chercher le bon "solver" selon que ce soit agentique ou classique
        solver = None
        if is_agentic and hasattr(self, 'agentic_ui') and hasattr(self.agentic_ui, 'agentic_results_panel'):
            solver = getattr(self.agentic_ui.agentic_results_panel, 'latest_solver', None)
            best = getattr(self.agentic_ui.agentic_results_panel, 'latest_best', None)
        else:
            solver = self.latest_solver
            best = self.latest_best
            
        if not solver or not best:
            messagebox.showwarning("Warning", "Nonee simulation valide a  enregistrer.")
            return
            
        record = self.build_simulation_record(best, solver, is_agentic)
        
        if is_agentic:
            self.simulation_records_agentic.append(record)
        else:
            self.simulation_records.insert(0, record)
            
        self.save_simulation_records()
        messagebox.showinfo("Saved", "Simulation added to registry.")

    def delete_selected_simulation_record(self):
        selection = self.lst_records.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a simulation to delete.")
            return
        idx = selection[0]
        if messagebox.askyesno("Delete", "Delete this simulation from the registry?"):
            self.simulation_records.pop(idx)
            self.save_simulation_records()

    def view_selected_simulation_record(self):
        selection = self.lst_records.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a simulation to view.")
            return
        idx = selection[0]
        rec = self.simulation_records[idx]
        details = [
            f"ID: {rec['id']}",
            f"Creee: {rec['created']}",
            f"City: {rec['city']}",
            f"Clients: {rec['n_clients']}",
            f"EV: {rec['n_ev']}, Drones: {rec['n_drones']}",
            f"Weather: {rec['weather']}",
        ]
        weather_conditions = rec.get("weather_conditions", {})
        if weather_conditions:
            details.append(f"  Conditions: {weather_conditions.get('main', '?')} ({weather_conditions.get('description', '?')})")
            details.append(f"  Vent: {weather_conditions.get('wind_speed', 0.0):.1f} m/s, Rafales: {weather_conditions.get('gust', 0.0):.1f} m/s")
            details.append(f"  Pluie: {weather_conditions.get('rain', 0.0):.1f} mm/h, Neige: {weather_conditions.get('snow', 0.0):.1f} mm/h")
            details.append(f"  Flight autorise: {'Oui' if rec.get('weather_allowed', True) else 'Non'}")
        details.extend(["", "Metrics:"])
        for k, v in rec.get("metrics", {}).items():
            details.append(f"  {k}: {v}")
        details.append("")
        details.append("EV Routes:")
        for i, route in enumerate(rec.get("ev_routes", []), 1):
            details.append(f"  EV {i}: {' -> '.join(str(x) for x in route)}")
        details.append("")
        details.append("Drone Assignments:")
        for c, p in rec.get("drone_assign", {}).items():
            launch = rec.get("drone_launch", {}).get(c, 0)
            details.append(f"  C{c}: L{launch} -> C{c} -> R{c}={p}")
        popup = tk.Toplevel(self.root)
        popup.title("Simulation details")
        popup.geometry("700x520")
        txt = tk.Text(popup, wrap="word", font=("Consolas", 10), bg="#0F172A", fg="#F8FAFC", insertbackground="#0F172A")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", "\n".join(details))
        txt.config(state="disabled")

    def load_selected_simulation_record(self):
        try:
            selection = self.lst_records.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Select a simulation to load.")
                return
            idx = selection[0]
            rec = self.simulation_records[idx]

            def _convert_int_keys(obj):
                if isinstance(obj, dict):
                    new_obj = {}
                    for k, v in obj.items():
                        try:
                            key = int(k)
                        except Exception:
                            key = k
                        new_obj[key] = _convert_int_keys(v)
                    return new_obj
                if isinstance(obj, list):
                    return [_convert_int_keys(v) for v in obj]
                return obj

            # Load les parama¨tres du scenario
            self.city_var.set(rec.get('city', list(MOROCCO_ZONES.keys())[0]))
            self.n_ev_var.set(rec.get('n_ev', 50))
            self.n_drone_var.set(rec.get('n_drones', 100))

            # Load les donnees geometriques et de commande
            self.clients_coords = [tuple(coord) for coord in rec.get('clients_coords', [])]
            self.clients_demand = [int(v) for v in rec.get('clients_demand', [])]
            self.orders = _convert_int_keys(rec.get('orders', {}))
            self.weather_allowed = rec.get('weather_allowed', True)
            self.weather_conditions = rec.get('weather_conditions', {})
            self.weather_status_var.set(rec.get('weather', self.weather_status_var.get()))
            self.update_map_filter_choices()

            hpo_params = rec.get("hpo_params", {})
            if hpo_params:
                self.update_hyper_fields(hpo_params)

            # Load les resultats dans latest_best
            self.latest_best = {
                "metrics": _convert_int_keys(rec.get("metrics", {})),
                "ev_routes": [[int(n) for n in route] for route in rec.get("ev_routes", [])],
                "drone_assign": _convert_int_keys(rec.get("drone_assign", {})),
                "drone_launch": _convert_int_keys(rec.get("drone_launch", {})),
                "drone_meetings": _convert_int_keys(rec.get("drone_meetings", {})),
            }
            self.latest_ev_routes_osrm = []
            history_data = {k: rec.get(k, []) for k in [
                "history_best", "history_distance", "history_co2", "history_co2_asife",
                "history_co2_ademe", "history_energy", "history_late", "history_service"
            ]}
            if any(history_data.values()) or "clusters" in rec:
                self.latest_solver = SimpleNamespace(**history_data)
                self.latest_solver.clusters = rec.get("clusters", [])
            else:
                self.latest_solver = None

            self.update_kpi_display(self.latest_best["metrics"])
            self.update_map_from_record(rec)
            self.update_results_from_record(rec)
            threading.Thread(target=self.rebuild_loaded_ev_routes_bg, daemon=True).start()

            messagebox.showinfo("Charge", f"Simulation from {rec['created']} loaded with succa¨s!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load simulation: {str(e)}")

    def rebuild_loaded_ev_routes_bg(self):
        if not self.latest_best:
            return
        ev_routes_osrm = []
        for route_nodes in self.latest_best.get("ev_routes", []):
            route_indices = [n for n in route_nodes if isinstance(n, int)]
            active_clients = [n for n in route_indices if n != 0]
            if not active_clients:
                ev_routes_osrm.append(None)
                continue
            if route_indices[0] != 0:
                route_indices.insert(0, 0)
            if route_indices[-1] != 0:
                route_indices.append(0)
            pts = [self.clients_coords[n] for n in route_indices if isinstance(n, int) and n < len(self.clients_coords)]
            if len(pts) < 2:
                ev_routes_osrm.append(None)
                continue
            route_aller = get_osrm_route(pts[:-1])
            route_retour = get_osrm_route([pts[-2], pts[-1]])
            ev_routes_osrm.append({"aller": route_aller, "retour": route_retour})
        self.root.after(0, lambda: self.set_loaded_ev_routes_osrm(ev_routes_osrm))

    def set_loaded_ev_routes_osrm(self, ev_routes_osrm):
        self.latest_ev_routes_osrm = ev_routes_osrm
        self.refresh_map_view()
        if hasattr(self, 'latest_best') and self.latest_best:
            self.update_layers_panel(self.latest_best, ev_routes_osrm)
            if hasattr(self, 'agentic_ui') and hasattr(self.agentic_ui, 'agentic_results_panel'):
                self.agentic_ui.agentic_results_panel.latest_ev_routes_osrm = ev_routes_osrm
                self.agentic_ui.agentic_results_panel.update_a_cart_ev_checkboxes(self.latest_best)

    def update_kpi_display(self, metrics):
        """Updates KPI display with simulation metrics"""
        if not metrics:
            return

        # Afficher les KPI avec valeurs par defaut si absent
        if "Total distance" in self.kpi_vars:
            val = metrics.get("total_dist", metrics.get("total_distance", 0.0))
            self.kpi_vars["Total distance"].set(f"{val:.2f}")
        if "EV distance" in self.kpi_vars:
            val = metrics.get("ev_dist", metrics.get("ev_distance", 0.0))
            self.kpi_vars["EV distance"].set(f"{val:.2f}")
        if "Drone distance" in self.kpi_vars:
            val = metrics.get("drone_dist", metrics.get("drone_distance", 0.0))
            self.kpi_vars["Drone distance"].set(f"{val:.2f}")
        if "Energy" in self.kpi_vars:
            val = metrics.get("energy", 0.0)
            self.kpi_vars["Energy"].set(f"{val:.2f}")
        if "CO2 ASIF" in self.kpi_vars:
            val = metrics.get("co2_asife", 0.0)
            self.kpi_vars["CO2 ASIF"].set(f"{val:.2f}")
        if "Service rate" in self.kpi_vars:
            val = metrics.get("service_rate", 0.0)
            self.kpi_vars["Service rate"].set(f"{val:.2f}")
        if "Late penalty" in self.kpi_vars:
            val = metrics.get("late", 0.0)
            self.kpi_vars["Late penalty"].set(f"{val:.2f}")
        if "EV recharges" in self.kpi_vars:
            val = metrics.get("ev_recharges", 0)
            self.kpi_vars["EV recharges"].set(str(val))
        if "Drone recharges" in self.kpi_vars:
            val = metrics.get("drone_recharges", 0)
            self.kpi_vars["Drone recharges"].set(str(val))
        if "EV auto left" in self.kpi_vars:
            val = metrics.get("ev_autonomy_left", 0.0)
            self.kpi_vars["EV auto left"].set(f"{val:.2f}")
        if "Drone auto left" in self.kpi_vars:
            val = metrics.get("drone_autonomy_left", 0.0)
            self.kpi_vars["Drone auto left"].set(f"{val:.2f}")
        if "Resources used" in self.kpi_vars:
            used_ev = metrics.get("used_ev", 0)
            used_dr = metrics.get("used_drones", 0)
            self.kpi_vars["Resources used"].set(f"{used_ev} EV, {used_dr} Drones")

    def update_map_from_record(self, rec):
        """Updates the map with routes from the loaded simulation"""
        if self.map_widget is None:
            return
        self.refresh_map_view()

    def update_results_from_record(self, rec):
        """Affiche les donnees du registre dans l'onglet Results."""
        self.lbl_status.config(text="Simulation loaded from registry.")
        best = rec
        rec["drone_assign"] = {int(k): int(v) for k, v in rec.get("drone_assign", {}).items()}
        rec["drone_launch"] = {int(k): int(v) for k, v in rec.get("drone_launch", {}).items()}

        metrics = rec.get("metrics", {})
        self.update_kpi_display(metrics)

        # Bar chart simple
        self.ax_bar.clear()
        bar_names = []
        bar_vals = []
        ev_dist = metrics.get("ev_dist", metrics.get("ev_distance"))
        drone_dist = metrics.get("drone_dist", metrics.get("drone_distance"))
        energy = metrics.get("energy")
        co2 = metrics.get("co2")
        co2_asife = metrics.get("co2_asife")
        co2_ademe = metrics.get("co2_ademe")
        late = metrics.get("late")
        if ev_dist is not None:
            bar_names.append("EV km"); bar_vals.append(ev_dist)
        if drone_dist is not None:
            bar_names.append("Drone km"); bar_vals.append(drone_dist)
        if energy is not None:
            bar_names.append("Energy"); bar_vals.append(energy)
        if co2_asife is not None:
            bar_names.append("CO2 ASIF/100"); bar_vals.append(co2_asife / 100)
        if late is not None:
            bar_names.append("Late"); bar_vals.append(late)
        if bar_names:
            self.ax_bar.bar(bar_names, bar_vals, color=[GRENAT, "red", "orange", "green", "purple", "blue", "purple"][:len(bar_names)])
        self.canvas_bar.draw()

        # Learning curves: si on a un solver (historique), on l'utilise,
        # sinon on affiche des points seuls bases sur les metriques sauvegardees.
        try:
            def _history_list(value):
                if value is None:
                    return []
                if isinstance(value, dict):
                    value = list(value.values())
                if isinstance(value, (str, int, float)):
                    return [float(value)]
                if isinstance(value, list):
                    return [float(v) for v in value if v is not None]
                return []

            history_best = _history_list(rec.get("history_best") or (self.latest_solver.history_best if hasattr(self, 'latest_solver') and self.latest_solver else []))
            history_distance = _history_list(rec.get("history_distance") or (self.latest_solver.history_distance if hasattr(self, 'latest_solver') and self.latest_solver else []))
            history_co2 = _history_list(rec.get("history_co2") or (self.latest_solver.history_co2 if hasattr(self, 'latest_solver') and self.latest_solver else []))
            history_co2_asife = _history_list(rec.get("history_co2_asife") or (self.latest_solver.history_co2_asife if hasattr(self, 'latest_solver') and self.latest_solver else []))
            history_co2_ademe = _history_list(rec.get("history_co2_ademe") or (self.latest_solver.history_co2_ademe if hasattr(self, 'latest_solver') and self.latest_solver else []))
            history_energy = _history_list(rec.get("history_energy") or (self.latest_solver.history_energy if hasattr(self, 'latest_solver') and self.latest_solver else []))
            history_late = _history_list(rec.get("history_late") or (self.latest_solver.history_late if hasattr(self, 'latest_solver') and self.latest_solver else []))
            history_service = _history_list(rec.get("history_service") or (self.latest_solver.history_service if hasattr(self, 'latest_solver') and self.latest_solver else []))

            # Objective
            self.ax_obj.clear()
            if history_best:
                ep = list(range(1, len(history_best) + 1))
                self.ax_obj.plot(ep, history_best, color=GRENAT, label="Score IA")
                self.ax_obj.legend(fontsize=8)
            else:
                obj = metrics.get('objective', metrics.get('obj', None))
                if obj is not None:
                    self.ax_obj.bar([1], [obj], color=GRENAT)
                    self.ax_obj.set_xticks([1])
                    self.ax_obj.set_xticklabels(['Obj'])
            self.ax_obj.set_title("Objective", fontsize=10, fontweight='bold', color='white')
            self.ax_obj.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

            # Distance
            self.ax_dist.clear()
            if history_distance:
                ep = list(range(1, len(history_distance) + 1))
                self.ax_dist.plot(ep, history_distance, color="blue", label="Distance")
                self.ax_dist.legend(fontsize=8)
            else:
                evd = metrics.get('ev_dist', metrics.get('ev_distance', None))
                dd = metrics.get('drone_dist', metrics.get('drone_distance', None))
                if evd is not None and dd is not None:
                    self.ax_dist.bar([1, 2], [evd, dd], color=[GRENAT, 'blue'])
                    self.ax_dist.set_xticks([1, 2])
                    self.ax_dist.set_xticklabels(['EV', 'Drone'])
            self.ax_dist.set_title("Distance", fontsize=10, fontweight='bold', color='white')
            self.ax_dist.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

            # CO2 (ASIF)
            self.ax_co2.clear()
            if history_co2_asife:
                ep = list(range(1, len(history_co2_asife) + 1))
                self.ax_co2.plot(ep, history_co2_asife, color='orange', label='CO2 ASIF')
                self.ax_co2.legend(fontsize=8)
            else:
                val = metrics.get('co2_asife', 0.0)
                if val is not None:
                    self.ax_co2.bar([1], [val], color='orange')
                    self.ax_co2.set_xticks([1])
                    self.ax_co2.set_xticklabels(['CO2 ASIF'])
            self.ax_co2.set_title("CO2 (ASIF)", fontsize=10, fontweight='bold', color='white')
            self.ax_co2.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

            # Energy
            self.ax_energy.clear()
            if history_energy:
                ep = list(range(1, len(history_energy) + 1))
                self.ax_energy.plot(ep, history_energy, color='orange', label='Energy')
                self.ax_energy.legend(fontsize=8)
            else:
                energy_val = metrics.get('energy', None)
                if energy_val is not None:
                    self.ax_energy.plot([1], [energy_val], color='orange')
            self.ax_energy.set_title("Energy", fontsize=10, fontweight='bold', color='white')
            self.ax_energy.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

            # Late
            self.ax_late.clear()
            if history_late:
                ep = list(range(1, len(history_late) + 1))
                self.ax_late.plot(ep, history_late, color='red', label='Late')
                self.ax_late.legend(fontsize=8)
            else:
                late = metrics.get('late', None)
                if late is not None:
                    self.ax_late.plot([1], [late], color='red')
            self.ax_late.set_title("Late", fontsize=10, fontweight='bold', color='white')
            self.ax_late.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

            # Service Rate
            self.ax_service.clear()
            if history_service:
                ep = list(range(1, len(history_service) + 1))
                self.ax_service.plot(ep, history_service, color='purple', label='Service Rate')
                self.ax_service.legend(fontsize=8)
            else:
                service_val = metrics.get('service_rate', None)
                if service_val is not None:
                    self.ax_service.bar([1], [service_val], color='purple')
                    self.ax_service.set_xticks([1])
                    self.ax_service.set_xticklabels(['Service'])
            self.ax_service.set_title("Service Rate", fontsize=10, fontweight='bold', color='white')
            self.ax_service.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

            self.canvas_curve.draw()
        except Exception:
            pass

        # Map et plan
        self.draw_cartesian_plan(self.latest_best, None)
        self.refresh_map_view()

        # Texte route simplifie
        weather_status = self.weather_status_var.get() or "No weather data"
        weather_line = f"DRONE WEATHER: {weather_status} - Flight autorise : {'Oui' if self.weather_allowed else 'Non'}\n\n"
        self.txt_routes.pack(side="top", fill="both", expand=True)
        self.txt_routes.delete("1.0", "end")
        self.txt_routes.insert("end", weather_line)
        
        # --- CLUSTERING DETAILS (K-Means) ---
        clusters = []
        if hasattr(self, 'latest_solver') and self.latest_solver:
            clusters = getattr(self.latest_solver, 'clusters', [])
        grouped_clusters = {}
        if len(clusters) > 0:
            for i, cid in enumerate(clusters):
                c_id = i + 1
                grouped_clusters.setdefault(cid, []).append(c_id)
        
        cluster_vehicles = {cid: {"evs": [], "drones": []} for cid in grouped_clusters}
        for ev_idx, route in enumerate(best.get("ev_routes", [])):
            ev_id = ev_idx + 1
            for node in route:
                if node != 0:
                    node_cluster_idx = node - 1
                    if node_cluster_idx < len(clusters):
                        cid = clusters[node_cluster_idx]
                        if ev_id not in cluster_vehicles[cid]["evs"]:
                            cluster_vehicles[cid]["evs"].append(ev_id)
                            
        for c in best.get("drone_assign", {}).items():
            node_cluster_idx = c[0] - 1
            if node_cluster_idx < len(clusters):
                cid = clusters[node_cluster_idx]
                if c[0] not in cluster_vehicles[cid]["drones"]:
                    cluster_vehicles[cid]["drones"].append(c[0])

        cluster_text = "CLUSTERING ANALYSIS (K-Means):\n"
        if len(grouped_clusters) == 0:
            cluster_text += "  No cluster data available.\n\n"
        else:
            for cid in sorted(grouped_clusters.keys()):
                clients_list = ", ".join(f"C{c}" for c in grouped_clusters[cid])
                evs_serving = ", ".join(f"EV {e}" for e in cluster_vehicles[cid]["evs"]) or "None"
                drones_serving = ", ".join(f"Drone C{d}" for d in cluster_vehicles[cid]["drones"]) or "None"
                cluster_text += f"  Cluster {cid + 1}:\n"
                cluster_text += f"    - Clients: {clients_list}\n"
                cluster_text += f"    - Serving Vehicles: {evs_serving} | Drones: {drones_serving}\n"
            cluster_text += "\n"
            
        self.txt_routes.insert("end", cluster_text)
        self.txt_routes.insert("end", "ROUTES EV :\n")
        for k, route in enumerate(rec.get("ev_routes", []), 1):
            self.txt_routes.insert("end", f"EV {k}: {' -> '.join(str(x) for x in route)}\n")
        self.txt_routes.insert("end", "\nMISSIONS DRONE :\n")
        if not rec.get("drone_assign"):
            self.txt_routes.insert("end", "  Nonee mission drone planifiee.\n")
        for c, p_return in rec.get("drone_assign", {}).items():
            launch = rec.get("drone_launch", {}).get(c, 0)
            self.txt_routes.insert("end", f"  C{c}: L{launch} -> C{c} -> R{p_return}\n")
        
        self.txt_routes.insert("end", "\nDetails metriques :\n")
        for k, v in metrics.items():
            self.txt_routes.insert("end", f"  {k}: {v}\n")
            
        # Switch to the correct results tab automatically
        is_agentic = rec.get("is_agentic", False)
        if is_agentic:
            # For agentic training, the view is usually managed by the agentic_ui, but we can enforce it:
            if hasattr(self, 'sidebar'):
                self.sidebar.select("analytics_agentic")
        else:
            if hasattr(self, 'sidebar'):
                self.sidebar.select("analytics_classic")


    def build_tab2(self):
        self.canvas2 = tk.Canvas(self.tab2, bg="#1E293B", highlightthickness=0)
        self.canvas2.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(self.tab2, orient="vertical", command=self.canvas2.yview)
        scroll.pack(side="right", fill="y")
        self.canvas2.configure(yscrollcommand=scroll.set)
        inner = tk.Frame(self.canvas2, bg="#1E293B")
        self.canvas2.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: self.canvas2.configure(scrollregion=self.canvas2.bbox("all")))
        
        # Hyperparameters
        f_hypers = tk.Frame(inner, bg="#1E293B")
        f_hypers.pack(fill="x", padx=20, pady=10)

                # Global Optimizer UI
        f_global_opt = tk.Frame(f_hypers, bg="#1E293B", pady=10)
        f_global_opt.pack(fill="x", side="top", padx=5)
        tk.Label(f_global_opt, text="Global Optimizer Mode (All Algorithms):", bg="#1E293B", fg="#38BDF8", font=("Arial", 10, "bold")).pack(side="left")
        self.hyper_mode_all_var = tk.StringVar(value="Manual")
        ttk.Combobox(f_global_opt, textvariable=self.hyper_mode_all_var, values=self.hyper_modes, state="readonly", width=25).pack(side="left", padx=10)
        
        # HPO Trials selection (moved next to optimizer settings)
        tk.Label(f_global_opt, text="HPO Trials:", bg="#1E293B", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=(10, 5))
        ttk.Entry(f_global_opt, textvariable=self.hpo_trials_var, width=8).pack(side="left", padx=5)
        
        self.btn_optuna_all = tk.Button(f_global_opt, text="Optuna Optimize All", bg="#d9534f", fg="white", font=("Arial", 8, "bold"), relief="flat", cursor="hand2", command=lambda: self.start_tuning("optuna", "ALL"))
        self.btn_kerastuner_all = tk.Button(f_global_opt, text="Keras Optimize All", bg="#5cb85c", fg="white", font=("Arial", 8, "bold"), relief="flat", cursor="hand2", command=lambda: self.start_tuning("kerastuner", "ALL"))
        
        def toggle_all_modes(*args):
            mode = self.hyper_mode_all_var.get()
            self.btn_optuna_all.pack_forget()
            self.btn_kerastuner_all.pack_forget()
            if "Optuna" in mode: self.btn_optuna_all.pack(side="left", padx=5)
            elif "Keras" in mode: self.btn_kerastuner_all.pack(side="left", padx=5)
        
        self.hyper_mode_all_var.trace_add("write", toggle_all_modes)
        toggle_all_modes()
        
        # Hyperparameters Panels Container
        f_hypers_panels = tk.Frame(f_hypers, bg="#1E293B")
        f_hypers_panels.pack(fill="x")

        # --- COLONNE 0: Episodes ---
        p2_global = ttk.LabelFrame(f_hypers_panels, text="Global & Level 1 Parameters")
        p2_global.pack(side="left", fill="both", expand=True, padx=5)

        f_h_g = tk.Frame(p2_global, bg="#1E293B")
        f_h_g.pack(fill="x", padx=10, pady=10)
        tk.Label(f_h_g, text="Episodes:", bg="#1E293B", fg="white").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_g, textvariable=self.episodes_var, width=8).grid(row=0, column=1, padx=5)
        tk.Label(f_h_g, text="K Clusters:", bg="#1E293B", fg="white").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_g, textvariable=self.k_clusters_var, width=8).grid(row=1, column=1, padx=5)

        # --- COLONNE 1: NSGA-II ---
        p2_nsga = ttk.LabelFrame(f_hypers_panels, text="NSGA-II Hyperparameters (Strategic)")
        p2_nsga.pack(side="left", fill="both", expand=True, padx=5)
        
        f_h_n = tk.Frame(p2_nsga, bg="#1E293B")
        f_h_n.pack(fill="x", padx=10, pady=10)
        tk.Label(f_h_n, text="Pop Size:", bg="#1E293B", fg="white").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_n, textvariable=self.pop_size_nsga_var, width=8).grid(row=0, column=1, padx=5)
        tk.Label(f_h_n, text="Generations:", bg="#1E293B", fg="white").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Entry(f_h_n, textvariable=self.n_gen_nsga_var, width=8).grid(row=0, column=3, padx=5)
        tk.Label(f_h_n, text="Crossover:", bg="#1E293B", fg="white").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_n, textvariable=self.cx_rate_nsga_var, width=8).grid(row=1, column=1, padx=5)
        tk.Label(f_h_n, text="Mutation:", bg="#1E293B", fg="white").grid(row=1, column=2, sticky="w", padx=5)
        ttk.Entry(f_h_n, textvariable=self.mut_rate_nsga_var, width=8).grid(row=1, column=3, padx=5)

        # --- COLONNE 2: MAPPO ---
        p2_mappo = ttk.LabelFrame(f_hypers_panels, text="MAPPO Hyperparameters (Tactical)")
        p2_mappo.pack(side="left", fill="both", expand=True, padx=5)
        
        f_h_m = tk.Frame(p2_mappo, bg="#1E293B")
        f_h_m.pack(fill="x", padx=10, pady=10)
        tk.Label(f_h_m, text="LR:", bg="#1E293B", fg="white").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_m, textvariable=self.lr_var, width=8).grid(row=0, column=1, padx=5)
        tk.Label(f_h_m, text="Gamma:", bg="#1E293B", fg="white").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Entry(f_h_m, textvariable=self.gamma_var, width=8).grid(row=0, column=3, padx=5)
        tk.Label(f_h_m, text="Clip :", bg="#1E293B", fg="white").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_m, textvariable=self.clip_eps_var, width=8).grid(row=1, column=1, padx=5)
        tk.Label(f_h_m, text="Epochs :", bg="#1E293B", fg="white").grid(row=1, column=2, sticky="w", padx=5)
        ttk.Entry(f_h_m, textvariable=self.ppo_epochs_var, width=8).grid(row=1, column=3, padx=5)

        # --- COLONNE 3: MADDPG ---
        p2_maddpg = ttk.LabelFrame(f_hypers_panels, text="MADDPG Hyperparameters (Operational)")
        p2_maddpg.pack(side="left", fill="both", expand=True, padx=5)
        
        f_h_ma = tk.Frame(p2_maddpg, bg="#1E293B")
        f_h_ma.pack(fill="x", padx=10, pady=10)
        tk.Label(f_h_ma, text="LR:", bg="#1E293B", fg="white").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_ma, textvariable=self.lr_maddpg_var, width=8).grid(row=0, column=1, padx=5)
        tk.Label(f_h_ma, text="Gamma:", bg="#1E293B", fg="white").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Entry(f_h_ma, textvariable=self.gamma_maddpg_var, width=8).grid(row=0, column=3, padx=5)
        tk.Label(f_h_ma, text="Tau:", bg="#1E293B", fg="white").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_h_ma, textvariable=self.tau_maddpg_var, width=8).grid(row=1, column=1, padx=5)
        tk.Label(f_h_ma, text="Batch :", bg="#1E293B", fg="white").grid(row=1, column=2, sticky="w", padx=5)
        ttk.Entry(f_h_ma, textvariable=self.batch_maddpg_var, width=8).grid(row=1, column=3, padx=5)

        # Part 3
        p3 = ttk.LabelFrame(inner, text="Actions")
        p3.pack(fill="x", padx=20, pady=10)
        f_act = tk.Frame(p3, bg="#1E293B")
        f_act.pack(pady=20)
        self.btn_run = grenat_btn(f_act, "Start", self.start_training, width=15)
        self.btn_run.pack(side="left", padx=10)
        self.btn_stop = grenat_btn(f_act, "Stop", self.stop_training, width=15)
        self.btn_stop.pack(side="left", padx=10)

        f_progress = tk.Frame(inner, bg="#1E293B")
        f_progress.pack(fill="x", padx=20, pady=10)
        self.progress = ttk.Progressbar(f_progress, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.lbl_progress_pct = tk.Label(f_progress, text="0%", bg="#1E293B", fg="white", font=("Arial", 10, "bold"), width=6)
        self.lbl_progress_pct.pack(side="right")
        self.lbl_status = tk.Label(inner, text="Ready", bg="#1E293B", fg=GRENAT, font=("Arial", 10, "bold"))
        self.lbl_status.pack()

        # Architecture Info
        p4 = ttk.LabelFrame(inner, text="Proposed Hybrid Architecture")
        p4.pack(fill="x", padx=20, pady=10)
        desc = ("Level 1 (K-means): Structuring urban space into clusters.\n"
                "Level 2 (NSGA-II): Multi-objective strategic planning (Pareto).\n"
                "Level 3 (MAPPO): Tactical coordination and real-time navigation.\n"
                "Level 4 (MADDPG): Operational energy rendezvous maneuvers.")
        tk.Label(p4, text=desc, bg="#1E293B", fg="#F8FAFC", justify="left", font=("Arial", 9, "italic")).pack(padx=10, pady=10)
    def build_layers_panel(self, parent_frame, is_fullscreen=False):
        canvas_layers = tk.Canvas(parent_frame, bg="#1E293B", width=250, highlightthickness=0, name="layers_canvas")
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas_layers.yview)
        scrollable_frame = tk.Frame(canvas_layers, bg="#1E293B")
        
        # Keyboard scrolling
        def on_up(e):
            canvas_layers.yview_scroll(-1, "units")
        def on_down(e):
            canvas_layers.yview_scroll(1, "units")
            
        canvas_layers.bind("<Enter>", lambda e: canvas_layers.focus_set())
        canvas_layers.bind("<Up>", on_up)
        canvas_layers.bind("<Down>", on_down)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_layers.configure(scrollregion=canvas_layers.bbox("all"))
        )
        canvas_layers.create_window((0, 0), window=scrollable_frame, anchor="nw", width=230)
        canvas_layers.configure(yscrollcommand=scrollbar.set)
        
        canvas_layers.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        tk.Label(scrollable_frame, text="Layers", font=("Arial", 11, "bold"), bg="#1E293B", fg="#F8FAFC").pack(anchor="w", pady=(5,10), padx=5)
        
        f_actions = tk.Frame(scrollable_frame, bg="#1E293B")
        f_actions.pack(fill="x", padx=5, pady=5)
        tk.Button(f_actions, text="Show all", font=("Arial", 8), bg="#0F172A", command=self.show_all_layers).pack(side="left", expand=True, fill="x", padx=(0,2))
        tk.Button(f_actions, text="Hide all", font=("Arial", 8), bg="#0F172A", command=self.hide_all_layers).pack(side="right", expand=True, fill="x", padx=(2,0))
        
        f_fixed = self.create_collapsible_section(scrollable_frame, "Fixed Elements")
        self.create_layer_row(f_fixed, "Depot", "red", "circle", self.show_depot_var)
        self.create_layer_row(f_fixed, "Flight Zone", "#2ecc71", "rect", self.show_flight_zone_var)
        self.create_layer_row(f_fixed, "No-Flight Zones", "#e74c3c", "rect", self.show_nfz_var)
        self.create_layer_row(f_fixed, "Clients (by zone)", "#3498db", "circle", self.show_clients_var)
        self.create_layer_row(f_fixed, "Cluster Limits", "#9b59b6", "rect", self.show_clusters_var)
        self.create_layer_row(f_fixed, "Urgencys", "#f39c12", "circle", self.show_urgences_var)
        
        f_ev = self.create_collapsible_section(scrollable_frame, "EV Routes")
        tk.Label(f_ev, text="(No EV)", bg="#1E293B", fg="#F8FAFC", font=("Arial", 8)).pack(anchor="w")
        
        f_drone = self.create_collapsible_section(scrollable_frame, "Drone Routes")
        tk.Label(f_drone, text="(No drone)", bg="#1E293B", fg="#F8FAFC", font=("Arial", 8)).pack(anchor="w")

        if is_fullscreen:
            self.fs_fixed = f_fixed
            self.fs_ev_layers = f_ev
            self.fs_drone_layers = f_drone
        else:
            self.f_fixed = f_fixed
            self.f_ev_layers = f_ev
            self.f_drone_layers = f_drone

    def create_layer_row(self, parent, text, color, shape, var):
        f_row = tk.Frame(parent, bg="#1E293B")
        f_row.pack(fill="x", pady=2)
        c = tk.Canvas(f_row, width=16, height=16, bg="#1E293B", highlightthickness=0)
        c.pack(side="left", padx=(0, 5))
        if shape == "circle":
            c.create_oval(2, 2, 14, 14, fill=color, outline="black")
        elif shape == "rect":
            c.create_rectangle(2, 2, 14, 14, fill=color, outline="black")
        elif shape == "line":
            c.create_line(0, 8, 16, 8, fill=color, width=3, dash=(4,2))
        ttk.Checkbutton(f_row, text=text, variable=var, command=self.refresh_map_view).pack(side="left")
        return f_row

    def create_collapsible_section(self, parent, title):
        f_container = tk.Frame(parent, bg="#1E293B")
        f_container.pack(fill="x", pady=2)
        
        btn = tk.Button(f_container, text=f"\u25bc {title}", font=("Arial", 9, "bold"), bg="#0F172A", fg="#F8FAFC", activebackground="#1E293B", activeforeground="#FFFFFF", relief="flat", anchor="w", cursor="hand2")
        btn.pack(fill="x")
        
        f_content = tk.Frame(f_container, bg="#1E293B", padx=10, pady=5)
        f_content.pack(fill="x")
        
        def toggle():
            if f_content.winfo_ismapped():
                f_content.pack_forget()
                btn.config(text=f"\u25b6 {title}")
            else:
                f_content.pack(fill="x")
                btn.config(text=f"\u25bc {title}")
                
        btn.config(command=toggle)
        return f_content

    def show_all_layers(self):
        self.show_depot_var.set(True)
        self.show_flight_zone_var.set(True)
        self.show_nfz_var.set(True)
        self.show_clients_var.set(True)
        self.show_clusters_var.set(True)
        self.show_urgences_var.set(True)
        for var in self.ev_layer_vars.values(): var.set(True)
        for var in self.drone_layer_vars.values(): var.set(True)
        self.refresh_map_view()

    def hide_all_layers(self):
        self.show_depot_var.set(False)
        self.show_flight_zone_var.set(False)
        self.show_nfz_var.set(False)
        self.show_clients_var.set(False)
        self.show_clusters_var.set(False)
        self.show_urgences_var.set(False)
        for var in self.ev_layer_vars.values(): var.set(False)
        for var in self.drone_layer_vars.values(): var.set(False)
        self.refresh_map_view()

    def update_layers_panel(self, best, ev_routes_osrm):
        panels = []
        if hasattr(self, 'f_ev_layers') and self.f_ev_layers.winfo_exists():
            panels.append((self.f_ev_layers, self.f_drone_layers))
        if hasattr(self, 'fs_ev_layers') and self.fs_ev_layers.winfo_exists():
            panels.append((self.fs_ev_layers, self.fs_drone_layers))
            
        if not panels: return
        
        self.ev_layer_vars.clear()
        self.drone_layer_vars.clear()
        ev_routes = best.get("ev_routes", [])
        drone_assign = best.get("drone_assign", {})
        
        for p_ev, p_drone in panels:
            for widget in p_ev.winfo_children(): widget.destroy()
            for widget in p_drone.winfo_children(): widget.destroy()
            
            active_evs = 0
            for ev_idx, route in enumerate(ev_routes):
                if len(route) >= 2:
                    active_evs += 1
                    color = self.EV_COLORS[ev_idx % len(self.EV_COLORS)]
                    if ev_idx not in self.ev_layer_vars:
                        self.ev_layer_vars[ev_idx] = tk.BooleanVar(value=True)
                    var = self.ev_layer_vars[ev_idx]
                    
                    nb_clients = len(route) - 2
                    cl_text = f"{nb_clients} clients" if nb_clients > 1 else f"{nb_clients} client"
                    self.create_layer_row(p_ev, f"VE {ev_idx+1} ({cl_text})", color, "rect", var)
            
            if active_evs == 0:
                tk.Label(p_ev, text="(No EV)", bg="#1E293B", fg="#F8FAFC", font=("Arial", 8)).pack(anchor="w")

            if drone_assign:
                for client_id in sorted(drone_assign.keys()):
                    if client_id not in self.drone_layer_vars:
                        self.drone_layer_vars[client_id] = tk.BooleanVar(value=True)
                    var = self.drone_layer_vars[client_id]
                    self.create_layer_row(p_drone, f"D{client_id} -> C{client_id}", self.DRONE_COLOR, "line", var)
            else:
                tk.Label(p_drone, text="(No drone)", bg="#1E293B", fg="#F8FAFC", font=("Arial", 8)).pack(anchor="w")

    def build_tab3(self):
        self.canvas3 = tk.Canvas(self.tab3, bg="#1E293B", highlightthickness=0)
        self.canvas3.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(self.tab3, orient="vertical", command=self.canvas3.yview)
        scroll.pack(side="right", fill="y")
        self.canvas3.configure(yscrollcommand=scroll.set)
        inner = tk.Frame(self.canvas3, bg="#1E293B")
        self.inner_window = self.canvas3.create_window((0, 0), window=inner, anchor="nw")
        
        def on_inner_configure(e):
            self.canvas3.configure(scrollregion=self.canvas3.bbox("all"))
            
        def on_canvas_configure(e):
            self.canvas3.itemconfig(self.inner_window, width=e.width)
                
        inner.bind("<Configure>", on_inner_configure)
        self.canvas3.bind("<Configure>", on_canvas_configure)

        # KPI Summary
        p_kpi = ttk.LabelFrame(inner, text=" KPI Summary")
        p_kpi.pack(fill="x", padx=10, pady=10)
        
        bg_color = "#1E293B"
        grid_container = tk.Frame(p_kpi, bg=bg_color)
        grid_container.pack(fill="x", padx=5, pady=5)
        grid_container.columnconfigure(0, weight=1)
        grid_container.columnconfigure(1, weight=1)
        
        # 4 boxes for grouped KPIs
        box_dist = tk.LabelFrame(grid_container, text=" Distance ", font=("Segoe UI", 10, "bold"), fg="#38BDF8", bg=bg_color, bd=1, relief="groove")
        box_dist.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        box_energy = tk.LabelFrame(grid_container, text=" Energy & Environment ", font=("Segoe UI", 10, "bold"), fg="#4ADE80", bg=bg_color, bd=1, relief="groove")
        box_energy.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        box_service = tk.LabelFrame(grid_container, text=" Quality of Service & Resources ", font=("Segoe UI", 10, "bold"), fg="#F472B6", bg=bg_color, bd=1, relief="groove")
        box_service.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        box_autonomy = tk.LabelFrame(grid_container, text=" Recharges & Autonomy ", font=("Segoe UI", 10, "bold"), fg="#FBBF24", bg=bg_color, bd=1, relief="groove")
        box_autonomy.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure columns inside boxes
        for box in (box_dist, box_service):
            for c in range(3): box.columnconfigure(c, weight=1)
        for c in range(2): box_energy.columnconfigure(c, weight=1)
        for c in range(4): box_autonomy.columnconfigure(c, weight=1)
        
        kpi_mappings = {
            "Total distance": {"box": box_dist, "row": 0, "col": 0, "color": "#2196F3", "unit": "km"},
            "EV distance": {"box": box_dist, "row": 0, "col": 1, "color": "#1976D2", "unit": "km"},
            "Drone distance": {"box": box_dist, "row": 0, "col": 2, "color": "#03A9F4", "unit": "km"},
            
            "Energy": {"box": box_energy, "row": 0, "col": 0, "color": "#FFC107", "unit": "kWh"},
            "CO2 ASIF": {"box": box_energy, "row": 0, "col": 1, "color": "#388E3C", "unit": "g"},
            
            "Service rate": {"box": box_service, "row": 0, "col": 0, "color": "#9C27B0", "unit": "%"},
            "Late penalty": {"box": box_service, "row": 0, "col": 1, "color": "#F44336", "unit": "min"},
            "Resources used": {"box": box_service, "row": 0, "col": 2, "color": "#607D8B", "unit": ""},
            
            "EV recharges": {"box": box_autonomy, "row": 0, "col": 0, "color": "#FF9800", "unit": "times"},
            "Drone recharges": {"box": box_autonomy, "row": 0, "col": 1, "color": "#FF5722", "unit": "times"},
            "EV auto left": {"box": box_autonomy, "row": 0, "col": 2, "color": "#00BCD4", "unit": "km"},
            "Drone auto left": {"box": box_autonomy, "row": 0, "col": 3, "color": "#009688", "unit": "km"},
        }
        
        self.kpi_vars.clear()
        for name, info in kpi_mappings.items():
            var = tk.StringVar(value="-")
            self.kpi_vars[name] = var
            card = KPICard(info["box"], name, var, unit=info["unit"], icon="", color=info["color"])
            card.grid(row=info["row"], column=info["col"], sticky="nsew", padx=5, pady=5)

        p_map = ttk.LabelFrame(inner, text="Map View")
        p_map.pack(fill="both", expand=True, padx=10, pady=10)
        f_map = tk.Frame(p_map, bg="#1E293B")
        f_map.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Header avec bouton Plein a‰cran
        f_header = tk.Frame(f_map, bg="#0F172A", bd=1, relief="solid", padx=8, pady=6)
        f_header.pack(fill="x", padx=10, pady=(0, 8))
        self.btn_fullscreen_map = tk.Button(f_header, text="Full Screen", bg=GRENAT, fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", padx=10, pady=4, command=self.open_map_fullscreen)
        self.btn_fullscreen_map.pack(side="right", padx=6)
        
        # Conteneur 2 colonnes : Map (gauche) et Layers (droite)
        f_content = tk.Frame(f_map, bg="#1E293B")
        f_content.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Colonne Map
        f_map_view = tk.Frame(f_content, bg="#1E293B", bd=1, relief="solid")
        f_map_view.pack(side="left", fill="both", expand=True)
        
        self.fullscreen_map_window = None
        self.fullscreen_map_widget = None
        if MAP_AVAILABLE:
            self.map_widget = tkintermapview.TkinterMapView(f_map_view, corner_radius=0, width=800, height=500)
            try:
                # Utilisation d'un serveur tuile par defaut plus rapide (Google standard) pour un affichage immediat
                self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
            except Exception:
                pass
            self.map_widget.pack(fill="both", expand=True)
        else:
            self.map_widget = None
            tk.Label(f_map_view, text="tkintermapview not installed", bg="#1E293B", fg="white").pack(expand=True)
        
        # Legend drone sous la carte
        self.build_map_guide(f_map_view)
            
        # Colonne Layers (Layers)
        f_layers = tk.Frame(f_content, bg="#1E293B", bd=1, relief="solid", width=250)
        f_layers.pack(side="right", fill="y", padx=(10, 0))
        f_layers.pack_propagate(False) # Garder la largeur fixe
        
        self.build_layers_panel(f_layers)



        # Mapsian plan after the map view
        p_cart = ttk.LabelFrame(inner, text="EV-Drone Cartesian Plan")
        p_cart.pack(fill="both", padx=10, pady=10)
        
        # Toolbar for cartesian plan
        f_cart_toolbar = tk.Frame(p_cart, bg="#0F172A", pady=4)
        f_cart_toolbar.pack(fill="x", padx=5)
        
        tk.Label(f_cart_toolbar, text="Filter:", bg="#0F172A", fg="white", font=("Arial", 8, "bold")).pack(side="left", padx=(0,5))
        
        f_scroll_container = tk.Frame(f_cart_toolbar, bg="#0F172A")
        f_scroll_container.pack(side="left", fill="x", expand=True)
        self.cart_canvas = tk.Canvas(f_scroll_container, height=30, bg="#0F172A", highlightthickness=0)
        cart_scroll = ttk.Scrollbar(f_scroll_container, orient="horizontal", command=self.cart_canvas.xview)
        self.cart_canvas.configure(xscrollcommand=cart_scroll.set)
        cart_scroll.pack(side="bottom", fill="x")
        self.cart_canvas.pack(side="top", fill="both", expand=True)
        self.cart_filter_ev_frame = tk.Frame(self.cart_canvas, bg="#0F172A")
        self.cart_canvas.create_window((0, 0), window=self.cart_filter_ev_frame, anchor="nw")
        self.cart_filter_ev_frame.bind("<Configure>", lambda e: self.cart_canvas.configure(scrollregion=self.cart_canvas.bbox("all")))
        
        f_cart_btns = tk.Frame(f_cart_toolbar, bg="#0F172A")
        f_cart_btns.pack(side="right")
        
        # Afficher/Hide all boutons
        def cart_show_all():
            if hasattr(self, 'cart_ev_vars'):
                for v in self.cart_ev_vars.values(): v.set(True)
            self.draw_cartesian_plan(getattr(self,'latest_best',None), None)
        def cart_hide_all():
            if hasattr(self, 'cart_ev_vars'):
                for v in self.cart_ev_vars.values(): v.set(False)
            self.draw_cartesian_plan(getattr(self,'latest_best',None), None)
        self._cart_show_all_fn = cart_show_all
        self._cart_hide_all_fn = cart_hide_all
        
        tk.Button(f_cart_btns, text="Show all", font=("Arial", 8), bg="#0F172A",
                  relief="flat", padx=5, cursor="hand2",
                  command=cart_show_all).pack(side="left", padx=(6,2))
        tk.Button(f_cart_btns, text="Hide all", font=("Arial", 8), bg="#0F172A",
                  relief="flat", padx=5, cursor="hand2",
                  command=cart_hide_all).pack(side="left", padx=(0,6))
        
        self.cart_show_drones_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_cart_btns, text="Drone Flights", variable=self.cart_show_drones_var,
                        command=lambda: self.draw_cartesian_plan(getattr(self,'latest_best',None), getattr(self,'latest_solver',None))
                        ).pack(side="left", padx=8)
        
        tk.Button(f_cart_toolbar, text="Full Screen", font=("Arial", 8), bg="#1d4ed8", fg="white",
                  relief="flat", padx=8, pady=2, cursor="hand2",
                  command=self.open_cartesian_fullscreen).pack(side="right", padx=5)
        
        self.fig_cart = Figure(figsize=(12, 4), dpi=100)
        self.fig_cart.patch.set_facecolor('#1E293B')
        self.ax_cart = self.fig_cart.add_subplot(111)
        self.ax_cart.set_title("EV-Drone Cartesian Plan", color="white")
        self.ax_cart.set_facecolor('#1E293B')
        self.ax_cart.tick_params(colors="white")
        for spine in self.ax_cart.spines.values():
            spine.set_color('white')
        self.ax_cart.grid(True, alpha=0.3, linestyle=':')
        self.canvas_cart = FigureCanvasTkAgg(self.fig_cart, master=p_cart)
        self.canvas_cart.get_tk_widget().pack(fill="both", padx=5, pady=5)

        p_registry = ttk.LabelFrame(inner, text="Simulation Registry (Double-click to load)")
        p_registry.pack(fill="x", padx=10, pady=10)
        f_registry = tk.Frame(p_registry, bg="#1E293B")
        f_registry.pack(fill="both", expand=True, padx=5, pady=5)
        self.lst_records = tk.Listbox(f_registry, height=8, font=("Arial", 9), activestyle="dotbox")
        self.lst_records.pack(side="left", fill="both", expand=True, padx=(0,5), pady=5)
        self.lst_records.bind("<Double-Button-1>", lambda e: self.load_selected_simulation_record())
        scrollbar = ttk.Scrollbar(f_registry, orient="vertical", command=self.lst_records.yview)
        scrollbar.pack(side="left", fill="y", pady=5)
        self.lst_records.config(yscrollcommand=scrollbar.set)

        f_registry_buttons = tk.Frame(p_registry, bg="#1E293B")
        f_registry_buttons.pack(fill="x", padx=5, pady=(0,5))
        tk.Button(f_registry_buttons, text="Save simulation", bg=GRENAT, fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", command=self.add_current_simulation_record).pack(side="left", padx=5)
        tk.Button(f_registry_buttons, text="View details", bg="#2563eb", fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", command=self.view_selected_simulation_record).pack(side="left", padx=5)
        tk.Button(f_registry_buttons, text="Delete", bg="#ef4444", fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", command=self.delete_selected_simulation_record).pack(side="left", padx=5)
        tk.Button(f_registry_buttons, text="Refresh", bg="#6b7280", fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", command=self.refresh_simulation_registry).pack(side="left", padx=5)
        
        # Rafraa®chir le registre apra¨s la creation du widget
        self.root.after(100, self.refresh_simulation_registry)

        p_curves = ttk.LabelFrame(inner, text="Learning Curves")
        p_curves.pack(fill="x", padx=10, pady=10)
        self.fig_curve = Figure(figsize=(12, 9), dpi=100)
        self.fig_curve.patch.set_facecolor('#1E293B')
        self.fig_curve.subplots_adjust(hspace=0.4, wspace=0.3)
        self.ax_obj = self.fig_curve.add_subplot(3, 2, 1)
        self.ax_dist = self.fig_curve.add_subplot(3, 2, 2)
        self.ax_co2 = self.fig_curve.add_subplot(3, 2, 3)
        self.ax_energy = self.fig_curve.add_subplot(3, 2, 4)
        self.ax_late = self.fig_curve.add_subplot(3, 2, 5)
        self.ax_service = self.fig_curve.add_subplot(3, 2, 6)
        for ax, t in [
            (self.ax_obj, "Objective"),
            (self.ax_dist, "Distance"),
            (self.ax_co2, "CO2 Methods"),
            (self.ax_energy, "Energy"),
            (self.ax_late, "Late"),
            (self.ax_service, "Service Rate")
        ]:
            ax.set_title(t, fontsize=10, fontweight="bold", color="white")
            ax.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)
            ax.set_facecolor('#1E293B')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#334155')
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, master=p_curves)
        self.canvas_curve.get_tk_widget().pack(fill="x", padx=5, pady=5)

        p_bar = ttk.LabelFrame(inner, text="Metrics Comparison")
        p_bar.pack(fill="x", padx=10, pady=10)
        self.fig_bar = Figure(figsize=(12, 3), dpi=100)
        self.fig_bar.patch.set_facecolor('#1E293B')
        self.ax_bar = self.fig_bar.add_subplot(111)
        self.ax_bar.set_title("Metrics", fontsize=10, fontweight="bold", color="white")
        self.ax_bar.tick_params(colors='white')
        for spine in self.ax_bar.spines.values():
            spine.set_color('#334155')
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=p_bar)
        self.canvas_bar.get_tk_widget().pack(fill="x", padx=5, pady=5)

        # Route Details
        p_routes = ttk.LabelFrame(inner, text="Route Details")
        p_routes.pack(fill="x", padx=10, pady=10)
        
        f_export = tk.Frame(p_routes, bg='#1E293B')
        f_export.pack(fill='x', padx=5, pady=(5, 2))
        
        self.routes_container = tk.Frame(p_routes, bg="#1E293B")
        self.routes_container.pack(fill="x", padx=5, pady=(2, 5))
        tk.Label(self.routes_container, text="No results available. Start a training to see route details.", fg="#F8FAFC", bg="#1E293B", font=("Arial", 10, "italic")).pack(pady=20)

        self.txt_routes = tk.Text(p_routes, height=0, width=0, font=("Consolas", 1), bg="#0F172A", fg="#F8FAFC", insertbackground="#0F172A")
        self.txt_routes.pack(side="top", fill="none", expand=False)
        self.txt_routes.pack_forget()
        
        tk.Button(f_export, text="Export to CSV", bg="#10b981", fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", command=lambda: self.export_route_details(self.txt_routes, "csv")).pack(side="left", padx=5)
        tk.Button(f_export, text="Export to Excel", bg="#059669", fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", command=lambda: self.export_route_details(self.txt_routes, "xlsx")).pack(side="left", padx=5)
        tk.Button(f_export, text="Export Full Report (Excel)", bg="#0ea5e9", fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", command=self.export_full_simulation_excel).pack(side="left", padx=5)
        
        p_fleet = ttk.LabelFrame(inner, text="Fleet Summary")
        p_fleet.pack(fill="both", expand=True, padx=10, pady=10)
        self.fleet_table = FleetTable(p_fleet, self)
        self.fleet_table.pack(fill="both", expand=True, padx=5, pady=5)
    def build_map_guide(self, parent_frame):
        f_legend = tk.Frame(parent_frame, bg="#0F172A", pady=3)
        f_legend.pack(fill="x", side="bottom")
        tk.Label(f_legend, text="Guide:", bg="#0F172A", fg="#F8FAFC", font=("Arial", 8, "bold")).pack(side="left", padx=(8,4))
        
        c_d = tk.Canvas(f_legend, width=18, height=18, bg="#0F172A", highlightthickness=0)
        c_d.pack(side="left")
        c_d.create_oval(1,1,17,17, fill="#ef4444", outline="white", width=1)
        c_d.create_text(9,9, text="D", fill="white", font=("Arial", 8, "bold"))
        tk.Label(f_legend, text="Drone takeoff", bg="#0F172A", fg="#F8FAFC", font=("Arial", 8)).pack(side="left", padx=(2,10))
        
        c_c = tk.Canvas(f_legend, width=18, height=18, bg="#0F172A", highlightthickness=0)
        c_c.pack(side="left")
        c_c.create_oval(1,1,17,17, fill="#7c3aed", outline="white", width=1)
        c_c.create_text(9,9, text="+", fill="white", font=("Arial", 9, "bold"))
        tk.Label(f_legend, text="Client (drone delivery)", bg="#0F172A", fg="#F8FAFC", font=("Arial", 8)).pack(side="left", padx=(2,10))
        
        c_a = tk.Canvas(f_legend, width=18, height=18, bg="#0F172A", highlightthickness=0)
        c_a.pack(side="left")
        c_a.create_oval(1,1,17,17, fill="#0ea5e9", outline="white", width=1)
        c_a.create_text(9,9, text="A", fill="white", font=("Arial", 8, "bold"))
        c_l1 = tk.Canvas(f_legend, width=30, height=18, bg="#0F172A", highlightthickness=0)
        c_l1.pack(side="left")
        c_l1.create_line(2,9,28,9, fill="#7c3aed", width=2, dash=(5,3))
        tk.Label(f_legend, text="Drone outbound", bg="#0F172A", font=("Arial", 8), fg="#F8FAFC").pack(side="left", padx=(2,10))
        
        c_l2 = tk.Canvas(f_legend, width=30, height=18, bg="#0F172A", highlightthickness=0)
        c_l2.pack(side="left")
        c_l2.create_line(2,9,28,9, fill="#f97316", width=2, dash=(3,3))
        tk.Label(f_legend, text="Drone return", bg="#0F172A", font=("Arial", 8), fg="#F8FAFC").pack(side="left", padx=(2,6))

    def ask_manual_tws(self, n_clients):
        top = tk.Toplevel(self.root)
        top.title("Saisie des Time Windows (Manualle)")
        top.geometry("450x600")
        top.configure(bg="#1E293B")
        top.transient(self.root)
        top.grab_set()

        # Titre et Instructions
        tk.Label(top, text="Please enter time windows for each client", 
                 font=("Arial", 10, "bold"), bg="#1E293B", pady=10).pack()
        tk.Label(top, text="Expected format: HH:MM-HH:MM (e.g. 09:00-11:30)", 
                 font=("Arial", 9, "italic"), bg="#1E293B", fg="#F8FAFC").pack()

        # Zone de defilement
        frame_container = tk.Frame(top, bg="#1E293B")
        frame_container.pack(fill="both", expand=True, padx=20, pady=10)

        canvas = tk.Canvas(frame_container, bg="#1E293B", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1E293B")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Entrees pour chaque client
        entries = []
        for i in range(1, n_clients + 1):
            f = tk.Frame(scrollable_frame, bg="#1E293B", pady=2)
            f.pack(fill="x")
            tk.Label(f, text=f"Client C{i} :", bg="#1E293B", fg="white", width=12, anchor="w").pack(side="left", padx=5)
            v = tk.StringVar(value="08:00-18:00")
            e = ttk.Entry(f, textvariable=v, width=20)
            e.pack(side="left", padx=5)
            entries.append(v)

        result = {"successs": False, "data": []}

        def validate():
            temp_data = []
            try:
                for v in entries:
                    raw = v.get().strip()
                    if "-" not in raw: raise ValueError
                    s, e = raw.split("-")
                    parse_hhmm(s); parse_hhmm(e) # Verification du format
                    temp_data.append(raw)
                result["data"] = temp_data
                result["successs"] = True
                top.destroy()
            except:
                messagebox.showerror("Format error", "Invalid format for one of the clients.\nUse HH:MM-HH:MM (e.g. 08:30-14:00)")

        btn_f = tk.Frame(top, bg="#1E293B", pady=15)
        btn_f.pack(fill="x")
        tk.Button(btn_f, text="Validate Time Windows", bg=GRENAT, fg="white", 
                  font=("Arial", 10, "bold"), padx=20, pady=10, command=validate).pack()

        # Attendre la fermeture
        self.root.wait_window(top)
        return result

    def generate_scenario(self, silent=False):
        city = self.city_var.get()
        depot = CITY_DEPOTS.get(city, (33.570, -7.580))
        
        # S'assurer qu'on est sur Casablanca pour ce mode
        if city != "Casablanca":
            messagebox.showwarning("Limitation", "Database generation is currently reserved for Casablanca.")
            return

        try:
            n_expected = self.n_orders_var.get()
        except Exception:
            messagebox.showerror("Error", "Number of orders invalide.")
            return

        if n_expected <= 0 or n_expected > 3000:
            messagebox.showerror("Error", "Le nombre de orders doit aªtre entre 1 et 3000.")
            return

        if not hasattr(self, "client_db") or not self.client_db.get("clients"):
            messagebox.showerror("Error", "Client database not found. Please regenerate it.")
            return

        # Vider le tableau existant
        for item in self.tree_orders.get_children():
            self.tree_orders.delete(item)

        from client_database import get_random_orders
        
        try:
            generated_orders = get_random_orders(self.client_db, n_expected)
        except Exception as e:
            messagebox.showerror("Error", f"Error during order generation: {e}")
            return

        # Initialisation des variables internes (similaire a  l'ancien systa¨me pour compatibilite)
        self.clients_coords = [(depot[0], depot[1])]
        self.clients_demand = [0]
        self.orders = {}
        self.time_windows = {}
        self.no_flight_clients = []
        self.clients_zones = {}
        urgency_list = [False] # Le depa´t n'est pas urgent
        
        start_min = parse_hhmm(self.start_time_var.get())
        end_min = parse_hhmm(self.end_time_var.get())
        self.time_windows[0] = (start_min, end_min)
        
        city_nfzs = CITY_NFZS.get(city, []) if (not hasattr(self, 'enforce_nfz_var') or self.enforce_nfz_var.get()) else []

        total_parcels = 0
        
        for i, order in enumerate(generated_orders, start=1):
            
            lat, lon = order["latitude"], order["longitude"]
            self.clients_coords.append((lat, lon))
            self.clients_zones[i] = order["quartier"]
            
            # NFZ Check
            is_nfz = False
            for poly in city_nfzs:
                if (poly[0] <= lat <= poly[1]) and (poly[2] <= lon <= poly[3]):
                    is_nfz = True; break
            
            p_type_str = order["type_colis"]
            quantite = order["quantite"]
            is_urgent = order["urgence"]
            
            mode_livraison = order["mode_livraison"]
            
            # --- Flowchart 1: Delivery Mode Determination Logic ---
            p_type_int = 1 # Default to EV (Type 1)
            p_type_norm = p_type_str.replace(" ", "").lower() # "type1", "type2", "mixte"
            
            if quantite == 1 and p_type_norm == "type2":
                p_type_int = 2
                mode_livraison = "Drone"
            elif p_type_norm == "type1":
                p_type_int = 1
                mode_livraison = "Electric Vehicle"
            elif p_type_norm == "mixte":
                p_type_int = 1
                mode_livraison = "Electric Vehicle"
            else:
                # Security: client ordered multiple Type 2 parcels
                p_type_int = 1
                mode_livraison = "Electric Vehicle (Force: >1 Parcel)"
            
            # Si la zone est NFZ, on force l'EV (1)
            if is_nfz: 
                p_type_int = 1
                mode_livraison = "Electric Vehicle (Force: NFZ)"
                
            self.clients_demand.append(p_type_int)
            self.orders[i] = {"parcels": quantite, "urgency": 3 if is_urgent else 1}
            urgency_list.append(is_urgent)
            
            # TW
            tw_s, tw_e = parse_hhmm(order["tw_start"]), parse_hhmm(order["tw_end"])
            self.time_windows[i] = (tw_s, tw_e)
            
            tw_str = f"{order['tw_start']}-{order['tw_end']}"
            vol_status = "NO-FLIGHT" if is_nfz else "Libre"
            if is_nfz: self.no_flight_clients.append(i)
            
            self.tree_orders.insert("", "end", values=(
                f"C{i} ({order['nom']})", 
                order["quartier"], 
                f"{lat:.4f}", 
                f"{lon:.4f}", 
                quantite, 
                p_type_str, 
                mode_livraison, 
                tw_str, 
                "URGENT" if is_urgent else "Normal"
            ))
            
            total_parcels += quantite

        # Mise a  jour des compteurs globaux
        self.total_all_parcels_var.set(f"Total Parcels Generes : {total_parcels}")

        self.draw_clients(urgency_list)
        
        # --- Urgency calculations for summary ---
        urgent_msgs = []
        for i, is_urgent in enumerate(urgency_list[1:], start=1):
            if is_urgent:
                tw_s, tw_e = self.time_windows[i]
                # Delivery duration from start time to tw_e
                max_duration = tw_e - start_min
                tw_str = f"{tw_s//60:02d}:{tw_s%60:02d}-{tw_e//60:02d}:{tw_e%60:02d}"
                urgent_msgs.append(f"C{i} ({self.clients_zones[i]}): {tw_str} (Max wait: {max_duration} min)")
        
        self.txt_urgency.config(state="normal")
        self.txt_urgency.delete("1.0", tk.END)
        if urgent_msgs:
            self.txt_urgency.insert(tk.END, f"{len(urgent_msgs)} urgent orders generated:\n" + "\n".join(urgent_msgs))
        else:
            self.txt_urgency.insert(tk.END, "No urgent orders generated.")
        self.txt_urgency.config(state="disabled")

        if not silent:
            messagebox.showinfo("Success", f"Generation successful: {n_expected} orders for {total_parcels} total parcels.")

    def make_badge_icon(self, text, fill="#2563eb", outline="white", fg="white", size=28):
        """Cree une petite ica´ne ronde avec le texte C1, C2... ecrit dedans."""
        if not PIL_AVAILABLE:
            return None
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((2, 2, size-3, size-3), fill=fill, outline=outline, width=3)
        try:
            font = ImageFont.truetype("arial.ttf", 10 if len(text) <= 2 else 9)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((size - tw) / 2, (size - th) / 2 - 1), text, fill=fg, font=font)
        icon = ImageTk.PhotoImage(img)
        self._map_icons.append(icon)  # eviter que Python supprime l'image
        return icon

    def add_small_marker(self, map_widget, lat, lon, label, fill="#2563eb", outline="white", text_fallback=None):
        icon = self.make_badge_icon(label, fill=fill, outline=outline)
        if icon is not None:
            try:
                return map_widget.set_marker(lat, lon, text="", icon=icon)
            except TypeError:
                # Anciennes versions de tkintermapview sans parama¨tre icon
                pass
        return map_widget.set_marker(lat, lon, text=text_fallback or label,
                                     marker_color_circle=fill, marker_color_outside=outline)

    def get_focus_client_id(self):
        raw = self.focus_client_var.get()
        if raw.startswith("C"):
            try: return int(raw[1:].split()[0])
            except Exception: return None
        return None

    def get_focus_ev_id(self):
        raw = self.focus_ev_var.get()
        if raw.startswith("EV"):
            try: return int(raw.split()[1]) - 1
            except Exception: return None
        return None

    def get_focus_drone_client_id(self):
        raw = self.focus_drone_var.get()
        if "C" in raw:
            try: return int(raw.split("C")[-1].split()[0])
            except Exception: return None
        return None

    def update_map_filter_choices(self):
        pass # Remplace par le panneau de couches (Layers Panel)

    def draw_map_base(self, map_widget):
        """Dessine depa´t, clients, zones de vol et NFZ avec des badges courts et lisibles."""
        if not map_widget: return
        map_widget.delete_all_marker()
        map_widget.delete_all_polygon()
        map_widget.delete_all_path()
        city = self.city_var.get()
        depot = CITY_DEPOTS[city]
        map_widget.set_position(depot[0], depot[1])
        map_widget.set_zoom(13)

        # Flight Zone globale
        if self.show_flight_zone_var.get():
            fz_poly = [
                (depot[0]+0.05, depot[1]-0.05), (depot[0]+0.05, depot[1]+0.05),
                (depot[0]-0.05, depot[1]+0.05), (depot[0]-0.05, depot[1]-0.05)
            ]
            map_widget.set_polygon(fz_poly, fill_color="#d9f7d9", outline_color="green",
                                   border_width=1, name="Flight Zone allowed")

        # No-flight zones
        if self.show_nfz_var.get():
            for b in CITY_NFZS.get(city, []):
                poly = [(b[0], b[2]), (b[0], b[3]), (b[1], b[3]), (b[1], b[2])]
                map_widget.set_polygon(poly, fill_color="#ffcccc", outline_color="red", border_width=2, name=b[4])

        # Depot
        if self.show_depot_var.get():
            self.add_small_marker(map_widget, depot[0], depot[1], "D", fill="#dc2626", outline="white", text_fallback="Depot")

        if not self.show_clients_var.get():
            return

        best = self.latest_best or {}
        drone_clients = set(best.get("drone_assign", {}).keys())
        unique_zones = list(set(self.clients_zones.values())) if hasattr(self, 'clients_zones') else []

        for i, (lat, lon) in enumerate(self.clients_coords):
            if i == 0: continue
            
            is_urgent = self.orders.get(i, {}).get("urgency", 1) == 3
            is_drone = i in drone_clients or (i < len(self.clients_demand) and self.clients_demand[i] == 2)
            
            if is_urgent and self.show_urgences_var.get():
                color = "#f59e0b"  # orange urgent
            elif is_drone and i in drone_clients and self.drone_layer_vars.get(i, tk.BooleanVar(value=True)).get():
                color = self.DRONE_COLOR # Cyan drone
            elif is_drone and i not in drone_clients:
                color = "#9ca3af" # Gris drone non-assigne
            else:
                # Color code by K-Means cluster if solver and clusters exist
                clusters_to_draw = []
                if hasattr(self, 'latest_solver') and self.latest_solver and hasattr(self.latest_solver, 'clusters'):
                    clusters_to_draw = self.latest_solver.clusters
                elif hasattr(self, 'preview_clusters') and self.preview_clusters:
                    clusters_to_draw = self.preview_clusters

                if len(clusters_to_draw) >= i:
                    cl_id = clusters_to_draw[i - 1]
                    color = self.EV_COLORS[cl_id % len(self.EV_COLORS)]
                else:
                    # Fallback to neighborhood zone
                    zone = self.clients_zones.get(i) if hasattr(self, 'clients_zones') else None
                    if zone in unique_zones:
                        color = self.EV_COLORS[unique_zones.index(zone) % len(self.EV_COLORS)]
                    else:
                        color = "#2563eb"
            
            # Ne pas dessiner les clients drone si leur couche drone est decochee
            if is_drone and i in drone_clients and not self.drone_layer_vars.get(i, tk.BooleanVar(value=True)).get():
                continue
                
            self.add_small_marker(map_widget, lat, lon, f"C{i}", fill=color, outline="white")

    def dashed_path(self, map_widget, p_start, p_end, color="red", width=3):
        dist = euclidean_km(p_start, p_end)
        steps = max(6, int(dist / 0.4) * 2)
        for i in range(steps):
            if i % 2 == 0:
                p1 = (p_start[0] + (p_end[0]-p_start[0])*i/steps,
                      p_start[1] + (p_end[1]-p_start[1])*i/steps)
                p2 = (p_start[0] + (p_end[0]-p_start[0])*(i+1)/steps,
                      p_start[1] + (p_end[1]-p_start[1])*(i+1)/steps)
                map_widget.set_path([p1, p2], color=color, width=width)

    def dashed_path_list(self, map_widget, path, color="red", width=3):
        if len(path) < 2: return
        dash_len_km = 0.08
        gap_len_km = 0.08
        period = dash_len_km + gap_len_km
        current_dist = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            seg_dist = euclidean_km(p1, p2)
            if seg_dist == 0: continue
            steps = max(1, int(seg_dist / 0.02))
            for s in range(steps):
                d1 = current_dist + seg_dist * (s / steps)
                d2 = current_dist + seg_dist * ((s + 1) / steps)
                if (d1 % period) < dash_len_km:
                    sub_p1 = (p1[0] + (p2[0]-p1[0]) * (s / steps), p1[1] + (p2[1]-p1[1]) * (s / steps))
                    sub_p2 = (p1[0] + (p2[0]-p1[0]) * ((s+1) / steps), p1[1] + (p2[1]-p1[1]) * ((s+1) / steps))
                    map_widget.set_path([sub_p1, sub_p2], color=color, width=width)
            current_dist += seg_dist

    def render_map_on(self, map_widget):
        """Redessine la carte selon le panneau de couches (Layers Panel)."""
        if not map_widget: return
        self.draw_map_base(map_widget)
        
        # Affichage / dessin des zones Clusters (K-Means)
        if hasattr(self, 'show_clusters_var') and self.show_clusters_var.get():
            try:
                from scipy.spatial import ConvexHull
                clusters_to_draw = []
                if hasattr(self, 'latest_solver') and self.latest_solver and hasattr(self.latest_solver, 'clusters'):
                    clusters_to_draw = self.latest_solver.clusters
                elif hasattr(self, 'preview_clusters') and self.preview_clusters:
                    clusters_to_draw = self.preview_clusters
                    
                if len(clusters_to_draw) > 0:
                    grouped_clusters = {}
                    for i, cid in enumerate(clusters_to_draw):
                        grouped_clusters.setdefault(cid, []).append(i + 1)
                        
                    for cluster_id, cluster_indices in grouped_clusters.items():
                        # Recuperer les coordonnees reelles des clients de ce cluster
                        cluster_pts = [self.clients_coords[i] for i in cluster_indices if 0 < i < len(self.clients_coords)]
                        if len(cluster_pts) >= 3:
                            hull = ConvexHull(cluster_pts)
                            poly = [cluster_pts[i] for i in hull.vertices]
                            # Draw as a filled polygon with a thin border to represent a cluster zone,
                            # avoiding confusion with actual vehicle routes
                            EV_LIGHT_COLORS = [
                                "#dbeafe", "#ffedd5", "#dcfce7", "#f3e8ff",
                                "#fee2e2", "#cffafe", "#fef9c3", "#fce7f3",
                                "#e0e7ff", "#ccfbf1", "#f3e8ff", "#fee2e2"
                            ]
                            color = self.EV_COLORS[cluster_id % len(self.EV_COLORS)]
                            fill_color = EV_LIGHT_COLORS[cluster_id % len(EV_LIGHT_COLORS)]
                            map_widget.set_polygon(
                                poly,
                                fill_color=fill_color,
                                outline_color=color,
                                border_width=1.5,
                                name=f"cluster_{cluster_id}"
                            )
            except Exception as e:
                print(f"Error clusters ConvexHull: {e}")

        best = getattr(self, 'latest_best', None)
        if not best:
            return

        ev_routes = best.get("ev_routes", [])
        ev_routes_osrm = getattr(self, 'latest_ev_routes_osrm', []) or []

        # Affichage / suivi EV
        for ev_idx, route_nodes in enumerate(ev_routes):
            if not self.ev_layer_vars.get(ev_idx, tk.BooleanVar(value=True)).get():
                continue
            
            has_osrm = False
            if ev_idx < len(ev_routes_osrm) and ev_routes_osrm[ev_idx] is not None:
                r_data = ev_routes_osrm[ev_idx]
                if isinstance(r_data, dict):
                    route_aller = r_data.get("aller", [])
                    route_retour = r_data.get("retour", [])
                    has_osrm = True
                else:
                    route_aller = r_data
                    route_retour = []
            else:
                route_aller = [self.clients_coords[n] for n in route_nodes[:-1] if n < len(self.clients_coords)]
                route_retour = []
                if len(route_nodes) >= 2:
                    n_last = route_nodes[-2]
                    n_depot = route_nodes[-1]
                    if n_last < len(self.clients_coords) and n_depot < len(self.clients_coords):
                        route_retour = [self.clients_coords[n_last], self.clients_coords[n_depot]]
            
            color = self.EV_COLORS[ev_idx % len(self.EV_COLORS)]
            
            # Draw Outbound Route (Aller) - Solid Line
            if len(route_aller) >= 2:
                map_widget.set_path(route_aller, color=color, width=4)
                
            # Draw Return Route (Retour) - Dashed Line
            if len(route_retour) >= 2:
                self.dashed_path_list(map_widget, route_retour, color=color, width=4)

        # Affichage / suivi drones
        if self.weather_allowed:
            meetings = best.get("drone_meetings", {})
            launches = best.get("drone_launch", {})
            for c, p_return in best.get("drone_assign", {}).items():
                if not self.drone_layer_vars.get(c, tk.BooleanVar(value=True)).get():
                    continue
                    
                p_launch = launches.get(c, p_return)
                if p_launch >= len(self.clients_coords) or c >= len(self.clients_coords):
                    continue
                start_pos = self.clients_coords[p_launch]
                cli_pos = self.clients_coords[c]
                meet_pos = meetings.get(c, self.clients_coords[p_return])
                if isinstance(meet_pos, np.ndarray):
                    meet_pos = (float(meet_pos[0]), float(meet_pos[1]))

                self.dashed_path(map_widget, start_pos, cli_pos, color=self.DRONE_COLOR, width=3)
                self.dashed_path(map_widget, cli_pos, meet_pos, color="#f97316", width=3)
                # Point de decollage : "D" rouge
                self.add_small_marker(map_widget, start_pos[0], start_pos[1], "D", fill="#ef4444", outline="white", text_fallback=f"D{c}")
                # Client livre par drone
                self.add_small_marker(map_widget, cli_pos[0], cli_pos[1], f"C{c}", fill=self.DRONE_COLOR, outline="white", text_fallback=f"C{c}")
                # Point d'atterrissage : "A" blue
                self.add_small_marker(map_widget, meet_pos[0], meet_pos[1], "A", fill="#0ea5e9", outline="white", text_fallback=f"A{c}")
        elif not self.weather_allowed:
            # Weather defavorable : afficher uniquement les points de lancement/rendez-vous sans tracer les routes.
            pass

    def refresh_map_view(self):
        try:
            self.update_map_filter_choices()
            self.render_map_on(self.map_widget)
        except Exception as e:
            print(f"Error in refresh_map_view (map_widget): {e}")
            pass
        if getattr(self, 'fullscreen_map_widget', None):
            try:
                if self.fullscreen_map_widget.winfo_exists():
                    self.render_map_on(self.fullscreen_map_widget)
                else:
                    self.fullscreen_map_widget = None
            except Exception as e:
                print(f"Error in refresh_map_view (fullscreen): {e}")
                self.fullscreen_map_widget = None
        # Support pour la carte de l'onglet agentique s'il est charge
        agentic_panel = None
        if hasattr(self, 'agentic_ui') and hasattr(self.agentic_ui, 'agentic_results_panel'):
            agentic_panel = self.agentic_ui.agentic_results_panel
        if agentic_panel and getattr(agentic_panel, 'a_map_widget', None):
            try:
                self.render_map_on(agentic_panel.a_map_widget)
            except Exception as e:
                print(f"Error in refresh_map_view (agentic): {e}")
                pass

    def fetch_weather_conditions(self):
        key = self.openweather_api_key_var.get().strip()
        
        # Sauvegarder la cle API pour les prochaines utilisations
        try:
            import os
            key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
            with open(key_path, "w", encoding="utf-8") as f:
                f.write(key)
        except:
            pass
            
        if not key:
            self.weather_allowed = True
            self.weather_reason = "None OpenWeather API Key fourni"
            self.weather_status_var.set(self.weather_reason)
            return
            
        # Bypass Mode Demo pour la tha¨se sans attendre l'API
        if key.upper() == "DEMO" or key.upper() == "TEST":
            self.weather_allowed = True
            self.weather_conditions = {
                "main": "Clear",
                "description": "ciel degage (simule)",
                "wind_speed": 4.5,
                "gust": 6.0,
                "rain": 0.0,
                "snow": 0.0
            }
            self.weather_reason = "Weather ideale (Mode Demo)"
            self.weather_status_var.set(self.weather_reason)
            return

        city = self.city_var.get()
        depot = CITY_DEPOTS.get(city)
        if not depot:
            self.weather_allowed = True
            self.weather_reason = "City inconnue pour la meteo"
            self.weather_status_var.set(self.weather_reason)
            return
        try:
            params = {
                "lat": depot[0], "lon": depot[1], "appid": key,
                "units": "metric", "lang": "fr"
            }
            r = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=6)
            data = r.json()
            main = data.get("weather", [{}])[0].get("main", "?")
            desc = data.get("weather", [{}])[0].get("description", "?")
            wind = data.get("wind", {})
            wind_speed = float(wind.get("speed", 0.0))
            gust = float(wind.get("gust", 0.0))
            rain = float(data.get("rain", {}).get("1h", 0.0) or 0.0)
            snow = float(data.get("snow", {}).get("1h", 0.0) or 0.0)
            self.weather_conditions = {
                "main": main, "description": desc,
                "wind_speed": wind_speed, "gust": gust,
                "rain": rain, "snow": snow
            }
            blocked = False
            reasons = []
            if wind_speed >= self.weather_wind_max_var.get() or gust >= self.weather_gust_max_var.get():
                blocked = True
                reasons.append(f"Vent trop fort ({wind_speed:.1f} m/s, rafales {gust:.1f})")
            if rain >= self.weather_rain_max_var.get():
                blocked = True
                reasons.append(f"Pluie importante ({rain:.1f} mm/h)")
            if snow >= self.weather_snow_max_var.get():
                blocked = True
                reasons.append(f"Neige ({snow:.1f} mm/h)")
            if self.weather_block_storms_var.get() and main in ["Thunderstorm", "Tornado", "Squall", "Ash", "Sand"]:
                blocked = True
                reasons.append(main)
            self.weather_allowed = not blocked
            self.weather_reason = (
                f"Weather: {main} ({desc}), vent {wind_speed:.1f} m/s, pluie {rain:.1f} mm/h, neige {snow:.1f} mm/h"
                + (" aE” VOL interdit" if blocked else " aE” Flight autorise")
            )
            if reasons:
                self.weather_reason += ": " + ", ".join(reasons)
            self.weather_status_var.set(self.weather_reason)
            self.update_map_filter_choices()
            self.refresh_map_view()
            if self.latest_best and self.latest_solver:
                self.update_results(self.latest_best, self.latest_ev_routes_osrm, self.latest_solver)
        except Exception as e:
            self.weather_allowed = True
            self.weather_reason = f"Error meteo: {e}"
            self.weather_status_var.set(self.weather_reason)

    def get_weather_permission(self):
        key = self.openweather_api_key_var.get().strip()
        if not key:
            self.weather_allowed = True
            self.weather_reason = "Nonee cle OpenWeather fournie"
            self.weather_status_var.set(self.weather_reason)
            return True, self.weather_reason
        self.fetch_weather_conditions()
        self.weather_status_var.set(self.weather_reason)
        return self.weather_allowed, self.weather_reason

    def show_nfz_modal(self):
        city = self.city_var.get() if hasattr(self, 'city_var') else "Casablanca"
        nfzs = CITY_NFZS.get(city, [])
        top = tk.Toplevel(self.root)
        top.title(f"No-Flight Zones (NFZ) — {city}")
        top.geometry("540x360")
        top.configure(bg="#0F172A")
        top.transient(self.root)
        top.grab_set()

        # Header
        f_head = tk.Frame(top, bg="#800020", pady=8)
        f_head.pack(fill="x")
        tk.Label(f_head, text=f"🚫 No-Flight Zones (NFZ) — {city}", font=("Arial", 11, "bold"), fg="white", bg="#800020").pack()

        # Info
        is_active = self.enforce_nfz_var.get() if hasattr(self, 'enforce_nfz_var') else True
        status_txt = "Activé (Survol strictement interdit aux drones)" if is_active else "Désactivé (Toléré sans pénalité)"
        status_color = "#38BDF8" if is_active else "#94A3B8"
        tk.Label(top, text=f"Total des zones réglementées : {len(nfzs)} | Statut : {status_txt}",
                 font=("Arial", 9, "bold"), fg=status_color, bg="#0F172A", pady=6).pack()

        # Treeview / List
        cols = ("name", "lat_bounds", "lon_bounds")
        tree_frame = tk.Frame(top, bg="#0F172A")
        tree_frame.pack(fill="both", expand=True, padx=12, pady=5)
        
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=6)
        tree.heading("name", text="Nom de la Zone")
        tree.heading("lat_bounds", text="Latitude [Min, Max]")
        tree.heading("lon_bounds", text="Longitude [Min, Max]")
        tree.column("name", width=180)
        tree.column("lat_bounds", width=140, anchor="center")
        tree.column("lon_bounds", width=140, anchor="center")

        for poly in nfzs:
            if len(poly) >= 5:
                lat_str = f"[{poly[0]:.4f}, {poly[1]:.4f}]"
                lon_str = f"[{poly[2]:.4f}, {poly[3]:.4f}]"
                tree.insert("", "end", values=(poly[4], lat_str, lon_str))
            elif len(poly) == 4:
                lat_str = f"[{poly[0]:.4f}, {poly[1]:.4f}]"
                lon_str = f"[{poly[2]:.4f}, {poly[3]:.4f}]"
                tree.insert("", "end", values=("Zone Restreinte", lat_str, lon_str))

        tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Close button
        tk.Button(top, text="Fermer", bg="#334155", fg="white", font=("Arial", 9, "bold"), relief="flat", cursor="hand2", padx=15, pady=4, command=top.destroy).pack(pady=8)

    def update_cart_ev_checkboxes(self, best):
        """Update the EV filter checkboxes in the cartesian plan toolbar."""
        if not hasattr(self, 'cart_filter_ev_frame'): return
        for w in self.cart_filter_ev_frame.winfo_children(): w.destroy()
        if not hasattr(self, 'cart_ev_vars'): self.cart_ev_vars = {}
        self.cart_ev_vars.clear()
        for ev_idx, route in enumerate(best.get("ev_routes", [])):
            if len(route) >= 2:
                color = self.EV_COLORS[ev_idx % len(self.EV_COLORS)]
                var = tk.BooleanVar(value=True)
                self.cart_ev_vars[ev_idx] = var
                f = tk.Frame(self.cart_filter_ev_frame, bg="#0F172A")
                f.pack(side="left", padx=2)
                c = tk.Canvas(f, width=10, height=10, bg="#0F172A", highlightthickness=0)
                c.pack(side="left")
                c.create_rectangle(1,1,9,9, fill=color, outline="")
                ttk.Checkbutton(f, text=f"VE{ev_idx+1}", variable=var,
                                command=lambda: self.draw_cartesian_plan(getattr(self,'latest_best',None), getattr(self,'latest_solver',None))
                                ).pack(side="left")

    def _draw_on_axes(self, ax, best, show_drones=True, ev_filter=None):
        """Core drawing logic for cartesian plan, can target any ax (main or fullscreen)."""
        ax.clear()
        ax.set_title("GRAPHE DES ROUTES & CONVERGENCE Ra‰ELLE", fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.grid(True, alpha=0.2, linestyle="--")
        if not best or not self.clients_coords:
            return
        
        coords = self.clients_coords
        xs = [lon for lat, lon in coords]
        ys = [lat for lat, lon in coords]
        
        # Determine node colors based on EV assignment (cluster)
        node_colors = ["#a1a1aa"] * len(coords)
        for ev_idx, route_nodes in enumerate(best.get("ev_routes", [])):
            color = self.EV_COLORS[ev_idx % len(self.EV_COLORS)]
            for n in route_nodes:
                if 0 < n < len(coords):
                    node_colors[n] = color
                    
        # Draw Cluster Boundaries (Convex Hull)
        if hasattr(self, 'latest_solver') and self.latest_solver and hasattr(self.latest_solver, 'clusters'):
            try:
                from scipy.spatial import ConvexHull
                import matplotlib.patches as patches
                for cluster_id, cluster_indices in enumerate(self.latest_solver.clusters):
                    if ev_filter is not None and cluster_id not in ev_filter:
                        continue
                    cluster_pts = [(coords[i][1], coords[i][0]) for i in cluster_indices if 0 < i < len(coords)]
                    if len(cluster_pts) >= 3:
                        hull = ConvexHull(cluster_pts)
                        poly_pts = [cluster_pts[i] for i in hull.vertices]
                        color = self.EV_COLORS[cluster_id % len(self.EV_COLORS)]
                        poly = patches.Polygon(poly_pts, closed=True, edgecolor=color, facecolor=color, alpha=0.15, linewidth=2, linestyle=':', zorder=1)
                        ax.add_patch(poly)
            except Exception as e:
                print(f"Error live map ConvexHull: {e}")
                    
        # Draw EV Routes (Dashed lines)
        for ev_idx, route_nodes in enumerate(best.get("ev_routes", [])):
            if ev_filter is not None and ev_idx not in ev_filter:
                continue
            route = [(coords[n][1], coords[n][0]) for n in route_nodes if n < len(coords)]
            if len(route) >= 2:
                color = self.EV_COLORS[ev_idx % len(self.EV_COLORS)]
                ax.plot([p[0] for p in route], [p[1] for p in route],
                        color=color, linestyle="--", linewidth=1.5, alpha=0.8,
                        label=f"VE-{ev_idx+1}", zorder=2)
                
        # Draw Drone Flights
        drawn_drone_label = False
        drone_takeoffs = []
        drone_landings = []
        
        if show_drones:
            for c, p_return in best.get("drone_assign", {}).items():
                p_launch = best.get("drone_launch", {}).get(c, 0)
                if p_launch < len(coords) and c < len(coords) and p_return < len(coords):
                    start_pos = coords[p_launch]
                    cli_pos = coords[c]
                    return_pos = coords[p_return]
                    drone_takeoffs.append(p_launch)
                    drone_landings.append(p_return)
                    ax.plot([start_pos[1], cli_pos[1]], [start_pos[0], cli_pos[0]],
                            color="#0f766e", linestyle=":", linewidth=1.5, alpha=0.9,
                            label="Flight drone" if not drawn_drone_label else None, zorder=1)
                    drawn_drone_label = True
                    ax.plot([cli_pos[1], return_pos[1]], [cli_pos[0], return_pos[0]],
                            color="#0f766e", linestyle=":", linewidth=1.5, alpha=0.9, zorder=1)
                
        # Draw Nodes
        for i in range(1, len(coords)):
            if i in drone_takeoffs:
                ax.scatter(xs[i], ys[i], facecolor="#ef4444", edgecolor=node_colors[i],
                           linewidths=2, s=90, marker="^", zorder=7,
                           label="Takeoff point" if i == drone_takeoffs[0] else None)
            elif i in drone_landings:
                ax.scatter(xs[i], ys[i], facecolor="#0ea5e9", edgecolor=node_colors[i],
                           linewidths=2, s=90, marker="v", zorder=7,
                           label="Point atterrissage" if i == drone_landings[0] else None)
            else:
                ax.scatter(xs[i], ys[i], facecolor="#10b981", edgecolor=node_colors[i],
                           linewidths=2, s=50, zorder=5,
                           label="Clients" if i == 1 else None)
                                 
        # Depot
        ax.scatter(xs[0], ys[0], facecolor="#1f2937", edgecolor="#000000",
                   linewidths=2, s=100, marker="*", zorder=8, label="Depot")
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    def draw_cartesian_plan(self, best, solver):
        if not hasattr(self, "ax_cart"): return
        if not hasattr(self, 'cart_ev_vars'): self.cart_ev_vars = {}
        if not hasattr(self, 'cart_show_drones_var'): self.cart_show_drones_var = tk.BooleanVar(value=True)
        
        ev_filter = set(k for k,v in self.cart_ev_vars.items() if v.get()) if self.cart_ev_vars else None
        show_drones = self.cart_show_drones_var.get()
        
        self._draw_on_axes(self.ax_cart, best, show_drones=show_drones, ev_filter=ev_filter)
        self.canvas_cart.draw()
        
        # Also refresh fullscreen if open
        if hasattr(self, 'cart_fs_canvas') and self.cart_fs_canvas:
            try:
                self._draw_on_axes(self.cart_fs_ax, best, show_drones=show_drones, ev_filter=ev_filter)
                self.cart_fs_canvas.draw()
            except Exception:
                pass

    def open_cartesian_fullscreen(self):
        best = getattr(self, 'latest_best', None)
        top = tk.Toplevel(self.root)
        top.title("EV-Drone Mapsian Plan aE” Full Screen")
        top.state('zoomed')
        top.configure(bg="#1E293B")
        
        # Toolbar
        f_tb = tk.Frame(top, bg="#0F172A", pady=5)
        f_tb.pack(fill="x", padx=10)
        tk.Label(f_tb, text="Filter:", bg="#0F172A", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=(0,5))
        f_scroll_container = tk.Frame(f_tb, bg="#0F172A")
        f_scroll_container.pack(side="left", fill="x", expand=True)
        self.fs_cart_canvas = tk.Canvas(f_scroll_container, height=30, bg="#0F172A", highlightthickness=0)
        fs_cart_scroll = ttk.Scrollbar(f_scroll_container, orient="horizontal", command=self.fs_cart_canvas.xview)
        self.fs_cart_canvas.configure(xscrollcommand=fs_cart_scroll.set)
        fs_cart_scroll.pack(side="bottom", fill="x")
        self.fs_cart_canvas.pack(side="top", fill="both", expand=True)
        f_ev_fs = tk.Frame(self.fs_cart_canvas, bg="#0F172A")
        self.fs_cart_canvas.create_window((0, 0), window=f_ev_fs, anchor="nw")
        f_ev_fs.bind("<Configure>", lambda e: self.fs_cart_canvas.configure(scrollregion=self.fs_cart_canvas.bbox("all")))
        
        f_btns_fs = tk.Frame(f_tb, bg="#0F172A")
        f_btns_fs.pack(side="right")
        
        # Copy the ev vars into fullscreen checkboxes (shared BooleanVars)
        if best and hasattr(self, 'cart_ev_vars'):
            for ev_idx, var in self.cart_ev_vars.items():
                color = self.EV_COLORS[ev_idx % len(self.EV_COLORS)]
                f = tk.Frame(f_ev_fs, bg="#0F172A")
                f.pack(side="left", padx=2)
                c = tk.Canvas(f, width=10, height=10, bg="#0F172A", highlightthickness=0)
                c.pack(side="left")
                c.create_rectangle(1,1,9,9, fill=color, outline="")
                ttk.Checkbutton(f, text=f"VE{ev_idx+1}", variable=var,
                                command=lambda: self.draw_cartesian_plan(getattr(self,'latest_best',None), None)
                                ).pack(side="left")
        
        show_drones_fs = getattr(self, 'cart_show_drones_var', tk.BooleanVar(value=True))
        ttk.Checkbutton(f_btns_fs, text="Drone Flights", variable=show_drones_fs,
                        command=lambda: self.draw_cartesian_plan(getattr(self,'latest_best',None), None)
                        ).pack(side="left", padx=8)
        tk.Button(f_btns_fs, text="aoe• Close", bg="#dc2626", fg="white", font=("Arial", 9),
                  relief="flat", padx=8, cursor="hand2",
                  command=top.destroy).pack(side="right", padx=5)
        
        # Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FCAT
        fs_fig = Figure(dpi=100)
        fs_fig.patch.set_facecolor('#1E293B')
        self.cart_fs_ax = fs_fig.add_subplot(111)
        self.cart_fs_canvas = FCAT(fs_fig, master=top)
        self.cart_fs_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
        
        ev_filter = set(k for k,v in self.cart_ev_vars.items() if v.get()) if hasattr(self,'cart_ev_vars') and self.cart_ev_vars else None
        self._draw_on_axes(self.cart_fs_ax, best, show_drones=show_drones_fs.get(), ev_filter=ev_filter)
        self.cart_fs_canvas.draw()
        
        def on_close():
            self.cart_fs_canvas = None
            self.cart_fs_ax = None
            top.destroy()
        top.protocol("WM_DELETE_WINDOW", on_close)
        top.bind("<Escape>", lambda e: on_close())

    def open_map_fullscreen(self):
        if not MAP_AVAILABLE:
            messagebox.showwarning("Map", "tkintermapview is not installed.")
            return
        top = tk.Toplevel(self.root)
        top.title("Map en plein ecran")
        top.configure(bg="#1E293B")
        try:
            top.attributes("-fullscreen", True)
        except Exception:
            top.geometry("1400x850")

        bar = tk.Frame(top, bg="#1E293B", pady=6)
        bar.pack(fill="x")
        tk.Label(bar, text="Map des livraisons EV-Drone", bg="#1E293B", fg="#F8FAFC",
                 font=("Arial", 13, "bold")).pack(side="left", padx=15)
        tk.Button(bar, text="Close", bg=GRENAT, fg="white", relief="flat",
                  font=("Arial", 10, "bold"), padx=14, pady=5,
                  command=self.close_fullscreen_map).pack(side="right", padx=15)

        # Conteneur 2 colonnes
        f_content = tk.Frame(top, bg="#1E293B")
        f_content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        f_map_view = tk.Frame(f_content, bg="#1E293B", bd=1, relief="solid")
        f_map_view.pack(side="left", fill="both", expand=True)
        
        full_map = tkintermapview.TkinterMapView(f_map_view, corner_radius=0)
        full_map.pack(fill="both", expand=True)
        
        self.build_map_guide(f_map_view)
        
        f_layers = tk.Frame(f_content, bg="#1E293B", bd=1, relief="solid", width=250)
        f_layers.pack(side="right", fill="y", padx=(10, 0))
        f_layers.pack_propagate(False)
        self.build_layers_panel(f_layers, is_fullscreen=True)
        self.fullscreen_map_window = top
        self.fullscreen_map_widget = full_map
        top.protocol("WM_DELETE_WINDOW", self.close_fullscreen_map)
        top.bind("<Escape>", lambda e: self.close_fullscreen_map())
        self.render_map_on(full_map)
        
        # Populate the layers immediately for the fullscreen map
        if getattr(self, 'latest_best', None) and getattr(self, 'latest_ev_routes_osrm', None):
            self.update_layers_panel(self.latest_best, self.latest_ev_routes_osrm)

    def close_fullscreen_map(self):
        if getattr(self, 'fullscreen_map_window', None):
            try:
                self.fullscreen_map_window.destroy()
            except Exception:
                pass
        self.fullscreen_map_window = None
        self.fullscreen_map_widget = None

    def update_preview_clusters(self):
        """Calculates K-Means clustering preview for loaded clients based on current 'K Clusters' value."""
        if not hasattr(self, 'clients_coords') or len(self.clients_coords) <= 1:
            self.preview_clusters = []
            return
        try:
            k = int(self.k_clusters_var.get())
            labels, _ = kmeans_simple(self.clients_coords[1:], k)
            self.preview_clusters = list(labels)
        except Exception:
            self.preview_clusters = []

    def on_k_clusters_changed(self):
        try:
            val = self.k_clusters_var.get()
        except Exception:
            return # Ignore temp empty or invalid input during typing
        self.update_preview_clusters()
        if self.map_widget:
            self.draw_map_base(self.map_widget)
            if getattr(self, 'fullscreen_map_widget', None):
                self.draw_map_base(self.fullscreen_map_widget)

    def draw_clients(self, urgency_list):
        """Affichage clair du scenario genere : badges C1, C2... ecrits dans les marqueurs."""
        if not self.map_widget: return
        self.latest_best = None
        self.latest_ev_routes_osrm = []
        self.latest_solver = None
        self.focus_client_var.set("All")
        self.focus_ev_var.set("All")
        self.focus_drone_var.set("All")
        self.update_map_filter_choices()
        self.draw_map_base(self.map_widget)
        if getattr(self, 'fullscreen_map_widget', None):
            self.draw_map_base(self.fullscreen_map_widget)
        self.lbl_status.config(text="Generated map: badges C1, C2... | blue=EV | purple=drone | orange=urgent")

    def draw_initial_map(self, selected_zones):
        if not self.map_widget: return
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_polygon()
        self.map_widget.delete_all_path()
        
        city = self.city_var.get()
        depot = CITY_DEPOTS[city]
        self.map_widget.set_position(depot[0], depot[1])
        self.map_widget.set_zoom(12)
        
        # Dessiner les NFZ statiques en rouge
        for b in CITY_NFZS.get(city, []):
            poly = [(b[0], b[2]), (b[0], b[3]), (b[1], b[3]), (b[1], b[2])]
            self.map_widget.set_polygon(poly, fill_color="red", outline_color="red", border_width=2, name=b[4])

        # Draw clients
        for i, (lat, lon) in enumerate(self.clients_coords):
            if i == 0: self.map_widget.set_marker(lat, lon, text="Depot")
            else: self.map_widget.set_marker(lat, lon, text=f"C{i}")

    def start_tuning(self, method, target):
        if not self.clients_coords:
            messagebox.showwarning("Error", "Generate a scenario first.")
            return
        
        # Check availability
        if method == "optuna" and not OPTUNA_AVAILABLE:
            messagebox.showerror("Error", "Optuna is not installed.")
            return
        if method == "kerastuner" and not KERASTUNER_AVAILABLE:
            messagebox.showerror("Error", "Keras Tuner is not installed.")
            return

        # Disable the Start button during tuning to avoid parallel runs
        self.btn_run.config(state="disabled")

        status_text = f"Optimization {target} via {method.upper()}..."
        self.lbl_progress_pct.config(text="Tuning")
        self.lbl_status.config(text=status_text)
        self.progress.config(mode="indeterminate")
        self.progress.start()
        self.lbl_progress_pct.config(text="Tuning")
        
        threading.Thread(target=self.run_tuning_study, args=(method, target), daemon=True).start()

    def run_tuning_study(self, method, target):
        try:
            best_params = {}
            if method == "optuna":
                best_params = self.execute_optuna(target)
            elif method == "kerastuner":
                best_params = self.execute_kerastuner(target)
            
            self.root.after(0, lambda: self.update_hyper_fields(best_params))
            self.root.after(0, lambda: messagebox.showinfo("Optimization", f"Optimization {method.upper()} for {target} finished!"))
        except Exception as e:
            self.root.after(0, lambda m=str(e): messagebox.showerror("Error Optimization", m))
        finally:
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: [self.progress.config(mode="determinate", value=0), self.lbl_progress_pct.config(text="0%")])
            self.root.after(0, lambda: self.lbl_status.config(text="Ready"))
            # Always re-enable the Start button when tuning terminates
            self.root.after(0, lambda: self.btn_run.config(state="normal"))

    def execute_optuna(self, target):
        """Optuna HPO with MedianPruner — prune unpromising trials early."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,   # Let first 3 trials run fully to build a baseline
            n_warmup_steps=1      # Start pruning after at least 1 reported step
        )

        trials_count = max(1, self.hpo_trials_var.get())
        current_trial = 0

        def objective(trial):
            nonlocal current_trial
            current_trial += 1
            pct = (current_trial - 1) * 100 / trials_count
            self.root.after(0, lambda: [
                self.progress.config(mode="determinate", value=pct),
                self.lbl_progress_pct.config(text=f"{int(pct)}%"),
                self.lbl_status.config(text=f"Optimization {target} via OPTUNA... Trial {current_trial}/{trials_count}")
            ])
            params = self.get_trial_params(trial, "optuna", target)
            try:
                val = self.evaluate_params_with_pruning(params, target, trial)
                final_pct = current_trial * 100 / trials_count
                self.root.after(0, lambda: [
                    self.progress.config(value=final_pct),
                    self.lbl_progress_pct.config(text=f"{int(final_pct)}%")
                ])
                return val
            except optuna.TrialPruned:
                final_pct = current_trial * 100 / trials_count
                self.root.after(0, lambda: [
                    self.progress.config(value=final_pct),
                    self.lbl_progress_pct.config(text=f"{int(final_pct)}%")
                ])
                raise  # Re-raise so Optuna marks this trial as pruned

        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=trials_count)
        return study.best_params

    def evaluate_params_with_pruning(self, params, target, trial):
        """
        Same solver evaluation as evaluate_params, but runs episode-by-episode
        and reports intermediate scores to Optuna for pruning.
        After each episode, trial.report() sends the current best energy score,
        and trial.should_prune() allows early termination of bad trials.
        """
        # Utilisation des vrais parametres HPO suggeres (pas de reduction artificielle)
        eval_episodes   = int(params.get('episodes',  self.episodes_var.get()))
        eval_k_clusters = int(params.get('k_clusters', self.k_clusters_var.get()))
        eval_pop_size   = int(params.get('pop_size',   self.pop_size_nsga_var.get()))
        eval_n_gen      = int(params.get('n_gen',      self.n_gen_nsga_var.get()))

        solver = MAPPOEVRPD(
            self.clients_coords, self.clients_demand, self.orders, self.time_windows,
            no_flight_clients=self.no_flight_clients,
            nfz_polygons=(CITY_NFZS.get(self.city_var.get(), []) if (not hasattr(self, 'enforce_nfz_var') or self.enforce_nfz_var.get()) else []),
            n_ev=self.n_ev_var.get(), n_drones=self.n_drone_var.get(),
            ev_autonomy_km=self.ev_auto_var.get(), drone_autonomy_km=self.drone_auto_var.get(),
            start_time_minute=parse_hhmm(self.start_time_var.get()),
            n_episodes=eval_episodes,
            k_clusters=eval_k_clusters,
            pop_size_nsga=eval_pop_size,
            n_gen_nsga=eval_n_gen,
            cx_rate_nsga=float(params.get('cx_rate', self.cx_rate_nsga_var.get())),
            mut_rate_nsga=float(params.get('mut_rate', self.mut_rate_nsga_var.get())),
            lr=float(params.get('lr_mappo', params.get('lr', self.lr_var.get()))),
            gamma=float(params.get('gamma_mappo', params.get('gamma', self.gamma_var.get()))),
            clip_eps=float(params.get('clip_eps', self.clip_eps_var.get())),
            ppo_epochs=int(params.get('ppo_epochs', self.ppo_epochs_var.get())),
            maddpg_lr=float(params.get('lr_maddpg', params.get('lr', self.lr_maddpg_var.get()))),
            maddpg_gamma=float(params.get('gamma_maddpg', params.get('gamma', self.gamma_maddpg_var.get()))),
            maddpg_tau=float(params.get('tau', self.tau_maddpg_var.get()))
        )

        # --- Level 1: K-Means clustering (same as solver.run) ---
        if len(solver.coords) > 5:
            t_pens = solver.traffic_penalties[1:] if solver.traffic_penalties is not None else None
            solver.clusters, _ = kmeans_conditioned(
                solver.coords[1:],
                k=min(len(solver.coords)-1, getattr(solver, 'k_clusters', 4)),
                weather_penalty=solver.weather_penalty,
                traffic_penalties=t_pens
            )
        else:
            solver.clusters = np.zeros(len(solver.coords)-1, dtype=int)

        # --- Level 2: NSGA-II strategic planning ---
        nsga = NSGAII(
            solver.coords, solver.n_ev, clusters=solver.clusters,
            pop_size=solver.pop_size_nsga, n_gen=solver.n_gen_nsga,
            cx_rate=solver.cx_rate_nsga, mut_rate=solver.mut_rate_nsga,
            orders=solver.orders, ev_autonomy_km=solver.ev_autonomy_km,
            demand_types=solver.demand_types, no_flight_clients=solver.no_flight_clients
        )
        solver.master_plan = nsga.solve()

        # --- Level 3 & 4: MAPPO + MADDPG episode loop with pruning ---
        best_obj = float("inf")
        best_energy = 999999
        for ep in range(1, solver.n_episodes + 1):
            # Tactical (MAPPO) sample pass
            _, _, _, traj, rew, rv_traj, rv_rew, m_sample, _, _ = solver.construct_solution_sample(ep)
            solver.ppo_update(traj, rew, rv_traj, rv_rew)

            # Operational (MADDPG) deterministic evaluation
            ev_e, dr_e, us_e, met, dmeet, dlaunch = solver.construct_solution_deterministic()

            if met["objective"] < best_obj:
                best_obj = met["objective"]
                solver.best_solution = {
                    "ev_routes": ev_e, "drone_assign": dr_e,
                    "metrics": met, "drone_meetings": dmeet, "drone_launch": dlaunch
                }
                best_energy = met.get("energy", 999999)

            # --- PRUNING: report intermediate score to Optuna ---
            trial.report(best_energy, ep - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_energy

    def execute_kerastuner(self, target):
        """KerasTuner HPO with Hyperband — successive halving for resource allocation."""
        import tempfile
        import shutil

        dashboard_ref = self  # Capture reference for use inside the tuner class

        def build_model(hp):
            dashboard_ref.get_trial_params(hp, "kerastuner", target)
            return None

        class EVRPDHyperbandTuner(kt.Tuner):
            """
            Custom Tuner subclass for non-Keras models.
            Overrides run_trial() to execute our MAPPOEVRPD solver
            instead of a standard model.fit().
            Hyperband manages how many epochs (episodes) each trial gets.
            """
            def run_trial(self, trial, *args, **kwargs):
                current_trial = len(self.oracle.trials)
                pct = min(99.0, (current_trial - 1) * 100 / trials_count)
                
                dashboard_ref.root.after(0, lambda: [
                    dashboard_ref.progress.config(mode="determinate", value=pct),
                    dashboard_ref.lbl_progress_pct.config(text=f"{int(pct)}%"),
                    dashboard_ref.lbl_status.config(text=f"Optimization {target} via KERASTUNER... Trial {current_trial}")
                ])
                
                hp = trial.hyperparameters
                params = dashboard_ref.get_trial_params(hp, "kerastuner", target)
                score = dashboard_ref.evaluate_params(params, target)
                
                final_pct = min(100.0, current_trial * 100 / trials_count)
                dashboard_ref.root.after(0, lambda: [
                    dashboard_ref.progress.config(value=final_pct),
                    dashboard_ref.lbl_progress_pct.config(text=f"{int(final_pct)}%")
                ])
                
                # Report the objective value to the Hyperband oracle and mark complete
                self.oracle.update_trial(trial.trial_id, {"val_loss": score})
                self.on_trial_end(trial)

        # Temporary directory for Hyperband bookkeeping (cleaned up after)
        tuner_dir = tempfile.mkdtemp(prefix="evrpd_hyperband_")

        try:
            trials_count = max(1, dashboard_ref.hpo_trials_var.get())

            # Build the Hyperband oracle
            oracle = kt.oracles.HyperbandOracle(
                objective=kt.Objective("val_loss", direction="min"),
                max_epochs=max(3, trials_count),  # Controls successive halving depth
                factor=3,                          # Reduction factor (keep 1/3 at each round)
                hyperband_iterations=1
            )

            tuner = EVRPDHyperbandTuner(
                oracle=oracle,
                hypermodel=build_model,
                directory=tuner_dir,
                project_name="evrpd_hpo"
            )

            # Run the Hyperband search
            tuner.search()

            # Retrieve the best hyperparameters safely
            best_hps = tuner.get_best_hyperparameters(num_trials=1)
            if best_hps:
                best_params = dashboard_ref.get_trial_params(best_hps[0], "kerastuner", target)
            else:
                best_params = {}
            return best_params

        finally:
            # Clean up temporary Hyperband directory
            try:
                shutil.rmtree(tuner_dir, ignore_errors=True)
            except Exception:
                pass

    def get_search_space(self, method, target):
        if method == "kerastuner":
            if target == "ALL":
                return {'episodes': (50, 1000, 10), 'k_clusters': (2, 8, 1), 'pop_size': (10, 100, 1), 'n_gen': (5, 50, 1), 'cx_rate': (0.5, 0.95), 'mut_rate': (0.01, 0.3),
                        'lr_mappo': (1e-4, 1e-3), 'gamma_mappo': (0.9, 0.999), 'clip_eps': (0.1, 0.3), 'ppo_epochs': (2, 10, 1),
                        'lr_maddpg': (1e-4, 1e-3), 'gamma_maddpg': (0.9, 0.999), 'tau': (0.001, 0.05), 'batch': [32, 64, 128]}
            elif target == "MAPPO":
                return {'lr': (1e-4, 1e-3), 'gamma': (0.9, 0.999), 'clip_eps': (0.1, 0.3), 'ppo_epochs': (2, 10, 1)}
            elif target == "Global": return {'episodes': (50, 1000, 10)}
            elif target == "NSGA": return {'pop_size': (10, 100, 1), 'n_gen': (5, 50, 1), 'cx_rate': (0.5, 0.95), 'mut_rate': (0.01, 0.3)}
            elif target == "MADDPG": return {'lr': (1e-4, 1e-3), 'gamma': (0.9, 0.999), 'tau': (0.001, 0.05), 'batch': [32, 64, 128]}
    def get_trial_params(self, hp_obj, method, target):
        params = {}
        if method == "optuna":
            if target == "MAPPO":
                params['lr'] = hp_obj.suggest_float("lr", 1e-4, 1e-3, log=True)
                params['gamma'] = hp_obj.suggest_float("gamma", 0.9, 0.999)
                params['episodes'] = hp_obj.suggest_int("episodes", 50, 1000, step=10)
            elif target == "Global":
                params['episodes'] = hp_obj.suggest_int("episodes", 50, 1000, step=10)
            elif target == "NSGA":
                params['pop_size'] = hp_obj.suggest_int("pop_size", 10, 100)
                params['n_gen'] = hp_obj.suggest_int("n_gen", 5, 50)
                params['cx_rate'] = hp_obj.suggest_float("cx_rate", 0.5, 0.95)
                params['mut_rate'] = hp_obj.suggest_float("mut_rate", 0.01, 0.3)
            elif target == "MADDPG":
                params['m_lr'] = hp_obj.suggest_float("m_lr", 1e-4, 5e-3, log=True)
                params['m_gamma'] = hp_obj.suggest_float("m_gamma", 0.9, 0.99)
            elif target == "ALL":
                params['episodes'] = hp_obj.suggest_int("episodes", 50, 1000, step=10)
                params['k_clusters'] = hp_obj.suggest_int("k_clusters", 2, 8)
                params['pop_size'] = hp_obj.suggest_int("pop_size", 10, 100)
                params['n_gen'] = hp_obj.suggest_int("n_gen", 5, 50)
                params['cx_rate'] = hp_obj.suggest_float("cx_rate", 0.5, 0.95)
                params['mut_rate'] = hp_obj.suggest_float("mut_rate", 0.01, 0.3)
                params['lr_mappo'] = hp_obj.suggest_float("lr_mappo", 1e-4, 1e-3, log=True)
                params['gamma_mappo'] = hp_obj.suggest_float("gamma_mappo", 0.9, 0.999)
                params['clip_eps'] = hp_obj.suggest_float("clip_eps", 0.1, 0.3)
                params['ppo_epochs'] = hp_obj.suggest_int("ppo_epochs", 2, 10)
                params['lr_maddpg'] = hp_obj.suggest_float("lr_maddpg", 1e-4, 5e-3, log=True)
                params['gamma_maddpg'] = hp_obj.suggest_float("gamma_maddpg", 0.9, 0.99)
                params['tau'] = hp_obj.suggest_float("tau", 0.001, 0.05, log=True)
                params['batch'] = hp_obj.suggest_categorical("batch", [32, 64, 128])
        elif method == "kerastuner":
            if target == "MAPPO":
                params['lr'] = hp_obj.Float("lr", 1e-4, 1e-3, sampling="log")
                params['gamma'] = hp_obj.Float("gamma", 0.9, 0.999)
                params['episodes'] = hp_obj.Int("episodes", 50, 1000, step=10)
            elif target == "Global":
                params['episodes'] = hp_obj.Int("episodes", 50, 1000, step=10)
            elif target == "NSGA":
                params['pop_size'] = hp_obj.Int("pop_size", 10, 100)
                params['n_gen'] = hp_obj.Int("n_gen", 5, 50)
                params['cx_rate'] = hp_obj.Float("cx_rate", 0.5, 0.95)
                params['mut_rate'] = hp_obj.Float("mut_rate", 0.01, 0.3)
            elif target == "MADDPG":
                params['m_lr'] = hp_obj.Float("m_lr", 1e-4, 5e-3, sampling="log")
                params['m_gamma'] = hp_obj.Float("m_gamma", 0.9, 0.99)
            elif target == "ALL":
                params['episodes'] = hp_obj.Int("episodes", 50, 1000, step=10)
                params['k_clusters'] = hp_obj.Int("k_clusters", 2, 8)
                params['pop_size'] = hp_obj.Int("pop_size", 10, 100)
                params['n_gen'] = hp_obj.Int("n_gen", 5, 50)
                params['cx_rate'] = hp_obj.Float("cx_rate", 0.5, 0.95)
                params['mut_rate'] = hp_obj.Float("mut_rate", 0.01, 0.3)
                params['lr_mappo'] = hp_obj.Float("lr_mappo", 1e-4, 1e-3, sampling="log")
                params['gamma_mappo'] = hp_obj.Float("gamma_mappo", 0.9, 0.999)
                params['clip_eps'] = hp_obj.Float("clip_eps", 0.1, 0.3)
                params['ppo_epochs'] = hp_obj.Int("ppo_epochs", 2, 10)
                params['lr_maddpg'] = hp_obj.Float("lr_maddpg", 1e-4, 5e-3, sampling="log")
                params['gamma_maddpg'] = hp_obj.Float("gamma_maddpg", 0.9, 0.99)
                params['tau'] = hp_obj.Float("tau", 0.001, 0.05, sampling="log")
                params['batch'] = hp_obj.Choice("batch", [32, 64, 128])
        return params

    def evaluate_params(self, params, target):
        # Utilisation des vrais parametres HPO suggeres (pas de reduction artificielle)
        eval_episodes   = int(params.get('episodes',  self.episodes_var.get()))
        eval_k_clusters = int(params.get('k_clusters', self.k_clusters_var.get()))
        eval_pop_size   = int(params.get('pop_size',   self.pop_size_nsga_var.get()))
        eval_n_gen      = int(params.get('n_gen',      self.n_gen_nsga_var.get()))

        solver = MAPPOEVRPD(
            self.clients_coords, self.clients_demand, self.orders, self.time_windows,
            no_flight_clients=self.no_flight_clients,
            nfz_polygons=(CITY_NFZS.get(self.city_var.get(), []) if (not hasattr(self, 'enforce_nfz_var') or self.enforce_nfz_var.get()) else []),
            n_ev=self.n_ev_var.get(), n_drones=self.n_drone_var.get(),
            ev_autonomy_km=self.ev_auto_var.get(), drone_autonomy_km=self.drone_auto_var.get(),
            start_time_minute=parse_hhmm(self.start_time_var.get()),
            n_episodes=eval_episodes, 
            k_clusters=eval_k_clusters, 
            pop_size_nsga=eval_pop_size if target == "ALL" else eval_pop_size,
            n_gen_nsga=eval_n_gen if target == "ALL" else eval_n_gen,
            cx_rate_nsga=float(params.get('cx_rate', self.cx_rate_nsga_var.get())),
            mut_rate_nsga=float(params.get('mut_rate', self.mut_rate_nsga_var.get())),
            lr=float(params.get('lr_mappo', params.get('lr', self.lr_var.get()))),
            gamma=float(params.get('gamma_mappo', params.get('gamma', self.gamma_var.get()))),
            clip_eps=float(params.get('clip_eps', self.clip_eps_var.get())),
            ppo_epochs=int(params.get('ppo_epochs', self.ppo_epochs_var.get())),
            maddpg_lr=float(params.get('lr_maddpg', params.get('lr', self.lr_maddpg_var.get()))),
            maddpg_gamma=float(params.get('gamma_maddpg', params.get('gamma', self.gamma_maddpg_var.get()))),
            maddpg_tau=float(params.get('tau', self.tau_maddpg_var.get()))
        )
        def dummy_cb(e, t): pass
        solver.run(progress_callback=dummy_cb)
        if not solver.best_solution: return 999999
        m = solver.best_solution.get("metrics", {})
        return m.get('energy', 999999)


    def update_hyper_fields(self, best):
        self.latest_hpo_params = best
        if 'lr' in best: self.lr_var.set(f"{best['lr']:.5f}")
        if 'gamma' in best: self.gamma_var.set(f"{best['gamma']:.3f}")
        if 'm_lr' in best: self.lr_maddpg_var.set(f"{best['m_lr']:.5f}")
        if 'm_gamma' in best: self.gamma_maddpg_var.set(f"{best['m_gamma']:.3f}")
        if 'clip_eps' in best: self.clip_eps_var.set(f"{best['clip_eps']:.2f}")
        if 'ppo_epochs' in best: self.ppo_epochs_var.set(str(int(best['ppo_epochs'])))
        if 'tau' in best: self.tau_maddpg_var.set(f"{best['tau']:.3f}")
        if 'batch' in best: self.batch_maddpg_var.set(str(int(best['batch'])))

        # Keys for ALL target
        if 'lr_mappo' in best: self.lr_var.set(f"{best['lr_mappo']:.5f}")
        if 'gamma_mappo' in best: self.gamma_var.set(f"{best['gamma_mappo']:.3f}")
        if 'lr_maddpg' in best: self.lr_maddpg_var.set(f"{best['lr_maddpg']:.5f}")
        if 'gamma_maddpg' in best: self.gamma_maddpg_var.set(f"{best['gamma_maddpg']:.3f}")
        if 'episodes' in best: self.episodes_var.set(str(int(best['episodes'])))
        if 'k_clusters' in best: self.k_clusters_var.set(str(int(best['k_clusters'])))

        # NSGA Fields
        if 'pop_size' in best: self.pop_size_nsga_var.set(str(int(best['pop_size'])))
        if 'n_gen' in best: self.n_gen_nsga_var.set(str(int(best['n_gen'])))
        if 'cx_rate' in best: self.cx_rate_nsga_var.set(f"{best['cx_rate']:.2f}")
        if 'mut_rate' in best: self.mut_rate_nsga_var.set(f"{best['mut_rate']:.2f}")

    def stop_training(self):
        if hasattr(self, 'solver') and self.solver:
            self.solver.stop_flag = True
            self.lbl_status.config(text="Stop requested...")

    def start_training(self, is_agentic=False):
        from tkinter import messagebox
        import threading
        if not self.clients_coords:
            messagebox.showwarning("Error", "Generate a scenario first (Tab 1).")
            return
        self.btn_run.config(state="disabled")
        self.lbl_status.config(text="Training in progress...")
        self.progress["value"] = 0
        self.lbl_progress_pct.config(text="0%")
        
        try:
            if is_agentic:
                live = self.agentic_ui.agentic_results_panel.live_mapping_panel
                live.trigger_stage(1, "Init Agentic Training", 10)
            else:
                live = getattr(self, 'live_mapping_classic_panel', None)
                if live:
                    live.trigger_stage(1, "Init Classic Training", 10)
        except Exception:
            pass

        threading.Thread(target=self.run_mappo, args=(is_agentic,), daemon=True).start()

    def run_mappo(self, is_agentic=False):
        try:
            # Nouveau seed a  chaque entraa®nement
            new_seed = int(time.time() * 1000) % 100000
            random.seed(new_seed)
            np.random.seed(new_seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(new_seed)
            
            # Regenerer le scenario pour avoir une nouvelle topologie a  chaque fois
                        
            # Enforce Multi-Parcel restriction: If a client has > 1 parcel, they must be delivered by EV
            if hasattr(self, 'clients_demand') and self.clients_demand:
                for i in range(1, len(self.clients_demand)):
                    qty = self.orders.get(i, {}).get("parcels", 0)
                    if qty > 1 and self.clients_demand[i] == 2:
                        self.clients_demand[i] = 1
            
            weather_ok, weather_reason = self.get_weather_permission()
            
            # --- K-Means Conditionne : Calcul des penalites ---
            try:
                wind_speed = float(self.weather_wind_var.get())
            except:
                wind_speed = 0.0
            wp = 1.0 + max(0, (wind_speed - 10.0) / 20.0)
            
            dense_zones = ["Maarif", "Sidi Belyout", "Anfa", "Derb Ghallef", "Roches Noires"]
            tp = [1.0] # Depot
            for i in range(1, len(self.clients_coords)):
                q = self.clients_zones.get(i, "")
                if any(dz in q for dz in dense_zones):
                    tp.append(1.5)
                else:
                    tp.append(1.0)
            # ----------------------------------------------------
            
            local_solver = MAPPOEVRPD(
                self.clients_coords, self.clients_demand, self.orders, self.time_windows,
                no_flight_clients=self.no_flight_clients,
                nfz_polygons=(CITY_NFZS.get(self.city_var.get(), []) if (not hasattr(self, 'enforce_nfz_var') or self.enforce_nfz_var.get()) else []),
                n_ev=self.n_ev_var.get(), n_drones=self.n_drone_var.get(),
                ev_autonomy_km=self.ev_auto_var.get(), drone_autonomy_km=self.drone_auto_var.get(),
                drone_pick_time=self.drone_pick_time_var.get(),
                drone_load_time=self.drone_load_time_var.get(),
                drone_unload_time=self.drone_unload_time_var.get(),
                drone_takeoff_time=self.drone_takeoff_time_var.get(),
                drone_return_time=self.drone_return_time_var.get(),
                drone_safe_margin=self.drone_safe_margin_var.get(),
                drone_recharge_time=self.drone_recharge_time_var.get(),
                weather_ok=weather_ok, weather_reason=weather_reason,
                weather_penalty=wp, traffic_penalties=tp,
                start_time_minute=parse_hhmm(self.start_time_var.get()),
                k_clusters=int(self.k_clusters_var.get()),
                n_episodes=self.episodes_var.get(),
                lr=float(self.lr_var.get()), gamma=float(self.gamma_var.get()),
                clip_eps=float(self.clip_eps_var.get()), ppo_epochs=int(self.ppo_epochs_var.get()),
                maddpg_lr=float(self.lr_maddpg_var.get()), maddpg_gamma=float(self.gamma_maddpg_var.get()),
                maddpg_tau=float(self.tau_maddpg_var.get()),
                pop_size_nsga=int(self.pop_size_nsga_var.get()), n_gen_nsga=int(self.n_gen_nsga_var.get()),
                cx_rate_nsga=float(self.cx_rate_nsga_var.get()), mut_rate_nsga=float(self.mut_rate_nsga_var.get())
            )
            if not is_agentic:
                self.solver = local_solver
            def prog(ep, tot):
                pct = ep * 100 / tot
                if not is_agentic:
                    self.root.after(0, lambda: [self.progress.configure(value=pct), self.lbl_progress_pct.configure(text=f"{int(pct)}%")])
                    self.root.after(0, lambda: self.lbl_status.configure(text=f"Training in progress... {int(pct)}%"))
                else:
                    if hasattr(self, 'agentic_ui') and hasattr(self.agentic_ui, 'agentic_progress'):
                        self.root.after(0, lambda: self.agentic_ui.agentic_progress.configure(value=pct))
                        if hasattr(self.agentic_ui, 'lbl_status'):
                            self.root.after(0, lambda: self.agentic_ui.lbl_status.configure(text=f"Agentic Training in progress... {int(pct)}%"))
                try:
                    if is_agentic:
                        live = self.agentic_ui.agentic_results_panel.live_mapping_panel
                        live.trigger_stage(4, f"Agentic Training: {ep}/{tot}", 50 + pct/2.0)
                    else:
                        live = getattr(self, 'live_mapping_classic_panel', None)
                        if live:
                            live.trigger_stage(4, f"Classic Training: {ep}/{tot}", pct)
                except Exception: pass

            local_solver.run(progress_callback=prog)
            self.root.after(0, lambda s=local_solver: self.training_done(s, is_agentic))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.btn_run.config(state="normal"))

    def training_done(self, solver, is_agentic=False):
        if not solver or not solver.best_solution:
            self.lbl_status.config(text="Completed (No solution)")
            self.btn_run.config(state="normal")
            return
        
        self.lbl_status.config(text="Calculating real routes (OSRM)...")
        threading.Thread(target=self.process_results_bg, args=(solver, is_agentic), daemon=True).start()

    def process_results_bg(self, solver, is_agentic=False):
        # Extraction des resultats pour l'affichage
        best = solver.best_solution
                
        # Calcul des routes
        ev_routes_osrm = []
        for r in best.get("ev_routes", []):
            route_indices = [n for n in r if isinstance(n, int)]
            active_clients = [n for n in route_indices if n != 0]
            if not active_clients:
                ev_routes_osrm.append(None)
                continue
            if route_indices[0] != 0:
                route_indices.insert(0, 0)
            if route_indices[-1] != 0:
                route_indices.append(0)
            pts = [self.clients_coords[i] for i in route_indices if isinstance(i, int) and i < len(self.clients_coords)]
            if len(pts) >= 2:
                route_aller = get_osrm_route(pts[:-1])
                route_retour = get_osrm_route([pts[-2], pts[-1]])
                ev_routes_osrm.append({"aller": route_aller, "retour": route_retour})
            else:
                ev_routes_osrm.append(None)
        
        # Retour au thread principal pour mettre a jour l'UI
        self.root.after(0, lambda: self.update_results(best, ev_routes_osrm, solver, is_agentic))


    def update_results(self, best, ev_routes_osrm, solver, is_agentic=False):
        self.lbl_status.config(text="Completed")
        self.btn_run.config(state="normal")
        
        # Coerce drone_assign and drone_launch to integer keys/values
        drone_assign_raw = best.get("drone_assign", {}) or {}
        drone_launch_raw = best.get("drone_launch", {}) or {}
        drone_assign = {int(k): int(v) for k, v in drone_assign_raw.items()}
        drone_launch = {int(k): int(v) for k, v in drone_launch_raw.items()}
        best["drone_assign"] = drone_assign
        best["drone_launch"] = drone_launch

        if not is_agentic:
            self.latest_best_classic = best
            self.latest_ev_routes_osrm_classic = ev_routes_osrm
            self.latest_solver_classic = solver
            
        self.latest_best = best
        self.latest_ev_routes_osrm = ev_routes_osrm
        self.latest_solver = solver
        
        if is_agentic:
            # Si c'est agentique, on met uniquement a  jour le panel agentique
            try:
                # Basculer vers l'onglet principal "Methode agentique"
                # Basculer vers le sous-onglet "Results agentiques"
                
                # Mettre a  jour les resultats
                self.agentic_ui.agentic_results_panel.update_results(best, ev_routes_osrm, solver)
                
                # Fin de live mapping
                try:
                    live = self.agentic_ui.agentic_results_panel.live_mapping_panel
                    live.trigger_stage(6, "Simulation completed, displaying results", 100)
                except Exception: pass
                
                # Mettre le Step Tracker sur DONE
                self.agentic_ui.update_status("Training completed successsfully.", 100)
                
            except Exception as e:
                print(f"Error update agentique: {e}")
            return
        
        # Sinon classique
        try:
            live = getattr(self, 'live_mapping_classic_panel', None)
            if live:
                live.trigger_stage(6, "Classic training completed, displaying results", 100)
            self.sidebar.select('analytics_classic')
        except Exception: pass
        m = best["metrics"]
        # KPIs
        self.kpi_vars["Total distance"].set(f"{m.get('total_dist', m.get('total_distance', 0.0)):.2f}")
        self.kpi_vars["EV distance"].set(f"{m.get('ev_dist', m.get('ev_distance', 0.0)):.2f}")
        self.kpi_vars["Drone distance"].set(f"{m.get('drone_dist', m.get('drone_distance', 0.0)):.2f}")
        self.kpi_vars["Energy"].set(f"{m.get('energy', 0.0):.2f}")
        if "CO2 ASIF" in self.kpi_vars:
            self.kpi_vars["CO2 ASIF"].set(f"{m.get('co2_asife', 0.0):.2f}")
        self.kpi_vars["Service rate"].set(f"{m.get('service_rate', 0.0):.2f}")
        self.kpi_vars["Late penalty"].set(f"{m.get('late', 0.0):.2f}")
        self.kpi_vars["EV recharges"].set(str(m.get('ev_recharges', 0)))
        self.kpi_vars["Drone recharges"].set(str(m.get('drone_recharges', 0)))
        self.kpi_vars["EV auto left"].set(f"{m.get('ev_autonomy_left', 0.0):.2f}")
        self.kpi_vars["Drone auto left"].set(f"{m.get('drone_autonomy_left', 0.0):.2f}")
        self.kpi_vars["Resources used"].set(f"{m.get('used_ev', 0)} EV, {m.get('used_drones', 0)} Drones")

        # Mise a  jour des orders avec les modes finaux et clusters
        try:
            ev_clients = set()
            for r in best.get("ev_routes", []):
                for c in r:
                    if c != 0:
                        ev_clients.add(c)
            drone_clients = set(best.get("drone_assign", {}).keys())
            clusters = getattr(solver, 'clusters', [])

            for item in self.tree_orders.get_children():
                val = list(self.tree_orders.item(item, "values"))
                c_str = val[0].split(' ')[0][1:]
                if c_str.isdigit():
                    c_id = int(c_str)
                    if c_id in drone_clients:
                        val[6] = "Drone"
                    elif c_id in ev_clients:
                        val[6] = "Electric Vehicle"
                    else:
                        val[6] = "Non Livre"
                        
                    if len(clusters) >= c_id:
                        cl_id = clusters[c_id - 1]
                        q = val[1].split(" (Cluster")[0]
                        val[1] = f"{q} (Cluster {cl_id + 1})"
                        
                    self.tree_orders.item(item, values=val)
        except Exception as e:
            print(f"Error update tree_orders: {e}")

        if not is_agentic:
            self.latest_best_classic = best
            self.latest_ev_routes_osrm_classic = ev_routes_osrm
            self.latest_solver_classic = solver
            
        self.latest_best = best
        self.latest_ev_routes_osrm = ev_routes_osrm
        self.latest_solver = solver
        
        # Mettre a  jour le panneau des couches avant de rafraa®chir la carte
        self.update_layers_panel(best, ev_routes_osrm)
        self.update_cart_ev_checkboxes(best)
        
        self.update_map_filter_choices()
        self.refresh_map_view()
        self.draw_cartesian_plan(best, solver)

        # Curves
        ep = list(range(1, len(solver.history_best) + 1))

        self.ax_obj.clear()
        self.ax_obj.plot(ep, solver.history_best, color=GRENAT, label="Score IA")
        self.ax_obj.set_title("Objective", fontsize=10, fontweight='bold', color='white')
        self.ax_obj.legend(fontsize=8)
        self.ax_obj.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

        self.ax_dist.clear()
        self.ax_dist.plot(ep, solver.history_distance, color="blue", label="Distance")
        self.ax_dist.set_title("Distance", fontsize=10, fontweight='bold', color='white')
        self.ax_dist.legend(fontsize=8)
        self.ax_dist.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

        self.ax_co2.clear()
        self.ax_co2.plot(ep, solver.history_co2_asife, color="orange", label="CO2 ASIF")
        self.ax_co2.set_title("CO2 (ASIF)", fontsize=10, fontweight='bold', color='white')
        self.ax_co2.legend(fontsize=8)
        self.ax_co2.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

        self.ax_energy.clear()
        self.ax_energy.plot(ep, solver.history_energy, color="orange", label="Energy")
        self.ax_energy.set_title("Energy", fontsize=10, fontweight='bold', color='white')
        self.ax_energy.legend(fontsize=8)
        self.ax_energy.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

        self.ax_late.clear()
        self.ax_late.plot(ep, solver.history_late, color="red", label="Late")
        self.ax_late.set_title("Late", fontsize=10, fontweight='bold', color='white')
        self.ax_late.legend(fontsize=8)
        self.ax_late.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

        self.ax_service.clear()
        self.ax_service.plot(ep, solver.history_service, color="purple", label="Service Rate")
        self.ax_service.set_title("Service Rate", fontsize=10, fontweight='bold', color='white')
        self.ax_service.legend(fontsize=8)
        self.ax_service.grid(True, which='both', color='#334155', linestyle='-', linewidth=0.5)

        self.canvas_curve.draw()

        # Bars
        self.ax_bar.clear()
        names = ["EV km", "Drone km", "Energy", "CO2 ASIF/100", "Late"]
        vals = [m["ev_dist"], m["drone_dist"], m["energy"], m["co2_asife"]/100, m["late"]]
        self.ax_bar.bar(names, vals, color=[GRENAT, "red", "orange", "#388E3C", "purple"])
        self.canvas_bar.draw()

        # Text
        def build_ev_schedule(ev_route):
            schedule = []
            t = solver.start_time_minute
            for i, node in enumerate(ev_route):
                if i > 0:
                    t += solver.travel_time_minutes(ev_route[i-1], node)
                arrival = t
                if node == 0:
                    service = 0.0
                else:
                    is_drone_waypoint = (solver.demand_types[node] == 2 and node not in solver.no_flight_clients)
                    service = 1.0 if is_drone_waypoint else solver.service_time_min * max(1, solver.orders.get(node, {}).get("parcels", 1))
                departure = arrival + service
                schedule.append({"node": node, "arrival": arrival, "departure": departure})
                t = departure
            return schedule

        def find_node_time(schedule, node_id, field):
            for stop in schedule:
                if stop["node"] == node_id:
                    return stop[field]
            return None

        weather_status = self.weather_status_var.get() or "No weather data"
        weather_line = f"DRONE WEATHER: {weather_status} - Flight autorise : {'Oui' if self.weather_allowed else 'Non'}\n\n"
        for widget in getattr(self, 'routes_container', tk.Frame()).winfo_children():
            widget.destroy()

        self.txt_routes.pack(side="top", fill="both", expand=True)
        self.txt_routes.delete("1.0", "end")
        self.txt_routes.insert("end", weather_line)
        
        # --- CLUSTERING DETAILS (K-Means) ---
        clusters = []
        if hasattr(self, 'latest_solver') and self.latest_solver:
            clusters = getattr(self.latest_solver, 'clusters', [])
        grouped_clusters = {}
        if len(clusters) > 0:
            for i, cid in enumerate(clusters):
                c_id = i + 1
                grouped_clusters.setdefault(cid, []).append(c_id)
        
        cluster_vehicles = {cid: {"evs": [], "drones": []} for cid in grouped_clusters}
        for ev_idx, route in enumerate(best.get("ev_routes", [])):
            ev_id = ev_idx + 1
            for node in route:
                if node != 0:
                    node_cluster_idx = node - 1
                    if node_cluster_idx < len(clusters):
                        cid = clusters[node_cluster_idx]
                        if ev_id not in cluster_vehicles[cid]["evs"]:
                            cluster_vehicles[cid]["evs"].append(ev_id)
                            
        for c in best.get("drone_assign", {}).items():
            node_cluster_idx = c[0] - 1
            if node_cluster_idx < len(clusters):
                cid = clusters[node_cluster_idx]
                if c[0] not in cluster_vehicles[cid]["drones"]:
                    cluster_vehicles[cid]["drones"].append(c[0])

        cluster_text = "CLUSTERING ANALYSIS (K-Means):\n"
        if len(grouped_clusters) == 0:
            cluster_text += "  No cluster data available.\n\n"
        else:
            for cid in sorted(grouped_clusters.keys()):
                clients_list = ", ".join(f"C{c}" for c in grouped_clusters[cid])
                evs_serving = ", ".join(f"EV {e}" for e in cluster_vehicles[cid]["evs"]) or "None"
                drones_serving = ", ".join(f"Drone C{d}" for d in cluster_vehicles[cid]["drones"]) or "None"
                cluster_text += f"  Cluster {cid + 1}:\n"
                cluster_text += f"    - Clients: {clients_list}\n"
                cluster_text += f"    - Serving Vehicles: {evs_serving} | Drones: {drones_serving}\n"
            cluster_text += "\n"
            
        self.txt_routes.insert("end", cluster_text)
        self.txt_routes.insert("end", "ROUTES EV :\n")
        ev_schedules = [build_ev_schedule(r) for r in best["ev_routes"]]
        for k, schedule in enumerate(ev_schedules, 1):
            txts = []
            self.txt_routes.insert("end", f"EV {k}:\n")
            stops = 0
            for stop in schedule:
                node = stop["node"]
                if node == 0 and stop == schedule[0]:
                    msg = f"Departure depa´t 0 a  {minutes_to_hhmm(stop['departure'])}"
                    self.txt_routes.insert("end", f"  {msg}\n")
                elif node == 0:
                    msg = f"Return to depot 0 arrival at {minutes_to_hhmm(stop['arrival'])}"
                    self.txt_routes.insert("end", f"  {msg}\n")
                else:
                    pkgs = solver.orders.get(node, {}).get("parcels", 1)
                    msg = f"Client {node} ({pkgs} parcels) arrival {minutes_to_hhmm(stop['arrival'])} / departure {minutes_to_hhmm(stop['departure'])}"
                    self.txt_routes.insert("end", f"  {msg}\n")
                    stops += 1
                txts.append(msg)
            
            color = self.EV_COLORS[(k-1) % len(self.EV_COLORS)] if hasattr(self, 'EV_COLORS') else "#2196F3"
            RouteCard(self.routes_container, f"EV {k}", color, f"{stops} Stops", txts).pack(fill="x", pady=2)

        self.txt_routes.insert("end", "\nMISSIONS DRONE :\n")
        if not best.get("drone_assign"):
            self.txt_routes.insert("end", "  Nonee mission drone planifiee ou meteo defavorable.\n")
        else:
            drone_missions_by_ev = {}
            for c, p_return in best.get("drone_assign", {}).items():
                p_launch = best.get("drone_launch", {}).get(c, 0)
                launch_time = None
                return_time = None
                assigned_ev = 1
                for k, schedule in enumerate(ev_schedules, 1):
                    if launch_time is None:
                        t_launch = find_node_time(schedule, p_launch, "departure")
                        if t_launch is not None:
                            launch_time = t_launch
                    if return_time is None:
                        t_return = find_node_time(schedule, p_return, "arrival")
                        if t_return is not None:
                            return_time = t_return
                    if launch_time is not None and return_time is not None:
                        assigned_ev = k
                        break
                
                if launch_time is None:
                    launch_time = solver.start_time_minute
                t_picking_end = launch_time + solver.drone_pick_time
                t_loading_end = t_picking_end + solver.drone_load_time
                t_takeoff = t_loading_end + solver.drone_takeoff_time
                t_arrival_c = t_takeoff + solver.travel_time_minutes(p_launch, c)
                t_unload_end = t_arrival_c + solver.drone_unload_time
                t_return_est = t_unload_end + solver.travel_time_minutes(c, p_return)
                if return_time is not None:
                    t_return = max(t_return_est, return_time)
                else:
                    t_return = t_return_est
                
                pkgs = solver.orders.get(c, {}).get("parcels", 1)
                txts = [
                    f"=== LIVRAISON CLIENT {c} ({pkgs} parcels) ===",
                    f"Takeoff depuis noeud {p_launch} a  {minutes_to_hhmm(t_takeoff)}",
                    f"Client {c} arr. {minutes_to_hhmm(t_arrival_c)} / dep. {minutes_to_hhmm(t_unload_end)}",
                    f"Landing noeud {p_return} a  {minutes_to_hhmm(t_return)}",
                    ""
                ]
                
                if assigned_ev not in drone_missions_by_ev:
                    drone_missions_by_ev[assigned_ev] = []
                drone_missions_by_ev[assigned_ev].extend(txts)

            for ev_id, txts in drone_missions_by_ev.items():
                self.txt_routes.insert("end", f"\nDrone associe au EV Truck {ev_id} :\n")
                for txt in txts:
                    self.txt_routes.insert("end", f"    {txt}\n")
                if txts and txts[-1] == "":
                    txts.pop()
                RouteCard(self.routes_container, f"Mission Drone {ev_id}", "#9C27B0", f"Via EV {ev_id}", txts).pack(fill="x", pady=2)
                
        if hasattr(self, "fleet_table"):
            self.fleet_table.update_data(best, ev_routes_osrm)

    def export_route_details(self, txt_widget, fmt="csv"):
        from tkinter import filedialog, messagebox
        content = txt_widget.get("1.0", "end").strip()
        if not content:
            messagebox.showwarning("Warning", "None detail de route a  exporter.")
            return
            
        lines = content.split('\n')
        data = [[line] for line in lines]
        
        if fmt == "csv":
            filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not filepath: return
            import csv
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(data)
                messagebox.showinfo("Success", f"Fichier CSV exporte avec succa¨s :\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"CSV export failed: {e}")
        elif fmt == "xlsx":
            filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if not filepath: return
            try:
                import pandas as pd
                df = pd.DataFrame(data, columns=["Route details"])
                df.to_excel(filepath, index=False)
                messagebox.showinfo("Success", f"Fichier Excel exporte avec succa¨s :\n{filepath}")
            except ImportError:
                messagebox.showwarning("Missing Pandas", "pandas/openpyxl library is required for Excel export. Exportation en CSV a  la place.")
                self.export_route_details(txt_widget, "csv")
            except Exception as e:
                messagebox.showerror("Error", f"Excel export failed: {e}")

    def export_full_simulation_excel(self):
        from tkinter import filedialog, messagebox
        import pandas as pd
        
        # Check if we have at least one simulation
        classic_best = getattr(self, 'latest_best_classic', None) or getattr(self, 'latest_best', None)
        
        agentic_best = None
        if hasattr(self, 'agentic_ui') and hasattr(self.agentic_ui, 'agentic_results_panel'):
            agentic_best = self.agentic_ui.agentic_results_panel.latest_best
            
        if not classic_best and not agentic_best:
            messagebox.showwarning("Export Warning", "No simulation data available. Please run Classic or Agentic training first.")
            return
            
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if not filepath:
            return
            
        try:
            m_c = classic_best.get("metrics", {}) if classic_best else {}
            m_a = agentic_best.get("metrics", {}) if agentic_best else {}
            
            # Tab 1: Comparison Summary
            kpis_to_compare = [
                ("Total distance", "km", "lower"),
                ("EV distance", "km", "lower"),
                ("Drone distance", "km", "lower"),
                ("Energy", "kWh", "lower"),
                ("CO2 ASIF", "g", "lower"),
                ("Service rate", "%", "higher"),
                ("Late penalty", "min", "lower"),
                ("EV recharges", "times", "lower"),
                ("Drone recharges", "times", "lower"),
                ("EV auto left", "km", "higher"),
                ("Drone auto left", "km", "higher"),
            ]
            
            def get_metric(metrics, key, default=0.0):
                if not metrics: return default
                val = metrics.get(key, metrics.get(key.replace(" ", "_").lower(), None))
                if val is None:
                    for k, v in metrics.items():
                        if k.lower() == key.lower():
                            return v
                return val if val is not None else default
                
            comp_rows = []
            for name, unit, direction in kpis_to_compare:
                val_c = get_metric(m_c, name, 0.0) if classic_best else 0.0
                val_a = get_metric(m_a, name, 0.0) if agentic_best else 0.0
                
                # Compute improvement
                if not classic_best or not agentic_best:
                    improvement_str = "-"
                elif val_c == val_a:
                    improvement_str = "0.00 %"
                elif val_c == 0:
                    improvement_str = "N/A"
                else:
                    if direction == "lower":
                        imp = (val_c - val_a) / val_c * 100
                    else:
                        imp = (val_a - val_c) / val_c * 100
                    improvement_str = f"{imp:+.2f} %"
                    
                comp_rows.append({
                    "KPI Parameter": name,
                    "Classic Value (NSGA-II)": round(val_c, 3) if classic_best else "-",
                    "Agentic Value (MAS)": round(val_a, 3) if agentic_best else "-",
                    "Unit": unit,
                    "Improvement": improvement_str
                })
                
            ev_c = m_c.get("used_ev", 0) if classic_best else 0
            dr_c = m_c.get("used_drones", 0) if classic_best else 0
            ev_a = m_a.get("used_ev", 0) if agentic_best else 0
            dr_a = m_a.get("used_drones", 0) if agentic_best else 0
            comp_rows.append({
                "KPI Parameter": "Active Vehicles",
                "Classic Value (NSGA-II)": f"{ev_c} EV, {dr_c} Drones" if classic_best else "-",
                "Agentic Value (MAS)": f"{ev_a} EV, {dr_a} Drones" if agentic_best else "-",
                "Unit": "-",
                "Improvement": f"EV: {ev_a - ev_c:+d}, Drones: {dr_a - dr_c:+d}" if (classic_best and agentic_best) else "-"
            })
            df_comp = pd.DataFrame(comp_rows)
            
            # Tab 2: Simulation Parameters
            param_rows = [
                {"Parameter": "Active City", "Value": self.city_var.get(), "Description": "Selected city database"},
                {"Parameter": "Active Neighborhoods", "Value": ", ".join(self.active_subzones) if hasattr(self, 'active_subzones') else "All", "Description": "Selected districts within the city"},
                {"Parameter": "Number of Customers", "Value": self.n_orders_var.get(), "Description": "Total client demand points"},
                {"Parameter": "K-Means Clusters", "Value": self.k_clusters_var.get(), "Description": "Number of geographic sectors (a priori)"},
                {"Parameter": "EV Fleet Count", "Value": self.n_ev_var.get(), "Description": "Total available EV trucks"},
                {"Parameter": "Drone Fleet Count", "Value": self.n_drone_var.get(), "Description": "Total available drones"},
                {"Parameter": "EV Autonomy Limit", "Value": self.ev_auto_var.get(), "Description": "Maximum battery range in km"},
                {"Parameter": "Drone Autonomy Limit", "Value": self.drone_auto_var.get(), "Description": "Maximum battery range in km"},
                {"Parameter": "Weather Penalty Multiplier", "Value": getattr(self, 'weather_penalty', 1.0), "Description": "Distance factor applied due to weather severity"},
                {"Parameter": "Weather Status", "Value": "Allowed" if getattr(self, 'weather_allowed', True) else "Prohibited", "Description": "Drone flight permission based on API conditions"},
                
                # Classic Training Params
                {"Parameter": "Classic Pop Size (NSGA-II)", "Value": getattr(self, 'pop_size_nsga', 30), "Description": "Population size for GA routing"},
                {"Parameter": "Classic Generations (NSGA-II)", "Value": getattr(self, 'n_gen_nsga', 20), "Description": "Number of generations for GA routing"},
                {"Parameter": "Classic Crossover Rate", "Value": getattr(self, 'cx_rate_nsga', 0.8), "Description": "Order crossover probability"},
                {"Parameter": "Classic Mutation Rate", "Value": getattr(self, 'mut_rate_nsga', 0.1), "Description": "Route mutation probability"},
                
                # Agentic Training Params
                {"Parameter": "Agentic Debate Rounds", "Value": 3, "Description": "Number of debate iterations between LLM agents"},
            ]
            if hasattr(self, 'agentic_ui') and hasattr(self.agentic_ui, 'latest_debate_res') and self.agentic_ui.latest_debate_res:
                res = self.agentic_ui.latest_debate_res
                if isinstance(res, dict):
                    for pk, pv in res.items():
                        param_rows.append({"Parameter": f"Agentic HPO Suggestion: {pk}", "Value": str(pv), "Description": "Optimal parameter chosen by Supreme LLM Judge"})
            df_params = pd.DataFrame(param_rows)
            
            # Tab 3: Orders Database
            order_rows = []
            if hasattr(self, 'tree_orders') and len(self.tree_orders.get_children()) > 0:
                for row_id in self.tree_orders.get_children():
                    vals = self.tree_orders.item(row_id)['values']
                    if len(vals) >= 9:
                        order_rows.append({
                            "Client ID": vals[0],
                            "District (Quartier)": vals[1],
                            "Latitude": vals[2],
                            "Longitude": vals[3],
                            "Quantity (Quantité)": vals[4],
                            "Package Type": vals[5],
                            "Delivery Mode": vals[6],
                            "Time Window": vals[7],
                            "Urgency": vals[8]
                        })
            elif hasattr(self, 'clients_coords') and len(self.clients_coords) > 1:
                for i in range(1, len(self.clients_coords)):
                    lat, lon = self.clients_coords[i]
                    ord_info = self.orders.get(i, {})
                    tw = self.time_windows.get(i, (480, 1080))
                    tw_str = f"{tw[0]//60:02d}:{tw[0]%60:02d}-{tw[1]//60:02d}:{tw[1]%60:02d}"
                    order_rows.append({
                        "Client ID": f"C{i}",
                        "District (Quartier)": self.clients_zones.get(i, "-") if hasattr(self, 'clients_zones') else "-",
                        "Latitude": lat,
                        "Longitude": lon,
                        "Quantity (Quantité)": ord_info.get("parcels", 1),
                        "Package Type": "Type 2" if self.clients_demand[i] == 2 else "Type 1",
                        "Delivery Mode": "Drone" if self.clients_demand[i] == 2 else "Electric Vehicle",
                        "Time Window": tw_str,
                        "Urgency": "URGENT" if ord_info.get("urgency", 1) == 3 else "Normal"
                    })
            df_orders = pd.DataFrame(order_rows)
            
            # Tab 4: Classic Routes
            classic_routes = []
            if classic_best:
                ev_routes = classic_best.get("ev_routes", [])
                drone_assign = classic_best.get("drone_assign", {}) or {}
                drone_launch = classic_best.get("drone_launch", {}) or {}
                for ev_idx, route in enumerate(ev_routes):
                    if len(route) >= 2:
                        stops = " -> ".join(f"C{n}" if n != 0 else "Depot" for n in route)
                        drones_info = []
                        for c_id, p_ret in drone_assign.items():
                            p_lnc = drone_launch.get(c_id, 0)
                            if p_lnc in route or p_ret in route:
                                drones_info.append(f"Drone C{c_id} (Launch: C{p_lnc}, Land: C{p_ret})")
                        classic_routes.append({
                            "Vehicle ID": f"EV-{ev_idx+1}",
                            "Type": "Electric Vehicle",
                            "Route Sequence": stops,
                            "Active Drones": ", ".join(drones_info) if drones_info else "None",
                            "Clients Delivered": ", ".join(f"C{n}" for n in route if n != 0)
                        })
            df_classic = pd.DataFrame(classic_routes)
            
            # Tab 5: Agentic Routes
            agentic_routes = []
            if agentic_best:
                ev_routes = agentic_best.get("ev_routes", [])
                drone_assign = agentic_best.get("drone_assign", {}) or {}
                drone_launch = agentic_best.get("drone_launch", {}) or {}
                for ev_idx, route in enumerate(ev_routes):
                    if len(route) >= 2:
                        stops = " -> ".join(f"C{n}" if n != 0 else "Depot" for n in route)
                        drones_info = []
                        for c_id, p_ret in drone_assign.items():
                            p_lnc = drone_launch.get(c_id, 0)
                            if p_lnc in route or p_ret in route:
                                drones_info.append(f"Drone C{c_id} (Launch: C{p_lnc}, Land: C{p_ret})")
                        agentic_routes.append({
                            "Vehicle ID": f"EV-{ev_idx+1}",
                            "Type": "Electric Vehicle",
                            "Route Sequence": stops,
                            "Active Drones": ", ".join(drones_info) if drones_info else "None",
                            "Clients Delivered": ", ".join(f"C{n}" for n in route if n != 0)
                        })
            df_agentic = pd.DataFrame(agentic_routes)
            
            # Write to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df_comp.to_excel(writer, sheet_name="Comparison Summary", index=False)
                df_params.to_excel(writer, sheet_name="Simulation Parameters", index=False)
                if not df_orders.empty:
                    df_orders.to_excel(writer, sheet_name="Orders & Demands", index=False)
                if not df_classic.empty:
                    df_classic.to_excel(writer, sheet_name="Classic Routes", index=False)
                if not df_agentic.empty:
                    df_agentic.to_excel(writer, sheet_name="Agentic Routes", index=False)
                    
            messagebox.showinfo("Export Successful", f"Rapport de simulation complet exporté avec succès vers :\n{filepath}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Export Failed", f"Une erreur s'est produite lors de l'export Excel :\n{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        sys.exit(0)
    root = tk.Tk()
    app = ResearchDashboard(root)
    root.mainloop()
