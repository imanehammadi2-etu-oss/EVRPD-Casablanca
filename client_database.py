"""
client_database.py
==================
Module de generation et gestion d'une base de donnees de 3000 clients
repartis dans les quartiers reels de Casablanca, Maroc.

Fonctionnalites :
  - Generation de clients avec coordonnees GPS realistes par quartier
  - Fenetres horaires de livraison realistes
  - Systeme de orders avec attribution automatique du mode de livraison
  - Persistance JSON avec chargement/creation automatique

Auteur : Genere automatiquement
Date   : 2026-07-03
"""

import json
import os
import random
from datetime import datetime

# ============================================================================
# DONNÉES GÉOGRAPHIQUES DES QUARTIERS DE CASABLANCA
# Chaque quartier est defini par ses bornes GPS : (lat_min, lat_max, lon_min, lon_max)
# ============================================================================

CASABLANCA_QUARTIERS = {
    "Maarif":          (33.575, 33.595, -7.640, -7.610),
    "Ain Chok":        (33.540, 33.570, -7.640, -7.580),
    "Ain Sebaa":       (33.585, 33.599, -7.560, -7.520),
    "Oulfa":           (33.540, 33.570, -7.670, -7.630),
    "Hay Hassani":     (33.560, 33.590, -7.670, -7.630),
    "Sidi Maarouf":    (33.520, 33.550, -7.650, -7.600),
    "Anfa":            (33.585, 33.598, -7.670, -7.640),
    "Bernoussi":       (33.590, 33.599, -7.510, -7.480),
    "Derb Sultan":     (33.565, 33.585, -7.610, -7.580),
    "Bourgogne":       (33.590, 33.598, -7.650, -7.630),
    "Belvedere":       (33.575, 33.590, -7.600, -7.580),
    "Gauthier":        (33.590, 33.598, -7.630, -7.610),
    "Sidi Bernoussi":  (33.580, 33.599, -7.520, -7.490),
    "Sidi Othmane":    (33.545, 33.570, -7.580, -7.550),
    "Hay Mohammadi":   (33.575, 33.595, -7.560, -7.530),
    "Medina":          (33.590, 33.598, -7.620, -7.595),
    "Roches Noires":   (33.595, 33.599, -7.580, -7.560),
    "Sbata":           (33.555, 33.575, -7.580, -7.555),
    "Lissasfa":        (33.520, 33.545, -7.600, -7.570),
    "Moulay Rachid":   (33.545, 33.570, -7.555, -7.525),
    "Ben Msick":       (33.560, 33.580, -7.570, -7.545),
    "Al Fida":         (33.570, 33.590, -7.590, -7.565),
    "Sidi Belyout":    (33.585, 33.598, -7.625, -7.605),
    "Oasis":           (33.555, 33.575, -7.640, -7.615),
    "Mers Sultan":     (33.585, 33.598, -7.610, -7.590),
}

def is_port_or_sea(lat: float, lon: float) -> bool:
    """Détecte et élimine les coordonnées dans le port de Casablanca ou dans l'eau."""
    if lat >= 33.5990 and -7.635 <= lon <= -7.585:
        return True
    if lat >= 33.6040:
        return True
    return False

# ============================================================================
# DISTRIBUTION DE LA POPULATION PAR QUARTIER
# ============================================================================

_HAUTE_DENSITE = {
    "Maarif":        160,
    "Ain Chok":      165,
    "Hay Hassani":   160,
    "Derb Sultan":   155,
    "Sidi Othmane":  150,
    "Hay Mohammadi": 155,
    "Ben Msick":     150,
    "Al Fida":       150,
    "Moulay Rachid": 155,
}

_MOYENNE_DENSITE = {
    "Ain Sebaa":       120,
    "Oulfa":           125,
    "Sidi Maarouf":    120,
    "Bernoussi":       115,
    "Sidi Bernoussi":  120,
    "Sbata":           110,
    "Lissasfa":        110,
    "Medina":          120,
}

_BASSE_DENSITE = {
    "Anfa":          75,
    "Bourgogne":     80,
    "Belvedere":     85,
    "Gauthier":      75,
    "Roches Noires": 85,
    "Sidi Belyout":  80,
    "Oasis":         90,
    "Mers Sultan":   90,
}
# Sous-total basse densite : 660

# Fusionner toutes les distributions dans un seul dictionnaire
DISTRIBUTION_QUARTIERS = {}
DISTRIBUTION_QUARTIERS.update(_HAUTE_DENSITE)
DISTRIBUTION_QUARTIERS.update(_MOYENNE_DENSITE)
DISTRIBUTION_QUARTIERS.update(_BASSE_DENSITE)

# Verifier et ajuster le total pour atteindre exactement 3000
_total_brut = sum(DISTRIBUTION_QUARTIERS.values())
_difference = 3000 - _total_brut

if _difference != 0:
    # Ajuster le quartier le plus peuple pour compenser l'ecart
    _quartier_ajustement = max(DISTRIBUTION_QUARTIERS, key=DISTRIBUTION_QUARTIERS.get)
    DISTRIBUTION_QUARTIERS[_quartier_ajustement] += _difference

assert sum(DISTRIBUTION_QUARTIERS.values()) == 3000, (
    f"Error : le total des clients doit etre 3000, obtenu {sum(DISTRIBUTION_QUARTIERS.values())}"
)

# ============================================================================
# FENÊTRES HORAIRES DE LIVRAISON
# Repartition realiste :
#   - 70% creneaux matinaux
#   - 20% creneaux apres-midi
#   - 10% journee complete
# ============================================================================

FENETRES_MATIN = [
    ("08:00", "12:00"),
    ("09:00", "13:00"),
    ("08:00", "11:00"),
]

FENETRES_APRES_MIDI = [
    ("12:00", "16:00"),
    ("13:00", "17:00"),
    ("14:00", "18:00"),
]

FENETRE_JOURNEE = [
    ("08:00", "18:00"),
]


def _choisir_fenetre_horaire() -> tuple:
    """
    Choisit une fenetre horaire de livraison selon la distribution definie :
      - 70% matin, 20% apres-midi, 10% journee complete.

    Returns:
        tuple: (tw_start, tw_end) sous forme de chaines 'HH:MM'.
    """
    tirage = random.random()

    if tirage < 0.70:
        # Creneau matinal (70% de probabilite)
        return random.choice(FENETRES_MATIN)
    elif tirage < 0.90:
        # Creneau apres-midi (20% de probabilite)
        return random.choice(FENETRES_APRES_MIDI)
    else:
        # Journee complete (10% de probabilite)
        return random.choice(FENETRE_JOURNEE)


# ============================================================================
# GÉNÉRATION DES CLIENTS
# ============================================================================

def generate_casablanca_clients(n: int = 3000) -> list:
    """
    Genere une liste de `n` clients repartis dans les quartiers de Casablanca.

    Chaque client possede :
      - id         : identifiant entier unique (commencant a 1)
      - nom        : nom formate 'Client_XXXX'
      - quartier   : nom du quartier de Casablanca
      - latitude   : coordonnee GPS (latitude) realiste dans le quartier
      - longitude  : coordonnee GPS (longitude) realiste dans le quartier
      - tw_start   : debut de la fenetre de livraison (format 'HH:MM')
      - tw_end     : fin de la fenetre de livraison (format 'HH:MM')

    La graine aleatoire est fixee a 42 pour la reproductibilite.

    Args:
        n: Nombre total de clients a generer (par defaut 3000).

    Returns:
        list: Liste de dictionnaires, un par client.
    """
    # Fixer la graine pour garantir la reproductibilite des resultats
    random.seed(42)

    clients = []
    client_id = 1

    # Calculer le facteur d'echelle si n != 3000
    # On garde les proportions relatives entre quartiers
    if n != 3000:
        facteur = n / 3000.0
        distribution = {}
        total_ajuste = 0

        quartiers_list = list(DISTRIBUTION_QUARTIERS.keys())
        for i, quartier in enumerate(quartiers_list):
            if i < len(quartiers_list) - 1:
                nb = round(DISTRIBUTION_QUARTIERS[quartier] * facteur)
                distribution[quartier] = nb
                total_ajuste += nb
            else:
                # Dernier quartier : compenser pour atteindre exactement n
                distribution[quartier] = n - total_ajuste
    else:
        distribution = DISTRIBUTION_QUARTIERS.copy()

    # Generer les clients quartier par quartier
    for quartier, nb_clients in distribution.items():
        # Recuperer les bornes GPS du quartier
        lat_min, lat_max, lon_min, lon_max = CASABLANCA_QUARTIERS[quartier]

        for _ in range(nb_clients):
            # Generer des coordonnees aleatoires terrestres hors du port
            while True:
                latitude = round(random.uniform(lat_min, lat_max), 6)
                longitude = round(random.uniform(lon_min, lon_max), 6)
                if not is_port_or_sea(latitude, longitude):
                    break

            # Choisir une fenetre horaire de livraison
            tw_start, tw_end = _choisir_fenetre_horaire()

            # Creer l'enregistrement client
            client = {
                "id": client_id,
                "nom": f"Client_{client_id:04d}",
                "quartier": quartier,
                "latitude": latitude,
                "longitude": longitude,
                "tw_start": tw_start,
                "tw_end": tw_end,
            }

            clients.append(client)
            client_id += 1

    return clients


# ============================================================================
# PERSISTANCE JSON - CHARGEMENT OU CRÉATION DE LA BASE
# ============================================================================

def load_or_create_database(filepath: str = None) -> dict:
    """
    Charge la base de donnees clients depuis un fichier JSON existant,
    ou la cree si le fichier n'existe pas.
    """
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "clients_database.json")

    # Tenter de charger la base existante
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            database = json.load(f)
        # Assurer qu'aucun client chargé n'est dans la zone du port
        for c in database.get("clients", []):
            if is_port_or_sea(c.get("latitude", 0.0), c.get("longitude", 0.0)):
                c["latitude"] = round(random.uniform(33.585, 33.597), 6)
                c["longitude"] = round(random.uniform(-7.620, -7.600), 6)
                c["quartier"] = "Sidi Belyout"
        print(f"[INFO] Database loaded depuis : {filepath}")
        print(f"       -> {len(database['clients'])} clients found.")
        return database

    # Sinon, generer une nouvelle base de donnees
    print("[INFO] File not found. Generating a new database...")
    clients = generate_casablanca_clients()

    # Collecter les quartiers uniques presents dans la base
    quartiers_uniques = sorted(set(c["quartier"] for c in clients))

    # Construire la structure complete avec metadonnees
    database = {
        "metadata": {
            "nombre_total_clients": len(clients),
            "nombre_quartiers": len(quartiers_uniques),
            "quartiers": quartiers_uniques,
            "ville": "Casablanca",
            "pays": "Maroc",
            "date_generation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "graine_aleatoire": 42,
        },
        "clients": clients,
    }

    # Sauvegarder la base au format JSON (encodage UTF-8, formate lisiblement)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Database generated and saved in: {filepath}")
    print(f"       -> {len(clients)} clients repartis dans {len(quartiers_uniques)} quartiers.")

    return database


# ============================================================================
# GÉNÉRATION DE COMMANDES ALÉATOIRES
# ============================================================================

def get_random_orders(database: dict, n_orders: int) -> list:
    """
    Selectionne aleatoirement `n_orders` clients uniques dans la base de donnees
    et genere une commande pour chacun d'eux.

    Chaque commande contient :
      - Toutes les informations du client (id, nom, quartier, coordonnees, fenetre horaire)
      - type_colis     : "Type 2" (40%) ou "Type 1" (60%)
      - quantite       : entier entre 1 et 5
      - urgence        : True (20%) ou False (80%)
      - mode_livraison : determine automatiquement selon les regles :
          * quantite == 1 ET type_colis == "Type 2"  → "Drone"
          * type_colis == "Type 1"                    → "Electric Vehicle"
          * Sinon (plusieurs Type 2 ou mixte)         → "Electric Vehicle"

    Args:
        database: Dictionnaire de la base de donnees (format load_or_create_database).
        n_orders: Number of orders a generer.

    Returns:
        list: Liste de dictionnaires de orders.

    Raises:
        ValueError: Si n_orders depasse le nombre total de clients disponibles.
    """
    clients = database["clients"]

    # Verifier que le nombre de orders ne depasse pas le nombre de clients
    if n_orders > len(clients):
        raise ValueError(
            f"Impossible to select {n_orders} orders uniques parmi "
            f"{len(clients)} clients. Reduisez n_orders."
        )

    # Selectionner n_orders clients uniques aleatoirement
    clients_selectionnes = random.sample(clients, n_orders)

    orders = []

    for client in clients_selectionnes:
        # --- Determiner le type de colis ---
        # 40% de chance d'etre "Type 2" (Drone), 60% d'etre "Type 1" (EV)
        type_colis = "Type 2" if random.random() < 0.40 else "Type 1"

        # --- Determiner la quantite (jusqu'a 100 colis inclus pour saturer la capacite des EVs) ---
        # 60% de petites commandes (1 a 15 colis)
        # 25% de commandes moyennes (15 a 60 colis)
        # 15% de grandes commandes (60 a 100 colis)
        r_val = random.random()
        if r_val < 0.60:
            quantite = random.randint(1, 15)
        elif r_val < 0.85:
            quantite = random.randint(15, 60)
        else:
            quantite = random.randint(60, 100)

        # --- Determiner l'urgence selon le Time Window ---
        # Si la fenetre horaire est strictement inferieure a 4 heures (240 minutes) => URGENT
        h_s, m_s = map(int, client["tw_start"].split(':'))
        h_e, m_e = map(int, client["tw_end"].split(':'))
        duration_minutes = (h_e * 60 + m_e) - (h_s * 60 + m_s)
        
        urgence = duration_minutes < 240

        # --- Determiner le mode de livraison selon les regles metier ---
        if quantite == 1 and type_colis == "Type 2":
            # Un seul colis Type 2 → livraison par drone possible
            mode_livraison = "Drone"
        elif type_colis == "Type 1":
            # Parcels Type 1 → necessite un vehicule electrique
            mode_livraison = "Electric Vehicle"
        else:
            # Plusieurs colis Type 2 → vehicule electrique requis
            mode_livraison = "Electric Vehicle"

        # Construire le dictionnaire de la commande
        # On copie d'abord les donnees du client puis on ajoute les infos de commande
        commande = {
            "id": client["id"],
            "nom": client["nom"],
            "quartier": client["quartier"],
            "latitude": client["latitude"],
            "longitude": client["longitude"],
            "tw_start": client["tw_start"],
            "tw_end": client["tw_end"],
            "type_colis": type_colis,
            "quantite": quantite,
            "urgence": urgence,
            "mode_livraison": mode_livraison,
        }

        orders.append(commande)

    return orders


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL - pour tester le module directement
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  GÉNÉRATION DE LA BASE DE DONNÉES CLIENTS - CASABLANCA, MAROC")
    print("=" * 70)
    print()

    # Load ou creer la base de donnees
    db = load_or_create_database()

    print()
    print("-" * 70)
    print("  STATISTIQUES DE LA BASE")
    print("-" * 70)

    # Afficher la repartition par quartier
    from collections import Counter
    repartition = Counter(c["quartier"] for c in db["clients"])
    print(f"\n  {'District':<20} {'Num Clients':>12}")
    print(f"  {'─' * 20} {'─' * 12}")
    for quartier in sorted(repartition.keys()):
        print(f"  {quartier:<20} {repartition[quartier]:>12}")
    print(f"  {'─' * 20} {'─' * 12}")
    print(f"  {'TOTAL':<20} {sum(repartition.values()):>12}")

    # Afficher la repartition des fenetres horaires
    print(f"\n  Time windows distribution:")
    fenetres = Counter(f"{c['tw_start']}-{c['tw_end']}" for c in db["clients"])
    for fenetre, count in sorted(fenetres.items()):
        pct = count / len(db["clients"]) * 100
        print(f"    {fenetre:<15} : {count:>5} clients ({pct:5.1f}%)")

    # Tester la generation de orders
    print()
    print("-" * 70)
    print("  EXEMPLE DE COMMANDES GÉNÉRÉES")
    print("-" * 70)

    orders = get_random_orders(db, 10)
    for cmd in orders[:5]:
        print(f"\n   {cmd['nom']} - {cmd['quartier']}")
        print(f"     GPS : ({cmd['latitude']}, {cmd['longitude']})")
        print(f"     Window : {cmd['tw_start']} → {cmd['tw_end']}")
        print(f"     Parcels : {cmd['type_colis']} x{cmd['quantite']}")
        print(f"     Urgent : {'Oui ⚡' if cmd['urgence'] else 'Non'}")
        print(f"     Mode : {cmd['mode_livraison']}")

    print()
    print("=" * 70)
    print("  Module client_database.py pret a l'emploi.")
    print("=" * 70)
