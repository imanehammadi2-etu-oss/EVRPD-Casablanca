import json
import requests
import os

cleaned_coords = [
    [33.5700, -7.5800],  # 0: Dépôt Maarif
    [33.5955, -7.6140],  # 1: Sidi Belyout / Bd des FAR
    [33.5880, -7.5937],  # 2: Mers Sultan / Derb Omar
    [33.5621, -7.5448],  # 3: Ben Msick
    [33.5480, -7.6137],  # 4: Oasis / Ain Chok
    [33.5907, -7.6296],  # 5: Gauthier / Bd Moulay Youssef
    [33.5524, -7.5904],  # 6: Ain Chok
    [33.5347, -7.5780],  # 7: Californie
    [33.5747, -7.5562],  # 8: Hay Mohammadi
    [33.5607, -7.6126],  # 9: Derb Ghallef
    [33.5613, -7.6352],  # 10: Maarif Extension
    [33.5620, -7.6490],  # 11: Maarif Université
    [33.5865, -7.6674],  # 12: Anfa
    [33.5915, -7.6170],  # 13: Centre-Ville / Bd de Paris
    [33.5960, -7.5020],  # 14: Sidi Bernoussi
    [33.5935, -7.6185],  # 15: Ancienne Médina / Bab Marrakech
    [33.5512, -7.5391],  # 16: Sbata
    [33.5950, -7.4980],  # 17: Bernoussi Ouest
    [33.5920, -7.5680],  # 18: Belvédère / Place Albert 1er
    [33.5792, -7.6627],  # 19: Anfa Supérieur
    [33.5950, -7.5294],  # 20: Ain Sebaa
    [33.5875, -7.6240],  # 21: Gauthier / Rue Moussa Ben Noussair
    [33.5910, -7.5080],  # 22: Sidi Bernoussi Z.I.
    [33.5577, -7.5611],  # 23: Sidi Othmane
    [33.5722, -7.6381],  # 24: Maarif Val Fleuri
    [33.5919, -7.4946],  # 25: Sidi Bernoussi Centre
    [33.5276, -7.5827],  # 26: Sidi Maarouf / Zénith
    [33.5766, -7.5395],  # 27: Hay Mohammadi
    [33.5925, -7.6330],  # 28: Bourgogne / Bd Zerktouni
    [33.5936, -7.5572],  # 29: Roches Noires
    [33.5722, -7.5697]   # 30: Al Fida
]

VE_CONFIG = [
    {'id': 1, 'name': 'VE #01 (Maarif / Anfa)', 'route_nodes': [0, 24, 11, 19, 12, 0]},
    {'id': 2, 'name': 'VE #02 (Bourgogne / Gauthier)', 'route_nodes': [0, 15, 1, 13, 28, 21, 5, 10, 0]},
    {'id': 3, 'name': 'VE #03 (Sidi Maarouf / Technopark)', 'route_nodes': [0, 30, 23, 2, 0]},
    {'id': 4, 'name': 'VE #04 (Ain Sebaa / Belvédère)', 'route_nodes': [0, 8, 18, 20, 22, 17, 25, 14, 0]},
    {'id': 5, 'name': 'VE #05 (Hay Hassani)', 'route_nodes': [0, 9, 4, 0]},
    {'id': 6, 'name': 'VE #06 (Oulfa / Lissasfa)', 'route_nodes': [0, 29, 27, 3, 16, 0]},
    {'id': 7, 'name': 'VE #07 (Californie / Oasis)', 'route_nodes': [0, 6, 7, 26, 0]}
]

def get_osrm_detailed_route(points):
    if len(points) < 2:
        return points
    try:
        coords = ';'.join([f"{p[1]:.6f},{p[0]:.6f}" for p in points])
        url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            coords_out = r.json()["routes"][0]["geometry"]["coordinates"]
            return [[round(c[1], 6), round(c[0], 6)] for c in coords_out]
    except Exception as e:
        print("OSRM error:", e)
    
    # Fallback by segments
    full_road = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        try:
            coords = f"{p1[1]:.6f},{p1[0]:.6f};{p2[1]:.6f},{p2[0]:.6f}"
            url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                seg_coords = [[round(c[1], 6), round(c[0], 6)] for c in r.json()["routes"][0]["geometry"]["coordinates"]]
                if full_road:
                    full_road.extend(seg_coords[1:])
                else:
                    full_road.extend(seg_coords)
                continue
        except Exception:
            pass
        if not full_road:
            full_road.append(p1)
        full_road.append(p2)
    return full_road

cache = {}
for ve in VE_CONFIG:
    pts = [cleaned_coords[n] for n in ve['route_nodes']]
    road_pts = get_osrm_detailed_route(pts)
    cache[ve['name']] = road_pts
    max_lat = max(p[0] for p in road_pts)
    print(f"{ve['name']}: {len(pts)} stops -> {len(road_pts)} waypoints, max_lat={max_lat}")

out_dir = r"C:\Users\MSI\Desktop\Forecasting\Forecasting\data"
os.makedirs(out_dir, exist_ok=True)
cache_file = os.path.join(out_dir, "osrm_casablanca_routes.json")
with open(cache_file, "w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2)

print("Saved cache successfully to", cache_file)
