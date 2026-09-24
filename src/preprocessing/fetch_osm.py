"""
Downloads the drivable OpenStreetMap road network around each IO-VNBD trip (once) via the
Overpass API and stores it as GeoJSON LineStrings in data/maps/<trip>.geojson for offline
map matching.

The corridor (default 400 m either side of the reference route) contains all nearby roads,
including parallel streets, junction arms and service roads, so the matcher has to pick
the right road rather than being handed the answer.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.parse_trip import OUT_DIR as INTERMEDIATE_DIR, load_trip

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
HIGHWAY_TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|road|"
                 "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
MAP_DIR = "data/maps"


def route_points(lat, lon, spacing_m=150.0):
    """Down-samples the GT track to points ~spacing_m apart (keeps the Overpass query small)."""
    pts = [(lat[0], lon[0])]
    k = 111320.0
    for la, lo in zip(lat[::10], lon[::10]):
        dy = (la - pts[-1][0]) * k
        dx = (lo - pts[-1][1]) * k * np.cos(np.radians(la))
        if dx * dx + dy * dy >= spacing_m ** 2:
            pts.append((la, lo))
    return pts


def overpass(query):
    data = urllib.parse.urlencode({"data": query}).encode()
    last_err = None
    for url in OVERPASS_URLS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "PS26168-IDR/1.0"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read().decode())
            except Exception as e:  # rate limits / timeouts: back off and retry
                last_err = e
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last_err}")


def fetch_trip_roads(trip, corridor_m=400.0, chunk=120):
    d = load_trip(trip, INTERMEDIATE_DIR)
    pts = route_points(d["lat"], d["lon"])
    ways = {}
    for i in range(0, len(pts), chunk):
        coords = ",".join(f"{la:.6f},{lo:.6f}" for la, lo in pts[i:i + chunk + 1])
        q = f'[out:json][timeout:240];way["highway"~"^({HIGHWAY_TYPES})$"](around:{corridor_m:.0f},{coords});out geom tags;'
        res = overpass(q)
        for el in res.get("elements", []):
            if el.get("type") == "way" and "geometry" in el:
                ways[el["id"]] = el
        print(f"  {trip}: chunk {i // chunk + 1}/{(len(pts) + chunk - 1) // chunk} -> {len(ways)} ways")
        time.sleep(2)
    features = []
    for wid, el in ways.items():
        tags = el.get("tags", {})
        features.append({
            "type": "Feature",
            "properties": {"id": wid, "highway": tags.get("highway"), "name": tags.get("name"),
                           "oneway": tags.get("oneway"), "junction": tags.get("junction")},
            "geometry": {"type": "LineString", "coordinates": [[round(p["lon"], 7), round(p["lat"], 7)]
                                                               for p in el["geometry"]]},
        })
    return {"type": "FeatureCollection",
            "properties": {"trip": trip, "corridor_m": corridor_m, "source": "OpenStreetMap contributors (ODbL)"},
            "features": features}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", default=["S1", "Y1", "S2", "S4", "M", "Vw02", "Vw03", "Vta01b", "Vta06", "Vtb01"])
    ap.add_argument("--corridor", type=float, default=400.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(MAP_DIR, exist_ok=True)
    for trip in a.trips:
        path = os.path.join(MAP_DIR, f"{trip}.geojson")
        if os.path.exists(path) and not a.force:
            print(f"Skipping {trip} (exists)")
            continue
        fc = fetch_trip_roads(trip, a.corridor)
        with open(path, "w") as f:
            json.dump(fc, f, separators=(",", ":"))
        print(f"Saved {len(fc['features'])} roads to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
