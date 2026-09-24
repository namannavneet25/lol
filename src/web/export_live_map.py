"""
Downloads drivable OpenStreetMap roads around a location into src/web/nav/maps/live.geojson, so the
navigator's Live mode can do map matching offline (the replay maps only cover the IO-VNBD trips).

Usage: python src/web/export_live_map.py --lat 26.1445 --lon 91.7362 --radius-km 5
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.fetch_osm import HIGHWAY_TYPES, overpass

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav", "maps", "live.geojson")


def fetch_area(lat, lon, radius_km):
    q = (f'[out:json][timeout:240];way["highway"~"^({HIGHWAY_TYPES})$"]'
         f'(around:{radius_km * 1000:.0f},{lat:.6f},{lon:.6f});out geom tags;')
    feats = []
    for el in overpass(q).get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        feats.append({"type": "Feature",
                      "properties": {"id": el["id"], "highway": tags.get("highway"), "name": tags.get("name"),
                                     "oneway": tags.get("oneway"), "junction": tags.get("junction")},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[round(p["lon"], 7), round(p["lat"], 7)] for p in el["geometry"]]}})
    return {"type": "FeatureCollection",
            "properties": {"center": [lat, lon], "radius_km": radius_km, "source": "OpenStreetMap contributors (ODbL)"},
            "features": feats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius-km", type=float, default=5.0)
    a = ap.parse_args()
    fc = fetch_area(a.lat, a.lon, a.radius_km)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    print(f"Saved {len(fc['features'])} roads to {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
