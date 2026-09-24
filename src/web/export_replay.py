"""
Exports an IO-VNBD trip segment for the web navigation app's Replay mode:
  src/web/nav/replay/<trip>.json  - 10 Hz IMU, 1 Hz simulated GNSS fixes, 1 Hz ground-truth track
  src/web/nav/maps/<trip>.geojson - OSM roads around the segment (clipped from data/maps/)

Usage: python src/web/export_replay.py --trip S1 --start 600 --minutes 30
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.evaluation.benchmark import load_bench_trip, synthetic_gnss

NAV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav")


def export(trip_name, start_s, minutes, margin_m=400.0):
    trip = load_bench_trip(trip_name)
    fixes = synthetic_gnss(trip)
    i0 = int(start_s * 10)
    i1 = min(len(trip["t"]), i0 + int(minutes * 600))
    sl = slice(i0, i1)
    imu = np.concatenate([trip["acc"][sl], trip["gyro"][sl]], axis=1)
    gnss = []
    for i in range(i0, i1):
        f = fixes[i]
        if f is not None:
            gnss.append([i - i0, round(f["lat"], 7), round(f["lon"], 7), round(f["speed"], 2),
                         None if f["heading"] is None else round(f["heading"], 1), f["accuracy"]])
    gt = [[i - i0, round(float(trip["lat"][i]), 7), round(float(trip["lon"][i]), 7)] for i in range(i0, i1, 10)]
    data = {"trip": trip_name, "rate_hz": 10, "start_s": start_s, "frames": i1 - i0,
            "imu": np.round(imu, 4).ravel().tolist(), "gnss": gnss, "gt": gt,
            "note": "IO-VNBD phone IMU (aligned, 10 Hz). GNSS = ground truth + simulated receiver noise, 1 Hz."}
    os.makedirs(os.path.join(NAV_DIR, "replay"), exist_ok=True)
    out = os.path.join(NAV_DIR, "replay", f"{trip_name}.json")
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    e, n = trip["east"][sl], trip["north"][sl]
    src = os.path.join("data/maps", f"{trip_name}.geojson")
    with open(src) as f:
        gj = json.load(f)
    feats = []
    for feat in gj["features"]:
        coords = np.asarray(feat["geometry"]["coordinates"])
        fe, fn = trip["ltp"].to_enu(coords[:, 1], coords[:, 0])
        if np.any((fe > e.min() - margin_m) & (fe < e.max() + margin_m) & (fn > n.min() - margin_m) &
                  (fn < n.max() + margin_m)):
            feats.append(feat)
    os.makedirs(os.path.join(NAV_DIR, "maps"), exist_ok=True)
    map_out = os.path.join(NAV_DIR, "maps", f"{trip_name}.geojson")
    with open(map_out, "w") as f:
        json.dump({"type": "FeatureCollection", "properties": gj.get("properties", {}), "features": feats}, f,
                  separators=(",", ":"))
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {i1 - i0} frames) and {map_out} "
          f"({len(feats)} roads, {os.path.getsize(map_out) / 1e6:.1f} MB)")

    index_path = os.path.join(NAV_DIR, "replay", "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else []
    index = [x for x in index if x["trip"] != trip_name] + [
        {"trip": trip_name, "minutes": round((i1 - i0) / 600, 1), "file": f"{trip_name}.json",
         "map": f"../maps/{trip_name}.geojson"}]
    with open(index_path, "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="S1")
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--minutes", type=float, default=30.0)
    a = ap.parse_args()
    export(a.trip, a.start, a.minutes)
