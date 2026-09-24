"""
Quick end-to-end replay: one IO-VNBD trip through the full navigation pipeline with a simulated
GNSS tunnel, printing mode statistics, sample NavigationState outputs and the position error.
For the full PS benchmark (many outages, ablations, plots) use src/evaluation/benchmark.py.

Usage: python src/navigation/replay.py [--trip S1] [--outage-start 1200] [--outage-seconds 60]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.evaluation.benchmark import load_bench_trip, make_engine, replay, score_outages, synthetic_gnss


def run_sensor_replay(trip_name="S1", outage_start_s=1200.0, outage_s=60.0, config="ekf_ai_map"):
    print(f"--- Replaying trip {trip_name} ({config}) with a {outage_s:.0f} s GNSS blackout ---")
    trip = load_bench_trip(trip_name)
    fixes = synthetic_gnss(trip)
    s = int(outage_start_s * 10)
    outage = [(s, s + int(outage_s * 10))]
    t0 = time.time()
    est, modes, lat_ms = replay(trip, make_engine(config, trip_name), fixes, outage)
    elapsed = time.time() - t0
    n = len(modes)
    print(f"Replayed {n} frames in {elapsed:.1f} s ({n / elapsed:.0f} frames/s, {lat_ms.mean():.2f} ms/frame)")
    for mode in sorted(set(modes)):
        c = int(np.sum(modes == mode))
        print(f"  {mode:16s} {c:6d} frames ({100 * c / n:.1f} %)")
    row = score_outages(trip, est, modes, outage)[0]
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, indent=1))
    gt = np.stack([trip["east"], trip["north"]], axis=1)
    err = np.hypot(*(est - gt).T)
    print(f"GNSS-aided position error: median {np.nanmedian(err[modes == 'GNSS_AIDED']):.2f} m")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="S1")
    ap.add_argument("--outage-start", type=float, default=1200.0)
    ap.add_argument("--outage-seconds", type=float, default=60.0)
    ap.add_argument("--config", default="ekf_ai_map", choices=["ekf_only", "ekf_ai", "ekf_ai_map"])
    a = ap.parse_args()
    run_sensor_replay(a.trip, a.outage_start, a.outage_seconds, a.config)
