"""
Dead-reckoning benchmark against the PS26168 performance targets.

Replays IO-VNBD trips through a navigation engine with a simulated 1 Hz GNSS receiver
(ground truth + correlated noise) and injects GNSS blackouts. For every outage it records the
position error when GNSS returns, as % of the distance driven during the outage
(target < 10%), plus absolute error, PS example scenarios and timing.

Configurations (ablations)
  legacy       - the original pipeline (snapshot of the code before this optimisation)
  ekf_only     - new EKF + gyro, no AI speed, no map ("classical INS")
  ekf_ai       - new EKF + AI speed/ZUPT, no map
  ekf_ai_map   - full system: EKF + AI + HMM map matching on OpenStreetMap

Usage
  python src/evaluation/benchmark.py                      # test trip S1, all configs
  python src/evaluation/benchmark.py --trips S1 Y1 --configs ekf_ai_map
  python src/evaluation/benchmark.py --legacy-root <dir>  # include the legacy engine
  python src/evaluation/benchmark.py --rate-test          # 10 Hz vs 200 Hz engine throughput
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)
from src.navigation.frames import LocalTangentPlane
from src.preprocessing.parse_trip import load_trip

RESULTS_DIR = "results"
DT = 0.1
GNSS_PERIOD = 10          # frames (1 Hz)
RECOVERY_S = 90           # GNSS-aided time between outages
MIN_OUTAGE_SPEED = 3.0    # m/s; drift-% is undefined for a parked car


# ------------------------------------------------------------------------------------------ GNSS
def synthetic_gnss(trip, seed=0, sigma_white=1.0, sigma_gm=2.0, tau_gm=30.0, latency_s=0.0):
    """
    Idealised 1 Hz fixes: GT + first-order Gauss-Markov (multipath-like) + white noise on position,
    speed noise 0.1 m/s, course noise 1 deg, constant 3 m accuracy. latency_s delays the content of
    each fix (it describes where the car was latency_s earlier), like a phone's fused location.
    """
    lag = int(round(latency_s / DT))
    rng = np.random.default_rng(seed)
    n = len(trip["t"])
    phi = math.exp(-GNSS_PERIOD * DT / tau_gm)
    gm = np.zeros(2)
    fixes = [None] * n
    for i in range(0, n, GNSS_PERIOD):
        gm = phi * gm + math.sqrt(1 - phi ** 2) * sigma_gm * rng.standard_normal(2)
        de, dn = gm + sigma_white * rng.standard_normal(2)
        j = max(0, i - lag)
        lat, lon = trip["ltp"].to_geodetic(trip["east"][j] + de, trip["north"][j] + dn)
        spd = max(0.0, float(trip["speed"][j]) + 0.1 * rng.standard_normal())
        crs = (float(trip["course_deg"][j]) + rng.normal(0, 1.0)) % 360 if spd > 1.0 else None
        fixes[i] = {"lat": lat, "lon": lon, "speed": spd, "heading": crs, "accuracy": 3.0}
    return fixes


def phone_gnss(trip):
    """
    The phone's own logged GPS from IO-VNBD: a new position only about every 9 s (0.11 Hz), real reported
    accuracy, the phone's speed and bearing (missing values -> None).
    """
    fixes = [None] * len(trip["t"])
    for i in np.where(trip["phone_gps_new"])[0]:
        lat, lon, spd, acc, brg = (float(v) for v in trip["phone_gps"][i])
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        spd = spd if math.isfinite(spd) else None
        fixes[i] = {"lat": lat, "lon": lon, "speed": spd,
                    "heading": brg if (spd is not None and spd > 1.0 and math.isfinite(brg)) else None,
                    "accuracy": acc if math.isfinite(acc) else None}
    return fixes


class OracleSpeedModel:
    """Perfect speed: returns the ground-truth speed of the current frame (set by replay) with small variance."""

    def __init__(self, trip, window=40):
        self.speed, self.window, self.i = trip["speed"], window, 0

    def predict(self, window, scalars):
        from src.models.inference import AIOutput
        v = float(self.speed[self.i])
        return AIOutput(v, 0.05 ** 2, 1.0 if v < 0.3 else 0.0)


def plan_outages(trip, duration_s, first_s=120, recovery_s=RECOVERY_S):
    """Tiles the trip with outages of `duration_s` separated by `recovery_s` of GNSS."""
    n = len(trip["t"])
    dur, rec = int(duration_s / DT), int(recovery_s / DT)
    outages, i = [], int(first_s / DT)
    while i + dur < n:
        outages.append((i, i + dur))
        i += dur + rec
    return outages


def plan_distance_outages(trip, distance_m, min_speed, max_speed, max_duration_s, spacing_s=150):
    """Outages that end once the vehicle has driven `distance_m`, restricted to a speed band (PS examples)."""
    n = len(trip["t"])
    cum = np.concatenate([[0.0], np.cumsum(trip["speed"] * DT)])
    outages, i = [], int(120 / DT)
    while i < n:
        j = int(np.searchsorted(cum, cum[i] + distance_m))
        if j >= n:
            break
        dur = (j - i) * DT
        mean_v = distance_m / max(dur, DT)
        if dur <= max_duration_s and min_speed <= mean_v <= max_speed and trip["valid"][i:j].all():
            outages.append((i, j))
            i = j + int(spacing_s / DT)
        else:
            i += int(5 / DT)
    return outages


# ------------------------------------------------------------------------------------------ engines
MODEL_PATH = "models/idr_tcn.onnx"     # the deployed artefact (identical outputs to the .pth, ~10x faster)


def make_engine(config, trip_name, map_dir="data/maps", trip=None):
    from src.models.inference import SpeedModel
    from src.navigation.pipeline import CoreNavigationPipeline
    map_path = os.path.join(map_dir, f"{trip_name}.geojson")
    model = SpeedModel(MODEL_PATH)
    if config == "ekf_only":
        return CoreNavigationPipeline(calibration_path=None, use_ai=False, use_map=False, model=model)
    if config == "ekf_ai":
        return CoreNavigationPipeline(calibration_path=None, use_map=False, model=model)
    if config == "ekf_ai_map":
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"{map_path} missing - run src/preprocessing/fetch_osm.py")
        return CoreNavigationPipeline(calibration_path=None, map_path=map_path, model=model)
    if config == "ekf_ai_map_oracle":    # full system fed the ground-truth speed (upper bound for any speed model)
        return CoreNavigationPipeline(calibration_path=None, map_path=map_path, model=OracleSpeedModel(trip))
    raise ValueError(config)


def frame_iter(trip, fixes, outage_mask, legacy_axes=False):
    acc, gyro, t = trip["acc"], trip["gyro"], trip["t"]
    for i in range(len(t)):
        g = gyro[i]
        if legacy_axes:   # the legacy parser read "Yaw" as gz and "Pitch" as gx
            g = (g[2], g[1], g[0])
        yield i, {"timestamp": float(t[i] - t[0]), "ax": float(acc[i, 0]), "ay": float(acc[i, 1]),
                  "az": float(acc[i, 2]), "gx": float(g[0]), "gy": float(g[1]), "gz": float(g[2]),
                  "gnss": None if (fixes[i] is None or outage_mask[i]) else fixes[i]}


def replay(trip, engine, fixes, outages, legacy_axes=False):
    n = len(trip["t"])
    mask = np.zeros(n, dtype=bool)
    for s, e in outages:
        mask[s:e] = True
    est = np.full((n, 2), np.nan)
    modes = np.empty(n, dtype=object)
    lat_ms = []
    oracle = getattr(engine, "model", None) if isinstance(getattr(engine, "model", None), OracleSpeedModel) else None
    for i, frame in frame_iter(trip, fixes, mask, legacy_axes):
        if oracle is not None:
            oracle.i = i
        t0 = time.perf_counter()
        out = engine.process_frame(frame)
        lat_ms.append((time.perf_counter() - t0) * 1000)
        if out.get("latitude") is not None:
            est[i] = trip["ltp"].to_enu(out["latitude"], out["longitude"])
        modes[i] = out["mode"]
    return est, modes, np.asarray(lat_ms)


def score_outages(trip, est, modes, outages):
    gt = np.stack([trip["east"], trip["north"]], axis=1)
    rows = []
    for s, e in outages:
        last = e - 1
        dist = float(np.sum(trip["speed"][s:e]) * DT)
        err_end = float(np.hypot(*(est[last] - gt[last])))
        err_start = float(np.hypot(*(est[s - 1] - gt[s - 1])))
        err_max = float(np.nanmax(np.hypot(*(est[s:e] - gt[s:e]).T)))
        dr_frac = float(np.mean([m in ("DEAD_RECKONING", "REACQUISITION") for m in modes[s:e]]))
        # time from GNSS loss until the engine reported dead reckoning
        switch = next((k for k in range(s, e) if modes[k] in ("DEAD_RECKONING", "REACQUISITION")), None)
        rows.append({"start": s, "end": e, "duration_s": (e - s) * DT, "distance_m": dist,
                     "mean_speed_mps": dist / ((e - s) * DT), "valid": bool(trip["valid"][s:e].all()),
                     "err_start_m": err_start, "err_end_m": err_end, "err_max_m": err_max,
                     "drift_pct": 100 * err_end / dist if dist > 0 else float("nan"),
                     "dr_fraction": dr_frac,
                     "switch_latency_s": None if switch is None else (switch - s) * DT})
    return rows


def summarize(rows):
    rows = [r for r in rows if r["valid"] and r["mean_speed_mps"] >= MIN_OUTAGE_SPEED]
    if not rows:
        return {"n": 0}
    d = np.array([r["drift_pct"] for r in rows])
    e = np.array([r["err_end_m"] for r in rows])
    sw = [r["switch_latency_s"] for r in rows if r["switch_latency_s"] is not None]
    return {"n": len(rows), "drift_pct_median": float(np.median(d)), "drift_pct_mean": float(np.mean(d)),
            "drift_pct_p90": float(np.percentile(d, 90)), "pass_rate_10pct": float(np.mean(d < 10.0)),
            "err_end_m_median": float(np.median(e)), "err_end_m_p90": float(np.percentile(e, 90)),
            "distance_m_median": float(np.median([r["distance_m"] for r in rows])),
            "switch_latency_s_median": float(np.median(sw)) if sw else None}


def load_bench_trip(name):
    trip = load_trip(name)
    trip["ltp"] = LocalTangentPlane(*trip["origin"])
    return trip


# ------------------------------------------------------------------------------------------ plots
def plot_outages(trip, results, outages, trip_name, scenario, out_dir, max_plots=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    gt = np.stack([trip["east"], trip["north"]], axis=1)
    ref_cfg = next(iter(results))
    rows = [r for r in results[ref_cfg]["rows"] if r["valid"] and r["mean_speed_mps"] >= MIN_OUTAGE_SPEED]
    rows = sorted(rows, key=lambda r: -r["distance_m"])[:max_plots]
    colors = {"legacy": "#b0413e", "ekf_only": "#9a9a9a", "ekf_ai": "#e59c24", "ekf_ai_map": "#2a6fdb"}
    for r in rows:
        s, e = r["start"], r["end"]
        pad = int(15 / DT)
        fig, ax = plt.subplots(figsize=(6, 6))
        seg = slice(max(0, s - pad), min(len(gt), e + pad))
        ax.plot(gt[seg, 0], gt[seg, 1], color="#222", lw=2.5, label="ground truth")
        for cfg, res in results.items():
            est = res["est"]
            row = next(x for x in res["rows"] if x["start"] == s)
            ax.plot(est[s:e, 0], est[s:e, 1], color=colors.get(cfg, None), lw=1.6,
                    label=f"{cfg}: {row['err_end_m']:.1f} m ({row['drift_pct']:.1f}%)")
        ax.plot(*gt[s], "go", ms=8, label="GNSS lost")
        ax.plot(*gt[e - 1], "rs", ms=8, label="GNSS back")
        if trip.get("roads") is not None:
            pts = np.concatenate([gt[seg], *[res["est"][s:e] for res in results.values()]])
            lo, hi = np.nanmin(pts, axis=0) - 60, np.nanmax(pts, axis=0) + 60
            for (a, b) in trip["roads"]:
                if np.all(a >= lo) and np.all(a <= hi) or np.all(b >= lo) and np.all(b <= hi):
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="#ccc", lw=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
        ax.set_title(f"{trip_name} - {scenario}: {r['duration_s']:.0f} s / {r['distance_m']:.0f} m outage")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{trip_name}_{scenario}_{s}.png"), dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------------------------------ main
def scenarios_for(trip):
    sc = {f"outage_{d}s": plan_outages(trip, d) for d in (30, 60, 120)}
    sc["ps_50m_under_1min"] = plan_distance_outages(trip, 50.0, 0.8, 100.0, 60.0)
    sc["ps_1km_highway"] = plan_distance_outages(trip, 1000.0, 13.0, 40.0, 90.0)
    return sc


# The legacy engine hard-codes its local origin (Guwahati) with a flat-earth lat/lon conversion. As in its
# own replay.py, trips are relocated to start at that origin; otherwise it starts ~9000 km away and never
# converges, which would unfairly penalise it.
LEGACY_REF_LAT, LEGACY_REF_LON = 26.1445, 91.7362
_M_PER_DEG = 111320.0


def _to_legacy_latlon(east, north):
    return (LEGACY_REF_LAT + north / _M_PER_DEG,
            LEGACY_REF_LON + east / (_M_PER_DEG * math.cos(math.radians(LEGACY_REF_LAT))))


def _from_legacy_latlon(lat, lon):
    return ((lon - LEGACY_REF_LON) * _M_PER_DEG * math.cos(math.radians(LEGACY_REF_LAT)),
            (lat - LEGACY_REF_LAT) * _M_PER_DEG)


def run_legacy(legacy_root, trip, trip_name, fixes, outages):
    """Runs the legacy engine in a subprocess (its `src` package clashes with the new code)."""
    import pickle
    import tempfile
    helper = os.path.join(ROOT, "src/evaluation/_legacy_runner.py")
    n = len(trip["t"])
    mask = np.zeros(n, dtype=bool)
    for s, e in outages:
        mask[s:e] = True
    frames, last_course = [], 0.0
    for _, f in frame_iter(trip, fixes, mask, legacy_axes=True):
        if f["gnss"] is not None:          # the legacy engine requires a course on every fix
            fe, fn = trip["ltp"].to_enu(f["gnss"]["lat"], f["gnss"]["lon"])
            lat, lon = _to_legacy_latlon(fe, fn)
            f["gnss"] = {**f["gnss"], "lat": lat, "lon": lon}
            if f["gnss"]["heading"] is None:
                f["gnss"] = {**f["gnss"], "heading": last_course}
            last_course = f["gnss"]["heading"]
        frames.append(f)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "frames.pkl"), os.path.join(tmp, "out.pkl")
        with open(inp, "wb") as f:
            pickle.dump(frames, f)
        subprocess.run([sys.executable, helper, legacy_root, inp, out], check=True)
        with open(out, "rb") as f:
            outs, lat_ms = pickle.load(f)
    est = np.full((n, 2), np.nan)
    modes = np.empty(n, dtype=object)
    for i, o in enumerate(outs):
        est[i] = _from_legacy_latlon(o["latitude"], o["longitude"])
        modes[i] = o["mode"]
    return est, modes, np.asarray(lat_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", default=["S1"])
    ap.add_argument("--configs", nargs="*", default=["ekf_only", "ekf_ai", "ekf_ai_map"])
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--legacy-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "benchmark.json"))
    ap.add_argument("--rate-test", action="store_true")
    ap.add_argument("--gnss-source", choices=["synthetic", "phone"], default="synthetic")
    ap.add_argument("--gnss-latency", type=float, default=0.0, help="seconds (synthetic source only)")
    a = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if a.rate_test:
        return rate_test()
    configs = (["legacy"] if a.legacy_root else []) + a.configs
    gnss_desc = ("IO-VNBD phone GPS (~0.11 Hz, real accuracy)" if a.gnss_source == "phone" else
                 f"synthetic 1 Hz, GT + GM(2 m, 30 s) + white 1 m, latency {a.gnss_latency:g} s")
    report = {"configs": configs, "seed": a.seed, "gnss_source": a.gnss_source, "gnss_latency_s": a.gnss_latency,
              "trips": {}, "gnss": gnss_desc}
    for trip_name in a.trips:
        trip = load_bench_trip(trip_name)
        map_path = os.path.join("data/maps", f"{trip_name}.geojson")
        if os.path.exists(map_path):
            from src.navigation.map_matching import RoadNetwork
            net = RoadNetwork.from_geojson(map_path, trip["ltp"])
            trip["roads"] = list(zip(net.p1, net.p2))
        fixes = phone_gnss(trip) if a.gnss_source == "phone" else \
            synthetic_gnss(trip, seed=a.seed, latency_s=a.gnss_latency)
        report["trips"][trip_name] = {}
        for sc_name, outages in scenarios_for(trip).items():
            if a.scenarios and sc_name not in a.scenarios:
                continue
            results = {}
            for cfg in configs:
                t0 = time.time()
                if cfg == "legacy":
                    est, modes, lat_ms = run_legacy(a.legacy_root, trip, trip_name, fixes, outages)
                else:
                    est, modes, lat_ms = replay(trip, make_engine(cfg, trip_name, trip=trip), fixes, outages)
                rows = score_outages(trip, est, modes, outages)
                summ = summarize(rows)
                summ["frame_latency_ms_mean"] = float(np.mean(lat_ms))
                summ["frame_latency_ms_p99"] = float(np.percentile(lat_ms, 99))
                results[cfg] = {"est": est, "rows": rows, "summary": summ}
                print(f"[{trip_name} | {sc_name:18s} | {cfg:10s}] n={summ.get('n', 0):3d} "
                      f"drift median {summ.get('drift_pct_median', float('nan')):5.1f}% "
                      f"p90 {summ.get('drift_pct_p90', float('nan')):5.1f}% "
                      f"pass<10% {100 * summ.get('pass_rate_10pct', float('nan')):5.1f}% "
                      f"| end err median {summ.get('err_end_m_median', float('nan')):6.1f} m "
                      f"| {summ['frame_latency_ms_mean']:.2f} ms/frame | {time.time() - t0:.0f}s", flush=True)
            report["trips"][trip_name][sc_name] = {cfg: {"summary": r["summary"], "outages": r["rows"]}
                                                   for cfg, r in results.items()}
            if a.plots and outages:
                plot_outages(trip, results, outages, trip_name, sc_name, os.path.join(RESULTS_DIR, "plots"),
                             a.plots)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1)
    print(f"Saved {a.out}")


def rate_test(seconds=300):
    """
    Engine throughput at the phone rate (10 Hz) and an edge IMU rate (200 Hz), with and without map
    matching. The 200 Hz stream is S1 upsampled by linear interpolation, so this measures compute load
    only: it says nothing about accuracy with genuinely high-rate sensor data.
    """
    from src.navigation.pipeline import CoreNavigationPipeline
    trip = load_bench_trip("S1")
    fixes = synthetic_gnss(trip)
    n = int(seconds / DT)
    s0 = int(600 / DT)
    res = {}
    runs = [(rate, use_map) for use_map in (False, True) for rate in (10, 200)]
    for rate, use_map in runs:
        from src.models.inference import SpeedModel
        eng = CoreNavigationPipeline(calibration_path=None, use_map=use_map, map_path="data/maps/S1.geojson",
                                     model=SpeedModel(MODEL_PATH))
        key = f"{rate}Hz_{'full_system' if use_map else 'no_map'}"
        t_src = trip["t"][s0:s0 + n] - trip["t"][s0]
        t_new = np.arange(0, t_src[-1], 1.0 / rate)
        acc = np.stack([np.interp(t_new, t_src, trip["acc"][s0:s0 + n, k]) for k in range(3)], 1)
        gyr = np.stack([np.interp(t_new, t_src, trip["gyro"][s0:s0 + n, k]) for k in range(3)], 1)
        fix_idx = {round(i * DT * rate): fixes[s0 + i] for i in range(n) if fixes[s0 + i] is not None}
        lat = []
        for k in range(len(t_new)):
            f = {"timestamp": float(t_new[k]), "ax": acc[k, 0], "ay": acc[k, 1], "az": acc[k, 2],
                 "gx": gyr[k, 0], "gy": gyr[k, 1], "gz": gyr[k, 2], "gnss": fix_idx.get(k)}
            t0 = time.perf_counter()
            eng.process_frame(f)
            lat.append(time.perf_counter() - t0)
        lat = np.asarray(lat) * 1000
        res[key] = {"frames": len(lat), "latency_ms_mean": float(lat.mean()),
                    "latency_ms_p99": float(np.percentile(lat, 99)),
                    "max_sustainable_rate_hz": float(1000 / lat.mean()),
                    "realtime_factor": float((1000 / rate) / lat.mean())}
        print(f"{key:18s}: {lat.mean():.3f} ms/frame (p99 {np.percentile(lat, 99):.2f} ms) -> "
              f"{1000 / lat.mean():.0f} frames/s sustainable, {res[key]['realtime_factor']:.1f}x real time")
    with open(os.path.join(RESULTS_DIR, "rate_test.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
