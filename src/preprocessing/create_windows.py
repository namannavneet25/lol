"""
Builds the model dataset from the aligned trips in data/intermediate/.

Outputs (data/processed/):
  normalization.json   - feature mean/std (TRAIN trips only) + model I/O contract
  sequences.npz        - per-trip raw feature sequences, speed and validity, used by the
                         trainer to sample windows and GNSS-reference scalars on the fly
  {train,val,test,ood}_windows.npz - fixed, seeded windows (normalized) for evaluation/baselines:
                         X (N, T, C), S (N, 3) scalars, y (N,) speed, trip (N,) trip index

Splits are by trip (never by window), per plan.md section 36:
  train: S2, S4, M     (Drivers A, B)
  val:   Y1            (Driver D)
  test:  S1            (Driver A, unseen trip)
  ood:   Vw02, Vw03, Vta01b, Vta06, Vtb01  (Driver E, different vehicle, loosely mounted phone)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.features import (CHANNEL_NAMES, MODEL_RATE_HZ, SCALAR_NAMES, T_SINCE_REF_MAX,
                                        sequence_features)
from src.preprocessing.parse_trip import OUT_DIR as INTERMEDIATE_DIR, load_trip

SPLITS = {
    "train": ["S2", "S4", "M"],
    "val": ["Y1"],
    "test": ["S1"],
    "ood": ["Vw02", "Vw03", "Vta01b", "Vta06", "Vtb01"],
}
WINDOW = 40                      # 4 s at 10 Hz
STATIONARY_SPEED = 0.3           # m/s
SCALAR_SCALE = [20.0, 60.0, 1.0] # v_ref, t_since_ref, ref_valid
PROCESSED_DIR = "data/processed"


def run_starts(valid):
    """For every index, the first index of its contiguous valid run (or -1 if invalid)."""
    starts = np.full(len(valid), -1, dtype=np.int64)
    start = -1
    for i, ok in enumerate(valid):
        if ok:
            start = i if start < 0 else start
            starts[i] = start
        else:
            start = -1
    return starts


def sample_reference(rng, end_idx, run_start, speed, p_valid=0.75):
    """
    Draws the GNSS-reference conditioning for a window ending at end_idx (exclusive):
    the speed at the last trusted GNSS fix and the time since it. 20% of references
    are recent (<2 s, GNSS-aided operation), the rest spread over a 120 s outage.
    """
    if rng.random() > p_valid:
        return 0.0, 0.0, 0.0
    t_since = rng.uniform(0, 2.0) if rng.random() < 0.2 else rng.uniform(0, T_SINCE_REF_MAX)
    ref = max(end_idx - 1 - int(round(t_since * MODEL_RATE_HZ)), run_start)
    t_since = (end_idx - 1 - ref) / MODEL_RATE_HZ
    v_ref = max(0.0, float(speed[ref]) + rng.normal(0, 0.1))
    return v_ref, t_since, 1.0


def load_sequences(trips):
    seqs = {}
    for name in trips:
        d = load_trip(name, INTERMEDIATE_DIR)
        seqs[name] = {"feats": sequence_features(d["acc"], d["gyro"]), "speed": d["speed"].astype(np.float32),
                      "valid": d["valid"].astype(bool)}
    return seqs


def fixed_windows(seqs, trips, mean, std, window, step, seed, ignore_valid=False):
    rng = np.random.default_rng(seed)
    X, S, y, trip_idx = [], [], [], []
    for ti, name in enumerate(trips):
        sq = seqs[name]
        valid = np.ones_like(sq["valid"]) if ignore_valid else sq["valid"]
        starts = run_starts(valid)
        norm = (sq["feats"] - mean) / std
        for end in range(window, len(norm) + 1, step):
            if starts[end - 1] < 0 or starts[end - 1] > end - window:
                continue
            X.append(norm[end - window:end])
            S.append(sample_reference(rng, end, starts[end - 1], sq["speed"]))
            y.append(sq["speed"][end - 1])
            trip_idx.append(ti)
    S = np.asarray(S, dtype=np.float32) / np.asarray(SCALAR_SCALE, dtype=np.float32)
    return (np.asarray(X, dtype=np.float32), S, np.asarray(y, dtype=np.float32), np.asarray(trip_idx))


def build_processed_dataset(window=WINDOW, output_dir=PROCESSED_DIR):
    os.makedirs(output_dir, exist_ok=True)
    all_trips = [t for split in SPLITS.values() for t in split]
    seqs = load_sequences(all_trips)

    train_feats = np.concatenate([seqs[t]["feats"][seqs[t]["valid"]] for t in SPLITS["train"]])
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std[std < 1e-6] = 1e-6

    norm_params = {
        "version": 2,
        "channel_names": CHANNEL_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_size": window,
        "sampling_rate_hz": MODEL_RATE_HZ,
        "scalar_names": SCALAR_NAMES,
        "scalar_scale": SCALAR_SCALE,
        "t_since_ref_max_s": T_SINCE_REF_MAX,
        "stationary_speed_mps": STATIONARY_SPEED,
        "splits": SPLITS,
        "frame_convention": "gravity-levelled (z along gravity); horizontal axes up to mount azimuth",
    }
    with open(os.path.join(output_dir, "normalization.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    flat = {}
    for name, sq in seqs.items():
        for k, v in sq.items():
            flat[f"{name}/{k}"] = v
    np.savez_compressed(os.path.join(output_dir, "sequences.npz"), **flat)

    for i, (split, trips) in enumerate(SPLITS.items()):
        X, S, y, trip_idx = fixed_windows(seqs, trips, mean, std, window, step=5, seed=100 + i,
                                          ignore_valid=(split == "ood"))
        np.savez_compressed(os.path.join(output_dir, f"{split}_windows.npz"), X=X, S=S, y=y, trip=trip_idx,
                            trip_names=np.array(trips))
        print(f"{split:5s}: {len(y):6d} windows from {trips} | mean speed {y.mean():.1f} m/s | "
              f"stationary {100 * np.mean(y < STATIONARY_SPEED):.0f}%")
    print(f"Saved normalization.json, sequences.npz and split windows to {output_dir}/")


def load_normalization(path=os.path.join(PROCESSED_DIR, "normalization.json")):
    with open(path) as f:
        p = json.load(f)
    p["mean"] = np.asarray(p["mean"], dtype=np.float32)
    p["std"] = np.asarray(p["std"], dtype=np.float32)
    p["scalar_scale"] = np.asarray(p["scalar_scale"], dtype=np.float32)
    return p


def load_sequences_npz(path=os.path.join(PROCESSED_DIR, "sequences.npz")):
    seqs = {}
    with np.load(path) as d:
        for key in d.files:
            trip, field = key.split("/")
            seqs.setdefault(trip, {})[field] = d[key]
    return seqs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=WINDOW, help="window length in 10 Hz frames")
    build_processed_dataset(window=ap.parse_args().window)
