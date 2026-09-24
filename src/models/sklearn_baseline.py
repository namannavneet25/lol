"""
Classical ML baselines for the speed model (plan.md section 37, experiment E0):
Ridge regression and histogram gradient boosting on hand-crafted window statistics
(mean, std, mean |diff|, 5/95th percentiles and 4 spectral band energies per channel)
plus the same GNSS-reference scalars the TCN receives.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.tiny_tcn import outage_distance_errors
from src.preprocessing.create_windows import PROCESSED_DIR, SPLITS, load_normalization, load_sequences_npz


def window_stats(X):
    """X: (N, T, C) raw features -> (N, 9*C) statistics."""
    d = np.abs(np.diff(X, axis=1)).mean(1)
    spec = np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), axis=1))[:, 1:]
    bands = [b.mean(1) for b in np.array_split(spec, 4, axis=1)]
    return np.concatenate([X.mean(1), X.std(1), d, np.percentile(X, 5, axis=1), np.percentile(X, 95, axis=1)]
                          + bands, axis=1)


def load_split(split, norm):
    d = np.load(os.path.join(PROCESSED_DIR, f"{split}_windows.npz"))
    X_raw = d["X"] * norm["std"] + norm["mean"]
    return np.concatenate([window_stats(X_raw), d["S"]], axis=1), d["y"]


def train_sklearn_baselines():
    norm = load_normalization()
    seqs = load_sequences_npz()
    Xtr, ytr = load_split("train", norm)
    Xte, yte = load_split("test", norm)
    print(f"Train {Xtr.shape}, test {Xte.shape}")
    results = {}
    for name, model in [("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                        ("hist_gbr", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0))]:
        t0 = time.time()
        model.fit(Xtr, ytr)
        p = np.maximum(model.predict(Xte), 0)
        fn = lambda Xr, S, m=model: np.maximum(m.predict(np.concatenate([window_stats(Xr), S], 1)), 0)
        errs, hold = outage_distance_errors(None, seqs, SPLITS["test"], norm, norm["window_size"], predict_fn=fn)
        results[name] = {"test_mae": float(np.mean(np.abs(p - yte))), "test_rmse": float(np.sqrt(np.mean((p - yte) ** 2))),
                         "outage60_median_pct": float(np.median(errs)), "outage60_mean_pct": float(np.mean(errs)),
                         "hold_last_speed_median_pct": float(np.median(hold)), "fit_s": time.time() - t0}
        r = results[name]
        print(f"{name:9s} test MAE {r['test_mae']:.2f} m/s | RMSE {r['test_rmse']:.2f} | 60 s outage error "
              f"median {r['outage60_median_pct']:.1f}% mean {r['outage60_mean_pct']:.1f}% "
              f"(hold-speed {r['hold_last_speed_median_pct']:.1f}%)")
    os.makedirs("results", exist_ok=True)
    with open("results/baselines_sklearn.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    train_sklearn_baselines()
