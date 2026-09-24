"""
Evaluates the trained speed model on the held-out test trip (S1) and the out-of-distribution
Driver E trips, and writes results/model_eval.json plus data/processed/speed_prediction_benchmark.png.

Metrics
  window MAE / RMSE / bias (m/s), MAE by speed band, stationary-detection F1,
  uncertainty calibration (std of (pred - true) / sigma, ideal 1.0),
  60 s simulated-outage along-track distance error (% of distance) with and without the
  GNSS speed reference, vs. the "hold last GNSS speed" baseline.
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.tiny_tcn import CHECKPOINT, load_model, outage_distance_errors, predict
from src.preprocessing.create_windows import PROCESSED_DIR, SPLITS, STATIONARY_SPEED, load_normalization, \
    load_sequences_npz

RESULTS = "results/model_eval.json"


def window_metrics(sp, var, pst, y):
    err = sp - y
    stat_true, stat_pred = y < STATIONARY_SPEED, pst > 0.5
    tp = np.sum(stat_true & stat_pred)
    f1 = 2 * tp / max(np.sum(stat_true) + np.sum(stat_pred), 1)
    bands = {}
    for lo, hi in [(0, 0.3), (0.3, 5), (5, 10), (10, 20), (20, 40)]:
        m = (y >= lo) & (y < hi)
        if m.any():
            bands[f"{lo}-{hi} m/s"] = round(float(np.mean(np.abs(err[m]))), 3)
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)), "mae_by_speed": bands, "stationary_f1": float(f1),
            "z_std": float(np.std(err / np.sqrt(var))),
            "predict_mean_mae": float(np.mean(np.abs(y - y.mean())))}


def evaluate(weights=CHECKPOINT, plot=True):
    norm = load_normalization()
    model = load_model(weights)
    seqs = load_sequences_npz()
    window = norm["window_size"]
    report = {"weights": weights}
    for split in ("test", "ood"):
        d = np.load(os.path.join(PROCESSED_DIR, f"{split}_windows.npz"))
        X_raw = d["X"] * norm["std"] + norm["mean"]
        sp, lv, stl = predict(model, X_raw, d["S"], norm)
        wm = window_metrics(sp, np.exp(lv), 1 / (1 + np.exp(-stl)), d["y"])
        no_ref = d["S"][:, 2] == 0
        wm["mae_without_gnss_ref"] = float(np.mean(np.abs(sp[no_ref] - d["y"][no_ref])))
        wm["mae_with_gnss_ref"] = float(np.mean(np.abs(sp[~no_ref] - d["y"][~no_ref])))
        trips = SPLITS[split]
        if split == "ood":   # Driver E: evaluate every sample, the phone was loosely mounted
            for t in trips:
                seqs[t]["valid"] = np.ones_like(seqs[t]["valid"])
        e_ref, hold = outage_distance_errors(model, seqs, trips, norm, window)
        e_noref, _ = outage_distance_errors(model, seqs, trips, norm, window, ref_valid=False)
        wm["outage60"] = {
            "n": int(len(e_ref)),
            "model_with_ref_median_pct": float(np.median(e_ref)), "model_with_ref_mean_pct": float(np.mean(e_ref)),
            "model_without_ref_median_pct": float(np.median(e_noref)),
            "hold_last_speed_median_pct": float(np.median(hold)), "hold_last_speed_mean_pct": float(np.mean(hold)),
        }
        report[split] = wm
        o = wm["outage60"]
        print(f"[{split}] {trips}: MAE {wm['mae']:.2f} m/s (with ref {wm['mae_with_gnss_ref']:.2f}, IMU-only "
              f"{wm['mae_without_gnss_ref']:.2f}; predict-mean {wm['predict_mean_mae']:.2f}) | RMSE {wm['rmse']:.2f} | "
              f"bias {wm['bias']:+.2f} | stationary F1 {wm['stationary_f1']:.2f} | z-std {wm['z_std']:.2f}")
        print(f"        60 s outage along-track error: model {o['model_with_ref_median_pct']:.1f}% median "
              f"({o['model_with_ref_mean_pct']:.1f}% mean), IMU-only {o['model_without_ref_median_pct']:.1f}%, "
              f"hold-last-speed {o['hold_last_speed_median_pct']:.1f}%  (n={o['n']})")
        print(f"        MAE by speed: {wm['mae_by_speed']}")
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(report, f, indent=1)
    if plot:
        plot_trace(model, seqs, norm)
    return report


def plot_trace(model, seqs, norm, trip="S1", start_s=1200, seconds=600):
    """GT vs predicted speed (IMU-only and GNSS-referenced over repeated 60 s outages) with 2-sigma band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = norm["window_size"]
    sq = seqs[trip]
    idx = np.arange(start_s * 10, (start_s + seconds) * 10)
    X = np.stack([sq["feats"][i - w + 1:i + 1] for i in idx])
    S0 = np.zeros((len(idx), 3), dtype=np.float32)
    S1 = np.array([[sq["speed"][i - (i - idx[0]) % 600], ((i - idx[0]) % 600) / 10.0, 1.0] for i in idx],
                  dtype=np.float32)
    sp0, lv0, _ = predict(model, X, S0 / norm["scalar_scale"], norm)
    sp1, lv1, _ = predict(model, X, S1 / norm["scalar_scale"], norm)
    t = (idx - idx[0]) / 10.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, sq["speed"][idx] * 3.6, color="#222", lw=1.8, label="ground truth (CAN/GNSS reference)")
    ax.plot(t, sp1 * 3.6, color="#2a6fdb", lw=1.2, label="TCN, GNSS reference reset every 60 s")
    s1 = np.sqrt(np.exp(lv1)) * 3.6
    ax.fill_between(t, np.maximum(sp1 * 3.6 - 2 * s1, 0), sp1 * 3.6 + 2 * s1, color="#2a6fdb", alpha=0.12, label="±2σ")
    ax.plot(t, sp0 * 3.6, color="#e59c24", lw=1.0, alpha=0.9, label="TCN, IMU only (no GNSS ever)")
    for k in range(0, seconds, 60):
        ax.axvline(k, color="#bbb", lw=0.6, zorder=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (km/h)")
    ax.set_title(f"Forward speed on unseen trip {trip} (vertical lines: simulated outage starts)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(PROCESSED_DIR, "speed_prediction_benchmark.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    evaluate(sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT)
