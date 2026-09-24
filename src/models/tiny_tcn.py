"""
Tiny causal TCN speed estimator (the "AI Speed & Vibration Filter").

Inputs
  imu_window  (B, C=9, T)  normalized levelled IMU features at 10 Hz (see preprocessing/features.py)
  ref_scalars (B, 3)       [v_ref, t_since_ref, ref_valid] / scalar_scale - the speed at the last
                           trusted GNSS fix and the time since it; ref_valid=0 means "no GNSS
                           reference", and the model then estimates speed from the IMU alone.
Outputs (AIOutput)
  speed       (B,)  forward speed v_f in m/s (softplus, never negative)
  log_var     (B,)  log sigma_v^2 used as the EKF measurement variance
  stat_logit  (B,)  logit of P(vehicle stationary), drives ZUPT in the EKF

Training loss = Huber + beta-NLL (uncertainty) + BCE (stationary) + rollout loss.
The rollout loss compares predicted vs true mean speed over contiguous 60 s simulated
outages, i.e. it directly optimises the along-track dead-reckoning drift.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.create_windows import (PROCESSED_DIR, SPLITS, STATIONARY_SPEED, load_normalization,
                                              load_sequences_npz, run_starts, sample_reference)
from src.preprocessing.features import G, MODEL_RATE_HZ, NUM_CHANNELS

LOG_VAR_MIN, LOG_VAR_MAX = -6.0, 5.0
CHECKPOINT = "models/tiny_tcn_best.pth"


class CausalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.GroupNorm(4, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(F.pad(x, (self.pad, 0)))
        return x + self.drop(F.gelu(self.norm(h)))


class TinyTCN(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, num_scalars=3, hidden=32, dilations=(1, 2, 4, 8, 16),
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.Sequential(*[CausalBlock(hidden, kernel_size, d, dropout) for d in dilations])
        self.scalar_net = nn.Sequential(nn.Linear(num_scalars, 16), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(2 * hidden + 16, 64), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(64, 64), nn.GELU())
        self.speed_head = nn.Linear(64, 1)
        self.uncertainty_head = nn.Linear(64, 1)
        self.stationary_head = nn.Linear(64, 1)
        nn.init.constant_(self.speed_head.bias, 5.0)

    def forward(self, x, scalars):
        h = self.blocks(self.inp(x))
        pooled = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)
        feat = self.trunk(torch.cat([pooled, self.scalar_net(scalars)], dim=1))
        speed = F.softplus(self.speed_head(feat)).squeeze(-1)
        raw = self.uncertainty_head(feat).squeeze(-1)
        log_var = LOG_VAR_MIN + (LOG_VAR_MAX - LOG_VAR_MIN) * torch.sigmoid(raw)
        stat_logit = self.stationary_head(feat).squeeze(-1)
        return speed, log_var, stat_logit


# ----------------------------------------------------------------------------------------------
# Augmentation on raw (un-normalized) features, keeping derived channels physically consistent
# ----------------------------------------------------------------------------------------------
def augment_raw(x, gen):
    """
    x: (B, T, 9) raw features. Random rotation about the vertical (mount azimuth), sensor bias,
    scale and noise are applied to the base channels; magnitude channels are then recomputed.
    """
    B = x.shape[0]
    theta = torch.rand(B, 1, generator=gen) * 2 * math.pi
    c, s = torch.cos(theta), torch.sin(theta)
    a1, a2, av = x[..., 0], x[..., 1], x[..., 2]
    w1, w2, wy = x[..., 3], x[..., 4], x[..., 5]
    a1, a2 = c * a1 - s * a2, s * a1 + c * a2
    w1, w2 = c * w1 - s * w2, s * w1 + c * w2

    def jitter(v, bias, scale, noise):
        b = (torch.rand(B, 1, generator=gen) * 2 - 1) * bias
        k = 1 + (torch.rand(B, 1, generator=gen) * 2 - 1) * scale
        return v * k + b + torch.randn(v.shape, generator=gen) * noise

    a1, a2, av = (jitter(a1, 0.15, 0.08, 0.05), jitter(a2, 0.15, 0.08, 0.05), jitter(av, 0.1, 0.08, 0.05))
    w1, w2, wy = (jitter(w1, 0.01, 0.05, 0.005), jitter(w2, 0.01, 0.05, 0.005), jitter(wy, 0.01, 0.05, 0.005))
    a_mag = torch.sqrt(a1 ** 2 + a2 ** 2 + (av + G) ** 2) - G
    a_h = torch.sqrt(a1 ** 2 + a2 ** 2)
    w_mag = torch.sqrt(w1 ** 2 + w2 ** 2 + wy ** 2)
    out = torch.stack([a1, a2, av, w1, w2, wy, a_mag, a_h, w_mag], dim=-1)
    # sample dropout: repeat the previous frame for ~2% of samples (missed sensor events)
    drop = torch.rand(out.shape[:2], generator=gen) < 0.02
    drop[:, 0] = False
    idx = torch.arange(out.shape[1]).expand(B, -1).clone()
    idx[drop] -= 1
    return torch.gather(out, 1, idx.unsqueeze(-1).expand_as(out))


# ----------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------
class SpeedLoss(nn.Module):
    def __init__(self, lambda_nll=0.2, lambda_stat=0.3, beta=0.5):
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_stat = lambda_stat
        self.beta = beta

    def forward(self, speed, log_var, stat_logit, target):
        huber = F.huber_loss(speed, target, delta=1.0)
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + (speed.detach() - target) ** 2 / var)
        nll = (nll * var.detach() ** self.beta).mean()
        stat = F.binary_cross_entropy_with_logits(stat_logit, (target < STATIONARY_SPEED).float())
        return huber + self.lambda_nll * nll + self.lambda_stat * stat, {
            "huber": huber.item(), "nll": nll.item(), "stat": stat.item()}


# ----------------------------------------------------------------------------------------------
# Data sampling
# ----------------------------------------------------------------------------------------------
class TripSampler:
    """Samples training windows / rollouts from per-trip feature sequences with GNSS-reference scalars."""

    def __init__(self, seqs, trips, norm, window, seed=0, highspeed_weight=1.0, highspeed_mps=15.0):
        self.seqs = {t: seqs[t] for t in trips}
        self.window = window
        self.norm = norm
        self.rng = np.random.default_rng(seed)
        self.starts = {t: run_starts(self.seqs[t]["valid"]) for t in trips}
        self.ends = {}
        for t in trips:
            st = self.starts[t]
            ends = np.arange(window, len(st) + 1)
            ok = (st[ends - 1] >= 0) & (st[ends - 1] <= ends - window)
            self.ends[t] = ends[ok]
        self.trips = list(trips)
        # Highway windows are rare in the (mostly urban) training trips; optionally oversample them.
        self.end_p, sizes = {}, []
        for t in self.trips:
            w = np.where(self.seqs[t]["speed"][self.ends[t] - 1] > highspeed_mps, highspeed_weight, 1.0)
            self.end_p[t] = w / w.sum()
            sizes.append(w.sum())
        self.trip_p = np.asarray(sizes) / np.sum(sizes)

    def windows(self, n):
        X = np.empty((n, self.window, NUM_CHANNELS), dtype=np.float32)
        S = np.empty((n, 3), dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        trip_choice = self.rng.choice(len(self.trips), size=n, p=self.trip_p)
        ends = np.empty(n, dtype=np.int64)
        for ti, t in enumerate(self.trips):          # one vectorised draw per trip
            sel = trip_choice == ti
            ends[sel] = self.rng.choice(self.ends[t], size=int(sel.sum()), p=self.end_p[t])
        for i, ti in enumerate(trip_choice):
            t = self.trips[ti]
            end = int(ends[i])
            sq = self.seqs[t]
            X[i] = sq["feats"][end - self.window:end]
            S[i] = sample_reference(self.rng, end, self.starts[t][end - 1], sq["speed"])
            y[i] = sq["speed"][end - 1]
        return X, S / self.norm["scalar_scale"], y

    def rollouts(self, n, seconds=60):
        """n simulated outages: windows every 1 s for `seconds`, all sharing the outage-start reference."""
        steps = int(seconds)
        span = self.window + steps * int(MODEL_RATE_HZ)
        X, S, y = [], [], []
        for _ in range(n):
            ti = self.rng.choice(len(self.trips), p=self.trip_p)
            t = self.trips[ti]
            sq, st = self.seqs[t], self.starts[t]
            for _attempt in range(50):
                end0 = int(self.rng.choice(self.ends[t], p=self.end_p[t]))
                last = end0 + (steps - 1) * int(MODEL_RATE_HZ)
                if last <= len(st) and st[last - 1] >= 0 and st[last - 1] <= end0 - self.window:
                    break
            ref_valid = self.rng.random() < 0.85
            v0 = max(0.0, float(sq["speed"][end0 - 1]) + self.rng.normal(0, 0.1))
            for k in range(steps):
                end = end0 + k * int(MODEL_RATE_HZ)
                X.append(sq["feats"][end - self.window:end])
                S.append([v0, float(k), 1.0] if ref_valid else [0.0, 0.0, 0.0])
                y.append(sq["speed"][end - 1])
        X = np.asarray(X, dtype=np.float32)
        S = np.asarray(S, dtype=np.float32) / self.norm["scalar_scale"]
        return X, S, np.asarray(y, dtype=np.float32), steps


def to_model_input(X_raw, norm):
    """(N, T, C) raw features -> (N, C, T) normalized tensor."""
    Xn = (X_raw - torch.as_tensor(norm["mean"])) / torch.as_tensor(norm["std"])
    return Xn.permute(0, 2, 1).contiguous()


# ----------------------------------------------------------------------------------------------
# Evaluation helpers (also used by evaluate_models.py)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X_raw, S, norm, batch=4096):
    model.eval()
    outs = [[], [], []]
    for i in range(0, len(X_raw), batch):
        xb = to_model_input(torch.as_tensor(X_raw[i:i + batch]), norm)
        sp, lv, st = model(xb, torch.as_tensor(S[i:i + batch]))
        for o, v in zip(outs, (sp, lv, st)):
            o.append(v.numpy())
    return tuple(np.concatenate(o) for o in outs)


@torch.no_grad()
def outage_distance_errors(model, seqs, trips, norm, window, seconds=60, spacing_s=30, ref_valid=True,
                           min_speed=3.0, predict_fn=None):
    """
    Along-track distance error (% of distance travelled) of the model's speed integrated over
    simulated GNSS outages of `seconds`, starting every `spacing_s` along each trip.
    predict_fn(X_raw, S_scaled) -> speed overrides the model (used for baselines).
    Returns (errors_pct array, hold_last_speed_errors_pct array).
    """
    errs, hold = [], []
    rate = int(MODEL_RATE_HZ)
    for t in trips:
        sq = seqs[t]
        st = run_starts(sq["valid"])
        n_out = seconds * rate
        starts = []
        for s0 in range(window, len(sq["speed"]) - n_out, spacing_s * rate):
            last = s0 + n_out
            if st[last - 1] >= 0 and st[last - 1] <= s0 - window and sq["speed"][s0:last].mean() >= min_speed:
                starts.append(s0)
        if not starts:
            continue
        ends = np.array([s0 + k + 1 for s0 in starts for k in range(n_out)])
        X = np.stack([sq["feats"][e - window:e] for e in ends])
        S = np.array([[sq["speed"][s0], (k + 1) / rate, 1.0] if ref_valid else [0, 0, 0]
                      for s0 in starts for k in range(n_out)], dtype=np.float32) / norm["scalar_scale"]
        sp = predict_fn(X, S) if predict_fn is not None else predict(model, X, S, norm)[0]
        sp = sp.reshape(len(starts), n_out)
        for i, s0 in enumerate(starts):
            true_d = sq["speed"][s0:s0 + n_out].sum() / rate
            errs.append(100 * abs(sp[i].sum() / rate - true_d) / true_d)
            hold.append(100 * abs(sq["speed"][s0] * seconds - true_d) / true_d)
    return np.asarray(errs), np.asarray(hold)


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def train_tiny_tcn(epochs=40, lr=2e-3, batch_size=256, samples_per_epoch=60000, rollout_batch=8,
                   lambda_rollout=0.5, seed=0, save_path=CHECKPOINT, data_dir=PROCESSED_DIR, highspeed_weight=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    norm = load_normalization(os.path.join(data_dir, "normalization.json"))
    window = norm["window_size"]
    seqs = load_sequences_npz(os.path.join(data_dir, "sequences.npz"))
    sampler = TripSampler(seqs, SPLITS["train"], norm, window, seed=seed, highspeed_weight=highspeed_weight)

    model = TinyTCN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"--- Tiny TCN training | {n_params} params | window {window} | epochs {epochs} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps_per_epoch = samples_per_epoch // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    criterion = SpeedLoss()

    val = np.load(os.path.join(data_dir, "val_windows.npz"))
    best, best_epoch, history = float("inf"), 0, []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        X_all, S_all, y_all = sampler.windows(steps_per_epoch * batch_size)
        tot = 0.0
        for step in range(steps_per_epoch):
            sl = slice(step * batch_size, (step + 1) * batch_size)
            xb = to_model_input(augment_raw(torch.as_tensor(X_all[sl]), gen), norm)
            sp, lv, stl = model(xb, torch.as_tensor(S_all[sl]))
            loss, _ = criterion(sp, lv, stl, torch.as_tensor(y_all[sl]))
            if lambda_rollout > 0:
                Xr, Sr, yr, n_steps = sampler.rollouts(rollout_batch)
                xr = to_model_input(augment_raw(torch.as_tensor(Xr), gen), norm)
                spr, _, _ = model(xr, torch.as_tensor(Sr))
                mean_pred = spr.view(rollout_batch, n_steps).mean(1)
                mean_true = torch.as_tensor(yr).view(rollout_batch, n_steps).mean(1)
                loss = loss + lambda_rollout * F.smooth_l1_loss(mean_pred, mean_true)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()

        sp, lv, _ = predict(model, val["X"] * norm["std"] + norm["mean"], val["S"], norm)
        mae = float(np.mean(np.abs(sp - val["y"])))
        errs, _ = outage_distance_errors(model, seqs, SPLITS["val"], norm, window, spacing_s=60)
        score = float(np.mean(errs)) + 5.0 * mae
        rec = {"epoch": epoch, "loss": tot / steps_per_epoch, "val_mae": mae,
               "val_outage60_median_pct": float(np.median(errs)), "val_outage60_mean_pct": float(np.mean(errs))}
        history.append(rec)
        tag = ""
        if score < best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), save_path)
            tag = " *"
        print(f"Epoch {epoch:02d} | loss {rec['loss']:.3f} | val MAE {mae:.2f} m/s | 60s outage err "
              f"median {rec['val_outage60_median_pct']:.1f}% mean {rec['val_outage60_mean_pct']:.1f}% "
              f"| {time.time() - t0:.0f}s{tag}")
    print(f"Best epoch {best_epoch}, checkpoint saved to {save_path}")
    with open(os.path.splitext(save_path)[0] + "_history.json", "w") as f:
        json.dump({"params": n_params, "window": window, "best_epoch": best_epoch, "history": history}, f, indent=1)
    return model


def load_model(path=CHECKPOINT):
    model = TinyTCN()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the Tiny TCN speed model")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rollout", type=float, default=0.5, help="weight of the 60 s rollout loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--highspeed-weight", type=float, default=1.0, help="sampling weight of windows > 15 m/s")
    ap.add_argument("--save", default=CHECKPOINT)
    ap.add_argument("--train", action="store_true", help="kept for backward compatibility (always trains)")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    train_tiny_tcn(epochs=a.epochs, lr=a.lr, lambda_rollout=a.rollout, seed=a.seed, save_path=a.save,
                   highspeed_weight=a.highspeed_weight)
