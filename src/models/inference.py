"""
Runtime wrapper around the speed model: normalization + PyTorch or ONNX Runtime backend.
Produces the AIOutput interface: forward speed v_f (m/s), variance sigma_v^2, P(stationary).
"""
import json
import math

import numpy as np


class AIOutput:
    __slots__ = ("speed", "variance", "p_stationary")

    def __init__(self, speed, variance, p_stationary):
        self.speed, self.variance, self.p_stationary = speed, variance, p_stationary

    def as_dict(self):
        return {"speed": self.speed, "variance": self.variance, "p_stationary": self.p_stationary}


class SpeedModel:
    def __init__(self, weights_path="models/tiny_tcn_best.pth", norm_path="data/processed/normalization.json",
                 backend=None):
        with open(norm_path) as f:
            self.norm = json.load(f)
        self.mean = np.asarray(self.norm["mean"], dtype=np.float32)
        self.std = np.asarray(self.norm["std"], dtype=np.float32)
        self.scalar_scale = np.asarray(self.norm["scalar_scale"], dtype=np.float32)
        self.window = int(self.norm["window_size"])
        backend = backend or ("onnx" if weights_path.endswith(".onnx") else "torch")
        self.backend = backend
        if backend == "onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self.session = ort.InferenceSession(weights_path, opts, providers=["CPUExecutionProvider"])
        else:
            import torch
            from src.models.tiny_tcn import TinyTCN
            self.torch = torch
            self.model = TinyTCN()
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
            self.model.eval()

    def _inputs(self, windows, scalars):
        x = (np.asarray(windows, dtype=np.float32) - self.mean) / self.std        # (B, T, C)
        x = np.ascontiguousarray(np.transpose(x, (0, 2, 1)))
        s = np.asarray(scalars, dtype=np.float32) / self.scalar_scale
        return x, s

    def predict_batch(self, windows, scalars):
        """windows: (B, T, C) raw features; scalars: (B, 3) raw [v_ref, t_since_ref, ref_valid]."""
        x, s = self._inputs(windows, scalars)
        if self.backend == "onnx":
            speed, log_var, stat_logit = self.session.run(None, {"imu_window": x, "ref_scalars": s})
        else:
            with self.torch.no_grad():
                out = self.model(self.torch.from_numpy(x), self.torch.from_numpy(s))
            speed, log_var, stat_logit = (o.numpy() for o in out)
        return speed, np.exp(log_var), 1.0 / (1.0 + np.exp(-stat_logit))

    def predict(self, window, scalars):
        sp, var, ps = self.predict_batch(window[None], np.asarray(scalars)[None])
        return AIOutput(float(sp[0]), float(var[0]), float(ps[0]))


def ai_confidence(out):
    """Maps sigma_v to a 0..1 display confidence (1 m/s sigma -> 0.5)."""
    return 1.0 / (1.0 + math.sqrt(max(out.variance, 0.0)))


if __name__ == "__main__":
    m = SpeedModel()
    w = np.zeros((m.window, len(m.mean)), dtype=np.float32)
    print(m.backend, m.predict(w, [10.0, 5.0, 1.0]).as_dict())
