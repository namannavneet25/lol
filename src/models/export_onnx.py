"""
Exports the Tiny TCN to ONNX for on-device inference (onnxruntime-web in the PWA, ONNX Runtime
Mobile / Python on edge devices) and verifies it against PyTorch.

Inputs : imu_window (B, 9, T) float32, ref_scalars (B, 3) float32   (normalized, see normalization.json)
Outputs: speed (B,), log_var (B,), stat_logit (B,)
Also writes models/model_config.json - everything a runtime needs besides the weights.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.tiny_tcn import CHECKPOINT, TinyTCN

ONNX_PATH = "models/idr_tcn.onnx"
NORM_PATH = "data/processed/normalization.json"
WEB_MODEL_DIR = "src/web/nav/model"


def export_to_onnx(weights_path=CHECKPOINT, output_onnx_path=ONNX_PATH, norm_path=NORM_PATH):
    with open(norm_path) as f:
        norm = json.load(f)
    window, channels = norm["window_size"], len(norm["channel_names"])
    model = TinyTCN(in_channels=channels)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    x = torch.randn(1, channels, window)
    s = torch.tensor([[0.5, 0.2, 1.0]])
    torch.onnx.export(model, (x, s), output_onnx_path, export_params=True, opset_version=17,
                      do_constant_folding=True, input_names=["imu_window", "ref_scalars"],
                      output_names=["speed", "log_var", "stat_logit"],
                      dynamic_axes={"imu_window": {0: "batch"}, "ref_scalars": {0: "batch"},
                                    "speed": {0: "batch"}, "log_var": {0: "batch"}, "stat_logit": {0: "batch"}},
                      dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    xb = np.random.default_rng(0).standard_normal((256, channels, window)).astype(np.float32)
    sb = np.random.default_rng(1).uniform(0, 1, (256, 3)).astype(np.float32)
    sb[:, 2] = (sb[:, 2] > 0.3).astype(np.float32)
    with torch.no_grad():
        ref = [o.numpy() for o in model(torch.from_numpy(xb), torch.from_numpy(sb))]
    got = sess.run(None, {"imu_window": xb, "ref_scalars": sb})
    max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(ref, got))
    assert max_diff < 1e-4, f"ONNX/PyTorch mismatch {max_diff}"

    x1, s1 = xb[:1], sb[:1]
    for _ in range(20):
        sess.run(None, {"imu_window": x1, "ref_scalars": s1})
    t0 = time.perf_counter()
    for _ in range(500):
        sess.run(None, {"imu_window": x1, "ref_scalars": s1})
    latency_ms = (time.perf_counter() - t0) / 500 * 1000

    size_kb = os.path.getsize(output_onnx_path) / 1024
    config = {
        "onnx": os.path.basename(output_onnx_path),
        "inputs": {"imu_window": [None, channels, window], "ref_scalars": [None, 3]},
        "outputs": ["speed", "log_var", "stat_logit"],
        "params": int(sum(p.numel() for p in model.parameters())),
        "size_kb": round(size_kb, 1),
        "cpu_latency_ms": round(latency_ms, 3),
        "parity_max_abs_diff": max_diff,
        "normalization": norm,
    }
    with open(os.path.join(os.path.dirname(output_onnx_path), "model_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    if os.path.isdir(os.path.dirname(WEB_MODEL_DIR)):
        os.makedirs(WEB_MODEL_DIR, exist_ok=True)
        shutil.copy(output_onnx_path, os.path.join(WEB_MODEL_DIR, os.path.basename(output_onnx_path)))
        with open(os.path.join(WEB_MODEL_DIR, "model_config.json"), "w") as f:
            json.dump(config, f, indent=1)
    print(f"Exported {output_onnx_path}: {size_kb:.1f} KB, {config['params']} params, "
          f"ONNX/PyTorch max diff {max_diff:.2e}, single-window CPU latency {latency_ms:.3f} ms")
    return config


if __name__ == "__main__":
    export_to_onnx()
