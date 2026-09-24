"""End-to-end smoke tests of CoreNavigationPipeline at phone (10 Hz) and edge-IMU (200 Hz) rates, and ONNX parity."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)

HAVE_MODEL = os.path.exists("models/tiny_tcn_best.pth") and os.path.exists("data/processed/normalization.json")
pytestmark = pytest.mark.skipif(not HAVE_MODEL, reason="trained model / normalization.json not available")


class CountingModel:
    def __init__(self, inner):
        self.inner, self.window, self.calls = inner, inner.window, 0

    def predict(self, window, scalars):
        self.calls += 1
        return self.inner.predict(window, scalars)


def _drive(rate_hz, seconds=90, outage=(40, 60), speed=12.0):
    """Straight drive east at `speed`, 1 Hz GNSS, GNSS blackout during `outage` (s)."""
    from src.models.inference import SpeedModel
    from src.navigation.frames import LocalTangentPlane
    from src.navigation.pipeline import CoreNavigationPipeline
    rng = np.random.default_rng(0)
    ltp = LocalTangentPlane(52.4, -1.5)
    model = CountingModel(SpeedModel())
    eng = CoreNavigationPipeline(calibration_path=None, model=model, use_map=False)
    states = []
    n = int(seconds * rate_hz)
    for k in range(n):
        t = k / rate_hz
        acc = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.4, 3)
        gyro = rng.normal(0, 0.01, 3)
        frame = {"timestamp": t, "ax": acc[0], "ay": acc[1], "az": acc[2], "gx": gyro[0], "gy": gyro[1], "gz": gyro[2]}
        on_second = abs(t - round(t)) < 0.5 / rate_hz
        if on_second and not (outage[0] <= t < outage[1]):
            lat, lon = ltp.to_geodetic(speed * t, 0.0)
            frame["gnss"] = {"lat": lat, "lon": lon, "speed": speed, "heading": 90.0, "accuracy": 3.0}
        states.append(eng.process_frame(frame))
    return states, model, ltp


@pytest.mark.parametrize("rate", [10, 200])
def test_pipeline_runs_at_rate_and_switches_modes(rate):
    states, model, _ = _drive(rate)
    assert len(states) == 90 * rate                                   # one NavigationState per IMU frame
    assert abs(model.calls - (90 * 10 - 40)) <= 3                     # model always runs at 10 Hz
    modes = [s["mode"] for s in states]
    t_dr = next(i for i, m in enumerate(modes) if i > 40 * rate and m == "DEAD_RECKONING") / rate
    assert 40.0 < t_dr <= 41.6                                        # GNSS loss detected within 1.5 fix intervals
    assert modes[int(45 * rate)] == "DEAD_RECKONING"
    assert modes[-1] == "GNSS_AIDED"
    e = np.array([s["east"] for s in states])
    assert np.all(np.isfinite(e))
    step = np.abs(np.diff(e))
    assert step[int(41 * rate):int(59 * rate)].max() < 40.0 / rate   # smooth, no jumps while dead reckoning


def test_heading_output_is_compass():
    states, _, _ = _drive(10, seconds=30, outage=(100, 101))
    assert abs(((states[-1]["heading"] - 90.0 + 180) % 360) - 180) < 5.0


def test_onnx_matches_torch():
    if not os.path.exists("models/idr_tcn.onnx"):
        pytest.skip("run src/models/export_onnx.py first")
    from src.models.inference import SpeedModel
    torch_m, onnx_m = SpeedModel(), SpeedModel("models/idr_tcn.onnx")
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((8, torch_m.window, 9)) * torch_m.std + torch_m.mean).astype(np.float32)
    s = np.column_stack([rng.uniform(0, 30, 8), rng.uniform(0, 120, 8), rng.integers(0, 2, 8)]).astype(np.float32)
    a, b = torch_m.predict_batch(w, s), onnx_m.predict_batch(w, s)
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-3, rtol=1e-4)
    assert math.isfinite(float(a[0][0]))

