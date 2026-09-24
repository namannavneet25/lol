"""
Records a Python reference run for the JS parity test (tests/js/parity.test.mjs):
8 minutes of trip S1 with a 2-minute GNSS outage, the OSM map around it, every model output,
and the expected NavigationState per frame. Run after training: python tests/js/make_fixture.py
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.evaluation.benchmark import frame_iter, load_bench_trip, synthetic_gnss
from src.models.inference import SpeedModel
from src.navigation.pipeline import CoreNavigationPipeline

START, END, OUTAGE = 3000, 7800, (5000, 6200)
OUT = os.path.join(os.path.dirname(__file__), "fixture.json")


INPUT_SAMPLE_EVERY = 20


class RecordingModel:
    """Records every model output, and the exact model inputs (window features + scalars) every 20th call."""

    def __init__(self, inner):
        self.inner, self.window, self.log, self.inputs = inner, inner.window, [], {}

    def predict(self, window, scalars):
        if len(self.log) % INPUT_SAMPLE_EVERY == 0:
            self.inputs[len(self.log)] = {"window": np.asarray(window, dtype=float).round(9).tolist(),
                                          "scalars": [float(v) for v in scalars]}
        out = self.inner.predict(window, scalars)
        self.log.append([out.speed, out.variance, out.p_stationary])
        return out


def clip_map(path, ltp, e0, n0, e1, n1, margin=300):
    with open(path) as f:
        gj = json.load(f)
    keep = []
    for feat in gj["features"]:
        coords = feat["geometry"]["coordinates"]
        en = [ltp.to_enu(la, lo) for lo, la in coords]
        if any(e0 - margin <= e <= e1 + margin and n0 - margin <= n <= n1 + margin for e, n in en):
            keep.append(feat)
    return {"type": "FeatureCollection", "features": keep}


def main():
    trip = load_bench_trip("S1")
    fixes = synthetic_gnss(trip)
    mask = np.zeros(len(trip["t"]), dtype=bool)
    mask[OUTAGE[0]:OUTAGE[1]] = True
    sl = slice(START, END)
    gmap = clip_map("data/maps/S1.geojson", trip["ltp"], trip["east"][sl].min(), trip["north"][sl].min(),
                    trip["east"][sl].max(), trip["north"][sl].max())
    model = RecordingModel(SpeedModel())
    eng = CoreNavigationPipeline(calibration_path=None, model=model)
    frames, states = [], []
    for i, fr in frame_iter(trip, fixes, mask):
        if i < START:
            continue
        if i >= END:
            break
        if eng.ltp is None and fr["gnss"] is not None:
            eng._set_origin(fr["gnss"]["lat"], fr["gnss"]["lon"])
            eng.load_map(gmap)
        st = eng.process_frame(fr)
        frames.append(fr)
        states.append([eng.ekf.x[0], eng.ekf.x[1], eng.ekf.x[2], eng.ekf.x[3], st["mode"], st["mapConfidence"]])
    with open(OUT, "w") as f:
        json.dump({"frames": frames, "ai": model.log, "ai_inputs": model.inputs, "states": states, "map": gmap,
                   "window": model.window}, f)
    print(f"Wrote {OUT}: {len(frames)} frames, {len(model.log)} model calls, {len(gmap['features'])} roads, "
          f"modes {sorted(set(s[4] for s in states))}")


if __name__ == "__main__":
    main()
