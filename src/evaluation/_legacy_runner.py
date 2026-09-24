"""Runs the pre-optimisation navigation engine on pickled SensorFrames (used by benchmark.py)."""
import os
import pickle
import sys
import time

legacy_root, inp, out = sys.argv[1:4]
with open(inp, "rb") as f:
    frames = pickle.load(f)
sys.path.insert(0, legacy_root)
os.chdir(legacy_root)
from src.navigation.pipeline import CoreNavigationPipeline  # noqa: E402  (legacy package)

engine = CoreNavigationPipeline()
outs, lat_ms = [], []
for fr in frames:
    t0 = time.perf_counter()
    o = engine.process_frame(fr)
    lat_ms.append((time.perf_counter() - t0) * 1000)
    outs.append({"latitude": o["latitude"], "longitude": o["longitude"], "mode": o["mode"]})
with open(out, "wb") as f:
    pickle.dump((outs, lat_ms), f)
