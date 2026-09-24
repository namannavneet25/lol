"""
Random GNSS fix sequences + the Python monitor's decisions, for tests/js/gnss_monitor.test.mjs.
Covers dense (1 Hz) and sparse (up to 20 s) receivers, outages, position jumps, missing/poor accuracy.
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.navigation.gnss_quality import GNSSQualityMonitor

OUT = os.path.join(os.path.dirname(__file__), "gnss_fixture.json")


def sequence(rng):
    period = float(rng.choice([0.2, 1.0, 1.0, 5.0, 9.0, 20.0]))
    mon = GNSSQualityMonitor()
    t, e, events = 0.0, 0.0, []
    for _ in range(int(rng.integers(40, 120))):
        gap = period * rng.uniform(0.9, 1.1)
        if rng.random() < 0.08:
            gap += rng.uniform(5, 90)                       # outage
        t_next = t + gap
        while t + 0.1 < t_next:                              # frame ticks between fixes
            t += 0.1
            events.append({"type": "tick", "t": round(t, 6), "mode": mon.tick(round(t, 6))})
        t = round(t_next, 6)
        v = float(rng.uniform(0, 30))
        e += v * gap
        jump = float(rng.normal(0, 300)) if rng.random() < 0.1 else 0.0
        acc = None if rng.random() < 0.05 else float(rng.uniform(1, 30))
        nis = float(rng.exponential(5) if rng.random() < 0.7 else rng.uniform(10, 1000))
        fs = float(max(0.0, v + rng.normal(0, 2)))
        fix = {"t": t, "e": e + jump, "n": float(rng.normal(0, 3)), "speed": v, "acc": acc, "nis": nis, "fs": fs}
        d = mon.on_fix(t, fix["e"], fix["n"], v, acc, nis, filter_speed=fs)
        events.append({"type": "fix", **fix, "decision": d, "mode": mon.mode})
        events.append({"type": "tick", "t": t, "mode": mon.tick(t)})
    return events


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    seqs = [sequence(rng) for _ in range(300)]
    with open(OUT, "w") as f:
        json.dump(seqs, f)
    print(f"Wrote {OUT}: {len(seqs)} sequences, {sum(len(s) for s in seqs)} events")
