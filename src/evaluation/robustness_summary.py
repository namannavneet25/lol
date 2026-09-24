"""
Robustness summary of the benchmark: pools GNSS seeds and reports bootstrap 95 % confidence intervals
(cluster bootstrap over distinct outages, since every seed replays the same outage windows).

Inputs
  results/benchmark_<trip>.json      seed 0, idealised synthetic GNSS (the original run)
  results/v2/<trip>_seed{1,2}.json   seeds 1-2, same GNSS model
  results/v2/<trip>_latency1s.json   synthetic GNSS delayed by 1 s
  results/v2/<trip>_phonegps.json    the phone's own logged GPS (~0.11 Hz)
  results/v2/<trip>_oracle.json      full system with perfect (ground-truth) speed

Output: results/robustness_summary.json and Markdown tables on stdout.
"""
import json
import os

import numpy as np

MIN_OUTAGE_SPEED = 3.0
TRIPS = ("S1", "Y1")
SCENARIOS = {"outage_30s": "30 s", "outage_60s": "60 s", "outage_120s": "120 s",
             "ps_50m_under_1min": "PS 50 m < 1 min", "ps_1km_highway": "PS 1 km highway"}
CONFIGS = ("ekf_only", "ekf_ai", "ekf_ai_map")
RNG = np.random.default_rng(0)


def drifts(path, trip, sc, cfg):
    """{outage start frame: drift %} for one run, or None if the file/config is missing."""
    if not os.path.exists(path):
        return None
    rep = json.load(open(path))
    r = rep["trips"].get(trip, {}).get(sc, {}).get(cfg)
    if r is None:
        return None
    return {o["start"]: o["drift_pct"] for o in r["outages"] if o["valid"] and o["mean_speed_mps"] >= MIN_OUTAGE_SPEED}


def stats(runs, n_boot=2000):
    """
    runs: per-seed {outage start: drift}. The seeds replay the SAME outages with different GNSS noise, so
    observations of one outage are correlated. The bootstrap therefore resamples whole outages (clusters),
    taking every seed's result for each sampled outage; n_outages counts distinct outages.
    """
    runs = [r for r in runs if r]
    if not runs:
        return None
    starts = sorted(set().union(*[r.keys() for r in runs]))
    clusters = [np.array([r[s] for r in runs if s in r]) for s in starts]
    pooled = np.concatenate(clusters)
    meds, passes = [], []
    for _ in range(n_boot):
        pick = RNG.integers(0, len(clusters), size=len(clusters))
        sample = np.concatenate([clusters[i] for i in pick])
        meds.append(np.median(sample))
        passes.append(np.mean(sample < 10.0))
    return {"n_outages": len(starts), "n_observations": int(len(pooled)), "seeds": len(runs),
            "median": float(np.median(pooled)),
            "median_ci": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "pass_rate": float(np.mean(pooled < 10.0)),
            "pass_ci": [float(np.percentile(passes, 2.5)), float(np.percentile(passes, 97.5))],
            "p90": float(np.percentile(pooled, 90))}


def main():
    out = {}
    for trip in TRIPS:
        seed_files = [f"results/benchmark_{trip}.json", f"results/v2/{trip}_seed1.json", f"results/v2/{trip}_seed2.json"]
        for sc in SCENARIOS:
            row = {}
            for cfg in CONFIGS:
                row[f"{cfg}|3 seeds"] = stats([drifts(p, trip, sc, cfg) for p in seed_files])
                row[f"{cfg}|latency 1 s"] = stats([drifts(f"results/v2/{trip}_latency1s.json", trip, sc, cfg)])
                row[f"{cfg}|phone GPS"] = stats([drifts(f"results/v2/{trip}_phonegps.json", trip, sc, cfg)])
            row["ekf_ai_map_oracle|perfect speed"] = stats([drifts(f"results/v2/{trip}_oracle.json", trip, sc,
                                                                   "ekf_ai_map_oracle")])
            out.setdefault(trip, {})[sc] = row
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/robustness_summary.json", "w"), indent=1)

    def fmt(r):
        if r is None:
            return "–"
        return (f"{r['median']:.1f} % [{r['median_ci'][0]:.1f}–{r['median_ci'][1]:.1f}] · "
                f"{100 * r['pass_rate']:.0f} % [{100 * r['pass_ci'][0]:.0f}–{100 * r['pass_ci'][1]:.0f}] "
                f"(n={r['n_outages']})")

    for trip, scs in out.items():
        print(f"\n**{trip}** — median drift [95 % CI] · share of outages < 10 % [95 % CI]\n")
        print("| Scenario | Variant | EKF only | EKF + AI | Full system |")
        print("|---|---|---|---|---|")
        for sc, label in SCENARIOS.items():
            for variant in ("3 seeds", "latency 1 s", "phone GPS"):
                cells = [fmt(scs[sc].get(f"{c}|{variant}")) for c in CONFIGS]
                print(f"| {label} | {variant} | " + " | ".join(cells) + " |")
            print(f"| {label} | perfect speed | – | – | {fmt(scs[sc].get('ekf_ai_map_oracle|perfect speed'))} |")


if __name__ == "__main__":
    main()
