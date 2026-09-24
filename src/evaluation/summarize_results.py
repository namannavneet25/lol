"""Renders results/benchmark_*.json into Markdown tables (pasted into docs/RESULTS.md)."""
import glob
import json
import os
import sys

CONFIG_LABEL = {"legacy": "Original pipeline", "ekf_only": "EKF + gyro (no AI, no map)",
                "ekf_ai": "EKF + AI speed", "ekf_ai_map": "**Full IDR (EKF + AI + map)**"}
SCENARIO_LABEL = {"outage_30s": "30 s outages", "outage_60s": "60 s outages", "outage_120s": "120 s outages",
                  "ps_50m_under_1min": "PS example: 50 m in < 1 min", "ps_1km_highway": "PS example: 1 km at 47-144 km/h"}


def fmt(v, pct=False, digits=1):
    if v is None:
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.{digits}f}"


def table(report):
    lines = []
    for trip, scenarios in report["trips"].items():
        for sc, cfgs in scenarios.items():
            lines.append(f"\n**{trip} – {SCENARIO_LABEL.get(sc, sc)}**\n")
            lines.append("| Engine | n | drift median | drift mean | drift p90 | pass < 10 % | end error median | ms/frame |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for cfg, r in cfgs.items():
                s = r["summary"]
                if not s.get("n"):
                    lines.append(f"| {CONFIG_LABEL.get(cfg, cfg)} | 0 | – | – | – | – | – | – |")
                    continue
                lines.append(f"| {CONFIG_LABEL.get(cfg, cfg)} | {s['n']} | {fmt(s['drift_pct_median'])} % | "
                             f"{fmt(s['drift_pct_mean'])} % | {fmt(s['drift_pct_p90'])} % | {fmt(s['pass_rate_10pct'], True)} | "
                             f"{fmt(s['err_end_m_median'])} m | {fmt(s['frame_latency_ms_mean'], digits=2)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join("results", "benchmark_*.json")))
    for p in paths:
        with open(p) as f:
            print(table(json.load(f)))
