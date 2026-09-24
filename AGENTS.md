# AGENTS.md — SIH PS26168: Intelligent Dead Reckoning (IDR) Project Guidelines

Project scope, repository layout, data/model contracts and rules for humans and AI assistants working on
**SIH PS26168: Intelligent Dead Reckoning System with GNSS Fusion**.
Overview: `docs/SUMMARY.md`, which lists every document. Rationale and evidence for the current design: `docs/OPTIMIZATION_PLAN.md`. What changed and why: `docs/CHANGES.md`. Measured results: `docs/RESULTS.md`. How to test: `docs/TESTING.md`.

---

## 🎯 Project Identity & Overview

* **Problem Statement**: PS26168 (`ProblemStatement.md`) — continuous smartphone vehicle navigation through GNSS
  outages; target **< 10 % drift of distance travelled** while dead reckoning, 10 Hz fusion on phones and higher rates
  (e.g. 200 Hz FOG IMUs) on edge devices.
* **Stack**: Python 3.10+ (NumPy, pandas, PyTorch, scikit-learn, Matplotlib), ONNX Runtime (Python / Web WASM),
  vanilla JS PWA (no build step), OpenStreetMap road data.

---

## 📁 Repository Structure

```text
PS26168/
├── AGENTS.md, ProblemStatement.md, plan.md   # rules, PS text, original design plan
├── docs/                                     # SUMMARY (index), OPTIMIZATION_PLAN, CHANGES, RESULTS, TESTING,
│                                             # NEW_MODEL_WITH_OLD_PIPELINE, NEXT_STEPS, img/
├── calibration.json                         # written by the web calibrator (/api/save_calibration)
├── requirements.txt
├── data/
│   ├── raw/IO-VNBD/                          # downloaded benchmark data (untracked, never modify)
│   ├── intermediate/<trip>.npz               # time-aligned 10 Hz trips + trips_meta.json
│   ├── maps/<trip>.geojson                   # offline OSM road network per trip (untracked)
│   └── processed/
│       ├── normalization.json                # feature mean/std (TRAIN trips) + model I/O contract
│       ├── sequences.npz                     # per-trip feature sequences for the trainer
│       └── {train,val,test,ood}_windows.npz  # fixed evaluation windows
├── models/
│   ├── tiny_tcn_best.pth                     # speed model checkpoint
│   ├── idr_tcn.onnx, model_config.json       # deployable model + runtime config
├── results/                                  # benchmark.json, model_eval.json, plots/ (generated)
├── src/
│   ├── preprocessing/
│   │   ├── download_trip.py                  # fetch IO-VNBD trips (Git LFS media URLs)
│   │   ├── parse_trip.py                     # clock alignment, gyro axis mapping, validity mask
│   │   ├── fetch_osm.py                      # Overpass -> data/maps/<trip>.geojson
│   │   ├── features.py                       # shared 10 Hz levelled IMU features (train == runtime)
│   │   └── create_windows.py                 # trip-level splits, normalization, datasets
│   ├── models/
│   │   ├── tiny_tcn.py                       # causal TCN, loss, augmentation, trainer
│   │   ├── inference.py                      # SpeedModel runtime (PyTorch or ONNX) -> AIOutput
│   │   ├── evaluate_models.py                # speed metrics + outage along-track error
│   │   ├── sklearn_baseline.py               # Ridge / HistGBR baselines
│   │   └── export_onnx.py                    # ONNX export + parity check + web copy
│   ├── navigation/
│   │   ├── frames.py                         # ENU / heading conventions, WGS84 tangent plane
│   │   ├── ins_ekf.py                        # 6-state EKF (pos, heading, speed, gyro bias & scale)
│   │   ├── gnss_quality.py                   # GNSS gating + mode state machine + re-anchoring
│   │   ├── map_matching.py                   # OSM road graph + online HMM matcher
│   │   ├── orientation.py                    # stand calibration helpers (web calibrator parity)
│   │   ├── pipeline.py                       # CoreNavigationPipeline: SensorFrame -> NavigationState
│   │   └── replay.py                         # quick end-to-end replay of one trip
│   ├── evaluation/benchmark.py               # PS benchmark: drift % over simulated outages, ablations
│   └── web/
│       ├── server.py                         # HTTPS server (/nav, /calibration, /api/*)
│       ├── export_replay.py, setup_vendor.py # replay data for the PWA, offline onnxruntime-web
│       ├── export_live_map.py                # offline OSM map for Live mode (nav/maps/live.geojson)
│       ├── calibration/                      # pre-drive stand calibrator
│       └── nav/                              # IDR Navigator PWA (js/ = port of src/navigation)
└── tests/                                    # pytest suite + tests/js/parity.test.mjs (Node)
```

---

## 📐 Data & Model Contracts

* **IO-VNBD quirks** (handled in `parse_trip.py`, do not "fix" elsewhere): phone and vehicle logs are aligned by
  wall-clock time (phone `DATE` − 1 h timezone + auto-estimated lag), **not by row**; the phone column labelled
  `GYROSCOPE Pitch` is the axis about gravity (vehicle yaw). Driver E trips (Vw*, Vta*, Vtb*) have a loosely mounted
  phone and are used only as an out-of-distribution test.
* **Splits (by trip)**: train S2, S4, M · val Y1 · test S1 · ood Vw02, Vw03, Vta01b, Vta06, Vtb01.
* **Model rate** 10 Hz; window **40 frames (4 s)**; higher-rate IMUs are block-averaged to 10 Hz (`features.Decimator`).
* **Input channels (C = 9)**, gravity-levelled (`features.py`): `a_h1, a_h2, a_v−g, w_h1, w_h2, w_yaw, |a|−g, |a_h|, |w|`.
  Horizontal axes are defined only up to the mount azimuth — the model is trained invariant to it.
* **Scalar inputs**: `[v_ref, t_since_ref, ref_valid]` — speed at the last trusted GNSS fix and seconds since it
  (≤ 120 s); `ref_valid = 0` means IMU-only.
* **Outputs (`AIOutput`)**: `speed` v_f (m/s, ≥ 0), `log_var` (EKF variance σ_v²), `stat_logit` (P(stationary) → ZUPT).
* **Target**: vehicle reference speed (m/s) at the window's last frame.
* **Conventions (`frames.py`)**: ENU metres; ψ = heading CCW from East (rad) inside filters; compass degrees
  (clockwise from North) in all external I/O (GNSS, NavigationState, OSM).

---

## ⚙️ Rules for AI Assistants

1. **Never modify raw data** in `data/raw/`; write derived data to `data/intermediate/`, `data/processed/`, `data/maps/`.
2. **One feature implementation**: training and runtime both use `src/preprocessing/features.py`; the JS port
   (`src/web/nav/js/features.js`) must stay numerically identical — run the parity test after any change.
3. **Normalization** comes only from `data/processed/normalization.json` (TRAIN trips); never recompute it at runtime.
4. **Python ↔ JS parity**: `src/web/nav/js/{frames,features,ekf,gnss,mapmatch,pipeline}.js` mirror the Python modules.
   Change both, then `python tests/js/make_fixture.py && node tests/js/parity.test.mjs`.
5. **Claims need measurements**: report accuracy only from `src/evaluation/benchmark.py` / `evaluate_models.py`
   outputs, never estimated numbers.
6. **Keep heavy/generated files untracked** (see `.gitignore`): datasets, maps, `.npz`, replay exports, WASM, TLS keys.
7. **Interfaces**:
   - `SensorFrame`: `{timestamp s, ax, ay, az (m/s², incl. gravity), gx, gy, gz (rad/s), gnss: null | {lat, lon, speed m/s, heading compass°, accuracy m}}`
   - `AIOutput`: `speed`, `variance`, `p_stationary`
   - `NavigationState`: `{timestamp ms, latitude, longitude, east, north, speed km/h, heading compass°, mode, gnssQuality, aiSpeed, aiConfidence, mapConfidence, positionStdM}`
     with `mode ∈ {GNSS_AIDED, DEGRADED, DEAD_RECKONING, REACQUISITION}`.

---

## 🚀 Commands (from the repo root, inside `.venv`)

| Step | Command |
|---|---|
| Environment | `python -m venv .venv && .venv/bin/pip install -r requirements.txt` |
| Fetch dataset | `python src/preprocessing/download_trip.py` |
| Align trips | `python src/preprocessing/parse_trip.py` |
| Fetch OSM maps | `python src/preprocessing/fetch_osm.py` |
| Build datasets | `python src/preprocessing/create_windows.py` |
| Train speed model | `python src/models/tiny_tcn.py --epochs 40` |
| Evaluate model | `python src/models/evaluate_models.py` · baselines: `python src/models/sklearn_baseline.py` |
| Export ONNX | `python src/models/export_onnx.py` |
| PS benchmark | `python src/evaluation/benchmark.py --trips S1 Y1`. Options: `--seed N`, `--gnss-latency S`, `--gnss-source phone`, `--configs ekf_ai_map_oracle`, `--legacy-root <snapshot>`, `--rate-test`. Pool seeds and CIs: `python src/evaluation/robustness_summary.py` |
| Tests (full guide: `docs/TESTING.md`) | `python -m pytest tests` · `python tests/js/make_fixture.py && node tests/js/parity.test.mjs` · `cd tests/e2e && npm install && node ui_e2e.mjs` |
| Web app | `python src/web/setup_vendor.py && python src/web/export_replay.py --trip S1` (and, for Live map matching, `python src/web/export_live_map.py --lat .. --lon ..`), then `python src/web/server.py` → `https://<LAN-IP>:8443/nav/`. Run the server only on trusted networks: the calibration upload has no login. |
