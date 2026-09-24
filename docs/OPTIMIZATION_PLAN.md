# IDR Optimisation Plan (`docs/OPTIMIZATION_PLAN.md`)

> Status: **implemented** (2026-09-23). This file records *why* each change is being made and
> the evidence behind it, so reviewers can check the reasoning. See also: [`SUMMARY.md`](SUMMARY.md) (one-page
> overview), [`CHANGES.md`](CHANGES.md) (per-file change log) and [`RESULTS.md`](RESULTS.md) (measured results).


## Context

The goal is to make the code and models fit the problem statement: smartphone dead reckoning through GNSS outages with **<10% drift of distance travelled**, GNSS+INS fusion at 10 Hz on phone and higher rates on edge devices, map matching, and a real-time navigation UI. Agreed scope: the full stack including the web app, a changed model I/O contract, and an offline OpenStreetMap extract for map matching.

Findings. Every item below was verified by running code on the repo's own data (2026-09-23):

**Data**
- **Row pairing is wrong.** The IO-VNBD "synchronised" S/V files are paired row by row, which misaligns targets:
  - S2 is off by 8.7 s.
  - M drifts from 0.9 s to 3.2 s within the trip.
  - Y1 and the second half of S4 are fully decorrelated, because the V file has 254 s and 384 s gaps.
  - Aligning on wall-clock timestamps fixes every trip. The fix is `phone DATE − 1 h`, then a lag refinement of −0.8 to +0.3 s. After alignment, gyro-vs-CAN yaw correlation is 0.9+ throughout.
- **Gyro axes are mislabelled.** The CSV column `GYROSCOPE Pitch` is the vertical (yaw) axis: r = 0.93–0.99 with the CAN yaw rate, gain ≈ 0.95. The code integrates heading from `GYROSCOPE Yaw`, which has r ≈ 0.05.
- **Phone GPS lags by ~4.2 s** and updates at 1 Hz. *Corrected 2026-09-24: this holds for the phone's GPS **speed** column only. The phone's GPS **position** updates only about every 9 s (0.11 Hz), with 3–5 m error and no measurable lag.*
- **The vehicle `Heading` column is a compass course**, confirmed against the lat/lon track.

**Model**
- The shipped checkpoint scores **val MAE 7.83 m/s and test MAE 5.17 m/s**. That is worse than always predicting the mean (7.41 and 4.41 m/s). The documented 0.98 m/s does not reproduce.
- The motion head only ever emits 2 of its 5 classes, because the labels are synthetic.
- The train/val/test split is sequential across concatenated trips, which contradicts `plan.md` §36.

**Fusion**
- The GNSS heading update and re-anchoring use compass degrees directly as the ENU ψ, so they fight the velocity update.
- `pipeline.py:40` writes the gyro bias into `x[7]` (accelerometer y bias) instead of `x[5]`. `tests/test_orientation_calibration.py` asserts this bug, and it also unpacks 4 of the 6 return values.
- The GNSS update runs before the quality gate.
- The ENU origin is hard-coded to Guwahati.
- There is no map file, so map matching is a no-op.
- `replay.py` feeds exact ground truth as GNSS and computes **no position error**. The accuracy tables in doc2/doc4 are unmeasured.

**Feasibility (leave-one-trip-out, 60 s outages)**

| Estimator | Result |
|---|---|
| Accelerometer integration | 35–55% distance error (unusable) |
| Gyro heading with pre-outage bias estimate | ~5–10% median drift, heavy tails |
| Absolute speed model | 13–20% along-track |
| Speed model conditioned on last GNSS speed | 8–21% along-track |

To hit <10% reliably, the design needs all four of:
- aligned data
- a GNSS-conditioned speed model with learned uncertainty and ZUPT
- online gyro-bias estimation in the EKF
- HMM map matching on real OSM roads, for cross-track error plus along-track correction at turns

## Approach, in executable order

### 1. Data pipeline — `src/preprocessing/`
- **`parse_trip.py`** (rewrite with pandas):
  - Parse the phone `DATE` into seconds of day and remove the whole-hour offset.
  - **Auto-estimate the fine lag** by cross-correlating the vertical gyro with the CAN yaw rate. It is automatic so future organiser datasets work too.
  - Resample S and V onto a common 10 Hz grid.
  - Add a `valid` mask for V/phone gaps and for 60 s segments where the moving gyro/CAN correlation is below 0.5.
  - Map gyro axes via a per-dataset axis map (IO-VNBD: vertical = `Pitch` column), with an assertion based on the correlation check.
  - Keep GT fields: ENU position from V lat/lon, speed, compass course, and CAN yaw. Also keep phone GPS fields (lat, lon, speed, accuracy, bearing).
  - Write `data/intermediate/<trip>.npz` plus `trips_meta.json` (lag, correlation, valid %).
- **`download_trip.py`:**
  - Fix Vta06 and Vtb01, whose V files are missing. Check the upstream names and drop them if they are unavailable.
  - Add **`fetch_osm.py`**: an Overpass query for drivable `highway=*` within a 1 km corridor of each trip, saved to `data/maps/<trip>.geojson`.
- **`create_windows.py`:**
  - Build **trip-level splits**: train S2, S4, M, Vw02; val Y1; test S1, plus Vw03 and Vta01b as the out-of-distribution driver/vehicle.
  - Add a leave-one-trip-out CV mode for model selection.
  - Features are computed in a leveled vehicle frame; see §2.
  - Save `normalization.json` with the new schema: channels, mean/std, window, rate, scalar-input stats, axis conventions, model version.
  - Delete the stale root `normalization.json`.

### 2. Speed model — `src/models/`
- **New I/O contract** (`AIOutput` = `v_f`, `σ_v²`, `p_stationary`), 10 Hz model rate:
  - Sequence input of 6 leveled channels: `a_h1, a_h2, a_v−g, ω_h1, ω_h2, ω_yaw`, plus `|a|−g` and `|ω|`. Window length is chosen from {2, 4, 6} s by LOTO; default 4 s.
  - Scalar inputs: `v_ref` (last trusted GNSS speed), `t_since_ref` (clipped to 120 s), `ref_valid`.
  - Training samples `t_since_ref` uniformly from 0–120 s and sets `ref_valid=0` 25% of the time, so the model also works with no GNSS at all.
- **`tiny_tcn.py`:**
  - **Causal** dilated TCN, ~30–50k params. Scalars are fused via FiLM/concat at the dense layer.
  - Heads: speed through a softplus (always ≥0), clamped `log σ²`, and `p_stationary` through a sigmoid. The 5-class head is dropped.
  - Loss: β-NLL (Gaussian) + Huber + BCE (stationary) + a **rollout loss** on the mean speed of contiguous 60 s sequences, which directly optimises the drift metric.
  - Remove the "physics" and smoothness terms, which are meaningless on shuffled batches.
  - Augmentation per `plan.md` §33–35: random yaw rotation about the vertical (mount-azimuth invariance), small tilt, bias, scale, noise, sample dropout and time jitter.
  - Training uses AdamW, cosine schedule, early stopping on val 60 s distance error, and fixed seeds.
- **`sklearn_baseline.py` / `mlp_baseline.py`:** Port to the new data. Replace the RF with a HistGBR on window statistics, then run the `plan.md` §37 experiment matrix and log results to `models/experiments.json`.
- **`evaluate_models.py`:** Per-trip MAE/RMSE, speed-binned MAE, σ calibration (z-std), stationary F1, and plots saved to `data/processed/`.
- **`export_onnx.py`:**
  - Two inputs (`imu_window`, `ref_scalars`) and three outputs.
  - Verify with onnxruntime parity (≤1e-4) and time a single inference.
  - Emit `models/model_config.json`.

### 3. Navigation engine — `src/navigation/`
- **Conventions module (`frames.py`):**
  - ENU with ψ measured CCW from East.
  - `compass_to_psi` and `psi_to_compass` helpers, and a proper local tangent-plane converter.
  - The origin is set from the first GNSS fix instead of a hard-coded location.
  - All outputs use compass heading.
- **`ins_ekf.py`** (reformulate):
  - State `[px, py, ψ, v, b_g, k_v]`, where `k_v` is the AI speed scale. Velocity along ψ makes NHC a hard constraint; this gets documented as the NHC implementation.
  - Predict: gyro yaw rate minus bias, and `v` driven weakly by the leveled forward acceleration.
  - Updates:
    - AI speed with R = σ², inflated for window overlap
    - ZUPT when `p_stationary` > 0.8, which also makes the gyro bias observable
    - GNSS position with R from accuracy and χ² gating
    - GNSS course/speed above 2 m/s
    - map pseudo-measurement, cross-track only, with R from map confidence
  - Use the Joseph-form covariance update.
- **`gnss_quality.py`:**
  - Gate before fusing.
  - Timeout of 1.5× the GNSS interval, for an instant DR switch.
  - Re-anchor after 2 consistent fixes, then reset the counters.
  - Record the time of each mode change.
- **`map_matching.py`:**
  - Online HMM in the Newson–Krumm style over the OSM graph.
  - Grid spatial index.
  - Emissions from distance and heading difference; transitions from network distance vs. EKF-travelled distance.
  - Outputs a matched point, road heading, and confidence.
  - Turns at junctions give an along-track correction.
- **`pipeline.py`:**
  - Wire in the leveled features from `orientation.py`, reusing `PhoneOrientationEstimator`.
  - Fix the `x[5]` bias bug.
  - Add a **rate-agnostic front end**: IMU at any rate (e.g. 200 Hz) gets EKF prediction at the native rate plus an anti-aliased decimation to the 10 Hz model rate.
  - Model backend is either PyTorch or ONNX Runtime.
  - Keep the `SensorFrame` → `NavigationState` schemas and add `positionStdM`.

### 4. Benchmark harness (new) — `src/evaluation/benchmark.py`
- Replay each test trip with GNSS generated as GT + noise at 1 Hz, with configurable latency. A phone-GPS variant is also available. *(The latency and phone-GPS options were only built on 2026-09-24, after the independent review; the first results used idealised GNSS.)*
- Outage scenarios:
  - 30, 60 and 120 s outages tiled across the trip
  - "tunnel" outages on segments above 50 km/h
  - the PS examples: 50 m in under 1 min, and 1 km at 60 km/h
- Metrics:
  - end-of-outage error as % of distance travelled
  - max error
  - pass rate below 10%
  - mode-switch latency
  - per-frame latency, and throughput at 10 Hz and 200 Hz
- Ablations: hold-last-speed, old pipeline (via git stash of the originals), new pipeline without map, new pipeline with map.
- Outputs `results/benchmark.json` and **trajectory plots per outage**, which the PS requires for the proposal.
- `replay.py` becomes a thin wrapper around this.

### 5. Web app — `src/web/`
- New **`nav/` PWA**, fully offline-capable:
  - `index.html`
  - `app.js`
  - `features.js`: a port of the leveling and window features
  - `ekf.js`
  - `mapmatch.js`
  - `model.js`: onnxruntime-web, vendored into `src/web/vendor/` so it works without a network
  - `manifest.json` and a service worker
- UI:
  - canvas map drawn from the OSM GeoJSON, with no tile server
  - smooth vehicle icon, interpolated at 60 fps between 10 Hz states
  - mode badge, GNSS quality, AI speed ±σ, and map confidence
  - a "Simulate tunnel" toggle that drops GNSS
- Modes:
  - **Live**: DeviceMotion + Geolocation. Reuses `calibration/orientation_calibrator.js` and `calibration.json`.
  - **Replay**: plays an IO-VNBD trip exported by `src/web/export_replay.py`.
- `server.py` also serves `/nav`, `/models` and `/maps`.
- Make sure `.certs/key.pem` is gitignored.

### 6. Tests, docs and housekeeping
- **pytest:**
  - alignment lag recovery on synthetic data
  - compass/ψ round trip
  - EKF: straight-line DR, bias convergence, ZUPT
  - HMM junction choice
  - pipeline smoke test at 10 and 200 Hz
  - ONNX parity
  - fix `tests/test_orientation_calibration.py`
  - convert `test_map_matching.py` to real asserts
- **Node parity test** (`tests/js/parity.test.mjs`): the JS features and EKF against a Python-recorded fixture, with model outputs stubbed. Node v24 is available.
- **Docs:**
  - Update `AGENTS.md`: I/O contract, structure, commands, Python 3.10+ (the machine has 3.10).
  - Add `docs/RESULTS.md` with **measured** numbers.
  - Mark the doc2/doc4 benchmark tables as superseded.
- **Housekeeping:** `requirements.txt`; `.gitignore` for `.venv`, `__pycache__`, `data/`, `results/`, `.certs`; `.venv` already exists with torch 2.x CPU and onnxruntime.

## Critical files
- **Rewrite:** `src/preprocessing/parse_trip.py`, `create_windows.py`, `src/models/tiny_tcn.py`, `src/navigation/ins_ekf.py`, `map_matching.py`, `pipeline.py`
- **Modify:** `gnss_quality.py`, `orientation.py`, `export_onnx.py`, `evaluate_models.py`, `src/web/server.py`, `AGENTS.md`
- **New:** `fetch_osm.py`, `src/navigation/frames.py`, `src/evaluation/benchmark.py`, `src/web/nav/*`, `src/web/export_replay.py`, `tests/*`
- **Never modify** `data/raw/`, per AGENTS.md.

## Verification
1. `.venv/bin/python src/preprocessing/parse_trip.py` prints the per-trip lag and correlation. Every kept trip should have moving gyro/CAN correlation ≥0.85.
2. `create_windows.py` → `tiny_tcn.py --train` → `evaluate_models.py`: the test MAE must beat the old model's 5.17 m/s and the HistGBR baseline (~2–3 m/s LOTO).
3. `src/evaluation/benchmark.py` produces the ablation table. **Success criterion:** median 60 s outage drift below 10% with the map on the test trips, compared against hold-speed and the old pipeline. The pass rate gets reported honestly even where it falls short.
4. `pytest tests/` and `node tests/js/parity.test.mjs` pass. `export_onnx.py` parity is ≤1e-4.
5. `python src/web/server.py`: open `/nav` in Chrome, run the replay of S1 with "Simulate tunnel" on, and confirm the icon stays on the road and the mode switches between DR and GNSS. Take screenshots via Chrome.

## Expectations
The <10% target is realistic for highway or tunnel outages and with map constraints. In urban stop-and-go traffic without a map, the evidence says 10–20%, and results will be reported as measured.


## Implementation status & deviations (2026-09-23)

Everything above was implemented. Where the evidence pointed elsewhere, the plan was changed as follows:

| Plan item | What was done | Why |
|---|---|---|
| EKF state `[px, py, ψ, v, b_g, k_v]` (AI speed scale) | State `[px, py, ψ, v, b_g, s_g]`: **gyro scale factor** instead of an AI scale | Phone gyros read 1–7.5 % low against the GNSS course rate (S1 0.94, Y1 0.94, S2 0.97). An AI-scale self-calibration was tried separately and hurt validation (below). |
| OSM corridor 1 km | 400 m corridor | Keeps maps at 0.3–3 MB for the phone while still including parallel roads, junction arms and service roads. |
| MLP baseline port | Replaced by Ridge + HistGradientBoosting on window statistics | HistGBR is the stronger classical baseline (1.72 m/s test MAE), so it sets a tougher bar for the TCN. |
| Driver E trips in the splits | Used only as an out-of-distribution test | Their phone gyro disagrees with the CAN yaw rate (r = 0.1–0.4) and is 4× noisier: the phone was not rigidly mounted. |
| Final model on train+val | Kept the model trained on train trips only | Keeps the validation (Y1) and test (S1) numbers clean for the proposal. Retraining on train+val is a cheap next step. |
| Vta06 / Vtb01 "missing" | Fixed: upstream files are named `V-vta6.csv` / `V-vtb1.csv` | Added to the OOD set. |
| — (new) | Tuned on Y1 only: map update period 0.5 s, map-feedback confidence ≥ 0.4, AI variance ×4 | Selected by 60 s outage drift on the validation trip, never on the test trip. |
| — (tried, rejected) | "AI trust monitor": re-run the model on a virtual 30 s outage at every fix, learn a speed scale + variance inflation | Made Y1 worse (median 8.8 % → 9.6–13.9 % across variants), so it was removed. |
| Legacy comparison | Old engine replayed on the same aligned data and GNSS; it gets the course on every fix (it crashes on stationary fixes without one). Trips are relocated to its hard-coded Guwahati origin, as its own replay did, and it runs with the original `calibration.json`. | So the ablation compares engines, not data handling. The first run omitted the relocation and overstated the old engine's error; `RESULTS.md` was corrected. |

### Changes after the independent review (2026-09-24)
A fresh subagent reviewed the whole folder. It judged the code largely sound and found that every reported number matched `results/*.json`, but it also found overstated claims and robustness gaps. What changed:

| Finding | Change |
|---|---|
| Headline "meets < 10 %" rests on 1 trip, 1 seed, n = 23–32, and idealised GNSS | Added GNSS seeds, `--gnss-latency`, `--gnss-source phone` and a perfect-speed config to the benchmark, plus `robustness_summary.py` with bootstrap 95 % CIs. `RESULTS.md` now reports pooled results with intervals. |
| Live mode ran without map matching | `export_live_map.py` downloads an offline map for the user's area; Live mode loads it and says whether it did |
| Calibration upload: no origin/type/size checks, key file world-readable, directory listings | Same-origin JSON only, 64 KB cap, schema validation, threaded server with timeouts, web-asset allow-list, key mode 0600. `tests/test_server.py` added. |
| Parity test did not check model inputs; yaw-sign test drove straight | Parity now compares JS vs Python model inputs; a real turning test was added |
| `accuracy: null` crashed the Python pipeline; reference speed not refreshed in DEGRADED mode; iOS sign missing in the calibrator | Fixed (Python and JS) |
| Perfect-speed and "16× in Chrome" claims had no saved evidence | `results/v2/*_oracle.json`; `tests/e2e/ui_e2e.mjs` with results in `results/ui_e2e.json` |
| Y1 presented as confirmation although it is the tuning trip | Docs now say so |
| (found while re-measuring) Engine never re-anchored with sparse GNSS (phone GPS, one fix per ~9 s): errors grew to kilometres | Plausibility check for fix pairs > 5 s apart; fix-interval cap raised to 15 s (Python and JS, with tests). 1 Hz results verified bit-identical. |
| Not fixed (documented as known limitations) | Training/runtime sampling difference (10 Hz point samples vs block-averaged phone data; unverified, needs high-rate recordings); Dijkstra ignores one-way streets; the EKF's state clamps don't adjust P; map pseudo-measurements are treated as independent, so `positionStdM` is optimistic; minor trainer edge cases (rollout fallback, `t_since` off by 0.1 s) |

### Second independent re-check (2026-09-24)
A second fresh reviewer verified the fixes above: the code fixes were correct, 1 Hz results were bit-identical, JS and Python agreed, and the doc numbers matched. It then found the following, all now addressed:

| Finding | Change |
|---|---|
| Bootstrap CIs too narrow, n inflated ~3×: the 3 seeds replay the same outages | Cluster bootstrap over distinct outages; n now counts distinct outages (e.g. the 1 km example has only 4). The same flaw in the ML-only paired tests was fixed too, which reduced that experiment's "6 of 10 significant" to "30 s outages only". |
| Sparse re-anchor too loose: one fix 400 m off could be adopted | Sparse pairs must fit the filter's own speed (× 1.5 + 3 m/s, capped at 60 m/s). |
| The 15 s interval cap slowed loss detection after short reacquisitions | 15 s cap only once the receiver is evidently sparse (estimated interval > 3 s); 5 s otherwise, which is the original behaviour. |
| Very sparse receivers (> 22 s) still never re-anchored | The pending fix now expires by age (60 s), not by the loss timeout. |
| Server edge cases (NaN, huge numbers, deep nesting, NUL byte, boolean components, non-atomic writes) | All rejected with 400 or 404; writes are atomic; covered by tests. |
| Weak tests | e2e cross-origin check now verifies the file is unchanged; JS/Python GNSS monitor fuzz test (300 sequences, 2.46 M events); monitor tests tick the clock between fixes; forbidden-path tests require 404. |
| Doc details (a stale 0.97 %, an unsourced highway figure, "3–4×", map-corridor caveat, detection time after reacquisition, Live double watch) | Corrected; `app.js` stops the previous watch before starting Live mode. |

