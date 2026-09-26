# Change Log — IDR Optimisation (2026-09-23)

A quick overview of every change: what changed, **why**, and which part of
[`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) it implements (§1 data, §2 model, §3 navigation, §4 benchmark,
§5 web app, §6 tests/docs). Measured outcomes are in [`RESULTS.md`](RESULTS.md); one-page overview in [`SUMMARY.md`](SUMMARY.md).
The plan's closing table, "Implementation status & deviations", lists where evidence changed the plan.

## Why anything changed (plan → Context)
Five problems were measured before any change:
1. **The training data was misaligned.** Phone and vehicle logs were paired by row, which put targets up to 8.7 s off, and two trips were fully scrambled.
2. **Heading used the wrong gyro axis.** The column labelled `GYROSCOPE Pitch` is the actual yaw axis.
3. **The shipped model was worse than predicting the mean speed** (test MAE 5.17 m/s against 4.41).
4. **The EKF mixed compass and ENU heading conventions.**
5. **Position error was never measured.**

Every change below addresses one of these, or a requirement from `ProblemStatement.md`.

## §1 Data pipeline — `src/preprocessing/`
| File | Change | Why |
|---|---|---|
| `parse_trip.py` (rewritten) | Aligns trips by wall clock, with an auto-estimated lag. Auto-detects the vertical gyro axis. Masks logging gaps and bad segments. Writes `data/intermediate/<trip>.npz`. | Fixes problems 1 and 2. After the fix, gyro-vs-CAN yaw correlation is 0.9+ on every well-mounted trip. |
| `features.py` (new) | One implementation of the 9 gravity-levelled features, the gravity tracker and a decimator for any IMU rate. | Training and runtime used to compute features differently. Levelling makes the model independent of the phone mount. Decimation handles 200 Hz IMUs. |
| `create_windows.py` (rewritten) | Trip-level splits (train S2/S4/M · val Y1 · test S1 · OOD Driver E). Train-only normalization in the new schema. Per-trip sequences for on-the-fly sampling. | The old sequential split leaked across trips and ignored `plan.md` §36. |
| `fetch_osm.py` (new) | Offline OpenStreetMap road network per trip, including oneway and roundabout tags. | Map matching needs real roads; there was no map before. |
| `download_trip.py` | Fixed the names of the Vta06/Vtb01 vehicle files. | The files exist upstream in lowercase; two trips were silently missing. |
| `normalization.json` (root, deleted) | Removed the stale duplicate. | Only `data/processed/normalization.json` is valid (AGENTS rule 3). |

## §2 Speed model — `src/models/`
| File | Change | Why |
|---|---|---|
| `tiny_tcn.py` (rewritten) | Causal TCN (25.8 k params) conditioned on the last GNSS speed. Outputs speed, log σ² and P(stationary). Loss is Huber + β-NLL + BCE + **60 s rollout loss**. Augmentations include random mount yaw, bias, scale, noise and sample dropout. Early stopping uses validation drift. | The rollout loss optimises the PS metric directly. σ feeds the EKF (z-std 1.01), and P(stationary) drives ZUPT. Result: test MAE 5.17 → 1.59 m/s. |
| `inference.py` (new) | `SpeedModel` runtime wrapping PyTorch or ONNX. Produces `AIOutput`. | A single model interface for the pipeline, the benchmark and tests. |
| `export_onnx.py` | Two-input ONNX, a parity check (≤ 1e-4), a latency measurement, `model_config.json`, and a copy into the web app. | The deployable artefact is verified, not assumed. |
| `evaluate_models.py` | Test/OOD metrics, calibration, stationary F1, outage along-track error and a plot. | Reports what matters for dead reckoning, not just MAE. |
| `sklearn_baseline.py` | Ridge + HistGBR baselines on the new data. `mlp_baseline.py` deleted. | HistGBR is a stronger bar than the MLP (plan deviation). |

## §3 Navigation engine — `src/navigation/`
| File | Change | Why |
|---|---|---|
| `frames.py` (new) | ENU/ψ ↔ compass helpers and an exact WGS84 tangent plane. | Fixes problem 4. The origin comes from the first fix instead of hard-coded Guwahati. |
| `ins_ekf.py` (rewritten) | State `[p, ψ, v, gyro bias, gyro scale]`. The hard NHC (velocity along heading) replaces the soft pseudo-measurement. Adds ZUPT + ZARU, χ²-gated GNSS, a cross-track map update, and Joseph-form covariance. | Gyro bias and scale (phones read 1–7.5 % low) were the dominant heading errors. The perfect-speed benchmark gives about 1 % drift once speed is right (`results/v2/*_oracle.json`). |
| `gnss_quality.py` (rewritten) | Gate-before-fuse. Loss timeout of 1.5 × the fix interval. Re-anchors on 2 self-consistent fixes. Logs transitions. Sparse receivers (fixes > 5 s apart) re-anchor on a plausibility check — added after the phone-GPS benchmark showed the engine never recovering. | The old monitor fused GNSS before checking it, and flickered into dead-reckoning mode between 1 Hz fixes. |
| `map_matching.py` (rewritten) | Online HMM (Newson–Krumm) over the OSM graph, with a grid index, one-way awareness and network-distance transitions. | This was the biggest single gain: 17 % → 9.2 % median drift on S1 at 60 s (3 seeds). Turns correct along-track error. |
| `pipeline.py` (rewritten) | Native-rate prediction, 10 Hz model branch, GNSS state machine, map feedback in DR mode, and the `x[5]` bias-index bug fixed. Map period (0.5 s) and confidence threshold (0.4) tuned on Y1. | Wires the pieces together, and runs the same engine for phone and edge IMUs (PS requirement). |
| `replay.py` | Now a thin wrapper around the benchmark that reports the actual error. | The old replay printed modes only. |
| `test_map_matching.py` (moved) | Became `tests/test_map_matching.py`, with asserts. | It had no assertions. |

## §4 Benchmark (new) — `src/evaluation/`
| File | Why |
|---|---|
| `benchmark.py` | Fixes problem 5. Simulates 1 Hz GNSS with outages of 30/60/120 s plus the two PS examples. Ablates legacy / EKF / +AI / +map. Produces plots and 10/200 Hz timing. |
| `benchmark.py` options, `robustness_summary.py` (added after review) | `--seed`, `--gnss-latency`, `--gnss-source phone` and a perfect-speed config; seeds are pooled with bootstrap 95 % CIs. | The first headline came from idealised GNSS, one seed and small n. |
| `_legacy_runner.py` | Replays the original engine in isolation, so the comparison is on identical data. |
| `summarize_results.py` | Renders the benchmark JSON into the tables in `RESULTS.md`. |

## §5 Web app — `src/web/`
| File | Change | Why |
|---|---|---|
| `nav/` (new PWA) | `js/{frames,features,ekf,gnss,mapmatch,pipeline}.js` port the Python engine. `model.js` runs ONNX in the browser. `app.js`/`index.html` provide the canvas map, a smooth vehicle icon, mode/GNSS/AI/map meters, a "Simulate tunnel" toggle, an outage log, and Live and Replay modes. Also `sw.js` + `manifest.json` for offline use. | The PS requires a real-time navigation UI running on the phone. Previously only a calibrator existed. |
| `server.py` | Serves the whole web root (`/nav`, `/calibration`) and adds `GET /api/calibration`. Hardened after review: web-asset allow-list, no directory listings, TLS key blocked on the resolved path (key mode 0600), and the calibration upload accepts only same-origin JSON ≤ 64 KB that passes validation. Threaded, with timeouts. | Serving the web root would otherwise have exposed `.certs/key.pem`, and the upload was writable by any page on the LAN. |
| `export_replay.py`, `setup_vendor.py` (new) | Generate replay data and clipped maps. Fetch the pinned onnxruntime-web (offline). | Reproducible demo assets without committing large binaries. |
| `export_live_map.py` (new) | Downloads an offline OSM map around the user's location for Live mode. | Live mode previously ran without map matching (≈ 17 % instead of ≈ 9 % median drift at 60 s). |

## §6 Tests, docs, housekeeping
| File | Why |
|---|---|
| `tests/test_navigation.py`, `test_pipeline.py`, `test_map_matching.py`, `test_server.py` (new) | Conventions, EKF (including a real yaw-sign turn test), GNSS state machine (including sparse receivers and outlier fixes), HMM, 10/200 Hz pipeline, ONNX parity, and web-server security with edge cases. 58 tests pass. |
| `tests/test_orientation_calibration.py` | Previously asserted the bias-index bug and unpacked the wrong tuple size. |
| `tests/js/make_fixture.py`, `parity.test.mjs` (new) | JS ↔ Python engine parity. It checks model inputs (features and reference scalars, 238 sampled calls, ≤ 5e-10) and then the EKF, GNSS modes and map matching per frame (2e-10 m). The model itself is stubbed; ONNX parity is covered in `test_pipeline.py`. |
| `tests/js/make_gnss_fixture.py`, `gnss_monitor.test.mjs` (new) | JS GNSS monitor == Python on 300 random dense and sparse fix sequences (2.46 M events). |
| `tests/e2e/ui_e2e.mjs` (new) | Headless-Chrome end-to-end test of the navigator (replay, tunnel, recovery, live mode with and without a map) and the calibration upload rules. 12 checks. |
| `AGENTS.md` | New structure, I/O contracts, data quirks, parity rule and commands. |
| `docs/SUMMARY.md`, `OPTIMIZATION_PLAN.md`, `RESULTS.md`, `CHANGES.md`, `TESTING.md`, `NEW_MODEL_WITH_OLD_PIPELINE.md`, `NEXT_STEPS.md`, `img/` | Overview and document index, plan, measured results, this log, testing guide, model-compatibility Q&A, next steps, and figures. |
| `doc1.md`–`doc4.md` | Added a "superseded" banner, because their accuracy claims don't reproduce. |
| `.gitignore`, `requirements.txt` (new) | Keep data, maps, WASM and TLS keys untracked. Pin the environment (`.venv`, Python 3.10). |

## Windows support (2026-09-25, from the first team run on Windows)
| File | Why |
|---|---|
| `src/web/server.py` | The startup banner's emoji crashed the server on Windows when its output was redirected (cp1252 encoding). The server now replaces characters that can't be printed. |
| `tests/e2e/ui_e2e.mjs` | It now uses the activated virtual environment, or else `.venv` on Windows or Unix, and looks for Chrome in the usual install locations on Windows, macOS and Linux. It waits up to 60 s for the server, and stops with the server's own error output (and a hint when packages are missing) instead of a vague "connection refused". The e2e test passed on Windows. |
| `src/web/server.py`, `tests/test_server.py` | Two fixes. The server now rejects any URL containing a NUL byte: newer Python on Windows (3.14) no longer raises on it during path resolution, so a request could reach the stdlib file handler and drop the connection instead of returning 404. The server tests' two oversized payload cases now get short IDs: the long ones exceeded the Windows limit on environment variable length, since pytest stores the current test's ID in an environment variable. |

## Tried and rejected (evidence in the plan's deviations table)
- **AI "trust monitor"** (online self-check of the model): made validation worse (8.8 % → 9.6–13.9 %), so it was removed.
- **High-speed oversampling:** no gain on validation. The checkpoint is kept in `results/experiments/`.
