# IDR Optimisation — Summary (updated 2026-09-24)

The code and models were reworked against the PS26168 problem statement. An independent subagent then reviewed the whole folder. Its findings were fixed or documented, and results were re-measured with 3 GNSS seeds, confidence intervals and realistic GNSS.
- **Tests:** 58 Python tests pass; the browser engine matches the Python one (model inputs, filter state, and a 2.46 M-event GNSS-monitor fuzz); a 12-check headless-Chrome end-to-end test passes.
- **Documents:** see [Documents](#documents) at the end for the full list.

## What was wrong before
- **Training data was misaligned.** The phone and vehicle files were paired by row number, which put speed labels up to 8.7 s out of sync.
- **Heading used the wrong gyro axis.** The column labelled `GYROSCOPE Pitch` in the dataset is actually the yaw axis.
- **The shipped model was worse than guessing the average speed** (5.17 m/s error against 4.41). The documented 0.98 m/s does not reproduce.
- **The EKF mixed compass and map heading conventions.**
- **Position error was never measured.** The drift tables in `doc1.md`–`doc4.md` were never measured; those docs now carry a "superseded" note.

## Results on the held-out test trip (S1)
Drift is position error when GNSS returns, as a share of distance driven during the outage. The target is under 10 %.
Medians are pooled over 3 GNSS seeds, with 95 % confidence intervals and the share of outages under 10 % ([`RESULTS.md`](RESULTS.md)).

| Engine (idealised 1 Hz GNSS) | 60 s outages | 120 s outages |
|---|---|---|
| Original pipeline | 97 % median | 81 % |
| New, without map | 17 % | 14 % |
| **New, full system** | **9.2 % [5.6–12.5], 52 % pass** (32 outages) | **9.1 % [6.8–13.6], 51 % pass** (23 outages) |
| New, full system, *perfect speed* | 1.0 % | 0.3 % |

- **At the target, not safely below it.** The median sits on the 10 % line, about half of outages pass, and the problem statement's own examples are missed: 50 m → 8.3 m median error against < 5 m; 1 km → 146 m against < 100 m.
- **Realistic GNSS matters.** With fixes 1 s late: 11 % at 60 s. With the phone's own logged GPS (one fix per ~9 s, poor speed): 22–42 %. That variant exposed a real bug: the engine never re-anchored with sparse fixes. It is fixed.
- **Speed model:** 1.59 m/s error, with calibrated uncertainty and stop detection. On outage distance error it ties a gradient-boosted baseline (9.9 % vs 10.0 %).
- **Real time:**
  - The full system runs 113× faster than real time on 10 Hz data (0.89 ms/frame) and 43× on 200 Hz data.
  - Dead reckoning is declared about 1.5 s after the last fix (up to ~2.3 s in the first seconds after GNSS returns), with no jump in position.
- **Validation trip (Y1):** similar (8.8 % at 60 s), but Y1 was used for tuning, so it is not independent confirmation.

**Against the problem statement:** every listed capability exists in working form. The 10 % drift target is reached only at the median with good GNSS; the two worked examples (50 m and 1 km) are not met. A requirement-by-requirement comparison (Met, Partly met, Not met, Not tested) is in [`RESULTS.md` §7](RESULTS.md#7-comparison-with-the-problem-statements-expectations).

## What doesn't meet the target yet
- **Speed accuracy:** it is the dominant remaining error (perfect speed gives ≈ 1 %). Highway speeds are the weakest (7.7 m/s error above 72 km/h on S1, `results/model_eval.json`).
- **GNSS quality before the outage:** sparse or delayed fixes raise drift a lot.
- **Mounting:** a loosely mounted phone makes the AI speed worse than holding the last GNSS speed. The phone must be rigidly mounted; calibration does not fix a loose mount.
- **Tried and dropped:** an automatic check to detect an unreliable model hurt validation accuracy, so it was removed.

## Structural changes
- **Model format:** 4-second windows of 9 gravity-levelled channels, plus the last GNSS speed as an extra input. `AGENTS.md` describes the new contract.
- **Maps:** OpenStreetMap roads for each trip are stored offline in `data/maps/` (`src/preprocessing/fetch_osm.py`).
- **Navigation app:** `src/web/nav/` is an offline phone app (PWA) with live and replay modes and a "Simulate tunnel" button. Start it with `python src/web/server.py`, then open `/nav/`.
- **Live mode map matching:** live mode loads an offline road map for your area (`src/web/export_live_map.py`) and tells you whether it did. The map works offline once it has been loaded once while online. Without one, live mode runs without map matching (≈ 17 % instead of 9 %).
- **Security:**
  - The server refuses its TLS key folder and server sources, and lists no directories.
  - The key file is owner-only.
  - The calibration upload accepts only same-origin, size-limited, validated JSON.
  - It has no login, so run the server on trusted networks only.

## Housekeeping
- Nothing has been committed.
- A `.gitignore` keeps datasets, maps and the 12 MB browser runtime out of git. Regenerate them with the commands in `AGENTS.md`.
- `docs/OPTIMIZATION_PLAN.pdf` predates the "Implementation status" section of the `.md` version and may be out of date.

## Next steps
The gap to the problem statement and a prioritised path to closing it are in [`NEXT_STEPS.md`](NEXT_STEPS.md), in six workstreams:
- **A:** a high-rate IMU recording mode and field data.
- **B:** a better speed model.
- **C:** GNSS latency compensation and quality gating.
- **D:** mount-yaw estimation and loose-mount detection.
- **E:** validation on real phones and real tunnels.
- **F:** the remaining requirement gaps (magnetometer, AI-based fusion, faster blackout detection).

It also proposes concrete acceptance criteria for "meeting the expectation".

## Documents
| Document | Purpose |
|---|---|
| [`SUMMARY.md`](SUMMARY.md) | This one-page overview |
| [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) | The plan, the evidence behind it, and the implementation status and deviations |
| [`CHANGES.md`](CHANGES.md) | Short per-file change log: what changed, why, and which plan section it comes from |
| [`RESULTS.md`](RESULTS.md) | All measured numbers (speed model, drift benchmark with CIs, robustness, timing) and a requirement-by-requirement comparison with the problem statement (§7) |
| [`TESTING.md`](TESTING.md) | How to test: automated checks, browser end-to-end test, accuracy and robustness benchmarks, manual app testing |
| [`NEW_MODEL_WITH_OLD_PIPELINE.md`](NEW_MODEL_WITH_OLD_PIPELINE.md) | Q&A: why the new model can't be dropped into the old pipeline, and the minimum adapter needed |
| [`NEXT_STEPS.md`](NEXT_STEPS.md) | Gap to the problem statement, measured causes, prioritised path to close it, acceptance criteria, and the speed-model options |
| [`../../PS26168-ModelOnlyUpdated/docs/ML_ONLY_RESULTS.md`](../../PS26168-ModelOnlyUpdated/docs/ML_ONLY_RESULTS.md) | Separate experiment: only the ML changes applied to the original code. Result: significant drift reduction only on short outages; drift stays at 52–92 %, so the fusion fixes are what bring drift to the target line |
| `img/` | Figures used in the docs |
