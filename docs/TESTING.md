# Testing Guide

You can still test by hand, but a few manual steps changed, and there's now an automated layer that should run first.
Other documents are listed in [`SUMMARY.md`](SUMMARY.md#documents). Run all commands from the repo root. `python` means `.venv/bin/python`.

## 1. One-time setup
If the environment or data isn't there yet:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python src/preprocessing/download_trip.py      # IO-VNBD trips
python src/preprocessing/parse_trip.py         # align phone + vehicle logs
python src/preprocessing/fetch_osm.py          # offline OpenStreetMap roads
python src/preprocessing/create_windows.py     # datasets + normalization.json
```

## 2. Automated checks (run these first, about 30 seconds)
```bash
python -m pytest tests                  # 58 tests: EKF (incl. yaw sign), GNSS modes (incl. sparse receivers), map matching, features,
                                        # 10/200 Hz pipeline, ONNX parity, web-server security
python tests/js/make_fixture.py         # re-run after changing engine code or retraining
node tests/js/parity.test.mjs           # browser engine == Python engine: model inputs (features + reference
                                        # scalars) on every 20th call, then EKF / GNSS modes / map matching per frame
python tests/js/make_gnss_fixture.py && node tests/js/gnss_monitor.test.mjs   # GNSS monitor: JS == Python on
                                        # 300 random dense/sparse fix sequences (2.46 M events)
```

## 2b. Browser end-to-end test (about 1 minute)
Drives the real web apps in headless Chrome. It starts its own server on port 8547, and backs up and restores `calibration.json`.
```bash
npm install --prefix tests/e2e      # one-time, from the repo root
node tests/e2e/ui_e2e.mjs            # finds Chrome in its default place on Linux/macOS/Windows, or set CHROME=...
```
It runs 12 checks:
- **Navigator, replay:** GNSS-aided → tunnel → dead reckoning → recovery, with an outage logged and the replay keeping up at 16×.
- **Navigator, live mode:** without an offline map, and with one.
- **Page health:** no page errors or failed requests.
- **Calibration upload:** a same-origin save is accepted, an invalid calibration is rejected, and a cross-origin page is blocked.

Results go to `results/ui_e2e.json`, with a screenshot.
The test starts the server with the activated virtual environment's Python, so activate it first. If Chrome is not found, set `CHROME` to the "Executable Path" shown at `chrome://version`. If the server fails to start, the test prints the server's error output. Passed on Linux and Windows. Windows setup (PowerShell): see `TEAM_SETUP_AND_TESTING.md`, which is shared alongside the zips.
It needs the replay data and vendored runtime from §5 step 1.

## 3. Accuracy against the problem-statement target
Use this instead of judging accuracy by eye:
```bash
python src/evaluation/benchmark.py --trips S1 Y1
```
It reports drift % per outage (target < 10 %) and saves plots in `results/plots/`.

For a quick check, taking about a minute per trip:
```bash
python src/evaluation/benchmark.py --trips S1 --scenarios outage_60s --configs ekf_ai_map
```

The default GNSS simulation is idealised: 1 Hz, no latency, constant 3 m accuracy. Robustness variants:
```bash
python src/evaluation/benchmark.py --trips S1 --seed 1 --out results/v2/S1_seed1.json          # other noise seeds
python src/evaluation/benchmark.py --trips S1 --gnss-latency 1.0 --out results/v2/S1_latency1s.json
python src/evaluation/benchmark.py --trips S1 --gnss-source phone --out results/v2/S1_phonegps.json   # real phone GPS
python src/evaluation/benchmark.py --trips S1 --configs ekf_ai_map_oracle --out results/v2/S1_oracle.json  # perfect speed
python src/evaluation/robustness_summary.py   # pooled seeds + bootstrap 95 % CIs -> results/robustness_summary.json
python src/evaluation/benchmark.py --rate-test   # throughput at 10 / 200 Hz, with and without map matching
```

The reference numbers to compare against are in `RESULTS.md`. Tune on Y1 only; S1 is the test trip.

## 4. What changed in manual testing

| Before | Now |
|---|---|
| `python src/navigation/test_map_matching.py` printed a snapping table to read by eye | Removed. It's now `tests/test_map_matching.py`, with pass/fail checks. |
| `python src/navigation/replay.py` printed modes and sample JSON | Still works, and now also reports the position error after a simulated tunnel (e.g. `--trip S1 --outage-seconds 60`). |
| `python tests/test_orientation_calibration.py` | Still works as a script, now fixed (it previously checked for the bias bug). |
| Calibrator at `https://<LAN-IP>:8443/` | Moved to `https://<LAN-IP>:8443/calibration/`. The root now redirects to the navigation app at `/nav/`. |
| `python src/models/tiny_tcn.py --train` | Still works. `--train` is now optional, and training writes `models/tiny_tcn_best.pth` plus a history file. |

## 5. Manual app testing
1. Run once:
   ```bash
   python src/web/setup_vendor.py
   python src/web/export_replay.py --trip S1
   ```
2. Start the server: `python src/web/server.py`.
3. Open `https://<LAN-IP>:8443/nav/` (or `https://localhost:8443/nav/` on the desktop):
   - **Replay:** press Play, toggle "Simulate tunnel", and the outage log shows the drift % against ground truth.
   - **Live on a phone:** put the phone in its mount and run `/calibration/` first, then open `/nav/` and choose Live.
     For map matching in live mode, first download an offline map for your area:
     `python src/web/export_live_map.py --lat <lat> --lon <lon> --radius-km 5`.
     Live mode shows whether a map was loaded.

## 6. After retraining the model
```bash
python src/models/evaluate_models.py    # speed metrics + outage along-track error
python src/models/export_onnx.py        # refresh models/ and the web app copy
```
Then re-run sections 2 and 3.
