# Next Steps: Closing the Gap to the Problem Statement

Other documents are listed in [`SUMMARY.md`](SUMMARY.md#documents). All numbers come from [`RESULTS.md`](RESULTS.md); §7 there compares each requirement in turn.

**Short answer:** every capability in the problem statement is built and working, but the performance target is not reliably met. The largest measured lever is **speed accuracy**, followed by **GNSS quality before the outage** and **mounting robustness**. Real-phone validation has not been done yet.

## 1. Where we stand

| Target (problem statement) | Current (test trip S1, full system) | Gap |
|---|---|---|
| Drift < 10 % of distance during blackouts ("restrict") | Idealised GNSS: median 9.2 % [95 % CI 5.6–12.5] at 60 s (32 outages), **52 % of outages under 10 %**. Fixes 1 s late: 11 %. Phone's own GPS: 22–42 %. | Median at the line; about half of outages fail; not robust to real GNSS |
| < 5 m over 50 m in under 1 min | Median 8.3 m; 33 % within 5 m | About 1.7× too large |
| < 100 m over 1 km at 60 km/h | Median 146 m; 25 % within 100 m (only 4 distinct outages) | About 1.5× too large; very small sample |
| 10 Hz on smartphones | 0.89 ms/frame on a desktop CPU | Not measured on a phone |
| ~200 Hz on edge with FOG IMU | 0.12 ms/frame (compute) | Accuracy untested, no real data |
| Alignment: pitch, roll **and yaw** relative to the vehicle | Pitch and roll automatic; yaw not estimated (the model is invariant to it instead) | Yaw estimation missing |
| AI-based fusion | AI enters through learned uncertainty and stop detection | No learned fusion beyond that |
| Transition "within milliseconds" | Seamless output, but blackout detection takes about 1.5 s at 1 Hz GNSS (up to ~2.3 s just after reacquiring) | Detection bounded by the GNSS rate |
| Magnetometer/compass input | Not used | Missing |

## 2. Why: the measured causes

| Cause | Evidence | Share of the gap |
|---|---|---|
| **Speed estimation error** | Perfect speed in the same engine: 1.0 % median at 60 s, 100 % pass, and ≈ 3 m on the 1 km example. The model's along-track error alone is ≈ 10 % at 60 s. | **Dominant** with good GNSS |
| **GNSS quality before the outage** | A 1 s fix latency doubles drift on the 50 m example (17 % → 35 %). Sparse phone GPS (one fix per ~9 s, poor speed) gives 22–42 %. | Large on short outages and with real receivers |
| **Speed model limits** | The TCN ties a gradient-boosted baseline; highway speeds are the weakest (7.7 m/s error above 72 km/h on S1). IO-VNBD's 10 Hz logging removes the vibration frequencies above 5 Hz that carry speed. | This is why speed is hard to improve on current data |
| **Mounting and out-of-distribution phones** | A loosely mounted phone (Driver E) makes the AI speed worse than holding the last GNSS speed (47 % vs 10.7 % along-track). All training phones lay flat. | Unknown share on real phones; potentially large |
| **Not yet validated** | No on-phone timing, no drive through a real tunnel, no tilted mounts, no external IMU data | Risk of unknown size |

## 3. Path to closing the gap
Workstreams are in priority order. The **verify** column says how progress would be measured, using the benchmark protocol in [`TESTING.md`](TESTING.md): tune on Y1, report S1 with 3 seeds and CIs, and always include `--gnss-latency 1.0` and `--gnss-source phone`.

| # | Workstream | What | Gap it targets | Verify | Effort |
|---|---|---|---|---|---|
| A | **High-rate data collection** | Add a recording mode to the web app: raw IMU at the phone's native rate (60–200 Hz), GNSS and timestamps, saved to a file. Record drives with a rigid mount, **tilted mounts**, several phones, highway sections, and real underpasses or tunnels. Add the organisers' datasets when released. | Speed accuracy, mount robustness, missing validation | New trips join the benchmark as extra test and validation trips | Recorder: ~1 day. Driving: ongoing. |
| B | **Better speed model** (options in §5) | Retrain on high-rate data. Meanwhile, on IO-VNBD: a stateful outage model and a small ensemble. | Dominant cause | Along-track error at 60 s (currently ≈ 10 %) and pass rate < 10 % | Days per experiment |
| C | **GNSS handling** | Latency compensation (fuse each fix at its own timestamp using a short state history); speed and course quality gating for poor receivers; reference speed from filtered GNSS rather than the last raw fix. | Short-outage (50 m) example, realistic-GNSS drift | `--gnss-latency 1.0` and `--gnss-source phone` results approach the idealised ones | ~2–3 days |
| D | **Mount robustness** | Estimate mount yaw automatically (from GNSS course vs levelled horizontal acceleration when accelerating); detect a loose mount (gyro vs GNSS-course consistency, vibration level) and fall back to holding the last speed; train with tilted and rotated mounts. | Alignment requirement; the Driver E failure | Driver E trips (currently 54–66 % with AI); tilted-mount recordings from A | ~3–5 days |
| E | **Real-device validation** | Measure per-frame timing on 2–3 phones in the browser; drive through real GNSS-denied places with the app logging; replay the logs through the benchmark. Test an external IMU if one is available. | Untested requirements (10 Hz on phone, edge, field behaviour) | On-phone ms/frame; field drift against GNSS after exit | Depends on access |
| F | **Remaining requirement gaps** | (1) **Magnetometer** as a heading aid when stopped or at low speed, gated for magnetic disturbance (cars and tunnels distort it). (2) **AI-based fusion:** let the network also set process noise and heading trust ("AI-IMU dead reckoning", Brossard et al. 2020). (3) **Faster blackout detection:** use the receiver's own signal-loss status where the platform exposes it, not just a fix timeout. | Magnetometer, AI fusion, "within milliseconds" | Drift change on the benchmark; detection time on device | ~1 week total |

**What we can say about expected impact:**
- **Speed (A + B):** perfect speed gives ≈ 1 %, so the upper bound of the payoff is large. How much of it real models capture is unknown until high-rate data exists.
- **GNSS handling (C):** targets the gap between idealised and realistic GNSS (about 2× on the 50 m example).
- **Mounting (D):** decides whether results hold on real phones.

## 4. What "meeting the expectation" would look like
Proposed acceptance criteria. They are stricter than a median, because the problem statement says "restrict":
- **Drift:** at least **90 % of outages under 10 %** at 60 s and 120 s, on S1 **and** at least one new independent trip, with 1 s GNSS latency; pooled over 3 seeds, with the CI reported.
- **The two worked examples:**
  - 50 m in under 1 min: median < 5 m, and at least 80 % of outages within 5 m.
  - 1 km at 60 km/h: median < 100 m, with at least 20 outages.
- **Robustness:** the same criteria on recordings with a tilted mount and on a second phone model.
- **Timing:** 10 Hz sustained on a mid-range phone in the browser (per-frame time well under 100 ms), measured on device.
- **Field:** at least one real underpass or tunnel drive, with drift measured at exit against GNSS.

## 5. Speed model options (analysis)
What was explored, on the held-out test trip S1:

| Model / experiment | Test speed error (MAE) | 60 s outage along-track error |
|---|---|---|
| Ridge regression | 2.57 m/s | – |
| Gradient-boosted trees on window statistics | 1.72 m/s | 10.0 % |
| **Tiny TCN (current)**, with calibrated uncertainty and stop detection | **1.59 m/s** | **9.9 %** |
| Window lengths 2, 5 and 10 s (with gradient-boosted trees) | little difference | – |
| Oversampling highway windows | no gain on validation | – |

**Interpretation:**
- Two very different models land within 1 % of each other, which usually means the input information has run out, not the model's capacity.
- The input is thin: 10 Hz logging, about 7 hours of training driving from two drivers, mostly urban.
- So a bigger TCN, a transformer or an LSTM on the same input would likely give only small gains. A transformer would also struggle with this little data and the phone's compute budget.

| # | Idea | Why | Effort |
|---|---|---|---|
| 1 | **Train on 100–200 Hz IMU data** (workstream A) | Restores the vibration frequencies that encode speed. The pipeline already accepts any rate; only retraining is needed. | After data collection |
| 2 | **Self-supervised fine-tuning on the user's own phone**, using GNSS speed as free labels during normal driving | Targets the loose-mount and different-phone failure. No manual labelling is needed. | Medium |
| 3 | **Model the whole outage**: a small GRU that keeps state from the moment GNSS drops, or a model predicting the speed change since the last fix (`plan.md` §28) | The current model sees a 4 s window plus the last GNSS speed. A stateful model can use the speed trend over the outage. | Medium, about 1 day |
| 4 | **Learned fusion noise** (workstream F2) | Addresses the "AI-based fusion" requirement and the heading-drift part of the error. | Medium |
| 5 | **Small ensemble** (3–5 TCNs) | A reliable few-percent gain and better uncertainty. The cost is 3–5× inference, still under 1 ms. | Low, about 2 hours of CPU |
| 6 | Bigger TCN or transformer on the same input | Small expected gain for the reasons above. | Low, but low value |

## 6. Suggested order
1. **Now (days):**
   - the high-rate recording mode (A);
   - GNSS latency compensation and quality gating (C);
   - the ensemble and stateful-outage experiments on IO-VNBD (B3, B5).
2. **Next (weeks):**
   - record drives (A);
   - retrain on high-rate data (B1);
   - mount-yaw estimation and loose-mount fallback (D);
   - on-phone timing (E).
3. **Then:**
   - field tunnel or underpass tests (E);
   - magnetometer and learned fusion (F);
   - judge against the §4 acceptance criteria.
