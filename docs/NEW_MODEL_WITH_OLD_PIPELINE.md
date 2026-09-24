# Can the new ML model be used with the old pipeline?

Other documents are listed in [`SUMMARY.md`](SUMMARY.md#documents).

**Short answer:** not as a drop-in. The new model's inputs and outputs changed, so the old pipeline can't call it without an adapter. Even with one, most of the old system's error comes from the fusion code, not the model.

## Why it won't plug in directly

| | Old pipeline expects | New model needs |
|---|---|---|
| Input window | 20 frames (2 s) of raw phone-axis channels | 40 frames (4 s) of gravity-levelled channels |
| Normalization | old `normalization.json` | new `data/processed/normalization.json` |
| Extra input | none | last GNSS speed, time since that fix, valid flag |
| Outputs | speed, log σ², 5 motion-class logits | speed, log σ², stationary logit |
| Constructor | `TinyTCN(in_channels=9, num_motion_classes=5)` | `TinyTCN()` / ONNX with two inputs |

If you fed it the old buffer, it would run but give meaningless speeds, because the channels mean different things.

## The minimum needed to use the new model on its own
- `src/preprocessing/features.py`: computes the input channels. This is required, or training and runtime won't match.
- `src/models/inference.py` plus `models/idr_tcn.onnx` (or `src/models/tiny_tcn.py` plus `models/tiny_tcn_best.pth`)
- `data/processed/normalization.json` (the new one)
- A roughly 20-line adapter in the old `pipeline.py`, which would:
  - build features with `GravityTracker` + `feature_frame`
  - keep the last GNSS speed and its time
  - call `SpeedModel.predict`
  - use `p_stationary` in place of the motion-class confidence

One more catch: if you replay IO-VNBD through the old parser, it still reads the wrong gyro column (it treats `GYROSCOPE Yaw` as the yaw axis; the real yaw axis is the column labelled `GYROSCOPE Pitch`). Live phone data isn't affected.

## Why that alone won't get you the results
This was later measured directly: the folder `../PS26168-ModelOnlyUpdated` applies only the ML changes to the original code, with a thin adapter like the one described above (see its `docs/ML_ONLY_RESULTS.md`). On 60 s outages on the test trip S1:

| Engine | Median drift |
|---|---|
| Old engine, old model | 97 % (3 seeds) |
| **Old engine, new model** (ML-only update) | 85 % (3 seeds) |
| New EKF + new model, no map | 17 % |
| Full new system | 9.2 % (3 seeds) |

The new model alone helps only a little inside the old engine: drift falls significantly only on short (30 s) outages, and other changes are not statistically significant. Drift stays at 52–92 %, because the old engine's heading is wrong. Even a perfect speed input leaves it at about 80 %.

The old EKF still has three bugs that would sabotage even a perfect speed input:
- It mixes compass and map heading conventions.
- It writes the gyro bias to the wrong state (`x[7]` instead of `x[5]`).
- It fuses GNSS before checking it.

## Recommendation
Use the full new pipeline; the measured results in `RESULTS.md` are based on it. The ML-only experiment confirms that the model alone is not enough: the fusion fixes are what turn a good speed model into low drift.
