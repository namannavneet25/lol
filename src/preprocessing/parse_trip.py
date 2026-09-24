"""
IO-VNBD trip parser: aligns the smartphone (S-*.csv) and vehicle reference (V-*.csv)
logs on a common 10 Hz time grid and writes data/intermediate/<trip>.npz.

Why not pair rows?  The "synchronised" S/V files are not row-aligned: pairing by row
index leaves S2 8.7 s out of sync, drifts by >2 s within M, and scrambles Y1 and the
second half of S4 (the V logs contain 254 s / 384 s gaps).  Instead:
  1. phone wall-clock (DATE column) -> seconds of day, minus the whole-hour timezone
     offset to the vehicle clock ("Time Since Start of Day");
  2. the residual lag (a few 100 ms) is estimated by cross-correlating the phone's
     vertical gyro with the vehicle CAN yaw rate;
  3. both logs are interpolated onto one 10 Hz grid, and samples inside logging gaps or
     inside 60 s segments where phone and vehicle disagree are masked invalid.

Gyro axis labels in IO-VNBD are permuted: the column named "GYROSCOPE Pitch" is the
axis about gravity (vehicle yaw rate, r > 0.9 with the CAN yaw rate); "GYROSCOPE Yaw"
is not.  The vertical axis is detected from the data and asserted.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.navigation.frames import LocalTangentPlane

RAW_DIR = "data/raw/IO-VNBD/IO-VNBD-master/Synchronised V abd S datasets/Categorised IOVNB Dataset"
OUT_DIR = "data/intermediate"
DT = 0.1
MAX_LAG_S = 5.0
SEGMENT_S = 60.0
MIN_SEGMENT_CORR = 0.5

# Phone gyro columns in file order, and the phone axis each one is mapped to.
# "Pitch" is the axis about gravity (phone z, the phone lies flat in IO-VNBD).
GYRO_COLUMNS = {"GYROSCOPE Yaw (rad/s)": 0, "GYROSCOPE Roll (rad/s)": 1, "GYROSCOPE Pitch (rad/s)": 2}


def _read_csv(path):
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _col(df, prefix):
    return next(c for c in df.columns if c.startswith(prefix))


def _phone_time_of_day(df):
    stamp = df["DATE (YYYY-MO-DD HH-MI-SS_SSS)"].astype(str).str.strip().str.slice(11)
    parts = stamp.str.split(r"[-:_]", expand=True).astype(float)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2] + parts[3] / 1000.0).to_numpy()


def _xcorr_lag(a, b, max_lag):
    """Lag L (samples) maximising corr(a[t], b[t + L]); returns (L, r)."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.mean(a[:len(a) - lag] * b[lag:]) if lag else np.mean(a * b)
        else:
            r = np.mean(a[-lag:] * b[:lag])
        if r > best[1]:
            best = (lag, r)
    return best


def trip_name(trip_dir):
    base = os.path.basename(trip_dir)
    return "M" if "Driver B" in base else base


def list_trip_dirs(raw_dir=RAW_DIR):
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(raw_dir, "**", "S-*.csv"), recursive=True)})
    return [d for d in dirs if glob.glob(os.path.join(d, "V-*.csv"))]


def parse_trip(trip_dir, verbose=True):
    s = _read_csv(glob.glob(os.path.join(trip_dir, "S-*.csv"))[0])
    v = _read_csv(glob.glob(os.path.join(trip_dir, "V-*.csv"))[0])
    name = trip_name(trip_dir)

    t_phone = _phone_time_of_day(s)
    t_veh = v["Time Since Start of Day (seconds)"].to_numpy(dtype=float)
    hour_offset = round((t_phone[0] - t_veh[0]) / 3600.0) * 3600.0
    t_phone = t_phone - hour_offset

    keep = np.r_[True, np.diff(t_phone) > 0]
    s, t_phone = s[keep].reset_index(drop=True), t_phone[keep]
    keep = np.r_[True, np.diff(t_veh) > 0]
    v, t_veh = v[keep].reset_index(drop=True), t_veh[keep]

    t0 = max(t_phone[0], t_veh[0])
    t1 = min(t_phone[-1], t_veh[-1])
    t = np.arange(np.ceil(t0 * 10) / 10, t1, DT)

    def on_grid(times, values):
        return np.interp(t, times, np.asarray(values, dtype=float))

    can_yaw = np.radians(on_grid(t_veh, v["Yaw Rate (deg/sec)"]))
    speed = on_grid(t_veh, v["Velocity (km/hr)"]) / 3.6

    # Detect the gyro axis about gravity and the residual clock lag.
    moving = speed > 2.0
    gyro_raw = {c: s[c].to_numpy(dtype=float) for c in GYRO_COLUMNS}
    corr = {}
    for c, vals in gyro_raw.items():
        g = on_grid(t_phone, vals)
        corr[c] = np.corrcoef(g[moving], can_yaw[moving])[0, 1] if moving.sum() > 100 else 0.0
    vertical_col = max(corr, key=lambda c: abs(corr[c]))
    if moving.sum() > 600:
        assert GYRO_COLUMNS[vertical_col] == 2, f"{name}: unexpected vertical gyro column {vertical_col} ({corr})"

    lag_s, lag_r = 0.0, float("nan")
    if moving.sum() > 600:
        lag, lag_r = _xcorr_lag(on_grid(t_phone, gyro_raw[vertical_col]), can_yaw, int(MAX_LAG_S / DT))
        lag_s = lag * DT
    t_phone = t_phone + lag_s

    acc = np.stack([on_grid(t_phone, s[_col(s, f"ACCELEROMETER {ax}")]) for ax in "XYZ"], axis=1)
    gyro = np.zeros((len(t), 3))
    for c, axis in GYRO_COLUMNS.items():
        gyro[:, axis] = on_grid(t_phone, gyro_raw[c])

    # Validity: inside logging gaps, or in segments where phone and vehicle disagree.
    valid = np.ones(len(t), dtype=bool)
    for times in (t_veh, t_phone):
        gap_after = np.diff(times) > 0.5
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, len(gap_after) - 1)
        valid &= ~gap_after[idx]
    seg = int(SEGMENT_S / DT)
    seg_corr = []
    for start in range(0, len(t), seg):
        sl = slice(start, start + seg)
        m = moving[sl] & valid[sl]
        if m.sum() > 100 and np.std(can_yaw[sl][m]) > 0.02:
            r = np.corrcoef(gyro[sl, 2][m], can_yaw[sl][m])[0, 1]
            seg_corr.append(r)
            if r < MIN_SEGMENT_CORR:
                valid[sl] = False

    lat = on_grid(t_veh, v["Latitude (degrees)"])
    lon = on_grid(t_veh, v["Longitude (degrees)"])
    course_unwrapped = np.degrees(np.unwrap(np.radians(v["Heading (degrees)"].to_numpy(dtype=float))))
    course = on_grid(t_veh, course_unwrapped) % 360.0
    ltp = LocalTangentPlane(lat[0], lon[0])
    east, north = ltp.to_enu(lat, lon)

    # Phone GNSS (1 Hz, ~4 s latency in IO-VNBD). A new fix is flagged when any field changes.
    g_lat = s["GPS LATITUDE (degrees)"].to_numpy(dtype=float)
    g_lon = s["GPS LONGITUDE (degrees)"].to_numpy(dtype=float)
    g_spd = s["GPS SPEED (Kmh)"].to_numpy(dtype=float) / 3.6
    g_acc = s["GPS ACCURACY (m)"].to_numpy(dtype=float)
    g_brg = s[_col(s, "GPS ORIENTATION")].to_numpy(dtype=float)
    new_fix_raw = np.r_[True, (np.diff(g_lat) != 0) | (np.diff(g_lon) != 0) | (np.diff(g_spd) != 0)]
    idx = np.clip(np.searchsorted(t_phone, t, side="right") - 1, 0, len(t_phone) - 1)
    fix_id = np.cumsum(new_fix_raw)[idx]
    phone_gps = np.stack([g_lat[idx], g_lon[idx], g_spd[idx], g_acc[idx], g_brg[idx]], axis=1)
    phone_gps_new = np.r_[True, np.diff(fix_id) > 0]

    out = {
        "t": t, "acc": acc.astype(np.float32), "gyro": gyro.astype(np.float32),
        "speed": speed.astype(np.float32), "course_deg": course.astype(np.float32),
        "can_yaw": can_yaw.astype(np.float32), "lat": lat, "lon": lon,
        "east": east, "north": north, "valid": valid,
        "phone_gps": phone_gps, "phone_gps_new": phone_gps_new,
        "origin": np.array([lat[0], lon[0]]),
    }
    meta = {
        "trip": name, "dir": trip_dir, "samples": int(len(t)), "duration_min": round(len(t) * DT / 60, 1),
        "hour_offset_s": hour_offset, "lag_s": round(lag_s, 2), "lag_corr": round(float(lag_r), 3),
        "vertical_gyro_column": vertical_col, "gyro_corr": {k: round(float(r), 3) for k, r in corr.items()},
        "segment_corr_median": round(float(np.median(seg_corr)), 3) if seg_corr else None,
        "valid_pct": round(100.0 * valid.mean(), 1),
        "distance_km": round(float(np.sum(speed[valid]) * DT / 1000), 1),
    }
    if verbose:
        print(f"[{name:7s}] {meta['duration_min']:6.1f} min | lag {lag_s:+.1f} s (r={lag_r:.2f}) | "
              f"segment r~{meta['segment_corr_median']} | valid {meta['valid_pct']:5.1f}% | {meta['distance_km']} km")
    return out, meta


def parse_all(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    metas = {}
    for trip_dir in list_trip_dirs(raw_dir):
        data, meta = parse_trip(trip_dir)
        np.savez_compressed(os.path.join(out_dir, f"{meta['trip']}.npz"), **data)
        metas[meta["trip"]] = meta
    with open(os.path.join(out_dir, "trips_meta.json"), "w") as f:
        json.dump(metas, f, indent=2)
    print(f"\nParsed {len(metas)} trips into {out_dir}/")
    return metas


def load_trip(name, data_dir=OUT_DIR):
    with np.load(os.path.join(data_dir, f"{name}.npz")) as d:
        return {k: d[k] for k in d.files}


if __name__ == "__main__":
    parse_all()
