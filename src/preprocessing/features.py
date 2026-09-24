"""
IMU feature extraction shared by offline training (create_windows.py) and live
inference (navigation/pipeline.py, web/nav/features.js).

Everything here runs at the 10 Hz model rate. Higher-rate sensors (e.g. a 200 Hz
FOG IMU) are block-averaged down to 10 Hz first (see Decimator), which doubles as
the anti-aliasing filter.

Feature channels (C = 9), all in a gravity-levelled frame so the model does not
depend on how the phone sits in its holder:
  0 a_h1        levelled horizontal accel, axis 1       (m/s^2)
  1 a_h2        levelled horizontal accel, axis 2       (m/s^2)
  2 a_v         vertical accel minus 1 g                (m/s^2)
  3 w_h1        levelled horizontal angular rate, ax 1  (rad/s)
  4 w_h2        levelled horizontal angular rate, ax 2  (rad/s)
  5 w_yaw       angular rate about gravity (vehicle yaw rate)  (rad/s)
  6 a_mag       |a| - 1 g                               (m/s^2)
  7 a_h_mag     |a_horizontal|                          (m/s^2)
  8 w_mag       |w|                                     (rad/s)

The horizontal axes are only defined up to the (unknown) mount azimuth, so training
applies random rotations about the vertical axis and the model learns to be
invariant to it.
"""
import numpy as np

G = 9.80665
MODEL_RATE_HZ = 10.0
CHANNEL_NAMES = ["a_h1", "a_h2", "a_v", "w_h1", "w_h2", "w_yaw", "a_mag", "a_h_mag", "w_mag"]
NUM_CHANNELS = len(CHANNEL_NAMES)
SCALAR_NAMES = ["v_ref", "t_since_ref", "ref_valid"]
T_SINCE_REF_MAX = 120.0


def leveling_matrices(g_hat):
    """
    Rows [x_l, y_l, z_l] of the levelling rotation for unit gravity vectors g_hat (N,3) or (3,).
    Same construction as PhoneOrientationEstimator.compute_leveling_matrix: z along gravity,
    y = phone-top projected onto the horizontal plane (phone-x when lying flat).
    """
    g = np.atleast_2d(np.asarray(g_hat, dtype=np.float64))
    z = g / np.linalg.norm(g, axis=1, keepdims=True)
    y_ref = np.array([0.0, 1.0, 0.0])
    y = y_ref - z[:, 1:2] * z
    flat = np.linalg.norm(y, axis=1) < 1e-3
    if np.any(flat):
        x_ref = np.array([1.0, 0.0, 0.0])
        x_flat = x_ref - z[flat, 0:1] * z[flat]
        y[flat] = np.cross(z[flat], x_flat)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    return R[0] if np.ndim(g_hat) == 1 else R


class GravityTracker:
    """
    Causal gravity-direction estimator: exponential average of the accelerometer,
    updated only while the specific force is close to 1 g and the phone is not rotating
    quickly (so braking/cornering does not tilt the estimate).
    """

    def __init__(self, tau_s=10.0, rate_hz=MODEL_RATE_HZ, g_init=None):
        self.alpha = 1.0 / (tau_s * rate_hz)
        self.g = None if g_init is None else np.asarray(g_init, dtype=np.float64).copy()

    def update(self, acc, gyro):
        a_mag = float(np.linalg.norm(acc))
        if self.g is None:
            if 0.8 * G < a_mag < 1.2 * G:
                self.g = np.asarray(acc, dtype=np.float64).copy()
            else:
                return np.array([0.0, 0.0, 1.0])
        elif 0.9 * G < a_mag < 1.1 * G and float(np.linalg.norm(gyro)) < 0.2:
            self.g += self.alpha * (acc - self.g)
        return self.g / np.linalg.norm(self.g)


def track_gravity(acc, gyro, tau_s=10.0, g_init=None):
    """Runs GravityTracker over a whole (N,3) sequence. Returns (N,3) unit vectors."""
    tracker = GravityTracker(tau_s=tau_s, g_init=g_init)
    out = np.empty_like(acc, dtype=np.float64)
    for i in range(len(acc)):
        out[i] = tracker.update(acc[i], gyro[i])
    return out


def feature_frame(acc, gyro, g_hat):
    """Feature channels for a single 10 Hz sample (or (N,3) arrays with per-row g_hat)."""
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    R = leveling_matrices(g_hat)
    if acc.ndim == 1:
        a_l, w_l = R @ acc, R @ gyro
    else:
        a_l = np.einsum("nij,nj->ni", R, acc)
        w_l = np.einsum("nij,nj->ni", R, gyro)
    a_mag = np.linalg.norm(acc, axis=-1) - G
    a_h_mag = np.hypot(a_l[..., 0], a_l[..., 1])
    w_mag = np.linalg.norm(gyro, axis=-1)
    return np.stack([a_l[..., 0], a_l[..., 1], a_l[..., 2] - G,
                     w_l[..., 0], w_l[..., 1], w_l[..., 2],
                     a_mag, a_h_mag, w_mag], axis=-1).astype(np.float32)


def sequence_features(acc, gyro, g_init=None):
    """Feature matrix (N, 9) for a whole 10 Hz trip, identical to running the streaming path."""
    g_hat = track_gravity(acc, gyro, g_init=g_init)
    return feature_frame(acc, gyro, g_hat)


def build_scalars(v_ref, t_since_ref, ref_valid):
    """Scalar conditioning vector [v_ref (m/s), t_since_ref (s, clipped), ref_valid (0/1)]."""
    valid = 1.0 if ref_valid else 0.0
    return np.array([v_ref * valid, min(max(t_since_ref, 0.0), T_SINCE_REF_MAX) * valid, valid],
                    dtype=np.float32)


class Decimator:
    """
    Converts an IMU stream of any rate into 10 Hz samples by block-averaging on
    the timestamp grid (a boxcar anti-alias filter). Feeding 10 Hz input yields each
    sample unchanged.
    """

    def __init__(self, rate_hz=MODEL_RATE_HZ):
        self.period = 1.0 / rate_hz
        self.bin_end = None
        self.acc_sum = np.zeros(3)
        self.gyro_sum = np.zeros(3)
        self.count = 0

    def push(self, t, acc, gyro):
        """Returns (t_bin_end, acc_mean, gyro_mean) when a 10 Hz bin completes, else None."""
        if self.bin_end is None:
            self.bin_end = t + self.period * 0.5
        out = None
        if t >= self.bin_end and self.count > 0:
            out = (self.bin_end, self.acc_sum / self.count, self.gyro_sum / self.count)
            self.acc_sum = np.zeros(3)
            self.gyro_sum = np.zeros(3)
            self.count = 0
            while self.bin_end <= t:
                self.bin_end += self.period
        self.acc_sum += acc
        self.gyro_sum += gyro
        self.count += 1
        return out
