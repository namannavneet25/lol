"""
2D INS Extended Kalman Filter for smartphone/vehicle dead reckoning.

State x (6):  [p_e, p_n, psi, v, b_g, s_g]
  p_e, p_n  position in the local ENU plane (m)
  psi       vehicle heading, CCW from East (rad)  - see frames.py
  v         forward speed (m/s)
  b_g       bias of the yaw-rate gyro (rad/s)
  s_g       scale-factor error of the yaw-rate gyro (phone gyros read 1-8% low on IO-VNBD)

Non-Holonomic Constraint (NHC): velocity is modelled as v * [cos psi, sin psi], i.e. the
vehicle can only move along its heading - lateral and vertical velocity are exactly zero.
This is the "hard" form of the NHC pseudo-measurement; it cannot drift sideways.

Prediction : psi += (1 + s_g)(w_yaw - b_g) dt ; p += v [cos psi, sin psi] dt ; v random walk
Updates    : AI forward speed (R = sigma_v^2 from the network)   -> update_speed
             ZUPT + ZARU when the network says the car is stopped -> update_zupt
             GNSS position with chi-square gating                 -> update_gnss_position
             GNSS speed / course                                  -> update_gnss_velocity
             map-matching cross-track pseudo-measurement          -> update_cross_track
Covariance updates use the Joseph form for numerical robustness.
"""
import math

import numpy as np

from src.navigation.frames import wrap_pi

PE, PN, PSI, V, BG, SG = range(6)
CHI2_2DOF_999 = 13.82


class EKF2D:
    def __init__(self, init_e=0.0, init_n=0.0, init_psi=0.0, init_v=0.0, init_bg=0.0,
                 gyro_noise=0.005, speed_process_noise=1.0, bias_walk=3e-5):
        self.dim_x = 6
        self.x = np.array([init_e, init_n, init_psi, init_v, init_bg, 0.0], dtype=np.float64)
        self.P = np.diag([25.0, 25.0, 0.5 ** 2, 4.0, 0.01 ** 2, 0.05 ** 2])
        self.q_psi = gyro_noise ** 2          # (rad/s)^2 per s  -> heading random walk
        self.q_v = speed_process_noise ** 2   # (m/s^2)^2 per s  -> unmodelled acceleration
        self.q_bg = bias_walk ** 2            # (rad/s)^2 per s  -> gyro bias drift
        self.q_pos = 0.05 ** 2

    # ------------------------------------------------------------------ prediction
    def predict(self, dt, yaw_rate):
        if dt <= 0:
            return
        e, n, psi, v, bg, sg = self.x
        w = (1.0 + sg) * (yaw_rate - bg)
        c, s = math.cos(psi), math.sin(psi)
        self.x[PE] = e + v * c * dt
        self.x[PN] = n + v * s * dt
        self.x[PSI] = wrap_pi(psi + w * dt)
        F = np.eye(6)
        F[PE, PSI] = -v * s * dt
        F[PE, V] = c * dt
        F[PN, PSI] = v * c * dt
        F[PN, V] = s * dt
        F[PSI, BG] = -(1.0 + sg) * dt
        F[PSI, SG] = (yaw_rate - bg) * dt
        Q = np.diag([self.q_pos * dt, self.q_pos * dt, self.q_psi * dt, self.q_v * dt, self.q_bg * dt, 1e-8 * dt])
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------ generic update
    def _update(self, H, residual, R, gate=None):
        H = np.atleast_2d(H)
        residual = np.atleast_1d(np.asarray(residual, dtype=np.float64))
        R = np.atleast_2d(R)
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        nis = float(residual @ S_inv @ residual)
        if gate is not None and nis > gate:
            return nis, False
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ residual
        self.x[PSI] = wrap_pi(self.x[PSI])
        self.x[V] = max(self.x[V], 0.0)
        self.x[SG] = min(max(self.x[SG], -0.2), 0.2)
        I_KH = np.eye(self.dim_x) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return nis, True

    # ------------------------------------------------------------------ measurements
    def update_speed(self, v_meas, variance):
        H = np.zeros(6); H[V] = 1.0
        return self._update(H, v_meas - self.x[V], max(variance, 1e-4))

    def update_zupt(self, yaw_rate_meas, v_var=0.01, zaru_var=0.002 ** 2):
        """Zero-velocity update plus zero-angular-rate update: the raw gyro reading is pure bias."""
        self.update_speed(0.0, v_var)
        H = np.zeros(6); H[BG] = 1.0
        return self._update(H, yaw_rate_meas - self.x[BG], zaru_var)

    def update_gnss_position(self, e, n, std_m, gate=CHI2_2DOF_999):
        H = np.zeros((2, 6)); H[0, PE] = 1.0; H[1, PN] = 1.0
        r = np.array([e - self.x[PE], n - self.x[PN]])
        return self._update(H, r, np.eye(2) * max(std_m, 1.0) ** 2, gate)

    def gnss_position_nis(self, e, n, std_m):
        H = np.zeros((2, 6)); H[0, PE] = 1.0; H[1, PN] = 1.0
        r = np.array([e - self.x[PE], n - self.x[PN]])
        S = H @ self.P @ H.T + np.eye(2) * max(std_m, 1.0) ** 2
        return float(r @ np.linalg.inv(S) @ r)

    def update_gnss_velocity(self, speed, psi_course, speed_std=0.3, min_speed_for_course=2.0):
        self.update_speed(speed, speed_std ** 2)
        if speed > min_speed_for_course:
            course_std = 0.02 + 0.5 / speed        # rad; course gets noisier at low speed
            H = np.zeros(6); H[PSI] = 1.0
            self._update(H, wrap_pi(psi_course - self.x[PSI]), course_std ** 2, gate=16.0)

    def update_cross_track(self, road_e, road_n, road_psi, std_m):
        """Constrains the position component normal to the matched road (along-track is left free)."""
        nx, ny = -math.sin(road_psi), math.cos(road_psi)
        H = np.zeros(6); H[PE] = nx; H[PN] = ny
        r = nx * (road_e - self.x[PE]) + ny * (road_n - self.x[PN])
        return self._update(H, r, std_m ** 2, gate=16.0)

    def update_heading(self, psi_meas, std_rad):
        H = np.zeros(6); H[PSI] = 1.0
        return self._update(H, wrap_pi(psi_meas - self.x[PSI]), std_rad ** 2, gate=16.0)

    def reanchor(self, e, n, psi=None, v=None, pos_std=5.0):
        """Resets position (and optionally heading/speed) to a trusted GNSS fix after an outage."""
        self.x[PE], self.x[PN] = e, n
        self.P[PE, :] = self.P[:, PE] = 0.0
        self.P[PN, :] = self.P[:, PN] = 0.0
        self.P[PE, PE] = self.P[PN, PN] = pos_std ** 2
        if psi is not None:
            self.x[PSI] = wrap_pi(psi)
            self.P[PSI, :] = self.P[:, PSI] = 0.0
            self.P[PSI, PSI] = 0.1 ** 2
        if v is not None:
            self.x[V] = max(v, 0.0)
            self.P[V, :] = self.P[:, V] = 0.0
            self.P[V, V] = 0.5 ** 2

    # ------------------------------------------------------------------ accessors
    @property
    def position(self):
        return float(self.x[PE]), float(self.x[PN])

    @property
    def speed(self):
        return float(self.x[V])

    @property
    def psi(self):
        return float(self.x[PSI])

    @property
    def position_std(self):
        return float(math.sqrt(max(self.P[PE, PE] + self.P[PN, PN], 0.0)))
