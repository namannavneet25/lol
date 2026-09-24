"""
Core IDR navigation pipeline: SensorFrame in, NavigationState out, at the IMU rate.

  IMU (any rate) --native rate--> gravity-levelled yaw rate --> EKF prediction
        |
        +--> Decimator (block mean) --10 Hz--> feature buffer --> Tiny TCN --> AI speed / ZUPT update
  GNSS fix --> quality monitor / state machine --> fuse | reject | re-anchor
  every 0.5 s --> HMM map matcher --> cross-track (+ heading) pseudo-measurement while dead reckoning

The same engine serves the phone (10 Hz) and external IMUs (e.g. 200 Hz FOG units): prediction
runs at the native rate, the network always sees 10 Hz features.

SensorFrame  : {"timestamp": s, "ax","ay","az" (m/s^2, incl. gravity), "gx","gy","gz" (rad/s),
                "gnss": None | {"lat","lon","speed" (m/s),"heading" (compass deg),"accuracy" (m)}}
NavigationState : see _nav_state().
"""
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.inference import SpeedModel, ai_confidence
from src.navigation.frames import LocalTangentPlane, compass_to_psi, psi_to_compass
from src.navigation.gnss_quality import GNSSDecision, GNSSQualityMonitor, NavigationStateMode
from src.navigation.ins_ekf import EKF2D
from src.navigation.map_matching import HMMMapMatcher, RoadNetwork
from src.preprocessing.features import (MODEL_RATE_HZ, T_SINCE_REF_MAX, Decimator, GravityTracker,
                                        feature_frame)

DR_MODES = (NavigationStateMode.DEAD_RECKONING, NavigationStateMode.REACQUISITION)


class CoreNavigationPipeline:
    def __init__(self, model_weights_path="models/tiny_tcn_best.pth",
                 norm_json_path="data/processed/normalization.json", calibration_path="calibration.json",
                 map_path=None, origin=None, use_map=True, ai_variance_inflation=4.0, zupt_threshold=0.8,
                 map_update_period_s=0.5, map_min_confidence=0.4, map_heading_update=True, use_ai=True,
                 model=None):
        self.model = model or SpeedModel(model_weights_path, norm_json_path)
        self.window = self.model.window
        self.use_ai = use_ai
        self.ekf = EKF2D()
        self.gnss_monitor = GNSSQualityMonitor()
        self.gravity = GravityTracker()
        self.decimator = Decimator(MODEL_RATE_HZ)
        self.buffer = collections.deque(maxlen=self.window)
        self.ai_variance_inflation = ai_variance_inflation
        self.zupt_threshold = zupt_threshold
        self.map_period = map_update_period_s
        self.map_min_conf = map_min_confidence
        self.map_heading_update = map_heading_update
        self.use_map = use_map
        self.map_path = map_path
        self.matcher = HMMMapMatcher()
        self.ltp = None
        if origin is not None:
            self._set_origin(*origin)

        self.gyro_bias_phone = np.zeros(3)
        if calibration_path and os.path.exists(calibration_path):
            self._load_calibration(calibration_path)

        self.last_t = None
        self.v_ref, self.t_ref, self.has_ref = 0.0, None, False
        self.last_ai = None
        self.last_match = None
        self.travelled_since_match = 0.0
        self.last_map_t = None
        self.bin_yaw = []

    # ---------------------------------------------------------------- setup
    def _load_calibration(self, path):
        with open(path) as f:
            cal = json.load(f)
        if not cal.get("is_calibrated", False):
            return
        b = cal.get("gyro_bias_rad_s", {})
        self.gyro_bias_phone = np.array([b.get("gx", 0.0), b.get("gy", 0.0), b.get("gz", 0.0)])
        g = cal.get("resting_gravity_vector")
        if g:
            self.gravity = GravityTracker(g_init=[g.get("x", 0.0), g.get("y", 0.0), g.get("z", 9.81)])
            g_hat = self.gravity.g / np.linalg.norm(self.gravity.g)
            # the yaw-gyro bias seen by the EKF is the phone bias projected on the vertical
            self.ekf.x[4] = float(g_hat @ self.gyro_bias_phone)

    def _set_origin(self, lat, lon):
        self.ltp = LocalTangentPlane(lat, lon)
        if self.use_map and self.map_path and os.path.exists(self.map_path):
            self.matcher = HMMMapMatcher(RoadNetwork.from_geojson(self.map_path, self.ltp))

    def load_map(self, geojson):
        """Loads a road network (path or GeoJSON dict); requires the origin to be set."""
        self.matcher = HMMMapMatcher(RoadNetwork.from_geojson(geojson, self.ltp))

    # ---------------------------------------------------------------- main entry
    def process_frame(self, frame):
        t = float(frame["timestamp"])
        acc = np.array([frame.get("ax", 0.0), frame.get("ay", 0.0), frame.get("az", 9.81)], dtype=np.float64)
        gyro = np.array([frame.get("gx", 0.0), frame.get("gy", 0.0), frame.get("gz", 0.0)], dtype=np.float64)

        g_hat = self.gravity.g / np.linalg.norm(self.gravity.g) if self.gravity.g is not None \
            else np.array([0.0, 0.0, 1.0])
        yaw_rate = float(g_hat @ gyro)

        # 1. INS prediction at the native IMU rate
        if self.last_t is not None:
            dt = t - self.last_t
            if 0.0 < dt < 1.0:
                self.ekf.predict(dt, yaw_rate)
                self.travelled_since_match += self.ekf.speed * dt
        self.last_t = t
        self.bin_yaw.append(yaw_rate)

        # 2. 10 Hz model branch
        sample = self.decimator.push(t, acc, gyro)
        if sample is not None:
            self._model_step(sample)

        # 3. GNSS + state machine
        fix = frame.get("gnss")
        if fix is not None and "lat" in fix:
            self._gnss_step(t, fix)
        mode = self.gnss_monitor.tick(t)

        # 4. map matching (~1 Hz); feedback only while dead reckoning
        if self.ltp is not None and self.matcher.has_map and \
                (self.last_map_t is None or t - self.last_map_t >= self.map_period):
            self._map_step(mode)
            self.last_map_t = t

        return self._nav_state(t, mode)

    # ---------------------------------------------------------------- steps
    def _model_step(self, sample):
        _, acc10, gyro10 = sample
        g_hat = self.gravity.update(acc10, gyro10)
        self.buffer.append(feature_frame(acc10, gyro10, g_hat))
        raw_yaw = float(np.mean(self.bin_yaw)) if self.bin_yaw else 0.0
        self.bin_yaw = []
        if len(self.buffer) < self.window or not self.use_ai:
            return
        t_since = (self.last_t - self.t_ref) if self.has_ref else 0.0
        ref_valid = self.has_ref and t_since <= T_SINCE_REF_MAX
        scalars = [self.v_ref if ref_valid else 0.0, t_since if ref_valid else 0.0, 1.0 if ref_valid else 0.0]
        out = self.model.predict(np.asarray(self.buffer, dtype=np.float32), scalars)
        self.last_ai = out
        if out.p_stationary > self.zupt_threshold and out.speed < 0.5:
            self.ekf.update_zupt(raw_yaw)
        else:
            self.ekf.update_speed(out.speed, out.variance * self.ai_variance_inflation)

    def _gnss_step(self, t, fix):
        if self.ltp is None:
            self._set_origin(fix["lat"], fix["lon"])
        e, n = self.ltp.to_enu(fix["lat"], fix["lon"])
        acc_m = fix.get("accuracy")
        acc_m = 10.0 if acc_m is None else float(acc_m)      # receivers may omit accuracy
        speed = fix.get("speed")
        course = fix.get("heading")
        nis = self.ekf.gnss_position_nis(e, n, acc_m)
        decision = self.gnss_monitor.on_fix(t, e, n, speed or 0.0, acc_m, nis, filter_speed=self.ekf.speed)
        psi = compass_to_psi(course) if (course is not None and speed is not None and speed > 2.0) else None
        if decision == GNSSDecision.REANCHOR:
            self.ekf.reanchor(e, n, psi=psi, v=speed)
            self.matcher.reset()
        elif decision in (GNSSDecision.FUSE, GNSSDecision.FUSE_INFLATED):
            std = acc_m * (2.0 if decision == GNSSDecision.FUSE_INFLATED else 1.0)
            self.ekf.update_gnss_position(e, n, std, gate=None)
            if speed is not None:
                self.ekf.update_gnss_velocity(speed, psi if psi is not None else self.ekf.psi)
        if decision in (GNSSDecision.REANCHOR, GNSSDecision.FUSE, GNSSDecision.FUSE_INFLATED) and speed is not None:
            self.v_ref, self.t_ref, self.has_ref = float(speed), t, True

    def _map_step(self, mode):
        e, n = self.ekf.position
        match = self.matcher.step(e, n, self.ekf.psi, self.ekf.speed, self.travelled_since_match,
                                  self.ekf.position_std)
        self.travelled_since_match = 0.0
        self.last_match = match
        if match is None or mode not in DR_MODES or match.confidence < self.map_min_conf:
            return
        self.ekf.update_cross_track(match.e, match.n, match.road_psi, std_m=3.0 / match.confidence)
        if self.map_heading_update and self.ekf.speed > 3.0 and match.confidence > 0.8:
            self.ekf.update_heading(match.road_psi, math.radians(6.0) / match.confidence)

    # ---------------------------------------------------------------- output
    def _nav_state(self, t, mode):
        e, n = self.ekf.position
        lat, lon = self.ltp.to_geodetic(e, n) if self.ltp is not None else (None, None)
        match = self.last_match
        return {
            "timestamp": int(round(t * 1000)),
            "latitude": None if lat is None else round(lat, 7),
            "longitude": None if lon is None else round(lon, 7),
            "east": round(e, 2), "north": round(n, 2),
            "speed": round(self.ekf.speed * 3.6, 2),
            "heading": round(psi_to_compass(self.ekf.psi), 1),
            "mode": mode,
            "gnssQuality": round(self.gnss_monitor.gnss_quality, 2),
            "aiSpeed": None if self.last_ai is None else round(self.last_ai.speed * 3.6, 2),
            "aiConfidence": 0.0 if self.last_ai is None else round(ai_confidence(self.last_ai), 2),
            "mapConfidence": 0.0 if match is None else round(match.confidence, 2),
            "positionStdM": round(self.ekf.position_std, 2),
        }
