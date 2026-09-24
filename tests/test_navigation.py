"""Unit tests for frames, EKF, GNSS state machine, map matcher and feature extraction."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.navigation.frames import LocalTangentPlane, compass_to_psi, psi_to_compass, wrap_pi
from src.navigation.gnss_quality import GNSSDecision, GNSSQualityMonitor, NavigationStateMode
from src.navigation.ins_ekf import EKF2D
from src.navigation.map_matching import HMMMapMatcher, RoadNetwork
from src.preprocessing.features import Decimator, G, feature_frame, leveling_matrices, sequence_features
from src.preprocessing.parse_trip import _xcorr_lag


# ------------------------------------------------------------------ frames
@pytest.mark.parametrize("course,psi", [(0, math.pi / 2), (90, 0.0), (180, -math.pi / 2), (270, math.pi)])
def test_compass_psi_conventions(course, psi):
    assert abs(wrap_pi(compass_to_psi(course) - psi)) < 1e-9
    assert abs(((psi_to_compass(compass_to_psi(course)) - course + 180) % 360) - 180) < 1e-9


def test_local_tangent_plane_roundtrip():
    ltp = LocalTangentPlane(52.40, -1.50)
    e, n = ltp.to_enu(52.45, -1.40)            # ~7 km east, ~5.6 km north
    assert 6700 < e < 6900 and 5500 < n < 5600
    lat, lon = ltp.to_geodetic(e, n)
    assert abs(lat - 52.45) < 1e-7 and abs(lon + 1.40) < 1e-7


# ------------------------------------------------------------------ features
def test_leveling_removes_gravity_for_tilted_phone():
    tilt = math.radians(35)
    g_phone = np.array([0.0, math.sin(tilt), math.cos(tilt)]) * G
    R = leveling_matrices(g_phone / G)
    assert np.allclose(R @ g_phone, [0, 0, G], atol=1e-9)
    # a pure rotation about gravity must appear only on the yaw channel
    f = feature_frame(g_phone, 0.3 * g_phone / G, g_phone / G)
    assert abs(f[5] - 0.3) < 1e-6 and abs(f[3]) < 1e-6 and abs(f[4]) < 1e-6
    assert abs(f[0]) < 1e-5 and abs(f[1]) < 1e-5 and abs(f[2]) < 1e-5


def test_sequence_features_match_streaming():
    rng = np.random.default_rng(0)
    acc = np.array([0.3, 0.2, G]) + 0.3 * rng.standard_normal((200, 3))
    gyro = 0.02 * rng.standard_normal((200, 3))
    batch = sequence_features(acc, gyro)
    from src.preprocessing.features import GravityTracker
    tr = GravityTracker()
    stream = np.array([feature_frame(acc[i], gyro[i], tr.update(acc[i], gyro[i])) for i in range(200)])
    assert np.allclose(batch, stream, atol=1e-5)


def test_decimator_200hz_to_10hz():
    dec = Decimator(10.0)
    outs = []
    for k in range(2001):                      # 10 s at 200 Hz
        o = dec.push(k / 200.0, np.array([1.0, 0, 0]) * k, np.zeros(3))
        if o is not None:
            outs.append(o)
    assert 99 <= len(outs) <= 100
    assert all(abs(b[0] - a[0] - 0.1) < 1e-9 for a, b in zip(outs, outs[1:]))


def test_lag_estimation_recovers_shift():
    rng = np.random.default_rng(1)
    base = np.convolve(rng.standard_normal(3000), np.ones(10) / 10, "same")
    lag, r = _xcorr_lag(base[:-7], base[7:], 20)
    assert lag == -7 and r > 0.99


# ------------------------------------------------------------------ EKF
def _drive(ekf, seconds, speed, yaw_rate, gyro_bias=0.0, dt=0.1, ai_std=0.3):
    for _ in range(int(seconds / dt)):
        ekf.predict(dt, yaw_rate + gyro_bias)
        ekf.update_speed(speed, ai_std ** 2)


def test_ekf_straight_line_dead_reckoning():
    ekf = EKF2D(init_psi=0.0, init_v=15.0)
    _drive(ekf, 60, 15.0, 0.0)
    e, n = ekf.position
    assert abs(e - 900.0) < 5.0 and abs(n) < 1.0


def test_ekf_positive_yaw_rate_turns_left():
    """A positive (counter-clockwise, z-up) yaw rate must turn a north-bound car towards the west."""
    ekf = EKF2D(init_psi=math.pi / 2, init_v=10.0)        # heading north
    _drive(ekf, 9.0, 10.0, math.radians(10.0))             # +10 deg/s for 9 s -> +90 deg CCW
    assert abs(psi_to_compass(ekf.psi) - 270.0) < 3.0     # compass: now heading west
    e, n = ekf.position
    assert e < -40.0 and n > 40.0                          # quarter circle up and to the left (radius ~57 m)


def test_ekf_north_heading_uses_compass_convention():
    ekf = EKF2D(init_psi=math.pi / 2, init_v=10.0)        # heading north
    _drive(ekf, 10, 10.0, 0.0)
    assert abs(psi_to_compass(ekf.psi) - 0.0) < 1e-6 or abs(psi_to_compass(ekf.psi) - 360.0) < 1e-6
    e, n = ekf.position
    assert abs(n - 100.0) < 2.0 and abs(e) < 1.0


def test_zaru_estimates_gyro_bias_when_stopped():
    ekf = EKF2D()
    for _ in range(300):
        ekf.predict(0.1, 0.01)
        ekf.update_zupt(0.01)
    assert abs(ekf.x[4] - 0.01) < 1e-3
    assert ekf.speed < 0.05


def test_gnss_course_update_bounds_heading():
    ekf = EKF2D(init_psi=0.3, init_v=15.0)
    for k in range(30):
        ekf.predict(0.1, 0.0)
        if k % 10 == 0:
            ekf.update_gnss_velocity(15.0, 0.0)
    assert abs(ekf.psi) < 0.1


def test_cross_track_update_only_moves_normal_component():
    ekf = EKF2D(init_e=10.0, init_n=5.0)
    ekf.P[:2, :2] = np.eye(2) * 100.0
    ekf.update_cross_track(0.0, 0.0, 0.0, std_m=0.5)       # road along East through origin
    e, n = ekf.position
    assert abs(n) < 0.5 and abs(e - 10.0) < 1e-6


# ------------------------------------------------------------------ GNSS state machine
def test_mode_switches_to_dead_reckoning_after_timeout_and_reanchors():
    mon = GNSSQualityMonitor()
    t = 0.0
    assert mon.on_fix(t, 0, 0, 10, 3.0, 0.0) == GNSSDecision.REANCHOR
    for k in range(1, 5):
        t = float(k)
        assert mon.on_fix(t, 10 * k, 0, 10, 3.0, 1.0) == GNSSDecision.FUSE
        mon.tick(t)
    mon.tick(t + 1.4)
    assert mon.mode == NavigationStateMode.GNSS_AIDED
    mon.tick(t + 1.6)
    assert mon.mode == NavigationStateMode.DEAD_RECKONING
    # GNSS returns, but the filter drifted 40 m: first fix pending, second consistent fix re-anchors
    assert mon.on_fix(t + 30, 340, 40, 10, 3.0, 200.0) == GNSSDecision.REJECT
    assert mon.mode == NavigationStateMode.REACQUISITION
    assert mon.on_fix(t + 31, 350, 40, 10, 3.0, 200.0) == GNSSDecision.REANCHOR
    assert mon.mode == NavigationStateMode.GNSS_AIDED


def _feed(mon, t0, t1, dt=0.1):
    """Advance the monitor's clock frame by frame (as the pipeline does) from t0 to t1."""
    t = t0
    while t < t1 - 1e-9:
        t += dt
        mon.tick(t)


def test_sparse_fixes_reanchor_after_outage():
    """A receiver with one fix every ~9 s must still re-anchor after dead reckoning (reacquisition trap)."""
    mon = GNSSQualityMonitor()
    t = 0.0
    for k in range(8):                                   # GNSS-aided, 9 s fixes, car at 10 m/s
        t = 9.0 * k
        mon.on_fix(t, 90 * k, 0, 10, 3.0, 1.0, filter_speed=10.0)
        _feed(mon, t, t + 9.0 if k < 7 else t)
    _feed(mon, t, t + 60)                                # outage: dead reckoning
    assert mon.mode == NavigationStateMode.DEAD_RECKONING
    t += 60
    assert mon.on_fix(t, 1300, 50, 2, 3.0, 400.0, filter_speed=10.0) == GNSSDecision.REJECT   # drifted: pending
    _feed(mon, t, t + 9)
    assert mon.on_fix(t + 9, 1390, 50, 2, 3.0, 400.0, filter_speed=10.0) == GNSSDecision.REANCHOR
    assert mon.mode == NavigationStateMode.GNSS_AIDED


def test_sparse_single_outlier_fix_is_not_adopted():
    """After an outage, one far-off fix (400 m in 9 s at 10 m/s) must not re-anchor the filter."""
    mon = GNSSQualityMonitor()
    for k in range(8):
        mon.on_fix(9.0 * k, 90 * k, 0, 10, 3.0, 1.0, filter_speed=10.0)
        _feed(mon, 9.0 * k, 9.0 * k + 9.0)
    t = 9.0 * 8 + 60
    _feed(mon, 9.0 * 8, t)
    assert mon.on_fix(t, 1300, 0, 10, 3.0, 400.0, filter_speed=10.0) == GNSSDecision.REJECT
    _feed(mon, t, t + 9)
    assert mon.on_fix(t + 9, 1390, 400, 10, 20.0, 400.0, filter_speed=10.0) == GNSSDecision.REJECT   # outlier
    assert mon.mode == NavigationStateMode.REACQUISITION


def test_loss_detection_fast_after_short_reacquisition_at_1hz():
    """
    At 1 Hz, a second outage 3 s after reacquiring is detected within ~2.3 s of the last fix (the original
    behaviour). It must not regress to the ~4.2 s that a 15 s gap cap for all receivers caused.
    """
    mon = GNSSQualityMonitor()
    t = 0.0
    for k in range(20):
        t = float(k)
        mon.on_fix(t, 10 * k, 0, 10, 3.0, 1.0)
        _feed(mon, t, t + 1.0 if k < 19 else t)
    _feed(mon, t, t + 30)                                # 30 s outage
    t += 30
    for k in range(3):                                   # GNSS back for only 3 s
        mon.on_fix(t + k, 0, 0, 10, 3.0, 1.0)
        _feed(mon, t + k, t + k + (1.0 if k < 2 else 0.0))
    last = t + 2
    t_lost = last
    while mon.mode != NavigationStateMode.DEAD_RECKONING:
        t_lost += 0.1
        mon.tick(t_lost)
    assert t_lost - last <= 2.4


def test_very_sparse_receiver_can_reanchor():
    """Fixes 30 s apart (beyond the old ~22 s pending expiry) can still form a re-anchor pair."""
    mon = GNSSQualityMonitor()
    mon.on_fix(0.0, 0, 0, 10, 3.0, 0.0)
    _feed(mon, 0.0, 100.0)
    assert mon.on_fix(100.0, 1000, 0, 10, 3.0, 400.0, filter_speed=10.0) == GNSSDecision.REJECT
    _feed(mon, 100.0, 130.0)
    assert mon.on_fix(130.0, 1300, 0, 10, 3.0, 400.0, filter_speed=10.0) == GNSSDecision.REANCHOR


def test_outlier_fix_rejected_in_aided_mode():
    mon = GNSSQualityMonitor()
    mon.on_fix(0.0, 0, 0, 10, 3.0, 0.0)
    assert mon.on_fix(1.0, 10, 0, 10, 3.0, 500.0) == GNSSDecision.REJECT
    assert mon.mode == NavigationStateMode.GNSS_AIDED


# ------------------------------------------------------------------ map matching
def _t_junction():
    # main road north (x=0) up to y=200, branch east at y=100, parallel road at x=25
    segs = [((0, 0), (0, 100)), ((0, 100), (0, 200)), ((0, 100), (150, 100)), ((25, 0), (25, 200))]
    return RoadNetwork.from_segments(segs)


def test_hmm_prefers_road_consistent_with_heading_and_history():
    m = HMMMapMatcher(_t_junction())
    res = None
    for y in range(10, 90, 10):                          # driving north, drifting 9 m east
        res = m.step(9.0, float(y), math.pi / 2, 10.0, 10.0, 5.0)
    assert res.segment == 0 and abs(res.e) < 1e-6          # main road, not the parallel one at x=25
    for x in range(10, 80, 10):                          # turn east onto the branch
        res = m.step(float(x), 104.0, 0.0, 10.0, 10.0, 5.0)
    assert res.segment == 2 and abs(res.n - 100.0) < 1e-6 and res.confidence > 0.6


def test_hmm_respects_oneway_direction():
    net = RoadNetwork.from_segments([((0, 0), (0, 100), 1), ((6, 100), (6, 0), 1)])   # dual carriageway
    m = HMMMapMatcher(net)
    res = None
    for y in range(10, 90, 10):
        res = m.step(4.0, float(y), math.pi / 2, 10.0, 10.0, 5.0)                   # northbound, closer to x=6
    assert res.segment == 0


def test_hmm_without_map_returns_none():
    assert HMMMapMatcher().step(0, 0, 0, 0, 0, 1) is None
