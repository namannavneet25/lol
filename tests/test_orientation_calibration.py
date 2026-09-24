#!/usr/bin/env python3
"""
Unit test & verification script for PhoneOrientationEstimator and calibration integration.
Tests:
1. Resting Gravity Separation (Total |a| = 9.81, |a_horiz| = 0 across arbitrary 3D tilt).
2. Tilt-Immune Yaw Rate Extraction (wobble immunity & true vertical projection).
3. Calibration JSON export & pipeline loading.
"""

import math
import json
import numpy as np
import tempfile
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.navigation.orientation import PhoneOrientationEstimator
from src.navigation.pipeline import CoreNavigationPipeline


def test_gravity_separation():
    print("\n--- Test 1: Gravity Separation Across Tilts ---")
    estimator = PhoneOrientationEstimator(alpha_gravity=0.1)

    # Simulate phone tilted at 30 deg pitch, 45 deg roll
    pitch = math.radians(30.0)
    roll = math.radians(45.0)

    # Unit gravity vector in phone frame
    gx = math.sin(pitch)
    gy = math.sin(roll) * math.cos(pitch)
    gz = math.cos(roll) * math.cos(pitch)

    # Scale by 9.81 m/s^2
    ax, ay, az = 9.81 * gx, 9.81 * gy, 9.81 * gz

    # Initialize estimator with 10 samples
    for _ in range(10):
        yaw_rate, _, _, a_horiz, est_pitch, est_roll = estimator.update(ax, ay, az, 0.0, 0.0, 0.0)

    print(f"True Pitch: 30.0°, Estimated Pitch: {est_pitch:.2f}°")
    print(f"True Roll:  45.0°, Estimated Roll:  {est_roll:.2f}°")
    print(f"Total |a|:  {math.sqrt(ax**2 + ay**2 + az**2):.2f} m/s²")
    print(f"Horiz |a|:  {a_horiz:.4f} m/s² (Target: ~0 m/s²)")

    assert abs(est_pitch - 30.0) < 1.0, f"Pitch error too high: {est_pitch}"
    assert abs(est_roll - 45.0) < 1.0, f"Roll error too high: {est_roll}"
    assert a_horiz < 0.01, f"Gravity removal failed, residual horiz accel: {a_horiz}"
    print("✅ Test 1 PASSED: Gravity cleanly isolated and removed!")


def test_tilt_immune_yaw():
    print("\n--- Test 2: Tilt-Immune Vertical Yaw Rate ---")
    estimator = PhoneOrientationEstimator(alpha_gravity=0.1)

    # Phone tilted at 45 degrees roll
    pitch = 0.0
    roll = math.radians(45.0)
    gx = 0.0
    gy = math.sin(roll)
    gz = math.cos(roll)

    ax, ay, az = 0.0, 9.81 * gy, 9.81 * gz

    # Prime gravity tracker
    for _ in range(10):
        estimator.update(ax, ay, az, 0.0, 0.0, 0.0)

    # Case A: Vehicle turns at 0.5 rad/s (28.6 deg/s) around TRUE VERTICAL
    # In phone frame, omega = 0.5 * [gx, gy, gz]
    turn_rate = 0.5
    omega_x = turn_rate * gx
    omega_y = turn_rate * gy
    omega_z = turn_rate * gz

    yaw_rate = estimator.update(ax, ay, az, omega_x, omega_y, omega_z)[0]
    print(f"True Turn Rate: {turn_rate:.4f} rad/s")
    print(f"Raw Phone Gz:   {omega_z:.4f} rad/s ({math.degrees(omega_z):.2f}°/s)")
    print(f"True Yaw Rate:  {yaw_rate:.4f} rad/s ({math.degrees(yaw_rate):.2f}°/s)")

    assert abs(yaw_rate - turn_rate) < 0.01, f"Yaw rate error: {yaw_rate} vs {turn_rate}"

    # Case B: Phone wobbles (tilt oscillates) without any vehicle turn
    # Angular velocity is perpendicular to gravity (pure tilt wobble)
    wobble_x = 0.8
    wobble_y = 0.0
    wobble_z = 0.0  # pure pitch wobble
    yaw_rate_wobble = estimator.update(ax, ay, az, wobble_x, wobble_y, wobble_z)[0]
    print(f"Wobble Raw Gx:  {wobble_x:.2f} rad/s, True Yaw Rate: {yaw_rate_wobble:.4f} rad/s")
    assert abs(yaw_rate_wobble) < 0.01, f"Wobble leaked into yaw rate: {yaw_rate_wobble}"
    print("✅ Test 2 PASSED: Tilt-immune yaw rate successfully verified!")


def test_calibration_pipeline_integration():
    print("\n--- Test 3: Calibration JSON Pipeline Integration ---")
    sample_calibration = {
        "timestamp": "2026-09-21T18:00:00.000Z",
        "gyro_bias_rad_s": {
            "gx": 0.005,
            "gy": -0.002,
            "gz": 0.015
        },
        "resting_gravity_vector": {
            "x": 0.1,
            "y": 6.9,
            "z": 6.9,
            "magnitude": 9.81
        },
        "mounting_yaw_offset_deg": 12.5,
        "is_calibrated": True,
        "is_mounting_aligned": True
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(sample_calibration, tf)
        temp_cal_path = tf.name

    try:
        pipeline = CoreNavigationPipeline(calibration_path=temp_cal_path)
        g = np.array([0.1, 6.9, 6.9])
        bias = np.array([0.005, -0.002, 0.015])
        expected_bg = float(g @ bias / np.linalg.norm(g))
        # the EKF yaw-gyro bias state (x[4]) is the phone gyro bias projected on the vertical
        assert abs(pipeline.ekf.x[4] - expected_bg) < 1e-6
        assert np.allclose(pipeline.gravity.g, g)
        print("✅ Test 3 PASSED: Calibration JSON loaded into the EKF bias state and gravity tracker!")
    finally:
        Path(temp_cal_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_gravity_separation()
    test_tilt_immune_yaw()
    test_calibration_pipeline_integration()
    print("\n🎉 ALL ORIENTATION & CALIBRATION TESTS PASSED!")
