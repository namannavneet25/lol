# Intelligent Dead Reckoning (IDR) — Master Technical Documentation (`doc3.md`)

> **⚠️ Superseded (2026-09-23).** This document describes the pre-optimisation system. Its accuracy figures (e.g. 0.98 m/s validation MAE, the drift tables) could not be reproduced: the shipped checkpoint scored 5.17 m/s test MAE on its own split, and no position error was ever measured. See `docs/SUMMARY.md` for an overview, `docs/OPTIMIZATION_PLAN.md` and `docs/CHANGES.md` for what changed and why, `docs/RESULTS.md` for measured results, and `AGENTS.md` for the current contracts and commands.

This master technical document consolidates the complete system architecture, multi-driver dataset pipeline, Model #4 (Tiny TCN) specifications, sensor fusion algorithms, real-world edge fixes (orientation alignment, reacquisition trap resolution), and execution commands for **Intelligent Dead Reckoning (SIH 2026 / PS26168)**.

---

## 📑 Table of Contents
1. [System Architecture & Core Principles](#1-system-architecture--core-principles)
2. [End-to-End Execution & Retraining Commands](#2-end-to-end-execution--retraining-commands)
3. [Multi-Driver Dataset & Preprocessing Pipeline](#3-multi-driver-dataset--preprocessing-pipeline)
4. [Model #4: Tiny TCN Specifications](#4-model-4-tiny-tcn-specifications)
5. [Sensor Fusion & Navigation Engine (EKF2D + NHC)](#5-sensor-fusion--navigation-engine-ekf2d--nhc)
6. [Real-World Edge Problem Solvers](#6-real-world-edge-problem-solvers)
   - 6.1 [Auto-Gravity & Tilt-Immune Orientation Engine](#61-auto-gravity--tilt-immune-orientation-engine)
   - 6.2 [The Reacquisition Trap & State Re-Anchoring](#62-the-reacquisition-trap--state-re-anchoring)
   - 6.3 [Vibration Damping for Unmounted Devices (e.g., Bicycles)](#63-vibration-damping-for-unmounted-devices-eg-bicycles)
7. [I/O Schemas: Runtime Contracts](#7-io-schemas-runtime-contracts)
8. [Hardware Roadmap: External 200 Hz IoT Sensor Pod](#8-hardware-roadmap-external-200-hz-iot-sensor-pod)

---

## 1. System Architecture & Core Principles

The core design principle follows modern learned inertial odometry (RoNIN / TLIO):
> **AI estimates what classical physics is bad at (forward speed & uncertainty from raw IMU vibrations); classical physics & filtering determine how that information becomes position.**

```
 ┌────────────────────────────────────────────────────────┐
 │                   SMARTPHONE SENSORS                   │
 │       Accelerometer (ax, ay, az) & Gyroscope (gx, gy, gz) @ 10 Hz       │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           ORIENTATION & GRAVITY ALIGNMENT              │
 │   • Tracks gravity vector g_hat via low-pass filter    │
 │   • Extracts true vertical yaw rate: omega_yaw = w · g │
 │   • Removes 1g gravity component -> a_horiz            │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │          TINY TEMPORAL CONVOLUTIONAL NETWORK           │
 │               Input: (Batch, 9, 20) — 2.0s             │
 │   ┌───────────────────────┼────────────────────────┐   │
 │   ▼                       ▼                        ▼   │
 │ Forward Speed v_f    Uncertainty σ_v^2       Motion Class │
 └───────────┬───────────────────────┬────────────────────┘
             │                       │
             ▼                       ▼
 ┌────────────────────────────────────────────────────────┐
 │                2D EXTENDED KALMAN FILTER               │
 │   State: [p_x, p_y, v_x, v_y, psi, b_g, b_ax, b_ay]^T  │
 │   • INS Prediction with tilt-immune yaw rate           │
 │   • AI velocity update weighted by adaptive σ_v^2      │
 │   • Non-Holonomic Constraint (NHC: v_lateral ≈ 0)      │
 │   • GNSS Course heading update (when moving > 1.2 m/s) │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │             STATE MACHINE & MAP MATCHING               │
 │   • Modes: GNSS_AIDED -> DEGRADED -> DEAD_RECKONING    │
 │   • Instant Re-Anchoring on GPS recovery               │
 │   • OpenStreetMap road centerline candidate snapping   │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │             STANDARDIZED NAVIGATION STATE              │
 │   { timestamp, lat, lon, speed, heading, mode, ... }   │
 └────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Execution & Retraining Commands

Run all commands from the repository root ([`/Users/shudhanshuranjangupta/.gemini/antigravity/scratch/PS26168`](file:///Users/shudhanshuranjangupta/.gemini/antigravity/scratch/PS26168)):

```bash
# Step 1: Download raw multi-driver trip CSVs (8 trips across Drivers A, B, D, E)
python3 src/preprocessing/download_trip.py

# Step 2: Extract 2-second rolling windows (94,775 windows @ 10 Hz) & calculate train statistics
python3 src/preprocessing/create_windows.py

# Step 3: Train Tiny TCN model with Cosine Annealing scheduler & best-checkpoint saving
python3 src/models/tiny_tcn.py --train --epochs 30 --lr 0.001

# Step 4: Evaluate the best trained checkpoint on the held-out test set
python3 src/models/evaluate_models.py

# Step 5: Export PyTorch checkpoint to lightweight, dynamic-batch ONNX
python3 src/models/export_onnx.py

# Step 6: Run full end-to-end replay test simulating a GNSS blackout tunnel
python3 src/navigation/replay.py
```

---

## 3. Multi-Driver Dataset & Preprocessing Pipeline

### **3.1 9 Input Channels at 10 Hz**
Each 2.0-second window comprises **20 temporal frames** across **9 channels**:
1. `ax`: Longitudinal / Forward Acceleration ($\text{m/s}^2$)
2. `ay`: Lateral Acceleration ($\text{m/s}^2$)
3. `az`: Vertical Acceleration ($\text{m/s}^2$)
4. `gx`: Pitch Angular Velocity ($\text{rad/s}$)
5. `gy`: Roll Angular Velocity ($\text{rad/s}$)
6. `gz`: Yaw Angular Velocity ($\text{rad/s}$)
7. `accel_mag`: Total Acceleration Magnitude $\|a\| = \sqrt{a_x^2 + a_y^2 + a_z^2}$
8. `gyro_mag`: Total Angular Velocity Magnitude $\|\omega\| = \sqrt{g_x^2 + g_y^2 + g_z^2}$
9. `jerk`: Acceleration derivative $\frac{d\|a\|}{dt}$ ($\text{m/s}^3$)

### **3.2 Dataset Partitions**
Data is pooled from 8 distinct driving trips across 4 different drivers to ensure generalization:
- **Total Windows Extracted**: $94,775$ ($20 \times 9$ floats)
- **Train Set** (`train_windows.npz`): **$66,342$ windows** ($70\%$)
- **Validation Set** (`val_windows.npz`): **$14,216$ windows** ($15\%$)
- **Test Set** (`test_windows.npz`): **$14,217$ windows** ($15\%$)

> [!IMPORTANT]
> Normalization parameters ($\mu, \sigma$) stored in `normalization.json` are computed **strictly from the Train Set** to guarantee zero data leakage.

---

## 4. Model #4: Tiny TCN Specifications

### **4.1 Architecture Specs**
* **Model Class**: `TinyTCN` in [`src/models/tiny_tcn.py`](file:///Users/shudhanshuranjangupta/.gemini/antigravity/scratch/PS26168/src/models/tiny_tcn.py)
* **Total Parameters**: $45,223$ parameters
* **Exported Model Size**: **$62.75\text{ KB}$** (`idr_tcn.onnx`)
* **Layer Hierarchy**:
  1. `Conv1D`: $9 \to 32$ channels, kernel size $5$, padding $2$, ReLU
  2. `Dilated Conv1D`: $32 \to 32$ channels, kernel size $3$, dilation $2$, padding $2$ + Residual Skip Connection
  3. `Dilated Conv1D`: $32 \to 64$ channels, kernel size $3$, dilation $4$, padding $4$, ReLU
  4. `AdaptiveAvgPool1D(1)`: Global Average Pooling $\to (B, 64)$
  5. `Dense`: $64 \to 64$, ReLU
  6. **Multi-Task Heads**:
     - `speed_head`: Linear($64 \to 1$) $\to$ Forward velocity $v_f$ ($\text{m/s}$)
     - `uncertainty_head`: Linear($64 \to 1$) $\to$ Log variance $\log \sigma_v^2$
     - `motion_head`: Linear($64 \to 5$) $\to$ Motion class logits (`stationary`, `accelerating`, `cruising`, `braking`, `turning`)

### **4.2 Loss Function**
$$\mathcal{L} = \mathcal{L}_{\text{data}} + 0.1 \mathcal{L}_{\text{unc}} + 0.05 \mathcal{L}_{\text{smooth}} + 0.1 \mathcal{L}_{\text{motion}}$$
* $\mathcal{L}_{\text{data}} = \text{Huber}(v_f, v_{\text{GT}})$
* $\mathcal{L}_{\text{unc}} = \frac{(v_f - v_{\text{GT}})^2}{\exp(\log \sigma_v^2) + \epsilon} + \log \sigma_v^2$ (Heteroscedastic aleatoric loss)
* $\mathcal{L}_{\text{smooth}} = |v_{f, t+1} - v_{f, t}|$

### **4.3 Empirical Benchmark**
* **Validation Speed MAE**: **$0.98\text{ m/s}$ ($3.53\text{ km/h}$)**
* **Inference Latency**: $< 1.0\text{ ms}$ on CPU/Mobile Browser

---

## 5. Sensor Fusion & Navigation Engine (EKF2D + NHC)

### **5.1 State Vector (8 Dimensions)**
$$\mathbf{x} = \begin{bmatrix} p_x & p_y & v_x & v_y & \psi & b_g & b_{ax} & b_{ay} \end{bmatrix}^T$$
* $(p_x, p_y)$: 2D position in local East-North-Up (m)
* $(v_x, v_y)$: 2D velocity vector in navigation frame ($\text{m/s}$)
* $\psi$: Vehicle yaw / heading angle ($\text{rad}$)
* $b_g$: Gyroscope bias drift ($\text{rad/s}$)
* $(b_{ax}, b_{ay})$: Accelerometer biases ($\text{m/s}^2$)

### **5.2 Update Cycles**
1. **INS Propagation Step**: Integrates corrected accelerations and gravity-aligned yaw rate.
2. **AI Forward Velocity Update**: $z = v_f$, measurement variance $R = \sigma_v^2$.
3. **Non-Holonomic Constraint (NHC)**: Enforces $v_{\text{lateral}} = -v_x \sin\psi + v_y \cos\psi \approx 0$ with constraint covariance $R_{\text{NHC}} = 0.01$.
4. **GNSS Heading Fusion**: When vehicle speed $> 1.2\text{ m/s}$, GNSS bearing directly updates heading state $\psi$, bounding gyro drift.
5. **Map Matching Soft Correction**: Snaps local $(p_x, p_y)$ onto OpenStreetMap candidate segments minimizing composite score $S = w_d D + w_h H + w_t T$.

---

## 6. Real-World Edge Problem Solvers

### **6.1 Auto-Gravity & Tilt-Immune Orientation Engine**
* **Problem**: In handheld devices or bikes without mounts, the phone tilts arbitrarily. Raw gyro $g_z$ is contaminated by roll and pitch, causing open-loop yaw to spin uncontrollably.
* **Solution ([`orientation.py`](file:///Users/shudhanshuranjangupta/.gemini/antigravity/scratch/PS26168/src/navigation/orientation.py))**:
  1. Low-pass filters the 3D accelerometer to track the true gravity unit vector $\hat{\mathbf{g}} = \frac{\mathbf{a}}{\|\mathbf{a}\|}$.
  2. Extracts the true vertical yaw rate via projection:
     $$\omega_{\text{yaw}} = \boldsymbol{\omega} \cdot \hat{\mathbf{g}}$$
  3. Eliminates tilt bleed: even at $45^\circ$ phone tilt, pitch/roll wobbles are completely isolated from vehicle yaw.

### **6.2 The Reacquisition Trap & State Re-Anchoring**
* **Problem**: After dead reckoning through a tunnel, position drifts by $10\text{--}30\text{ m}$. When GNSS returns, the large residual creates high Normalized Innovation Squared ($NIS > 20$). Standard filters mistake this for a GPS anomaly, penalizing quality and staying trapped in `REACQUISITION` forever.
* **Solution ([`gnss_quality.py`](file:///Users/shudhanshuranjangupta/.gemini/antigravity/scratch/PS26168/src/navigation/gnss_quality.py))**:
  1. Distinguishes between an outage recovery and a temporary GPS glitch.
  2. Upon receiving 2 consecutive valid fixes ($\text{accuracy} < 8\text{ m}$), triggers `reanchor_to_gnss(...)`.
  3. Immediately resets state vector $\mathbf{x}$ and covariance $P$ to the verified GNSS anchor, returning to `GNSS_AIDED` in **1 single frame**.

### **6.3 Vibration Damping for Unmounted Devices (e.g., Bicycles)**
* **Problem**: Rigid bicycle forks and hand tremors create violent acceleration shocks ($> 15\text{ m/s}^2$). The TCN misinterprets these spikes as high-speed car vibrations, causing massive speed overshooting ($> 30\text{ km/h}$).
* **Solution ([`pipeline.py`](file:///Users/shudhanshuranjangupta/.gemini/antigravity/scratch/PS26168/src/navigation/pipeline.py))**:
  - Clamps jerk and high-frequency acceleration shocks ($jerk \in [-25, 25]\text{ m/s}^3$) before passing to the TCN buffer.
  - Ensures robust speed bounds regardless of road surface roughness.

---

## 7. I/O Schemas: Runtime Contracts

### **Input Schema (`SensorFrame`)**
Received by `CoreNavigationPipeline.process_frame(frame)`:
```json
{
  "timestamp": 12.921,
  "ax": 0.1002,
  "ay": 1.5390,
  "az": 9.8338,
  "gx": -0.0607,
  "gy": -0.0069,
  "gz": 0.0187,
  "gnss": {
    "lat": 26.144501,
    "lon": 91.736211,
    "speed": 5.54,
    "heading": 241.7,
    "accuracy": 3.5
  }
}
```
*(When entering a tunnel, `"gnss": null` is passed).*

### **Output Schema (`NavigationState` — Section 56 Compliant)**
Emitted to the UI / Frontend Web Map:
```json
{
  "timestamp": 12921,
  "latitude": 26.144971,
  "longitude": 91.736710,
  "speed": 19.94,
  "heading": 241.7,
  "mode": "GNSS_AIDED",
  "gnssQuality": 0.93,
  "aiConfidence": 1.0,
  "mapConfidence": 0.5
}
```

---

## 8. Hardware Roadmap: External 200 Hz IoT Sensor Pod

```
┌───────────────────────────────────────────────┐
│     EXTERNAL 200 Hz IoT SENSOR POD            │
│   • Low-noise industrial IMU (BNO085 / BMI088)│
│   • 200 Hz high-rate sampling                 │
│   • Fixed chassis mounting                    │
└───────────────────────┬───────────────────────┘
                        │ BLE / Wi-Fi / USB Telemetry
                        ▼
┌───────────────────────────────────────────────┐
│           SMARTPHONE COMPUTE HUB              │
│   • Edge AI Model (Tiny TCN @ 10 Hz)          │
│   • 2D EKF State Estimation Engine            │
│   • Offline MapLibre Vector Map Rendering     │
│   • Progressive Web App (PWA) UI              │
└───────────────────────────────────────────────┘
```
* **Architecture Strategy**: Keep all heavy compute, deep learning, Kalman filtering, and mapping on the driver's smartphone. The external IoT module serves solely as a dedicated, high-frequency, low-drift inertial sensing node.
