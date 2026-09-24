# Intelligent Dead Reckoning (IDR) — Core Model & Pipeline Documentation (`doc1.md`)

> **⚠️ Superseded (2026-09-23).** This document describes the pre-optimisation system. Its accuracy figures (e.g. 0.98 m/s validation MAE, the drift tables) could not be reproduced: the shipped checkpoint scored 5.17 m/s test MAE on its own split, and no position error was ever measured. See `docs/SUMMARY.md` for an overview, `docs/OPTIMIZATION_PLAN.md` and `docs/CHANGES.md` for what changed and why, `docs/RESULTS.md` for measured results, and `AGENTS.md` for the current contracts and commands.

This document provides a comprehensive end-to-end technical guide for **Model #4 (Tiny TCN)** and the **Core Inertial Navigation Pipeline** as defined in **Part II** of `plan.pdf`.

---

## 📑 Table of Contents
1. [End-to-End Execution & Retraining Commands](#1-end-to-end-execution--retraining-commands)
2. [Dataset Preprocessing & Splits](#2-dataset-preprocessing--splits)
3. [Model Architecture & Hyperparameters](#3-model-architecture--hyperparameters)
4. [Physics-Informed Loss Function](#4-physics-informed-loss-function)
5. [End-to-End Pipeline Data Flow & Schemas](#5-end-to-end-pipeline-data-flow--schemas)
6. [ONNX Model Export](#6-onnx-model-export)

---

## 1. End-to-End Execution & Retraining Commands

Follow these steps from the root directory of the `PS26168` repository (`/Users/shudhanshuranjangupta/.gemini/antigravity/scratch/PS26168`):

### **Step 1: Download Raw Multi-Driver IO-VNBD Trips**
Downloads raw CSV sensor files for 8 diverse trips across Drivers A, B, D, and E into `data/raw/IO-VNBD/IO-VNBD-master/`:
```bash
python3 src/preprocessing/download_trip.py
```

### **Step 2: Parse & Extract 2-Second Rolling Windows**
Parses raw sensor logs, extracts 9 input channels at $10\text{ Hz}$ in 20-frame rolling windows (50% overlap), computes normalization statistics ($\mu, \sigma$) from the **Train set only**, and generates compressed datasets:
```bash
python3 src/preprocessing/create_windows.py
```
*Outputs generated:*
- `data/processed/normalization.json`
- `data/processed/train_windows.npz`
- `data/processed/val_windows.npz`
- `data/processed/test_windows.npz`

### **Step 3: Train Tiny TCN Model**
Trains the multi-task Tiny TCN model with Cosine Annealing learning rate scheduling, physics-informed loss, IMU data augmentation, and saves the best model checkpoint based on validation MAE:
```bash
python3 src/models/tiny_tcn.py --train --epochs 50 --lr 0.001
```
*Outputs generated:*
- `models/tiny_tcn_best.pth`

### **Step 4: Evaluate Model on Held-Out Test Set**
Evaluates the best trained checkpoint on unseen test windows (`test_windows.npz`):
```bash
python3 src/models/evaluate_models.py
```

### **Step 5: Export Trained Model to ONNX**
Exports the trained PyTorch checkpoint into dynamic-batch ONNX format for mobile/browser inference:
```bash
python3 src/models/export_onnx.py
```
*Outputs generated:*
- `models/idr_tcn.onnx` ($62.75\text{ KB}$)

### **Step 6: Run End-to-End Navigation Replay**
Runs full sensor replay through the core pipeline (`EKF2D` + State Machine + Map Matcher) simulating a GNSS tunnel blackout:
```bash
python3 src/navigation/replay.py
```

---

## 2. Dataset Preprocessing & Splits

### **Input Channel Specifications (9 Channels @ 10 Hz)**
Every $2$-second rolling window consists of **$20$ temporal frames** across **$9$ feature channels**:

| Channel Index | Symbol | Description | Unit |
| :--- | :--- | :--- | :--- |
| `Ch 0` | $a_x$ | Body Accelerometer X (forward/longitudinal) | $\text{m/s}^2$ |
| `Ch 1` | $a_y$ | Body Accelerometer Y (lateral) | $\text{m/s}^2$ |
| `Ch 2` | $a_z$ | Body Accelerometer Z (vertical) | $\text{m/s}^2$ |
| `Ch 3` | $g_x$ | Body Gyroscope Pitch | $\text{rad/s}$ |
| `Ch 4` | $g_y$ | Body Gyroscope Roll | $\text{rad/s}$ |
| `Ch 5` | $g_z$ | Body Gyroscope Yaw | $\text{rad/s}$ |
| `Ch 6` | $\|a\|$ | Total Acceleration Magnitude $\sqrt{a_x^2 + a_y^2 + a_z^2}$ | $\text{m/s}^2$ |
| `Ch 7` | $\|\omega\|$ | Total Gyroscope Magnitude $\sqrt{g_x^2 + g_y^2 + g_z^2}$ | $\text{rad/s}$ |
| `Ch 8` | $jerk$ | Acceleration Jerk $\frac{d\|a\|}{dt}$ | $\text{m/s}^3$ |

### **Dataset Partition Summary**
- **Window Length**: $20$ frames ($2.0$ seconds at $10\text{ Hz}$)
- **Step Size**: $5$ frames ($0.5$ seconds, $50\%$ overlap)
- **Train Set** (`train_windows.npz`): **$66,342$ windows** ($70\%$)
- **Validation Set** (`val_windows.npz`): **$14,216$ windows** ($15\%$)
- **Test Set** (`test_windows.npz`): **$14,217$ windows** ($15\%$)

---

## 3. Model Architecture & Hyperparameters

### **Tiny TCN Layer Specifications (`src/models/tiny_tcn.py`)**

```
Input: Tensor (Batch, Channels=9, Seq_Len=20)
 │
 ├── 1D Convolution: 9 -> 32 channels, kernel_size=5, padding=2, ReLU
 ├── 1D Dilated Convolution: 32 -> 32 channels, kernel_size=3, dilation=2, padding=2 + Residual Add
 ├── 1D Dilated Convolution: 32 -> 64 channels, kernel_size=3, dilation=4, padding=4, ReLU
 ├── Global Average Pooling (GAP) -> (Batch, 64)
 └── Shared Dense Layer: Linear(64, 64), ReLU
      │
      ├── Head 1 (Speed): Linear(64, 1)          => Forward Speed v_f (m/s)
      ├── Head 2 (Uncertainty): Linear(64, 1)    => log(sigma_v^2) for EKF
      └── Head 3 (Motion Class): Linear(64, 5)   => Logits for [stationary, accelerating, cruising, braking, turning]
```

### **Hyperparameter Configurations**

| Argument / Parameter | Value | Rationale / Reference |
| :--- | :--- | :--- |
| **Input Shape** | `(Batch, 9, 20)` | 9 sensor channels, 2-sec window at 10 Hz |
| **Optimizer** | `AdamW` | Weight decay $1\text{e-}4$ for regularization |
| **Learning Rate** | `1e-3` | Initial learning rate |
| **Scheduler** | `CosineAnnealingLR` | Smooth learning rate decay to $0$ over epochs |
| **Batch Size** | `64` | Optimal mini-batch gradient stability |
| **Default Epochs** | `50` | Full convergence point |
| **Model Size** | `45,223` params | Highly lightweight ($62.75\text{ KB}$ ONNX) |

---

## 4. Physics-Informed Loss Function

The total loss $L$ combines data supervised loss with physical constraint penalties:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{unc}} \mathcal{L}_{\text{unc}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{motion}} \mathcal{L}_{\text{motion}}$$

Where:
1. **Primary Data Loss**: $\mathcal{L}_{\text{data}} = \text{Huber}(v_f, v_{\text{GT}})$
2. **Heteroscedastic Uncertainty Loss**: $\mathcal{L}_{\text{unc}} = \frac{(v_f - v_{\text{GT}})^2}{\exp(\log \sigma_v^2)} + \log \sigma_v^2$
3. **Smoothness Loss**: $\mathcal{L}_{\text{smooth}} = |v_{f, t+1} - v_{f, t}|$
4. **Physics Consistency Loss**: $\mathcal{L}_{\text{phys}} = \left|v_{f, t+1} - (v_{f, t} + a_{x, t} \Delta t)\right|^2$
5. **Motion Classification Loss**: $\mathcal{L}_{\text{motion}} = \text{CrossEntropy}(c_{\text{pred}}, c_{\text{target}})$

**Loss Weights**:
- $\lambda_{\text{unc}} = 0.1$
- $\lambda_{\text{smooth}} = 0.05$
- $\lambda_{\text{phys}} = 0.05$
- $\lambda_{\text{motion}} = 0.1$

---

## 5. End-to-End Pipeline Data Flow & Schemas

### **Pipeline Architecture**

```
 ┌─────────────────┐
 │   SensorFrame   │  (Timestamp, Accelerometer, Gyroscope, GNSS)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ IMU Windowing   │  (Normalizes 9 channels, keeps rolling 20-frame buffer)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │    Tiny TCN     │  (Outputs speed v_f, uncertainty log_var, motion class)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │     EKF2D       │  (State: [px, py, vx, vy, psi, bg, bax, bay])
 └────────┬────────┘  (Updates: INS Propagation + AI Velocity + NHC + GNSS)
          │
          ▼
 ┌─────────────────┐
 │ GNSS State Mach │  (Transitions: GNSS_AIDED -> DEGRADED -> DEAD_RECKONING -> REACQUISITION)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Map Matching   │  (Composite candidate score S = w_d * D + w_h * H + w_t * T)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ NavigationState │  (Standardized Section 56 JSON Output Schema)
 └─────────────────┘
```

### **Input Schema (`SensorFrame`)**
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
    "lat": 26.1445,
    "lon": 91.7362,
    "speed": 5.54,
    "heading": 241.7,
    "accuracy": 3.5
  }
}
```

### **Output Schema (`NavigationState` — Section 56 Compliant)**
```json
{
  "timestamp": 12921,
  "latitude": 26.144971,
  "longitude": 91.73671,
  "speed": 22.6,
  "heading": 241.7,
  "mode": "DEAD_RECKONING",
  "gnssQuality": 0.0,
  "aiConfidence": 1.0,
  "mapConfidence": 0.5
}
```

---

## 6. ONNX Model Export

- **Script**: `src/models/export_onnx.py`
- **Output File**: `models/idr_tcn.onnx`
- **ONNX Opset Version**: `14`
- **File Size**: `62.75 KB`
- **Dynamic Axes**: Supported on batch dimension `[batch_size, 9, 20]`
- **Input Node**: `imu_window`
- **Output Nodes**: `speed`, `log_var`, `motion_logits`
