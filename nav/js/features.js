// IMU feature extraction - JS port of src/preprocessing/features.py (keep in sync).
// 9 gravity-levelled channels at the 10 Hz model rate; see the Python module for definitions.

export const G = 9.80665;
export const MODEL_RATE_HZ = 10.0;
export const T_SINCE_REF_MAX = 120.0;

const norm3 = (v) => Math.hypot(v[0], v[1], v[2]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function levelingMatrix(gHat) {
  const gn = norm3(gHat);
  const z = [gHat[0] / gn, gHat[1] / gn, gHat[2] / gn];
  let y = [-z[1] * z[0], 1 - z[1] * z[1], -z[1] * z[2]];
  if (norm3(y) < 1e-3) {
    const xFlat = [1 - z[0] * z[0], -z[0] * z[1], -z[0] * z[2]];
    y = cross(z, xFlat);
  }
  const yn = norm3(y);
  y = [y[0] / yn, y[1] / yn, y[2] / yn];
  const x = cross(y, z);
  return [x, y, z];
}

export class GravityTracker {
  constructor(tauS = 10.0, rateHz = MODEL_RATE_HZ, gInit = null) {
    this.alpha = 1.0 / (tauS * rateHz);
    this.g = gInit ? gInit.slice() : null;
  }

  update(acc, gyro) {
    const aMag = norm3(acc);
    if (this.g === null) {
      if (aMag > 0.8 * G && aMag < 1.2 * G) this.g = acc.slice();
      else return [0, 0, 1];
    } else if (aMag > 0.9 * G && aMag < 1.1 * G && norm3(gyro) < 0.2) {
      for (let k = 0; k < 3; k++) this.g[k] += this.alpha * (acc[k] - this.g[k]);
    }
    return this.gHat();
  }

  gHat() {
    if (this.g === null) return [0, 0, 1];
    const n = norm3(this.g);
    return [this.g[0] / n, this.g[1] / n, this.g[2] / n];
  }
}

export function featureFrame(acc, gyro, gHat) {
  const R = levelingMatrix(gHat);
  const a = [dot(R[0], acc), dot(R[1], acc), dot(R[2], acc)];
  const w = [dot(R[0], gyro), dot(R[1], gyro), dot(R[2], gyro)];
  return Float32Array.from([a[0], a[1], a[2] - G, w[0], w[1], w[2],
    norm3(acc) - G, Math.hypot(a[0], a[1]), norm3(gyro)]);
}

export class Decimator {
  constructor(rateHz = MODEL_RATE_HZ) {
    this.period = 1.0 / rateHz;
    this.binEnd = null;
    this.accSum = [0, 0, 0]; this.gyroSum = [0, 0, 0]; this.count = 0;
  }

  push(t, acc, gyro) {
    if (this.binEnd === null) this.binEnd = t + this.period * 0.5;
    let out = null;
    if (t >= this.binEnd && this.count > 0) {
      const c = this.count;
      out = [this.binEnd, this.accSum.map((v) => v / c), this.gyroSum.map((v) => v / c)];
      this.accSum = [0, 0, 0]; this.gyroSum = [0, 0, 0]; this.count = 0;
      while (this.binEnd <= t) this.binEnd += this.period;
    }
    for (let k = 0; k < 3; k++) { this.accSum[k] += acc[k]; this.gyroSum[k] += gyro[k]; }
    this.count += 1;
    return out;
  }
}
