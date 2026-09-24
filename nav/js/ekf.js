// 2D INS EKF - JS port of src/navigation/ins_ekf.py (keep in sync).
// State [p_e, p_n, psi, v, b_g, s_g]; velocity along heading = hard non-holonomic constraint.
import { wrapPi } from "./frames.js";

export const PE = 0, PN = 1, PSI = 2, V = 3, BG = 4, SG = 5;
const N = 6;
export const CHI2_2DOF_999 = 13.82;

const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = (n) => { const m = zeros(n, n); for (let i = 0; i < n; i++) m[i][i] = 1; return m; };
function matmul(A, B) {
  const r = A.length, k = B.length, c = B[0].length, out = zeros(r, c);
  for (let i = 0; i < r; i++) for (let p = 0; p < k; p++) {
    const a = A[i][p]; if (a === 0) continue;
    for (let j = 0; j < c; j++) out[i][j] += a * B[p][j];
  }
  return out;
}
const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
function inv(S) {      // 1x1 or 2x2
  if (S.length === 1) return [[1 / S[0][0]]];
  const [[a, b], [c, d]] = S; const det = a * d - b * c;
  return [[d / det, -b / det], [-c / det, a / det]];
}

export class EKF2D {
  constructor({ gyroNoise = 0.005, speedProcessNoise = 1.0, biasWalk = 3e-5 } = {}) {
    this.x = [0, 0, 0, 0, 0, 0];
    this.P = zeros(N, N);
    [25, 25, 0.25, 4, 1e-4, 0.0025].forEach((v, i) => { this.P[i][i] = v; });
    this.qPsi = gyroNoise ** 2; this.qV = speedProcessNoise ** 2; this.qBg = biasWalk ** 2; this.qPos = 0.05 ** 2;
  }

  predict(dt, yawRate) {
    if (dt <= 0) return;
    const [e, n, psi, v, bg, sg] = this.x;
    const w = (1 + sg) * (yawRate - bg);
    const c = Math.cos(psi), s = Math.sin(psi);
    this.x[PE] = e + v * c * dt;
    this.x[PN] = n + v * s * dt;
    this.x[PSI] = wrapPi(psi + w * dt);
    const F = eye(N);
    F[PE][PSI] = -v * s * dt; F[PE][V] = c * dt;
    F[PN][PSI] = v * c * dt; F[PN][V] = s * dt;
    F[PSI][BG] = -(1 + sg) * dt; F[PSI][SG] = (yawRate - bg) * dt;
    const P = matmul(matmul(F, this.P), transpose(F));
    const q = [this.qPos, this.qPos, this.qPsi, this.qV, this.qBg, 1e-8];
    for (let i = 0; i < N; i++) P[i][i] += q[i] * dt;
    this.P = P;
  }

  // H: array of rows, r: residual array, R: matrix. Returns [nis, accepted].
  _update(H, r, R, gate = null) {
    const Ht = transpose(H);
    const PHt = matmul(this.P, Ht);
    const S = matmul(H, PHt);
    for (let i = 0; i < S.length; i++) for (let j = 0; j < S.length; j++) S[i][j] += R[i][j];
    const Si = inv(S);
    const rv = r.map((x) => [x]);
    const nis = matmul(matmul(transpose(rv), Si), rv)[0][0];
    if (gate !== null && nis > gate) return [nis, false];
    const K = matmul(PHt, Si);
    const dx = matmul(K, rv);
    for (let i = 0; i < N; i++) this.x[i] += dx[i][0];
    this.x[PSI] = wrapPi(this.x[PSI]);
    this.x[V] = Math.max(this.x[V], 0);
    this.x[SG] = Math.min(Math.max(this.x[SG], -0.2), 0.2);
    const IKH = eye(N);
    const KH = matmul(K, H);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) IKH[i][j] -= KH[i][j];
    const KRKt = matmul(matmul(K, R), transpose(K));
    const P = matmul(matmul(IKH, this.P), transpose(IKH));
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) P[i][j] += KRKt[i][j];
    this.P = P;
    return [nis, true];
  }

  _row(idx, val = 1) { const h = new Array(N).fill(0); h[idx] = val; return h; }

  updateSpeed(v, variance) { return this._update([this._row(V)], [v - this.x[V]], [[Math.max(variance, 1e-4)]]); }

  updateZupt(yawRateMeas, vVar = 0.01, zaruVar = 0.002 ** 2) {
    this.updateSpeed(0, vVar);
    return this._update([this._row(BG)], [yawRateMeas - this.x[BG]], [[zaruVar]]);
  }

  gnssPositionNis(e, n, std) {
    const s = Math.max(std, 1) ** 2;
    const S = [[this.P[PE][PE] + s, this.P[PE][PN]], [this.P[PN][PE], this.P[PN][PN] + s]];
    const Si = inv(S); const r = [e - this.x[PE], n - this.x[PN]];
    return r[0] * (Si[0][0] * r[0] + Si[0][1] * r[1]) + r[1] * (Si[1][0] * r[0] + Si[1][1] * r[1]);
  }

  updateGnssPosition(e, n, std, gate = CHI2_2DOF_999) {
    const s = Math.max(std, 1) ** 2;
    return this._update([this._row(PE), this._row(PN)], [e - this.x[PE], n - this.x[PN]], [[s, 0], [0, s]], gate);
  }

  updateGnssVelocity(speed, psiCourse, speedStd = 0.3, minSpeedForCourse = 2.0) {
    this.updateSpeed(speed, speedStd ** 2);
    if (speed > minSpeedForCourse) {
      const cs = 0.02 + 0.5 / speed;
      this._update([this._row(PSI)], [wrapPi(psiCourse - this.x[PSI])], [[cs * cs]], 16.0);
    }
  }

  updateCrossTrack(roadE, roadN, roadPsi, std) {
    const nx = -Math.sin(roadPsi), ny = Math.cos(roadPsi);
    const h = new Array(N).fill(0); h[PE] = nx; h[PN] = ny;
    const r = nx * (roadE - this.x[PE]) + ny * (roadN - this.x[PN]);
    return this._update([h], [r], [[std * std]], 16.0);
  }

  updateHeading(psi, std) { return this._update([this._row(PSI)], [wrapPi(psi - this.x[PSI])], [[std * std]], 16.0); }

  reanchor(e, n, psi = null, v = null, posStd = 5.0) {
    const clear = (k) => { for (let i = 0; i < N; i++) { this.P[k][i] = 0; this.P[i][k] = 0; } };
    this.x[PE] = e; this.x[PN] = n;
    clear(PE); clear(PN); this.P[PE][PE] = this.P[PN][PN] = posStd ** 2;
    if (psi !== null) { this.x[PSI] = wrapPi(psi); clear(PSI); this.P[PSI][PSI] = 0.01; }
    if (v !== null) { this.x[V] = Math.max(v, 0); clear(V); this.P[V][V] = 0.25; }
  }

  get position() { return [this.x[PE], this.x[PN]]; }
  get speed() { return this.x[V]; }
  get psi() { return this.x[PSI]; }
  get positionStd() { return Math.sqrt(Math.max(this.P[PE][PE] + this.P[PN][PN], 0)); }
}
