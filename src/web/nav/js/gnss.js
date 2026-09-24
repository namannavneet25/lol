// GNSS quality monitor & mode state machine - JS port of src/navigation/gnss_quality.py (keep in sync).

export const Mode = {
  GNSS_AIDED: "GNSS_AIDED", DEGRADED: "DEGRADED",
  DEAD_RECKONING: "DEAD_RECKONING", REACQUISITION: "REACQUISITION",
};
const MAX_PAIR_GAP_S = 60, MAX_PLAUSIBLE_SPEED = 60, SPARSE_INTERVAL_S = 3;
export const Decision = { FUSE: "fuse", FUSE_INFLATED: "fuse_inflated", REANCHOR: "reanchor", REJECT: "reject" };

export class GNSSQualityMonitor {
  constructor({ maxAccuracyM = 25, degradedAccuracyM = 12, timeoutFactor = 1.5, nominalIntervalS = 1.0,
    nisGate = 13.82, maxConsecutiveRejects = 3 } = {}) {
    Object.assign(this, { maxAccuracyM, degradedAccuracyM, timeoutFactor, nisGate, maxConsecutiveRejects });
    this.interval = nominalIntervalS;
    this.mode = Mode.GNSS_AIDED;
    this.gnssQuality = 0;
    this.lastFixTime = null; this.lastFix = null; this.pendingFix = null;
    this.consecutiveRejects = 0; this.initialized = false; this.transitions = [];
  }

  _setMode(t, m) { if (m !== this.mode) { this.transitions.push([t, this.mode, m]); this.mode = m; } }

  timeoutS() { return this.timeoutFactor * Math.max(this.interval, 0.1); }

  tick(t) {
    if (this.lastFixTime === null || t - this.lastFixTime > this.timeoutS()) {
      if (this.mode !== Mode.DEAD_RECKONING) this._setMode(t, Mode.DEAD_RECKONING);
      this.gnssQuality = 0;
    }
    return this.mode;
  }

  // Fixes <= 5 s apart must match their speeds. Sparse pairs are bounded by the fastest of the fix speeds and
  // the filter's own speed (x1.5 + 3 m/s, capped at 60 m/s), so a single far-off fix is not accepted.
  static consistent(a, b, filterSpeed = 0) {
    const dt = b[0] - a[0];
    if (dt <= 0 || dt > MAX_PAIR_GAP_S) return false;
    const moved = Math.hypot(b[1] - a[1], b[2] - a[2]);
    if (dt <= 5) {
      const expected = 0.5 * (a[3] + b[3]) * dt;
      return Math.abs(moved - expected) < 3 * Math.hypot(a[4], b[4]) + 2 * dt;
    }
    const v = Math.max(filterSpeed || 0, a[3] || 0, b[3] || 0);
    const limit = Math.min(MAX_PLAUSIBLE_SPEED, 1.5 * v + 3) * dt;
    return moved <= limit + 3 * Math.hypot(a[4], b[4]) + 10;
  }

  onFix(t, e, n, speed, acc, nis, filterSpeed = 0) {
    if (acc === null || acc === undefined || acc > this.maxAccuracyM) return Decision.REJECT;
    if (this.lastFixTime !== null) {
      // cap gaps at 5 s for normal receivers (an outage gap must not inflate the interval); 15 s once sparse
      const cap = this.interval > SPARSE_INTERVAL_S ? 15 : 5;
      this.interval = 0.8 * this.interval + 0.2 * Math.min(Math.max(t - this.lastFixTime, 0.1), cap);
    }
    this.lastFixTime = t;
    const fix = [t, e, n, speed, acc];
    const baseQ = Math.max(0.1, Math.min(1, 1 - (acc - 3) / 25));
    if (!this.initialized) {
      this.initialized = true; this.lastFix = fix; this.gnssQuality = baseQ;
      this._setMode(t, Mode.GNSS_AIDED);
      return Decision.REANCHOR;
    }
    if (this.mode === Mode.DEAD_RECKONING || this.mode === Mode.REACQUISITION) {
      if (nis <= this.nisGate) {
        this.pendingFix = null;
        this._setMode(t, Mode.GNSS_AIDED); this.gnssQuality = baseQ; this.lastFix = fix;
        return Decision.FUSE;
      }
      if (this.pendingFix !== null && GNSSQualityMonitor.consistent(this.pendingFix, fix, filterSpeed)) {
        this.pendingFix = null; this.consecutiveRejects = 0;
        this._setMode(t, Mode.GNSS_AIDED); this.gnssQuality = baseQ; this.lastFix = fix;
        return Decision.REANCHOR;
      }
      this.pendingFix = fix;
      this._setMode(t, Mode.REACQUISITION); this.gnssQuality = 0.5 * baseQ;
      return Decision.REJECT;
    }
    if (nis > this.nisGate) {
      this.consecutiveRejects += 1;
      this.gnssQuality = Math.max(0, baseQ - 0.2 * this.consecutiveRejects);
      if (this.consecutiveRejects >= this.maxConsecutiveRejects && GNSSQualityMonitor.consistent(this.lastFix, fix, filterSpeed)) {
        this.consecutiveRejects = 0; this.lastFix = fix;
        return Decision.REANCHOR;
      }
      this.lastFix = fix;
      return Decision.REJECT;
    }
    this.consecutiveRejects = 0; this.lastFix = fix; this.gnssQuality = baseQ;
    if (acc > this.degradedAccuracyM) { this._setMode(t, Mode.DEGRADED); return Decision.FUSE_INFLATED; }
    this._setMode(t, Mode.GNSS_AIDED);
    return Decision.FUSE;
  }
}
