// Core IDR navigation pipeline - JS port of src/navigation/pipeline.py (keep in sync).
// SensorFrame in (any IMU rate), NavigationState out. `model` needs async predict(window, scalars).
import { LocalTangentPlane, compassToPsi, psiToCompass } from "./frames.js";
import { Decimator, GravityTracker, MODEL_RATE_HZ, T_SINCE_REF_MAX, featureFrame } from "./features.js";
import { EKF2D } from "./ekf.js";
import { Decision, GNSSQualityMonitor, Mode } from "./gnss.js";
import { HMMMapMatcher, RoadNetwork } from "./mapmatch.js";

const DR_MODES = [Mode.DEAD_RECKONING, Mode.REACQUISITION];

export class CoreNavigationPipeline {
  constructor(model, { calibration = null, useMap = true, useAi = true, aiVarianceInflation = 4.0,
    zuptThreshold = 0.8, mapUpdatePeriodS = 0.5, mapMinConfidence = 0.4, mapHeadingUpdate = true } = {}) {
    this.model = model; this.window = model.window;
    Object.assign(this, { useMap, useAi, aiVarianceInflation, zuptThreshold, mapMinConfidence, mapHeadingUpdate });
    this.mapPeriod = mapUpdatePeriodS;
    this.ekf = new EKF2D();
    this.gnssMonitor = new GNSSQualityMonitor();
    this.gravity = new GravityTracker();
    this.decimator = new Decimator(MODEL_RATE_HZ);
    this.buffer = [];
    this.matcher = new HMMMapMatcher();
    this.ltp = null; this.pendingMap = null;
    this.lastT = null; this.vRef = 0; this.tRef = null; this.hasRef = false;
    this.lastAi = null; this.lastMatch = null; this.travelledSinceMatch = 0; this.lastMapT = null; this.binYaw = [];
    if (calibration && calibration.is_calibrated) this._loadCalibration(calibration);
  }

  _loadCalibration(cal) {
    const g = cal.resting_gravity_vector, b = cal.gyro_bias_rad_s || {};
    if (!g) return;
    this.gravity = new GravityTracker(10, MODEL_RATE_HZ, [g.x, g.y, g.z]);
    const gh = this.gravity.gHat();
    this.ekf.x[4] = gh[0] * (b.gx || 0) + gh[1] * (b.gy || 0) + gh[2] * (b.gz || 0);
  }

  setOrigin(lat, lon) {
    this.ltp = new LocalTangentPlane(lat, lon);
    if (this.pendingMap && this.useMap) this.matcher = new HMMMapMatcher(RoadNetwork.fromGeoJSON(this.pendingMap, this.ltp));
  }

  // Road map (GeoJSON); applied as soon as the ENU origin is known.
  setMap(geojson) {
    this.pendingMap = geojson;
    if (this.ltp && this.useMap) this.matcher = new HMMMapMatcher(RoadNetwork.fromGeoJSON(geojson, this.ltp));
  }

  async processFrame(frame) {
    const t = frame.timestamp;
    const acc = [frame.ax, frame.ay, frame.az], gyro = [frame.gx, frame.gy, frame.gz];
    const gh = this.gravity.gHat();
    const yawRate = gh[0] * gyro[0] + gh[1] * gyro[1] + gh[2] * gyro[2];
    if (this.lastT !== null) {
      const dt = t - this.lastT;
      if (dt > 0 && dt < 1) { this.ekf.predict(dt, yawRate); this.travelledSinceMatch += this.ekf.speed * dt; }
    }
    this.lastT = t;
    this.binYaw.push(yawRate);

    const sample = this.decimator.push(t, acc, gyro);
    if (sample !== null) await this._modelStep(sample);

    const fix = frame.gnss;
    if (fix && fix.lat !== undefined) this._gnssStep(t, fix);
    const mode = this.gnssMonitor.tick(t);

    if (this.ltp && this.matcher.hasMap && (this.lastMapT === null || t - this.lastMapT >= this.mapPeriod)) {
      this._mapStep(mode);
      this.lastMapT = t;
    }
    return this._navState(t, mode);
  }

  async _modelStep([, acc10, gyro10]) {
    const gh = this.gravity.update(acc10, gyro10);
    this.buffer.push(featureFrame(acc10, gyro10, gh));
    if (this.buffer.length > this.window) this.buffer.shift();
    const rawYaw = this.binYaw.length ? this.binYaw.reduce((a, b) => a + b, 0) / this.binYaw.length : 0;
    this.binYaw = [];
    if (this.buffer.length < this.window || !this.useAi) return;
    const tSince = this.hasRef ? this.lastT - this.tRef : 0;
    const refValid = this.hasRef && tSince <= T_SINCE_REF_MAX;
    const out = await this.model.predict(this.buffer, refValid ? [this.vRef, tSince, 1] : [0, 0, 0]);
    this.lastAi = out;
    if (out.pStationary > this.zuptThreshold && out.speed < 0.5) this.ekf.updateZupt(rawYaw);
    else this.ekf.updateSpeed(out.speed, out.variance * this.aiVarianceInflation);
  }

  _gnssStep(t, fix) {
    if (!this.ltp) this.setOrigin(fix.lat, fix.lon);
    const [e, n] = this.ltp.toEnu(fix.lat, fix.lon);
    const acc = fix.accuracy ?? 10;
    const speed = fix.speed ?? null, course = fix.heading ?? null;
    const nis = this.ekf.gnssPositionNis(e, n, acc);
    const decision = this.gnssMonitor.onFix(t, e, n, speed ?? 0, acc, nis, this.ekf.speed);
    const psi = course !== null && speed !== null && speed > 2 ? compassToPsi(course) : null;
    if (decision === Decision.REANCHOR) {
      this.ekf.reanchor(e, n, psi, speed);
      this.matcher.reset();
    } else if (decision === Decision.FUSE || decision === Decision.FUSE_INFLATED) {
      this.ekf.updateGnssPosition(e, n, acc * (decision === Decision.FUSE_INFLATED ? 2 : 1), null);
      if (speed !== null) this.ekf.updateGnssVelocity(speed, psi !== null ? psi : this.ekf.psi);
    }
    if ((decision === Decision.REANCHOR || decision === Decision.FUSE || decision === Decision.FUSE_INFLATED) && speed !== null) {
      this.vRef = speed; this.tRef = t; this.hasRef = true;
    }
  }

  _mapStep(mode) {
    const [e, n] = this.ekf.position;
    const m = this.matcher.step(e, n, this.ekf.psi, this.ekf.speed, this.travelledSinceMatch, this.ekf.positionStd);
    this.travelledSinceMatch = 0;
    this.lastMatch = m;
    if (!m || !DR_MODES.includes(mode) || m.confidence < this.mapMinConfidence) return;
    this.ekf.updateCrossTrack(m.e, m.n, m.roadPsi, 3.0 / m.confidence);
    if (this.mapHeadingUpdate && this.ekf.speed > 3 && m.confidence > 0.8) {
      this.ekf.updateHeading(m.roadPsi, (6 * Math.PI / 180) / m.confidence);
    }
  }

  _navState(t, mode) {
    const [e, n] = this.ekf.position;
    const ll = this.ltp ? this.ltp.toGeodetic(e, n) : [null, null];
    const ai = this.lastAi;
    return {
      timestamp: Math.round(t * 1000), latitude: ll[0], longitude: ll[1], east: e, north: n,
      speed: this.ekf.speed * 3.6, heading: psiToCompass(this.ekf.psi), mode,
      gnssQuality: this.gnssMonitor.gnssQuality,
      aiSpeed: ai ? ai.speed * 3.6 : null,
      aiConfidence: ai ? 1 / (1 + Math.sqrt(Math.max(ai.variance, 0))) : 0,
      mapConfidence: this.lastMatch ? this.lastMatch.confidence : 0,
      positionStdM: this.ekf.positionStd,
      matched: this.lastMatch ? [this.lastMatch.e, this.lastMatch.n] : null,
    };
  }
}
