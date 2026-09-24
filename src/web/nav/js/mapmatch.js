// Online HMM map matcher - JS port of src/navigation/map_matching.py (keep in sync).
import { wrapPi } from "./frames.js";

export function onewayDirection(props) {
  const tag = String(props.oneway ?? "").toLowerCase();
  if (["yes", "true", "1"].includes(tag)) return 1;
  if (tag === "-1") return -1;
  const implied = ["roundabout", "circular"].includes(props.junction) || ["motorway", "motorway_link"].includes(props.highway);
  return implied && !["no", "false", "0"].includes(tag) ? 1 : 0;
}

class MinHeap {
  constructor() { this.a = []; }
  push(item) {
    const a = this.a; a.push(item); let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] <= a[i][0]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; }
  }
  pop() {
    const a = this.a; const top = a[0]; const last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break; [a[m], a[i]] = [a[i], a[m]]; i = m;
      }
    }
    return top;
  }
  get size() { return this.a.length; }
}

export class RoadNetwork {
  constructor(cell = 50) {
    this.cell = cell;
    this.p1 = []; this.p2 = []; this.n1 = []; this.n2 = []; this.oneway = []; this.length = []; this.psi = [];
    this.grid = new Map(); this.nodeSegs = new Map();
    this.highway = [];
  }

  static fromGeoJSON(geojson, ltp, cell = 50) {
    const net = new RoadNetwork(cell);
    const nodeIds = new Map();
    for (const feat of geojson.features || []) {
      const g = feat.geometry || {};
      const lines = g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : [];
      const props = feat.properties || {};
      const ow = onewayDirection(props);
      for (const coords of lines) {
        if (!coords || coords.length < 2) continue;
        const pts = coords.map(([lon, lat]) => ltp.toEnu(lat, lon));
        const ids = coords.map(([lon, lat]) => {
          const key = `${lon.toFixed(7)},${lat.toFixed(7)}`;
          if (!nodeIds.has(key)) nodeIds.set(key, nodeIds.size);
          return nodeIds.get(key);
        });
        for (let i = 0; i < coords.length - 1; i++) {
          if (ids[i] === ids[i + 1]) continue;
          net._add(pts[i], pts[i + 1], ids[i], ids[i + 1], ow, props.highway);
        }
      }
    }
    net._index();
    return net;
  }

  _add(a, b, n1, n2, ow, highway) {
    this.p1.push(a); this.p2.push(b); this.n1.push(n1); this.n2.push(n2); this.oneway.push(ow);
    this.length.push(Math.hypot(b[0] - a[0], b[1] - a[1]));
    this.psi.push(Math.atan2(b[1] - a[1], b[0] - a[0]));
    this.highway.push(highway || "");
  }

  _index() {
    const c = this.cell;
    for (let i = 0; i < this.length.length; i++) {
      const [a, b] = [this.p1[i], this.p2[i]];
      const x0 = Math.floor(Math.min(a[0], b[0]) / c), x1 = Math.floor(Math.max(a[0], b[0]) / c);
      const y0 = Math.floor(Math.min(a[1], b[1]) / c), y1 = Math.floor(Math.max(a[1], b[1]) / c);
      for (let gx = x0; gx <= x1; gx++) for (let gy = y0; gy <= y1; gy++) {
        const k = `${gx},${gy}`;
        if (!this.grid.has(k)) this.grid.set(k, []);
        this.grid.get(k).push(i);
      }
      for (const nd of [this.n1[i], this.n2[i]]) {
        if (!this.nodeSegs.has(nd)) this.nodeSegs.set(nd, []);
        this.nodeSegs.get(nd).push(i);
      }
    }
  }

  get size() { return this.length.length; }

  segmentsInBox(e0, n0, e1, n1) {
    const c = this.cell, out = new Set();
    for (let gx = Math.floor(e0 / c); gx <= Math.floor(e1 / c); gx++)
      for (let gy = Math.floor(n0 / c); gy <= Math.floor(n1 / c); gy++)
        for (const s of this.grid.get(`${gx},${gy}`) || []) out.add(s);
    return out;
  }

  nearby(e, n, radius) {
    const ids = this.segmentsInBox(e - radius, n - radius, e + radius, n + radius);
    const out = [];
    for (const s of ids) {
      const [a, b] = [this.p1[s], this.p2[s]];
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const L2 = Math.max(dx * dx + dy * dy, 1e-9);
      const t = Math.min(1, Math.max(0, ((e - a[0]) * dx + (n - a[1]) * dy) / L2));
      const pe = a[0] + t * dx, pn = a[1] + t * dy;
      const dist = Math.hypot(e - pe, n - pn);
      if (dist <= radius) out.push({ seg: s, t, pe, pn, dist });
    }
    return out;
  }

  networkDistances(seg, t, cutoff) {
    const L = this.length[seg], dist = new Map(), heap = new MinHeap();
    heap.push([t * L, this.n1[seg]]); heap.push([(1 - t) * L, this.n2[seg]]);
    while (heap.size) {
      const [d, node] = heap.pop();
      if (dist.has(node) || d > cutoff) continue;
      dist.set(node, d);
      for (const s of this.nodeSegs.get(node) || []) {
        const other = this.n1[s] === node ? this.n2[s] : this.n1[s];
        if (!dist.has(other)) heap.push([d + this.length[s], other]);
      }
    }
    return dist;
  }
}

const LOG_ZERO = -1e9;

export class HMMMapMatcher {
  constructor(net = null, { sigmaHeadingDeg = 25, minSigmaDist = 4, maxCandidates = 12, betaM = 8 } = {}) {
    this.net = net; this.sigmaH = sigmaHeadingDeg * Math.PI / 180; this.minSigmaD = minSigmaDist;
    this.maxCandidates = maxCandidates; this.beta = betaM; this.cands = null;
  }

  get hasMap() { return this.net !== null && this.net.size > 0; }
  reset() { this.cands = null; }

  roadHeading(seg, psi) {
    const fwd = this.net.psi[seg], ow = this.net.oneway[seg];
    if (ow === 1) return fwd;
    const rev = wrapPi(fwd + Math.PI);
    if (ow === -1) return rev;
    return Math.abs(wrapPi(psi - fwd)) <= Math.abs(wrapPi(psi - rev)) ? fwd : rev;
  }

  step(e, n, psi, speed, travelled, posStd) {
    if (!this.hasMap) return null;
    const sigmaD = Math.max(this.minSigmaD, posStd);
    const radius = Math.min(Math.max(3 * sigmaD, 25), 120);
    let c = this.net.nearby(e, n, radius);
    if (!c.length) { this.reset(); return null; }
    c.sort((a, b) => a.dist - b.dist);
    c = c.slice(0, this.maxCandidates);
    for (const k of c) {
      k.roadPsi = this.roadHeading(k.seg, psi);
      k.emis = -0.5 * (k.dist / sigmaD) ** 2;
      if (speed > 2) k.emis -= 0.5 * (Math.abs(wrapPi(psi - k.roadPsi)) / this.sigmaH) ** 2;
      k.score = k.emis;
    }
    if (this.cands) {
      const beta = this.beta + 0.25 * travelled, cutoff = 2 * travelled + 60;
      const best = new Array(c.length).fill(LOG_ZERO);
      for (const [pseg, [pt, pscore]] of this.cands) {
        const nd = this.net.networkDistances(pseg, pt, cutoff);
        c.forEach((k, j) => {
          let route;
          if (k.seg === pseg) route = Math.abs(k.t - pt) * this.net.length[k.seg];
          else {
            const L = this.net.length[k.seg];
            const d1 = (nd.has(this.net.n1[k.seg]) ? nd.get(this.net.n1[k.seg]) : Infinity) + k.t * L;
            const d2 = (nd.has(this.net.n2[k.seg]) ? nd.get(this.net.n2[k.seg]) : Infinity) + (1 - k.t) * L;
            route = Math.min(d1, d2);
          }
          if (!Number.isFinite(route)) return;
          best[j] = Math.max(best[j], pscore - Math.abs(route - travelled) / beta);
        });
      }
      if (!best.every((b) => b <= LOG_ZERO / 2)) c.forEach((k, j) => { k.score = k.emis + best[j]; });
    }
    const mx = Math.max(...c.map((k) => k.score));
    let sum = 0;
    for (const k of c) { k.score -= mx; k.post = Math.exp(k.score); sum += k.post; }
    let bi = 0;
    c.forEach((k, j) => { k.post /= sum; if (k.score > c[bi].score) bi = j; });
    const b = c[bi];
    this.cands = new Map(c.map((k) => [k.seg, [k.t, k.score]]));
    let conf = 0;
    for (const k of c) {
      if (Math.hypot(k.pe - b.pe, k.pn - b.pn) < 3 && Math.abs(wrapPi(k.roadPsi - b.roadPsi)) < 20 * Math.PI / 180) conf += k.post;
    }
    return { e: b.pe, n: b.pn, roadPsi: b.roadPsi, confidence: conf, segment: b.seg, distance: b.dist };
  }
}
