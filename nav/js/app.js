// IDR Navigator UI: replay (IO-VNBD) and live (phone sensors) sources, canvas map, smooth vehicle icon.
import { CoreNavigationPipeline } from "./pipeline.js";
import { SpeedModel } from "./model.js";
import { wrapPi } from "./frames.js";
import { Mode } from "./gnss.js";

const $ = (id) => document.getElementById(id);
const canvas = $("map"), ctx = canvas.getContext("2d");
const MODE_STYLE = {
  [Mode.GNSS_AIDED]: ["GNSS-AIDED", "var(--aided)"], [Mode.DEAD_RECKONING]: ["DEAD RECKONING", "var(--dr)"],
  [Mode.REACQUISITION]: ["REACQUIRING", "var(--reacq)"], [Mode.DEGRADED]: ["GNSS DEGRADED", "var(--degraded)"],
};
const ROAD_WIDTH = { motorway: 4, trunk: 3.5, primary: 3, secondary: 2.6, tertiary: 2.2, motorway_link: 2, trunk_link: 2,
  primary_link: 2, residential: 1.6, unclassified: 1.6, living_street: 1.2, service: 0.9 };

const state = {
  source: "replay", playing: false, rate: 4, tunnel: false, follow: true, scale: 1.6, center: [0, 0],
  engine: null, model: null, prev: null, cur: null, curWall: 0, trail: [], fixes: [], gtTrail: [],
  replay: null, frame: 0, simT: 0, busy: false, outages: [], outageStart: null, calibration: null, map: null,
};

function toast(msg, ms = 2500) {
  const t = $("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(toast.h); if (ms) toast.h = setTimeout(() => { t.style.display = "none"; }, ms);
}

// ------------------------------------------------------------------ engine lifecycle
async function newEngine() {
  if (!state.model) state.model = await SpeedModel.load("./model/");
  state.engine = new CoreNavigationPipeline(state.model, { calibration: state.calibration });
  if (state.map) state.engine.setMap(state.map);
  state.prev = state.cur = null; state.trail = []; state.fixes = []; state.gtTrail = []; state.outages = [];
  state.outageStart = null; renderLog();
}

async function loadReplay(entry) {
  toast(`Loading ${entry.trip} replay…`, 0);
  const [rep, map] = await Promise.all([
    fetch(`./replay/${entry.file}`).then((r) => r.json()),
    fetch(`./replay/${entry.map}`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  state.replay = rep; state.map = map; state.frame = 0; state.simT = 0;
  await newEngine();
  state.gtIndex = new Map(rep.gt.map((g) => [g[0], [g[1], g[2]]]));
  state.fixIndex = new Map(rep.gnss.map((g) => [g[0], g]));
  toast(`${entry.trip}: ${entry.minutes} min of IO-VNBD data. Press Play, then "Simulate tunnel".`, 4000);
  state.center = [0, 0];
}

function replayFrame(i) {
  const r = state.replay, o = i * 6, im = r.imu;
  const f = { timestamp: i / r.rate_hz, ax: im[o], ay: im[o + 1], az: im[o + 2], gx: im[o + 3], gy: im[o + 4], gz: im[o + 5], gnss: null };
  const g = state.fixIndex.get(i);
  if (g && !state.tunnel) f.gnss = { lat: g[1], lon: g[2], speed: g[3], heading: g[4], accuracy: g[5] };
  return f;
}

function gtEnu(i) {
  const k = Math.round(i / 10) * 10, g = state.gtIndex.get(k);
  if (!g || !state.engine.ltp) return null;
  return state.engine.ltp.toEnu(g[0], g[1]);
}

async function onNav(nav, frameIdx = null) {
  state.prev = state.cur; state.cur = nav; state.curWall = performance.now();
  if (nav.latitude === null) return;
  const last = state.trail[state.trail.length - 1];
  if (!last || Math.hypot(nav.east - last[0], nav.north - last[1]) > 1) state.trail.push([nav.east, nav.north, nav.mode]);
  if (state.trail.length > 20000) state.trail.splice(0, 5000);
  if (frameIdx !== null && frameIdx % 10 === 0) {
    const gt = gtEnu(frameIdx);
    if (gt) {
      state.gtTrail.push(gt);
      state.lastErr = Math.hypot(nav.east - gt[0], nav.north - gt[1]);
    }
  }
}

async function stepReplay(budgetMs) {
  if (state.busy || !state.replay) return;
  state.busy = true;
  const t0 = performance.now();
  const r = state.replay;
  try {
    while (state.frame < r.frames && state.frame / r.rate_hz <= state.simT && performance.now() - t0 < budgetMs) {
      const f = replayFrame(state.frame);
      if (f.gnss) state.fixes.push(f.gnss);
      const nav = await state.engine.processFrame(f);
      await onNav(nav, state.frame);
      trackOutage(state.frame);
      state.frame += 1;
    }
    if (state.frame >= r.frames) { state.playing = false; $("play").textContent = "↺ Restart"; }
  } finally { state.busy = false; }
}

// ------------------------------------------------------------------ outage bookkeeping (replay only)
function trackOutage(i) {
  if (state.tunnel && state.outageStart === null && state.cur) {
    state.outageStart = { frame: i, dist: 0, last: gtEnu(i) };
  }
  if (state.outageStart && i % 10 === 0) {
    const g = gtEnu(i);
    if (g && state.outageStart.last) state.outageStart.dist += Math.hypot(g[0] - state.outageStart.last[0], g[1] - state.outageStart.last[1]);
    state.outageStart.last = g || state.outageStart.last;
  }
  if (!state.tunnel && state.outageStart) {
    const g = gtEnu(i), nav = state.cur;
    const o = state.outageStart; state.outageStart = null;
    if (g && nav && o.dist > 20) {
      const err = Math.hypot(nav.east - g[0], nav.north - g[1]);
      state.outages.push({ secs: (i - o.frame) / 10, dist: o.dist, err, drift: 100 * err / o.dist });
      renderLog(); $("log").classList.add("show");
    }
  }
}

function renderLog() {
  $("logBody").innerHTML = state.outages.map((o, k) => `<tr><td>${k + 1}</td><td>${o.secs.toFixed(0)} s</td>` +
    `<td>${o.dist.toFixed(0)} m</td><td>${o.err.toFixed(1)} m</td>` +
    `<td class="${o.drift < 10 ? "pass" : "fail"}">${o.drift.toFixed(1)}%</td></tr>`).join("") ||
    `<tr><td colspan="5" style="text-align:center;color:var(--muted)">Toggle "Simulate tunnel" while playing</td></tr>`;
}

// ------------------------------------------------------------------ live sensors
const IOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
let liveQueue = Promise.resolve(), lastFix = null, watchId = null;

async function startLive() {
  stopLive();                          // clicking Live again must not start a second sensor/geolocation watch
  state.replay = null; state.playing = false;
  try { state.calibration = await (await fetch("/api/calibration")).json(); } catch { state.calibration = null; }
  if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
    const p = await DeviceMotionEvent.requestPermission();
    if (p !== "granted") { toast("Motion sensor permission denied"); return; }
  }
  // Offline road map for map matching (src/web/export_live_map.py writes maps/live.geojson)
  try {
    const r = await fetch("./maps/live.geojson");
    state.map = r.ok ? await r.json() : null;
  } catch { state.map = null; }
  await newEngine();
  window.addEventListener("devicemotion", onMotion, true);
  if ("geolocation" in navigator) {
    watchId = navigator.geolocation.watchPosition((p) => {
      lastFix = { lat: p.coords.latitude, lon: p.coords.longitude, accuracy: p.coords.accuracy,
        speed: Number.isFinite(p.coords.speed) ? p.coords.speed : null,
        heading: Number.isFinite(p.coords.heading) ? p.coords.heading : null };
    }, (e) => toast(`GNSS: ${e.message}`), { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 });
  }
  const cal = state.calibration && state.calibration.is_calibrated ? "stand calibration loaded" : "no stand calibration";
  const map = state.map ? `offline map (${state.map.features.length} roads)` :
    "NO offline map - map matching off (run src/web/export_live_map.py)";
  toast(`Live: ${cal}; ${map}`, 5000);
}

function stopLive() {
  window.removeEventListener("devicemotion", onMotion, true);
  if (watchId !== null) navigator.geolocation.clearWatch(watchId);
  watchId = null;
}

function onMotion(ev) {
  const a = ev.accelerationIncludingGravity, r = ev.rotationRate;
  if (!a || !r) return;
  const s = IOS ? -1 : 1;           // iOS reports the opposite sign convention
  const D = Math.PI / 180;
  const f = { timestamp: ev.timeStamp / 1000, ax: s * (a.x || 0), ay: s * (a.y || 0), az: s * (a.z || 0),
    gx: (r.beta || 0) * D, gy: (r.gamma || 0) * D, gz: (r.alpha || 0) * D, gnss: null };
  if (lastFix && !state.tunnel) { f.gnss = lastFix; state.fixes.push(lastFix); }
  lastFix = null;
  liveQueue = liveQueue.then(() => state.engine.processFrame(f)).then((nav) => onNav(nav)).catch((e) => console.error(e));
}

// ------------------------------------------------------------------ rendering
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function pose() {
  const c = state.cur, p = state.prev;
  if (!c) return null;
  if (!p) return [c.east, c.north, c.heading];
  const period = Math.max(1, c.timestamp - p.timestamp) / (state.source === "replay" ? state.rate : 1);
  const a = Math.min(1, (performance.now() - state.curWall) / period);
  const dh = wrapPi(((c.heading - p.heading) * Math.PI) / 180) * 180 / Math.PI;
  return [p.east + a * (c.east - p.east), p.north + a * (c.north - p.north), p.heading + a * dh];
}

function draw() {
  const W = innerWidth, H = innerHeight;
  ctx.fillStyle = "#0f1720"; ctx.fillRect(0, 0, W, H);
  const eng = state.engine, ps = pose();
  if (ps && state.follow) state.center = [ps[0], ps[1]];
  const [cx, cy] = state.center, k = state.scale;
  const toScr = (e, n) => [W / 2 + (e - cx) * k, H / 2 - (n - cy) * k];
  const net = eng && eng.matcher && eng.matcher.net;
  if (net) {
    const segs = net.segmentsInBox(cx - W / 2 / k, cy - H / 2 / k, cx + W / 2 / k, cy + H / 2 / k);
    ctx.lineCap = "round"; ctx.strokeStyle = "#2b3a4a";
    for (const s of segs) {
      ctx.lineWidth = Math.max(1, (ROAD_WIDTH[net.highway[s]] || 1.2) * Math.min(k, 3));
      const a = toScr(...net.p1[s]), b = toScr(...net.p2[s]);
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    }
  }
  const path = (pts, color, width, dash = []) => {
    if (pts.length < 2) return;
    ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(...toScr(p[0], p[1])) : ctx.moveTo(...toScr(p[0], p[1]))));
    ctx.stroke(); ctx.setLineDash([]);
  };
  path(state.gtTrail.slice(-3000), "#8a9aab", 2, [6, 5]);
  // estimated trail, coloured by mode
  for (let i = 1; i < state.trail.length; i++) {
    const a = state.trail[i - 1], b = state.trail[i];
    ctx.strokeStyle = b[2] === Mode.GNSS_AIDED ? "#3b8bff" : "#f0a020"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(...toScr(a[0], a[1])); ctx.lineTo(...toScr(b[0], b[1])); ctx.stroke();
  }
  if (eng && eng.ltp) {
    ctx.fillStyle = "#2fb67c";
    for (const f of state.fixes.slice(-120)) {
      const [e, n] = eng.ltp.toEnu(f.lat, f.lon); const [x, y] = toScr(e, n);
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2 * Math.PI); ctx.fill();
    }
  }
  if (ps) {
    const [x, y] = toScr(ps[0], ps[1]);
    const sd = state.cur.positionStdM * k;
    ctx.fillStyle = "rgba(59,139,255,0.12)"; ctx.beginPath(); ctx.arc(x, y, Math.max(sd, 6), 0, 2 * Math.PI); ctx.fill();
    ctx.save(); ctx.translate(x, y); ctx.rotate((ps[2] * Math.PI) / 180);
    ctx.fillStyle = state.cur.mode === Mode.GNSS_AIDED ? "#3b8bff" : "#f0a020";
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, -14); ctx.lineTo(9, 10); ctx.lineTo(0, 5); ctx.lineTo(-9, 10); ctx.closePath();
    ctx.fill(); ctx.stroke(); ctx.restore();
  }
  // scale bar
  const m = [10, 20, 50, 100, 200, 500, 1000, 2000].find((v) => v * k > 60) || 5000;
  ctx.fillStyle = "#8a9aab"; ctx.fillRect(16, H - 150, m * k, 3); ctx.font = "11px system-ui";
  ctx.fillText(m >= 1000 ? `${m / 1000} km` : `${m} m`, 16, H - 156);
}

function updateHud() {
  const c = state.cur;
  if (!c) return;
  const [label, color] = MODE_STYLE[c.mode] || [c.mode, "var(--muted)"];
  const mode = $("mode"); mode.textContent = label; mode.style.background = color;
  $("speed").innerHTML = `${c.speed.toFixed(0)}<small>km/h</small>`;
  $("headingBox").textContent = `${c.heading.toFixed(0)}° · ${c.latitude !== null ? `${c.latitude.toFixed(5)}, ${c.longitude.toFixed(5)}` : "waiting for first fix"}`;
  $("gnssQ").textContent = `${(c.gnssQuality * 100).toFixed(0)}%`; $("gnssBar").style.width = `${c.gnssQuality * 100}%`;
  $("aiSpeed").textContent = c.aiSpeed === null ? "—" : `${c.aiSpeed.toFixed(0)} km/h`;
  $("aiBar").style.width = `${c.aiConfidence * 100}%`;
  $("mapQ").textContent = `${(c.mapConfidence * 100).toFixed(0)}%`; $("mapBar").style.width = `${c.mapConfidence * 100}%`;
  if (state.replay && state.lastErr !== undefined) {
    $("errBox").firstChild.textContent = "Error vs GT";
    $("posStd").textContent = `${state.lastErr.toFixed(1)} m`;
    $("errBar").style.width = `${Math.min(100, state.lastErr * 2)}%`;
  } else {
    $("posStd").textContent = `±${c.positionStdM.toFixed(1)} m`;
    $("errBar").style.width = `${Math.min(100, c.positionStdM * 2)}%`;
  }
  if (state.replay) $("progress").firstChild.style.width = `${(100 * state.frame) / state.replay.frames}%`;
}

let lastTick = performance.now();
async function loop(now) {
  const dt = (now - lastTick) / 1000; lastTick = now;
  if (state.source === "replay" && state.playing) {
    state.simT += dt * state.rate;
    await stepReplay(12);
  }
  draw(); updateHud();
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ controls
function bindControls() {
  $("play").onclick = async () => {
    if (!state.replay) return;
    if (state.frame >= state.replay.frames) { state.frame = 0; state.simT = 0; await newEngine(); }
    state.playing = !state.playing;
    if (state.playing) state.simT = Math.max(state.simT, state.frame / state.replay.rate_hz);
    $("play").textContent = state.playing ? "❚❚ Pause" : "▶ Play";
  };
  $("rateSeg").querySelectorAll("button").forEach((b) => (b.onclick = () => {
    state.rate = +b.dataset.r;
    $("rateSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  }));
  $("tunnel").onclick = () => { state.tunnel = !state.tunnel; $("tunnel").classList.toggle("on", state.tunnel); };
  $("follow").onclick = () => { state.follow = !state.follow; $("follow").classList.toggle("on", state.follow); };
  $("logBtn").onclick = () => $("log").classList.toggle("show");
  $("srcReplay").onclick = async () => {
    stopLive(); state.source = "replay";
    $("srcReplay").classList.add("on"); $("srcLive").classList.remove("on");
    await loadReplay(state.trips[+$("tripSel").value]);
  };
  $("srcLive").onclick = async () => {
    state.source = "live";
    $("srcLive").classList.add("on"); $("srcReplay").classList.remove("on");
    await startLive();
  };
  $("tripSel").onchange = () => loadReplay(state.trips[+$("tripSel").value]);
  // pan & zoom
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    state.follow = false; $("follow").classList.remove("on");
    state.center[0] -= (e.clientX - drag[0]) / state.scale; state.center[1] += (e.clientY - drag[1]) / state.scale;
    drag = [e.clientX, e.clientY];
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault(); state.scale = Math.min(20, Math.max(0.05, state.scale * Math.exp(-e.deltaY * 0.0015)));
  }, { passive: false });
  addEventListener("keydown", (e) => { if (e.key === "t") $("tunnel").click(); if (e.key === " ") $("play").click(); });
}

async function main() {
  resize(); addEventListener("resize", resize);
  bindControls(); renderLog();
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
  try {
    state.trips = await (await fetch("./replay/index.json")).json();
    $("tripSel").innerHTML = state.trips.map((t, i) => `<option value="${i}">${t.trip} (${t.minutes} min)</option>`).join("");
    await loadReplay(state.trips[0]);
  } catch (e) {
    console.error(e); toast("No replay data found - run src/web/export_replay.py", 5000);
  }
  requestAnimationFrame(loop);
}

main();
