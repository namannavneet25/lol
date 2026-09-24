// End-to-end test of the IDR web apps in headless Chrome (uses the system Chrome via puppeteer-core).
//   cd tests/e2e && npm install && node ui_e2e.mjs      (needs: python src/web/setup_vendor.py,
//                                                        python src/web/export_replay.py --trip S1)
// Starts src/web/server.py itself, checks the navigator (replay, simulated tunnel, recovery, live mode)
// and the calibration upload (same-origin accepted, cross-origin blocked), then writes results/ui_e2e.json.
// calibration.json is backed up and restored.
import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";      // the local server uses a self-signed certificate
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const PORT = 8547, BASE = `https://127.0.0.1:${PORT}`;
const WIN = process.platform === "win32";
const VENV_PY = join(ROOT, WIN ? ".venv/Scripts/python.exe" : ".venv/bin/python");
const PY = existsSync(VENV_PY) ? VENV_PY : (WIN ? "python" : "python3");
const CHROME = process.env.CHROME || {
  win32: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  darwin: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
}[process.platform] || "/usr/bin/google-chrome";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = { checks: [], started: new Date().toISOString() };
let failures = 0;
const check = (name, ok, detail = "") => {
  results.checks.push({ name, ok: !!ok, detail });
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
};

const calPath = join(ROOT, "calibration.json"), calBackup = join(ROOT, ".calibration.e2e-backup.json");
const hadCal = existsSync(calPath);
if (hadCal) copyFileSync(calPath, calBackup);

const server = spawn(PY, ["src/web/server.py", "--port", String(PORT), "--ip", "127.0.0.1"], { cwd: ROOT, stdio: "ignore" });
let browser;
try {
  for (let i = 0; i < 40; i++) {                        // wait for the server
    try { await fetch(`${BASE}/nav/`); break; } catch { await sleep(250); }
  }
  browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
    args: ["--ignore-certificate-errors", "--no-sandbox", "--window-size=430,900"] });
  await browser.defaultBrowserContext().overridePermissions(BASE, ["geolocation", "accelerometer", "gyroscope"]);
  const page = await browser.newPage();
  await page.setViewport({ width: 430, height: 900 });
  await page.setGeolocation({ latitude: 52.4017, longitude: -1.5053, accuracy: 5 });
  const errors = [], failedRequests = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error" && !/cpuid_info|Unknown CPU vendor|Failed to load resource/.test(m.text())) errors.push(m.text());
  });
  // the optional offline live map may legitimately be absent (404); any other failed request is an error
  page.on("response", (r) => { if (r.status() >= 400 && !r.url().endsWith("/maps/live.geojson")) failedRequests.push(`${r.status()} ${r.url()}`); });

  // ---------------------------------------------------------------- navigator: replay
  await page.goto(`${BASE}/nav/`, { waitUntil: "networkidle0" });
  await page.waitForFunction(() => document.querySelector("#tripSel").options.length > 0, { timeout: 20000 });
  await sleep(1500);
  const hud = () => page.evaluate(() => ({ mode: document.querySelector("#mode").textContent,
    rows: [...document.querySelectorAll("#logBody tr")].map((r) => r.innerText.replace(/\s+/g, " ").trim()) }));
  const simT = () => page.evaluate(() => performance.now());
  await page.click('#rateSeg button[data-r="16"]');
  await page.click("#play");
  const t0 = await simT();
  await sleep(10000);
  let h = await hud();
  check("replay runs GNSS-aided before the tunnel", h.mode === "GNSS-AIDED", h.mode);
  await page.click("#tunnel");                           // GNSS off
  await sleep(1500);
  h = await hud();
  check("dead reckoning after GNSS is removed", h.mode === "DEAD RECKONING", h.mode);
  await sleep(3000);
  await page.click("#tunnel");                           // GNSS back
  await sleep(2500);
  h = await hud();
  check("GNSS-aided again after recovery", h.mode === "GNSS-AIDED", h.mode);
  const row = h.rows.find((r) => /%/.test(r));
  check("outage logged with a drift value", !!row, row || JSON.stringify(h.rows));
  const frames = await page.evaluate(() => document.querySelector("#progress i").style.width);
  const wall = ((await simT()) - t0) / 1000;
  const progress = parseFloat(frames) / 100;                        // share of the 30-min (1800 s) replay done
  const speedup = (progress * 1800) / wall;
  check("replay keeps up with the 16x setting", speedup > 12, `${speedup.toFixed(1)}x real time over ${wall.toFixed(0)} s`);
  results.replay = { speedup: +speedup.toFixed(1), outage_row: row || null };
  await page.screenshot({ path: join(ROOT, "results/ui_e2e_replay.png") });

  // ---------------------------------------------------------------- navigator: live mode (no sensors headless)
  await page.click("#srcLive");
  await sleep(2500);
  const toast = await page.evaluate(() => document.querySelector("#toast").textContent);
  check("live mode reports whether an offline map is loaded", /offline map|map matching off/.test(toast), toast);
  // live mode again with an offline map installed (temporarily reuse the S1 replay map as live.geojson)
  const liveMap = join(ROOT, "src/web/nav/maps/live.geojson");
  const tempMap = !existsSync(liveMap);
  if (tempMap) copyFileSync(join(ROOT, "src/web/nav/maps/S1.geojson"), liveMap);
  try {
    await page.click("#srcReplay");
    await sleep(2500);
    await page.click("#srcLive");
    await sleep(2500);
    const toast2 = await page.evaluate(() => document.querySelector("#toast").textContent);
    check("live mode loads the offline map when present", /offline map \(\d+ roads\)/.test(toast2), toast2);
  } finally { if (tempMap) rmSync(liveMap); }
  check("no page errors in the navigator", errors.length === 0, errors.join(" | "));
  check("no failed requests in the navigator", failedRequests.length === 0, failedRequests.join(" | "));

  // ---------------------------------------------------------------- calibration upload
  await page.goto(`${BASE}/calibration/`, { waitUntil: "networkidle0" });
  const cal = { is_calibrated: true, resting_gravity_vector: { x: 0, y: 0.1, z: 9.8 },
    gyro_bias_rad_s: { gx: 0, gy: 0, gz: 0.001 } };
  const same = await page.evaluate(async (body) => (await fetch("/api/save_calibration", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).status, cal);
  check("calibrator page can save (same origin)", same === 200, `HTTP ${same}`);
  const bad = await page.evaluate(async () => (await fetch("/api/save_calibration", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resting_gravity_vector: { x: 0, y: 0, z: 0 } }) })).status);
  check("invalid calibration is rejected", bad === 400, `HTTP ${bad}`);
  const calBefore = await (await fetch(`${BASE}/api/calibration`)).text();
  const other = await browser.newPage();
  await other.goto("data:text/html,<p>other site</p>");
  const cross = await other.evaluate(async (url) => {
    try {
      const r = await fetch(url, { method: "POST", headers: { "Content-Type": "text/plain" }, body: "{}" });
      return `HTTP ${r.status}`;
    } catch (e) { return "blocked"; }
  }, `${BASE}/api/save_calibration`);
  const calAfter = await (await fetch(`${BASE}/api/calibration`)).text();
  check("cross-origin page cannot write calibration", calAfter === calBefore,
    `browser: ${cross}; calibration.json ${calAfter === calBefore ? "unchanged" : "CHANGED"}`);
} finally {
  if (browser) await browser.close();
  server.kill();
  if (hadCal) { copyFileSync(calBackup, calPath); rmSync(calBackup); } else if (existsSync(calPath)) rmSync(calPath);
  mkdirSync(join(ROOT, "results"), { recursive: true });
  results.failures = failures;
  writeFileSync(join(ROOT, "results/ui_e2e.json"), JSON.stringify(results, null, 1));
}
console.log(failures ? `${failures} check(s) failed` : "All UI end-to-end checks passed");
process.exit(failures ? 1 : 0);
