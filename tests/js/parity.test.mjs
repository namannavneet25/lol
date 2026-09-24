// JS engine parity with the Python reference (node tests/js/parity.test.mjs).
// The model is stubbed with the outputs Python recorded. The stub also checks that the JS engine
// computes the same model inputs (levelled window features + reference scalars) as Python on every
// 20th call, then the EKF, GNSS state machine and HMM map matching are compared frame by frame.
import { readFileSync } from "node:fs";
import { CoreNavigationPipeline } from "../../src/web/nav/js/pipeline.js";
import { compassToPsi, psiToCompass, LocalTangentPlane, wrapPi } from "../../src/web/nav/js/frames.js";

let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.error("FAIL:", msg); } };

// unit checks
check(Math.abs(wrapPi(compassToPsi(90) - 0)) < 1e-12, "compass 90 -> psi 0");
check(Math.abs(psiToCompass(Math.PI / 2)) < 1e-9, "psi pi/2 -> compass 0");
const ltp = new LocalTangentPlane(52.4, -1.5);
const [e, n] = ltp.toEnu(52.45, -1.4);
const [la, lo] = ltp.toGeodetic(e, n);
check(Math.abs(la - 52.45) < 1e-7 && Math.abs(lo + 1.4) < 1e-7, "LTP round trip");

const fx = JSON.parse(readFileSync(new URL("./fixture.json", import.meta.url)));
let call = 0, maxFeat = 0, maxScalar = 0, checkedInputs = 0;
const stub = { window: fx.window, async predict(windowRows, scalars) {
  const ref = fx.ai_inputs[String(call)];
  if (ref) {
    checkedInputs++;
    ref.window.forEach((row, t) => row.forEach((v, c) => { maxFeat = Math.max(maxFeat, Math.abs(windowRows[t][c] - v)); }));
    ref.scalars.forEach((v, i) => { maxScalar = Math.max(maxScalar, Math.abs(scalars[i] - v)); });
  }
  const [speed, variance, pStationary] = fx.ai[call++]; return { speed, variance, pStationary };
} };
const eng = new CoreNavigationPipeline(stub);
let maxPos = 0, maxPsi = 0, maxV = 0, modeMismatch = 0;
for (let i = 0; i < fx.frames.length; i++) {
  const fr = fx.frames[i];
  if (!eng.ltp && fr.gnss) { eng.setOrigin(fr.gnss.lat, fr.gnss.lon); eng.setMap(fx.map); }
  const st = await eng.processFrame(fr);
  const [pe, pn, psi, v, mode] = fx.states[i];
  maxPos = Math.max(maxPos, Math.hypot(eng.ekf.x[0] - pe, eng.ekf.x[1] - pn));
  maxPsi = Math.max(maxPsi, Math.abs(wrapPi(eng.ekf.x[2] - psi)));
  maxV = Math.max(maxV, Math.abs(eng.ekf.x[3] - v));
  if (st.mode !== mode) modeMismatch++;
}
check(call === fx.ai.length, `model call count ${call} vs ${fx.ai.length}`);
check(checkedInputs === Object.keys(fx.ai_inputs).length && checkedInputs > 100, `checked ${checkedInputs} model inputs`);
check(maxFeat < 1e-5, `max model feature difference ${maxFeat}`);     // float32 features
check(maxScalar < 1e-9, `max reference-scalar difference ${maxScalar}`);
check(maxPos < 0.01, `max position difference ${maxPos} m`);
check(maxPsi < 1e-4, `max heading difference ${maxPsi} rad`);
check(maxV < 1e-3, `max speed difference ${maxV} m/s`);
check(modeMismatch === 0, `${modeMismatch} frames with a different navigation mode`);
console.log(`model inputs checked ${checkedInputs}: max |dfeature| ${maxFeat.toExponential(2)}, max |dscalar| ${maxScalar.toExponential(2)}`);
console.log(`frames ${fx.frames.length}, model calls ${call}, max |dpos| ${maxPos.toExponential(2)} m, ` +
  `max |dpsi| ${maxPsi.toExponential(2)} rad, max |dv| ${maxV.toExponential(2)} m/s, mode mismatches ${modeMismatch}`);
if (failures) { console.error(`${failures} parity check(s) failed`); process.exit(1); }
console.log("JS/Python parity OK");
