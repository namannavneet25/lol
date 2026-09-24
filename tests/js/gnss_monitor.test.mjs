// JS GNSS monitor == Python monitor on random dense/sparse fix sequences (node tests/js/gnss_monitor.test.mjs).
// Generate the fixture first: python tests/js/make_gnss_fixture.py
import { readFileSync } from "node:fs";
import { GNSSQualityMonitor } from "../../src/web/nav/js/gnss.js";

const seqs = JSON.parse(readFileSync(new URL("./gnss_fixture.json", import.meta.url)));
let events = 0, mismatches = 0, reanchors = 0, sparseReanchors = 0;
for (const seq of seqs) {
  const mon = new GNSSQualityMonitor();
  let lastFixT = null;
  for (const ev of seq) {
    events++;
    if (ev.type === "tick") {
      if (mon.tick(ev.t) !== ev.mode) mismatches++;
    } else {
      const d = mon.onFix(ev.t, ev.e, ev.n, ev.speed, ev.acc, ev.nis, ev.fs);
      if (d !== ev.decision || mon.mode !== ev.mode) {
        if (mismatches < 5) console.error("mismatch", ev, d, mon.mode);
        mismatches++;
      }
      if (d === "reanchor") { reanchors++; if (lastFixT !== null && ev.t - lastFixT > 5) sparseReanchors++; }
      lastFixT = ev.t;
    }
  }
}
console.log(`${seqs.length} sequences, ${events} events, ${reanchors} re-anchors (${sparseReanchors} after gaps > 5 s), ${mismatches} mismatches`);
if (mismatches || sparseReanchors === 0) { console.error("GNSS monitor parity FAILED"); process.exit(1); }
console.log("GNSS monitor JS/Python parity OK");
