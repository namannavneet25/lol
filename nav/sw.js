// Offline cache for the IDR Navigator PWA: the app shell, model, WASM runtime, maps and replays.
const CACHE = "idr-nav-v1";
const SHELL = ["./", "./index.html", "./manifest.json", "./icon.svg", "./js/app.js", "./js/pipeline.js",
  "./js/model.js", "./js/ekf.js", "./js/gnss.js", "./js/mapmatch.js", "./js/features.js", "./js/frames.js",
  "./model/model_config.json", "./model/idr_tcn.onnx", "./replay/index.json",
  "../vendor/onnxruntime-web/ort.wasm.min.mjs", "../vendor/onnxruntime-web/ort-wasm-simd-threaded.mjs",
  "../vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm"];

self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
// Network first (fresh code while developing), falling back to the cache when offline.
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET" || new URL(e.request.url).pathname.startsWith("/api/")) return;
  e.respondWith(fetch(e.request).then((r) => {
    if (r.ok) { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request)));
});
