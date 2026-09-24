// Speed model (Tiny TCN) running in the browser with onnxruntime-web (WASM, single thread, offline).
// AIOutput = { speed (m/s), variance (m/s)^2, pStationary }.

export class SpeedModel {
  static async load(baseUrl = "./model/") {
    const ort = await import("../../vendor/onnxruntime-web/ort.wasm.min.mjs");
    ort.env.wasm.wasmPaths = new URL("../../vendor/onnxruntime-web/", import.meta.url).href;
    ort.env.wasm.numThreads = 1;
    const cfg = await (await fetch(baseUrl + "model_config.json")).json();
    const session = await ort.InferenceSession.create(baseUrl + cfg.onnx, { executionProviders: ["wasm"] });
    return new SpeedModel(ort, session, cfg);
  }

  constructor(ort, session, cfg) {
    this.ort = ort; this.session = session; this.cfg = cfg;
    const nm = cfg.normalization;
    this.mean = nm.mean; this.std = nm.std; this.scalarScale = nm.scalar_scale;
    this.window = nm.window_size; this.channels = nm.channel_names.length;
  }

  // windowRows: array of Float32Array(C) oldest..newest; scalars: [vRef, tSinceRef, refValid]
  async predict(windowRows, scalars) {
    const T = this.window, C = this.channels;
    const x = new Float32Array(C * T);
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) x[c * T + t] = (windowRows[t][c] - this.mean[c]) / this.std[c];
    const s = Float32Array.from(scalars.map((v, i) => v / this.scalarScale[i]));
    const out = await this.session.run({
      imu_window: new this.ort.Tensor("float32", x, [1, C, T]),
      ref_scalars: new this.ort.Tensor("float32", s, [1, 3]),
    });
    const speed = out.speed.data[0], logVar = out.log_var.data[0], stat = out.stat_logit.data[0];
    return { speed, variance: Math.exp(logVar), pStationary: 1 / (1 + Math.exp(-stat)) };
  }
}
