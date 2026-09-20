// AudioWorklet capture processor, authored as a source STRING and registered via
// a Blob URL (audioWorklet.addModule). Blob-URL registration is bundler-agnostic
// and CSP/LAN-friendly — it sidesteps Vite's worklet-asset handling entirely and
// works the same in dev and in the packaged offline app.
//
// The processor runs in the AudioWorkletGlobalScope (no window / imports). It
// coalesces the tiny 128-sample render quanta into ~2048-sample Float32 batches
// and posts them (transferring the buffer) to the main thread, which resamples
// 48k→16k and quantizes to PCM16 before sending on the WS.

export const CAPTURE_WORKLET_SOURCE = `
class MastCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._n = 0;
    this._target = 2048;
  }
  process(inputs) {
    const input = inputs[0];
    const ch = input && input[0];
    if (ch && ch.length) {
      this._buf.push(ch.slice(0)); // copy: the render buffer is reused
      this._n += ch.length;
      if (this._n >= this._target) {
        const merged = new Float32Array(this._n);
        let o = 0;
        for (let i = 0; i < this._buf.length; i++) {
          merged.set(this._buf[i], o);
          o += this._buf[i].length;
        }
        this.port.postMessage(merged, [merged.buffer]);
        this._buf = [];
        this._n = 0;
      }
    }
    return true; // keep the processor alive
  }
}
registerProcessor('mast-capture', MastCaptureProcessor);
`;
