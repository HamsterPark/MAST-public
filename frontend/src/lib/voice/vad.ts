// Lightweight energy-based Voice Activity Detection for full-duplex mode.
//
// Drives endpointing (auto-commit on speech→silence) and barge-in (speech while
// the assistant is speaking). Dependency-free and fully offline — a deliberate
// choice over Silero/@ricky0123/vad-web (WASM + ONNX assets that are awkward to
// bundle for the packaged LAN app). With getUserMedia echoCancellation on, an RMS
// threshold + hangover is robust enough; the interface is kept minimal so a
// Silero implementation can be dropped in later behind the same callbacks.

export interface VadOptions {
  threshold?: number; // RMS (0..1) above which a frame counts as speech
  startFrames?: number; // consecutive speech frames to confirm speech start
  silenceMs?: number; // trailing silence to confirm speech end
  frameMs?: number; // approx ms represented by one push() sample
}

export class EnergyVad {
  private threshold: number;
  private readonly startFrames: number;
  private readonly silenceFrames: number;
  private speech = false;
  private speechCount = 0;
  private silenceCount = 0;

  onSpeechStart?: () => void;
  onSpeechEnd?: () => void;

  constructor(opts: VadOptions = {}) {
    this.threshold = opts.threshold ?? 0.02;
    this.startFrames = opts.startFrames ?? 3;
    const frameMs = opts.frameMs ?? 43; // ~2048 samples @ 48 kHz
    this.silenceFrames = Math.max(1, Math.round((opts.silenceMs ?? 700) / frameMs));
  }

  push(rms: number): void {
    if (rms >= this.threshold) {
      this.speechCount++;
      this.silenceCount = 0;
      if (!this.speech && this.speechCount >= this.startFrames) {
        this.speech = true;
        this.onSpeechStart?.();
      }
    } else {
      this.silenceCount++;
      this.speechCount = 0;
      if (this.speech && this.silenceCount >= this.silenceFrames) {
        this.speech = false;
        this.onSpeechEnd?.();
      }
    }
  }

  get active(): boolean {
    return this.speech;
  }

  setThreshold(t: number): void {
    this.threshold = t;
  }

  reset(): void {
    this.speech = false;
    this.speechCount = 0;
    this.silenceCount = 0;
  }
}
