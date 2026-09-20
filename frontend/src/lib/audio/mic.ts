// Microphone capture → 16 kHz PCM16 frames. Uses an AudioWorklet (via Blob URL)
// so audio processing stays off the main thread. echoCancellation is ON so the
// speaker's TTS doesn't feed straight back into the mic (matters for barge-in).

import { CAPTURE_WORKLET_SOURCE } from "./workletSource";
import { floatTo16BitPCM, resampleLinear } from "./pcm";

export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private readonly onPcm: (buf: ArrayBuffer) => void;
  private readonly targetRate: number;
  private readonly onLevel?: (rms: number) => void;

  constructor(
    onPcm: (buf: ArrayBuffer) => void,
    targetRate = 16000,
    onLevel?: (rms: number) => void,
  ) {
    this.onPcm = onPcm;
    this.targetRate = targetRate;
    this.onLevel = onLevel;
  }

  get active(): boolean {
    return this.node != null;
  }

  async start(): Promise<void> {
    if (this.node) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    this.ctx = new AudioContext();
    const url = URL.createObjectURL(
      new Blob([CAPTURE_WORKLET_SOURCE], { type: "application/javascript" }),
    );
    try {
      await this.ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "mast-capture");
    const inRate = this.ctx.sampleRate;
    this.node.port.onmessage = (e: MessageEvent) => {
      const float = e.data as Float32Array;
      if (this.onLevel) {
        let sum = 0;
        for (let i = 0; i < float.length; i++) sum += float[i]! * float[i]!;
        this.onLevel(Math.sqrt(sum / Math.max(1, float.length)));
      }
      const down = resampleLinear(float, inRate, this.targetRate);
      this.onPcm(floatTo16BitPCM(down).buffer as ArrayBuffer);
    };
    src.connect(this.node);
    // Keep the worklet scheduled: connect to a muted sink (never audible — do NOT
    // connect the mic to destination or it would echo through the speakers).
    const sink = this.ctx.createGain();
    sink.gain.value = 0;
    this.node.connect(sink);
    sink.connect(this.ctx.destination);
  }

  async stop(): Promise<void> {
    try {
      this.node?.disconnect();
    } catch {
      /* ignore */
    }
    try {
      this.stream?.getTracks().forEach((t) => t.stop());
    } catch {
      /* ignore */
    }
    try {
      await this.ctx?.close();
    } catch {
      /* ignore */
    }
    this.node = null;
    this.stream = null;
    this.ctx = null;
  }
}
