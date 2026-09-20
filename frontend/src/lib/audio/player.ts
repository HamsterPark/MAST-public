// Gapless PCM playback via scheduled AudioBufferSourceNodes. Incoming 24 kHz
// PCM16 frames are declared as 24k AudioBuffers; the Web Audio graph resamples to
// the hardware rate automatically, so we never force the AudioContext rate.
//
// clear() implements barge-in: stop everything scheduled/playing immediately.

import { pcm16ToFloat32 } from "./pcm";

export class PcmPlayer {
  private ctx: AudioContext | null = null;
  private nextTime = 0;
  private readonly sources = new Set<AudioBufferSourceNode>();
  private readonly rate: number;

  constructor(rate = 24000) {
    this.rate = rate;
  }

  private ensure(): AudioContext {
    if (!this.ctx || this.ctx.state === "closed") {
      this.ctx = new AudioContext();
      this.nextTime = 0;
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  /** Create + resume the context inside a user gesture so later (non-gesture)
   *  enqueue() calls aren't blocked by the browser autoplay policy. */
  prime(): void {
    this.ensure();
  }

  enqueue(pcm16: ArrayBuffer): void {
    const float = pcm16ToFloat32(pcm16);
    if (float.length === 0) return;
    const ctx = this.ensure();
    const buf = ctx.createBuffer(1, float.length, this.rate);
    buf.getChannelData(0).set(float);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const start = Math.max(ctx.currentTime + 0.02, this.nextTime);
    src.start(start);
    this.nextTime = start + buf.duration;
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  /** True while audio is scheduled/playing. */
  get playing(): boolean {
    return this.sources.size > 0;
  }

  /** barge-in / stop: kill everything now. */
  clear(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.nextTime = this.ctx ? this.ctx.currentTime : 0;
  }

  async close(): Promise<void> {
    this.clear();
    try {
      await this.ctx?.close();
    } catch {
      /* ignore */
    }
    this.ctx = null;
  }
}
