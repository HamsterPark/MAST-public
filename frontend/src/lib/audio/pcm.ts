// PCM / resampling helpers for the voice pipeline. Pure, dependency-free.
//
// The voice WS carries raw little-endian PCM16 mono: mic up @ 16 kHz, TTS down
// @ 24 kHz. Browsers capture at the hardware rate (usually 48 kHz) as Float32, so
// we downsample + quantize on the way up and de-quantize on the way down.

/** Float32 [-1,1] → Int16 PCM. */
export function floatTo16BitPCM(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i] ?? 0));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/** Int16 PCM (as an ArrayBuffer) → Float32 [-1,1]. */
export function pcm16ToFloat32(buf: ArrayBuffer): Float32Array {
  const i16 = new Int16Array(buf);
  const out = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) out[i] = (i16[i] ?? 0) / 0x8000;
  return out;
}

/** Linear-interpolation mono resample inRate → outRate. Good enough for speech;
 *  avoids pulling in an FFT/polyphase dependency. */
export function resampleLinear(
  input: Float32Array,
  inRate: number,
  outRate: number,
): Float32Array {
  if (inRate === outRate || input.length === 0) return input;
  const ratio = inRate / outRate;
  const outLen = Math.max(1, Math.floor(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = pos - i0;
    out[i] = (input[i0] ?? 0) * (1 - frac) + (input[i1] ?? 0) * frac;
  }
  return out;
}
