// 曲线几何 — turning a sample series into SVG polyline points inside a unit box.
//
// Deliberately NOT shared with components/vision/Sparkline.tsx even though the
// arithmetic rhymes. Two differences that matter:
//
//   · Sparkline pins a constant series to the FLOOR (`rng = 1` when flat), which
//     is right for an events/s strip where flat means zero. For a current trace
//     flat does not mean zero — a steady 30 pA drawn along the bottom edge reads
//     as "no signal", which is a lie about the one quantity the monitor exists
//     to report. A constant series is centred here.
//   · this one emits a UNIT box (0..100 × 0..100) drawn with
//     `preserveAspectRatio="none"`, so the SVG scales to whatever the tile got
//     without anyone measuring pixels. Sparkline takes explicit px.
//
// Non-finite samples are dropped rather than plotted at 0: the monitor stores
// nulls for readings it did not get, and a gap drawn as zero is a fabricated
// excursion in exactly the channel that triggers CRITICALs.

export const CURVE_BOX = 100;

export interface CurveGeometry {
  /** `x,y x,y …` for an SVG <polyline>, in a 0..100 box. Empty when unplottable. */
  points: string;
  /** Smallest and largest finite sample, for the caption. Null when none. */
  min: number | null;
  max: number | null;
  /** Finite samples actually plotted. */
  n: number;
}

export function curveGeometry(values: readonly (number | null | undefined)[]): CurveGeometry {
  const ys: number[] = [];
  for (const v of values) {
    if (typeof v === "number" && Number.isFinite(v)) ys.push(v);
  }
  if (ys.length < 2) {
    return { points: "", min: ys.length ? ys[0]! : null, max: ys.length ? ys[0]! : null, n: ys.length };
  }
  let min = ys[0]!;
  let max = ys[0]!;
  for (const v of ys) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min;
  const pad = 2; // keep the stroke off the box edge
  const usable = CURVE_BOX - 2 * pad;
  const pts = ys.map((v, i) => {
    const x = (i * CURVE_BOX) / (ys.length - 1);
    // A flat series sits in the MIDDLE — see the header. Anything else is
    // normalised into the padded band, y inverted for SVG's downward axis.
    const frac = span === 0 ? 0.5 : (v - min) / span;
    const y = CURVE_BOX - pad - frac * usable;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return { points: pts.join(" "), min, max, n: ys.length };
}
