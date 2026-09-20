// 图表尺寸 — how wide and how tall a plot should be drawn, given the space it
// actually got.
//
// Every chart and canvas in this app used to be a hardcoded pixel pair (860×300,
// 620×460, …) with no caller ever overriding it, so a 2560-wide monitor showed
// the same postage stamp as a laptop and the rest of the row was dead space
// (「扫描地图这个占网页的面积太小」, 「图的大小没有充分利用屏幕空间，
// 考虑到是浏览器，应该增加根据比例的自适应」).
//
// The measuring itself is a DOM concern and lives in hooks/useElementWidth.ts.
// What is here is only the arithmetic — which is the part that can be wrong in a
// way nobody notices (a chart 3 px tall still "renders"), so it is the part that
// gets tested.
//
// Why height is derived rather than measured: these plots sit in a scrolling
// page, so their container has no height to measure — it is whatever the chart
// takes. Tying height to width keeps the aspect sane at every viewport instead
// of leaving a 2400-wide, 300-tall letterbox. The viewport cap exists so a very
// wide window cannot push the x axis below the fold.

/** Below this the axes and labels stop fitting; scroll instead of shrinking. */
export const MIN_CHART_W = 320;

export interface ChartBox {
  width: number;
  height: number;
}

export interface SizeChartOpts {
  /** height / width. 0.35 is a wide trend strip; 0.74 is the map's old 620×460. */
  ratio: number;
  /** Never shorter than this, however narrow the container. */
  minHeight: number;
  /** Never taller than this, however wide the container. */
  maxHeight: number;
  /** Viewport height, when known. Height is additionally capped at 70% of it so
   *  the x axis stays above the fold on short windows. */
  viewportH?: number | null;
  /** Floor for width. Defaults to MIN_CHART_W. */
  minWidth?: number;
}

/**
 * Size a plot for the width its container reports.
 *
 * `measured` is null until the ResizeObserver has fired once (SSR, or the first
 * paint). Callers pass a fallback so the chart renders at a sane size on that
 * first frame rather than collapsing to zero — uPlot given width 0 draws a
 * canvas with no plotting area at all and never recovers on its own.
 */
export function sizeChart(
  measured: number | null | undefined,
  fallbackW: number,
  opts: SizeChartOpts,
): ChartBox {
  const minW = opts.minWidth ?? MIN_CHART_W;
  const raw = typeof measured === "number" && Number.isFinite(measured) && measured > 0
    ? measured
    : fallbackW;
  const width = Math.max(minW, Math.round(raw));

  let height = Math.round(width * opts.ratio);
  height = Math.max(opts.minHeight, Math.min(opts.maxHeight, height));
  // A short window (a laptop in a 13" lid, the browser sharing the screen with
  // Nanonis) matters more than the ratio: better a squat chart than one whose
  // time axis is off-screen.
  const vh = opts.viewportH;
  if (typeof vh === "number" && Number.isFinite(vh) && vh > 0) {
    height = Math.min(height, Math.max(opts.minHeight, Math.round(vh * 0.7)));
  }
  return { width, height };
}

/** Trend / spectrum strips: wide and short, they are read left-to-right. */
export const TREND_RATIO = 0.3;

/** The scan map is a spatial view — keep it close to the 620×460 it shipped as. */
export const MAP_RATIO = 0.74;

/** Small multiples (the aux channels): several stacked strips read as one block,
 *  so each has to stay short enough that the stack fits without scrolling. */
export const SMALL_MULTIPLE_RATIO = 0.14;
