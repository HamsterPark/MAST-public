// ════════════════════════════════════════════════════════════════════════════
// Pure-logic tests for src/lib/chartSize.ts — how a plot sizes itself to the
// space it was given.
//
// Same house rules as scanMapView.test.ts: no test framework is installed in
// this frontend, so these run on `node --test` with Node's native TypeScript
// stripping:
//
//     cd frontend && npm run test:unit
//
// Why this arithmetic is worth pinning: every failure here is silent. uPlot
// handed width 0 draws a canvas with no plotting area and never recovers; a
// height that ignores its floor renders a few pixels of axis and looks like a
// chart that "didn't load". Neither throws, and neither is obvious in a
// screenshot — which is exactly how the app shipped six hardcoded sizes that no
// caller overrode (/ #58).
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { MAP_RATIO, MIN_CHART_W, TREND_RATIO, sizeChart } from "../src/lib/chartSize.ts";

const TREND = { ratio: TREND_RATIO, minHeight: 240, maxHeight: 520 };

describe("sizeChart — width", () => {
  it("uses the measured width when there is one", () => {
    assert.equal(sizeChart(1200, 860, TREND).width, 1200);
  });

  it("falls back before the first measurement rather than collapsing to 0", () => {
    // null is the pre-ResizeObserver state. Zero width is the one value that
    // leaves uPlot permanently broken, so it must never be produced.
    assert.equal(sizeChart(null, 860, TREND).width, 860);
    assert.equal(sizeChart(undefined, 860, TREND).width, 860);
    assert.equal(sizeChart(0, 860, TREND).width, 860);
  });

  it("ignores a measurement that is not a finite positive number", () => {
    assert.equal(sizeChart(NaN, 860, TREND).width, 860);
    assert.equal(sizeChart(-40, 860, TREND).width, 860);
    assert.equal(sizeChart(Infinity, 860, TREND).width, 860);
  });

  it("floors at MIN_CHART_W — a narrow pane scrolls, it does not squeeze", () => {
    assert.equal(sizeChart(120, 860, TREND).width, MIN_CHART_W);
  });

  it("honours an explicit minWidth over the default floor", () => {
    assert.equal(sizeChart(100, 860, { ...TREND, minWidth: 500 }).width, 500);
  });

  it("rounds to whole pixels — flexbox reports fractions", () => {
    assert.equal(sizeChart(863.4062, 860, TREND).width, 863);
  });
});

describe("sizeChart — height", () => {
  it("follows the ratio between the two clamps", () => {
    // 1000 * 0.3 = 300, inside [240, 520].
    assert.equal(sizeChart(1000, 860, TREND).height, 300);
  });

  it("clamps to minHeight on a narrow container", () => {
    // 400 * 0.3 = 120 — below the floor.
    assert.equal(sizeChart(400, 860, TREND).height, 240);
  });

  it("clamps to maxHeight on a very wide container", () => {
    // 2560 * 0.3 = 768 — above the cap. Without this an ultrawide monitor gets
    // a chart taller than the window.
    assert.equal(sizeChart(2560, 860, TREND).height, 520);
  });

  it("caps at 70% of the viewport so the x axis stays above the fold", () => {
    // Ratio wants 480; a 600 px window allows 420.
    assert.equal(sizeChart(1600, 860, { ...TREND, viewportH: 600 }).height, 420);
  });

  it("never lets the viewport cap push height below minHeight", () => {
    // 70% of 200 is 140, under the 240 floor. A tiny window is not a reason to
    // render an unreadable sliver.
    assert.equal(sizeChart(1600, 860, { ...TREND, viewportH: 200 }).height, 240);
  });

  it("ignores a nonsense viewport height", () => {
    const want = sizeChart(1600, 860, TREND).height;
    assert.equal(sizeChart(1600, 860, { ...TREND, viewportH: 0 }).height, want);
    assert.equal(sizeChart(1600, 860, { ...TREND, viewportH: NaN }).height, want);
    assert.equal(sizeChart(1600, 860, { ...TREND, viewportH: null }).height, want);
  });
});

describe("sizeChart — invariants that hold for any input", () => {
  const widths = [null, 0, 1, 120, 320, 640, 860, 1440, 2560, 5120, 863.5];
  const viewports = [null, 200, 600, 900, 1440];

  it("always returns a positive, finite, integral box", () => {
    for (const w of widths) {
      for (const vh of viewports) {
        for (const opts of [TREND, { ratio: MAP_RATIO, minHeight: 320, maxHeight: 900 }]) {
          const box = sizeChart(w, 620, { ...opts, viewportH: vh });
          assert.ok(Number.isInteger(box.width), `width not integral for ${w}`);
          assert.ok(Number.isInteger(box.height), `height not integral for ${w}`);
          assert.ok(box.width > 0 && Number.isFinite(box.width), `bad width for ${w}`);
          assert.ok(box.height > 0 && Number.isFinite(box.height), `bad height for ${w}`);
          assert.ok(box.width >= MIN_CHART_W, `below floor for ${w}`);
          assert.ok(box.height >= opts.minHeight, `below height floor for ${w}/${vh}`);
          assert.ok(box.height <= opts.maxHeight, `above height cap for ${w}/${vh}`);
        }
      }
    }
  });

  it("is monotonic in width — a wider pane never yields a narrower chart", () => {
    let prev = 0;
    for (const w of [320, 400, 640, 860, 1440, 2560, 5120]) {
      const { width } = sizeChart(w, 620, TREND);
      assert.ok(width >= prev, `width went backwards at ${w}`);
      prev = width;
    }
  });
});
