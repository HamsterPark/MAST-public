// ════════════════════════════════════════════════════════════════════════════
// Pure-geometry tests for src/lib/curvePath.ts — the thumbnail curve drawn for a
// current-monitor event .
//
//     cd frontend && npm run test:unit
//
// A wrong curve is the worst possible failure here, because it is the picture
// attached to a CRITICAL: it will be believed. So the invariants pinned below
// are the ones whose violation still LOOKS like a plot — a gap rendered as zero,
// a steady current drawn along the floor, an inverted y axis.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { CURVE_BOX, curveGeometry } from "../src/lib/curvePath.ts";

/** `"x,y x,y"` → `[[x, y], …]` */
function pts(s: string): [number, number][] {
  return s
    .split(" ")
    .filter(Boolean)
    .map((p) => p.split(",").map(Number) as [number, number]);
}

describe("curveGeometry — basics", () => {
  it("spans the full width and stays inside the box", () => {
    const g = curveGeometry([1, 5, 2, 9, 3]);
    const p = pts(g.points);
    assert.equal(p.length, 5);
    assert.equal(p[0]![0], 0);
    assert.equal(p[4]![0], CURVE_BOX);
    for (const [x, y] of p) {
      assert.ok(x >= 0 && x <= CURVE_BOX, `x out of box: ${x}`);
      assert.ok(y >= 0 && y <= CURVE_BOX, `y out of box: ${y}`);
    }
  });

  it("puts the maximum ABOVE the minimum — SVG y grows downward", () => {
    const g = curveGeometry([0, 10]);
    const [lo, hi] = pts(g.points);
    assert.ok(hi![1] < lo![1], "the larger sample must sit higher on screen");
  });

  it("reports the real min/max for the caption", () => {
    const g = curveGeometry([3e-12, -7e-12, 1e-12]);
    assert.equal(g.min, -7e-12);
    assert.equal(g.max, 3e-12);
    assert.equal(g.n, 3);
  });
});

describe("curveGeometry — the ways a plot can lie", () => {
  it("drops non-finite samples instead of plotting them as zero", () => {
    // The monitor stores nulls for readings it did not get. Zero is a REAL
    // current value, so a gap drawn at zero is a fabricated excursion in the
    // exact channel that raises CRITICALs.
    const g = curveGeometry([5, null, 6, NaN, 7, undefined, Infinity]);
    assert.equal(g.n, 3);
    assert.equal(g.min, 5);
    assert.equal(g.max, 7);
    assert.equal(pts(g.points).length, 3);
  });

  it("centres a constant series rather than pinning it to the floor", () => {
    // A steady 30 pA drawn along the bottom edge reads as "no signal".
    const g = curveGeometry([3e-11, 3e-11, 3e-11]);
    for (const [, y] of pts(g.points)) {
      assert.ok(Math.abs(y - CURVE_BOX / 2) < 1, `flat trace should be centred, got y=${y}`);
    }
  });

  it("returns no points — not a degenerate line — below two samples", () => {
    for (const v of [[], [1], [null, NaN]]) {
      const g = curveGeometry(v as (number | null)[]);
      assert.equal(g.points, "", "caller must show a caption, not a fake plot");
    }
  });

  it("handles a negative-only series (the current can be either sign)", () => {
    const g = curveGeometry([-1e-12, -5e-12, -3e-12]);
    const p = pts(g.points);
    assert.equal(g.min, -5e-12);
    assert.equal(g.max, -1e-12);
    assert.ok(p[0]![1] < p[1]![1], "-1e-12 is the larger value → drawn higher");
  });

  it("keeps the stroke off the edge so a spike is not clipped away", () => {
    const g = curveGeometry([0, 1]);
    for (const [, y] of pts(g.points)) {
      assert.ok(y >= 1 && y <= CURVE_BOX - 1, `y=${y} touches the frame`);
    }
  });
});
