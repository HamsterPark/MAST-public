// ════════════════════════════════════════════════════════════════════════════
// Pure-logic tests for src/lib/scanMapView.ts — which scan-map layers are
// drawn, when the analysis is refetched, and what area is in view.
//
// Same house rules as monitoring.test.ts: NO test framework is installed in
// this frontend and adding one is not our decision, so these run on
// `node --test` with Node's native TypeScript stripping:
//
//     cd frontend && npm run test:unit
//
// What that buys: the map's failure modes here are all silent. A view that
// still fits switched-off content looks like a map that "just won't zoom in";
// an analysis that never refetches looks like a keep-out zone that simply is
// not there; a persisted layer set missing a newly added key hides an overlay
// with no control to bring it back. None of those throw, and none of them are
// obvious in a screenshot.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  DEFAULT_LAYERS,
  MAP_LAYER_IDS,
  MAP_PRESETS,
  MIN_AUTO_ANALYSIS_INTERVAL_MS,
  analysisAutoEnabled,
  analysisSignature,
  boundsForMode,
  fmtNm,
  markerDetailRows,
  inverseStageTransform,
  isStaleEpoch,
  matchPreset,
  nextAnalysisDelayMs,
  niceStepNm,
  pinnedLabel,
  stageToScreen,
  visibleStepNm,
  withLayerDefaults,
  type MapLayers,
} from "../src/lib/scanMapView.ts";

// ── fixtures ────────────────────────────────────────────────────────────────

const nm = (v: number) => v * 1e-9;

function layers(over: Partial<MapLayers> = {}): MapLayers {
  return { ...DEFAULT_LAYERS, ...over };
}

function mapWith(over: Record<string, unknown> = {}) {
  return {
    frame: null,
    tip_xyz: null,
    markers: [],
    scan_images: [],
    current_epoch: 0,
    marker_count: 0,
    ...over,
  };
}

// ── layer sets and presets ──────────────────────────────────────────────────

describe("layer defaults", () => {
  it("starts with every layer on", () => {
    assert.equal(MAP_LAYER_IDS.every((id) => DEFAULT_LAYERS[id]), true);
  });

  it("fills in a layer the stored state predates", () => {
    // Exactly what an older localStorage looks like after a layer is added.
    const restored = withLayerDefaults({ underlay: false } as Partial<MapLayers>);
    assert.equal(restored.underlay, false, "the stored choice is kept");
    assert.equal(restored.upcoming, true, "the unknown layer defaults to visible");
  });

  it("ignores junk in stored state rather than reading it as off", () => {
    const restored = withLayerDefaults({ grid: "yes" } as unknown as Partial<MapLayers>);
    assert.equal(restored.grid, true);
  });

  it("survives no stored state at all", () => {
    assert.deepEqual(withLayerDefaults(null), DEFAULT_LAYERS);
    assert.deepEqual(withLayerDefaults(undefined), DEFAULT_LAYERS);
  });
});

describe("presets", () => {
  it("every preset assigns every layer", () => {
    for (const p of MAP_PRESETS) {
      for (const id of MAP_LAYER_IDS) {
        assert.equal(typeof p.layers[id], "boolean", `${p.id} is missing ${id}`);
      }
    }
  });

  it("精简 drops the crowding overlays but keeps the surface", () => {
    const p = MAP_PRESETS.find((x) => x.id === "minimal")!;
    assert.equal(p.layers.underlay, true);
    assert.equal(p.layers.history, true);
    assert.equal(p.layers.grid, false);
    assert.equal(p.layers.plan, false);
    assert.equal(p.layers.upcoming, false);
    assert.equal(p.layers.stale, false);
  });

  it("规划聚焦 keeps both planning channels and what they must avoid", () => {
    const p = MAP_PRESETS.find((x) => x.id === "planning")!;
    assert.equal(p.layers.plan, true);
    assert.equal(p.layers.upcoming, true);
    assert.equal(p.layers.next, true);
    assert.equal(p.layers.avoid, true, "planning without the keep-out zones is a lie");
    assert.equal(p.layers.history, true, "already-scanned ground is planning input");
    assert.equal(p.layers.underlay, false);
  });

  it("highlights the preset whose layer set is exactly the current one", () => {
    assert.equal(matchPreset(DEFAULT_LAYERS), "all");
    for (const p of MAP_PRESETS) assert.equal(matchPreset(p.layers), p.id);
  });

  it("shows no preset once a single switch is flipped", () => {
    assert.equal(matchPreset(layers({ grid: false })), null);
  });
});

// ── analysis refresh ────────────────────────────────────────────────────────

describe("analysis signature", () => {
  it("changes when an operation is recorded", () => {
    const before = analysisSignature(mapWith({ marker_count: 3 }));
    assert.notEqual(before, analysisSignature(mapWith({ marker_count: 4 })));
  });

  it("changes when a coarse move invalidates every coordinate", () => {
    const before = analysisSignature(mapWith({ marker_count: 3, current_epoch: 0 }));
    assert.notEqual(
      before,
      analysisSignature(mapWith({ marker_count: 3, current_epoch: 1 })),
    );
  });

  it("changes when a route is published, and again when it advances", () => {
    const none = analysisSignature(mapWith({ markers: [{ status: "done" }] }));
    const published = analysisSignature(
      mapWith({ markers: [{ status: "done" }, { status: "planned" }, { status: "planned" }] }),
    );
    const advanced = analysisSignature(
      mapWith({ markers: [{ status: "done" }, { status: "planned" }] }),
    );
    assert.notEqual(none, published);
    assert.notEqual(published, advanced);
  });

  it("does not change when nothing that matters moved", () => {
    assert.equal(analysisSignature(mapWith({ marker_count: 7 })),
                 analysisSignature(mapWith({ marker_count: 7 })));
  });

  it("is empty with no map yet", () => {
    assert.equal(analysisSignature(null), "");
  });
});

describe("analysis auto-enable", () => {
  it("is off when no layer needs the payload", () => {
    assert.equal(
      analysisAutoEnabled(layers({ avoid: false, next: false, upcoming: false })),
      false,
      "with all three analysis layers off the endpoint must not be called at all",
    );
  });

  it("is on if any one of the three is showing", () => {
    for (const id of ["avoid", "next", "upcoming"] as const) {
      const only = layers({ avoid: false, next: false, upcoming: false, [id]: true });
      assert.equal(analysisAutoEnabled(only), true, `${id} alone should enable it`);
    }
  });
});

describe("analysis throttle", () => {
  it("goes immediately on the first call", () => {
    assert.equal(nextAnalysisDelayMs(1_000_000, null), 0);
  });

  it("goes immediately once the interval has passed", () => {
    const t = 1_000_000;
    assert.equal(nextAnalysisDelayMs(t + MIN_AUTO_ANALYSIS_INTERVAL_MS, t), 0);
  });

  it("defers rather than drops inside the interval", () => {
    // A burst of markers must not be lost — the LAST state has to land.
    const t = 1_000_000;
    const wait = nextAnalysisDelayMs(t + 2_000, t);
    assert.equal(wait, MIN_AUTO_ANALYSIS_INTERVAL_MS - 2_000);
    assert.ok(wait > 0);
  });

  it("never asks for a negative wait when the clock jumps", () => {
    assert.equal(nextAnalysisDelayMs(0, 1_000_000), 0);
  });
});

// ── view bounds ─────────────────────────────────────────────────────────────

describe("fit mode", () => {
  it("returns nothing to fit on an empty map", () => {
    assert.equal(boundsForMode("fit", mapWith(), null, DEFAULT_LAYERS).bounds, null);
  });

  it("includes a marker's whole footprint, not just its centre", () => {
    const { bounds } = boundsForMode(
      "fit",
      mapWith({ markers: [{ status: "done", x_m: 0, y_m: 0, w_m: nm(100), h_m: nm(100) }] }),
      null,
      DEFAULT_LAYERS,
    );
    assert.ok(bounds);
    assert.equal(bounds!.minX, nm(-50));
    assert.equal(bounds!.maxX, nm(50));
  });

  it("stops fitting around a layer that was switched off", () => {
    // The whole point of 精简: a far-away underlay must not keep the view
    // zoomed out around a thumbnail nobody is drawing.
    const map = mapWith({
      markers: [{ status: "done", x_m: 0, y_m: 0, w_m: nm(20), h_m: nm(20) }],
      scan_images: [{ center_x_m: nm(5000), center_y_m: 0, width_m: nm(100), height_m: nm(100) }],
    });
    const on = boundsForMode("fit", map, null, DEFAULT_LAYERS).bounds!;
    const off = boundsForMode("fit", map, null, layers({ underlay: false })).bounds!;
    assert.ok(on.maxX > nm(4000));
    assert.ok(off.maxX < nm(100));
  });

  it("drops stale-generation markers from the fit when they are hidden", () => {
    const map = mapWith({
      current_epoch: 1,
      markers: [
        { status: "done", coord_epoch: 1, x_m: 0, y_m: 0, w_m: nm(20), h_m: nm(20) },
        { status: "done", coord_epoch: 0, x_m: nm(9000), y_m: 0, w_m: nm(20), h_m: nm(20) },
      ],
    });
    assert.ok(boundsForMode("fit", map, null, DEFAULT_LAYERS).bounds!.maxX > nm(8000));
    assert.ok(boundsForMode("fit", map, null, layers({ stale: false })).bounds!.maxX < nm(100));
  });

  it("keeps the route ahead inside the view, or it would point off-canvas", () => {
    const map = mapWith({ tip_xyz: { x_m: 0, y_m: 0 } });
    const analysis = {
      frame_size_m: nm(100),
      upcoming: [{ x_m: nm(800), y_m: 0 }, { x_m: nm(-800), y_m: 0 }],
    };
    const on = boundsForMode("fit", map, analysis, DEFAULT_LAYERS).bounds!;
    assert.ok(on.maxX >= nm(850));
    assert.ok(on.minX <= nm(-850));
    const off = boundsForMode("fit", map, analysis, layers({ upcoming: false })).bounds!;
    assert.equal(off.maxX, 0);
  });
});

describe("follow mode", () => {
  it("puts a window around the live scan frame", () => {
    const map = mapWith({
      frame: { center_x_m: nm(200), center_y_m: nm(-100), width_m: nm(100), height_m: nm(100) },
      markers: [{ status: "done", x_m: nm(9000), y_m: 0, w_m: nm(20), h_m: nm(20) }],
    });
    const { bounds, fellBack } = boundsForMode("follow", map, null, DEFAULT_LAYERS);
    assert.equal(fellBack, false);
    // Centred on the frame, and unaffected by far-away history.
    assert.equal((bounds!.minX + bounds!.maxX) / 2, nm(200));
    assert.equal((bounds!.minY + bounds!.maxY) / 2, nm(-100));
    assert.ok(bounds!.maxX - bounds!.minX > nm(100), "some context around the frame");
    assert.ok(bounds!.maxX - bounds!.minX < nm(200), "but not the whole record");
  });

  it("falls back to fit — and says so — when nothing is being scanned", () => {
    const map = mapWith({
      markers: [{ status: "done", x_m: 0, y_m: 0, w_m: nm(100), h_m: nm(100) }],
    });
    const { bounds, fellBack } = boundsForMode("follow", map, null, DEFAULT_LAYERS);
    assert.equal(fellBack, true, "the UI has to be able to explain the mismatch");
    assert.deepEqual(bounds, boundsForMode("fit", map, null, DEFAULT_LAYERS).bounds);
  });

  it("falls back when the frame has no size yet", () => {
    const map = mapWith({
      frame: { center_x_m: 0, center_y_m: 0, width_m: 0, height_m: 0 },
      markers: [{ status: "done", x_m: 0, y_m: 0, w_m: nm(100), h_m: nm(100) }],
    });
    assert.equal(boundsForMode("follow", map, null, DEFAULT_LAYERS).fellBack, true);
  });
});

describe("full mode", () => {
  it("shows the whole reachable piezo area regardless of content", () => {
    const { bounds, fellBack } = boundsForMode(
      "full", mapWith(), null, DEFAULT_LAYERS, 1.5e-6,
    );
    assert.equal(fellBack, false);
    assert.deepEqual(bounds, { minX: -1.5e-6, maxX: 1.5e-6, minY: -1.5e-6, maxY: 1.5e-6 });
  });

  it("falls back to fit when the range is unknown", () => {
    const map = mapWith({
      markers: [{ status: "done", x_m: 0, y_m: 0, w_m: nm(100), h_m: nm(100) }],
    });
    const { bounds, fellBack } = boundsForMode("full", map, null, DEFAULT_LAYERS, null);
    assert.equal(fellBack, true);
    assert.deepEqual(bounds, boundsForMode("fit", map, null, DEFAULT_LAYERS).bounds);
  });
});

describe("stale epoch", () => {
  it("is only stale when the generation is genuinely behind", () => {
    assert.equal(isStaleEpoch({ coord_epoch: 0 }, 1), true);
    assert.equal(isStaleEpoch({ coord_epoch: 1 }, 1), false);
    assert.equal(isStaleEpoch({ coord_epoch: null }, 1), false);
    assert.equal(isStaleEpoch({ coord_epoch: 0 }, null), false);
  });
});

// ── map chrome under wheel zoom  ───────────────────────────────────────

describe("visibleStepNm / niceStepNm", () => {
  it("is unchanged at 1× — the whole-map behaviour is the baseline", () => {
    assert.equal(visibleStepNm(500, 1), niceStepNm(500));
    assert.equal(niceStepNm(500), 100);
  });

  it("subdivides as the operator zooms in, instead of holding whole-map spacing", () => {
    assert.ok(visibleStepNm(500, 10) < visibleStepNm(500, 1));
    assert.ok(visibleStepNm(500, 25) < visibleStepNm(500, 10));
  });

  it("keeps the scale bar a roughly constant fraction of the canvas", () => {
    // Screen length = step · (px per nm) · k. Because the step tracks 1/k, that
    // product must not run away with zoom — the old code multiplied by k with a
    // fixed step, so at 25× the bar was 25× too long and left the canvas.
    const spanNm = 500;
    const pxPerNm = 532 / spanNm; // CANVAS_W - 2·PAD over the span
    const lengths = [1, 2, 5, 10, 25].map((k) => visibleStepNm(spanNm, k) * pxPerNm * k);
    for (const L of lengths) {
      assert.ok(L > 40 && L < 300, `scale bar ${L}px must stay on a 620px canvas`);
    }
  });

  it("survives a nonsense zoom rather than producing NaN spacing", () => {
    assert.equal(visibleStepNm(500, 0), niceStepNm(500));
    assert.equal(visibleStepNm(500, Number.NaN), niceStepNm(500));
  });
});

describe("inverseStageTransform / stageToScreen", () => {
  it("composes with the stage transform to the identity", () => {
    const zoom = { k: 7.3, x: -211, y: 88 };
    const inv = inverseStageTransform(zoom);
    for (const p of [{ x: 0, y: 0 }, { x: 310, y: 230 }, { x: 619, y: 459 }]) {
      // stage(layer(p)) — what Konva actually paints for a layer carrying inv.
      const lx = p.x * inv.scaleX + inv.x;
      const ly = p.y * inv.scaleY + inv.y;
      const out = stageToScreen({ x: lx, y: ly }, zoom);
      assert.ok(Math.abs(out.x - p.x) < 1e-9, `${out.x} ≈ ${p.x}`);
      assert.ok(Math.abs(out.y - p.y) < 1e-9, `${out.y} ≈ ${p.y}`);
    }
  });

  it("is the identity at 1× so an un-zoomed map is untouched", () => {
    assert.deepEqual(inverseStageTransform({ k: 1, x: 0, y: 0 }), {
      scaleX: 1, scaleY: 1, x: 0, y: 0,
    });
    assert.deepEqual(stageToScreen({ x: 12, y: 34 }, { k: 1, x: 0, y: 0 }), { x: 12, y: 34 });
  });

  it("moves a world anchor off-canvas when it really is off-canvas", () => {
    // The culling the tick labels rely on: at 10× with a big pan, a tick that
    // used to sit at x=300 is nowhere near the 620px canvas.
    const s = stageToScreen({ x: 300, y: 100 }, { k: 10, x: -2800, y: 0 });
    assert.equal(s.x, 200);
    const far = stageToScreen({ x: 300, y: 100 }, { k: 10, x: -3400, y: 0 });
    assert.ok(far.x < 0, `${far.x} must be culled`);
  });

  it("degrades a bad zoom to the identity instead of dividing by zero", () => {
    assert.deepEqual(inverseStageTransform({ k: 0, x: 5, y: 5 }), {
      scaleX: 1, scaleY: 1, x: -5, y: -5,
    });
  });
});

describe("fmtNm", () => {
  it("rounds to whole nm when the grid step is whole nm", () => {
    assert.equal(fmtNm(137.4, 50), "137");
    assert.equal(fmtNm(-0.2, 50), "0"); // never a stray "-0"
  });

  it("keeps sub-nm precision once the zoom makes the step sub-nm", () => {
    // The bug this replaces: Math.round() printed "0" for every tick on a
    // zoomed-in map, and "0 nm" on the scale bar.
    assert.equal(fmtNm(0.25, 0.05), "0.25");
    assert.equal(fmtNm(1.5, 0.5), "1.5");
    assert.notEqual(fmtNm(0.25, 0.05), "0");
  });

  it("is never more precise than the step it labels", () => {
    assert.equal(fmtNm(1.23456, 1), "1");
    assert.equal(fmtNm(1.23456, 0.1), "1.2");
  });

  it("returns empty for a non-finite value rather than 'NaN'", () => {
    assert.equal(fmtNm(Number.NaN, 1), "");
    assert.equal(fmtNm(1.0, Number.NaN), "1");
  });
});

// ── world-anchored labels under zoom ───────────────────────────────────────
//
// #52 pinned the GLOBAL chrome (ticks, scale bar) with inverseStageTransform.
// The five labels that belong to a moving feature stayed in the zoomed layer,
// so they kept scaling with the stage: at 25× a 10 px caption paints 250 px
// tall over the surface it names — the same complaint as #52:
// captions don't stay put when the wheel zooms.

describe("pinnedLabel", () => {
  it("renders at constant screen size whatever the zoom", () => {
    for (const k of [1, 2.5, 25]) {
      const p = pinnedLabel({ x: 100, y: 100 }, 8, -6, k);
      // Konva paints glyphs at fontSize · stageScale · nodeScale.
      assert.ok(Math.abs(k * p.scaleX - 1) < 1e-12, `${k}× 下字号被放大了 ${k * p.scaleX}`);
      assert.ok(Math.abs(k * p.scaleY - 1) < 1e-12);
    }
  });

  it("keeps the gap between feature and caption constant on screen", () => {
    // The label's screen offset from its anchor must not depend on k.
    for (const k of [1, 4, 25]) {
      const p = pinnedLabel({ x: 100, y: 100 }, 8, -6, k);
      assert.ok(Math.abs((p.x - 100) * k - 8) < 1e-12, `${k}× 下横向间距变成 ${(p.x - 100) * k}`);
      assert.ok(Math.abs((p.y - 100) * k - -6) < 1e-12);
    }
  });

  it("stays anchored to the feature — the anchor is NOT counter-scaled", () => {
    // The whole reason this is not just inverseStageTransform on the layer:
    // the anchor has to keep riding the world, or the caption drifts off the
    // thing it names as soon as the operator pans.
    const p = pinnedLabel({ x: 640, y: 480 }, 0, 0, 25);
    assert.deepEqual({ x: p.x, y: p.y }, { x: 640, y: 480 });
  });

  it("is a no-op offset at 1× so an un-zoomed map is untouched", () => {
    assert.deepEqual(pinnedLabel({ x: 10, y: 20 }, 8, -6, 1), {
      x: 18, y: 14, scaleX: 1, scaleY: 1,
    });
  });

  it("degrades a bad zoom to 1× instead of dividing by zero", () => {
    for (const bad of [0, -3, Number.NaN, Number.POSITIVE_INFINITY]) {
      const p = pinnedLabel({ x: 10, y: 20 }, 8, -6, bad);
      assert.deepEqual(p, { x: 18, y: 14, scaleX: 1, scaleY: 1 }, `k=${bad}`);
    }
  });
});

// ── marker detail popup  ──────────────────────────────────────

describe("markerDetailRows", () => {
  const row = (rows: { label: string; value: string }[], label: string) =>
    rows.find((r) => r.label === label);

  it("carries what the one-line tooltip could not", () => {
    // The hover tip is capped at 200 px with wrap="none" — a label and a clock
    // time and nothing else. Skill, outcome and frame size had nowhere to go.
    const rows = markerDetailRows({
      kind: "scan",
      skill_name: "StartScan",
      status: "done",
      timestamp: "2026-08-05T13:42:11",
      x_m: 1.2e-7,
      y_m: -3.4e-8,
      w_m: 1e-7,
      h_m: 1e-7,
      coord_epoch: 2,
    }, 2);
    assert.equal(row(rows, "技能")?.value, "StartScan");
    assert.equal(row(rows, "状态")?.value, "已完成");
    assert.match(row(rows, "时间")!.value, /2026-08-05 13:42:11/);
    assert.match(row(rows, "位置")!.value, /120\.00 nm/);
    assert.match(row(rows, "尺寸")!.value, /100\.00 nm × 100\.00 nm/);
  });

  it("says a stale epoch IS stale, in words", () => {
    // Fading the marker says "something is different"; it does not say the
    // coordinates no longer point at where the operation happened. That is the
    // one thing here that changes what the operator should do next.
    const stale = markerDetailRows({ coord_epoch: 1 }, 3);
    assert.match(row(stale, "坐标代次")!.value, /旧坐标系/);
    const fresh = markerDetailRows({ coord_epoch: 3 }, 3);
    assert.equal(row(fresh, "坐标代次")!.value, "#3");
  });

  it("omits what it does not have rather than printing a zero", () => {
    // A marker with no size is a POINT operation (a pulse, a spectrum), not a
    // zero-sized scan. 「尺寸 0 × 0」 would be a false statement.
    const rows = markerDetailRows({ kind: "pulse", x_m: 0, y_m: 0 });
    assert.equal(row(rows, "尺寸"), undefined);
    assert.equal(row(rows, "时间"), undefined);
    assert.equal(row(rows, "技能"), undefined);
    assert.ok(row(rows, "位置"), "a point operation still has a position");
  });

  it("survives a marker with nothing on it at all", () => {
    assert.deepEqual(markerDetailRows({}), []);
    assert.deepEqual(markerDetailRows({ x_m: null, y_m: undefined }), []);
  });

  it("drops a non-finite coordinate instead of rendering NaN", () => {
    const rows = markerDetailRows({ x_m: Number.NaN, y_m: 1e-9 });
    assert.equal(row(rows, "位置"), undefined);
  });
});
