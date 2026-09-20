// ════════════════════════════════════════════════════════════════════════════
// Pure-logic tests for src/lib/visionFrames.ts — the caption a 近期标注帧 tile
// shows when it has no picture.
//
//     cd frontend && npm run test:unit
//
// was「近期标注帧 / 无图像」. The panel was not broken: it was
// printing one flat string for two situations that mean opposite things — a
// deliberate refusal to borrow a stale image for an in-scan judgement (correct,
// and the fix for #76/#78), and a completed scan whose file could not be found
// (a real failure). These tests pin that they never collapse back into one
// message, because the only symptom of that regression is an operator filing
// the same report again.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { framePlaceholder, parseCauseRef } from "../src/lib/visionFrames.ts";

describe("framePlaceholder — when there IS an image", () => {
  it("returns null so the caller draws the picture", () => {
    assert.equal(framePlaceholder({ kind: "scan_complete", image_b64: "iVBORw0KGgo=" }), null);
  });

  it("prefers the image even when a path is also present", () => {
    assert.equal(
      framePlaceholder({ kind: "scan_complete", image_b64: "iVBORw0KGgo=", file_path: "a.sxm" }),
      null,
    );
  });
});

describe("framePlaceholder — a file exists but no thumbnail rendered", () => {
  it("says the thumbnail is missing, not that there is no image", () => {
    const p = framePlaceholder({ kind: "scan_complete", file_path: "D:/exp/scan_042.sxm" });
    assert.ok(p);
    assert.equal(p.label, "缩略图未生成");
    // The path is the actionable part — it belongs in the hover text.
    assert.match(p.hint, /scan_042\.sxm/);
  });
});

describe("framePlaceholder — no image and no path", () => {
  it("explains that an in-scan pulse deliberately has no borrowed picture", () => {
    const p = framePlaceholder({ kind: "feature_of_interest" });
    assert.ok(p);
    assert.match(p.label, /扫描中/);
    // The WHY is the whole point: without it this reads as a bug.
    assert.match(p.hint, /不会借用/);
  });

  it("reports a completed scan with no locatable file as a lookup failure", () => {
    const p = framePlaceholder({ kind: "scan_complete" });
    assert.ok(p);
    assert.match(p.label, /找不到/);
  });

  it("keeps the two cases distinguishable — that is the entire fix for #55", () => {
    const pulse = framePlaceholder({ kind: "feature_of_interest" });
    const done = framePlaceholder({ kind: "scan_complete" });
    assert.ok(pulse && done);
    assert.notEqual(pulse.label, done.label);
    assert.notEqual(pulse.hint, done.hint);
  });

  it("covers the non-imaging event kinds without calling them failures", () => {
    for (const kind of ["tip_quality_drop", "tip_shape_verdict"]) {
      const p = framePlaceholder({ kind });
      assert.ok(p, kind);
      assert.match(p.label, /针尖判读/);
    }
    for (const kind of ["sensor_fault", "vision_error"]) {
      const p = framePlaceholder({ kind });
      assert.ok(p, kind);
      assert.match(p.label, /故障事件/);
    }
  });

  it("falls back to a plain statement for an unknown or missing kind", () => {
    // A kind added to the backend enum later must not produce an empty caption.
    for (const frame of [{ kind: "e_stop" }, { kind: "" }, { kind: null }, {}]) {
      const p = framePlaceholder(frame);
      assert.ok(p);
      assert.ok(p.label.length > 0);
      assert.ok(p.hint.length > 0);
    }
  });

  it("matches kind case-insensitively — severity/kind arrive unnormalised", () => {
    const upper = framePlaceholder({ kind: "FEATURE_OF_INTEREST" });
    const lower = framePlaceholder({ kind: "feature_of_interest" });
    assert.deepEqual(upper, lower);
  });
});

// ── cause_ref  ────────────────────────────────────────────────
// 「无图像 / tip_quality_drop / critical」was CORRECT — that alert came from
// `current_monitor#80391`, i.e. a 1 kHz current segment with no scan frame
// anywhere near it. But the same `tip_quality_drop` kind is also emitted by the
// vision path, so the KIND cannot decide what to draw. cause_ref can.

describe("parseCauseRef", () => {
  it("reads the segment id out of a current-monitor ref", () => {
    assert.deepEqual(parseCauseRef("current_monitor#80391"), {
      kind: "current_monitor",
      segmentId: 80391,
    });
  });

  it("accepts the bare form the emitter falls back to with no segment", () => {
    // monitoring/alerts.py: `f"current_monitor#{segment_id}" if segment_id else "current_monitor"`
    assert.deepEqual(parseCauseRef("current_monitor"), {
      kind: "current_monitor",
      segmentId: null,
    });
  });

  it("refuses a non-numeric segment id rather than passing NaN to a URL", () => {
    const got = parseCauseRef("current_monitor#abc");
    assert.deepEqual(got, { kind: "current_monitor", segmentId: null });
  });

  it("knows the other producers", () => {
    assert.deepEqual(parseCauseRef("scan#run-7"), { kind: "scan", scanId: "run-7" });
    assert.deepEqual(parseCauseRef("tip_status#412"), { kind: "tip_status", seqno: 412 });
    assert.deepEqual(parseCauseRef("skill:ConditionTip"), { kind: "skill", name: "ConditionTip" });
  });

  it("returns null for absent, and 'unknown' for a shape nobody emits yet", () => {
    for (const v of [null, undefined, "", "   "]) assert.equal(parseCauseRef(v), null);
    assert.deepEqual(parseCauseRef("weird"), { kind: "unknown", raw: "weird" });
  });
});

describe("framePlaceholder — current-monitor events ", () => {
  it("asks for the segment's OWN waveform instead of saying 无图像", () => {
    const p = framePlaceholder({
      kind: "tip_quality_drop",
      cause_ref: "current_monitor#80391",
    });
    assert.ok(p);
    assert.deepEqual(p.source, { kind: "current_monitor", segmentId: 80391 });
    assert.match(p.label, /电流监控/);
    // The hint has to say WHY there is no frame, or a drawn curve just raises
    // the same question in a new form.
    assert.match(p.hint, /不是扫描画面/);
  });

  it("beats the kind — the vision path emits tip_quality_drop too", () => {
    const fromMonitor = framePlaceholder({
      kind: "tip_quality_drop",
      cause_ref: "current_monitor#5",
    });
    const fromVision = framePlaceholder({ kind: "tip_quality_drop", cause_ref: "scan#run-1" });
    assert.ok(fromMonitor && fromVision);
    assert.ok(fromMonitor.source, "monitor event must offer a curve");
    assert.equal(fromVision.source, undefined, "a scan-sourced event has no segment to draw");
  });

  it("says so plainly when the segment id is missing, and offers no curve", () => {
    const p = framePlaceholder({ kind: "tip_quality_drop", cause_ref: "current_monitor" });
    assert.ok(p);
    assert.equal(p.source, undefined);
    assert.match(p.label, /无分段/);
  });

  it("never overrides a real image", () => {
    const p = framePlaceholder({
      kind: "tip_quality_drop",
      cause_ref: "current_monitor#7",
      image_b64: "iVBORw0KGgo=",
    });
    assert.equal(p, null);
  });

  it("prefers a real file path over the curve — the file is the better picture", () => {
    const p = framePlaceholder({
      kind: "tip_quality_drop",
      cause_ref: "current_monitor#7",
      file_path: "D:/exp/evidence.png",
    });
    assert.ok(p);
    assert.equal(p.source, undefined);
    assert.equal(p.label, "缩略图未生成");
  });
});

describe("framePlaceholder — empty strings are not values", () => {
  it("treats an empty image_b64 as absent", () => {
    const p = framePlaceholder({ kind: "scan_complete", image_b64: "" });
    assert.ok(p, "empty base64 must not be rendered as an <img>");
  });

  it("treats an empty file_path as absent", () => {
    const p = framePlaceholder({ kind: "feature_of_interest", file_path: "" });
    assert.ok(p);
    assert.match(p.label, /扫描中/);
  });
});
