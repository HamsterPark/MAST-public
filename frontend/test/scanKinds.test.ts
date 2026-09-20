// ════════════════════════════════════════════════════════════════════════════
// Pure-logic tests for src/lib/scanKinds.ts — the chip → `ext` mapping the 数据
// tab sends to /api/scans/latest.
//
//     cd frontend && npm run test:unit
//
// and its 数据面板 twin  are both "the filter and the truncation
// live on opposite sides of the wire". Moving the filter to the server fixes
// that; what these tests pin is the part that stayed on the client — 其他, which
// has no extension of its own and is therefore derived. Derive it wrong and the
// tab shows the WRONG files, which looks exactly like showing the right ones.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  EXT_MATCH_NONE,
  SCAN_GROUPS,
  extParam,
  groupOf,
  groupScans,
  isKnownKind,
  otherCount,
} from "../src/lib/scanKinds.ts";

const RIG_COUNTS = { ".sxm": 26, ".dat": 30, ".csv": 4, ".txt": 2 };

describe("extParam", () => {
  it("sends no filter for 全部", () => {
    assert.equal(extParam("all", Object.keys(RIG_COUNTS)), undefined);
  });

  it("passes a named extension straight through", () => {
    assert.equal(extParam(".sxm", Object.keys(RIG_COUNTS)), ".sxm");
  });

  it("expands 其他 from what the SERVER reported, not a hardcoded list", () => {
    // .csv/.txt are not named chips, so they are 其他. Nothing in the frontend
    // may enumerate the backend's recognised extensions — that copy would drift
    // and files would silently stop appearing under 其他.
    const got = extParam("other", Object.keys(RIG_COUNTS));
    assert.deepEqual(got?.split(",").sort(), [".csv", ".txt"]);
  });

  it("never lists a named kind under 其他", () => {
    const got = extParam("other", [".sxm", ".dat", ".3ds", ".sm4", ".asc"]);
    assert.equal(got, ".asc");
  });

  it("matches NOTHING when 其他 is empty, rather than dropping the filter", () => {
    // Dropping it would return every file on disk under an 其他 chip — the same
    // answering-a-different-question failure #53 was.
    assert.equal(extParam("other", [".sxm", ".dat"]), EXT_MATCH_NONE);
    assert.equal(extParam("other", []), EXT_MATCH_NONE);
  });
});

describe("otherCount", () => {
  it("sums only the unnamed extensions", () => {
    assert.equal(otherCount(RIG_COUNTS), 6);
  });

  it("is 0 when everything on disk has its own chip", () => {
    assert.equal(otherCount({ ".sxm": 3, ".dat": 1 }), 0);
  });

  it("counts the whole disk, not a page — the caller must pass full-result counts", () => {
    // The count is only meaningful because the server computes counts_by_ext
    // over the FULL discovery result. If it ever starts counting the returned
    // page instead, this number becomes the "SXM (0)" lie again.
    assert.equal(otherCount({ ".csv": 120 }), 120);
  });
});

describe("isKnownKind", () => {
  it("recognises exactly the four named chips", () => {
    for (const e of [".sxm", ".dat", ".3ds", ".sm4"]) assert.equal(isKnownKind(e), true);
    for (const e of [".csv", ".txt", ".asc", ".tsv", ".png", ""]) {
      assert.equal(isKnownKind(e), false);
    }
  });
});

// ── grouping (「数据被电流的 csv 占据了！看不到 sxm 了」) ─────────────

describe("groupScans", () => {
  const f = (name: string, ext: string) => ({ name, ext });

  it("keeps scans in their own section however many CSVs are newer", () => {
    // The rig shape: telemetry CSVs carry an mtime of *now* forever, so a
    // mtime-desc list puts every one of them ahead of every scan. Grouping is
    // what survives the NEXT telemetry writer, which the source-side blocklist
    // by construction cannot know about.
    const files = [
      f("monitor_current_1.csv", ".csv"),
      f("monitor_bias_1.csv", ".csv"),
      f("topo_0001.sxm", ".sxm"),
      f("sts_0001.dat", ".dat"),
    ];
    const groups = groupScans(files);
    assert.deepEqual(groups.map((g) => g.group.id), ["image", "spectrum", "other"]);
    assert.deepEqual(groups[0]!.files.map((x) => x.name), ["topo_0001.sxm"]);
  });

  it("is a reordering — never drops a file", () => {
    // Dropping one here is the #53/#60 shape a third time: a list that looks
    // right while a file the operator can see on disk is missing from it.
    const files = [
      f("a.sxm", ".sxm"), f("b.sm4", ".sm4"), f("c.dat", ".dat"),
      f("d.3ds", ".3ds"), f("e.csv", ".csv"), f("f.txt", ".txt"),
      f("g.weird", ".weird"), f("h", ""),
    ];
    const out = groupScans(files).flatMap((g) => g.files);
    assert.equal(out.length, files.length);
    assert.deepEqual(new Set(out.map((x) => x.name)), new Set(files.map((x) => x.name)));
  });

  it("preserves the server's order inside a group", () => {
    const files = [f("new.sxm", ".sxm"), f("mid.sxm", ".sxm"), f("old.sxm", ".sxm")];
    assert.deepEqual(
      groupScans(files)[0]!.files.map((x) => x.name),
      ["new.sxm", "mid.sxm", "old.sxm"],
    );
  });

  it("omits a group with no members rather than showing an empty heading", () => {
    assert.deepEqual(groupScans([f("a.sxm", ".sxm")]).map((g) => g.group.id), ["image"]);
    assert.deepEqual(groupScans([]), []);
  });

  it("classifies every extension the server can return", () => {
    // _SCAN_EXTS in webui/scan_preview.py. An unclassified one would silently
    // vanish if groupOf ever returned something SCAN_GROUPS does not list.
    const ids = new Set(SCAN_GROUPS.map((g) => g.id));
    for (const e of [".sxm", ".sm4", ".dat", ".3ds", ".txt", ".csv", ".asc", ".tsv"]) {
      assert.ok(ids.has(groupOf(e)), e);
    }
    assert.equal(groupOf(".SXM"), "image", "extension case must not matter");
    assert.ok(ids.has(groupOf("")));
  });
});
