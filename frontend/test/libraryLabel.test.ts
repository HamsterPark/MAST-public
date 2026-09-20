// ════════════════════════════════════════════════════════════════════════════
// Pure-logic tests for src/lib/libraryLabel.ts — how 「当前生效的库」 is named.
//
// Runs on `node --test` (no test framework is installed in this frontend):
//
//     cd frontend && npm run test:unit
//
// The failure this covers is silent by construction: the effective library id
// is DERIVED from the active experiment and the library row is created lazily,
// so before the first ingest there is an id with no row behind it. Every naming
// site did `libraries.find(...) ?? id` and quietly fell back to the bare
// `exp_example` — which is exactly what the operator saw.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { libraryLabel } from "../src/lib/libraryLabel.ts";

describe("libraryLabel", () => {
  it("prefers the experiment name over the frozen library name", () => {
    const out = libraryLabel(
      "exp_example",
      "experiment",
      [{ library_id: "exp_example", name: "旧名 文献库", experiment_name: "v6.0.0 真机验收" }],
    );
    assert.equal(out.text, "《v6.0.0 真机验收》的文献库");
    assert.equal(out.pending, false);
  });

  it("falls back to the library's own name when it is not an experiment library", () => {
    const out = libraryLabel("reading", "fallback", [
      { library_id: "reading", name: "reading library" },
    ]);
    assert.equal(out.text, "reading library");
    assert.equal(out.pending, false);
  });

  it("names a not-yet-created experiment library after the active experiment", () => {
    // Synthetic experiment scope whose library has not been created yet.
    const out = libraryLabel(
      "exp_example",
      "experiment",
      [{ library_id: "reading", name: "reading library" }],
      "示例实验 · sample",
    );
    assert.equal(out.text, "《示例实验 · sample》的文献库");
    // Derived, not on disk — the UI must not claim the library already exists.
    assert.equal(out.pending, true);
  });

  it("keeps the bare id when there is no name to be had, and still flags pending", () => {
    const out = libraryLabel("exp_example", "experiment", [], null);
    assert.equal(out.text, "exp_example");
    assert.equal(out.pending, true);
  });

  it("does not invent a pending state for a manual pointer", () => {
    const out = libraryLabel("gone", "manual", [], "某实验");
    assert.equal(out.text, "gone");
    assert.equal(out.pending, false);
  });

  it("returns empty for an empty id rather than a stray 《》", () => {
    assert.deepEqual(libraryLabel("", "experiment", [], "某实验"), { text: "", pending: false });
  });

  // ── 「已经落到哪」的回显（2026-08-05） ──────────────────────────────────
  //
  // #49 修了三个「将会落到哪」的横幅，漏了两个结果回显（摄取的「· 库 …」、
  // 上传的「· 已入库 …」）。它们通过 LibraryRef 走同一个函数，但**没有
  // source** 可传 —— 摄取结果里根本没有这个字段。下面两条钉住那条调用形态，
  // 免得有人看到 `source` 在这些站点恒为空就把它当冗余参数删掉。

  it("names a library from the row alone when the caller has no source", () => {
    const out = libraryLabel("exp_example", "", [
      { library_id: "exp_example", name: "v6.0.0 真机验收 文献库",
        experiment_name: "v6.0.0 真机验收" },
    ]);
    assert.equal(out.text, "《v6.0.0 真机验收》的文献库");
    assert.equal(out.pending, false);
  });

  it("without a source, an unknown id is NOT flagged pending", () => {
    // 摄取刚写完的库一定在列表里（回调会 invalidate）。查不到就是真查不到，
    // 说「还没建出来」是编的。
    const out = libraryLabel("exp_example", "", [], "某实验");
    assert.equal(out.text, "exp_example");
    assert.equal(out.pending, false);
  });
});
