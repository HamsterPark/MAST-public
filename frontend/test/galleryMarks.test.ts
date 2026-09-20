// ════════════════════════════════════════════════════════════════════════════
// 数据图库的标记 — src/lib/gallery/marks.ts
//
//     cd frontend && npm run test:unit
//
// 标记是操作员几个小时的判断。这里钉住存盘协议里「不丢」的那几条：删除写墓碑、
// 失败的一批不盖掉请求在路上时又改的新版本、坏的 localStorage 当作没有而不是崩。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  DEFAULT_TAGS,
  applyPatchLocal,
  composeMark,
  emptyPatch,
  isEmptyMark,
  isTombstone,
  mergePatch,
  normaliseDoc,
  parsePending,
  parseTagList,
  patchEmpty,
  ratingToggle,
  requeue,
  toggledTags,
  tsClock,
  withDir,
  withMark,
  withSeries,
} from "../src/lib/gallery/marks.ts";

const doc = () => normaliseDoc({ rev: 3, tags: ["x", "y"], items: { a: { r: 1, tags: [], note: "" } }, series: {}, days: {} });

describe("空与墓碑", () => {
  it("patchEmpty", () => {
    assert.equal(patchEmpty(emptyPatch()), true);
    assert.equal(patchEmpty({ ...emptyPatch(), tags: ["a"] }), false);
    assert.equal(patchEmpty({ ...emptyPatch(), days: { d: { done: true, note: "" } } }), false);
  });

  it("isTombstone", () => {
    assert.equal(isTombstone({ del: true, ts: 1 }), true);
    assert.equal(isTombstone({ r: 1, tags: [], note: "" }), false);
    assert.equal(isTombstone(null), false);
  });

  it("空标记：无评级无标签无备注无系定；只有系定也算有", () => {
    assert.equal(isEmptyMark({ r: 0, tags: [], note: "  " }), true);
    assert.equal(isEmptyMark(null), true);
    assert.equal(isEmptyMark({ r: 0, tags: [], note: "", anchor: { id: "f", fn: "f", rel: "prev", dt: 1, desc: "" } }), false);
    assert.equal(isEmptyMark({ r: -1, tags: [], note: "" }), false);
  });
});

describe("文档", () => {
  it("缺标签表时用默认；坏字段补空", () => {
    const d = normaliseDoc({ items: [1, 2], series: "x" });
    assert.deepEqual(d.tags, [...DEFAULT_TAGS]);
    assert.deepEqual(d.items, {});
    assert.deepEqual(d.series, {});
    assert.deepEqual(normaliseDoc(null).days, {});
    assert.deepEqual(normaliseDoc({ tags: ["a", 3, "b"] }).tags, ["a", "b"]);
  });

  it("withMark：设 / 墓碑删 / null 删，不改入参", () => {
    const d0 = doc();
    const d1 = withMark(d0, "b", { r: 2, tags: [], note: "" });
    assert.ok(d1.items.b && !d0.items.b);
    assert.ok(!withMark(d1, "b", { del: true, ts: 9 }).items.b);
    assert.ok(!withMark(d1, "a", null).items.a);
    assert.ok(d1.items.a, "入参没被改");
  });

  it("withSeries：成员为空等于删除", () => {
    const d = withSeries(doc(), "S", { name: "n", k: "f", ids: ["a"], r: 0, tags: [], note: "" });
    assert.ok(d.series.S);
    assert.ok(!withSeries(d, "S", { name: "n", k: "f", ids: [], r: 0, tags: [], note: "" }).series.S);
    assert.ok(!withSeries(d, "S", { del: true, ts: 1 }).series.S);
  });

  it("withDir：没勾已过完又没备注就删", () => {
    const d = withDir(doc(), "R/1", { done: true, note: "" });
    assert.ok(d.days["R/1"]);
    assert.ok(!withDir(d, "R/1", { done: false, note: " " }).days["R/1"]);
    assert.ok(withDir(d, "R/2", { done: false, note: "针尖变了" }).days["R/2"]);
  });

  it("applyPatchLocal 四类都应用", () => {
    const p = { items: { a: { del: true as const, ts: 1 }, c: { r: 1, tags: [], note: "" } }, series: { S: { name: "s", k: "f", ids: ["c"], r: 0, tags: [], note: "" } }, days: { D: { done: true, note: "" } }, tags: ["z"] };
    const d = applyPatchLocal(doc(), p);
    assert.deepEqual(Object.keys(d.items), ["c"]);
    assert.ok(d.series.S);
    assert.ok(d.days.D);
    assert.deepEqual(d.tags, ["z"]);
  });
});

describe("待存队列", () => {
  it("mergePatch：新的覆盖旧的", () => {
    const a = { ...emptyPatch(), items: { x: { r: 1, tags: [], note: "old" } }, tags: ["t1"] };
    const b = { ...emptyPatch(), items: { x: { r: 2, tags: [], note: "new" } } };
    const m = mergePatch(a, b);
    assert.equal((m.items.x as { note: string }).note, "new");
    assert.deepEqual(m.tags, ["t1"]);
  });

  it("requeue：失败的一批不盖掉在路上时又改的新版本", () => {
    const failed = { ...emptyPatch(), items: { x: { r: 1, tags: [], note: "失败的旧版" }, y: { r: 1, tags: [], note: "只在失败批里" } }, tags: ["old"] };
    const pending = { ...emptyPatch(), items: { x: { r: 2, tags: [], note: "路上又改的" } } };
    const out = requeue(pending, failed);
    assert.equal((out.items.x as { note: string }).note, "路上又改的");
    assert.equal((out.items.y as { note: string }).note, "只在失败批里");
    assert.deepEqual(out.tags, ["old"]);
    assert.deepEqual(requeue({ ...pending, tags: ["new"] }, failed).tags, ["new"]);
  });

  it("parsePending：坏数据、空 patch 都当没有", () => {
    assert.equal(parsePending("{broken"), null);
    assert.equal(parsePending(null), null);
    assert.equal(parsePending(JSON.stringify(emptyPatch())), null);
    const p = parsePending(JSON.stringify({ items: { a: { r: 1, tags: [], note: "" } }, tags: ["a", 1] }));
    assert.ok(p && p.items.a);
    assert.deepEqual(p?.tags, ["a"]);
  });
});

describe("改一条标记", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const it0: any = { id: "R/20010910/a_0010.sxm", k: "f", d: "R/20010910", fn: "a_0010.sxm", t: 1000, w: 3, b: 2, sp: 100, nx: 192, p: "", pf: "", ad: "", th: "" };

  it("写上 t ts tt k meta", () => {
    const m = composeMark(null, { r: 2 }, it0, "2001-09-13 20:00:00", 42);
    assert.ok(m);
    assert.equal(m!.r, 2);
    assert.equal(m!.t, "2001-09-13 20:00:00");
    assert.equal(m!.ts, 42);
    assert.equal(m!.tt, 1000);
    assert.equal(m!.k, "f");
    assert.ok(m!.meta!.startsWith("R/20010910 · "));
  });

  it("清掉唯一的评级 = 删除；有系定就留着", () => {
    const old = { r: 1, tags: [], note: "" };
    assert.equal(composeMark(old, { r: 0 }, it0, "t", 1), null);
    const anc = { id: "f", fn: "f", rel: "prev", dt: -5, desc: "" };
    assert.ok(composeMark({ ...old, anchor: anc }, { r: 0 }, it0, "t", 1));
  });

  it("toggledTags：取反 / 强制开 / 强制关 / 无变化回 null", () => {
    assert.deepEqual(toggledTags(["a"], "b"), ["a", "b"]);
    assert.deepEqual(toggledTags(["a", "b"], "a"), ["b"]);
    assert.equal(toggledTags(["a"], "a", true), null);
    assert.equal(toggledTags(["a"], "b", false), null);
  });

  it("ratingToggle：再点同一个取消", () => {
    assert.equal(ratingToggle(2, 2), 0);
    assert.equal(ratingToggle(1, 2), 2);
    assert.equal(ratingToggle(undefined, -1), -1);
  });

  it("parseTagList：中英文逗号、去空白、去重、保序", () => {
    assert.deepEqual(parseTagList(" a，b, a ,, c"), ["a", "b", "c"]);
  });

  it("tsClock 同一毫秒里也严格递增", () => {
    const next = tsClock(() => 1000);
    assert.deepEqual([next(), next(), next()], [1000, 1001, 1002]);
  });
});
