// ════════════════════════════════════════════════════════════════════════════
// 数据图库的目录总览 — src/lib/gallery/dirs.ts
//
//     cd frontend && npm run test:unit
//
// 目录卡片上的每一个数都来自这里。目录名在 MAST 里不一定是日期，所以排序按目录内
// 最晚时刻，不按名字——一个名字排在前面、内容却是半年前的目录不该压在今天上面。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { newBatch, shortList, summariseDirs } from "../src/lib/gallery/dirs.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mk = (o: Record<string, unknown>): any => ({ fn: "x", p: "", pf: "", ad: "", th: "", ...o });

const items = [
  mk({ id: "A/1", d: "R/zzz_old", k: "f", t: 100, mt: 150, b: 2, w: 5, at: 30, pf: "p", ad: "B0" }),
  mk({ id: "A/2", d: "R/zzz_old", k: "f", t: 200, mt: 260, b: 2.001, w: 5.02, dup: "A/1", pf: "p", ad: "B0" }),
  mk({ id: "A/3", d: "R/zzz_old", k: "s", t: 300, mt: 310, b: 9, pf: "q", ad: "B0" }),
  mk({ id: "B/1", d: "R/aaa_new", k: "f", t: 1000, mt: 1100, b: -0.05, w: 10, hf: 1.8, pf: "p", ad: "B1" }),
  mk({ id: "B/2", d: "R/aaa_new", k: "g", t: 1200, t1: 5000, mt: 5001, pf: "Grid", ad: "B1" }),
];

describe("summariseDirs", () => {
  const marks: Record<string, unknown> = { "A/1": { r: 2 }, "A/3": { r: 1 } };
  const out = summariseDirs(items, "B1", (id) => marks[id] as never);

  it("按目录内最晚时刻倒序，不按名字", () => {
    assert.deepEqual(out.map((s) => s.d), ["R/aaa_new", "R/zzz_old"]);
  });

  it("计数", () => {
    const old = out.find((s) => s.d === "R/zzz_old")!;
    assert.deepEqual([old.f, old.s, old.g, old.dup, old.atom, old.half, old.nw, old.mk, old.star], [2, 1, 0, 1, 1, 0, 0, 2, 1]);
    const nw = out.find((s) => s.d === "R/aaa_new")!;
    assert.deepEqual([nw.f, nw.g, nw.half, nw.nw], [1, 1, 1, 2]);
  });

  it("时间范围：网格用结束时刻", () => {
    const nw = out.find((s) => s.d === "R/aaa_new")!;
    assert.equal(nw.t0, 1000);
    assert.equal(nw.t1, 5000);
  });

  it("偏压按分组键去重（2 与 2.001 是同一组）、只算帧；帧宽同理", () => {
    const old = out.find((s) => s.d === "R/zzz_old")!;
    assert.equal(old.biases.length, 1);
    assert.equal(old.widths.length, 1);
  });

  it("前缀按条目数从多到少", () => {
    const old = out.find((s) => s.d === "R/zzz_old")!;
    assert.deepEqual(old.prefixes, ["p", "q"]);
  });

  it("没有批次时「新」一律为 0", () => {
    assert.ok(summariseDirs(items, "", () => null).every((s) => s.nw === 0));
  });
});

describe("shortList / newBatch", () => {
  it("超过 9 个加省略号，升序", () => {
    assert.equal(shortList([3, 1, 2], String), "1 2 3");
    assert.equal(shortList([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], String), "1 2 3 4 5 6 7 8 9 …");
  });

  it("最新一批", () => {
    assert.deepEqual(newBatch(items, "B1").map((x) => x.id), ["B/1", "B/2"]);
    assert.deepEqual(newBatch(items, ""), []);
  });
});
