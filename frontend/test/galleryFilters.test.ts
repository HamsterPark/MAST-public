// ════════════════════════════════════════════════════════════════════════════
// 数据图库的筛选 — src/lib/gallery/filters.ts
//
//     cd frontend && npm run test:unit
//
// 每一个分支一条用例。最要紧的一条是「重复保存默认隐藏，但带标记或在系列里的照样
// 显示」：它被删掉时没有任何东西会报错，只会让一张打过 ★ 的图从目录里消失。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  DEFAULT_FILTERS,
  applyFilters,
  facets,
  parseStoredFilters,
  sanitizeFilters,
  serializeFilters,
  type FilterContext,
  type Filters,
} from "../src/lib/gallery/filters.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;
const mk = (o: Record<string, unknown>): Any => ({ d: "R/20010910", p: "", pf: "", ad: "", th: "", ...o });

const f1 = mk({ id: "R/a_0001.sxm", k: "f", fn: "a_0001.sxm", pf: "a", b: 2, w: 5, at: 300, hf: 0, rows: 256, rall: 256 });
const f2 = mk({ id: "R/a_0002.sxm", k: "f", fn: "a_0002.sxm", pf: "a", b: 2, w: 5, at: 0, hf: 0, rows: 256, rall: 256, dup: "R/a_0001.sxm" });
const f3 = mk({ id: "R/b_0003.sxm", k: "f", fn: "b_0003.sxm", pf: "b", b: -0.05, w: 10, at: 0, hf: 2.3, rows: 100, rall: 256 });
const f4 = mk({ id: "R/a_0004.sxm", k: "f", fn: "a_0004.sxm", pf: "a", b: 1, w: 5.04, at: 0, hf: 0, rows: 256, rall: 256, li: "/api/x.li.jpg" });
const s1 = mk({ id: "R/rep1.dat", k: "s", fn: "rep1.dat", pf: "rep", b: 2, lic: 1 });
const s2 = mk({ id: "R/rep2.dat", k: "s", fn: "rep2.dat", pf: "rep", b: 2, lic: 0 });
const g1 = mk({ id: "R/Grid001.3ds", k: "g", fn: "Grid001.3ds", pf: "Grid", b: 2, w: 3, have: 75, gx: 36, gy: 36 });
const ALL = [f1, f2, f3, f4, s1, s2, g1];

function ctx(view: FilterContext["view"], marks: Record<string, Any> = {}, series: Record<string, string[]> = {}): FilterContext {
  return { view, mark: (id) => marks[id] ?? null, seriesOf: (id) => series[id] ?? [] };
}
const F = (o: Partial<Filters>): Filters => ({ ...DEFAULT_FILTERS, nodup: false, ...o });
const ids = (xs: Any[]) => xs.map((x) => x.id);

describe("类型 / 偏压 / 帧宽 / 前缀", () => {
  it("类型在目录视图里起作用，在「全部X」视图里不起作用", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ k: "s" }), ctx("dir"))), ["R/rep1.dat", "R/rep2.dat"]);
    assert.equal(applyFilters(ALL, F({ k: "s" }), ctx("all")).length, ALL.length);
  });

  it("偏压筛选排除谱（谱的偏压是扫描范围不是工作点）", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ b: "2.00" }), ctx("dir"))), ["R/a_0001.sxm", "R/a_0002.sxm", "R/Grid001.3ds"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ b: "-0.050" }), ctx("dir"))), ["R/b_0003.sxm"]);
  });

  it("帧宽按 0.1 nm 分组", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ w: "5" }), ctx("dir"))), ["R/a_0001.sxm", "R/a_0002.sxm", "R/a_0004.sxm"]);
  });

  it("文件名前缀", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ pf: "rep" }), ctx("dir"))), ["R/rep1.dat", "R/rep2.dat"]);
  });
});

describe("衬度与完整性", () => {
  it("atom / half / li", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ c: "atom" }), ctx("dir"))), ["R/a_0001.sxm"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ c: "half" }), ctx("dir"))), ["R/b_0003.sxm"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ c: "li" }), ctx("dir"))), ["R/a_0004.sxm", "R/rep1.dat", "R/Grid001.3ds"]);
  });

  it("只看扫完：排除未扫完的帧与没做完的网格", () => {
    const out = ids(applyFilters(ALL, F({ full: true }), ctx("dir")));
    assert.ok(!out.includes("R/b_0003.sxm"));
    assert.ok(!out.includes("R/Grid001.3ds"));
    assert.ok(out.includes("R/a_0001.sxm"));
  });
});

describe("重复保存", () => {
  const nodup = F({ nodup: true });

  it("目录视图默认隐藏重复保存", () => {
    assert.ok(!ids(applyFilters(ALL, nodup, ctx("dir"))).includes("R/a_0002.sxm"));
  });

  it("带标记的重复帧照样显示", () => {
    const marks = { "R/a_0002.sxm": { r: 2, tags: [], note: "" } };
    assert.ok(ids(applyFilters(ALL, nodup, ctx("dir", marks))).includes("R/a_0002.sxm"));
  });

  it("在系列里的重复帧照样显示", () => {
    const series = { "R/a_0002.sxm": ["S1"] };
    assert.ok(ids(applyFilters(ALL, nodup, ctx("dir", {}, series))).includes("R/a_0002.sxm"));
  });

  it("已标记页、系列页里不隐藏", () => {
    assert.ok(ids(applyFilters(ALL, nodup, ctx("marked"))).includes("R/a_0002.sxm"));
    assert.ok(ids(applyFilters(ALL, nodup, ctx("series"))).includes("R/a_0002.sxm"));
  });
});

describe("标记 / 标签 / 搜索 / 排序", () => {
  const marks = {
    "R/a_0001.sxm": { r: 2, tags: ["原子分辨佳"], note: "好帧 Defect" },
    "R/b_0003.sxm": { r: -1, tags: [], note: "" },
    "R/rep1.dat": { r: 1, tags: [], note: "", anchor: { id: "R/a_0001.sxm", fn: "a", rel: "next", dt: 4, desc: "" } },
  };
  const c = ctx("dir", marks, { "R/a_0004.sxm": ["S9"] });

  it("none / any", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ m: "any" }), c)), ["R/a_0001.sxm", "R/b_0003.sxm", "R/rep1.dat"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ m: "none" }), c)), ["R/a_0002.sxm", "R/a_0004.sxm", "R/rep2.dat", "R/Grid001.3ds"]);
  });

  it("按评级", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ m: "2" }), c)), ["R/a_0001.sxm"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ m: "1" }), c)), ["R/rep1.dat"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ m: "-1" }), c)), ["R/b_0003.sxm"]);
  });

  it("隐藏 ✗", () => {
    assert.ok(!ids(applyFilters(ALL, F({ m: "hidex" }), c)).includes("R/b_0003.sxm"));
    assert.equal(applyFilters(ALL, F({ m: "hidex" }), c).length, ALL.length - 1);
  });

  it("属于系列 / 位置已系定", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ m: "ser" }), c)), ["R/a_0004.sxm"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ m: "anc" }), c)), ["R/rep1.dat"]);
  });

  it("标签", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ tag: "原子分辨佳" }), c)), ["R/a_0001.sxm"]);
  });

  it("搜索文件名或备注，不区分大小写", () => {
    assert.deepEqual(ids(applyFilters(ALL, F({ q: "REP2" }), c)), ["R/rep2.dat"]);
    assert.deepEqual(ids(applyFilters(ALL, F({ q: "defect" }), c)), ["R/a_0001.sxm"]);
  });

  it("时间 ↓ 整体反转，不修改输入", () => {
    const copy = [...ALL];
    assert.deepEqual(ids(applyFilters(ALL, F({ sort: "desc" }), c)), ids([...ALL].reverse()));
    assert.deepEqual(ids(ALL), ids(copy));
  });
});

describe("下拉选项与持久化", () => {
  it("facets：谱不进偏压/帧宽，按数值升序；前缀带计数", () => {
    const fac = facets(ALL);
    assert.deepEqual(fac.biases.map(([k]) => k), ["-0.050", "1.00", "2.00"]);
    assert.deepEqual(fac.widths.map(([k]) => k), ["3", "5", "10"]);
    assert.deepEqual(fac.prefixes, [["a", 3], ["b", 1], ["Grid", 1], ["rep", 2]]);
  });

  it("选中的值在当前列表里没有时退回全部", () => {
    const fac = facets([s1, s2]);
    const out = sanitizeFilters(F({ b: "2.00", w: "5", pf: "a", tag: "gone" }), fac, ["x"]);
    assert.deepEqual([out.b, out.w, out.pf, out.tag], ["", "", "", ""]);
    const kept = sanitizeFilters(F({ pf: "rep", tag: "x" }), fac, ["x"]);
    assert.deepEqual([kept.pf, kept.tag], ["rep", "x"]);
  });

  it("读回来的筛选逐项校验，搜索词不留", () => {
    assert.deepEqual(parseStoredFilters("{not json"), DEFAULT_FILTERS);
    assert.deepEqual(parseStoredFilters(null), DEFAULT_FILTERS);
    const back = parseStoredFilters(JSON.stringify({ k: "zzz", m: "2", c: "half", q: "keep?", cw: 9999, nodup: "no", sort: "desc" }));
    assert.equal(back.k, "all");
    assert.equal(back.m, "2");
    assert.equal(back.c, "half");
    assert.equal(back.q, "");
    assert.equal(back.cw, 480);
    assert.equal(back.nodup, true);
    assert.equal(back.sort, "desc");
  });

  it("存的时候不存搜索词", () => {
    assert.equal(JSON.parse(serializeFilters(F({ q: "secret" }))).q, "");
  });
});
