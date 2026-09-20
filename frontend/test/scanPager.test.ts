// ════════════════════════════════════════════════════════════════════════════
// 数据 tab 的翻页累积 — src/lib/scanPager.ts
//
//     cd frontend && npm run test:unit
//
// The listing asked for a fixed 60 files and printed "只显示最近 60 个"
// underneath; file 61 was unreachable. Paging replaced that, and the accumulator
// has one job beyond concatenation: the rig writes to these directories WHILE
// the operator pages, so a scan saved between page 1 and page 2 shifts every
// later entry down a slot and page 2 arrives starting with a row already on
// screen. Appending blindly shows it twice — on the very tab whose other fix
// this round was to stop showing one measurement twice.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { appendScans, nextOffset, pageSummary } from "../src/lib/scanPager.ts";

const f = (path: string) => ({ path });

describe("appendScans", () => {
  it("把下一页接在后面", () => {
    const out = appendScans([f("a"), f("b")], [f("c"), f("d")]);
    assert.deepEqual(out.map((s) => s.path), ["a", "b", "c", "d"]);
  });

  it("吸收翻页期间落盘造成的重叠", () => {
    // A new scan landed after page 1 was served: everything shifted down one, so
    // page 2 repeats the last row of page 1.
    const page1 = [f("newest"), f("second"), f("third")];
    const page2 = [f("third"), f("fourth")];
    const out = appendScans(page1, page2);
    assert.deepEqual(out.map((s) => s.path), ["newest", "second", "third", "fourth"]);
  });

  it("完全重复的一页不改变任何东西", () => {
    const page1 = [f("a"), f("b")];
    assert.deepEqual(appendScans(page1, page1).map((s) => s.path), ["a", "b"]);
  });

  it("不修改传进来的数组", () => {
    const prev = [f("a")];
    appendScans(prev, [f("b")]);
    assert.equal(prev.length, 1);
  });

  it("空页 / 空累积都不特殊", () => {
    assert.deepEqual(appendScans([], [f("a")]).map((s) => s.path), ["a"]);
    assert.deepEqual(appendScans([f("a")], []).map((s) => s.path), ["a"]);
    assert.deepEqual(appendScans([], []), []);
  });

  it("保持服务端给的顺序（mtime 倒序）", () => {
    const out = appendScans([f("t3"), f("t2")], [f("t1"), f("t0")]);
    assert.deepEqual(out.map((s) => s.path), ["t3", "t2", "t1", "t0"]);
  });
});

describe("nextOffset", () => {
  it("按已经拿到的条数算，而不是按点了几次", () => {
    // Page 2 arrived one entry short because of the overlap above. Asking for
    // offset 60 next would skip a file; asking for what we actually hold does not.
    assert.equal(nextOffset([f("a"), f("b"), f("c")]), 3);
    assert.equal(nextOffset([]), 0);
  });
});

describe("pageSummary", () => {
  it("没有副本时不提副本", () => {
    const s = pageSummary(30, 100, 100);
    assert.ok(s.includes("30 / 100"));
    assert.ok(!s.includes("折叠"), `不该提折叠：${s}`);
  });

  it("有副本时说清折叠了几个、原本几个文件", () => {
    const s = pageSummary(30, 100, 130);
    assert.ok(s.includes("折叠 30"), s);
    assert.ok(s.includes("130"), s);
  });

  it("什么都没有就不说话", () => {
    assert.equal(pageSummary(0, 0, 0), "");
  });
});
