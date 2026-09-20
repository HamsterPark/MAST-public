// ════════════════════════════════════════════════════════════════════════════
// 数据图库的视图 ↔ URL — src/lib/gallery/route.ts
//
//     cd frontend && npm run test:unit
//
// 深链要能收藏、能发给别人。认不出的地址回 null（由页面退回上次的视图），而不是
// 渲染一个空白页——那样用户报上来的会是「图库坏了」。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { isListView, navId, parseView, viewKey, viewSearch, type GalleryView } from "../src/lib/gallery/route.ts";

const VIEWS: GalleryView[] = [
  { v: "dirs" },
  { v: "dir", d: "SPM/2001/200109/20010910" },
  { v: "dir", d: "带 空格/与&符号" },
  { v: "all", k: "f" },
  { v: "all", k: "s" },
  { v: "all", k: "g" },
  { v: "new" },
  { v: "marked" },
  { v: "series", s: "fixture_series_00" },
  { v: "figures" },
  { v: "setup" },
];

describe("往返", () => {
  for (const view of VIEWS) {
    it(JSON.stringify(view), () => {
      assert.deepEqual(parseView(new URLSearchParams(viewSearch(view))), view);
    });
  }
});

describe("读时校验", () => {
  it("缺参数或不认识的视图回 null", () => {
    assert.equal(parseView(new URLSearchParams("")), null);
    assert.equal(parseView(new URLSearchParams("v=bogus")), null);
    assert.equal(parseView(new URLSearchParams("v=dir")), null);
    assert.equal(parseView(new URLSearchParams("v=series&s=")), null);
  });

  it("「全部」缺 k 或 k 不认识时看帧（原版默认）", () => {
    assert.deepEqual(parseView(new URLSearchParams("v=all")), { v: "all", k: "f" });
    assert.deepEqual(parseView(new URLSearchParams("v=all&k=x")), { v: "all", k: "f" });
  });
});

describe("身份与高亮", () => {
  it("viewKey 区分目录与系列", () => {
    assert.notEqual(viewKey({ v: "dir", d: "a" }), viewKey({ v: "dir", d: "b" }));
    assert.equal(viewKey({ v: "marked" }), "marked");
  });

  it("navId：某目录 / 最新一批 / 系列页不高亮顶栏", () => {
    assert.equal(navId({ v: "all", k: "s" }), "all-s");
    assert.equal(navId({ v: "setup" }), "setup");
    assert.equal(navId({ v: "dir", d: "x" }), "");
    assert.equal(navId({ v: "series", s: "S" }), "");
  });

  it("列表类视图", () => {
    assert.deepEqual(VIEWS.filter(isListView).map((v) => v.v), ["dir", "dir", "all", "all", "all", "new", "series"]);
  });
});
