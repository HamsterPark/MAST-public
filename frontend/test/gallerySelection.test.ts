// ════════════════════════════════════════════════════════════════════════════
// 数据图库的选中 — src/lib/gallery/selection.ts
//
//     cd frontend && npm run test:unit
//
// 连选是「存为系列」的前一步：148 帧的转角系列就是 Shift 点首尾选出来的。连选选错
// 一张不会报错，只会让系列里多一张或少一张。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { pickClick, pickKey, rangeIds, type SelectionState } from "../src/lib/gallery/selection.ts";

const L = ["a", "b", "c", "d", "e"].map((id) => ({ id }));
const empty: SelectionState = { ids: new Set(), lastId: null, startId: null };
const sorted = (st: SelectionState) => [...st.ids].sort();

describe("rangeIds", () => {
  it("两个方向都含两端", () => {
    assert.deepEqual(rangeIds(L, "b", 3), ["b", "c", "d"]);
    assert.deepEqual(rangeIds(L, "d", 1), ["b", "c", "d"]);
  });

  it("起点不在列表里只含终点", () => {
    assert.deepEqual(rangeIds(L, "zzz", 2), ["c"]);
    assert.deepEqual(rangeIds(L, null, 0), ["a"]);
  });
});

describe("卡片勾选框", () => {
  it("单击选、再点取消，记住上次点的", () => {
    const s1 = pickClick(empty, L, 1, false, true);
    assert.deepEqual(sorted(s1), ["b"]);
    assert.equal(s1.lastId, "b");
    assert.deepEqual(sorted(pickClick(s1, L, 1, false, false)), []);
  });

  it("Shift 从上次点的那张连选到这一张，只加不减", () => {
    const s1 = pickClick(empty, L, 1, false, true);
    const s2 = pickClick(s1, L, 3, true, true);
    assert.deepEqual(sorted(s2), ["b", "c", "d"]);
    const s3 = pickClick(s2, L, 4, true, false);
    assert.deepEqual(sorted(s3), ["b", "c", "d", "e"]);
  });

  it("没点过任何一张时 Shift 等于单击", () => {
    assert.deepEqual(sorted(pickClick(empty, L, 2, true, true)), ["c"]);
  });

  it("不修改入参", () => {
    const s1 = pickClick(empty, L, 0, false, true);
    pickClick(s1, L, 4, true, true);
    assert.deepEqual(sorted(s1), ["a"]);
  });
});

describe("大图键盘", () => {
  it("S 取反", () => {
    const s1 = pickKey(empty, L, 2, "KeyS");
    assert.deepEqual(sorted(s1), ["c"]);
    assert.deepEqual(sorted(pickKey(s1, L, 2, "KeyS")), []);
  });

  it("[ 定起点并选中，] 从起点选到当前", () => {
    const s1 = pickKey(empty, L, 1, "BracketLeft");
    assert.equal(s1.startId, "b");
    const s2 = pickKey({ ...s1, lastId: "a" }, L, 4, "BracketRight");
    assert.deepEqual(sorted(s2), ["b", "c", "d", "e"]);
  });

  it("没有起点时 ] 从上次那张选起", () => {
    const s1 = pickKey(empty, L, 0, "KeyS");
    assert.deepEqual(sorted(pickKey(s1, L, 2, "BracketRight")), ["a", "b", "c"]);
  });
});
