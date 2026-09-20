// ════════════════════════════════════════════════════════════════════════════
// 数据图库的系列 — src/lib/gallery/series.ts
//
//     cd frontend && npm run test:unit
//
// 系列是一段连续数据的名字。它最怕的失效是「并入」时把索引里暂时没有的成员静默
// 删掉（换数据根、某个目录还没重建）——下面有一条专门钉它。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  autoName,
  kindOf,
  mergedMemberIds,
  newSeriesId,
  rangeText,
  seriesCountByDir,
  seriesIndex,
  seriesMembers,
  seriesOf,
  shortFn,
  sortSeriesEntries,
} from "../src/lib/gallery/series.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mk = (o: Record<string, unknown>): any => ({ p: "", pf: "", ad: "", th: "", ...o });
const local = (mo: number, d: number, h: number, mi: number) => new Date(2001, mo - 1, d, h, mi).getTime() / 1000;

const f1 = mk({ id: "R/d1/s_0001.sxm", d: "R/20010907", k: "f", fn: "s_0001.sxm", b: 2, t: local(9, 7, 23, 1) });
const f2 = mk({ id: "R/d1/s_0002.sxm", d: "R/20010907", k: "f", fn: "s_0002.sxm", b: -2, t: local(9, 7, 23, 7) });
const s1 = mk({ id: "R/d2/rep1.dat", d: "R/20010908", k: "s", fn: "rep1.dat", t: local(9, 8, 1, 0), mt: local(9, 8, 1, 5) });
const ITEMS = [f1, f2, s1];

describe("系列索引", () => {
  const series = { S1: { name: "a", k: "f", ids: [f1.id, f2.id], r: 0, tags: [], note: "" }, S2: { name: "b", k: "mix", ids: [f2.id, s1.id], r: 0, tags: [], note: "" } };

  it("id → 系列号，按 series 对象身份缓存", () => {
    assert.deepEqual(seriesOf(series, f2.id), ["S1", "S2"]);
    assert.deepEqual(seriesOf(series, "nope"), []);
    assert.equal(seriesIndex(series), seriesIndex(series));
    assert.notEqual(seriesIndex(series), seriesIndex({ ...series }));
  });

  it("成员按索引顺序", () => {
    assert.deepEqual(seriesMembers({ name: "", k: "", ids: [s1.id, f1.id], r: 0, tags: [], note: "" }, ITEMS).map((x) => x.id), [f1.id, s1.id]);
    assert.deepEqual(seriesMembers(undefined, ITEMS), []);
  });

  it("每个目录涉及几个系列，跨目录两边各算一次", () => {
    const byId = new Map(ITEMS.map((x) => [x.id, x]));
    assert.deepEqual(seriesCountByDir(series, byId), { "R/20010907": 2, "R/20010908": 1 });
  });
});

describe("命名与文案", () => {
  it("kindOf", () => {
    assert.equal(kindOf([f1, f2]), "f");
    assert.equal(kindOf([f1, s1]), "mix");
    assert.equal(kindOf([]), "mix");
  });

  it("默认名：帧的偏压不止一个时附「第一个成员 … 最后一个成员」的偏压（原版如此，偏压系列读得出方向）", () => {
    assert.equal(autoName([f1, f2]), "2001-09-07 0001–0002 · 2 帧 · +2.00 V…−2.00 V");
    assert.equal(autoName([s1]), "2001-09-08 rep1–rep1 · 1 谱");
    assert.equal(autoName([f1, s1]), "2001-09-07 0001–rep1 · 2 个");
  });

  it("系定小标签：长文件名末尾是编号时只留编号", () => {
    assert.equal(shortFn("sample_STM_STS_002_0003.sxm"), "0003");
    assert.equal(shortFn("unnamed0033.sxm"), "unnamed0033");
    assert.equal(shortFn(undefined), "");
  });

  it("范围文案跨目录时写两边的日期", () => {
    assert.equal(rangeText([f1, s1]), " 2001-09-07 0001 → 2001-09-08 rep1（09-07 23:01 → 09-08 01:05）");
    assert.equal(rangeText([]), "");
  });
});

describe("存为 / 并入系列", () => {
  it("索引里有的按索引顺序，旧成员里索引没有的留在末尾", () => {
    const out = mergedMemberIds(["R/old/missing.sxm", s1.id], [f2, f1], ITEMS);
    assert.deepEqual(out, [f1.id, f2.id, s1.id, "R/old/missing.sxm"]);
  });

  it("系列号", () => {
    assert.equal(newSeriesId(1000382400000), `S${(1000382400000).toString(36)}`);
  });

  it("系列表按第一个成员的开始时刻排", () => {
    const series = {
      Late: { name: "l", k: "s", ids: [s1.id], r: 0, tags: [], note: "" },
      Early: { name: "e", k: "f", ids: [f1.id], r: 0, tags: [], note: "" },
    };
    assert.deepEqual(sortSeriesEntries(series, ITEMS).map((e) => e.sid), ["Early", "Late"]);
  });
});
