// ════════════════════════════════════════════════════════════════════════════
// 数据图库的文字格式 — src/lib/gallery/format.ts
//
//     cd frontend && npm run test:unit
//
// 这些小函数决定卡片上印什么。它们错了不会报错，只会让「+20 mV」印成「+0.02 V」、
// 让两个目录的小标签印成同一个字——而操作员是靠这些字挑图的。
// 时间按浏览器本地时区显示，所以期望值也用本地时区的 Date 构造。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  autoTags,
  bKey,
  caption,
  dirLabel,
  dirShort,
  dirnameOf,
  dur,
  fmtB,
  fmtBShort,
  fmtT,
  fmtTs,
  fmtV,
  gridIncomplete,
  meta,
  nm,
  nowStr,
  num,
  partial,
  wKey,
} from "../src/lib/gallery/format.ts";

const local = (y: number, mo: number, d: number, h: number, mi: number, s = 0) =>
  new Date(y, mo - 1, d, h, mi, s).getTime() / 1000;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const item = (o: Record<string, unknown>): any => ({ id: "R/x", d: "R/20010910", fn: "x.sxm", p: "", pf: "", ad: "", th: "", ...o });

describe("时间", () => {
  it("fmtT / fmtTs 按本地时区", () => {
    const t = local(2001, 9, 10, 15, 7, 5);
    assert.equal(fmtT(t), "09-10 15:07");
    assert.equal(fmtTs(t), "15:07:05");
  });

  it("没有时刻印问号", () => {
    assert.equal(fmtT(null), "?");
    assert.equal(fmtT(0), "?");
    assert.equal(fmtTs(undefined), "?");
  });

  it("nowStr 是 YYYY-MM-DD HH:MM:SS", () => {
    assert.equal(nowStr(new Date(2001, 8, 3, 4, 5, 6)), "2001-09-03 04:05:06");
  });

  it("dur 分三档，负数取绝对值", () => {
    assert.equal(dur(5), "5 秒");
    assert.equal(dur(79), "1 分 19 秒");
    assert.equal(dur(3700), "1 小时 1 分");
    assert.equal(dur(-6), "6 秒");
  });
});

describe("偏压与帧宽", () => {
  it("小偏压用 mV（三位有效数字），其余 V 两位小数", () => {
    assert.equal(fmtB(-2), "−2.00 V");
    assert.equal(fmtB(0.5), "+0.50 V");
    assert.equal(fmtB(0.02), "+20 mV");
    assert.equal(fmtB(-0.0123), "−12.3 mV");
    assert.equal(fmtB(null), "?");
  });

  it("目录卡片上的紧凑写法", () => {
    assert.equal(fmtBShort(0.05), "+50mV");
    assert.equal(fmtBShort(-2.5), "−2.5");
  });

  it("fmtV 两位小数带符号", () => {
    assert.equal(fmtV(-2.5), "−2.50");
    assert.equal(fmtV(2), "+2.00");
    assert.equal(fmtV(undefined), "?");
  });

  it("分组键：小偏压保留到 mV，帧宽到 0.1 nm", () => {
    assert.equal(bKey(0.05), "0.050");
    assert.equal(bKey(-2), "-2.00");
    assert.equal(bKey(null), "");
    assert.equal(wKey(5.04), "5");
    assert.equal(wKey(29.96), "30");
    assert.equal(wKey(0), "");
    assert.equal(nm(150), 150);
    assert.equal(nm(3.14159), 3.142);
  });
});

describe("编号与目录", () => {
  it("帧取文件名最后四位，谱取整个主干", () => {
    assert.equal(num({ k: "f", fn: "sample_scan_0010.sxm" }), "0010");
    assert.equal(num({ k: "s", fn: "rep00002.dat" }), "rep00002");
    assert.equal(num({ k: "g", fn: "Grid Spectroscopy001.3ds" }), "Grid Spectroscopy001");
  });

  it("日期形目录显示成日期、MMDD；非日期形原样", () => {
    assert.equal(dirLabel("SPM/2001/200109/20010910"), "2001-09-10");
    assert.equal(dirShort("SPM/2001/200109/20010910"), "0910");
    assert.equal(dirLabel("WS/session_a"), "session_a");
    assert.equal(dirShort("WS/session_a"), "session_a");
    assert.equal(dirLabel("SPM"), "SPM");
  });

  it("dirnameOf 认两种分隔符", () => {
    assert.equal(dirnameOf("E:\\raw\\20010910\\a.sxm"), "E:\\raw\\20010910");
    assert.equal(dirnameOf("E:/raw/20010910/a.sxm"), "E:/raw/20010910");
  });
});

describe("未扫完 / 网格未完成", () => {
  it("有效行不到 95% 算未扫完", () => {
    assert.equal(partial(item({ k: "f", rows: 100, rall: 256 })), true);
    assert.equal(partial(item({ k: "f", rows: 250, rall: 256 })), false);
    assert.equal(partial(item({ k: "f", rall: 256 })), false, "没有 rows 不下结论");
    assert.equal(partial(item({ k: "s", rows: 1, rall: 256 })), false);
  });

  it("网格 have < gx·gy 算未完成", () => {
    assert.equal(gridIncomplete(item({ k: "g", have: 75, gx: 36, gy: 36 })), true);
    assert.equal(gridIncomplete(item({ k: "g", have: 1296, gx: 36, gy: 36 })), false);
    assert.equal(gridIncomplete(item({ k: "f", have: 0, gx: 3, gy: 3 })), false);
  });
});

describe("参数摘要与卡片说明", () => {
  const t = local(2001, 9, 10, 15, 7);
  it("帧", () => {
    const f = item({ k: "f", fn: "a_0010.sxm", t, w: 3, b: 2, sp: 100, nx: 192 });
    assert.equal(meta(f), "09-10 15:07 · 3 nm · +2.00 V · 100 pA · 192px");
    assert.deepEqual(caption(f), { num: "0010", when: "09-10 15:07", lines: ["3 nm · +2.00 V · 100 pA · 192px"] });
  });

  it("偏压谱：sweeps > 1 才写 sw", () => {
    const s = item({ k: "s", fn: "rep1.dat", t, n: 251, v0: -2.5, v1: 2, zo: 150, x: -65.36, y: 210.67, sw: 2 });
    assert.equal(meta(s), "09-10 15:07 · 谱 251 点 · −2.50…+2.00 V · Zoff 150 pm · (-65.36, 210.67) nm");
    assert.deepEqual(caption(s).lines, ["251 点 · −2.50…+2.00 V · Zoff 150 pm · 2 sw", "(-65.36, 210.67) nm"]);
    assert.equal(caption({ ...s, sw: 1 }).lines[0], "251 点 · −2.50…+2.00 V · Zoff 150 pm");
  });

  it("不是偏压谱的 .dat 写实验名", () => {
    const s = item({ k: "s", fn: "noise.dat", t, n: 128, ex: "Spectrum" });
    assert.equal(meta(s), "09-10 15:07 · Spectrum · 128 点（不是偏压谱）");
  });

  it("网格写起止时刻", () => {
    const g = item({ k: "g", fn: "Grid001.3ds", t, t1: t + 600, gx: 36, gy: 36, w: 3, n: 11, v0: 2, v1: -2.5, zo: 0, b: 2, sp: 300 });
    assert.equal(meta(g), "09-10 15:07 · 网格 36×36 · 3 nm · 11 点 +2.00→−2.50 V");
    assert.equal(caption(g).when, "09-10 15:07 → 09-10 15:17");
  });
});

describe("自动标签", () => {
  const numOf = (id: string) => (id === "R/a_0048.sxm" ? "0048" : null);

  it("每一种都在，顺序固定", () => {
    const f = item({
      k: "f", ad: "B1", at: 312.4, hf: 2.31, hl: "(0.5,0)", rows: 10, rall: 256,
      dup: "R/a_0048.sxm", seg: 3, cp: 2,
    });
    assert.deepEqual(
      autoTags(f, "B1", numOf).map((t) => [t.kind, t.text]),
      [
        ["new", "新"],
        ["atom", "原子 ×312"],
        ["half", "超结构 (0.5,0) ×2.31"],
        ["part", "未扫完 10/256"],
        ["dup", "重复保存 = 0048"],
        ["seg", "同一次采集 ×3"],
        ["copies", "副本 ×2"],
      ],
    );
  });

  it("原帧不在索引里时编号印问号", () => {
    const f = item({ k: "f", dup: "R/gone.sxm", rows: 256, rall: 256 });
    assert.equal(autoTags(f, "", numOf)[0]?.text, "重复保存 = ?");
  });

  it("网格未完成、谱的 LI", () => {
    assert.deepEqual(autoTags(item({ k: "g", have: 4, gx: 2, gy: 2 }), "", numOf).map((t) => t.text), []);
    assert.deepEqual(autoTags(item({ k: "g", have: 3, gx: 36, gy: 36 }), "", numOf).map((t) => t.text), ["未完成 3/1296"]);
    assert.deepEqual(autoTags(item({ k: "s", lic: 1 }), "", numOf).map((t) => t.text), ["LI"]);
  });

  it("没有批次时不打「新」；seg=1、cp=1 不打", () => {
    assert.deepEqual(autoTags(item({ k: "f", ad: "", seg: 1, cp: 1, rows: 256, rall: 256 }), "", numOf), []);
  });
});
