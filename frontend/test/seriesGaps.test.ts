// ════════════════════════════════════════════════════════════════════════════
// src/lib/seriesGaps.ts — 一条时间序列在哪里不再连续。
//
// 这些断言里有一半是**变异测试**：它们钉住的不是「功能能用」，而是「那个更简单的
// 写法是错的」。已知问题「数据中断的时候…强行连线」正是因为这条规则本来只存在于
// 环境历史页里，而每一个「本可以更简单」的子句都对应一种把它悄悄写坏的方式。
//
//     cd frontend && npm run test:unit
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { GAP_FACTOR, medianStep, spliceGaps } from "../src/lib/seriesGaps.ts";

describe("medianStep", () => {
  it("reads the cadence off the data", () => {
    assert.equal(medianStep([0, 1.3, 2.6, 3.9]), 1.3);
  });

  it("is not dragged by the very gap it has to find", () => {
    // 均值写法在这里给 (1+1+1+600)/4 ≈ 151 s，于是那个 600 s 的空档在它自己
    // 算出来的步长面前只有 4 倍——阈值一乘就再也不是空档了。空档越大越不像空档。
    const t = [0, 1, 2, 3, 603, 604, 605];
    assert.equal(medianStep(t), 1);
  });

  it("says 0 rather than guessing when there is nothing to measure", () => {
    assert.equal(medianStep([]), 0);
    assert.equal(medianStep([5]), 0);
    assert.equal(medianStep([5, 5, 5]), 0); // 时间戳不前进 = 没有间隔可言
  });
});

describe("spliceGaps", () => {
  const step = 1;

  it("leaves a continuous series alone", () => {
    const out = spliceGaps([0, 1, 2], [[10, 11, 12]], { step });
    assert.equal(out.gaps, 0);
    assert.deepEqual(out.t, [0, 1, 2]);
    assert.deepEqual(out.cols[0], [10, 11, 12]);
  });

  it("punches a null through EVERY column at a break", () => {
    // 少插一列的后果不是少一条线断不掉：band 图的两条边缘只要有一条没断，
    // uPlot 就会把中间那片色块糊过整个空档——而一片色块比一条直线更像「测过」。
    const out = spliceGaps([0, 1, 60, 61], [[1, 2, 3, 4], [9, 8, 7, 6]], { step });
    assert.equal(out.gaps, 1);
    assert.deepEqual(out.t, [0, 1, 2, 60, 61]);
    assert.deepEqual(out.cols[0], [1, 2, null, 3, 4]);
    assert.deepEqual(out.cols[1], [9, 8, null, 7, 6]);
  });

  it("keeps x strictly increasing even when the caller picks a silly factor", () => {
    // factor < 1 时 prev+step 会越过下一个真实样本，x 轴不再单调，uPlot 画出一条
    // 往回走的线。调用方守不守规矩不该由调用方决定。
    const out = spliceGaps([0, 1], [[1, 2]], { step: 10, factor: 0.01 });
    assert.equal(out.gaps, 1);
    assert.ok(out.t[0]! < out.t[1]! && out.t[1]! < out.t[2]!);
  });

  it("draws no breaks at all when the caller cannot say how dense the series is", () => {
    // step ≤ 0 = 「我不知道正常间隔是多少」。这时任何断点判定都是在编，
    // 所以一处都不划——环境历史在桶宽为 0 时就走这一支。
    assert.equal(spliceGaps([0, 3600], [[1, 2]], { step: 0 }).gaps, 0);
    assert.equal(spliceGaps([0, 3600], [[1, 2]], { step: -1 }).gaps, 0);
  });

  it("tolerates a skipped sample or two, then breaks", () => {
    // 采样是机会式的（角色锁被扫描占着就跳过），漏一两拍是正常运行不是中断——
    // 面板另有一处如实报「因通道被占跳过 N 次」。GAP_FACTOR 就是这条界线。
    assert.equal(spliceGaps([0, 2 * step], [[1, 2]], { step }).gaps, 0);
    assert.equal(spliceGaps([0, GAP_FACTOR * step], [[1, 2]], { step }).gaps, 0);
    assert.equal(spliceGaps([0, GAP_FACTOR * step + 0.001], [[1, 2]], { step }).gaps, 1);
  });

  it("passes a channel's own nulls through instead of closing the hole", () => {
    // 这一条就是辅助通道那个 bug 的最小复现：把 null 那一行**丢掉**的话，
    // 它两边的样本变成相邻，uPlot 直接连过去——空洞被填平了。
    const out = spliceGaps([0, 1, 2], [[1, null, 3]], { step });
    assert.equal(out.gaps, 0);
    assert.deepEqual(out.t, [0, 1, 2]);
    assert.deepEqual(out.cols[0], [1, null, 3]);
  });

  it("never lets a non-finite value reach the chart as a number", () => {
    const out = spliceGaps([0, 1, 2], [[1, NaN, Infinity]], { step });
    assert.deepEqual(out.cols[0], [1, null, null]);
  });

  it("drops rows whose timestamp is unusable, keeping columns aligned", () => {
    const out = spliceGaps([0, NaN, 2], [[1, 2, 3]], { step: 10 });
    assert.deepEqual(out.t, [0, 2]);
    assert.deepEqual(out.cols[0], [1, 3]);
  });
});
