// ════════════════════════════════════════════════════════════════════════════
// STS 曲线的数据整形 — src/lib/spectrumSeries.ts
//
// A .dat used to reach this app as a ~160 px PNG of its first two columns. The
// chart replaced that, and the distinction that must survive is that **dI/dV has
// three states**: a measured lock-in channel, a numeric derivative of I(V), and
// nothing. Drawing the middle one under a bare "dI/dV" label shows the operator
// a derivative of noise where they expect a modulated measurement.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  availableModes,
  buildSpectrumData,
  numericDidvNote,
  resolveMode,
} from "../src/lib/spectrumSeries.ts";

const IV = {
  kind: "iv",
  sweep_name: "Bias calc (V)",
  sweep: [-1, 0, 1],
  didv_source: "lockin",
  columns: ["Bias calc (V)", "Current (A)", "Current [bwd] (A)", "LIX 1 omega (A)"],
  series: [
    { id: "current", name: "Current (A)", values: [-1e-9, 0, 1e-9], source: "file" },
    { id: "current_bwd", name: "Current [bwd] (A)", values: [-1.1e-9, 0, 1.1e-9], source: "file" },
    { id: "didv", name: "LIX 1 omega (A)", values: [1e-9, 2e-9, 1e-9], source: "file" },
  ],
};

describe("availableModes", () => {
  it("只给这份文件真的画得出来的模式", () => {
    assert.deepEqual(availableModes(IV).map((m) => m.id), ["current", "didv"]);
  });

  it("I(z) 谱不叫 I-V", () => {
    // `kind` comes from which column is actually sweeping, not from the file
    // header — 字段标签会说谎. Labelling an I(z) curve "I-V" mislabels the physics.
    const iz = { ...IV, kind: "iz", series: [IV.series[0]!] };
    assert.equal(availableModes(iz)[0]!.label, "I-z");
    assert.equal(availableModes(IV)[0]!.label, "I-V");
  });

  it("认不出角色的文件走「全部通道」", () => {
    const generic = {
      sweep: [0, 1],
      series: [{ id: "col1", name: "Signal A", values: [1, 2], source: "file" }],
    };
    assert.deepEqual(availableModes(generic).map((m) => m.id), ["other"]);
  });

  it("没有数据就没有模式（绝不给一张空图）", () => {
    assert.deepEqual(availableModes(null), []);
    assert.deepEqual(availableModes({ sweep: [], series: [] }), []);
  });
});

describe("resolveMode", () => {
  it("这份文件支持就沿用操作员的选择", () => {
    assert.equal(resolveMode(IV, "didv"), "didv");
  });

  it("不支持就退回第一个可用的，而不是画空", () => {
    const ivOnly = { ...IV, series: [IV.series[0]!], didv_source: null };
    assert.equal(resolveMode(ivOnly, "didv"), "current");
  });

  it("什么都没有时返回 null", () => {
    assert.equal(resolveMode(null, "current"), null);
  });
});

describe("buildSpectrumData", () => {
  it("第一列是扫描轴，后面每条曲线一列", () => {
    const p = buildSpectrumData(IV, "current")!;
    assert.equal(p.data.length, 3);                       // x + fwd + bwd
    assert.deepEqual(p.data[0], [-1, 0, 1]);
    assert.deepEqual(p.series.map((s) => s.id), ["current", "current_bwd"]);
    assert.equal(p.xLabel, "Bias calc (V)");
  });

  it("dI/dV 模式只取 dI/dV", () => {
    const p = buildSpectrumData(IV, "didv")!;
    assert.deepEqual(p.series.map((s) => s.id), ["didv"]);
    assert.equal(p.yLabel, "dI/dV (A/V)");
  });

  it("长度对不上时补 null，而不是错位配对", () => {
    // uPlot pairs by index without checking. A short series silently drawn
    // against the wrong x values is a plot that looks fine and is wrong.
    const short = {
      ...IV,
      series: [{ id: "current", name: "I", values: [1e-9], source: "file" }],
    };
    const p = buildSpectrumData(short, "current")!;
    assert.equal(p.data[1]!.length, 3);
    assert.deepEqual(p.data[1], [1e-9, null, null]);
  });

  it("null 保留下来（图上空档不能连线）", () => {
    const gap = {
      ...IV,
      series: [{ id: "current", name: "I", values: [1e-9, null, 3e-9], source: "file" }],
    };
    const p = buildSpectrumData(gap, "current")!;
    assert.equal(p.data[1]![1], null);
  });

  it("标出画的是不是数值微分", () => {
    const numeric = {
      ...IV,
      didv_source: "numeric",
      series: [{ id: "didv", name: "dI/dV（数值微分）", values: [1, 2, 3], source: "numeric" }],
    };
    assert.equal(buildSpectrumData(numeric, "didv")!.hasNumeric, true);
    assert.equal(buildSpectrumData(IV, "didv")!.hasNumeric, false);
  });

  it("认不出角色时用真实列名当纵轴标签", () => {
    // Not "I (A)": inventing a label for a column we cannot identify is how an
    // earlier bug came to label every spectrum figure "I (A)", dI/dV included.
    const generic = {
      sweep: [0, 1],
      series: [{ id: "col1", name: "Signal A", values: [1, 2], source: "file" }],
    };
    assert.equal(buildSpectrumData(generic, "other")!.yLabel, "Signal A");
  });

  it("没有扫描轴 / 没有匹配曲线就返回 null", () => {
    assert.equal(buildSpectrumData({ ...IV, sweep: [] }, "current"), null);
    assert.equal(buildSpectrumData(IV, "other"), null);
    assert.equal(buildSpectrumData(null, "current"), null);
    assert.equal(buildSpectrumData(IV, null), null);
  });
});

describe("numericDidvNote", () => {
  it("实测 lock-in 不加注解", () => {
    assert.equal(numericDidvNote(IV), null);
  });

  it("数值微分时说清它不是实测调制信号", () => {
    const note = numericDidvNote({ ...IV, didv_source: "numeric" });
    assert.ok(note && note.includes("数值微分"), String(note));
    assert.ok(note!.includes("lock-in"), note!);
  });
});
