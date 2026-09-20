import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  RANGES,
  asdUnit,
  axisLabel,
  channelLabel,
  coverageOf,
  ctxStabilityNote,
  excludedCount,
  fmtBytes,
  fmtSpan,
  isDutyCycleSeries,
  preferLogScale,
  rangeWindow,
  spectrumPickerLabel,
  spectrumSeries,
  statusLabel,
  statusTone,
  toAlignedData,
  toAsd,
  tunnellingLabel,
  worstOf,
  type EnvPointLike,
} from "../src/lib/envHistory.ts";
import { buildKnobPayload, clampKnob, fmtKnobValue, type KnobLike } from "../src/lib/knobs.ts";

const pt = (ts: number, extra: Partial<EnvPointLike> = {}): EnvPointLike => ({
  ts,
  mean: 1,
  min: 0,
  max: 2,
  n: 30,
  n_excluded: 0,
  worst_status: "ok",
  ...extra,
});

describe("rangeWindow", () => {
  it("turns a named range into a since, and leaves until open", () => {
    const now = 1_800_000_000_000;
    assert.deepEqual(rangeWindow(3600, now), { since: 1_800_000_000 - 3600 });
  });

  it("returns an empty window for 全部", () => {
    assert.deepEqual(rangeWindow(null, Date.now()), {});
  });

  it("every shipped range is usable", () => {
    for (const r of RANGES) {
      const w = rangeWindow(r.s, 1_800_000_000_000);
      if (r.s == null) assert.equal(w.since, undefined);
      else assert.ok((w.since ?? 0) > 0);
    }
  });

  // 「如果我想回看过去 48 小时的温度……」。钉的是**这一档在不在
  // 出厂列表里**，不是 rangeWindow(172800) 算得对不对 —— 后者从来没错过，缺的
  // 一直是那个按钮。
  it("ships a 48 小时 range ", () => {
    const r = RANGES.find((x) => x.s === 2 * 86400);
    assert.ok(r, "48 小时 档不在 RANGES 里");
    assert.equal(r.label, "48 小时");
  });

  // 页面的默认档是按**跨度值**选的（不是下标），所以这个值必须真的在菜单里。
  // 从前默认写的是 `useState(1)`，往中间插一档就会静默换掉默认；这条断言让
  // 「默认档」与「菜单」不可能各说各的。
  it("the page's default span is one of the shipped ranges", () => {
    assert.ok(RANGES.some((r) => r.s === 86400), "默认档 24 小时 不在菜单里");
  });

  // 菜单从短到长、「全部」压轴。乱序不会让任何东西报错，只会让那一排按钮读起来
  // 像随手堆的；而真正会咬人的是 s 写错量级（`2 * 8640`），它同样只表现为
  // 顺序不对 —— 一条断言两件事都管。
  it("ranges are strictly increasing, with 全部 last and no duplicates", () => {
    const spans = RANGES.map((r) => r.s);
    assert.equal(spans.indexOf(null), spans.length - 1, "只允许有一个 null，且必须在最后");
    const finite = spans.slice(0, -1) as number[];
    for (let i = 1; i < finite.length; i += 1) {
      assert.ok(finite[i]! > finite[i - 1]!, `RANGES 不是严格递增：${finite[i - 1]} → ${finite[i]}`);
    }
    assert.equal(new Set(RANGES.map((r) => r.label)).size, RANGES.length, "档位标签有重复");
  });
});

describe("toAlignedData", () => {
  it("keeps contiguous buckets contiguous", () => {
    const [t, mean] = toAlignedData([pt(0), pt(60), pt(120)], 60);
    assert.deepEqual(t, [0, 60, 120]);
    assert.deepEqual(mean, [1, 1, 1]);
  });

  it("inserts a null so a recording gap draws as a HOLE, not a straight line", () => {
    // Six hours missing between two buckets: joining them would draw a line
    // through time when nothing was recorded.
    const [t, mean, lo, hi] = toAlignedData([pt(0), pt(21600)], 60);
    assert.equal(t.length, 3);
    assert.equal(mean[1], null);
    assert.equal(lo[1], null);
    assert.equal(hi[1], null);
    assert.equal(t[1], 60); // the break sits right after the last real bucket
  });

  it("passes nulls through for buckets with no usable readings", () => {
    const [, mean] = toAlignedData([pt(0, { mean: null })], 60);
    assert.equal(mean[0], null);
  });

  it("survives a zero bucket width without inventing gaps", () => {
    const [t] = toAlignedData([pt(0), pt(3600)], 0);
    assert.deepEqual(t, [0, 3600]);
  });
});

describe("preferLogScale", () => {
  it("chooses log for a vacuum gauge spanning decades", () => {
    const pts = [pt(0, { mean: 1e-9 }), pt(60, { mean: 1e-3 })];
    assert.equal(preferLogScale("Pa", pts), true);
  });

  it("stays linear for a temperature that barely moves", () => {
    const pts = [pt(0, { mean: 77.1 }), pt(60, { mean: 77.4 })];
    assert.equal(preferLogScale("K", pts), false);
  });

  it("never picks log when a value is zero or negative", () => {
    // log(0) is what makes uPlot render an entirely empty plot.
    const pts = [pt(0, { mean: 0 }), pt(60, { mean: 1e6 })];
    assert.equal(preferLogScale("Pa", pts), false);
  });
});

describe("worstOf / excludedCount", () => {
  it("uses the backend's severity order, not lexicographic", () => {
    // 'warning' > 'error' as strings, but 'error' is worse.
    assert.equal(worstOf([pt(0, { worst_status: "warning" }), pt(60, { worst_status: "error" })]),
                 "error");
    assert.equal(worstOf([pt(0, { worst_status: "alarm" }), pt(60, { worst_status: "error" })]),
                 "alarm");
  });

  it("treats unavailable as worse than ok but better than a warning", () => {
    assert.equal(worstOf([pt(0, { worst_status: "unavailable" }), pt(60)]), "unavailable");
    assert.equal(
      worstOf([pt(0, { worst_status: "unavailable" }), pt(60, { worst_status: "warning" })]),
      "warning",
    );
  });

  it("sums the readings the quiet gate kept out", () => {
    assert.equal(excludedCount([pt(0, { n_excluded: 3 }), pt(60, { n_excluded: 4 })]), 7);
  });
});

describe("statusTone / statusLabel — severity, not a binary ", () => {
  it("does not paint the least severe non-ok state in the alarm colour", () => {
    // The bug: `worst === "warning" ? "WARN" : "DANGEROUS"` gave `unavailable`
    // (severity 1) the same red as `alarm` (severity 4).
    assert.equal(statusTone("unavailable"), "INFO");
    assert.equal(statusTone("warning"), "WARN");
    assert.equal(statusTone("error"), "DANGEROUS");
    assert.equal(statusTone("alarm"), "DANGEROUS");
  });

  it("orders tones monotonically with the severity ladder worstOf uses", () => {
    const ladder = ["ok", "unavailable", "warning", "error", "alarm"];
    const rank: Record<string, number> = { AUTO: 0, INFO: 1, WARN: 2, DANGEROUS: 3 };
    const tones = ladder.map((s) => rank[statusTone(s)]);
    for (let i = 1; i < tones.length; i += 1) {
      assert.ok(tones[i]! >= tones[i - 1]!, `${ladder[i]} must not be milder than ${ladder[i - 1]}`);
    }
  });

  it("says what unavailable means instead of showing the raw enum", () => {
    assert.equal(statusLabel("unavailable"), "部分时段无读数");
    assert.equal(statusLabel("ok"), "正常");
  });
});

describe("coverageOf", () => {
  it("reports counted alongside excluded — a bare excluded count is unreadable", () => {
    const out = coverageOf([
      pt(0, { n: 20, n_excluded: 10 }),
      pt(60, { n: 30, n_excluded: 5 }),
    ]);
    assert.deepEqual(out, { counted: 50, excluded: 15, buckets: 2, drawable: 2 });
  });

  it("separates 'buckets exist but nothing is drawable' from 'no buckets'", () => {
    // The rig's vacuum/helium/noise series: 1682 buckets, every reading excluded,
    // every mean null — the chart rendered blank with no explanation.
    const dead = coverageOf([
      pt(0, { mean: null, n: 0, n_excluded: 26, worst_status: "unavailable" }),
      pt(60, { mean: null, n: 0, n_excluded: 26, worst_status: "unavailable" }),
    ]);
    assert.equal(dead.buckets, 2);
    assert.equal(dead.drawable, 0);
    assert.equal(dead.counted, 0);
    assert.deepEqual(coverageOf([]), { counted: 0, excluded: 0, buckets: 0, drawable: 0 });
  });

  it("counts a bucket as drawable only when its mean is a finite number", () => {
    assert.equal(coverageOf([pt(0, { mean: Number.NaN })]).drawable, 0);
    assert.equal(coverageOf([pt(0, { mean: 0 })]).drawable, 1);
  });
});

describe("spectrum helpers", () => {
  it("drops non-positive PSD samples instead of losing the whole curve", () => {
    const [f, p] = spectrumSeries([1, 2, 3, 4], [1e-24, 0, -1, 4e-24]);
    assert.deepEqual(f, [1, 4]);
    assert.deepEqual(p, [1e-24, 4e-24]);
  });

  it("drops points where either array is short or non-finite", () => {
    const [f] = spectrumSeries([1, 2, 3], [1e-24, NaN]);
    assert.deepEqual(f, [1]);
  });

  it("converts PSD to ASD by square root", () => {
    const [a, b] = toAsd([4e-24, 9e-24]);
    // Tolerance, not equality: sqrt(9e-24) is 2.9999999999999997e-12 in IEEE
    // doubles, and a test that demands the decimal answer would be testing
    // floating point rather than the conversion.
    assert.ok(Math.abs((a as number) - 2e-12) < 1e-24);
    assert.ok(Math.abs((b as number) - 3e-12) < 1e-24);
  });

  it("leaves non-positive PSD as NaN rather than inventing a floor", () => {
    const out = toAsd([0, -1, 4e-24]);
    assert.ok(Number.isNaN(out[0]));
    assert.ok(Number.isNaN(out[1]));
    assert.ok(Number.isFinite(out[2] as number));
  });

  it("renames the unit to match the conversion", () => {
    assert.equal(asdUnit("A^2/Hz"), "A/√Hz");
    assert.equal(asdUnit("m^2/Hz"), "m/√Hz");
    assert.equal(asdUnit(""), "");
  });

  it("labels both channels in Chinese", () => {
    assert.equal(channelLabel("current"), "电流噪声谱");
    assert.equal(channelLabel("z"), "Z 噪声谱");
  });
});

describe("formatting", () => {
  it("spells durations the way an operator reads them", () => {
    assert.equal(fmtSpan(60), "1 分");
    assert.equal(fmtSpan(3600), "1 小时");
    assert.equal(fmtSpan(3660), "1 小时 1 分");
    assert.equal(fmtSpan(90000), "1 天 1 小时");
    assert.equal(fmtSpan(0), "—");
  });

  it("formats byte counts", () => {
    assert.equal(fmtBytes(0), "0 B");
    assert.equal(fmtBytes(1536), "1.5 KB");
    assert.equal(fmtBytes(null), "—");
  });

  it("labels an axis with its unit when there is one", () => {
    assert.equal(axisLabel("temperature", "K"), "temperature (K)");
    assert.equal(axisLabel("instrument_quiet", ""), "instrument_quiet");
  });

  it("knows the synthetic duty-cycle series", () => {
    assert.equal(isDutyCycleSeries("instrument_quiet"), true);
    assert.equal(isDutyCycleSeries("temperature"), false);
  });
});

// The environment-history knobs go through exactly the same store semantics as
// the current monitor's, so the guard is re-verified against THIS catalogue —
// the failure it prevents (eh_raw_keep_days silently snapping back to 14 and
// deleting more history than the operator asked) is worse here than there.
const EH_KNOBS: KnobLike[] = [
  { key: "eh_enabled", min: 0, max: 1, value: 1, default: 1, is_bool: true },
  { key: "eh_z_enabled", min: 0, max: 1, value: 0, default: 0, is_bool: true },
  { key: "eh_bucket_s", min: 10, max: 3600, value: 60, default: 60 },
  { key: "eh_raw_keep_days", min: 1, max: 3650, value: 14, default: 14 },
];

describe("buildKnobPayload — whole-replace guard, env_history catalogue", () => {
  it("sends every key, not only the one that changed", () => {
    const out = buildKnobPayload(EH_KNOBS, {}, { key: "eh_bucket_s", value: 300 });
    assert.deepEqual(Object.keys(out).sort(), EH_KNOBS.map((k) => k.key).sort());
    assert.equal(out.eh_bucket_s, 300);
  });

  it("carries a persisted retention through an unrelated edit", () => {
    // The whole point: changing the bucket width must NOT reset the operator's
    // 90-day retention back to the 14-day default.
    const persisted = { eh_raw_keep_days: 90 };
    const out = buildKnobPayload(EH_KNOBS, persisted, { key: "eh_bucket_s", value: 300 });
    assert.equal(out.eh_raw_keep_days, 90);
  });

  it("clamps into the catalogue's bounds", () => {
    const out = buildKnobPayload(EH_KNOBS, {}, { key: "eh_raw_keep_days", value: 0 });
    assert.equal(out.eh_raw_keep_days, 1); // 0 天 would delete today's readings
  });

  it("coerces booleans to exactly 0 or 1", () => {
    assert.equal(clampKnob(EH_KNOBS[0] as KnobLike, 0.7), 1);
    assert.equal(clampKnob(EH_KNOBS[1] as KnobLike, 0.2), 0);
  });

  it("round-trips small values without rendering them as 0", () => {
    assert.equal(fmtKnobValue(2e-11), "2e-11");
  });
});

// ── spectrum working point (需求:噪声谱需要记录当时的状态) ──────────

describe("tunnellingLabel", () => {
  it("keeps 「没测到」 separate from 「没有隧穿」", () => {
    // The whole reason this returns four answers and not two. A spectrum with
    // no junction measures the AMPLIFIER, not the tip; one whose state was
    // unreadable measures we-do-not-know. Collapsing the second into the first
    // would let the page state a fact nobody observed.
    assert.equal(tunnellingLabel({ quietness: "no_tunnel" }).label, "未隧穿");
    assert.equal(tunnellingLabel({}).label, "状态未知");
    assert.notEqual(
      tunnellingLabel({ quietness: "no_tunnel" }).label,
      tunnellingLabel({}).label,
    );
  });

  it("falls back to the Z-controller flag when quietness is missing", () => {
    assert.equal(tunnellingLabel({ ctx_zctrl_on: true }).label, "Z 反馈开");
    assert.equal(tunnellingLabel({ ctx_zctrl_on: false }).label, "未隧穿");
  });

  it("only ever names a tone the Badge actually defines", () => {
    // BADGE_TONE keys are UPPERCASE and an unknown key silently falls back to
    // `default` — a lowercase "warn" would render as a neutral chip and quietly
    // stop warning. Same silence as /#61.
    const KNOWN = new Set(["AUTO", "INFO", "WARN", "DANGEROUS", "default"]);
    const cases = [
      { quietness: "no_tunnel" }, { quietness: "quiet" }, { quietness: "active" },
      { quietness: "unknown" }, { quietness: "" }, { quietness: "junk" },
      { ctx_zctrl_on: true }, { ctx_zctrl_on: false }, {},
    ];
    for (const c of cases) assert.ok(KNOWN.has(tunnellingLabel(c).tone), JSON.stringify(c));
  });
});

describe("ctxStabilityNote", () => {
  it("says nothing when the working point held", () => {
    assert.equal(ctxStabilityNote({ ctx_stable: true }), "");
  });

  it("warns when it moved mid-window", () => {
    assert.match(ctxStabilityNote({ ctx_stable: false }), /变过/);
  });

  it("treats a missing flag as unknown, not as stable", () => {
    // Old rows predate the tracking. Reassuring about them would be inventing
    // evidence — the same distinction as 「没测到」 vs 「测出来是零」.
    assert.notEqual(ctxStabilityNote({}), "");
    assert.notEqual(ctxStabilityNote({}), ctxStabilityNote({ ctx_stable: false }));
  });
});

describe("spectrumPickerLabel", () => {
  const ts = (t: number) => `T${t}`;

  it("carries the working point, not just a timestamp", () => {
    const s = spectrumPickerLabel(
      { ts: 5, quietness: "quiet", ctx_bias_v: -1.2 }, ts);
    assert.match(s, /T5/);
    assert.match(s, /隧穿中/);
    assert.match(s, /-1\.2/);
  });

  it("omits a bias it does not have rather than printing 0", () => {
    const s = spectrumPickerLabel({ ts: 5, quietness: "quiet" }, ts);
    assert.ok(!s.includes("0.00"));
    assert.ok(!s.includes("V"));
  });
});
