// ════════════════════════════════════════════════════════════════════════════
// Pure-logic tests for src/lib/monitoring.ts — the 电流监控 helpers.
//
// NO TEST FRAMEWORK IS INSTALLED in this frontend (no vitest, no jest), and
// adding one is a dependency decision that is not ours to make. These run on
// `node --test` with Node's native TypeScript stripping — zero new packages:
//
//     cd frontend && npm run test:unit
//
// What that buys: the rules that decide whether an operator sees a live number
// or a grey 「无数据」 are exercised as CODE, not only through a rendered page.
// The merge helper in particular has a failure mode no screenshot would catch —
// silently fabricating a status envelope from a partial push — so it gets the
// most cases here. Rendering itself is out of scope (no DOM).
//
// fmtCurrent is covered too even though it lives in lib/units.ts: this page is
// its heaviest consumer and a regression there would misprint every current on
// screen. It is imported, not re-implemented, on purpose.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import {
  AUX_CHANNELS,
  AUX_WINDOWS,
  MAX_AUX_WINDOW_S,
  MAX_TRACE_WINDOW_S,
  METRIC_TILES,
  TRACE_WINDOWS,
  auxAmpSamplingNote,
  auxLockinSplit,
  auxSeries,
  auxVerdictView,
  buildKnobPayload,
  clampAuxWindow,
  clampKnob,
  clampWindow,
  pollForWindow,
  fmtBytes,
  fmtKnobValue,
  fmtSeconds,
  isStale,
  metricSeries,
  patchStatusFromWsEvent,
  stateView,
  verdictView,
  type KnobLike,
  type MonitoringStatusLike,
} from "../src/lib/monitoring.ts";
import { fmtCurrent } from "../src/lib/units.ts";

// ── fixtures ────────────────────────────────────────────────────────────────

function baseStatus(): MonitoringStatusLike & { connected: boolean; segments_done: number } {
  return {
    state: "running",
    detail: "采集中",
    retry_in_s: 0,
    strategy: "osci1t",
    fs_hz: 1000,
    channel_name: "Current",
    last_segment_ts: 1_700_000_000,
    latest: {
      ts: 1_700_000_000,
      seg_id: 41,
      mean_a: 1e-10,
      rms_detrended_a: 5e-12,
      min_a: -2e-11,
      max_a: 3e-10,
      verdict: "ok",
      metrics: { mean_a: 1e-10, rms_detrended_a: 5e-12, kurtosis: 3.1, inv_f_slope: -1.02 },
    },
    connected: true,
    segments_done: 41,
  };
}

const SEGMENT_EVENT = {
  kind: "segment",
  seg_id: 42,
  ts: 1_700_000_001,
  fs_hz: 1000,
  rms_pa: 8.5,
  mean_na: 0.12,
  spike_sigma: 4.2,
  rtn_score: 0.31,
  sat_frac: 0.0,
  level: "warn",
  gap_s: 0,
  ctx_scanning: true,
  ctx_skill: "",
};

// ── currents ────────────────────────────────────────────────────────────────

describe("fmtCurrent (from lib/units.ts — not duplicated here)", () => {
  it("picks the engineering prefix instead of scientific notation", () => {
    assert.equal(fmtCurrent(9.63e-11), "96.3 pA");
    assert.equal(fmtCurrent(1e-9), "1.00 nA");
    assert.equal(fmtCurrent(2.5e-6), "2.50 µA");
  });

  it("does not park everything in pA", () => {
    assert.notEqual(fmtCurrent(1e-9), "1000.0 pA");
  });

  it("has a placeholder for absent readings rather than printing a zero", () => {
    assert.equal(fmtCurrent(null, { placeholder: "—" }), "—");
    assert.equal(fmtCurrent(undefined, { placeholder: "—" }), "—");
    assert.equal(fmtCurrent(Number.NaN, { placeholder: "—" }), "—");
  });
});

// ── vocabulary ──────────────────────────────────────────────────────────────

describe("verdictView", () => {
  it("maps every verdict the backend can emit", () => {
    assert.deepEqual(verdictView("ok"), { label: "正常", tone: "AUTO" });
    assert.deepEqual(verdictView("warn"), { label: "警告", tone: "WARN" });
    assert.deepEqual(verdictView("critical"), { label: "严重", tone: "DANGEROUS" });
    assert.deepEqual(verdictView("suppressed"), { label: "已抑制", tone: "INFO" });
    assert.deepEqual(verdictView("unknown"), { label: "未知", tone: "default" });
  });

  it("falls back to 未知 for anything unrecognised, including null", () => {
    assert.equal(verdictView("something_new").label, "未知");
    assert.equal(verdictView(null).label, "未知");
    assert.equal(verdictView(undefined).label, "未知");
    assert.equal(verdictView("").label, "未知");
  });

  it("never returns a tone the Badge component cannot render", () => {
    const allowed = new Set(["AUTO", "INFO", "WARN", "DANGEROUS", "default"]);
    for (const v of ["ok", "warn", "critical", "suppressed", "unknown", "junk"]) {
      assert.ok(allowed.has(verdictView(v).tone), `bad tone for ${v}`);
    }
  });
});

describe("stateView", () => {
  // Every state monitoring/service.py can publish. "unknown" is a real state
  // with its own entry, so it is checked separately from the fall-through.
  const STATES = [
    "disabled", "no_pool", "comms_down", "probing",
    "running", "unavailable", "paused",
  ];

  it("covers every state in the daemon's state machine", () => {
    for (const s of [...STATES, "unknown"]) {
      assert.ok(stateView(s).hint.length > 0, `${s} has no hint`);
    }
    for (const s of STATES) {
      assert.notEqual(stateView(s).label, "未知", `${s} fell through to the unknown branch`);
    }
  });

  it("falls back for an unrecognised state instead of throwing", () => {
    assert.equal(stateView("teleporting").label, "未知");
    assert.equal(stateView(null).label, "未知");
  });

  it("does not dress a dead link as healthy", () => {
    assert.equal(stateView("running").tone, "AUTO");
    assert.equal(stateView("comms_down").tone, "DANGEROUS");
    assert.notEqual(stateView("no_pool").tone, "AUTO");
  });
});

// ── the merge ───────────────────────────────────────────────────────────────

describe("patchStatusFromWsEvent — merge, never replace", () => {
  it("refuses to invent a status when the cache is empty", () => {
    assert.equal(patchStatusFromWsEvent(undefined, SEGMENT_EVENT), undefined);
    assert.equal(patchStatusFromWsEvent(null, SEGMENT_EVENT), undefined);
    assert.equal(
      patchStatusFromWsEvent(undefined, { kind: "status", state: "running" }),
      undefined,
    );
  });

  it("ignores payloads it does not recognise", () => {
    const prev = baseStatus();
    for (const junk of [
      null, undefined, 42, "segment", [],
      {}, { kind: "banana" }, { kind: 7 }, { state: "running" },
    ]) {
      assert.equal(patchStatusFromWsEvent(prev, junk), undefined, `accepted ${JSON.stringify(junk)}`);
    }
  });

  it("keeps every field the event does not carry", () => {
    const prev = baseStatus();
    const next = patchStatusFromWsEvent(prev, SEGMENT_EVENT);
    assert.ok(next);
    assert.equal(next.connected, true);
    assert.equal(next.state, "running");
    assert.equal(next.channel_name, "Current");
  });

  it("does not advance counters — a reconnect replays events and would double-count", () => {
    const prev = baseStatus();
    const once = patchStatusFromWsEvent(prev, SEGMENT_EVENT)!;
    const twice = patchStatusFromWsEvent(once, SEGMENT_EVENT)!;
    assert.equal(once.segments_done, 41);
    assert.equal(twice.segments_done, 41);
  });

  it("leaves the previous object untouched (react-query needs a new reference)", () => {
    const prev = baseStatus();
    const next = patchStatusFromWsEvent(prev, SEGMENT_EVENT)!;
    assert.notEqual(next, prev);
    assert.equal(prev.latest?.seg_id, 41, "prev was mutated in place");
    assert.equal(prev.last_segment_ts, 1_700_000_000);
  });

  describe("segment events", () => {
    it("converts the bus's display units into SI amps", () => {
      const next = patchStatusFromWsEvent(baseStatus(), SEGMENT_EVENT)!;
      // 8.5 pA and 0.12 nA — a pA left sitting in an `_a` field is a 1000× lie.
      assert.equal(next.latest?.rms_detrended_a, 8.5e-12);
      assert.ok(Math.abs(next.latest!.mean_a! - 1.2e-10) < 1e-22);
      assert.equal(next.latest?.metrics.rms_detrended_a, 8.5e-12);
    });

    it("carries the verdict, seg_id and timestamp through", () => {
      const next = patchStatusFromWsEvent(baseStatus(), SEGMENT_EVENT)!;
      assert.equal(next.latest?.seg_id, 42);
      assert.equal(next.latest?.verdict, "warn");
      assert.equal(next.latest?.ts, 1_700_000_001);
      assert.equal(next.last_segment_ts, 1_700_000_001);
    });

    it("does NOT carry the old segment's metrics onto the new seg_id", () => {
      const next = patchStatusFromWsEvent(baseStatus(), SEGMENT_EVENT)!;
      assert.equal(next.latest?.metrics.kurtosis, undefined);
      assert.equal(next.latest?.metrics.inv_f_slope, undefined);
      assert.equal(next.latest?.min_a, null);
      assert.equal(next.latest?.max_a, null);
    });

    it("keeps the renamed spike field under its REST name", () => {
      const next = patchStatusFromWsEvent(baseStatus(), SEGMENT_EVENT)!;
      assert.equal(next.latest?.metrics.spike_max_sigma, 4.2);
      assert.equal(next.latest?.metrics.rtn_score, 0.31);
      assert.equal(next.latest?.metrics.sat_frac, 0);
    });

    it("drops a segment with no usable timestamp", () => {
      assert.equal(
        patchStatusFromWsEvent(baseStatus(), { ...SEGMENT_EVENT, ts: null }),
        undefined,
      );
      assert.equal(
        patchStatusFromWsEvent(baseStatus(), { ...SEGMENT_EVENT, ts: "now" }),
        undefined,
      );
    });

    it("tolerates null metric fields without writing NaN into the map", () => {
      const next = patchStatusFromWsEvent(baseStatus(), {
        ...SEGMENT_EVENT, rms_pa: null, rtn_score: null,
      })!;
      assert.equal(next.latest?.rms_detrended_a, null);
      assert.equal("rms_detrended_a" in next.latest!.metrics, false);
      assert.equal("rtn_score" in next.latest!.metrics, false);
      for (const v of Object.values(next.latest!.metrics)) {
        assert.ok(Number.isFinite(v));
      }
    });

    it("falls back to unknown when the level is missing", () => {
      const next = patchStatusFromWsEvent(baseStatus(), { ...SEGMENT_EVENT, level: null })!;
      assert.equal(next.latest?.verdict, "unknown");
      assert.equal(verdictView(next.latest!.verdict).label, "未知");
    });
  });

  describe("status events", () => {
    const EVT = {
      kind: "status",
      state: "comms_down",
      detail: "TCP 熔断",
      retry_in_s: 12.5,
      strategy: null,
      fs_hz: 0,
      channel_name: "",
    };

    it("moves the six fields it carries", () => {
      const next = patchStatusFromWsEvent(baseStatus(), EVT)!;
      assert.equal(next.state, "comms_down");
      assert.equal(next.detail, "TCP 熔断");
      assert.equal(next.retry_in_s, 12.5);
      assert.equal(next.strategy, null);
      assert.equal(next.fs_hz, 0);
      assert.equal(next.channel_name, "");
    });

    it("does not touch the latest reading", () => {
      const next = patchStatusFromWsEvent(baseStatus(), EVT)!;
      assert.equal(next.latest?.seg_id, 41);
      assert.equal(next.last_segment_ts, 1_700_000_000);
    });

    it("keeps the previous value when a field is the wrong type", () => {
      const next = patchStatusFromWsEvent(baseStatus(), {
        kind: "status", state: 5, detail: null, retry_in_s: "soon",
      })!;
      assert.equal(next.state, "running");
      assert.equal(next.detail, "采集中");
      assert.equal(next.retry_in_s, 0);
    });
  });
});

// ── freshness ───────────────────────────────────────────────────────────────

describe("isStale", () => {
  const NOW = 1_700_000_000_000; // ms

  it("treats a missing timestamp as stale — no data is not fresh data", () => {
    assert.equal(isStale(null, NOW, 15_000), true);
    assert.equal(isStale(undefined, NOW, 15_000), true);
    assert.equal(isStale(Number.NaN, NOW, 15_000), true);
    assert.equal(isStale(0, NOW, 15_000), true);
  });

  it("compares a unix-SECONDS payload against a millisecond clock", () => {
    assert.equal(isStale(1_700_000_000, NOW, 15_000), false);
    assert.equal(isStale(1_699_999_990, NOW, 15_000), false); // 10 s old
    assert.equal(isStale(1_699_999_980, NOW, 15_000), true); // 20 s old
  });

  it("does not silently answer 'fresh forever' when handed milliseconds", () => {
    // dataUpdatedAt is in ms and lives on the same page — the guard exists so a
    // mixed-clock call degrades to a correct answer, not a permanent green.
    assert.equal(isStale(NOW, NOW, 15_000), false);
    assert.equal(isStale(NOW - 20_000, NOW, 15_000), true);
  });

  it("uses a strict threshold so the boundary is not flapping", () => {
    assert.equal(isStale(1_699_999_985, NOW, 15_000), false); // exactly 15 s
    assert.equal(isStale(1_699_999_984, NOW, 15_000), true);
  });
});

// ── window clamp ────────────────────────────────────────────────────────────

describe("clampWindow", () => {
  it("keeps every window the UI offers", () => {
    for (const w of TRACE_WINDOWS) assert.equal(clampWindow(w.s), w.s);
  });

  it("clamps to the range the endpoint accepts", () => {
    assert.equal(clampWindow(0), 1);
    assert.equal(clampWindow(-5), 1);
    assert.equal(clampWindow(9999999), MAX_TRACE_WINDOW_S);
    assert.equal(clampWindow(MAX_TRACE_WINDOW_S), MAX_TRACE_WINDOW_S);
  });

  it("falls back to the default rather than sending NaN", () => {
    assert.equal(clampWindow(Number.NaN), 60);
    assert.equal(clampWindow(Number.POSITIVE_INFINITY), 60);
  });
});

describe("clampAuxWindow", () => {
  it("keeps every window the UI offers", () => {
    for (const w of AUX_WINDOWS) assert.equal(clampAuxWindow(w.s), w.s);
  });

  it("clamps to the range /monitoring/aux/series accepts (10…86400 s)", () => {
    assert.equal(clampAuxWindow(1), 10);
    assert.equal(clampAuxWindow(999999), MAX_AUX_WINDOW_S);
  });

  it("falls back to the default rather than sending NaN", () => {
    assert.equal(clampAuxWindow(Number.NaN), 900);
  });
});

// The two ceilings are ONE number that happens to be written twice. When they
// drift the failure is silent in the worst way: the client asks for six hours,
// the server quietly serves ten minutes, and the chart just looks like the
// instrument was idle. Read the Python and compare, rather than trusting that
// whoever changes one will remember the other.
describe("the client's window ceilings match the server's", () => {
  const ROUTES = join(
    fileURLToPath(new URL(".", import.meta.url)),
    "..", "..", "MASTv2", "mast", "api", "routes", "monitoring.py",
  );

  it("live-trace: MAX_TRACE_WINDOW_S === _MAX_TRACE_WINDOW_S", () => {
    const py = readFileSync(ROUTES, "utf8");
    const m = /^_MAX_TRACE_WINDOW_S\s*:\s*float\s*=\s*([\d.]+)/m.exec(py);
    assert.ok(m, "routes/monitoring.py no longer declares _MAX_TRACE_WINDOW_S");
    assert.equal(MAX_TRACE_WINDOW_S, Number(m[1]));
  });

  it("live-trace: the route actually clamps with that constant", () => {
    // A constant nothing reads would make the check above pass while the
    // endpoint kept its old literal.
    const py = readFileSync(ROUTES, "utf8");
    assert.match(py, /min\(_MAX_TRACE_WINDOW_S,\s*float\(window_s\)\)/);
  });

  it("aux series: MAX_AUX_WINDOW_S is the ceiling the route clamps to", () => {
    const py = readFileSync(ROUTES, "utf8");
    const m = /w = max\(10\.0,\s*min\(([\d.]+),\s*float\(window_s\)\)\)/.exec(py);
    assert.ok(m, "the aux route no longer clamps window_s the way this test reads");
    assert.equal(MAX_AUX_WINDOW_S, Number(m[1]));
  });
});

describe("pollForWindow", () => {
  it("leaves short windows on the live cadence", () => {
    assert.equal(pollForWindow(30, 3000), 3000);
    assert.equal(pollForWindow(300, 3000), 3000);
  });

  it("slows down as the window grows", () => {
    assert.ok(pollForWindow(3600, 3000) > 3000);
    assert.ok(pollForWindow(21600, 3000) > pollForWindow(3600, 3000));
  });

  it("never polls FASTER than the caller asked", () => {
    // The caller's interval already encodes whether the WS is carrying events;
    // this function may only stretch it.
    for (const w of [30, 300, 3600, 86400]) {
      assert.ok(pollForWindow(w, 60_000) >= 60_000);
    }
  });

  it("survives a junk window rather than returning NaN", () => {
    assert.equal(pollForWindow(Number.NaN, 3000), 3000);
  });
});

// ── formatting ──────────────────────────────────────────────────────────────

describe("fmtBytes", () => {
  it("steps through binary units", () => {
    assert.equal(fmtBytes(0), "0 B");
    assert.equal(fmtBytes(512), "512 B");
    assert.equal(fmtBytes(1024), "1.00 KB");
    assert.equal(fmtBytes(1536), "1.50 KB");
    assert.equal(fmtBytes(20 * 1024 * 1024), "20.0 MB");
    assert.equal(fmtBytes(4 * 1024 ** 3), "4.00 GB");
  });

  it("stops at TB instead of inventing a prefix", () => {
    assert.equal(fmtBytes(5 * 1024 ** 4), "5.00 TB");
    assert.ok(fmtBytes(5000 * 1024 ** 4).endsWith("TB"));
  });

  it("renders an em dash for an absent size", () => {
    assert.equal(fmtBytes(null), "—");
    assert.equal(fmtBytes(undefined), "—");
    assert.equal(fmtBytes(Number.NaN), "—");
  });
});

describe("fmtSeconds", () => {
  it("spans sub-second gaps to multi-hour uptimes", () => {
    assert.equal(fmtSeconds(0.4), "400 ms");
    assert.equal(fmtSeconds(1.5), "1.5 s");
    assert.equal(fmtSeconds(42), "42 s");
    assert.equal(fmtSeconds(90), "1 分 30 秒");
    assert.equal(fmtSeconds(7200), "2 时 0 分");
  });

  it("renders an em dash for an absent duration", () => {
    assert.equal(fmtSeconds(null), "—");
    assert.equal(fmtSeconds(Number.NaN), "—");
  });
});

// ── feature tiles ───────────────────────────────────────────────────────────

describe("METRIC_TILES / metricSeries", () => {
  it("names only keys the extractor actually produces", () => {
    // Mirrors monitoring/features.py FEATURE_COLUMNS — a typo here shows up as a
    // permanently empty tile, which looks like missing data rather than a bug.
    const known = new Set([
      "rms_detrended_a", "mean_a", "inv_f_slope", "jump_rate_hz",
      "rtn_score", "kurtosis",
    ]);
    for (const t of METRIC_TILES) {
      assert.ok(known.has(t.key), `unknown metric key ${t.key}`);
      assert.ok(t.label && t.hint, `${t.key} missing label/hint`);
    }
  });

  it("pulls one metric's history in row order", () => {
    const rows = [
      { metrics: { rms_detrended_a: 1e-12 } },
      { metrics: { rms_detrended_a: 2e-12 } },
      { metrics: { rms_detrended_a: 3e-12 } },
    ];
    assert.deepEqual(metricSeries(rows, "rms_detrended_a"), [1e-12, 2e-12, 3e-12]);
  });

  it("skips rows lacking the column instead of plotting a hole as zero", () => {
    const rows = [
      { metrics: { kurtosis: 3 } },
      { metrics: {} },
      { metrics: null },
      {},
      { metrics: { kurtosis: 4 } },
    ];
    assert.deepEqual(metricSeries(rows, "kurtosis"), [3, 4]);
  });

  it("drops non-finite values so a sparkline cannot go blank on one NaN", () => {
    const rows = [
      { metrics: { rtn_score: 0.2 } },
      { metrics: { rtn_score: Number.NaN } },
      { metrics: { rtn_score: Number.POSITIVE_INFINITY } },
      { metrics: { rtn_score: 0.5 } },
    ];
    assert.deepEqual(metricSeries(rows, "rtn_score"), [0.2, 0.5]);
  });

  it("returns an empty series for an unknown key rather than throwing", () => {
    assert.deepEqual(metricSeries([{ metrics: { a: 1 } }], "nope"), []);
    assert.deepEqual(metricSeries([], "rms_detrended_a"), []);
  });
});

// ── settings knobs ──────────────────────────────────────────────────────────

// A trimmed copy of the live catalogue (mast/monitoring/thresholds.py
// knob_catalog), including the two booleans and the amp-valued thresholds that
// are the reason this page cannot reuse SettingsPage's _fmtThreshold.
const KNOBS: KnobLike[] = [
  { key: "cm_enabled", min: 0, max: 1, default: 1, value: 1, is_bool: true },
  { key: "cm_alerts_enabled", min: 0, max: 1, default: 1, value: 1, is_bool: true },
  { key: "cm_segment_s", min: 0.2, max: 10, default: 1, value: 1 },
  { key: "cm_target_fs_hz", min: 0, max: 1e6, default: 0, value: 0 },
  { key: "cm_rms_warn_a", min: 1e-13, max: 1e-6, default: 2e-11, value: 2e-11 },
  { key: "cm_keep_gb", min: 0.1, max: 500, default: 4, value: 4 },
];

describe("buildKnobPayload — whole-replace guard", () => {
  it("emits EVERY knob, not just the one that changed", () => {
    // The bug this prevents: the backend fills absent keys with defaults, so a
    // one-key POST silently resets the other eighteen.
    const out = buildKnobPayload(KNOBS, {}, { key: "cm_keep_gb", value: 8 });
    assert.deepEqual(Object.keys(out).sort(), KNOBS.map((k) => k.key).sort());
    assert.equal(out.cm_keep_gb, 8);
  });

  it("carries a calibrated threshold through untouched when another knob changes", () => {
    const persisted = { cm_rms_warn_a: 7.5e-12, cm_segment_s: 2 };
    const out = buildKnobPayload(KNOBS, persisted, { key: "cm_keep_gb", value: 8 });
    assert.equal(out.cm_rms_warn_a, 7.5e-12, "hand-calibrated noise threshold was lost");
    assert.equal(out.cm_segment_s, 2);
  });

  it("prefers the persisted value over the catalogue's effective value", () => {
    // A save can land in the settings cache before the config query refetches.
    const out = buildKnobPayload(KNOBS, { cm_keep_gb: 16 }, null);
    assert.equal(out.cm_keep_gb, 16);
  });

  it("falls back catalogue value → default → 0 when nothing is persisted", () => {
    const out = buildKnobPayload(
      [{ key: "a", value: 5, default: 9 }, { key: "b", default: 9 }, { key: "c" }],
      null,
    );
    assert.deepEqual(out, { a: 5, b: 9, c: 0 });
  });

  it("lets the explicit change outrank both", () => {
    const out = buildKnobPayload(KNOBS, { cm_segment_s: 2 }, { key: "cm_segment_s", value: 5 });
    assert.equal(out.cm_segment_s, 5);
  });

  it("ignores a persisted value that is not a finite number", () => {
    const persisted = { cm_keep_gb: "8" as unknown as number, cm_segment_s: Number.NaN };
    const out = buildKnobPayload(KNOBS, persisted, null);
    assert.equal(out.cm_keep_gb, 4); // catalogue value, not the string
    assert.equal(out.cm_segment_s, 1);
  });

  it("normalises booleans to exactly 0 or 1", () => {
    const on = buildKnobPayload(KNOBS, {}, { key: "cm_enabled", value: 1 });
    const off = buildKnobPayload(KNOBS, {}, { key: "cm_enabled", value: 0 });
    assert.equal(on.cm_enabled, 1);
    assert.equal(off.cm_enabled, 0);
    // A stray 0.4 from anywhere must not persist as a fractional "enabled".
    assert.equal(buildKnobPayload(KNOBS, { cm_enabled: 0.4 }, null).cm_enabled, 0);
    assert.equal(buildKnobPayload(KNOBS, { cm_enabled: 3 }, null).cm_enabled, 1);
  });

  it("holds values inside the catalogue's bounds", () => {
    assert.equal(buildKnobPayload(KNOBS, {}, { key: "cm_segment_s", value: 999 }).cm_segment_s, 10);
    assert.equal(buildKnobPayload(KNOBS, {}, { key: "cm_segment_s", value: 0 }).cm_segment_s, 0.2);
    assert.equal(buildKnobPayload(KNOBS, {}, { key: "cm_keep_gb", value: -3 }).cm_keep_gb, 0.1);
  });

  it("keeps a change whose key the catalogue does not know", () => {
    // Only happens with a stale catalogue; dropping the edit silently is worse.
    const out = buildKnobPayload(KNOBS, {}, { key: "cm_brand_new", value: 3 });
    assert.equal(out.cm_brand_new, 3);
  });

  it("survives a junk catalogue without throwing", () => {
    const out = buildKnobPayload(
      [{ key: "" }, { key: "ok", value: 1 }] as KnobLike[],
      null,
    );
    assert.deepEqual(out, { ok: 1 });
    assert.deepEqual(buildKnobPayload([], null), {});
  });

  it("produces only finite numbers — the backend takes dict[str, float]", () => {
    const out = buildKnobPayload(KNOBS, { cm_rms_warn_a: 7.5e-12 }, { key: "cm_keep_gb", value: 8 });
    for (const [k, v] of Object.entries(out)) {
      assert.equal(typeof v, "number", `${k} is not a number`);
      assert.ok(Number.isFinite(v), `${k} is not finite`);
    }
  });
});

describe("clampKnob", () => {
  const amp = KNOBS[4]!; // cm_rms_warn_a

  it("does not clamp a legitimate picoamp threshold to zero", () => {
    assert.equal(clampKnob(amp, 2e-11), 2e-11);
    assert.equal(clampKnob(amp, 1e-13), 1e-13);
  });

  it("pulls an out-of-range value to the nearest bound", () => {
    assert.equal(clampKnob(amp, 1), 1e-6);
    assert.equal(clampKnob(amp, 0), 1e-13);
  });

  it("recovers to a sane value when handed NaN", () => {
    assert.equal(clampKnob(amp, Number.NaN), 2e-11);
    assert.equal(clampKnob({ key: "x" }, Number.NaN), 0);
  });

  it("treats max=0 as 'no upper bound' rather than clamping everything to zero", () => {
    // knob_catalog emits (0, 0) for a key with no FIELD_BOUNDS entry.
    assert.equal(clampKnob({ key: "x", min: 0, max: 0 }, 42), 42);
  });
});

describe("fmtKnobValue", () => {
  it("round-trips the amp thresholds that _fmtThreshold destroys", () => {
    // Math.round(2e-11 * 1000) / 1000 === 0 — the row would then commit that 0.
    for (const v of [2e-11, 9e-8, 1e-13, 1e-6, 7.5e-12]) {
      assert.equal(Number.parseFloat(fmtKnobValue(v)), v, `lost ${v}`);
      assert.notEqual(fmtKnobValue(v), "0");
    }
  });

  it("keeps ordinary knobs readable", () => {
    assert.equal(fmtKnobValue(1), "1");
    assert.equal(fmtKnobValue(0.2), "0.2");
    assert.equal(fmtKnobValue(120), "120");
    assert.equal(fmtKnobValue(0), "0");
  });

  it("renders nothing for a non-finite value instead of 'NaN'", () => {
    assert.equal(fmtKnobValue(Number.NaN), "");
    assert.equal(fmtKnobValue(Number.POSITIVE_INFINITY), "");
  });
});

// ── auxiliary channels (Z 位置 / qPlus 振幅 / Δf) ───────────────────────────

describe("auxVerdictView", () => {
  it("never paints an unjudged or absent channel as 正常", () => {
    // Green here would claim a judgement that never happened. Both of these
    // mean "no verdict", and neither means "fine".
    assert.equal(auxVerdictView("unjudged").tone, "INFO");
    assert.equal(auxVerdictView("unavailable").tone, "default");
    assert.notEqual(auxVerdictView("unjudged").tone, "AUTO");
    assert.notEqual(auxVerdictView("unavailable").tone, "AUTO");
  });

  it("says 本机没有 rather than an error for a rig without qPlus", () => {
    // An STM with no qPlus sensor is a normal configuration.
    assert.equal(auxVerdictView("unavailable").label, "本机没有");
  });

  it("keeps ok / warn / suppressed aligned with the current channel", () => {
    for (const v of ["ok", "warn", "suppressed"] as const) {
      assert.equal(auxVerdictView(v).tone, verdictView(v).tone);
    }
  });

  it("falls back to 未知 for anything it does not recognise", () => {
    // NOT the `unknown` row: that one means "the signal table has not been read
    // yet", a specific claim. Version skew must not be dressed as that.
    assert.equal(auxVerdictView("nonsense").label, "未知");
    assert.equal(auxVerdictView(null).label, "未知");
    assert.equal(auxVerdictView(undefined).label, "未知");
    assert.notEqual(auxVerdictView("nonsense").label, auxVerdictView("unknown").label);
  });
});

describe("AUX_CHANNELS", () => {
  it("maps every channel to a distinct series column", () => {
    const cols = AUX_CHANNELS.map((c) => c.series);
    assert.equal(new Set(cols).size, cols.length);
  });

  it("puts Z first — it is the one with an actionable failure mode", () => {
    assert.equal(AUX_CHANNELS[0]!.kind, "z");
  });

  it("labels Δf as record-only in its hint", () => {
    const df = AUX_CHANNELS.find((c) => c.kind === "df")!;
    assert.match(df.hint, /不判级/);
  });
});

// 「辅助通道中 z 和振幅应该以 nm 为单位」。
//
// 这一组钉的是**纵轴前缀**,不是存储 —— 库里和判据里永远是 SI 裸值,这里只管
// 画出来的那个标签和那个乘数。从前这份映射按**单位**键控住在 AuxChannels.tsx
// 里(`Record<"m"|"A"|"Hz", …>`),于是 Z 和 qPlus 振幅因为同为米被迫共用一个前缀,
// 而它俩差四个数量级;搬到 AUX_CHANNELS 上之后它才第一次是可测的。
describe("AUX_CHANNELS 纵轴前缀 ", () => {
  const PREFIX_MUL: Record<string, number> = {
    T: 1e-12, G: 1e-9, M: 1e-6, k: 1e-3, "": 1,
    m: 1e3, µ: 1e6, u: 1e6, n: 1e9, p: 1e12, f: 1e15,
  };

  it("z 与振幅都用 nm", () => {
    for (const kind of ["z", "amplitude"]) {
      const c = AUX_CHANNELS.find((x) => x.kind === kind)!;
      assert.ok(c, `没有 ${kind} 通道`);
      assert.equal(c.plot.suffix, "nm", `${kind} 的纵轴不是 nm`);
    }
  });

  // 「顺带别误伤」写成断言。dI/dV 是安培、Δf 是赫兹,#37 一个字都没提它们,
  // 而把 z 和振幅改成 nm 的那次改动物理上碰得到这两条(它们本来同住一张查找表)。
  it("dI/dV 仍是 pA、Δf 仍是 Hz —— #37 没点它们", () => {
    assert.equal(AUX_CHANNELS.find((c) => c.kind === "lockin")!.plot.suffix, "pA");
    assert.equal(AUX_CHANNELS.find((c) => c.kind === "df")!.plot.suffix, "Hz");
  });

  // 每条通道都必须自带前缀。从前是 `PLOT_SCALE[unit] ?? {mul: 1, suffix: unit}` ——
  // 查不到就**静默**画裸 SI 值、标一个光秃秃的 "m",也就是说它失败成的样子
  // 恰好就是 #37 报的那个症状。现在漏填是类型错误,这条是给「填了但填空」兜底。
  it("每条通道都声明了纵轴前缀", () => {
    for (const c of AUX_CHANNELS) {
      assert.ok(c.plot, `${c.kind} 没有 plot`);
      assert.ok(Number.isFinite(c.plot.mul) && c.plot.mul > 0, `${c.kind} 的 mul 不是正数`);
      assert.ok(c.plot.suffix.length > 0, `${c.kind} 的 suffix 是空的`);
    }
  });

  // 最会咬人的一条:标签和乘数各写各的。`{mul: 1e12, suffix: "nm"}` 编译通过、
  // 长得完全正常,画出来的每一个数都差 1000 倍 —— 而纵轴上写着 "nm",没有任何
  // 东西会说出这件事。所以从后缀里把前缀抠出来,反推它应该是多少。
  it("后缀里的 SI 前缀与乘数必须自洽", () => {
    for (const c of AUX_CHANNELS) {
      const base = c.unit; // "m" | "Hz" | "A"
      const suffix = c.plot.suffix;
      assert.ok(suffix.endsWith(base), `${c.kind}: 后缀 "${suffix}" 的基本单位不是 "${base}"`);
      const prefix = suffix.slice(0, suffix.length - base.length);
      const expected = PREFIX_MUL[prefix];
      assert.ok(expected !== undefined, `${c.kind}: 认不出 SI 前缀 "${prefix}"`);
      assert.equal(
        c.plot.mul, expected,
        `${c.kind}: 纵轴写着 "${suffix}" 但乘的是 ${c.plot.mul}（应为 ${expected}）`,
      );
    }
  });
});

describe("auxSeries", () => {
  const body = {
    t_s: [1, 2, 3, 4],
    series: {
      z_m: [1e-9, 2e-9, null, 4e-9],
      amp_m: [null, null, null, null],
    },
  };

  it("keeps a missing reading in place as a hole", () => {
    // 这一条以前是反过来写的（「drop null readings」），而它就是操作员报的那个
    // bug：不画成 0 是对的（塌掉的 qPlus 振幅**真的**是 0，两者不能同高），
    // 但**丢掉那一行**让它的左右邻居变成相邻，uPlot 直接连了过去 —— 空洞被填平，
    // 「没测到」画成了一条平稳的曲线。null 留在原位，两件事才都成立。
    const { t, v, n } = auxSeries(body, "z_m");
    assert.deepEqual(t, [1, 2, 3, 4]);
    assert.deepEqual(v, [1e-9, 2e-9, null, 4e-9]);
    assert.equal(n, 3); // 真实读数三条，不是四条
  });

  it("splices a break where the timestamps themselves jump", () => {
    // 守护停过 / 服务重启：连行都没有，时间戳直接跳过去。载荷里没有任何东西
    // 标出这件事，所以只能按节奏自己判。
    const stopped = { t_s: [0, 1.3, 2.6, 600, 601.3], series: { z_m: [1, 2, 3, 4, 5] } };
    const { t, v, gaps } = auxSeries(stopped, "z_m");
    assert.equal(gaps, 1);
    assert.equal(v.filter((x) => x == null).length, 1);
    assert.ok(t.length === 6 && t.every((x, i) => i === 0 || x > t[i - 1]!));
  });

  it("counts real readings, not array slots", () => {
    // `t.length` 现在把补进去的断点也算进来，拿它问「这一路有没有数据」会让
    // 一条全是空洞的曲线显示成「有数据」。
    assert.equal(auxSeries(body, "amp_m").n, 0);
    assert.equal(auxSeries(body, "z_m").n, 3);
  });

  it("returns empty for a missing column, a missing body and junk", () => {
    const empty = { t: [], v: [], n: 0, gaps: 0 };
    assert.deepEqual(auxSeries(body, "no_such_column"), empty);
    assert.deepEqual(auxSeries(null, "z_m"), empty);
    assert.deepEqual(auxSeries(undefined, "z_m"), empty);
    assert.deepEqual(auxSeries({ t_s: null, series: null }, "z_m"), empty);
  });

  it("stops at the shorter of the two arrays", () => {
    const ragged = { t_s: [1, 2], series: { z_m: [1e-9, 2e-9, 3e-9] } };
    assert.equal(auxSeries(ragged, "z_m").t.length, 2);
  });

  it("turns NaN and Infinity into holes, which would otherwise blank a chart", () => {
    const bad = { t_s: [1, 2, 3], series: { z_m: [1e-9, NaN, Infinity] } };
    assert.deepEqual(auxSeries(bad, "z_m").v, [1e-9, null, null]);
    assert.equal(auxSeries(bad, "z_m").n, 1);
  });
});

describe("auxLockinSplit", () => {
  // 已知需求「辅助通道 didv 应该以某种方式标出 lockin 开没开」。调制关着的时候
  // 解调器输出的是噪声与串扰底 —— 同一个量纲、同一个数量级、同一种形状,
  // 只有「它是不是测量值」这一件事不一样,而那件事在数字里看不出来。
  const body = (mod: (number | null)[]) => ({
    t_s: [0, 1.3, 2.6, 3.9],
    series: { lockin_a: [1e-12, 2e-12, 3e-12, 4e-12], lockin_mod_on: mod },
  });

  it("draws modulating and non-modulating stretches as different curves", () => {
    const s = auxLockinSplit(body([1, 1, 0, 0]));
    // 过渡点(index 2)两条都放:否则两种线型之间会留一个采样点的洞,
    // 而那个洞和「采集中断」画出来一模一样。
    assert.deepEqual(s.on, [1e-12, 2e-12, 3e-12, null]);
    assert.deepEqual(s.off, [null, null, 3e-12, 4e-12]);
    assert.deepEqual([s.nOn, s.nOff, s.nUnknown], [2, 2, 0]);
  });

  it("counts 'never read' apart from 'off' while drawing them the same", () => {
    // 图上两者是同一句话（别当 dI/dV 读）,但下一步动作不同:关闭是有人拧过的
    // 旋钮,没读到是一条要去查的链路。所以线型合并、计数分开。
    const s = auxLockinSplit(body([1, null, 0, null]));
    assert.deepEqual([s.nOn, s.nOff, s.nUnknown], [1, 1, 2]);
    // 图上「关闭」和「没读到」是同一条虚线：index 1..3 共三点,加上过渡点
    // index 1 同时也留在实线上,所以实线两点、虚线三点。
    assert.deepEqual(s.on, [1e-12, 2e-12, null, null]);
    assert.deepEqual(s.off, [null, 2e-12, 3e-12, 4e-12]);
  });

  it("treats a missing flag column as 'not confirmed', never as 'on'", () => {
    // 老库、或者这台机器读不到 LockIn_ModOnOffGet。默认必须落在**不声称**那一侧:
    // 把没读到当成「在调制」,就是替仪器说了一句它没说过的话。
    //
    // 而且**曲线本身必须还在**:第一版里注释列缺失把 n 拉成了 0,于是一台读不到
    // 调制状态的机器连 dI/dV 曲线都看不见了 —— 那条曲线明明就在载荷里。
    const s = auxLockinSplit({
      t_s: [0, 1.3],
      series: { lockin_a: [1e-12, 2e-12] },
    });
    assert.deepEqual(s.on, [null, null]);
    assert.deepEqual(s.off, [1e-12, 2e-12]);
    assert.deepEqual([s.nOn, s.nOff, s.nUnknown], [0, 0, 2]);
    assert.equal(s.n, 2);
  });

  it("still draws the curve when the flag column is only partly filled", () => {
    // schema v6 之前的老行没有这一列。补 null,不截断 —— 「后面几拍没有这个注释」
    // 不是「后面几拍没有数据」。
    const s = auxLockinSplit({
      t_s: [0, 1.3, 2.6, 3.9],
      series: { lockin_a: [1e-12, 2e-12, 3e-12, 4e-12], lockin_mod_on: [1, 1] },
    });
    assert.equal(s.n, 4);
    assert.deepEqual([s.nOn, s.nOff, s.nUnknown], [2, 0, 2]);
  });

  it("keeps the flag aligned to the value across a spliced gap", () => {
    // 两列必须**一起**补断点。分别调用 spliceGaps 会得到两个长度不同的数组,
    // 于是每个读数配到的是另一个时刻的调制状态 —— 一种不会报错的错位。
    const s = auxLockinSplit({
      t_s: [0, 1.3, 600, 601.3],
      series: { lockin_a: [1e-12, 2e-12, 3e-12, 4e-12], lockin_mod_on: [1, 1, 0, 0] },
    });
    assert.equal(s.gaps, 1);
    assert.equal(s.t.length, 5);
    assert.equal(s.on.length, 5);
    assert.equal(s.off.length, 5);
    assert.equal(s.n, 4);
  });

  it("survives an empty payload", () => {
    const s = auxLockinSplit(undefined);
    assert.deepEqual([s.t, s.on, s.off], [[], [], []]);
    assert.deepEqual([s.n, s.nOn, s.nOff, s.nUnknown], [0, 0, 0, 0]);
  });
});

describe("auxAmpSamplingNote", () => {
  it("reports oversampling when the sample interval is inside τ", () => {
    // τ = Q/(π f₀) is how fast a high-Q qPlus amplitude can physically move.
    // 106 ms with 1 s sampling is UNDERsampled, 106 ms with 50 ms is not.
    const note = auxAmpSamplingNote(0.106, 0.05)!;
    assert.match(note, /过采样/);
    assert.match(note, /106 ms/);
  });

  it("says plainly when sampling is slower than τ", () => {
    const note = auxAmpSamplingNote(0.106, 1.0)!;
    assert.match(note, /比 τ 还慢/);
  });

  it("returns null when the resonance was never swept — unknown, not fine", () => {
    assert.equal(auxAmpSamplingNote(null, 1.0), null);
    assert.equal(auxAmpSamplingNote(undefined, 1.0), null);
    assert.equal(auxAmpSamplingNote(0, 1.0), null);
    assert.equal(auxAmpSamplingNote(0.1, null), null);
    assert.equal(auxAmpSamplingNote(NaN, 1.0), null);
  });

  it("prints τ in seconds once it is longer than one", () => {
    assert.match(auxAmpSamplingNote(2.5, 1.0)!, /2\.50 s/);
  });
});

describe("patchStatusFromWsEvent — aux events", () => {
  it("leaves the cache to the poll rather than merging an aux push", () => {
    // Deliberate: nothing on these channels moves fast enough to need a push
    // path, and a partial merge is how a stale number ends up looking live.
    const prev = baseStatus();
    assert.equal(
      patchStatusFromWsEvent(prev, { kind: "aux", ts: 1, z_m: 1e-9 }),
      undefined,
    );
  });
});

describe("auxVerdictView — unknown vs unavailable", () => {
  it("keeps 尚未探测 distinct from 本机没有", () => {
    // "we have not looked" and "this rig does not have one" are different
    // claims, and the second is the one that makes someone stop investigating.
    assert.notEqual(auxVerdictView("unknown").label, auxVerdictView("unavailable").label);
    assert.equal(auxVerdictView("unknown").label, "尚未探测");
    assert.equal(auxVerdictView("unavailable").label, "本机没有");
  });
});
