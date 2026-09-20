// ════════════════════════════════════════════════════════════════════════════
// monitoring.ts — pure helpers behind 电流监控 (pages/MonitoringPage.tsx).
//
// NO RUNTIME DEPENDENCIES beyond sibling `lib/*.ts` modules, on purpose — same
// rule lib/ws.ts follows. These run under `node --test` with Node's native
// TypeScript stripping and no DOM (frontend/test/monitoring.test.ts), so the
// WS→cache merge rules and the verdict/state vocabulary are covered as CODE
// instead of only through a rendered page. Anything that needs React, uPlot or
// the api client stays in the components.
//
// NOT HERE, deliberately: current formatting. `fmtCurrent` already exists in
// lib/units.ts with a documented engineering-notation rule (96.3 pA, not
// 9.63e-11 A and not 1000.0 pA) that came out of . A second
// implementation would drift from it, and a current printed two different ways
// on the same page is exactly the class of bug that unit confusion causes here.
// Components import it from "@/lib/units".
// ════════════════════════════════════════════════════════════════════════════
import { medianStep, spliceGaps } from "./seriesGaps.ts";

/** Badge tones as `components/ui.tsx` defines them. */
export type Tone = "AUTO" | "INFO" | "WARN" | "DANGEROUS" | "default";

// ── verdict / state vocabulary ──────────────────────────────────────────────

export interface VerdictView {
  label: string;
  tone: Tone;
}

const VERDICTS: Record<string, VerdictView> = {
  ok: { label: "正常", tone: "AUTO" },
  warn: { label: "警告", tone: "WARN" },
  critical: { label: "严重", tone: "DANGEROUS" },
  // The daemon judged the segment but withheld the alert because a tip-shaping
  // skill held the instrument — the noise was ours. Showing it as 正常 would
  // hide that a judgement happened; showing it as 警告 would cry wolf.
  suppressed: { label: "已抑制", tone: "INFO" },
  unknown: { label: "未知", tone: "default" },
};

export function verdictView(v: string | null | undefined): VerdictView {
  return VERDICTS[String(v ?? "")] ?? VERDICTS.unknown!;
}

export interface StateView {
  label: string;
  tone: Tone;
  hint: string;
}

// The daemon's own state machine (monitoring/service.py). Every entry gets a
// hint that says what the operator can DO about it — a bare "no_pool" chip
// tells them nothing, and "unavailable" in particular is not a fault to chase.
const STATES: Record<string, StateView> = {
  disabled: {
    label: "已关闭",
    tone: "default",
    hint: "监控在设置里被关闭。到「设置 → 电流监控」打开后即可启动。",
  },
  no_pool: {
    label: "无连接",
    tone: "WARN",
    hint: "内核未连接 Nanonis，采集无法开始。历史数据照常可浏览。",
  },
  comms_down: {
    label: "通信中断",
    tone: "DANGEROUS",
    hint: "Nanonis TCP 链路已熔断，采集暂停中；链路恢复后自动重连。",
  },
  probing: {
    label: "探测中",
    tone: "INFO",
    hint: "正在协商采集通道与采样率，稍候即进入运行。",
  },
  running: {
    label: "运行中",
    tone: "AUTO",
    hint: "正在采集，每段落库一次。",
  },
  unavailable: {
    label: "不可用",
    tone: "WARN",
    hint: "本机 Nanonis 没有可用的示波器接口，无法原生采集——这不是故障，是该机型的能力限制。",
  },
  paused: {
    label: "已暂停",
    tone: "WARN",
    hint: "采集被暂停（通常是其他任务占用了仪器），让出后自动恢复。",
  },
  unknown: {
    label: "未知",
    tone: "default",
    hint: "拿不到守护进程状态。",
  },
};

export function stateView(state: string | null | undefined): StateView {
  return STATES[String(state ?? "")] ?? STATES.unknown!;
}

// ── WS → react-query cache merge ────────────────────────────────────────────

/** The slice of `MonitoringStatus` a pushed event is allowed to touch. */
export interface MonitoringLatestLike {
  ts: number;
  seg_id: number;
  mean_a?: number | null;
  rms_detrended_a?: number | null;
  min_a?: number | null;
  max_a?: number | null;
  verdict: string;
  // Optional to match the generated schema: pydantic fields with a default come
  // out of openapi-typescript as optional, and a structural constraint that is
  // stricter than the real type would reject the very object it must accept.
  metrics?: { [k: string]: number };
}

export interface MonitoringStatusLike {
  state: string;
  detail: string;
  retry_in_s: number;
  strategy?: string | null;
  fs_hz: number;
  channel_name: string;
  last_segment_ts?: number | null;
  latest?: MonitoringLatestLike | null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Fold one `current_monitor` WS event into a cached `/api/monitoring/status`.
 *
 * MERGE, NEVER REPLACE — and refuse to create. Returning `undefined` tells
 * react-query to leave the cache untouched, which is the right answer twice:
 *
 *   · `prev` is empty. The event knows six fields; the envelope also carries
 *     `connected`, `degraded`, retention totals and the storage counters. To
 *     hand back a whole `MonitoringStatus` we would have to invent those, and
 *     `connected: false` invented next to a real segment reading would put a
 *     fabricated connection state on screen. The poll is ≤3 s away and fills it
 *     in properly. (Same reasoning as TopBar.tsx:163-188.)
 *
 *   · the payload is not one we know. An unrecognised push degrades to
 *     poll-only behaviour rather than blanking a live readout.
 *
 * Counters (`segments_done`, `segments_total`, `store_bytes`) are NOT advanced
 * here even though a `segment` event proves one more segment exists: the socket
 * replays up to 100 events on reconnect (`?since=`), so incrementing would
 * double-count a catch-up. Counters stay REST-owned; the push only moves what
 * it literally states.
 */
export function patchStatusFromWsEvent<T extends MonitoringStatusLike>(
  prev: T | null | undefined,
  evt: unknown,
): T | undefined {
  if (!prev) return undefined;
  if (typeof evt !== "object" || evt === null) return undefined;
  const d = evt as Record<string, unknown>;

  if (d.kind === "status") {
    return {
      ...prev,
      state: typeof d.state === "string" ? d.state : prev.state,
      detail: typeof d.detail === "string" ? d.detail : prev.detail,
      retry_in_s: num(d.retry_in_s) ?? prev.retry_in_s,
      strategy: typeof d.strategy === "string" ? d.strategy : null,
      fs_hz: num(d.fs_hz) ?? prev.fs_hz,
      channel_name:
        typeof d.channel_name === "string" ? d.channel_name : prev.channel_name,
    } as T;
  }

  if (d.kind === "segment") {
    const ts = num(d.ts);
    if (ts == null) return undefined; // a segment with no time is not a segment

    // The bus carries DISPLAY units (pA / nA) because it feeds a readout; the
    // REST contract is SI amps throughout. Convert on the way in so nothing
    // downstream has to know which source a number came from — a pA value
    // sitting in an `_a` field is how a 1000× error reaches an operator.
    const rmsA = num(d.rms_pa) == null ? null : num(d.rms_pa)! * 1e-12;
    const meanA = num(d.mean_na) == null ? null : num(d.mean_na)! * 1e-9;

    const metrics: { [k: string]: number } = {};
    if (rmsA != null) metrics.rms_detrended_a = rmsA;
    if (meanA != null) metrics.mean_a = meanA;
    for (const k of ["spike_max_sigma", "rtn_score", "sat_frac"] as const) {
      const src = k === "spike_max_sigma" ? "spike_sigma" : k;
      const v = num(d[src]);
      if (v != null) metrics[k] = v;
    }

    // A fresh `latest`, not a merge onto the old one: `metrics` from REST holds
    // ~35 columns and this event holds five. Carrying the other thirty over
    // would attribute the PREVIOUS segment's numbers to this seg_id, which is
    // worse than showing fewer of them. min/max stay null — unknown, not zero.
    return {
      ...prev,
      last_segment_ts: ts,
      latest: {
        ts,
        seg_id: num(d.seg_id) ?? 0,
        mean_a: meanA,
        rms_detrended_a: rmsA,
        min_a: null,
        max_a: null,
        verdict: typeof d.level === "string" ? d.level : "unknown",
        metrics,
      },
    } as T;
  }

  return undefined;
}

// ── freshness ───────────────────────────────────────────────────────────────

/**
 * Has the newest data point aged past `thresholdMs`?
 *
 * `lastTs` is a UNIX timestamp in **seconds** — what every backend field here
 * uses (`last_segment_ts`, `FeatureRow.ts`, `LiveTraceResponse.t_s`). `now` is
 * `Date.now()`, i.e. milliseconds. Missing/NaN counts as stale: no data is not
 * fresh data, and the banner must go grey rather than keep a frozen number
 * looking live.
 *
 * The magnitude guard exists because mixing the two clocks is a live hazard on
 * this page (`dataUpdatedAt` from react-query is in ms, the payload is in s).
 * A seconds timestamp does not reach 1e12 until the year 33658, so anything
 * that large is milliseconds and gets treated as such — a silent "never stale"
 * is the one failure mode this function must not have.
 */
export function isStale(
  lastTs: number | null | undefined,
  now: number,
  thresholdMs: number,
): boolean {
  if (lastTs == null || !Number.isFinite(lastTs) || lastTs <= 0) return true;
  const ms = lastTs > 1e12 ? lastTs : lastTs * 1000;
  return now - ms > thresholdMs;
}

// ── trace window ────────────────────────────────────────────────────────────

/** Longest window `/api/monitoring/live-trace` will serve (routes/monitoring.py
 *  `_MAX_TRACE_WINDOW_S`). Envelope rows are permanent, so the limit is about
 *  what is worth drawing on a *live* chart, not about what data survives. */
export const MAX_TRACE_WINDOW_S = 21600;

/** Window lengths offered by the live chart, in seconds. The long end answers
 *  Live current needed a longer visible window: a tip degrading over an afternoon is
 *  invisible in five minutes, and the envelope band keeps every spike visible
 *  no matter how far out the window goes. */
export const TRACE_WINDOWS: ReadonlyArray<{ s: number; label: string }> = [
  { s: 30, label: "30 秒" },
  { s: 60, label: "1 分钟" },
  { s: 300, label: "5 分钟" },
  { s: 1800, label: "30 分钟" },
  { s: 7200, label: "2 小时" },
  { s: 21600, label: "6 小时" },
];

/** Clamp to the range `/api/monitoring/live-trace` accepts, so the client never
 *  asks for a window the server will silently shorten. */
export function clampWindow(s: number): number {
  if (!Number.isFinite(s)) return 60;
  return Math.max(1, Math.min(MAX_TRACE_WINDOW_S, s));
}

/** Longest window `/api/monitoring/aux/series` will serve
 *  (routes/monitoring.py clamps `window_s` to 10…86400). */
export const MAX_AUX_WINDOW_S = 86400;

/**
 * Window lengths offered by the auxiliary channels.
 *
 * A different scale from {@link TRACE_WINDOWS} on purpose, and the short end
 * starts where the current chart's long end is: these sample at ~1 Hz, so
 * 30 seconds would be thirty points, and the phenomena they exist to show —
 * Z drift, amplitude wander — are minutes-to-hours things.
 *
 * The long end is real data, not an empty axis: aux samples are persisted to
 * the `aux_samples` table and swept at `cm_aux_keep_hours` (default 168 h),
 * so a 24 h window has a week of history behind it. The in-memory ring is only
 * `cm_aux_window_s` (default 300 s) — that ring feeds the verdict tiles, not
 * these charts.
 */
export const AUX_WINDOWS: ReadonlyArray<{ s: number; label: string }> = [
  { s: 300, label: "5 分钟" },
  { s: 900, label: "15 分钟" },
  { s: 3600, label: "1 小时" },
  { s: 21600, label: "6 小时" },
  { s: 86400, label: "24 小时" },
];

export function clampAuxWindow(s: number): number {
  if (!Number.isFinite(s)) return 900;
  return Math.max(10, Math.min(MAX_AUX_WINDOW_S, s));
}

/**
 * Poll interval for a windowed series, in ms.
 *
 * A six-hour window costs the server thousands of envelope rows per request and
 * moves by 0.03% between three-second polls. Refetching it on the live cadence
 * would be pure load for no visible change, so the interval scales with the
 * window — while short windows keep exactly the cadence they had.
 */
export function pollForWindow(windowS: number, liveMs: number): number {
  if (!Number.isFinite(windowS) || windowS <= 300) return liveMs;
  if (windowS <= 3600) return Math.max(liveMs, 15_000);
  return Math.max(liveMs, 60_000);
}

// ── formatting ──────────────────────────────────────────────────────────────

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

/** Store footprint. Binary steps — this counts `.npy` on disk, not marketing GB. */
export function fmtBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const neg = n < 0;
  let v = Math.abs(n);
  let i = 0;
  while (v >= 1024 && i < BYTE_UNITS.length - 1) {
    v /= 1024;
    i += 1;
  }
  const digits = i === 0 ? 0 : v < 10 ? 2 : 1;
  return `${neg ? "-" : ""}${v.toFixed(digits)} ${BYTE_UNITS[i]}`;
}

/** Durations that span six orders of magnitude (a 0.4 s gap, a 3 h uptime). */
export function fmtSeconds(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  const neg = s < 0;
  const v = Math.abs(s);
  let out: string;
  if (v < 1) out = `${(v * 1000).toFixed(0)} ms`;
  else if (v < 60) out = `${v.toFixed(v < 10 ? 1 : 0)} s`;
  else if (v < 3600) out = `${Math.floor(v / 60)} 分 ${Math.round(v % 60)} 秒`;
  else out = `${Math.floor(v / 3600)} 时 ${Math.round((v % 3600) / 60)} 分`;
  return neg ? `-${out}` : out;
}

// ── feature tiles ───────────────────────────────────────────────────────────

/** How a metric should be rendered. The component owns the actual formatters
 *  (currents go through lib/units.ts) so this table stays import-free. */
export type MetricUnit = "A" | "hz" | "ratio" | "plain";

export interface MetricTile {
  /** Key inside `FeatureRow.metrics` / `LatestFeature.metrics`. */
  key: string;
  label: string;
  hint: string;
  unit: MetricUnit;
}

/** The six the operator watches. The extractor produces ~35 columns
 *  (monitoring/features.py FEATURE_COLUMNS); the rest are for the corpus. */
export const METRIC_TILES: readonly MetricTile[] = [
  { key: "rms_detrended_a", label: "RMS 噪声", hint: "去趋势后的电流起伏，针尖状态最直接的读数", unit: "A" },
  { key: "mean_a", label: "平均电流", hint: "段内均值——偏离设定点说明回路在追", unit: "A" },
  { key: "inv_f_slope", label: "1/f 斜率", hint: "低频噪声谱的斜率，越接近 -1 越像典型 1/f", unit: "plain" },
  { key: "jump_rate_hz", label: "跳变率", hint: "每秒台阶式跳变次数，针尖不稳的特征", unit: "hz" },
  { key: "rtn_score", label: "RTN 判分", hint: "双能级随机电报噪声的强度，高即针尖有活动位点", unit: "ratio" },
  { key: "kurtosis", label: "峰度", hint: "远离 3 表示分布有重尾——尖峰或跳变", unit: "plain" },
];

// ── auxiliary channels (Z 位置 / qPlus 振幅 / Δf) ───────────────────────────
//
// These are NOT drawn on the current chart, and not on each other. Metres,
// metres and hertz are incommensurable, so an overlay needs two or three Y axes
// — and the relative vertical position of two curves on separate axes is
// whatever the person who drew it chose, which invites the reader to see a
// correlation nobody measured. Small multiples on a shared time axis say the
// same thing without the invitation.

/**
 * Y-axis scaling for one aux chart: multiply by `mul`, label with `suffix`.
 *
 * A FIXED prefix per channel, not a per-tick SI prefix picked from the data —
 * that would relabel the axis every time the window moved, so the same curve
 * would read "40" and then "0.04" with nothing having changed on the instrument.
 *
 * It hangs off the CHANNEL rather than off the unit, and that is the whole
 * point of it living here. It used to be a `Record<unit, …>` lookup inside
 * AuxChannels.tsx, which forced Z and qPlus amplitude to share one prefix
 * because both are measured in metres — and they are four orders of magnitude
 * apart. One `m` entry cannot serve a ±100 nm piezo range and a 10 pm
 * oscillation amplitude; picking pm for both is what made the Z axis read
 * ±100000 . Being a field of the object the renderer is already
 * iterating also removes the lookup entirely — the old `PLOT_SCALE[unit] ??
 * {mul: 1, suffix: unit}` fallback meant a key that did not match silently
 * rendered raw metres with a bare "m" label, i.e. it failed into exactly the
 * symptom being reported.
 */
export interface AuxPlotScale {
  mul: number;
  suffix: string;
}

export interface AuxChannelView {
  /** `AuxChannelState.kind` from the backend. */
  kind: string;
  /** Column name inside `AuxSeriesResponse.series`. */
  series: string;
  unit: "m" | "Hz" | "A";
  /** Fixed y-axis prefix for this channel's chart. SI storage is untouched. */
  plot: AuxPlotScale;
  /** What the number is FOR — the tile subtitle. */
  hint: string;
}

/** Draw order, top to bottom. Z first: it is the one most often checked
 *  first and the one with an actionable failure mode (drift ruins a scan). */
export const AUX_CHANNELS: readonly AuxChannelView[] = [
  // nm，要求要的。压电 Z 的行程是 ±100 nm 量级，画成 pm 就是一排
  // 六位数,而这条曲线要读的是漂移斜率和台阶,不是绝对位置的皮米位。
  { kind: "z", series: "z_m", unit: "m", plot: { mul: 1e9, suffix: "nm" },
    hint: "压电 Z 位置——看漂移与台阶" },
  // 只在进针期间判「归零」。STM 模式下这一路平时是噪声（隧穿不阻尼它），
  // 扎针与电脉冲反而会让它大起振。
  //
  // 用 nm 显示振幅，避免根据未驱动时的噪声底选择单位。
  // 实际幅度范围和激励状态应由使用者确认。
  { kind: "amplitude", series: "amp_m", unit: "m", plot: { mul: 1e9, suffix: "nm" },
    hint: "qPlus 振幅——进针时归零即撞针" },
  { kind: "df", series: "df_hz", unit: "Hz", plot: { mul: 1, suffix: "Hz" },
    hint: "频率偏移——只记录，不判级" },
  // 显示带符号的解调分量，穿零不等于信号缺失；保持与实时电流图相同的单位。
  { kind: "lockin", series: "lockin_a", unit: "A", plot: { mul: 1e12, suffix: "pA" },
    hint: "dI/dV lock-in——带符号，会穿零；只记录" },
];

/**
 * Verdict vocabulary for an aux channel.
 *
 * Two of these do not exist on the current path, and the difference is the
 * whole point: neither one means "fine".
 *
 *   · `unavailable` — the rig does not expose this signal. An STM with no qPlus
 *     sensor is a normal configuration, not a fault, so it is toned neutral.
 *   · `unjudged` — recorded but deliberately not judged (Δf carries no rules by
 *     design; Z and amplitude have no calibrated thresholds on this instrument
 *     yet; or the amplitude baseline is missing). Toned INFO, never AUTO —
 *     showing it green would claim a judgement that never happened.
 *   · `unknown` — the signal table has not been read yet. Distinct from
 *     `unavailable` on purpose: "we have not looked" is not "this rig does not
 *     have one", and the second reading is the one that makes someone stop
 *     investigating.
 */
const AUX_VERDICTS: Record<string, VerdictView> = {
  ok: { label: "正常", tone: "AUTO" },
  warn: { label: "警告", tone: "WARN" },
  suppressed: { label: "已抑制", tone: "INFO" },
  unjudged: { label: "未判级", tone: "INFO" },
  unavailable: { label: "本机没有", tone: "default" },
  unknown: { label: "尚未探测", tone: "default" },
};

/** For a verdict string this frontend does not know (backend/UI version skew).
 *  Deliberately NOT the `unknown` row: that one means a specific thing — "the
 *  signal table has not been read yet" — and putting it on every unrecognised
 *  value would state a fact nobody established. */
const AUX_FALLBACK: VerdictView = { label: "未知", tone: "default" };

export function auxVerdictView(v: string | null | undefined): VerdictView {
  return AUX_VERDICTS[String(v ?? "")] ?? AUX_FALLBACK;
}

/**
 * Pull one channel's series out of an `/aux/series` payload, with every
 * interruption preserved as a hole.
 *
 * Two different holes, and both used to be paved over — the aux channel did not
 * mark a data gap as missing, it drew a straight line across it:
 *
 *   · **This channel had no reading while others did.** The row exists, its
 *     column is `null`. That null is now KEPT at its own timestamp instead of
 *     being dropped, because dropping it closed the hole — the two neighbours
 *     became adjacent and uPlot joined them. Still never plotted as zero: a
 *     collapsed qPlus amplitude genuinely IS ~0, so "not measured" drawn at the
 *     same height would paint a tip crash that never happened.
 *   · **No rows at all.** Daemon stopped, service restarted, or the window
 *     reaches back past when recording began. Nothing marks this in the payload
 *     — the timestamps simply jump — so a null row gets spliced in
 *     (`lib/seriesGaps.ts`).
 *
 * The step is self-calibrated from the data. It cannot come from
 * `aux.interval_s`: that is the configured 1 s, the real cadence is ~1.33 s
 * (sampling rides the segment boundary), and `/aux/series` thins server-side to
 * `max_points`, so on a 24 h window the drawn spacing is minutes.
 *
 * `n` is the count of REAL readings — the spliced nulls must not be reported as
 * points the instrument produced, and "did this channel measure anything at
 * all" is no longer answerable from `t.length`.
 */
export function auxSeries(
  body: AuxSeriesBody,
  column: string,
): { t: number[]; v: (number | null)[]; n: number; gaps: number } {
  const { t, v, gaps } = auxColumns(body, column);
  return { t, v, n: countReadings(v), gaps };
}

type AuxSeriesBody =
  | { t_s?: number[] | null; series?: { [k: string]: (number | null)[] } | null }
  | null
  | undefined;

function countReadings(v: readonly (number | null)[]): number {
  return v.reduce<number>((acc, y) => acc + (y == null ? 0 : 1), 0);
}

/**
 * One value column plus any number of ANNOTATION columns, gap-spliced together.
 *
 * Together, not one call each: `spliceGaps` inserts break rows, so two
 * independent calls would return arrays of different lengths that no longer line
 * up index-for-index — and a caller pairing a reading with its flag would be
 * pairing it with a different instant. No error, just a quiet misalignment.
 *
 * The value column and the annotations are NOT symmetric, and the asymmetry is
 * the point. A missing value column means there is nothing to draw. A missing
 * annotation means only that this annotation is unknown — letting it shorten the
 * result would blank the curve itself, so a rig whose lock-in state cannot be
 * read would lose the dI/dV trace entirely, when the trace is right there.
 */
function auxColumns(
  body: AuxSeriesBody,
  column: string,
  extras: readonly string[] = [],
): { t: number[]; v: (number | null)[]; extras: (number | null)[][]; gaps: number } {
  const t_s = body?.t_s ?? [];
  const col = body?.series?.[column] ?? [];
  const n = Math.min(t_s.length, col.length);
  const ts = t_s.slice(0, n);
  const ex = extras.map((c) => {
    const raw = body?.series?.[c];
    if (!Array.isArray(raw)) return new Array<number | null>(n).fill(null);
    // 短了就补 null（不是截断整体）：注释列比值列短，说的是「后面那几拍没有这个
    // 注释」，不是「后面那几拍没有数据」。
    return raw.length >= n
      ? raw.slice(0, n)
      : [...raw, ...new Array<number | null>(n - raw.length).fill(null)];
  });
  const out = spliceGaps(ts, [col.slice(0, n), ...ex], { step: medianStep(ts) });
  return { t: out.t, v: out.cols[0]!, extras: out.cols.slice(1), gaps: out.gaps };
}

/**
 * The dI/dV curve, split by whether the lock-in was actually modulating.
 *
 * The dI/dV auxiliary channel needs to mark whether the lock-in was actually on. This is not a
 * display preference. With modulation off the demodulator puts out crosstalk and
 * noise — same unit, same order of magnitude, same shape as a real dI/dV — and
 * the one thing that differs, whether it is a measurement at all, is invisible
 * in the numbers. Drawn as one uniform curve, the cheap reading is always "this
 * is dI/dV".
 *
 * TWO drawn series, not three. `off` holds everything that is **not confirmed
 * modulating**, which lumps "confirmed off" together with "never read" — on the
 * chart both mean the same thing: do not read this as dI/dV. The two are still
 * counted separately (`nOff` / `nUnknown`) and said apart in words, because the
 * ACTIONS differ: modulation off is a knob someone turned, an unread state is a
 * link to go check. Never collapse "not read" into "off" — that reading throws
 * away a real dI/dV as if it were noise.
 *
 * A transition point belongs to BOTH series, so the two styles meet instead of
 * leaving a one-sample hole that would look like the gap markers.
 */
export function auxLockinSplit(body: AuxSeriesBody): {
  t: number[];
  on: (number | null)[];
  off: (number | null)[];
  n: number;
  gaps: number;
  nOn: number;
  nOff: number;
  nUnknown: number;
} {
  const { t, v, extras, gaps } = auxColumns(body, "lockin_a", ["lockin_mod_on"]);
  const flag = extras[0]!;
  const on: (number | null)[] = [];
  const off: (number | null)[] = [];
  let nOn = 0;
  let nOff = 0;
  let nUnknown = 0;

  const modulating = (i: number) => flag[i] === 1;
  for (let i = 0; i < v.length; i += 1) {
    const y = v[i] ?? null;
    const mine = modulating(i);
    const prev = i > 0 ? modulating(i - 1) : mine;
    // 过渡点两条都放：否则两种线型之间会留一个采样点的洞，而那个洞和「采集中断」
    // 画出来一模一样 —— 刚修好的那个谎会从另一头回来。
    const bridge = prev !== mine;
    on.push(mine || (bridge && prev) ? y : null);
    off.push(!mine || (bridge && !prev) ? y : null);
    if (y != null) {
      if (flag[i] === 1) nOn += 1;
      else if (flag[i] === 0) nOff += 1;
      else nUnknown += 1;
    }
  }
  return { t, on, off, n: countReadings(v), gaps, nOn, nOff, nUnknown };
}

/**
 * Is the sampler oversampling the qPlus amplitude, and by how much?
 *
 * `tauS` is the amplitude's 1/e relaxation time Q/(π f₀). The amplitude of a
 * high-Q resonator physically cannot move faster than that, so sampling faster
 * than τ buys demodulator noise and nothing else. Returns null when the
 * resonance has never been swept — that is "unknown", not "fine".
 *
 * `intervalS` 要传**实测**节奏（`aux.observed_interval_s`），不是设置值。
 * 采样是机会式的，设置只是节流上限，两个数一直不相等 —— 传设置值的话这句话
 * 回答的是「假如每次机会都采得到，够不够」，一个没人问的问题，而且长得像答案。
 */
export function auxAmpSamplingNote(
  tauS: number | null | undefined,
  intervalS: number | null | undefined,
): string | null {
  if (tauS == null || !Number.isFinite(tauS) || tauS <= 0) return null;
  if (intervalS == null || !Number.isFinite(intervalS) || intervalS <= 0) return null;
  const ratio = tauS / intervalS;
  const tau = tauS >= 1 ? `${tauS.toFixed(2)} s` : `${(tauS * 1000).toFixed(0)} ms`;
  return ratio >= 1
    ? `振幅弛豫 τ = ${tau}，采样间隔 ${intervalS.toFixed(2)} s —— 过采样 ${ratio.toFixed(1)}×，够了`
    : `振幅弛豫 τ = ${tau}，采样间隔 ${intervalS.toFixed(2)} s —— 比 τ 还慢，会漏掉快速归零`;
}

// ── settings knobs ──────────────────────────────────────────────────────────

/** One row of `GET /api/monitoring/config`'s `knobs[]`. */
// ── knob helpers ────────────────────────────────────────────────────────────
// Moved to lib/knobs.ts when the environment-history recorder grew a second
// knob catalogue: the whole-replace guard is a property of the settings STORE,
// not of the current monitor, and a copied guard is one that drifts. Re-exported
// here so existing imports (SettingsPage, MonitoringPage, the node tests) keep
// working unchanged.
export {
  buildKnobPayload,
  clampKnob,
  fmtKnobValue,
  type KnobLike,
} from "./knobs.ts";

/** Pull one metric's history out of a feature series, skipping rows that lack
 *  it (a column can appear mid-history when the extractor gains a feature). */
export function metricSeries(
  rows: ReadonlyArray<{ metrics?: { [k: string]: number } | null }>,
  key: string,
): number[] {
  const out: number[] = [];
  for (const r of rows) {
    const v = r.metrics?.[key];
    if (typeof v === "number" && Number.isFinite(v)) out.push(v);
  }
  return out;
}
