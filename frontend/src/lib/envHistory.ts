/**
 * 环境历史 — pure logic behind the 环境历史 page. `node --test`able.
 *
 * Everything here is a transformation of what `/api/env-history/*` returns.
 * Nothing here fetches, and nothing here decides *when* to fetch — that lives
 * in the page, with the intervals in `lib/pollRates.ts`.
 *
 * The one import is `lib/seriesGaps.ts`, which used to be inlined below. It got
 * pulled out when the monitoring charts turned out to need the same rule and did
 * not have it — two copies of "where does a
 * series stop being continuous" is the shape this repo has already paid for.
 */
import { spliceGaps } from "./seriesGaps.ts";

/** One aggregated bucket as the series endpoint returns it. */
export interface EnvPointLike {
  ts: number;
  mean?: number | null;
  min?: number | null;
  max?: number | null;
  std?: number | null;
  n?: number;
  n_excluded?: number;
  worst_status?: string;
}

/**
 * Named time ranges. `s` is seconds back from now; `null` = everything.
 *
 * 48 小时 was missing for no reason at all: the server takes `since` as
 * a bare epoch float with no window constraint whatsoever
 * (`api/routes/env_history.py:184`), and re-aggregates buckets by weight for any
 * span, so this list is a menu of convenient spans and nothing more. The jump it
 * fills was 24 小时 → 7 天, a factor of seven — an overnight-plus-a-day question
 * (did the cryostat hold overnight?) had no button.
 *
 * Kept as ONE list rather than a free-form date picker on purpose: the aux and
 * live-current charts already switched windows this way , and a second
 * mechanism for "how far back am I looking" is the thing this repo keeps paying
 * for. Anything not on this menu is what 环境历史 → 全部 plus a drag-zoom is for.
 */
export const RANGES: ReadonlyArray<{ s: number | null; label: string }> = [
  { s: 3600, label: "1 小时" },
  { s: 86400, label: "24 小时" },
  { s: 2 * 86400, label: "48 小时" },
  { s: 7 * 86400, label: "7 天" },
  { s: 30 * 86400, label: "30 天" },
  { s: 365 * 86400, label: "1 年" },
  { s: null, label: "全部" },
];

/** `[since, until]` for a named range. `until` stays undefined so the server
 *  returns everything up to the newest bucket — pinning it to `now` would clip
 *  a bucket whose timestamp is its START. */
export function rangeWindow(
  seconds: number | null,
  nowMs: number,
): { since?: number; until?: number } {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return {};
  return { since: nowMs / 1000 - seconds };
}

/**
 * Split a series into uPlot's aligned-data columns.
 *
 * Gaps matter: a recorder that was off for six hours must draw a HOLE, not a
 * straight line joining the two sides of it. That rule now lives in
 * `lib/seriesGaps.ts` — see there for why null breaks a uPlot series *and* its
 * band fill.
 *
 * `bucketS` is the DECLARED bucket width the server computed, so it is passed
 * straight through rather than re-derived from the timestamps. That is the
 * difference from the aux charts, which have no declared step and have to
 * self-calibrate: here a bucket width of 0 means "the server did not say", and
 * the honest response to that is to draw no breaks at all.
 *
 * `gapFactor` stays at 2.5 rather than the shared default of 3: these buckets
 * are already aggregated, so a single missing bucket IS the recorder having been
 * down for a whole bucket — there is no per-sample jitter left to absorb.
 */
export function toAlignedData(
  points: readonly EnvPointLike[],
  bucketS: number,
  gapFactor = 2.5,
): [number[], (number | null)[], (number | null)[], (number | null)[]] {
  const ts: number[] = [];
  const mean: (number | null)[] = [];
  const lo: (number | null)[] = [];
  const hi: (number | null)[] = [];
  for (const p of points) {
    if (!p || !Number.isFinite(p.ts)) continue;
    ts.push(p.ts);
    mean.push(num(p.mean));
    lo.push(num(p.min));
    hi.push(num(p.max));
  }
  const out = spliceGaps(ts, [mean, lo, hi], { step: bucketS, factor: gapFactor });
  return [out.t, out.cols[0]!, out.cols[1]!, out.cols[2]!];
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Should this series be drawn on a log Y axis?
 *
 * Pressure spans decades (1e-10 … 1e2 Pa) and is unreadable linearly. Decided
 * from the DATA, not the sensor name: a site can name its gauge anything, but
 * four decades of span always means log. Requires strictly positive values —
 * `log(0)` is what makes uPlot draw nothing at all.
 */
export function preferLogScale(unit: string, points: readonly EnvPointLike[]): boolean {
  const vals: number[] = [];
  for (const p of points) {
    const v = num(p.mean) ?? num(p.max);
    if (v != null && v > 0) vals.push(v);
  }
  if (vals.length < 2) return false;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of vals) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(lo > 0) || !(hi > 0)) return false;
  const decades = Math.log10(hi / lo);
  return decades >= 3 || (unit === "Pa" && decades >= 1.5);
}

/** Y-axis label: the unit if there is one, else the bare series name. */
export function axisLabel(sensor: string, unit: string): string {
  return unit ? `${sensor} (${unit})` : sensor;
}

const SEVERITY: Readonly<Record<string, number>> = {
  ok: 0,
  unavailable: 1,
  warning: 2,
  error: 3,
  alarm: 4,
};

/** Worst status across a set of buckets — for the "this window contained an
 *  alarm" badge. Uses the same ordering as the backend, so the badge cannot
 *  disagree with the row colour. */
export function worstOf(points: readonly EnvPointLike[]): string {
  let worst = "ok";
  for (const p of points) {
    const s = String(p?.worst_status || "ok");
    if ((SEVERITY[s] ?? 0) > (SEVERITY[worst] ?? 0)) worst = s;
  }
  return worst;
}

/**
 * Badge tone for a bucket status — BY SEVERITY, which is the whole point.
 *
 * The page used to do `worst === "warning" ? "WARN" : "DANGEROUS"`, so
 * `unavailable` — severity 1, the *least* severe non-ok state — was painted in
 * exactly the same red as `alarm`, severity 4. On the rig that is not a corner
 * case: the two Lake Shore channels share one COM13 handle and only ONE of them
 * can hold it at a time — which one changes occasionally (4 handovers in the
 * 29 h to 2026-08-04, the two series exactly anti-phased, a perfect partition
 * of the same 43779 reading slots). So each channel is legitimately
 * `unavailable` for every bucket the other one owned the port. A red alarm
 * badge over that is the page crying wolf (operator ).
 */
export function statusTone(status: string): string {
  switch (status) {
    case "alarm":
    case "error":
      return "DANGEROUS";
    case "warning":
      return "WARN";
    case "unavailable":
      return "INFO";
    default:
      return "AUTO";
  }
}

/** Human label for a bucket status. `unavailable` is not an error condition —
 *  it means no reading arrived — and the badge should say so in words, not
 *  leave the operator to infer it from a colour. */
export function statusLabel(status: string): string {
  switch (status) {
    case "alarm":
      return "报警";
    case "error":
      return "读取错误";
    case "warning":
      return "告警";
    case "unavailable":
      return "部分时段无读数";
    default:
      return "正常";
  }
}

/** How many readings the quiet gate / sensor faults kept out of a window.
 *  Surfaced because "the tunnelling-current line has a hole here" and "the
 *  instrument was busy here" are the same fact, and the operator should not
 *  have to guess which. */
export function excludedCount(points: readonly EnvPointLike[]): number {
  let n = 0;
  for (const p of points) n += Number(p?.n_excluded || 0);
  return n;
}

/**
 * Counted vs excluded readings over a window, plus how many buckets actually
 * have a value to draw.
 *
 * A bare "1521 条读数未计入" reads like data loss. It is only interpretable
 * next to how many DID count: 1521-of-1800 is a sensor that is barely being
 * read, 1521-of-43779 is a normal duty cycle. `drawable` separates the third
 * case the page could not express at all — buckets exist, so the series is
 * offered in the dropdown, but every one of them is empty, and the chart
 * renders blank with no explanation (operator ).
 */
export function coverageOf(points: readonly EnvPointLike[]): {
  counted: number;
  excluded: number;
  buckets: number;
  drawable: number;
} {
  let counted = 0;
  let excluded = 0;
  let drawable = 0;
  for (const p of points) {
    counted += Number(p?.n || 0);
    excluded += Number(p?.n_excluded || 0);
    if (typeof p?.mean === "number" && Number.isFinite(p.mean)) drawable += 1;
  }
  return { counted, excluded, buckets: points.length, drawable };
}

/** `instrument_quiet` is a synthetic 0/1 series; its bucket mean is the duty
 *  cycle. Rendered as a percentage rather than "0.83". */
export function isDutyCycleSeries(sensor: string): boolean {
  return sensor === "instrument_quiet";
}

/** Compact absolute timestamp for axis ticks and readouts (local time). */
export function fmtTs(epochS: number, withDate = true): string {
  if (!Number.isFinite(epochS)) return "";
  const d = new Date(epochS * 1000);
  const p2 = (n: number) => String(n).padStart(2, "0");
  const hm = `${p2(d.getHours())}:${p2(d.getMinutes())}`;
  if (!withDate) return hm;
  return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${hm}`;
}

/** Human duration for "covers 3 天 4 小时". */
export function fmtSpan(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return h > 0 ? `${d} 天 ${h} 小时` : `${d} 天`;
  if (h > 0) return m > 0 ? `${h} 小时 ${m} 分` : `${h} 小时`;
  if (m > 0) return `${m} 分`;
  return `${Math.round(seconds)} 秒`;
}

/** Bytes → human. Local copy so this module keeps its zero-import property. */
export function fmtBytes(n: number | null | undefined): string {
  if (typeof n !== "number" || !Number.isFinite(n) || n < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

/** Channel label for a spectrum. */
export function channelLabel(ch: string): string {
  if (ch === "current") return "电流噪声谱";
  if (ch === "z") return "Z 噪声谱";
  return ch || "—";
}

/** State a spectrum was recorded under — the `ctx_*` columns of `env_spectra`. */
export interface SpectrumCtx {
  quietness?: string;
  ctx_bias_v?: number | null;
  ctx_setpoint_a?: number | null;
  ctx_zctrl_on?: boolean | null;
  ctx_stable?: boolean | null;
}

/**
 * Was the tip in tunnelling contact while this spectrum was accumulated?
 *
 * Four answers, not two, and the fourth is the one that matters: a spectrum
 * taken with no junction is a measurement of the AMPLIFIER, not of the tip, and
 * comparing it against a tunnelling one is comparing two different instruments.
 * `unknown` must therefore never collapse into "no" — "we could not tell" is
 * not evidence of absence, and the operator reads this to decide whether two
 * spectra are comparable at all.
 *
 * Mirrors `mast.envhistory.quiet.classify` (quiet / active / no_tunnel /
 * unknown); the Z-controller flag refines it when quietness is absent.
 */
export function tunnellingLabel(c: SpectrumCtx): { label: string; tone: string } {
  const q = String(c.quietness || "");
  // Tone names are the Badge vocabulary (ui.tsx BADGE_TONE) — UPPERCASE, and an
  // unknown key silently falls back to `default`, so a lowercase "warn" here
  // would render as a neutral chip and quietly stop warning.
  if (q === "no_tunnel") return { label: "未隧穿", tone: "WARN" };
  if (q === "quiet") return { label: "隧穿中 · 安静", tone: "AUTO" };
  if (q === "active") return { label: "隧穿中 · 有操作", tone: "INFO" };
  if (q === "unknown" || q === "") {
    if (c.ctx_zctrl_on === true) return { label: "Z 反馈开", tone: "INFO" };
    if (c.ctx_zctrl_on === false) return { label: "未隧穿", tone: "WARN" };
    return { label: "状态未知", tone: "default" };
  }
  return { label: q, tone: "default" };
}

/**
 * One-line caveat when the working point moved mid-accumulation, else "".
 *
 * A spectrum can span half an hour. If bias or setpoint changed inside that
 * window, the recorded numbers describe the START of the window and the curve
 * itself is a mixture — so the honest thing is to say so rather than print a
 * precise-looking number. `null` (old rows, recorded before this was tracked)
 * is NOT the same as `true` and must not be reassured about.
 */
/**
 * Dropdown label for one archived spectrum: timestamp plus working point.
 *
 * A list of bare timestamps cannot answer the question the dropdown exists for
 * — "which of these is comparable to the one I am looking at?" Two spectra
 * taken at different bias, or one with no junction at all, are measurements of
 * different things, and picking them side by side produces a difference that
 * means nothing.
 */
export function spectrumPickerLabel(
  m: SpectrumCtx & { ts: number },
  fmtTsFn: (t: number) => string,
): string {
  const bits = [fmtTsFn(m.ts), tunnellingLabel(m).label];
  if (typeof m.ctx_bias_v === "number") bits.push(`${m.ctx_bias_v.toPrecision(3)} V`);
  return bits.join(" · ");
}

export function ctxStabilityNote(c: SpectrumCtx): string {
  if (c.ctx_stable === false) {
    return "攒谱期间偏压 / 设定点 / Z 反馈变过——下面的数是窗口起点的值，这条谱是混合的";
  }
  if (c.ctx_stable == null) return "这条谱记录时还没有跟踪工作点是否变化";
  return "";
}

/**
 * Log-log points for a spectrum, dropping non-positive samples.
 *
 * A single zero or negative PSD value (a fully saturated run, or a bin that
 * underflowed) makes uPlot's log scale render an empty plot — dropping those
 * points loses one bin, keeping them loses the whole curve.
 */
export function spectrumSeries(
  freqs: readonly number[],
  psd: readonly number[],
): [number[], number[]] {
  const f: number[] = [];
  const p: number[] = [];
  const n = Math.min(freqs.length, psd.length);
  for (let i = 0; i < n; i += 1) {
    const x = freqs[i];
    const y = psd[i];
    if (typeof x !== "number" || typeof y !== "number") continue;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x <= 0 || y <= 0) continue;
    f.push(x);
    p.push(y);
  }
  return [f, p];
}

/**
 * Amplitude spectral density from power spectral density: √PSD.
 *
 * The stored quantity is A²/Hz (or m²/Hz) because that is what integrates to a
 * variance. Operators read noise floors in A/√Hz, and every SPM datasheet is
 * written that way, so the viewer offers both and this is the conversion.
 */
export function toAsd(psd: readonly number[]): number[] {
  return psd.map((v) => (Number.isFinite(v) && v > 0 ? Math.sqrt(v) : NaN));
}

/** Unit string after the √ conversion above. `A^2/Hz` → `A/√Hz`. */
export function asdUnit(psdUnit: string): string {
  const m = /^([A-Za-z]+)\^2\/Hz$/.exec(psdUnit || "");
  return m ? `${m[1]}/√Hz` : psdUnit || "";
}
