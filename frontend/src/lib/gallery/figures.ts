// ════════════════════════════════════════════════════════════════════════════
//
// 出图是图库流程的最后一环：预处理 → 展示 → 人工标记 → 出图。图由服务端
// `mast.gallery.figures` 生成，前端负责：
// 发起、看进度、列产物。这里放其中不碰 React 的部分——分组建议、表单 ↔ options、
// 预填匹配、产物列表的排序与文字。
//
// 两条值得记住的规则：
//   * **拉线谱的「同一条线」只是建议**。用户勾选系列：
//     系列名去掉「区组k」一类记号与末尾括号注释后相同的，默认归为
//     一条线。用户确认之后才出图，站位由服务端按几何推断（D17），弹层里可以先预演。
//   * **预填按集合比，不按顺序**。同一组系列勾选的先后不同，是同一张图。
//
// 无 React、无值 import 以外的依赖 ⇒ node --test 直接加载。
// ════════════════════════════════════════════════════════════════════════════

import type { components } from "@/api/schema";
import type { GalleryItem, Mark, Series } from "./types.ts";

export type FigureKind = components["schemas"]["GalleryFigureRequest"]["kind"];
export type FigureCategoryKey = components["schemas"]["GalleryFigureCategory"]["key"];
export type FigureCategory = components["schemas"]["GalleryFigureCategory"];
export type FigureEntry = components["schemas"]["GalleryFigureEntry"];
export type FigureFile = components["schemas"]["GalleryFigureFile"];
export type FigureJobStatus = components["schemas"]["GalleryFigureJobStatus"];
export type StsLinePlan = components["schemas"]["GalleryStsLinePlan"];

// ── 类别与种类 ──────────────────────────────────────────────────────────

/** 设计 D16 的类别顺序。服务端漏回某一类时前端照样摆出空的那一栏（告诉人还能出什么）。 */
export const CATEGORY_ORDER: readonly FigureCategoryKey[] = ["frames", "grids", "sts_lines", "sts_stitch", "series"];

export const CATEGORY_TITLE: Record<FigureCategoryKey, string> = {
  frames: "标记帧",
  grids: "网格谱",
  sts_lines: "拉线谱",
  sts_stitch: "单根谱拼接",
  series: "旋转系列",
};

/** 空类别下的一句「怎么出」。 */
export const CATEGORY_HINT: Record<FigureCategoryKey, string> = {
  frames: "顶部「为已标记出图」，或在大图侧栏点「出对比图」。",
  grids: "顶部「全部网格谱出图」，或在网格谱的大图侧栏点「出逐层图」。",
  sts_lines: "在一个谱系列的系列页点「拉线谱出图…」，勾上同一条线的各个区组。",
  sts_stitch: "在「已标记」页的「不属于任何系列的单根谱」里按目录拼接，或选中几条谱后在底部点「拼接出图」。",
  series: "在一个帧系列的系列页点「16:9 拼图」或「叠加…」。",
};

export const KIND_LABEL: Record<FigureKind, string> = {
  marked_frames: "为已标记出图",
  frame_sheet: "帧对比图",
  grid_sheets: "网格逐层图",
  sts_lines: "拉线谱",
  sts_stitch: "单根谱拼接",
  series_slides: "16:9 拼图",
  series_stack: "旋转叠加",
};

export const JOB_PHASE: Record<string, string> = {
  idle: "空闲",
  running: "出图中",
  done: "完成",
  cancelled: "已取消",
  error: "出错",
};

export function categoryTitle(key: string, fallback?: string | null): string {
  return (CATEGORY_TITLE as Record<string, string>)[key] ?? (fallback || key);
}

const categoryRank = (key: string): number => {
  const i = (CATEGORY_ORDER as readonly string[]).indexOf(key);
  return i < 0 ? CATEGORY_ORDER.length : i;
};

/** 按 D16 排序并补齐五个类别；认不出的类别排在最后，标题用服务端给的。 */
export function withAllCategories(cats: readonly FigureCategory[]): FigureCategory[] {
  const have = new Map(cats.map((c) => [c.key as string, c]));
  const out: FigureCategory[] = cats.map((c) => ({ ...c, title: categoryTitle(c.key, c.title) }));
  for (const key of CATEGORY_ORDER) {
    if (!have.has(key)) out.push({ key, title: CATEGORY_TITLE[key], figures: [] });
  }
  return out.sort((a, b) => categoryRank(a.key) - categoryRank(b.key) || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

/** 全部条目（跨类别），给预填匹配用。 */
export function allEntries(cats: readonly FigureCategory[] | undefined): FigureEntry[] {
  return (cats ?? []).flatMap((c) => c.figures ?? []);
}

// ── 同一条线的分组建议 ─────────────────────────────────────────────────

/** 区组记号：`区组0` / `组 2` / `block 1` / `blk3`（大小写不敏感）。 */
export const BLOCK_TOKEN = /(区组|组|block|blk)\s*\d+/i;
const TRAILING_NOTE = /\s*[（(][^（）()]*[）)]\s*$/;

/** `batch A · block 2 · line X` → `batch A · line X`；末尾括号注释同样去除。 */
export function normaliseLineName(name: string): string {
  let s = String(name ?? "").replace(new RegExp(BLOCK_TOKEN.source, "gi"), "");
  // 末尾括号注释先去：「A · block 2 (retry)」去掉记号后是「A ·  (retry)」，先收拾分隔的话，
  // 末尾那个「·」被括号挡着，会留下来。
  s = s.replace(TRAILING_NOTE, "");
  // 记号删掉之后留下的空分隔：「A ·  · B」→「A · B」，首尾的分隔去掉。
  s = s.replace(/(\s*·\s*){2,}/g, " · ").replace(/^\s*·\s*/, "").replace(/\s*·\s*$/, "");
  return s.replace(/\s+/g, " ").trim();
}

/** 分组键：归一化之后什么都不剩（名字就是一个区组记号）时用原名，不和别人并。 */
export function lineKey(name: string): string {
  return normaliseLineName(name) || String(name ?? "");
}

/** 名字里的区组号（没有记号时为 Infinity，排在最后）。 */
export function blockNumber(name: string): number {
  const m = String(name ?? "").match(BLOCK_TOKEN);
  const d = m ? m[0].match(/\d+/) : null;
  return d ? Number(d[0]) : Number.POSITIVE_INFINITY;
}

/** 名字里的区组记号本身（「区组0」），没有就用整个名字。 */
export function blockLabel(name: string): string {
  const m = String(name ?? "").match(BLOCK_TOKEN);
  return m ? m[0] : String(name ?? "");
}

export interface LineGroup {
  /** 归一化后的线名，也是分组键。 */
  lineName: string;
  /** 按区组号、再按名字排。 */
  sids: string[];
}

const byString = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * 谱系列按「同一条线」分组。`isSpectra` 决定哪些系列参与；缺省看系列自己的 `k === "s"`
 * （帧系列、混合系列不参与——拉线谱只对谱有意义）。
 */
export function suggestLineGroups(
  series: Record<string, Series>,
  isSpectra: (sid: string, s: Series) => boolean = (_sid, s) => s.k === "s",
): LineGroup[] {
  const groups = new Map<string, string[]>();
  for (const [sid, s] of Object.entries(series)) {
    if (!isSpectra(sid, s)) continue;
    const key = lineKey(s.name);
    const cur = groups.get(key);
    if (cur) cur.push(sid);
    else groups.set(key, [sid]);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => byString(a, b))
    .map(([lineName, sids]) => ({
      lineName,
      sids: sids.sort((x, y) => {
        const nx = series[x]?.name ?? "";
        const ny = series[y]?.name ?? "";
        return blockNumber(nx) - blockNumber(ny) || byString(nx, ny) || byString(x, y);
      }),
    }));
}

/** 某个系列所在的建议组（找不到就只含它自己）。 */
export function groupFor(groups: readonly LineGroup[], sid: string): LineGroup {
  return groups.find((g) => g.sids.includes(sid)) ?? { lineName: "", sids: [sid] };
}

/** 勾选的这些系列的默认线名：归一化名都一样就用它，否则取公共前缀（去掉末尾分隔）。 */
export function commonLineName(series: Record<string, Series>, sids: readonly string[]): string {
  const names = sids.map((sid) => series[sid]?.name).filter((n): n is string => !!n).map(lineKey);
  const first = names[0];
  if (first === undefined) return "";
  if (names.every((n) => n === first)) return first;
  let p = first;
  for (const n of names.slice(1)) {
    while (p && !n.startsWith(p)) p = p.slice(0, -1);
  }
  return p.replace(/[\s·]+$/, "").trim();
}

// ── 拉线谱表单 ↔ options ────────────────────────────────────────────────

export interface MarkRow {
  station: string;
  label: string;
}

const parseStation = (s: string): number | null => {
  const t = String(s ?? "").trim();
  return /^\d+$/.test(t) ? Number(t) : null;
};

/** `{"5": "V"}` → 表格行（按站位号排）。 */
export function stationMarksToRows(marks: unknown): MarkRow[] {
  if (!marks || typeof marks !== "object" || Array.isArray(marks)) return [];
  return Object.entries(marks as Record<string, unknown>)
    .map(([k, v]) => ({ station: String(k), label: String(v ?? "") }))
    .sort((a, b) => (parseStation(a.station) ?? Infinity) - (parseStation(b.station) ?? Infinity));
}

/** 表格行 → `{"5": "V"}`。站位号不是非负整数、或文字为空的行丢掉；同一站位后写的赢。 */
export function rowsToStationMarks(rows: readonly MarkRow[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const r of rows) {
    const n = parseStation(r.station);
    const label = String(r.label ?? "").trim();
    if (n == null || !label) continue;
    out[String(n)] = label;
  }
  return out;
}

/** options.bad_from → 表单（系列号 → 文字）。 */
export function badFromToForm(badFrom: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!badFrom || typeof badFrom !== "object" || Array.isArray(badFrom)) return out;
  for (const [sid, v] of Object.entries(badFrom as Record<string, unknown>)) {
    const n = Number(v);
    if (Number.isInteger(n) && n >= 0) out[sid] = String(n);
  }
  return out;
}

/** 表单 → options.bad_from：只收勾选了的系列、只收非负整数（空 = 这组没有针尖在变）。 */
export function formToBadFrom(form: Record<string, string>, checked: readonly string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const sid of checked) {
    const t = String(form[sid] ?? "").trim();
    if (/^\d+$/.test(t)) out[sid] = Number(t);
  }
  return out;
}

export interface StsLinesForm {
  lineName: string;
  marks: MarkRow[];
  badFrom: Record<string, string>;
  excludeRejected: boolean;
}

export function stsLinesOptions(form: StsLinesForm, checked: readonly string[]): Record<string, unknown> {
  const o: Record<string, unknown> = {
    station_marks: rowsToStationMarks(form.marks),
    bad_from: formToBadFrom(form.badFrom, checked),
    exclude_rejected: form.excludeRejected,
  };
  const ln = form.lineName.trim();
  if (ln) o.line_name = ln;
  return o;
}

export function formFromOptions(options: Record<string, unknown> | null | undefined, defaultLineName: string): StsLinesForm {
  const ln = options?.line_name;
  return {
    lineName: typeof ln === "string" && ln.trim() ? ln : defaultLineName,
    marks: stationMarksToRows(options?.station_marks),
    badFrom: badFromToForm(options?.bad_from),
    excludeRejected: options?.exclude_rejected === undefined ? true : !!options.exclude_rejected,
  };
}

// ── 预填 ───────────────────────────────────────────────────────────────

/** 两组系列号是不是同一个集合（与顺序、重复无关）。 */
export function sameSet(a: readonly string[], b: readonly string[]): boolean {
  const sa = new Set(a);
  const sb = new Set(b);
  if (sa.size !== sb.size) return false;
  for (const x of sa) if (!sb.has(x)) return false;
  return true;
}

/** 同一组系列最近一次的拉线谱出图（给弹层预填），没有就 null。 */
export function findPrefill(entries: readonly FigureEntry[], sids: readonly string[]): FigureEntry | null {
  if (!sids.length) return null;
  const hits = entries.filter((e) => e.kind === "sts_lines" && sameSet(e.series ?? [], sids));
  if (!hits.length) return null;
  return [...hits].sort((a, b) => byString(b.created || "", a.created || ""))[0] ?? null;
}

// ── 产物列表的文字 ─────────────────────────────────────────────────────

/** 整数原样；|v| ≥ 100 取整；其余三位有效数字。 */
export function fmtNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) >= 100) return v.toFixed(0);
  return String(Number(v.toPrecision(3)));
}

/** summary 里的一个值 → 小标签文字；嵌套对象、空值不上标签（回 null）。 */
export function fmtSummaryValue(v: unknown): string | null {
  if (v == null) return null;
  if (typeof v === "number") return fmtNumber(v);
  if (typeof v === "boolean") return v ? "是" : "否";
  if (typeof v === "string") return v.length > 40 ? `${v.slice(0, 39)}…` : v;
  if (Array.isArray(v)) {
    if (v.length && v.length <= 4 && v.every((x) => typeof x === "number")) {
      return (v as number[]).map(fmtNumber).join(" / ");
    }
    return `${v.length} 项`;
  }
  return null;
}

export function summaryChips(summary: Record<string, unknown> | null | undefined, limit = 8): { key: string; text: string }[] {
  const out: { key: string; text: string }[] = [];
  for (const [k, v] of Object.entries(summary ?? {})) {
    const t = fmtSummaryValue(v);
    if (t == null || t === "") continue;
    out.push({ key: k, text: `${k} ${t}` });
    if (out.length >= limit) break;
  }
  return out;
}

/** options 的一行摘要：`key=value · …`，空值与空对象不写。 */
export function optionsText(options: Record<string, unknown> | null | undefined): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(options ?? {})) {
    if (v == null || v === "") continue;
    if (typeof v === "object") {
      const j = JSON.stringify(v);
      if (j === "{}" || j === "[]") continue;
      parts.push(`${k}=${j.length > 60 ? `${j.slice(0, 59)}…` : j}`);
    } else {
      parts.push(`${k}=${typeof v === "number" ? fmtNumber(v) : String(v)}`);
    }
  }
  return parts.join(" · ");
}

export function fmtBytes(n: number | null | undefined): string {
  const b = Number(n) || 0;
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

const EXT_ORDER = ["png", "jpg", "csv", "npy", "json"];
const extRank = (ext: string): number => {
  const i = EXT_ORDER.indexOf(String(ext).toLowerCase());
  return i < 0 ? EXT_ORDER.length : i;
};

/** 下载链接：png → jpg → csv → npy → json，同类按文件名。 */
export function fileLinks(files: readonly FigureFile[] | null | undefined): { name: string; url: string; label: string }[] {
  return [...(files ?? [])]
    .sort((a, b) => extRank(a.ext) - extRank(b.ext) || byString(a.name, b.name))
    .map((f) => ({ name: f.name, url: f.url, label: `${String(f.ext).toUpperCase()} ${fmtBytes(f.size)}` }));
}

/** 这张图的「主文件」：`<base>.png`，没有就第一张 png，再没有就 jpg。 */
export function mainImage(entry: Pick<FigureEntry, "base" | "files">): FigureFile | null {
  const files = entry.files ?? [];
  const byExt = (ext: string) => files.filter((f) => String(f.ext).toLowerCase() === ext).sort((a, b) => byString(a.name, b.name));
  return (
    files.find((f) => f.name === `${entry.base}.png`) ?? byExt("png")[0] ?? byExt("jpg")[0] ?? null
  );
}

export function previewUrl(entry: Pick<FigureEntry, "base" | "files">): string | null {
  const main = mainImage(entry);
  if (main?.preview_url) return main.preview_url;
  return (entry.files ?? []).find((f) => f.preview_url)?.preview_url ?? null;
}

export function fullImageUrl(entry: Pick<FigureEntry, "base" | "files">): string | null {
  return mainImage(entry)?.url ?? null;
}

// ── 单根谱（不属于任何系列的标记谱）按目录 ─────────────────────────────

/** 绘图实现 draw_sts.py 的 singles：有标记、是谱、不在任何系列里、索引里有；按目录分组、组内按时间。 */
export function singleSpectraByDir(
  markItems: Record<string, Mark>,
  series: Record<string, Series>,
  byId: ReadonlyMap<string, GalleryItem>,
): { d: string; ids: string[] }[] {
  const inSeries = new Set(Object.values(series).flatMap((s) => s.ids || []));
  const groups = new Map<string, GalleryItem[]>();
  for (const id of Object.keys(markItems)) {
    const it = byId.get(id);
    if (!it || it.k !== "s" || inSeries.has(id)) continue;
    const cur = groups.get(it.d);
    if (cur) cur.push(it);
    else groups.set(it.d, [it]);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => byString(a, b))
    .map(([d, its]) => ({
      d,
      ids: its.sort((a, b) => (a.t ?? a.mt ?? 0) - (b.t ?? b.mt ?? 0) || byString(a.fn, b.fn)).map((x) => x.id),
    }));
}

// ── 发起出图 ───────────────────────────────────────────────────────────

/**
 * 点了一个出图按钮之后该怎么说。出图只有一个任务槽：已经有任务在跑时，服务端回的是
 * **那个**任务的状态，所以「请求成功」不等于「我要的开始了」。
 */
export function describeStart(
  prev: FigureJobStatus | null | undefined,
  next: FigureJobStatus | null | undefined,
  kind: FigureKind,
): { ok: boolean; text: string } {
  if (!next) return { ok: false, text: "出图没有开始：请求失败" };
  if (next.degraded) return { ok: false, text: `出图没有开始：${next.detail || "出图后端不可用"}` };
  const busy = (k: FigureKind | null | undefined) => `已有出图任务在跑（${KIND_LABEL[k ?? kind] ?? k}），等它结束再出`;
  if (prev?.running) return { ok: false, text: busy(prev.kind) };
  if (next.running && next.kind && next.kind !== kind) return { ok: false, text: busy(next.kind) };
  if (next.phase === "error") return { ok: false, text: `出图出错：${next.message || next.detail || ""}` };
  return { ok: true, text: `已开始：${KIND_LABEL[kind]}` };
}

/** κ（nm⁻¹）输入框：正数才收。 */
export function parseKappa(text: string): number | null {
  const t = String(text ?? "").trim();
  if (!/^\d*\.?\d+$/.test(t)) return null;
  const v = Number(t);
  return Number.isFinite(v) && v > 0 ? v : null;
}

/** 正整数输入框（系列帧数上限等）。 */
export function parsePositiveInt(text: string): number | null {
  const t = String(text ?? "").trim();
  if (!/^\d+$/.test(t)) return null;
  const v = Number(t);
  return v >= 1 ? v : null;
}
