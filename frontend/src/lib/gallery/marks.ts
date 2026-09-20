// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 标记的纯逻辑（旧版兼容格式 app.js 的 Store / updateMark / toggleTag）。
//
// 存盘协议（设计 D11/D12，与 mast/gallery/marks.py 的 apply_patch 成对）：
//   * 每次改动带一个**单调递增的毫秒戳** ts；服务端丢弃比它手里旧的改动。
//   * 删除写成 `{del: true, ts}`，服务端据此记墓碑——迟到的旧改动不能把删掉的复活。
//   * 空标记（无评级、无标签、无备注、无系定）不存，等同删除。
//   * 失败的一批改动**不覆盖**失败期间又产生的新改动（requeue）。
//
// React / zustand 那一层在 components/gallery/marksStore.ts；这里只有能单测的部分。
// ════════════════════════════════════════════════════════════════════════════

import type { DirMark, GalleryItem, Mark, MarksDoc, Patch, Series, Tombstone } from "./types.ts";
import { meta } from "./format.ts";

/** 没有 marks.json 时的标签表（通用，设计 §4.2；与 mast/gallery/marks.py 的 DEFAULT_TAGS 同一份）。 */
export const DEFAULT_TAGS: readonly string[] = [
  "原子分辨佳", "超结构", "空位/缺陷", "台阶/边界", "偏压系列", "针尖变化", "dI/dV", "作图候选", "漂移",
];

/** 没存上的改动在浏览器里的落点。 */
export const PENDING_KEY = "mast.gallery.pending";

export const emptyPatch = (): Patch => ({ items: {}, series: {}, days: {}, tags: null });

export function patchEmpty(p: Patch): boolean {
  return (
    !Object.keys(p.items).length &&
    !Object.keys(p.series).length &&
    !Object.keys(p.days).length &&
    !p.tags
  );
}

export function isTombstone(x: unknown): x is Tombstone {
  return !!x && typeof x === "object" && (x as { del?: unknown }).del === true;
}

export function isEmptyMark(m: Partial<Mark> | null | undefined): boolean {
  return !m || (!m.r && !(m.tags || []).length && !(m.note || "").trim() && !m.anchor);
}

/** 服务端文档 → 字段齐全的本地文档（缺的表补空、标签表缺了用默认）。 */
export function normaliseDoc(raw: unknown): MarksDoc {
  const o = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const obj = <T>(x: unknown): Record<string, T> =>
    x && typeof x === "object" && !Array.isArray(x) ? (x as Record<string, T>) : {};
  const tags = Array.isArray(o.tags) ? o.tags.filter((t): t is string => typeof t === "string") : null;
  return {
    version: typeof o.version === "number" ? o.version : 1,
    rev: typeof o.rev === "number" ? o.rev : 0,
    updated: typeof o.updated === "string" ? o.updated : "",
    tags: tags ?? [...DEFAULT_TAGS],
    items: obj<Mark>(o.items),
    series: obj<Series>(o.series),
    days: obj<DirMark>(o.days),
  };
}

/** 换一条单条标记（null / 墓碑 = 删除），返回新文档；不改动入参。 */
export function withMark(doc: MarksDoc, id: string, m: Mark | Tombstone | null): MarksDoc {
  const items = { ...doc.items };
  if (m && !isTombstone(m)) items[id] = m;
  else delete items[id];
  return { ...doc, items };
}

/** 换一个系列；null / 墓碑 / 成员为空 = 删除。 */
export function withSeries(doc: MarksDoc, sid: string, s: Series | Tombstone | null): MarksDoc {
  const series = { ...doc.series };
  if (s && !isTombstone(s) && (s.ids || []).length) series[sid] = s;
  else delete series[sid];
  return { ...doc, series };
}

/** 换一个目录的状态；既没勾「已过完」又没有备注 = 删除。 */
export function withDir(doc: MarksDoc, d: string, v: DirMark): MarksDoc {
  const days = { ...doc.days };
  if (!v.done && !(v.note || "").trim()) delete days[d];
  else days[d] = v;
  return { ...doc, days };
}

/** 把一整份 patch 应用到本地文档（加载时恢复上次没存上的改动用）。 */
export function applyPatchLocal(doc: MarksDoc, p: Patch): MarksDoc {
  let out = doc;
  for (const [id, m] of Object.entries(p.items)) out = withMark(out, id, m);
  for (const [sid, s] of Object.entries(p.series)) out = withSeries(out, sid, s);
  for (const [d, v] of Object.entries(p.days)) out = withDir(out, d, v);
  if (p.tags) out = { ...out, tags: [...p.tags] };
  return out;
}

/** 两份 patch 合并，`newer` 的同名键覆盖 `older`。 */
export function mergePatch(older: Patch, newer: Patch): Patch {
  return {
    items: { ...older.items, ...newer.items },
    series: { ...older.series, ...newer.series },
    days: { ...older.days, ...newer.days },
    tags: newer.tags ?? older.tags,
  };
}

/**
 * 一批改动没存上：放回待存队列，但**只填队列里还没有的键**——请求在路上的这段时间
 * 里操作员可能又改了同一条，那一条更新，不能被失败的旧版本盖掉（原版 flush 的 catch）。
 */
export function requeue(pending: Patch, failed: Patch): Patch {
  const fill = <T>(cur: Record<string, T>, old: Record<string, T>) => {
    const out = { ...cur };
    for (const [k, v] of Object.entries(old)) if (!(k in out)) out[k] = v;
    return out;
  };
  return {
    items: fill(pending.items, failed.items),
    series: fill(pending.series, failed.series),
    days: fill(pending.days, failed.days),
    tags: pending.tags ?? failed.tags,
  };
}

/** localStorage 里的待存改动 → Patch；坏数据当作没有。 */
export function parsePending(raw: string | null | undefined): Patch | null {
  if (!raw) return null;
  try {
    const o: unknown = JSON.parse(raw);
    if (!o || typeof o !== "object") return null;
    const r = o as Partial<Patch>;
    const obj = <T>(x: unknown): Record<string, T> =>
      x && typeof x === "object" && !Array.isArray(x) ? (x as Record<string, T>) : {};
    const p: Patch = {
      items: obj(r.items),
      series: obj(r.series),
      days: obj(r.days),
      tags: Array.isArray(r.tags) ? r.tags.filter((t): t is string => typeof t === "string") : null,
    };
    return patchEmpty(p) ? null : p;
  } catch {
    return null;
  }
}

/**
 * 改一条标记的纯部分（原版 updateMark）：在旧标记上叠加 patch，写上 `t ts tt k meta`；
 * 结果为空标记时回 null（= 删除）。
 */
export function composeMark(
  old: Mark | null | undefined,
  patch: Partial<Mark>,
  it: GalleryItem,
  now: string,
  ts: number,
): Mark | null {
  const base: Mark = old ?? { r: 0, tags: [], note: "" };
  const m: Mark = {
    ...base,
    ...patch,
    t: now,
    ts,
    tt: it.t ?? null,
    k: it.k,
    meta: `${it.d} · ${meta(it)}`,
  };
  return isEmptyMark(m) ? null : m;
}

/** 开关一个标签；`on` 缺省 = 取反。没有变化时回 null。 */
export function toggledTags(tags: readonly string[], t: string, on?: boolean): string[] | null {
  const out = [...tags];
  const i = out.indexOf(t);
  const want = on === undefined ? i < 0 : on;
  if (want && i < 0) out.push(t);
  else if (!want && i >= 0) out.splice(i, 1);
  else return null;
  return out;
}

/** 卡片上的 ✓★✗：再点同一个 = 取消。 */
export const ratingToggle = (current: number | undefined, clicked: number): number =>
  current === clicked ? 0 : clicked;

/** 标签表编辑框：逗号或中文逗号分隔，去空白、去重、保持顺序。 */
export function parseTagList(text: string): string[] {
  return [...new Set(text.split(/[,，]/).map((s) => s.trim()).filter(Boolean))];
}

/** 单调毫秒戳：同一毫秒里连按两次也严格递增（服务端据此排序改动）。 */
export function tsClock(now: () => number = Date.now): () => number {
  let last = 0;
  return () => {
    last = Math.max(now(), last + 1);
    return last;
  };
}
