// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 系列（旧版兼容格式 series.js 的查询部分）。
//
// 系列 = 一段连续的帧 / 谱 / 网格，存在 marks.json 的 series 里：
// `{name, k, ids, r, tags, note, anchor?}`。成员自己的单条标记与系列互不影响。
// ════════════════════════════════════════════════════════════════════════════

import type { GalleryItem, Series } from "./types.ts";
import { KLAB, bKey, dirLabel, fmtB, fmtT, num, stem } from "./format.ts";

const INDEX_CACHE = new WeakMap<object, Map<string, string[]>>();

/**
 * 条目 id → 它所在的系列号列表。按 `series` 对象身份缓存：marks 文档每改一次
 * series 就换一个新对象，于是缓存自然失效，而几百张卡片各自查询时不会各算一遍。
 */
export function seriesIndex(series: Record<string, Series>): Map<string, string[]> {
  let idx = INDEX_CACHE.get(series);
  if (!idx) {
    idx = new Map();
    for (const [sid, s] of Object.entries(series)) {
      for (const id of s.ids || []) {
        const cur = idx.get(id);
        if (cur) cur.push(sid);
        else idx.set(id, [sid]);
      }
    }
    INDEX_CACHE.set(series, idx);
  }
  return idx;
}

const EMPTY: readonly string[] = [];

export function seriesOf(series: Record<string, Series>, id: string): readonly string[] {
  return seriesIndex(series).get(id) ?? EMPTY;
}

/** 系列成员，按**索引顺序**（即时间顺序），只含索引里有的。 */
export function seriesMembers(s: Series | undefined, items: readonly GalleryItem[]): GalleryItem[] {
  if (!s) return [];
  const set = new Set(s.ids || []);
  return items.filter((it) => set.has(it.id));
}

/** 每个目录涉及几个系列（一个系列跨两个目录时两边各算一次）。 */
export function seriesCountByDir(
  series: Record<string, Series>,
  byId: ReadonlyMap<string, GalleryItem>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of Object.values(series)) {
    const dirs = new Set<string>();
    for (const id of s.ids || []) {
      const it = byId.get(id);
      if (it) dirs.add(it.d);
    }
    for (const d of dirs) out[d] = (out[d] ?? 0) + 1;
  }
  return out;
}

export function kindOf(items: readonly Pick<GalleryItem, "k">[]): "f" | "s" | "g" | "mix" {
  const ks = new Set(items.map((it) => it.k));
  if (ks.size !== 1) return "mix";
  return [...ks][0] ?? "mix";
}

/** 默认系列名：`2001-01-01 0001–0003 · 3 帧`，帧的偏压不止一个时附上范围。 */
export function autoName(items: readonly GalleryItem[]): string {
  const a = items[0];
  const b = items[items.length - 1];
  if (!a || !b) return "新系列";
  const k = kindOf(items);
  let s = `${dirLabel(a.d)} ${num(a)}–${num(b)} · ${items.length} ${k === "mix" ? "个" : KLAB[k]}`;
  const bs = [...new Set(items.filter((it) => it.k === "f" && it.b != null).map((it) => bKey(it.b)))];
  if (bs.length > 1) s += ` · ${fmtB(+(bs[0] ?? 0))}…${fmtB(+(bs[bs.length - 1] ?? 0))}`;
  return s;
}

/** 系定小标签上的帧名：长文件名末尾是编号时只留编号。 */
export function shortFn(fn: string | undefined): string {
  const st = stem(String(fn || ""));
  return st.length > 12 && /\d{4}$/.test(st) ? st.slice(-4) : st;
}

/** 底部操作条上「从哪张到哪张」的文案（原版 selBarUpdate）。 */
export function rangeText(items: readonly GalleryItem[]): string {
  const a = items[0];
  const b = items[items.length - 1];
  if (!a || !b) return "";
  return (
    ` ${dirLabel(a.d)} ${num(a)} → ${b.d !== a.d ? `${dirLabel(b.d)} ` : ""}${num(b)}` +
    `（${fmtT(a.t)} → ${fmtT(b.mt || b.t)}）`
  );
}

/**
 * 存为系列 / 并入系列时的成员表：索引里有的按索引顺序排，旧成员里索引没有的**原样留在
 * 末尾**。原版只保留索引里有的——换过数据根、或者某个目录还没重建时，一次「并入」会把
 * 那些成员静默删掉。
 */
export function mergedMemberIds(
  oldIds: readonly string[],
  add: readonly GalleryItem[],
  items: readonly GalleryItem[],
): string[] {
  const want = new Set([...oldIds, ...add.map((it) => it.id)]);
  const known = items.filter((it) => want.has(it.id)).map((it) => it.id);
  const knownSet = new Set(known);
  return [...known, ...oldIds.filter((id) => !knownSet.has(id))];
}

/** 新系列号：`S` + 毫秒戳的 36 进制（原版同款）。 */
export const newSeriesId = (ts: number): string => `S${ts.toString(36)}`;

/** 已标记页的系列表：按第一个成员的开始时刻排。 */
export function sortSeriesEntries(
  series: Record<string, Series>,
  items: readonly GalleryItem[],
): { sid: string; s: Series; mem: GalleryItem[] }[] {
  const all = Object.entries(series).map(([sid, s]) => ({ sid, s, mem: seriesMembers(s, items) }));
  all.sort((x, y) => (x.mem[0]?.t ?? 0) - (y.mem[0]?.t ?? 0));
  return all;
}
