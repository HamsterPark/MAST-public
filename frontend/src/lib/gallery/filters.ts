// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 筛选（旧版兼容格式 app.js 的 applyF + filterBar 的选项部分）。
//
// 语义逐字照原版，只多一项「文件名前缀」（设计 D15，替代原版按文件名硬编码的
// 样品种类）。最容易被「顺手整理」掉的一条是：
//
//     重复保存的帧默认隐藏 —— 但**自己带标记或在系列里的**照样显示
//
// 没有它，一张被操作员打过 ★ 的重复帧会在他下一次打开这个目录时凭空消失。
// ════════════════════════════════════════════════════════════════════════════

import type { GalleryItem, Mark } from "./types.ts";
import type { ViewName } from "./route.ts";
import { bKey, gridIncomplete, partial, wKey } from "./format.ts";

export type FilterKind = "all" | "f" | "s" | "g";
export type ContrastFilter = "" | "atom" | "half" | "li";
export type MarkFilter = "" | "none" | "any" | "2" | "1" | "-1" | "hidex" | "ser" | "anc";

export interface Filters {
  /** 类型（「全部X」视图里不起作用——那里类型由视图定）。 */
  k: FilterKind;
  /** 偏压分组键（format.bKey）。 */
  b: string;
  /** 帧宽分组键（format.wKey）。 */
  w: string;
  /** 衬度：有原子分辨 / 有超结构 / 有 lock-in dI/dV。 */
  c: ContrastFilter;
  /** 只看扫完（帧有效行 ≥ 95%、网格做完）。 */
  full: boolean;
  /** 隐藏重复保存。 */
  nodup: boolean;
  m: MarkFilter;
  tag: string;
  /** 文件名前缀。 */
  pf: string;
  /** 搜索：文件名或备注，不区分大小写。**不持久化。** */
  q: string;
  sort: "asc" | "desc";
  /** 卡片最小宽度 px。 */
  cw: number;
}

export const DEFAULT_FILTERS: Filters = {
  k: "all",
  b: "",
  w: "",
  c: "",
  full: false,
  nodup: true,
  m: "",
  tag: "",
  pf: "",
  q: "",
  sort: "asc",
  cw: 220,
};

export const CARD_WIDTH_MIN = 150;
export const CARD_WIDTH_MAX = 480;

export interface FilterContext {
  view: ViewName;
  mark: (id: string) => Mark | null | undefined;
  seriesOf: (id: string) => readonly string[];
}

/** 当前视图 + 筛选 → 要显示的条目（保持输入顺序；`sort=desc` 时整体反转）。 */
export function applyFilters(
  base: readonly GalleryItem[],
  F: Filters,
  ctx: FilterContext,
): GalleryItem[] {
  const q = F.q.trim().toLowerCase();
  const out = base.filter((it) => {
    if (ctx.view !== "all" && F.k !== "all" && it.k !== F.k) return false;
    if (F.b && (it.k === "s" || bKey(it.b) !== F.b)) return false;
    if (F.w && (it.k === "s" || wKey(it.w) !== F.w)) return false;
    if (F.pf && (it.pf || "") !== F.pf) return false;
    if (F.c === "atom" && !it.at) return false;
    if (F.c === "half" && !it.hf) return false;
    if (F.c === "li" && !(it.li || it.lic || it.k === "g")) return false;
    if (F.full && (partial(it) || gridIncomplete(it))) return false;
    const m = ctx.mark(it.id);
    if (
      F.nodup &&
      it.dup &&
      ctx.view !== "marked" &&
      ctx.view !== "series" &&
      !m &&
      !ctx.seriesOf(it.id).length
    ) {
      return false;
    }
    if (F.m === "none" && m) return false;
    if (F.m === "any" && !m) return false;
    if ((F.m === "2" || F.m === "1" || F.m === "-1") && !(m && String(m.r) === F.m)) return false;
    if (F.m === "hidex" && m && m.r === -1) return false;
    if (F.m === "ser" && !ctx.seriesOf(it.id).length) return false;
    if (F.m === "anc" && !(m && m.anchor)) return false;
    if (F.tag && !(m && (m.tags || []).includes(F.tag))) return false;
    if (q && !(it.fn.toLowerCase().includes(q) || (m && (m.note || "").toLowerCase().includes(q)))) {
      return false;
    }
    return true;
  });
  if (F.sort === "desc") out.reverse();
  return out;
}

export interface Facets {
  /** `[分组键, 代表值]`，按数值升序。谱不参与（原版如此：谱没有帧宽，偏压是扫描范围）。 */
  biases: [string, number][];
  widths: [string, number][];
  /** `[前缀, 条目数]`，按前缀名排序。 */
  prefixes: [string, number][];
}

export function facets(base: readonly GalleryItem[]): Facets {
  const bs = new Map<string, number>();
  const ws = new Map<string, number>();
  const ps = new Map<string, number>();
  for (const it of base) {
    if (it.k !== "s") {
      if (it.b != null) bs.set(bKey(it.b), it.b);
      if (it.w) ws.set(wKey(it.w), it.w);
    }
    if (it.pf) ps.set(it.pf, (ps.get(it.pf) ?? 0) + 1);
  }
  const byValue = (mp: Map<string, number>) => [...mp.entries()].sort((a, b) => a[1] - b[1]);
  return {
    biases: byValue(bs),
    widths: byValue(ws),
    prefixes: [...ps.entries()].sort((a, b) => a[0].localeCompare(b[0])),
  };
}

/**
 * 选中值在当前列表里已经不存在时退回「全部」（原版 filterBar 开头那几行）。
 *
 * 不退的话：从 0910 带着「偏压 +2.00 V」进了一个根本没有 +2 V 的目录，列表是空的，
 * 而下拉框里又找不到那个值——看起来像这个目录没有数据。
 */
export function sanitizeFilters(F: Filters, fac: Facets, tags: readonly string[]): Filters {
  const has = (arr: [string, number][], key: string) => arr.some(([k]) => k === key);
  const out = { ...F };
  if (out.b && !has(fac.biases, out.b)) out.b = "";
  if (out.w && !has(fac.widths, out.w)) out.w = "";
  if (out.pf && !has(fac.prefixes, out.pf)) out.pf = "";
  if (out.tag && !tags.includes(out.tag)) out.tag = "";
  return out;
}

export const FILTERS_KEY = "mast.gallery.filters";

const KINDS: readonly string[] = ["all", "f", "s", "g"];
const CONTRASTS: readonly string[] = ["", "atom", "half", "li"];
const MARK_FILTERS: readonly string[] = ["", "none", "any", "2", "1", "-1", "hidex", "ser", "anc"];

/**
 * 读 localStorage 里存的筛选。**逐项读时校验**，认不出的值退默认；搜索词永远清空
 * （原版 `F.q = ''`：带着上次的搜索词回来，列表莫名其妙少了一大半）。
 */
export function parseStoredFilters(raw: string | null | undefined): Filters {
  const F: Filters = { ...DEFAULT_FILTERS };
  if (!raw) return F;
  let o: Record<string, unknown>;
  try {
    const v: unknown = JSON.parse(raw);
    if (!v || typeof v !== "object") return F;
    o = v as Record<string, unknown>;
  } catch {
    return F;
  }
  const str = (x: unknown) => (typeof x === "string" ? x : null);
  if (KINDS.includes(str(o.k) ?? "\0")) F.k = o.k as FilterKind;
  if (str(o.b) != null) F.b = o.b as string;
  if (str(o.w) != null) F.w = o.w as string;
  if (CONTRASTS.includes(str(o.c) ?? "\0")) F.c = o.c as ContrastFilter;
  if (typeof o.full === "boolean") F.full = o.full;
  if (typeof o.nodup === "boolean") F.nodup = o.nodup;
  if (MARK_FILTERS.includes(str(o.m) ?? "\0")) F.m = o.m as MarkFilter;
  if (str(o.tag) != null) F.tag = o.tag as string;
  if (str(o.pf) != null) F.pf = o.pf as string;
  if (o.sort === "asc" || o.sort === "desc") F.sort = o.sort;
  if (typeof o.cw === "number" && Number.isFinite(o.cw)) {
    F.cw = Math.max(CARD_WIDTH_MIN, Math.min(CARD_WIDTH_MAX, Math.round(o.cw)));
  }
  return F;
}

export function serializeFilters(F: Filters): string {
  return JSON.stringify({ ...F, q: "" });
}
