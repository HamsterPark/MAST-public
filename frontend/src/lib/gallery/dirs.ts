// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 目录总览的汇总（旧版兼容格式 app.js renderDays 的计数部分）。
//
// 原版的「日期」在 MAST 里是「目录」（设计 D5）：目录名不一定是日期，所以卡片按
// 目录内**最晚时刻**倒序排，而不是按名字。
// ════════════════════════════════════════════════════════════════════════════

import type { GalleryItem, Mark } from "./types.ts";
import { bKey, wKey } from "./format.ts";

export interface DirSummary {
  d: string;
  f: number;
  s: number;
  g: number;
  /** 重复保存的帧数。 */
  dup: number;
  /** 原子分辨帧数（at > 0）。 */
  atom: number;
  /** 有超结构的帧数（hf > 0）。 */
  half: number;
  /** 属于最新一批的条目数。 */
  nw: number;
  /** 已标记条目数、其中 ★ 的数目。 */
  mk: number;
  star: number;
  /** 最早开始 / 最晚结束（epoch 秒）；没有时刻时 t0 = Infinity、t1 = 0。 */
  t0: number;
  t1: number;
  /** 帧的偏压（按 bKey 去重）与帧宽（按 wKey 去重），各自升序。 */
  biases: number[];
  widths: number[];
  /** 文件名前缀，按条目数从多到少。 */
  prefixes: string[];
  /** 这个目录里的一条（给「原始目录」取绝对路径用）。 */
  sample: GalleryItem;
}

export function summariseDirs(
  items: readonly GalleryItem[],
  lastBatch: string,
  mark: (id: string) => Mark | null | undefined,
): DirSummary[] {
  const acc = new Map<
    string,
    DirSummary & { _b: Map<string, number>; _w: Map<string, number>; _p: Map<string, number> }
  >();
  for (const it of items) {
    let s = acc.get(it.d);
    if (!s) {
      s = {
        d: it.d, f: 0, s: 0, g: 0, dup: 0, atom: 0, half: 0, nw: 0, mk: 0, star: 0,
        t0: Infinity, t1: 0, biases: [], widths: [], prefixes: [], sample: it,
        _b: new Map(), _w: new Map(), _p: new Map(),
      };
      acc.set(it.d, s);
    }
    s[it.k]++;
    if (it.dup) s.dup++;
    if (it.at) s.atom++;
    if (it.hf) s.half++;
    if (lastBatch && it.ad === lastBatch) s.nw++;
    const m = mark(it.id);
    if (m) {
      s.mk++;
      if (m.r === 2) s.star++;
    }
    if (it.t) {
      s.t0 = Math.min(s.t0, it.t);
      s.t1 = Math.max(s.t1, it.t1 || it.mt || it.t);
    }
    if (it.k === "f") {
      if (it.b != null) s._b.set(bKey(it.b), it.b);
      if (it.w) s._w.set(wKey(it.w), it.w);
    }
    if (it.pf) s._p.set(it.pf, (s._p.get(it.pf) ?? 0) + 1);
  }
  const out: DirSummary[] = [];
  for (const s of acc.values()) {
    const { _b, _w, _p, ...rest } = s;
    out.push({
      ...rest,
      biases: [..._b.values()].sort((a, b) => a - b),
      widths: [..._w.values()].sort((a, b) => a - b),
      prefixes: [..._p.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([p]) => p),
    });
  }
  out.sort((a, b) => b.t1 - a.t1 || b.d.localeCompare(a.d));
  return out;
}

/** 升序取前 `max` 个、空格连接，多了加「 …」（原版 renderDays 的 short）。 */
export function shortList(values: readonly number[], fmt: (v: number) => string, max = 9): string {
  const v = [...values].sort((a, b) => a - b);
  return v.slice(0, max).map(fmt).join(" ") + (v.length > max ? " …" : "");
}

/** 最新一批的条目。 */
export function newBatch(items: readonly GalleryItem[], lastBatch: string): GalleryItem[] {
  return lastBatch ? items.filter((it) => it.ad === lastBatch) : [];
}
