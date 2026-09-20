// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 谱 ↔ 前后帧对照的几何与时间（旧版兼容格式 context.js 的纯函数部分）。
//
// 时间（设计 T22）：帧的开始 = 头里 REC 时刻，保存 = 文件 mtime（仪器写盘时刻）；
// REC 晚于 mtime（中途停过）时按 `mt − ACQ × 已扫行比例` 估。谱的开始 = Start time，
// 结束 = mtime；网格 = Start / End time。
//
// 坐标：SCAN_ANGLE 为正 = 扫描框相对压电坐标顺时针转。
// 帧的 y 向上，帧内分数坐标 v 自上沿量起。
//
// 缩略图几何（设计 T5）：帧缩略图 = 定向后第 [r0, r1) 行 + 底部 STRIP_PX 信息条。
// 原版用 `sd === 'u' ? rall − rows : 0` 猜有效行的起点，有效行不从扫描起点开始时
// 十字圈会画偏；索引现在显式给 r0/r1，这里只用它们。
// ════════════════════════════════════════════════════════════════════════════

import type { Anchor, GalleryItem } from "./types.ts";
import { dur, fmtTs } from "./format.ts";

/**
 * 帧缩略图底部信息条的像素高度。**必须**与 MASTv2/mast/gallery/render.py 的
 * `STRIP_PX` 相等——frontend/test/galleryContext.test.ts 读那个文件的源码核对。
 */
export const STRIP_PX = 34;

export interface UV {
  u: number;
  v: number;
  /** 系列页里画在点旁边的编号。 */
  lab?: string;
}

/** 参与前后帧查找的帧：有保存时刻、不是重复保存；按保存时刻升序。 */
export function framesTimeline(items: readonly GalleryItem[]): GalleryItem[] {
  return items.filter((it) => it.k === "f" && it.mt && !it.dup).sort((a, b) => (a.mt ?? 0) - (b.mt ?? 0));
}

/** 帧开始扫描的时刻。很久以后才保存的帧也按 REC 算，才看得出它跨着谱。 */
export function fStart(f: GalleryItem): number {
  const mt = f.mt ?? 0;
  if (f.t && f.t <= mt) return f.t;
  const rall = f.rall || 1;
  return mt - (f.acq || 0) * Math.min(1, (f.rows || f.rall || 1) / rall);
}

export const tBeg = (it: GalleryItem): number => it.t ?? it.mt ?? 0;

export function tEnd(it: GalleryItem): number {
  if (it.k === "g") return it.t1 || it.mt || it.t || 0;
  return it.mt || it.t || 0;
}

/** 保存时刻 ≤ t 的最后一张帧的下标；没有回 -1。 */
export function lastBefore(frames: readonly GalleryItem[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if ((frames[m]?.mt ?? 0) <= t) {
      ans = m;
      lo = m + 1;
    } else hi = m - 1;
  }
  return ans;
}

/** 谱结束后**才开始扫描**的第一张帧的下标（跳过跨着谱的帧）；没有回 frames.length。 */
export function firstStartAfter(frames: readonly GalleryItem[], t: number): number {
  for (let i = Math.max(0, lastBefore(frames, t - 1)); i < frames.length; i++) {
    const f = frames[i];
    if (f && fStart(f) >= t - 1) return i;
  }
  return frames.length;
}

export interface Relation {
  rel: "prev" | "next";
  /** 秒。prev：帧保存 − 谱开始（负数）；next：帧开始 − 谱结束。 */
  dt: number;
  desc: string;
}

/** 这张帧相对谱（或系列）的时间关系与说明文字。 */
export function relation(f: GalleryItem, t0: number, t1: number, noun = "谱"): Relation {
  const mt = f.mt ?? 0;
  if (mt <= t0 + 1) return { rel: "prev", dt: mt - t0, desc: `${noun}开始前 ${dur(t0 - mt)}保存` };
  const st = fStart(f);
  if (st >= t1 - 1) return { rel: "next", dt: st - t1, desc: `${noun}结束后 ${dur(st - t1)}开始扫描` };
  return {
    rel: mt < t1 ? "prev" : "next",
    dt: mt - t0,
    desc: `与${noun}时间交叠：${fmtTs(st)} 开始、${fmtTs(mt)} 保存`,
  };
}

/** 实验室坐标（nm）→ 帧内分数坐标（u 自左，v 自上）。正角 = 帧顺时针转。 */
export function toFrame(f: GalleryItem, x: number, y: number): UV {
  const th = ((f.ang || 0) * Math.PI) / 180;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const dx = x - (f.cx ?? 0);
  const dy = y - (f.cy ?? 0);
  const xp = dx * c - dy * s;
  const yp = dx * s + dy * c;
  const w = f.w || 1;
  const h = f.hn || (w * (f.ny || 1)) / (f.nx || 1);
  return { u: 0.5 + xp / w, v: 0.5 - yp / h };
}

/** 网格的四个角（实验室坐标 nm）。用 w × hn——原版用 w × w，非方网格会画错。 */
export function gridCorners(g: GalleryItem): [number, number][] {
  const th = ((g.ang || 0) * Math.PI) / 180;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const hw = (g.w || 0) / 2;
  const hh = (g.hn || g.w || 0) / 2;
  const cx = g.cx ?? 0;
  const cy = g.cy ?? 0;
  const local: [number, number][] = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  return local.map(([a, b]): [number, number] => [cx + a * c + b * s, cy - a * s + b * c]);
}

export const inside = (p: UV): boolean => p.u >= 0 && p.u <= 1 && p.v >= 0 && p.v <= 1;

/**
 * 帧内分数坐标 → 缩略图像素 `[x, y, 图像区高]`。
 *
 * 图像区高 = 图高 − STRIP_PX；缩略图只画了定向后第 [r0, r1) 行，所以
 * `y = (v·rall − r0) / (r1 − r0) · 图像区高`。
 */
export function imgXY(f: GalleryItem, p: UV, natW: number, natH: number): [number, number, number] {
  const hh = natH - STRIP_PX;
  const rall = f.rall || f.ny || 1;
  const r0 = f.r0 ?? 0;
  const r1 = f.r1 ?? rall;
  const shown = Math.max(1, r1 - r0);
  return [p.u * natW, ((p.v * rall - r0) / shown) * hh, hh];
}

/** 一条谱 / 网格在某张帧上的位置：谱是一个点，网格是中心点 + 四角多边形。 */
export function itemGeometry(it: GalleryItem, f: GalleryItem): { pts: UV[]; poly: UV[] | null } {
  if (it.k === "g") {
    return {
      pts: [toFrame(f, it.cx ?? 0, it.cy ?? 0)],
      poly: gridCorners(it).map(([x, y]) => toFrame(f, x, y)),
    };
  }
  return { pts: [toFrame(f, it.x ?? 0, it.y ?? 0)], poly: null };
}

/** 把这条谱系于这张帧时要存的锚点（字段照原版 anchorTo）。 */
export function anchorOf(it: GalleryItem, f: GalleryItem): Anchor {
  const r = relation(f, tBeg(it), tEnd(it));
  const p = itemGeometry(it, f).pts[0] ?? { u: 0.5, v: 0.5 };
  return {
    id: f.id,
    fn: f.fn,
    rel: r.rel,
    dt: Math.round(r.dt),
    desc: r.desc,
    u: +p.u.toFixed(4),
    v: +p.v.toFixed(4),
    inside: inside(p),
  };
}

/** 整个系列系于一张帧时的锚点（没有 u/v：系列不是一个点）。 */
export function seriesAnchorOf(f: GalleryItem, t0: number, t1: number): Anchor {
  const r = relation(f, t0, t1, "系列");
  return { id: f.id, fn: f.fn, rel: r.rel, dt: Math.round(r.dt), desc: r.desc };
}

/**
 * 打开对照时两侧先摆哪两张帧：前一张 = 开始前最后保存的，后一张 = 结束后第一张开始的；
 * 已系定的帧若不是最近那张，也把它摆到对应的一侧（原版 ctxStage / ctxSeries 开头）。
 */
export function initialNeighbours(
  frames: readonly GalleryItem[],
  t0: number,
  t1: number,
  anchorId?: string | null,
): { prev: number; next: number } {
  const out = { prev: lastBefore(frames, t0 + 1), next: firstStartAfter(frames, t1) };
  if (anchorId) {
    const i = frames.findIndex((f) => f.id === anchorId);
    const f = frames[i];
    if (f) {
      if ((f.mt ?? 0) <= t0 + 1) out.prev = i;
      else out.next = i;
    }
  }
  return out;
}

/** 系列的时间跨度：成员里谱与网格的最早开始、最晚结束。 */
export function spanOf(members: readonly GalleryItem[]): { t0: number; t1: number } {
  const sp = members.filter((it) => it.k === "s" || it.k === "g");
  if (!sp.length) return { t0: 0, t1: 0 };
  return { t0: Math.min(...sp.map(tBeg)), t1: Math.max(...sp.map(tEnd)) };
}

export interface OverlayPoint {
  x: number;
  y: number;
  out: boolean;
  label: string | null;
}

export interface OverlayShapes {
  /** 网格范围多边形（y 夹在图像区内）。 */
  polygon: string | null;
  /** 多点时按顺序连起来的折线（不夹）。 */
  polyline: string | null;
  points: OverlayPoint[];
  /** 只有一个点：画十字圈而不是编号小圆点。 */
  single: boolean;
  width: number;
  height: number;
  imageHeight: number;
}

/**
 * SVG 标记层要画的东西（原版 drawMarks 的计算部分）。视野外的点夹到边上 4 px 内并
 * 标 `out`（组件画成黄色虚线）；多点时首尾两点总有编号，20 个以内全部编号。
 */
export function overlayShapes(
  f: GalleryItem,
  pts: readonly UV[],
  poly: readonly UV[] | null,
  natW: number,
  natH: number,
): OverlayShapes {
  const hh = natH - STRIP_PX;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  let polygon: string | null = null;
  if (poly && poly.length) {
    polygon = poly
      .map((q) => {
        const [x, y] = imgXY(f, q, natW, natH);
        return `${x.toFixed(1)},${clamp(y, 0, hh).toFixed(1)}`;
      })
      .join(" ");
  }
  const many = pts.length > 1;
  const polyline = many
    ? pts
        .map((q) => {
          const [x, y] = imgXY(f, q, natW, natH);
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ")
    : null;
  const points = pts.map((q, i) => {
    const [x0, y0] = imgXY(f, q, natW, natH);
    const out = x0 < 0 || x0 > natW || y0 < 0 || y0 > hh;
    const labelled = many && (i === 0 || i === pts.length - 1 || pts.length <= 20);
    return {
      x: clamp(x0, 4, natW - 4),
      y: clamp(y0, 4, hh - 4),
      out,
      label: labelled ? q.lab || String(i + 1) : null,
    };
  });
  return { polygon, polyline, points, single: !many, width: natW, height: natH, imageHeight: hh };
}
