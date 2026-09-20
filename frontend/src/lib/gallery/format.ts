// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 文字格式（旧版兼容格式 app.js 顶部那一串小函数的逐个移植）。
//
// 两处与原版不同，都写在函数上：
//   * 目录键不再是 `20010910` 这样的日期名，而是 `<根名>/<相对父目录>`（设计 D5）。
//     显示时取最后一段；日期形的照原版排成 YYYY-MM-DD / MMDD。
//   * 自动标签去掉了原版按文件名硬编码的样品种类（设计 T10），多了「同一次采集」
//     「副本」两种（设计 D15）。
//
// 时间一律按**浏览器本地时区**显示——原版如此，仪器与操作员通常在同一时区。
// ════════════════════════════════════════════════════════════════════════════

import type { GalleryItem, Kind } from "./types.ts";

export const KLAB: Record<Kind, string> = { f: "帧", s: "谱", g: "网格谱" };
export const RSYM: Record<string, string> = { "2": "★", "1": "✓", "-1": "✗" };
export const RLAB: Record<string, string> = { "2": "★ 重点", "1": "✓ 可用", "-1": "✗ 排除" };

export const pad2 = (n: number): string => String(n).padStart(2, "0");

/** `MM-DD HH:MM`；没有时刻给 `?`。 */
export function fmtT(t?: number | null): string {
  if (!t) return "?";
  const d = new Date(t * 1000);
  return `${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

/** `HH:MM:SS`。 */
export function fmtTs(t?: number | null): string {
  if (!t) return "?";
  const d = new Date(t * 1000);
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

/** `YYYY-MM-DD HH:MM:SS`，标记的 `t` 字段用它。 */
export function nowStr(d: Date = new Date()): string {
  return (
    `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ` +
    `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
  );
}

/** 正负号用真的减号（U+2212），与原版一致——等宽字体里 `-` 太短看不见。 */
const sgn = (v: number): string => (v >= 0 ? "+" : "−");

/** 偏压：|b| < 0.1 V 用 mV（三位有效数字），否则 V 两位小数。 */
export function fmtB(b?: number | null): string {
  if (b == null) return "?";
  const a = Math.abs(b);
  return a < 0.1 ? `${sgn(b)}${+(a * 1000).toPrecision(3)} mV` : `${sgn(b)}${a.toFixed(2)} V`;
}

/** 目录卡片上那一串偏压用的紧凑写法（原版 renderDays 里的内联表达式）。 */
export function fmtBShort(b: number): string {
  return Math.abs(b) < 0.1 ? fmtB(b).replace(" ", "") : `${sgn(b)}${+Math.abs(b).toFixed(2)}`;
}

export const fmtV = (v?: number | null): string =>
  v == null ? "?" : `${sgn(v)}${Math.abs(v).toFixed(2)}`;

/** 偏压分组键：小偏压保留到 mV。筛选下拉与目录汇总共用这一个，别另写。 */
export const bKey = (b?: number | null): string =>
  b == null ? "" : Math.abs(b) < 0.1 ? b.toFixed(3) : b.toFixed(2);

/** 帧宽分组键：0.1 nm。 */
export const wKey = (w?: number | null): string => (w ? String(+w.toFixed(1)) : "");

/** 帧宽显示：四位有效数字。 */
export const nm = (w?: number | null): number => +(Number(w) || 0).toPrecision(4);

export const stem = (fn: string): string => fn.replace(/\.[^.]+$/, "");

/** 卡片上的编号：帧取文件名最后四位，谱与网格取整个文件名主干（原版 num）。 */
export function num(it: Pick<GalleryItem, "k" | "fn">): string {
  return it.k === "f" ? stem(it.fn).slice(-4) : stem(it.fn);
}

export function lastSeg(d: string): string {
  const parts = d.split("/");
  return parts[parts.length - 1] || d;
}

export const isDateSeg = (s: string): boolean => /^\d{8}$/.test(s);

/** 目录标题：`SPM/2001/200109/20010910` → `2001-09-10`；非日期形的目录名原样。 */
export function dirLabel(d: string): string {
  const s = lastSeg(d);
  return isDateSeg(s) ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : s;
}

/** 卡片上的目录小标签：日期形取 MMDD（原版 `it.d.slice(4)`），否则目录名。 */
export function dirShort(d: string): string {
  const s = lastSeg(d);
  return isDateSeg(s) ? s.slice(4) : s;
}

/** 绝对路径的父目录（Windows 与 POSIX 分隔符都认）。 */
export function dirnameOf(p: string): string {
  return p.replace(/[\\/][^\\/]*$/, "");
}

/** 复制路径用：优先条目自带的绝对路径。 */
export const fullPath = (it: Pick<GalleryItem, "p" | "id">): string => it.p || it.id;

/** 帧没扫完：整行有效的行数不到 95%（原版 partial）。 */
export function partial(it: GalleryItem): boolean {
  return it.k === "f" && !!it.rall && it.rows != null && it.rows < 0.95 * it.rall;
}

/** 网格谱没做完。 */
export function gridIncomplete(it: GalleryItem): boolean {
  return it.k === "g" && (it.have ?? 0) < (it.gx ?? 0) * (it.gy ?? 0);
}

/** 秒数 → `N 秒` / `N 分 M 秒` / `N 小时 M 分`（原版 context.js 的 dur）。 */
export function dur(s: number): string {
  const x = Math.round(Math.abs(s));
  if (x < 60) return `${x} 秒`;
  if (x < 3600) return `${Math.floor(x / 60)} 分 ${x % 60} 秒`;
  return `${Math.floor(x / 3600)} 小时 ${Math.floor((x % 3600) / 60)} 分`;
}

const fix2 = (v?: number | null): string => (Number(v) || 0).toFixed(2);
const round0 = (v?: number | null): number => Math.round(Number(v) || 0);

/** 一行参数摘要。写进标记的 `meta`，也用在大图侧栏与标记表里（原版 meta）。 */
export function meta(it: GalleryItem): string {
  if (it.k === "f") {
    return `${fmtT(it.t)} · ${nm(it.w)} nm · ${fmtB(it.b)} · ${round0(it.sp)} pA · ${it.nx}px`;
  }
  if (it.k === "s" && it.ex) return `${fmtT(it.t)} · ${it.ex} · ${it.n} 点（不是偏压谱）`;
  if (it.k === "s") {
    return (
      `${fmtT(it.t)} · 谱 ${it.n} 点 · ${fmtV(it.v0)}…${fmtV(it.v1)} V · ` +
      `Zoff ${round0(it.zo)} pm · (${fix2(it.x)}, ${fix2(it.y)}) nm`
    );
  }
  return `${fmtT(it.t)} · 网格 ${it.gx}×${it.gy} · ${nm(it.w)} nm · ${it.n} 点 ${fmtV(it.v0)}→${fmtV(it.v1)} V`;
}

/** 卡片说明：`num` 加粗、`when` 跟在后面，`lines` 各占一行（原版 capHtml 的文字部分）。 */
export function caption(it: GalleryItem): { num: string; when: string; lines: string[] } {
  const n = num(it);
  if (it.k === "f") {
    return {
      num: n,
      when: fmtT(it.t),
      lines: [`${nm(it.w)} nm · ${fmtB(it.b)} · ${round0(it.sp)} pA · ${it.nx}px`],
    };
  }
  if (it.k === "s" && it.ex) {
    return { num: n, when: fmtT(it.t), lines: [`${it.ex} · ${it.n} 点（不是偏压谱）`] };
  }
  if (it.k === "s") {
    return {
      num: n,
      when: fmtT(it.t),
      lines: [
        `${it.n} 点 · ${fmtV(it.v0)}…${fmtV(it.v1)} V · Zoff ${round0(it.zo)} pm` +
          ((it.sw ?? 0) > 1 ? ` · ${it.sw} sw` : ""),
        `(${fix2(it.x)}, ${fix2(it.y)}) nm`,
      ],
    };
  }
  return {
    num: n,
    when: `${fmtT(it.t)} → ${fmtT(it.t1)}`,
    lines: [
      `${it.gx}×${it.gy} · ${nm(it.w)} nm · ${it.n} 点 ${fmtV(it.v0)}→${fmtV(it.v1)} V · ` +
        `Zoff ${round0(it.zo)} pm · ${fmtB(it.b)}/${round0(it.sp)} pA`,
    ],
  };
}

export type AutoTagKind =
  | "new" | "atom" | "half" | "part" | "dup" | "seg" | "copies" | "gridpart" | "li";

export interface AutoTag {
  kind: AutoTagKind;
  text: string;
  title?: string;
}

/**
 * 卡片上的自动标签（原版 autoTags）。
 *
 * `numOf(id)` 把「重复保存 = 哪一张」的原帧 id 换成编号；原帧不在索引里时回 `?`。
 * 原版的「原子 N」数字是 (1,0) 峰 / 本底；MAST 版是原子相判据的**角向集中度**
 * （设计 D6），所以 title 要说清是哪个量，否则同样一个「原子 300」两边意思不同。
 */
export function autoTags(
  it: GalleryItem,
  lastBatch: string,
  numOf: (id: string) => string | null,
): AutoTag[] {
  const out: AutoTag[] = [];
  if (lastBatch && it.ad === lastBatch) out.push({ kind: "new", text: "新" });
  if (it.at) {
    out.push({
      kind: "atom",
      text: `原子 ×${Math.round(it.at)}`,
      title: "原子相判据的角向集中度（离散布拉格点 vs 弥散环；通过阈值 20）",
    });
  }
  if (it.hf) {
    out.push({
      kind: "half",
      text: `超结构 ${it.hl || ""} ×${it.hf}`.replace("  ", " "),
      title: "分数序位置的相干幅值 / 空白对照最大值（> 1.5 判为存在）",
    });
  }
  if (partial(it)) out.push({ kind: "part", text: `未扫完 ${it.rows}/${it.rall}` });
  if (it.dup) {
    const n = numOf(it.dup);
    out.push({
      kind: "dup",
      text: `重复保存 = ${n ?? "?"}`,
      title: `Z 数据与 ${it.dup} 逐字节相同（REC 时刻相同、之后又存了一次）`,
    });
  }
  if ((it.seg ?? 0) >= 2) {
    out.push({
      kind: "seg",
      text: `同一次采集 ×${it.seg}`,
      title: "REC 时刻、视野、中心、转角都相同的不同文件（自动保存 / 手动保存 / 中途停）",
    });
  }
  if ((it.cp ?? 0) > 1) {
    out.push({ kind: "copies", text: `副本 ×${it.cp}`, title: "字节级相同的副本（自动拷贝进实验文件夹）" });
  }
  if (gridIncomplete(it)) {
    out.push({ kind: "gridpart", text: `未完成 ${it.have}/${(it.gx ?? 0) * (it.gy ?? 0)}` });
  }
  if (it.k === "s" && it.lic) out.push({ kind: "li", text: "LI" });
  return out;
}
