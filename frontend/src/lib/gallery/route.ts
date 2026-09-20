// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 视图 ↔ URL。
//
// 旧版兼容格式用 location.hash（`#d=20010910`、`#s=<系列号>`、`#v=all&k=f`）。MAST 版挂在
// 真路由段 `/records/gallery` 下，视图放进 query：
//
//     ?v=dirs | dir&d=<目录键> | all&k=f|s|g | new | marked | series&s=<系列号> | figures | setup
//
// `figures` 是出图页（设计 §10，D22）：从标记与系列生成的正式图。
//
// 深链可收藏、前进后退可用。解析**读时校验**：参数缺了或值不认识就回 null，由调用方
// 退回「上次停的视图 / 目录总览」——一个认不出的地址渲染成空白页，用户报上来的会是
// 「图库坏了」而不是「链接过期了」（与 lib/stickyTab.ts 同一条纪律）。
// ════════════════════════════════════════════════════════════════════════════

import type { Kind } from "./types.ts";

export type GalleryView =
  | { v: "dirs" }
  | { v: "dir"; d: string }
  | { v: "all"; k: Kind }
  | { v: "new" }
  | { v: "marked" }
  | { v: "series"; s: string }
  | { v: "figures" }
  | { v: "setup" };

export type ViewName = GalleryView["v"];

const isKind = (k: string | null): k is Kind => k === "f" || k === "s" || k === "g";

/** query → 视图；认不出就 null。 */
export function parseView(sp: URLSearchParams): GalleryView | null {
  switch (sp.get("v")) {
    case "dirs":
      return { v: "dirs" };
    case "dir": {
      const d = sp.get("d");
      return d ? { v: "dir", d } : null;
    }
    case "all": {
      const k = sp.get("k");
      // 原版 `#v=all` 不带 k 时默认看帧。
      return { v: "all", k: isKind(k) ? k : "f" };
    }
    case "new":
      return { v: "new" };
    case "marked":
      return { v: "marked" };
    case "series": {
      const s = sp.get("s");
      return s ? { v: "series", s } : null;
    }
    case "figures":
      return { v: "figures" };
    case "setup":
      return { v: "setup" };
    default:
      return null;
  }
}

/** 视图 → query 字符串（不带 `?`）。 */
export function viewSearch(view: GalleryView): string {
  const sp = new URLSearchParams();
  sp.set("v", view.v);
  if (view.v === "dir") sp.set("d", view.d);
  if (view.v === "all") sp.set("k", view.k);
  if (view.v === "series") sp.set("s", view.s);
  return sp.toString();
}

/** 视图身份：变了就清空选中（原版 route() 里 `before !== viewKey(VIEW)`）。 */
export function viewKey(view: GalleryView): string {
  if (view.v === "dir") return `dir:${view.d}`;
  if (view.v === "all") return `all:${view.k}`;
  if (view.v === "series") return `series:${view.s}`;
  return view.v;
}

/** 顶栏哪一项高亮；某目录 / 最新一批 / 系列页不高亮任何一项（与原版一致）。 */
export function navId(view: GalleryView): string {
  if (view.v === "all") return `all-${view.k}`;
  if (view.v === "dirs" || view.v === "marked" || view.v === "figures" || view.v === "setup") return view.v;
  return "";
}

/** 列表类视图（有卡片网格、有选中操作条）。 */
export function isListView(view: GalleryView): boolean {
  return view.v === "dir" || view.v === "all" || view.v === "new" || view.v === "series";
}
