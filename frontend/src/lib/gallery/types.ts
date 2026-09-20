// ════════════════════════════════════════════════════════════════════════════
//
// 线上形状的真源是 `components["schemas"]["Gallery*"]`（Pydantic → openapi →
// schema.d.ts）。条目、索引、构建状态、配置直接用生成的类型。
//
// **标记另立一层宽松类型**，理由只有一个：生成的 `GalleryMark` 把带默认值的字段
// 标成了必填（`del` / `meta` / `k` / `t` …），而服务端存下来的标记只含客户端真正
// 发过的键（路由里 `model_dump(exclude_unset=True)`）。照生成类型写，每读一条
// 存量标记都得假装它有 `del`，而删除条目 `{del, ts}` 又根本不是一个 Mark。
// 发请求时在调用点做一次显式转换（见 components/gallery/marksStore.ts）。
//
// 无 React、无值 import ⇒ node --test 可直接加载（`import type` 会被整句擦除）。
// ════════════════════════════════════════════════════════════════════════════

import type { components } from "@/api/schema";

export type GalleryItem = components["schemas"]["GalleryItem"];
export type GalleryIndexDoc = components["schemas"]["GalleryIndex"];
export type GalleryBuildStatus = components["schemas"]["GalleryBuildStatus"];
export type GalleryConfig = components["schemas"]["GalleryConfigResponse"];
export type GalleryRoot = components["schemas"]["GalleryRoot"];

export type Kind = "f" | "s" | "g";

/** 谱（或一个系列）的位置系于哪一张帧。字段照原版 context.js 的 anchorTo。 */
export interface Anchor {
  id: string;
  fn: string;
  /** "prev" | "next"：帧在谱之前还是之后。 */
  rel: string;
  /** 帧相对谱的秒数（整数）。 */
  dt: number;
  desc: string;
  /** 帧内分数坐标：u 自左，v 自上。系列锚点没有。 */
  u?: number;
  v?: number;
  inside?: boolean;
}

export interface Mark {
  /** 2 重点 / 1 可用 / -1 排除 / 0 无。 */
  r: number;
  tags: string[];
  note: string;
  t?: string;
  ts?: number;
  tt?: number | null;
  k?: string;
  meta?: string;
  anchor?: Anchor | null;
}

export interface Series {
  name: string;
  /** f / s / g / mix */
  k: string;
  ids: string[];
  r: number;
  tags: string[];
  note: string;
  t?: string;
  ts?: number;
  anchor?: Anchor | null;
}

export interface DirMark {
  done: boolean;
  note: string;
  ts?: number;
}

/** patch 里「删掉这一条」的写法。服务端据 ts 记墓碑，挡住迟到的旧改动。 */
export interface Tombstone {
  del: true;
  ts: number;
}

export interface MarksDoc {
  version?: number;
  rev?: number;
  updated?: string;
  tags: string[];
  items: Record<string, Mark>;
  series: Record<string, Series>;
  /** 按目录键存。名字沿用旧版兼容格式的 days，文件格式可往返。 */
  days: Record<string, DirMark>;
}

export interface Patch {
  items: Record<string, Mark | Tombstone>;
  series: Record<string, Series | Tombstone>;
  days: Record<string, DirMark>;
  tags: string[] | null;
}
