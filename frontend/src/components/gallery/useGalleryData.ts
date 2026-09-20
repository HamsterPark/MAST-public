// 数据图库的服务端数据：索引 / 构建状态 / 配置（react-query），以及从索引派生的模型。
//
// 索引一次拉全（设计 D9：几千条、gzip 后几百 KB），之后筛选、目录汇总、前后帧查找都在
// 浏览器里做——切筛选不该有网络往返。索引只在构建完成时失效重拉（GalleryApp 盯着
// status 的 running → 非 running 边沿），平时 staleTime 无穷大。

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { GalleryItem, GalleryRoot } from "@/lib/gallery/types";
import { num } from "@/lib/gallery/format";
import { framesTimeline } from "@/lib/gallery/context";

export const GALLERY_KEYS = {
  index: ["gallery", "index"] as const,
  status: ["gallery", "status"] as const,
  config: ["gallery", "config"] as const,
};

export function useGalleryIndex() {
  return useQuery({
    queryKey: GALLERY_KEYS.index,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/gallery/index");
      if (error) throw error;
      return data;
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

export function useGalleryStatus() {
  return useQuery({
    queryKey: GALLERY_KEYS.status,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/gallery/status");
      if (error) throw error;
      return data;
    },
    // 构建在跑时 1.5 s 一次看进度，否则 15 s 一次（别的标签页、CLI 也可能在构建）。
    refetchInterval: (q) => (q.state.data?.running ? 1500 : 15000),
  });
}

export function useGalleryConfig() {
  return useQuery({
    queryKey: GALLERY_KEYS.config,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/gallery/config");
      if (error) throw error;
      return data;
    },
  });
}

export interface GalleryModel {
  /** 按 (目录, 开始时刻, 文件名) 排好的全部条目——原版 data.js 的顺序。 */
  items: GalleryItem[];
  byId: Map<string, GalleryItem>;
  /** 参与前后帧查找的帧（有保存时刻、非重复保存，按保存时刻升序）。 */
  frames: GalleryItem[];
  lastBatch: string;
  generated: string;
  built: boolean;
  roots: GalleryRoot[];
  counts: { f: number; s: number; g: number; dup: number };
  numOf: (id: string) => string | null;
}

export function buildModel(data: {
  items?: GalleryItem[];
  last_batch?: string;
  generated?: string;
  built?: boolean;
  roots?: GalleryRoot[];
}): GalleryModel {
  const items = [...(data.items ?? [])].sort(
    (a, b) =>
      (a.d < b.d ? -1 : a.d > b.d ? 1 : 0) ||
      (a.t || a.mt || 0) - (b.t || b.mt || 0) ||
      (a.fn < b.fn ? -1 : a.fn > b.fn ? 1 : 0),
  );
  const byId = new Map(items.map((it) => [it.id, it]));
  const counts = { f: 0, s: 0, g: 0, dup: 0 };
  for (const it of items) {
    counts[it.k]++;
    if (it.dup) counts.dup++;
  }
  return {
    items,
    byId,
    frames: framesTimeline(items),
    lastBatch: data.last_batch ?? "",
    generated: data.generated ?? "",
    built: !!data.built,
    roots: data.roots ?? [],
    counts,
    numOf: (id) => {
      const it = byId.get(id);
      return it ? num(it) : null;
    },
  };
}

/** 索引查询 + 派生模型（模型只在索引数据换了时重算）。 */
export function useGalleryModel() {
  const q = useGalleryIndex();
  const model = useMemo(() => (q.data ? buildModel(q.data) : null), [q.data]);
  return { q, model };
}
