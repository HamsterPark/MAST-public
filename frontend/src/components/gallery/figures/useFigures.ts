//
// 状态键不以 ["gallery", "figures"] 开头：「刷新产物列表」的 invalidate 按前缀匹配，
// 不该顺带把任务状态也打回重拉。

import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

export const FIGURE_KEYS = {
  list: ["gallery", "figures"] as const,
  status: ["gallery", "figure-status"] as const,
};

/** 出图产物列表（纯读）。任务结束时由 FigureJobChip 让它失效重拉。 */
export function useGalleryFigures() {
  return useQuery({
    queryKey: FIGURE_KEYS.list,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/gallery/figures");
      if (error) throw error;
      return data;
    },
    staleTime: 30_000,
  });
}

/** 出图任务槽：在跑时 1.5 s 一次看进度，否则 15 s 一次（任务可能是别的标签页发起的）。 */
export function useFigureStatus() {
  return useQuery({
    queryKey: FIGURE_KEYS.status,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/gallery/figures/status");
      if (error) throw error;
      return data;
    },
    refetchInterval: (q) => (q.state.data?.running ? 1500 : 15000),
  });
}
