import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

// 粗动大地图的取数。**按需**，不进 3 秒轮询：它只在换区前后变化，而每次都要读一遍
// marker 表并派生站点 —— 轮询它是为一个几分钟才动一次的数字付每 3 秒一次的代价。
//
// 抽成共享 hook：从前它只住在 VisionPage 里，于是「扫描地图」
// 的另一处（对话页 → 视觉缓冲）没有粗动图 —— 而那一处恰好是唯一进得去的那一处。
// 两处各写一遍的话，间隔与端点迟早会分叉；同一个 queryKey 两套配置只有一套生效，
// 而哪一套生效取决于谁先挂载。
export function useCoarseMap() {
  return useQuery({
    queryKey: ["coarse-map"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/coarse-map");
      if (error) throw error;
      return data;
    },
    refetchInterval: 30000,
  });
}
