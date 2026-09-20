import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

/** 新仪器初始化清单。横幅（AppLayout）与页面（SetupPage）共用同一份查询，
 *  所以填完一项之后横幅会跟着两边一起更新，不会出现「页面说好了、横幅还在喊」。
 *
 *  后端永不 500：读不出来会降级成 degraded=true 而不是报错。一份读不出来的
 *  清单如果表现成「没什么要填的」，那是这一页最坏的失败方式。 */
export const INSTRUMENT_INIT_KEY = ["instrument-init"] as const;

export function useInstrumentInit() {
  return useQuery({
    queryKey: INSTRUMENT_INIT_KEY,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/instrument-init");
      if (error) throw error;
      return data;
    },
    // 这份清单只在有人保存时才变；不需要轮询，但换页回来时重取一次，
    // 免得在设置页改完 profile 回到横幅还看着旧计数。
    staleTime: 15_000,
    refetchOnWindowFocus: true,
  });
}
