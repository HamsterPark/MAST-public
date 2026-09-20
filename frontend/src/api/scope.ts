// 当前实验 / 当前样品 —— 前端的唯一真源。
//
// 设计文档：docs/v2/design/experiment_folder_persistence.md §12
//
// 在这之前没有 /api/experiments/current 端点，于是每个组件自己从实验列表里猜
// 当前实验，而且猜法不一致：RightPanel 取 experiments[0]，TopBar 用
// find(status === "active")。两处能显示不同的实验。
//
// 现在服务端有一个显式指针（active_scope 单行表）。所有组件读同一个端点、同一个
// queryKey —— 结构上不可能再分歧。
//
// 刻意不放进 zustand：这是服务端状态。store.ts 的 persist 一旦缓存它，重启后就是
// 一份陈旧副本 —— 那正是我们在修的 bug。react-query 的缓存就是共享 store。

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";

export const SCOPE_KEY = ["scope", "current"] as const;

export type CurrentScope = {
  experiment?: {
    id: string;
    name: string;
    goal: string;
    start_time?: string | null;
    last_active_at?: string | null;
    dir_name?: string | null;
  } | null;
  sample?: {
    id: string;
    name: string;
    sample_type: string;
    sample_subtype: string;
    description: string;
    start_time?: string | null;
    last_active_at?: string | null;
    dir_name?: string | null;
    index?: number | null;
  } | null;
  has_experiment: boolean;
  has_sample: boolean;
  hint: string;
  folder_path?: string | null;
  degraded: boolean;
};

/** 当前作用域。所有组件共用这一个 queryKey。 */
export function useCurrentScope() {
  return useQuery({
    queryKey: SCOPE_KEY,
    queryFn: async (): Promise<CurrentScope> => {
      const { data, error } = await api.GET("/api/experiments/current");
      if (error) throw error;
      return data as CurrentScope;
    },
    // 作用域可以被 agent 从对话里改（start_experiment / start_sample 是 agent
    // 工具），所以前端必须轮询才能跟上；10 秒与 ["experiments"] 保持一致。
    refetchInterval: 10_000,
  });
}

/** 切换器的数据源：按【上次活动时间】倒序，不是创建时间。 */
export function useRecentExperiments(q = "", enabled = true) {
  return useQuery({
    queryKey: ["scope", "recent", q],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experiments/recent", {
        params: { query: { limit: 40, q } },
      });
      if (error) throw error;
      return data;
    },
    enabled,
  });
}

export function useSamplesOf(experimentId: string | null | undefined, enabled = true) {
  return useQuery({
    queryKey: ["scope", "samples", experimentId ?? ""],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experiments/{experiment_id}/samples", {
        params: { path: { experiment_id: experimentId as string } },
      });
      if (error) throw error;
      return data;
    },
    enabled: enabled && !!experimentId,
  });
}

/** 新建实验前的重名检测 —— 「严谨」的实质。 */
export function usePreflight(name: string, enabled: boolean) {
  const trimmed = name.trim();
  return useQuery({
    queryKey: ["scope", "preflight", trimmed],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experiments/preflight", {
        params: { query: { name: trimmed } },
      });
      if (error) throw error;
      return data;
    },
    enabled: enabled && trimmed.length >= 2,
  });
}

/** 切换前的忙碌状态。本地内存读，毫秒级，永不阻塞。 */
export function useSwitchAdvisory(enabled: boolean) {
  return useQuery({
    queryKey: ["scope", "advisory"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/scope/switch-advisory");
      if (error) throw error;
      return data;
    },
    enabled,
    refetchInterval: enabled ? 4000 : false,
  });
}

/** 切换后要刷新的所有缓存。 */
function invalidateScope(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: ["scope"] });
  qc.invalidateQueries({ queryKey: ["experiments"] });
}

export type ScopeChangeResult = {
  ok: boolean;
  changed: boolean;
  blocked: boolean;
  block_code: string;
  block_reason: string;
  can_force: boolean;
  warnings: string[];
  degraded: boolean;
  experiment?: CurrentScope["experiment"];
  sample?: CurrentScope["sample"];
};

export function useSwitchExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (v: { experimentId: string; sampleId?: string | null; force?: boolean }) => {
      const { data, error } = await api.POST("/api/experiments/{experiment_id}/activate", {
        params: { path: { experiment_id: v.experimentId } },
        body: { sample_id: v.sampleId ?? null, force: !!v.force, reason: "" },
      });
      if (error) throw error;
      return data as ScopeChangeResult;
    },
    onSuccess: () => invalidateScope(qc),
  });
}

export function useSwitchSample() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (v: { experimentId: string; sampleId: string; force?: boolean }) => {
      const { data, error } = await api.POST(
        "/api/experiments/{experiment_id}/samples/{sample_id}/activate",
        {
          params: { path: { experiment_id: v.experimentId, sample_id: v.sampleId } },
          body: { force: !!v.force, reason: "" },
        },
      );
      if (error) throw error;
      return data as ScopeChangeResult;
    },
    onSuccess: () => invalidateScope(qc),
  });
}

/** 取消选中样品 —— 物理出样时用，不结束样品（样品没有终态）。 */
export function useClearSample() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/scope/clear-sample", {
        body: { reason: "" },
      });
      if (error) throw error;
      return data as ScopeChangeResult;
    },
    onSuccess: () => invalidateScope(qc),
  });
}

export function useCreateExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (v: { name: string; goal: string }) => {
      const { data, error } = await api.POST("/api/experiments", {
        body: { name: v.name, goal: v.goal },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => invalidateScope(qc),
  });
}

export function useCreateSample() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (v: {
      experimentId: string;
      name: string;
      description?: string;
      sample_type?: string;
      sample_subtype?: string;
    }) => {
      const { data, error } = await api.POST("/api/experiments/{experiment_id}/samples", {
        params: { path: { experiment_id: v.experimentId } },
        body: {
          name: v.name,
          description: v.description ?? "",
          sample_type: v.sample_type ?? "",
          sample_subtype: v.sample_subtype ?? "",
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => invalidateScope(qc),
  });
}
