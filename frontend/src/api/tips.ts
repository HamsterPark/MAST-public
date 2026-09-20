// 针尖登记 —— 当前装的是哪根针，以及换针史。
//
// 设计文档：docs/v2/design/tip_registry_and_hardware_profile.md
//
// 作用域是**仪器**而不是实验：换实验、换样品都未必换针尖。所以这里没有
// experiment_id，也没有「切换针尖」——针尖服役期是线性的，当前针尖就是
// 唯一还没退役的那一行。
//
// 与 scope.ts 同样刻意不进 zustand：这是服务端状态，persist 一缓存就是陈旧副本。

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";

export const TIP_KEY = ["tips", "current"] as const;

export type Tip = {
  id: string;
  tip_index?: number | null;
  name: string;
  material: string;
  material_detail: string;
  fabrication: string;
  form: string;
  wire_diameter_mm?: number | null;
  qplus_sensor_model: string;
  qplus_f0_hz?: number | null;
  qplus_q?: number | null;
  qplus_k_n_per_m?: number | null;
  installed_at?: string | null;
  removed_at?: string | null;
  installed_by: string;
  note: string;
  retire_snapshot: Record<string, unknown>;
};

export type CurrentTip = {
  tip?: Tip | null;
  registered: boolean;
  degraded: boolean;
  hint: string;
};

export type TipChangeResult = {
  ok: boolean;
  tip?: Tip | null;
  changed: boolean;
  cleared_calibration: string[];
  warnings: string[];
  error: string;
};

/** 当前针尖。所有组件共用这一个 queryKey。 */
export function useCurrentTip() {
  return useQuery({
    queryKey: TIP_KEY,
    queryFn: async (): Promise<CurrentTip> => {
      const { data, error } = await api.GET("/api/tips/current");
      if (error) throw error;
      return data as CurrentTip;
    },
    // agent 也能登记针尖（register_tip 是 meta 工具），所以要轮询才跟得上。
    // 10 秒与作用域保持一致。
    refetchInterval: 10_000,
  });
}

/** 换针史。只有面板展开时才拉。 */
export function useTipHistory(enabled = true) {
  return useQuery({
    queryKey: ["tips", "history"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/tips", {
        params: { query: { limit: 50 } },
      });
      if (error) throw error;
      return data as { tips: Tip[]; degraded: boolean };
    },
    enabled,
  });
}

/** 受控词表（后端是唯一真源，免得前后端各写一份分叉）。 */
export function useTipVocabulary(enabled = true) {
  return useQuery({
    queryKey: ["tips", "vocabulary"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/tips/vocabulary");
      if (error) throw error;
      return data as {
        materials: { value: string; label: string }[];
        fabrications: { value: string; label: string }[];
        forms: { value: string; label: string }[];
      };
    },
    enabled,
    staleTime: Infinity, // 词表在一次会话里不会变
  });
}

function invalidateTips(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: ["tips"] });
  // 换针清掉了学习标定 —— 设置页显示的 dI/dV 标定也跟着变了。
  qc.invalidateQueries({ queryKey: ["settings"] });
}

export type RegisterTipInput = {
  material?: string;
  fabrication?: string;
  form?: string;
  name?: string;
  material_detail?: string;
  wire_diameter_mm?: number | null;
  qplus_sensor_model?: string;
  qplus_f0_hz?: number | null;
  qplus_q?: number | null;
  qplus_k_n_per_m?: number | null;
  installed_at?: string;
  note?: string;
};

/** 登记装入一根针尖。会退役上一根并清掉绑它的学习标定。 */
export function useRegisterTip() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (v: RegisterTipInput) => {
      const { data, error } = await api.POST("/api/tips", {
        body: {
          material: v.material ?? "",
          fabrication: v.fabrication ?? "",
          form: v.form ?? "",
          name: v.name ?? "",
          material_detail: v.material_detail ?? "",
          wire_diameter_mm: v.wire_diameter_mm ?? null,
          qplus_sensor_model: v.qplus_sensor_model ?? "",
          qplus_f0_hz: v.qplus_f0_hz ?? null,
          qplus_q: v.qplus_q ?? null,
          qplus_k_n_per_m: v.qplus_k_n_per_m ?? null,
          installed_at: v.installed_at ?? "",
          installed_by: "",
          note: v.note ?? "",
        },
      });
      if (error) throw error;
      return data as TipChangeResult;
    },
    onSuccess: () => invalidateTips(qc),
  });
}

/** 补记/更正属性。不清任何标定。 */
export function useUpdateTip() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (v: { tipId: string; fields: Partial<RegisterTipInput> }) => {
      const { data, error } = await api.PATCH("/api/tips/{tip_id}", {
        params: { path: { tip_id: v.tipId } },
        body: v.fields as Record<string, never>,
      });
      if (error) throw error;
      return data as TipChangeResult;
    },
    onSuccess: () => invalidateTips(qc),
  });
}

/** 记录「针尖已取出、还没装新的」。 */
export function useRemoveCurrentTip() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/tips/current/remove", {});
      if (error) throw error;
      return data as TipChangeResult;
    },
    onSuccess: () => invalidateTips(qc),
  });
}

/** 「W · 电化学腐蚀 · 普通针尖」这样的一行摘要。 */
export function tipSummary(tip: Tip | null | undefined): string {
  if (!tip) return "未登记";
  const bits = [tip.material, tip.fabrication, tip.form].filter(
    (x) => x && x !== "unknown",
  );
  return bits.join(" · ") || "未记录属性";
}

/** 已服役天数；装入日期解析不出返回 null。 */
export function serviceDays(tip: Tip | null | undefined): number | null {
  if (!tip?.installed_at) return null;
  const t = Date.parse(tip.installed_at);
  if (Number.isNaN(t)) return null;
  return Math.max(0, Math.floor((Date.now() - t) / 86_400_000));
}
