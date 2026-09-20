import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Card, DegradedNote, Section } from "../ui";
import { Button, Field, SelectField, useToast } from "../controls";
import { JsonBlock } from "./shared";

// 训练日志导出 sub-tab — POST /api/trajectories/export.
//   format: jsonl | sft | dpo | failure_mining
//   thread_id (jsonl only) · limit · inline · max_inline

type Fmt = "jsonl" | "sft" | "dpo" | "failure_mining";

// OLD Gradio 导出类型 dropdown options (verbatim labels), mapped onto the
// /api/trajectories/export `format` enum (failure → failure_mining).
const FORMAT_OPTIONS: { value: Fmt; label: string }[] = [
  { value: "jsonl", label: "全量 JSONL（每行一条完整轨迹）" },
  { value: "sft", label: "SFT 样本（指令+上下文→步骤序列）" },
  { value: "dpo", label: "DPO 偏好对（HITL 改参：模型 vs 人）" },
  { value: "failure_mining", label: "失败挖掘（失败/中止/回滚轨迹）" },
];

export function TrajectoryExportPane() {
  const { toast, node } = useToast();
  const [format, setFormat] = useState<Fmt>("jsonl");

  const m = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/trajectories/export", {
        body: {
          format,
          // OLD GUI exposed only 导出类型; limit / max_inline were server-side
          // defaults. Keep them out of the UI but satisfy the request schema.
          limit: 10000,
          inline: true,
          max_inline: 500,
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d.ok) toast(`导出完成：${d.count} 条`);
      else toast(d.detail || "导出降级", "err");
    },
    onError: () => toast("导出失败", "err"),
  });

  function download() {
    const rows = m.data?.rows ?? [];
    if (!rows.length) return;
    const text = rows.map((r) => JSON.stringify(r)).join("\n");
    const blob = new Blob([text], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trajectories_${format}.jsonl`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-4">
      {/* OLD 训练日志 help markdown (verbatim) restored for parity. */}
      <p className="text-sm text-mast-muted">
        <strong className="text-mast-text">训练日志 Agent Trajectories</strong> —
        把记录的「用户指令 → 路由 → agent 思考 → skill 调用 → 安全门控 → HITL → 结果」因果轨迹导出成训练集（SFT
        / DPO 偏好对 / 失败挖掘）。数据来自 <code className="font-mono">mast_experiments_v2.db</code> 的
        trajectories / trajectory_steps 表，自有产权、本地存储。
      </p>
      <Section title="训练日志">
        <Card>
          <div className="grid items-end gap-4 sm:grid-cols-[3fr_1fr]">
            <Field label="导出类型">
              <SelectField value={format} onChange={setFormat} options={FORMAT_OPTIONS} />
            </Field>
            <Button variant="primary" onClick={() => m.mutate()} disabled={m.isPending}>
              {m.isPending ? "导出中…" : "导出训练集"}
            </Button>
          </div>
          {m.data?.rows && m.data.rows.length > 0 && (
            <div className="mt-4">
              <Button onClick={download}>下载导出文件 ({m.data.rows.length})</Button>
            </div>
          )}
        </Card>
      </Section>

      {m.data?.degraded && <DegradedNote what="训练日志导出" />}

      {m.data && !m.data.degraded && (
        <Section title="导出结果">
          <Card>
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-mast-muted">
              <span>格式：{m.data.format}</span>
              <span className="tabular-nums">总数：{m.data.count}</span>
              <span className="tabular-nums">内联：{m.data.rows?.length ?? 0}</span>
              {m.data.truncated && <span className="text-mast-warn">已截断</span>}
            </div>
            {m.data.rows && m.data.rows.length > 0 && (
              <div className="mt-3 space-y-2">
                {m.data.rows.slice(0, 20).map((r, i) => (
                  <JsonBlock key={i} value={r} />
                ))}
                {m.data.rows.length > 20 && (
                  <div className="text-xs text-mast-muted">
                    仅预览前 20 条（共 {m.data.rows.length} 条内联）；点击“下载 JSONL”获取全部。
                  </div>
                )}
              </div>
            )}
          </Card>
        </Section>
      )}
      {node}
    </div>
  );
}
