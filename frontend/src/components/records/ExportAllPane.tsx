import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Card, DegradedNote, Section } from "../ui";
import { Button, Toggle, useToast } from "../controls";
import { Field as KV, fmtTime } from "./shared";

// 导出全部 sub-tab — POST /api/experiments/export.
//   include_heavy (model weights / caches / manuals) · dest (optional override).
//   Returns the export_all manifest roll-up.

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function ExportAllPane() {
  const { toast, node } = useToast();
  const [includeHeavy, setIncludeHeavy] = useState(false);

  const m = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/experiments/export", {
        body: { include_heavy: includeHeavy, dest: null },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d.ok) toast(`导出完成：${d.file_count} 个文件`);
      else toast(d.detail || "导出降级", "err");
    },
    onError: () => toast("导出失败", "err"),
  });

  return (
    <div className="space-y-4">
      {/* OLD 导出全部 help markdown (verbatim) restored for parity. */}
      <p className="text-sm text-mast-muted">
        <strong className="text-mast-text">📦 一键导出全部历史数据</strong> —
        把全部实验记录、实验数据、所有对话交互、评价/反馈、训练轨迹、日志、计划与快照打包成一个
        ZIP（含 v1+v2 数据库的一致快照）。实验室内部工具，不做隐私过滤，通通导出。
      </p>
      <Section title="导出全部">
        <Card>
          <div className="flex items-start gap-2">
            <Toggle checked={includeHeavy} onChange={setIncludeHeavy} label="include_heavy" />
            <span className="text-sm text-mast-muted">
              同时包含大型模型权重 / 文献索引 / 缓存（体积可达 GB 级，通常不需要）
            </span>
          </div>
          <div className="mt-4">
            <Button variant="primary" onClick={() => m.mutate()} disabled={m.isPending}>
              {m.isPending ? "打包中…" : "打包导出全部历史数据"}
            </Button>
          </div>
        </Card>
      </Section>

      {m.data?.degraded && <DegradedNote what="全量导出" />}

      {m.data && !m.data.degraded && (
        <Section title="导出清单">
          <Card>
            <KV label="目标">
              <span className="break-all font-mono text-xs">{m.data.dest || "—"}</span>
            </KV>
            <KV label="文件数">
              <span className="tabular-nums">{m.data.file_count}</span>
            </KV>
            <KV label="数据库数">
              <span className="tabular-nums">{m.data.database_count}</span>
            </KV>
            <KV label="总大小">
              <span className="tabular-nums">{fmtBytes(m.data.total_bytes)}</span>
            </KV>
            <KV label="跳过">
              <span className="tabular-nums">{m.data.skipped_count}</span>
            </KV>
            <KV label="错误">
              <span className="tabular-nums text-mast-danger">{m.data.error_count}</span>
            </KV>
            <KV label="包含大文件">{m.data.include_heavy ? "是" : "否"}</KV>
            <KV label="创建时间">{fmtTime(m.data.created_at)}</KV>
            {m.data.detail && <KV label="详情">{m.data.detail}</KV>}
          </Card>
        </Section>
      )}
      {node}
    </div>
  );
}
