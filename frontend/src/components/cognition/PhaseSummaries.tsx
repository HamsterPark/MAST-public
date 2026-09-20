import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, Badge, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { Field, TextField, Button } from "@/components/controls";

// 对话阶段摘要 (sharding output) — mirrors cognition_panel.render_phase_summaries_html
// over PhaseManager.list_phases via GET /api/cognition/phases.
//
// Long conversations are auto/manually sharded; each shard yields a compressed
// summary. Optionally scope to one experiment_id (read-only; query is debounced
// to an explicit "查询" click so a stray keystroke never refetches).

export function PhaseSummaries() {
  const [draft, setDraft] = useState("");
  const [experimentId, setExperimentId] = useState("");

  const q = useQuery({
    queryKey: ["cognition", "phases", experimentId],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/cognition/phases", {
        params: { query: { experiment_id: experimentId.trim() || null } },
      });
      if (error) throw error;
      return data;
    },
  });

  const phases = q.data?.phases ?? [];

  return (
    <Card>
      <div className="mb-3 flex items-end gap-2">
        <div className="flex-1">
          <Field label="实验 ID（可选）" hint="留空 = 全部实验的对话阶段">
            <TextField
              value={draft}
              onChange={setDraft}
              placeholder="例如：exp-2026-06-..."
            />
          </Field>
        </div>
        <Button variant="primary" onClick={() => setExperimentId(draft)}>
          查询
        </Button>
        {experimentId && (
          <Button
            variant="ghost"
            onClick={() => {
              setDraft("");
              setExperimentId("");
            }}
          >
            清除筛选
          </Button>
        )}
      </div>

      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="分片管理" />}
      {q.data && !q.data.degraded && phases.length === 0 && (
        <EmptyNote label="尚无对话阶段。长对话会自动/手动分片，每段产出压缩摘要。" />
      )}

      {q.data && !q.data.degraded && phases.length > 0 && (
        <div className="space-y-1">
          <p className="px-1 text-xs text-mast-muted">
            {q.data.count} 段
            {experimentId && (
              <>
                {" "}
                · 实验 <code>{experimentId}</code>
              </>
            )}
          </p>
          <div className="divide-y divide-mast-border overflow-hidden rounded border border-mast-border">
            {phases.map((p, i) => (
              <div key={`${p.phase_index ?? i}`} className="px-3 py-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-semibold tabular-nums">
                    #{p.phase_index ?? i}
                  </span>
                  <span>{p.title}</span>
                  {p.open && <Badge tone="AUTO">进行中</Badge>}
                </div>
                <div className="mt-1 whitespace-pre-wrap text-mast-muted">
                  {p.summary}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
