import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, Field, SelectField, Toggle, useToast } from "@/components/controls";
import { agentLabel } from "@/components/agents/registry";
import { settingsWriteProblem } from "@/lib/settingsWrite";

// ════════════════════════════════════════════════════════════════════════════
// 后台任务 — true-parallel background runs.
//
// A long/independent task (a literature survey, offline analysis) runs as a
// DETACHED orchestrator invocation (its own thread_id + isolated checkpointer)
// so the FOREGROUND supervisor + instrument_control stay responsive — it breaks
// the LangGraph super-step barrier WITHOUT changing graph semantics. Results
// merge back into the group transcript, tagged 「后台」.
//
// This panel is a thin client over the relay endpoints:
//   POST /api/agents/run-task/background           — launch
//   GET  /api/agents/background-runs               — list (polled)
//   POST /api/agents/run-task/background/{id}/abort — stop
// instrument_control is intentionally NOT offerable — it is the foreground
// hardware agent and can never be detached (the backend rejects it too).
// ════════════════════════════════════════════════════════════════════════════

// Backgroundable agents = everything EXCEPT instrument_control (mirrors the
// backend BackgroundRunManager.BACKGROUNDABLE set).
const BACKGROUNDABLE = [
  "literature",
  "data_processing",
  "experiment_design",
  "paper_writing",
  "paper_review",
] as const;

const STATUS_TONE: Record<string, string> = {
  queued: "INFO",
  running: "AUTO",
  done: "AUTO",
  failed: "DANGEROUS",
  aborted: "WARN",
};
const STATUS_LABEL: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  done: "已完成",
  failed: "失败",
  aborted: "已中止",
};
const ACTIVE = new Set(["queued", "running"]);

type BgRun = {
  run_id: string;
  conversation_id: string;
  instruction: string;
  agents: string[];
  title: string;
  priority?: string;
  status: string;
  created_at?: number | null;
  started_at?: number | null;
  finished_at?: number | null;
  final_text: string;
  error: string;
  thread_id: string;
  // fine-grained progress (resource-governance ①)
  steps?: number;
  progress?: number | null;
  last_activity?: string;
};

function fmtTime(t?: number | null): string {
  if (!t) return "—";
  return new Date(t * 1000).toLocaleTimeString();
}

function elapsed(r: BgRun): string {
  const start = r.started_at ?? r.created_at;
  if (!start) return "";
  const end = r.finished_at ?? Date.now() / 1000;
  const s = Math.max(0, Math.round(end - start));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${s % 60}s`;
}

export function BackgroundRunsPanel({ conversationId }: { conversationId?: string | null }) {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const [instruction, setInstruction] = useState("");
  const [agent, setAgent] = useState<string>("literature");

  const runsQuery = useQuery({
    queryKey: ["agents", "background-runs", conversationId ?? ""],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/background-runs", {
        params: { query: conversationId ? { conversation_id: conversationId } : {} },
      });
      if (error) throw error;
      return data;
    },
    // poll briskly while anything is active, back off to idle otherwise.
    refetchInterval: (q) => {
      const runs = (q.state.data?.runs ?? []) as BgRun[];
      return runs.some((r) => ACTIVE.has(r.status)) ? 2500 : 8000;
    },
  });

  const runs = (runsQuery.data?.runs ?? []) as BgRun[];
  const activeCount = useMemo(() => runs.filter((r) => ACTIVE.has(r.status)).length, [runs]);

  // Conservative auto-background toggle (default OFF). When ON, the supervisor
  // auto-detaches a literature survey paired with instrument_control so the
  // instrument foreground isn't barrier-blocked. Read/write the live setting.
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });
  const autoOn = settings.data?.orchestrator_auto_background ?? false;
  const saveAuto = useMutation({
    mutationFn: async (v: boolean) => {
      const { data, error } = await api.POST("/api/settings", {
        body: { orchestrator_auto_background: v },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d, v) => {
      // 这句话以前无条件说出口。下面 spawn 那个 mutation 查了 `ok`，这个没查 ——
      // 同一个文件里一个查一个不查，正是「每一页各自记得」那种接线的样子。
      const problem = settingsWriteProblem(d);
      if (problem) { toast(problem, "err"); return; }
      toast(v ? "已开启自动后台化（保守：仅文献综述搭配仪器时）" : "已关闭自动后台化", "ok");
      qc.invalidateQueries({ queryKey: ["settings"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  const spawn = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/agents/run-task/background", {
        body: {
          instruction: instruction.trim(),
          agents: [agent],
          conversation_id: conversationId ?? null,
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok && !data.degraded) {
        toast(`已在后台启动（${agentLabel(agent)}）`, "ok");
        setInstruction("");
        qc.invalidateQueries({ queryKey: ["agents", "background-runs"] });
      } else {
        toast(`未启动：${data?.detail ?? "后端降级 / 无可用模型"}`, "err");
      }
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  const abort = useMutation({
    mutationFn: async (runId: string) => {
      const { data, error } = await api.POST("/api/agents/run-task/background/{run_id}/abort", {
        params: { path: { run_id: runId } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok && data.aborted) toast("已发出中止信号", "ok");
      else toast(`未中止（${data?.detail ?? "可能已结束"}）`, "err");
      qc.invalidateQueries({ queryKey: ["agents", "background-runs"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  function submitSpawn() {
    if (!instruction.trim() || spawn.isPending) return;
    spawn.mutate();
  }

  return (
    <div className="space-y-4">
      {node}

      {/* what this is */}
      <Card className="space-y-1 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium">后台任务 · 真并行</span>
          <Badge tone={activeCount > 0 ? "AUTO" : "INFO"}>
            {activeCount > 0 ? `${activeCount} 个进行中` : "空闲"}
          </Badge>
        </div>
        <p className="text-xs text-mast-muted">
          把独立的长任务（文献综述 / 离线分析 / 起草报告）甩到后台异步运行，前台仪器对话不被阻塞；
          结果会带回当前群聊并标注「后台」。仪器控制（instrument_control）留在前台，不能后台化。
        </p>
        <div className="flex items-center justify-between gap-3 border-t border-mast-border pt-2">
          <div className="min-w-0">
            <div className="text-xs font-medium text-mast-text">自动后台化（保守 · 实验性）</div>
            <div className="text-[11px] text-mast-muted">
              开启后：当编排器把「文献综述」和「仪器操作」放到同一批时，自动把文献综述转入后台，
              让仪器前台不被阻塞。仅限文献（其它分析依赖前台产物，不自动后台）；默认关。
              想强制某次留前台？在对话里 @文献 直接点名即可（定向永不自动后台）。
            </div>
          </div>
          <Toggle
            checked={!!autoOn}
            onChange={(v) => saveAuto.mutate(v)}
            label={autoOn ? "已开启" : "已关闭"}
          />
        </div>
      </Card>

      {/* launcher */}
      <Card className="space-y-3 px-4 py-3">
        <Field
          label="启动后台任务"
          hint="选一个非仪器智能体，描述一个不依赖当前硬件步骤、又比较耗时的独立任务。"
        >
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
            <textarea
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submitSpawn();
                }
              }}
              placeholder="例：综述 Au(111) 上单分子磁体的 STM/STS 研究进展"
              rows={2}
              disabled={spawn.isPending}
              className="flex-1 resize-none rounded-md border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent disabled:opacity-50"
            />
            <div className="flex items-end gap-2">
              <div className="w-40">
                <SelectField
                  value={agent}
                  onChange={setAgent}
                  options={BACKGROUNDABLE.map((id) => ({ value: id, label: agentLabel(id) }))}
                />
              </div>
              <Button
                variant="primary"
                onClick={submitSpawn}
                disabled={!instruction.trim() || spawn.isPending}
              >
                {spawn.isPending ? "启动中…" : "后台启动"}
              </Button>
            </div>
          </div>
        </Field>
      </Card>

      {/* run list */}
      <div>
        <div className="mb-2 flex items-center gap-3">
          <h4 className="text-sm font-medium">
            后台运行
            <span className="ml-2 text-xs text-mast-muted">共 {runs.length} 个</span>
          </h4>
        </div>
        {runsQuery.isPending && <Spinner />}
        {runsQuery.isError && <ErrorNote error={runsQuery.error} />}
        {runsQuery.data?.degraded && <DegradedNote what="后台任务列表" />}
        {runsQuery.data && !runsQuery.data.degraded && runs.length === 0 && (
          <EmptyNote label="暂无后台任务。用上方表单启动一个，或让仪器智能体在对话中自行甩到后台。" />
        )}
        {runs.length > 0 && (
          <div className="space-y-2">
            {runs.map((r) => (
              <Card key={r.run_id} className="px-4 py-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0 flex-1 space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge tone={STATUS_TONE[r.status] ?? "INFO"}>
                        {STATUS_LABEL[r.status] ?? r.status}
                      </Badge>
                      {r.priority === "high" && <Badge tone="WARN">高优先</Badge>}
                      {r.agents.map((a) => (
                        <span
                          key={a}
                          className="rounded border border-mast-border px-1.5 py-0.5 font-mono text-[10px] text-mast-muted"
                        >
                          {agentLabel(a)}
                        </span>
                      ))}
                      <span className="font-mono text-[10px] text-mast-muted">
                        {elapsed(r)}
                      </span>
                    </div>
                    <div className="truncate text-sm text-mast-text" title={r.instruction}>
                      {r.title || r.instruction}
                    </div>
                    {/* live progress bar (item ①) — only while active, when the
                        backend has published a percent. */}
                    {ACTIVE.has(r.status) && typeof r.progress === "number" && (
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-mast-border/60">
                            <div
                              className="h-full rounded-full bg-mast-accent transition-all"
                              style={{ width: `${Math.max(0, Math.min(100, r.progress))}%` }}
                            />
                          </div>
                          <span className="font-mono text-[10px] tabular-nums text-mast-muted">
                            {Math.max(0, Math.min(100, r.progress))}%
                          </span>
                        </div>
                        {r.last_activity && (
                          <div
                            className="truncate text-[10px] text-mast-muted/80"
                            title={r.last_activity}
                          >
                            {r.last_activity}
                          </div>
                        )}
                      </div>
                    )}
                    {(r.final_text || r.error) && (
                      <p
                        className={
                          "line-clamp-2 text-xs " +
                          (r.error ? "text-mast-danger" : "text-mast-muted")
                        }
                      >
                        {r.error || r.final_text}
                      </p>
                    )}
                    <div className="flex flex-wrap gap-3 font-mono text-[10px] text-mast-muted/80">
                      <span>启动 {fmtTime(r.started_at ?? r.created_at)}</span>
                      {r.finished_at && <span>结束 {fmtTime(r.finished_at)}</span>}
                      {r.conversation_id && <span className="truncate">会话 {r.conversation_id.slice(0, 8)}</span>}
                    </div>
                  </div>
                  <div>
                    {ACTIVE.has(r.status) && (
                      <Button
                        variant="danger"
                        onClick={() => abort.mutate(r.run_id)}
                        disabled={abort.isPending && abort.variables === r.run_id}
                      >
                        {abort.isPending && abort.variables === r.run_id ? "中止中…" : "中止"}
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
