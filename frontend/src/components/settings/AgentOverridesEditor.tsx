import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Badge, Card, DegradedNote, ErrorNote, Spinner } from "../ui";
import { Button, SelectField } from "../controls";
import { thinkingDisplay, type ThinkingMode } from "../../pages/SettingsPage";

// 设置 → 各 Agent 模型 / Thinking（与 Agents 页同源）。Per-agent MODEL is
// selectable; Thinking is a READ-ONLY display of the locked value (consistent
// with 全局 / 查询助手 in 设置). Reads GET /api/agents/models (effective table)
// and writes POST /api/agents/{agent_id}/model-override per row (model only).

// Agent id → human label (mirrors old _agent_labels; instrument_control added).
const AGENT_LABELS: Record<string, string> = {
  _supervisor: "编排 SUP",
  literature: "文献 LIT",
  experiment_design: "实验设计 XD",
  instrument_control: "仪器控制 IC",
  data_processing: "数据处理 DP",
  paper_writing: "论文写作 PW",
  paper_review: "论文审稿 PR",
  buffer_summarizer: "视觉摘要 BUF",
};

// Old _agent_model_choices.
const AGENT_MODEL_CHOICES = [
  "kimi-k2.6",
  "kimi-k2.7-code",
  "deepseek-v4-pro",
  "sonnet-4.6",
  "haiku-4.5",
  "minimax-m3",
  "glm-5.2",
];

// Per-agent thinking is NOT user-selectable: the system locks each agent's
// model to its strongest supported thinking value. We only DISPLAY the actual
// locked value (derived from the selected model's thinking_mode via
// thinkingDisplay) — identical style/wording to 全局 / 查询助手 in 设置.
type RowState = { model?: string };

function AgentRow({
  agentId,
  label,
  effModel,
  effThinking,
  modeForAlias,
  toast,
}: {
  agentId: string;
  label: string;
  effModel: string;
  effThinking: string | null | undefined;
  modeForAlias: (alias: string) => ThinkingMode;
  toast: (text: string, tone?: "ok" | "err") => void;
}) {
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<RowState>({});

  const model = draft.model ?? (AGENT_MODEL_CHOICES.includes(effModel) ? effModel : "kimi-k2.6");
  // Thinking display reflects the CURRENTLY-SELECTED model in this row (locked).
  const thinkingMode = modeForAlias(model);
  const thinkInfo = thinkingDisplay(thinkingMode);

  const mut = useMutation({
    mutationFn: async (body: { model?: string }) => {
      const { data, error } = await api.POST("/api/agents/{agent_id}/model-override", {
        params: { path: { agent_id: agentId } },
        body,
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.degraded) {
        toast(`${label}：内核未连接，改动未应用`, "err");
        return;
      }
      if (data?.ok) {
        toast(`${label} 已更新`, "ok");
        setDraft({});
        queryClient.invalidateQueries({ queryKey: ["agents", "models"] });
      } else {
        toast(`${label} 更新失败`, "err");
      }
    },
    onError: (e) => toast(`${label} 更新失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const dirty = draft.model !== undefined;

  return (
    <div className="flex flex-wrap items-end gap-3 border-t border-mast-border py-3 first:border-t-0">
      <div className="w-32 shrink-0 text-sm font-medium">{label}</div>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-mast-muted">模型</span>
        <SelectField
          value={model}
          onChange={(v) => setDraft((d) => ({ ...d, model: v }))}
          options={AGENT_MODEL_CHOICES.map((m) => ({ value: m, label: m }))}
        />
      </label>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-mast-muted">Thinking</span>
        {/* READ-ONLY display of the actual locked value (not selectable) —
            identical wording/style to 全局 / 查询助手 in 设置. */}
        <span className="inline-flex items-center rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5">
          <Badge tone={thinkInfo.tone}>{thinkInfo.label}</Badge>
        </span>
      </label>
      <Button
        variant="primary"
        disabled={!dirty || mut.isPending}
        // Thinking is locked (display-only), so only the model choice persists.
        onClick={() => mut.mutate({ model })}
      >
        {mut.isPending ? "保存中…" : "保存"}
      </Button>
      <span className="text-xs text-mast-muted">
        当前生效：{effModel}
        {effThinking ? ` · ${effThinking}` : ""}
      </span>
    </div>
  );
}

export function AgentOverridesEditor({
  toast,
}: {
  toast: (text: string, tone?: "ok" | "err") => void;
}) {
  const q = useQuery({
    queryKey: ["agents", "models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/models");
      if (error) throw error;
      return data;
    },
  });

  // Model-capability table → alias → thinking_mode, so each row's READ-ONLY
  // thinking display reflects the selected model's locked value. Default tunable
  // when unknown.
  const modelsQ = useQuery({
    queryKey: ["config", "models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/config/models");
      if (error) throw error;
      return data;
    },
  });
  const modeForAlias = useMemo(() => {
    const byAlias = new Map(
      (modelsQ.data?.models ?? []).map((m) => [m.alias, m.thinking_mode as ThinkingMode]),
    );
    return (alias: string): ThinkingMode => byAlias.get(alias) ?? "tunable";
  }, [modelsQ.data]);

  // Merge live effective table with the canonical 8-agent ordering so every
  // agent renders even if the registry omits one.
  const rows = useMemo(() => {
    const byId = new Map((q.data?.agents ?? []).map((a) => [a.agent_id, a]));
    const ordered = Object.keys(AGENT_LABELS);
    const extra = (q.data?.agents ?? [])
      .map((a) => a.agent_id)
      .filter((id) => !AGENT_LABELS[id]);
    return [...ordered, ...extra].map((id) => ({
      agent_id: id,
      label: AGENT_LABELS[id] ?? id,
      model: byId.get(id)?.model ?? "kimi-k2.6",
      thinking: byId.get(id)?.thinking ?? null,
    }));
  }, [q.data]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="各 Agent 模型覆盖" />;

  return (
    <Card>
      <p className="mb-2 text-xs text-mast-muted">
        每个 agent 可独立指定模型；改动经 ConfigOverrideRegistry 持久化，并在下一个编排任务生效。
        思考强度由系统锁定为该模型支持的最强值，仅作只读显示、不可手动选择（可调模型锁定 max，
        推理模型服务端固定，无思考模型不传 thinking 参数）。
      </p>
      <div>
        {rows.map((r) => (
          <AgentRow
            key={r.agent_id}
            agentId={r.agent_id}
            label={r.label}
            effModel={r.model}
            effThinking={r.thinking}
            modeForAlias={modeForAlias}
            toast={toast}
          />
        ))}
      </div>
    </Card>
  );
}
