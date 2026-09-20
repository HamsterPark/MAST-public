import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";

/** Per-agent LIGHT admin panel for the 5 non-IC tool-based agents
 *  (LIT / XD / DP / PW / PR).
 *
 *  Verbatim parity with the OLD admin_panel.py:_build_light_agent_panel — those
 *  agents are LangGraph @tool agents auto-bridged into the registry as AUTO-level
 *  skills. They carry NO editable hardware governance (safety_level / preconditions
 *  / rollback are not applicable), so their tools are shown READ-ONLY here, with a
 *  note that everything is AUTO-level (no safety gating). Model / thinking is the
 *  single source in 设置 → 各 Agent 模型; we only DISPLAY it (GET /api/agents/models).
 *
 *  Tool list: GET /api/agents/tools (per-agent catalog), filtered to this agent.
 */
export function LightAgentPanel({ agentId, label }: { agentId: string; label: string }) {
  const modelsQ = useQuery({
    queryKey: ["agents", "models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/models");
      if (error) throw error;
      return data;
    },
  });

  const toolsQ = useQuery({
    queryKey: ["agents", "tools"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/tools");
      if (error) throw error;
      return data;
    },
  });

  const me = useMemo(
    () => (modelsQ.data?.agents ?? []).find((a) => a.agent_id === agentId) ?? null,
    [modelsQ.data, agentId],
  );
  const myTools = useMemo(
    () =>
      [...((toolsQ.data?.agents ?? []).find((a) => a.agent_id === agentId)?.tools ?? [])].sort(
        (a, b) => a.name.localeCompare(b.name),
      ),
    [toolsQ.data, agentId],
  );

  const modelTxt = me?.model ?? "—";
  const thinkTxt = me?.thinking ?? "—";

  return (
    <div className="space-y-4">
      {/* model / thinking summary (display-only — edited in 设置) */}
      <Card>
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <span className="text-sm font-semibold text-mast-text">{label}</span>
          <span className="text-sm text-mast-muted">— 工具型 agent（LangGraph tools）</span>
        </div>
        <div className="mt-2 text-sm text-mast-muted">
          {modelsQ.isPending ? (
            <Spinner label="读取模型…" />
          ) : modelsQ.error ? (
            <ErrorNote error={modelsQ.error} />
          ) : (
            <p>
              当前模型 <span className="font-mono text-mast-text">{modelTxt}</span> · thinking{" "}
              <span className="font-mono text-mast-text">{thinkTxt}</span>
              {modelsQ.data?.degraded && <span className="ml-2 text-mast-warn">（registry 未接入）</span>}
              <span className="ml-1">（在「设置 → 各 Agent 模型 / Thinking」调整——单一真源）。</span>
            </p>
          )}
        </div>
      </Card>

      {/* read-only tool codex */}
      <Card>
        <div className="mb-1 flex items-center gap-2">
          <span className="text-sm font-semibold text-mast-text">技能管理（只读）</span>
          {toolsQ.data && !toolsQ.data.degraded && <Badge tone="INFO">{myTools.length}</Badge>}
        </div>
        <p className="mb-3 text-xs text-mast-muted">
          该 agent 的工具均为 <Badge tone="AUTO">AUTO</Badge> 级（纯分析/检索，不碰硬件），无需安全门控 /
          前置条件 / 回滚等可治理项，故此处只读展示；如需编辑技能元数据见「仪器控制 IC」。
        </p>
        {toolsQ.isPending && <Spinner />}
        {toolsQ.error && <ErrorNote error={toolsQ.error} />}
        {toolsQ.data?.degraded && <DegradedNote what="工具目录" />}
        {toolsQ.data && !toolsQ.data.degraded && (
          myTools.length === 0 ? (
            <EmptyNote label="该 agent 暂无可列出的工具。" />
          ) : (
            <ul className="grid grid-cols-1 gap-0.5 sm:grid-cols-2">
              {myTools.map((t) => (
                <li
                  key={t.name}
                  className="flex items-center justify-between gap-2 rounded-md px-2 py-1.5 hover:bg-mast-bg/60"
                >
                  <span className="truncate font-mono text-xs text-mast-text">{t.name}</span>
                  <div className="flex shrink-0 items-center gap-1">
                    {t.composition_level && <Badge>{t.composition_level}</Badge>}
                    <Badge tone={String(t.safety_level).toUpperCase() === "AUTO" ? "AUTO" : "WARN"}>
                      {String(t.safety_level).toUpperCase()}
                    </Badge>
                  </div>
                </li>
              ))}
            </ul>
          )
        )}
      </Card>
    </div>
  );
}
