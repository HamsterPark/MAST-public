import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useStickyTab } from "@/hooks/useStickyTab";
import type { ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import {
  Badge,
  Card,
  DegradedNote,
  EmptyNote,
  ErrorNote,
  Section,
  Spinner,
} from "@/components/ui";
import {
  Button,
  Field,
  SelectField,
  SubTabs,
  useToast,
} from "@/components/controls";
import { DataTable } from "@/components/DataTable";
import {
  AGENTS,
  Avatar,
  PIPELINE,
  SUP_ID,
  SUPERVISOR,
  THINKING_LEVELS,
  agentDef,
  agentLabel,
  safetyTone,
} from "@/components/agents/registry";
import {
  ALL_VIEW as CONTEXT_ALL,
  ContextInjectionView,
} from "@/components/agents/ContextInjectionView";
import { TopologyGraph, type AgentLive } from "@/components/agents/TopologyGraph";
import { AgentChatPanel } from "@/components/agents/AgentChatPanel";
import { GroupActivityFeed } from "@/components/agents/GroupActivityFeed";
import { InterruptsModal } from "@/components/agents/InterruptsModal";
import { RunTaskPanel } from "@/components/agents/RunTaskPanel";
import { BackgroundRunsPanel } from "@/components/agents/BackgroundRunsPanel";
import { PendingActivationsPanel } from "@/components/agents/PendingActivationsPanel";
import { ArtifactEditor } from "@/components/agents/ArtifactEditor";
// Replaces ArtifactConnectionGraph (2026-07-11): that drew a single-writer
// bipartite blob from a model that was not true of the real pipeline.
import { ArtifactFlowGraph } from "@/components/agents/ArtifactFlowGraph";

// ════════════════════════════════════════════════════════════════════════════
// Agents inspector — FULL parity rebuild of the old Gradio 智能体 surface.
//
// Sub-tabs (flat React state via <SubTabs>, never nested gr.Tabs → no freeze):
//   对话     — DEFAULT/primary. INTERACTIVE multi-agent conversation (POST
//              /api/agents/run-task SSE + live POST /api/agents/{target}/interject):
//              one scrolling thread with a persistent bottom input — first message
//              starts the task, later messages steer the run live;
//              inline handoffs + HITL approve/reject/edit + abort; legacy per-agent
//              SUP private chat kept as a secondary mode
//   拓扑     — supervisor + 7-agent topology graph (live model/active/threads)
//   代理对话 — per-agent private chat threads
//   检查器   — per-agent model+thinking picker + tool catalog DataTable (IC ~224)
//   制品     — artifact list + R/W permission matrix + per-artifact operator EDITOR
//              (save/revert/history/diff/export via /api/artifacts/{id}/edit…)
//   QA 助手  — scoped single-turn READ-ONLY helper (POST /api/qa)
//
// Plus a global HITL interrupts modal (badge-gated) reachable from the header.
// All reads render loading / error / degraded / empty. All writes toast.
// ════════════════════════════════════════════════════════════════════════════

const ROSTER = [SUP_ID, ...AGENTS.map((a) => a.id)];

// ── shared read hooks ───────────────────────────────────────────────────────

function useSnapshot() {
  return useQuery({
    queryKey: ["agents", "snapshot"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/snapshot");
      if (error) throw error;
      return data;
    },
    refetchInterval: 4000,
  });
}

function useAgentModels() {
  return useQuery({
    queryKey: ["agents", "models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/models");
      if (error) throw error;
      return data;
    },
  });
}

function useAgentTools() {
  return useQuery({
    queryKey: ["agents", "tools"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/tools");
      if (error) throw error;
      return data;
    },
    refetchInterval: (q) => (q.state.data && !q.state.data.degraded ? false : 4000),
  });
}

// ── live overlay (model/thinking/active/held/threads) keyed by agent id ──────

function useLiveByAgent(): {
  live: Record<string, AgentLive>;
  snapshot: ReturnType<typeof useSnapshot>;
} {
  const snapshot = useSnapshot();
  const models = useAgentModels();
  const live = useMemo(() => {
    const m: Record<string, AgentLive> = {};
    // model seam (near-static defaults)
    for (const a of models.data?.agents ?? []) {
      m[a.agent_id] = { model: a.model, thinking: a.thinking };
    }
    // snapshot maps override with live values
    const snap = snapshot.data;
    if (snap) {
      for (const [id, model] of Object.entries(snap.models ?? {})) {
        m[id] = { ...(m[id] ?? {}), model };
      }
      for (const [id, th] of Object.entries(snap.thinking ?? {})) {
        m[id] = { ...(m[id] ?? {}), thinking: th };
      }
      for (const [id, held] of Object.entries(snap.holds ?? {})) {
        m[id] = { ...(m[id] ?? {}), held };
      }
      for (const [id, n] of Object.entries(snap.threads_index ?? {})) {
        m[id] = { ...(m[id] ?? {}), threads: n };
      }
      for (const node of snap.agents ?? []) {
        m[node.id] = {
          ...(m[node.id] ?? {}),
          model: node.model ?? m[node.id]?.model,
          thinking: node.thinking ?? m[node.id]?.thinking,
          active: node.active,
          held: node.held,
          threads: node.thread_count,
        };
      }
      if (snap.active_agent_id) {
        m[snap.active_agent_id] = { ...(m[snap.active_agent_id] ?? {}), active: true };
      }
    }
    return m;
  }, [models.data, snapshot.data]);
  return { live, snapshot };
}

type SubTab =
  | "topology" | "chat" | "background" | "waking"
  | "agentchat" | "inspector" | "context" | "workspace" | "editor";

// 子页表抽成一份数据，因为它同时是**两样东西**：渲染 tab 条用的 {id,label}，
// 以及 useStickyTab 的合法 id 名单。另抄一份名单出来是行不通的 —— 抄的那份会和
// tab 条漂开，而漂开的症状是「某一个子页记不住」，一个没人会去查存储键的症状。
const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: "chat", label: "对话" },
  { id: "background", label: "后台" },
  // 待唤醒: a parked agent that nobody can see is indistinguishable from
  // a hung system, so the scheduler's parking mechanism is not usable
  // without this tab existing.
  { id: "waking", label: "待唤醒" },
  { id: "topology", label: "拓扑" },
  { id: "agentchat", label: "代理对话" },
  { id: "inspector", label: "Inspector" },
  // 上下文注入：只读，回答「不同角色有针对性的注入吗」与「这一次调用由什么
  // 构成」。编辑仍在 高级管理 → 上下文注入（PIN 门后面）。
  { id: "context", label: "上下文注入" },
  { id: "workspace", label: "对象总览" },
  { id: "editor", label: "对象编辑" },
];

export default function AgentsPage() {
  // 从代理对话离开再回来，应该还停在代理对话栏目。
  // 路由元素由 AppLayout 的 <Outlet/> 渲染，每次回来都是一次全新挂载，
  // 所以 useState("chat") 每次都把他打回「对话」。
  const [tab, setTab] = useStickyTab<SubTab>(
    "agents", SUB_TABS.map((t) => t.id), "chat");
  const [selected, setSelected] = useState<string>("instrument_control");
  // 「上下文注入」子页选的是哪个角色。名单从 ROSTER 派生 —— AdminPage 里那份
  // 手抄的六人名单（漏了 research_director）是这个仓的反面教材。
  const CONTEXT_PICKS = useMemo(
    () => [{ id: CONTEXT_ALL, label: "全部" },
           ...ROSTER.map((id) => ({ id, label: agentDef(id).cn }))],
    []);
  const [ctxPick, setCtxPick] = useStickyTab<string>(
    "agents.context.pick", CONTEXT_PICKS.map((p) => p.id), CONTEXT_ALL);
  const [interruptAgent, setInterruptAgent] = useState<string | null>(null);
  // shared between 对象总览 and 对象编辑 (the old workspace card's 打开编辑器
  // button navigated to the editor sub-tab focused on the selected artifact).
  const [editArtifact, setEditArtifact] = useState<string | null>(null);

  const { live, snapshot } = useLiveByAgent();
  const pending = snapshot.data?.pending_interrupt_count ?? 0;

  return (
    <div className="space-y-6">
      <Section
        title="智能体"
        actions={
          <div className="flex items-center gap-2">
            {snapshot.data && (
              <Badge tone={snapshot.data.session_active ? "AUTO" : "INFO"}>
                {snapshot.data.session_active ? "会话进行中" : "空闲"}
              </Badge>
            )}
            <Button
              variant={pending > 0 ? "danger" : "default"}
              onClick={() => setInterruptAgent(selected || SUP_ID)}
            >
              人机交互{pending > 0 ? ` (${pending})` : ""}
            </Button>
          </div>
        }
      >
        <SubTabs<SubTab> tabs={SUB_TABS} value={tab} onChange={setTab} />

        {tab === "chat" && <SupervisorChatView snapshot={snapshot} onOpenInterrupts={() => setInterruptAgent(SUP_ID)} />}
        {tab === "background" && <BackgroundRunsPanel />}
        {tab === "context" && (
          <ContextInjectionView picks={CONTEXT_PICKS} pick={ctxPick} onPick={setCtxPick} />
        )}
        {tab === "waking" && <PendingActivationsPanel />}
        {tab === "topology" && (
          <TopologyView live={live} snapshot={snapshot} selected={selected} onSelect={setSelected} />
        )}
        {tab === "agentchat" && (
          <AgentChatView selected={selected} onSelect={setSelected} />
        )}
        {tab === "inspector" && (
          <InspectorView live={live} selected={selected} onSelect={setSelected} />
        )}
        {tab === "workspace" && (
          <WorkspaceView
            onEdit={(id) => {
              setEditArtifact(id);
              setTab("editor");
            }}
          />
        )}
        {tab === "editor" && (
          <EditorView selected={editArtifact} onSelect={setEditArtifact} />
        )}
      </Section>

      <InterruptsModal
        open={interruptAgent != null}
        agentId={interruptAgent}
        onClose={() => setInterruptAgent(null)}
      />
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// 拓扑 — topology graph + per-agent live status strip
// ════════════════════════════════════════════════════════════════════════════

function TopologyView({
  live,
  snapshot,
  selected,
  onSelect,
}: {
  live: Record<string, AgentLive>;
  snapshot: ReturnType<typeof useSnapshot>;
  selected: string;
  onSelect: (id: string) => void;
}) {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const snap = snapshot.data;

  // hold/release per agent — LIVE-only orchestrator control, degrade-safe.
  const hold = useMutation({
    mutationFn: async (vars: { agent_id: string; held: boolean }) => {
      const opts = { params: { path: { agent_id: vars.agent_id } } };
      const { data, error } = vars.held
        ? await api.POST("/api/agents/{agent_id}/hold", opts)
        : await api.POST("/api/agents/{agent_id}/release", opts);
      if (error) throw error;
      return data;
    },
    onSuccess: (data, vars) => {
      const verb = vars.held ? "暂停" : "恢复";
      if (data?.ok && !data.degraded) toast(`已${verb} ${agentLabel(vars.agent_id)}`, "ok");
      else toast(`${verb}未生效（${data?.detail ?? "后端降级"}）`, "err");
      qc.invalidateQueries({ queryKey: ["agents", "snapshot"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  return (
    <div className="space-y-5">
      {node}
      {/* header — 多智能体拓扑 (mirrors the old AGTopologyView title row) */}
      <div className="flex flex-wrap items-baseline gap-3">
        <h2 className="text-lg font-semibold tracking-tight">多智能体拓扑</h2>
        <span className="text-xs text-mast-muted">
          用户 → 编排协调 → 六 agent 流水线 · BUF 侧通道
        </span>
      </div>
      {snapshot.isPending && <Spinner />}
      {snapshot.isError && <ErrorNote error={snapshot.error} />}
      {snap && snap.degraded && <DegradedNote what="智能体拓扑快照" />}

      <TopologyGraph
        selected={selected}
        live={live}
        onPick={onSelect}
        onToggleHold={(id, held) => hold.mutate({ agent_id: id, held })}
      />

      {snap && (
        <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3 lg:grid-cols-6">
          <StatCell label="后端" value={snap.backend || "—"} />
          <StatCell label="会话" value={snap.session_active ? "活跃" : "空闲"} />
          <StatCell label="活动智能体" value={snap.active_agent_id ? agentLabel(snap.active_agent_id) : "—"} />
          <StatCell label="制品数" value={String(snap.artifacts_count ?? 0)} />
          <StatCell label="待中断" value={String(snap.pending_interrupt_count ?? 0)} />
          <StatCell label="插话次数" value={String(snap.interject_count ?? 0)} />
        </div>
      )}

      {/* per-agent live status strip */}
      <div>
        <h4 className="mb-2 text-sm font-medium text-mast-muted">逐智能体实时状态</h4>
        <div className="overflow-x-auto rounded-lg border border-mast-border">
          <table className="w-full text-sm">
            <thead className="border-b border-mast-border text-left text-xs text-mast-muted">
              <tr>
                <th className="px-3 py-2">智能体</th>
                <th className="px-3 py-2">模型</th>
                <th className="px-3 py-2">思考</th>
                <th className="px-3 py-2">状态</th>
                <th className="px-3 py-2 text-right tabular-nums">线程深度</th>
                <th className="px-3 py-2 text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {ROSTER.map((id) => {
                const lv = live[id] ?? {};
                return (
                  <tr
                    key={id}
                    className={
                      "cursor-pointer border-b border-mast-border/50 hover:bg-mast-bg/60 " +
                      (selected === id ? "bg-mast-accent/5" : "")
                    }
                    onClick={() => onSelect(id)}
                  >
                    <td className="px-3 py-2">
                      <span className="flex items-center gap-2">
                        <Avatar id={id} size={18} active={lv.active} />
                        {agentLabel(id)}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{lv.model ?? "—"}</td>
                    <td className="px-3 py-2">
                      <Badge tone={lv.thinking ? "AUTO" : "default"}>{lv.thinking ?? "—"}</Badge>
                    </td>
                    <td className="px-3 py-2">
                      {lv.active ? (
                        <Badge tone="AUTO">运行中</Badge>
                      ) : lv.held ? (
                        <Badge tone="WARN">已暂留</Badge>
                      ) : (
                        <span className="text-xs text-mast-muted">空闲</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">{lv.threads ?? 0}</td>
                    <td className="px-3 py-2 text-right" onClick={(e) => e.stopPropagation()}>
                      <button
                        disabled={hold.isPending && hold.variables?.agent_id === id}
                        onClick={() => hold.mutate({ agent_id: id, held: !lv.held })}
                        className={
                          "rounded-md border px-2.5 py-1 text-xs font-medium disabled:opacity-50 " +
                          (lv.held
                            ? "border-mast-auto-border text-mast-auto hover:bg-mast-auto-bg"
                            : "border-mast-warn-border text-mast-warn hover:bg-mast-warn-bg")
                        }
                      >
                        {hold.isPending && hold.variables?.agent_id === id
                          ? "…"
                          : lv.held
                            ? "恢复"
                            : "暂停"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <Card className="px-3 py-2">
      <div className="text-xs text-mast-muted">{label}</div>
      <div className="mt-0.5 truncate font-mono text-sm tabular-nums">{value}</div>
    </Card>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// 对话 — INTERACTIVE multi-agent conversation (run-task SSE + live interject) with
//   a persistent bottom input + handoff timeline; legacy SUP private chat as a
//   secondary mode.
// ════════════════════════════════════════════════════════════════════════════

type ChatMode = "runtask" | "supchat";

const CHAT_MODES: { id: ChatMode; label: string }[] = [
  { id: "runtask", label: "多智能体对话" },
  { id: "supchat", label: "编排器私聊" },
];

// Per-event-kind icon + dot colour + badge tone for the 交接事件时间线
// (— a wall of identical dots was unscannable).
const TIMELINE_STYLE: Record<
  string,
  { icon: string; label?: string; dot: string; tone: React.ComponentProps<typeof Badge>["tone"] }
> = {
  start: { icon: "🚀", label: "开始", dot: "bg-sky-400", tone: "INFO" },
  handoff: { icon: "🔀", label: "交接", dot: "bg-violet-400", tone: "AUTO" },
  // A parallel fan-out is the ONE event the operator most needs to spot, and it
  // was landing on the anonymous fallback bullet — i.e. straight back into the
  // "wall of identical dots" #101 was about. The dispatch kind is new (2026-07-11);
  // the style map was not updated with it.
  dispatch: { icon: "‖", label: "并行下发", dot: "bg-cyan-400", tone: "INFO" },
  held: { icon: "⏸️", label: "暂停", dot: "bg-amber-400", tone: "WARN" },
  resumed: { icon: "▶️", label: "继续", dot: "bg-emerald-400", tone: "AUTO" },
  interrupt: { icon: "🛡️", label: "待批准", dot: "bg-rose-400", tone: "DANGEROUS" },
  dangerous: { icon: "🛡️", label: "待批准", dot: "bg-rose-400", tone: "DANGEROUS" },
  buffer_hitl: { icon: "🛡️", label: "待批准", dot: "bg-rose-400", tone: "DANGEROUS" },
  workflow_human: { icon: "🙋", label: "待人工", dot: "bg-rose-300", tone: "WARN" },
  done: { icon: "✅", label: "完成", dot: "bg-emerald-500", tone: "AUTO" },
  aborted: { icon: "🛑", label: "已中止", dot: "bg-rose-500", tone: "DANGEROUS" },
  error: { icon: "⚠️", label: "出错", dot: "bg-rose-500", tone: "DANGEROUS" },
};
const TIMELINE_FALLBACK: {
  icon: string;
  label?: string;
  dot: string;
  tone: React.ComponentProps<typeof Badge>["tone"];
} = {
  icon: "•",
  dot: "bg-mast-accent",
  tone: "INFO",
};

function SupervisorChatView({
  snapshot,
  onOpenInterrupts,
}: {
  snapshot: ReturnType<typeof useSnapshot>;
  onOpenInterrupts: () => void;
}) {
  const { node } = useToast();
  const snap = snapshot.data;
  const task = snap?.active_task ?? null;
  // Primary action = the REAL multi-agent orchestrator runner (run-task SSE).
  // The legacy per-agent SUP private chat is kept reachable as a secondary mode.
  // 第二层也记 —— 停在「编排器私聊」的人离开一次就被弹回「多智能体对话」，
  // 和 #32 抱怨的是同一件事，只是深了一层。
  const [mode, setMode] = useStickyTab<ChatMode>(
    "agents.chat.mode", CHAT_MODES.map((m) => m.id), "runtask");

  const handoffs = snap?.handoff_events ?? [];

  return (
    <div className="space-y-4">
      {node}
      {/* mode selector + status bar */}
      <Card className="space-y-2 px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="flex items-center gap-2 text-sm font-medium">
            <Avatar id={SUP_ID} size={20} /> 编排与协调
          </span>
          {task?.active ? (
            <Badge tone="AUTO">任务进行中</Badge>
          ) : (
            <Badge tone="INFO">无活动任务</Badge>
          )}
          <div className="ml-auto">
            <Button variant="default" onClick={onOpenInterrupts}>
              人机交互
            </Button>
          </div>
        </div>
        <SubTabs<ChatMode> tabs={CHAT_MODES} value={mode} onChange={setMode} />
        {mode === "runtask" ? (
          <p className="text-xs text-mast-muted">
            与 6 智能体编排器持续对话（POST /api/agents/run-task）：第一条消息下发任务，运行期间继续输入即可
            插话引导（POST /api/agents/&#123;target&#125;/interject），逐智能体进展、交接与 HITL 批准
            内联呈现；与下方「编排器私聊」（单代理对话）不同。
          </p>
        ) : (
          <p className="text-xs text-mast-muted">
            与编排智能体的单代理私聊线程（不触发多智能体编排）。
          </p>
        )}
      </Card>

      {/* PRIMARY: real multi-agent orchestrator runner */}
      {mode === "runtask" && (
        <RunTaskPanel conversationId={null} />
      )}

      {/* SECONDARY: legacy per-agent supervisor private chat (kept as-is) */}
      {mode === "supchat" && <AgentChatPanel agentId={SUP_ID} />}

      {/* handoff event timeline (from snapshot) — per-kind icon + colour so the
          operator can scan it at a glance. */}
      <div>
        <h4 className="mb-2 text-sm font-medium text-mast-muted">交接事件时间线</h4>
        {handoffs.length === 0 ? (
          <EmptyNote label="暂无交接事件（编排器空闲或未接入）。" />
        ) : (
          <ol className="space-y-1.5 border-l border-mast-border pl-4">
            {handoffs.map((ev, i) => {
              const st = TIMELINE_STYLE[ev.kind ?? ""] ?? TIMELINE_FALLBACK;
              return (
                <li key={i} className="relative text-sm">
                  <span
                    className={`absolute -left-[21px] top-1.5 h-2 w-2 rounded-full ${st.dot}`}
                  />
                  <span className="mr-2 font-mono text-xs text-mast-muted">
                    {ev.t != null ? new Date(ev.t * 1000).toLocaleTimeString() : "—"}
                  </span>
                  {ev.kind && (
                    <Badge tone={st.tone}>
                      {st.icon} {st.label ?? ev.kind}
                      {/* the fan-out width: the event carries targets[], and the
                          count is the whole point of a parallel dispatch */}
                      {(ev.targets?.length ?? 0) > 1 && ` ×${ev.targets!.length}`}
                    </Badge>
                  )}
                  <span className="ml-2 text-mast-text">{ev.text}</span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// 代理对话 — per-agent private chat threads
// ════════════════════════════════════════════════════════════════════════════

function AgentChatView({ selected, onSelect }: { selected: string; onSelect: (id: string) => void }) {
  // 编排器(_supervisor) is selectable here too so its 群聊 contributions are
  // viewable per-agent like everyone else (the bridge covers it: agent_activity
  // ("_supervisor") returns its group messages). Its 1:1 private chat lives in the
  // 对话 → 编排器私聊 tab, so the private-chat panel below is shown only for the
  // pipeline agents.
  const isSup = selected === SUP_ID;
  const { toast, node } = useToast();
  const [interjectText, setInterjectText] = useState("");

  // 插话 — queue an operator interjection for the RUNNING supervisor; it is
  // delivered as an operator message on the supervisor's next super-step.
  // LIVE-only; degrade-safe (typed no-op when no live app / not running).
  const interject = useMutation({
    mutationFn: async (text: string) => {
      const { data, error } = await api.POST("/api/agents/{agent_id}/interject", {
        params: { path: { agent_id: SUP_ID } },
        body: { text },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok && !data.degraded) {
        toast("已插话，将于编排器下一步送达", "ok");
        setInterjectText("");
      } else {
        toast(`插话未生效（${data?.detail ?? "编排器未在运行 / 后端降级"}）`, "err");
      }
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  function submitInterject() {
    const t = interjectText.trim();
    if (!t || interject.isPending) return;
    interject.mutate(t);
  }

  return (
    <div className="space-y-4">
      {node}
      <div className="flex flex-wrap gap-2">
        {[SUPERVISOR, ...AGENTS].map((a) => (
          <button
            key={a.id}
            onClick={() => onSelect(a.id)}
            className={
              "flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm " +
              (selected === a.id
                ? "border-mast-accent bg-mast-accent/10 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            <Avatar id={a.id} size={16} /> {a.cn}
          </button>
        ))}
      </div>

      {/* 插话 — operator interjection to the running supervisor */}
      <Card className="space-y-2 px-4 py-3">
        <Field
          label="插话"
          hint="向正在运行的编排智能体注入一条用户消息（不打断当前回合，于下一步送达）。"
        >
          <div className="flex items-end gap-2">
            <textarea
              value={interjectText}
              onChange={(e) => setInterjectText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submitInterject();
                }
              }}
              placeholder="例：优先处理 Au(111)，跳过当前样品的剩余扫描"
              rows={2}
              disabled={interject.isPending}
              className="flex-1 resize-none rounded-md border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent disabled:opacity-50"
            />
            <Button
              variant="primary"
              onClick={submitInterject}
              disabled={!interjectText.trim() || interject.isPending}
            >
              {interject.isPending ? "插话中…" : "插话"}
            </Button>
          </div>
        </Field>
      </Card>

      {/* 群聊中的活动 — what this agent said/did in multi-agent (群聊) runs, the
          per-agent view of the team conversation (the bridge restored). Covers
          编排器(_supervisor) too — its routing/decision turns are persisted with
          agent_id="_supervisor" and surface here. */}
      <GroupActivityFeed agentId={selected} />

      {isSup ? (
        <Card className="px-4 py-3 text-sm text-mast-muted">
          编排器的 1:1 私聊在「对话 → 编排器私聊」标签。此处展示的是它在多智能体群聊中的发言。
        </Card>
      ) : (
        <AgentChatPanel agentId={selected} />
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// 检查器 — per-agent model+thinking picker + tool catalog DataTable
// ════════════════════════════════════════════════════════════════════════════

type ToolRow = {
  name: string;
  safety_level: string;
  composition_level?: string | null;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const TOOL_COLUMNS: ColumnDef<ToolRow, any>[] = [
  {
    accessorKey: "name",
    header: "工具名",
    cell: (c) => <span className="font-mono text-xs">{c.getValue() as string}</span>,
  },
  {
    accessorKey: "safety_level",
    header: "安全等级",
    cell: (c) => {
      const sl = (c.getValue() as string) || "AUTO";
      return <Badge tone={safetyTone(sl)}>{sl}</Badge>;
    },
  },
  {
    accessorKey: "composition_level",
    header: "组合层级",
    cell: (c) => {
      const lvl = c.getValue() as string | null | undefined;
      return lvl ? (
        <span className="font-mono text-xs text-mast-muted">{lvl}</span>
      ) : (
        <span className="text-xs text-mast-muted">—</span>
      );
    },
  },
];

function InspectorView({
  live,
  selected,
  onSelect,
}: {
  live: Record<string, AgentLive>;
  selected: string;
  onSelect: (id: string) => void;
}) {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const tools = useAgentTools();
  const models = useQuery({
    queryKey: ["config", "models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/config/models");
      if (error) throw error;
      return data;
    },
  });
  const [filter, setFilter] = useState("");

  const toolsByAgent = useMemo(() => {
    const m = new Map<string, ToolRow[]>();
    for (const cat of tools.data?.agents ?? []) {
      m.set(
        cat.agent_id,
        (cat.tools ?? []).map((t) => ({
          name: t.name,
          safety_level: t.safety_level,
          composition_level: t.composition_level,
        })),
      );
    }
    return m;
  }, [tools.data]);

  const override = useMutation({
    mutationFn: async (vars: { agent_id: string; model?: string; thinking?: string }) => {
      const { data, error } = await api.POST("/api/agents/{agent_id}/model-override", {
        params: { path: { agent_id: vars.agent_id } },
        body: { model: vars.model ?? null, thinking: vars.thinking ?? null },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok && !data.degraded) toast("已保存并将于下个任务生效", "ok");
      else toast("后端未接入 / 降级，未生效", "err");
      qc.invalidateQueries({ queryKey: ["agents", "snapshot"] });
      qc.invalidateQueries({ queryKey: ["agents", "models"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  const a = agentDef(selected);
  const lv = live[selected] ?? {};
  const selectedTools = toolsByAgent.get(selected) ?? [];

  // model options from /api/config/models (fall back to current live model only)
  const modelOptions = useMemo(() => {
    const opts: { value: string; label: string }[] = [];
    const seen = new Set<string>();
    for (const m of models.data?.models ?? []) {
      const id = m.alias;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      opts.push({ value: id, label: m.description ? `${id} · ${m.description}` : id });
    }
    if (lv.model && !seen.has(String(lv.model))) {
      opts.unshift({ value: String(lv.model), label: String(lv.model) });
    }
    if (opts.length === 0 && lv.model) opts.push({ value: String(lv.model), label: String(lv.model) });
    return opts;
  }, [models.data, lv.model]);

  return (
    <div className="space-y-5">
      {node}
      {/* agent picker chips */}
      <div className="flex flex-wrap gap-2">
        {ROSTER.map((id) => (
          <button
            key={id}
            onClick={() => onSelect(id)}
            className={
              "flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm " +
              (selected === id
                ? "border-mast-accent bg-mast-accent/10 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            <Avatar id={id} size={16} active={lv && live[id]?.active} /> {agentDef(id).cn}
          </button>
        ))}
      </div>

      {/* model + thinking picker */}
      <Card className="space-y-3 px-4 py-4">
        <div className="flex items-center gap-3">
          <Avatar id={selected} size={26} active={lv.active} />
          <div>
            <div className="font-mono text-sm font-semibold">{selected}</div>
            <div className="text-xs text-mast-muted">{a.role}</div>
          </div>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Field label="模型" hint="保存后重建编排器，下个任务生效（不影响进行中的会话）">
            <SelectField
              value={String(lv.model ?? "")}
              onChange={(v) => override.mutate({ agent_id: selected, model: v })}
              options={
                modelOptions.length
                  ? modelOptions
                  : [{ value: String(lv.model ?? ""), label: String(lv.model ?? "（未知）") }]
              }
            />
          </Field>
          <Field
            label="思考强度"
            hint="对 Claude 映射为扩展思考预算；Kimi/DeepSeek 等推理模型自带推理、固定最高强度。"
          >
            <div className="flex gap-1">
              {THINKING_LEVELS.map((tl) => (
                <button
                  key={tl}
                  onClick={() => override.mutate({ agent_id: selected, thinking: tl })}
                  className={
                    "rounded-md px-3 py-1.5 font-mono text-xs " +
                    ((lv.thinking ?? "").toLowerCase() === tl
                      ? "bg-mast-accent/25 text-mast-accent"
                      : "border border-mast-border text-mast-muted hover:text-mast-text")
                  }
                >
                  {tl}
                </button>
              ))}
            </div>
          </Field>
        </div>
      </Card>

      {/* tool catalog */}
      <div>
        <div className="mb-3 flex items-center justify-between gap-3">
          <h4 className="text-sm font-medium">
            工具目录 · {a.cn}
            <span className="ml-2 text-xs text-mast-muted">共 {selectedTools.length} 个</span>
          </h4>
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="搜索工具名 / 安全等级 / 层级…"
            className="w-64 rounded border border-mast-border bg-mast-bg px-3 py-1.5 text-sm text-mast-text outline-none focus:border-mast-accent"
          />
        </div>
        {tools.isPending && <Spinner />}
        {tools.isError && <ErrorNote error={tools.error} />}
        {tools.data && tools.data.degraded && <DegradedNote what="工具目录" />}
        {tools.data && !tools.data.degraded && (
          <DataTable
            data={selectedTools}
            columns={TOOL_COLUMNS}
            globalFilter={filter}
            empty="该智能体暂无工具（或目录尚未预热）。"
          />
        )}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// 对象总览 — 共享对象 · Workspace: R/W permission matrix + artifact list + the
//   selected-artifact detail card (read-only overview). Mirrors the old
//   AGWorkspaceView: title "共享对象 · Workspace", access matrix with an
//   operator column, per-artifact detail with a "打开编辑器" button that jumps
//   to the 对象编辑 sub-tab.
// ════════════════════════════════════════════════════════════════════════════

// Permission-matrix columns = the 6 pipeline agents. The old table also carried
// operator / SUP / BUF columns, but in the REAL flow model none of them read or
// write a pipeline artifact (the supervisor routes; BUF narrates the buffer;
// the operator edits through the editor, not through an agent edge) — so those
// columns were three permanently-empty stripes pretending to carry information.
const WS_COLS: string[] = [...PIPELINE];

function WorkspaceView({ onEdit }: { onEdit: (id: string) => void }) {
  const artifacts = useQuery({
    queryKey: ["artifacts"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/artifacts");
      if (error) throw error;
      return data;
    },
    refetchInterval: 6000,
  });
  const perms = useQuery({
    queryKey: ["artifacts", "permissions"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/artifacts/permissions");
      if (error) throw error;
      return data;
    },
  });

  const entries = artifacts.data?.artifacts ?? [];
  const matrix = perms.data?.permissions ?? [];
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  // ── 制品分组 (+ #29, 2026-07-28) ────────────────────────────
  //
  // This used to compute "which classes are still 待产出" in the browser, like
  // this:
  //
  //     const producedIds = new Set(entries.map((e) => e.id));   // "draft:Au111_v001"
  //     matrix.filter((r) => !producedIds.has(r.artifact_id))    // "draft"
  //
  // Those are two different id spaces — a produced id is per-FILE and always
  // contains a colon, a permission row's id is the CLASS and never does — so
  // the filter removed nothing, ever. All nine classes rendered as 待产出 on
  // every load no matter what had been produced. (Measured on the real repo at
  // the time of the report: six of the nine stores had content; the panel said
  // nothing had been produced.) The twin component below, EditorView, had
  // already been fixed to compare on `kind`; this copy was missed.
  //
  // The class state is no longer derived here at all. `groups` comes from
  // GET /api/artifacts, which asks each class its OWN store — the only way the
  // four non-file classes (文献库 / 实验记录 / 长期记忆 / 视觉缓冲) can report
  // themselves at all, since nothing walks a directory to find them.
  const sessionActive = !!artifacts.data?.session_active;
  const groups = useMemo(() => {
    const byClass = new Map<string, typeof entries>();
    for (const e of entries) {
      const k = e.kind ?? "";
      const bucket = byClass.get(k);
      if (bucket) bucket.push(e);
      else byClass.set(k, [e]);
    }
    return (artifacts.data?.groups ?? []).map((g) => ({
      ...g,
      items: byClass.get(g.artifact_id) ?? [],
      producer:
        (matrix.find((r) => r.artifact_id === g.artifact_id)?.writers ?? []).find(
          (w) => w !== "operator",
        ) ?? null,
    }));
  }, [entries, matrix, artifacts.data?.groups]);

  // A group starts open when it holds a manageable number of files. `scan_files`
  // is flagged high_volume by the backend (one row per SaveScan — a night's run
  // is hundreds), which is the pile that made the list unreadable: "如果可以算
  // 的话，这个制品区域就应该分目录，否则看起来太乱了" . Classes with no
  // enumerable files (the DB/registry ones) have nothing to expand.
  function isOpen(g: { artifact_id: string; high_volume: boolean; items: unknown[] }) {
    const explicit = openGroups[g.artifact_id];
    if (explicit !== undefined) return explicit;
    return g.items.length > 0 && !g.high_volume && g.items.length <= 6;
  }
  const pendingCount = groups.filter((g) => !g.produced).length;

  // (The old graphArtifacts memo fed ArtifactConnectionGraph, which is gone —
  // ArtifactFlowGraph consumes the permission rows directly, since those now
  // carry the label / kind / store / multi_writer the picture needs.)

  // permission lookup for the selected-artifact detail card. The selection is a
  // per-FILE id ("draft:Au111_v001"); permissions are keyed by CLASS ("draft"),
  // so resolve through the entry's kind. Matching the file id against the class
  // id directly — which is what this line used to do — never hit, so the 访问权限
  // panel read "无权限信息" for every produced artifact. Same id-space confusion
  // as the roster bug above, in a second place.
  const selEntry = entries.find((e) => e.id === selectedId);
  const selRow = matrix.find((r) => r.artifact_id === (selEntry?.kind ?? selectedId));
  const writers = new Set(selRow?.writers ?? []);
  const readers = new Set([...(selRow?.readers ?? []), ...(selRow?.writers ?? [])]);

  return (
    <div className="space-y-5">
      {/* header — 共享对象 · Workspace */}
      <div className="flex flex-wrap items-baseline gap-3">
        <h2 className="text-lg font-semibold tracking-tight">共享对象 · Workspace</h2>
        <span className="text-xs text-mast-muted">
          本次运行的客体 · agent 和用户均可直接读写 · 写权限收紧
        </span>
        {artifacts.data && (
          <span className="ml-auto font-mono text-xs text-mast-muted">
            {artifacts.data.count} 对象
          </span>
        )}
      </div>

      {/* 数据流图 (redrawn 2026-07-11, ): the REAL directed flow —
          production is the skeleton (solid, in the writer's hue), consumption
          the detail (dashed, on hover). Multi-writer artifacts are called out;
          the two artifacts everyone touches are drawn as a shared bus / side
          rail rather than as nodes with a dozen edges each. */}
      <div>
        <h4 className="mb-2 text-sm font-medium">
          数据流 · 谁写 → 谁读
          <span className="ml-2 text-xs font-normal text-mast-muted">
            · 悬停聚焦 · 单击选中 · 双击打开编辑器
          </span>
        </h4>
        {perms.isPending && <Spinner />}
        {perms.isError && <ErrorNote error={perms.error} />}
        {perms.data && perms.data.degraded && <DegradedNote what="数据流图" />}
        {perms.data && !perms.data.degraded && matrix.length === 0 && (
          <EmptyNote label="无产物定义。" />
        )}
        {matrix.length > 0 && (
          <ArtifactFlowGraph
            permissions={matrix}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onOpen={onEdit}
          />
        )}
      </div>

      {/* R/W matrix — the precise lookup that complements the flow graph.
          A grid is still the fastest way to answer "may agent X write artifact
          Y?"; the graph answers "where does the data go". Both read from the
          same (now truthful) flow model. */}
      <div>
        <h4 className="mb-2 text-sm font-medium">
          读写矩阵
          <span className="ml-2 text-xs font-normal text-mast-muted">
            · W=写入 · R=读取 · 点击行聚焦
          </span>
        </h4>
        {perms.isPending && <Spinner />}
        {perms.isError && <ErrorNote error={perms.error} />}
        {perms.data && perms.data.degraded && <DegradedNote what="读写矩阵" />}
        {perms.data && !perms.data.degraded && matrix.length === 0 && (
          <EmptyNote label="无产物定义。" />
        )}
        {matrix.length > 0 && (
          <div className="overflow-x-auto rounded-lg border border-mast-border">
            <table className="w-full text-sm">
              <thead className="border-b border-mast-border text-left text-xs text-mast-muted">
                <tr>
                  <th className="px-3 py-2">产物</th>
                  {WS_COLS.map((id) => (
                    <th key={id} className="px-2 py-2 text-center">
                      {agentDef(id).short}
                    </th>
                  ))}
                  <th className="px-3 py-2 text-left">存储位置</th>
                </tr>
              </thead>
              <tbody>
                {matrix.map((row) => {
                  const w = new Set(row.writers ?? []);
                  const r = new Set(row.readers ?? []);
                  const isSel = row.artifact_id === selectedId;
                  return (
                    <tr
                      key={row.artifact_id}
                      onClick={() => setSelectedId(row.artifact_id)}
                      className={
                        "cursor-pointer border-b border-mast-border/50 hover:bg-mast-bg/60 " +
                        (isSel ? "bg-mast-accent-soft" : "")
                      }
                    >
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-1.5">
                          <span className="text-mast-text">{row.label || row.artifact_id}</span>
                          {row.multi_writer && (
                            <span
                              className="rounded border border-mast-accent px-1 font-mono text-[9px] text-mast-accent"
                              title={`多个智能体都写入：${(row.writers ?? []).map((x) => agentDef(x).cn).join("、")}`}
                            >
                              多写
                            </span>
                          )}
                        </div>
                        <div className="font-mono text-[10px] text-mast-muted">
                          {row.artifact_id}
                        </div>
                      </td>
                      {WS_COLS.map((id) => {
                        const canW = w.has(id);
                        const canR = r.has(id);
                        return (
                          <td key={id} className="px-2 py-2 text-center">
                            {canW && canR ? (
                              <span className="font-mono text-xs text-mast-warn" title="读 + 写">
                                W·R
                              </span>
                            ) : canW ? (
                              <span className="font-mono text-xs text-mast-warn" title="写入">
                                W
                              </span>
                            ) : canR ? (
                              <span className="font-mono text-xs text-mast-info" title="只读">
                                R
                              </span>
                            ) : (
                              <span className="text-mast-muted/40">·</span>
                            )}
                          </td>
                        );
                      })}
                      <td className="px-3 py-2 font-mono text-[10px] text-mast-muted">
                        {row.store || "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="px-3 py-2 text-xs text-mast-muted/80">
              <span className="text-mast-warn">W</span> 读 + 写 ·{" "}
              <span className="text-mast-info">R</span> 只读
            </p>
          </div>
        )}
      </div>

      {/* artifact list */}
      <div>
        <div className="mb-2 flex items-center gap-3">
          <h4 className="text-sm font-medium">工作区制品</h4>
          {artifacts.data && (
            <Badge tone={artifacts.data.session_active ? "AUTO" : "INFO"}>
              {artifacts.data.session_active ? "会话活跃" : "空闲"} · {artifacts.data.count} 项
            </Badge>
          )}
          {pendingCount > 0 && (
            <Badge tone="INFO">{pendingCount} 类尚未产出</Badge>
          )}
        </div>
        {artifacts.isPending && <Spinner />}
        {artifacts.isError && <ErrorNote error={artifacts.error} />}
        {artifacts.data && artifacts.data.degraded && <DegradedNote what="制品列表" />}
        {artifacts.data && !artifacts.data.degraded && groups.length === 0 && (
          <EmptyNote label="无产物定义。" />
        )}
        <p className="mb-2 text-xs text-mast-muted">
          按类型分组 · 点组标题展开/折叠 · 计数来自各自的存储（不是猜的）
          {sessionActive && " · 实验进行中"}
        </p>
        {/* #29 — grouped by class instead of one flat grid. A run saves one row
            per scan, so the .sxm pile used to bury the drafts, figures and plans
            underneath it: "sxm文件算不算工作区制品？可以算，但是…这个制品区域就
            应该分目录，否则看起来太乱了". */}
        <div className="space-y-2">
          {groups.map((g) => {
            const open = isOpen(g);
            const expandable = g.items.length > 0;
            return (
              <div
                key={g.artifact_id}
                className={
                  "overflow-hidden rounded-lg border " +
                  (g.produced ? "border-mast-border" : "border-dashed border-mast-border/60")
                }
              >
                <button
                  type="button"
                  disabled={!expandable}
                  onClick={() =>
                    setOpenGroups((s) => ({ ...s, [g.artifact_id]: !open }))
                  }
                  title={g.store}
                  className={
                    "flex w-full items-center gap-2 px-3 py-2 text-left " +
                    (expandable ? "hover:bg-mast-bg/60" : "cursor-default")
                  }
                >
                  <span className="w-3 shrink-0 font-mono text-xs text-mast-muted">
                    {expandable ? (open ? "▾" : "▸") : ""}
                  </span>
                  <span className="text-sm text-mast-text">{g.label || g.artifact_id}</span>
                  <span className="font-mono text-[10px] text-mast-muted">
                    {g.artifact_id}
                  </span>
                  {/* Honest three-way state. 待产出 is now only shown when the
                      store really is empty; 无法读取 covers a store we could not
                      open, which must never be rendered as "0 / 待产出". */}
                  {!g.known ? (
                    <Badge tone="WARN">无法读取</Badge>
                  ) : g.produced ? (
                    <Badge tone="AUTO">已产出 · {g.count}</Badge>
                  ) : (
                    <Badge tone="INFO">待产出</Badge>
                  )}
                  {g.high_volume && g.items.length > 0 && (
                    <span className="text-[10px] text-mast-muted/70">量大 · 默认折叠</span>
                  )}
                  <span className="ml-auto truncate pl-2 text-xs text-mast-muted">
                    {g.detail}
                  </span>
                </button>

                {/* Classes whose contents are not files (实验记录 / 长期记忆 /
                    视觉缓冲 / 文献库) have a real count but nothing to list here
                    — say where they live instead of showing a fake empty folder.
                    They also get NO edit button: the old 编辑/填充 button on
                    these slots called POST /api/artifacts/<class>/edit, which
                    the backend rejects for a bare class id every single time
                    ("artifact_id 必须是 '<draft|review>:<文件名>'"). That button
                    could not work — 编辑/填充这些东西还是不会自动生产 —
                    and a button that cannot work is worse than none. */}
                {!expandable && (
                  <div className="border-t border-mast-border/50 px-3 py-2 text-xs text-mast-muted/80">
                    {g.produced
                      ? `内容存于 ${g.store}（此处不逐条列出）`
                      : `尚未产出 · 由 ${g.producer ? agentLabel(g.producer) : "系统"} 写入 ${g.store}`}
                  </div>
                )}

                {expandable && open && (
                  <div className="grid grid-cols-1 gap-2 border-t border-mast-border/50 p-2 sm:grid-cols-2 lg:grid-cols-3">
                    {g.items.map((e) => (
                      <div
                        key={e.id}
                        onClick={() => setSelectedId(e.id)}
                        className={
                          "cursor-pointer space-y-1 rounded-md border border-mast-border bg-mast-panel px-3 py-2 " +
                          (e.id === selectedId ? "ring-1 ring-mast-accent" : "")
                        }
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate font-mono text-xs" title={e.path || undefined}>
                            {e.preview || e.id}
                          </span>
                          {e.edited && <Badge tone="WARN">已编辑</Badge>}
                        </div>
                        {e.producer && (
                          <div className="text-xs text-mast-muted">
                            产出 · {agentLabel(e.producer)}
                          </div>
                        )}
                        <div className="pt-1" onClick={(ev) => ev.stopPropagation()}>
                          {/* Only file-backed EDITABLE artifacts get a button.
                              Every row used to get 打开编辑器 including .sxm
                              scans , and the backend refuses those — so the
                              button could only ever open a degraded editor. */}
                          {e.editable ? (
                            <Button variant="default" onClick={() => onEdit(e.id)}>
                              打开编辑器
                            </Button>
                          ) : (
                            <Badge tone="INFO">只读</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* selected-artifact detail card with 访问权限 panel + 打开编辑器 */}
      {selectedId && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
          <Card className="space-y-3 border-l-4 border-l-mast-accent px-5 py-4">
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-base font-semibold">{selectedId}</span>
              {selEntry?.kind && (
                <span className="font-mono text-xs text-mast-muted">· {selEntry.kind}</span>
              )}
              {selEntry?.producer && (
                <span className="text-xs text-mast-muted">· 产出 {agentLabel(selEntry.producer)}</span>
              )}
            </div>
            {selEntry?.preview ? (
              <pre className="max-h-60 overflow-auto whitespace-pre-wrap rounded-md border border-mast-border bg-mast-bg/60 p-3 font-mono text-xs leading-relaxed">
                {selEntry.preview}
              </pre>
            ) : (
              <EmptyNote label="无预览正文。" />
            )}
            <div className="flex flex-wrap items-center gap-2">
              {selEntry?.editable ? (
                <Button variant="primary" onClick={() => onEdit(selectedId)}>
                  打开编辑器
                </Button>
              ) : (
                <span className="text-xs text-mast-muted">
                  只读 · {selEntry?.path || "该产物不可手工编辑"}
                </span>
              )}
            </div>
          </Card>

          <Card className="space-y-0 px-0 py-0">
            <div className="border-b border-mast-border px-4 py-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-mast-muted">
                访问权限
              </div>
              <div className="mt-0.5 text-xs text-mast-muted">
                {writers.size} 写 · {readers.size} 读
              </div>
            </div>
            {selRow
              ? WS_COLS.filter((id) => readers.has(id)).map((id) => {
                  const canWrite = writers.has(id);
                  const def = id === "operator" ? null : agentDef(id);
                  return (
                    <div
                      key={id}
                      className="flex items-center gap-3 border-b border-mast-border/50 px-4 py-2 last:border-b-0"
                    >
                      {def ? (
                        <Avatar id={id} size={18} />
                      ) : (
                        <span className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-full bg-mast-muted/40 font-mono text-[8px] font-bold text-mast-text">
                          OP
                        </span>
                      )}
                      <div className="min-w-0 flex-1">
                        <div className="font-mono text-xs font-semibold">
                          {def ? def.short : "OP"}
                        </div>
                        <div className="text-xs text-mast-muted">
                          {def ? def.cn : "用户"}
                        </div>
                      </div>
                      <span
                        className={
                          "rounded px-2 py-0.5 font-mono text-xs font-semibold " +
                          (canWrite
                            ? "bg-mast-warn-bg text-mast-warn"
                            : "bg-mast-bg text-mast-muted")
                        }
                      >
                        {canWrite ? "读 + 写" : "只读"}
                      </span>
                    </div>
                  );
                })
              : (
                <div className="px-4 py-3">
                  <EmptyNote label="无权限信息。" />
                </div>
              )}
          </Card>
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// 对象编辑 — inline per-artifact editor (left artifact list + center editor).
//   Mirrors the old AGWorkspaceEditorView 3-column layout: left = artifact
//   roster, center = the operator editor (edit / diff / history). The center
//   editor reuses the fully-wired ArtifactEditor logic, rendered inline (not a
//   modal) so it lives on its own sub-tab exactly like the old build.
// ════════════════════════════════════════════════════════════════════════════

function EditorView({
  selected,
  onSelect,
}: {
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const artifacts = useQuery({
    queryKey: ["artifacts"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/artifacts");
      if (error) throw error;
      return data;
    },
    refetchInterval: 6000,
  });
  // The artifact CLASSES that nothing has produced yet are shown as
  // informational rows — NOT as editable slots.
  //
  // They used to be clickable "seed" slots, which was incoherent: you cannot
  // bring a manuscript into existence by typing into a textarea (paper_writing
  // creates it with save_draft), and you cannot hand-edit the experiment DB or a
  // Nanonis .sxm at all. Clicking one opened an editor over nothing. Showing them
  // is still useful — they say what the system CAN produce and where it lands —
  // so they stay, as rows you read rather than buttons that lie.
  const perms = useQuery({
    queryKey: ["artifacts", "permissions"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/artifacts/permissions");
      if (error) throw error;
      return data;
    },
  });
  const entries = artifacts.data?.artifacts ?? [];
  const cur = entries.find((e) => e.id === selected) ?? null;
  const [showReadonly, setShowReadonly] = useState(false);
  // Only file-backed EDITABLE artifacts (drafts / reviews) are clickable edit
  // targets. Produced-but-read-only artifacts (scan .sxm / figures) are listed
  // as informational rows, not buttons: the backend refuses to edit them, so
  // clicking one only opens a degraded editor — the same "button that lies" the
  // not-yet-produced classes below are careful to avoid. Read-only scans are
  // usually the newest files (list is mtime-sorted), so they sit at the top of
  // this roster and were the first thing an operator clicked.
  const editableEntries = entries.filter((e) => e.editable);
  // #29 — collapsed behind one summary row instead of N flat rows. This rail is
  // 260px wide and a run's scans push everything else off the screen.
  const readonlyEntries = entries.filter((e) => !e.editable);
  // 待产出 comes from the backend's per-class status, not from subtracting the
  // produced FILES: 文献库 / 实验记录 / 长期记忆 / 视觉缓冲 have no files to
  // enumerate, so subtracting files marked them 待产出 permanently even with
  // 77 experiment records in the DB. Sharing `groups` also keeps
  // this tab and 对象总览 from disagreeing about the same class.
  const permById = new Map(
    (perms.data?.permissions ?? []).map((r) => [r.artifact_id, r]),
  );
  const rosterSlots = (artifacts.data?.groups ?? [])
    .filter((g) => !g.produced)
    .map((g) => ({
      id: g.artifact_id,
      label: g.label || g.artifact_id,
      store: g.store ?? "",
      known: g.known,
      producer:
        (permById.get(g.artifact_id)?.writers ?? []).find((w) => w !== "operator") ??
        null,
    }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline gap-3">
        <h2 className="text-lg font-semibold tracking-tight">对象编辑</h2>
        <span className="text-xs text-mast-muted">
          直接编辑智能体产出的文档（论文草稿 / 评审报告）。保存 = 存为新版本，下一个打开该文档的智能体会读到你的修改。
        </span>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[260px_1fr]">
        {/* left: artifact roster */}
        <div className="space-y-1">
          {artifacts.isPending && <Spinner />}
          {artifacts.isError && <ErrorNote error={artifacts.error} />}
          {artifacts.data && artifacts.data.degraded && <DegradedNote what="制品列表" />}
          {artifacts.data && !artifacts.data.degraded && entries.length === 0 && rosterSlots.length === 0 && (
            <EmptyNote label="当前会话尚无制品。" />
          )}
          {editableEntries.map((e) => (
            <button
              key={e.id}
              onClick={() => onSelect(e.id)}
              className={
                "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-sm " +
                (e.id === selected
                  ? "border-mast-accent bg-mast-accent/10 text-mast-accent"
                  : "border-mast-border text-mast-muted hover:text-mast-text")
              }
            >
              <span className="min-w-0 flex-1 truncate font-mono text-xs">{e.id}</span>
              {e.edited && <Badge tone="WARN">已编辑</Badge>}
            </button>
          ))}
          {/* Produced but READ-ONLY (scan .sxm / figures) — shown so the operator
              knows they exist and where (title=path), but NOT clickable: the
              backend cannot edit them and the editor would only degrade. Behind
              one collapsed summary row : a run's scans are read-only and
              would otherwise push the two editable drafts off the rail. */}
          {readonlyEntries.length > 0 && (
            <button
              type="button"
              onClick={() => setShowReadonly((s) => !s)}
              className="flex w-full items-center gap-2 rounded-md border border-dashed border-mast-border/50 px-3 py-2 text-left text-sm text-mast-muted/70 hover:text-mast-text"
            >
              <span className="w-3 shrink-0 font-mono text-xs">
                {showReadonly ? "▾" : "▸"}
              </span>
              <span className="min-w-0 flex-1 truncate text-xs">
                只读产物 · {readonlyEntries.length} 项
              </span>
            </button>
          )}
          {showReadonly &&
            readonlyEntries.map((e) => (
              <div
                key={e.id}
                title={e.path || undefined}
                className="flex w-full items-center gap-2 rounded-md border border-dashed border-mast-border/50 py-2 pl-6 pr-3 text-left text-sm text-mast-muted/70"
              >
                <span className="min-w-0 flex-1 truncate font-mono text-xs">{e.id}</span>
                <Badge tone="INFO">只读</Badge>
              </div>
            ))}
          {/* Not-yet-produced artifact CLASSES — informational, not clickable.
              A dashed row that opens an editor over a file that does not exist
              is a button that lies; these say what will be produced and where. */}
          {rosterSlots.map((slot) => (
            <div
              key={slot.id}
              title={slot.store}
              className="flex w-full items-center gap-2 rounded-md border border-dashed border-mast-border/50 px-3 py-2 text-left text-sm text-mast-muted/70"
            >
              <span className="min-w-0 flex-1 truncate font-mono text-xs">{slot.id}</span>
              <Badge tone={slot.known ? "INFO" : "WARN"}>
                {slot.known ? "待产出" : "无法读取"}
              </Badge>
            </div>
          ))}
        </div>

        {/* center: inline editor */}
        <div>
          {selected ? (
            <ArtifactEditor
              inline
              artifactId={selected}
              producer={cur?.producer ?? rosterSlots.find((s) => s.id === selected)?.producer ?? null}
            />
          ) : (
            <EmptyNote label="从左侧选择一个对象进行编辑。" />
          )}
        </div>
      </div>
    </div>
  );
}
