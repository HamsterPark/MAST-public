import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, Toggle, useToast } from "@/components/controls";
import { agentLabel } from "@/components/agents/registry";
import { settingsWriteProblem } from "@/lib/settingsWrite";

// ════════════════════════════════════════════════════════════════════════════
// 待唤醒 — agents parked until their inputs exist.
//
// WHY THIS PANEL IS A PRECONDITION, NOT A NICE-TO-HAVE.
// The scheduler may decline to dispatch an agent whose upstream products do not
// exist yet (reviewing a manuscript nobody has written can only produce an
// invented review). The design that introduced that says of itself that its most
// likely failure is a beautiful SILENT DEATH CHANNEL — an agent waiting forever
// while the system looks idle and healthy. This repo has already paid once for
// "nothing happened" and "it hung" being indistinguishable.
//
// So three properties the backend and this panel share, none of them optional:
// a parked agent is VISIBLE; its wait has a DEADLINE; when the deadline passes it
// surfaces to a human instead of resuming silently or vanishing. An expired row
// keeps demanding attention until someone acknowledges it — a one-shot toast is
// not a delivery when nobody is watching, and nobody watching is the normal case
// for a mechanism whose whole purpose is to work while you are away.
//
//   GET  /api/agents/pending-activations                  — list (polled)
//   POST /api/agents/pending-activations/{id}/acknowledge — clear attention
// ════════════════════════════════════════════════════════════════════════════

type Park = {
  park_id: string;
  agent: string;
  experiment_id: string;
  waiting_for: string[];
  waiting_for_labels: string[];
  blocked_by: string;
  reason: string;
  instruction: string;
  hard: boolean;
  status: string;
  created_at: number;
  deadline_at: number;
  seconds_left: number;
  waited_human: string;
  declines: number;
  woken_run_id: string;
  needs_attention: boolean;
  campaign_id?: string;
  /** 最近一次目标判据求值。`verdict==="unknown"` 时 `reason` 说的是**读不到
   *  什么** —— 判不了必须看得见，否则它和「一切正常」长得一样。 */
  goal_check?: { verdict?: string; reason?: string; checked_at?: number };
};

// 这两张表要盖住 `mast/core/park_board.py` 的 `STATUSES` 全集 —— 有一条 Python
// parity 测试钉着。漏一个的后果不是报错，是那一行显示成一个裸的英文状态码，
// 而用户看到 `done_by_goal` 时并不知道它是好事。
const STATUS_TONE: Record<string, string> = {
  waiting: "INFO",
  woken: "AUTO",
  done: "AUTO",
  done_by_goal: "AUTO",
  expired: "DANGEROUS",
  cancelled: "INFO",
};
const STATUS_LABEL: Record<string, string> = {
  waiting: "等待中",
  woken: "已唤醒",
  done: "已完成",
  // 与「已完成」分开：那个说「醒了、干完了、产物回来了」，这个说「一次都没醒，
  // 而且不用醒了 —— 它服务的目标已经达成」。
  done_by_goal: "目标已达成",
  expired: "已超时",
  cancelled: "已撤销",
};

function countdown(secondsLeft: number): string {
  if (!Number.isFinite(secondsLeft)) return "—";
  if (secondsLeft <= 0) return "已超时";
  const s = Math.round(secondsLeft);
  if (s < 3600) return `还有 ${Math.max(1, Math.floor(s / 60))} 分钟`;
  if (s < 172800) return `还有 ${Math.floor(s / 3600)} 小时`;
  return `还有 ${Math.floor(s / 86400)} 天`;
}

export function PendingActivationsPanel() {
  const qc = useQueryClient();
  const { toast, node } = useToast();

  const parksQuery = useQuery({
    queryKey: ["agents", "pending-activations"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/pending-activations");
      if (error) throw error;
      return data;
    },
    // Slow poll: a park's timescale is hours. The backend sweeps deadlines on
    // read, so this poll is ALSO what makes a timeout fire while the system is
    // otherwise idle — but 20 s is ample for an hours-long wait.
    refetchInterval: 20000,
  });

  const parks = (parksQuery.data?.parks ?? []) as Park[];
  const attention = useMemo(() => parks.filter((p) => p.needs_attention), [parks]);

  // The gate that decides whether parking happens at all. Default OFF, and
  // surfaced HERE rather than buried in a settings page so the switch sits next
  // to the consequence of flipping it.
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });
  const gateOn = settings.data?.orchestrator_activation_gating ?? false;
  const saveGate = useMutation({
    mutationFn: async (v: boolean) => {
      const { data, error } = await api.POST("/api/settings", {
        body: { orchestrator_activation_gating: v },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d, v) => {
      const problem = settingsWriteProblem(d);
      if (problem) { toast(problem, "err"); return; }
      toast(
        v
          ? "已开启：缺上游资料的智能体会被搁置，并显示在这里"
          : "已关闭：编排器照旧派发，不再搁置任何智能体",
        "ok",
      );
      qc.invalidateQueries({ queryKey: ["settings"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  const ack = useMutation({
    mutationFn: async (parkId: string) => {
      const { data, error } = await api.POST(
        "/api/agents/pending-activations/{park_id}/acknowledge",
        { params: { path: { park_id: parkId } } },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      toast("已确认（记录保留，不会自动继续）", "ok");
      qc.invalidateQueries({ queryKey: ["agents", "pending-activations"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  return (
    <div className="space-y-4">
      {node}

      {/* what this is + the gate that produces it */}
      <Card className="space-y-1 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium">待唤醒 · 环境驱动搁置</span>
          {attention.length > 0 ? (
            <Badge tone="DANGEROUS">{attention.length} 项超时待处理</Badge>
          ) : (
            <Badge tone={parks.length > 0 ? "INFO" : "AUTO"}>
              {parks.length > 0 ? `${parks.length} 个在等` : "无搁置"}
            </Badge>
          )}
        </div>
        <p className="text-xs text-mast-muted">
          编排器不会把一个「缺上游资料」的智能体派去编内容 —— 例如还没有草稿就让评审去评。
          这类智能体会被记在这里等资料；资料到位后会重新问它要不要开工，它也可以选择继续等。
          每个等待都有上限，超时会浮到这里要你处理，<strong>不会自动继续、也不会悄悄消失</strong>。
        </p>
        <div className="flex items-center justify-between gap-3 border-t border-mast-border pt-2">
          <div className="min-w-0">
            <div className="text-xs font-medium text-mast-text">启用环境驱动搁置（实验性）</div>
            <div className="text-[11px] text-mast-muted">
              默认关闭 —— 关着时编排器的调度与以前完全一样。开启后才会出现搁置。
              想强制某个智能体现在就干活？在对话里 @它 直接点名即可（定向永不被搁置）。
            </div>
          </div>
          <Toggle
            checked={!!gateOn}
            onChange={(v) => saveGate.mutate(v)}
            label={gateOn ? "已开启" : "已关闭"}
          />
        </div>
      </Card>

      {parksQuery.isLoading ? <Spinner /> : null}
      {parksQuery.error ? <ErrorNote error={parksQuery.error} label="读不到搁置记录" /> : null}
      {parksQuery.data?.degraded ? <DegradedNote what="待唤醒列表" /> : null}

      {!parksQuery.isLoading && parks.length === 0 && !parksQuery.data?.degraded ? (
        <EmptyNote label="目前没有被搁置的智能体。" />
      ) : null}

      {parks.map((p) => (
        <Card
          key={p.park_id}
          className={
            p.needs_attention
              ? "space-y-1.5 border-mast-danger-border px-4 py-3"
              : "space-y-1.5 px-4 py-3"
          }
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium">{agentLabel(p.agent)}</span>
            <Badge tone={STATUS_TONE[p.status] ?? "INFO"}>
              {STATUS_LABEL[p.status] ?? p.status}
            </Badge>
            {p.hard ? <Badge tone="WARN">硬依赖缺失</Badge> : null}
          </div>

          <div className="text-xs text-mast-muted">
            等：{p.waiting_for_labels.join("、") || "—"}
            {p.blocked_by ? `（由${agentLabel(p.blocked_by)}产出）` : ""}
          </div>

          <div className="text-[11px] text-mast-faint">
            已等 {p.waited_human || "—"} · {countdown(p.seconds_left)}
            {p.declines > 0 ? ` · 已拒绝 ${p.declines} 次` : ""}
            {p.woken_run_id ? ` · 运行 ${p.woken_run_id.slice(0, 8)}` : ""}
          </div>

          {p.reason ? (
            <div className="text-[11px] text-mast-faint">理由：{p.reason}</div>
          ) : null}

          {/* 目标判据（2026-08-28）。后端每 tick 都在算，只在结论变化时落盘。
              **unknown 必须显示出来** —— 判不了和「还没到」驱动的下一步不同，
              而一个不显示的 unknown 与「一切正常」长得一模一样。 */}
          {p.goal_check?.verdict ? (
            <div
              className={
                "text-[11px] " +
                (p.goal_check.verdict === "unknown"
                  ? "text-mast-warn"
                  : "text-mast-faint")
              }
            >
              目标判据：
              {p.goal_check.verdict === "done"
                ? "已达成"
                : p.goal_check.verdict === "unknown"
                  ? "读不到"
                  : "还没满足"}
              {p.goal_check.reason ? `（${p.goal_check.reason}）` : ""}
            </div>
          ) : null}

          {p.needs_attention ? (
            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-mast-danger-border pt-2">
              <span className="text-[11px] text-mast-danger">
                等待已超时。不会自动继续 —— 需要你决定：补上缺的资料，或在对话里 @它 直接让它开工。
              </span>
              <Button variant="default" onClick={() => ack.mutate(p.park_id)} loading={ack.isPending}>
                我知道了
              </Button>
            </div>
          ) : null}
        </Card>
      ))}
    </div>
  );
}
