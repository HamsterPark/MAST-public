import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { api } from "@/api/client";
import { Section, Card, Badge, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { Sparkline } from "@/components/vision/Sparkline";
import { fmtBias, fmtCurrent, fmtSI } from "@/lib/units";
import { LIVE_READINGS_POLL_MS } from "@/lib/pollRates";

// DashboardPage — Domain J overview (Lab Console), FULL parity rebuild.
//   · 状态卡: Nanonis 连接 / 实验存储 / 主模型 (IC)
//   · 硬件实时读数: bias / current / z / setpoint (+ x/y/控制器/退针/扫描),
//     /api/hardware/live-readings, poll LIVE_READINGS_POLL_MS, with sparklines (mirrors
//     gui/status_panel.get_status_html — the v1 header read these straight off
//     InstrumentState).
//   · 系统自检表: /api/system/check (mirrors gui/dashboard.run_system_check).
//   · TCP 端口 + 各 Agent 模型.
// Every read renders loading / error / degraded / empty.

const POLL_MS = 2000;

function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/health");
      if (error) throw error;
      return data;
    },
    refetchInterval: POLL_MS,
  });
}
function useConnection() {
  return useQuery({
    queryKey: ["nanonis", "connection"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/nanonis/connection");
      if (error) throw error;
      return data;
    },
    refetchInterval: POLL_MS,
  });
}
function useLiveReadings() {
  return useQuery({
    queryKey: ["hardware", "live-readings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/hardware/live-readings");
      if (error) throw error;
      return data;
    },
    // Faster than the other cards on purpose: this one is a cached snapshot the
    // request never pays TCP for, so it can track the ~1 s producer instead of
    // sampling it at half rate  — see lib/pollRates.ts.
    refetchInterval: LIVE_READINGS_POLL_MS,
  });
}
function useSystemCheck() {
  return useQuery({
    queryKey: ["system", "check"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/system/check");
      if (error) throw error;
      return data;
    },
    refetchInterval: 15000,
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
    refetchInterval: 10000,
  });
}

// ── value formatters ─────────────────────────────────────────────────────────
// Delegated to @/lib/units. This page, TopBar and RightPanel each carried their
// OWN copy, and each was wrong in its own way — this one pinned currents to pA
// (a 1 nA setpoint read "1000.0 pA"), the other two dropped to raw
// `toExponential` ("9.63e-11 A", ). One implementation is the
// only shape that stays fixed.
const fmtSetpoint = fmtCurrent;
const fmtZ = (m?: number | null) => fmtSI(m, "m", { digits: 5 });

// ── small presentational helpers ─────────────────────────────────────────────
function StatCard({
  label,
  children,
  tone,
}: {
  label: string;
  children: ReactNode;
  tone?: "ok" | "err" | "warn" | "dim";
}) {
  const toneCls =
    tone === "ok"
      ? "text-mast-auto"
      : tone === "err"
        ? "text-mast-danger"
        : tone === "warn"
          ? "text-mast-warn"
          : "text-mast-text";
  return (
    <Card>
      <div className="text-xs uppercase tracking-wide text-mast-muted">{label}</div>
      <div className={`mt-2 text-lg font-semibold tabular-nums ${toneCls}`}>{children}</div>
    </Card>
  );
}

function Reading({
  label,
  value,
  valueClass,
  spark,
}: {
  label: ReactNode;
  value: ReactNode;
  valueClass?: string;
  spark?: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-mast-border py-2 last:border-b-0">
      <span className="text-sm text-mast-muted">{label}</span>
      <span className="flex items-center gap-3">
        {spark}
        <span className={`text-sm font-medium tabular-nums ${valueClass ?? "text-mast-text"}`}>
          {value}
        </span>
      </span>
    </div>
  );
}

const CHECK_ICON: Record<string, string> = {
  ok: "✅",
  warning: "⚠️",
  error: "❌",
  unavailable: "🔧",
};
const CHECK_TONE: Record<string, string> = {
  ok: "text-mast-auto",
  warning: "text-mast-warn",
  error: "text-mast-danger",
  unavailable: "text-mast-muted",
};

const DOT = "●";

export default function DashboardPage() {
  const health = useHealth();
  const connection = useConnection();
  const live = useLiveReadings();
  const check = useSystemCheck();
  const models = useAgentModels();

  const conn = connection.data;
  const lr = live.data;
  const hw = lr?.readings;

  const expWired = health.data?.experiment_storage_wired ?? false;
  const connConnected = conn?.connected ?? false;
  const connPorts = conn?.ports ?? [];

  const agents = models.data?.agents ?? [];
  const icAgent = agents.find((a) => a.agent_id === "ic") ?? agents[0];

  const checkItems = check.data?.items ?? [];
  const checkCounts = checkItems.reduce<Record<string, number>>((acc, it) => {
    acc[it.status] = (acc[it.status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-2">
      {/* ── Status cards ─────────────────────────────────────────────────── */}
      <Section title="总览">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <StatCard
            label="Nanonis 连接"
            tone={connection.isError ? "err" : connConnected ? "ok" : "dim"}
          >
            {connection.isPending ? (
              <span className="text-base font-normal text-mast-muted">加载中…</span>
            ) : connection.isError ? (
              <span className="text-base font-normal text-mast-danger">读取失败</span>
            ) : connConnected ? (
              <>
                已连接
                {conn?.host ? <span className="ml-1 text-sm text-mast-muted">{conn.host}</span> : null}
              </>
            ) : (
              "未连接"
            )}
          </StatCard>

          <StatCard label="实验存储" tone={health.isError ? "err" : expWired ? "ok" : "dim"}>
            {health.isPending ? (
              <span className="text-base font-normal text-mast-muted">加载中…</span>
            ) : health.isError ? (
              <span className="text-base font-normal text-mast-danger">读取失败</span>
            ) : expWired ? (
              "已就绪"
            ) : (
              "未接入"
            )}
          </StatCard>

          <StatCard label="主模型 (IC)" tone={models.isError ? "err" : icAgent ? undefined : "dim"}>
            {models.isPending ? (
              <span className="text-base font-normal text-mast-muted">加载中…</span>
            ) : models.isError ? (
              <span className="text-base font-normal text-mast-danger">读取失败</span>
            ) : icAgent ? (
              <span className="text-base">
                {icAgent.model}
                {icAgent.thinking ? (
                  <span className="ml-2 text-sm text-mast-muted">{icAgent.thinking}</span>
                ) : null}
              </span>
            ) : (
              <span className="text-base font-normal text-mast-muted">无</span>
            )}
          </StatCard>
        </div>

        {health.data?.version && (
          <p className="mt-2 text-xs text-mast-muted">
            {health.data.service} · v{health.data.version} · 技能注册表
            {health.data.skill_registry_wired ? "已接入" : "未接入"} · 设置存储
            {health.data.settings_store_wired ? "已接入" : "未接入"}
          </p>
        )}
      </Section>

      {/* ── Hardware live readings ───────────────────────────────────────── */}
      <Section title="硬件实时读数">
        {live.isPending && <Spinner />}
        {live.isError && <ErrorNote error={live.error} />}
        {lr?.degraded && <DegradedNote what="硬件实时读数" />}
        {lr && !lr.degraded && (
          <Card>
            <div className="mb-2 flex items-center gap-2 text-xs text-mast-muted">
              <span className={lr.connected ? "text-mast-auto" : "text-mast-danger"}>{DOT}</span>
              {lr.connected ? "已连接（每 2 秒刷新）" : "主端口未连接"}
              {hw?.timestamp ? <span className="ml-auto">{hw.timestamp}</span> : null}
            </div>
            <Reading
              label="Bias"
              value={fmtBias(hw?.bias_v)}
              spark={<Sparkline values={lr.bias_history ?? []} color="var(--mast-ag-ic)" />}
            />
            <Reading
              label="Current"
              value={fmtCurrent(hw?.current_a)}
              spark={<Sparkline values={lr.current_history ?? []} color="var(--mast-dream)" />}
            />
            <Reading
              label="Z pos."
              value={fmtZ(hw?.z_m)}
              spark={<Sparkline values={lr.z_history ?? []} color="var(--mast-ag-ic)" />}
            />
            <Reading label="Setpoint" value={fmtSetpoint(hw?.setpoint_a)} />
            <Reading
              label="Z 控制器"
              value={
                hw?.z_controller_status
                  ? hw.z_controller_status.toUpperCase()
                  : hw?.z_controller_on == null
                    ? "---"
                    : hw.z_controller_on
                      ? "ON"
                      : "OFF"
              }
              valueClass={
                (hw?.z_controller_status ?? "").toLowerCase() === "on" || hw?.z_controller_on === true
                  ? "text-mast-auto"
                  : (hw?.z_controller_status ?? "").toLowerCase() === "off" || hw?.z_controller_on === false
                    ? "text-mast-danger"
                    : "text-mast-text"
              }
            />
            {hw?.x_m != null && <Reading label="X pos." value={fmtZ(hw.x_m)} />}
            {hw?.y_m != null && <Reading label="Y pos." value={fmtZ(hw.y_m)} />}
            {hw?.withdrawn != null && (
              <Reading
                label="探针"
                value={hw.withdrawn ? "WITHDRAWN" : "ENGAGED"}
                valueClass={hw.withdrawn ? "text-mast-danger" : "text-mast-auto"}
              />
            )}
            {hw?.scan_running != null && (
              <Reading
                label="扫描"
                value={hw.scan_running ? "RUNNING" : "STOPPED"}
                valueClass={hw.scan_running ? "text-mast-auto" : "text-mast-muted"}
              />
            )}
          </Card>
        )}
      </Section>

      {/* ── System self-check ────────────────────────────────────────────── */}
      <Section
        title="系统自检"
        actions={
          checkItems.length > 0 ? (
            <span className="flex gap-3 text-xs">
              {(["ok", "warning", "error", "unavailable"] as const).map((s) =>
                checkCounts[s] ? (
                  <span key={s} className={CHECK_TONE[s]}>
                    {CHECK_ICON[s]} {checkCounts[s]}
                  </span>
                ) : null,
              )}
            </span>
          ) : undefined
        }
      >
        {check.isPending && <Spinner />}
        {check.isError && <ErrorNote error={check.error} />}
        {check.data?.degraded && <DegradedNote what="系统自检" />}
        {check.data && !check.data.degraded && checkItems.length === 0 && (
          <EmptyNote label="暂无自检结果。" />
        )}
        {check.data && !check.data.degraded && checkItems.length > 0 && (
          <div className="overflow-auto rounded-lg border border-mast-border">
            <table className="w-full text-sm">
              <thead className="bg-mast-panel text-mast-muted">
                <tr>
                  <th className="w-12 px-3 py-2 text-center font-medium">状态</th>
                  <th className="px-3 py-2 text-left font-medium">组件</th>
                  <th className="px-3 py-2 text-left font-medium">详情</th>
                </tr>
              </thead>
              <tbody>
                {checkItems.map((it, i) => (
                  <tr key={`${it.name}-${i}`} className="border-t border-mast-border hover:bg-mast-bg/40">
                    <td className="px-3 py-2 text-center" title={it.status}>
                      {CHECK_ICON[it.status] ?? "?"}
                    </td>
                    <td className={`px-3 py-2 font-medium ${CHECK_TONE[it.status] ?? ""}`}>{it.name}</td>
                    <td className="px-3 py-2 text-mast-muted">{it.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* ── TCP ports detail ─────────────────────────────────────────────── */}
      <Section title="TCP 端口">
        {connection.isPending && <Spinner />}
        {connection.isError && <ErrorNote error={connection.error} />}
        {conn?.degraded && <DegradedNote what="Nanonis 连接" />}
        {conn && !conn.degraded && (
          <Card>
            {connPorts.length === 0 ? (
              <EmptyNote label="无端口信息" />
            ) : (
              connPorts.map((p) => (
                <Reading
                  key={p.role}
                  label={
                    <span>
                      <span className={p.connected ? "text-mast-auto" : "text-mast-danger"}>{DOT}</span>{" "}
                      <span className="capitalize">{p.role}</span>
                      {p.port != null ? <span className="ml-1 text-mast-muted">:{p.port}</span> : null}
                    </span>
                  }
                  value={
                    <Badge tone={p.connected ? "AUTO" : "DANGEROUS"}>
                      {p.connected ? "已连接" : "断开"}
                    </Badge>
                  }
                />
              ))
            )}
            {conn.host && <p className="mt-2 text-xs text-mast-muted">主机：{conn.host}</p>}
          </Card>
        )}
      </Section>

      {/* ── Per-agent models ─────────────────────────────────────────────── */}
      <Section title="各 Agent 模型">
        {models.isPending && <Spinner />}
        {models.isError && <ErrorNote error={models.error} />}
        {models.data?.degraded && <DegradedNote what="Agent 模型注册表" />}
        {models.data &&
          !models.data.degraded &&
          (agents.length === 0 ? (
            <EmptyNote label="暂无 Agent" />
          ) : (
            <Card>
              {agents.map((a) => (
                <Reading
                  key={a.agent_id}
                  label={a.agent_id}
                  value={
                    <span>
                      {a.model}
                      {a.thinking ? <span className="ml-2 text-mast-muted">{a.thinking}</span> : null}
                    </span>
                  }
                />
              ))}
            </Card>
          ))}
      </Section>
    </div>
  );
}
