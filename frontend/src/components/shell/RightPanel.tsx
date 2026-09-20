import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { useCurrentScope } from "@/api/scope";
import { ScopeControls } from "@/components/scope/ScopeControls";
import { fmtBias, fmtCurrent, fmtSI } from "@/lib/units";
import { LIVE_READINGS_POLL_MS, LIVE_READINGS_STALE_AFTER_MS } from "@/lib/pollRates";
import { headlineSensors, readingRows, stubKinds, type EnvSensor } from "@/lib/envPanel";

// Persistent right rail — faithful to the old mast-right-panel: INSTRUMENT
// (get_status_html) / ENVIRONMENT (get_environment_html) / EXPERIMENT (active
// exp + sample lifecycle) / SETTINGS (read-only summary) / SYSTEM (TCP). Visible
// on every tab. Read-only summaries here; editing lives in the 设置 / 实验记录 tabs.

// One label/value pair rendered as two cells of the parent's auto/1fr grid.
// The value is right-aligned, font-mono tabular-nums; an optional sparkline or
// a status-pill node may replace the plain text value.
function Row({
  label,
  value,
  tone,
  spark,
  sparkColor,
  valueNode,
}: {
  label: string;
  value?: string;
  tone?: string;
  spark?: number[];
  sparkColor?: string;
  valueNode?: React.ReactNode;
}) {
  return (
    <>
      <span className="text-mast-muted">{label}</span>
      <div className="flex items-center justify-end gap-2 text-right">
        {spark && spark.length > 1 && <Sparkline data={spark} color={sparkColor} />}
        {valueNode ?? (
          <span className={`font-mono tabular-nums ${tone ?? "text-mast-text"}`}>{value}</span>
        )}
      </div>
    </>
  );
}

// Section label — mono micro-caps with wide tracking (canvas right-panel labels).
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-[9.5px] tracking-[1.4px] text-mast-muted">{children}</span>
  );
}

// Inline status pill — semantic triple (warn / muted), matching the canvas
// Z ctrl OFF and Tip 撤回·安全 badges.
function StatusPill({ tone, children }: { tone: "warn" | "muted" | "auto" | "danger"; children: React.ReactNode }) {
  const cls =
    tone === "warn"
      ? "text-mast-warn bg-mast-warn-bg border-mast-warn-border"
      : tone === "auto"
        ? "text-mast-auto bg-mast-auto-bg border-mast-auto-border"
        : tone === "danger"
          ? "text-mast-danger bg-mast-danger-bg border-mast-danger-border"
          : "text-mast-muted border-mast-border-strong";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-mast-badge border px-[7px] py-px text-[10.5px] ${cls}`}
    >
      {children}
    </span>
  );
}

// Compact inline sparkline — mirrors the old _svg_sparkline next to Bias /
// Current / Z pos. in get_status_html.
function Sparkline({ data, color = "var(--mast-accent)" }: { data: number[]; color?: string }) {
  const w = 48;
  const h = 14;
  const lo = Math.min(...data);
  const hi = Math.max(...data);
  const span = hi - lo || 1;
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - lo) / span) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={w} height={h} className="shrink-0">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1" />
    </svg>
  );
}

function PanelSection({
  title,
  action,
  grid = true,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  /** Wrap children in the auto/1fr readout grid (default). Set false for free-form bodies. */
  grid?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="border-b border-mast-border px-[18px] py-4 last:border-0">
      <div className="mb-3 flex items-center justify-between">
        <SectionLabel>{title}</SectionLabel>
        {action}
      </div>
      {grid ? (
        <div className="grid grid-cols-[auto_1fr] items-center gap-x-2.5 gap-y-[var(--mast-row-py)] text-[12.5px]">
          {children}
        </div>
      ) : (
        children
      )}
    </div>
  );
}

export function RightPanel() {
  const readings = useQuery({
    queryKey: ["hardware", "live-readings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/hardware/live-readings");
      if (error) throw error;
      return data;
    },
    // 500 ms, not 2 s : the endpoint is a cached snapshot with no TCP on
    // the request path, and the producer refreshes at ~1 s — polling at half
    // the producer's rate was throwing away every other reading. Costs the
    // instrument nothing. See lib/pollRates.ts.
    refetchInterval: LIVE_READINGS_POLL_MS,
  });
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });
  const conn = useQuery({
    queryKey: ["nanonis", "connection"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/nanonis/connection");
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });

  const health = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/health");
      if (error) throw error;
      return data;
    },
    staleTime: 5 * 60_000,
  });

  // ITEM 14 — live ENVIRONMENT readings (vacuum / temperature / helium / noise).
  // Cached relay endpoint (no serial I/O on the request); degrades to N/A.
  const env = useQuery({
    queryKey: ["environment", "readings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/environment/readings");
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });
  const qcEnv = useQueryClient();
  const rescan = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.GET("/api/environment/sensors/rescan");
      if (error) throw error;
      return data;
    },
    onSettled: () => qcEnv.invalidateQueries({ queryKey: ["environment", "readings"] }),
  });

  const r = readings.data?.readings;
  const s = settings.data;
  const biasHist = readings.data?.bias_history ?? [];
  const curHist = readings.data?.current_history ?? [];
  const zHist = readings.data?.z_history ?? [];
  const zCtrl = r?.z_controller_status
    ? r.z_controller_status.toUpperCase()
    : r?.z_controller_on == null
      ? "---"
      : r.z_controller_on
        ? "ON"
        : "OFF";
  // 当前实验/样品来自服务端的单一指针（2026-07-28）。
  //
  // 从前这里取 experiments[0]，而 TopBar 取 find(status === "active") —— 两处
  // 各自推断，可以显示不同的实验。根因是「哪个是当前」被编码在 status 列里，
  // 于是切换必须顺手把上一个标成 superseded，一次崩溃就留下陈旧的 active 行。
  // 现在有一个显式的 active_scope 指针，两处读同一个端点、同一个 queryKey，
  // 结构上不可能再分歧。
  const scope = useCurrentScope();
  const activeExp = scope.data?.experiment
    ? {
        id: scope.data.experiment.id,
        name: scope.data.experiment.name,
        sample_name: scope.data.sample?.name ?? null,
        sample_id: scope.data.sample?.id ?? null,
      }
    : null;

  // ITEM 14 — render one headline gauge: live value when present, else N/A.
  // `degraded` (no monitor wired) collapses everything to a muted N/A.
  const envDegraded = env.data?.degraded ?? true;
  function gauge(
    sv?: {
      value?: number | null;
      unit?: string | null;
      connected?: boolean | null;
      status?: string | null;
    } | null,
  ) {
    if (envDegraded || !sv || sv.value == null) {
      return { value: "N/A", tone: "text-mast-muted" };
    }
    const unit = sv.unit ? ` ${sv.unit}` : "";
    const num = Math.abs(sv.value) >= 1e4 || (sv.value !== 0 && Math.abs(sv.value) < 1e-3)
      ? sv.value.toExponential(2)
      : sv.value.toFixed(2);
    // Reflect the sensor's alert status (the old gauge ignored it — a vacuum/
    // thermal ALARM read the same as a healthy value, 审查).
    const tone =
      sv.status === "alarm" || sv.status === "error"
        ? "text-red-400 font-semibold"
        : sv.status === "warning"
          ? "text-amber-400"
          : sv.connected
            ? "text-mast-text"
            : "text-mast-warn";
    return { value: `${num}${unit}`, tone };
  }
  // The four headline gauges, used ONLY as the fallback when no real sensor of
  // that kind is connected (see the ENVIRONMENT section below).
  const ENV_KINDS = [
    { key: "vacuum", label: "Vacuum", headline: gauge(env.data?.vacuum) },
    { key: "temperature", label: "Temperature", headline: gauge(env.data?.temperature) },
    { key: "helium_level", label: "Helium Level", headline: gauge(env.data?.helium_level) },
    { key: "noise_level", label: "Noise Level", headline: gauge(env.data?.noise_level) },
  ];
  const ENV_KIND_KEYS = ENV_KINDS.map((k) => k.key);
  // 选行的判断全在 lib/envPanel.ts（纯函数，有测试）。这里只负责画。
  const envSensors: EnvSensor[] = (env.data?.sensors ?? []) as EnvSensor[];
  // 只有占位撑着的那几位：它们不是「读不到」，是 MAST 里没有这个量的驱动。
  const envStubbed = stubKinds(envSensors, ENV_KIND_KEYS);
  const envStubLabels = ENV_KINDS.filter((k) => envStubbed.includes(k.key));
  // Overall environment alert (worst sensor) — surfaced as a banner so an
  // unattended vacuum failure / thermal runaway is visible, not just a number.
  const envOverall = env.data?.overall_status ?? "";
  const envAlert = envOverall === "alarm" || envOverall === "error"
    ? "alarm" : envOverall === "warning" ? "warning" : "";

  // Updating 4× faster makes these numbers look MORE live, so it has to be
  // impossible for them to look live when they are not . The backend flags
  // `stale` when the monitor link is down and it is carrying values forward; an
  // errored or long-silent query means the same thing from this side. TopBar
  // already gated on exactly this — the panel the operator actually watches did
  // not.
  const readingsStale =
    (r as { stale?: boolean } | undefined)?.stale === true ||
    readings.isError ||
    (readings.dataUpdatedAt > 0 &&
      Date.now() - readings.dataUpdatedAt > LIVE_READINGS_STALE_AFTER_MS);

  return (
    <aside className="relative w-72 shrink-0 overflow-auto bg-mast-panel">
      {/* accent bar down the left edge (canvas right-panel) */}
      <span className="pointer-events-none absolute inset-y-0 left-0 w-1 bg-mast-accent" />
      <PanelSection title="INSTRUMENT">
        {readingsStale && (
          <div className="col-span-2 mb-1 rounded bg-mast-warn-bg px-1.5 py-1 text-[10.5px] leading-snug text-mast-warn">
            读数已停更 —— 下面是最后一次读到的值，不是当前值。
          </div>
        )}
        <Row label="Bias"
          value={fmtBias(r?.bias_v)}
          tone={r?.bias_v == null ? "text-mast-muted" : "text-mast-text"}
          spark={biasHist.length ? biasHist : undefined} />
        <Row label="Current"
          value={fmtCurrent(r?.current_a)}
          tone={r?.current_a == null ? "text-mast-muted" : "text-mast-text"}
          spark={curHist.length ? curHist : undefined} sparkColor="var(--mast-accent)" />
        <Row label="Z pos."
          value={fmtSI(r?.z_m, "m", { digits: 5 })}
          tone={r?.z_m == null ? "text-mast-muted" : "text-mast-text"}
          spark={zHist.length ? zHist : undefined} />
        <Row label="Z ctrl"
          valueNode={
            <StatusPill tone={zCtrl === "ON" ? "auto" : zCtrl === "OFF" ? "warn" : "muted"}>{zCtrl}</StatusPill>
          } />
        <Row label="Setpoint" value={fmtCurrent(r?.setpoint_a)} />
        <Row label="Tip"
          valueNode={
            <StatusPill tone={r?.withdrawn ? "muted" : r?.withdrawn === false ? "auto" : "muted"}>
              {r?.withdrawn == null ? "---" : r.withdrawn ? "撤回 · 安全" : "接近"}
            </StatusPill>
          } />
        <Row label="Scan"
          valueNode={
            r?.scan_running ? (
              <StatusPill tone="auto">RUNNING</StatusPill>
            ) : (
              <span className="font-mono text-[11.5px] tabular-nums text-mast-muted">
                {r?.scan_running == null ? "---" : "STOPPED"}
              </span>
            )
          } />
      </PanelSection>

      <PanelSection
        title="ENVIRONMENT"
        action={
          <button
            type="button"
            onClick={() => rescan.mutate()}
            disabled={rescan.isPending}
            title="重新扫描传感器"
            aria-label="重新扫描传感器"
            className={`inline-flex text-mast-muted hover:text-mast-accent disabled:opacity-50 ${rescan.isPending ? "animate-spin" : ""}`}
          >
            ↻
          </button>
        }
      >
        {envAlert && (
          <div
            className={`col-span-2 mb-1 rounded px-2 py-1 text-[11px] font-semibold ${
              envAlert === "alarm"
                ? "bg-red-600/20 text-red-300"
                : "bg-amber-500/15 text-amber-300"
            }`}
          >
            {envAlert === "alarm"
              ? "⚠ 环境告警：某传感器越限（真空/温度）——已触发止损"
              : "环境警告：某传感器接近限值"}
          </div>
        )}
        {/* Every reading ONCE, under the name the instrument reports .
            The panel used to print four fixed headline rows AND then the raw
            sensor list, so the operator saw the same 77.42 K twice — once as
            "Temperature", once as "MODEL335 (COM13)" — with three placeholder
            rows duplicated underneath as "vacuum / helium_level / noise_level".
            A real sensor now REPLACES its headline row and keeps its own name
            (a two-input Lake Shore reads out as "SPM" and "Magnet", which is
            what its inputs are actually called). */}
        {ENV_KINDS.map(({ key, label, headline }) => {
          // 只有占位撑着的那一位不画读数行 —— 一个从来没返回过数字的桩挂着
          // 「N/A」，读起来和「表坏了」一模一样。它去下面那行「未接入」。
          if (envStubbed.includes(key)) return null;
          const real = headlineSensors(envSensors, key);
          if (!real.length) {
            return <Row key={key} label={label} value={headline.value} tone={headline.tone} />;
          }
          return real.map((sn) => {
            const g = gauge({ value: sn.value, unit: sn.unit, connected: sn.connected, status: sn.status });
            return <Row key={`${key}:${sn.name}`} label={sn.name || label} value={g.value} tone={g.tone} />;
          });
        })}
        {/* Anything that is not one of the four known kinds — extra configured
            gauges — still gets a row. `readingRows` drops the InstrumentState
            mirrors here: tunnel_current is the SAME number as INSTRUMENT →
            Current four rows above . 只去掉展示，采集与判据不动。 */}
        {readingRows(envSensors)
          .filter((sn) => !ENV_KIND_KEYS.includes(sn.kind ?? ""))
          .map((sn) => {
            const g = gauge({ value: sn.value, unit: sn.unit, connected: sn.connected, status: sn.status });
            return <Row key={sn.name} label={sn.name} value={g.value} tone={g.tone} />;
          })}
        {envStubLabels.length > 0 && (
          <div className="col-span-2 mt-1 text-[11px] leading-snug text-mast-faint">
            未接入：
            {envStubLabels.map((k, i) => (
              <span key={k.key}>
                {i > 0 && " · "}
                {/* 噪声这一位没有硬件，但这台机器的噪声**是**被测的 ——
                    电流监控算去趋势 RMS。指过去，不在这里编一个数出来。 */}
                {k.key === "noise_level" ? (
                  <Link to="/monitoring/current" className="text-mast-accent hover:underline">
                    {k.label}
                  </Link>
                ) : (
                  k.label
                )}
              </span>
            ))}
          </div>
        )}
        {envDegraded && (
          <div className="col-span-2 mt-2 text-[11px] leading-snug text-mast-faint">
            传感器监控未连接
          </div>
        )}
      </PanelSection>

      <PanelSection title="EXPERIMENT" grid={false}>
        {activeExp ? (
          <div className="grid grid-cols-[auto_1fr] items-center gap-x-2.5 gap-y-2 text-[12.5px]">
            <Row label="项目" value={activeExp.name ?? "—"} />
            {/* 见下方 ScopeControls：显示与切换都读 /api/experiments/current，
                不再从实验列表推断（那是 TopBar 与本面板显示不同实验的根因）。 */}
            {/* 样品名 replaces the bare UUID. "441ebe7c-2a7" told the operator
                nothing — not even whether it was an experiment, sample or
                conversation id . The id stays reachable as a tooltip for
                when it IS what you need (filing a bug, grepping a log).
                The "└" indent is not decoration: this panel showed 项目 and 样品
                as two equal rows, which read as siblings — "一个实验若干样品"
                . The sample hangs off the experiment, and so does every
                chat started under it. */}
            <Row label="└ 样品" value={activeExp.sample_name || "—"} />
            <Row
              label="实验 ID"
              valueNode={
                <span
                  className="cursor-help font-mono text-[11px] tabular-nums text-mast-muted"
                  title={`实验 ID: ${activeExp.id ?? "—"}${
                    activeExp.sample_id ? `\n样品 ID: ${activeExp.sample_id}` : ""
                  }`}
                >
                  {String(activeExp.id ?? "—").slice(0, 8)}…
                </span>
              }
            />
          </div>
        ) : (
          <div className="text-xs text-mast-muted">No active experiment</div>
        )}
        <ScopeControls />
        <a href="/records" className="mt-2 inline-block text-[11.5px] text-mast-accent hover:underline">实验记录 →</a>
      </PanelSection>

      <PanelSection title="SETTINGS">
        <Row label="Model" value={s?.model_alias ?? "—"} tone="text-mast-accent text-[11.5px]" />
        <Row label="Thinking" value={s?.thinking ? s.thinking.charAt(0).toUpperCase() + s.thinking.slice(1) : "—"} />
        <Row label="Font" value={s?.font_scale ?? "中"} />
        <Row label="Theme" value={s?.theme ?? "Dark"} />
        <p className="col-span-2 mt-2 text-[11px] leading-snug text-mast-faint">
          在「设置」标签中修改模型 / 外观 / 硬件连接 / 知识 / 语音。
        </p>
      </PanelSection>

      <PanelSection title="SYSTEM" grid={false}>
        <div className="text-[12.5px] font-semibold text-mast-text">MAST v{health.data?.version ?? "?"}</div>
        <div className="mb-2 text-[11.5px] text-mast-muted">
          {health.data?.skill_registry_wired ? "技能注册已就绪" : "技能注册未连接"}
        </div>
        <div className="flex gap-3">
          <a href="/settings" className="inline-block text-[11.5px] text-mast-accent hover:underline">设置 →</a>
          {/* `/dashboard`（实验台总览：状态卡 / 实时读数 / **系统自检表** /
              TCP / 各 agent 模型）在顶栏里没有位置，而在这一行之前，全仓渲染出来的
              链接里也没有一个指向它 —— 只有知道地址的人进得去。与 #44 的粗动大
              地图同一个形状，是那条闸门（test/scanMapPairing.test.ts）顺手逮到的
              第二处。系统自检表在别处没有第二个家。 */}
          <Link to="/dashboard" className="inline-block text-[11.5px] text-mast-accent hover:underline">
            系统自检 →
          </Link>
        </div>
      </PanelSection>

      {/* TCP footer — separate from System, like the old get_connection_compact_html */}
      <PanelSection title="TCP">
        <Row label="连接"
          valueNode={
            <StatusPill tone={conn.data?.connected ? "auto" : "muted"}>
              {conn.data?.connected ? "已连接" : "未连接"}
            </StatusPill>
          } />
        {(conn.data?.ports ?? []).map((p) => (
          <Row key={p.role} label={p.role} value={`${p.port ?? ""} ${p.connected ? "●" : "○"}`}
            tone={p.connected ? "text-mast-auto" : "text-mast-muted"} />
        ))}
      </PanelSection>
    </aside>
  );
}
// ExperimentControls 已被 components/scope/ScopeControls 取代 (2026-07-28)：
// 自由文本 + New 是 create-always 的源头（库里 5 条同名实验相隔 2 分钟），
// End 按钮对应的「结束实验/样品」动作已不存在 —— 实验永久可继续。
