import { useCallback, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import type { components } from "@/api/schema";
import { api } from "@/api/client";
import { useCurrentScope, useSwitchExperiment } from "@/api/scope";
import { ExperimentPicker } from "@/components/scope/ScopePickers";
import { useAutonomyMode } from "@/hooks/useAutonomyMode";
import { useWsConnection, useWsEvent } from "@/hooks/useWsEvents";
import { splitSI } from "@/lib/units";
import {
  LIVE_READINGS_POLL_MS,
  LIVE_READINGS_STALE_AFTER_MS,
  LIVE_READINGS_WS_KEEPALIVE_POLL_MS,
} from "@/lib/pollRates";
import { useUiStore } from "@/store";

// Top header bar — faithful to the old build_compact_header_html: logo + product,
// inline live hardware readings (B / I / Z / setpoint), and right-aligned
// connection · model · clock.
//
// ── Live readings: WebSocket push, HTTP poll underneath ────────────────────
// This is the first consumer of the
// shared /ws/events socket (lib/ws.ts) and the reference for the rest.
//
// The poll is NOT deleted, it is de-rated. When push is live we fall back from
// 500 ms to a slow keep-alive poll instead of stopping outright, because the
// window between "socket silently died" and "idle watchdog noticed" is up to
// WS_IDLE_MS (45 s), and 45 s of numbers that are frozen but look live is the
// exact failure 项目铁律「UI 绝不冻结」 forbids. The keep-alive costs nothing —
// the endpoint is a cached dict read with no TCP on the request path (see
// lib/pollRates.ts) — and it self-heals any event the socket missed.
//
// Both sources write the SAME react-query cache entry, so there is one timeline
// and nothing to arbitrate — see the note on the push handler for why keeping
// the pushed value beside the cache does not work.

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return now;
}

function fmt(v: number | null | undefined, digits: number, suffix = ""): string {
  return v == null ? "—" : `${v.toFixed(digits)}${suffix}`;
}

// Global operating-mode segmented control — safe (green) / semi (amber) / auto
// (none). Mirrors the belief + tip-processing gate wired on the backend
// (autonomy_mode). The active tint uses the semantic auto/warn/accent tokens;
// the whole-app colored border lives in AppLayout.
const MODES = ["safe", "semi", "auto"] as const;

// The operator asked, in the field: "是不是安全模式下不动粗逼近？" . They
// asked because the UI never said. The answer is NO — and it is not a mode thing
// at all: open-loop coarse Z approach toward the sample (MotorMove z-approach) is
// hard-blocked by SafetyGate Layer-0 in ALL THREE modes, before the mode gate is
// even consulted. The AUTO tooltip used to read "全部允许（默认，无限制）", which
// is simply false, and false in the direction that gets a tip crashed: it invites
// the operator to believe auto mode will drive the coarse motor for them.
const COARSE_NOTE = "开环粗逼近（MotorMove z-approach）在三种模式下一律禁止，须用户手动执行";

type LiveReadings = components["schemas"]["LiveReadings"];
type LiveReadingsResponse = components["schemas"]["LiveReadingsResponse"];

/** Shared by TopBar + RightPanel + DashboardPage — one cache entry, one fetch.
 *  A WS push writes here too, so all three see it without changing. */
const LIVE_READINGS_KEY = ["hardware", "live-readings"] as const;

/**
 * Pull a readings object out of a `hardware_state` event payload.
 *
 * Accepts BOTH `{...fields}` and `{readings: {...fields}}` because the event's
 * envelope is the backend's to choose and this component must not break the top
 * bar over a wrapper key. Anything unrecognised returns null and the poll simply
 * stays authoritative — an unknown push shape degrades to the old behaviour
 * rather than blanking the readout.
 */
function readingsFromEvent(data: unknown): LiveReadings | null {
  if (typeof data !== "object" || data === null) return null;
  const d = data as Record<string, unknown>;
  const inner =
    typeof d.readings === "object" && d.readings !== null
      ? (d.readings as Record<string, unknown>)
      : d;
  // Require at least one field we actually render, so an unrelated payload that
  // happens to be an object can't blank the bar.
  const KNOWN = ["bias_v", "current_a", "z_m", "setpoint_a", "z_controller_on", "scan_running"];
  if (!KNOWN.some((k) => k in inner)) return null;
  return inner as LiveReadings;
}

const MODE_SEG: Record<
  (typeof MODES)[number],
  { label: string; title: string; active: string; dot: string }
> = {
  safe: {
    label: "安全",
    title: `安全模式：视针尖良好，不修针、不电脉冲，专注实验。${COARSE_NOTE}`,
    active: "bg-mast-auto-bg text-mast-auto",
    dot: "var(--mast-auto)",
  },
  semi: {
    label: "半自动",
    title: `半自动模式：允许浅层机械修针；电脉冲需人工确认。${COARSE_NOTE}`,
    active: "bg-mast-warn-bg text-mast-warn",
    dot: "var(--mast-warn)",
  },
  auto: {
    label: "自动",
    title: `自动模式：修针与电脉冲均自动执行，不再逐条确认。${COARSE_NOTE}`,
    active: "bg-mast-accent-soft text-mast-accent",
    dot: "var(--mast-accent)",
  },
};

export function TopBar() {
  const theme = useUiStore((s) => s.theme);
  const toggleTheme = useUiStore((s) => s.toggleTheme);
  const { mode: autonomyMode, setMode: setAutonomyMode } = useAutonomyMode();
  const now = useClock();

  // Hardware EMERGENCY STOP — the one-click safety action (abort run + stop
  // motion + retract tip). There was NO hardware e-stop entry point in the UI
  // before. Always visible in the top bar.
  const [estopMsg, setEstopMsg] = useState<string>("");
  const estop = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/safety/emergency-stop");
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      const r = d as { retracted?: boolean; degraded?: boolean } | undefined;
      setEstopMsg(
        r?.degraded ? "急停：未连接硬件" : r?.retracted ? "已急停并退针" : "已急停",
      );
      setTimeout(() => setEstopMsg(""), 4000);
    },
    onError: () => {
      setEstopMsg("急停请求失败");
      setTimeout(() => setEstopMsg(""), 4000);
    },
  });

  // 中止锁的状态。轮询而不是只在出错时才发现 —— 从前「闩着」这件事在界面上
  // 没有任何表现，症状只有一个：每次仪器调用都被拒，而拒绝语说是用户干的。
  const latch = useQuery({
    queryKey: ["safety", "emergency-latch"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/safety/emergency-latch");
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });
  const clearLatch = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/safety/clear-emergency", {
        body: { reason: "从顶栏解除" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      const r = d as { was_latched?: boolean } | undefined;
      setEstopMsg(r?.was_latched ? "中止锁已解除" : "本来就没锁");
      setTimeout(() => setEstopMsg(""), 4000);
      void latch.refetch();
    },
    onError: () => {
      setEstopMsg("解除失败");
      setTimeout(() => setEstopMsg(""), 4000);
    },
  });

  // ── live readings: pushed over /ws/events, polled as the floor ────────────
  //
  // The push is written INTO the react-query cache rather than kept beside it.
  // The first attempt kept a local `pushed` value and rendered whichever of the
  // two had the newer timestamp — which looks right and is not: RightPanel is
  // mounted on the same screen and polls this very queryKey every 500 ms, so a
  // pushed value was overwritten on screen twice a second and the display
  // flickered between two sources. One cache, one timeline, no race.
  //
  // Side effect worth stating: RightPanel and DashboardPage read this same key,
  // so they get the pushed values too, without either file changing. Their own
  // polls keep running and would correct anything wrong within 500 ms.
  const ws = useWsConnection();
  const queryClient = useQueryClient();
  const [source, setSource] = useState<"ws" | "poll">("poll");

  const onHardwareState = useCallback(
    (event: { data: unknown }) => {
      const r = readingsFromEvent(event.data);
      if (!r) return; // unrecognised payload → leave the poll authoritative
      queryClient.setQueryData(
        LIVE_READINGS_KEY,
        (prev: LiveReadingsResponse | undefined): LiveReadingsResponse | undefined =>
          // No envelope cached yet → decline. The response also carries
          // `connected` / `degraded`, which this event does not know; inventing
          // them to satisfy the type would put a fabricated connection state on
          // screen. Returning undefined tells react-query to leave the cache
          // alone, and the first poll (≤500 ms away) fills it in properly.
          prev
            ? {
                // Merge, never replace: the sparkline histories belong to the
                // REST payload and a push that dropped them would blank
                // DashboardPage's charts.
                ...prev,
                readings: { ...(prev.readings ?? {}), ...r },
              }
            : undefined,
      );
      setSource("ws");
    },
    [queryClient],
  );
  useWsEvent("hardware_state", onHardwareState);

  const readings = useQuery({
    // NOTE: shares its queryKey with RightPanel + DashboardPage, so all three
    // read ONE deduped request — raising the rate does not multiply traffic.
    //
    // Each observer owns its OWN refetch timer, so de-rating this one does not
    // slow the other two down (nor they it): while RightPanel is mounted the
    // shared cache keeps ticking at 500 ms regardless. That is fine — this
    // component reads from the push — but it does mean the traffic saving only
    // lands once those two switch over as well.
    queryKey: LIVE_READINGS_KEY,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/hardware/live-readings");
      if (error) throw error;
      setSource("poll");
      return data;
    },
    refetchInterval: ws.polling ? LIVE_READINGS_POLL_MS : LIVE_READINGS_WS_KEEPALIVE_POLL_MS,
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

  const models = useQuery({
    queryKey: ["config", "models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/config/models");
      if (error) throw error;
      return data;
    },
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

  const scope = useCurrentScope();
  const switchExp = useSwitchExperiment();
  const [scopePicker, setScopePicker] = useState(false);

  const r = readings.data?.readings;
  const connected = conn.data?.connected ?? false;
  const zOff = r?.z_controller_on === false || r?.withdrawn === true;
  const currentModel = models.data?.default_alias ?? "—";
  const version = health.data?.version ?? "";

  // Live-reading staleness: the backend flags a snapshot `stale` when the
  // monitor link is down (values carried forward). Also treat an errored /
  // stopped readings query as stale so the topbar never shows hours-old numbers
  // as if live.
  //
  // Unchanged by the WS switch, and that is the payoff of writing pushes into
  // the cache: `dataUpdatedAt` now advances on a push as well as on a fetch, so
  // one age check covers both transports. A push that stops arriving ages out
  // exactly like a poll that stops returning — push cannot become a way to make
  // stale numbers look permanently live.
  const readingsStale =
    (r as { stale?: boolean } | undefined)?.stale === true ||
    readings.isError ||
    (readings.dataUpdatedAt > 0 &&
      Date.now() - readings.dataUpdatedAt > LIVE_READINGS_STALE_AFTER_MS);

  // Mode pill text. localhost does NOT imply a simulator — Nanonis normally runs
  // on the SAME machine as MAST, so the old "localhost → Simulator" label marked
  // every real local rig as a simulator. Show a truthful
  // connected/offline state; a real simulator flag can drive a "Sim" tag later.
  const modeText = !connected
    ? "Nanonis · Offline"
    : readingsStale
      ? "Nanonis · Stale"
      : "Nanonis · Connected";

  // 作用域读服务端的单一指针（2026-07-28）。
  //
  // 这里从前是 find(status === "active") ?? [0]，而 RightPanel 用 [0] ——
  // 两处推断规则不同，顶栏和右栏可以显示不同的实验。现在两边读同一个
  // /api/experiments/current。
  const expName = scope.data?.experiment?.name ?? "";
  const smpName = scope.data?.sample?.name ?? "";
  const hasExp = !!scope.data?.experiment;
  const hasSmp = !!scope.data?.sample;

  return (
    <header
      className="sticky top-0 z-30 flex flex-col border-b border-mast-border bg-mast-topbar text-sm"
      // Transport state as a DOM attribute: lets the e2e assert the WS/fallback
      // switch without a visible debug widget, and gives the operator something
      // concrete to read back over a screen share when push misbehaves.
      data-ws-status={ws.status}
      data-readings-source={source}
    >
      {/* row 1 — logo · wordmark · subtitle · version | spacer | mode · model · clock · theme · exp chip */}
      <div className="flex items-center gap-[11px] px-[18px] pb-2 pt-[9px]">
        <svg width="24" height="24" viewBox="0 0 24 24" className="block shrink-0 text-mast-accent">
          <circle cx="12" cy="12" r="9.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
          <circle cx="12" cy="12" r="2.9" fill="currentColor" />
        </svg>
        <span className="text-[18px] font-bold tracking-tight text-mast-text">MAST</span>
        <span className="hidden text-[11.5px] text-mast-muted sm:inline">Modular Autonomous SPM Toolkit</span>
        {version && <span className="font-mono text-[10.5px] text-mast-faint">v{version}</span>}

        {/* 这一排是 E‑STOP + 三档模式 + 状态点 + 模型 + 时钟 + 主题 + 作用域 chip，
            从前不换行：窗口一窄，唯一能压缩的那个 chip 就替所有人挨压。宽度不够时
            换行比把实验名挤成两个字更好读。 */}
        <div className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-x-3 gap-y-1.5">
          {/* EMERGENCY STOP — always reachable; retracts the tip + aborts the run */}
          <button
            type="button"
            onClick={() => estop.mutate()}
            disabled={estop.isPending}
            title="紧急停止：中止任务 + 停止运动 + 退针"
            className="rounded-mast-badge border border-red-500/60 bg-red-600/90 px-2.5 py-0.5 text-[11.5px] font-bold uppercase tracking-wide text-white shadow-[0_0_8px_rgba(220,38,38,0.5)] hover:bg-red-600 disabled:opacity-60"
          >
            {estop.isPending ? "…" : "E‑STOP"}
          </button>
          {estopMsg && (
            <span className="text-[11px] font-medium text-red-400">{estopMsg}</span>
          )}
          {/* 急停闩挂着 —— 这一块曾经不存在。
              闩有三个来源会挂上它（急停按钮 / 任意 E_STOP 事件 / 环境告警），
              其中两个不需要人参与；挂上之后**每一次仪器动作都被拒**。而它在
              界面上没有任何痕迹，解除口也不存在。症状因此是：
              操作没有任何反应，而机器已经退了针、锁死到重启为止。
              所以这里两件事必须一起给：**挂着**，以及**为什么**。 */}
          {latch.data?.latched && (
            <span className="inline-flex items-center gap-1.5 rounded-mast-badge border border-amber-500/60 bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-300">
              <span
                title={
                  latch.data.why
                    ? `中止原因：${latch.data.why}`
                    : "没有留下中止原因——别假定是人停的。去看服务日志里最近的 CRITICAL 行。"
                }
              >
                中止锁定中{latch.data.why ? `：${latch.data.why}` : "（未留原因）"}
              </span>
              <button
                type="button"
                onClick={() => clearLatch.mutate()}
                disabled={clearLatch.isPending}
                title="解除中止锁，让仪器动作重新被允许。只放开状态，不碰硬件——不进针、不开反馈、不恢复运行。"
                className="rounded-mast-badge border border-amber-400/70 px-1.5 py-px text-[10.5px] font-semibold text-amber-200 hover:bg-amber-500/25 disabled:opacity-60"
              >
                {clearLatch.isPending ? "…" : "解除"}
              </button>
            </span>
          )}
          {/* Global operating mode — safe/semi/auto tip-processing gate. Writes
              autonomy_mode via POST /api/settings; the whole-app colored border
              (AppLayout) reflects the same value. */}
          <div
            className="inline-flex items-center gap-0.5 rounded-mast-ctl border border-mast-border bg-mast-panel-2 p-0.5"
            role="group"
            aria-label="操作模式"
          >
            {MODES.map((m) => {
              const seg = MODE_SEG[m];
              const active = autonomyMode === m;
              return (
                <button
                  key={m}
                  type="button"
                  onClick={() => setAutonomyMode.mutate(m)}
                  disabled={setAutonomyMode.isPending}
                  title={seg.title}
                  aria-pressed={active}
                  className={clsx(
                    "inline-flex items-center gap-1 rounded px-2 py-[3px] text-[11px] font-medium transition-colors disabled:opacity-60",
                    active ? seg.active : "text-mast-muted hover:text-mast-text",
                  )}
                >
                  <span
                    className="h-[6px] w-[6px] rounded-full"
                    style={{ background: active ? seg.dot : "var(--mast-faint)" }}
                  />
                  {seg.label}
                </button>
              );
            })}
          </div>
          {/* A failed mode switch used to roll the cache back in SILENCE, so the
              operator saw the mode snap back and nothing else — indistinguishable
              from "卡住" . Say what happened. */}
          {setAutonomyMode.isError && (
            <span
              className="text-[11px] font-medium text-mast-danger"
              title={String((setAutonomyMode.error as Error)?.message ?? "")}
            >
              模式未切换（后端无响应）· 请重试
            </span>
          )}
          {/* Push is down and staying down. Say so — the readings below are
              still moving (the poll took over), so without this line the only
              evidence is a slower tick, which reads as "卡了" over Tailscale. */}
          {ws.warn && (
            <span
              className="text-[11px] font-medium text-mast-warn"
              title="实时推送连不上，已自动改用定时刷新：读数仍在更新，只是延迟略高。后台还在尝试恢复。"
              data-testid="ws-degraded"
            >
              实时推送不可用 · 已改定时刷新
            </span>
          )}
          <span className="inline-flex items-center gap-1.5 text-xs text-mast-text">
            <span
              className="h-[7px] w-[7px] rounded-full bg-mast-auto shadow-[0_0_7px_var(--mast-auto)]"
              style={
                !connected
                  ? { background: "var(--mast-muted)", boxShadow: "none" }
                  : readingsStale
                    ? { background: "var(--mast-warn, #d99)", boxShadow: "none" }
                    : undefined
              }
            />
            {modeText}
          </span>
          <span className="rounded-mast-badge bg-mast-accent-soft px-2 py-0.5 font-mono text-[11.5px] text-mast-accent">
            {currentModel}
          </span>
          <span className="font-mono text-[11px] tabular-nums text-mast-faint">
            {now.toLocaleDateString("zh-CN")} · {now.toLocaleTimeString("zh-CN", { hour12: false })}
          </span>
          <span className="h-[18px] w-px bg-mast-border" />
          <button
            onClick={toggleTheme}
            className="inline-flex rounded-mast-ctl border border-mast-border bg-mast-panel-2 px-[7px] py-[5px] text-mast-muted hover:border-mast-accent hover:text-mast-accent"
            title="切换主题"
          >
            {theme === "dark" ? "☀" : "🌙"}
          </button>
          {/* 两段式作用域 chip，**永远可见**。
              从前是 expName 为空就整个不渲染 —— 在样品硬门控下那恰恰是错的：
              「没样品」是一个会阻断扫描/谱学的状态，必须始终在场，而不是安静
              地什么都不显示。点它打开与右栏同一个选择器。 */}
          <button
            data-testid="scope-chip"
            onClick={() => setScopePicker(true)}
            /* 名字会被截断，所以悬浮必须给得回来。从前 title 里只有两个 ID ——
               看不清名字的人悬上去拿到的是一串他更读不出来的编号。 */
            title={
              `实验：${expName || "未选择"}\n实验 ID: ${scope.data?.experiment?.id ?? "—"}` +
              `\n样品：${smpName || "未选择"}` +
              (scope.data?.sample?.id ? `\n样品 ID: ${scope.data.sample.id}` : "")
            }
            /* 外壳**不再设固定 ch 上限**（#50 那轮把 26ch 调到 38ch，这轮用户
               还是看到「› 样品 Au(111」被拦腰截断）。任何写死的数字都要 ≥ 五个
               子元素 + 4 段 gap + 左右 padding 的总和，而其中两个标签是 CJK：
               `实验` 不是 2ch 是约 3.6ch（1ch = "0" 的宽度，汉字是 1em）。上一轮
               的算式按 2ch 记，一开始就少算了 ~3.6ch，再加上 gap(≈3.8ch) 和
               padding(≈2.9ch)，38ch 仍然小于实际需要的 ~43ch —— 于是两个名字继续
               替所有人挨压，声明的 14ch 依旧兑现不了。
               这次不是换个更大的数字，而是把这道算术**删掉**：外壳只受本行宽度
               约束（父容器 flex-wrap，挤不下就整块换行），能压缩的仍然只有两个
               名字自己的上限。没有算式，就没有算错的算式。 */
            className="inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-mast-badge border border-mast-border px-[9px] py-[3px] text-[11.5px] text-mast-muted hover:border-mast-accent"
          >
            <span className="shrink-0 text-mast-faint">实验</span>
            {/* 混合中英文名称需要显示余量；超长名称可通过悬浮提示读取。 */}
            <span
              className={
                "min-w-0 max-w-[18ch] truncate font-medium " +
                (hasExp ? "text-mast-text" : "text-mast-warn")
              }
            >
              {expName || "未选择"}
            </span>
            <span className="shrink-0 text-mast-faint">›</span>
            <span className="shrink-0 text-mast-faint">样品</span>
            <span
              className={
                "min-w-0 max-w-[18ch] truncate " +
                (hasSmp ? "text-mast-text" : "animate-pulse text-mast-danger")
              }
            >
              {smpName || "未选择"}
            </span>
          </button>
          <ExperimentPicker
            open={scopePicker}
            onClose={() => setScopePicker(false)}
            onPick={(id) => {
              setScopePicker(false);
              switchExp.mutate({ experimentId: id });
            }}
            onCreate={() => setScopePicker(false)}
            currentId={scope.data?.experiment?.id}
          />
        </div>
      </div>

      {/* row 2 — grouped live hardware readouts: [tip state] | [feedback loop] | [scan] */}
      <div className="mast-scroll flex items-center overflow-x-auto border-t border-mast-border px-[18px] py-[7px] font-mono text-xs tabular-nums">
        {/* group: tip state — Bias, Current */}
        <div className="flex flex-none items-center gap-3.5 whitespace-nowrap pr-4">
          <span className="flex items-center gap-1.5">
            <span className="text-mast-faint">B</span>
            <span className="text-mast-text" data-testid="reading-bias">{fmt(r?.bias_v, 4)}</span>
            <span className="text-[10.5px] text-mast-muted">V</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="text-mast-faint">I</span>
            <span className="text-mast-text">{splitSI(r?.current_a, "A").num}</span>
            <span className="text-[10.5px] text-mast-muted">{splitSI(r?.current_a, "A").unit}</span>
          </span>
        </div>

        <span className="h-5 w-px flex-none bg-mast-border" />

        {/* group: feedback loop — Z pos, ctrl, setpoint */}
        <div className="flex flex-none items-center gap-3.5 whitespace-nowrap px-4">
          <span className="flex items-center gap-1.5">
            <span className="text-mast-faint">Z</span>
            <span className="text-mast-text">{splitSI(r?.z_m, "m", { digits: 5 }).num}</span>
            <span className="text-[10.5px] text-mast-muted">{splitSI(r?.z_m, "m", { digits: 5 }).unit}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="text-mast-faint">ctrl</span>
            <span className={zOff ? "text-mast-warn" : "text-mast-text"}>{zOff ? "OFF" : "ON"}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="text-mast-faint">setpoint</span>
            <span className="text-mast-text">{splitSI(r?.setpoint_a, "A").num}</span>
            <span className="text-[10.5px] text-mast-muted">{splitSI(r?.setpoint_a, "A").unit}</span>
          </span>
        </div>

        <span className="h-5 w-px flex-none bg-mast-border" />

        {/* group: scan */}
        <div className="flex flex-none items-center gap-3.5 whitespace-nowrap pl-4">
          <span className="flex items-center gap-1.5">
            <span className="text-mast-faint">scan</span>
            {r?.scan_running ? (
              <span className="inline-flex items-center gap-1.5 text-mast-auto">
                <span className="h-[7px] w-[7px] rounded-full bg-mast-auto" />
                RUNNING
              </span>
            ) : (
              <span className="text-mast-muted">STOPPED</span>
            )}
          </span>
        </div>
      </div>
    </header>
  );
}
