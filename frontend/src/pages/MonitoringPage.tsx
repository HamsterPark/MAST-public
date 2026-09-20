import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Card, DegradedNote, ErrorNote, Section, Spinner } from "@/components/ui";
import { Accordion, useToast } from "@/components/controls";
import { useWsConnection, useWsEvent } from "@/hooks/useWsEvents";
import { VerdictBanner } from "@/components/monitoring/VerdictBanner";
import { DaemonControls } from "@/components/monitoring/DaemonControls";
import { LiveCurrentChart } from "@/components/monitoring/LiveCurrentChart";
import { FeatureSparklines } from "@/components/monitoring/FeatureSparklines";
import { AlertsTable } from "@/components/monitoring/AlertsTable";
import { SegmentBrowser } from "@/components/monitoring/SegmentBrowser";
import { AuxChannels } from "@/components/monitoring/AuxChannels";
import {
  clampAuxWindow,
  clampWindow,
  isStale,
  patchStatusFromWsEvent,
  pollForWindow,
} from "@/lib/monitoring";
import {
  MONITORING_STALE_AFTER_MS,
  MONITORING_STATUS_POLL_MS,
  MONITORING_STATUS_WS_KEEPALIVE_POLL_MS,
  MONITORING_TRACE_POLL_MS,
  MONITORING_TRACE_WS_KEEPALIVE_POLL_MS,
} from "@/lib/pollRates";

type MonitoringStatus = components["schemas"]["MonitoringStatus"];

// 电流监控 — the tunnelling current, recorded natively and continuously.
//
// ── where each number comes from ───────────────────────────────────────────
// The daemon holds the ONE oscilloscope subscription and writes a segment per
// second into its own SQLite store. Everything on this page reads that store:
// routes/monitoring.py never touches the Nanonis TCP port, so nothing here can
// steal lock time from the tip-crash watchdog no matter how hard we poll.
//
// The WebSocket carries SCALARS ONLY, by design — the bus replays just the last
// 100 events and a 1 kHz waveform would evict everything else on it. So push
// tells us WHEN something happened and REST fetches WHAT it was. Concretely:
//   · a `segment` event patches the cached status (verdict + headline numbers)
//     and nudges the trace to refetch, throttled to ≤1/s;
//   · an `alert` event invalidates the alert list;
//   · curves are never pushed.
//
// ── what must never happen ─────────────────────────────────────────────────
// A stopped daemon must not blank the page: the store is read-only-usable and
// browsing history with acquisition down is a supported state. Conversely a
// frozen reading must never look live — hence the stale gate on the banner.

const STATUS_KEY = ["monitoring", "status"] as const;
const TRACE_KEY = ["monitoring", "live-trace"] as const;

/** How far back the feature sparklines look. */
const FEATURE_WINDOW_S = 600;

/**
 * Where the auxiliary channels start.
 *
 * Fifteen minutes, deliberately longer than the current chart's default: Z drift
 * and amplitude wander are minutes-scale phenomena, and a one-minute window of
 * a 20 pm/s drift is a flat line. Same reason these sample at 1 Hz rather than
 * riding the oscilloscope — see docs/v2/design/monitoring_aux_channels.md.
 *
 * Adjustable — the auxiliary channels needed both a short- and long-term view: the picker
 * lives in AuxChannels and reaches 24 h, which is real history — aux samples are
 * persisted for `cm_aux_keep_hours` (168 h), not held in the 300 s ring.
 */
const AUX_WINDOW_DEFAULT_S = 900;

export default function MonitoringPage() {
  const ws = useWsConnection();
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const [windowS, setWindowS] = useState(60);
  const [auxWindowS, setAuxWindowS] = useState(AUX_WINDOW_DEFAULT_S);
  const [paused, setPaused] = useState(false);
  // Read once, not tracked: SegmentBrowser rewrites `?seg=` as the operator
  // clicks around, and re-evaluating this would fight the accordion's own state.
  const [deepLinkedSeg] = useState(
    () => new URLSearchParams(window.location.search).has("seg"),
  );

  // ── status ────────────────────────────────────────────────────────────────
  const status = useQuery({
    queryKey: STATUS_KEY,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/status");
      if (error) throw error;
      return data;
    },
    refetchInterval: ws.polling
      ? MONITORING_STATUS_POLL_MS
      : MONITORING_STATUS_WS_KEEPALIVE_POLL_MS,
  });

  const degraded = status.data?.degraded ?? false;

  // ── live trace ────────────────────────────────────────────────────────────
  // Pausing stops the refetch rather than hiding new data: a paused chart that
  // keeps fetching would jump forward the moment it resumes, and the operator
  // paused it precisely to hold one moment still.
  const trace = useQuery({
    queryKey: [...TRACE_KEY, windowS],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/live-trace", {
        params: { query: { window_s: clampWindow(windowS), max_points: 1200 } },
      });
      if (error) throw error;
      return data;
    },
    // A 6 h window moves by 0.03% between three-second polls and costs the
    // server thousands of envelope rows to answer, so the cadence follows the
    // window. Short windows keep exactly the cadence they had.
    refetchInterval: paused
      ? false
      : pollForWindow(
          windowS,
          ws.polling ? MONITORING_TRACE_POLL_MS : MONITORING_TRACE_WS_KEEPALIVE_POLL_MS,
        ),
  });

  // ── feature series (sparklines) ───────────────────────────────────────────
  //
  // `since` is computed at FETCH time, not baked into the queryKey: the store
  // orders features ASC and applies LIMIT after, so a bare `limit` would return
  // the OLDEST rows in the table, not the newest. Asking for a time window is
  // the only way to get recent history — and putting a moving `now` in the key
  // would mint a new cache entry on every render.
  const features = useQuery({
    queryKey: ["monitoring", "features", FEATURE_WINDOW_S],
    refetchInterval: ws.polling ? 10_000 : 60_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/features", {
        params: {
          query: {
            since: Date.now() / 1000 - FEATURE_WINDOW_S,
            limit: 2000,
            max_points: 300,
          },
        },
      });
      if (error) throw error;
      return data;
    },
  });

  // ── auxiliary channels (Z / qPlus amplitude / Δf) ─────────────────────────
  //
  // Its own query rather than a widened live-trace: different cadence (the
  // daemon samples these at 1 Hz), different window (minutes, not seconds), and
  // a different degradation story — a rig with no qPlus sensor still has Z.
  // Refetch is slow on purpose: the fastest thing on these channels is an
  // amplitude collapse, and 1 Hz sampling plus a 5 s poll already puts that on
  // screen well inside the time it takes anyone to react.
  const auxSeriesQ = useQuery({
    queryKey: ["monitoring", "aux-series", auxWindowS],
    refetchInterval: paused
      ? false
      : pollForWindow(auxWindowS, ws.polling ? 5_000 : 30_000),
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/aux/series", {
        params: { query: { window_s: clampAuxWindow(auxWindowS), max_points: 900 } },
      });
      if (error) throw error;
      return data;
    },
  });

  // ── alerts ────────────────────────────────────────────────────────────────
  const alerts = useQuery({
    queryKey: ["monitoring", "alerts"],
    refetchInterval: ws.polling ? 15_000 : 60_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/alerts", {
        params: { query: { limit: 50 } },
      });
      if (error) throw error;
      return data;
    },
  });

  // ── WS wiring ─────────────────────────────────────────────────────────────
  //
  // The throttle is the load-bearing part. Segments arrive at ~1 Hz and each
  // one invalidating the trace would be one extra request per second on top of
  // the poll — over a Tailscale tunnel that is worse than the poll it replaces.
  // A leading-edge throttle with a trailing catch-up keeps the chart responsive
  // without letting a burst (a reconnect replays up to 100 events) turn into a
  // request storm.
  const lastTraceInvalidate = useRef(0);
  const pendingTrace = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Read inside the trailing timer, not captured by it: a bump queued a moment
  // before the operator hit 暂停 would otherwise still fire and move the chart
  // they just froze. `refetchInterval: false` does not stop an explicit
  // invalidate, so the check has to live here.
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    return () => {
      if (pendingTrace.current) clearTimeout(pendingTrace.current);
    };
  }, []);

  const bumpTrace = useCallback(() => {
    if (pausedRef.current) return;
    const now = Date.now();
    const wait = 1000 - (now - lastTraceInvalidate.current);
    if (wait <= 0) {
      lastTraceInvalidate.current = now;
      qc.invalidateQueries({ queryKey: TRACE_KEY });
      return;
    }
    if (pendingTrace.current) return;
    pendingTrace.current = setTimeout(() => {
      pendingTrace.current = null;
      if (pausedRef.current) return;
      lastTraceInvalidate.current = Date.now();
      qc.invalidateQueries({ queryKey: TRACE_KEY });
    }, wait);
  }, [qc]);

  const onMonitorEvent = useCallback(
    (event: { data: unknown }) => {
      const d = event.data as { kind?: unknown } | null;
      const kind = d && typeof d === "object" ? d.kind : undefined;

      if (kind === "alert") {
        qc.invalidateQueries({ queryKey: ["monitoring", "alerts"] });
      }
      // Both `segment` and `status` fold into the same cache entry; the helper
      // returns undefined for anything it cannot honour (including an empty
      // cache), which leaves the poll authoritative — see lib/monitoring.ts.
      qc.setQueryData(STATUS_KEY, (prev: MonitoringStatus | undefined) =>
        patchStatusFromWsEvent(prev, event.data),
      );
      if (kind === "segment") bumpTrace();
    },
    [qc, bumpTrace],
  );
  useWsEvent("current_monitor", onMonitorEvent);

  // ── controls ──────────────────────────────────────────────────────────────
  const start = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/monitoring/start");
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(d?.note || "已启动", d?.ok ? "ok" : "err");
      qc.invalidateQueries({ queryKey: STATUS_KEY });
    },
    onError: (e) => toast(`启动失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const stop = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/monitoring/stop");
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(d?.note || "已停止", d?.ok ? "ok" : "err");
      qc.invalidateQueries({ queryKey: STATUS_KEY });
    },
    onError: (e) => toast(`停止失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  // ── freshness ─────────────────────────────────────────────────────────────
  //
  // Driven off the payload's own timestamp, not react-query's dataUpdatedAt: a
  // successful fetch that returns the SAME old segment refreshes dataUpdatedAt
  // while the data is exactly as dead as before. A 1 s tick re-evaluates it so
  // the banner greys out on its own, without waiting for the next response.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const lastTs = status.data?.latest?.ts ?? status.data?.last_segment_ts ?? null;
  const stale = isStale(lastTs, nowMs, MONITORING_STALE_AFTER_MS);
  const ageS = lastTs ? Math.max(0, nowMs / 1000 - lastTs) : null;

  return (
    <div>
      <Section
        title="电流监控"
        subtitle="常开采集隧道电流：逐段落库、判级告警，并留作针尖状态的训练语料"
      >
        {/* 实现细节归代码注释，页面只留操作者视角的事实。
            删掉的是「独立守护进程 / 曲线走 REST / WebSocket 推标量」——
            那是这一页怎么取数，不是用户需要知道的事。
            留下的是「不占用 Nanonis 通信端口」：它回答的是一个真问题
            ——「我一直开着这一页，会不会影响正在跑的实验」。 */}
        <p className="mb-4 text-sm text-mast-muted">
          本页只读已落库的数据，<strong>不占用 Nanonis 通信端口</strong>——
          开着它不会影响正在跑的实验。
        </p>

        {status.isPending ? (
          <Spinner />
        ) : status.isError ? (
          <ErrorNote error={status.error} />
        ) : (
          <div className="space-y-4">
            {degraded && <DegradedNote what="电流监控" />}

            <VerdictBanner status={status.data} stale={stale} ageS={ageS} />

            <Card>
              <DaemonControls
                status={status.data}
                onStart={() => start.mutate()}
                onStop={() => stop.mutate()}
                starting={start.isPending}
                stopping={stop.isPending}
                wsWarn={ws.warn}
              />
            </Card>

            <Card>
              <div className="mb-2 text-sm font-medium text-mast-text">实时电流</div>
              <LiveCurrentChart
                tS={trace.data?.t_s ?? []}
                iMinA={trace.data?.i_min_a ?? []}
                iMaxA={trace.data?.i_max_a ?? []}
                nSegments={trace.data?.n_segments ?? 0}
                nGaps={trace.data?.n_gaps ?? 0}
                windowS={windowS}
                onWindowChange={setWindowS}
                paused={paused}
                onPausedChange={setPaused}
                stale={stale}
                degraded={degraded || (trace.data?.degraded ?? false)}
              />
            </Card>

            <Card>
              <div className="mb-2 text-sm font-medium text-mast-text">
                辅助通道
                <span className="ml-2 text-xs font-normal text-mast-muted">
                  Z 位置 · qPlus 振幅 · 频率偏移 · dI/dV
                </span>
              </div>
              {/* No spinner: by the time this renders the status poll has
                  already answered, and AuxChannels has an honest empty state for
                  every reason the panel can be blank (daemon down, this rig has
                  no qPlus, window not filled yet). A spinner would only be able
                  to say "wait" where those say WHY. */}
              <AuxChannels
                aux={status.data?.aux ?? null}
                series={auxSeriesQ.data}
                degraded={degraded || (auxSeriesQ.data?.degraded ?? false)}
                windowS={auxWindowS}
                onWindowChange={setAuxWindowS}
              />
            </Card>

            <Card>
              <div className="mb-2 text-sm font-medium text-mast-text">
                特征趋势
                <span className="ml-2 text-xs font-normal text-mast-muted">
                  近 {FEATURE_WINDOW_S / 60} 分钟
                  {features.data?.thinned && "（已按告警等级抽稀，最严重的一条必被保留）"}
                </span>
              </div>
              {features.isPending ? (
                <Spinner />
              ) : (
                <FeatureSparklines
                  rows={features.data?.rows ?? []}
                  degraded={degraded || (features.data?.degraded ?? false)}
                  detail={features.data?.detail}
                />
              )}
            </Card>

            <Card>
              <div className="mb-2 text-sm font-medium text-mast-text">
                告警历史
                {alerts.data?.total ? (
                  <span className="ml-2 text-xs font-normal text-mast-muted">
                    共 {alerts.data.total} 条
                  </span>
                ) : null}
              </div>
              {alerts.isPending ? (
                <Spinner />
              ) : (
                // ⚠️ `degraded` 这一个变量来自 **status** 端点。告警表读的是
                // 另一个端点，它有自己的 `degraded` —— 漏掉它，一次告警查询
                // 失败就会画成「暂无告警——这是好消息」。上面实时曲线与辅助
                // 通道两处都写了 `degraded || (…data?.degraded)`，只有这里没写。
                <AlertsTable
                  alerts={alerts.data?.alerts ?? []}
                  degraded={degraded || (alerts.data?.degraded ?? false)}
                  detail={alerts.data?.detail}
                />
              )}
            </Card>

            {/* Opened by a `?seg=` deep link : a current-monitor event
                tile links straight at one segment's waveform, and landing on a
                page where the panel holding it is collapsed is a dead link. */}
            <Accordion
              title="段浏览 · 人工标注（好针尖 / 坏针尖）"
              defaultOpen={deepLinkedSeg}
            >
              <SegmentBrowser degraded={degraded} />
            </Accordion>
          </div>
        )}
      </Section>
      {node}
    </div>
  );
}
