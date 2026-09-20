import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, ErrorNote, DegradedNote } from "@/components/ui";
import { Button } from "@/components/controls";
import { useToast } from "@/components/controls";
import { TimeTraceChart } from "@/components/experimental/TimeTraceChart";

// 长期监控 — periodic single-channel reads appended to a CSV by a core daemon.
// Typed replacement for gui/exp_capture.build_monitor_panel:
//  · channel (string: index or 'current') + interval slider
//  · 开始监控 / 停止监控 (POST monitor/start | monitor/stop)
//  · live status (GET monitor/status, poll 2s) — running / count / last value / csv path
// The daemon lives in the core; closing this page does NOT stop it.

function Reading({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between border-b border-mast-border py-1.5 last:border-b-0">
      <span className="text-sm text-mast-muted">{label}</span>
      <span className="text-sm font-medium tabular-nums">{value}</span>
    </div>
  );
}

export function MonitorPanel() {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const [channel, setChannel] = useState("current");
  const [intervalS, setIntervalS] = useState(5);

  const status = useQuery({
    queryKey: ["experimental", "monitor", "status"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experimental/monitor/status");
      if (error) throw error;
      return data;
    },
    refetchInterval: 2000,
  });

  const st = status.data;
  const running = st?.running ?? false;

  // Improvement 1 — real channel dropdown, reusing the capture tab's channel list
  // (was a free-text index box because the monitor had no channel enumeration).
  const signals = useQuery({
    queryKey: ["experimental", "signals"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experimental/signals");
      if (error) throw error;
      return data;
    },
    staleTime: 60_000,
  });
  const channels = signals.data?.channels ?? [];

  // Improvement 2 — live chart. The monitor daemon reports only the LAST value per
  // poll, so accumulate a client-side series (one point per NEW sample = count
  // change) while this page is open. The full history is always in the CSV.
  const [series, setSeries] = useState<{ t: number[]; v: number[] }>({ t: [], v: [] });
  const lastCount = useRef(-1);
  const t0 = useRef<number | null>(null);
  const clearSeries = () => {
    setSeries({ t: [], v: [] });
    lastCount.current = -1;
    t0.current = null;
  };
  useEffect(() => {
    if (!st || !running) return;
    const count = st.count ?? 0;
    if (count !== lastCount.current && st.last_value != null) {
      lastCount.current = count;
      if (t0.current == null) t0.current = Date.now() / 1000;
      const elapsed = Date.now() / 1000 - t0.current;
      setSeries((s) => ({
        t: [...s.t, elapsed].slice(-3000),
        v: [...s.v, st.last_value as number].slice(-3000),
      }));
    }
  }, [st, running]);

  const startMut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/experimental/monitor/start", {
        body: { channel, interval_s: intervalS },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d.ok) {
        toast("已开始监控");
        clearSeries();
      } else toast(d.message || "无法开始监控", "err");
      qc.invalidateQueries({ queryKey: ["experimental", "monitor", "status"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  const stopMut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/experimental/monitor/stop", {});
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d.ok) toast("已停止监控");
      else toast(d.message || "停止失败", "err");
      qc.invalidateQueries({ queryKey: ["experimental", "monitor", "status"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  return (
    <div className="space-y-3">
      {node}
      <p className="text-xs text-mast-muted">
        长期监控 / Long-term monitor — 每隔设定秒数读取一次所选信号通道，持续追加到 CSV
        长期保存。后台线程记录，切走本页也继续，直到点「停止监控」或程序退出。适合过夜观察漂移 /
        稳定性。
      </p>

      {status.isError && <ErrorNote error={status.error} />}
      {st?.degraded && <DegradedNote what="长期监控" />}

      <Card className="space-y-3">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">监控通道</span>
            <select
              value={channel}
              onChange={(e) => setChannel(e.target.value)}
              disabled={running}
              className="w-60 rounded-md border border-mast-border bg-mast-bg px-2 py-1.5 text-sm disabled:opacity-50"
            >
              <option value="current">current（隧道电流）</option>
              {channels.map((c) => (
                <option key={c.index} value={String(c.index)}>
                  {c.index}: {c.name}
                  {c.unit ? ` (${c.unit})` : ""}
                </option>
              ))}
            </select>
            <span className="text-mast-muted">
              {signals.isLoading
                ? "加载通道…"
                : signals.data?.degraded
                  ? "⚠ 未连硬件，为默认通道列表"
                  : `${channels.length} 个通道`}
              {" · "}
              <button
                type="button"
                onClick={() => signals.refetch()}
                disabled={running || signals.isFetching}
                className="text-mast-accent hover:underline disabled:opacity-50"
              >
                刷新
              </button>
            </span>
          </label>
          <label className="flex flex-1 flex-col gap-1 text-xs">
            <span className="text-mast-muted">采样间隔：{intervalS.toFixed(1)} 秒/点</span>
            <input
              type="range"
              min={0.5}
              max={600}
              step={0.5}
              value={intervalS}
              onChange={(e) => setIntervalS(Number(e.target.value))}
              disabled={running}
              className="w-full"
            />
          </label>
          <Button
            variant="primary"
            onClick={() => startMut.mutate()}
            disabled={running || startMut.isPending}
          >
            {startMut.isPending ? "启动中…" : "开始监控"}
          </Button>
          <Button variant="danger" onClick={() => stopMut.mutate()} disabled={!running || stopMut.isPending}>
            {stopMut.isPending ? "停止中…" : "停止监控"}
          </Button>
        </div>
      </Card>

      <Card>
        <div className="mb-2 flex items-center gap-2">
          <span className="text-sm font-semibold">监控状态</span>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs ${
              running ? "bg-mast-auto-bg text-mast-auto" : "bg-mast-bg text-mast-muted"
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${running ? "bg-mast-auto" : "bg-mast-muted"}`} />
            {running ? "运行中" : "未运行"}
          </span>
        </div>
        {!st?.started && !running ? (
          <p className="text-sm text-mast-muted">未开始监控。</p>
        ) : (
          <>
            <Reading label="通道" value={st?.channel || "—"} />
            <Reading label="采样间隔" value={st ? `${st.interval_s.toFixed(1)} s` : "—"} />
            <Reading label="已采集点数" value={st?.count ?? 0} />
            <Reading
              label="最新值"
              value={
                st?.last_value != null
                  ? `${st.last_value.toPrecision(6)}${st.unit ? " " + st.unit : ""}`
                  : "—"
              }
            />
            <Reading label="最新时间" value={st?.last_t || "—"} />
            <Reading
              label="CSV 路径"
              value={
                st?.csv_path ? (
                  <span className="font-mono text-xs" title={st.csv_path}>
                    {st.csv_path}
                  </span>
                ) : (
                  "—"
                )
              }
            />
            {st?.error && <p className="mt-2 text-sm text-mast-danger">{st.error}</p>}
          </>
        )}
      </Card>

      {/* Improvement 2 — real-time chart of the monitored channel */}
      <Card>
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm font-semibold">实时曲线</span>
          {series.v.length > 0 && (
            <button
              type="button"
              onClick={clearSeries}
              className="text-xs text-mast-muted hover:text-mast-accent"
            >
              清空
            </button>
          )}
        </div>
        {series.v.length === 0 ? (
          <p className="py-6 text-center text-sm text-mast-muted">
            {running
              ? `等待采样点…（每 ${intervalS.toFixed(1)} 秒一个）`
              : "开始监控后，这里实时显示所选通道随时间的曲线。"}
          </p>
        ) : (
          <>
            <TimeTraceChart
              timestampsS={series.t}
              samples={series.v}
              unit={st?.unit || ""}
              channelName={st?.channel || channel}
            />
            <p className="mt-1 text-xs text-mast-faint">
              本页打开期间累积的 {series.v.length} 个点（横轴＝经过秒数）；完整历史见 CSV。
            </p>
          </>
        )}
      </Card>
    </div>
  );
}
