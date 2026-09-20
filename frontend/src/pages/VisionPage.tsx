import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Section, Card, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { SubTabs } from "@/components/controls";
import { ScanMapPanel } from "@/components/vision/ScanMapPanel";
import { CoarseMapPanel } from "@/components/vision/CoarseMapPanel";
import { VacuumInterlockStrip } from "@/components/vision/VacuumInterlockStrip";
import { PulseRibbon } from "@/components/vision/PulseRibbon";
import { FftChart } from "@/components/vision/FftChart";
import { VisionBufferPanel } from "@/components/vision/VisionBufferPanel";
import { RecentFrameThumb } from "@/components/vision/RecentFrameThumb";
import { MonitorPanel } from "@/components/vision/MonitorPanel";
import { useCoarseMap } from "@/hooks/useCoarseMap";
import { useStickyTab } from "@/hooks/useStickyTab";

// Domain I — VisionPage (FULL parity rebuild, Wave B). Reproduces every old
// Gradio surface that lived under 视觉 / 扫描 / 实验性功能, organized as flat
// in-page SubTabs (no nested gr.Tabs → no freeze):
//   扫描地图   — Konva live map, /api/scan-map, poll 3s
//   视觉脉冲   — /api/vision/pulse ribbon, poll 2s
//   近期帧     — /api/vision/recent thumbnail grid, poll 5s
//   视觉缓冲   — full event table + filters + sparkline, /api/vision/buffer
//   信号捕获+FFT — POST /api/experimental/fft (uPlot)
//   长期监控   — start/stop + status poll
//   拼图       — POST /api/experimental/mosaic
// Every read renders loading / error / degraded / empty.

// ── number formatting helpers (metres → human) ───────────────────────────────
function fmtNm(m?: number | null): string {
  if (m == null) return "—";
  return `${(m * 1e9).toLocaleString(undefined, { maximumFractionDigits: 1 })} nm`;
}
function fmtSize(w?: number | null, h?: number | null): string {
  if (w == null || h == null) return "—";
  return `${(w * 1e9).toFixed(0)}×${(h * 1e9).toFixed(0)} nm`;
}
function fmtTime(epochS?: number | null): string {
  if (!epochS) return "—";
  try {
    return new Date(epochS * 1000).toLocaleTimeString();
  } catch {
    return "—";
  }
}

// ── queries ──────────────────────────────────────────────────────────────────
// (the scan-map + analysis queries live in components/vision/ScanMapPanel.tsx,
//  shared with the chat page's embedded copy of the same map)
function usePulse() {
  return useQuery({
    queryKey: ["vision", "pulse"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/vision/pulse");
      if (error) throw error;
      return data;
    },
    refetchInterval: 2000,
  });
}
function useRecent() {
  return useQuery({
    queryKey: ["vision", "recent"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/vision/recent");
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });
}

type SubTab = "map" | "pulse" | "recent" | "buffer" | "fft" | "monitor" | "mosaic";

const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: "map", label: "扫描地图" },
  { id: "pulse", label: "视觉脉冲" },
  { id: "recent", label: "近期帧" },
  { id: "buffer", label: "视觉缓冲" },
  { id: "fft", label: "信号捕获 + FFT" },
  { id: "monitor", label: "长期监控" },
  { id: "mosaic", label: "拼图" },
];

// ─────────────────────────────────────────────────────────────────────────────
export default function VisionPage() {
  const [tab, setTab] = useStickyTab<SubTab>(
    "vision", SUB_TABS.map((t) => t.id), "map");
  return (
    <div>
      <SubTabs<SubTab> value={tab} onChange={setTab} tabs={SUB_TABS} />
      {tab === "map" && <ScanMapSection />}
      {tab === "pulse" && <PulseSection />}
      {tab === "recent" && <RecentFramesSection />}
      {tab === "buffer" && (
        <Section title="视觉缓冲">
          <VisionBufferPanel />
        </Section>
      )}
      {tab === "fft" && <FftSection />}
      {tab === "monitor" && (
        <Section title="长期监控">
          <MonitorPanel />
        </Section>
      )}
      {tab === "mosaic" && <MosaicSection />}
    </div>
  );
}

// ── 视觉脉冲 ───────────────────────────────────────────────────────────────────
function PulseSection() {
  const q = usePulse();
  return (
    <Section title="视觉脉冲">
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data && (q.data.degraded ? <DegradedNote what="视觉脉冲" /> : <PulseRibbon pulse={q.data} />)}
    </Section>
  );
}

// ── 扫描地图 ───────────────────────────────────────────────────────────────────
function ScanMapImporter() {
  const qc = useQueryClient();
  const [path, setPath] = useState("");
  const mut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/scan-map/import", { body: { path, recursive: true } });
      if (error) throw error;
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["scan-map"] }),
  });
  const res = mut.data;
  return (
    <Card className="space-y-2">
      <div className="text-xs text-mast-muted">
        导入你自己的扫描 / 谱到地图（存在搜索目录之外、或较旧的文件）：填一个 .sxm/.dat/.3ds 文件或文件夹路径。
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <input
          className="min-w-[280px] flex-1 rounded border border-mast-border bg-mast-bg px-2 py-1.5 font-mono text-sm"
          placeholder="D:\\MAST-data\\working-sessions  或  D:\\data\\scan_042.sxm"
          value={path}
          onChange={(e) => setPath(e.target.value)}
        />
        <button
          className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
          onClick={() => mut.mutate()}
          disabled={mut.isPending || !path.trim()}
        >
          {mut.isPending ? "导入中…" : "导入到地图"}
        </button>
      </div>
      {mut.isError && <ErrorNote error={mut.error} />}
      {res && (
        <div className={`text-xs ${res.ok ? "text-mast-accent" : "text-mast-warn"}`}>{res.message}</div>
      )}
    </Card>
  );
}

// 地图分析：把 AI 用来决策的那套程序判据摆出来给人看。
//
// 调的是和 agent 工具完全同一个后端计算（mast.io.map_analysis），所以面板上
// 看到的结论就是 agent 会照着做的结论 —— 这正是这个按钮存在的意义：那些判断
// 由程序做出，人有权检查。按需触发，不进 3 秒轮询（查询在 ScanMapPanel）。

function fmtPct(v?: number | null): string {
  if (v == null) return "—";
  return `${v.toFixed(v < 1 ? 2 : 1)}%`;
}

const STRATEGY_LABEL: Record<string, string> = {
  center_first: "中心优先（压电蠕变最小）",
  perimeter_inward: "外圈→内圈（可用面积利用最大化）",
};

function MapAnalysisPanel({
  analysis,
  onAnalyze,
  isFetching,
  isError,
  error,
}: {
  analysis?: components["schemas"]["ScanMapAnalysisResponse"] | null;
  onAnalyze: () => void;
  isFetching: boolean;
  isError: boolean;
  error: unknown;
}) {
  const qc = useQueryClient();
  const backfill = useMutation({
    mutationFn: async (body: { direction: string; steps: number; note: string }) => {
      const { data, error: e } = await api.POST("/api/scan-map/coarse-move", { body });
      if (e) throw e;
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["scan-map"] });
      onAnalyze();
    },
  });
  const [dir, setDir] = useState("");
  const [steps, setSteps] = useState("");

  const zones = analysis?.damage_counts ?? {};
  const zoneLabels: Record<string, string> = {
    tip_shape: "修针尖",
    pulse: "电脉冲",
    crash: "撞针",
    approach: "进针",
    manual_avoid: "人工避让",
  };

  return (
    <Card className="min-w-[280px] max-w-[420px] space-y-2 text-sm">
      <div className="flex items-center justify-between gap-3">
        <span className="font-medium">地图分析</span>
        <button
          className="rounded bg-mast-accent/20 px-2.5 py-1 text-xs text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
          onClick={onAnalyze}
          disabled={isFetching}
        >
          {isFetching ? "分析中…" : analysis ? "重新分析" : "分析地图"}
        </button>
      </div>
      <div className="text-xs text-mast-muted">
        AI 判断「扫了哪 / 哪不能去 / 下一步去哪 / 该不该换区」用的就是这份程序算出的结论。
      </div>

      {isError && <ErrorNote error={error} />}
      {analysis?.degraded && <DegradedNote what="地图分析" />}
      {analysis && !analysis.degraded && (
        <div className="space-y-2">
          <div className="flex justify-between gap-4">
            <span className="text-mast-muted">已扫面积</span>
            <span className="tabular-nums">{fmtPct(analysis.coverage_pct)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-mast-muted" title="未被破坏的面积。已扫过 ≠ 不可用">
              可用面积
            </span>
            <span className="tabular-nums">{fmtPct(analysis.usable_pct)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-mast-muted" title="既没被破坏、也还没扫过 —— 还有多少新地方">
              可用且未扫
            </span>
            <span className="tabular-nums">{fmtPct(analysis.usable_unscanned_pct)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-mast-muted">选点策略</span>
            <span className="text-right text-xs">
              {STRATEGY_LABEL[analysis.strategy] ?? analysis.strategy ?? "—"}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-mast-muted" title="XY 粗动会让旧坐标整体失效；旧代次标记在图上淡显">
              坐标代次
            </span>
            <span className="tabular-nums">
              第 {analysis.current_epoch} 代
              {analysis.markers_total > analysis.markers_current_epoch && (
                <span className="text-mast-muted">
                  {" "}
                  （{analysis.markers_total - analysis.markers_current_epoch} 条旧标记已淡显）
                </span>
              )}
            </span>
          </div>

          {Object.keys(zones).length > 0 && (
            <div className="border-t border-mast-border pt-2">
              <div className="mb-1 text-xs text-mast-muted">避让区</div>
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                {Object.entries(zones).map(([k, n]) => (
                  <span key={k}>
                    {zoneLabels[k] ?? k} × {n}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="border-t border-mast-border pt-2">
            <div className="mb-1 text-xs text-mast-muted">建议下一个位置</div>
            {analysis.next_position?.x_m != null ? (
              <div className="text-xs">
                <div className="tabular-nums">
                  ({fmtNm(analysis.next_position.x_m)}, {fmtNm(analysis.next_position.y_m)})
                </div>
                <div className="text-mast-muted">{analysis.next_position.reason}</div>
                <div className="text-mast-muted">
                  剩余候选 {analysis.next_position.candidates_left}
                  {analysis.route_truncated && "（已达搜索上限，实际更多）"}
                </div>
              </div>
            ) : (
              <div className="text-xs text-mast-warn">当前策略下已无可用位置。</div>
            )}
          </div>

          {(analysis.coarse_advice?.reasons?.length ?? 0) > 0 && (
            <div className="border-t border-mast-border pt-2">
              <div
                className={`mb-1 text-xs ${
                  analysis.coarse_advice?.suggest ? "text-mast-warn" : "text-mast-muted"
                }`}
              >
                {analysis.coarse_advice?.suggest ? "建议粗动换区" : "换区说明"}
              </div>
              <ul className="list-inside list-disc space-y-0.5 text-xs text-mast-muted">
                {(analysis.coarse_advice?.reasons ?? []).map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          {(analysis.sts_total > 0 || analysis.pending_plan_steps > 0) && (
            <div className="flex flex-wrap gap-x-4 border-t border-mast-border pt-2 text-xs text-mast-muted">
              {analysis.sts_total > 0 && <span>谱点 {analysis.sts_total}</span>}
              {analysis.pending_plan_steps > 0 && (
                <span>计划待办 {analysis.pending_plan_steps} 步</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* 手动粗动补记：MAST 侦测不到直接在 Nanonis 里做的粗动，
          而那之后所有旧标记的坐标都已经指向另一片表面了。 */}
      <div className="border-t border-mast-border pt-2">
        <div className="mb-1 text-xs text-mast-muted">
          在 Nanonis 里手动粗动过？补记一次，地图从此进入新坐标代次（不会改历史记录）。
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <select
            className="rounded border border-mast-border bg-mast-bg px-1.5 py-1 text-xs"
            value={dir}
            onChange={(e) => setDir(e.target.value)}
          >
            <option value="">方向未知</option>
            <option value="x+">x+</option>
            <option value="x-">x-</option>
            <option value="y+">y+</option>
            <option value="y-">y-</option>
          </select>
          <input
            className="w-20 rounded border border-mast-border bg-mast-bg px-1.5 py-1 text-xs tabular-nums"
            placeholder="步数"
            value={steps}
            onChange={(e) => setSteps(e.target.value.replace(/[^0-9]/g, ""))}
          />
          <button
            // `mast-fg` is not a token (the body colour is `mast-text`) — the
            // hover state did nothing at all. Same shape as ①/#61.
            className="rounded bg-mast-panel px-2 py-1 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50"
            onClick={() =>
              backfill.mutate({ direction: dir, steps: Number(steps) || 0, note: "面板补记" })
            }
            disabled={backfill.isPending}
          >
            {backfill.isPending ? "记录中…" : "补记一次粗动"}
          </button>
        </div>
        {backfill.isError && <ErrorNote error={backfill.error} />}
        {backfill.data && (
          <div className={`mt-1 text-xs ${backfill.data.ok ? "text-mast-accent" : "text-mast-warn"}`}>
            {backfill.data.message}
          </div>
        )}
      </div>
    </Card>
  );
}

function ScanMapSection() {
  const coarseQ = useCoarseMap();
  return (
    <Section title="扫描地图">
      <p className="mb-3 text-xs text-mast-muted">
        实验助手的可视化核心：实时扫描框 + 针尖位置 + 所有带位置的历史操作（扫图 / STS / 电脉冲 / 修针尖 /
        进针 / 撞针 / 粗动换区 / 移动 / 手动），并把真实 .sxm 缩略图按其舞台坐标铺成表面地图。每 3 秒刷新。
        计划路线与候选序列按执行顺序标号；图层与显示范围可在下方切换。
      </p>
      <ScanMapPanel
        importer={<ScanMapImporter />}
        analysisPanel={(a) => (
          <MapAnalysisPanel
            analysis={a.analysis}
            onAnalyze={a.onAnalyze}
            isFetching={a.isFetching}
            isError={a.isError}
            error={a.error}
          />
        )}
      />

      {/* 第二张地图：样品台尺度。上面那张是压电量程内的 ±1.5 µm（单位米，
          只看当前代次），这张是整个样品（单位步，跨全部代次）。刻意分开画：
          两者回答的是不同问题，混在一起只会让尺度差消失。 */}
      <div className="flex flex-wrap items-start gap-4 pt-2">
        <CoarseMapPanel
          data={coarseQ.data}
          isPending={coarseQ.isPending}
          isError={coarseQ.isError}
          error={coarseQ.error}
          isFetching={coarseQ.isFetching}
          onRefresh={() => void coarseQ.refetch()}
        />
        <div className="min-w-[300px] flex-1">
          <VacuumInterlockStrip />
        </div>
      </div>
    </Section>
  );
}

// ── 近期标注帧 ─────────────────────────────────────────────────────────────────
function RecentFramesSection() {
  const q = useRecent();
  const data = q.data;
  const frames = data?.frames ?? [];
  return (
    <Section title="近期标注帧">
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {data && data.degraded && <DegradedNote what="近期标注帧" />}
      {data && !data.degraded && frames.length === 0 && <EmptyNote label="暂无近期视觉事件。" />}
      {data && !data.degraded && frames.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {frames.map((f) => (
            <Card key={f.seqno} className="space-y-2 p-3">
              <RecentFrameThumb frame={f} alt={f.summary || f.kind} />
              <div className="space-y-1 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{f.kind || "事件"}</span>
                  <span
                    className={
                      f.severity?.toLowerCase() === "critical"
                        ? "text-mast-danger"
                        : f.severity?.toLowerCase() === "warn" || f.severity?.toLowerCase() === "warning"
                          ? "text-mast-warn"
                          : "text-mast-muted"
                    }
                  >
                    {f.severity || "info"}
                  </span>
                </div>
                <div className="text-mast-muted">{fmtTime(f.t_wall)} · #{f.seqno}</div>
                {f.summary && <div className="line-clamp-2 text-mast-text">{f.summary}</div>}
                {f.file_path && (
                  <div className="truncate text-mast-muted" title={f.file_path}>
                    {f.file_path}
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </Section>
  );
}

// ── 信号捕获 + FFT ─────────────────────────────────────────────────────────────
// Parity note: the live hardware capture (Osci1T / poll fast-path) runs INSIDE
// the core and has no read-only HTTP endpoint exposed (it streams large sample
// arrays that must never cross the wire). What IS exposed is the pure-compute FFT
// (POST /api/experimental/fft). We reproduce the old FFT controls (窗函数 / 输出 /
// 通道名 / 单位 / 采样率) and let the operator paste/load a captured trace.
function FftSection() {
  const [samplesText, setSamplesText] = useState("");
  const [fsHz, setFsHz] = useState("1000");
  const [windowFn, setWindowFn] = useState<"hann" | "hamming" | "rect">("hann");
  const [output, setOutput] = useState<"magnitude" | "power">("magnitude");
  const [channelName, setChannelName] = useState("signal");
  const [unit, setUnit] = useState("A");

  const mut = useMutation({
    mutationFn: async () => {
      const samples = samplesText
        .split(/[\s,;]+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map(Number)
        .filter((n) => Number.isFinite(n));
      const fs = Number(fsHz);
      const { data, error } = await api.POST("/api/experimental/fft", {
        body: {
          samples,
          fs_hz: Number.isFinite(fs) && fs > 0 ? fs : null,
          window: windowFn,
          output,
          detrend: true,
          unit,
          channel_name: channelName,
        },
      });
      if (error) throw error;
      return data;
    },
  });

  const result = mut.data;
  return (
    <Section title="信号捕获 + FFT">
      <p className="mb-3 text-xs text-mast-muted">
        选任意信号通道、短时高频记录其 trace，再对采集数据做 FFT 频谱。硬件捕获在内核中运行（大样本数组永不过线）；
        此处对一段已采集的时域 trace 做单边 rfft：粘贴样本（空格 / 逗号 / 换行分隔），填采样率即可得频谱。
      </p>
      <Card className="space-y-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">采样率 fs (Hz)</span>
            <input
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={fsHz}
              onChange={(e) => setFsHz(e.target.value)}
              inputMode="decimal"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">窗函数</span>
            <select
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={windowFn}
              onChange={(e) => setWindowFn(e.target.value as "hann" | "hamming" | "rect")}
            >
              <option value="hann">Hann</option>
              <option value="hamming">Hamming</option>
              <option value="rect">矩形 / 无窗</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">输出</span>
            <select
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={output}
              onChange={(e) => setOutput(e.target.value as "magnitude" | "power")}
            >
              <option value="magnitude">幅度 |FFT|</option>
              <option value="power">功率谱 PSD</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">通道名</span>
            <input
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={channelName}
              onChange={(e) => setChannelName(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">单位</span>
            <input
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={unit}
              onChange={(e) => setUnit(e.target.value)}
            />
          </label>
        </div>
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-mast-muted">时域样本</span>
          <textarea
            className="h-24 rounded border border-mast-border bg-mast-bg px-2 py-1.5 font-mono text-xs"
            placeholder="0.0, 0.12, 0.23, …"
            value={samplesText}
            onChange={(e) => setSamplesText(e.target.value)}
          />
        </label>
        <div>
          <button
            className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
            onClick={() => mut.mutate()}
            disabled={mut.isPending}
          >
            {mut.isPending ? "计算中…" : "计算 FFT"}
          </button>
        </div>

        {mut.isError && <ErrorNote error={mut.error} />}
        {result && result.degraded && <DegradedNote what="FFT 计算" />}
        {result && !result.degraded && !result.ok && (
          <EmptyNote label="无法计算（样本太少，至少 4 个，或采样率缺失）。" />
        )}
        {result && !result.degraded && result.ok && (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-mast-muted">
              <span>N={result.n_samples}</span>
              <span>fs={result.fs_hz.toFixed(2)} Hz</span>
              <span>Nyquist={result.nyquist_hz.toFixed(2)} Hz</span>
              <span>Δf={result.df_hz.toFixed(3)} Hz</span>
              <span>窗={result.window}</span>
            </div>
            <FftChart fft={result} />
          </div>
        )}
      </Card>
    </Section>
  );
}

// ── 拼图（图像拼接） ─────────────────────────────────────────────────────────────
function MosaicSection() {
  const [directory, setDirectory] = useState("");
  const [channel, setChannel] = useState("Z");
  const [cmap, setCmap] = useState("viridis");
  const [recursive, setRecursive] = useState(false);
  const [lineNormalize, setLineNormalize] = useState(true);
  const [label, setLabel] = useState("");

  const mut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/experimental/mosaic", {
        body: {
          directory,
          channel,
          recursive,
          line_normalize: lineNormalize,
          cmap,
          label,
        },
      });
      if (error) throw error;
      return data;
    },
  });

  const result = mut.data;
  const CMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "afmhot", "hot"];
  return (
    <Section title="拼图（图像拼接）">
      <p className="mb-3 text-xs text-mast-muted">
        按每张 .sxm 的真实 stage xy 把一个目录里的扫描拼成一张大画布总览（非网格预览）。返回画布 PNG（ndarray
        永不过线）。换样品 = 换目录 = 新画布。
      </p>
      <Card className="space-y-3">
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-mast-muted">数据目录（.sxm 文件夹）</span>
          <input
            className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 font-mono text-sm"
            placeholder="D:\\data\\sample-A"
            value={directory}
            onChange={(e) => setDirectory(e.target.value)}
          />
        </label>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">通道</span>
            <input
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={channel}
              onChange={(e) => setChannel(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">配色 (cmap)</span>
            <select
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={cmap}
              onChange={(e) => setCmap(e.target.value)}
            >
              {CMAPS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">样品标签（写入文件名）</span>
            <input
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
          </label>
          <div className="flex flex-col justify-end gap-1.5 text-xs">
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={lineNormalize}
                onChange={(e) => setLineNormalize(e.target.checked)}
              />
              逐行展平 (Z 推荐)
            </label>
            <label className="inline-flex items-center gap-2">
              <input type="checkbox" checked={recursive} onChange={(e) => setRecursive(e.target.checked)} />
              递归子目录
            </label>
          </div>
        </div>
        <div>
          <button
            className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
            onClick={() => mut.mutate()}
            disabled={mut.isPending || !directory.trim()}
          >
            {mut.isPending ? "拼接中…" : "生成拼图"}
          </button>
        </div>

        {mut.isError && <ErrorNote error={mut.error} />}
        {result && result.degraded && <DegradedNote what="拼图" />}
        {result && !result.degraded && !result.ok && (
          <EmptyNote label={result.error || "未找到可拼接的有效扫描。"} />
        )}
        {result && !result.degraded && result.ok && (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-mast-muted">
              <span>已放置 {result.placed}/{result.n_input}</span>
              {result.canvas_px && (
                <span>
                  画布 {result.canvas_px[0]}×{result.canvas_px[1]} px
                </span>
              )}
              {result.res_m_per_px > 0 && (
                <span>分辨率 {(result.res_m_per_px * 1e9).toFixed(2)} nm/px</span>
              )}
              {result.angle_warning && <span className="text-mast-warn">⚠ 存在旋转扫描，拼接为近似</span>}
            </div>
            {result.image_b64 ? (
              <img
                src={`data:image/png;base64,${result.image_b64}`}
                alt="mosaic canvas"
                className="max-w-full rounded-lg border border-mast-border"
              />
            ) : (
              <EmptyNote label="拼接成功但未生成预览图。" />
            )}
            {result.scans_meta && result.scans_meta.length > 0 && (
              <details className="text-xs">
                <summary className="cursor-pointer text-mast-muted">
                  输入扫描 ({result.scans_meta.length})
                </summary>
                <ul className="mt-2 space-y-1 font-mono text-mast-muted">
                  {result.scans_meta.map((s, i) => (
                    <li key={`${s.path}-${i}`} className="truncate" title={s.path}>
                      {s.path} · {s.channel} · {fmtSize(s.w, s.h)}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </Card>
    </Section>
  );
}
