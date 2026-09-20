import { useMemo, useState } from "react";
import UplotReact from "uplot-react";
import type uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useUiStore } from "@/store";
import { Badge, EmptyNote } from "@/components/ui";
import {
  asdUnit,
  channelLabel,
  ctxStabilityNote,
  fmtSpan,
  fmtTs,
  spectrumSeries,
  toAsd,
  tunnellingLabel,
} from "@/lib/envHistory";
import { fmtBias, fmtCurrent } from "@/lib/units";
import { useElementWidth, useViewportHeight } from "@/hooks/useElementWidth";
import { TREND_RATIO, sizeChart } from "@/lib/chartSize";

// 噪声谱快照 — one archived spectrum, optionally against an older one.
//
// Log-log, always: a noise floor spans decades on both axes and is meaningless
// on linear scales. Two curves at most, because the question this view answers
// is "is it worse than it was?" — a pile of curves answers nothing.
//
// The default readout is ASD (A/√Hz), not the stored PSD (A²/Hz). PSD is what
// integrates to a variance and therefore what we store; A/√Hz is what every
// preamp datasheet and every operator's intuition is written in.

// Token names carry the `--mast-` prefix (index.css). They used to read
// `--fg-muted` / `--border` / `--accent`, none of which are defined anywhere in
// the app, so every lookup returned "" and the chart silently painted its
// fallback literals on both themes — same note as EnvSeriesChart.tsx .
function token(name: string): string {
  if (typeof window === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export interface SpectrumLike {
  id: number;
  ts: number;
  channel: string;
  unit: string;
  fs_hz: number;
  span_s: number;
  n_segments: number;
  freqs_hz: number[];
  psd: number[];
  // 当时的仪器状态。一直在库里，只是从来没送到眼前来过——需要记录
  // 当时进针与否、偏压、setpoint 等状态。
  // 每一个都可以是 null = 当时读不到，**不是**零/关。
  quietness?: string;
  ctx_bias_v?: number | null;
  ctx_setpoint_a?: number | null;
  ctx_zctrl_on?: boolean | null;
  ctx_stable?: boolean | null;
}

/** 这条谱是在什么条件下测的。没有这一行，两条谱之间无法判断可不可比。 */
function StateRow({ s }: { s: SpectrumLike }) {
  const tun = tunnellingLabel(s);
  const caveat = ctxStabilityNote(s);
  return (
    <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      <Badge tone={tun.tone as never}>{tun.label}</Badge>
      <span className="text-mast-muted">
        偏压 <span className="font-mono text-mast-text">{fmtBias(s.ctx_bias_v)}</span>
      </span>
      <span className="text-mast-muted">
        设定点 <span className="font-mono text-mast-text">{fmtCurrent(s.ctx_setpoint_a)}</span>
      </span>
      {caveat && <span className="text-mast-warn">{caveat}</span>}
    </div>
  );
}

export function SpectrumViewer({
  primary,
  compare,
}: {
  primary: SpectrumLike | null;
  compare?: SpectrumLike | null;
}) {
  const theme = useUiStore((s) => s.theme);
  const [boxRef, measuredW] = useElementWidth<HTMLDivElement>();
  const viewportH = useViewportHeight();
  const { width, height } = sizeChart(measuredW, 860, {
    ratio: TREND_RATIO,
    minHeight: 260,
    maxHeight: 540,
    viewportH,
  });
  const [asd, setAsd] = useState(true);
  const [hover, setHover] = useState<{ f: number; a: number | null; b: number | null } | null>(null);

  const prepared = useMemo(() => {
    if (!primary) return null;
    const [f, p] = spectrumSeries(primary.freqs_hz, primary.psd);
    const y = asd ? toAsd(p) : p;
    // The comparison curve is resampled onto the primary's grid by nearest
    // frequency. Both come from the same log-binning with the same bin count,
    // so in practice the grids already coincide; this only covers a snapshot
    // taken after someone changed eh_spectrum_bins.
    let y2: (number | null)[] | null = null;
    if (compare) {
      const [cf, cp] = spectrumSeries(compare.freqs_hz, compare.psd);
      const cy = asd ? toAsd(cp) : cp;
      y2 = f.map((x) => {
        const j = nearestIndex(cf, x);
        if (j < 0) return null;
        const v = cy[j];
        return typeof v === "number" && Number.isFinite(v) ? v : null;
      });
    }
    return { f, y, y2 };
  }, [primary, compare, asd]);

  const data = useMemo<uPlot.AlignedData>(() => {
    if (!prepared) return [[], []] as unknown as uPlot.AlignedData;
    const cols: unknown[] = [prepared.f, prepared.y];
    if (prepared.y2) cols.push(prepared.y2);
    return cols as uPlot.AlignedData;
  }, [prepared]);

  const options = useMemo<uPlot.Options>(() => {
    const muted = token("--mast-muted") || "#94a3b8";
    const grid = token("--mast-border") || "#2c3440";
    const accent = token("--mast-accent") || "#28d0e6";
    const dim = token("--mast-faint") || "#6b7588";
    const unit = primary ? (asd ? asdUnit(primary.unit) : primary.unit) : "";
    const series: uPlot.Series[] = [
      {},
      { stroke: accent, width: 1.5, points: { show: false }, label: "当前" },
    ];
    if (prepared?.y2) {
      series.push({ stroke: dim, width: 1, dash: [4, 3], points: { show: false }, label: "对比" });
    }
    return {
      width,
      height,
      // distr: 3 is uPlot's log distribution. Both axes — a noise spectrum is
      // only legible log-log.
      scales: { x: { time: false, distr: 3 }, y: { distr: 3 } },
      legend: { show: false },
      cursor: { drag: { x: true, y: false }, points: { show: false } },
      axes: [
        { stroke: muted, label: "频率 (Hz)", grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } },
        { stroke: muted, label: unit, labelSize: 52, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } },
      ],
      series,
      hooks: {
        setCursor: [
          (u: uPlot) => {
            const i = u.cursor.idx;
            if (i == null) {
              setHover(null);
              return;
            }
            setHover({
              f: u.data[0][i] as number,
              a: (u.data[1]?.[i] ?? null) as number | null,
              b: (u.data[2]?.[i] ?? null) as number | null,
            });
          },
        ],
      },
    } as uPlot.Options;
  }, [width, height, primary, prepared, asd, theme]);

  if (!primary) {
    return (
      <div ref={boxRef}>
        <EmptyNote label="还没有噪声谱快照。仪器安静地隧穿一段时间之后才会出现第一条。" />
      </div>
    );
  }

  const unit = asd ? asdUnit(primary.unit) : primary.unit;

  return (
    <div ref={boxRef}>
      <StateRow s={primary} />
      <div className="mb-2 flex flex-wrap items-center gap-3 text-xs">
        <span className="font-medium">{channelLabel(primary.channel)}</span>
        <span className="text-mast-muted">
          {fmtTs(primary.ts)} · {primary.n_segments} 段 · 窗口 {fmtSpan(primary.span_s)} ·
          采样率 {Math.round(primary.fs_hz)} Hz
        </span>
        <label className="ml-auto flex items-center gap-1">
          <input type="checkbox" checked={asd} onChange={(e) => setAsd(e.target.checked)} />
          <span title="幅度谱密度 = √功率谱密度。前置放大器手册用的是这个单位。">
            按 {asdUnit(primary.unit)} 显示
          </span>
        </label>
      </div>
      <UplotReact options={options} data={data} />
      <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-mast-muted">
        <span>{prepared?.f.length ?? 0} 个频点</span>
        {compare && <span>虚线为 {fmtTs(compare.ts)} 的快照</span>}
        {hover && (
          <span className="ml-auto tabular-nums">
            {fmtHz(hover.f)} · {fmtSci(hover.a)} {unit}
            {hover.b != null ? ` · 对比 ${fmtSci(hover.b)} ${unit}` : ""}
          </span>
        )}
      </div>
    </div>
  );
}

function nearestIndex(sorted: readonly number[], x: number): number {
  if (!sorted.length) return -1;
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((sorted[mid] ?? Infinity) < x) lo = mid + 1;
    else hi = mid;
  }
  const here = sorted[lo];
  const prev = sorted[lo - 1];
  if (lo > 0 && prev != null && here != null && Math.abs(prev - x) < Math.abs(here - x)) {
    return lo - 1;
  }
  return lo;
}

function fmtHz(f: number): string {
  if (!Number.isFinite(f)) return "—";
  if (f >= 1000) return `${(f / 1000).toFixed(2)} kHz`;
  if (f >= 10) return `${f.toFixed(1)} Hz`;
  return `${f.toFixed(2)} Hz`;
}

function fmtSci(v: number | null | undefined): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return v.toExponential(2);
}
