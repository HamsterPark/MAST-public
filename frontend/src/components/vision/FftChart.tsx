import { useMemo, useState } from "react";
import UplotReact from "uplot-react";
import type uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { components } from "@/api/schema";
import { DataTable } from "@/components/DataTable";
import type { ColumnDef } from "@tanstack/react-table";
import { useUiStore } from "@/store";
import { useElementWidth, useViewportHeight } from "@/hooks/useElementWidth";
import { TREND_RATIO, sizeChart } from "@/lib/chartSize";

type FFT = components["schemas"]["FFTResponse"];

// One-sided spectrum line chart (freqs_hz × spectrum) via uPlot. Dark theme to
// match the Lab-Console look; replaces the matplotlib FFT plot in exp_capture.
//
// ITEM 6 — fuller 信号捕获 + FFT. Beyond the original single-trace render this
// now supports:
//   · logY      — log-scaled Y axis (distr:3) vs linear (parity with semilogy)
//   · logX      — log-scaled X axis (decade view of the spectrum)
//   · dropDc    — drop the f=0 bin (mandatory for a log axis; optional on linear)
//   · fMin/fMax — frequency-axis zoom window (Hz); null = auto
//   · peaks     — top-N spectral peaks as a sortable DataTable readout (freq +
//                 magnitude + Δ from fundamental), not just inline pills
//   · OVERLAY   — pass `series` (2+ named spectra) to compare channels on one plot
//   · cursor    — hover readout (freq + value at the nearest bin) below the plot
//   · units     — V/√Hz (PSD) vs V (|FFT|) scale note, unit-aware
// Never freezes: empty / degraded spectra render a note, not a blank canvas.

// uPlot paints to <canvas> and cannot resolve CSS vars, so the axis colours are
// read off <html> and re-read on theme change. They used to be dark-theme hex
// literals inlined at the call site, which left the light theme with a near
// invisible grid (same family of defect as ).
function token(name: string): string {
  if (typeof window === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Distinct strokes for overlaid channel spectra (channel-compare mode).
const OVERLAY_STROKES = ["#38bdf8", "#f472b6", "#facc15", "#34d399", "#a78bfa", "#fb923c"];

export interface FftPeak {
  freq: number;
  value: number;
}

/** One named spectrum for the overlay / channel-compare mode. */
export interface FftSeries {
  fft: FFT;
  label: string;
}

/** Local-maximum peak picker over (freq, value). Returns the `topN` strongest
 *  interior local maxima, sorted by descending value — cheap, no deps. */
export function findPeaks(freqs: number[], spec: number[], topN = 5): FftPeak[] {
  const peaks: FftPeak[] = [];
  for (let i = 1; i < spec.length - 1; i++) {
    const v = spec[i]!;
    if (v > spec[i - 1]! && v >= spec[i + 1]!) {
      peaks.push({ freq: freqs[i]!, value: v });
    }
  }
  peaks.sort((a, b) => b.value - a.value);
  return peaks.slice(0, topN);
}

export function fmtHz(hz: number): string {
  if (!Number.isFinite(hz)) return "—";
  if (hz >= 1e6) return `${(hz / 1e6).toFixed(3)} MHz`;
  if (hz >= 1e3) return `${(hz / 1e3).toFixed(3)} kHz`;
  return `${hz.toFixed(2)} Hz`;
}

export function fmtVal(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(3);
  return v.toPrecision(5);
}

/** The amplitude-axis label + a units note that names the physical scale
 *  (PSD ≈ unit/√Hz, |FFT| ≈ unit). */
export function spectrumUnits(fft: Pick<FFT, "output" | "unit">): { yLabel: string; scaleNote: string } {
  const unit = fft.unit || "V";
  if (fft.output === "power") {
    return {
      yLabel: `PSD (${unit}²/Hz)`,
      scaleNote: `功率谱密度 PSD，单位 ${unit}²/Hz（幅度谱密度 ≈ ${unit}/√Hz）`,
    };
  }
  return { yLabel: `幅值 (${unit})`, scaleNote: `单边幅度谱 |FFT|，单位 ${unit}` };
}

interface PeakRow {
  rank: number;
  series: string;
  freq: number;
  value: number;
  ratio: number;
}

export function FftChart({
  fft,
  series,
  logY = true,
  logX = false,
  dropDc = true,
  showPeaks = true,
  topN = 5,
  fMin = null,
  fMax = null,
}: {
  /** Single spectrum (back-compat). Ignored when `series` is provided. */
  fft?: FFT;
  /** 2+ named spectra to overlay (channel compare). Takes priority over `fft`. */
  series?: FftSeries[];
  logY?: boolean;
  logX?: boolean;
  dropDc?: boolean;
  showPeaks?: boolean;
  topN?: number;
  fMin?: number | null;
  fMax?: number | null;
}) {
  // Cursor readout state (nearest-bin freq + per-series value under the pointer).
  const [hover, setHover] = useState<{ f: number; vs: (number | null)[] } | null>(null);
  const theme = useUiStore((s) => s.theme);
  const [boxRef, measuredW] = useElementWidth<HTMLDivElement>();
  const viewportH = useViewportHeight();
  const { width, height } = sizeChart(measuredW, 760, {
    ratio: TREND_RATIO,
    minHeight: 260,
    maxHeight: 520,
    viewportH,
  });

  // Normalise to a list of named spectra (single `fft` → one unnamed series).
  const specs: FftSeries[] = useMemo(() => {
    if (series && series.length) return series;
    if (fft) return [{ fft, label: fft.channel_name || "signal" }];
    return [];
  }, [series, fft]);

  const primary = specs[0]?.fft;

  // Build aligned data: a shared frequency axis (union of bins, but in practice
  // all share Δf when same fs/N) + one Y column per series. We use the FIRST
  // series' freq grid as the X axis and index-align the rest (they share grids
  // in channel-compare since N/fs are common); mismatched lengths are clamped.
  const { data, perSeriesArrays, xFreqs } = useMemo(() => {
    if (!specs.length) {
      return { data: [[], []] as uPlot.AlignedData, perSeriesArrays: [] as { f: number[]; s: number[] }[], xFreqs: [] as number[] };
    }
    const baseFreqs = specs[0]!.fft.freqs_hz ?? [];
    const start = dropDc || logY || logX ? 1 : 0;
    // Apply the frequency-axis zoom window on top of the DC drop.
    const lo = fMin != null && Number.isFinite(fMin) ? fMin : -Infinity;
    const hi = fMax != null && Number.isFinite(fMax) ? fMax : Infinity;
    const keepIdx: number[] = [];
    for (let i = start; i < baseFreqs.length; i++) {
      const f = baseFreqs[i]!;
      if (f >= lo && f <= hi) keepIdx.push(i);
    }
    const fAxis = keepIdx.map((i) => baseFreqs[i]!);
    const cols: number[][] = [];
    const arrs: { f: number[]; s: number[] }[] = [];
    for (const sp of specs) {
      const spec = sp.fft.spectrum ?? [];
      const col = keepIdx.map((i) => {
        const v = spec[i];
        if (v == null) return null as unknown as number;
        return logY ? Math.max(v, 1e-30) : v;
      });
      cols.push(col);
      arrs.push({ f: fAxis, s: col });
    }
    return {
      data: [fAxis, ...cols] as unknown as uPlot.AlignedData,
      perSeriesArrays: arrs,
      xFreqs: fAxis,
    };
  }, [specs, logY, logX, dropDc, fMin, fMax]);

  // Peak table: top-N per series, with Δ ratio against the strongest peak.
  const peakRows: PeakRow[] = useMemo(() => {
    if (!showPeaks) return [];
    const rows: PeakRow[] = [];
    perSeriesArrays.forEach((arr, si) => {
      const pk = findPeaks(arr.f, arr.s, topN);
      const fund = pk[0]?.value ?? 0;
      pk.forEach((p, i) => {
        rows.push({
          rank: i + 1,
          series: specs[si]?.label ?? `#${si + 1}`,
          freq: p.freq,
          value: p.value,
          ratio: fund > 0 ? p.value / fund : 0,
        });
      });
    });
    return rows;
  }, [perSeriesArrays, specs, showPeaks, topN]);

  const units = primary ? spectrumUnits(primary) : { yLabel: "幅值", scaleNote: "" };

  const options = useMemo<uPlot.Options>(
    () => {
      // getComputedStyle is a layout read — keep it inside the memo so it runs
      // on theme change, not on every hover-driven re-render.
      const axisStroke = token("--mast-muted") || "#94a3b8";
      const gridStroke = token("--mast-border") || "#1e293b";
      const tickStroke = token("--mast-border-strong") || "#334155";
      return {
      width,
      height,
      scales: {
        // logX → log-scaled X axis (distr:3) for a decade view of the spectrum.
        x: { time: false, distr: logX ? 3 : 1 },
        // logY → log-scaled Y axis (distr:3, parity with matplotlib semilogy).
        y: { distr: logY ? 3 : 1 },
      },
      axes: [
        {
          stroke: axisStroke,
          grid: { stroke: gridStroke },
          ticks: { stroke: tickStroke },
          label: logX ? "频率 (Hz, 对数)" : "频率 (Hz)",
          labelGap: 6,
        },
        {
          stroke: axisStroke,
          grid: { stroke: gridStroke },
          ticks: { stroke: tickStroke },
          label: units.yLabel,
          labelGap: 6,
        },
      ],
      series: [
        { label: "Hz" },
        // Stroke only, in every mode. A single spectrum used to get a `fill`,
        // which floods from the curve down to the baseline — on a log Y axis
        // that is most of the panel, and it made the one-channel view look
        // categorically different from the compare view of the same data
        // (「不宜图线下方填色」).
        ...specs.map((sp, i) => ({
          label: sp.label || "signal",
          stroke: OVERLAY_STROKES[i % OVERLAY_STROKES.length],
          width: 1.5,
        })),
      ],
      legend: { show: true },
      cursor: { drag: { x: true, y: false } },
      hooks: {
        setCursor: [
          (u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null) {
              setHover(null);
              return;
            }
            const f = u.data[0]?.[idx];
            if (typeof f !== "number") {
              setHover(null);
              return;
            }
            const vs: (number | null)[] = [];
            for (let s = 1; s <= specs.length; s++) {
              const v = u.data[s]?.[idx];
              vs.push(typeof v === "number" ? v : null);
            }
            setHover({ f, vs });
          },
        ],
      },
      };
    },
    [width, height, logX, logY, units.yLabel, specs, theme],
  );

  if (!xFreqs.length) {
    return <p className="text-sm text-mast-muted">频谱为空。</p>;
  }

  const peakColumns: ColumnDef<PeakRow, unknown>[] = [
    { accessorKey: "rank", header: "#", cell: (c) => <span className="font-mono">{c.getValue<number>()}</span> },
    ...(specs.length > 1
      ? [{ accessorKey: "series", header: "通道", cell: (c) => <span className="truncate">{c.getValue<string>()}</span> } as ColumnDef<PeakRow, unknown>]
      : []),
    { accessorKey: "freq", header: "频率", cell: (c) => <span className="font-mono text-mast-accent">{fmtHz(c.getValue<number>())}</span> },
    { accessorKey: "value", header: units.yLabel, cell: (c) => <span className="font-mono">{fmtVal(c.getValue<number>())}</span> },
    { accessorKey: "ratio", header: "相对峰", cell: (c) => <span className="font-mono text-mast-muted">{(c.getValue<number>() * 100).toFixed(1)}%</span> },
  ];

  return (
    <div className="space-y-2">
      <div ref={boxRef} className="overflow-x-auto rounded-lg border border-mast-border bg-mast-bg p-2">
        <UplotReact options={options} data={data} />
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-mast-muted">
        <span>
          游标：
          {hover ? (
            <span className="ml-1 font-mono text-mast-text">
              {fmtHz(hover.f)}
              {hover.vs.map((v, i) => (
                <span key={i} className="ml-2">
                  <span
                    className="mr-0.5 inline-block h-2 w-2 rounded-full align-middle"
                    style={{ background: OVERLAY_STROKES[i % OVERLAY_STROKES.length] }}
                  />
                  {v == null ? "—" : fmtVal(v)}
                </span>
              ))}
            </span>
          ) : (
            <span className="ml-1">移动鼠标到曲线上读取</span>
          )}
        </span>
        <span>X 轴：{logX ? "对数" : "线性"}</span>
        <span>Y 轴：{logY ? "对数 (semilogy)" : "线性"}</span>
        {(dropDc || logY || logX) && <span>已去除 DC 分量 (f=0)</span>}
        {(fMin != null || fMax != null) && (
          <span className="text-mast-warn">
            频率窗 [{fMin != null ? fmtHz(fMin) : "auto"} – {fMax != null ? fmtHz(fMax) : "auto"}]
          </span>
        )}
      </div>
      {units.scaleNote && <p className="text-xs text-mast-muted/80">标度：{units.scaleNote}</p>}

      {showPeaks && peakRows.length > 0 && (
        <div className="rounded-md border border-mast-border bg-mast-bg/40 p-2">
          <span className="mb-1.5 block text-xs font-semibold text-mast-text">
            峰值检测（每通道前 {topN}）
          </span>
          <DataTable<PeakRow> data={peakRows} columns={peakColumns} empty="未检出峰值。" />
        </div>
      )}
    </div>
  );
}
