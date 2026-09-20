import { useMemo, useState } from "react";
import UplotReact from "uplot-react";
import type uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useUiStore } from "@/store";
import { fmtCurrent } from "@/lib/units";
import { TRACE_WINDOWS } from "@/lib/monitoring";
import { EmptyNote } from "@/components/ui";
import { useElementWidth, useViewportHeight } from "@/hooks/useElementWidth";
import { TREND_RATIO, sizeChart } from "@/lib/chartSize";

// 实时电流包络 — the min/max band `/api/monitoring/live-trace` returns.
//
// A band, not a line, because the daemon stores an envelope per segment: at
// 1 kHz a 5-minute window is 300k samples, and any single-value reduction of
// that (mean, or one decimated sample) throws away exactly the spikes the
// monitor exists to catch. Drawing min and max and filling between them keeps
// every excursion visible at ~1 point per segment.
//
// `null` in either band array is an ACQUISITION GAP, and it arrives already
// marked from the store — this component must not try to work gaps out from the
// timestamps. Inside a segment consecutive points are one envelope step apart
// (10 ms); between segments they are a whole acquisition pause apart. Flattened
// into one array the two are indistinguishable here, so any spacing rule
// applied at this layer breaks the line at EVERY segment boundary. Only
// `store.trace_gap_marks` can see where the segments are.

// uPlot paints to <canvas>, which cannot resolve CSS vars — read the token's
// computed value off <html> and feed concrete colors, keyed on the active theme
// (same approach as experimental/TimeTraceChart.tsx:10-13).
function token(name: string): string {
  if (typeof window === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}


export function LiveCurrentChart({
  tS,
  iMinA,
  iMaxA,
  nSegments,
  nGaps,
  windowS,
  onWindowChange,
  paused,
  onPausedChange,
  stale,
  degraded,
}: {
  tS: number[];
  /** `null` = 那一刻没在采集。**绝不能当 0 画** —— 0 是一个合法的电流读数。 */
  iMinA: (number | null)[];
  iMaxA: (number | null)[];
  /** Segments behind the band. NOT the point count — the store expands each
   *  segment's stored envelope into several points at `envelope_dt_s`. */
  nSegments: number;
  /** Acquisition gaps in view. Drawn as holes; also said in words below,
   *  because an unlabelled break in a live chart reads as a rendering bug. */
  nGaps: number;
  windowS: number;
  onWindowChange: (s: number) => void;
  paused: boolean;
  onPausedChange: (p: boolean) => void;
  stale: boolean;
  degraded: boolean;
}) {
  const [hover, setHover] = useState<{ t: number; lo: number; hi: number } | null>(null);
  const theme = useUiStore((s) => s.theme);
  const [boxRef, measuredW] = useElementWidth<HTMLDivElement>();
  const viewportH = useViewportHeight();
  const { width, height } = sizeChart(measuredW, 860, {
    ratio: TREND_RATIO,
    minHeight: 220,
    maxHeight: 460,
    viewportH,
  });

  // Plot in pA. The whole window shares one scale, so a fixed prefix keeps the
  // Y axis readable; the cursor readout below re-renders in engineering units.
  const data = useMemo<uPlot.AlignedData>(() => {
    const n = Math.min(tS.length, iMinA.length, iMaxA.length);
    const t = tS.slice(0, n);
    const pa = (v: number | null) => (v == null ? null : v * 1e12);
    const hi = iMaxA.slice(0, n).map(pa);
    const lo = iMinA.slice(0, n).map(pa);
    return [t, hi, lo] as uPlot.AlignedData;
  }, [tS, iMinA, iMaxA]);

  /** Points that are actual readings. `data[0].length` counts the gap markers
   *  too, and reporting those as measured points is how a chart claims data it
   *  never had. */
  const nPoints = useMemo(
    () => iMinA.reduce<number>((acc, v) => acc + (v == null ? 0 : 1), 0),
    [iMinA],
  );

  const options = useMemo<uPlot.Options>(() => {
    const axisStroke = token("--mast-muted") || "#94a3b8";
    const gridStroke = token("--mast-border") || "#1e293b";
    const tickStroke = token("--mast-border-strong") || "#334155";
    const stroke = token("--mast-accent") || "#28d0e6";
    const bandFill = `color-mix(in srgb, ${stroke} 22%, transparent)`;
    return {
      width,
      height,
      // Absolute unix seconds on x — uPlot's time scale reads seconds natively,
      // so the axis shows wall-clock and a paused chart is obviously behind.
      scales: { x: { time: true }, y: {} },
      axes: [
        { stroke: axisStroke, grid: { stroke: gridStroke }, ticks: { stroke: tickStroke } },
        {
          stroke: axisStroke,
          grid: { stroke: gridStroke },
          ticks: { stroke: tickStroke },
          label: "电流 (pA)",
          labelGap: 6,
        },
      ],
      series: [
        { label: "时间" },
        { label: "最大", stroke, width: 1 },
        { label: "最小", stroke, width: 1 },
      ],
      // [high, low] — uPlot fills the area between the two series.
      bands: [{ series: [1, 2], fill: bandFill }],
      legend: { show: false },
      cursor: { drag: { x: true, y: false } },
      hooks: {
        setCursor: [
          (u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null) {
              setHover(null);
              return;
            }
            const t = u.data[0]?.[idx];
            const hi = u.data[1]?.[idx];
            const lo = u.data[2]?.[idx];
            if (typeof t === "number" && typeof hi === "number" && typeof lo === "number") {
              setHover({ t, hi, lo });
            } else setHover(null);
          },
        ],
      },
    };
  }, [width, height, theme]);

  const controls = (
    <div className="flex flex-wrap items-center gap-2">
      <div className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
        {TRACE_WINDOWS.map((w) => (
          <button
            key={w.s}
            onClick={() => onWindowChange(w.s)}
            className={
              "px-2.5 py-1 text-xs " +
              (windowS === w.s
                ? "bg-mast-accent-soft font-medium text-mast-accent"
                : "text-mast-muted hover:text-mast-text")
            }
          >
            {w.label}
          </button>
        ))}
      </div>
      <button
        onClick={() => onPausedChange(!paused)}
        className={
          "rounded-mast-ctl border px-2.5 py-1 text-xs " +
          (paused
            ? "border-mast-warn-border bg-mast-warn-bg text-mast-warn"
            : "border-mast-border text-mast-muted hover:text-mast-text")
        }
      >
        {paused ? "已暂停 · 继续" : "暂停"}
      </button>
      {paused && <span className="text-xs text-mast-warn">画面已冻结，不再跟随新数据</span>}
      {!paused && stale && (
        <span className="text-xs text-mast-warn">数据已停更，曲线停在最后一段</span>
      )}
    </div>
  );

  return (
    <div className="space-y-2">
      {controls}
      {degraded ? (
        <EmptyNote label="监控模块未装载，无实时曲线。" />
      ) : !nPoints ? (
        // 判据是真实读数条数，不是数组长度：一窗口全是断点标记时数组不空，
        // 但那正是「还没有采集到数据」。
        <EmptyNote label="窗口内还没有采集到数据——启动采集后曲线会自动出现。" />
      ) : (
        <>
          <div
            ref={boxRef}
            className="overflow-x-auto rounded-lg border border-mast-border bg-mast-bg p-2"
          >
            <UplotReact options={options} data={data} />
          </div>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-mast-muted">
            <span>
              游标：
              {hover ? (
                <span className="ml-1 font-mono text-mast-text">
                  {new Date(hover.t * 1000).toLocaleTimeString("zh-CN", { hour12: false })} →{" "}
                  {fmtCurrent(hover.lo * 1e-12)} ～ {fmtCurrent(hover.hi * 1e-12)}
                </span>
              ) : (
                <span className="ml-1">移动鼠标到曲线上读取</span>
              )}
            </span>
            <span>
              {nSegments} 段 / {nPoints} 点
            </span>
            {nGaps > 0 && (
              <span className="text-mast-warn">
                曲线断开 {nGaps} 处 = 那几段时间没有采集（守护停过 / 服务重启），
                不是电流为零
              </span>
            )}
            <span>带宽即每桶的最小/最大值，尖峰不会被抽稀抹掉</span>
          </div>
        </>
      )}
    </div>
  );
}
