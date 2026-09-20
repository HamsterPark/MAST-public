import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import UplotReact from "uplot-react";
import type uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { api } from "@/api/client";
import { useUiStore } from "@/store";
import { Badge, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { useElementWidth, useViewportHeight } from "@/hooks/useElementWidth";
import { TREND_RATIO, sizeChart } from "@/lib/chartSize";
import {
  availableModes,
  buildSpectrumData,
  numericDidvNote,
  resolveMode,
  type SpectrumMode,
} from "@/lib/spectrumSeries";

// STS 曲线 —— 一条谱的交互式显示。
//
// Until now a .dat reached this app as a ~160 px PNG of its first two columns:
// no axes, no units, no zoom, and the backward sweep and lock-in channel thrown
// away before the wire. The operator asked for 「sts曲线的初步处理和显示」; this
// is that. Drag on the plot to zoom, double-click to reset (uPlot built-ins).
//
// Series colours come from the theme tokens, not literals — `--mast-*` names,
// because the earlier charts in this repo looked up `--accent` / `--border`,
// which are defined nowhere, so every lookup returned "" and both themes got the
// hard-coded fallbacks .

function token(name: string): string {
  if (typeof window === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function useSpectrum(path: string | null) {
  return useQuery({
    enabled: !!path,
    queryKey: ["scans", "spectrum", path],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/scans/spectrum", {
        params: { query: { path: path as string } },
      });
      if (error) throw error;
      return data;
    },
    staleTime: 5 * 60_000,
  });
}

function fmtSci(v: number | null | undefined): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return v.toExponential(3);
}

export function SpectrumChart({ path }: { path: string }) {
  const theme = useUiStore((s) => s.theme);
  const [boxRef, measuredW] = useElementWidth<HTMLDivElement>();
  const viewportH = useViewportHeight();
  const { width, height } = sizeChart(measuredW, 640, {
    ratio: TREND_RATIO,
    minHeight: 220,
    maxHeight: 420,
    viewportH,
  });
  const [wanted, setWanted] = useState<SpectrumMode | null>(null);
  const [hover, setHover] = useState<{ x: number; ys: (number | null)[] } | null>(null);

  const q = useSpectrum(path);
  const resp = q.data ?? null;
  const modes = availableModes(resp);
  const mode = resolveMode(resp, wanted);
  const prepared = useMemo(() => buildSpectrumData(resp, mode), [resp, mode]);
  const numericNote = numericDidvNote(resp);

  const data = useMemo<uPlot.AlignedData>(
    () => (prepared ? (prepared.data as unknown as uPlot.AlignedData) : ([[], []] as unknown as uPlot.AlignedData)),
    [prepared],
  );

  const options = useMemo<uPlot.Options>(() => {
    const muted = token("--mast-muted") || "#94a3b8";
    const grid = token("--mast-border") || "#2c3440";
    const accent = token("--mast-accent") || "#28d0e6";
    const dim = token("--mast-faint") || "#6b7588";
    const series: uPlot.Series[] = [{}];
    (prepared?.series ?? []).forEach((s, i) => {
      series.push({
        // Backward sweeps are the dashed dim ones: they are the SAME quantity
        // measured the other way, and giving them an equal-weight colour makes
        // a normal trace/retrace pair look like two different signals.
        stroke: i === 0 ? accent : dim,
        width: i === 0 ? 1.5 : 1,
        dash: i === 0 ? undefined : [4, 3],
        points: { show: false },
        label: s.name,
      });
    });
    return {
      width,
      height,
      scales: { x: { time: false } },
      legend: { show: false },
      cursor: { drag: { x: true, y: false }, points: { show: false } },
      axes: [
        {
          stroke: muted,
          label: prepared?.xLabel || "",
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
        },
        {
          stroke: muted,
          label: prepared?.yLabel || "",
          labelSize: 60,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          // Currents are 1e-12..1e-9 and dI/dV smaller still; default tick
          // formatting renders every one of them as "0".
          values: (_u, splits) => splits.map((v) => (v === 0 ? "0" : v.toExponential(1))),
        },
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
              x: u.data[0][i] as number,
              ys: u.data.slice(1).map((col) => (col?.[i] ?? null) as number | null),
            });
          },
        ],
      },
    } as uPlot.Options;
  }, [width, height, prepared, theme]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;

  return (
    <div ref={boxRef}>
      {q.data?.degraded && <DegradedNote what="谱数据" />}
      {q.data?.detail && !q.data.degraded && (
        <div className="mb-2 text-xs text-mast-warn">{q.data.detail}</div>
      )}

      {!prepared ? (
        <EmptyNote
          label={
            q.data?.degraded
              ? `读不到这条谱${q.data?.detail ? `：${q.data.detail}` : ""}`
              : "这个文件里没有可画的数值列。"
          }
        />
      ) : (
        <>
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            {modes.length > 1 && (
              <div className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
                {modes.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setWanted(m.id)}
                    className={
                      "px-2.5 py-1 " +
                      (mode === m.id
                        ? "bg-mast-accent-soft font-medium text-mast-accent"
                        : "text-mast-muted hover:text-mast-text")
                    }
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            )}
            {prepared.hasNumeric && <Badge tone="warn">数值微分</Badge>}
            <span className="text-mast-muted">
              {q.data?.n_points ?? 0} 点
              {q.data?.decimated ? "（已抽稀）" : ""}
              {q.data?.kind_evidence ? ` · ${q.data.kind_evidence}` : ""}
            </span>
          </div>

          <UplotReact options={options} data={data} />

          <div className="mt-1 space-y-1 text-xs text-mast-muted">
            {numericNote && <div className="text-mast-warn">{numericNote}</div>}
            <div className="flex flex-wrap items-center gap-x-3">
              {prepared.series.map((s, i) => (
                <span key={s.id} className="truncate" title={s.name}>
                  <span
                    className={
                      "mr-1 inline-block h-[2px] w-4 align-middle " +
                      (i === 0 ? "bg-mast-accent" : "bg-mast-faint")
                    }
                  />
                  {s.name}
                </span>
              ))}
              {hover && (
                <span className="ml-auto tabular-nums">
                  {fmtSci(hover.x)} → {hover.ys.map((y) => fmtSci(y)).join(" / ")}
                </span>
              )}
            </div>
            {/* Every column in the file, including the ones no curve shows —
                so what is on screen can be checked against what was measured. */}
            {(q.data?.columns?.length ?? 0) > prepared.series.length + 1 && (
              <div className="text-mast-faint">
                文件通道：{(q.data?.columns ?? []).join(" · ")}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
