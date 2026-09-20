import { useMemo } from "react";
import UplotReact from "uplot-react";
import type uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { components } from "@/api/schema";
import { useUiStore } from "@/store";
import { Badge, EmptyNote } from "@/components/ui";
import { fmtCurrent, fmtLength } from "@/lib/units";
import {
  AUX_CHANNELS,
  AUX_WINDOWS,
  type AuxPlotScale,
  auxAmpSamplingNote,
  auxLockinSplit,
  auxSeries,
  auxVerdictView,
} from "@/lib/monitoring";
import { useElementWidth, useViewportHeight } from "@/hooks/useElementWidth";
import { SMALL_MULTIPLE_RATIO, sizeChart } from "@/lib/chartSize";

type AuxSnapshot = components["schemas"]["AuxSnapshot"];
type AuxChannelState = components["schemas"]["AuxChannelState"];
type AuxSeriesResponse = components["schemas"]["AuxSeriesResponse"];

// 辅助通道 — Z 位置 / qPlus 振幅 / 频率偏移.
//
// ── why these are not on the current chart ─────────────────────────────────
// Metres, metres and hertz do not share an axis with amperes, and they do not
// share one with each other either. Overlaying them needs two or three Y axes,
// and where two curves sit relative to one another on separate axes is a choice
// the person drawing made — it invites the reader to see a correlation nobody
// measured. Small multiples on ONE time axis show the same data and make no
// such claim. That is also the answer to "同时看多路但不要变成一团".
//
// ── why the numbers arrive at 1 Hz and not 2 kHz ───────────────────────────
// The daemon samples these over `Signals_ValsGet` once per segment and never
// touches the oscilloscope, so the current channel keeps its whole acquisition
// cadence. That is a physics decision as much as a plumbing one: a high-Q qPlus
// amplitude cannot move faster than Q/(π f₀) — hundreds of milliseconds — and Z
// drift is a minutes-scale phenomenon. See docs/v2/design/monitoring_aux_channels.md.
//
// ── what must never happen ─────────────────────────────────────────────────
// A channel this rig does not have must read "本机没有", not "正常". A channel
// recorded but not judged must read "未判级", not "正常". Green here would be a
// claim that a judgement happened.

/** Width used only for the first frame, before the ResizeObserver has measured
 *  the container. uPlot given width 0 never recovers, so it needs a real number. */
const FALLBACK_W = 860;

function token(name: string): string {
  if (typeof window === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Value in the channel's own unit, at a scale a human reads. */
function fmtAux(v: number | null | undefined, unit: string): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (unit === "m") return fmtLength(v);
  if (unit === "A") return fmtCurrent(v);
  if (unit === "Hz") return `${v.toPrecision(4)} Hz`;
  return v.toPrecision(4);
}

/** Drift in the unit an operator thinks in: nm per minute, not m per second. */
function fmtDrift(v: number | null | undefined): string | null {
  if (v == null || !Number.isFinite(v)) return null;
  const nmPerMin = v * 1e9 * 60;
  return `${nmPerMin >= 0 ? "+" : ""}${nmPerMin.toFixed(2)} nm/min`;
}

/**
 * 调制开没开，紧挨着那个读数。
 *
 * 三态各说各的，**不塌成两态**：`undefined` 是「这一拍没读到 ``LockIn_ModOnOffGet``」，
 * 把它显示成「关闭」会让人把一条真 dI/dV 当噪声底扔掉，而这一路本来就是做谱学的。
 * 后端同样以 NULL 而非 0 记录它（``store`` schema v6）。
 */
function LockinState({ mod }: { mod: number | undefined }) {
  if (mod === 1) {
    return <div className="mt-0.5 text-xs text-mast-muted">调制开 · 这是 dI/dV</div>;
  }
  if (mod === 0) {
    return (
      <div className="mt-0.5 text-xs text-mast-warn">调制关 · 此读数不是 dI/dV</div>
    );
  }
  return (
    <div className="mt-0.5 text-xs text-mast-muted">调制状态未读到（不等于关闭）</div>
  );
}

function ChannelTile({ ch, hint }: { ch: AuxChannelState | undefined; hint: string }) {
  const view = auxVerdictView(ch?.verdict);
  const drift = fmtDrift(ch?.metrics?.z_drift_m_per_s);
  const frac = ch?.metrics?.amp_frac_of_baseline;
  return (
    <div className="rounded-mast-card border border-mast-border bg-mast-panel p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-mast-muted">{ch?.label_zh || "—"}</span>
        <Badge tone={view.tone}>{view.label}</Badge>
      </div>
      <div className="mt-1 font-mono text-lg text-mast-text">
        {fmtAux(ch?.value, ch?.unit ?? "")}
      </div>
      {drift && (
        <div className="mt-0.5 font-mono text-xs text-mast-muted">漂移 {drift}</div>
      )}
      {/* 「基线」是**未接触本底**，不是「健康的被驱动振幅」——这台机器的 qPlus
          传感器根本没被驱动，那 8 pm 是未驱动解调器的噪声底。它唯一的用处是
          回答「不为零长什么样」，所以标签必须这么写。 */}
      {typeof frac === "number" && (
        <div className="mt-0.5 font-mono text-xs text-mast-muted">
          未接触本底的 {(frac * 100).toFixed(1)}%
        </div>
      )}
      {/* dI/dV 的读数在调制关着的时候仍然是一个漂亮的 pA 数字 —— 量纲、量级都
          对，只是它不是 dI/dV。所以状态要紧挨着那个数字，不能只藏在下面的说明里。 */}
      {ch?.kind === "lockin" && <LockinState mod={ch?.metrics?.lockin_mod_on} />}
      <div className="mt-1 text-[11px] leading-snug text-mast-faint">
        {ch?.note || hint}
      </div>
    </div>
  );
}

/** 一条画出来的曲线：值 + 线型。第二条用来把「不是测量值」的那些段区分开。 */
interface Trace {
  label: string;
  /** `null` = 这一点不属于这条曲线（或那一刻根本没有读数）。 */
  v: (number | null)[];
  dashed?: boolean;
  muted?: boolean;
}

function MiniChart({
  label,
  plot,
  t,
  traces,
  n,
  gaps,
  footer,
}: {
  label: string;
  /** 这条通道声明的固定纵轴前缀。来自 `AUX_CHANNELS`，不在这里按单位反查 ——
   *  按单位反查会让 Z 与 qPlus 振幅(都是米，量级差四个数量级)被迫共用一个前缀。 */
  plot: AuxPlotScale;
  t: number[];
  /** 一条或多条曲线，共用 `t`。`null` **绝不能变成 0** 画出来。 */
  traces: Trace[];
  /** 真实读数的条数（不含为了断线补进去的 null）。 */
  n: number;
  /** 窗口内的采集中断处数。 */
  gaps: number;
  /** 图下额外一行（如 lock-in 的调制状态构成）。 */
  footer?: React.ReactNode;
}) {
  const theme = useUiStore((s) => s.theme);
  const [boxRef, measuredW] = useElementWidth<HTMLDivElement>();
  const viewportH = useViewportHeight();
  const { width, height } = sizeChart(measuredW, FALLBACK_W, {
    ratio: SMALL_MULTIPLE_RATIO,
    minHeight: 96,
    maxHeight: 200,
    viewportH,
  });

  const scale = plot.mul;
  const axisLabel = `${label} (${plot.suffix})`;

  const data = useMemo<uPlot.AlignedData>(
    () =>
      [
        t,
        ...traces.map((tr) => tr.v.map((y) => (y == null ? null : y * scale))),
      ] as uPlot.AlignedData,
    [t, traces, scale],
  );

  const shape = traces.map((tr) => `${tr.label}|${tr.dashed}|${tr.muted}`).join(",");
  const options = useMemo<uPlot.Options>(() => {
    const axisStroke = token("--mast-muted") || "#94a3b8";
    const gridStroke = token("--mast-border") || "#1e293b";
    const stroke = token("--mast-accent") || "#28d0e6";
    return {
      width,
      height,
      scales: { x: { time: true }, y: {} },
      axes: [
        { stroke: axisStroke, grid: { stroke: gridStroke } },
        {
          stroke: axisStroke,
          grid: { stroke: gridStroke },
          label: axisLabel,
          labelGap: 4,
          size: 62,
        },
      ],
      series: [
        { label: "时间" },
        ...traces.map((tr) => ({
          label: tr.label,
          // 「不是测量值」的那些段既换颜色又换线型。只换颜色的话，深浅两个主题下
          // 都得赌那两个 token 的对比度够；虚线在两个主题、以及打印出来时都成立。
          stroke: tr.muted ? axisStroke : stroke,
          width: tr.muted ? 1 : 1.2,
          ...(tr.dashed ? { dash: [4, 3] } : {}),
        })),
      ],
      legend: { show: false },
      cursor: { drag: { x: true, y: false } },
    };
  }, [theme, axisLabel, shape, width, height]);

  // 判据是**真实读数条数**，不是 t.length：为了断线补进去的 null 也占一个时间戳，
  // 拿数组长度问「有没有数据」会让一条全是空洞的曲线显示成「有数据」。
  if (!n) {
    return (
      <div className="rounded-lg border border-mast-border bg-mast-bg p-3 text-xs text-mast-muted">
        {label}：窗口内还没有采到数据。
      </div>
    );
  }
  return (
    <div
      ref={boxRef}
      className="overflow-x-auto rounded-lg border border-mast-border bg-mast-bg p-2"
    >
      <UplotReact options={options} data={data} />
      {footer}
      {gaps > 0 && (
        // 说出来，否则一条断掉的曲线看起来像画崩了。这一行同时是「断点是被判出来的，
        // 不是渲染故障」这句话本身。
        <div className="px-1 pt-1 text-[11px] text-mast-faint">
          曲线断开 {gaps} 处 = 那几段时间没有采到数据（守护停过 / 服务重启 / 窗口早于
          开始记录的时刻），不是读数为零。
        </div>
      )}
    </div>
  );
}

/**
 * dI/dV 图下的一行：虚线段是什么，以及窗口里三种状态各占多少点。
 *
 * 图上只有两种线型（**确认在调制** vs 其余），因为在图上「关闭」和「没读到」要
 * 说的是同一句话：别把它当 dI/dV 读。但两者的**下一步动作不同** —— 关闭是有人
 * 拧过的旋钮，没读到是一条要去查的链路 —— 所以数字上分开报。
 *
 * 「没读到」绝不写成「关闭」：那句话会让人把一条真 dI/dV 当噪声底扔掉。
 */
function LockinLegend({ nOn, nOff, nUnknown }: {
  nOn: number; nOff: number; nUnknown: number;
}) {
  if (!nOn && !nOff && !nUnknown) return null;
  return (
    <div className="px-1 pt-1 text-[11px] leading-snug text-mast-faint">
      <span className="text-mast-muted">灰色虚线段＝未确认在调制，不是 dI/dV</span>
      （解调器的噪声与串扰底）。本窗口：调制开 {nOn} 点
      {nOff > 0 && <> · 调制关 {nOff} 点</>}
      {nUnknown > 0 && <> · 调制状态未读到 {nUnknown} 点（不等于关闭）</>}。
    </div>
  );
}

/** 短期↔长期 switcher for the charts below. Deliberately NOT labelled 统计窗口:
 *  it changes how far back the small multiples are drawn and nothing else. The
 *  verdict tiles keep using the daemon's own window (`aux.window_s`), which is a
 *  threshold-calibration parameter and lives in 设置, not here. */
function WindowPicker({
  windowS,
  onWindowChange,
}: {
  windowS: number;
  onWindowChange: (s: number) => void;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
      {AUX_WINDOWS.map((w) => (
        <button
          key={w.s}
          type="button"
          aria-pressed={windowS === w.s}
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
  );
}

export function AuxChannels({
  aux,
  series,
  degraded,
  windowS,
  onWindowChange,
}: {
  aux: AuxSnapshot | null | undefined;
  series: AuxSeriesResponse | undefined;
  degraded: boolean;
  windowS: number;
  onWindowChange: (s: number) => void;
}) {
  if (degraded) return <EmptyNote label="监控模块未装载，无辅助通道。" />;
  if (!aux) {
    // Not the same as "this rig has no Z channel" — say which one it is.
    return <EmptyNote label="采集守护未运行，辅助通道没有在采样。历史数据照常可浏览。" />;
  }

  const byKind = new Map<string, AuxChannelState>();
  for (const c of aux.channels ?? []) byKind.set(c.kind, c);
  const available = AUX_CHANNELS.filter((c) => byKind.get(c.kind)?.available);
  // 采样是机会式的；设置只是节流上限，按观测节奏评估采样速度。
  const observed = aux.observed_interval_s ?? null;
  const sampling = auxAmpSamplingNote(aux.amp_tau_s, observed ?? aux.interval_s);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {AUX_CHANNELS.map((c) => (
          <ChannelTile key={c.kind} ch={byKind.get(c.kind)} hint={c.hint} />
        ))}
      </div>

      <p className="text-xs leading-relaxed text-mast-muted">
        {/* 同时报设置间隔与观测间隔；样本不足时明确说明。 */}
        设定每 {aux.interval_s.toFixed(2)} s 采一次
        {observed != null
          ? `，实测 ${observed.toFixed(2)} s`
          : "（实测节奏样本还不够，暂时给不出）"}
        ，<strong>不占用示波器</strong>——它搭在电流泵等缓冲的空隙里，
        与采集主环、急停看门狗都不共用锁。判级窗口 {Math.round(aux.window_s)} s，
        已采 {aux.sampled} 次
        {aux.skipped_busy > 0 && `，因通道被占跳过 ${aux.skipped_busy} 次`}。
        {sampling && <> {sampling}。</>}
      </p>

      {!aux.alerts_enabled && (
        <p className="rounded-mast-card border border-mast-warn-border bg-mast-warn-bg p-2 text-xs leading-relaxed text-mast-warn">
          辅助通道的<strong>告警当前是关的</strong>——出厂状态，不是故障。
          Z 与振幅的阈值从未在本机标定过，先跑
          <code className="mx-1">python -m mast.monitoring.commission</code>
          看【辅助通道】那一段，再到「设置 → 电流监控」把
          <code className="mx-1">cm_aux_alerts_enabled</code>打开。记录一直在进行。
        </p>
      )}

      {available.length === 0 ? (
        // "we have not looked yet" and "this rig does not have one" get
        // different sentences. The second makes someone stop investigating, so
        // it must not be shown before the signal table has actually been read.
        <EmptyNote
          label={
            (aux.channels ?? []).some((c) => c.verdict === "unknown")
              ? `还没读到信号表（${aux.detail || "探测中"}）——尚未判定这台机器有没有这些通道。`
              : "这台机器的信号表里没有 Z / qPlus 振幅通道——不是故障，是本机没有这些信号。"
          }
        />
      ) : (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <WindowPicker windowS={windowS} onWindowChange={onWindowChange} />
            <span className="text-xs text-mast-faint">
              图的时间跨度。判级仍按守护自己的 {Math.round(aux.window_s)} s 窗口。
            </span>
          </div>
          {available.map((c) => {
            const label = byKind.get(c.kind)?.label_zh || c.kind;
            // lock-in 是唯一一路「同一条曲线有时是测量值、有时不是」的通道，
            // 所以只有它拆两条线。别的通道拆了只会多一条永远为空的序列。
            if (c.kind === "lockin") {
              const s = auxLockinSplit(series);
              return (
                <MiniChart
                  key={c.kind}
                  label={label}
                  plot={c.plot}
                  t={s.t}
                  n={s.n}
                  gaps={s.gaps}
                  traces={[
                    { label: "调制开（dI/dV）", v: s.on },
                    { label: "未确认在调制", v: s.off, dashed: true, muted: true },
                  ]}
                  footer={<LockinLegend {...s} />}
                />
              );
            }
            const { t, v, n, gaps } = auxSeries(series, c.series);
            return (
              <MiniChart
                key={c.kind}
                label={label}
                plot={c.plot}
                t={t}
                n={n}
                gaps={gaps}
                traces={[{ label, v }]}
              />
            );
          })}
          <div className="text-xs text-mast-muted">
            共用同一条时间轴。<strong>刻意不叠在一起</strong>：米、米、赫兹不通约，
            叠加要用双 Y 轴，而两条曲线在各自轴上的相对高低是画图人定的，
            会诱导出没人测过的相关性。
          </div>
        </div>
      )}
    </div>
  );
}
