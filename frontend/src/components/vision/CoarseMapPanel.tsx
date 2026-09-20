import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Card, Spinner, ErrorNote, DegradedNote } from "@/components/ui";
import {
  COARSE_ZOOM_MAX,
  COARSE_ZOOM_RESET,
  type CoarseZoom,
  chromeAt,
  coarseViewBox,
  fitCoarse,
  fmtZoom,
  isZoomed,
  panByFraction,
  toStage,
  zoomAtPoint,
} from "@/lib/coarseMapView";

// 粗动大地图 —— 样品台尺度的第二张地图。
//
// 扫描地图画的是压电量程内的 ±1.5 µm，单位是米，只显示当前坐标代次。
// 这张画的是整个样品：每一次横向粗动 = 一个新站点，单位是**步**，跨全部代次。
// 两张图回答的是不同问题，所以刻意画成两块，不合并。
//
// 关键的视觉决定：站点是**模糊斑**不是点。粗动开环，位置来自步数累加，
// 误差随走过的路程增长。把它画成一个精确的点，等于把估计伪装成坐标 ——
// 而这张图的用途正是「不要回到已经去过的地方」，模糊半径就是判据本身。
//
// 压电量程按真实比例画在当前站点里：那个几乎看不见的小方块就是扫描地图的全部
// 范围。尺度差本身是最该被看见的信息 —— 它解释了为什么「换区」不能靠压电。

type CoarseMap = components["schemas"]["CoarseMapResponse"];
type Site = components["schemas"]["CoarseSiteView"];

function fmtSteps(n: number): string {
  return n.toLocaleString();
}

/** 步→µm 只作注释：粗动开环，步长随驱动幅度/负载/温度漂移。 */
function approx(site: Site): string | null {
  if (site.approx_x_um == null || site.approx_y_um == null) return null;
  return `≈ (${site.approx_x_um.toFixed(1)}, ${site.approx_y_um.toFixed(1)}) µm`;
}

export function CoarseMapPanel({
  data,
  isPending,
  isError,
  error,
  onRefresh,
  isFetching,
  defaultCompact = false,
}: {
  data?: CoarseMap;
  isPending: boolean;
  isError: boolean;
  error?: unknown;
  onRefresh: () => void;
  isFetching: boolean;
  /**
   * 初始是否紧凑（「粗动地图占用空间过大，搞一个小地图就行了」）。
   *
   * 是 prop 不是硬编码：对话页上这张图和聊天抢地方，`/vision` 上它是主角。
   * 是**初值**不是常态：用户随时能展开 —— 紧凑模式藏的是尺寸，不是信息。
   */
  defaultCompact?: boolean;
}) {
  const [compact, setCompact] = useState(defaultCompact);
  const [zoom, setZoom] = useState<CoarseZoom>(COARSE_ZOOM_RESET);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const fit = useMemo(
    () => fitCoarse(data?.sites ?? null, data?.site_spacing_steps),
    [data],
  );

  // 站点集换了（换站点、换样品）⇒ 缩放复位。
  // 缩放是相对**当时那个拟合框**的，留着它会把用户丢在新地图的某个角落 ——
  // 和双击复位存在的理由是同一个。
  const fitKey = fit ? `${fit.cx}|${fit.cy}|${fit.span}` : "";
  useEffect(() => {
    setZoom(COARSE_ZOOM_RESET);
  }, [fitKey]);

  /** 指针在 SVG 元素内的归一化位置。元素是正方形、viewBox 也是，所以是直的线性映射。 */
  const frac = useCallback((clientX: number, clientY: number) => {
    const el = svgRef.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return null;
    return { fx: (clientX - r.left) / r.width, fy: (clientY - r.top) / r.height };
  }, []);

  // 滚轮缩放。**必须走 addEventListener + passive:false** —— React 的 onWheel
  // 在根节点上是 passive 的，里面的 preventDefault() 不生效，于是滚轮会同时缩放
  // 地图和滚动页面。这不是洁癖：一边缩放一边整页乱跳，比没有缩放更难用。
  useEffect(() => {
    const el = svgRef.current;
    if (!el || !fit) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const f = frac(e.clientX, e.clientY);
      if (!f) return;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      setZoom((z) => zoomAtPoint(fit, z, f.fx, f.fy, factor));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [fit, frac]);

  // 拖动平移。只在放大之后有意义（k=1 时 panByFraction 本身就是空操作，
  // 这里的 guard 只是为了不无谓地改光标）。
  const drag = useRef<{ x: number; y: number } | null>(null);
  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!fit || zoom.k <= 1) return;
    drag.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    const el = svgRef.current;
    if (!d || !fit || !el) return;
    const r = el.getBoundingClientRect();
    if (r.width <= 0) return;
    const dfx = (e.clientX - d.x) / r.width;
    const dfy = (e.clientY - d.y) / r.height;
    drag.current = { x: e.clientX, y: e.clientY };
    setZoom((z) => panByFraction(fit, z, dfx, dfy));
  };
  const endDrag = (e: React.PointerEvent<SVGSVGElement>) => {
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  // 按钮缩放：以视图中心为锚点。滚轮不是每个人都顺手，而这张图的用途
  // （「别回到已经去过的地方」）不该只有一种进入方式。
  const nudge = (factor: number) =>
    setZoom((z) => (fit ? zoomAtPoint(fit, z, 0.5, 0.5, factor) : z));

  const zoomed = isZoomed(zoom);
  /** chrome 的基准尺寸都由拟合跨度派生；`ch()` 是它们**唯一**的出口。 */
  const ch = (denom: number) => (fit ? chromeAt(fit.span / denom, zoom) : 0);

  // 压电量程（米）折算成步，只有在用户填了步长标定时才画得出来。
  const piezoSteps =
    data?.step_m && data.step_m > 0
      ? (data.piezo_half_range_m ?? 1.5e-6) / data.step_m
      : null;

  const btn =
    "rounded border border-mast-border bg-mast-panel px-2 py-1 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50";

  return (
    <Card
      className={
        "min-w-[300px] flex-1 space-y-2 text-sm " + (compact ? "" : "lg:basis-full")
      }
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-semibold">粗动大地图（样品台尺度）</span>
        <div className="flex items-center gap-1">
          {/* 缩放控件。滚轮之外还给按钮：这张图回答的是「别回到已经去过的地方」，
              不该只有一种进入方式（触控板上的滚轮语义各家不一）。 */}
          <button
            type="button"
            onClick={() => nudge(1 / 1.6)}
            disabled={!fit || zoom.k <= 1}
            className={btn}
            aria-label="缩小"
            title="缩小"
          >
            −
          </button>
          <span className="min-w-[2.5rem] text-center text-xs tabular-nums text-mast-muted">
            {fmtZoom(zoom)}
          </span>
          <button
            type="button"
            onClick={() => nudge(1.6)}
            disabled={!fit || zoom.k >= COARSE_ZOOM_MAX}
            className={btn}
            aria-label="放大"
            title="放大"
          >
            +
          </button>
          <button
            type="button"
            onClick={() => setZoom(COARSE_ZOOM_RESET)}
            disabled={!zoomed}
            className={btn}
            title="回到能看见全部站点的视野"
          >
            复位
          </button>
          <button
            type="button"
            onClick={() => setCompact((v) => !v)}
            className={btn}
            title={compact ? "展开成大图" : "收成小地图"}
          >
            {compact ? "展开" : "紧凑"}
          </button>
          <button
            type="button"
            onClick={onRefresh}
            disabled={isFetching}
            className={btn}
          >
            {isFetching ? "读取中…" : "刷新"}
          </button>
        </div>
      </div>

      {isPending && <Spinner />}
      {isError && <ErrorNote error={error} />}
      {data?.degraded && <DegradedNote what="粗动大地图" />}

      {data && !data.degraded && (() => {
        // Narrow once, up front: every optional field below is required for the
        // drawing to mean anything, and threading `?.` through the geometry is
        // how a missing site silently becomes a plotted point at the origin.
        const sites = data.sites ?? [];
        const current = sites.find((s) => s.is_current);
        const budget = data.axis_step_budget ?? 0;
        const lands = data.suggestion?.lands_at_steps ?? [];
        return (
        <>
          {!compact && (
            <p className="text-xs text-mast-muted">
              每一次横向粗动 = 一个新站点 = 一个新坐标代次。单位是<b>步</b>，不是米：
              粗动开环，步长随驱动幅度 / 负载 / 温度漂移，所以站点画成<b>模糊斑</b>而不是点，
              半径就是位置不确定量。
            </p>
          )}

          {fit && (
            <svg
              ref={svgRef}
              viewBox={coarseViewBox(fit, zoom)}
              className="w-full touch-none select-none rounded border border-mast-border bg-mast-bg"
              style={{
                aspectRatio: "1 / 1",
                // 「占用空间过大」。压高度而不是砍内容 —— 能缩放之后，
                // 小尺寸不再意味着看不清。
                maxHeight: compact ? 280 : undefined,
                cursor: zoom.k > 1 ? (drag.current ? "grabbing" : "grab") : "default",
              }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              onDoubleClick={() => setZoom(COARSE_ZOOM_RESET)}
              role="img"
              aria-label="粗动站点分布（滚轮缩放，拖动平移，双击复位）"
            >
              {/* 行程预算边界。方框本身是 data（真实步数），描边是 chrome。 */}
              {budget > 0 && (
                <rect
                  x={-budget - fit.cx}
                  y={-budget + fit.cy}
                  width={budget * 2}
                  height={budget * 2}
                  fill="none"
                  stroke="currentColor"
                  strokeOpacity={0.18}
                  strokeDasharray={ch(60)}
                  strokeWidth={ch(300)}
                />
              )}

              {sites.map((s) => {
                const p = toStage(fit, s);
                const cur = s.is_current;
                // 半径是 data（真实不确定量），只有那个**下限**是 chrome ——
                // 它存在的理由是「小到看不见的斑也要看得见」，而放大之后
                // 真实半径自己就够大了，所以下限必须跟着缩。
                const r = Math.max(s.uncertainty_steps, ch(90));
                return (
                  <g key={s.index}>
                    {/* 模糊斑：位置不确定量。故意画得比标记大。 */}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={r}
                      fill={cur ? "#14b8a6" : "#8b5cf6"}
                      fillOpacity={cur ? 0.22 : 0.12}
                      stroke={cur ? "#14b8a6" : "#8b5cf6"}
                      strokeOpacity={s.position_known ? 0.5 : 0.25}
                      strokeWidth={ch(400)}
                      strokeDasharray={s.position_known ? undefined : ch(80)}
                    />
                    <circle cx={p.x} cy={p.y} r={ch(160)} fill={cur ? "#14b8a6" : "#8b5cf6"} />
                    {/* 压电量程按真实比例——那个小方块就是扫描地图的全部范围。
                        它是 data：放大到看得清它，正是这次加缩放要解决的事。 */}
                    {cur && piezoSteps && (
                      <rect
                        x={p.x - piezoSteps}
                        y={p.y - piezoSteps}
                        width={piezoSteps * 2}
                        height={piezoSteps * 2}
                        fill="none"
                        stroke="#22d3ee"
                        strokeWidth={ch(400)}
                      />
                    )}
                    <text
                      x={p.x}
                      y={p.y - r - ch(60)}
                      textAnchor="middle"
                      fill="currentColor"
                      fillOpacity={0.65}
                      fontSize={ch(34)}
                    >
                      #{s.index}
                      {s.summary?.scans ? ` · ${s.summary.scans}图` : ""}
                      {s.summary?.damage ? ` · ${s.summary.damage}破坏` : ""}
                    </text>
                  </g>
                );
              })}

              {/* 建议落点：十字 + 虚线 */}
              {data.suggestion && current && (() => {
                const from = toStage(fit, current);
                const to = toStage(fit, {
                  x_steps: lands[0] ?? 0,
                  y_steps: lands[1] ?? 0,
                });
                const k = ch(45);
                return (
                  <g stroke="#f59e0b" strokeWidth={ch(300)} fill="none">
                    <line
                      x1={from.x}
                      y1={from.y}
                      x2={to.x}
                      y2={to.y}
                      strokeDasharray={ch(70)}
                    />
                    <line x1={to.x - k} y1={to.y} x2={to.x + k} y2={to.y} />
                    <line x1={to.x} y1={to.y - k} x2={to.x} y2={to.y + k} />
                  </g>
                );
              })()}
            </svg>
          )}

          <div className="space-y-1 text-xs">
            <Row
              label="当前站点"
              value={`#${data.current_index}${data.position_known ? "" : "（里程表已失效）"}`}
            />
            <Row
              label="已用行程"
              value={`x ${fmtSteps(data.budget_used_steps?.x ?? 0)} / y ${fmtSteps(
                data.budget_used_steps?.y ?? 0,
              )} 步（预算 ${fmtSteps(budget)}）`}
            />
            {current?.temperature_k != null && (
              <Row label="移动时温度" value={`${current.temperature_k.toFixed(1)} K`} />
            )}
            {(() => {
              const a = current ? approx(current) : null;
              return a ? <Row label="粗略位置" value={`${a}（仅注释，不作几何）`} /> : null;
            })()}
          </div>

          {/* 站点履历。「大地图应包含过去小地图的历史记录」。
              这些字段（first_ts / last_ts / summary）端点一直在返回，只是从来
              没画出来过 —— SVG 上那行 `#0 · 3图` 已经小到读不出来，而「哪个站点
              什么时候做过什么」正是换站点前要问的问题。
              位置一律写「步」：与图上同一套开环里程表，不换算成米。 */}
          {sites.length > 0 && (
            <details className="text-xs" open={!compact}>
              {/* 紧凑模式把履历收起来,**不删掉** —— 他要的是「小地图」,
                  不是「少一半信息的地图」。收起 ≠ 不存在:`<details>` 自带
                  一行「共 N 个站点」,一眼能看出下面还有东西。 */}
              <summary
                className={
                  "cursor-pointer text-mast-muted marker:text-mast-muted " +
                  (compact ? "" : "list-none [&::-webkit-details-marker]:hidden")
                }
              >
                {compact ? `站点履历（共 ${sites.length} 个）` : ""}
              </summary>
              <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-mast-muted">
                    <th className="py-1 pr-3 font-normal">站点</th>
                    <th className="py-1 pr-3 font-normal">位置（步）</th>
                    <th className="py-1 pr-3 font-normal">扫描</th>
                    <th className="py-1 pr-3 font-normal">STS</th>
                    <th className="py-1 pr-3 font-normal">破坏</th>
                    <th className="py-1 font-normal">时间</th>
                  </tr>
                </thead>
                <tbody>
                  {sites.map((s) => (
                    <tr
                      key={s.index}
                      className={
                        "border-t border-mast-border " +
                        (s.is_current ? "bg-mast-accent-soft" : "")
                      }
                    >
                      <td className="py-1 pr-3 font-mono text-mast-text">
                        #{s.index}
                        {s.is_current ? " ·当前" : ""}
                        {s.position_known ? "" : " ·位置未知"}
                      </td>
                      <td className="py-1 pr-3 font-mono tabular-nums text-mast-muted">
                        {s.x_steps}, {s.y_steps}
                        {s.uncertainty_steps > 0
                          ? ` ±${Math.round(s.uncertainty_steps)}`
                          : ""}
                      </td>
                      <td className="py-1 pr-3 tabular-nums">{s.summary?.scans ?? 0}</td>
                      <td className="py-1 pr-3 tabular-nums">{s.summary?.sts ?? 0}</td>
                      <td
                        className={
                          "py-1 pr-3 tabular-nums " +
                          (s.summary?.damage ? "text-mast-warn" : "")
                        }
                      >
                        {s.summary?.damage ?? 0}
                      </td>
                      <td className="py-1 tabular-nums text-mast-muted">
                        {tsRange(s.first_ts, s.last_ts)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            </details>
          )}

          <div
            className={`rounded border px-2 py-1.5 text-xs ${
              data.suggestion
                ? "border-mast-border bg-mast-panel"
                : "border-mast-warn/40 bg-mast-warn/10"
            }`}
          >
            {data.suggestion ? (
              <>
                <div className="font-semibold text-mast-text">
                  建议：{data.suggestion.direction} × {fmtSteps(data.suggestion.steps)} 步
                </div>
                <div className="mt-0.5 text-mast-muted">{data.suggestion.reason}</div>
              </>
            ) : (
              <>
                <div className="font-semibold text-mast-warn">无可用落点</div>
                <div className="mt-0.5 text-mast-muted">{data.note || "—"}</div>
              </>
            )}
          </div>

          {/* 两道闸门。地图上有个好落点、而互锁在拒绝时，光看地图是看不出来的。 */}
          <div
            className={`rounded border px-2 py-1.5 text-xs ${
              data.vacuum_allow
                ? "border-mast-border bg-mast-panel text-mast-muted"
                : "border-mast-warn/40 bg-mast-warn/10 text-mast-warn"
            }`}
          >
            {data.vacuum_reason || "真空互锁状态未知"}
          </div>
          {!data.coarse_drive_declared && (
            <div className="rounded border border-mast-warn/40 bg-mast-warn/10 px-2 py-1.5 text-xs text-mast-warn">
              {data.coarse_drive_note}
            </div>
          )}

          <CoarseSelfCheck />
        </>
        );
      })()}
    </Card>
  );
}

// 验收自检。同一条只读探测,agent 走技能、用户走这个按钮 —— 一份计算两个受众,
// 和扫描地图分析同一个理由:能被检查的结论才值得被信任。
//
// 按需触发,不进任何轮询:它每次都要问一遍仪器,而它回答的是「装配对不对」,
// 那种事一分钟内不会变。
function CoarseSelfCheck() {
  const [open, setOpen] = useState(false);
  const q = useQuery({
    queryKey: ["coarse-map", "selfcheck"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/coarse-map/selfcheck");
      if (error) throw error;
      return data as {
        ready?: boolean;
        summary?: string;
        blocking?: string[];
        todo?: string[];
        degraded?: boolean;
        detail?: string;
      };
    },
    enabled: open,
    refetchOnWindowFocus: false,
  });

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="text-xs text-mast-muted underline hover:text-mast-text"
      >
        运行粗动自检（只读，进针/扫描中也可以跑）…
      </button>
    );
  }

  const d = q.data;
  return (
    <div className="space-y-2 rounded border border-mast-border bg-mast-panel p-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold">粗动自检</span>
        <button
          type="button"
          onClick={() => void q.refetch()}
          disabled={q.isFetching}
          className="rounded border border-mast-border bg-mast-bg px-2 py-0.5 text-[11px] text-mast-muted hover:text-mast-text disabled:opacity-50"
        >
          {q.isFetching ? "检查中…" : "重跑"}
        </button>
      </div>
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {d?.degraded && (
        <p className="text-xs text-mast-warn">{d.detail || "内核未就绪"}</p>
      )}
      {d && !d.degraded && (
        <>
          <p className={`text-xs font-semibold ${d.ready ? "text-mast-accent" : "text-mast-warn"}`}>
            {d.summary}
          </p>
          {!!d.blocking?.length && (
            <ul className="space-y-1 text-xs text-mast-warn">
              {d.blocking.map((b, i) => (
                <li key={i}>· {b}</li>
              ))}
            </ul>
          )}
          {!!d.todo?.length && (
            <details className="text-xs text-mast-muted">
              <summary className="cursor-pointer">
                还有 {d.todo.length} 项只有真机能回答
              </summary>
              <ul className="mt-1 space-y-1">
                {d.todo.map((t, i) => (
                  <li key={i}>· {t}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </div>
  );
}

/**
 * 「这个站点从什么时候做到什么时候」，写成一格。
 *
 * 只有一个时间戳时不编造区间：一个站点可以只有一条记录，写成「X ~ X」会让人
 * 以为在那里待了一段时间。两个都没有就留破折号 —— 没记过就是没记过。
 */
function tsRange(first?: string | null, last?: string | null): string {
  const a = short(first);
  const b = short(last);
  if (!a && !b) return "—";
  if (!a) return b;
  if (!b || a === b) return a;
  return `${a} ~ ${b}`;
}

function short(ts?: string | null): string {
  if (!ts) return "";
  // "2026-08-05T13:42:11" → "08-05 13:42"
  return ts.replace("T", " ").slice(5, 16);
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-mast-muted">{label}</span>
      <span className="tabular-nums text-right">{value}</span>
    </div>
  );
}
