import { useEffect, useMemo, useState } from "react";
import { Stage, Layer, Rect, Circle, Line, Text, Group, Image as KonvaImage } from "react-konva";
import type { components } from "@/api/schema";
import { useUiStore } from "@/store";
import { useElementWidth, useViewportHeight } from "@/hooks/useElementWidth";
import { MAP_RATIO, sizeChart } from "@/lib/chartSize";
import {
  DEFAULT_LAYERS,
  boundsForMode,
  fmtNm,
  inverseStageTransform,
  isStaleEpoch,
  markerDetailRows,
  pinnedLabel,
  stageToScreen,
  visibleStepNm,
  type MapLayers,
  type MapViewMode,
} from "@/lib/scanMapView";
import { Modal } from "@/components/controls";

type ScanMap = components["schemas"]["ScanMapResponse"];
type Marker = components["schemas"]["MapMarkerView"];
type ScanImg = components["schemas"]["ScanImage"];
type MapAnalysis = components["schemas"]["ScanMapAnalysisResponse"];

// Native Konva render of the live experiment map. Coordinates are the Nanonis
// stage frame, in METRES; we auto-fit every spatial element (frame + tip +
// markers + plan) into the canvas with a metres→pixels affine and flip Y so +y
// points up (lab convention). Redone 2026-07-01:
//   • rotated footprints now pivot about their CENTRE (was: top-left corner →
//     rotated scans landed in the wrong place)
//   • marker colours match the backend KIND_STYLE exactly (was: 'tip_shape'
//     fell through to grey; 'plan'/'frame' had no colour)
//   • hover tooltip (label · kind · time), an nm grid, axis ticks and a scale
//     bar so the map actually reads as a scientific surface map
//   • honest empty / degraded states

// The canvas used to be a hardcoded 620×460 that no caller ever overrode, so a
// wide monitor showed the same postage stamp as a laptop and the rest of the row
// was dead space (「扫描地图这个占网页的面积太小」). It is now sized
// from the width its container actually got; 620×460 survives as the fallback
// for the first frame, before the ResizeObserver has reported.
const FALLBACK_W = 620;
const PAD = 44;

// kind → colour, mirroring mast/io/exp_map.py KIND_STYLE so the Konva map and
// the backend matplotlib legend never disagree.
const KIND_COLOR = {
  scan: "#3b82f6",
  sts: "#22c55e",
  pulse: "#f59e0b",
  tip_shape: "#ef4444",
  move: "#94a3b8",
  manual: "#e879f9",
  plan: "#38bdf8",
  frame: "#06b6d4",
  tip: "#f43f5e",
  coarse_move: "#8b5cf6",
  approach: "#14b8a6",
  crash: "#991b1b",
  manual_avoid: "#e879f9",
};
const KIND_LABEL = {
  scan: "扫图",
  sts: "STS 点谱",
  pulse: "电脉冲",
  tip_shape: "修针尖",
  move: "移动",
  manual: "手动操作",
  plan: "计划",
  frame: "扫描框",
  tip: "针尖",
  coarse_move: "粗动换区",
  approach: "进针",
  crash: "撞针",
  manual_avoid: "人工避让区",
};
// Opacity multiplier for a marker whose coordinate belongs to a superseded
// coordinate generation. A lateral coarse move slides the sample stage, so those
// numbers now address a different patch of surface — the marker is real history
// but it is NOT where it is drawn any more. Faded rather than hidden: the
// operator should still be able to see that work happened.
const STALE_EPOCH_OPACITY = 0.22;
// Chrome colours — grid, ticks, scale bar, tooltip. Konva paints onto a canvas
// and cannot read a CSS variable, so the palette has to be resolved in JS;
// hard-coding the dark one is what left the map a black slab in light mode
// (「扫描地图的黑色底不好看，应该跟随主界面的白天/黑夜模式」#24).
//
// The MARKER colours above are deliberately NOT themed: they mirror
// mast/io/exp_map.py KIND_STYLE so the Konva map and the backend matplotlib
// legend never disagree, and "扫图 is blue" must mean the same thing in a
// screenshot regardless of who took it.
const CHROME = {
  dark: {
    bg: "#0b1017", grid: "#1e293b", tick: "#64748b", axis: "#94a3b8",
    bar: "#e2e8f0", tipRing: "#ffffff",
    tooltipBg: "#111827", tooltipBorder: "#334155", tooltipText: "#e2e8f0",
  },
  light: {
    bg: "#f8fafc", grid: "#dde3ec", tick: "#64748b", axis: "#475569",
    bar: "#1e293b", tipRing: "#ffffff",
    tooltipBg: "#ffffff", tooltipBorder: "#cbd5e1", tooltipText: "#0f172a",
  },
} as const;

function kindColor(kind?: string | null): string {
  return (KIND_COLOR as Record<string, string>)[(kind ?? "").toLowerCase()] ?? "#94a3b8";
}
function kindLabel(kind?: string | null): string {
  return (KIND_LABEL as Record<string, string>)[kind ?? ""] ?? "";
}

interface Pt {
  x: number;
  y: number;
}

// Which elements are in view — and which are drawn at all — now lives in
// lib/scanMapView.ts (boundsForMode), where it is covered by npm run test:unit.
// So do the grid step and the chrome transform (visibleStepNm /
// inverseStageTransform / stageToScreen).

export function ScanMapCanvas({
  map,
  analysis,
  layers = DEFAULT_LAYERS,
  viewMode = "fit",
}: {
  map: ScanMap;
  analysis?: MapAnalysis | null;
  layers?: MapLayers;
  viewMode?: MapViewMode;
}) {
  const c = CHROME[useUiStore((st) => st.theme) === "light" ? "light" : "dark"];
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  // 「扫描地图点击标记弹出具体信息、时间等」. The hover tooltip is
  // one un-wrapped line capped at 200 px, so everything else the marker knows
  // had nowhere to go.
  const [detail, setDetail] = useState<Marker | null>(null);
  const [boxRef, measuredW] = useElementWidth<HTMLDivElement>();
  const viewportH = useViewportHeight();
  // Konva's Stage needs concrete pixels — `width: 100%` does nothing to a canvas.
  const { width: CANVAS_W, height: CANVAS_H } = sizeChart(measuredW, FALLBACK_W, {
    ratio: MAP_RATIO,
    minHeight: 320,
    maxHeight: 900,
    viewportH,
  });
  // Wheel zoom + drag pan ("扫描地图不能放大缩小").
  // Zoom is about the cursor; drag pans when zoomed; double-click resets.
  const [zoom, setZoom] = useState<{ k: number; x: number; y: number }>({ k: 1, x: 0, y: 0 });
  // Zoom is a screen-space transform anchored to the extent it was applied to.
  // Keeping it across a view change would drag the operator to an unpredictable
  // corner of the new one — the same reason double-click resets.
  useEffect(() => {
    setZoom({ k: 1, x: 0, y: 0 });
  }, [viewMode]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function onWheel(e: any) {
    e.evt.preventDefault();
    const stage = e.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer) return;
    const factor = e.evt.deltaY < 0 ? 1.15 : 1 / 1.15;
    setZoom((z) => {
      const k = Math.min(25, Math.max(1, z.k * factor));
      if (k === z.k) return z;
      if (k === 1) return { k: 1, x: 0, y: 0 };
      // keep the world point under the cursor fixed while scaling
      const wx = (pointer.x - z.x) / z.k;
      const wy = (pointer.y - z.y) / z.k;
      return { k, x: pointer.x - wx * k, y: pointer.y - wy * k };
    });
  }

  const fit = useMemo(
    () => boundsForMode(viewMode, map, analysis, layers, map.piezo_half_range_m),
    [viewMode, map, analysis, layers],
  );

  const view = useMemo(() => {
    const b = fit.bounds;
    if (!b) return null;
    let { minX, maxX, minY, maxY } = b;
    let spanX = maxX - minX;
    let spanY = maxY - minY;
    // Degenerate (single point) → a fixed ~100 nm window rather than a dot.
    const FIXED = 1e-7;
    if (spanX <= 0 && spanY <= 0) {
      minX -= FIXED / 2; maxX += FIXED / 2; minY -= FIXED / 2; maxY += FIXED / 2;
    }
    // Square the data box so equal x/y scale doesn't distort footprints.
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const half = Math.max(maxX - minX, maxY - minY, FIXED) / 2;
    const pad = half * 0.14;
    minX = cx - half - pad; maxX = cx + half + pad;
    minY = cy - half - pad; maxY = cy + half + pad;
    spanX = maxX - minX; spanY = maxY - minY;

    const s = Math.min((CANVAS_W - 2 * PAD) / spanX, (CANVAS_H - 2 * PAD) / spanY);
    const offX = PAD + (CANVAS_W - 2 * PAD - spanX * s) / 2;
    const offY = PAD + (CANVAS_H - 2 * PAD - spanY * s) / 2;
    const toPx = (xm: number, ym: number): Pt => ({
      x: offX + (xm - minX) * s,
      y: CANVAS_H - (offY + (ym - minY) * s), // flip Y so +y is up
    });
    return { toPx, scale: s, minX, maxX, minY, maxY };
  }, [fit]);

  if (!view) {
    return (
      <div
        ref={boxRef}
        className="flex w-full items-center justify-center rounded-lg border border-mast-border text-center text-sm text-mast-muted"
        style={{ height: CANVAS_H, background: c.bg }}
      >
        等待扫描范围…
        <br />
        开始扫描或读取仪器状态后，这里会显示扫描框、针尖与所有带位置的操作。
      </div>
    );
  }
  const { toPx, scale, minX, maxX, minY, maxY } = view;

  const f = map.frame;
  const t = map.tip_xyz;
  const allMarkers = map.markers ?? [];
  // Planned steps ride in `markers` in EXECUTION ORDER (PlanOverlay.snapshot is
  // next-first and the API preserves it), which is where the step numbers and
  // the "next" highlight come from — there is no separate ordering field.
  const planned = allMarkers.filter(
    (m) => (m.status ?? "") === "planned" && m.x_m != null && m.y_m != null,
  );
  const markers = allMarkers.filter((m) => {
    if ((m.status ?? "") === "planned") return false; // drawn by PlanRoute
    return isStaleEpoch(m, map.current_epoch) ? layers.stale : layers.history;
  });

  // Grid lines on nice nm boundaries. The step follows the VISIBLE span, so
  // zooming in subdivides the grid instead of leaving it at whole-map spacing
  // — and keeps the scale bar a constant fraction of the canvas .
  const spanNm = (maxX - minX) * 1e9;
  const stepNm = visibleStepNm(spanNm, zoom.k);
  const gridXs: number[] = [];
  const gridYs: number[] = [];
  const startNm = Math.ceil((minX * 1e9) / stepNm) * stepNm;
  for (let gx = startNm; gx <= maxX * 1e9; gx += stepNm) gridXs.push(gx);
  const startYNm = Math.ceil((minY * 1e9) / stepNm) * stepNm;
  for (let gy = startYNm; gy <= maxY * 1e9; gy += stepNm) gridYs.push(gy);

  // Chrome lives in an un-zoomed layer: the numbers that tell the operator the
  // scale must stay on the canvas at 25× (see inverseStageTransform).
  const chrome = inverseStageTransform(zoom);
  const onCanvas = (x: number) => x >= -20 && x <= CANVAS_W + 20;
  const scaleBarPx = stepNm * 1e-9 * scale * zoom.k;
  // Grid lines span the VISIBLE window inset by PAD/2 — identical to the old
  // fixed [PAD/2, CANVAS_H-PAD/2] box at 1×, but panning no longer walks off
  // the end of them.
  const gridTop = (PAD / 2 - zoom.y) / zoom.k;
  const gridBottom = (CANVAS_H - PAD / 2 - zoom.y) / zoom.k;
  const gridLeft = (PAD / 2 - zoom.x) / zoom.k;
  const gridRight = (CANVAS_W - PAD / 2 - zoom.x) / zoom.k;

  return (
    // `w-full` (not `inline-block`) so the box width comes from the layout and
    // never from the canvas inside it — measuring a shrink-to-fit box that
    // contains the thing being sized is a feedback loop.
    <div
      ref={boxRef}
      className="relative w-full overflow-x-auto rounded-lg border border-mast-border"
      style={{ background: c.bg }}
    >
      <Stage
        width={CANVAS_W}
        height={CANVAS_H}
        scaleX={zoom.k}
        scaleY={zoom.k}
        x={zoom.x}
        y={zoom.y}
        draggable={zoom.k > 1}
        onWheel={onWheel}
        onDblClick={() => setZoom({ k: 1, x: 0, y: 0 })}
        onDragEnd={(e) => {
          const st = e.target.getStage();
          if (st && e.target === st) setZoom((z) => ({ ...z, x: st.x(), y: st.y() }));
        }}
        onMouseLeave={() => setHover(null)}
      >
        {/* Grid LINES are surface geometry — they belong to the zoomed world.
            Their extent is the visible stage window, not the canvas box, or
            panning walks off the end of them. */}
        <Layer listening={false} visible={layers.grid}>
          {gridXs.map((gx, i) => {
            const px = toPx(gx * 1e-9, minY).x;
            return <Line key={`gx${i}`} points={[px, gridTop, px, gridBottom]} stroke={c.grid} strokeWidth={0.6 / zoom.k} />;
          })}
          {gridYs.map((gy, i) => {
            const py = toPx(minX, gy * 1e-9).y;
            return <Line key={`gy${i}`} points={[gridLeft, py, gridRight, py]} stroke={c.grid} strokeWidth={0.6 / zoom.k} />;
          })}
        </Layer>

        <Layer>
          {/* the reachable piezo area — only worth drawing when the view is the
              whole range, where it is the one thing giving the dots a scale */}
          {viewMode === "full" && map.piezo_half_range_m != null && (
            <PiezoRangeShape half={map.piezo_half_range_m} toPx={toPx} stroke={c.tick} zoomK={zoom.k} />
          )}

          {/* saved-scan surface mosaic — real .sxm thumbnails placed by footprint */}
          {layers.underlay &&
            (map.scan_images ?? []).map((s, i) => (
              <ScanImageShape key={`si-${i}`} img={s} toPx={toPx} scale={scale} />
            ))}

          {/* keep-out discs: where the surface is damaged or contaminated.
              Under the markers so the events that caused them stay readable. */}
          {layers.avoid &&
            (analysis?.avoid_zones ?? []).map((z, i) => (
              <AvoidZoneShape key={`az-${i}`} zone={z} toPx={toPx} scale={scale} />
            ))}

          {/* the route the strategy would walk next — ghost frames, numbered */}
          {layers.upcoming && (
            <UpcomingRoute
              positions={analysis?.upcoming ?? []}
              frameSizeM={analysis?.frame_size_m ?? 0}
              /* the first upcoming position IS next_position; drawing both
                 would double-strike one frame with two different labels */
              skipFirst={layers.next && analysis?.next_position?.x_m != null}
              toPx={toPx}
              scale={scale}
              zoomK={zoom.k}
            />
          )}

          {/* planned route: dashed polyline through plan markers, in order */}
          {layers.plan && (
            <PlanRoute
              plan={planned}
              toPx={toPx}
              scale={scale}
              zoomK={zoom.k}
              onHover={(text, x, y) => setHover({ x, y, text })}
              onOut={() => setHover(null)}
            />
          )}

          {/* history markers (rect footprint when sized, else dot) */}
          {markers.map((m, i) => (
            <MarkerShape
              key={`${m.skill_name}-${i}`}
              m={m}
              currentEpoch={map.current_epoch}
              toPx={toPx}
              scale={scale}
              onHover={(text) => {
                if (m.x_m != null && m.y_m != null) {
                  const p = toPx(m.x_m, m.y_m);
                  setHover({ x: p.x, y: p.y, text });
                }
              }}
              onOut={() => setHover(null)}
              onSelect={() => setDetail(m)}
            />
          ))}

          {/* live current scan frame */}
          {f && f.center_x_m != null && f.center_y_m != null && f.width_m != null && f.height_m != null && (
            <FrameShape frame={f} toPx={toPx} scale={scale} />
          )}

          {/* the recommended next scan position, on top of everything it had to
              avoid to get there */}
          {layers.next && analysis?.next_position?.x_m != null && analysis.next_position.y_m != null && (
            <NextPositionShape
              at={toPx(analysis.next_position.x_m, analysis.next_position.y_m)}
              sizePx={(analysis.frame_size_m ?? 0) * scale}
              zoomK={zoom.k}
            />
          )}

          {/* tip position + crosshair */}
          {t && t.x_m != null && t.y_m != null && (
            <TipShape at={toPx(t.x_m, t.y_m)} ring={c.tipRing} zoomK={zoom.k} />
          )}
        </Layer>

        {/* Chrome — axis numbers, scale bar, tooltip. This layer carries the
            INVERSE of the stage transform, so it stays pinned to the viewport
            at any zoom/pan: at 25× the operator still gets a readable scale
            instead of a scale that slid off the canvas . */}
        <Layer listening={false} {...chrome}>
          {layers.grid && (
            <>
              {/* tick labels ride the world in x, the canvas edge in y */}
              {gridXs.map((gx, i) => {
                const sx = stageToScreen(toPx(gx * 1e-9, minY), zoom).x;
                if (!onCanvas(sx)) return null;
                return <Text key={`lx${i}`} x={sx - 12} y={CANVAS_H - 16} text={fmtNm(gx, stepNm)} fontSize={9} fill={c.tick} />;
              })}
              {gridYs.map((gy, i) => {
                const sy = stageToScreen(toPx(minX, gy * 1e-9), zoom).y;
                if (sy < -20 || sy > CANVAS_H + 20) return null;
                return <Text key={`ly${i}`} x={4} y={sy - 5} text={fmtNm(gy, stepNm)} fontSize={9} fill={c.tick} />;
              })}
              <Text x={CANVAS_W - 34} y={CANVAS_H - 15} text="X nm" fontSize={9} fill={c.axis} />
            </>
          )}
          <Line points={[PAD, CANVAS_H - 26, PAD + scaleBarPx, CANVAS_H - 26]} stroke={c.bar} strokeWidth={2} />
          <Line points={[PAD, CANVAS_H - 30, PAD, CANVAS_H - 22]} stroke={c.bar} strokeWidth={2} />
          <Line points={[PAD + scaleBarPx, CANVAS_H - 30, PAD + scaleBarPx, CANVAS_H - 22]} stroke={c.bar} strokeWidth={2} />
          <Text x={PAD} y={CANVAS_H - 44} text={`${fmtNm(stepNm, stepNm)} nm`} fontSize={10} fill={c.bar} />
          {hover && (() => {
            const h = stageToScreen({ x: hover.x, y: hover.y }, zoom);
            return (
              <Group x={Math.min(h.x + 10, CANVAS_W - 150)} y={Math.max(h.y - 28, 4)}>
                <Rect width={Math.min(hover.text.length * 7 + 12, 200)} height={22} fill={c.tooltipBg} stroke={c.tooltipBorder} strokeWidth={1} cornerRadius={4} opacity={0.95} />
                <Text x={6} y={5} text={hover.text} fontSize={11} fill={c.tooltipText} width={188} ellipsis wrap="none" />
              </Group>
            );
          })()}
        </Layer>
      </Stage>
      {/* zoom affordance — visible hint + live multiplier.
          Also where a view mode says it could not be honoured: silently showing
          a different area than the selected button claims is worse than empty. */}
      <div className="pointer-events-none absolute right-2 top-2 rounded bg-mast-panel/80 px-1.5 py-0.5 text-[10px] text-mast-muted">
        {fit.fellBack
          ? viewMode === "follow"
            ? "无实时扫描框 · 已自动拟合"
            : "压电范围未知 · 已自动拟合"
          : zoom.k > 1
            ? `${zoom.k.toFixed(1)}× · 拖拽平移 · 双击复位`
            : "滚轮缩放"}
      </div>

      <Modal
        open={detail != null}
        onClose={() => setDetail(null)}
        title={
          detail
            ? detail.label || kindLabel(detail.kind) || detail.kind || "地图标记"
            : ""
        }
      >
        {detail && (
          <div className="space-y-1 text-sm">
            {markerDetailRows(detail, map.current_epoch).map((r) => (
              <div key={r.label} className="flex gap-3">
                <span className="w-20 shrink-0 text-mast-muted">{r.label}</span>
                <span className={r.mono ? "font-mono tabular-nums text-mast-text" : "text-mast-text"}>
                  {r.value}
                </span>
              </div>
            ))}
            {markerDetailRows(detail, map.current_epoch).length === 0 && (
              <p className="text-mast-muted">这个标记只记录了位置，没有别的信息。</p>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}

function _footprintPx(
  m: { x_m?: number | null; y_m?: number | null; w_m?: number | null; h_m?: number | null; angle_deg?: number | null },
  toPx: (x: number, y: number) => Pt,
  scale: number,
) {
  const c = toPx(m.x_m!, m.y_m!);
  return {
    // Position the Rect at the CENTRE and shift its origin there via offset, so
    // Konva rotates about the centre (not the top-left corner) — the core fix.
    cx: c.x,
    cy: c.y,
    w: Math.abs(m.w_m!) * scale,
    h: Math.abs(m.h_m!) * scale,
    rotation: -(m.angle_deg ?? 0), // lab CCW-positive → Konva CW-positive screen
  };
}

function ScanImageShape({
  img,
  toPx,
  scale,
}: {
  img: ScanImg;
  toPx: (x: number, y: number) => Pt;
  scale: number;
}) {
  const [el, setEl] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!img.image_b64) return;
    const image = new window.Image();
    image.src = `data:image/png;base64,${img.image_b64}`;
    const done = () => setEl(image);
    if (image.complete) done();
    else image.onload = done;
    return () => {
      image.onload = null;
    };
  }, [img.image_b64]);
  if (!el || img.center_x_m == null || img.center_y_m == null || !img.width_m || !img.height_m) {
    return null;
  }
  const c = toPx(img.center_x_m, img.center_y_m);
  const w = img.width_m * scale;
  const h = img.height_m * scale;
  return (
    <KonvaImage
      image={el}
      x={c.x}
      y={c.y}
      width={w}
      height={h}
      offsetX={w / 2}
      offsetY={h / 2}
      rotation={-(img.angle_deg ?? 0)}
      opacity={0.92}
      listening={false}
    />
  );
}

function MarkerShape({
  m,
  currentEpoch,
  toPx,
  scale,
  onHover,
  onOut,
  onSelect,
}: {
  m: Marker;
  currentEpoch?: number | null;
  toPx: (x: number, y: number) => Pt;
  scale: number;
  onHover: (text: string) => void;
  onOut: () => void;
  onSelect: () => void;
}) {
  if (m.x_m == null || m.y_m == null) return null;
  const color = kindColor(m.kind);
  const planned = (m.status ?? "done") === "planned";
  const failed = (m.status ?? "done") === "failed";
  // A superseded coordinate generation: real history, but the sample stage has
  // moved since, so this is not where it happened any more.
  const stale = isStaleEpoch(m, currentEpoch);
  const fade = stale ? STALE_EPOCH_OPACITY : 1;
  const dim = (planned ? 0.5 : 1) * fade;
  const tip =
    `${m.label || kindLabel(m.kind) || m.kind || "操作"}` +
    `${m.timestamp ? " · " + m.timestamp.slice(11, 19) : ""}` +
    (stale ? " · 粗动前的旧坐标系" : "");
  const hoverProps = {
    onMouseEnter: (e: { target: { getStage: () => { container: () => HTMLElement } | null } }) => {
      onHover(tip);
      // The cursor is the only affordance saying these are clickable — Konva
      // shapes get none from CSS.
      const stage = e.target.getStage();
      if (stage) stage.container().style.cursor = "pointer";
    },
    onMouseMove: () => onHover(tip),
    onMouseLeave: (e: { target: { getStage: () => { container: () => HTMLElement } | null } }) => {
      onOut();
      const stage = e.target.getStage();
      if (stage) stage.container().style.cursor = "";
    },
    onClick: onSelect,
    onTap: onSelect,
  };

  if (m.w_m != null && m.h_m != null && m.w_m > 0 && m.h_m > 0) {
    const fp = _footprintPx(m, toPx, scale);
    return (
      <Rect
        x={fp.cx}
        y={fp.cy}
        width={fp.w}
        height={fp.h}
        offsetX={fp.w / 2}
        offsetY={fp.h / 2}
        rotation={fp.rotation}
        stroke={color}
        strokeWidth={1.5}
        fill={m.kind === "scan" && !failed ? color : undefined}
        opacity={(m.kind === "scan" ? (failed ? 0.15 : 0.22) : dim) * (stale ? STALE_EPOCH_OPACITY : 1)}
        dash={planned || failed ? [4, 4] : undefined}
        {...hoverProps}
      />
    );
  }
  const c = toPx(m.x_m, m.y_m);
  return <Circle x={c.x} y={c.y} radius={4.5} fill={color} opacity={dim} {...hoverProps} />;
}

/** A keep-out disc from the map analysis. */
function AvoidZoneShape({
  zone,
  toPx,
  scale,
}: {
  zone: NonNullable<MapAnalysis["avoid_zones"]>[number];
  toPx: (x: number, y: number) => Pt;
  scale: number;
}) {
  const p = toPx(zone.x_m, zone.y_m);
  const r = zone.radius_m * scale;
  if (!Number.isFinite(r) || r <= 0) return null;
  const color = kindColor(zone.kind);
  return (
    <Circle
      x={p.x}
      y={p.y}
      radius={r}
      fill={color}
      opacity={0.12}
      stroke={color}
      strokeWidth={1}
      dash={[3, 3]}
      listening={false}
    />
  );
}

/** Where the analysis says to scan next: a crosshair plus the frame outline. */
function NextPositionShape({ at, sizePx, zoomK }: { at: Pt; sizePx: number; zoomK: number }) {
  const half = Math.max(sizePx, 6) / 2;
  const C = "#38bdf8"; // same "this has not happened yet" hue as the plan route
  return (
    <Group listening={false}>
      <Rect
        x={at.x}
        y={at.y}
        width={half * 2}
        height={half * 2}
        offsetX={half}
        offsetY={half}
        stroke={C}
        strokeWidth={1.6}
        dash={[5, 3]}
      />
      <Line points={[at.x - half - 6, at.y, at.x + half + 6, at.y]} stroke={C} strokeWidth={1} />
      <Line points={[at.x, at.y - half - 6, at.x, at.y + half + 6]} stroke={C} strokeWidth={1} />
      {/* half is a WORLD offset (the frame grows with zoom, the label must keep
          clearing it); 4/-14 are the screen gap, so they do not. */}
      <Text
        {...pinnedLabel({ x: at.x + half, y: at.y - half }, 4, -14, zoomK)}
        text="建议下一个位置"
        fontSize={10}
        fill={C}
      />
    </Group>
  );
}

/** The published route: dashed polyline, numbered steps, next step highlighted.
 *
 *  The steps arrive in execution order and shrink from the front as operations
 *  land (PlanOverlay.advance), so the first element is always what happens next
 *  and the length is always how much is left — no extra fields needed. Matches
 *  what the backend matplotlib renderer has always drawn (exp_map.render_map_figure). */
function PlanRoute({
  plan,
  toPx,
  scale,
  zoomK,
  onHover,
  onOut,
}: {
  plan: Marker[];
  toPx: (x: number, y: number) => Pt;
  scale: number;
  zoomK: number;
  onHover: (text: string, x: number, y: number) => void;
  onOut: () => void;
}) {
  const first = plan[0];
  if (!first) return null;
  const C = "#38bdf8";
  const NUM = "#7dd3fc";
  const pts = plan.flatMap((m) => {
    const p = toPx(m.x_m!, m.y_m!);
    return [p.x, p.y];
  });
  const head = toPx(first.x_m!, first.y_m!);
  return (
    <Group>
      {plan.length >= 2 && (
        <Line points={pts} stroke={C} strokeWidth={1.4} dash={[6, 4]} opacity={0.7} listening={false} />
      )}
      {plan.map((m, i) => {
        const p = toPx(m.x_m!, m.y_m!);
        const w = Math.abs(m.w_m ?? 0) * scale;
        const h = Math.abs(m.h_m ?? 0) * scale;
        const tip =
          `计划 ${i + 1}/${plan.length}` +
          (m.label ? ` · ${m.label}` : "") +
          (i === 0 ? " · 下一步" : "");
        const hoverProps = {
          onMouseEnter: () => onHover(tip, p.x, p.y),
          onMouseMove: () => onHover(tip, p.x, p.y),
          onMouseLeave: onOut,
        };
        return (
          <Group key={`plan-${i}`}>
            {w > 0 && h > 0 ? (
              <Rect
                x={p.x}
                y={p.y}
                width={w}
                height={h}
                offsetX={w / 2}
                offsetY={h / 2}
                rotation={-(m.angle_deg ?? 0)}
                stroke={C}
                strokeWidth={i === 0 ? 1.8 : 1.2}
                dash={[4, 4]}
                opacity={i === 0 ? 0.95 : 0.5}
                {...hoverProps}
              />
            ) : (
              <Circle x={p.x} y={p.y} radius={4} stroke={C} strokeWidth={1.2}
                      opacity={i === 0 ? 0.95 : 0.5} {...hoverProps} />
            )}
            <Text
              {...pinnedLabel(
                { x: p.x + Math.max(w, 8) / 2, y: p.y - Math.max(h, 8) / 2 },
                3,
                -11,
                zoomK,
              )}
              text={`${i + 1}`}
              fontSize={10}
              fill={NUM}
              listening={false}
            />
          </Group>
        );
      })}
      {/* The step about to be executed, ringed so it reads at a glance. NO text
          label here: the route, the candidate sequence and 建议下一个位置 all
          converge on the same few hundred nanometres, and a sentence pinned to
          this point lands on top of them. 「计划 · <title> · 剩 N 步」 is a
          status fact, not a map feature — it lives in the chip row above the
          canvas, where it is readable at any zoom. */}
      <Circle x={head.x} y={head.y} radius={9} stroke={C} strokeWidth={1.6} opacity={0.9} listening={false} />
    </Group>
  );
}

/** The positions the current strategy would walk next — ghost frames, numbered.
 *
 *  Distinct from the published plan: nobody committed to these, they are what
 *  the analysis recomputes from the record every time. Same "not yet" hue,
 *  fainter, so the two channels read as related but not the same claim. */
function UpcomingRoute({
  positions,
  frameSizeM,
  skipFirst,
  toPx,
  scale,
  zoomK,
}: {
  positions: NonNullable<MapAnalysis["upcoming"]>;
  frameSizeM: number;
  skipFirst: boolean;
  toPx: (x: number, y: number) => Pt;
  scale: number;
  zoomK: number;
}) {
  const pts = positions.filter((p) => p.x_m != null && p.y_m != null);
  if (!pts.length) return null;
  const C = "#38bdf8";
  const side = Math.max((frameSizeM || 0) * scale, 6);
  const half = side / 2;
  return (
    <Group listening={false}>
      {pts.length >= 2 && (
        <Line
          points={pts.flatMap((p) => {
            const q = toPx(p.x_m!, p.y_m!);
            return [q.x, q.y];
          })}
          stroke={C}
          strokeWidth={1}
          dash={[2, 4]}
          opacity={0.35}
        />
      )}
      {pts.map((p, i) => {
        if (i === 0 && skipFirst) return null;
        const q = toPx(p.x_m!, p.y_m!);
        return (
          <Group key={`up-${i}`}>
            <Rect
              x={q.x}
              y={q.y}
              width={side}
              height={side}
              offsetX={half}
              offsetY={half}
              stroke={C}
              strokeWidth={1}
              dash={[2, 3]}
              opacity={0.45}
            />
            <Text
              {...pinnedLabel({ x: q.x - half, y: q.y - half }, 2, 2, zoomK)}
              text={`${i + 1}`}
              fontSize={9}
              fill={C}
              opacity={0.6}
            />
          </Group>
        );
      })}
    </Group>
  );
}

/** The reachable piezo area, as a dashed square. Drawn only in the whole-range
 *  view, where without it the nanometre-scale markers have nothing to scale
 *  against and the map reads as a few dots in an empty field. */
function PiezoRangeShape({
  half,
  toPx,
  stroke,
  zoomK,
}: {
  half: number;
  toPx: (x: number, y: number) => Pt;
  stroke: string;
  zoomK: number;
}) {
  const a = toPx(-half, -half);
  const b = toPx(half, half);
  return (
    <Group listening={false}>
      <Rect
        x={Math.min(a.x, b.x)}
        y={Math.min(a.y, b.y)}
        width={Math.abs(b.x - a.x)}
        height={Math.abs(b.y - a.y)}
        stroke={stroke}
        strokeWidth={1}
        dash={[4, 4]}
        opacity={0.7}
      />
      <Text
        {...pinnedLabel({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y) }, 4, 4, zoomK)}
        text={`压电范围 ±${(half * 1e6).toFixed(2)} µm`}
        fontSize={9}
        fill={stroke}
      />
    </Group>
  );
}

function FrameShape({
  frame,
  toPx,
  scale,
}: {
  frame: NonNullable<ScanMap["frame"]>;
  toPx: (x: number, y: number) => Pt;
  scale: number;
}) {
  const fp = _footprintPx(
    { x_m: frame.center_x_m, y_m: frame.center_y_m, w_m: frame.width_m, h_m: frame.height_m, angle_deg: frame.angle_deg },
    toPx,
    scale,
  );
  return (
    <Group>
      <Rect x={fp.cx} y={fp.cy} width={fp.w} height={fp.h} offsetX={fp.w / 2} offsetY={fp.h / 2} rotation={fp.rotation} stroke="#06b6d4" strokeWidth={2} />
      <Line points={[fp.cx - 8, fp.cy, fp.cx + 8, fp.cy]} stroke="#06b6d4" strokeWidth={1} />
      <Line points={[fp.cx, fp.cy - 8, fp.cx, fp.cy + 8]} stroke="#06b6d4" strokeWidth={1} />
    </Group>
  );
}

function TipShape({ at, ring, zoomK }: { at: Pt; ring: string; zoomK: number }) {
  return (
    <Group>
      <Line points={[at.x - 10, at.y, at.x + 10, at.y]} stroke="#f43f5e" strokeWidth={0.8} opacity={0.5} />
      <Line points={[at.x, at.y - 10, at.x, at.y + 10]} stroke="#f43f5e" strokeWidth={0.8} opacity={0.5} />
      <Circle x={at.x} y={at.y} radius={5} fill="#f43f5e" stroke={ring} strokeWidth={1} />
      <Text {...pinnedLabel(at, 8, -6, zoomK)} text="针尖" fontSize={11} fill="#f43f5e" />
    </Group>
  );
}

// Legend reused by the page header — matches the backend KIND_STYLE palette.
export function ScanMapLegend() {
  const items: [string, string][] = [
    ["扫描框", KIND_COLOR.frame],
    ["针尖", KIND_COLOR.tip],
    ["扫图", KIND_COLOR.scan],
    ["STS", KIND_COLOR.sts],
    ["电脉冲", KIND_COLOR.pulse],
    ["修针尖", KIND_COLOR.tip_shape],
    ["进针", KIND_COLOR.approach],
    ["撞针", KIND_COLOR.crash],
    ["粗动换区", KIND_COLOR.coarse_move],
    ["移动", KIND_COLOR.move],
    ["手动", KIND_COLOR.manual],
    ["计划", KIND_COLOR.plan],
  ];
  return (
    <div className="flex flex-wrap gap-3 text-xs text-mast-muted">
      {items.map(([label, color]) => (
        <span key={label} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color }} />
          {label}
        </span>
      ))}
    </div>
  );
}
