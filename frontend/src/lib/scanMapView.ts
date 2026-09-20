// ════════════════════════════════════════════════════════════════════════════
// Scan-map view rules — which layers are drawn, and what area is in view.
//
// Pure functions, deliberately: the map grew to eight kinds of overlay on one
// 620×460 canvas (surface mosaic, keep-out discs, planned route, the route
// ahead, twelve marker kinds, grid, live frame, tip), and "which of those is
// visible" plus "what patch of surface is in view" are now decisions with
// enough edge cases to deserve tests. Konva rendering itself is not testable
// here (no DOM), so everything that CAN be decided in plain data is decided in
// this file and the component just draws the answer.
//
// No imports from ../api/schema on purpose: the structural types below are the
// small subset actually read, which keeps this module runnable under
// `node --test` (npm run test:unit) with no build step.
// ════════════════════════════════════════════════════════════════════════════

/** A drawable layer the operator can switch off. */
export type MapLayerId =
  | "underlay" // saved .sxm thumbnails placed by real stage footprint
  | "grid" // nm grid + axis ticks
  | "avoid" // keep-out discs from the analysis
  | "history" // executed markers in the LIVE coordinate generation
  | "stale" // executed markers from a superseded generation (faded)
  | "plan" // the published route (status="planned")
  | "upcoming" // the route the strategy would walk next (ghost frames)
  | "next"; // the single recommended next position

export type MapLayers = Record<MapLayerId, boolean>;

/** How much surface the canvas fits into view. */
export type MapViewMode =
  | "fit" // everything there is to draw (the historical behaviour)
  | "follow" // the live scan frame and its immediate surroundings
  | "full"; // the whole reachable piezo area

export const MAP_LAYER_IDS: MapLayerId[] = [
  "underlay", "grid", "avoid", "history", "stale", "plan", "upcoming", "next",
];

/** Human labels for the layer switches (kept next to the ids on purpose). */
export const MAP_LAYER_LABELS: Record<MapLayerId, string> = {
  underlay: "扫描底图",
  grid: "网格刻度",
  avoid: "避让区",
  history: "历史操作",
  stale: "旧坐标系",
  plan: "计划路线",
  upcoming: "候选序列",
  next: "建议位置",
};

/** One-line rationale shown under each switch. */
export const MAP_LAYER_HINTS: Record<MapLayerId, string> = {
  underlay: "已保存 .sxm 按真实舞台坐标铺底",
  grid: "nm 网格与坐标轴刻度",
  avoid: "破坏/污染区域，规划会绕开",
  history: "当前坐标系里已执行的操作",
  stale: "粗动前的旧坐标（淡显；关掉则完全隐藏）",
  plan: "已发布的计划路线，按执行顺序编号",
  upcoming: "当前策略接下来会走的位置",
  next: "分析给出的下一个推荐位置",
};

export const DEFAULT_LAYERS: MapLayers = {
  underlay: true, grid: true, avoid: true, history: true,
  stale: true, plan: true, upcoming: true, next: true,
};

export interface MapPreset {
  id: string;
  label: string;
  hint: string;
  layers: MapLayers;
}

/**
 * Preset layer sets.
 *
 * A preset is a one-shot setter, NOT a mode: clicking it writes these eight
 * booleans and then every switch stays independent. That is why there is no
 * "custom" state to store — {@link matchPreset} derives the highlight from the
 * booleans themselves, so the two controls can never disagree about what is
 * actually being drawn.
 */
export const MAP_PRESETS: MapPreset[] = [
  {
    id: "all",
    label: "全部",
    hint: "所有图层",
    layers: { ...DEFAULT_LAYERS },
  },
  {
    id: "minimal",
    label: "精简",
    hint: "只看表面与当前坐标系里做过什么",
    layers: {
      underlay: true, grid: false, avoid: false, history: true,
      stale: false, plan: false, upcoming: false, next: false,
    },
  },
  {
    id: "planning",
    label: "规划聚焦",
    hint: "去掉缩略图与旧坐标，突出去哪儿",
    layers: {
      underlay: false, grid: true, avoid: true, history: true,
      stale: false, plan: true, upcoming: true, next: true,
    },
  },
];

export const MAP_VIEW_MODES: { id: MapViewMode; label: string; hint: string }[] = [
  { id: "fit", label: "自动拟合", hint: "把所有内容装进画面" },
  { id: "follow", label: "跟随扫描框", hint: "只看当前扫描框附近" },
  { id: "full", label: "压电全域", hint: "整个可达范围，看清位置在量程的哪里" },
];

/**
 * Fill in any layer the stored value is missing.
 *
 * Persisted UI state outlives the code that wrote it: a layer added later would
 * otherwise arrive as `undefined` and read as OFF, so an operator who used the
 * map once would never see the new overlay and would have no way to know it
 * existed.
 */
export function withLayerDefaults(stored?: Partial<MapLayers> | null): MapLayers {
  const out = { ...DEFAULT_LAYERS };
  for (const id of MAP_LAYER_IDS) {
    const v = stored?.[id];
    if (typeof v === "boolean") out[id] = v;
  }
  return out;
}

/** The preset whose layer set is exactly this one, or null (= 自定义). */
export function matchPreset(layers: MapLayers): string | null {
  for (const p of MAP_PRESETS) {
    if (MAP_LAYER_IDS.every((id) => p.layers[id] === layers[id])) return p.id;
  }
  return null;
}

// ── analysis refresh ───────────────────────────────────────────────────────

/**
 * Smallest gap between two AUTOMATIC analysis calls.
 *
 * `analyze_map` rasterises the whole piezo area and walks the candidate route —
 * a ~1 s budget the backend deliberately keeps OFF the 3 s map poll. But the
 * overlays it feeds (keep-out discs, the recommendation, the route ahead) are
 * now layers the operator can leave switched on, and a layer that only updates
 * when someone remembers to press a button is a layer that quietly lies.
 *
 * 10 s resolves that: markers arrive in bursts (the manual-activity watcher
 * ticks at 1.5 s, an import writes many at once), and this collapses a burst
 * into at most one analysis per 10 s while still guaranteeing the LAST change
 * lands — see {@link nextAnalysisDelayMs}, which asks for a trailing call
 * rather than dropping it.
 */
export const MIN_AUTO_ANALYSIS_INTERVAL_MS = 10_000;

/**
 * How stale an analysis may get while its layers are on, even with no new
 * markers. Not everything it depends on produces one: the frame size comes from
 * the live scan width, so changing the scan size in Nanonis moves every
 * candidate position without touching the record.
 */
export const MAX_ANALYSIS_AGE_MS = 60_000;

interface MapLike {
  marker_count?: number | null;
  current_epoch?: number | null;
  degraded?: boolean | null;
  markers?: { status?: string | null }[] | null;
}

/**
 * A value that changes exactly when the analysis would come out different.
 *
 * Everything in it rides on the 3 s map poll that is happening anyway, so
 * watching for changes costs no extra request:
 *   • `marker_count` — an operation was recorded (executed markers only)
 *   • `current_epoch` — a coarse move invalidated every old coordinate
 *   • planned-step count — a route was published, cleared, or advanced
 */
export function analysisSignature(map?: MapLike | null): string {
  if (!map) return "";
  const planned = (map.markers ?? []).filter(
    (m) => (m?.status ?? "") === "planned",
  ).length;
  return `${map.marker_count ?? 0}|${map.current_epoch ?? 0}|${planned}`;
}

/** Whether any layer on screen actually needs the analysis payload. */
export function analysisAutoEnabled(layers: MapLayers): boolean {
  return layers.avoid || layers.next || layers.upcoming;
}

/**
 * Milliseconds to wait before the next automatic analysis: 0 = go now.
 *
 * Leading-edge with a trailing catch-up. A burst of markers fires this many
 * times; the first call goes immediately, the rest collapse into ONE delayed
 * call after the interval, so the operator ends up looking at the state after
 * the burst rather than the state in the middle of it.
 */
export function nextAnalysisDelayMs(
  now: number,
  lastFetchedAt: number | null | undefined,
  minIntervalMs: number = MIN_AUTO_ANALYSIS_INTERVAL_MS,
): number {
  if (lastFetchedAt == null || !Number.isFinite(lastFetchedAt)) return 0;
  const elapsed = now - lastFetchedAt;
  if (!Number.isFinite(elapsed) || elapsed >= minIntervalMs) return 0;
  // Clock stepped backwards (NTP correction, the operator fixing the system
  // time). The stored timestamp is now in the future, and subtracting it would
  // schedule the next analysis that far out — the overlays would sit frozen for
  // as long as the jump, which is exactly the silent stall 「UI 绝不冻结」
  // forbids. An unusable timestamp means "no idea when we last ran": go now.
  if (elapsed < 0) return 0;
  return Math.max(0, minIntervalMs - elapsed);
}

// ── view bounds ────────────────────────────────────────────────────────────

export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface BoundsResult {
  bounds: Bounds | null;
  /** True when the requested mode could not be honoured and fit was used. */
  fellBack: boolean;
}

interface XY {
  x_m?: number | null;
  y_m?: number | null;
  w_m?: number | null;
  h_m?: number | null;
}

interface FrameLike {
  center_x_m?: number | null;
  center_y_m?: number | null;
  width_m?: number | null;
  height_m?: number | null;
}

interface BoundsMap {
  frame?: FrameLike | null;
  tip_xyz?: { x_m?: number | null; y_m?: number | null } | null;
  markers?: ({ status?: string | null; coord_epoch?: number | null } & XY)[] | null;
  scan_images?: {
    center_x_m?: number | null; center_y_m?: number | null;
    width_m?: number | null; height_m?: number | null;
  }[] | null;
  current_epoch?: number | null;
}

interface BoundsAnalysis {
  avoid_zones?: { x_m: number; y_m: number; radius_m: number }[] | null;
  next_position?: { x_m?: number | null; y_m?: number | null } | null;
  upcoming?: { x_m?: number | null; y_m?: number | null }[] | null;
  frame_size_m?: number | null;
}

/** Half-width of the window that "follow" puts around the live scan frame:
 *  the frame plus enough context to see what is next to it. */
const FOLLOW_FRAME_FACTOR = 1.4;

/** Accumulates the metre-space extent of everything actually drawn. */
class Extent {
  readonly xs: number[] = [];
  readonly ys: number[] = [];

  add(x?: number | null, y?: number | null, w?: number | null, h?: number | null): void {
    if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) return;
    const hw = Math.abs(w ?? 0) / 2;
    const hh = Math.abs(h ?? 0) / 2;
    this.xs.push(x - hw, x + hw);
    this.ys.push(y - hh, y + hh);
  }

  bounds(): Bounds | null {
    if (!this.xs.length || !this.ys.length) return null;
    return {
      minX: Math.min(...this.xs), maxX: Math.max(...this.xs),
      minY: Math.min(...this.ys), maxY: Math.max(...this.ys),
    };
  }
}

/** True when this marker's coordinate predates a lateral coarse move. */
export function isStaleEpoch(
  m: { coord_epoch?: number | null },
  currentEpoch?: number | null,
): boolean {
  return currentEpoch != null && m.coord_epoch != null && m.coord_epoch < currentEpoch;
}

/**
 * The metre-space box to fit into the canvas, for the requested mode.
 *
 * Only what is actually DRAWN is fitted — a switched-off layer must not keep
 * pushing the view open around content nobody can see, which is the whole
 * reason 「精简」 is a usable escape from a crowded map.
 */
export function boundsForMode(
  mode: MapViewMode,
  map: BoundsMap | null | undefined,
  analysis: BoundsAnalysis | null | undefined,
  layers: MapLayers,
  piezoHalfM?: number | null,
): BoundsResult {
  if (mode === "full") {
    const half = Number(piezoHalfM ?? 0);
    if (Number.isFinite(half) && half > 0) {
      return { bounds: { minX: -half, maxX: half, minY: -half, maxY: half }, fellBack: false };
    }
    // No range to show: fitting is better than an empty canvas, but say so.
    return { ...fitBounds(map, analysis, layers), fellBack: true };
  }

  if (mode === "follow") {
    const f = map?.frame;
    const cx = f?.center_x_m;
    const cy = f?.center_y_m;
    const w = Math.abs(Number(f?.width_m ?? 0));
    const h = Math.abs(Number(f?.height_m ?? 0));
    const span = Math.max(w, h);
    if (cx != null && cy != null && Number.isFinite(cx) && Number.isFinite(cy) && span > 0) {
      const half = (span * FOLLOW_FRAME_FACTOR) / 2;
      return {
        bounds: { minX: cx - half, maxX: cx + half, minY: cy - half, maxY: cy + half },
        fellBack: false,
      };
    }
    // Nothing is being scanned right now — there is no frame to follow.
    return { ...fitBounds(map, analysis, layers), fellBack: true };
  }

  return fitBounds(map, analysis, layers);
}

function fitBounds(
  map: BoundsMap | null | undefined,
  analysis: BoundsAnalysis | null | undefined,
  layers: MapLayers,
): BoundsResult {
  const ext = new Extent();

  const f = map?.frame;
  if (f) ext.add(f.center_x_m, f.center_y_m, f.width_m, f.height_m);
  const t = map?.tip_xyz;
  if (t) ext.add(t.x_m, t.y_m, 0, 0);

  for (const m of map?.markers ?? []) {
    const planned = (m?.status ?? "") === "planned";
    if (planned) {
      if (!layers.plan) continue;
    } else if (isStaleEpoch(m, map?.current_epoch)) {
      if (!layers.stale) continue;
    } else if (!layers.history) {
      continue;
    }
    ext.add(m.x_m, m.y_m, m.w_m, m.h_m);
  }

  if (layers.underlay) {
    for (const s of map?.scan_images ?? []) {
      ext.add(s.center_x_m, s.center_y_m, s.width_m, s.height_m);
    }
  }

  if (analysis) {
    // Keep-out discs and the recommendation must be inside the fitted view, or
    // the overlay silently points off-canvas. NOT piezo_half_range_m: fitting
    // the whole ±1.5 µm range would collapse every real nanometre-scale marker
    // — that is what the explicit 「压电全域」 mode is for.
    if (layers.avoid) {
      for (const z of analysis.avoid_zones ?? []) {
        ext.add(z.x_m, z.y_m, z.radius_m * 2, z.radius_m * 2);
      }
    }
    const size = analysis.frame_size_m ?? 0;
    if (layers.next) {
      const np = analysis.next_position;
      if (np) ext.add(np.x_m, np.y_m, size, size);
    }
    if (layers.upcoming) {
      for (const p of analysis.upcoming ?? []) ext.add(p.x_m, p.y_m, size, size);
    }
  }

  return { bounds: ext.bounds(), fellBack: false };
}

// ── map chrome under wheel zoom ────────────────────────────────────────────
//
// The map's textual chrome — axis tick numbers, the "X nm" axis label, the nm
// scale bar and its caption — was laid out in STAGE coordinates pinned to the
// canvas edges (`y = CANVAS_H - 16`, `x = 4`, `x = PAD`). The wheel zoom is a
// transform on the Stage, so those coordinates are scaled and translated with
// everything else: the moment the operator zooms in to read detail, every
// number that told them the scale slides off the canvas
// (「滚轮缩放的时候上面的字不跟着缩放」— operator , 2026-08-03).
//
// Chrome is viewport furniture, not surface geometry. These two helpers are
// what the component needs to keep it that way, and they are pure so the
// arithmetic is checked here rather than by eye at 25×.

/** "Nice" round nm step for the grid / scale bar given a visible span in nm. */
export function niceStepNm(spanNm: number): number {
  const raw = spanNm / 5;
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-6))));
  const norm = raw / pow;
  const nice = norm >= 5 ? 5 : norm >= 2 ? 2 : 1;
  return nice * pow;
}

/**
 * Grid / scale-bar step for what is ACTUALLY visible at zoom `k`.
 *
 * At k=1 this is `niceStepNm(spanNm)`, unchanged. Zoomed in, the visible span
 * is `spanNm / k`, so the step shrinks with it — which is what keeps the scale
 * bar a constant fraction of the canvas instead of growing k× until it runs off
 * the edge, and what makes the grid subdivide as the operator zooms in rather
 * than staying at the whole-map spacing.
 */
export function visibleStepNm(spanNm: number, zoomK: number): number {
  const k = Number.isFinite(zoomK) && zoomK > 0 ? zoomK : 1;
  return niceStepNm(spanNm / k);
}

/**
 * Layer transform that CANCELS a stage transform, so a layer carrying it is
 * pinned to the viewport however the stage is zoomed and panned.
 *
 * Konva composes stage∘layer, so with stage `p ↦ k·p + t` this layer's
 * `p ↦ p/k − t/k` composes to the identity. Cheaper and far less error-prone
 * than counter-scaling every Text node and recomputing its anchor.
 */
export function inverseStageTransform(zoom: { k: number; x: number; y: number }): {
  scaleX: number;
  scaleY: number;
  x: number;
  y: number;
} {
  const k = Number.isFinite(zoom?.k) && zoom.k > 0 ? zoom.k : 1;
  const tx = Number.isFinite(zoom?.x) ? zoom.x : 0;
  const ty = Number.isFinite(zoom?.y) ? zoom.y : 0;
  // `+` normalises -0 → 0: harmless to Konva, but a -0 offset on an un-zoomed
  // map is a confusing thing to read back out of a debug dump.
  return { scaleX: 1 / k, scaleY: 1 / k, x: +(-tx / k) + 0, y: +(-ty / k) + 0 };
}

/**
 * Konva props that keep a Text anchored to a WORLD feature while rendering at a
 * constant SCREEN size under stage zoom `k`.
 *
 * Why this exists on top of {@link inverseStageTransform} (2026-08-05). #52
 * moved the map's GLOBAL chrome — axis ticks, the scale bar — into an
 * inverse-transformed layer, and that was right for furniture pinned to the
 * canvas edges. But five labels belong to a moving feature and therefore stayed
 * in the zoomed world layer: 「针尖」, 「建议下一个位置」, 「压电范围 ±… µm」, and
 * the plan / candidate step numbers. Those scale with the stage, so at the 25×
 * the zoom allows a 10 px caption renders 250 px tall and buries the surface it
 * was labelling. Same complaint, other half of it: captions again failed to
 * stay put under wheel zoom.
 *
 * The whole layer cannot simply be inverted here — that would unpin the labels
 * from the features they name. So the anchor stays in world space and only the
 * glyphs are counter-scaled:
 *
 *   • `dxWorld/dyWorld` — offsets that SHOULD grow with zoom (half a frame
 *     width: the label must clear the frame however big it is drawn);
 *   • `dxPx/dyPx` — the visual gap between feature and caption, which should
 *     look the same at 1× and 25×, so it is divided by `k`.
 *
 * Pure, so the arithmetic is checked here rather than by eye at 25×.
 */
export function pinnedLabel(
  anchor: { x: number; y: number },
  dxPx: number,
  dyPx: number,
  zoomK: number,
): { x: number; y: number; scaleX: number; scaleY: number } {
  const k = Number.isFinite(zoomK) && zoomK > 0 ? zoomK : 1;
  return {
    x: anchor.x + dxPx / k,
    y: anchor.y + dyPx / k,
    scaleX: 1 / k,
    scaleY: 1 / k,
  };
}

/** Where a stage-space point lands on screen under the stage transform. Used to
 *  anchor viewport-fixed chrome (tick labels) to moving world features. */
export function stageToScreen(
  p: { x: number; y: number },
  zoom: { k: number; x: number; y: number },
): { x: number; y: number } {
  const k = Number.isFinite(zoom?.k) && zoom.k > 0 ? zoom.k : 1;
  const tx = Number.isFinite(zoom?.x) ? zoom.x : 0;
  const ty = Number.isFinite(zoom?.y) ? zoom.y : 0;
  return { x: p.x * k + tx, y: p.y * k + ty };
}

/**
 * Format a nanometre value for an axis tick / scale caption, at a precision the
 * grid step actually justifies.
 *
 * The old code was `Math.round(nm)`. That was fine while the step was derived
 * from the whole-map span, but once it follows the zoom (visibleStepNm) a
 * zoomed-in map has sub-nanometre steps — and rounding then prints "0" for
 * every tick and "0 nm" on the scale bar. Decimals come from the step, so the
 * label is never more precise than the thing it is labelling.
 */
export function fmtNm(valueNm: number, stepNm: number): string {
  if (!Number.isFinite(valueNm)) return "";
  const step = Number.isFinite(stepNm) && stepNm > 0 ? stepNm : 1;
  const decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(step))));
  // `+` normalises -0 to 0 and drops trailing zeros ("1.50" → "1.5").
  return String(+(valueNm + 0).toFixed(decimals));
}

// ── marker detail (「扫描地图点击标记弹出具体信息、时间等」) ────

/** The subset of `MapMarkerView` this needs. Structural, so tests need no API types. */
export interface MarkerLike {
  kind?: string | null;
  label?: string | null;
  skill_name?: string | null;
  status?: string | null;
  source?: string | null;
  timestamp?: string | null;
  coord_epoch?: number | null;
  x_m?: number | null;
  y_m?: number | null;
  w_m?: number | null;
  h_m?: number | null;
  angle_deg?: number | null;
}

export interface DetailRow {
  label: string;
  value: string;
  /** Render in the mono/tabular face — coordinates and sizes. */
  mono?: boolean;
}

function nm(v: number | null | undefined): string | null {
  if (v == null || !Number.isFinite(v)) return null;
  return `${(v * 1e9).toFixed(2)} nm`;
}

/**
 * Rows for the marker popup.
 *
 * Why a popup at all: the hover tooltip is one line capped at 200 px with
 * `wrap="none"`, so it can carry a label and a clock time and nothing else.
 * Everything the marker actually knows — which skill put it there, whether it
 * succeeded, how big the frame was, which coordinate generation it belongs to
 * — had no way to reach the screen.
 *
 * Omits what it does not have rather than printing a placeholder: a marker with
 * no size is a point operation (a pulse, a spectrum), not a zero-sized scan,
 * and a row reading「尺寸 0 × 0」 would be a statement that is false.
 */
export function markerDetailRows(
  m: MarkerLike,
  currentEpoch?: number | null,
): DetailRow[] {
  const rows: DetailRow[] = [];
  if (m.timestamp) rows.push({ label: "时间", value: m.timestamp.replace("T", " "), mono: true });
  if (m.skill_name) rows.push({ label: "技能", value: m.skill_name, mono: true });
  if (m.status) rows.push({ label: "状态", value: statusLabelOf(m.status) });
  const x = nm(m.x_m);
  const y = nm(m.y_m);
  if (x && y) rows.push({ label: "位置", value: `X ${x} · Y ${y}`, mono: true });
  const w = nm(m.w_m);
  const h = nm(m.h_m);
  if (w && h) rows.push({ label: "尺寸", value: `${w} × ${h}`, mono: true });
  if (m.angle_deg != null && Number.isFinite(m.angle_deg) && m.angle_deg !== 0) {
    rows.push({ label: "角度", value: `${m.angle_deg.toFixed(1)}°`, mono: true });
  }
  if (m.coord_epoch != null) {
    // Stale epochs are the one thing here that changes what the operator should
    // DO: the stage has moved since, so these coordinates no longer point at
    // the place the operation happened. Say it in words, not just by fading.
    const stale = currentEpoch != null && m.coord_epoch !== currentEpoch;
    rows.push({
      label: "坐标代次",
      value: `#${m.coord_epoch}${stale ? "（粗动前的旧坐标系，位置已不再对应）" : ""}`,
    });
  }
  if (m.source) rows.push({ label: "来源", value: m.source });
  return rows;
}

function statusLabelOf(s: string): string {
  if (s === "planned") return "计划中";
  if (s === "failed") return "失败";
  if (s === "done") return "已完成";
  return s;
}
