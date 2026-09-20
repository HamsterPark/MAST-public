// ════════════════════════════════════════════════════════════════════════════
// 粗动大地图（样品台尺度）的视图数学 —— 拟合、缩放、平移，以及 chrome 的反缩放。
//
// 为什么单独一个模块：`CoarseMapPanel.tsx` 画的是 SVG，压电那张
// (`ScanMapCanvas.tsx`) 画的是 Konva canvas。两边的缩放机制**不可能共用**：
// Konva 靠 Stage 的 scale/offset，SVG 靠 `viewBox`。共用的是那条纪律 ——
// **缩放时 chrome 必须反向补偿** —— 而纪律没法靠共用代码强制，只能靠共用的测试。
//
// ── 需求列表：这张图历经多次迭代 ──────────────────────────────────────
//
//   xy 粗动的大地图需要一个可视化                      → 做了
//   大地图应包含过去小地图的历史记录                    → 做了
//   扫描地图页也要能进入粗动大地图                      → 接线（只画在进不去的那页）
//   粗动地图需要缩放，原尺度过大看不清                  → 本模块
//   粗动地图占用空间过大，需要一个更紧凑的小地图        → 本模块
//
// 缩放与紧凑模式是**同一个修法**：能缩放，就不必靠尺寸换清晰度。所以先做缩放，
// 紧凑模式才不是「把信息砍掉」。
//
// ── 这个模块存在的真正理由：一个已经犯过两次的错 ────────────────────────
//
// 已知问题是「滚轮缩放的时候上面的字**不跟着缩放**」。SVG 里这件事更隐蔽：
// stroke-width 和 font-size 的单位是**用户单位**，而缩放 = 把 viewBox 缩小 k 倍
// ⇒ 同一个用户单位在屏幕上变粗 k 倍。也就是说，**什么都不做地加一个滚轮缩放，
// 放大到 10× 时字号会变成 10 倍**，与另一张图上已经出现过的问题完全同形。
//
// 所以本模块把尺寸分成两类，并且**给这个区分起了名字**（否则下一个人会把
// `/ k` 当成冗余删掉）：
//
//   · **data**   —— 有物理意义的量：站点坐标、不确定半径、压电量程方框、行程
//                   预算边界。它们**必须**跟着缩放，那正是放大要看的东西。
//   · **chrome** —— 只为「看得见」而存在的量：描边宽度、虚线段长、字号、
//                   标记点半径、不确定斑的**最小**可见半径、十字臂长。
//                   它们必须除以 k，否则放大等于把家具糊在脸上。
//
// `chromeAt()` 是这条纪律的唯一入口。测试 `test/coarseMapView.test.ts` 钉的
// 就是「data 不动 / chrome 除 k」这个分界本身。
// ════════════════════════════════════════════════════════════════════════════

/** 拟合时在数据包围盒外留的边距，占跨度的比例。 */
export const COARSE_PAD = 0.14;

/** 缩放下限 = 1：拟合视图就是最外层，再往外拉没有信息，只会把图缩成一个点。 */
export const COARSE_ZOOM_MIN = 1;

/**
 * 缩放上限。
 *
 * 40× 不是拍的：这张图的用途是「别回到已经去过的地方」，而站点间距的出厂值是
 * 100 步、不确定半径动辄几十步。拟合跨度通常是几百到几千步，放大 40× 之后可见
 * 跨度落在十几步的量级 —— **比一个站点的不确定斑还小**，再放大就只剩一片色块，
 * 没有可读的信息了。压电那张图用 25×，这里更大是因为粗动的跨度动态范围更宽。
 */
export const COARSE_ZOOM_MAX = 40;

export type CoarseSitePt = {
  x_steps: number;
  y_steps: number;
  uncertainty_steps: number;
  position_known?: boolean;
};

/** 拟合结果：一个以 (cx, cy) 为中心、半边长 `half` 的**正方形** stage 坐标系。 */
export type CoarseFit = {
  /** 数据中心（步）。stage 坐标 = 相对它的偏移。 */
  cx: number;
  cy: number;
  /** 数据跨度（步），不含 PAD。所有 chrome 的基准尺寸都由它派生。 */
  span: number;
  /** 视口半边长（步），含 PAD。k=1 时 viewBox 就是 [−half, +half]²。 */
  half: number;
};

/** 缩放/平移状态。`x`/`y` 是**视口中心**在 stage 坐标里的位置（步）。 */
export type CoarseZoom = { k: number; x: number; y: number };

export const COARSE_ZOOM_RESET: CoarseZoom = { k: 1, x: 0, y: 0 };

function num(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

/**
 * 由站点集拟合出正方形视口。
 *
 * `null` = 画不出来（没有站点）。**不要**在这里编一个默认视图：一张画着空坐标系
 * 的地图和一张画着「还没有站点」的地图，传达的信息完全不同。
 */
export function fitCoarse(
  sites: readonly CoarseSitePt[] | null | undefined,
  siteSpacingSteps?: number | null,
): CoarseFit | null {
  if (!sites || sites.length === 0) return null;
  // 位置已知的优先；一个都没有时退回全体（画出来但会被虚线标成不确定）。
  const known = sites.filter((s) => s.position_known);
  const pts = known.length ? known : sites;
  const xs = pts.map((s) => num(s.x_steps));
  const ys = pts.map((s) => num(s.y_steps));
  const rMax = Math.max(...pts.map((s) => num(s.uncertainty_steps)), 0);
  // 至少留出一个站点间距，否则单站点时整张图会缩成一个点。
  const margin = Math.max(rMax * 2, num(siteSpacingSteps, 0) || 100);
  const minX = Math.min(...xs) - margin;
  const maxX = Math.max(...xs) + margin;
  const minY = Math.min(...ys) - margin;
  const maxY = Math.max(...ys) + margin;
  const span = Math.max(maxX - minX, maxY - minY, 1);
  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    span,
    half: span / 2 + span * COARSE_PAD,
  };
}

/** 站点（步）→ stage 坐标。SVG 的 y 轴向下、样品坐标向上，所以 y 取反。 */
export function toStage(
  fit: CoarseFit,
  p: { x_steps: number; y_steps: number },
): { x: number; y: number } {
  return { x: num(p.x_steps) - fit.cx, y: -(num(p.y_steps) - fit.cy) };
}

/** 当前可见的半边长（步）。放大 k 倍 = 可见跨度缩小 k 倍。 */
export function visibleHalf(fit: CoarseFit, zoom: CoarseZoom): number {
  const k = clampK(zoom?.k);
  return fit.half / k;
}

function clampK(k: unknown): number {
  const v = num(k, 1);
  return Math.min(COARSE_ZOOM_MAX, Math.max(COARSE_ZOOM_MIN, v));
}

/**
 * 把视口中心夹回拟合框内，**不许把地图拖出视野**。
 *
 * 允许范围随 k 收紧：k=1 时 h=half，唯一合法的中心是 0（拟合视图本身）。
 * 这条也顺带保证了「缩回 1× 自动归位」不需要单独一段代码。
 */
export function clampPan(fit: CoarseFit, zoom: CoarseZoom): CoarseZoom {
  const k = clampK(zoom?.k);
  const h = fit.half / k;
  const lim = Math.max(0, fit.half - h);
  const cx = Math.min(lim, Math.max(-lim, num(zoom?.x)));
  const cy = Math.min(lim, Math.max(-lim, num(zoom?.y)));
  // `+ 0` 把 -0 归一成 0：对 SVG 无害，但把一张没缩放的图 dump 出来看到
  // "-0 -0 …" 会让人以为哪里错了。
  return { k, x: cx + 0, y: cy + 0 };
}

/**
 * SVG `viewBox` 字符串。
 *
 * ⚠️ k=1 且未平移时，它必须**逐字等于**加缩放之前那一版的公式
 * （`${-half} ${-half} ${2*half} ${2*half}`）—— 否则这次改动会在没人缩放的
 * 情况下悄悄改变默认视图。测试里钉了这一条。
 */
export function coarseViewBox(fit: CoarseFit, zoom: CoarseZoom = COARSE_ZOOM_RESET): string {
  const z = clampPan(fit, zoom);
  const h = fit.half / z.k;
  return `${z.x - h} ${z.y - h} ${2 * h} ${2 * h}`;
}

/**
 * 以光标为锚点缩放：**光标下的那个点在屏幕上不动**。
 *
 * `fx`/`fy` 是光标在 SVG 元素内的归一化位置（0..1，左上角为 0）。元素是正方形
 * 且 viewBox 也是正方形，所以这里是一个直的线性映射，不需要 `getScreenCTM`。
 *
 * 不做锚点会怎样：放大时视口始终围绕拟合中心收缩，用户想看的那个站点会往外
 * 跑，于是「放大」变成「先放大再找回来」——这正是解决「尺度太大看不清」时
 * 想要的那个动作的反面。
 */
export function zoomAtPoint(
  fit: CoarseFit,
  zoom: CoarseZoom,
  fx: number,
  fy: number,
  factor: number,
): CoarseZoom {
  const z = clampPan(fit, zoom);
  const kNext = clampK(z.k * num(factor, 1));
  if (kNext === z.k) return z;
  const h = fit.half / z.k;
  const hNext = fit.half / kNext;
  // 光标下的 stage 点（缩放前）
  const wx = z.x - h + num(fx) * 2 * h;
  const wy = z.y - h + num(fy) * 2 * h;
  // 缩放后让同一个 stage 点仍落在同一个归一化位置上
  return clampPan(fit, {
    k: kNext,
    x: wx - hNext * (2 * num(fx) - 1),
    y: wy - hNext * (2 * num(fy) - 1),
  });
}

/**
 * 拖动平移。`dfx`/`dfy` 是指针位移占元素边长的比例。
 *
 * 视口中心朝**指针的反方向**移动 —— 拖动的手感是「抓住地图拖」，不是「移动窗口」。
 */
export function panByFraction(
  fit: CoarseFit,
  zoom: CoarseZoom,
  dfx: number,
  dfy: number,
): CoarseZoom {
  const z = clampPan(fit, zoom);
  const h = fit.half / z.k;
  return clampPan(fit, {
    k: z.k,
    x: z.x - num(dfx) * 2 * h,
    y: z.y - num(dfy) * 2 * h,
  });
}

/**
 * **chrome 的唯一入口** —— 描边宽、字号、虚线段、最小可见半径都走它。
 *
 * `base` 用拟合跨度派生（例如 `fit.span / 300`），除以 k 之后，它在**屏幕上**的
 * 大小与缩放无关。data 类的量（站点坐标、不确定半径、压电方框、行程预算框）
 * **一律不走这里** —— 它们该跟着放大，那正是放大要看的东西。
 *
 * 这个函数短到看起来像可以内联。**不要内联**：它的价值不在计算，在于让
 * 「这个数属于哪一类」在每个调用点都写出来，并且让测试能一处钉住整条纪律。
 */
export function chromeAt(base: number, zoom: CoarseZoom | number): number {
  const k = clampK(typeof zoom === "number" ? zoom : zoom?.k);
  return num(base) / k;
}

/** 缩放是否偏离了拟合视图 —— 决定要不要显示「复位」按钮。 */
export function isZoomed(zoom: CoarseZoom | null | undefined): boolean {
  if (!zoom) return false;
  return clampK(zoom.k) > COARSE_ZOOM_MIN + 1e-9 || num(zoom.x) !== 0 || num(zoom.y) !== 0;
}

/**
 * 缩放倍率的显示文本。
 *
 * 整数倍不写小数（`4×` 而不是 `4.0×`）：这行字挨着地图，读的是量级不是精度。
 */
export function fmtZoom(zoom: CoarseZoom | number): string {
  const k = clampK(typeof zoom === "number" ? zoom : zoom?.k);
  return (k < 10 && k % 1 !== 0 ? k.toFixed(1) : String(Math.round(k))) + "×";
}
