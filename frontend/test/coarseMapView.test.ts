// ════════════════════════════════════════════════════════════════════════════
// src/lib/coarseMapView.ts 的纯逻辑测试 —— 粗动大地图的缩放。
//
// 与 scanMapView.test.ts 同一套家规：本前端没有装测试框架，加一个不是我们的
// 决定，所以跑在 `node --test` + Node 原生 TS 剥离上：
//
//     cd frontend && npm run test:unit
//
// ── 这个文件真正在守什么 ────────────────────────────────────────────────
//
// 这张图的同一类问题反复出现（#65 → #66 → #92 → #95/#96）：
// 缺少缩放、尺度悬殊导致看不清，且占用空间过大。这两条是同一个修法：
// **能缩放，就不必靠尺寸换清晰度。**
//
// 而加缩放本身带着一个这个仓已经犯过两次的错：「滚轮缩放的时候上面的字
// 不跟着缩放」。SVG 里它更隐蔽 —— stroke/font 的单位是用户单位，缩小 viewBox
// 会让它们在屏幕上**变大 k 倍**。也就是说「什么都不做地加个滚轮缩放」不是
// 中性的，它会把 #52 在另一张图上原样重发。
//
// 所以本文件的核心是 `test_chrome_shrinks_and_data_does_not`：它钉的不是一个
// 数值，是**「哪些量属于 chrome」这个区分本身**。没有它，`chromeAt` 里那个
// `/ k` 看起来就是一行可以删掉的冗余。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  COARSE_PAD,
  COARSE_ZOOM_MAX,
  COARSE_ZOOM_MIN,
  COARSE_ZOOM_RESET,
  type CoarseSitePt,
  chromeAt,
  clampPan,
  coarseViewBox,
  fitCoarse,
  fmtZoom,
  isZoomed,
  panByFraction,
  toStage,
  visibleHalf,
  zoomAtPoint,
} from "../src/lib/coarseMapView.ts";

const SITES: CoarseSitePt[] = [
  { x_steps: 0, y_steps: 0, uncertainty_steps: 20, position_known: true },
  { x_steps: 600, y_steps: 0, uncertainty_steps: 35, position_known: true },
  { x_steps: 600, y_steps: 400, uncertainty_steps: 50, position_known: true },
];

function fitOrThrow(sites = SITES, spacing = 100) {
  const f = fitCoarse(sites, spacing);
  assert.ok(f, "fitCoarse 应该能拟合出视图");
  return f;
}

// ── 拟合 ───────────────────────────────────────────────────────────────────

describe("fitCoarse", () => {
  it("没有站点就说没有，不编一个空坐标系", () => {
    // 一张画着空坐标轴的地图和一张写着「还没有站点」的地图，传达的信息不同。
    assert.equal(fitCoarse([], 100), null);
    assert.equal(fitCoarse(null, 100), null);
    assert.equal(fitCoarse(undefined, 100), null);
  });

  it("视口是正方形 —— 否则模糊斑会被拉成椭圆", () => {
    // 数据是 600×400 的矩形；两轴比例必须一致，不确定半径才画得成圆。
    const f = fitOrThrow();
    const vb = coarseViewBox(f).split(" ").map(Number);
    assert.equal(vb[2], vb[3], "viewBox 的宽高不等 —— 圆会被画成椭圆");
  });

  it("单个站点不会缩成一个点", () => {
    const f = fitCoarse([{ x_steps: 12, y_steps: -7, uncertainty_steps: 0 }], 100);
    assert.ok(f);
    assert.ok(f.span >= 200, `单站点跨度 ${f.span} 太小 —— 至少要留出一个站点间距`);
  });

  it("站点间距读不到时用 100 步兜底，而不是 0", () => {
    // 0 会让 margin 变成 0 —— 单站点又缩成一个点。
    const f = fitCoarse([{ x_steps: 0, y_steps: 0, uncertainty_steps: 0 }], 0);
    assert.ok(f);
    assert.ok(f.span >= 200);
  });

  it("位置已知的站点优先决定视野；一个都没有时退回全体", () => {
    const mixed: CoarseSitePt[] = [
      { x_steps: 0, y_steps: 0, uncertainty_steps: 10, position_known: true },
      { x_steps: 99999, y_steps: 0, uncertainty_steps: 10, position_known: false },
    ];
    const f = fitOrThrow(mixed);
    assert.ok(f.span < 10000, "里程表失效的站点把视野拉走了 —— 它的坐标本来就不可信");

    const allUnknown = mixed.map((s) => ({ ...s, position_known: false }));
    const g = fitCoarse(allUnknown, 100);
    assert.ok(g && g.span > 10000, "全都位置未知时应该退回全体，而不是画不出来");
  });
});

// ── 回归钉：不缩放时视图必须和加缩放之前逐字一致 ─────────────────────────

describe("k=1 时的默认视图", () => {
  it("viewBox 逐字等于加缩放之前的那个公式", () => {
    // 这一条守的是「这次改动在没人缩放时是中性的」。
    // 旧代码（CoarseMapPanel.tsx 内联）：
    //     const lo = -span/2 - span*PAD, hi = span/2 + span*PAD
    //     viewBox = `${lo} ${lo} ${hi-lo} ${hi-lo}`
    const f = fitOrThrow();
    const lo = -f.span / 2 - f.span * COARSE_PAD;
    const hi = f.span / 2 + f.span * COARSE_PAD;
    assert.equal(coarseViewBox(f), `${lo} ${lo} ${hi - lo} ${hi - lo}`);
    assert.equal(coarseViewBox(f, COARSE_ZOOM_RESET), `${lo} ${lo} ${hi - lo} ${hi - lo}`);
  });

  it("chromeAt 在 k=1 时是恒等 —— 原来的尺寸一个都不变", () => {
    const f = fitOrThrow();
    for (const denom of [34, 60, 80, 90, 160, 300, 400]) {
      assert.equal(chromeAt(f.span / denom, COARSE_ZOOM_RESET), f.span / denom);
    }
  });
});

// ── 核心：chrome 缩、data 不缩 ─────────────────────────────────────────────

describe("chrome 与 data 的分界（#52 的教训）", () => {
  it("chrome 随缩放变小，data 原样不动", () => {
    // 屏幕上的表观大小 = 用户单位 × k（因为 viewBox 缩小了 k 倍）。
    // 所以 chrome 要保持屏幕尺寸不变，用户单位就必须 ÷ k。
    const f = fitOrThrow();
    const base = f.span / 300; // 一个典型的描边宽度
    for (const k of [1, 2, 5, 10, 40]) {
      const drawn = chromeAt(base, { k, x: 0, y: 0 });
      assert.equal(drawn, base / k);
      // 表观大小 = drawn × k —— 必须与 k 无关，这才是这条纪律的意思。
      assert.ok(
        Math.abs(drawn * k - base) < 1e-12,
        `k=${k} 时 chrome 的屏幕尺寸变了 —— 这正是「字不跟着缩放」那条反馈`,
      );
    }
  });

  it("放大 10× 而不补偿，字号会变成 10 倍 —— 记下这个反例", () => {
    // 这条不测产品代码，它测的是「为什么需要 chromeAt」。
    // 如果哪天有人把 `/ k` 删掉，上一条会红；这一条解释红的是什么。
    const f = fitOrThrow();
    const fontUnits = f.span / 34;
    const naiveApparent = fontUnits * 10; // 不补偿
    const fixedApparent = chromeAt(fontUnits, { k: 10, x: 0, y: 0 }) * 10;
    assert.ok(naiveApparent > fixedApparent * 9, "不补偿的字号应该大出接近一个数量级");
    assert.equal(fixedApparent, fontUnits);
  });

  it("站点坐标是 data —— 缩放不改变它", () => {
    // toStage 不接受 zoom 参数，这条钉的就是「它不该接受」。
    const f = fitOrThrow();
    const p = toStage(f, { x_steps: 600, y_steps: 400 });
    assert.equal(p.x, 600 - f.cx);
    assert.equal(p.y, -(400 - f.cy), "SVG 的 y 轴向下，样品坐标向上 —— 必须取反");
  });
});

// ── 缩放交互 ───────────────────────────────────────────────────────────────

describe("以光标为锚点的缩放", () => {
  /** 归一化位置 (fx,fy) 对应的 stage 点 —— 测试自己算一遍，不复用产品代码。 */
  function pointUnder(f: ReturnType<typeof fitOrThrow>, z: { k: number; x: number; y: number },
                      fx: number, fy: number) {
    const h = f.half / z.k;
    return { x: z.x - h + fx * 2 * h, y: z.y - h + fy * 2 * h };
  }

  it("光标下的那个点在缩放前后不动", () => {
    const f = fitOrThrow();
    // 取一个偏离中心的锚点；居中的锚点连错误实现都能通过。
    const fx = 0.25, fy = 0.7;
    let z = { ...COARSE_ZOOM_RESET };
    const before = pointUnder(f, z, fx, fy);
    z = zoomAtPoint(f, z, fx, fy, 1.15 ** 6);
    assert.ok(z.k > 2, `应该真的放大了，实际 k=${z.k}`);
    const after = pointUnder(f, z, fx, fy);
    assert.ok(
      Math.abs(after.x - before.x) < 1e-6 && Math.abs(after.y - before.y) < 1e-6,
      `锚点漂了: ${JSON.stringify(before)} → ${JSON.stringify(after)}`,
    );
  });

  it("倍率夹在 [1, 40] —— 缩不出拟合视图，也放不成一片色块", () => {
    const f = fitOrThrow();
    let z = { ...COARSE_ZOOM_RESET };
    for (let i = 0; i < 100; i++) z = zoomAtPoint(f, z, 0.5, 0.5, 1.15);
    assert.equal(z.k, COARSE_ZOOM_MAX);
    for (let i = 0; i < 200; i++) z = zoomAtPoint(f, z, 0.5, 0.5, 1 / 1.15);
    assert.equal(z.k, COARSE_ZOOM_MIN);
  });

  it("缩回 1× 时视口自动回到拟合中心", () => {
    // 不靠一段单独的「if k===1 then reset」，靠 clampPan 的边界自然得到。
    const f = fitOrThrow();
    let z = zoomAtPoint(f, COARSE_ZOOM_RESET, 0.1, 0.1, 8);
    z = panByFraction(f, z, 0.4, -0.3);
    assert.ok(z.x !== 0 || z.y !== 0, "先要真的平移出去");
    for (let i = 0; i < 100; i++) z = zoomAtPoint(f, z, 0.5, 0.5, 1 / 1.15);
    assert.equal(z.k, 1);
    assert.equal(z.x, 0);
    assert.equal(z.y, 0);
  });

  it("到顶之后再滚不会改变状态（同一个对象语义上的空操作）", () => {
    const f = fitOrThrow();
    let z = { ...COARSE_ZOOM_RESET };
    for (let i = 0; i < 100; i++) z = zoomAtPoint(f, z, 0.3, 0.3, 1.15);
    const again = zoomAtPoint(f, z, 0.3, 0.3, 1.15);
    assert.deepEqual(again, z);
  });
});

describe("平移", () => {
  it("往右拖，看到的是地图左边的内容", () => {
    // 手感是「抓住地图拖」：视口中心朝指针的反方向走。
    const f = fitOrThrow();
    const z0 = zoomAtPoint(f, COARSE_ZOOM_RESET, 0.5, 0.5, 8);
    const z1 = panByFraction(f, z0, 0.1, 0);
    assert.ok(z1.x < z0.x, "视口中心应该朝反方向移动");
  });

  it("拖不出拟合框 —— 地图不会被拖到看不见", () => {
    const f = fitOrThrow();
    let z = zoomAtPoint(f, COARSE_ZOOM_RESET, 0.5, 0.5, 4);
    for (let i = 0; i < 50; i++) z = panByFraction(f, z, 0.5, 0.5);
    const lim = f.half - f.half / z.k;
    assert.ok(Math.abs(z.x) <= lim + 1e-9, `x 越界: ${z.x} > ${lim}`);
    assert.ok(Math.abs(z.y) <= lim + 1e-9, `y 越界: ${z.y} > ${lim}`);
  });

  it("k=1 时平移是空操作 —— 拟合视图没有可平移的余地", () => {
    const f = fitOrThrow();
    const z = panByFraction(f, COARSE_ZOOM_RESET, 0.4, 0.4);
    assert.deepEqual(z, { k: 1, x: 0, y: 0 });
  });

  it("可见跨度确实随 k 收缩", () => {
    const f = fitOrThrow();
    assert.equal(visibleHalf(f, COARSE_ZOOM_RESET), f.half);
    assert.equal(visibleHalf(f, { k: 4, x: 0, y: 0 }), f.half / 4);
  });
});

// ── 显示辅助 ───────────────────────────────────────────────────────────────

describe("isZoomed / fmtZoom", () => {
  it("只有偏离拟合视图才算「缩放中」", () => {
    assert.equal(isZoomed(COARSE_ZOOM_RESET), false);
    assert.equal(isZoomed(null), false);
    assert.equal(isZoomed({ k: 2, x: 0, y: 0 }), true);
    // 倍率没变但平移了，也要给复位按钮 —— 否则操作员只能靠反复滚回去。
    assert.equal(isZoomed({ k: 1, x: 5, y: 0 }), true);
  });

  it("整数倍不写小数", () => {
    assert.equal(fmtZoom(1), "1×");
    assert.equal(fmtZoom(4), "4×");
    assert.equal(fmtZoom(2.5), "2.5×");
    assert.equal(fmtZoom(40), "40×");
    assert.equal(fmtZoom({ k: 12.4, x: 0, y: 0 }), "12×");
  });
});

// ── 垃圾输入 ───────────────────────────────────────────────────────────────

describe("坏数据不产生 NaN 的 viewBox", () => {
  it("NaN / undefined 的缩放状态退回拟合视图", () => {
    const f = fitOrThrow();
    // 一个含 NaN 的 viewBox 会让整张 SVG 消失 —— 静默，且没有任何报错。
    for (const bad of [
      { k: NaN, x: 0, y: 0 },
      { k: 4, x: NaN, y: 0 },
      { k: Infinity, x: 0, y: 0 },
      undefined as never,
    ]) {
      const vb = coarseViewBox(f, bad);
      assert.ok(!vb.includes("NaN"), `viewBox 里出现了 NaN: ${vb}`);
      assert.ok(!vb.includes("Infinity"), `viewBox 里出现了 Infinity: ${vb}`);
    }
  });

  it("站点坐标里的 NaN 不会污染拟合", () => {
    const f = fitCoarse(
      [
        { x_steps: 0, y_steps: 0, uncertainty_steps: 10, position_known: true },
        { x_steps: NaN as number, y_steps: 0, uncertainty_steps: 10, position_known: true },
      ],
      100,
    );
    assert.ok(f);
    assert.ok(Number.isFinite(f.span) && Number.isFinite(f.cx) && Number.isFinite(f.half));
    assert.ok(!coarseViewBox(f).includes("NaN"));
  });

  it("clampPan 不吐 -0", () => {
    const f = fitOrThrow();
    const z = clampPan(f, { k: 1, x: -0, y: -0 });
    assert.ok(Object.is(z.x, 0), "x 是 -0");
    assert.ok(Object.is(z.y, 0), "y 是 -0");
  });
});
