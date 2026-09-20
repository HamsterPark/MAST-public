// ════════════════════════════════════════════════════════════════════════════
// 谱 ↔ 前后帧对照 — src/lib/gallery/context.ts
//
//     cd frontend && npm run test:unit
//
// 两类错误在屏幕上都「看起来对」：
//   * 角度符号反了，十字圈会画在镜像的位置（正角 = 顺时针）；
//   * 缩略图几何猜错——原版用 `sd === 'u' ? rall − rows : 0` 猜有效行的起点，有效行
//     不从扫描起点开始时十字圈整体上下偏移（设计 T5）。索引现在显式给 r0/r1。
// 最后一条钉住测试读后端 render.py 的源码：信息条高度两边各写一份，改一边不改另一边
// 时，画出来的点会沿 y 偏几十个像素而不报任何错。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import {
  STRIP_PX,
  anchorOf,
  fStart,
  firstStartAfter,
  framesTimeline,
  gridCorners,
  imgXY,
  initialNeighbours,
  inside,
  itemGeometry,
  lastBefore,
  overlayShapes,
  relation,
  seriesAnchorOf,
  spanOf,
  tEnd,
  toFrame,
} from "../src/lib/gallery/context.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mk = (o: Record<string, unknown>): any => ({ d: "R/1", fn: "x.sxm", p: "", pf: "", ad: "", th: "", ...o });
const near = (a: number, b: number, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

describe("时间线", () => {
  const fa = mk({ id: "a", k: "f", t: 100, mt: 200, acq: 100, rows: 256, rall: 256 });
  const fb = mk({ id: "b", k: "f", t: 250, mt: 400, acq: 150, rows: 256, rall: 256 });
  const dup = mk({ id: "c", k: "f", t: 250, mt: 401, dup: "b" });
  const nomt = mk({ id: "d", k: "f", t: 50 });
  const spec = mk({ id: "s", k: "s", t: 300, mt: 320 });

  it("framesTimeline：去掉重复保存与没有保存时刻的，按保存时刻排", () => {
    assert.deepEqual(framesTimeline([fb, spec, dup, nomt, fa]).map((f) => f.id), ["a", "b"]);
  });

  it("fStart：REC 不晚于保存就用 REC；中途停的按已扫行比例估", () => {
    assert.equal(fStart(fa), 100);
    const stopped = mk({ id: "e", k: "f", t: 900, mt: 800, acq: 200, rows: 64, rall: 256 });
    assert.equal(fStart(stopped), 800 - 200 * 0.25);
  });

  it("tEnd：谱用保存时刻，网格用结束时刻", () => {
    assert.equal(tEnd(spec), 320);
    assert.equal(tEnd(mk({ id: "g", k: "g", t: 1, t1: 99, mt: 100 })), 99);
  });

  it("前一张 = 谱开始前最后保存的；后一张跳过跨着谱的帧", () => {
    const frames = [fa, fb, mk({ id: "z", k: "f", t: 330, mt: 500, acq: 170, rows: 256, rall: 256 })];
    // 谱 300–320：a 在前；b 250 开始、400 保存 —— 跨着谱，不算「之后」；z 330 开始才是。
    assert.equal(lastBefore(frames, 300 + 1), 0);
    assert.equal(firstStartAfter(frames, 320), 2);
    assert.equal(lastBefore([], 5), -1);
    assert.equal(firstStartAfter(frames, 10_000), 3);
  });

  it("relation 三种情况", () => {
    assert.deepEqual(relation(fa, 279, 320), { rel: "prev", dt: -79, desc: "谱开始前 1 分 19 秒保存" });
    const later = mk({ id: "n", k: "f", t: 326, mt: 700 });
    assert.deepEqual(relation(later, 300, 320), { rel: "next", dt: 6, desc: "谱结束后 6 秒开始扫描" });
    const r = relation(fb, 300, 320);
    assert.equal(r.rel, "next");
    assert.ok(r.desc.startsWith("与谱时间交叠："));
    assert.ok(relation(fa, 279, 320, "系列").desc.startsWith("系列开始前"));
  });
});

describe("坐标换算（正角 = 扫描框顺时针）", () => {
  const frame = (ang: number) => mk({ id: "f", k: "f", cx: 0, cy: 0, w: 10, hn: 10, nx: 256, ny: 256, ang });

  it("0°：右为 u 增，上为 v 减", () => {
    const p = toFrame(frame(0), 1, 2);
    near(p.u, 0.6);
    near(p.v, 0.3);
  });

  it("+90°：扫描框顺时针转，实验室的右方在帧里是上方", () => {
    const right = toFrame(frame(90), 1, 0);
    near(right.u, 0.5);
    near(right.v, 0.4);
    const up = toFrame(frame(90), 0, 1);
    near(up.u, 0.4);
    near(up.v, 0.5);
  });

  it("−90°：实验室的右方在帧里是下方", () => {
    const right = toFrame(frame(-90), 1, 0);
    near(right.u, 0.5);
    near(right.v, 0.6);
  });

  it("高度用 hn；没有 hn 按像素比例推", () => {
    const tall = mk({ id: "t", k: "f", cx: 0, cy: 0, w: 10, hn: 20, ang: 0 });
    near(toFrame(tall, 0, 2).v, 0.4);
    const noHn = mk({ id: "t", k: "f", cx: 0, cy: 0, w: 10, nx: 100, ny: 200, ang: 0 });
    near(toFrame(noHn, 0, 2).v, 0.4);
  });

  it("网格四角用 w × hn，且与 toFrame 互逆", () => {
    const g = mk({ id: "g", k: "g", cx: 5, cy: -3, w: 4, hn: 2, ang: 30 });
    const corners = gridCorners(g);
    assert.equal(corners.length, 4);
    const f = mk({ id: "f", k: "f", cx: 5, cy: -3, w: 4, hn: 2, ang: 30 });
    const uv = corners.map(([x, y]) => toFrame(f, x, y));
    [[0, 1], [1, 1], [1, 0], [0, 0]].forEach(([u, v], i) => {
      near(uv[i]!.u, u!, 1e-9);
      near(uv[i]!.v, v!, 1e-9);
    });
  });

  it("inside", () => {
    assert.equal(inside({ u: 0, v: 1 }), true);
    assert.equal(inside({ u: -0.01, v: 0.5 }), false);
  });
});

describe("缩略图几何用 r0 / r1", () => {
  // 定向后第 100..228 行被画进缩略图；rows（整行有效的行数）是 120，扫描方向 up。
  // 原版的猜法会取起点 rall − rows = 136 —— 这一组数专门让猜法与真值分开。
  const f = mk({ id: "f", k: "f", rall: 256, ny: 256, rows: 120, r0: 100, r1: 228, sd: "u" });
  const W = 400;
  const H = 200 + STRIP_PX;

  it("显示区上沿 / 下沿 / 中点", () => {
    const [x0, yTop, hh] = imgXY(f, { u: 0.25, v: 100 / 256 }, W, H);
    near(x0, 100);
    near(yTop, 0);
    near(hh, 200);
    near(imgXY(f, { u: 0, v: 228 / 256 }, W, H)[1], 200);
    near(imgXY(f, { u: 0, v: 164 / 256 }, W, H)[1], 100);
  });

  it("没有 r0/r1 时整帧都画了", () => {
    const whole = mk({ id: "w", k: "f", rall: 256 });
    near(imgXY(whole, { u: 0, v: 0.5 }, W, H)[1], 100);
  });

  it("overlayShapes：单点视野外夹到边上并标 out", () => {
    const sh = overlayShapes(f, [{ u: 1.5, v: 0 }], null, W, H);
    assert.equal(sh.single, true);
    assert.equal(sh.points[0]!.out, true);
    assert.equal(sh.points[0]!.x, W - 4);
    assert.equal(sh.points[0]!.y, 4);
    assert.equal(sh.polyline, null);
  });

  it("overlayShapes：多点有折线；超过 20 个只给首尾编号", () => {
    const pts = Array.from({ length: 25 }, (_, i) => ({ u: 0.5, v: (100 + i) / 256 }));
    const sh = overlayShapes(f, pts, null, W, H);
    assert.ok(sh.polyline);
    const labels = sh.points.map((p) => p.label).filter(Boolean);
    assert.deepEqual(labels, ["1", "25"]);
    const few = overlayShapes(f, pts.slice(0, 3).map((p, i) => ({ ...p, lab: `#${i}` })), null, W, H);
    assert.deepEqual(few.points.map((p) => p.label), ["#0", "#1", "#2"]);
  });

  it("overlayShapes：网格多边形的 y 夹在图像区内", () => {
    const sh = overlayShapes(f, [{ u: 0.5, v: 0.5 }], [{ u: 0, v: 0 }, { u: 1, v: 0 }, { u: 1, v: 1 }, { u: 0, v: 1 }], W, H);
    const ys = sh.polygon!.split(" ").map((xy) => Number(xy.split(",")[1]));
    assert.ok(ys.every((y) => y >= 0 && y <= 200));
  });
});

describe("位置与系定", () => {
  const frame = mk({ id: "R/f_0498.sxm", fn: "f_0498.sxm", k: "f", cx: 0, cy: 0, w: 10, hn: 10, ang: 0, t: 330, mt: 500 });
  const spec = mk({ id: "R/rep6.dat", k: "s", t: 300, mt: 326, x: 1, y: -2 });

  it("谱是一个点，网格是中心点加四角", () => {
    assert.equal(itemGeometry(spec, frame).poly, null);
    const g = mk({ id: "g", k: "g", cx: 0, cy: 0, w: 2, hn: 2, ang: 0 });
    assert.equal(itemGeometry(g, frame).poly!.length, 4);
  });

  it("anchorOf 的字段与取整照原版", () => {
    const a = anchorOf(spec, frame);
    assert.deepEqual(a, { id: "R/f_0498.sxm", fn: "f_0498.sxm", rel: "next", dt: 4, desc: "谱结束后 4 秒开始扫描", u: 0.6, v: 0.7, inside: true });
    const s = seriesAnchorOf(frame, 100, 320);
    assert.equal(s.desc, "系列结束后 10 秒开始扫描");
    assert.equal("u" in s, false);
  });

  it("initialNeighbours：已系定的帧摆到它所在的一侧", () => {
    const frames = [
      mk({ id: "p1", k: "f", t: 10, mt: 50 }),
      mk({ id: "p2", k: "f", t: 60, mt: 90 }),
      mk({ id: "n1", k: "f", t: 330, mt: 400 }),
      mk({ id: "n2", k: "f", t: 410, mt: 480 }),
    ];
    assert.deepEqual(initialNeighbours(frames, 300, 320), { prev: 1, next: 2 });
    assert.deepEqual(initialNeighbours(frames, 300, 320, "p1"), { prev: 0, next: 2 });
    assert.deepEqual(initialNeighbours(frames, 300, 320, "n2"), { prev: 1, next: 3 });
    assert.deepEqual(initialNeighbours(frames, 300, 320, "not-there"), { prev: 1, next: 2 });
  });

  it("spanOf 只看谱与网格", () => {
    const f = mk({ id: "f", k: "f", t: 1, mt: 9999 });
    assert.deepEqual(spanOf([f, spec, mk({ id: "g", k: "g", t: 250, t1: 900, mt: 901 })]), { t0: 250, t1: 900 });
    assert.deepEqual(spanOf([f]), { t0: 0, t1: 0 });
  });
});

describe("信息条高度两边一致", () => {
  it("STRIP_PX 与 MASTv2/mast/gallery/render.py 相等", () => {
    // 不许写成「文件不存在就跳过」：那样后端删掉或改名这个常量时，这条闸门会永远绿。
    const HERE = fileURLToPath(new URL(".", import.meta.url));
    const src = readFileSync(join(HERE, "..", "..", "MASTv2", "mast", "gallery", "render.py"), "utf8");
    const m = src.match(/^STRIP_PX\s*=\s*(\d+)/m);
    assert.ok(m, "render.py 里找不到顶层的 STRIP_PX = <整数>");
    assert.equal(Number(m[1]), STRIP_PX);
  });
});
