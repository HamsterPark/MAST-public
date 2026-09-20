/**
 * 结构闸门：**两张地图一起画**，以及没有 tab 的页至少要有一个链接。
 *
 * ── 为什么需要这一条 ──
 * 已知问题：扫描地图缺一张粗动大地图。查下来它一直存在，而且是全链路的
 * （`mast/io/coarse_map.py` → `GET /api/coarse-map` → `CoarseMapPanel`），
 * 只是**只画在 `/vision` 上**。而 `/vision` 在顶栏里没有位置，且在这一批之前，
 * 全仓渲染出来的链接里没有一个指向它 —— 唯一进得去的那张「扫描地图」是对话页
 * 「视觉缓冲」里的那张，那张只有压电尺度的一张图。
 *
 * 这是本仓第二次栽在同一个形状上（第一次是仪器初始化页）：
 * 路由解析正常、组件渲染正常、typecheck 全绿、单页截图完美 —— 唯一的症状是一个
 * 找不到它的人，而他的措辞永远不会是「导航里少了一项」。
 *
 * ── 判据 ──
 * 1. 谁画了压电尺度的扫描地图，谁就得画样品尺度的粗动地图。两张图刻意分开画
 *    （单位不同、代次范围不同），但**必须同时出现**：尺度差本身就是「换区不能
 *    靠压电」那句话的证据，只剩一张时它就消失了。
 * 2. `nav.ts` 里每一个 INTENTIONALLY_UNLISTED 的路径，都要有一个**渲染得出来**
 *    的链接指向它 —— 「没有 tab」可以是个决定，「没有任何入口」不是。
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import { INTENTIONALLY_UNLISTED } from "../src/lib/nav.ts";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) sourceFiles(p, out);
    else if (/\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/** 只看**渲染**出来的东西：注释里提到一个组件不等于画了它。 */
function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .split("\n")
    .map((line) => {
      const i = line.search(/(^|[^:])\/\//);
      return i === -1 ? line : line.slice(0, i);
    })
    .join("\n");
}

const FILES = sourceFiles(SRC).map((p) => {
  const text = readFileSync(p, "utf8");
  return {
    rel: relative(SRC, p).replace(/\\/g, "/"),
    text,
    code: stripComments(text),
  };
});

const SCAN = "<ScanMapPanel";
const COARSE = "<CoarseMapPanel";

/** 画压电尺度扫描地图的文件（组件自己的定义文件不算调用方）。 */
const SCAN_SITES = FILES.filter(
  (f) => f.code.includes(SCAN) && !f.rel.startsWith("components/vision/ScanMapPanel"),
);

describe("两张地图一起画 ", () => {
  // 闸门自检。一个匹配不到任何东西的判据永远是绿的 —— 而它和「全部通过」长得
  // 一模一样。组件改名、目录挪走、注释剥过头，都会让上面那条 filter 静默变空。
  it("闸门自己找得到东西（否则它就是个假警报）", () => {
    assert.ok(
      SCAN_SITES.length >= 2,
      `只找到 ${SCAN_SITES.length} 处画 ${SCAN} 的地方 —— 预期至少两处` +
        "（视觉页与对话页的视觉缓冲）。组件名或扫描范围可能已经过时。",
    );
    assert.ok(
      FILES.some((f) => f.code.includes(COARSE)),
      `全仓没有一处画 ${COARSE} —— 判据的另一半已经指不到东西了`,
    );
  });

  it("每一处扫描地图旁边都有粗动大地图", () => {
    const missing = SCAN_SITES.filter((f) => !f.code.includes(COARSE)).map((f) => f.rel);
    assert.deepEqual(
      missing,
      [],
      "这些界面只画了压电尺度的那一张（缺粗动大地图，" +
        `影响：${missing.join(", ")}）`,
    );
  });

  it("注释里提一句不算画 —— 判据剥注释", () => {
    assert.ok(
      !stripComments(`// <CoarseMapPanel />`).includes(COARSE),
      "注释没有被剥掉，于是「在注释里写下它」就能骗过这条闸门",
    );
  });
});

describe("没有 tab 的页至少要进得去 ", () => {
  it("每个 INTENTIONALLY_UNLISTED 的路径都有一个渲染得出来的链接", () => {
    for (const path of INTENTIONALLY_UNLISTED) {
      if (path === "/chat") continue; // `/` 的别名，顶栏那一项就是它
      const linked = FILES.filter(
        (f) => !f.rel.startsWith("lib/nav") && f.code.includes(`to="${path}"`),
      ).map((f) => f.rel);
      assert.ok(
        linked.length > 0,
        `${path} 既不在顶栏、也没有任何一处 <Link to="${path}"> —— ` +
          "它只有知道地址的人进得去。「没有 tab」可以是个决定，「没有入口」不是。",
      );
    }
  });

  // 方向的另一半：链接**画得出来**才算数。`/vision` 的那个链接曾经存在于
  // VisionRibbon 里，写在一个条件分支上（`onOpenBuffer ? 按钮 : 链接`），而唯一
  // 的调用方永远传 `onOpenBuffer` —— 于是它一次都没有被画出来过。
  // grep 找得到、页面上没有，这两件事必须能分开。
  it("`/vision` 的入口不是一个永远走不到的分支", () => {
    const ribbon = FILES.find((f) => f.rel === "components/chat/VisionRibbon.tsx");
    assert.ok(ribbon, "VisionRibbon.tsx 不见了 —— 这条断言的前提没了");
    const others = FILES.filter(
      (f) => f.rel !== "components/chat/VisionRibbon.tsx" && !f.rel.startsWith("lib/nav")
        && f.code.includes(`to="/vision"`),
    ).map((f) => f.rel);
    assert.ok(
      others.length > 0,
      "指向 /vision 的链接只剩 VisionRibbon 里那一个了 —— 它在 onOpenBuffer " +
        "恒被传入时永远画不出来，等于没有入口",
    );
  });
});
