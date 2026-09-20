/**
 * 结构闸门：近期标注帧的「点开放大」—— 一份实现，两页共享，且不自动弹出。
 *
 * ── 为什么需要 ──
 * 已知问题：近期标注帧不能点开放大。那时格子渲染的是一个裸 `<img>`，
 * 没有 handler；两页（视觉 / 对话）各自渲染同一个组件，所以只要有人把放大写进
 * **页面**而不是那个组件，下一页就还是打不开 —— 这个仓在这个形状上栽过三次
 * （#55 文案、#62 曲线、现在是放大）。
 *
 * ── 三条判据，方向各不相同 ──
 * 1. 谁读 `/api/vision/recent`，谁就得用 `RecentFrameThumb` 画格子。判据从**行为**
 *    派生：新加一页近期帧但自己拼 `<img>`，这条红。
 * 2. 放大取的是 `/api/vision/recent-frame/`，**不是**把 `image_b64` 拉大。那份
 *    b64 是 160 px 的缩略图 —— 拉大它是「同样的 160 px，格子更大」，一个看起来
 *    做完了的按钮。
 * 3. **绝不自动弹**。#38（「人工介入的时候看不到后面的界面显示了」）割掉的正是
 *    一个自动弹出的全屏遮罩，而这里用的是同一个 `Modal`。一张图永远不值得盖住
 *    读数。判据是这个文件里不许出现 `useEffect` —— 粗，但这个组件本来就不需要
 *    副作用；哪天真需要了，正确的动作是把这条收窄成「effect 里不许调 setZoom」，
 *    而不是删掉它。
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

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

/** 先剥注释再扫：讲清楚一个错误写法最好的办法就是把它写进注释里。 */
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

/**
 * 显式豁免，不是静默跳过。
 *
 * `api/schema.d.ts` 是 `npm run gen:api` 从 openapi.json 生成的，它**必然**提到
 * 每一个端点路径 —— 「唯一调用方」那条一旦把它算进去就永远红。踩到这个的方式很
 * 有说服力：闸门写完是绿的，跑完 `gen:api` 才红，而中间什么手写代码都没动。
 */
const GENERATED = new Set(["api/schema.d.ts"]);

const FILES = sourceFiles(SRC)
  .map((p) => {
    const text = readFileSync(p, "utf8");
    return {
      rel: relative(SRC, p).replace(/\\/g, "/"),
      text,
      code: stripComments(text),
    };
  })
  .filter((f) => !GENERATED.has(f.rel));

const THUMB = FILES.find((f) => f.rel === "components/vision/RecentFrameThumb.tsx");
const ENDPOINT = "/api/vision/recent-frame/";

describe("近期标注帧 · 点开放大 ", () => {
  it("自检：扫得到源码，且那个组件在", () => {
    assert.ok(FILES.length > 50, `只扫到 ${FILES.length} 个源文件`);
    assert.ok(THUMB, "找不到 RecentFrameThumb.tsx —— 闸门在检查一个不存在的文件");
    // 豁免名单只许放生成物。手写文件一旦进来，这道闸门就开始装样子。
    assert.deepEqual([...GENERATED], ["api/schema.d.ts"]);
  });

  it("谁读 /api/vision/recent，谁就用 RecentFrameThumb 画格子", () => {
    // useRecent 是那个端点唯一的 hook；从行为派生而不是列一张页面清单。
    const readers = FILES.filter(
      (f) => f.code.includes("useRecent(") && !f.rel.startsWith("hooks/"),
    );
    assert.ok(readers.length >= 2, `只有 ${readers.length} 处读近期帧 —— 自检失败，判据没落到东西上`);
    for (const f of readers) {
      assert.ok(
        f.code.includes("<RecentFrameThumb"),
        `${f.rel} 读了近期帧却自己画格子 —— 放大/文案/曲线会在这一页失效`,
      );
    }
  });

  it("放大取的是端点，不是把缩略图拉大", () => {
    assert.ok(
      THUMB!.code.includes(ENDPOINT),
      `RecentFrameThumb 里没有 ${ENDPOINT} —— 放大没有更大的图可取`,
    );
    // 唯一的调用方就是这个组件；别处再写一份就会和格子选的那一帧脱钩。
    const users = FILES.filter((f) => f.code.includes(ENDPOINT));
    assert.deepEqual(
      users.map((f) => f.rel),
      ["components/vision/RecentFrameThumb.tsx"],
      "放大端点有第二个调用方 —— 它选的帧未必是格子上那一帧",
    );
  });

  it("格子上的图是可点的（按钮，不是裸 img）", () => {
    assert.ok(
      /<button[\s\S]*?onClick=\{\(\) => setZoom\(true\)\}/.test(THUMB!.code),
      "缩略图外面没有那个把它打开的按钮",
    );
  });

  it("绝不自动弹出（#38 割掉的正是自动全屏遮罩）", () => {
    assert.ok(
      !THUMB!.code.includes("useEffect"),
      "RecentFrameThumb 里出现了 useEffect —— 自动打开放大框会把读数盖住，" +
        "这正是 #38 拆掉的东西",
    );
    // 反方向：这个仓确实用 useEffect，所以上面那条不是「这个词全仓不存在」。
    assert.ok(
      FILES.some((f) => f.code.includes("useEffect")),
      "全仓一个 useEffect 都没有 —— 上一条断言是空的",
    );
  });
});
