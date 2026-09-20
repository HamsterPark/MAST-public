// ════════════════════════════════════════════════════════════════════════════
// 扫描地图上的每一段文字都必须交代它在哪个空间里 —— 源码级守卫。
//
//     cd frontend && npm run test:unit
//
// 地图有两种文字，规矩相反：
//
//   · 画布家具（坐标刻度、比例尺、tooltip）住在带 `{...chrome}` 反变换的图层里，
//     那一层已经把 stage 变换抵消掉了，所以它们**应该**用裸 x/y —— 那已经是屏幕
//     坐标。
//   · 贴在某个实物上的标注（针尖 / 建议下一个位置 / 压电范围 / 计划与候选序号）
//     住在被缩放的世界图层里。裸 x/y 会跟着 stage 一起放大：25× 下一个 10 px 的
//     字要画到 250 px 高，把它本来要标注的表面整个盖住。它们必须走 `pinnedLabel`。
//
// 为什么是测试而不是注释：#52 修完全局家具之后，这五个标注**看起来**也一起修好了
// （它们确实跟着实物走，只是越来越大），没有任何东西会报错。下一个往地图上加标注
// 的人会照着最近的一段 `<Text x={…} y={…} />` 抄 —— 而最近的那一段恰好是 chrome
// 层里合法的那种。这条测试是唯一会在那一刻出声的东西。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const CANVAS = fileURLToPath(
  new URL("../src/components/vision/ScanMapCanvas.tsx", import.meta.url),
);

/** Every `<Text … />` tag in the file, paired with whether it sits in the
 *  inverse-transformed chrome layer. */
function textTags(): { tag: string; inChrome: boolean }[] {
  const src = readFileSync(CANVAS, "utf8");
  const chromeStart = src.indexOf("{...chrome}");
  assert.notEqual(chromeStart, -1, "找不到 chrome 图层 —— 切片假设失效，先修这条测试");
  const chromeEnd = src.indexOf("</Layer>", chromeStart);
  assert.notEqual(chromeEnd, -1, "chrome 图层没有闭合标签，切片取错了");

  const out: { tag: string; inChrome: boolean }[] = [];
  for (let i = src.indexOf("<Text"); i !== -1; i = src.indexOf("<Text", i + 1)) {
    const end = src.indexOf("/>", i);
    assert.notEqual(end, -1, "有一个 <Text> 不是自闭合的，切片取错了");
    out.push({ tag: src.slice(i, end + 2), inChrome: i > chromeStart && i < chromeEnd });
  }
  return out;
}

describe("扫描地图文字标注", () => {
  it("文件里确实有两种文字，测试没有在测空气", () => {
    const tags = textTags();
    assert.ok(tags.some((t) => t.inChrome), "chrome 层一段文字都没有？切片取错了");
    assert.ok(tags.some((t) => !t.inChrome), "世界层一段文字都没有？切片取错了");
  });

  it("世界层的每一段文字都走 pinnedLabel", () => {
    for (const { tag, inChrome } of textTags()) {
      if (inChrome) continue;
      const oneLine = tag.replace(/\s+/g, " ").slice(0, 120);
      assert.ok(
        tag.includes("pinnedLabel("),
        `世界层的这段文字用了裸坐标，25× 下会被放大 25 倍：${oneLine}\n` +
          `要么改用 pinnedLabel(锚点, 屏幕 dx, 屏幕 dy, zoomK)，` +
          `要么说明它为什么应该跟着缩放。`,
      );
    }
  });

  it("chrome 层的文字反过来不该用 pinnedLabel（会被抵消两次）", () => {
    for (const { tag, inChrome } of textTags()) {
      if (!inChrome) continue;
      assert.ok(
        !tag.includes("pinnedLabel("),
        `chrome 层已经带了整层反变换，再套一次 pinnedLabel 等于缩小两次：${tag.slice(0, 90)}`,
      );
    }
  });
});
