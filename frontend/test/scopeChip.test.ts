// ════════════════════════════════════════════════════════════════════════════
// 顶栏作用域 chip 的宽度决定 —— 把**被否掉的做法**钉住，不只是把当前值钉住。
//
//     cd frontend && npm run test:unit
//
// 这块地方被同一个毛病咬过两次：
//
//   · 外壳 `max-w-[26ch]`，里面两个名字各声明 14ch，长一点的实验名 + 样品名
//     拼起来会被截断。当时的修法是把 26ch 调成 38ch。
//   · 同一个位置又咬了一次：更长的实验名 + 样品名右上角排版不美观。
//     38ch 依然不够。
//
// 为什么调数字修不好：那个数字必须 ≥ 五个子元素 + 4 段 gap + 左右 padding 的
// 总和，而其中 `实验` / `样品` 是 CJK —— 1ch 是 "0" 的宽度，一个汉字是 1em，
// 约 1.8ch。上一轮的算式按 2ch 记这两个标签，起手就少算 ~3.6ch。任何人重新
// 算一遍都会再算错一次，因为这个换算根本不写在算式里。
//
// 所以这一轮删掉了外壳上的固定 ch 上限（改成 `max-w-full`，交给父容器的
// flex-wrap）。下一个人会重新想到「给它个上限吧」—— 那看起来显然对，而且他不会
// 先读上面这段注释。这条测试就是拦他的。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const TOPBAR = fileURLToPath(new URL("../src/components/shell/TopBar.tsx", import.meta.url));

/** The chip's own markup: from its test id to the end of its <button>. */
function chipBlock(): string {
  const src = readFileSync(TOPBAR, "utf8");
  const start = src.indexOf('data-testid="scope-chip"');
  assert.notEqual(start, -1, "顶栏里找不到 scope-chip —— 这条测试在测一个不存在的东西");
  const end = src.indexOf("</button>", start);
  assert.notEqual(end, -1, "scope-chip 的 </button> 找不到，切片取错了");
  return src.slice(start, end);
}

/** The outer <button>'s own class string (children use `className={`). */
function shellClasses(block: string): string {
  const m = /className="([^"]*)"/.exec(block);
  assert.ok(m, "scope-chip 外壳没有字面 className —— 切片假设失效，先修这条测试");
  return m![1];
}

describe("顶栏作用域 chip", () => {
  it("外壳不设固定 ch 宽上限（被否掉的做法）", () => {
    const cls = shellClasses(chipBlock());
    const fixed = /max-w-\[\d+(?:\.\d+)?ch\]/.exec(cls);
    assert.equal(
      fixed,
      null,
      `外壳又出现了固定 ch 上限：${fixed?.[0]}。要推翻这个决定，先回答：` +
        `这个数字怎么把两个 CJK 标签按 ~1.8ch/字、4 段 gap、左右 padding 一起算进去，` +
        `并且用什么观测证明两个名字真的拿到了它们声明的宽度？`,
    );
  });

  it("外壳仍然不许撑破所在行", () => {
    const cls = shellClasses(chipBlock());
    assert.ok(cls.includes("max-w-full"), "去掉 ch 上限不等于不要上限：溢出会顶破顶栏");
    assert.ok(cls.includes("min-w-0"), "没有 min-w-0，flex 子项压不下去，仍会溢出");
  });

  it("两个名字各自带声明的上限，且不是 14ch", () => {
    const block = chipBlock();
    const caps = [...block.matchAll(/max-w-\[(\d+)ch\]/g)].map((m) => Number(m[1]));
    assert.equal(caps.length, 2, "应当正好两个名字带 ch 上限（实验名 / 样品名）");
    for (const c of caps) {
      // 混合中英文名称保留显示余量。
      assert.ok(c >= 16, `名字上限 ${c}ch 太窄：常见实验名约 14.3ch，留不出余量`);
    }
  });

  it("悬浮提示要给得回被截断的名字，不能只给 ID", () => {
    const block = chipBlock();
    const title = block.slice(block.indexOf("title="), block.indexOf("className="));
    assert.ok(title.includes("expName"), "title 里没有实验名 —— 截断之后就没处可查了");
    assert.ok(title.includes("smpName"), "title 里没有样品名 —— 截断之后就没处可查了");
    assert.ok(title.includes("ID"), "ID 仍然要留着：名字会重名，ID 不会");
  });
});
