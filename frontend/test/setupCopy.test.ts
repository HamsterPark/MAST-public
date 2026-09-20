/**
 * 结构闸门：初始化这一套界面上的字，不许再长回来。
 *
 * ── 为什么是闸门而不是又删一次 ──
 * 同一条反馈来了**五次**（「废话略多」→ 「说明基本上可以删去」→ 「废话
 * 太多」→ 「这种话都删掉」→ 「这些废话还是没有清理？」）。前三轮各自认真
 * 删过、也确实改善了，然后字长了回来。
 *
 * 根因不是谁偷懒，是**记账单位错了**：每一次多写一句都有它自己的好理由，逐条审
 * 都通过，而「加起来太长了」不属于任何一次改动，于是没有任何一次 review 会拦它。
 *
 * 目录那一侧（字数 / 一句 / 总量 / 安全警示下限）钉在
 * `tests/v2/unit/core/test_instrument_init.py`。这里钉的是**前端写死的那些字**，
 * 后端管不到：横幅的措辞，以及页面还印不印那些已经被请出去的字段。
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");

/** 只看渲染出来的东西：注释里写下教训不该让闸门变红。 */
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

const BANNER = stripComments(readFileSync(join(SRC, "components/shell/SetupBanner.tsx"), "utf8"));
const PAGE = stripComments(readFileSync(join(SRC, "pages/SetupPage.tsx"), "utf8"));

/**
 * 带中文的字符串字面量 = 给人看的文案。className 之类的英文串不算。
 *
 * 判据落在**中文**上是刻意的：这一族缺陷全部是中文散文，而这个仓的 className
 * 动辄七八十个字符，按长度一刀切会把它们全扫进来，于是闸门要么被放宽到没用，
 * 要么被下一个人整条删掉。
 */
function chineseLiterals(src: string): string[] {
  const out: string[] = [];
  for (const m of src.matchAll(/["'`]([^"'`\n]*)["'`]/g)) {
    const s = m[1] ?? "";
    if (/[一-鿿]/.test(s)) out.push(s);
  }
  // JSX 里裸写的文字。**这一半是变异测试逼出来的**：只查带引号的字符串时，把那句
  // 被删掉的副标题原样写成 `<span>缺了它们，……</span>` 就绕过了整条闸门，而闸门
  // 报绿 —— 和「确实没有长句」长得一模一样。
  for (const m of src.matchAll(/>([^<>{}]*)</g)) {
    const s = (m[1] ?? "").replace(/\s+/g, " ").trim();
    if (/[一-鿿]/.test(s)) out.push(s);
  }
  return out;
}

/** 横幅一句话的上限。当前最长的一句是 24 字。 */
const BANNER_MAX = 32;

describe("初始化横幅只说一句话 ", () => {
  it("闸门自己找得到东西（否则它就是个假警报）", () => {
    const lits = chineseLiterals(BANNER);
    assert.ok(lits.length >= 5, `只在横幅里找到 ${lits.length} 句中文文案 —— 判据可能已经失效`);
  });

  // #39 操作员点名删掉的判据：「缺了它们，某些安全网是按出厂占位值在拦人的 ——
  // 出厂值与任何一台真实机器都不对应。」43 字，解释的是「为什么要有这块横幅」，
  // 而看见横幅的人已经不需要被说服了。三条分支的副标题都是这个形状。
  it("横幅里没有一句解释性的长句", () => {
    const tooLong = chineseLiterals(BANNER).filter((s) => s.length > BANNER_MAX);
    assert.deepEqual(
      tooLong,
      [],
      `横幅上出现了长句（>${BANNER_MAX} 字）。横幅只该给数字和那个按钮，` +
        `解释搬进代码注释：${tooLong.join(" | ")}`,
    );
  });
});

describe("初始化页不再印那些被请出去的字段 ", () => {
  // #40 原样贴回来的四段里，最后一段「可对账：GetPiezoConfig.range（Z 分量）」
  // 与第二段「哪里找：`GetPiezoConfig.range` 的 Z 分量」是同一句话说了两遍。
  // `probe` 字段留着（「从仪器读一次」按钮要用），但**不上屏**。
  it("`probe` 不上屏 —— 它和一行提示指的是同一个读法", () => {
    assert.ok(
      !/item\.probe/.test(PAGE),
      "页面又开始印 item.probe 了 —— 那是一行提示的第二个副本（#40 点名的就是它）",
    );
  });

  // 从前每一项都有一张四段式说明卡（这是什么 / 哪里找 / 填错会怎样 / 可对账）。
  // 前三轮删的是卡里的字，第四轮删的是卡 —— 只要格子还在，就总有理由填满它，
  // 而「加起来太长了」不属于任何一次改动。
  it("四段式说明卡没有回来", () => {
    for (const label of ["这是什么：", "哪里找：", "填错 / 不填会怎样：", "可对账"]) {
      assert.ok(
        !PAGE.includes(label),
        `页面上又出现了「${label}」—— 四段式说明卡回来了`,
      );
    }
  });

  // 方向的另一半。删字的时候最容易顺手删掉的恰好是安全警示，因为它们最长、
  // 最像废话 —— 而它们是这一页存在的理由。
  it("安全关键项的方向性警示仍然常显", () => {
    assert.match(
      PAGE,
      /item\.safety_critical/,
      "页面不再按 safety_critical 区分 —— 要么安全警示没了，要么所有后果又全摊开了",
    );
    assert.match(PAGE, /<ConsequenceLine item=\{item\} \/>/, "常显的后果那一行没了");
  });
});
