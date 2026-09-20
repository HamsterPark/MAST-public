/**
 * 「已保存」这句话什么时候可以说 —— 判据本身 + 一道结构闸门。
 *
 * ── 背景 ──
 * 2026-08-10 实测：「保存 Nanonis 连接配置」是彻底的 no-op（五个字段不在后端写
 * schema 上，pydantic `extra='ignore'` 全丢，`ui_settings.json` 连建都没建），而
 * 按钮 toast 绿字「已保存」。后端那一半已经补上；**前端这一半是它为什么没被发现**：
 * 那个 onSuccess 只看了 `degraded`，而这个端点的拒绝形状是 `ok:false,
 * degraded:false`（端口越界 / 档位表结构非法 / 参数组数值没带 SI 前缀 / 管理 PIN
 * 不对）—— 于是「拒绝」和「成功」在前端长得完全一样。
 *
 * 六处在 POST 这个端点，逐一去看的时候发现各写各的：
 *   SettingsPage      查 degraded + ok
 *   BackgroundRuns    `saveAuto` 一句都没查（同一个文件里 `spawn` 查了 —— 所以
 *                     「文件里出现过 .ok」这种判据会在这里给出假绿）
 *   PendingActivations 一句都没查
 *   CapabilityGates   查 pin_required + degraded，没查 ok
 *   useAutonomyMode   一句都没查（乐观更新留在屏幕上，直到重取悄悄换回去）
 *   HardwareManager   只查 degraded ← 出事的那一处
 *
 * 所以判据不再由各页自己记，收进 `lib/settingsWrite.ts`。闸门查的是**接线**：
 * 谁 POST 了这个端点，谁就得 import 那个判据。这一条按行为派生，新增的页面不接
 * 线就红，不需要任何人记得更新名单。
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import { settingsWriteProblem } from "../src/lib/settingsWrite.ts";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");

/** 写这个端点的唯一写法（`api.POST("/api/settings"…)`）。 */
const WRITE_CALL = 'api.POST("/api/settings"';
const HELPER = "settingsWriteProblem";
const HELPER_IMPORT = 'from "@/lib/settingsWrite"';
/** 判据自己住的那个文件，当然不用 import 自己。 */
const HELPER_MODULE = join("lib", "settingsWrite.ts");

/**
 * 判据要出现在 POST 调用后的多少行以内。
 *
 * 为什么要有这个距离，而不是「文件里出现过就算」：`BackgroundRunsPanel.tsx` 里
 * 有两个 mutation，`spawn` 查了响应而 `saveAuto` 一句没查 —— 文件级的判据在那里
 * 给出的是**假绿**。所有六处的写法都是 `mutationFn:` 里 POST、紧接着的
 * `onSuccess:` 里判断，实测最远的一处是 13 行，40 行留了三倍余量。
 * 这是一条明写出来的启发式，不是 TSX 语法树：按缩进重建嵌套只会得到一棵看起来
 * 很有道理的错误的树。
 */
const NEAR_LINES = 40;

/**
 * 明知故犯的豁免。**空的**，而且应该一直是空的。
 * 留这个口子是因为没有它的闸门会被下一个人整条删掉，而不是加一行豁免；但每加
 * 一项都必须在这里写下它为什么 POST 了设置却不需要判断有没有存进去。
 */
const EXEMPT = new Map<string, string>();

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) sourceFiles(p, out);
    else if (/\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/**
 * 去掉注释之后的源码。
 *
 * 这份闸门找的是一个**错误写法**，而讲清楚那个错误写法最好的办法就是把它原样写
 * 在注释里 —— 不剥注释的话，写下教训本身就会让闸门变红，下一个人的修法多半是
 * 删掉那段解释。`://` 不当行注释（`https://…`）。
 */
function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");
}

describe("settingsWriteProblem — 判据本身", () => {
  it("真的存进去了 → null", () => {
    assert.equal(settingsWriteProblem({ ok: true, degraded: false }), null);
    assert.equal(
      settingsWriteProblem({ ok: true, degraded: false, rejected: {} }),
      null,
    );
  });

  it("ok:false + degraded:false 是拒绝，不是成功", () => {
    // 出事的那个形状。只看 degraded 的旧写法在这里说的是「已保存」。
    assert.notEqual(settingsWriteProblem({ ok: false, degraded: false }), null);
  });

  it("rejected 里的中文原因原样带出来，不吞成一句「保存失败」", () => {
    const msg = settingsWriteProblem({
      ok: false,
      degraded: false,
      rejected: { nanonis_port_main: "端口必须是 1..65535 的整数（收到 70000）" },
    });
    assert.match(String(msg), /65535/);
  });

  it("多条 rejected 全部说出来", () => {
    const msg = String(
      settingsWriteProblem({
        ok: false,
        rejected: { a: "甲错了", b: "乙错了" },
      }),
    );
    assert.match(msg, /甲错了/);
    assert.match(msg, /乙错了/);
  });

  it("PIN 拒绝优先，用后端给的那句人话", () => {
    const msg = settingsWriteProblem({
      ok: false,
      pin_required: true,
      rebuild_note: "未设置管理 PIN，请先在高级里设一个。",
    });
    assert.equal(msg, "未设置管理 PIN，请先在高级里设一个。");
  });

  it("内核没接上 → 说的是「未生效」而不是别的", () => {
    const msg = String(settingsWriteProblem({ ok: false, degraded: true }));
    assert.match(msg, /未生效/);
  });

  it("没有回应也算没存进去（不是「成功」）", () => {
    assert.notEqual(settingsWriteProblem(undefined), null);
    assert.notEqual(settingsWriteProblem(null), null);
  });

  it("ok 缺席不算成功", () => {
    // 后端换了响应模型、字段没了 —— 那时候「不知道」必须落在失败一侧。
    assert.notEqual(settingsWriteProblem({ degraded: false }), null);
  });
});

describe("结构闸门：谁写设置，谁就得判断有没有写进去", () => {
  it("每一处 api.POST(\"/api/settings\") 旁边都判断了写没写进去", () => {
    const offenders: string[] = [];
    let writers = 0;
    let callSites = 0;
    for (const file of sourceFiles(SRC)) {
      const rel = relative(SRC, file);
      if (rel === HELPER_MODULE) continue;
      const src = stripComments(readFileSync(file, "utf8"));
      if (!src.includes(WRITE_CALL)) continue;
      writers += 1;
      if (EXEMPT.has(rel)) continue;
      if (!src.includes(HELPER_IMPORT)) {
        offenders.push(`${rel}（没有 import 判据）`);
        continue;
      }
      // 每一处 POST 后面 NEAR_LINES 行内都要有那一句 —— 一个文件里两个
      // mutation、只有一个查了的情况，文件级判据看不出来。
      const lines = src.split("\n");
      lines.forEach((ln, i) => {
        if (!ln.includes(WRITE_CALL)) return;
        callSites += 1;
        const near = lines.slice(i, i + NEAR_LINES).join("\n");
        if (!near.includes(HELPER)) offenders.push(`${rel}:${i + 1}`);
      });
    }
    // 自检：闸门必须真的找到了写入方。一个匹配不到任何东西的闸门会一直绿，
    // 和「确实没问题」输出一模一样 —— 这个仓为此付过学费。
    assert.ok(
      writers >= 5,
      `只找到 ${writers} 处 POST /api/settings —— 判据串 ${WRITE_CALL} 多半已失效`,
    );
    assert.ok(
      callSites >= 5,
      `只逐行定位到 ${callSites} 个调用点 —— 逐行扫描没生效`,
    );
    assert.deepEqual(
      offenders,
      [],
      `这些地方在写设置却没有用 ${HELPER} 判断写没写进去 —— ` +
        `被拒绝的写入会显示成绿色的「已保存」：${offenders.join(", ")}`,
    );
  });
});
