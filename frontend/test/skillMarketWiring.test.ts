/**
 * 技能市场的**接线**闸门 —— 三条，各防一个已经发生过的形状。
 *
 * 1. **谁写市场，谁就得判断有没有生效。** 与 `settingsWrite.test.ts` 同形，但这里
 *    的「成功」有三态：一次 `ok:true` 的写入可能仍然没生效（任务在跑 ⇒ 重建排队）。
 *    只看 `ok` 的 onSuccess 会把「已排队」显示成绿色的「已保存」。
 *
 * 2. **每个市场端点都得有人用。** 「已经记下来了」是生产方的话 —— 要问的是**谁读
 *    它**。后端加了一个端点、前端没接，openapi 里有、界面上没有，两边都不报错。
 *    这份清单是**手写**的，故意不从 schema.d.ts 派生：那一份是「实际接上的」，
 *    这一份是「测试认为应该存在的」—— 校验不能交给会犯这个错的那一方。
 *
 * 3. **新子 tab 必须有入口。** `VIEW_TABS` 里有而渲染分支里没有 ⇒ 点进去一片空白；
 *    反过来 ⇒ 死代码假装还在。两个方向都要查（`nav.test.ts` 的同一条道理）。
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");

const WRITE_CALLS = [
  'api.POST("/api/skill-market/subscription"',
  'api.POST("/api/skill-market/subscription/reset"',
  'api.POST("/api/skill-market/recommendations/{rec_id}/resolve"',
  'api.POST("/api/skill-market/import"',
];
const HELPER = "marketWriteProblem";
const HELPER_IMPORT = 'from "@/lib/skillMarket"';
const HELPER_MODULE = join("lib", "skillMarket.ts");

/**
 * 一个写调用点「用了判据」的两种合法写法。
 *
 * 直接写 `marketWriteProblem(...)` 是一种；把它包成一个**文件内唯一的**结算函数
 * （`settle`）再让每个 mutation 都 `onSuccess: settle` 是另一种 —— 后者更好，因为
 * 四个 mutation 之后要做的事完全一样（刷新缓存、逐字显示 rebuild_note、按三态
 * toast），内联四份迟早只有三份是对的。
 *
 * 闸门仍然是**逐调用点**的：一个新加的 mutation 如果写成
 * `onSuccess: () => toast("已保存")`，两个 token 一个都不出现，当场红。而
 * `settle` 自己必须调判据（下面另有一条断言钉着），所以这条捷径不是一个漏洞。
 */
const CONSUMERS = [HELPER, "settle"];

/**
 * 判据的作用范围是**一个 `useMutation` 块**，不是「POST 之后 N 行」。
 *
 * 先写的是 40 行窗口（照 `settingsWrite.test.ts`），实测**给了假绿**：把 reset 那个
 * mutation 的 `onSuccess: settle` 换成一句裸 toast，闸门仍然绿 —— 因为窗口里蹭到了
 * **邻近另一个 mutation** 的 `settle`。四个 mutation 挨着写，行距十来行，窗口再怎么
 * 调也分不开它们。
 *
 * 按 `useMutation(` 切块就分得开：一次调用和它的 `onSuccess` 本来就属于同一个对象，
 * 那才是「这个写有没有被判定」的天然单位。这依然是一条**明写出来的启发式**，不是
 * TSX 语法树 —— 但它切在一个真实的语法边界上，而不是切在行数上。
 */
const MUTATION_SPLIT = "useMutation(";

/**
 * 市场端点清单 —— **手写**，见文件头第 2 条。
 * 每条都要 (a) 在生成的 schema.d.ts 里存在，(b) 在 src 里有至少一个调用点。
 */
const MARKET_ENDPOINTS = [
  "/api/skill-market/catalog",
  "/api/skill-market/status",
  "/api/skill-market/subscription",
  "/api/skill-market/subscription/reset",
  "/api/skill-market/export",
  "/api/skill-market/import",
  "/api/skill-market/recommendations",
  "/api/skill-market/recommendations/{rec_id}/resolve",
  // 审计流。**它进这份清单的理由就是它差点没有消费方** —— 第一版把 audit 写进了
  // 盘、也写了往返测试，然后没有任何东西读它。
  "/api/skill-market/audit",
  // 二期：实验室中心索引
  "/api/skill-market/share/publish",
  "/api/skill-market/lab-index",
  "/api/skill-market/lab-fetch/{sub_id}",
  // 二期：从推送服务器拉一个签名包。**这一条是这份闸门存在的最好理由** ——
  // 它的后端 (`skillpack_client.fetch_and_install`) 写好之后在仓库里躺了六天
  // 零调用方，因为没有人会去检查「谁读它」。
  "/api/skill-overlay/packs/fetch",
];

/** 后端发出的帧名（core/events.py 的 SKILL_RECOMMENDATION）。 */
const WS_EVENT = "skill_recommendation";

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) sourceFiles(p, out);
    else if (/\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/** 去掉注释之后的源码 —— 否则把错误写法写进注释就会让闸门变红。 */
function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");
}

describe("结构闸门：谁写市场，谁就得判断有没有生效", () => {
  it("每一处市场写调用旁边都判断了生效没有", () => {
    const offenders: string[] = [];
    let callSites = 0;
    for (const file of sourceFiles(SRC)) {
      const rel = relative(SRC, file);
      if (rel === HELPER_MODULE) continue;
      const src = stripComments(readFileSync(file, "utf8"));
      if (!WRITE_CALLS.some((c) => src.includes(c))) continue;
      if (!src.includes(HELPER_IMPORT)) {
        offenders.push(`${rel}（没有 import 判据）`);
        continue;
      }
      // 每个 useMutation 块单独看：这个写有没有被它自己的 handler 判定。
      const blocks = src.split(MUTATION_SPLIT).slice(1);
      blocks.forEach((block, n) => {
        const hit = WRITE_CALLS.find((c) => block.includes(c));
        if (!hit) return;
        callSites += 1;
        if (!CONSUMERS.some((c) => block.includes(c))) {
          offenders.push(`${rel}（第 ${n + 1} 个 useMutation：${hit.slice(9)}）`);
        }
      });
    }
    // 自检：匹配不到任何东西的闸门会一直绿，和「确实没问题」输出一模一样。
    assert.ok(
      callSites >= 4,
      `只逐行定位到 ${callSites} 个市场写调用点 —— 判据串多半已失效`,
    );
    assert.deepEqual(
      offenders,
      [],
      `这些地方在写订阅却没有用 ${HELPER} —— 一次「已排队但没生效」会显示成绿色：` +
        offenders.join(", "),
    );
  });

  it("那个结算函数自己确实调了判据（否则上面的捷径就是个漏洞）", () => {
    const files = sourceFiles(SRC).filter((f) => {
      const src = readFileSync(f, "utf8");
      return WRITE_CALLS.some((c) => src.includes(c));
    });
    assert.ok(files.length >= 1, "一个写市场的文件都没找到 —— 判据串失效了");
    for (const f of files) {
      const src = stripComments(readFileSync(f, "utf8"));
      const rel = relative(SRC, f);
      if (!/\bsettle\b/.test(src)) continue;   // 没用这条捷径就不用查
      const decl = src.match(/const settle = [\s\S]{0,600}/)?.[0] ?? "";
      assert.ok(
        decl.includes(HELPER),
        `${rel} 的 settle 没有调用 ${HELPER} —— 那它只是一个名字好听的 onSuccess`,
      );
    }
  });
});

describe("结构闸门：每个市场端点都有人用", () => {
  const schema = readFileSync(join(SRC, "api", "schema.d.ts"), "utf8");
  const allSrc = sourceFiles(SRC)
    .filter((f) => !f.endsWith("schema.d.ts"))
    .map((f) => readFileSync(f, "utf8"))
    .join("\n");

  for (const ep of MARKET_ENDPOINTS) {
    it(`${ep} 在生成的 schema 里存在`, () => {
      assert.ok(
        schema.includes(`"${ep}"`),
        `${ep} 不在 schema.d.ts 里 —— 要么后端没这个端点（清单烂了），` +
          "要么 npm run gen:api 没跑",
      );
    });

    it(`${ep} 在前端有调用点`, () => {
      assert.ok(
        allSrc.includes(`"${ep}"`),
        `${ep} 后端有、前端没人调 —— 生产方接了，消费方不存在`,
      );
    });
  }
});

describe("结构闸门：推荐帧有人消费", () => {
  it("后端发出的 skill_recommendation 在前端有订阅者", () => {
    const allSrc = sourceFiles(SRC)
      .map((f) => stripComments(readFileSync(f, "utf8")))
      .join("\n");
    assert.ok(
      allSrc.includes(`"${WS_EVENT}"`),
      `后端发 ${WS_EVENT} 帧，前端一个订阅者都没有 —— 推荐卡片永远要等下一次手动刷新`,
    );
  });
});

describe("结构闸门：市场子 tab 有入口", () => {
  const page = readFileSync(join(SRC, "pages", "SkillsPage.tsx"), "utf8");

  /** VIEW_TABS 里声明的 id（一份数据同时喂 tab 条和 useStickyTab 白名单）。 */
  const tabIds = [...page.matchAll(/\{\s*id:\s*"([a-z-]+)",\s*label:/g)].map((m) => m[1]!);
  /** 实际渲染的分支。 */
  const branchIds = [...page.matchAll(/view === "([a-z-]+)"/g)].map((m) => m[1]!);
  /** type View 的字面量。 */
  const unionIds = [
    ...(page.match(/type View =([\s\S]*?);/)?.[1] ?? "").matchAll(/"([a-z-]+)"/g),
  ].map((m) => m[1]!);

  it("解析器确实解析到了东西（自检）", () => {
    assert.ok(tabIds.length >= 7, `只解析到 ${tabIds.length} 个 tab —— 正则失效了`);
    assert.ok(branchIds.length >= 7, `只解析到 ${branchIds.length} 个分支`);
  });

  it("市场 tab 在名单里", () => {
    assert.ok(tabIds.includes("market"), "SkillsPage 的 VIEW_TABS 里没有 market");
  });

  it("每个 tab 都有渲染分支，每个分支都有 tab", () => {
    assert.deepEqual(
      [...new Set(tabIds)].sort(),
      [...new Set(branchIds)].sort(),
      "有 tab 没分支 ⇒ 点进去空白；有分支没 tab ⇒ 死代码假装还在",
    );
  });

  it("type View 与 VIEW_TABS 同步", () => {
    assert.deepEqual([...new Set(unionIds)].sort(), [...new Set(tabIds)].sort());
  });
});
