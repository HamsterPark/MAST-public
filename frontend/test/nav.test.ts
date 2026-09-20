// ════════════════════════════════════════════════════════════════════════════
// Tests for src/lib/nav.ts — that every page an operator needs has a way in.
//
//     cd frontend && npm run test:unit
//
// This is the one class of frontend defect that leaves no trace anywhere else.
// A page absent from the nav still routes, still renders, still typechecks, and
// still looks perfect in a screenshot of itself. The only symptom is a human
// who cannot get to it — and the report does not say「导航里少了一项」, it says
// 「初始化页只能进去一次」, which sounds like a state bug and sends whoever picks
// it up looking for a one-way flag that does not exist.
//
// The gap was reported as a small thing worth fixing in passing. The route
// had always resolved. `/instrument-init/reopen` had always existed and had
// always been wired to a button. What did not exist was a nav entry, so once the
// banner did its designed disappearing act the page had no permanent home. The
// operator had to have 退针方向 written over the API — the field where a wrong
// answer drives the tip INTO the sample.
//
// 大标签合并（判据是「大标签太多」）:17 项合并成 10 项。合并**放大**了上面这类
// 缺陷:现在一个页面掉出导航有了第二种方式 —— 它可以还配着组件、却不属于任何一个
// 大标签的 sections。所以这个文件跟着扩:二级段对账、以及**每一条旧地址都重定向
// 到了真的存在的地方**(书签不能断)。
//
// ── 这个文件自己踩过的坑,留在这里 ─────────────────────────────────────────
// 第一版用一个按缩进重建嵌套的正则解析器去读 router.tsx，它给出了一棵**看起来
// 很有道理的错误的树**:把 `/wishlist` 认成了 `/skills` 的父节点,因为多行对象的
// `path:` 缩进比单行对象深一级。一个用正则重建语法树的校验,验的是它自己的猜测。
// 修法不是把正则写得更聪明,是**让路径只有一份真源**:router.tsx 现在从 nav.ts
// 派生路由路径,于是路径根本不可能漂,这里只需要对账「每一段配没配组件」。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import {
  INTENTIONALLY_UNLISTED,
  LEGACY_REDIRECTS,
  TABS,
  findTab,
  sectionIds,
  sectionPath,
} from "../src/lib/nav.ts";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const ROUTER_SRC = readFileSync(join(HERE, "..", "src", "router.tsx"), "utf8");

// router.tsx 里唯一还需要手写的那份映射:「段的完整路径 → 渲染它的组件」。
// 路由的**路径**是从 nav.ts 派生的(所以路径不可能漂),能漂的只剩这一件事。
// 直接读源文件而不是 import —— 那是个 .tsx，node --test 跑不了 JSX。
const SECTION_ELEMENTS: Record<string, string> = Object.fromEntries(
  [...ROUTER_SRC.matchAll(/^\s*"(\/[a-z0-9-]+\/[a-z0-9-]+)":\s*<(\w+)/gm)].map(
    (m) => [m[1]!, m[2]!],
  ),
);

/**
 * 没有二级页的大标签 + 刻意不做 tab 但可达的页面。
 *
 * 手写，而且**故意**不 import router.tsx 里那份 FLAT_ELEMENTS：这一份是「测试
 * 认为应该存在的」，那一份是「实际接上的」。让测试直接读那个常量的话，删掉一页
 * 会让测试跟着一起沉默 —— 校验就不该交给会犯这个错的那一方。
 */
const FLAT_PATHS = [
  "/agents",
  "/qa",
  "/literature",
  "/wishlist",
  "/dashboard",
  "/vision",
  "/chat",
];

/** 应用实际能解析到的全部路径。 */
function routerPaths(): string[] {
  return [
    "/",
    ...FLAT_PATHS,
    ...TABS.filter((t) => t.sections?.length).map((t) => t.to),
    ...Object.keys(SECTION_ELEMENTS),
    ...Object.keys(LEGACY_REDIRECTS),
  ];
}

/** 有 sections 的大标签，其组根本身不渲染内容 —— 它只把人送进某一段。 */
const GROUP_ROOTS = TABS.filter((t) => t.sections?.length).map((t) => t.to);

describe("nav covers every operator-facing route", () => {
  it("router.tsx declares no page that is neither a tab nor explicitly unlisted", () => {
    const reachable = new Set<string>([
      "/",
      ...TABS.map((t) => t.to),
      ...TABS.flatMap((t) => sectionIds(t).map((s) => sectionPath(t, s))),
      ...INTENTIONALLY_UNLISTED,
      // 旧地址本身不是「页面」，它们只是把人送走。
      ...Object.keys(LEGACY_REDIRECTS),
    ]);
    const orphans = routerPaths().filter((p) => !reachable.has(p));
    assert.deepEqual(
      orphans,
      [],
      "these routes have no way in — add a tab/section, or add them to " +
        "INTENTIONALLY_UNLISTED with a reason:\n" + orphans.join("\n"),
    );
  });

  it("仪器初始化 has a home in the nav", () => {
    // Pinned by name because the reason is field evidence, not taste: this page
    // was deliberately kept out of the nav on the premise that it is only used
    // when commissioning a rig. 退针方向 is measured AFTER commissioning and
    // corrected after that, so the premise was wrong. Do not put it back.
    //
    // 2026-08-06 它从顶栏降成「设置」下的一段（#34 是操作员自己给的分组）。
    // 那条理由**一个字都没变** —— 变的只是那个常驻位置在哪一层，所以这条断言
    // 跟着改成「它在导航里有个位置」，而不是「它是个顶栏 tab」。
    const settings = findTab("/settings");
    assert.ok(settings, "/settings must be a top tab");
    assert.ok(
      sectionIds(settings).includes("setup"),
      "仪器初始化 must stay reachable from the nav — see lib/nav.ts",
    );
  });

  it("has no duplicate destinations", () => {
    const seen = new Set<string>();
    for (const t of TABS) {
      assert.equal(seen.has(t.to), false, `duplicate nav entry: ${t.to}`);
      seen.add(t.to);
    }
  });

  it("marks only the index route `end`", () => {
    // Without `end`, "/" prefix-matches every path and the first tab would stay
    // highlighted everywhere; with it on a deeper route, that route would stop
    // highlighting for its own children.
    //
    // 合并之后这一条有了第二重意义:大标签**必须**靠前缀匹配才能在它的二级页上
    // 保持高亮。给 /settings 加上 end，操作员点进「高级管理」之后顶栏上就没有
    // 任何一项是亮的 —— 看起来像「我不在任何页面里」。
    assert.deepEqual(TABS.filter((t) => t.end).map((t) => t.to), ["/"]);
  });

  it("every tab points somewhere the router actually declares", () => {
    const declared = new Set(routerPaths());
    const dangling = TABS.map((t) => t.to).filter((to) => !declared.has(to));
    assert.deepEqual(dangling, [], `nav points at non-routes: ${dangling.join(", ")}`);
  });
});

// ── 合并后的二级结构 ─────────────────────────────────────────────

describe("大标签合并", () => {
  it("每一段都在 router.tsx 里配了组件", () => {
    // 路径不可能漂（router.tsx 从 nav.ts 派生），能漂的只有这一件事:在 nav.ts
    // 里加了一段、却忘了给它配组件。症状是点进去一片空白 —— 路由照样解析、
    // 类型检查照样绿（对象取不到的键是 undefined，不是错误）。
    const missing: string[] = [];
    for (const t of TABS) {
      for (const seg of sectionIds(t)) {
        const p = sectionPath(t, seg);
        if (!SECTION_ELEMENTS[p]) missing.push(`${t.label} → ${p}`);
      }
    }
    assert.deepEqual(missing, [], "这些段点进去会是空白:\n" + missing.join("\n"));
  });

  it("router.tsx 里没有多余的段映射", () => {
    // 反方向:从导航表里删掉一段之后，那行映射会留在 router.tsx 里，
    // 读起来像「这一页还在」。
    const known = new Set(
      TABS.flatMap((t) => sectionIds(t).map((s) => sectionPath(t, s))),
    );
    const orphan = Object.keys(SECTION_ELEMENTS).filter((p) => !known.has(p));
    assert.deepEqual(orphan, [], "这些映射没有对应的导航段:\n" + orphan.join("\n"));
  });

  it("每一段都配了不同的组件", () => {
    // 复制粘贴一行忘了改组件名，得到的是两个 tab 打开同一页 —— 而这在截图里
    // 也是对的（每一张单看都没问题）。
    const byComponent = new Map<string, string[]>();
    for (const [p, comp] of Object.entries(SECTION_ELEMENTS)) {
      byComponent.set(comp, [...(byComponent.get(comp) ?? []), p]);
    }
    const dupes = [...byComponent.entries()]
      .filter(([, paths]) => paths.length > 1)
      .map(([comp, paths]) => `${comp}: ${paths.join(" / ")}`);
    assert.deepEqual(dupes, [], "两个段渲染同一个组件:\n" + dupes.join("\n"));
  });

  it("没有 sections 的大标签不会被当成组处理", () => {
    // 一页到底的大标签（仪器 Chat / Agents / 查询助手 / 文献库 / 心愿单）不该
    // 长出一条只有一个段的二级 tab 条 —— 那是纯噪音。
    const oneSection = TABS.filter((t) => t.sections && t.sections.length < 2);
    assert.deepEqual(
      oneSection.map((t) => t.to),
      [],
      "只有一段的组等于给页面加了一条没用的 tab 条",
    );
  });

  it("五组合并每一组都在", () => {
    // 钉住的是分组背后**要保留的意图**，不是当前的实现:
    // 一次「顺手整理」把某一组拆回去，这里会红。
    const merged: Record<string, string[]> = {
      "/skills": ["library", "builder"],
      "/records": ["log", "memory"],
      "/settings": ["general", "setup", "admin", "usage"],
      "/monitoring": ["current", "env"],
      "/experimental": ["tools", "optics"],
    };
    for (const [to, segs] of Object.entries(merged)) {
      const tab = findTab(to);
      assert.ok(tab, `${to} 不再是一个大标签了`);
      // 给定的每一段都还在、**相对顺序也还在** —— 但允许后来新增的段插进来，
      // 前提是那一段在 ADDED_LATER 里写了理由（见下面那条测试）。
      //
      // 从 deepEqual 放宽成子序列，是因为这两件事必须分开:「把某一组拆回去」
      // （他的意图被推翻）和「往某一组里加一页」（他的意图仍然成立，只是多了
      // 一件东西）。原来的 deepEqual 把两者都报成同一句「分段变了」，于是
      // 加页的人唯一的出路是改这张表——而改完之后，拆组也不会红了。
      const actual = sectionIds(tab!);
      const kept = actual.filter((s) => segs.includes(s));
      assert.deepEqual(kept, segs, `${to} 的分段被拆了或顺序变了`);
    }
  });

  it("#34 之后新增的每一段都写了理由", () => {
    // 加一段的门槛不是「不许」，是「要写下来为什么」。没有这条的话，导航会
    // 一次一页地涨回 #34 之前那个挤到换行的样子，而每一次单看都很有道理。
    const ADDED_LATER: Record<string, Record<string, string>> = {
      "/records": {
        conduct:
          "多天 conduct 面板（M1-d；2026-08-20 由 campaign 改名而来——campaign " +
          "一词归还 logging/v2 的科研纲领）。放这一组是因为一份 conduct **必须**" +
          "绑定一个实验（experiment_id 必填，spec 快照与 progress.jsonl 都落在那个" +
          "实验的文件夹里）；不新开第 11 个大标签是因为已经判定「大标签太多了」。",
        gallery:
          "数据图库提供增量预处理、按目录和类型浏览、标记、系列与谱帧关联。" +
          "它使用用户配置的数据根，标记保存在图库状态目录中。",
      },
    };
    const known = new Set(
      Object.values({
        "/skills": ["library", "builder"],
        "/records": ["log", "memory"],
        "/settings": ["general", "setup", "admin", "usage"],
        "/monitoring": ["current", "env"],
        "/experimental": ["tools", "optics"],
      }).flat(),
    );
    for (const tab of TABS) {
      for (const seg of sectionIds(tab)) {
        if (known.has(seg)) continue;
        const why = ADDED_LATER[tab.to]?.[seg];
        assert.ok(
          why && why.length > 20,
          `${tab.to}/${seg} 是 #34 之后新增的段，但没有在 ADDED_LATER 里写理由。` +
            `加一段可以，但要能写出一句站得住的话——否则导航会一次一页地涨回去。`,
        );
      }
    }
  });
});

// ── 旧地址（书签不能断） ────────────────────────────────────────────────────

describe("合并之前的地址还能用", () => {
  it("每一条旧地址都指向一个真的存在的二级页", () => {
    // 指到一个不存在的地方，症状是操作员的书签打开一个白页 —— 看起来像
    // 「这个功能被删了」，而不是「重定向表里有个笔误」。
    const declared = new Set(routerPaths());
    const broken = Object.entries(LEGACY_REDIRECTS)
      .filter(([, to]) => !declared.has(to))
      .map(([from, to]) => `${from} → ${to}`);
    assert.deepEqual(broken, [], "重定向指向不存在的路由:\n" + broken.join("\n"));
  });

  it("被合并掉的每一个顶栏地址都在重定向表里", () => {
    // 这一条是整组的要害。合并的动作是「从 TABS 里删掉一项」，而删掉之后那个
    // 地址会安静地变成 404 —— TABS 少一项这件事没有任何东西会抗议。
    // 名单写死，因为它是**历史事实**:2026-08-06 之前顶栏上确实有这些项。
    const RETIRED_TOP_TABS = [
      "/builder",      // → 技能
      "/cognition",    // → 实验记录
      "/admin",        // → 设置
      "/setup",        // → 设置
      "/usage",        // → 设置
      "/env-history",  // → 监控
      "/optics",       // → 实验性功能
    ];
    const stillTop = new Set(TABS.map((t) => t.to));
    for (const old of RETIRED_TOP_TABS) {
      assert.equal(stillTop.has(old), false, `${old} 应该已经并进别的大标签了`);
      assert.ok(LEGACY_REDIRECTS[old], `${old} 被合并掉了却没有重定向，书签会 404`);
    }
  });

  it("重定向不指向组根", () => {
    // 指到组根的话，操作员的 /admin 书签会被送到「上次停的那一段」——
    // 大多数时候不是高级管理。链接照样打得开，所以没人会意识到它去错了地方。
    const toRoot = Object.entries(LEGACY_REDIRECTS)
      .filter(([, to]) => GROUP_ROOTS.includes(to))
      .map(([from, to]) => `${from} → ${to}`);
    assert.deepEqual(
      toRoot, [], "重定向落在组根上，会被再送去别处:\n" + toRoot.join("\n"));
  });

  it("重定向不成环、不接力", () => {
    for (const [from, to] of Object.entries(LEGACY_REDIRECTS)) {
      assert.notEqual(from, to, `${from} 重定向到自己`);
      assert.equal(
        LEGACY_REDIRECTS[to],
        undefined,
        `${from} → ${to} → ${LEGACY_REDIRECTS[to]}：两跳重定向，后退键会失灵`,
      );
    }
  });

  it("旧地址不和现有路由撞名", () => {
    const live = new Set([
      ...TABS.map((t) => t.to),
      ...TABS.flatMap((t) => sectionIds(t).map((s) => sectionPath(t, s))),
      ...FLAT_PATHS,
    ]);
    const clash = Object.keys(LEGACY_REDIRECTS).filter((p) => live.has(p));
    assert.deepEqual(clash, [], `旧地址盖住了真页面:${clash.join(", ")}`);
  });
});

// ── 顶栏本身 ────────────────────────────────────────────────────────────────

describe("顶栏挤不挤", () => {
  it("大标签不超过 12 项", () => {
    // 的判据是「大标签太多」。这条闸门存在的意义是:下一个加功能的人
    // 会顺手再加一项（每个人都只加一项，17 项就是这么来的），而这条会当场拦住他，
    // 逼他回答「它属于哪个大标签」—— 那正是操作员这次替我们回答的那个问题。
    assert.ok(
      TABS.length <= 12,
      `顶栏有 ${TABS.length} 项。加新页面之前先想清楚它属于哪个已有的大标签；`
        + `真要开新的一项，把这个上限连同理由一起改。`,
    );
  });

  it("每一项都有非空标签", () => {
    for (const t of TABS) assert.ok(t.label.length > 0, `${t.to} 没有标签`);
    for (const t of TABS) {
      for (const s of t.sections ?? []) {
        assert.ok(s.label.length > 0, `${t.to}/${s.seg} 没有标签`);
      }
    }
  });
});

// ── 应用内部的链接 ──────────────────────────────────────────────────────────

describe("内部链接指向具体的段，不是组根", () => {
  it("没有 <Link to=\"/组根\"> 或 navigate(\"/组根\")", () => {
    // 指到组根不会坏,只会**去错地方**:组根把人送到「上次停的那一段」。
    // 一个写着「去设置 → 扫描参数档位」的链接把人丢在「用量花销」上,
    // 链接照样打得开、页面照样渲染,所以没有任何东西会抗议 ——
    // 这正是 #34 合并时最容易留下的那种伤,而且要等操作员来报。
    //
    // 组根本身仍然是顶栏那一项的目标(顶栏点进去就该回到上次那一段),
    // 所以这条只查源码里的链接,不查导航表。
    const SRC = join(HERE, "..", "src");
    // 这三个文件**就是**负责把人送进组根的那一层。
    const skip = new Set(["router.tsx", "TabGroup.tsx", "nav.ts"]);
    const offenders: string[] = [];

    // 逐行找字面量,不拼正则。上一版这里拼了个正则,而反斜杠在写进文件的路上
    // 被吃掉了,于是它抛 SyntaxError —— 好在抛出来了。要是恰好拼成一个能编译、
    // 但匹配不到任何东西的模式,这条闸门会**一直是绿的**,而绿得毫无道理。
    const forms = (root: string) => [
      `to="${root}"`, `to='${root}'`, "to={`" + root + "`}",
      `navigate("${root}")`, `navigate('${root}')`, "navigate(`" + root + "`)",
      // 带 query / hash 的形式:`/monitoring?seg=` 就是这样漏掉的。
      `to="${root}?`, `to='${root}?`, "to={`" + root + "?",
      `navigate("${root}?`, "navigate(`" + root + "?",
    ];

    const walk = (dir: string): void => {
      for (const e of readdirSync(dir, { withFileTypes: true })) {
        const p = join(dir, e.name);
        if (e.isDirectory()) { walk(p); continue; }
        if (!/\.tsx?$/.test(e.name) || skip.has(e.name)) continue;
        for (const line of readFileSync(p, "utf8").split("\n")) {
          const t = line.trimStart();
          // 注释里提到组根不算 —— 这个仓的注释里到处是路径。
          if (t.startsWith("//") || t.startsWith("*") || t.startsWith("/*")) continue;
          for (const root of GROUP_ROOTS) {
            if (forms(root).some((f) => line.includes(f))) {
              offenders.push(`${e.name}: ${root}`);
            }
          }
        }
      }
    };
    walk(SRC);
    assert.deepEqual(
      [...new Set(offenders)], [],
      "这些链接会落到「上次停的那一段」,不是它们想去的地方:\n"
        + [...new Set(offenders)].join("\n"),
    );
  });
});
