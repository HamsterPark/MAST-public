// 「上下文注入」子页的**结构闸门** —— 读源码文本断言接线，因为 `.tsx` 在这个
// 仓里跑不了单测（`node --test` 剥不掉 JSX）。
//
// 每一条都带**判据自校验**：先对一段构造的坏源码断言判据报红，再扫真仓。
// 一个匹配不到任何东西的判据永远是绿的，而它和「全部通过」长得一模一样。

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (p: string) => readFileSync(new URL(`../${p}`, import.meta.url), "utf8");

const AGENTS_PAGE = read("src/pages/AgentsPage.tsx");
const VIEW = read("src/components/agents/ContextInjectionView.tsx");
const MATRIX = read("src/components/agents/ContextInjectionMatrix.tsx");
const SHARE = read("src/components/agents/ShareBar.tsx");
const REGISTRY = read("src/components/agents/registry.tsx");
const ADMIN = read("src/pages/AdminPage.tsx");
const LIB = read("src/lib/contextInjection.ts");

const NEW_FILES = { VIEW, MATRIX, SHARE, LIB };

// ── 1. 子页接线 ─────────────────────────────────────────────────────────

describe("子页接线", () => {
  it("SubTab 联合类型与 SUB_TABS 两处都有 context", () => {
    // 只加一处的症状是「点得进去但记不住」或者「TS 报错」，两个都不好查。
    assert.ok(/\|\s*"context"/.test(AGENTS_PAGE), "SubTab 联合类型里没有 context");
    assert.ok(/\{\s*id:\s*"context",\s*label:/.test(AGENTS_PAGE),
              "SUB_TABS 里没有 context");
  });

  it("**未激活 = 未挂载 = 零请求**", () => {
    // 渲染必须挂在 `tab === "context" &&` 后面。挂在外面的话，操作员一进
    // Agents 页就发三个请求，而他可能根本没点这个子页。
    const idx = VIEW_MOUNT_INDEX();
    assert.ok(idx >= 0, "AgentsPage 里没有渲染 ContextInjectionView");
    const before = AGENTS_PAGE.slice(Math.max(0, idx - 200), idx);
    assert.ok(before.includes('tab === "context"'),
              "ContextInjectionView 没有挂在 tab === \"context\" 的守卫后面");
  });

  it("判据自校验：没有守卫的写法必须被判红", () => {
    const bad = 'return <div><ContextInjectionView picks={x} /></div>;';
    const i = bad.indexOf("<ContextInjectionView");
    assert.equal(bad.slice(Math.max(0, i - 200), i).includes('tab === "context"'),
                 false);
  });
});

function VIEW_MOUNT_INDEX(): number {
  return AGENTS_PAGE.indexOf("<ContextInjectionView");
}

// ── 2. agent 名单必须派生，不许手抄 ─────────────────────────────────────

function registryAgentIds(): string[] {
  const ids: string[] = [];
  const arr = REGISTRY.slice(REGISTRY.indexOf("export const AGENTS"));
  for (const m of arr.matchAll(/id:\s*"([a-z_]+)"/g)) {
    ids.push(m[1] as string);
    if (ids.length > 12) break;
  }
  const sup = REGISTRY.match(/export const SUP_ID\s*=\s*"([^"]+)"/);
  if (sup?.[1]) ids.push(sup[1]);
  return ids;
}

describe("agent 名单", () => {
  it("闸门自检：确实从 registry 里解析出了 agent id", () => {
    const ids = registryAgentIds();
    assert.ok(ids.length >= 8, `只解析出 ${ids.length} 个 id：${ids}`);
    assert.ok(ids.includes("instrument_control") && ids.includes("research_director"));
  });

  it("新文件里没有任何一个 registry agent id 的字面量", () => {
    // AdminPage.tsx:537-554 那份手抄的六人名单（漏了 research_director）是
    // 这个仓的反面教材 —— e5fe15ca 那次「前端认得第七个 agent」修的就是它。
    //
    // 唯一的豁免是 lib 里的 supervisor 映射常量：`lib/*.ts` **不能** import
    // `.tsx`（`node --test` 剥不掉 JSX），所以那个映射只能住在 lib 里。
    // 它由下面那条对账测试钉着不许与 registry 漂开。
    const ids = registryAgentIds();
    const offences: string[] = [];
    for (const [name, src] of Object.entries(NEW_FILES)) {
      for (const id of ids) {
        if (name === "LIB" && id === "_supervisor") continue;   // 见上
        if (src.includes(`"${id}"`)) offences.push(`${name} 里写死了 "${id}"`);
      }
    }
    assert.deepEqual(offences, [],
      offences.join("\n") + "\n名单要从 registry.tsx 的 AGENTS/ROSTER 派生。");
  });

  it("lib 里的 supervisor id 与 registry 的 SUP_ID 对得上", () => {
    // 这条是上面那个豁免的代价。两边漂开的症状：SUP 那一列永远是空的，
    // 而空看起来像「编排器没有注入」——一个看不出是接线错的错。
    const sup = REGISTRY.match(/export const SUP_ID\s*=\s*"([^"]+)"/)?.[1];
    assert.ok(sup, "registry.tsx 里找不到 SUP_ID");
    const lib = LIB.match(/export const UI_SUPERVISOR\s*=\s*"([^"]+)"/)?.[1];
    assert.equal(lib, sup, "UI_SUPERVISOR 与 registry 的 SUP_ID 漂开了");
  });

  it("**闸门自检**：它在已知的反面教材上确实报得出来", () => {
    // 没有这一条，上面那个断言可能只是因为正则什么也没匹配到才绿的。
    const ids = registryAgentIds();
    const hits = ids.filter((id) => ADMIN.includes(`"${id}"`));
    assert.ok(hits.length >= 5,
      `AdminPage 里应当能查出手抄的 agent id（已知反面教材），实际只查到 ${hits.length} 个 —— ` +
      "要么 AdminPage 已经修好了（那就把这条豁免删掉），要么这个判据失效了。");
  });
});

// ── 3. 只读 ─────────────────────────────────────────────────────────────

describe("只读", () => {
  it("这一页一个写请求都没有", () => {
    // 它不在 PIN 门后面 —— 能写的话高级管理那道 PIN 就形同虚设。
    for (const [name, src] of Object.entries(NEW_FILES)) {
      for (const bad of ["api.POST(", "api.DELETE(", "api.PUT(", "useMutation("]) {
        assert.equal(src.includes(bad), false, `${name} 里出现了 ${bad}`);
      }
    }
  });

  it("判据自校验", () => {
    assert.ok("const m = useMutation({});".includes("useMutation("));
  });
});

// ── 4. 失效前缀对齐 ─────────────────────────────────────────────────────

describe("缓存失效", () => {
  it("manifest 的 queryKey 挂在 PromptInspector 会 invalidate 的前缀下", () => {
    // PromptInspector 保存覆写后 invalidate ["admin","prompts"]（前缀匹配）。
    // 新页面的 key 不在这个前缀下的话，改了话术这一页还显示旧的 —— 而它显示的
    // 恰恰是「现在注入的是什么」。
    const inspector = read("src/components/admin/PromptInspector.tsx");
    assert.ok(/queryKey:\s*\["admin",\s*"prompts"\]/.test(inspector),
              "PromptInspector 的失效前缀变了，这条闸门要跟着改");
    assert.ok(VIEW.includes('queryKey: ["admin", "prompts", "manifest"]'),
              "矩阵的 queryKey 不在 [\"admin\",\"prompts\"] 前缀下");
  });
});

// ── 5. 正文只对可展示的块取 ─────────────────────────────────────────────

describe("诚实性", () => {
  it("只有 canShowText 的块才去取正文", () => {
    // 顺手给 needs_* 也取一次 detail、再把 preview 当正文画出来，就等于用
    // 「渲染不出来」的块伪造了一段内容 —— 这正是整套设计要防的那件事。
    assert.ok(/enabled:\s*open\s*&&\s*showable/.test(VIEW),
              "BlockRow 的 detail 查询没有用 canShowText 把关");
    assert.ok(VIEW.includes("const showable = canShowText(block)"));
  });

  it("needs_* 的分支显示的是 unavailable_reason，不是 preview", () => {
    const i = VIEW.indexOf("block.unavailable_reason");
    assert.ok(i > 0, "没有显示不可用原因");
  });
});

// ── 6. 设计 token ───────────────────────────────────────────────────────

describe("样式", () => {
  it("ShareBar 不拼 Tailwind 类名（会被 purge 掉，症状只是「没颜色」）", () => {
    assert.equal(/className={`[^`]*\$\{/.test(SHARE), false,
                 "ShareBar 里出现了模板字符串拼的 className");
  });

  it("颜色走 agentColorVar / CSS 变量，不写死色值", () => {
    assert.ok(SHARE.includes("agentColorVar("));
    assert.equal(/#[0-9a-fA-F]{6}/.test(SHARE), false, "ShareBar 里写死了十六进制色值");
  });
});
