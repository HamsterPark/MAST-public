// ════════════════════════════════════════════════════════════════════════════
// src/lib/stickyTab.ts — 「离开再回来，还停在刚才那一页吗」。
//
// 要求：从代理对话离开再回来，仍停在代理对话栏目。
//
// 这一组几乎全在钉**读时校验**。存进去的 id 会过期(子页改名、被合并进别的栏目、
// 干脆删掉 —— #34 就要合并五组大标签),而一个页面拿着不认识的 id 去渲染,得到的
// 是一片空白或者一条所有段都不高亮的 tab 条。那不像「配置过期」,像「这一页坏了」,
// 报上来的也会是后面那句 —— 于是排查方向从一开始就是错的。
//
//     cd frontend && npm run test:unit
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import {
  pickValidTab,
  readStickyTab,
  stickyTabKey,
  writeStickyTab,
  STICKY_TAB_PREFIX,
} from "../src/lib/stickyTab.ts";

const HERE = fileURLToPath(new URL(".", import.meta.url));

const TABS = ["chat", "agentchat", "topology"] as const;
type Tab = (typeof TABS)[number];

/** 一个够用的 localStorage 替身。`fail` 打开就模拟隐私模式。 */
function fakeStorage(seed: Record<string, string> = {}, fail = false) {
  const map = new Map(Object.entries(seed));
  return {
    getItem(k: string) {
      if (fail) throw new DOMException("denied", "SecurityError");
      return map.has(k) ? map.get(k)! : null;
    },
    setItem(k: string, v: string) {
      if (fail) throw new DOMException("denied", "SecurityError");
      map.set(k, v);
    },
    _map: map,
  };
}

function withStorage<T>(s: ReturnType<typeof fakeStorage>, fn: () => T): T {
  const g = globalThis as { localStorage?: unknown };
  const prev = g.localStorage;
  g.localStorage = s as unknown as Storage;
  try {
    return fn();
  } finally {
    if (prev === undefined) delete g.localStorage;
    else g.localStorage = prev;
  }
}

describe("stickyTabKey", () => {
  it("命名空间前缀固定", () => {
    assert.equal(stickyTabKey("agents"), `${STICKY_TAB_PREFIX}agents`);
  });

  it("嵌套子页用点分层,键长得像它在界面里的位置", () => {
    // 下一个人打开 devtools 时得认得出哪个键是哪个 —— 不然改子页时没人知道
    // 该动哪一条。
    assert.equal(stickyTabKey("agents.chat.mode"), "mast.subtab.agents.chat.mode");
  });
});

describe("pickValidTab", () => {
  it("认得的 id 原样返回", () => {
    assert.equal(pickValidTab("agentchat", TABS, "chat"), "agentchat");
  });

  it("不认得的 id 退回默认值,而不是原样返回", () => {
    // 这一条是整个文件的要害。`stored ?? fallback` 这个更短的写法在这里是错的:
    // 它只处理「没存过」,不处理「存的是上一版的名字」。
    assert.equal(pickValidTab("this_tab_was_deleted", TABS, "chat"), "chat");
  });

  it("空串 / null / undefined 都退默认值", () => {
    assert.equal(pickValidTab("", TABS, "chat"), "chat");
    assert.equal(pickValidTab(null, TABS, "chat"), "chat");
    assert.equal(pickValidTab(undefined, TABS, "chat"), "chat");
  });

  it("非字符串不当成 id", () => {
    assert.equal(pickValidTab(7 as unknown as string, TABS, "chat"), "chat");
  });

  it("名单为空时一律退默认值,不抛", () => {
    assert.equal(pickValidTab("chat", [] as readonly Tab[], "chat"), "chat");
  });

  it("fallback 不在名单里也原样返回 —— 不替调用方改默认值", () => {
    // 悄悄改掉一个写错的默认值,只会让「默认值写错了」变成一个查不到的问题。
    assert.equal(
      pickValidTab(null, TABS, "not_a_tab" as Tab),
      "not_a_tab",
    );
  });

  it("不做前缀 / 大小写的宽松匹配", () => {
    // 宽松匹配会让「agentchat」和「agent」互相认领,而两者是不同的页。
    assert.equal(pickValidTab("AgentChat", TABS, "chat"), "chat");
    assert.equal(pickValidTab("agent", TABS, "chat"), "chat");
  });
});

describe("readStickyTab / writeStickyTab", () => {
  it("写进去读得回来", () => {
    const s = fakeStorage();
    withStorage(s, () => {
      writeStickyTab("agents", "agentchat");
      assert.equal(readStickyTab("agents", TABS, "chat"), "agentchat");
    });
  });

  it("两个栏目的记忆互不干扰", () => {
    const s = fakeStorage();
    withStorage(s, () => {
      writeStickyTab("agents", "agentchat");
      writeStickyTab("records", "export");
      assert.equal(readStickyTab("agents", TABS, "chat"), "agentchat");
      assert.equal(
        readStickyTab("records", ["experiments", "export"] as const, "experiments"),
        "export",
      );
    });
  });

  it("库里那条过期记忆不会卡住页面", () => {
    const s = fakeStorage({ "mast.subtab.agents": "workspace_v1_removed" });
    withStorage(s, () => {
      assert.equal(readStickyTab("agents", TABS, "chat"), "chat");
    });
  });

  it("storage 被禁用时读不抛,退默认值", () => {
    // 一个记不住偏好的浏览器仍然要能用这个软件 —— 这条和「UI 绝不冻结」
    // 是同一条纪律的两个面。
    const s = fakeStorage({}, true);
    withStorage(s, () => {
      assert.equal(readStickyTab("agents", TABS, "chat"), "chat");
    });
  });

  it("storage 被禁用时写不抛", () => {
    const s = fakeStorage({}, true);
    withStorage(s, () => {
      assert.doesNotThrow(() => writeStickyTab("agents", "agentchat"));
    });
  });

  it("根本没有 localStorage 这个全局(SSR / node)时也不抛", () => {
    const g = globalThis as { localStorage?: unknown };
    const prev = g.localStorage;
    delete g.localStorage;
    try {
      assert.equal(readStickyTab("agents", TABS, "chat"), "chat");
      assert.doesNotThrow(() => writeStickyTab("agents", "chat"));
    } finally {
      if (prev !== undefined) g.localStorage = prev;
    }
  });
});

// ════════════════════════════════════════════════════════════════════════════
// 结构闸门 —— 下一页别再悄悄漏掉
//
// 这个缺陷有一个讨厌的性质:**每一页都要各自记得**。它不崩、不报错、类型检查
// 全绿、截图完美,唯一的症状是一个人回到某一页发现自己被弹回第一个子页 ——
// 而他多半不会为这个报 bug,只会多点一下。#32 之所以被报上来,是因为「代理对话」
// 那一页他一天要回去几十次。
//
// 所以这一条不测行为,测**源码里有没有接线**。同 nav.test.ts:那类缺陷也是
// 「路由照样解析、组件照样渲染」,只能靠断言守。
// ════════════════════════════════════════════════════════════════════════════

const PAGES_DIR = join(HERE, "..", "src", "pages");

/**
 * 有子页切换器、却**故意**不记忆的页面。
 *
 * 空的。留着这个名单不是形式主义 —— 有了它,「不记忆」永远是某人写下来的决定,
 * 而不是一个没人注意到的遗漏。真有该豁免的页面时,在这里写清理由。
 */
const INTENTIONALLY_FORGETFUL: readonly string[] = [];

describe("每一页的子页切换器都接了记忆", () => {
  it("没有哪一页用了 SubTabs 却没用 useStickyTab", () => {
    const offenders: string[] = [];
    for (const f of readdirSync(PAGES_DIR)) {
      if (!f.endsWith(".tsx")) continue;
      if (INTENTIONALLY_FORGETFUL.includes(f)) continue;
      const src = readFileSync(join(PAGES_DIR, f), "utf8");
      if (!src.includes("<SubTabs")) continue;
      if (!src.includes("useStickyTab")) offenders.push(f);
    }
    assert.deepEqual(
      offenders,
      [],
      "这些页面的子页每次回来都会被复位。改用 useStickyTab,"
        + "或者把它加进 INTENTIONALLY_FORGETFUL 并写下理由:\n"
        + offenders.join("\n"),
    );
  });

  it("每个 useStickyTab 的合法名单都是从 tab 列表 .map 出来的", () => {
    // 另抄一份 id 名单出来是这套机制唯一的静默失效方式:抄的那份会和 tab 条漂开,
    // 而漂开之后受害的那个子页**只是记不住**,别的全都正常。所以调用点长什么样
    // 本身要被钉住 —— 名单必须是 `X.map(...)`,不能是手写的字面量数组。
    const bad: string[] = [];
    for (const f of readdirSync(PAGES_DIR)) {
      if (!f.endsWith(".tsx")) continue;
      const src = readFileSync(join(PAGES_DIR, f), "utf8");
      for (const m of src.matchAll(/useStickyTab<[^>]*>\(\s*([\s\S]{0,200}?)\);/g)) {
        const call = m[1] ?? "";
        if (!call.includes(".map(")) bad.push(`${f}: ${call.replace(/\s+/g, " ").trim()}`);
      }
    }
    assert.deepEqual(bad, [], "合法名单要从 tab 列表派生,别手抄:\n" + bad.join("\n"));
  });
});
