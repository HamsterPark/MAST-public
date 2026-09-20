// `lib/contextInjection.ts` 的单测。
//
// 这里算的每一件事都有**算错的方式**，所以每一条都配一句「换成那个更简单的写法
// 会怎么坏」：
//
//   · 占比各自 Math.round → 加总 99 或 101，而一条加不满的条会让人以为漏了一块；
//   · 用 .length 切文本   → 一个 emoji 就让所有段错一位（Python 数码点，JS 数码元）；
//   · needs_* 的块当成 0  → 条画得比实际短，而「未知」和「零」是两件事；
//   · 忘了剔 ai:response  → 历史多算一条，占比跟着错；
//   · 不映射 supervisor   → SUP 那一列永远空，看起来像「编排器没有注入」。

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  ALL_AGENTS,
  agentFootprint,
  appliesTo,
  buildMatrix,
  canShowText,
  composeShares,
  fmtAge,
  fmtChars,
  positionLabel,
  segmentAnchor,
  splitByBlocks,
  splitMessages,
  toBackendAgentId,
  toUiAgentId,
  whenLabel,
} from "../src/lib/contextInjection.ts";

// ── 替身 ────────────────────────────────────────────────────────────────

function block(over: Record<string, unknown> = {}) {
  return {
    id: "mw.x", label: "X", category: "middleware", agent: "",
    availability: "static", source: "", note: "", overridable: false,
    overridden: false, default_chars: 100, effective_chars: 100,
    preview: "", unavailable_reason: "",
    agents: [ALL_AGENTS], when: "always", position: "system",
    middleware: "XMiddleware", paths: ["group"], requires: "", exclusive: false,
    ...over,
  } as never;
}

function msg(role: string, content: string, over: Record<string, unknown> = {}) {
  return { role, content, chars: [...content].length, truncated: false,
           blocks: null, ...over } as never;
}

// ── supervisor 两个名字 ────────────────────────────────────────────────

describe("agent id 映射", () => {
  it("把前端的 _supervisor 换成后端的 orchestrator", () => {
    assert.equal(toBackendAgentId("_supervisor"), "orchestrator");
    assert.equal(toUiAgentId("orchestrator"), "_supervisor");
  });

  it("其余 id 原样往返", () => {
    for (const id of ["instrument_control", "literature", "buffer_summarizer"]) {
      assert.equal(toBackendAgentId(id), id);
      assert.equal(toUiAgentId(toBackendAgentId(id)), id);
    }
  });

  it("不映射的话 SUP 那一列会永远是空的（看起来像「编排器没有注入」）", () => {
    // 变异：去掉映射就等价于恒等函数，这条断言会红。
    assert.notEqual(toBackendAgentId("_supervisor"), "_supervisor");
  });
});

// ── 归属判定 ────────────────────────────────────────────────────────────

describe("appliesTo", () => {
  it("* 对任何 agent 都成立", () => {
    assert.ok(appliesTo(block({ agents: [ALL_AGENTS] }), "paper_review"));
  });

  it("**空名单对谁都不成立** —— 当成全员会把每一块发给每个人", () => {
    assert.equal(appliesTo(block({ agents: [] }), "paper_review"), false);
  });

  it("精确匹配，不做前缀匹配", () => {
    const b = block({ agents: ["paper_review"] });
    assert.ok(appliesTo(b, "paper_review"));
    assert.equal(appliesTo(b, "paper_review_extra"), false);
    assert.equal(appliesTo(b, "paper"), false);
  });
});

describe("canShowText", () => {
  it("只有 static / live 能直接显示", () => {
    assert.ok(canShowText(block({ availability: "static" })));
    assert.ok(canShowText(block({ availability: "live" })));
  });

  it("needs_* 一律不能 —— 这是诚实性铁律的类型化", () => {
    assert.equal(canShowText(block({ availability: "needs_hardware" })), false);
    assert.equal(canShowText(block({ availability: "needs_request" })), false);
  });
});

describe("词表", () => {
  it("已知值给中文", () => {
    assert.equal(whenLabel("on_mode"), "看模式");
    assert.equal(positionLabel("last_human"), "最后一条用户消息");
  });

  it("**未知值原样返回**，不写「未知」", () => {
    // 后端加一个枚举值时，页面不该变成一片「未知」——那会把新功能显示成故障。
    assert.equal(whenLabel("brand_new_value"), "brand_new_value");
    assert.equal(positionLabel("somewhere_else"), "somewhere_else");
  });
});

// ── 消息拆分 ────────────────────────────────────────────────────────────

describe("splitMessages", () => {
  it("把模型自己的输出剔出去", () => {
    const out = splitMessages([
      msg("system", "SYS"), msg("human", "hi"), msg("ai", "thinking"),
      msg("tool", "result"), msg("ai:response", "final"),
    ]);
    assert.equal(out.system.length, 1);
    assert.equal(out.history.length, 3);
    assert.equal(out.response?.content, "final");
  });

  it("**不剔的话历史会多算一条**，占比跟着错", () => {
    const out = splitMessages([msg("human", "hi"), msg("ai:response", "final")]);
    assert.equal(out.history.length, 1);
    assert.ok(!out.history.some((m: never) => (m as { role: string }).role === "ai:response"));
  });

  it("空输入不炸", () => {
    const out = splitMessages(undefined);
    assert.deepEqual([out.system, out.history, out.response], [[], [], null]);
  });
});

// ── 按块切分 ────────────────────────────────────────────────────────────

describe("splitByBlocks", () => {
  it("块乱序进来，输出按位置升序", () => {
    // 替身**刻意乱序**：顺序写对的替身会让「输出有序」这条断言恒真。
    const text = "AAABBCCCC";
    const { segments } = splitByBlocks(text, 9, [
      { id: "b", start: 3, end: 5, chars: 2 },
      { id: "a", start: 0, end: 3, chars: 3 },
    ] as never);
    assert.deepEqual(segments.map((s) => s.id), ["a", "b", null]);
    assert.deepEqual(segments.map((s) => s.text), ["AAA", "BB", "CCCC"]);
  });

  it("空洞补成未归属段，而不是丢掉", () => {
    const { segments } = splitByBlocks("xxYYzz", 6,
      [{ id: "mid", start: 2, end: 4, chars: 2 }] as never);
    assert.deepEqual(segments.map((s) => s.id), [null, "mid", null]);
    // 各段拼回去必须等于全文 —— 丢段的话「加起来对不上」就看不见了。
    assert.equal(segments.map((s) => s.text).join(""), "xxYYzz");
  });

  it("**按码点切**，不按 UTF-16 码元", () => {
    // 后端 Python 的 len() 数码点；JS 的 .length 把 astral 字符数成 2。
    // 用 .slice 会让 emoji 之后的每一段都错一位。
    const text = "🔬ABC";           // 码点 4 个，.length 是 5
    const { segments } = splitByBlocks(text, 4,
      [{ id: "head", start: 0, end: 1, chars: 1 }] as never);
    assert.equal(segments[0]?.text, "🔬");
    assert.equal(segments[1]?.text, "ABC");
  });

  it("块超出可见文本时钳位并说出来（原文被截断过）", () => {
    const { segments, issues } = splitByBlocks("short", 20_000,
      [{ id: "big", start: 0, end: 19_000, chars: 19_000 }] as never);
    assert.equal(segments[0]?.clipped, true);
    assert.equal(segments[0]?.text, "short");
    assert.ok(issues.some((i) => i.kind === "beyond_content"));
  });

  it("重叠的块记一条 issue，但不抛", () => {
    const { issues } = splitByBlocks("abcdef", 6, [
      { id: "a", start: 0, end: 4, chars: 4 },
      { id: "b", start: 2, end: 6, chars: 4 },
    ] as never);
    assert.ok(issues.some((i) => i.kind === "overlap"));
  });

  it("没有块信息 → 一整段未归属，且不报 issue", () => {
    const { segments, issues } = splitByBlocks("whole", 5, null);
    assert.deepEqual(segments.map((s) => s.id), [null]);
    assert.equal(issues.length, 0);
  });

  it("各块声称的字符多于消息本身 → 记一条 issue", () => {
    const { issues } = splitByBlocks("abc", 3,
      [{ id: "a", start: 0, end: 3, chars: 999 }] as never);
    assert.ok(issues.some((i) => i.kind === "chars_mismatch"));
  });
});

// ── 占比 ────────────────────────────────────────────────────────────────

describe("composeShares", () => {
  const labelFor = (id: string) => id.toUpperCase();

  it("百分比**恰好加总 100**（最大余数法）", () => {
    // 变异：各自 Math.round 时这三个 1/3 会加成 99 或 101。
    const shares = composeShares({
      segments: [
        { id: "a", start: 0, end: 1, text: "", chars: 1, clipped: false },
        { id: "b", start: 0, end: 1, text: "", chars: 1, clipped: false },
        { id: "c", start: 0, end: 1, text: "", chars: 1, clipped: false },
      ],
      history: [], toolsChars: null, labelFor,
    });
    assert.equal(shares.reduce((s, x) => s + x.pct, 0), 100);
  });

  it("agent 自己的系统提示归到 static，中间件块归到 block", () => {
    const shares = composeShares({
      segments: [
        { id: "agent.ic.system", start: 0, end: 1, text: "", chars: 10, clipped: false },
        { id: "mw.tip_context", start: 0, end: 1, text: "", chars: 5, clipped: false },
      ],
      history: [], toolsChars: null, labelFor,
    });
    assert.deepEqual(shares.map((s) => s.key), ["static", "block"]);
  });

  it("toolsChars 为 null 时**不画这一段**（不是画成 0%）", () => {
    const shares = composeShares({
      segments: [{ id: "a", start: 0, end: 1, text: "", chars: 10, clipped: false }],
      history: [], toolsChars: null, labelFor,
    });
    assert.equal(shares.some((s) => s.key === "tools"), false);
  });

  it("toolsChars 为 0 时画出来 —— 「没量到」和「量到了是 0」不一样", () => {
    const shares = composeShares({
      segments: [{ id: "a", start: 0, end: 1, text: "", chars: 10, clipped: false }],
      history: [], toolsChars: 0, labelFor,
    });
    assert.ok(shares.some((s) => s.key === "tools"));
  });

  it("只有工具面那一段标成估算", () => {
    const shares = composeShares({
      segments: [{ id: "a", start: 0, end: 1, text: "", chars: 10, clipped: false }],
      history: [msg("human", "hi")] as never, toolsChars: 5, labelFor,
    });
    assert.deepEqual(shares.filter((s) => s.estimated).map((s) => s.key), ["tools"]);
  });

  it("历史用**原长** chars，不用可能被截过的 content.length", () => {
    const truncated = msg("tool", "short-visible", { chars: 50_000, truncated: true });
    const shares = composeShares({
      segments: [], history: [truncated] as never, toolsChars: null, labelFor,
    });
    assert.equal(shares[0]?.chars, 50_000);
  });

  it("什么都没有时返回空数组，不返回一条 0% 的假条", () => {
    assert.deepEqual(
      composeShares({ segments: [], history: [], toolsChars: null, labelFor }), []);
  });
});

// ── 矩阵 ────────────────────────────────────────────────────────────────

describe("buildMatrix", () => {
  const matrix = {
    agents: [{ id: "instrument_control" }, { id: "paper_review" }],
    blocks: [
      block({ id: "mw.shared", agents: [ALL_AGENTS] }),
      block({ id: "mw.tip", agents: ["instrument_control"], exclusive: true }),
    ],
    cells: {
      "mw.shared": { instrument_control: true, paper_review: true },
      "mw.tip": { instrument_control: true, paper_review: false },
    },
  } as never;

  it("列数恒等于传进去的列", () => {
    const rows = buildMatrix(matrix, ["instrument_control", "paper_review"]);
    for (const r of rows) assert.equal(r.cells.length, 2);
  });

  it("分得出「全员」与「专属」", () => {
    const rows = buildMatrix(matrix, ["instrument_control", "paper_review"]);
    assert.deepEqual(rows.map((r) => r.shared), [true, false]);
    assert.deepEqual(rows.map((r) => r.exclusive), [false, true]);
  });

  it("cells 里没有的列当成 false，不当成 true", () => {
    const rows = buildMatrix(matrix, ["instrument_control", "brand_new_agent"]);
    assert.deepEqual(rows[0]?.cells, [true, false]);
  });

  it("空输入返回空数组", () => {
    assert.deepEqual(buildMatrix(undefined, ["a"]), []);
  });
});

describe("agentFootprint", () => {
  const matrix = {
    agents: [], cells: {},
    blocks: [
      block({ id: "s", agents: [ALL_AGENTS], effective_chars: 100 }),
      block({ id: "x", agents: ["instrument_control"], exclusive: true,
              effective_chars: 50 }),
      block({ id: "live", agents: ["instrument_control"], exclusive: true,
              availability: "needs_hardware", effective_chars: 0 }),
      block({ id: "other", agents: ["paper_review"], exclusive: true,
              effective_chars: 999 }),
    ],
  } as never;

  it("只算这个 agent 收得到的块", () => {
    const fp = agentFootprint(matrix, "instrument_control");
    assert.equal(fp.knownChars, 150);
  });

  it("**needs_* 计数，不当成 0** —— 「未知」和「零」是两件事", () => {
    const fp = agentFootprint(matrix, "instrument_control");
    assert.equal(fp.unknownCount, 1);
    // 变异：把它当 0 加进去，条会画得比实际短，而且看不出来。
    assert.equal(fp.knownChars, 150);
  });

  it("数得清全员块与专属块", () => {
    const fp = agentFootprint(matrix, "instrument_control");
    assert.equal(fp.sharedCount, 1);
    assert.equal(fp.exclusiveCount, 2);
  });
});

// ── 杂项 ────────────────────────────────────────────────────────────────

describe("格式化", () => {
  it("fmtChars", () => {
    assert.equal(fmtChars(0), "0 字符");
    assert.equal(fmtChars(999), "999 字符");
    assert.equal(fmtChars(1000), "1.0k 字符");
    assert.equal(fmtChars(null), "—");
    assert.equal(fmtChars(undefined), "—");
  });

  it("fmtAge 的三个边界", () => {
    assert.equal(fmtAge(59), "59 秒前");
    assert.equal(fmtAge(60), "1 分钟前");
    assert.equal(fmtAge(3600), "1.0 小时前");
    assert.equal(fmtAge(null), "—");
  });
});

describe("segmentAnchor", () => {
  it("确定性：同一个 id 每次都得到同一个锚点", () => {
    // 不是幂等 —— 它是「id → 锚点」的一次映射，拿自己的输出再调一次是调用方的
    // bug。这里钉的是「同一个 id 在跳转端与渲染端算出来的是同一个字符串」。
    assert.equal(segmentAnchor("mw.a"), segmentAnchor("mw.a"));
    assert.ok(segmentAnchor("mw.a").startsWith("ctx-seg-"));
  });

  it("**不同 id 不许撞** —— 把 . 换成 _ 会让 mw.a.b 和 mw.a_b 变成一个", () => {
    assert.notEqual(segmentAnchor("mw.a.b"), segmentAnchor("mw.a_b"));
  });
});
