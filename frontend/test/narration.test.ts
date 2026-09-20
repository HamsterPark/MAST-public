// ════════════════════════════════════════════════════════════════════════════
// 旁白插在**哪一条消息后面** —— src/lib/narration.ts。
//
// 这一份存在的理由是设计文档 §10.7：本仓刚数过「一下午四个零消费者」——
// 生产方接好了、事件发出去了、而**没有任何东西读它**。所以每个阶段的验收都要
// 有一条**消费者可达**的断言，不能只断言端点返回了那一行。这就是那条断言：
// 前端真正用来排版的那个纯函数，拿到那一批数据之后画出来的是什么。
//
// 这里的断言几乎全是**变异测试**：钉的不是「能插进去」，而是那几个更简单的写法
// 各自会怎么坏 —— 它们都不崩、类型全绿，症状只有「旁白偶尔排在奇怪的位置」。
//
//     cd frontend && npm run test:unit
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  byEventTime,
  fmtClock,
  isNarrationFor,
  mergeBatch,
  mergeNarration,
  narrationImageUrl,
  nextCursor,
  toneClass,
  type NarrationItem,
  NARRATION_MEMO_MAX,
  forgetAllNarration,
  recallNarration,
  rememberNarration,
} from "../src/lib/narration.ts";

const msg = (role: string, content: string) => ({ role, content });

// 替身自校验：这些 `nk` 必须是**后端模板注册表里真有的** kind。
// 一个真组件永远不会返回的值，可以让一条端到端测试「证明」一条结构上不可达的
// 快乐路径 —— 本仓今晚刚被这件事咬过（`sharpness(verdict="good")` 绿了几个月）。
const REAL_KINDS = [
  "scan_at",
  "scan_start",
  "scan_milestone",
  "tip_pulse",
  "bias_pulse",
  "poke",
  "relocate",
  "step_failed",
];

function n(seq: number, anchor: number, nk = "tip_pulse"): NarrationItem {
  assert.ok(
    REAL_KINDS.includes(nk),
    `测试替身用了一个后端不存在的 kind: ${nk} —— ` +
      `它会让这条用例证明一条真机上到不了的路径。真的有：${REAL_KINDS.join(", ")}`,
  );
  return { seq, t: seq, text: `第 ${seq} 条`, nk, anchor, tone: "info" };
}

describe("mergeNarration", () => {
  const msgs = [msg("user", "a"), msg("assistant", "b"), msg("assistant", "c")];

  it("anchor = n 插在第 n 条**之后**", () => {
    const lanes = mergeNarration(msgs, [n(1, 2)]);
    assert.deepEqual(
      lanes.map((l) => (l.kind === "message" ? l.message.content : "旁白")),
      ["a", "b", "旁白", "c"],
    );
  });

  it("anchor = 0 插在最前面 —— 那一刻转录确实还是空的", () => {
    const lanes = mergeNarration(msgs, [n(1, 0)]);
    assert.equal(lanes[0]?.kind, "narration");
  });

  it("anchor = -1（没有活跃回合）追加到末尾，不是插到最前面", () => {
    // 把 -1 当成 0 是最容易犯的那个错：唤醒调度/群跑发的旁白会全部堆在
    // 对话最顶上，看起来像是「历史被改写了」。
    const lanes = mergeNarration(msgs, [n(1, -1)]);
    assert.equal(lanes[lanes.length - 1]?.kind, "narration");
    assert.equal(lanes[0]?.kind, "message");
  });

  it("anchor 超过消息条数（回合正在跑）也落在末尾", () => {
    const lanes = mergeNarration(msgs, [n(1, 99)]);
    assert.equal(lanes[lanes.length - 1]?.kind, "narration");
  });

  it("同一个 anchor 上的多条按 seq 排，不按数组顺序", () => {
    const lanes = mergeNarration(msgs, [n(7, 1), n(3, 1), n(5, 1)]);
    const seqs = lanes.filter((l) => l.kind === "narration").map((l) => (l as { item: NarrationItem }).item.seq);
    assert.deepEqual(seqs, [3, 5, 7]);
  });

  it("一条消息都没有时，旁白照样画得出来", () => {
    // 长任务在操作员发完一句话之后跑起来，快照可能还是空的。这时旁白是屏幕上
    // 唯一的内容 —— 它绝不能因为「没有可挂靠的消息」而消失。
    const lanes = mergeNarration([], [n(1, -1), n(2, 0)]);
    assert.equal(lanes.length, 2);
    assert.ok(lanes.every((l) => l.kind === "narration"));
  });

  it("没有旁白时，输出就是原来的消息序列", () => {
    const lanes = mergeNarration(msgs, []);
    assert.deepEqual(
      lanes.map((l) => (l.kind === "message" ? l.message.content : "旁白")),
      ["a", "b", "c"],
    );
  });

  it("每条消息都恰好出现一次（不丢、不重）", () => {
    const lanes = mergeNarration(msgs, [n(1, 0), n(2, 1), n(3, 3), n(4, -1)]);
    const kept = lanes.filter((l) => l.kind === "message").map((l) => (l as { message: { content: string } }).message.content);
    assert.deepEqual(kept, ["a", "b", "c"]);
  });
});

describe("nextCursor", () => {
  it("只往前", () => {
    assert.equal(nextCursor(10, 12), 12);
    // 后退会让同一批旁白被重复插进列表 —— 症状是「旁白出现两遍」。
    assert.equal(nextCursor(10, 4), 10);
  });

  it("latest_seq 缺失/非数时原地不动", () => {
    assert.equal(nextCursor(10, undefined), 10);
    assert.equal(nextCursor(10, NaN), 10);
  });
});

describe("mergeBatch", () => {
  it("按 seq 去重 —— WS 推送和轮询兜底会送来同一条", () => {
    const merged = mergeBatch([n(1, 0), n(2, 0)], [n(2, 0), n(3, 0)]);
    assert.deepEqual(merged.map((i) => i.seq), [1, 2, 3]);
  });

  it("空批次不改变已有的（也不新建数组）", () => {
    const existing = [n(1, 0)];
    assert.equal(mergeBatch(existing, []), existing);
  });
});

describe("isNarrationFor", () => {
  it("只刷我正在看的那个会话", () => {
    assert.equal(isNarrationFor({ conversation_id: "c1" }, "c1"), true);
    assert.equal(isNarrationFor({ conversation_id: "c2" }, "c1"), false);
  });

  it("没选会话时什么都不刷", () => {
    assert.equal(isNarrationFor({ conversation_id: "c1" }, null), false);
    assert.equal(isNarrationFor(null, "c1"), false);
  });
});

describe("narrationImageUrl", () => {
  it("has_image 为假时**不发请求** —— 返回 null 而不是一个会 404 的 URL", () => {
    assert.equal(narrationImageUrl({ ...n(1, 0), has_image: false }, "c1"), null);
  });

  it("有图时带上会话 id（取图端点按会话鉴权）", () => {
    const url = narrationImageUrl({ ...n(1, 0), has_image: true }, "c 1");
    assert.equal(url, "/api/chat/narration-image/1?conversation_id=c%201&px=240");
  });
});

describe("toneClass", () => {
  it("未知 tone 退回中性，不猜", () => {
    assert.equal(toneClass("nonsense"), toneClass("info"));
    assert.notEqual(toneClass("warn"), toneClass("info"));
  });
});

// ════════════════════════════════════════════════════════════════════════════
//  按**事件发生时刻**排序（2026-08-12，操作员报的「输出顺序有点混乱」）
//
//  查下来三件事互相独立：
//    · 执行顺序没问题 —— `completed_steps` 里 A(脉冲)→B(验证) 严格交替；
//    · 存储顺序没问题 —— 53 条旁白，`seq` 序与 `t` 序零逆序；
//    · **错在两个发出方对同一时刻的观测有时差** —— composite 执行器同步发
//      （延迟 ~0），视觉监视器要先抓帧再跑模型才发（滞后 0.5–1.5 s）。
//      `seq` 记的是入队顺序，于是把「什么时候知道」当成了「什么时候发生」。
//
//  后端现在让知道真实时刻的发出方传 `event_t`；这一组钉的是**前端必须按它排**
//  —— 否则后端那半修了等于没修（「生产方接上了，消费方不存在」，本仓已四次）。
// ════════════════════════════════════════════════════════════════════════════

describe("按事件时刻排序", () => {
  /** 三条示例实录（时间戳已改为编造值），`t` 是后端现在会写的那个数。 */
  const REAL: NarrationItem[] = [
    // seq 10 入队最早，但它描述的事（要打脉冲）发生在扫描结束**之后**
    { seq: 10, t: 2000, nk: "bias_pulse", text: "我们要打一发 10 V / 500 ms 的偏压脉冲。", anchor: 1 },
    // seq 11 入队晚，但事件（图扫完）发生得早 —— 抓帧时刻，不是说出来的时刻
    { seq: 11, t: 1000, nk: "scan_done", text: "这张图扫完了。", anchor: 1 },
    { seq: 12, t: 3000, nk: "scan_start", text: "我们开始扫一张图。", anchor: 1 },
  ];

  it("扫完排在下一发脉冲**前面** —— 物理上它必须先发生", () => {
    const lane = mergeNarration([msg("user", "跑")], REAL);
    const order = lane
      .filter((l) => l.kind === "narration")
      .map((l) => (l as { item: NarrationItem }).item.nk);
    assert.deepEqual(order, ["scan_done", "bias_pulse", "scan_start"]);
  });

  it("按 seq 排会得到那个错的顺序 —— 记下反例", () => {
    // 这一条不测产品代码，它解释上一条在防什么。
    const bySeq = [...REAL].sort((a, b) => a.seq - b.seq).map((i) => i.nk);
    assert.deepEqual(bySeq, ["bias_pulse", "scan_done", "scan_start"]);
    assert.notDeepEqual(bySeq, [...REAL].sort(byEventTime).map((i) => i.nk));
  });

  it("同刻用 seq 裁决 —— 排序必须稳定", () => {
    const a: NarrationItem = { seq: 7, t: 100, text: "a" };
    const b: NarrationItem = { seq: 3, t: 100, text: "b" };
    assert.ok(byEventTime(a, b) > 0);
    assert.ok(byEventTime(b, a) < 0);
  });

  it("`t` 缺失/垃圾时只比 seq —— 不许把它顶到最前面", () => {
    // 缺 t 的那条若被当成 t=0，会永远排在最前，把一次真实事件挤到后面去。
    const good: NarrationItem = { seq: 5, t: 1786541600, text: "有时间" };
    for (const bad of [undefined, NaN, Infinity, "x"] as unknown[]) {
      const junk = { seq: 9, t: bad, text: "无时间" } as unknown as NarrationItem;
      assert.ok(byEventTime(good, junk) < 0, `t=${String(bad)} 被排到了前面`);
    }
  });

  it("mergeBatch 也按事件时刻排，且仍按 seq 去重", () => {
    const first: NarrationItem[] = [{ seq: 2, t: 200, text: "晚" }];
    const second: NarrationItem[] = [
      { seq: 3, t: 100, text: "早" },
      { seq: 2, t: 200, text: "晚（重复送达）" },
    ];
    const out = mergeBatch(first, second);
    assert.equal(out.length, 2, "同一 seq 被当成两条了");
    assert.deepEqual(out.map((i) => i.seq), [3, 2], "没有按事件时刻排");
  });
});

describe("fmtClock", () => {
  it("epoch 秒 → HH:MM:SS", () => {
    const t = new Date(2026, 7, 12, 21, 33, 21).getTime() / 1000;
    assert.equal(fmtClock(t), "21:33:21");
  });

  it("毫秒时间戳也认得出来 —— 否则会显示成 55000 年", () => {
    const ms = new Date(2026, 7, 12, 21, 33, 21).getTime();
    assert.equal(fmtClock(ms), "21:33:21");
  });

  it("取不到就不显示，**绝不显示 1970**", () => {
    // 一个假时间比没有时间坏：它会被当成真的去推理。
    for (const bad of [undefined, null, NaN, 0, -1, "x", {}] as unknown[]) {
      assert.equal(fmtClock(bad), "", `${String(bad)} 渲染出了一个时刻`);
    }
  });
});

// ── 切标签页再回来,旁白还在(2026-08-17) ────────────────────────────────────

describe("旁白跨挂载留存", () => {
  it("切走再回来:条目和游标都还在", () => {
    forgetAllNarration();
    rememberNarration("c1", [{ seq: 1 }, { seq: 2 }], 2);
    const back = recallNarration<{ seq: number }>("c1");
    assert.equal(back.items.length, 2);
    assert.equal(back.cursor, 2);
  });

  it("**这才是那个 bug**:游标丢了会让前面几百条永远补不回来", () => {
    // 要求:「切换标签页再回来的时候旁白就没有了。」
    //
    // 组件卸载丢的不只是条目,还有游标。回来时 react-query 先喂上一次**增量**
    // 请求的缓存(只含最后两条),`useEffect` 照单全收并把游标推到 latest_seq
    // —— 于是前面的条目既不在内存里,也永远不会再被请求。
    forgetAllNarration();
    const fresh = recallNarration("never-seen");
    assert.equal(fresh.cursor, 0, "没见过的会话必须从 0 开始,否则会跳过开头");
    assert.deepEqual(fresh.items, []);
  });

  it("换会话不串账", () => {
    forgetAllNarration();
    rememberNarration("a", [{ seq: 9 }], 9);
    rememberNarration("b", [{ seq: 1 }], 1);
    assert.equal(recallNarration<{ seq: number }>("a").items[0].seq, 9);
    assert.equal(recallNarration<{ seq: number }>("b").items[0].seq, 1);
  });

  it("只留最近几条会话 —— 一条长任务几百条旁白,无上限会吃内存", () => {
    forgetAllNarration();
    for (let i = 0; i < NARRATION_MEMO_MAX + 2; i++) {
      rememberNarration(`conv${i}`, [{ seq: i }], i);
    }
    // 最早的那条应当被挤掉,最新的必须还在。
    assert.deepEqual(recallNarration("conv0").items, []);
    assert.equal(
      recallNarration<{ seq: number }>(`conv${NARRATION_MEMO_MAX + 1}`).items.length, 1);
  });

  it("空会话 id 不写表(否则所有未选中的会话会互相覆盖)", () => {
    forgetAllNarration();
    rememberNarration("", [{ seq: 1 }], 1);
    assert.deepEqual(recallNarration("").items, []);
  });
});

// ════════════════════════════════════════════════════════════════════════════
//  转录被压缩之后，`anchor` 指向的是**另一个坐标系**（2026-08-23）
//
//  要求：有些旁白和聊天内容前后顺序错乱，未按时间顺序显示。
//
//  取证：一次会话（编号 `a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4`，示例值）
//  跑了三个多小时、中途被压缩过。屏幕上 15 条消息、36 条旁白，而旁白身上的
//  anchor 是 8 / 13 / 28 / 33 —— 那是**压缩前**的条数。照 anchor 排出来的顺序
//  里有三处时间倒挂，最大的一处 12868 秒（3 小时 34 分）。
//
//  `t` 是不会被压缩改写的坐标，两边现在都有它（后端 `MessageClockMiddleware`
//  2026-08-12 起给消息盖戳）。这一组钉的就是「有戳就按戳排」。
// ════════════════════════════════════════════════════════════════════════════

describe("压缩过的转录：按时间归并，不按 anchor", () => {
  /** 示例转录骨架（编号 a1b2c3d4，epoch 秒取自 messages 端点的 `t`；数值已替换为示例值）。 */
  const REAL_MSGS = [
    { role: "user", content: "…会话摘要…", t: null },      // 压缩摘要，没有戳
    { role: "assistant", content: "a1", t: 8861.86 },
    { role: "assistant", content: "a2", t: 8875.60 },
    { role: "assistant", content: "a3", t: 8890.16 },
    { role: "assistant", content: "a4", t: 8965.06 },
    { role: "assistant", content: "a5", t: 9039.69 },
    { role: "assistant", content: "a6", t: 13211.98 },
    { role: "assistant", content: "a7", t: 13284.39 },
  ];

  /** 四段扫描的头一条，anchor 取自同一份示例转录（8 / 13 / 28 / 33）。 */
  const REAL_NARR: NarrationItem[] = [
    { seq: 1, t: 415.87, nk: "scan_start", text: "04:26 的那张", anchor: 8 },
    { seq: 10, t: 4829.04, nk: "scan_start", text: "05:40 的那张", anchor: 13 },
    { seq: 19, t: 9040.15, nk: "scan_start", text: "06:50 的那张", anchor: 28 },
    { seq: 28, t: 13543.02, nk: "scan_start", text: "08:05 的那张", anchor: 33 },
  ];

  /** 渲染出来的那条序列上，每一项的时刻（消息取 `t`，旁白取 `t`）。 */
  function shownTimes(lanes: ReturnType<typeof mergeNarration>): number[] {
    return lanes
      .map((l) =>
        l.kind === "narration"
          ? (l as { item: NarrationItem }).item.t
          : ((l as { message: { t?: number | null } }).message.t ?? NaN),
      )
      .filter((t) => Number.isFinite(t)) as number[];
  }

  it("画出来的顺序里一处时间倒挂都没有", () => {
    const lanes = mergeNarration(REAL_MSGS, REAL_NARR);
    const ts = shownTimes(lanes);
    const bad = ts.filter((t, i) => i > 0 && t < ts[i - 1]!);
    assert.deepEqual(
      bad,
      [],
      `显示顺序里有 ${bad.length} 处时间倒挂 —— 这正是要防的「前后顺序错乱」`,
    );
  });

  it("照 anchor 排会得到那个错的顺序 —— 记下反例", () => {
    // 这一条不测产品代码，它证明上一条防的东西真的会发生。
    const n = REAL_MSGS.length;
    const byAnchor: number[] = [];
    const at = (i: number) =>
      REAL_NARR.filter((x) => (x.anchor! < 0 || x.anchor! > n ? n : x.anchor) === i);
    for (const x of at(0)) byAnchor.push(x.t);
    for (let i = 0; i < n; i++) {
      const mt = REAL_MSGS[i]!.t;
      if (typeof mt === "number") byAnchor.push(mt);
      for (const x of at(i + 1)) byAnchor.push(x.t);
    }
    const inv = byAnchor.filter((t, i) => i > 0 && t < byAnchor[i - 1]!);
    assert.ok(inv.length > 0, "反例不再成立 —— 这组真机数据被改动过？");
    const worst = Math.max(
      ...byAnchor.map((t, i) => (i > 0 ? byAnchor[i - 1]! - t : 0)),
    );
    assert.ok(worst > 3600, `最大倒挂只有 ${worst} 秒，示例数据里是 12868 秒`);
  });

  it("比所有消息都早的那条排在摘要之后、第一条有戳的消息之前", () => {
    const lanes = mergeNarration(REAL_MSGS, REAL_NARR);
    const first = lanes[0]!;
    assert.equal(first.kind, "message", "没戳的压缩摘要必须仍在最前面");
    const second = lanes[1]!;
    assert.equal(second.kind, "narration");
    assert.equal((second as { item: NarrationItem }).item.seq, 1);
  });

  it("消息一条戳都没有时，逐字退回 anchor", () => {
    // 重启前的存档 / 关掉时钟中间件 —— 行为必须与 2026-08-23 之前完全一样。
    const bare = REAL_MSGS.map((m) => ({ role: m.role, content: m.content }));
    const lanes = mergeNarration(bare, REAL_NARR);
    const idx = lanes.findIndex(
      (l) => l.kind === "narration" && (l as { item: NarrationItem }).item.seq === 1,
    );
    // anchor=8，转录 8 条 ⇒ 插在第 8 条消息之后 = 整条序列的第 9 位（下标 8）。
    assert.equal(idx, 8, "没有消息时刻时应当仍按 anchor 排");
  });

  it("旁白自己没有 t 时退回 anchor —— 不许把它当成 0 顶到最前面", () => {
    const noClock = { seq: 99, nk: "scan_start", text: "老行", anchor: 2 } as
      unknown as NarrationItem;
    const lanes = mergeNarration(REAL_MSGS, [noClock]);
    const idx = lanes.findIndex((l) => l.kind === "narration");
    assert.equal(idx, 2, "anchor=2 ⇒ 插在第 2 条消息之后");
  });

  it("未定时的消息跟着前一条走，不会把旁白顶到它上面", () => {
    // 中间插一条没戳的助手消息：它排在 a1 后面，就该和 a1 同期。
    const msgs = [
      { role: "user", content: "问", t: 100 },
      { role: "assistant", content: "无戳", t: null },
      { role: "assistant", content: "答", t: 300 },
    ];
    const item: NarrationItem = { seq: 1, t: 200, text: "中间发生的", anchor: 99 };
    const lanes = mergeNarration(msgs, [item]);
    const idx = lanes.findIndex((l) => l.kind === "narration");
    assert.equal(idx, 2, "应当排在那条无戳消息之后、300 那条之前");
  });
});
