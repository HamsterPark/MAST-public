// ════════════════════════════════════════════════════════════════════════════
// src/lib/transcriptRefresh.ts — 「刚才那件事，值得我重读一次转录吗」。
//
// 「智能体代理对话不会自动刷新」。
//
// 这些断言几乎全是**变异测试**:它们钉住的不是「能刷新」,而是「那几个更简单的
// 写法各自会怎么坏」。这一块的每一个 bug 都不崩、不报错、类型检查全绿,症状只有
// 「偶尔多跳一下」或者「就是不刷」—— 而这两句话的排查方向完全相反。
//
//     cd frontend && npm run test:unit
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  advanceCursor,
  isPrivateTurnFor,
  transcriptCursorFor,
  transcriptSource,
} from "../src/lib/transcriptRefresh.ts";

describe("isPrivateTurnFor", () => {
  const ev = { scope: "private_turn", conversation_id: "c1", agent_id: "ic" };

  it("刷新我正在看的那个会话", () => {
    assert.equal(isPrivateTurnFor(ev, "c1"), true);
  });

  it("别人的会话动了,不刷我这一页", () => {
    assert.equal(isPrivateTurnFor(ev, "c2"), false);
  });

  it("没选会话时什么都不刷", () => {
    assert.equal(isPrivateTurnFor(ev, null), false);
    assert.equal(isPrivateTurnFor(ev, undefined), false);
    assert.equal(isPrivateTurnFor(ev, ""), false);
  });

  it("群聊转录事件不算 —— 两种 scope 的 payload 形状不同", () => {
    // 借用 "transcript" 那个值的话,这一侧会按缺失的 seq 去做去重:
    // Number(undefined ?? 0) === 0,去重条件恒真,症状是「一条都不刷」。
    assert.equal(
      isPrivateTurnFor({ ...ev, scope: "transcript" }, "c1"),
      false,
    );
  });

  it("自己正在流的时候不抢", () => {
    // 抢的后果不是「多一次请求」:页面此刻显示的是 SSE 增量快照,插一次整
    // history 重读会让已经画出来的半句话跳回上一轮的结尾。
    assert.equal(isPrivateTurnFor(ev, "c1", true), false);
  });

  it("空 payload / 非字符串字段一律不刷,而不是抛", () => {
    assert.equal(isPrivateTurnFor(null, "c1"), false);
    assert.equal(isPrivateTurnFor(undefined, "c1"), false);
    assert.equal(isPrivateTurnFor({}, "c1"), false);
    assert.equal(isPrivateTurnFor({ scope: 7, conversation_id: 1 }, "c1"), false);
  });
});

describe("transcriptCursorFor", () => {
  it("认得出这个 agent 在群聊里说的话,并带回游标", () => {
    const got = transcriptCursorFor(
      { scope: "transcript", agent_id: "lit", t: 1234.5 },
      "lit",
    );
    assert.deepEqual(got, { ok: true, t: 1234.5 });
  });

  it("别的 agent 说的话不刷我这一页", () => {
    const got = transcriptCursorFor(
      { scope: "transcript", agent_id: "dp", t: 1 },
      "lit",
    );
    assert.equal(got.ok, false);
  });

  it("私聊回合事件不走这条路", () => {
    const got = transcriptCursorFor(
      { scope: "private_turn", agent_id: "lit" },
      "lit",
    );
    assert.equal(got.ok, false);
  });

  it("「没有游标」报成 t=0,而不是混进「游标很旧」", () => {
    // 这是本文件里最容易写错的一条。调用方的去重是 `t <= seen` —— 一条没带 t
    // 的事件如果被当成 t=0,那个条件恒真,它就永远刷不出东西。所以 ok=true 配
    // t=0 的含义必须是「照刷」,由调用方兑现,这里只负责把两者分开。
    for (const bad of [undefined, null, 0, -1, NaN, Infinity, "12"]) {
      const got = transcriptCursorFor(
        { scope: "transcript", agent_id: "lit", t: bad },
        "lit",
      );
      assert.equal(got.ok, true, `t=${String(bad)} 应当仍然认得出是我的事件`);
      assert.equal(got.t, 0, `t=${String(bad)} 应当报成「没有游标」`);
    }
  });

  it("agentId 为空时不认领任何事件", () => {
    const got = transcriptCursorFor({ scope: "transcript", agent_id: "" }, "");
    assert.equal(got.ok, false);
  });
});

describe("advanceCursor", () => {
  it("首见即记账、不刷", () => {
    // 挂载那一刻的 updated_at 描述的正是页面刚取回来的那份历史。把它当成
    // 「有更新」会让每一次挂载都白读一次 —— 而「代理对话」是条件渲染的,
    // 切一次子 tab 就是一次挂载。
    const r = advanceCursor(null, "2026-08-06T10:00:00");
    assert.deepEqual(r, { seen: "2026-08-06T10:00:00", refresh: false });
  });

  it("游标前进了就重读一次", () => {
    const r = advanceCursor("2026-08-06T10:00:00", "2026-08-06T10:00:05");
    assert.deepEqual(r, { seen: "2026-08-06T10:00:05", refresh: true });
  });

  it("同一个游标再来一次不重复读", () => {
    // 会话列表每 6 s 轮询一次,绝大多数次拿回来的是同一个戳。少了这一条,
    // 「有更新才读」就退化成「每 6 s 读一次整份 history」—— 正是不想要的那个。
    const r = advanceCursor("2026-08-06T10:00:00", "2026-08-06T10:00:00");
    assert.deepEqual(r, { seen: "2026-08-06T10:00:00", refresh: false });
  });

  it("空值不推进游标,也不刷", () => {
    // updated_at 缺失是「这一行没带这个字段」,不是「回到了从前」。拿它把 seen
    // 清成 null 的话,下一次真更新会被当成首见而吞掉 —— 一次静默丢失。
    for (const empty of [null, undefined, ""]) {
      const r = advanceCursor("2026-08-06T10:00:00", empty);
      assert.deepEqual(r, { seen: "2026-08-06T10:00:00", refresh: false });
    }
  });

  it("一串轮询里只有变化的那一拍会刷", () => {
    const stamps = ["t1", "t1", "t1", "t2", "t2", "t3"];
    let seen: string | null = null;
    let refreshes = 0;
    for (const s of stamps) {
      const r = advanceCursor(seen, s);
      seen = r.seen;
      if (r.refresh) refreshes += 1;
    }
    assert.equal(refreshes, 2, "t1→t2 和 t2→t3 各一次,首见的 t1 不算");
  });
});


// ════════════════════════════════════════════════════════════════════════════
// transcriptSource —— **取回来之后画哪一份**（#31 → #36 → #41 的第三次落点）
//
// 前两轮修的都是送达。这一族钉的是第三段：数据到了浏览器之后，屏幕上换不换。
// 每一条都写成「若这一条不成立，操作员会看见什么」，因为这里所有的 bug 都不崩、
// typecheck 全绿、单页截图完美。
// ════════════════════════════════════════════════════════════════════════════
describe("transcriptSource —— 取回来之后画哪一份 ", () => {
  it("**新消息到达 ⇒ 那一页换成新的那一份**（#41 报的就是这条）", () => {
    // 别人（后台 agent / 另一台机器）写进来 → 通知 → invalidate → 重取落地，
    // 于是 history 的 dataUpdatedAt 走到了这一轮流结束之后。
    assert.equal(
      transcriptSource({
        hasSnapshot: true,
        streaming: false,
        streamEndedAt: 1_000,
        historyUpdatedAt: 1_001,
      }),
      "history",
    );
  });

  it("从前那个写法（谁先有值谁赢）在这里会答错 —— 钉住它答不对", () => {
    // `streamMsgs ?? history.data?.messages` 等价于「只要有快照就用快照」。
    // 上面那条如果返回 "stream"，症状正是操作员第三次报的那句：新消息不显示，
    // 切到别的页面再回来才显示（重挂让快照归 null）。
    const shadowed = transcriptSource({
      hasSnapshot: true, streaming: false, streamEndedAt: 1_000, historyUpdatedAt: 1_001,
    });
    assert.notEqual(shadowed, "stream");
  });

  it("正在流的时候不换 —— 半句话不许跳回上一轮的结尾", () => {
    assert.equal(
      transcriptSource({
        hasSnapshot: true, streaming: true, streamEndedAt: 0, historyUpdatedAt: 9_999,
      }),
      "stream",
    );
  });

  it("这一轮还没结束过就不换（快照正是最新的那一份）", () => {
    assert.equal(
      transcriptSource({
        hasSnapshot: true, streaming: false, streamEndedAt: 0, historyUpdatedAt: 500,
      }),
      "stream",
    );
  });

  it("历史一次都没取回来时留住快照 —— 别把屏幕清空", () => {
    // `historyUpdatedAt === 0` 是「从来没有成功过」，不是「很旧」。混成一件事的
    // 症状是一次失败的重取把刚说完的一轮抹掉。
    assert.equal(
      transcriptSource({
        hasSnapshot: true, streaming: false, streamEndedAt: 1_000, historyUpdatedAt: 0,
      }),
      "stream",
    );
  });

  it("历史比这一轮旧（陈旧缓存）就不换", () => {
    assert.equal(
      transcriptSource({
        hasSnapshot: true, streaming: false, streamEndedAt: 2_000, historyUpdatedAt: 1_500,
      }),
      "stream",
    );
  });

  it("同一毫秒不算更新 —— 严格大于才换", () => {
    assert.equal(
      transcriptSource({
        hasSnapshot: true, streaming: false, streamEndedAt: 1_000, historyUpdatedAt: 1_000,
      }),
      "stream",
    );
  });

  it("没有快照时永远画历史（刚挂载 / 换了会话）", () => {
    assert.equal(
      transcriptSource({
        hasSnapshot: false, streaming: false, streamEndedAt: 0, historyUpdatedAt: 0,
      }),
      "history",
    );
    assert.equal(
      transcriptSource({
        hasSnapshot: false, streaming: true, streamEndedAt: 0, historyUpdatedAt: 0,
      }),
      "history",
    );
  });
});
