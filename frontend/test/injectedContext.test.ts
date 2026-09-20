// ════════════════════════════════════════════════════════════════════════════
// 「这一轮往模型那儿塞了什么」—— src/lib/injectedContext.ts 的对齐逻辑。
//
// 对齐是靠时间窗做的（理由见那个模块的自述），而时间窗只有两种坏法：
//
//   ① 边界错一格 → 这一轮显示的是**上一轮**的注入。一句看起来完全正常、
//      内容却是错的话 —— 调试时照着它找一晚上都不会怀疑它。
//   ② 对不上的时候给一句笼统的「暂无数据」→ 五种原因（关了 / 还没跑 / 挤掉了 /
//      还没发 / 没时间戳）揉成一种，而它们没有一种能靠再点一次解决。
//
// 所以下面的断言大半是变异测试：换成那几个更简单的写法各自会怎么坏。
//
//     cd frontend && npm run test:unit
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  matchInjectedContext,
  nextTurnAfter,
  splitInjected,
  userTurnTimes,
  type CaptureItem,
  type CaptureList,
} from "../src/lib/injectedContext.ts";

const AGENT = "instrument_control";

function snap(seq: number, ts: number, over: Partial<CaptureItem> = {}): CaptureItem {
  return {
    index: 0,
    seq,
    ts,
    source: AGENT,
    model_id: "kimi-k3",
    provider: "moonshot",
    message_count: 6,
    total_chars: 9000,
    system_chars: 7000,
    ...over,
  };
}

function list(items: CaptureItem[], over: Partial<CaptureList> = {}): CaptureList {
  // 后端按**新的在前**返回，替身也照这个顺序 —— 一个按时间正序的替身会让
  // 「取第一次调用」那条断言自动通过，而真数据下它会取到最后一次。
  const newestFirst = [...items].sort((a, b) => b.ts - a.ts);
  return {
    items: newestFirst,
    count: newestFirst.length,
    enabled: true,
    total_seen: newestFirst.length,
    capacity: 40,
    degraded: false,
    ...over,
  };
}

describe("对上这一轮的请求", () => {
  it("取的是这一轮的**第一次**调用，不是最后一次", () => {
    // 一轮里有工具调用 ⇒ 好几次模型请求。注入的那份上下文是第一次带过去的；
    // 取最后一次会拿到一份已经塞满工具结果的历史，那不是「系统注入了什么」。
    const m = matchInjectedContext(100, 200, list([
      snap(1, 101), snap(2, 130), snap(3, 160),
    ]), AGENT);
    assert.equal(m.kind, "match");
    if (m.kind !== "match") return;
    assert.equal(m.seq, 1);
    assert.equal(m.calls, 3, "这一轮发了几次请求也要说");
  });

  it("下一条用户消息之后的请求**不算**这一轮", () => {
    // 边界错一格的形状：把 `< hi` 写成 `<= hi`，或者干脆不封口。
    const m = matchInjectedContext(100, 200, list([snap(9, 200), snap(1, 101)]), AGENT);
    assert.equal(m.kind, "match");
    if (m.kind !== "match") return;
    assert.equal(m.seq, 1);
    assert.equal(m.calls, 1);
  });

  it("这一轮还在进行中（没有下一条用户消息）时窗口不封口", () => {
    const m = matchInjectedContext(100, null, list([snap(1, 101), snap(2, 900)]), AGENT);
    assert.equal(m.kind, "match");
    if (m.kind !== "match") return;
    assert.equal(m.calls, 2);
  });

  it("和用户消息**同一秒**发出的请求算这一轮", () => {
    // `ts >= t` 而不是 `>`：后端盖时间和发请求之间常常不到 1 秒，
    // 写成严格大于会让整整一类正常对话对不上。
    const m = matchInjectedContext(100, null, list([snap(1, 100)]), AGENT);
    assert.equal(m.kind, "match");
  });
});

describe("对不上的时候，说清是哪一种对不上", () => {
  const why = (r: ReturnType<typeof matchInjectedContext>) =>
    r.kind === "none" ? r.why : "(matched)";

  it("捕获被关掉 ≠ 没发生过调用", () => {
    const r = matchInjectedContext(100, null, list([], { enabled: false }), AGENT);
    assert.match(why(r), /已关闭/);
  });

  it("进程还没跑过模型调用", () => {
    const r = matchInjectedContext(100, null, list([]), AGENT);
    assert.match(why(r), /还没有发生过模型调用/);
  });

  it("模块读不到 ≠ 没有数据", () => {
    const r = matchInjectedContext(100, null, list([], { degraded: true }), AGENT);
    assert.match(why(r), /读不到/);
  });

  it("被挤出记录（过去式）和还没发出（将来式）是两句话", () => {
    // 环里最老的一条都比这条消息新 ⇒ 这一轮的记录已经被挤掉了。
    const evicted = matchInjectedContext(
      100, 200, list([snap(50, 5000)], { total_seen: 137 }), AGENT);
    assert.match(why(evicted), /挤出记录/);
    assert.match(why(evicted), /137/, "说清一共发生过多少次，才知道挤得有多快");

    // 环里有更老的记录，只是这一轮的窗口里一条都没有 ⇒ 还没发出去。
    const notYet = matchInjectedContext(
      1000, 1100, list([snap(1, 10)]), AGENT);
    assert.match(why(notYet), /还没有发出/);
  });

  it("没有时间戳的消息永远对不上，而且要说出来", () => {
    const r = matchInjectedContext(null, null, list([snap(1, 101)]), AGENT);
    assert.match(why(r), /没有时间戳/);
    // `undefined` 和 `NaN` 走同一支 —— 后端的 `t` 是 `float | None`，
    // FastAPI 序列化成 `null`，而 `Number(null) === 0` 会让它对上 1970 年那一轮。
    assert.equal(matchInjectedContext(undefined, null, list([snap(1, 101)]), AGENT).kind,
                 "none");
    assert.equal(matchInjectedContext(NaN, null, list([snap(1, 101)]), AGENT).kind,
                 "none");
  });
});

describe("窗口里混进别的 agent", () => {
  it("优先取本对话这个 agent 的请求", () => {
    const m = matchInjectedContext(100, null, list([
      snap(1, 101, { source: "data_processing" }),
      snap(2, 110),
    ]), AGENT);
    assert.equal(m.kind, "match");
    if (m.kind !== "match") return;
    assert.equal(m.seq, 2);
    assert.equal(m.foreignSource, false);
    assert.equal(m.calls, 1, "别人的调用不该算进这一轮的次数");
  });

  it("一条都没有时退而求其次，但**必须标出来不是它发的**", () => {
    // 悄悄展示一份别人的注入，比不展示更坏 —— 它看起来完全正常。
    const m = matchInjectedContext(100, null, list([
      snap(1, 101, { source: "literature" }),
    ]), AGENT);
    assert.equal(m.kind, "match");
    if (m.kind !== "match") return;
    assert.equal(m.foreignSource, true);
    assert.equal(m.source, "literature");
  });
});

describe("一轮的右边界", () => {
  it("只有用户消息封口，助手/工具消息不封口", () => {
    const msgs = [
      { role: "user", t: 10 },
      { role: "assistant", t: 12 },
      { role: "tool", t: 13 },
      { role: "user", t: 20 },
      { role: "assistant", t: 22 },
    ];
    const times = userTurnTimes(msgs);
    assert.deepEqual(times, [10, 20]);
    assert.equal(nextTurnAfter(times, 10), 20);
    assert.equal(nextTurnAfter(times, 20), null);
  });

  it("没有时间戳的用户消息不能当边界", () => {
    // 拿一个 null 当边界会把窗口右端点变成 0/NaN，整轮当场对不上。
    assert.deepEqual(
      userTurnTimes([{ role: "user", t: 10 }, { role: "user", t: null },
                     { role: "user", t: 30 }]),
      [10, 30]);
  });

  it("边界是**严格大于**，同一秒的另一条不提前封口", () => {
    assert.equal(nextTurnAfter([10, 10, 30], 10), 30);
  });

  it("边界集合与消息**顺序无关** —— 分块渲染时这是唯一正确的算法", () => {
    // NarrationLane 把消息切成好几段分别交给 ChatBubbles。按下标算的版本会把
    // 每一段的最后一条用户消息的窗口开到无穷大 —— 它会吞掉后面所有轮的请求。
    const whole = [
      { role: "user", t: 10 }, { role: "assistant", t: 11 },
      { role: "user", t: 20 }, { role: "assistant", t: 21 },
      { role: "user", t: 30 },
    ];
    const times = userTurnTimes(whole);
    // 第一段只有前两条，但 t=10 这一轮的右边界仍然是 20。
    assert.equal(nextTurnAfter(times, 10), 20);
  });

  it("空表不炸", () => {
    assert.deepEqual(userTurnTimes([]), []);
    assert.equal(nextTurnAfter([], 10), null);
    assert.equal(nextTurnAfter([10], null), null);
  });
});

describe("哪些消息算「注入」", () => {
  it("system 是注入，对话历史不是", () => {
    // 历史在屏幕上就看得见，把它也算成「系统注入的上下文」会让那个数字
    // 随对话越滚越大，看起来像注入在膨胀。
    const { injected, history } = splitInjected([
      { role: "system", content: "A", chars: 1, truncated: false },
      { role: "human", content: "B", chars: 1, truncated: false },
      { role: "ai", content: "C", chars: 1, truncated: false },
      { role: "system", content: "D", chars: 1, truncated: false },
    ]);
    assert.deepEqual(injected.map((m) => m.content), ["A", "D"]);
    assert.deepEqual(history.map((m) => m.content), ["B", "C"]);
  });
});
