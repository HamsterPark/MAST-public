/**
 * 技能市场的写入判据 + manifest 解析 —— 判据本身。
 *
 * ── 背景 ──
 * 这与 `settingsWrite.test.ts` 是同一族，但多了一层：那边只要回答「存进去了吗」，
 * 这边还要回答「**现在生效了吗**」。改订阅只改了一个 holder，而 agent 的工具表在
 * 建图时冻结 —— 一次 `ok:true` 的写入之后，模型手上那张表可能还是旧的（任务在跑
 * ⇒ 排队 / 重建失败 / 没有活的运行时）。把这三种都显示成绿色的「已保存」，操作员
 * 会基于「已生效」去做下一件事。
 *
 * 所以判据是三态的，而 `null`（判断不了）与 `false`（确实没跟上）必须说不一样的
 * 话 —— 本仓 [[unknown_is_not_an_answer]] 那一族里最常见的折叠就发生在这一步。
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  canUnsubscribe,
  diffManifestAgainstCatalog,
  liveBadge,
  marketWriteProblem,
  parseSubscriptionManifest,
  pendingBadgeCount,
  paletteHides,
  paletteHiddenCount,
  locallyAuthoredUnsubscribed,
  LOCAL_ORIGINS,
  MANIFEST_KIND,
} from "../src/lib/skillMarket.ts";

describe("marketWriteProblem — 判据本身", () => {
  it("存了而且生效了 → null", () => {
    assert.equal(
      marketWriteProblem({ ok: true, agent_path_pending: false, fingerprint_matches: true }),
      null,
    );
  });

  it("没有响应落在失败一侧", () => {
    assert.equal(marketWriteProblem(undefined)?.kind, "failed");
    assert.equal(marketWriteProblem(null)?.kind, "failed");
    assert.equal(marketWriteProblem({})?.kind, "failed");
  });

  it("ok:false 且 degraded:false 是拒绝，不是成功", () => {
    const p = marketWriteProblem({ ok: false, degraded: false, reason: "必装技能" });
    assert.equal(p?.kind, "failed");
    assert.match(p!.message, /必装/);
  });

  it("degraded 是失败，不是「已保存」", () => {
    assert.equal(marketWriteProblem({ ok: true, degraded: true })?.kind, "failed");
  });

  it("排队了 → pending，而且带出后端那句人话", () => {
    const p = marketWriteProblem({
      ok: true,
      agent_path_pending: true,
      rebuild_note: "已保存（任务运行中：已排队，当前任务结束后自动生效）",
    });
    assert.equal(p?.kind, "pending");
    assert.match(p!.message, /任务运行中/);
  });

  it("agent_path_pending 判断不了 → unsure，不是绿色", () => {
    const p = marketWriteProblem({ ok: true, agent_path_pending: null });
    assert.equal(p?.kind, "unsure");
  });

  it("指纹说没跟上 → pending（即使 pending 标志是 false）", () => {
    const p = marketWriteProblem({
      ok: true,
      agent_path_pending: false,
      fingerprint_matches: false,
    });
    assert.equal(p?.kind, "pending");
  });

  it("指纹 null 不等于 false —— 重建走完了就是好的", () => {
    assert.equal(
      marketWriteProblem({ ok: true, agent_path_pending: false, fingerprint_matches: null }),
      null,
    );
  });

  it("后端那句人话原样带出，不被前端改写", () => {
    const note = "已保存（没有活的运行时 —— 已存盘，下次启动生效）";
    const p = marketWriteProblem({ ok: true, agent_path_pending: true, rebuild_note: note });
    assert.equal(p!.message, note);
  });
});

describe("liveBadge — 三态各说各话", () => {
  it("true / false / null 三句话互不相同", () => {
    const t = liveBadge(true), f = liveBadge(false), n = liveBadge(null);
    assert.equal(t.tone, "ok");
    assert.equal(f.tone, "warn");
    assert.equal(n.tone, "unknown");
    assert.notEqual(f.text, n.text, "「没跟上」和「判断不了」说了同一句话");
    assert.notEqual(t.text, n.text);
  });

  it("undefined 与 null 同义（判断不了）", () => {
    assert.deepEqual(liveBadge(undefined), liveBadge(null));
  });
});

describe("canUnsubscribe / pendingBadgeCount", () => {
  it("必装项的开关是锁着的", () => {
    assert.equal(canUnsubscribe({ mandatory: true }), false);
    assert.equal(canUnsubscribe({ mandatory: false }), true);
    assert.equal(canUnsubscribe({}), true);
  });

  it("角标只数待确认的", () => {
    assert.equal(
      pendingBadgeCount([
        { status: "pending" },
        { status: "accepted" },
        { status: "rejected" },
        {},
      ]),
      2,
    );
    assert.equal(pendingBadgeCount(null), 0);
  });
});

describe("paletteHides — 「不知道」不能被折叠成「否」", () => {
  it("只看订阅时，明确未订阅的被藏起来", () => {
    assert.equal(paletteHides({ subscribed: false }, true), true);
    assert.equal(paletteHides({ subscribed: true }, true), false);
  });

  it("字段缺席时**不过滤** —— 这是那个真陷阱", () => {
    // 写成 `!e.subscribed` 的话，一个少了这个字段的响应（旧版本、目录降级、
    // 订阅子系统读不出来）会让整个 palette 空掉，而界面只显示「没有匹配的技能」。
    assert.equal(paletteHides({}, true), false);
    assert.equal(paletteHides({ subscribed: undefined }, true), false);
  });

  it("切到全市场就什么都不藏", () => {
    assert.equal(paletteHides({ subscribed: false }, false), false);
  });

  it("藏了几条要数得出来（不能静默截断）", () => {
    const rows = [{ subscribed: true }, { subscribed: false }, {}, { subscribed: false }];
    assert.equal(paletteHiddenCount(rows, true), 2);
    assert.equal(paletteHiddenCount(rows, false), 0);
  });
});

describe("locallyAuthoredUnsubscribed — 本机长出来却不在工具面上的", () => {
  const rows = [
    { name: "SetBias", source: "builtin", subscribed: false },
    { name: "MyFlow", source: "user_composite", subscribed: false },
    { name: "AgentFlow", source: "user_composite", subscribed: true },
    { name: "Patched", source: "overlay", subscribed: false },
    { name: "Hand", source: "custom", subscribed: false },
    { name: "Paper1", source: "paper", subscribed: false },
  ];

  it("只挑本机来源的，且只挑未订阅的", () => {
    assert.deepEqual(
      locallyAuthoredUnsubscribed(rows).map((r) => r.name),
      ["MyFlow", "Patched", "Hand"],
    );
  });

  it("随应用发布的来源不算 —— 它们是升级带来的，不是本机长出来的", () => {
    const names = locallyAuthoredUnsubscribed(rows).map((r) => r.name);
    for (const shipped of ["SetBias", "Paper1"]) {
      assert.ok(!names.includes(shipped), `${shipped} 是随应用发布的，不该出现在这里`);
    }
  });

  it("字段缺席时不算（与 paletteHides 同一条 fail-open）", () => {
    assert.deepEqual(locallyAuthoredUnsubscribed([{ name: "X" }]), []);
    assert.deepEqual(
      locallyAuthoredUnsubscribed([{ name: "X", source: "user_composite" }]),
      [],
      "subscribed 未知时不该断言它「不在工具面上」",
    );
  });

  it("来源词汇取自后端闭集", () => {
    assert.deepEqual([...LOCAL_ORIGINS], ["user_composite", "custom", "overlay"]);
  });
});

describe("parseSubscriptionManifest", () => {
  const good = JSON.stringify({
    kind: MANIFEST_KIND,
    schema_version: 1,
    machine: "rig-1",
    entries: [{ name: "SetBias", source: "builtin" }, { name: "X" }],
  });

  it("好文件解析出条目", () => {
    const r = parseSubscriptionManifest(good);
    assert.ok("manifest" in r);
    assert.equal(r.manifest.entries.length, 2);
    assert.equal(r.manifest.machine, "rig-1");
  });

  it("不是 JSON → 一句中文原因，不抛", () => {
    const r = parseSubscriptionManifest("<html>404</html>");
    assert.ok("error" in r);
    assert.match(r.error, /JSON/);
  });

  it("kind 不对 → 点名说它是什么", () => {
    const r = parseSubscriptionManifest(JSON.stringify({ kind: "conduct", entries: [] }));
    assert.ok("error" in r);
    assert.match(r.error, /conduct/);
  });

  it("没有 entries → 说出来", () => {
    const r = parseSubscriptionManifest(JSON.stringify({ kind: MANIFEST_KIND }));
    assert.ok("error" in r);
    assert.match(r.error, /entries/);
  });

  it("数组 / null 顶层不算对象", () => {
    assert.ok("error" in parseSubscriptionManifest("[]"));
    assert.ok("error" in parseSubscriptionManifest("null"));
  });

  it("条目里的垃圾被丢掉，好的留下", () => {
    const r = parseSubscriptionManifest(
      JSON.stringify({ kind: MANIFEST_KIND, entries: [{ name: "A" }, 3, null, { name: "  " }] }),
    );
    assert.ok("manifest" in r);
    assert.deepEqual(r.manifest.entries.map((e) => e.name), ["A"]);
  });
});

describe("diffManifestAgainstCatalog", () => {
  const man = {
    kind: MANIFEST_KIND,
    entries: [
      { name: "A" },
      { name: "B", source: "overlay" },
      { name: "C", source: "user_composite", spec: { nodes: [] } },
    ],
  };

  it("分出本机有的与没有的", () => {
    const d = diffManifestAgainstCatalog(man, ["A", "C"]);
    assert.deepEqual(d.matched, ["A", "C"]);
    assert.deepEqual(d.missing.map((m) => m.name), ["B"]);
    assert.equal(d.embedded, 1);
  });

  it("本机什么都没有时全部算缺失（不静默丢）", () => {
    const d = diffManifestAgainstCatalog(man, []);
    assert.equal(d.missing.length, 3);
    assert.equal(d.matched.length, 0);
  });
});
