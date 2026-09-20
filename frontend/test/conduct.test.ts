// src/lib/conduct.ts —— conduct 面板的纯判断。
//
// 两组:
//
// **① 双端镜像 parity。** 状态词表与意图词表的真源在 `mast/conduct/store.py`,
// 这里的 TS 是镜像。测试**直接读那个 .py 文件**对账 —— 不经 openapi(响应字段是
// `str`,枚举根本进不了 schema),也不靠人记得两边一起改。设计 §10-6:「状态枚举
// 会同时出现在后端与前端面板 —— 一开始就配 parity 测试」。
//
// 提取不到就**失败**,不是跳过:一个「没找到就当过了」的 parity 测试,在 Python
// 那边重命名之后会永远绿着,而那正是它要防的事。
//
// **② 面板判断。** 每一条都对应一个「不会崩、typecheck 抓不到、截图看不出来」的
// 错法:灰按钮没有理由、空预算条被读成没花钱、waive 标记不显示、把正常长步报成
// 停滞、拿 WS 帧当增量状态源。
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import {
  CONDUCT_FRAMES,
  CONDUCT_OPS,
  CONDUCT_STATUSES,
  OP_VALID_STATUSES,
  WISHLIST_AGENT_PREFIX,
  approveBlockReason,
  conductIdFromAgentId,
  conductPanelHref,
  badgeTone,
  budgetText,
  conditionText,
  fmtDuration,
  goalProgressText,
  frameAction,
  heartbeatBanner,
  ignitionText,
  lackingLabels,
  opEnabled,
  statusLabel,
  statusTone,
  waitGateRows,
} from "../src/lib/conduct.ts";

// ── ① parity：真源是 Python ──────────────────────────────────────────────────

const STORE_PY = fileURLToPath(
  new URL("../../MASTv2/mast/conduct/store.py", import.meta.url),
);
const DIRECTOR_PY = fileURLToPath(
  new URL("../../MASTv2/mast/conduct/director.py", import.meta.url),
);
const EVENTS_PY = fileURLToPath(
  new URL("../../MASTv2/mast/core/events.py", import.meta.url),
);

/** 抽一个 `NAME = ( "a", "b", ... )` 里的字符串。**抽不到就抛。** */
function pyTuple(file: string, name: string): string[] {
  const src = readFileSync(file, "utf8");
  const m = new RegExp(`^${name}\\s*(?::[^=]*)?=\\s*\\(([\\s\\S]*?)\\)`, "m").exec(src);
  assert.ok(
    m,
    `没能从 ${file} 里抽出 ${name} —— parity 测试的输入没了。` +
      `抽不到必须红:一个「没找到就当过了」的对账,在那边重命名之后会永远绿着。`,
  );
  const items = [...m![1].matchAll(/"([a-z_]+)"/g)].map((x) => x[1]);
  assert.ok(items.length > 0, `${name} 抽出来是空的 —— 正则对上了但没拿到值`);
  return items;
}

describe("conduct 双端镜像 parity（真源在 Python）", () => {
  it("状态词表与 store.STATUSES 逐字相同", () => {
    assert.deepEqual([...CONDUCT_STATUSES], pyTuple(STORE_PY, "STATUSES"));
  });

  it("意图词表与 store.OPS 逐字相同", () => {
    assert.deepEqual([...CONDUCT_OPS], pyTuple(STORE_PY, "OPS"));
  });

  it("每个状态都有中文名和配色，一个都不漏", () => {
    for (const s of CONDUCT_STATUSES) {
      assert.notEqual(statusLabel(s), s, `${s} 没有中文名（会原样显示英文）`);
      assert.ok(statusTone(s));
    }
  });

  it("按钮可用状态表与 director.OP_VALID_STATUSES 的键一致", () => {
    const src = readFileSync(DIRECTOR_PY, "utf8");
    const block = /OP_VALID_STATUSES[\s\S]*?\n\}/.exec(src);
    assert.ok(block, "没能从 director.py 里抽出 OP_VALID_STATUSES");
    const keys = [...block![0].matchAll(/^\s{4}"([a-z_]+)":/gm)].map((m) => m[1]);
    assert.deepEqual(keys.sort(), Object.keys(OP_VALID_STATUSES).sort());
  });

  it("abort 从任何非终态都够得着 —— 「能停不能解」的反面", () => {
    for (const s of CONDUCT_STATUSES) {
      const { enabled } = opEnabled("abort", s);
      assert.equal(
        enabled,
        s !== "completed" && s !== "aborted",
        `abort 在 ${s} 下的可达性不对`,
      );
    }
  });

  it("每个配色档都映射到 ui.tsx 真的有的那个 Badge tone", () => {
    // 拼错一个 tone 名字的症状是徽章**悄悄退回默认色** —— 看起来完全正常。
    const ui = readFileSync(
      fileURLToPath(new URL("../src/components/ui.tsx", import.meta.url)),
      "utf8",
    );
    const block = /const BADGE_TONE[\s\S]*?\n\};/.exec(ui);
    assert.ok(block, "没能从 ui.tsx 里抽出 BADGE_TONE");
    const known = new Set([...block![0].matchAll(/^\s{2}([A-Za-z]+):/gm)].map((m) => m[1]));
    for (const t of ["ok", "warn", "crit", "info", "neutral"] as const) {
      assert.ok(known.has(badgeTone(t)), `${t} → ${badgeTone(t)} 不在 ui.tsx 的词表里`);
    }
  });

  it("三种 WS 帧的名字与 core/events.py 逐字相同", () => {
    const src = readFileSync(EVENTS_PY, "utf8");
    const names = [...src.matchAll(/CONDUCT_[A-Z]+\s*=\s*"([a-z_]+)"/g)].map((m) => m[1]);
    assert.deepEqual(names.sort(), [...CONDUCT_FRAMES].sort());
  });
});

// ── ② 面板判断 ───────────────────────────────────────────────────────────────

describe("未知状态不许渲染成空白", () => {
  it("后端加了新状态时原样显示 + 中性色", () => {
    assert.equal(statusLabel("some_new_state"), "some_new_state");
    assert.equal(statusTone("some_new_state"), "neutral");
  });

  it("空状态给一句话，不是空串", () => {
    assert.equal(statusLabel(""), "（无状态）");
  });
});

describe("灰按钮必须说得出为什么灰", () => {
  it("PAUSED 下的暂停按钮：灰的，并说出现在能按什么", () => {
    const r = opEnabled("pause", "paused");
    assert.equal(r.enabled, false);
    assert.match(r.why, /已暂停/);
    assert.match(r.why, /执行中/); // 列出允许的状态
  });

  it("终态给的是「新建一份」而不是一句通用的不可用", () => {
    const r = opEnabled("pause", "completed");
    assert.equal(r.enabled, false);
    assert.match(r.why, /终态是终态/);
  });

  it("能按的时候没有多余的话", () => {
    assert.deepEqual(opEnabled("pause", "running"), { enabled: true, why: "" });
    assert.deepEqual(opEnabled("resume", "paused"), { enabled: true, why: "" });
  });

  it("ack 只在两个等待态下能按", () => {
    assert.equal(opEnabled("ack", "waiting_operator").enabled, true);
    assert.equal(opEnabled("ack", "waiting_condition").enabled, true);
    assert.equal(opEnabled("ack", "running").enabled, false);
  });
});

describe("双闸：两个证据回答两个问题，互不替代", () => {
  const bothWait = {
    wait_id: "w1",
    kind: "both",
    message: "请换样品并等降温",
    ack: { required: true, at: null, by: "" },
    condition: {
      desc: "温度 ≤ 5 K",
      threshold: 5,
      current_value: 78.2,
      met: false,
      stale: false,
    },
  };

  it("两个闸各占一行，不是一个「还差 1 项」的计数", () => {
    const rows = waitGateRows(bothWait);
    assert.deepEqual(rows.map((r) => r.label), ["人的确认", "物理条件"]);
    assert.deepEqual(lackingLabels(bothWait), ["人的确认", "物理条件"]);
  });

  it("人确认了，物理条件仍然缺 —— 确认不能替条件", () => {
    const acked = { ...bothWait, ack: { required: true, at: 1e9, by: "操作员" } };
    assert.deepEqual(lackingLabels(acked), ["物理条件"]);
    assert.match(waitGateRows(acked)[0].detail, /操作员/);
  });

  it("温度到位了，人的确认仍然缺 —— 条件不能替确认", () => {
    const met = {
      ...bothWait,
      condition: { ...bothWait.condition, met: true, met_since: 1e9 },
    };
    assert.deepEqual(lackingLabels(met), ["人的确认"]);
  });

  it("stale 是**判不了**，与「没到」分开显示", () => {
    const stale = {
      ...bothWait,
      condition: { ...bothWait.condition, stale: true, reading_age_s: 3600 },
    };
    const cond = waitGateRows(stale)[1];
    assert.equal(cond.ok, false);
    assert.equal(cond.unreadable, true, "stale 被当成了普通的「没到」");
    assert.match(cond.detail, /读不到不等于没到/);
  });

  it("waive 标记**持续**显示，不是 ack 那一刻显示一次", () => {
    const waived = {
      ...bothWait,
      condition: {
        ...bothWait.condition,
        waived: true,
        waived_by: "操作员",
        waive_reason: "温度计 COM13 被占，人工读数 4.3 K",
      },
    };
    const cond = waitGateRows(waived)[1];
    assert.equal(cond.ok, true);
    assert.match(cond.detail, /操作员/);
    assert.match(cond.detail, /COM13/, "放行理由没显示 —— 这个闸是人放过去的，要一直看得见");
  });

  it("没有等待就没有闸行", () => {
    assert.deepEqual(waitGateRows(null), []);
    assert.deepEqual(lackingLabels(undefined), []);
  });

  it("条件正文带上当前读数；读不到就说读不到", () => {
    assert.match(conditionText(bothWait.condition), /78\.2/);
    assert.match(conditionText({ desc: "温度 ≤ 5 K", current_value: null }), /读不到/);
  });
});

describe("预算：读不到 ≠ 花了 0", () => {
  it("读不到时写「读不到」并带上为什么", () => {
    const r = budgetText({ spent_usd: null, cap_usd: 80, reason: "还没有归集口径" });
    assert.equal(r.unreadable, true);
    assert.match(r.text, /读不到/);
    assert.doesNotMatch(r.text, /\$0\.00/, "读不到被渲染成了 $0.00");
    assert.match(r.hint, /归集/);
  });

  it("真的花了 0 就写 $0.00 —— 那是一个答得上来的数", () => {
    const r = budgetText({ spent_usd: 0, cap_usd: 80 });
    assert.equal(r.unreadable, false);
    assert.match(r.text, /\$0\.00/);
  });

  it("没有上限时不写一个假的 $0.00 上限", () => {
    assert.match(budgetText({ spent_usd: 1.5, cap_usd: 0 }).text, /未设上限/);
  });

  // ── 「拦不住」是第三件事(M3-d)──────────────────────────────────────
  it("上限拦不住的时候，连「上限 $20.00」这半句都不照原样印", () => {
    // `读不到 / 上限 $20.00` 读起来仍然像「有一道上限在那儿，只是这一刻没读到
    // 花了多少」。真相是那道上限不存在 —— 账本按 provider 原生币种实测，
    // 而它是 USD，合并需要汇率而本仓不自造汇率。
    const r = budgetText({
      spent_usd: null,
      cap_usd: 20,
      enforceable: false,
      not_enforceable_why: "计价单位对不上：账本按原生币种，上限是 USD",
    });
    assert.equal(r.unreadable, true);
    assert.doesNotMatch(r.text, /上限 \$20\.00/, "拦不住的上限被印成了一道守卫");
    assert.match(r.text, /不生效/);
    assert.match(r.hint, /计价单位/);
  });

  it("逐币种实测照实显示，不折成一个数", () => {
    const r = budgetText({
      spent_usd: null,
      cap_usd: 20,
      enforceable: false,
      measured_by_currency: { CNY: 3.5, USD: 0.4 },
      not_enforceable_why: "跨币种",
    });
    assert.match(r.text, /CNY 3\.50/);
    assert.match(r.text, /USD 0\.40/);
    assert.doesNotMatch(r.text, /3\.90|6\.4|25\.6/, "把两个币种加成了一个数");
  });

  it("旧响应没有 enforceable 字段时保持旧行为", () => {
    // 后端与前端不是同一次部署上去的。少一个字段就翻脸，等于把一次版本错配
    // 变成一个坏掉的面板。
    const r = budgetText({ spent_usd: 1.5, cap_usd: 80 });
    assert.equal(r.unreadable, false);
    assert.match(r.text, /\$1\.50 \/ 上限 \$80\.00/);
  });
});

describe("heartbeat 告警：不该报的时候不报", () => {
  it("没停滞就没有告警条", () => {
    assert.equal(heartbeatBanner({ stalled: false, age_s: 9999 }, "running"), null);
  });

  it("终态不报 —— 已经结束的东西没有停滞可言", () => {
    assert.equal(heartbeatBanner({ stalled: true }, "completed"), null);
  });

  it("在步里：说的是这一步跑了多久，并写明不会杀步", () => {
    const b = heartbeatBanner(
      { stalled: true, in_step: true, step_elapsed_s: 7200, threshold_s: 3600 },
      "running",
    );
    assert.ok(b);
    assert.equal(b!.tone, "warn");
    assert.match(b!.detail, /2 小时/);
    assert.match(b!.detail, /不会.*杀/, "没说清「不杀步」——那是一条诚实短板，不是待办");
  });

  it("不在步里：这是线程可能死了，级别更高", () => {
    const b = heartbeatBanner(
      { stalled: true, in_step: false, age_s: 600, threshold_s: 45 },
      "running",
    );
    assert.ok(b);
    assert.equal(b!.tone, "crit");
    assert.match(b!.title, /线程/);
  });

  it("后端的理由原样带出来，前端不重判一次", () => {
    const b = heartbeatBanner(
      { stalled: true, in_step: false, age_s: 600, reason: "引擎关着?" },
      "running",
    );
    assert.match(b!.detail, /引擎关着/);
  });
});

describe("approve 批不下去：两种，不是一种", () => {
  it("「有检查根本没跑」与「发现了错误」说的不是同一句话", () => {
    const notChecked = approveBlockReason({
      ok: false,
      validation_ok: true,
      validation_complete: false,
    });
    const hasErrors = approveBlockReason({
      ok: false,
      validation_ok: false,
      validation_complete: true,
    });
    assert.match(notChecked, /没跑/);
    assert.match(hasErrors, /模板本身/);
    assert.notEqual(
      notChecked,
      hasErrors,
      "两种批不下去被并成了同一句——那正是校验器要报三态的全部理由",
    );
  });

  it("批过了就没有那句话", () => {
    assert.equal(approveBlockReason({ ok: true }), "");
    assert.equal(approveBlockReason(null), "");
  });
});

describe("WS 帧只做触发，不做增量状态源", () => {
  it("认得的帧 → refetch", () => {
    for (const t of CONDUCT_FRAMES) {
      assert.equal(frameAction(t, { conduct_id: "c1" }, "c1"), "refetch");
    }
  });

  it("别的 conduct 的帧不管 —— 同一条总线上还跑着别人", () => {
    assert.equal(frameAction("conduct_status", { conduct_id: "other" }, "c1"), "ignore");
  });

  it("不认得的 type 忽略，不抛 —— 新帧种类不该弄坏老前端", () => {
    assert.equal(frameAction("hardware_state", { conduct_id: "c1" }, "c1"), "ignore");
    assert.equal(frameAction("", null, "c1"), "ignore");
  });

  it("帧里没有 conduct_id 时保守地刷一次", () => {
    assert.equal(frameAction("conduct_alert", {}, "c1"), "refetch");
    assert.equal(frameAction("conduct_alert", "not-an-object", "c1"), "refetch");
  });

  it("还没选中 conduct 时什么都不做", () => {
    assert.equal(frameAction("conduct_status", { conduct_id: "c1" }, ""), "ignore");
  });

  it("返回值里没有任何来自帧的状态 —— 只有 refetch/ignore 两个字", () => {
    const out = frameAction("conduct_status", { conduct_id: "c1", status: "aborted" }, "c1");
    assert.equal(typeof out, "string");
    assert.ok(out === "refetch" || out === "ignore");
  });
});

describe("心愿单 ↔ 面板:待办要有一条能点的出路", () => {
  it("agent_id 前缀与后端 adapters.WISHLIST_AGENT_PREFIX 逐字相同", () => {
    const src = readFileSync(
      fileURLToPath(new URL("../../MASTv2/mast/conduct/adapters.py", import.meta.url)),
      "utf8",
    );
    // 后端写的是 `WISHLIST_AGENT_PREFIX = OWNER_PREFIX`，真值在 OWNER_PREFIX 上。
    const m = /^OWNER_PREFIX\s*=\s*"([^"]+)"/m.exec(src);
    assert.ok(m, "没能从 adapters.py 里抽出 OWNER_PREFIX");
    assert.equal(WISHLIST_AGENT_PREFIX, m![1]);
    // 前缀对不上的症状不是报错，是那条链接**永远不出现** —— 而心愿单看起来
    // 完全正常，只是少了一个从没有人见过的按钮。
    assert.ok(
      /^WISHLIST_AGENT_PREFIX\s*=\s*OWNER_PREFIX\s*$/m.test(src),
      "后端把两个前缀拆开了，这条 parity 断言的前提没了",
    );
  });

  it("认得 conduct 发的请求，认不出别人的", () => {
    assert.equal(conductIdFromAgentId("conduct:abc123"), "abc123");
    assert.equal(conductIdFromAgentId("literature"), "");
    assert.equal(conductIdFromAgentId(""), "");
    assert.equal(conductIdFromAgentId(null), "");
  });

  it("深链带上 id —— 可收藏、可转发给下一个班的人", () => {
    assert.equal(conductPanelHref("abc123"), "/records/conduct?id=abc123");
    assert.equal(conductPanelHref(""), "/records/conduct");
    assert.equal(conductPanelHref(null), "/records/conduct");
  });

  it("id 里有奇怪字符时不会拼出一个坏 URL", () => {
    assert.match(conductPanelHref("a b&c"), /id=a%20b%26c/);
  });
});

describe("时长格式", () => {
  it("空值给「—」而不是 0 秒", () => {
    assert.equal(fmtDuration(null), "—");
    assert.equal(fmtDuration(undefined), "—");
    assert.equal(fmtDuration(NaN), "—");
  });

  it("秒/分/小时", () => {
    assert.equal(fmtDuration(45), "45 秒");
    assert.equal(fmtDuration(600), "10 分");
    assert.equal(fmtDuration(7200), "2 小时");
    assert.equal(fmtDuration(5400), "1 小时 30 分");
  });
});

describe("ignitionText —— supervised 的撤销窗要看得见", () => {
  it("窗口没开就不渲染（空串，不是一个空框）", () => {
    assert.equal(ignitionText(null), "");
    assert.equal(ignitionText(undefined), "");
    assert.equal(ignitionText({ by: "agent:xd", remaining_s: 0 }), "");
    assert.equal(ignitionText({ by: "agent:xd", remaining_s: -3 }), "");
  });

  it("窗口开着就说清楚：谁批的、还有多久、怎么撤回", () => {
    const t = ignitionText({ by: "agent:xd", remaining_s: 125, delay_s: 600 });
    assert.match(t, /agent:xd/);
    assert.match(t, /2 分 5 秒/);
    // 「怎么撤回」不能省：这一档的全部意义就是「来得及后悔」，
    // 而操作员未必知道 abort 在 approved 态是合法的。
    assert.match(t, /abort/);
  });

  it("不到一分钟就只说秒", () => {
    assert.match(ignitionText({ by: "a", remaining_s: 12 }), /12 秒后/);
  });

  it("坏读数当成没有窗口，而不是渲染一个 NaN", () => {
    assert.equal(ignitionText({ by: "a", remaining_s: Number.NaN }), "");
  });
});

describe("goalProgressText —— unknown 不许显示成 0/N", () => {
  it("三态各有各的话说", () => {
    assert.equal(
      goalProgressText({ verdict: "done", satisfied: 2, total: 2 }).text,
      "已达成 2/2",
    );
    assert.equal(
      goalProgressText({ verdict: "not_done", satisfied: 1, total: 3 }).text,
      "1/3",
    );
  });

  it("读不到就说读不到 —— **不是** 0/N", () => {
    // 「一条都没满足」和「读不到」驱动的下一步相反：前者接着做，后者去看
    // 为什么读不到。折叠成同一个数字，操作员会把一次库故障读成「还早着呢」。
    const g = goalProgressText({ verdict: "unknown", satisfied: 0, total: 3 });
    assert.equal(g.text, "读不到");
    assert.ok(!g.text.includes("0/"), "unknown 被渲染成了一个进度数字");
  });

  it("没有判据时是「—」，不是「0/0 已达成」", () => {
    assert.equal(goalProgressText(null).text, "—");
    assert.equal(goalProgressText({}).text, "—");
  });
});
