// ════════════════════════════════════════════════════════════════════════════
// Pure-logic tests for src/lib/initFilter.ts — 新仪器初始化 的「显示哪些项」。
//
//     cd frontend && npm run test:unit
//
// 2026-08-04, on the instrument: the operator wanted to change 退针方向 and could not
// find it. The value was there, in a group whose header read「完成」and which
// therefore rendered collapsed — so the item was in the payload, on the page,
// and invisible. It had to be edited over the API instead. That field is the one
// where being wrong drives the tip INTO the sample, which makes "I measured it,
// now let me correct it" the expected flow rather than an edge case.
//
// The tests below pin the two rules that fix it: a scope switch that can show
// answered items, and a search that ignores the scope entirely (the item being
// hunted is BY DEFINITION already answered, so a search confined to 待办 would
// find nothing and look broken).
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  groupsToOpen,
  isOutstanding,
  matchesQuery,
  visibleItems,
  type InitItemLike,
} from "../src/lib/initFilter.ts";

function item(over: Partial<InitItemLike> = {}): InitItemLike {
  return {
    id: "instrument_profile.retract_motor_dir",
    group: "retract",
    label: "退针方向（粗动马达远离样品）",
    key: "retract_motor_dir",
    store: "instrument_profile",
    severity: "required",
    complete: true,
    status: "set",
    hint: "装置接线约定，多数 Nanonis 是 Z+ = 远离。",
    consequence: "搞反了，一次「退针 3000 步」就是往样品里送 3000 步。**运行时有兜底**：…",
    ...over,
  };
}

const RETRACT = item();
const MISSING_GAIN = item({
  id: "instrument_profile.preamp_gain_v_per_a",
  group: "preamp",
  label: "前置放大器增益",
  key: "preamp_gain_v_per_a",
  severity: "required",
  complete: false,
  status: "missing",
  hint: "电流 = 读数电压 ÷ 此值；看前放**铭牌** / 手册。",
  consequence: "缺了整类电流读数都是错的。",
});
const NA_ITEM = item({
  id: "optics.pztc_port",
  group: "optics",
  label: "PZTC 串口",
  key: "pztc_port",
  severity: "recommended",
  complete: false,
  status: "n/a",
  hint: "",
  consequence: "",
});
const ALL = [RETRACT, MISSING_GAIN, NA_ITEM];

describe("isOutstanding", () => {
  it("counts an unanswered required item", () => {
    assert.equal(isOutstanding(MISSING_GAIN), true);
  });

  it("does not count an answered one", () => {
    assert.equal(isOutstanding(RETRACT), false);
  });

  it("never counts n/a — an earlier answer ruled it out, it is not work", () => {
    assert.equal(isOutstanding(NA_ITEM), false);
  });
});

describe("visibleItems — scope", () => {
  it("待办 hides what is already answered", () => {
    const got = visibleItems(ALL, { scope: "todo", query: "" });
    assert.deepEqual(got.map((i) => i.key), ["preamp_gain_v_per_a"]);
  });

  it("全部 shows answered items — this IS the 修改入口", () => {
    const got = visibleItems(ALL, { scope: "all", query: "" });
    assert.equal(got.length, 3);
    assert.ok(got.some((i) => i.key === "retract_motor_dir"));
  });
});

describe("visibleItems — search", () => {
  it("reaches an ANSWERED item even in 待办 scope", () => {
    // The whole point. 退针方向 is complete, so a scope-respecting search would
    // return nothing and the operator would conclude the field is not there.
    const got = visibleItems(ALL, { scope: "todo", query: "退针" });
    assert.deepEqual(got.map((i) => i.key), ["retract_motor_dir"]);
  });

  it("matches the storage key, not just the Chinese label", () => {
    const got = visibleItems(ALL, { scope: "todo", query: "preamp_gain" });
    assert.deepEqual(got.map((i) => i.key), ["preamp_gain_v_per_a"]);
  });

  it("matches across the `**` markup in the copy", () => {
    // "运行时有兜底" is written **emphasised** in the catalog; searching for it
    // must not miss on an asterisk the operator cannot see.
    const got = visibleItems(ALL, { scope: "all", query: "运行时有兜底" });
    assert.deepEqual(got.map((i) => i.key), ["retract_motor_dir"]);
  });

  it("requires every term, so two words narrow rather than widen", () => {
    assert.equal(matchesQuery(RETRACT, "退针 方向"), true);
    assert.equal(matchesQuery(RETRACT, "退针 前置"), false);
  });

  it("is case-insensitive for the ASCII half", () => {
    assert.equal(matchesQuery(MISSING_GAIN, "PREAMP_GAIN"), true);
  });

  // `hint` 是 what + where 合并来的。合并那一刻最容易掉的就是搜索：
  // 操作员记得住的往往是「铭牌」「面板」这类**去哪儿找**，而不是这一项叫什么。
  it("still reaches 「铭牌」 —— 合并字段之后搜索没有变窄", () => {
    const got = visibleItems(ALL, { scope: "todo", query: "铭牌" });
    assert.deepEqual(got.map((i) => i.key), ["preamp_gain_v_per_a"]);
  });

  it("an empty or whitespace query is not a filter", () => {
    assert.equal(visibleItems(ALL, { scope: "all", query: "   " }).length, 3);
  });
});

describe("groupsToOpen", () => {
  it("opens every group holding a search hit", () => {
    const visible = visibleItems(ALL, { scope: "todo", query: "退针" });
    assert.deepEqual([...groupsToOpen(visible, { query: "退针" })], ["retract"]);
  });

  it("without a search, opens only groups with outstanding REQUIRED work", () => {
    const visible = visibleItems(ALL, { scope: "all", query: "" });
    const open = groupsToOpen(visible, { query: "" });
    assert.equal(open.has("preamp"), true);
    assert.equal(open.has("retract"), false, "a finished group starts collapsed");
    assert.equal(open.has("optics"), false, "n/a is not outstanding work");
  });
});
