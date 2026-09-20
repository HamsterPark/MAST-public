// ════════════════════════════════════════════════════════════════════════════
// src/lib/setupNav.ts —— 「人是不是已经站在初始化页上了」。
//
//     cd frontend && npm run test:unit
//
// 这条判断有一个**写在别的文件里的前提**。2026-08-06初始化页从 `/setup`
// 挪到 `/settings/setup`，旧地址改成重定向，于是「判新地址就够了」成立 —— 但它
// 成立的理由是「`/setup` 只作为重定向存在」，而那句话住在 router.tsx 里。
//
// 两件事因此是安静的：`<Navigate>` 在 effect 里跳，走旧书签进来时有一帧 pathname
// 仍是 `/setup`（横幅闪一下，指着人正要去的地方）；更要紧的是哪天有人把 `/setup`
// 变回真路由，横幅会结结实实挂在初始化页自己头上，而改动的人不会想到来看横幅。
//
// 所以别名从 LEGACY_REDIRECTS 派生。这里钉住的就是「派生真的发生了」——
// 一个只剩一条的名单，行为上和正确的一模一样，直到有人真的用了那条旧书签。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { SETUP_ALIASES, SETUP_PATH, isSetupPath } from "../src/lib/setupNav.ts";
import { LEGACY_REDIRECTS } from "../src/lib/nav.ts";

describe("setupNav", () => {
  it("正规地址算初始化页", () => {
    assert.equal(isSetupPath(SETUP_PATH), true);
    assert.equal(isSetupPath(`${SETUP_PATH}/`), true);
  });

  it("旧地址也算 —— 它是从 LEGACY_REDIRECTS 派生的，不是手写的", () => {
    // 前提：那张表里确实有一条指向初始化页。没有的话这条测试会变成空转，
    // 所以先断言它存在（一个「碰巧通过」的测试比没有测试更坏）。
    const legacy = Object.entries(LEGACY_REDIRECTS)
      .filter(([, to]) => to === SETUP_PATH)
      .map(([from]) => from);
    assert.ok(legacy.length >= 1, "LEGACY_REDIRECTS 里没有指向初始化页的旧地址");
    assert.ok(legacy.includes("/setup"), "旧地址 /setup 不见了 —— 书签会 404");

    for (const from of legacy) {
      assert.equal(isSetupPath(from), true, `${from} 应当算初始化页`);
      assert.ok(SETUP_ALIASES.includes(from), `${from} 不在别名表里`);
    }
  });

  it("子段也算 —— 将来初始化页长出子页时横幅仍该闭嘴", () => {
    assert.equal(isSetupPath(`${SETUP_PATH}/step2`), true);
  });

  it("同前缀的别的页面**不算** —— 边界必须落在路径分隔符上", () => {
    // 没有这一条，一个碰巧同前缀的新页面会让横幅在那里神秘消失，
    // 而「横幅没出现」看起来和「没什么要填的」一模一样。
    assert.equal(isSetupPath("/settings/setupfoo"), false);
    assert.equal(isSetupPath("/setupx"), false);
  });

  it("别的页面不算", () => {
    for (const p of ["/", "/settings", "/settings/admin", "/monitoring", "/records/log"]) {
      assert.equal(isSetupPath(p), false, `${p} 不该算初始化页`);
    }
  });

  it("空 / 垃圾输入不算，也不抛", () => {
    assert.equal(isSetupPath(""), false);
    assert.equal(isSetupPath("///"), false);
  });
});
