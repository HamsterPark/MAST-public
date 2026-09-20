// src/lib/stickToBottom.ts —— 「跟到底部」的两个纯判断。
//
// ## 这个文件盯的是一个截图看不出来的毛病
//
// 准备录屏时发现:对话的滚动条和整个标签的滚动条都会
// 自动锁定,往上拉会被自动拉回去,**两个滚动条都是这样**。
//
// 两个成因,各自都会让功能看起来「完全正常」:
//
//   ① 跟随不问用户在看哪儿 —— 只有在往上翻的时候才显形;
//   ② `scrollIntoView()` 会滚**每一个可滚动祖先** —— 所以是「两个」滚动条,
//      而不是一个。这一条尤其阴:页面上少一层 overflow 容器,它就看不出来。
//
// 所以断言写在**关系**上(往上翻 ⇒ 不跟随;只认最近的一个容器),
// 而不是写在像素上。
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  STICK_THRESHOLD_PX,
  isAtBottom,
  nearestScrollable,
} from "../src/lib/stickToBottom.ts";

describe("isAtBottom —— 什么时候才该跟随", () => {
  it("贴着底部 ⇒ 跟随", () => {
    assert.equal(isAtBottom({ scrollTop: 900, scrollHeight: 1000, clientHeight: 100 }), true);
  });

  it("用户往上翻了一大截 ⇒ **不跟随**(这就是要防的那个问题)", () => {
    assert.equal(isAtBottom({ scrollTop: 100, scrollHeight: 1000, clientHeight: 100 }), false);
  });

  it("差几个像素仍算在底部 —— 阈值不许写 0", () => {
    // 内容重排 / 图片加载完 / 亚像素取整都会差上几个像素。判 0 会把
    // 「明明在底部」读成「用户翻上去了」,于是跟随永远不触发 —— 一个
    // **反方向**的同类 bug,而它同样看不出来。
    const gap = STICK_THRESHOLD_PX - 1;
    assert.equal(
      isAtBottom({ scrollTop: 900 - gap, scrollHeight: 1000, clientHeight: 100 }),
      true,
    );
    // 刚好越过阈值就该停 —— 否则「阈值」只是个摆设。
    assert.equal(
      isAtBottom({ scrollTop: 900 - STICK_THRESHOLD_PX - 1, scrollHeight: 1000, clientHeight: 100 }),
      false,
    );
  });

  it("内容还没撑满容器 ⇒ 算在底部(否则第一条消息就跟不上)", () => {
    assert.equal(isAtBottom({ scrollTop: 0, scrollHeight: 50, clientHeight: 100 }), true);
  });

  it("橡皮筋回弹的负间隙 ⇒ 仍然算在底部", () => {
    // macOS / 触屏惯性滚动会让 scrollTop 短暂越过底部。那显然不是「翻上去了」。
    assert.equal(isAtBottom({ scrollTop: 950, scrollHeight: 1000, clientHeight: 100 }), true);
  });
});

// ── 找容器 ────────────────────────────────────────────────────────────────
//
// 用一棵假树,`parentElement` 手工串起来 —— node --test 里没有 DOM,
// 而这个函数的全部逻辑就是「往上走,遇到第一个可滚的就停」。

function tree(overflows: (string | undefined)[]) {
  // overflows[0] 是最里层(锚点自己),依次往外。
  const nodes = overflows.map((oy, i) => ({ id: i, oy })) as any[];
  for (let i = 0; i < nodes.length - 1; i++) nodes[i].parentElement = nodes[i + 1];
  nodes[nodes.length - 1].parentElement = null;
  return {
    leaf: nodes[0] as Element,
    nodes,
    getStyle: (e: any) => ({ overflowY: e.oy }),
  };
}

describe("nearestScrollable —— 只认最近的那一个", () => {
  it("锚点自己不可滚 ⇒ 找到最近的可滚祖先,**并且就此打住**", () => {
    // 聊天那边正是这个形状:面板可滚,外面整页也可滚。
    // `scrollIntoView()` 会把两个都滚 —— 那就是「两个滚动条都被拉回去」。
    const t = tree([undefined, "auto", "auto"]);
    const got = nearestScrollable(t.leaf, t.getStyle);
    assert.equal((got as any).id, 1, "应当停在最近的那一个,而不是继续往上找到整页");
  });

  it("锚点自己就是滚动容器 ⇒ 用它自己", () => {
    // 任务面板传的就是容器本身。两种调用方式共用一个函数,
    // 免得多出一处「传错了也不报错、只是不跟随」的地方。
    const t = tree(["auto", "auto"]);
    assert.equal((nearestScrollable(t.leaf, t.getStyle) as any).id, 0);
  });

  it("认得 overlay(老 WebKit 的 auto)", () => {
    // 漏掉它就会一路找到 <body> —— 又变回「连整页一起滚」。
    const t = tree([undefined, "overlay", "auto"]);
    assert.equal((nearestScrollable(t.leaf, t.getStyle) as any).id, 1);
  });

  it("一个可滚的都没有 ⇒ null(而不是退回 body)", () => {
    const t = tree([undefined, "visible", "hidden"]);
    assert.equal(nearestScrollable(t.leaf, t.getStyle), null);
  });

  it("null 进 null 出 —— 组件卸载后 ref 是空的", () => {
    assert.equal(nearestScrollable(null), null);
  });
});
