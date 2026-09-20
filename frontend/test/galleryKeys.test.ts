// ════════════════════════════════════════════════════════════════════════════
// 数据图库大图的键盘 — src/lib/gallery/keys.ts
//
//     cd frontend && npm run test:unit
//
// 整张表钉住。键盘标记是这个页面最高频的操作：一个键悄悄换了含义不会报任何错，
// 只会让接下来几百条标记全部打错。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { HOT_CODES, HOT_LABELS, keyAction } from "../src/lib/gallery/keys.ts";

describe("整张键位表", () => {
  const table: [string, unknown][] = [
    ["ArrowRight", { t: "next" }],
    ["ArrowDown", { t: "next" }],
    ["PageDown", { t: "next" }],
    ["ArrowLeft", { t: "prev" }],
    ["ArrowUp", { t: "prev" }],
    ["PageUp", { t: "prev" }],
    ["Home", { t: "first" }],
    ["End", { t: "last" }],
    ["Digit1", { t: "rate", r: 1 }],
    ["Numpad1", { t: "rate", r: 1 }],
    ["Digit2", { t: "rate", r: 2 }],
    ["Numpad2", { t: "rate", r: 2 }],
    ["Digit3", { t: "rate", r: -1 }],
    ["Numpad3", { t: "rate", r: -1 }],
    ["Digit0", { t: "rate", r: 0 }],
    ["Numpad0", { t: "rate", r: 0 }],
    ["Backquote", { t: "rate", r: 0 }],
    ["KeyN", { t: "note" }],
    ["KeyS", { t: "pick", code: "KeyS" }],
    ["BracketLeft", { t: "pick", code: "BracketLeft" }],
    ["BracketRight", { t: "pick", code: "BracketRight" }],
    ["Escape", { t: "close" }],
    ["KeyQ", { t: "tag", index: 0 }],
    ["KeyO", { t: "tag", index: 8 }],
    ["KeyP", null],
    ["Digit4", null],
    ["Space", null],
  ];
  for (const [code, want] of table) {
    it(code, () => assert.deepEqual(keyAction({ code }), want));
  }

  it("九个标签键依次对应 QWERTYUIO", () => {
    assert.equal(HOT_CODES.length, 9);
    HOT_CODES.forEach((c, i) => {
      assert.equal(c, `Key${HOT_LABELS[i]}`);
      assert.deepEqual(keyAction({ code: c }), { t: "tag", index: i });
    });
  });
});

describe("上下文", () => {
  it("A / D 只对谱与网格起作用", () => {
    assert.deepEqual(keyAction({ code: "KeyA", spectrumLike: true }), { t: "anchor", side: "prev" });
    assert.deepEqual(keyAction({ code: "KeyD", spectrumLike: true }), { t: "anchor", side: "next" });
    assert.equal(keyAction({ code: "KeyA" }), null);
    assert.equal(keyAction({ code: "KeyD", spectrumLike: false }), null);
  });

  it("焦点在输入框里只认 Esc（失焦），其余键交还输入框", () => {
    assert.deepEqual(keyAction({ code: "Escape", inField: true }), { t: "blur" });
    assert.equal(keyAction({ code: "Digit1", inField: true }), null);
    assert.equal(keyAction({ code: "ArrowRight", inField: true }), null);
  });

  it("带修饰键不处理", () => {
    assert.equal(keyAction({ code: "KeyS", ctrl: true }), null);
    assert.equal(keyAction({ code: "Digit1", meta: true }), null);
    assert.equal(keyAction({ code: "ArrowRight", alt: true }), null);
  });
});
