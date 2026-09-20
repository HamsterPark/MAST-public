// ════════════════════════════════════════════════════════════════════════════
// Pure-parser tests for src/lib/inlineBold.ts.
//
//     cd frontend && npm run test:unit
//
// The strings this parses are safety copy written in Python
// (mast/core/instrument_init.py), e.g.
//   「搞反了，一次「退针 3000 步」就是往样品里送 3000 步。**运行时有兜底**：…」
// The page used to print the asterisks, which read as noisy and unprofessional,
// and raw markup in the middle of the one sentence that keeps a tip out of the
// sample is a large part of why.
//
// The failure that matters is not "no bold" — it is TEXT LOSS. A parser that
// eats a character produces a sentence that still reads fine and no longer says
// what it said, so every case below checks the round trip.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { splitBold, stripBold } from "../src/lib/inlineBold.ts";

/** Everything that went in must come back out. */
function assertLossless(src: string) {
  assert.equal(
    splitBold(src).map((s) => (s.bold ? `**${s.text}**` : s.text)).join(""),
    src,
    "splitBold lost or invented characters",
  );
}

describe("splitBold", () => {
  it("splits the real catalog string", () => {
    const src = "搞反了，一次退针就是往样品里送。**运行时有兜底**：第一级只走 1 步。";
    assert.deepEqual(splitBold(src), [
      { text: "搞反了，一次退针就是往样品里送。", bold: false },
      { text: "运行时有兜底", bold: true },
      { text: "：第一级只走 1 步。", bold: false },
    ]);
    assertLossless(src);
  });

  it("handles several emphases in one sentence", () => {
    const src = "**唯一一个**填错了硬件当场报废的数，**没有任何读数**告诉你。";
    const got = splitBold(src);
    assert.deepEqual(got.filter((s) => s.bold).map((s) => s.text), [
      "唯一一个",
      "没有任何读数",
    ]);
    assertLossless(src);
  });

  it("handles emphasis at both ends", () => {
    assert.deepEqual(splitBold("**a**"), [{ text: "a", bold: true }]);
    assert.deepEqual(splitBold("**a**b"), [
      { text: "a", bold: true },
      { text: "b", bold: false },
    ]);
  });
});

describe("splitBold — text that must survive untouched", () => {
  it("leaves an UNCLOSED ** as literal text", () => {
    // Half-typed markup must not reflow the rest of the paragraph into bold.
    const src = "设定值 **3e-12 未闭合";
    assert.deepEqual(splitBold(src), [{ text: src, bold: false }]);
    assertLossless(src);
  });

  it("never treats a single asterisk as emphasis", () => {
    // Physics copy is full of bare asterisks: footnote marks, `a*` states, glob
    // patterns. Swallowing one deletes a character from a label.
    for (const src of ["a* 态", "glob *.sxm 匹配", "脚注 *", "5 * 3"]) {
      assert.deepEqual(splitBold(src), [{ text: src, bold: false }]);
      assertLossless(src);
    }
  });

  it("keeps `****` as text rather than emitting an empty span", () => {
    assert.deepEqual(splitBold("a****b"), [
      { text: "a", bold: false },
      { text: "****", bold: false },
      { text: "b", bold: false },
    ]);
    assertLossless("a****b");
  });

  it("returns nothing for empty / absent input", () => {
    for (const v of ["", null, undefined]) assert.deepEqual(splitBold(v), []);
  });
});

describe("stripBold", () => {
  it("gives the plain sentence, for title= and for searching", () => {
    assert.equal(
      stripBold("**运行时有兜底**：第一级只走 1 步。"),
      "运行时有兜底：第一级只走 1 步。",
    );
  });

  it("is the identity on unmarked text", () => {
    assert.equal(stripBold("退针方向（粗动马达远离样品）"), "退针方向（粗动马达远离样品）");
  });
});
