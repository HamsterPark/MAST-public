// ════════════════════════════════════════════════════════════════════════════
// 信号索引下拉 — 「不要让用户选 86，让用户在下拉菜单中选择」。
//
//     cd frontend && npm run test:unit
//
// 这里钉的是 KEY 清单，因为漏一个是**静默**的：那一项照常渲染成裸数字输入框，
// 看起来和「这一页还没改」一模一样，没有任何东西会报错。
// 「加一个操作员可编辑的键永远是双边动作」这个形状本仓已经付过四次学费。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import {
  SIGNAL_AUTO_VALUE,
  SIGNAL_INDEX_KEYS,
  channelLabel,
  channelListNote,
  channelOptions,
} from "../src/lib/signalChannels.ts";

const PROFILE_PY = join(
  fileURLToPath(new URL(".", import.meta.url)),
  "..", "..", "MASTv2", "mast", "core", "instrument_profile.py",
);

describe("channelLabel", () => {
  it("puts the index first — that is what gets stored", () => {
    assert.equal(channelLabel({ index: 86, name: "LI Demod 1 X (A)" }),
                 "86 · LI Demod 1 X (A)");
  });

  it("keeps index 0 visible rather than falsy-dropping it", () => {
    // `0 · Current (A)` is a real, common choice. A template that tested the
    // index for truthiness would render 「 · Current (A)」 and the operator
    // would have no way to tell which slot he was picking.
    assert.match(channelLabel({ index: 0, name: "Current (A)" }), /^0 · /);
  });
});

describe("SIGNAL_INDEX_KEYS covers every signal-slot key the backend defines", () => {
  it("matches instrument_profile._CONFIG_SPEC", () => {
    // Derived from the Python rather than hand-listed twice: a key added there
    // and forgotten here keeps its bare number box in silence.
    const py = readFileSync(PROFILE_PY, "utf8");
    const declared = new Set(
      [...py.matchAll(/^\s*"([a-z0-9_]*signal_index)":\s*\(/gm)].map((m) => m[1]!),
    );
    assert.ok(declared.size >= 4, `只解出 ${declared.size} 个键，正则大概过时了`);
    const missing = [...declared].filter((k) => !SIGNAL_INDEX_KEYS.has(k));
    assert.deepEqual(missing, [], `后端有、前端没接下拉：${missing.join(", ")}`);
  });

  it("does not name a key the backend dropped", () => {
    const py = readFileSync(PROFILE_PY, "utf8");
    const stale = [...SIGNAL_INDEX_KEYS].filter((k) => !py.includes(`"${k}"`));
    assert.deepEqual(stale, [], `前端还在提、后端已经没有：${stale.join(", ")}`);
  });

  it("includes the two the operator actually fills on the instrument", () => {
    assert.ok(SIGNAL_INDEX_KEYS.has("lockin_signal_index"));
    assert.ok(SIGNAL_INDEX_KEYS.has("qplus_amplitude_signal_index"));
  });
});

describe("channelOptions", () => {
  const CH = [
    { index: 0, name: "Current (A)" },
    { index: 86, name: "LI Demod 1 X (A)" },
  ];

  it("keeps a stored value that is not in the table selectable", () => {
    // Otherwise opening the dropdown shows a setting the operator already made
    // as blank, and his next move is to re-enter it — with a number he cannot
    // look up at that moment, which is exactly the problem #27 is about.
    const opts = channelOptions(CH, "99");
    assert.ok(opts.some((o) => o.value === "99"), "已填的值从下拉里消失了");
    assert.match(opts.find((o) => o.value === "99")!.label, /不在当前名单/);
  });

  it("does not add a duplicate row for a value the table has", () => {
    const opts = channelOptions(CH, "86");
    assert.equal(opts.filter((o) => o.value === "86").length, 1);
  });

  it("offers 自动 only where the key has one, and does not double it", () => {
    assert.ok(!channelOptions(CH, "").some((o) => o.label.includes("自动")));
    const withAuto = channelOptions(CH, "-1", -1);
    assert.equal(withAuto.filter((o) => o.value === "-1").length, 1,
                 "-1 既是 auto 又被当成「不在名单里」，出现了两次");
    assert.ok(withAuto.find((o) => o.value === "-1")!.label.includes("自动"));
  });

  it("always offers 未选择 first", () => {
    assert.equal(channelOptions([], "").at(0)!.value, "");
  });
});

describe("channelListNote", () => {
  it("is empty when the table is usable — that is what shows the dropdown", () => {
    assert.equal(channelListNote({ count: 128 }), "");
  });

  it("never lets 名单没解全 read as 本机没有这个通道", () => {
    // A short decode is a parse failure, not a fact about the hardware. The
    // whole reason the backend cross-checks the declared count is so these two
    // stay different sentences.
    const s = channelListNote({ truncated: true, declared_n: 128, n_channels: 51, count: 51 });
    assert.match(s, /128/);
    assert.match(s, /51/);
    assert.match(s, /别把/);
  });

  it("distinguishes 还没读到 from 读不到", () => {
    assert.notEqual(channelListNote({ pending: true, count: 0 }),
                    channelListNote({ error: true, count: 0 }));
  });

  it("falls back to the input box when the list came back empty", () => {
    assert.notEqual(channelListNote({ count: 0 }), "");
    assert.notEqual(channelListNote({ degraded: true, count: 16 }), "");
  });
});

describe("SIGNAL_AUTO_VALUE", () => {
  it("only names keys that are actually signal-index keys", () => {
    for (const k of Object.keys(SIGNAL_AUTO_VALUE)) {
      assert.ok(SIGNAL_INDEX_KEYS.has(k), k);
    }
  });

  it("gives qPlus its -1 sentinel", () => {
    // -1 means「按通道名自动查找」in _CONFIG_SPEC, range (-1, 127). Without an
    // explicit option the operator could not express it from a dropdown at all.
    assert.equal(SIGNAL_AUTO_VALUE.qplus_amplitude_signal_index, -1);
  });
});
