/**
 * 设置页里的 conduct 一节 —— 以及那条指向不存在页面的指引。
 *
 * 在这一节存在之前：后端的读 / 写 / 类型三侧早就通了（`SettingsStore.KNOWN_KEYS`
 * 有 `conduct`、读写 schema 都有、`set_conduct_knobs` 也接了），而 SettingsPage 里
 * grep `conduct` **零命中** —— 操作员在界面上根本打不开这条线，只能直接 POST。
 * 与此同时 ConductPage 还写着「去『设置 → 常规设置』」，指向一个不存在的地方。
 *
 * 这份闸门钉三件事：
 *   1. 整表替换守卫在 conduct 目录上照样成立（漏发一个键 = 那个键被后端当成
 *      「没设」而回落到默认，静默改掉操作员没动过的旋钮）；
 *   2. 「已保存」与「真的在跑」四态各有各的话说；
 *   3. 两个页面引用**同一个**标题常量 —— 按构造保证指引指得到地方。
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it } from "node:test";

import { CONDUCT_SETTINGS_TITLE, directorStateNote } from "../src/lib/conduct.ts";
import { buildKnobPayload } from "../src/lib/knobs.ts";

const SRC = join(import.meta.dirname, "..", "src");

function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");
}

/** conduct 的旋钮目录（形状照 `mast/conduct/settings.py::knob_catalog`）。 */
const CATALOG = [
  { key: "cd_enabled", label_zh: "启用", hint_zh: "", min: 0, max: 1, step: 0, default: 0, value: 0, is_bool: true },
  { key: "cd_stall_grace_s", label_zh: "停滞宽限", hint_zh: "", min: 0, max: 3600, step: 0, default: 900, value: 900, is_bool: false },
  {
    key: "cd_autonomy",
    label_zh: "自主度",
    hint_zh: "",
    min: 0,
    max: 2,
    step: 0,
    default: 0,
    value: 0,
    is_bool: false,
    choices: [
      { value: 0, label_zh: "有人值守" },
      { value: 1, label_zh: "半自主" },
      { value: 2, label_zh: "自主" },
    ],
  },
  { key: "cd_ignition_delay_s", label_zh: "撤销窗", hint_zh: "", min: 0, max: 3600, step: 0, default: 600, value: 600, is_bool: false },
];

describe("整表替换守卫在 conduct 目录上成立", () => {
  it("只改一个档位，其余三个键照样一起发过去", () => {
    // 后端的 `set_conduct_knobs` 收的是整张表：少发一个键，那个键会被当成
    // 「没设」而回落到默认 —— 操作员没动过的旋钮被静默改掉，而且改的方向是
    // 「回到出厂值」，看起来完全正常。
    // 签名是 (knobs, persisted, change)，change 是 {key, value} —— 照生产
    // 函数写，不照记忆写。第一版把 persisted 当成了 change，于是断言绿在
    // 「什么都没改」上。
    const payload = buildKnobPayload(CATALOG, null, { key: "cd_autonomy", value: 2 });
    assert.deepEqual(payload, {
      cd_enabled: 0,
      cd_stall_grace_s: 900,
      cd_autonomy: 2,
      cd_ignition_delay_s: 600,
    });
  });

  it("已持久化的值优先于目录里的 value", () => {
    const payload = buildKnobPayload(CATALOG, { cd_enabled: 1 });
    assert.equal(payload.cd_enabled, 1);
  });
});

describe("directorStateNote —— 「已保存」与「在跑」是两件事", () => {
  it("开 + 跑 = 正常", () => {
    const n = directorStateNote(true, true);
    assert.equal(n.tone, "ok");
    assert.match(n.text, /正在运行/);
  });

  it("开 + 不跑 = 说清楚是「下次启动」还是「起不来」", () => {
    const n = directorStateNote(true, false);
    assert.equal(n.tone, "warn");
    assert.match(n.text, /下次启动|没能启动/);
  });

  it("关 + 跑 = 这不该发生，不能和「关着」长得一样", () => {
    const n = directorStateNote(false, true);
    assert.equal(n.tone, "crit");
    assert.match(n.text, /不该发生/);
  });

  it("关 + 不跑 = 关着，但读端点照常", () => {
    const n = directorStateNote(false, false);
    assert.match(n.text, /读端点/);
  });

  it("四态两两不同 —— 没有两种处境共用一句话", () => {
    const texts = [
      directorStateNote(true, true).text,
      directorStateNote(true, false).text,
      directorStateNote(false, true).text,
      directorStateNote(false, false).text,
    ];
    assert.equal(new Set(texts).size, 4);
  });
});

describe("接线：设置页有这一节，conduct 页指得到它", () => {
  it("SettingsPage 挂了 ConductSettingsSection", () => {
    const src = stripComments(readFileSync(join(SRC, "pages", "SettingsPage.tsx"), "utf8"));
    assert.match(src, /<ConductSettingsSection\b/);
    assert.match(src, /ConductSettingsSection.*from "@\/components\/settings\/ConductSettingsSection"/);
  });

  it("ConductPage 不再指向「常规设置」那个不存在的地方", () => {
    const src = stripComments(readFileSync(join(SRC, "pages", "ConductPage.tsx"), "utf8"));
    assert.ok(
      !src.includes("常规设置"),
      "ConductPage 还在把操作员往「设置 → 常规设置」送 —— 那里没有 conduct",
    );
    assert.ok(
      src.includes("CONDUCT_SETTINGS_TITLE"),
      "指引里写死了标题字符串 —— 标题一改，指引就又指错了",
    );
  });

  it("两页引用同一个常量（按构造一致，不靠人对齐）", () => {
    const settings = stripComments(readFileSync(join(SRC, "pages", "SettingsPage.tsx"), "utf8"));
    assert.ok(settings.includes("CONDUCT_SETTINGS_TITLE"));
    assert.ok(CONDUCT_SETTINGS_TITLE.length > 0);
  });

  it("扫描器自检：它确实读到了这两个文件的内容", () => {
    // 没有这一条，上面那些 `!includes(...)` 可能只是因为读到了空字符串。
    const a = readFileSync(join(SRC, "pages", "SettingsPage.tsx"), "utf8");
    const b = readFileSync(join(SRC, "pages", "ConductPage.tsx"), "utf8");
    assert.ok(a.includes("Accordion") && b.includes("conduct"));
  });
});

describe("档位名不在前端第二次定义", () => {
  it("ConductSettingsSection 里没有写死的三档中文名", () => {
    // 档位名的真源是后端 `autonomy.describe`；在前端再写一份，两处岔开时
    // 不会有任何东西报错 —— 界面写着「半自主」，存进去的却是别的档。
    const src = stripComments(
      readFileSync(join(SRC, "components", "settings", "ConductSettingsSection.tsx"), "utf8"),
    );
    for (const name of ["有人值守", "半自主"]) {
      assert.ok(
        !src.includes(name),
        `前端写死了档位名「${name}」—— 它应当来自后端目录的 choices`,
      );
    }
  });

  it("KnobCatalogSection 认得 choices（否则三档会退化成数字框）", () => {
    const src = readFileSync(
      join(SRC, "components", "settings", "KnobCatalogSection.tsx"),
      "utf8",
    );
    assert.match(src, /knob\.choices/);
    assert.match(src, /role="radiogroup"/);
  });
});
