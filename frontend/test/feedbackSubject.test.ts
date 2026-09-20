// ════════════════════════════════════════════════════════════════════════════
// FeedbackFloat 的「这条反馈算在哪个面上」。
//
// 这一块有过一次前科:悬浮窗飘在每一页上,而 agent 是写死的,于是反馈
// 一度**全部**被记成 instrument_control —— 包括那些明明在说
// 群聊 / 记录 / 视觉页的。不崩、不报错,只是一张读起来完全正常、
// 内容却错了的表。
//
// 大标签合并时差点把同一个坑挖回来:`/builder` 变成了
// `/skills/builder`,而匹配用的是「表里第一条命中的前缀」——`/skills` 排在前面,
// 于是构建器的反馈会被记成技能页的。**光把字符串改新是不够的。**
//
//     cd frontend && npm run test:unit
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { subjectForPath } from "../src/lib/feedbackSubject.ts";

describe("subjectForPath", () => {
  it("仪器 Chat 是真的和 IC 私聊,不是一个「面」", () => {
    assert.equal(subjectForPath("/"), "instrument_control");
    assert.equal(subjectForPath("/chat"), "instrument_control");
  });

  it("二级页赢过它的大标签(最长前缀)", () => {
    // 这是整个文件的要害。少了它,下面这三行全都会退化成大标签那一档,
    // 而反馈表照样有数据、照样能看 —— 只是内容是错的。
    assert.equal(subjectForPath("/skills/builder"), "page_builder");
    assert.equal(subjectForPath("/records/memory"), "page_cognition");
    assert.equal(subjectForPath("/settings/admin"), "page_admin");
    assert.equal(subjectForPath("/settings/setup"), "page_setup");
    assert.equal(subjectForPath("/settings/usage"), "page_usage");
    assert.equal(subjectForPath("/monitoring/env"), "page_env_history");
    assert.equal(subjectForPath("/experimental/optics"), "page_optics");
  });

  it("大标签自己的那一段还是大标签", () => {
    assert.equal(subjectForPath("/skills/library"), "page_skills");
    assert.equal(subjectForPath("/records/log"), "page_records");
    assert.equal(subjectForPath("/settings/general"), "page_settings");
    assert.equal(subjectForPath("/monitoring/current"), "page_monitoring");
    assert.equal(subjectForPath("/experimental/tools"), "page_experimental");
  });

  it("组根落在大标签上", () => {
    assert.equal(subjectForPath("/settings"), "page_settings");
    assert.equal(subjectForPath("/monitoring"), "page_monitoring");
  });

  it("更深的路径继承它所在的那一段", () => {
    // 页面内部的深链(?seg=、/records/log/xxx 之类)不该掉回默认值。
    assert.equal(subjectForPath("/settings/admin/anything"), "page_admin");
    assert.equal(subjectForPath("/skills/builder/x/y"), "page_builder");
  });

  it("没登记过的路径退回 IC,而不是抛", () => {
    assert.equal(subjectForPath("/nowhere"), "instrument_control");
    assert.equal(subjectForPath(""), "instrument_control");
  });

  it("每一个面都有自己的名字,没有两条路径撞在一起", () => {
    // 两条路径映射到同一个名字 = 反馈表里那两页的意见混成一堆,而分不开这件事
    // 只有等到有人想按页面看统计时才会发现,那时数据已经积了几个月。
    const paths = [
      "/agents", "/qa", "/literature", "/wishlist",
      "/skills/library", "/skills/builder",
      "/records/log", "/records/memory",
      "/settings/general", "/settings/setup", "/settings/admin", "/settings/usage",
      "/monitoring/current", "/monitoring/env",
      "/experimental/tools", "/experimental/optics",
    ];
    const seen = new Map<string, string>();
    for (const p of paths) {
      const s = subjectForPath(p);
      assert.equal(
        seen.has(s), false,
        `${p} 和 ${seen.get(s)} 都被记成 ${s} —— 反馈分不开了`,
      );
      seen.set(s, p);
    }
  });
});
