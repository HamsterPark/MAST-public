/**
 * 结构闸门：**重建 `ChatMsg` 时不许把 `t` 丢掉。**
 *
 * ── 为什么需要这一条 ──
 *
 * 2026-08-12 给消息加了 `t`（发生时刻，操作员：「agent 的发言也带上时间就更好
 * 了」）。后端盖上了、`render_history` 吐出来了、API schema 也声明了 ——
 * 而 `ChatPage.tsx` 里**两处**把消息重建成 `{role, content}` 的 `.map`
 * 原样把它丢了：
 *
 *     const base: ChatMsg[] = shownMessages.map((m) => ({
 *       role: m.role,
 *       content: m.content,          // ← t 没了
 *     }));
 *     setStreamMsgs(frame.messages.map((m) => ({ role: m.role, content: m.content })));
 *
 * 第二处更要命：那是**长任务跑的时候操作员实际盯着的那一路**，而旁白正是按 `t`
 * 排的 —— 丢掉它，气泡和旁白就不在同一根时间轴上，而「顺序看着有点乱」正是他
 * 一开始报的那个症状。
 *
 * **`tsc` 一声不吭**，因为 `t` 是可选字段：少一个可选字段的对象仍然是合法的
 * `ChatMsg`。这就是本仓「生产方接上了、消费方把它丢了」的同一个形状 ——
 * 全链路每一环都对，只有中间某一次手写的重建把它漏了，而漏掉的症状只是
 * 「时间偶尔没有」，没人会去查。
 *
 * ── 判据 ──
 *
 * 源码里凡是**同时**写了 `role:` 和 `content:` 的对象字面量（那就是在重建一条
 * 消息），必须也写 `t`。放行两种：显式写了 `t`，或者用了展开（`...m`）——
 * 后者本来就带上了全部字段。
 *
 * 这是**结构**判据不是类型判据：正因为类型系统在这里没有信息量，才需要它。
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const SRC = fileURLToPath(new URL("../src", import.meta.url));

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/**
 * 找出「把一条**已有**消息重新投影成新对象」的字面量 —— 也就是 bug 的确切形状：
 *
 *     xs.map((m) => ({ role: m.role, content: m.content }))
 *                            ↑ 从 m 抄，却没抄 m.t
 *
 * 判据是**两个值都取自同一个标识符**（`m.role` / `m.content`）。这样收窄之后：
 *
 *  · 抓得住三处真的（ChatPage ×2、AgentChatPanel ×2）；
 *  · 放过**新造**的字面量（`{ role: "user", content: escapeHtml(text) }`）——
 *    它不是在抄一条已有消息，「有没有 t」是它自己的决定（对话页给了本地时刻，
 *    QaPanel 压根没有时间概念）；
 *  · 放过**类型声明**（`{ role: "user" | "assistant"; content: string }`）。
 *
 * 第一版没收窄，误报了 QaPanel 与 CognitionPage —— 一条会误报的闸门会被下一个
 * 人加豁免、再下一个人删掉，那时它防的东西就没人守了。
 */
function reprojectedMessages(text: string): string[] {
  const out: string[] = [];
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    const body = m[0];
    const role = body.match(/\brole\s*:\s*([A-Za-z_$][\w$]*)\.role\b/);
    const content = body.match(/\bcontent\s*:\s*([A-Za-z_$][\w$]*)\.content\b/);
    if (!role || !content || role[1] !== content[1]) continue;
    // 一条聊天消息的重投影**就只有** role + content（+ t）。字段一多，那就是
    // 别的东西 —— `CognitionPage` 的辩论帧有 round / speaker / viewpoint…，
    // 它同样从一个对象里抄 role 和 content，但它不是聊天消息，也没有 `t`。
    // 不加这一条，闸门会对一个跟它无关的页面长期报警，而**一条会误报的闸门
    // 迟早会被人删掉**，那时它防的东西就没人守了。
    const keys = (body.match(/[{,]\s*[A-Za-z_$][\w$]*\s*:/g) || []).length;
    if (keys > 3) continue;
    out.push(body);
  }
  return out;
}

describe("重投影一条消息时必须带上 t", () => {
  it("全仓没有一处 {role: m.role, content: m.content} 把 t 漏掉", () => {
    const offenders: string[] = [];
    for (const file of walk(SRC)) {
      for (const lit of reprojectedMessages(readFileSync(file, "utf8"))) {
        // 放行：显式写了 t，或者展开了整条消息（`...m` 本来就带全部字段）。
        if (/\bt\s*:/.test(lit) || /\.\.\./.test(lit)) continue;
        offenders.push(`${relative(SRC, file)}: ${lit.replace(/\s+/g, " ").slice(0, 110)}`);
      }
    }
    assert.deepEqual(
      offenders,
      [],
      "这些地方把一条已有消息重投影成了新对象，却没抄上 `t`（发生时刻）。\n" +
        "tsc 不会报错——`t` 是可选字段——而症状只是「气泡上的时间偶尔没有」，\n" +
        "以及气泡和旁白不在同一根时间轴上（那正是操作员报的「顺序看着有点乱」）。\n" +
        offenders.join("\n"),
    );
  });

  it("闸门自身有判别力 —— 反例必须被抓到，而正例与无关形状必须放过", () => {
    // 没有这一条，上面那条可能因为正则永远匹配不到任何东西而「恒绿」。
    const bad = `xs.map((m) => ({ role: m.role, content: m.content }))`;
    assert.equal(reprojectedMessages(bad).length, 1, "正则连这个明显的反例都抓不到");

    const good = `xs.map((m) => ({ role: m.role, content: m.content, t: m.t }))`;
    assert.ok(/\bt\s*:/.test(reprojectedMessages(good)[0]!), "带了 t 的写法应当被放行");

    // 新造的字面量不是重投影 —— 它有没有时间是它自己的决定。
    assert.equal(reprojectedMessages(`({ role: "user", content: escapeHtml(t) })`).length, 0);
    // 类型声明不是值。
    assert.equal(reprojectedMessages(`type M = { role: string; content: string }`).length, 0);
    // 两个值取自**不同**对象 ⇒ 不是在抄同一条消息。
    assert.equal(reprojectedMessages(`({ role: a.role, content: b.content })`).length, 0);
    // 字段一多就不是聊天消息了（CognitionPage 的辩论帧就是这个形状）。
    assert.equal(
      reprojectedMessages(
        `({ round: f.round, speaker: f.speaker, role: f.role, viewpoint: f.v, content: f.content })`,
      ).length,
      0,
      "辩论帧被当成了聊天消息 —— 这条闸门会对一个无关页面长期报警",
    );
  });
});
