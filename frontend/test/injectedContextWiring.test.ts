/**
 * 结构闸门：**分段渲染的 `<ChatBubbles>` 必须拿到整份转录。**
 *
 * ── 为什么需要这一条 ──
 *
 * 「这一轮往模型那儿塞了什么」是按时间窗对齐的：一轮 = [这条用户消息, 下一条用户
 * 消息)。而 `NarrationLane` 把消息按旁白卡片切成好几段，分别丢给 `ChatBubbles`。
 * 一段里的「最后一条用户消息」在**整份**转录里后面往往还有 —— 只看这一段就找不到
 * 右边界，那一轮的窗口开到无穷大，于是它把后面所有轮的请求都算成自己的。
 *
 * 症状：折叠块照常打开、内容照常渲染、类型全绿，只是「这一轮发出 N 次调用」那个
 * 数偏大，而展开的那份注入**可能是后面某一轮的**。一句看起来完全正常、内容却是错
 * 的话 —— 拿它调试会往错的方向找一整晚。
 *
 * ── 判据为什么是「循环里渲染」而不是一张名单 ──
 *
 * 名单要人记得更新，而这个仓已经为「每一页各自记得」的接线付过四次学费。
 * 判据从行为派生：**在 `.map(` 里渲染 `<ChatBubbles>` 就是在分段**，分段就必须
 * 传 `allMessages`。下一个写分段视图的人不传就红，不需要任何人记得。
 *
 *     cd frontend && npm run test:unit
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/** `<ChatBubbles …/>` 的每一处渲染（连同它前面那段上下文，用来判断在不在循环里）。 */
function bubbleUsages(src: string): { start: number; tag: string }[] {
  const out: { start: number; tag: string }[] = [];
  const re = /<ChatBubbles\b/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    // 取到配对的 `/>` 或 `>`：属性都在这一段里。
    const end = src.indexOf(">", m.index);
    out.push({ start: m.index, tag: src.slice(m.index, end + 1) });
  }
  return out;
}

/** 这一处渲染是不是在 `.map(` 的回调里 —— 往前找最近的 `.map(`，看它有没有闭合。 */
function insideMap(src: string, at: number): boolean {
  const before = src.slice(0, at);
  const mapAt = before.lastIndexOf(".map(");
  if (mapAt < 0) return false;
  // `.map(` 之后到这一处之间，括号还没配平 ⇒ 我们还在回调里面。
  let depth = 0;
  for (let i = mapAt + ".map(".length - 1; i < at; i++) {
    const c = src[i];
    if (c === "(") depth++;
    else if (c === ")") depth--;
    if (depth === 0 && i > mapAt) return false;
  }
  return depth > 0;
}

describe("分段渲染的转录必须拿到整份转录", () => {
  const files = walk(SRC).filter((f) => readFileSync(f, "utf8").includes("<ChatBubbles"));

  it("至少有人在渲染 ChatBubbles（闸门自校验）", () => {
    // 一条永远为真的闸门等于没有闸门。组件改名 / 这条路径没了，先在这里红。
    assert.ok(files.length > 0, "一处 <ChatBubbles> 都找不到 —— 闸门在空跑");
  });

  for (const f of files) {
    const rel = relative(SRC, f).replace(/\\/g, "/");
    const src = readFileSync(f, "utf8");
    for (const use of bubbleUsages(src)) {
      if (!insideMap(src, use.start)) continue;
      it(`${rel}: 循环里的 <ChatBubbles> 传了 allMessages`, () => {
        assert.match(
          use.tag,
          /allMessages=/,
          `${rel} 在 .map( 里渲染 <ChatBubbles> 却没传 allMessages。\n` +
            `分段之后每一段都不知道整份转录长什么样，注入上下文那个折叠块会把\n` +
            `后面所有轮的请求算进这一轮 —— 不报错，只是内容错。\n` +
            `这一处是：${use.tag}`,
        );
      });
    }
  }

  it("闸门认得出「在循环里」这件事（判据自校验）", () => {
    // 判据本身要能分辨，否则上面那批断言可能是**恒真**的 —— 一条从不触发的
    // 闸门和没有闸门是同一件事（本仓 `guard_that_isnt`）。
    const inLoop = `blocks.map((b, i) => (\n  <ChatBubbles messages={b.msgs} />\n))`;
    const notInLoop = `if (x) {\n  return <ChatBubbles messages={messages} />;\n}`;
    assert.equal(insideMap(inLoop, inLoop.indexOf("<ChatBubbles")), true);
    assert.equal(insideMap(notInLoop, notInLoop.indexOf("<ChatBubbles")), false);
  });
});
