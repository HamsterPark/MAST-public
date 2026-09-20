/**
 * 结构闸门：每一个渲染**服务端**对话历史的界面，都必须接上自动刷新。
 *
 * ── 为什么需要这一条 ──
 * 「代理对话不会自动刷新」修好之后，仪器 chat 和代理
 * 对话是否也已经修好还不确定。去查，两处都还接着 —— 但**没有任何测试在守它**。
 * `lib/transcriptRefresh.ts` 有 27 条纯函数断言，钉的全是「什么时候不刷」的判定
 * 逻辑；后端有测试钉住 `publish_private_turn_finished` 真的发。中间那一段
 * ——「那一页确实调用了这个 hook」—— 一条都没有。把 ChatPage 和 AgentChatPanel
 * 里那两行 `useTranscriptRefresh({…})` 删掉，415 条测试全绿，而症状是回到 #31：
 * 别的机器、别的标签页、后台跑的回合写进去的东西一律看不见。
 *
 * 这正是「校验不能交给会犯这个错的那一方」的另一个面：判定层等价、只有**接线**
 * 不同的缺陷，判定层的钉子原理上抓不到，必须钉到接线本身那个字节。
 *
 * ── 判据为什么是端点而不是名单 ──
 * 人肉维护一张「哪些页面是对话」的清单，下一个新增的对话页不会记得往里加 ——
 * 这个仓已经为「每一页各自记得」的接线付过四次学费。所以判据从**行为**派生：
 * 谁去拉 `/api/agents/{agent_id}/messages`（服务端存的私聊正文，唯一真源），
 * 谁就得接自动刷新。新页面不接线就红，不需要任何人记得更新名单。
 *
 * 群聊走的是另一条推送（scope `transcript`，经 ConversationStore），不读这个
 * 端点，因此天然不在这条闸门里 —— 那是另一个机制，不该被这一条顺手管上。
 */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");

/** 服务端私聊正文的唯一读取端点。仪器 chat 和代理对话读的是同一个。 */
const HISTORY_ENDPOINT = "/api/agents/{agent_id}/messages";
const REFRESH_HOOK = "useTranscriptRefresh";

/**
 * 明知故犯的豁免。**空的**，而且应该一直是空的。
 *
 * 留这个口子是因为没有它的闸门会被下一个人整条删掉，而不是加一行豁免；
 * 但每加一项都必须在这里写下它为什么读了历史却不需要刷新。
 */
const EXEMPT = new Map<string, string>();

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) sourceFiles(p, out);
    else if (/\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/**
 * 去掉注释之后的源码。
 *
 * 有一条闸门找的是一个**错误写法**，而讲清楚那个错误写法的最好办法就是把它原样
 * 写在注释里 —— 于是不去注释的话，写下教训本身就会让闸门变红，下一个人的修法
 * 多半是删掉那段注释。`://` 不当行注释（`https://…`）。
 */
function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .split("\n")
    .map((line) => {
      const i = line.search(/(^|[^:])\/\//);
      return i === -1 ? line : line.slice(0, i);
    })
    .join("\n");
}

const FILES = sourceFiles(SRC).map((p) => {
  const text = readFileSync(p, "utf8");
  return {
    path: p,
    rel: relative(SRC, p).replace(/\\/g, "/"),
    text,
    code: stripComments(text),
  };
});

/** 读服务端私聊历史的文件。api/schema.d.ts 是生成的类型，不算调用方。 */
const READERS = FILES.filter(
  (f) => f.text.includes(HISTORY_ENDPOINT) && !f.rel.startsWith("api/"),
);

describe("私聊历史的自动刷新接线 (#31 / #36)", () => {
  // 闸门自检。一个匹配不到任何东西的判据永远是绿的 —— 而它和「全部通过」
  // 长得一模一样。端点串改名、目录挪走、后缀过滤写错，都会让上面那条
  // filter 静默变空，于是这条闸门在什么都不检查的状态下继续报绿。
  it("闸门自己找得到东西（否则它就是个假警报）", () => {
    assert.ok(
      READERS.length >= 2,
      `只找到 ${READERS.length} 个读 ${HISTORY_ENDPOINT} 的文件 —— ` +
        "预期至少两个（仪器 chat 与代理对话）。端点串或扫描范围可能已经过时。",
    );
  });

  it("仪器 chat 与代理对话都在扫描结果里", () => {
    const rels = READERS.map((f) => f.rel);
    assert.ok(rels.includes("pages/ChatPage.tsx"), `仪器 chat 不在: ${rels.join(", ")}`);
    assert.ok(
      rels.includes("components/agents/AgentChatPanel.tsx"),
      `代理对话不在: ${rels.join(", ")}`,
    );
  });

  it("每个读服务端历史的界面都调用了 useTranscriptRefresh", () => {
    const missing = READERS.filter(
      (f) => !EXEMPT.has(f.rel) && !f.text.includes(`${REFRESH_HOOK}({`),
    ).map((f) => f.rel);
    assert.deepEqual(
      missing,
      [],
      `这些界面渲染服务端对话历史却没接自动刷新（症状=回到 #31：只有本浏览器` +
        `自己发的那条会出现）：${missing.join(", ")}`,
    );
  });

  // hook 存在 ≠ 接上了。`useTranscriptRefresh` 只在传了 conversationId 与
  // updatedAt 时才有两个信号源；漏掉 updatedAt 会让 WS 断线后的兜底整条消失，
  // 而这在开发机上永远看不出来（本机 WS 从不断）。
  it("每个调用都同时给了会话 id 与 updated_at 兜底", () => {
    for (const f of READERS) {
      if (EXEMPT.has(f.rel)) continue;
      const i = f.text.indexOf(`${REFRESH_HOOK}({`);
      const call = f.text.slice(i, i + 600);
      assert.match(call, /conversationId:/, `${f.rel}: 调用里没有 conversationId`);
      assert.match(call, /updatedAt:/, `${f.rel}: 调用里没有 updatedAt（WS 断线就再也不刷了）`);
      assert.match(call, /onRefresh:/, `${f.rel}: 调用里没有 onRefresh（收到通知也不会去取）`);
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // 第三段：取回来之后**画不画**（，同一条反馈的第三次）
  //
  // 上面那几条钉的是「送达」：后端发得出、这一页接得住、参数没漏。#41 报的那次，
  // 这三条全是绿的 —— 通知发了、事件收了、`invalidateQueries` 也重取了 ——
  // 而屏幕不变。因为两页画的时候都写着
  //
  //     const shown = streamMsgs ?? history.data?.messages ?? [];
  //
  // 本浏览器上一轮 SSE 的快照只在换会话时清空，于是它一直压在重取回来的历史
  // 前面。「切到别的页面再回来才显示」= 组件重挂让快照归 null。
  //
  // 判定层的钉子抓不到这个：判定完全正确，坏的是**用不用它的结果**。所以这一条
  // 钉到渲染那一行的字节上。
  // ══════════════════════════════════════════════════════════════════════════

  /** 「谁先有值谁赢」的那个写法。它是能编译、能跑、且默默作废整条刷新链的。 */
  const SHADOW_PATTERN = /\w+\s*\?\?\s*history\.data\?\.messages/;
  const SOURCE_FN = "transcriptSource({";

  // 闸门自检（同上面那条）：一个匹配不到任何东西的正则永远是绿的，而它和
  // 「全部通过」长得一模一样。这里的失效方式尤其安静 —— `stripComments` 若把
  // 代码也削掉，下面那条就在什么都不看的状态下继续报绿。
  it("这条判据自己认得出那个错误写法", () => {
    assert.ok(
      SHADOW_PATTERN.test(stripComments("const shown = streamMsgs ?? history.data?.messages ?? [];")),
      "正则已经认不出它要拦的那个写法了",
    );
    assert.ok(
      !SHADOW_PATTERN.test(stripComments("// const shown = streamMsgs ?? history.data?.messages;")),
      "注释没有被剥掉",
    );
    const chat = FILES.find((f) => f.rel === "pages/ChatPage.tsx")!;
    assert.ok(
      chat.code.includes("shownMessages"),
      "stripComments 削掉了代码本身 —— 下面几条会在什么都不看的状态下报绿",
    );
  });

  it("没有任何一页用「谁先有值谁赢」决定画哪一份", () => {
    const offenders = FILES.filter((f) => SHADOW_PATTERN.test(f.code)).map((f) => f.rel);
    assert.deepEqual(
      offenders,
      [],
      "这些文件把 SSE 快照直接挡在服务端历史前面（症状：新消息不显示，" +
        `切到别的页面再回来才显示）：${offenders.join(", ")}`,
    );
  });

  it("每个读服务端历史的界面都用同一条判据决定画哪一份", () => {
    const missing = READERS.filter(
      (f) => !EXEMPT.has(f.rel) && !f.text.includes(SOURCE_FN),
    ).map((f) => f.rel);
    assert.deepEqual(
      missing,
      [],
      `这些界面重取了历史却没有说清楚什么时候该画它：${missing.join(", ")}`,
    );
  });

  // 判据要能分出「新旧」，就必须两个时间戳都在手上。少哪一个都会退化成一个
  // **恒定**的答案：没有 streamEndedAt 就永远不换，没有 dataUpdatedAt 就永远换
  // —— 后者会在流刚结束时把刚说完的一轮抹回上一轮。
  it("两个时间戳都真的传进去了", () => {
    for (const f of READERS) {
      if (EXEMPT.has(f.rel)) continue;
      const i = f.text.indexOf(SOURCE_FN);
      const call = f.text.slice(i, i + 400);
      assert.match(call, /streamEndedAt/, `${f.rel}: 判据里没有 streamEndedAt`);
      assert.match(call, /historyUpdatedAt:\s*\w+\.dataUpdatedAt/,
        `${f.rel}: historyUpdatedAt 不是查询真实的 dataUpdatedAt`);
      // 戳要被写过，否则它是个永远为 0 的常量（判据恒定，闸门却是绿的）。
      assert.match(f.text, /setStreamEndedAt\(Date\.now\(\)\)/,
        `${f.rel}: 流结束时没有记下时刻 —— 判据退化成「永远不换」`);
    }
  });

  it("豁免名单里的每一项都写了理由", () => {
    for (const [rel, why] of EXEMPT) {
      assert.ok(why.trim().length > 0, `${rel} 的豁免没写理由`);
      assert.ok(
        FILES.some((f) => f.rel === rel),
        `豁免名单里的 ${rel} 已经不存在了 —— 陈旧的豁免会静默盖住新文件`,
      );
    }
  });
});
