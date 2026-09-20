// ════════════════════════════════════════════════════════════════════════════
// transcriptRefresh — 「刚才那件事，值得我重读一次转录吗」的**纯**判断。
//
// 与 `hooks/useTranscriptRefresh.ts` 的关系，同 `lib/ws.ts` 与
// `hooks/useWsEvents.ts`：这里没有 React import，所以每一条判断都能在 node --test
// 里直接跑（`frontend/test/transcriptRefresh.test.ts`）。hook 那边只剩订阅与 ref。
//
// 之所以要把它抠出来：这些判断全是「什么时候**不**刷」，而漏掉一条的症状是
// 「偶尔跳一下」——不会崩、截图看不出、类型检查更管不着。只有测试能守住。
// ════════════════════════════════════════════════════════════════════════════

/** WS `experiment` 事件的 payload 里我们关心的那几个字段。 */
export interface TranscriptEventData {
  scope?: unknown;
  conversation_id?: unknown;
  agent_id?: unknown;
  t?: unknown;
  seq?: unknown;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

/**
 * 这条事件是不是「**我正在看的那个私聊**刚写完一个回合」。
 *
 * `scope` 必须严格等于 `"private_turn"`：群聊转录用的是 `"transcript"`，
 * 两者的 payload 形状不同（那边带 seq，这边没有），混用会让一方按另一方的
 * 字段去做去重 —— 而缺字段的那一侧 `Number(undefined ?? 0) === 0`，
 * 去重条件恒真，症状是「一条都不刷」。
 *
 * `streaming` 为真时一律否：正在流的时候页面显示的是 SSE 增量快照，中途插一次
 * 整 history 重读会让已经画出来的半句话跳回上一轮的结尾。流结束时它自己会失效。
 */
export function isPrivateTurnFor(
  data: TranscriptEventData | null | undefined,
  conversationId: string | null | undefined,
  streaming = false,
): boolean {
  if (!data || streaming) return false;
  if (str(data.scope) !== "private_turn") return false;
  if (!conversationId) return false;
  return str(data.conversation_id) === conversationId;
}

/**
 * 这条事件是不是「**我正在看的那个 agent** 在群聊里说了话」，以及它的时间游标。
 *
 * 返回 `t = 0` 表示「事件里没有可用的游标」。调用方据此**照刷不误** —— 拿 0 去和
 * 已见过的游标比会恒假（`0 <= seen`），于是一条没带 t 的事件永远刷不出东西。
 * 「没有游标」和「游标很旧」必须是两种处理。
 */
export function transcriptCursorFor(
  data: TranscriptEventData | null | undefined,
  agentId: string,
): { ok: boolean; t: number } {
  if (!data || !agentId) return { ok: false, t: 0 };
  if (str(data.scope) !== "transcript") return { ok: false, t: 0 };
  if (str(data.agent_id) !== agentId) return { ok: false, t: 0 };
  const raw = data.t;
  const t = typeof raw === "number" && Number.isFinite(raw) && raw > 0 ? raw : 0;
  return { ok: true, t };
}

// ── 取回来之后画哪一份（#31 → #36 → #41 的第三次落点） ───────────────────────
//
// 前两轮修的都是**送达**：#31 把「私聊正文不走那张表、所以那条推送对它一次都没
// 发过」补上，#36 补上「函数会发」与「那一页接了」中间那段闸门。两轮之后，
// 通知发得出、事件收得到、`invalidateQueries` 也确实重取了 —— 而屏幕不变。
//
// 因为**画的时候没用那份新数据**：两页都写着
//
//     const shown = streamMsgs ?? history.data?.messages ?? [];
//
// `streamMsgs` 是本浏览器自己上一轮 SSE 的最终快照，只在换会话/换 agent 时清空。
// 只要用户在这个会话里发过一句话，它就一直非空、一直压在重取回来的历史前面。
// 切到别的页面再回来之所以「就好了」，是因为组件卸载重挂，`streamMsgs` 归 null
// —— 这正是用户第三次报的那句话的形状。
//
// 于是判据不能再是「谁先有值」，得是「谁更新」。
export type TranscriptSource = "stream" | "history";

/**
 * 画流里的快照，还是画服务端历史。
 *
 * 规则只有一条：**服务端在这一轮流结束之后又被读过一次，就以它为准。**
 * 两者本来就是同一个真源（`/messages` 与 SSE 快照都出自 checkpointer 的
 * `render_history`），所以「换成历史」不会丢掉刚说完的这一轮。
 *
 * 三条护栏，方向各不相同：
 * * `streaming` 期间一律留在快照 —— 中途换成整 history 会让已经画出来的半句话
 *   跳回上一轮的结尾；
 * * `historyUpdatedAt === 0`（历史一次都没取回来）时留在快照 —— 否则一次
 *   失败的重取会把屏幕清空；
 * * `streamEndedAt === 0`（这一轮还没结束过）时留在快照。
 */
export function transcriptSource({
  hasSnapshot,
  streaming,
  streamEndedAt,
  historyUpdatedAt,
}: {
  /** 手上有没有一份 SSE 快照。 */
  hasSnapshot: boolean;
  /** 本浏览器此刻正在流这个会话吗。 */
  streaming: boolean;
  /** 本浏览器这一轮流结束的时刻（`Date.now()`）。0 = 还没流过。 */
  streamEndedAt: number;
  /** 服务端历史最近一次取回来的时刻（react-query `dataUpdatedAt`）。0 = 从未。 */
  historyUpdatedAt: number;
}): TranscriptSource {
  if (!hasSnapshot) return "history";
  if (streaming) return "stream";
  if (streamEndedAt <= 0 || historyUpdatedAt <= 0) return "stream";
  return historyUpdatedAt > streamEndedAt ? "history" : "stream";
}

/** 已经据之刷过的游标。`null` = 还没见过任何游标。 */
export type Cursor = string | null;

/**
 * `updated_at` 游标前进了吗 —— 前进就重读一次。
 *
 * **首见即记账、不刷**：挂载那一刻的 `updated_at` 描述的正是页面刚取回来的那份
 * 历史，把它当成「有更新」会让每一次挂载都白读一次（切一次子 tab 一次，
 * 而这一页恰恰是条件渲染、切走就卸载的）。
 *
 * 空值不推进游标：`updated_at` 缺失是「这一行没带这个字段」，不是「回到了从前」。
 * 拿它把 `seen` 清成 null，下一次真更新就会被当成首见而吞掉。
 */
export function advanceCursor(
  seen: Cursor,
  updatedAt: string | null | undefined,
): { seen: Cursor; refresh: boolean } {
  if (!updatedAt) return { seen, refresh: false };
  if (seen === null) return { seen: updatedAt, refresh: false };
  if (seen === updatedAt) return { seen, refresh: false };
  return { seen: updatedAt, refresh: true };
}
