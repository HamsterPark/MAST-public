import { useRef } from "react";
import clsx from "clsx";
import { AgentIcon, agentClasses, agentColorVar, agentDef } from "@/components/agents/registry";
import { useStickToBottom } from "@/hooks/useStickToBottom";
import { InjectedContext } from "@/components/chat/InjectedContext";
import { fmtClock } from "@/lib/narration";
import { userTurnTimes } from "@/lib/injectedContext";

/** Rendered chat history. Backend `content` is PRE-RENDERED, SANITIZED HTML
 *  (markdown → HTML, thinking wrapped in <details>; see mast.chat.render +
 *  chat.py _esc_llm). We therefore render it with dangerouslySetInnerHTML inside
 *  a styled wrapper — never re-escape, never re-parse. */

/** `t` = 这条消息发生的 epoch 秒，由后端 `MessageClockMiddleware` 盖上。
 *
 *  **可选，而且它的缺席有意义**：没有 `t` 表示「不知道它是什么时候说的」
 *  （重启前就在 checkpoint 里的历史），不表示「零时刻」。所以这里不给默认值，
 *  渲染时 `fmtClock` 对无效值返回空串——**绝不显示 1970**。
 *  Assistant messages carry a timestamp too, for the same reason.
 *
 *  `null` 必须写进类型里：后端 `ChatMessage.t` 是 `float | None`，FastAPI 会把
 *  它序列化成 **`"t": null`**（不是省略这个键）。只写 `number | undefined` 时
 *  `tsc` 当场报 TS2322——那是类型在说实话，不是碍事：运行时 `fmtClock` 早就
 *  认得 null，而类型不认，两者一旦分家，下一个人会照类型写出一个错的判据。 */
export type ChatMsg = { role: string; content: string; t?: number | null };

const ROLE_LABEL: Record<string, string> = {
  user: "你",
  assistant: "助手",
  system: "系统",
  tool: "工具",
};

// Tailwind arbitrary-variant restyle of the pre-rendered HTML body. We cannot
// touch the inner markup (it is sanitized backend HTML), so we reach into it:
//   • the <details>…</details> thinking / CoT block → code-bg card, muted text
//   • inline numeric values (bias/setpoint, wrapped in <code> by the renderer)
//     → mono + accent so params read like the canvas
// Class strings stay complete literals so Tailwind never purges them.
const CHAT_HTML_CLASS = clsx(
  "mast-chat-html text-sm leading-relaxed text-mast-text [overflow-wrap:anywhere]",
  // thinking / chain-of-thought (<details>) → bg-mast-code-bg rounded-mast-ctl text-mast-muted
  "[&_details]:my-2 [&_details]:rounded-mast-ctl [&_details]:bg-mast-code-bg [&_details]:px-3 [&_details]:py-2",
  "[&_details]:text-xs [&_details]:text-mast-muted [&_summary]:cursor-pointer [&_summary]:select-none",
  "[&_summary]:text-mast-dream [&_summary]:font-medium",
  // inline numeric values / params → font-mono text-mast-accent (tabular)
  "[&_code]:font-mono [&_code]:tabular-nums [&_code]:text-mast-accent",
  "[&_pre]:rounded-mast-ctl [&_pre]:bg-mast-code-bg [&_pre]:p-3 [&_pre]:text-xs [&_pre]:text-mast-muted",
  "[&_pre_code]:text-mast-text",
);

/** Small agent chip: 18px rounded-mast-badge filled with the agent hue, mono
 *  initials (matches the canvas assistant-message header chip). */
function AgentChip({ agentId }: { agentId: string }) {
  const a = agentDef(agentId);
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-mast-badge font-mono text-[9px] font-bold text-white"
        style={{ background: agentColorVar(agentId) }}
      >
        {a.short}
      </span>
      <span className={clsx("inline-flex items-center", agentClasses(agentId).text)}>
        <AgentIcon id={agentId} size={12} />
      </span>
      <span className="text-[11.5px] font-medium text-mast-text">{a.cn} 助手</span>
    </span>
  );
}

function Bubble({
  msg,
  agentId,
  turnTimes,
}: {
  msg: ChatMsg;
  agentId: string;
  /** 整份转录里每一轮的起点（升序）。见 `lib/injectedContext.ts::userTurnTimes`。 */
  turnTimes: number[];
}) {
  const isUser = msg.role === "user";
  // 取不到就整个不渲染。带时间的用处不是计时，是**让顺序可核**：旁白和消息
  // 现在共用一根时间轴，一条排在前面却时间更晚的条目会当场露出来——在此之前
  // 那种错只能靠一句「感觉有点乱」被发现。
  const clock = fmtClock(msg.t);

  // USER → accent-aligned (right), accent-soft fill — unchanged identity.
  if (isUser) {
    return (
      <div className="flex w-full justify-end">
        <div className="max-w-[80%] rounded-mast-card rounded-br-[3px] border border-mast-accent/40 bg-mast-accent-soft px-3.5 py-2.5">
          {clock && (
            <div className="mb-1 text-right font-mono text-[10px] tabular-nums text-mast-faint">
              {clock}
            </div>
          )}
          <div className="text-sm leading-relaxed text-mast-text [overflow-wrap:anywhere]">
            <div
              className="mast-chat-html"
              dangerouslySetInnerHTML={{ __html: msg.content }}
            />
          </div>
          {/* 「我这句话发出去之后，系统往模型那儿塞了什么」。折叠着不发任何请求。
              挂在**用户**气泡下而不是助手气泡下：注入发生在这句话之后、
              回答之前，而用户当场要问的就是「它是带着什么去想的」。 */}
          <InjectedContext t={msg.t} turnTimes={turnTimes} agentId={agentId} />
        </div>
      </div>
    );
  }

  // SYSTEM / TOOL → neutral panel (no agent identity).
  const isAgentMsg = msg.role === "assistant";
  const cls = agentClasses(agentId);

  return (
    <div className="flex w-full justify-start">
      <div
        className={clsx(
          "max-w-[80%] overflow-hidden rounded-mast-card border border-mast-border bg-mast-panel shadow-mast",
          isAgentMsg && [cls.borderL, "rounded-l-none"],
        )}
      >
        {/* header strip with the agent chip (assistant) or plain role label */}
        <div className="flex items-center gap-2 border-b border-mast-border bg-mast-panel-2 px-3.5 py-2">
          {isAgentMsg ? (
            <AgentChip agentId={agentId} />
          ) : (
            <span className="text-[11.5px] font-medium text-mast-muted">
              {ROLE_LABEL[msg.role] ?? msg.role}
            </span>
          )}
          {clock && (
            <span className="ml-auto font-mono text-[10px] tabular-nums text-mast-faint">
              {clock}
            </span>
          )}
        </div>
        <div className="px-3.5 py-3">
          <div
            className={CHAT_HTML_CLASS}
            dangerouslySetInnerHTML={{ __html: msg.content }}
          />
        </div>
      </div>
    </div>
  );
}

export function ChatBubbles({
  messages,
  pending,
  agentId = "instrument_control",
  autoScroll = true,
  allMessages,
}: {
  messages: ChatMsg[];
  pending?: boolean;
  // Which agent owns this transcript (drives the chip + left-border color).
  // The main console chat is the IC private chat, so IC is the default.
  agentId?: string;
  /**
   * Scroll this block's end into view on every update. Default true (the whole
   * transcript is one block). NarrationLane renders SEVERAL blocks with
   * narration cards between them, and every block scrolling itself into view
   * means N competing smooth-scrolls per render — the view lands wherever the
   * race ends. Only the last block sets this.
   */
  autoScroll?: boolean;
  /**
   * **整份**转录（`messages` 可能只是其中一段）。只用来算「每一轮从哪儿开始」，
   * 给注入上下文那个折叠块定右边界。
   *
   * 必须是整份：`NarrationLane` 按旁白卡片把消息切成好几段分别丢进来，一段里的
   * 最后一条用户消息在整份转录里后面往往还有。只看这一段 → 那一条的时间窗开到
   * 无穷大 → 它会把后面所有轮的请求都算成自己这一轮的。
   *
   * 不传就退回 `messages`（未分段的调用方，行为不变）。
   */
  allMessages?: ChatMsg[];
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const turnTimes = userTurnTimes(allMessages ?? messages);

  // ⚠️ 这里原来是无条件 `endRef.scrollIntoView({behavior:"smooth"})`。
  //
  // 已知问题:对话的滚动条和整个标签的滚动条都会自动锁定,
  // 想向下翻查看历史都不行,会自动被拉回,两个滚动条都是这样。两个毛病:
  //
  //   ① 不问用户在看哪儿 —— 往上翻历史,下一条消息就把他弹回底部;
  //   ② `scrollIntoView` 会滚**每一个可滚动祖先** —— 所以是「两个」滚动条。
  //
  // 换成只在用户本来就贴着底部时,滚**最近的那一个**容器。
  useStickToBottom(endRef, [messages, pending], autoScroll);

  const cls = agentClasses(agentId);

  return (
    <div className="flex flex-col gap-3.5">
      {messages.map((m, i) => (
        <Bubble key={`${i}-${m.role}`} msg={m} agentId={agentId}
                turnTimes={turnTimes} />
      ))}
      {pending && (
        <div className="flex justify-start">
          <div
            className={clsx(
              "rounded-mast-card rounded-l-none border border-mast-border bg-mast-panel px-3.5 py-2.5 text-sm text-mast-muted shadow-mast",
              cls.borderL,
            )}
          >
            <span className="inline-flex items-center gap-2">
              <span className={clsx("inline-flex", cls.text)}>
                <AgentIcon id={agentId} size={13} />
              </span>
              {agentDef(agentId).cn}思考中…
            </span>
          </div>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}
