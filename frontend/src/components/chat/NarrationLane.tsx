import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/api/client";
import { useWsEvent } from "@/hooks/useWsEvents";
import { ChatBubbles, type ChatMsg } from "@/components/chat/ChatBubbles";
import {
  fmtClock,
  isNarrationFor,
  mergeBatch,
  recallNarration,
  rememberNarration,
  mergeNarration,
  narrationImageUrl,
  nextCursor,
  toneClass,
  type NarrationItem,
} from "@/lib/narration";

// ════════════════════════════════════════════════════════════════════════
//  旁白（narration）—— 长任务跑的时候，持续说给用户听。
//
//  长任务跑的时候，中间 agent 只会返回一个最终结果，中途没有任何反馈会让人
//  没有安全感。这条道就是那个答案：要打一发脉冲了、扫到 50% 了、
//  这块地方不行换个地方。
//
//  它**刻意长得跟助手气泡完全不一样**：没有 agent 头像、没有左侧色条、字号更小、
//  带一个「旁白」小标签。把它画成助手说的话会让用户以为 agent 在自言自语 ——
//  而它其实是系统在解说，agent 一个字都看不见（后端 mast/chat/narration.py）。
//
//  排序规则是纯函数（src/lib/narration.ts + test/narration.test.ts）：
//  一个埋在 JSX 里的 filter 出了错的样子是「旁白偶尔排在奇怪的位置」，没人会查。
// ════════════════════════════════════════════════════════════════════════

function NarrationCard({
  item,
  conversationId,
}: {
  item: NarrationItem;
  conversationId: string | null;
}) {
  const src = narrationImageUrl(item, conversationId);
  const [imgFailed, setImgFailed] = useState(false);
  const clock = fmtClock(item.t);

  return (
    <div className="flex w-full justify-start">
      <div
        className={clsx(
          "max-w-[80%] rounded-mast-ctl border border-dashed bg-mast-panel-2/50 px-3 py-1.5",
          toneClass(item.tone),
        )}
      >
        <div className="flex items-baseline gap-2">
          <span className="shrink-0 rounded-mast-badge bg-mast-code-bg px-1.5 py-0.5 text-[10px] text-mast-faint">
            旁白
          </span>
          {/* 事件**发生**的时刻（不是说出来的时刻）。
              用处不是计时，是**让顺序可核**：一条排在前面的旁白如果时间更晚，
              那就是排序错了 —— 在此之前那种错只能靠「感觉有点乱」被发现。
              取不到就整个不渲染（`fmtClock` 返回空串），**不显示 1970**。 */}
          {clock && (
            <span
              className="shrink-0 font-mono text-[10px] tabular-nums text-mast-faint"
              title="这件事发生的时刻"
            >
              {clock}
            </span>
          )}
          <span className="text-[12.5px] leading-relaxed [overflow-wrap:anywhere]">
            {item.text}
          </span>
        </div>
        {src && !imgFailed && (
          <img
            src={src}
            loading="lazy"
            alt=""
            onError={() => setImgFailed(true)}
            className="mt-1.5 max-h-40 rounded border border-mast-border"
          />
        )}
        {src && imgFailed && (
          // 「取不到图」不是「没有图」。说清楚是哪一种，别画一个白框。
          <p className="mt-1 text-[11px] text-mast-faint">这一帧的画面取不到了。</p>
        )}
      </div>
    </div>
  );
}

/**
 * 转录 + 旁白的合并视图。`show=false` 时**完全等于**原来的 `<ChatBubbles>`，
 * 一个请求都不发 —— 「关掉」得是真的关掉，不是只把它藏起来。
 */
export function NarrationLane({
  messages,
  conversationId,
  pending,
  show,
}: {
  messages: ChatMsg[];
  conversationId: string | null;
  pending?: boolean;
  show: boolean;
}) {
  // 条目与游标**存在模块级 memo 里**,不是组件状态(2026-08-17)。
  //
  // 要求:「切换标签页再回来的时候旁白就没有了。」切走 = 组件卸载 =
  // useState/useRef 一起丢;而回来时 react-query 又先把上一次**增量**请求的
  // 缓存喂回来(只有最后两条),游标随即跳到末尾,前面几百条再也补不回来。
  // 见 lib/narration 里 recallNarration 的说明。
  const [items, setItems] = useState<NarrationItem[]>(
    () => recallNarration<NarrationItem>(conversationId).items);
  const cursor = useRef(recallNarration(conversationId).cursor);

  // 换会话 = 换一条转录：游标和已有条目都要切到那条会话自己的账上,
  // 否则新会话的前几条会被当成「已读」而永远不显示。
  useEffect(() => {
    const memo = recallNarration<NarrationItem>(conversationId);
    cursor.current = memo.cursor;
    setItems(memo.items);
  }, [conversationId]);

  const enabled = show && !!conversationId;
  const q = useQuery({
    queryKey: ["chat", "narration", conversationId],
    enabled,
    // 轮询是兜底：WS 断了（企业代理剥 Upgrade 头）旁白也得继续出现，只是慢几秒。
    refetchInterval: 4000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/chat/narration", {
        params: {
          query: { conversation_id: conversationId!, after_seq: cursor.current },
        },
      });
      if (error) throw error;
      return data;
    },
  });

  useEffect(() => {
    const data = q.data;
    if (!data) return;
    cursor.current = nextCursor(cursor.current, data.latest_seq);
    if (data.items?.length) {
      setItems((prev) => {
        const next = mergeBatch(prev, data.items as NarrationItem[]);
        rememberNarration(conversationId, next, cursor.current);
        return next;
      });
    } else {
      // 空批次也要记 —— 游标可能前进了(``latest_seq`` 跳过了非旁白行)。
      rememberNarration(conversationId, items, cursor.current);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q.data, conversationId]);

  // 实时推送：事件只带指针（会话 + seq），正文走上面那条已鉴权的读端点。
  // 总线只回放最后 100 条，一次长任务几百条旁白，正文放上去会把别的全挤掉。
  useWsEvent(
    "chat_narration",
    (event) => {
      if (isNarrationFor(event.data as { conversation_id?: unknown }, conversationId)) {
        void q.refetch();
      }
    },
    enabled,
  );

  if (!show || items.length === 0) {
    return <ChatBubbles messages={messages} pending={pending} />;
  }

  const lanes = mergeNarration(messages, items);
  // 连续的消息交给 ChatBubbles 整段渲染（它负责气泡样式与滚动到底），
  // 旁白卡片插在段之间。**不重画气泡** —— 那会变成第二份渲染实现。
  const blocks: Array<{ msgs: ChatMsg[]; after: NarrationItem[] }> = [];
  let current: { msgs: ChatMsg[]; after: NarrationItem[] } = { msgs: [], after: [] };
  for (const lane of lanes) {
    if (lane.kind === "message") {
      if (current.after.length) {
        blocks.push(current);
        current = { msgs: [], after: [] };
      }
      current.msgs.push(lane.message);
    } else {
      current.after.push(lane.item);
    }
  }
  blocks.push(current);

  return (
    <div className="flex flex-col gap-3.5">
      {blocks.map((block, i) => (
        <div key={i} className="flex flex-col gap-3.5">
          {(block.msgs.length > 0 || (pending && i === blocks.length - 1)) && (
            <ChatBubbles
              messages={block.msgs}
              pending={pending && i === blocks.length - 1}
              autoScroll={i === blocks.length - 1}
              // 分段渲染，但「每一轮从哪儿开始」必须按**整份**转录算 ——
              // 见 ChatBubbles 里 allMessages 的自述。
              allMessages={messages}
            />
          )}
          {block.after.map((item) => (
            <NarrationCard key={item.seq} item={item} conversationId={conversationId} />
          ))}
        </div>
      ))}
    </div>
  );
}
