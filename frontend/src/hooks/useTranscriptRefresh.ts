// ════════════════════════════════════════════════════════════════════════════
// useTranscriptRefresh — 「别人写进来的消息，这一页看得见吗」
//
// 已知问题：智能体代理对话不会自动刷新。根因是消息历史只在**本浏览器
// 自己发完一条**之后才失效一次（SSE 流结束 → invalidateQueries），别的来源写进去
// 的东西这一页永远看不到：另一台机器、同一台的另一个标签页、后台运行的 agent、
// 别人替这个 agent 插的话。
//
// 形状照抄仓里已经做对的那一处（`components/agents/RunTaskPanel.tsx` L109-135）：
// **事件只推游标，正文走已鉴权的读端点**。三条护栏一条都不能少 ——
//
//   1. 会话过滤：别的会话的事件不能刷我这一页；
//   2. 自流不刷：本浏览器正在流的时候不抢，流结束时它自己会失效一次；
//   3. 去重：一串事件合并成一次刷新，不是一次事件一个请求。
//
// 为什么同时还看 `updated_at` 而不是只信 WS：WS 断线时（Tailscale 隧道抖动、
// 服务重启）推送就没了，而会话列表那条 6 s 轮询**本来就在跑**——拿它已经取回来的
// 一个字符串当游标，兜底是免费的。这不是「两套机制」，是同一条规则的两个信号源：
// 「有个比我手上更新的游标出现了 ⇒ 去读一次」。
// ════════════════════════════════════════════════════════════════════════════

import { useEffect, useRef } from "react";
import { useWsEvent } from "@/hooks/useWsEvents";
import {
  advanceCursor,
  isPrivateTurnFor,
  type Cursor,
  type TranscriptEventData,
} from "@/lib/transcriptRefresh";

/** 一次刷新请求。调用方自己决定怎么重读（invalidate / refetch / hydrate）。 */
export type RefreshFn = () => void;

export interface TranscriptRefreshOptions {
  /** 当前正在看的会话。`null` = 什么都没选，不刷。 */
  conversationId: string | null | undefined;
  /**
   * 这个会话最近一次变动的戳（`Conversation.updated_at`）。
   *
   * 从**已经在轮询**的会话列表里取，不要为它单开一个请求 —— 那样这个兜底就从
   * 「免费」变成了「每 N 秒一次额外往返」，而它存在的理由正是不额外要钱。
   */
  updatedAt?: string | null;
  /**
   * 本浏览器此刻正在流这个会话吗。真 = 不刷。
   *
   * 抢刷的后果不是「多一次请求」：正在流的时候页面显示的是 SSE 的增量快照，
   * 中途插一次整history 重读会让已经画出来的半句话跳回上一轮的结尾。
   */
  streaming?: boolean;
  /** 去重之后真正要做的事。 */
  onRefresh: RefreshFn;
  /** 关掉整条订阅（组件在隐藏的子 tab 里时用）。 */
  enabled?: boolean;
}

/**
 * 让一页转录在**别人**写进来的时候也刷新。
 *
 * 两个信号源，同一条规则：
 * * WS `experiment` 事件，`scope === "private_turn"`（私聊回合写完，
 *   由 `mast.chat.store.publish_private_turn_finished` 发）—— 近实时；
 * * `updatedAt` 变化 —— WS 断了也还在，因为会话列表本来就在轮询。
 */
export function useTranscriptRefresh({
  conversationId,
  updatedAt,
  streaming = false,
  onRefresh,
  enabled = true,
}: TranscriptRefreshOptions): void {
  // handler 存 ref：调用方基本都会传内联箭头，把它放进依赖会让 useWsEvent 每次
  // 渲染都重订阅，而重订阅会把共享 socket 的引用计数打到 0 再拉起来。
  const refreshRef = useRef(onRefresh);
  refreshRef.current = onRefresh;
  const streamingRef = useRef(streaming);
  streamingRef.current = streaming;

  // 最近一次「已经据此刷过」的游标。会话一换就清空 —— 不清的话，切到另一个会话
  // 之后它手里还攥着上一个会话的戳，而两个会话的戳没有可比性。
  const seen = useRef<Cursor>(null);
  useEffect(() => {
    seen.current = null;
  }, [conversationId]);

  useWsEvent(
    "experiment",
    (event) => {
      const d = (event.data ?? {}) as TranscriptEventData;
      if (!isPrivateTurnFor(d, conversationId, streamingRef.current)) return;
      refreshRef.current();
    },
    enabled && !!conversationId,
  );

  useEffect(() => {
    if (!enabled || !conversationId || streamingRef.current) return;
    const next = advanceCursor(seen.current, updatedAt);
    seen.current = next.seen;
    if (next.refresh) refreshRef.current();
  }, [enabled, conversationId, updatedAt]);
}
