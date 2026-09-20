import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button } from "@/components/controls";
import { ConfirmDialog } from "@/components/scope/ConfirmDialog";
import { readSseStream, sseBroke, sseEndMessage } from "@/lib/sse";
import { useTranscriptRefresh } from "@/hooks/useTranscriptRefresh";
import { transcriptSource } from "@/lib/transcriptRefresh";
import { fmtClock } from "@/lib/narration";
import { AGENTS, AgentIcon, Avatar, SUPERVISOR, agentClasses, agentColorVar, agentDef } from "./registry";
import { InterruptCard, fromPollRow, type ResolveArgs } from "./InterruptCard";

// @-mention roster (ITEM 4): the 8 agents (SUP + 7) the operator can @-target.
// Selecting a row inserts "@<agent_id> " at the caret — a cosmetic targeting
// hint mirroring the old interject "(指向 X)" convention; no backend routing.
const MENTIONABLE = [SUPERVISOR, ...AGENTS];

// Per-agent private chat — reproduces the old 代理对话 (per-agent threads):
// session list (create / rename / delete) + a live SSE transcript. The SSE
// frame protocol matches ChatPage: POST /api/agents/{id}/chat → text/event-stream
// of {kind:"snapshot",messages:[...]} | {kind:"error"} | {kind:"done"}.

/** `t` = 这条消息发生的 epoch 秒（后端 `MessageClockMiddleware` 盖上）。
 *
 *  可选，而且**它的缺席有含义**：没有 `t` = 「不知道它是什么时候说的」
 *  （重启前就在 checkpoint 里的历史），不是「零时刻」。`null` 要写进类型：
 *  后端是 `float | None`，FastAPI 会序列化成 `"t": null`。 */
type Msg = { role: string; content: string; t?: number | null };

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function AgentChatPanel({ agentId }: { agentId: string }) {
  const qc = useQueryClient();
  const a = agentDef(agentId);
  const cls = agentClasses(agentId);
  const hue = agentColorVar(agentId);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [interruptNote, setInterruptNote] = useState<string | null>(null);
  const [streamMsgs, setStreamMsgs] = useState<Msg[] | null>(null);
  // 本浏览器这一轮流结束的时刻。历史比它新 = 服务端在这一轮之后又被读过一次，
  // 那份才是要画的（，判据在 lib/transcriptRefresh.ts）。
  const [streamEndedAt, setStreamEndedAt] = useState(0);
  const [streamErr, setStreamErr] = useState<string | null>(null);
  // 原生 window.confirm 阻塞主线程且无失败反馈 —— 换成统一弹窗。
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const abortCtrl = useRef<AbortController | null>(null);

  // @-mention autocomplete (ITEM 4): when the operator types "@" we track the
  // partial query after it and offer the 8-agent roster; selecting inserts
  // "@<agent_id> ". Closed on space, Escape, or selection.
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const [mentionQuery, setMentionQuery] = useState<string | null>(null);
  const [mentionIdx, setMentionIdx] = useState(0);

  // current "@query" fragment immediately before the caret (null = popup closed)
  function detectMention(value: string, caret: number): string | null {
    const upto = value.slice(0, caret);
    const at = upto.lastIndexOf("@");
    if (at === -1) return null;
    // "@" must start the token (preceded by start/whitespace) and have no space after
    const before = at === 0 ? "" : upto[at - 1];
    if (before && !/\s/.test(before)) return null;
    const frag = upto.slice(at + 1);
    if (/\s/.test(frag)) return null;
    return frag;
  }

  const mentionMatches = useMemo(() => {
    if (mentionQuery == null) return [];
    const q = mentionQuery.toLowerCase();
    return MENTIONABLE.filter(
      (m) =>
        !q ||
        m.id.toLowerCase().includes(q) ||
        m.short.toLowerCase().includes(q) ||
        m.cn.includes(mentionQuery),
    );
  }, [mentionQuery]);

  function onInputChange(value: string, caret: number) {
    setInput(value);
    const m = detectMention(value, caret);
    setMentionQuery(m);
    setMentionIdx(0);
  }

  // insert "@<agent_id> " replacing the active "@query" fragment at the caret
  function applyMention(agentId: string) {
    const el = inputRef.current;
    const caret = el ? el.selectionStart : input.length;
    const upto = input.slice(0, caret);
    const at = upto.lastIndexOf("@");
    if (at === -1) return;
    const next = input.slice(0, at) + `@${agentId} ` + input.slice(caret);
    setInput(next);
    setMentionQuery(null);
    setMentionIdx(0);
    // restore focus + place caret after the inserted mention
    requestAnimationFrame(() => {
      const node = inputRef.current;
      if (node) {
        const pos = at + agentId.length + 2;
        node.focus();
        node.setSelectionRange(pos, pos);
      }
    });
  }

  const convos = useQuery({
    queryKey: ["agent-chat", "conversations", agentId],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/{agent_id}/conversations", {
        params: { path: { agent_id: agentId } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 6000,
  });

  const list = convos.data?.conversations ?? [];

  // Auto-pick the first conversation when none is selected.
  useEffect(() => {
    if (!activeId && list[0]) setActiveId(list[0].conversation_id);
    if (activeId && !list.some((c) => c.conversation_id === activeId)) {
      setActiveId(list[0]?.conversation_id ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list, activeId]);

  // Reset transcript when switching agent or conversation.
  useEffect(() => {
    setStreamMsgs(null);
    setStreamEndedAt(0);
    setStreamErr(null);
  }, [agentId, activeId]);

  const history = useQuery({
    queryKey: ["agent-chat", "messages", agentId, activeId],
    enabled: !!activeId,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/{agent_id}/messages", {
        params: { path: { agent_id: agentId }, query: { conversation_id: activeId as string } },
      });
      if (error) throw error;
      return data;
    },
  });

  // 已知问题：智能体代理对话不会自动刷新。
  //
  // 这一页原来**只**在本浏览器自己发完一条、SSE 流结束时失效一次。别的来源写进去
  // 的东西一律看不见：另一台机器、同一台的另一个标签页、后台跑着的 agent、
  // 别人替这个 agent 插的话。刻意不加 refetchInterval —— 整份 history 每 N 秒重取
  // 一次，隧道那头是实打实的流量，而绝大多数次取回来的和手上的一模一样。
  // 规则是「有更新才读」，游标由 WS 推、会话列表那条 6 s 轮询兜底。
  useTranscriptRefresh({
    conversationId: activeId,
    updatedAt: list.find((c) => c.conversation_id === activeId)?.updated_at ?? null,
    streaming,
    onRefresh: () => {
      void qc.invalidateQueries({
        queryKey: ["agent-chat", "messages", agentId, activeId],
      });
    },
  });

  // 取回来之后画哪一份 —— 「有了新消息还是不会自动显示，必须切到别的页面再
  // 回来」的落点。前两轮（#31 / #36）修的都是送达：通知发得出、这一页也接着，
  // 而屏幕不变，因为这里从前写的是 `streamMsgs ?? history.data?.messages`：
  // 本浏览器上一轮 SSE 的快照只在换会话时清空，于是它一直压在重取回来的历史
  // 前面。切页再回来「就好了」= 组件重挂、快照归 null —— 症状本身就是证据。
  const source = transcriptSource({
    hasSnapshot: streamMsgs !== null,
    streaming,
    streamEndedAt,
    historyUpdatedAt: history.dataUpdatedAt,
  });
  const shown: Msg[] = source === "history" ? (history.data?.messages ?? []) : (streamMsgs ?? []);

  // What this agent is currently blocked on (approval to grant, or a question
  // it asked). Polled rather than streamed: the chat SSE cannot emit while the
  // turn is blocked waiting for exactly this answer.
  const interrupts = useQuery({
    queryKey: ["agent-chat", "interrupts", agentId],
    refetchInterval: 3000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/{agent_id}/interrupts", {
        params: { path: { agent_id: agentId } },
      });
      if (error) throw error;
      return data;
    },
  });
  const pendingInterrupts = interrupts.data?.interrupts ?? [];

  const resolveInterrupt = useMutation({
    mutationFn: async (vars: { interrupt_id: string; args: ResolveArgs }) => {
      const { data, error } = await api.POST(
        "/api/agents/{agent_id}/interrupts/{interrupt_id}/resolve",
        {
          params: { path: { agent_id: agentId, interrupt_id: vars.interrupt_id } },
          body: {
            decision: vars.args.decision,
            edited_args: vars.args.editedArgs ?? null,
            comment: vars.args.comment ?? null,
            selected: vars.args.selected ?? null,
            custom_text: vars.args.customText ?? null,
          },
        },
      );
      if (error) throw error;
      return data;
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["agent-chat", "interrupts", agentId] });
    },
    onSuccess: (data) => {
      if (!data?.applied) {
        // Never silent (). The backend keeps the worker blocked on an
        // invalid answer precisely so it can be corrected — say what was wrong.
        setInterruptNote(data?.detail ?? "未生效：后端降级或该中断已处理");
      } else {
        setInterruptNote(null);
      }
    },
    onError: (e) => setInterruptNote(String((e as Error)?.message ?? e)),
  });

  const createConvo = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/agents/{agent_id}/conversations", {
        params: { path: { agent_id: agentId } },
        body: { title: null, kind: "private", experiment_id: null },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["agent-chat", "conversations", agentId] });
      const cid = data?.conversation?.conversation_id;
      if (cid) setActiveId(cid);
    },
  });

  const renameConvo = useMutation({
    mutationFn: async (vars: { conversation_id: string; title: string }) => {
      const { data, error } = await api.PATCH(
        "/api/agents/{agent_id}/conversations/{conversation_id}",
        {
          params: { path: { agent_id: agentId, conversation_id: vars.conversation_id } },
          body: { title: vars.title },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agent-chat", "conversations", agentId] }),
  });

  const deleteConvo = useMutation({
    mutationFn: async (conversation_id: string) => {
      const { data, error } = await api.DELETE(
        "/api/agents/{agent_id}/conversations/{conversation_id}",
        { params: { path: { agent_id: agentId, conversation_id } } },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (_d, cid) => {
      qc.invalidateQueries({ queryKey: ["agent-chat", "conversations", agentId] });
      if (activeId === cid) setActiveId(null);
    },
  });

  async function send() {
    const text = input.trim();
    if (!text || streaming || !activeId) return;
    setInput("");
    setMentionQuery(null);
    setStreamErr(null);
    setStreaming(true);
    setStreamEndedAt(0);
    // 接在**屏幕上那一份**后面，不是接在上一轮的快照后面：那两者在别人也写过
    // 这个会话之后就不是同一个东西了。
    // `t` 必须一起搬 —— 漏掉它 tsc 不报错（可选字段），症状只是「时间偶尔
    // 没有」。同一个形状 2026-08-12 在 ChatPage 里出现过两次，这是第三处；
    // 现在由 test/chatMsgFields.test.ts 的结构闸门盯着。
    const base: Msg[] = shown.map((m) => ({ role: m.role, content: m.content, t: m.t }));
    setStreamMsgs([
      ...base,
      { role: "user", content: escapeHtml(text), t: Date.now() / 1000 },
    ]);
    const ctrl = new AbortController();
    abortCtrl.current = ctrl;
    try {
      const resp = await fetch(`/api/agents/${agentId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: activeId, user_text: text }),
        signal: ctrl.signal,
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
      const end = await readSseStream(resp, {
        controller: ctrl,
        onFrame: (frame: { kind?: string; messages?: Msg[]; message?: string }) => {
          if (frame.kind === "snapshot" && Array.isArray(frame.messages)) {
            setStreamMsgs(
              frame.messages.map((m) => ({ role: m.role, content: m.content, t: m.t })),
            );
          } else if (frame.kind === "error") {
            setStreamErr(frame.message ?? "对话出错");
          }
        },
      });
      // EOF without the server's `done` frame (or 75 s of silence past the 15 s
      // keep-alive) means the link was cut, not that the turn finished — the old
      // loop could not tell the two apart and silently showed a truncated answer
      // as a complete one .
      const cause = sseEndMessage(end);
      if (sseBroke(end) && cause) {
        setStreamErr(`${cause}本轮回答仍在后端继续，已自动重新载入历史。`);
      }
    } catch (err) {
      if ((err as Error)?.name !== "AbortError")
        setStreamErr(`连接出错：${String((err as Error)?.message ?? err)}`);
    } finally {
      setStreaming(false);
      // 这一轮交还给服务端历史：从这一刻起，任何一次比它更新的重取都盖过快照。
      setStreamEndedAt(Date.now());
      abortCtrl.current = null;
      qc.invalidateQueries({ queryKey: ["agent-chat", "messages", agentId, activeId] });
      qc.invalidateQueries({ queryKey: ["agent-chat", "conversations", agentId] });
    }
  }

  async function stop() {
    try {
      await api.POST("/api/agents/{agent_id}/chat/abort", { params: { path: { agent_id: agentId } } });
    } catch {
      /* best-effort */
    }
    abortCtrl.current?.abort();
    setStreaming(false);
  }

  const degraded = convos.data?.degraded;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[260px_1fr]">
      {/* session list */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span
            className={
              "inline-flex items-center gap-2 rounded-mast-ctl border bg-mast-panel px-2.5 py-1 text-sm font-medium text-mast-text " +
              cls.border
            }
            style={{ borderLeftWidth: 3, borderLeftColor: hue }}
          >
            <Avatar id={agentId} size={20} />
            <span className={"inline-flex " + cls.text}>
              <AgentIcon id={agentId} size={13} />
            </span>
            {a.cn} 会话
          </span>
          <Button variant="primary" onClick={() => createConvo.mutate()}>
            ＋ 新建
          </Button>
        </div>
        {convos.isPending && <Spinner />}
        {convos.isError && <ErrorNote error={convos.error} />}
        {convos.data && degraded && <DegradedNote what="代理对话" />}
        {convos.data && !degraded && list.length === 0 && <EmptyNote label="暂无会话，点「新建」开始。" />}
        <div className="max-h-[460px] space-y-1 overflow-auto">
          {list.map((c) => (
            <div
              key={c.conversation_id}
              className={
                "group flex items-center gap-1 rounded-mast-ctl border px-2 py-1.5 text-sm " +
                (c.conversation_id === activeId
                  ? cls.border + " bg-mast-panel-2"
                  : "border-mast-border hover:bg-mast-panel-2/40")
              }
              style={
                c.conversation_id === activeId
                  ? { borderLeftWidth: 3, borderLeftColor: hue }
                  : undefined
              }
            >
              <button
                className="min-w-0 flex-1 truncate text-left"
                onClick={() => setActiveId(c.conversation_id)}
                title={c.last_message_preview}
              >
                {c.title || "新对话"}
              </button>
              <button
                className="text-mast-muted opacity-0 hover:text-mast-text group-hover:opacity-100"
                title="重命名"
                onClick={() => {
                  const t = window.prompt("重命名会话", c.title);
                  if (t != null && t.trim()) renameConvo.mutate({ conversation_id: c.conversation_id, title: t.trim() });
                }}
              >
                ✎
              </button>
              <button
                className="text-mast-danger opacity-0 group-hover:opacity-100"
                title="删除"
                onClick={() => setPendingDelete(c.conversation_id)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* transcript */}
      <div
        className={"flex min-h-[460px] flex-col rounded-mast-card border border-mast-border bg-mast-panel shadow-mast " + cls.border}
        style={{ borderLeftWidth: 3, borderLeftColor: hue }}
      >
        <div className="flex-1 space-y-3 overflow-auto p-4">
          {/* Anything this agent is blocked on, INLINE. A private chat that hit
              an approval — or that asked the operator a question — used to have
              nowhere to show it: the modal was hardcoded to instrument_control
              and the turn just stopped. */}
          {pendingInterrupts.map((it) => (
            <InterruptCard
              key={it.event_id}
              it={fromPollRow(it, agentId)}
              busy={resolveInterrupt.isPending && resolveInterrupt.variables?.interrupt_id === it.event_id}
              compact
              onResolve={(args) => resolveInterrupt.mutate({ interrupt_id: it.event_id, args })}
            />
          ))}
          {interruptNote && (
            <p className="rounded-md border border-mast-danger-border bg-mast-danger-bg px-3 py-2 text-xs text-mast-danger">
              {interruptNote}
            </p>
          )}
          {!activeId && <EmptyNote label="选择或新建一个会话以开始私聊。" />}
          {activeId && history.isPending && !streamMsgs && <Spinner />}
          {activeId && shown.length === 0 && !history.isPending && (
            <EmptyNote label="该会话暂无消息。" />
          )}
          {shown.map((m, i) => {
            const isUser = m.role === "user";
            const isAgentMsg = m.role === "assistant";
            // 取不到就整个不渲染（`fmtClock` 对无效值返回空串）——**绝不显示 1970**。
            const clock = fmtClock(m.t);
            return (
              <div key={i} className={"flex " + (isUser ? "justify-end" : "justify-start")}>
                <div
                  className={
                    "max-w-[85%] text-sm text-mast-text " +
                    (isUser
                      ? "rounded-mast-card rounded-br-[3px] border border-mast-accent/40 bg-mast-accent-soft px-3.5 py-2.5"
                      : "overflow-hidden rounded-mast-card border border-mast-border bg-mast-panel shadow-mast " +
                        (isAgentMsg ? "rounded-l-none " + cls.borderL : ""))
                  }
                >
                  {isUser && clock && (
                    <div className="mb-1 text-right font-mono text-[10px] tabular-nums text-mast-faint">
                      {clock}
                    </div>
                  )}
                  {!isUser && (
                    <div className="mb-1 flex items-center gap-2 border-b border-mast-border bg-mast-panel-2 px-3 py-1.5">
                      {isAgentMsg ? (
                        <>
                          <span
                            className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-mast-badge font-mono text-[9px] font-bold text-white"
                            style={{ background: hue }}
                          >
                            {a.short}
                          </span>
                          <span className={"inline-flex " + cls.text}>
                            <AgentIcon id={agentId} size={12} />
                          </span>
                          <span className="text-[11.5px] font-medium text-mast-text">{a.cn}</span>
                        </>
                      ) : (
                        <Badge tone="INFO">{m.role}</Badge>
                      )}
                    </div>
                  )}
                  <div
                    className={
                      (isUser ? "" : "px-3.5 py-2.5 ") +
                      "prose-sm whitespace-pre-wrap break-words " +
                      "[&_code]:font-mono [&_code]:tabular-nums [&_code]:text-mast-accent " +
                      "[&_details]:my-1.5 [&_details]:rounded-mast-ctl [&_details]:bg-mast-code-bg [&_details]:px-3 [&_details]:py-2 [&_details]:text-xs [&_details]:text-mast-muted [&_summary]:cursor-pointer [&_summary]:text-mast-dream"
                    }
                    dangerouslySetInnerHTML={{ __html: m.content }}
                  />
                </div>
              </div>
            );
          })}
          {streamErr && <ErrorNote error={streamErr} label="运行出错" />}
        </div>
        <div className="relative flex items-end gap-2 border-t border-mast-border p-3">
          {/* @-mention autocomplete popup (ITEM 4) */}
          {mentionQuery != null && mentionMatches.length > 0 && (
            <div className="absolute bottom-full left-3 z-20 mb-1 w-64 overflow-hidden rounded-mast-ctl border border-mast-border bg-mast-panel shadow-mast">
              <div className="border-b border-mast-border bg-mast-panel-2 px-2.5 py-1 text-[11px] text-mast-muted">
                @ 指向代理
              </div>
              {mentionMatches.map((m, i) => {
                const mc = agentClasses(m.id);
                return (
                  <button
                    key={m.id}
                    type="button"
                    onMouseDown={(e) => {
                      e.preventDefault();
                      applyMention(m.id);
                    }}
                    className={
                      "flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-sm " +
                      (i === mentionIdx
                        ? "bg-mast-panel-2 text-mast-text"
                        : "text-mast-muted hover:bg-mast-panel-2/50")
                    }
                    style={i === mentionIdx ? { borderLeft: `3px solid ${agentColorVar(m.id)}` } : { borderLeft: "3px solid transparent" }}
                  >
                    <Avatar id={m.id} size={16} />
                    <span className={"font-mono text-xs font-semibold " + mc.text}>{m.short}</span>
                    <span className="text-xs">{m.cn}</span>
                    <span className="ml-auto font-mono text-[10px] text-mast-faint">@{m.id}</span>
                  </button>
                );
              })}
            </div>
          )}
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => onInputChange(e.target.value, e.target.selectionStart)}
            onKeyUp={(e) => {
              const el = e.currentTarget;
              setMentionQuery(detectMention(el.value, el.selectionStart));
            }}
            onKeyDown={(e) => {
              // when the @-popup is open, arrows/enter/tab drive selection
              if (mentionQuery != null && mentionMatches.length > 0) {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setMentionIdx((n) => (n + 1) % mentionMatches.length);
                  return;
                }
                if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setMentionIdx((n) => (n - 1 + mentionMatches.length) % mentionMatches.length);
                  return;
                }
                if (e.key === "Enter" || e.key === "Tab") {
                  e.preventDefault();
                  applyMention(mentionMatches[mentionIdx]!.id);
                  return;
                }
                if (e.key === "Escape") {
                  e.preventDefault();
                  setMentionQuery(null);
                  return;
                }
              }
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={activeId ? `私聊 ${a.cn}…（@ 指向代理 · Enter 发送，Shift+Enter 换行）` : "请先选择会话"}
            disabled={!activeId || streaming}
            rows={2}
            className="flex-1 resize-none rounded-mast-ctl border border-mast-border-strong bg-mast-panel px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent disabled:opacity-50"
          />
          {streaming ? (
            <Button variant="danger" onClick={stop}>
              停止
            </Button>
          ) : (
            <Button variant="primary" onClick={send} disabled={!activeId || !input.trim()}>
              发送
            </Button>
          )}
        </div>
      </div>
      <ConfirmDialog
        open={!!pendingDelete}
        onClose={() => setPendingDelete(null)}
        title="删除会话"
        tone="danger"
        confirmLabel="删除"
        busy={deleteConvo.isPending}
        busyLabel="删除中…"
        onConfirm={() => {
          if (pendingDelete) deleteConvo.mutate(pendingDelete);
          setPendingDelete(null);
        }}
      >
        <p className="text-sm text-mast-text">删除该会话及其转录？</p>
        <p className="text-xs text-mast-muted">
          若它已被导出到实验文件夹，文件夹里的那份不受影响。
        </p>
      </ConfirmDialog>
    </div>
  );
}
