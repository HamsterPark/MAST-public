import { useEffect, useMemo, useRef, useState } from "react";
import { Badge, EmptyNote, ErrorNote } from "@/components/ui";
import { Button, useToast } from "@/components/controls";
import { useDraft } from "@/hooks/useDraft";
import { useStickToBottom } from "@/hooks/useStickToBottom";
import { useWsEvent } from "@/hooks/useWsEvents";
import {
  AGENTS,
  Avatar,
  agentLabel,
  SUP_ID,
  SUPERVISOR,
} from "./registry";
import { InterruptCard, type ResolveArgs } from "./InterruptCard";
import {
  useRunTaskStore,
  mentionTarget,
  type Entry,
  type KeyedEntry,
  type Pending,
} from "./runTaskStore";

// @-mention roster: SUP + 7 agents the operator can @-target. Typing "@" opens a
// popup; a selection inserts "@<agent_id> " at the caret. The @ is the SOLE
// driver of the interject target (no standalone dropdown).
const MENTIONABLE = [SUPERVISOR, ...AGENTS];

// ════════════════════════════════════════════════════════════════════════════
// RunTaskPanel — INTERACTIVE multi-agent conversation, now backed by a durable
// store + server transcript (no longer an ephemeral demo).
//
// The conversation state lives in `useRunTaskStore` (module scope), so switching
// tabs / pages no longer clears it and no longer aborts a running orchestrator
// stream. The durable group conversation_id is persisted; on mount the panel
// replays the server-side transcript (the Python core is the source of truth),
// so a tab switch or reload is non-destructive.
//
// A single persistent input at the BOTTOM: the FIRST message (or any message
// while idle) starts a task (POST /api/agents/run-task, SSE); while a run is
// streaming a message is injected live (POST /api/agents/{target}/interject).
// Inline handoffs + HITL approve/reject/edit + abort. Degrade-safe throughout.
// ════════════════════════════════════════════════════════════════════════════

function fmtTime(t?: number): string {
  if (t == null) return "";
  try {
    return new Date(t * 1000).toLocaleTimeString();
  } catch {
    return "";
  }
}

export function RunTaskPanel({
  conversationId,
}: {
  conversationId?: string | null;
}) {
  const { toast, node } = useToast();

  // ── durable conversation state (module-scoped store) ──────────────────────
  const entries = useRunTaskStore((s) => s.entries);
  const running = useRunTaskStore((s) => s.running);
  const sending = useRunTaskStore((s) => s.sending);
  const degradedNote = useRunTaskStore((s) => s.degradedNote);
  const streamErr = useRunTaskStore((s) => s.streamErr);
  const storeConvId = useRunTaskStore((s) => s.conversationId);
  const serverActive = useRunTaskStore((s) => s.serverActive);
  const step = useRunTaskStore((s) => s.step);
  const stepLimit = useRunTaskStore((s) => s.stepLimit);
  const send = useRunTaskStore((s) => s.send);
  const abortRun = useRunTaskStore((s) => s.abort);
  const paused = useRunTaskStore((s) => s.paused);
  const togglePause = useRunTaskStore((s) => s.togglePause);
  const resolveInterrupt = useRunTaskStore((s) => s.resolveInterrupt);
  const hydrate = useRunTaskStore((s) => s.hydrate);
  const newConversation = useRunTaskStore((s) => s.newConversation);
  const switchConversation = useRunTaskStore((s) => s.switchConversation);

  // ── the persistent bottom input (always available, like a chat) ───────────
  // Draft survives tab switches — sessionStorage-backed.
  const [input, setInput] = useDraft("runtask-input");

  // ── @-mention autocomplete ────────────────────────────────────────────────
  const [mentionQuery, setMentionQuery] = useState<string | null>(null);
  const [mentionIdx, setMentionIdx] = useState(0);

  // The edit-args / answer drafts live inside each InterruptCard now — they are
  // per-card state and hoisting them here meant two cards shared one draft.
  const [resolving, setResolving] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Resolved interject/steer target — derived PURELY from any "@<agent>" mention.
  const target = useMemo(() => mentionTarget(input) ?? SUP_ID, [input]);
  const mentioned = mentionTarget(input);

  // Seed the store from a prop conversationId (e.g. resuming a specific group
  // run), then replay its server transcript. With no prop, hydrate the persisted
  // conversation so a reload/tab-switch shows the thread instead of a blank pane.
  useEffect(() => {
    if (conversationId && conversationId !== storeConvId) {
      void switchConversation(conversationId);
    } else {
      void hydrate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  // ── another viewer wrote to THIS conversation → catch up () ────────
  //
  // Multiple remote windows could show out-of-sync conversations. The transcript was read on mount
  // and then only when the operator clicked 「刷新进展」, so a second machine
  // watching the same group chat simply never saw new messages.
  //
  // The bus event carries only a cursor (conversation_id + seq), never the text —
  // so this re-reads through the normal authenticated endpoint rather than
  // rendering whatever arrived on a broadcast socket.
  //
  // Guards, both of which matter:
  //   * ignore events for OTHER conversations — an operator with two group chats
  //     must not have one refresh because the other moved;
  //   * ignore while THIS tab is the one streaming (`running`) — the SSE stream
  //     is already delivering those entries live, and hydrating mid-stream would
  //     race the append. Our own writes are what generate these events.
  const wsSeen = useRef(0);
  useWsEvent("experiment", (event) => {
    const d = (event.data ?? {}) as { scope?: string; conversation_id?: string; seq?: number };
    if (d.scope !== "transcript") return;
    if (!storeConvId || d.conversation_id !== storeConvId) return;
    if (running) return;
    const seq = Number(d.seq ?? 0);
    if (seq <= wsSeen.current) return;   // coalesce a burst into one refresh
    wsSeen.current = seq;
    void hydrate(true);
  });

  // 新条目就跟到底部 —— **但只在用户本来就在底部时**。
  //
  // 原来是无条件 `el.scrollTop = el.scrollHeight`:用户往上翻记录,下一条就把他
  // 弹回去。聊天那边有同一个毛病,
  // 这里是同一个形状,一并修 —— 一个已知错法留着第二份实现,下次就是它出事。
  useStickToBottom(scrollRef, [entries, running]);

  // NOTE: deliberately NO unmount cleanup that aborts the stream — the store
  // drives the SSE loop in module scope so the run survives navigation.

  const detectMention = (value: string, caret: number): string | null => {
    const upto = value.slice(0, caret);
    const at = upto.lastIndexOf("@");
    if (at === -1) return null;
    const before = at === 0 ? "" : upto[at - 1];
    if (before && !/\s/.test(before)) return null;
    const frag = upto.slice(at + 1);
    if (/\s/.test(frag)) return null;
    return frag;
  };

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

  function setInputAndTarget(value: string, caret: number) {
    setInput(value);
    setMentionQuery(detectMention(value, caret));
    setMentionIdx(0);
  }

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
    requestAnimationFrame(() => {
      const node = inputRef.current;
      if (node) {
        const pos = at + agentId.length + 2;
        node.focus();
        node.setSelectionRange(pos, pos);
      }
    });
  }

  // ── the single send handler — START or INJECT depending on run state ──────
  async function onSend() {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    setMentionQuery(null);
    const res = await send(text);
    if (res.action === "interject") {
      if (res.ok) {
        toast("已插话，将于编排器下一步送达", "ok");
      } else {
        toast(`插话未生效（${res.detail ?? "编排器未在运行 / 后端降级"}）`, "err");
        setInput(res.text); // restore the steer so it isn't silently lost
      }
    } else if (res.action === "busy") {
      // Name buttons that actually exist. The old hint said 「刷新」(there is no
      // such button — it reads 「刷新进展」) and 「＋新群聊」(only rendered when
      // hasConversation && !running), so the operator was told to click things
      // they could not find, on the one screen where they were already stuck
      //. The abort button now stays put after a dropped
      // stream, so it is a real way out.
      toast("该群聊后端仍在运行——点「中止后端运行」结束它，或点「刷新进展」看它进行到哪了", "err");
      setInput(res.text);
    }
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  async function onAbort() {
    const { degraded } = await abortRun();
    if (degraded) toast("中止已转发（后端降级）", "err");
    else toast("已请求中止编排任务", "ok");
  }

  async function onTogglePause() {
    const res = await togglePause();
    if (!res.ok) {
      toast("暂停/继续请求失败（后端降级）", "err");
    } else if (res.paused) {
      toast("已请求暂停——编排器将在下一个智能体边界停下", "ok");
    } else {
      toast("已继续执行", "ok");
    }
  }

  // ── resolve a pending HITL interrupt (inline; stream resumes on resolve) ──
  async function resolve(it: Pending, args: ResolveArgs) {
    setResolving(it.interrupt_id);
    try {
      const res = await resolveInterrupt(it, args.decision, {
        editedArgs: args.editedArgs ?? null,
        selected: args.selected ?? null,
        customText: args.customText ?? null,
        comment: args.comment ?? null,
      });
      const label =
        args.decision === "approve" ? "批准"
          : args.decision === "reject" ? "拒绝"
          : args.decision === "edit" ? "编辑并批准"
          : args.decision === "answer" ? "回答" : `选择「${args.decision}」`;
      if (res.ok && res.applied) {
        toast(`已${label}中断 ${it.interrupt_id}`, "ok");
      } else if (res.status === "answer_invalid" || res.status === "route_not_allowed") {
        // The backend deliberately kept the worker blocked so this can be
        // corrected — say what was wrong and leave the card interactive.
        toast(`未提交：${res.detail ?? "回答无效"}，请修改后重试`, "err");
      } else {
        toast(`中断未生效：${res.detail ?? "后端降级或中断已处理"}`, "err");
      }
    } catch (e) {
      // Without this the rejection escapes the async function and the operator
      // gets NO feedback at all — the button reads as dead. Approve and reject
      // buttons could look clickable but silently do nothing. A failed approval must SAY it failed;
      // silence is the one outcome this UI may never produce.
      toast(`中断未送达：${e instanceof Error ? e.message : String(e)}——` +
            "网络或后端不可达，请重试", "err");
    } finally {
      setResolving(null);
    }
  }

  const pendingCount = entries.filter(
    (e) => e.kind === "interrupt" && !e.pending.resolved,
  ).length;
  const hasConversation = entries.length > 0;

  return (
    <div className="flex flex-col gap-3">
      {node}

      {/* status strip — run state + pending interrupts + new-conversation */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {running ? (
          paused ? (
            <Badge tone="WARN">已暂停 · 编排器停在智能体边界</Badge>
          ) : (
            <Badge tone="AUTO">编排器运行中</Badge>
          )
        ) : hasConversation ? (
          <Badge tone="INFO">空闲 · 发送消息开始新任务</Badge>
        ) : (
          <Badge tone="INFO">空闲</Badge>
        )}
        {/* step budget — always visible once a run reports its limit, so the
            operator can see how close the task is to the recursion ceiling. */}
        {stepLimit > 0 && (
          <Badge
            tone={
              step >= stepLimit * 0.9
                ? "DANGEROUS"
                : step >= stepLimit * 0.7
                  ? "WARN"
                  : "default"
            }
          >
            步数 {step}/{stepLimit}
          </Badge>
        )}
        {pendingCount > 0 && (
          <Badge tone="DANGEROUS">{pendingCount} 个待批准中断</Badge>
        )}
        {storeConvId && (
          <span className="font-mono text-[10px] text-mast-muted/70" title="本群聊已持久化，可跨标签页/刷新恢复">
            群聊 {storeConvId.slice(0, 8)}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          {hasConversation && !running && (
            <button
              onClick={() => newConversation()}
              className="rounded-md border border-mast-border px-2.5 py-1 text-xs font-medium text-mast-muted hover:text-mast-text"
            >
              ＋ 新群聊
            </button>
          )}
          {running && (
            <button
              onClick={onTogglePause}
              title={
                paused
                  ? "释放暂停，编排器继续执行"
                  : "在下一个智能体边界暂停编排器（不丢状态，可随时继续）"
              }
              className="rounded-md border border-mast-border px-2.5 py-1 text-xs font-medium text-mast-muted hover:text-mast-text"
            >
              {paused ? "▶ 继续" : "⏸ 暂停"}
            </button>
          )}
          {/* `running` is the CLIENT's own stream flag. After a reload / dropped
              SSE it is false while the BACKEND run — and the instrument — keep
              going. Gating 中止 on it meant the operator could not stop a live
              hardware task: the send box bounced them with "请先中止" while no
              中止 button existed anywhere on the page. Abort
              does not need a live stream — it POSTs and sets the abort event. */}
          {(running || serverActive) && (
            <button
              onClick={onAbort}
              className="rounded-md border border-mast-danger-border px-2.5 py-1 text-xs font-medium text-mast-danger hover:bg-mast-danger-bg"
            >
              中止
            </button>
          )}
        </div>
      </div>

      {degradedNote && (
        <div className="rounded-md border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-xs text-mast-warn">
          {degradedNote}
        </div>
      )}

      {/* reconnect note: after a reload the client can't re-attach the live SSE,
          but the server keeps persisting — surface that honestly. */}
      {serverActive && !running && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-mast-danger-border bg-mast-danger-bg/30 px-3 py-2 text-xs text-mast-muted">
          <span>
            该群聊仍在后端运行（仪器可能正在动作）——本页已断开实时流，显示的是已持久化的进展。
          </span>
          <button
            onClick={() => hydrate(true)}
            className="rounded border border-mast-border px-2 py-0.5 font-medium text-mast-muted hover:text-mast-text"
          >
            刷新进展
          </button>
          {/* The escape hatch that was missing. Without it the operator was told
              "请先中止" by the send box while no 中止 button was on screen. */}
          <button
            onClick={onAbort}
            className="rounded border border-mast-danger-border px-2 py-0.5 font-medium text-mast-danger hover:bg-mast-danger-bg"
          >
            中止后端运行
          </button>
        </div>
      )}

      {/* one continuous, scrolling multi-agent conversation */}
      <div
        ref={scrollRef}
        className="max-h-[560px] min-h-[280px] space-y-2 overflow-auto rounded-lg border border-mast-border p-3"
      >
        {!hasConversation && !running && (
          <EmptyNote label="在下方输入框开始与多智能体团队对话——第一条消息会下发任务，运行期间继续输入即可向编排器插话引导。本对话会持久化，切换标签页或刷新都不会丢失。" />
        )}
        {!hasConversation && running && (
          <p className="text-sm text-mast-muted">编排器启动中…</p>
        )}

        {entries.map((e, i) => (
          <EntryRow
            key={e._k}
            entry={e}
            prev={entries[i - 1]}
            resolving={resolving}
            onResolve={resolve}
          />
        ))}

        {/* live "thinking" affordance so the stream never looks frozen */}
        {running && (
          <div className="flex items-center gap-2 pl-1 text-xs text-mast-muted">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-mast-accent" />
            编排器处理中…
          </div>
        )}

        {streamErr && <ErrorNote error={streamErr} label="运行出错" />}
      </div>

      {/* persistent bottom input — ALWAYS available (start task OR live steer) */}
      <div className="rounded-lg border border-mast-border bg-mast-panel p-2">
        <div className="relative flex items-end gap-2">
          {/* @-mention autocomplete popup */}
          {mentionQuery != null && mentionMatches.length > 0 && (
            <div className="absolute bottom-full left-0 z-20 mb-1 w-64 overflow-hidden rounded-md border border-mast-border bg-mast-panel shadow-lg">
              <div className="border-b border-mast-border px-2.5 py-1 text-[11px] text-mast-muted">
                @ 指向代理（设为插话/引导目标）
              </div>
              {mentionMatches.map((m, i) => (
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
                      ? "bg-mast-accent/15 text-mast-text"
                      : "text-mast-muted hover:bg-mast-bg/60")
                  }
                >
                  <Avatar id={m.id} size={16} />
                  <span className="font-mono text-xs font-semibold">{m.short}</span>
                  <span className="text-xs">{m.cn}</span>
                  <span className="ml-auto font-mono text-[10px] text-mast-muted">
                    @{m.id}
                  </span>
                </button>
              ))}
            </div>
          )}
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInputAndTarget(e.target.value, e.target.selectionStart)}
            onKeyUp={(e) => {
              const el = e.currentTarget;
              setMentionQuery(detectMention(el.value, el.selectionStart));
            }}
            onKeyDown={(e) => {
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
                onSend();
              }
            }}
            placeholder={
              running
                ? "运行中：输入即可向编排器插话引导（@ 指向代理 · Enter 发送，Shift+Enter 换行）…"
                : "输入任务开始多智能体对话，例：在 Au(111) 上测 Kondo——先查文献给参数，再扫描成像做 STS，最后汇总（@ 指向代理 · Enter 发送）"
            }
            rows={2}
            className="flex-1 resize-none rounded-md border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent disabled:opacity-50"
          />
          {mentioned && (
            <span
              className="mb-2 flex shrink-0 items-center gap-1 self-end rounded-md border border-mast-accent/40 bg-mast-accent/10 px-2 py-1 font-mono text-[11px] text-mast-text"
              title={`插话/引导目标：${agentLabel(mentioned)}`}
            >
              → {agentLabel(mentioned)}
            </span>
          )}
          <Button variant="primary" onClick={onSend} disabled={!input.trim() || sending}>
            {sending ? "发送中…" : running ? "插话" : "发送"}
          </Button>
        </div>
        <p className="mt-1 px-1 text-[11px] text-mast-muted/80">
          {running
            ? `运行中 · 消息将作为用户插话送达「${agentLabel(target)}」${mentioned ? "" : "（默认全部·编排器）"}，编排器每个超步消费排队插话（不打断当前回合）。输入 @ 指向某代理。`
            : "空闲 · 发送即下发新任务（沿用同一群组线程上下文，持久化保存）。运行后可继续输入实时引导；输入 @ 指向某代理（默认全部·编排器）。"}
        </p>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// EntryRow — render one entry of the unified conversation.
// ════════════════════════════════════════════════════════════════════════════

function EntryRow({
  entry,
  prev,
  resolving,
  onResolve,
}: {
  entry: KeyedEntry;
  prev?: KeyedEntry;
  resolving: string | null;
  onResolve: (it: Pending, args: ResolveArgs) => void;
}) {
  if (entry.kind === "operator") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-md border border-mast-accent/50 bg-mast-accent/10 px-3 py-2 text-sm text-mast-text">
          <div className="mb-1 flex items-center gap-2">
            <Badge tone="AUTO">用户</Badge>
            <span className="text-xs text-mast-muted">指向 {agentLabel(entry.target)}</span>
            <span className="font-mono text-xs text-mast-muted">{fmtTime(entry.t)}</span>
          </div>
          <div className="whitespace-pre-wrap break-words">{entry.text}</div>
        </div>
      </div>
    );
  }

  if (entry.kind === "status") {
    return (
      <div className="flex items-center gap-2 text-xs text-mast-muted">
        <span className="font-mono">{fmtTime(entry.t)}</span>
        <Badge tone="INFO">状态</Badge>
        <span>{entry.text}</span>
        {entry.backend && <span className="font-mono opacity-70">· {entry.backend}</span>}
      </div>
    );
  }

  if (entry.kind === "compaction") {
    return <CompactionDivider entry={entry} />;
  }

  if (entry.kind === "done") {
    // Render finalText ONLY when it isn't already the preceding agent message.
    // The backend emits the final answer BOTH as an agent 'message' and as the
    // 'done' frame, which used to render the answer twice ("第二次刷屏",
    // 2026-07-06 feedback). Bare markers ("完成"/"已中止" from a replayed
    // transcript row) are the badge's job, so drop those too.
    const ft = (entry.finalText ?? "").trim();
    const dup =
      !ft ||
      ft === "完成" ||
      ft === "已中止" ||
      ft.startsWith("运行出错") ||
      (prev?.kind === "message" && prev.text.trim() === ft);
    // THREE outcomes, not two. Until 2026-07-28 the client read only `aborted`,
    // so a backend crash and an operator abort shared the 「已中止」 badge — and
    // a run whose branch died silently (no hand-back, langgraph raises nothing)
    // was pixel-identical to a completed one. `failed` + `stopReason` come
    // straight from the done frame / the persisted terminal marker.
    const failed = !!entry.failed;
    const tone = failed ? "DANGEROUS" : entry.aborted ? "WARN" : "INFO";
    const label = failed ? "运行未完成" : entry.aborted ? "已中止" : "任务完成";
    return (
      <div className="my-1 rounded-md border border-mast-border bg-mast-bg/40 px-3 py-2">
        <div className="flex items-center gap-2">
          <Badge tone={tone}>{label}</Badge>
          <span className="font-mono text-xs text-mast-muted">{fmtTime(entry.t)}</span>
        </div>
        {failed && entry.stopReason && (
          <p className="mt-1 whitespace-pre-wrap break-words text-sm text-mast-warn">
            {entry.stopReason}
          </p>
        )}
        {!dup && (
          <p className="mt-1 whitespace-pre-wrap break-words text-sm text-mast-text">
            {entry.finalText}
          </p>
        )}
      </div>
    );
  }

  if (entry.kind === "interrupt") {
    const it = entry.pending;
    return (
      <InterruptCard
        it={it}
        busy={resolving === it.interrupt_id}
        agentLabel={agentLabel}
        onResolve={(args) => onResolve(it, args)}
      />
    );
  }

  // kind === "message" — per-agent / tool message, grouped by agent
  const showHeader = !prev || prev.kind !== "message" || prev.agent !== entry.agent;
  const isTool = entry.role === "tool";
  const isHandoff = entry.text.startsWith("[HANDOFF");
  return (
    <div className="space-y-1">
      {showHeader && (
        <div className="mt-2 flex items-center gap-2">
          <Avatar id={entry.agent} size={18} />
          <span className="text-sm font-medium">{agentLabel(entry.agent)}</span>
          <span className="font-mono text-xs text-mast-muted">{fmtTime(entry.t)}</span>
        </div>
      )}
      <div
        className={
          "ml-6 rounded-md border px-3 py-2 text-sm " +
          (isTool
            ? "border-mast-border/60 bg-mast-bg/40 text-xs text-mast-muted"
            : "border-mast-border bg-mast-bg/60 text-mast-text")
        }
      >
        {isTool && <Badge tone="default">{isHandoff ? "移交" : "工具"}</Badge>}
        {/* The summary line. No longer monospace for tool rows: these are
            sentences now, not code fragments . */}
        <div className="whitespace-pre-wrap break-words">{entry.text}</div>
        <ToolDetail entry={entry} />
      </div>
    </div>
  );
}

// ── context-compaction divider ──────────────────────────
//
// Compaction was always running in 群聊, and
// nothing in the UI showed it: scroll far enough up a long run and the earlier turns
// are simply gone, in a view that otherwise reads as a faithful log. This is the
// same failure mode as a silently clipped tool return  — the record looks
// complete because nothing marks what was removed.
//
// So the divider is a RULE ACROSS THE TIMELINE, not a status line: it has to be
// impossible to scroll past without noticing, because everything above it is a
// summary rather than the conversation.
//
// It states only what the backend actually sent. `removed`/`kept`/`tokensBefore`
// are each optional; when they are missing the divider still appears and simply
// says less. A count is never guessed — a wrong "压掉了 40 条" would be worse
// than no number at all.
function CompactionDivider({
  entry,
}: {
  entry: Extract<Entry, { kind: "compaction" }>;
}) {
  const [open, setOpen] = useState(false);
  const facts: string[] = [];
  if (typeof entry.removed === "number") facts.push(`压缩 ${entry.removed} 条`);
  if (typeof entry.kept === "number") facts.push(`保留最近 ${entry.kept} 条原文`);
  if (typeof entry.tokensBefore === "number") {
    facts.push(
      `触发时约 ${entry.tokensBefore.toLocaleString()} tokens` +
        (typeof entry.triggerTokens === "number"
          ? ` / 阈值 ${entry.triggerTokens.toLocaleString()}`
          : ""),
    );
  }
  return (
    <div className="my-3">
      <div className="flex items-center gap-2">
        <span className="h-px flex-1 bg-mast-warn-border" />
        <Badge tone="WARN">上下文压缩</Badge>
        <span className="font-mono text-xs text-mast-muted">{fmtTime(entry.t)}</span>
        {entry.agent && (
          <span className="text-xs text-mast-muted">@ {agentLabel(entry.agent)}</span>
        )}
        <span className="h-px flex-1 bg-mast-warn-border" />
      </div>
      <div className="mt-1 rounded-md border border-mast-warn-border bg-mast-warn-bg px-3 py-2">
        <p className="text-xs text-mast-text">{entry.text}</p>
        <p className="mt-1 text-[11px] text-mast-muted">
          此分隔线以上的较早对话已被系统替换为摘要，不再是原文。
          {facts.length > 0 && ` （${facts.join(" · ")}）`}
          {typeof entry.tokensBefore === "number" && " token 数为估算值。"}
        </p>
        {entry.summary ? (
          <div className="mt-1">
            <button
              type="button"
              onClick={() => setOpen((s) => !s)}
              className="font-mono text-[11px] text-mast-muted hover:text-mast-text"
            >
              {open ? "▾" : "▸"} 摘要正文
            </button>
            {open && (
              <pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded border border-mast-border/60 bg-mast-bg/60 p-2 font-mono text-[11px] leading-relaxed">
                {entry.summary}
              </pre>
            )}
          </div>
        ) : (
          // Not decoration: without this line an operator would read the bare
          // divider as "the summary is somewhere else", when in fact this run
          // has no copy of it to show.
          <p className="mt-1 text-[11px] text-mast-muted/80">（本条记录未附摘要正文）</p>
        )}
      </div>
    </div>
  );
}

// ── collapsed detail for a tool row ─────────────────────
//
// 「参数默认折叠」. The summary above is deliberately lossy, so everything it
// dropped has to be ONE CLICK away — hiding it outright would just be a
// different way of losing the run's record. Nothing renders when there is
// nothing to show (a plain-prose tool return, or a transcript row written
// before the sidecar existed).
function ToolDetail({ entry }: { entry: Entry }) {
  const [open, setOpen] = useState(false);
  if (entry.kind !== "message") return null;
  const args = entry.args ?? "";
  const detail = entry.detail ?? "";
  const body = args || detail;
  if (!body) return null;
  const label = args ? "参数" : "完整返回";
  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setOpen((s) => !s)}
        className="font-mono text-[11px] text-mast-muted hover:text-mast-text"
      >
        {open ? "▾" : "▸"} {label}
        {entry.tool ? ` · ${entry.tool}` : ""}
        {entry.argsClipped ? " · 已截断" : ""}
      </button>
      {open && (
        <>
          {entry.argsClipped && (
            <p className="mt-1 text-[11px] text-mast-warn">
              参数过长，下面显示的是**开头部分**，不是完整内容。
            </p>
          )}
          <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded border border-mast-border/60 bg-mast-bg/60 p-2 font-mono text-[11px] leading-relaxed">
            {body}
          </pre>
        </>
      )}
    </div>
  );
}

export type { Entry };
