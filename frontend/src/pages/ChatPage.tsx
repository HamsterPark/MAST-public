import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Section, Card, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { SubTabs } from "@/components/controls";
import { ConfirmDialog } from "@/components/scope/ConfirmDialog";
import { useCurrentScope } from "@/api/scope";
import { type ChatMsg } from "@/components/chat/ChatBubbles";
// 气泡本身仍然由 ChatBubbles 画；NarrationLane 只是在气泡之间插旁白卡片，
// 关掉旁白时它原样把 ChatBubbles 转发出去（不是第二份渲染实现）。
import { NarrationLane } from "@/components/chat/NarrationLane";
import { DataStrip } from "@/components/chat/DataStrip";
import { VisionRibbon } from "@/components/chat/VisionRibbon";
import { QuickPrompts } from "@/components/chat/QuickPrompts";
import { useVoiceOutput } from "@/components/chat/VoiceControls";
import { VoiceDock } from "@/components/chat/VoiceDock";
import {
  PendingInterrupts,
  useInterrupts,
  type PendingInterruptsHandle,
} from "@/components/chat/PendingInterrupts";
import { useUiStore } from "@/store";
import { useDraft } from "@/hooks/useDraft";
import { useTranscriptRefresh } from "@/hooks/useTranscriptRefresh";
import { transcriptSource } from "@/lib/transcriptRefresh";
import { useStickyTab } from "@/hooks/useStickyTab";
import { readSseStream, sseBroke, sseEndMessage } from "@/lib/sse";
import { ScanMapPanel } from "@/components/vision/ScanMapPanel";
import { CoarseMapPanel } from "@/components/vision/CoarseMapPanel";
import { useCoarseMap } from "@/hooks/useCoarseMap";
import { PulseRibbon } from "@/components/vision/PulseRibbon";
import { VisionBufferPanel } from "@/components/vision/VisionBufferPanel";
import { RecentFrameThumb } from "@/components/vision/RecentFrameThumb";
import { useRunTaskStore } from "@/components/agents/runTaskStore";
import { PendingRecommendations } from "@/components/skills/PendingRecommendations";

// ════════════════════════════════════════════════════════════════════════
//  Chat page — FULL parity with the old Gradio chat surface.
//
//  Kept (already worked): SSE private chat with the IC agent + session CRUD.
//  Restored to parity (old chat.py / build_data_strip / feedback panel / etc.):
//    · LATEST DATA 缩略图条 (最近 .sxm 扫描 + open data browser)  → DataStrip
//    · 对话选择器 (下拉 + ➕新对话 + 重命名 + ✎ + 🗑, 非左侧列表)  → conv bar
//    · 快速提示按钮行                                            → QuickPrompts
//    · 全双工语音 (流式 ASR/TTS + 三模式 + 打断, /ws/voice)      → VoiceDock
//    · 语音输出朗读 (/api/voice/synthesize + autoplay, 🔊 一次性) → useVoiceOutput
//    · 对话反馈悬浮窗 (可拖拽, 评分+评论→/api/feedback)           → FeedbackFloat
//    · 视觉脉冲 ribbon (/api/vision/pulse)                       → VisionRibbon
//    · 人工介入内联面板 (/api/agents/{id}/interrupts)          → PendingInterrupts
//
//  Endpoints consumed (all exist):
//    GET  /api/agents/{id}/conversations | /messages | /interrupts
//    POST /api/agents/{id}/conversations | PATCH/DELETE | /chat (SSE) | /chat/abort
//    GET  /api/chat/quick-prompts | /api/hardware/live-readings | /api/vision/pulse
//    GET  /api/config/models | /api/agents/models
//    POST /api/voice/transcribe | /api/voice/synthesize | /api/feedback
//
//  Restored (parity with old chat_subtabs, git 8bef1a1 app.py ~L1185/1326):
//    The Chat surface lives under in-page SubTabs 对话 | Vision Buffer.
//      · 对话         = the chat surface (ChatSurface), kept exactly as-is.
//      · Vision Buffer = scan map + vision pulse + recent frames + buffer events
//                        (ScanMapCanvas /api/scan-map · PulseRibbon
//                         /api/vision/pulse · recent /api/vision/recent ·
//                         VisionBufferPanel /api/vision/buffer).
//    The 对话 sub-tab's Vision-Pulse ribbon "打开 Vision Buffer →" switches to
//    the Vision Buffer sub-tab (no route change, no freeze).
// ════════════════════════════════════════════════════════════════════════

const AGENT_ID = "instrument_control";

type Conversation = {
  conversation_id: string;
  title: string;
  last_message_preview: string;
  updated_at?: string | null;
};

function useConversations() {
  return useQuery({
    queryKey: ["chat", "conversations", AGENT_ID],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/{agent_id}/conversations", {
        params: { path: { agent_id: AGENT_ID } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });
}

function useMessages(conversationId: string | null) {
  return useQuery({
    queryKey: ["chat", "messages", AGENT_ID, conversationId],
    enabled: !!conversationId,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/{agent_id}/messages", {
        params: {
          path: { agent_id: AGENT_ID },
          query: { conversation_id: conversationId as string },
        },
      });
      if (error) throw error;
      return data;
    },
  });
}

// ────────────────────────────────────────────────────────────────────────
//  IC group-activity strip — real-time mirror of what instrument_control is
//  doing in a RUNNING 群聊 (multi-agent orchestrator run). The 仪器 chat is the
//  IC PRIVATE chat, a separate stream from the group run, so without this strip
//  the operator watching the instrument chat sees nothing of IC's team-run work.
//  Subscribes to the module-scoped run-task store (live, no poll). Renders
//  nothing when there's no group activity — never occupies space / never freezes.
// ────────────────────────────────────────────────────────────────────────
function IcGroupActivityStrip() {
  // Subscribe to the STABLE store-owned `entries` ref, derive in useMemo. A
  // `useShallow` selector that `.map`s into new object literals returns a fresh
  // array every call → unstable getSnapshot → React #185 infinite render.
  const entries = useRunTaskStore((s) => s.entries);
  const running = useRunTaskStore((s) => s.running);
  const ic = useMemo(
    () =>
      entries
        .filter((e) => e.kind === "message" && e.agent === AGENT_ID)
        .map((e) => (e.kind === "message" ? { role: e.role as string, text: e.text } : null))
        .filter((x): x is { role: string; text: string } => x != null),
    [entries],
  );
  if (ic.length === 0) return null;
  const recent = ic.slice(-6);
  return (
    <div className="rounded-lg border border-mast-accent/30 bg-mast-accent/5 px-3 py-2">
      <div className="mb-1 flex items-center gap-2 text-xs font-medium text-mast-text">
        <span>仪器控制 · 群聊中的实时活动</span>
        {running && (
          <span className="inline-flex items-center gap-1 text-mast-accent">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-mast-accent" />
            进行中
          </span>
        )}
        <span className="ml-auto font-mono text-[10px] text-mast-muted">{ic.length} 条</span>
      </div>
      <div className="max-h-28 space-y-1 overflow-auto">
        {recent.map((e, i) => (
          <div key={i} className="flex items-start gap-2 text-xs">
            {e.role === "tool" ? (
              <span className="shrink-0 rounded bg-mast-bg px-1 font-mono text-mast-muted">工具</span>
            ) : (
              <span className="shrink-0 rounded bg-mast-accent/10 px-1 text-mast-accent">发言</span>
            )}
            <span
              className={
                "min-w-0 flex-1 truncate " +
                (e.role === "tool" ? "font-mono text-mast-muted" : "text-mast-text")
              }
              title={e.text}
            >
              {e.text}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
//  Page wrapper — in-page SubTabs 对话 | Vision Buffer (old chat_subtabs).
// ────────────────────────────────────────────────────────────────────────
type ChatSubTab = "chat" | "vision_buffer";

// Remember the last-selected sub-tab across navigation. The
// route element remounts every time the operator returns to 仪器 Chat (AppLayout
// renders it through <Outlet/>), so a plain useState reset the tab to 对话 each
// time.
//
// 这里原本是本仓**唯一**一份手写的 localStorage 读写,而另外六个大栏目一份都没有
// —— 抱怨的就是其中一个(代理对话)。抽成 useStickyTab 之后改用共用的那
// 一份:留着这份手写的话,读时校验那一条(存进去的 id 会随子页改名/合并而过期)
// 就变成两个人各记一次,而漏掉它的症状是一片空白,看起来像「这一页坏了」。
const CHAT_SUB_TABS: { id: ChatSubTab; label: string }[] = [
  { id: "chat", label: "对话" },
  { id: "vision_buffer", label: "Vision Buffer" },
];

export default function ChatPage() {
  const [tab, setTab] = useStickyTab<ChatSubTab>(
    "chat", CHAT_SUB_TABS.map((t) => t.id), "chat");
  return (
    <div>
      <SubTabs<ChatSubTab> value={tab} onChange={setTab} tabs={CHAT_SUB_TABS} />
      {/* Keep both mounted — never unmount the chat surface (live SSE stream
          + draggable feedback widget must survive a sub-tab switch). Hide the
          inactive one instead. */}
      <div className={tab === "chat" ? undefined : "hidden"}>
        <ChatSurface onOpenBuffer={() => setTab("vision_buffer")} />
      </div>
      <div className={tab === "vision_buffer" ? undefined : "hidden"}>
        <VisionBufferSurface />
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
//  对话 — the chat surface (unchanged behaviour).
// ────────────────────────────────────────────────────────────────────────
function ChatSurface({ onOpenBuffer }: { onOpenBuffer: () => void }) {
  const qc = useQueryClient();
  const convos = useConversations();
  const [activeId, setActiveId] = useState<string | null>(null);
  // 原生 confirm() 会阻塞主线程，且失败时没有任何反馈。改用统一的
  // ConfirmDialog（失败时弹窗不关闭 + 就地红字）。
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Auto-select the first conversation once the list loads.
  useEffect(() => {
    const list = convos.data?.conversations ?? [];
    if (!activeId && list[0]) setActiveId(list[0].conversation_id);
  }, [convos.data, activeId]);

  // Publish the active conversation to the UI store so the global 对话反馈 widget
  // (mounted once in AppLayout → visible on every tab) attaches feedback to it.
  // Clear on unmount so feedback given from other tabs isn't tied to a stale chat.
  const setActiveConversation = useUiStore((s) => s.setActiveConversation);
  useEffect(() => {
    setActiveConversation(activeId);
  }, [activeId, setActiveConversation]);
  useEffect(() => () => setActiveConversation(null), [setActiveConversation]);

  const history = useMessages(activeId);

  // Live message buffer driven by the SSE stream. While streaming this overrides
  // the fetched history; otherwise the fetched history is the source of truth.
  const [streamMsgs, setStreamMsgs] = useState<ChatMsg[] | null>(null);
  // ……而「otherwise」以前根本没实现：快照只在换会话时清空，于是它一直压在
  // 重取回来的历史前面。这个戳让判据从「谁先有值」变成「谁更新」。
  const [streamEndedAt, setStreamEndedAt] = useState(0);
  const [streaming, setStreaming] = useState(false);
  const [streamErr, setStreamErr] = useState<string | null>(null);
  const abortCtrl = useRef<AbortController | null>(null);

  // 画哪一份 —— 判据在 lib/transcriptRefresh.ts，与「代理对话」共用一条。
  // 定在这里（`send` 之前）不是排版偏好：发送时那句乐观追加要接在**屏幕上
  // 那一份**后面，而不是接在上一轮的快照后面。
  const shownMessages: ChatMsg[] =
    transcriptSource({
      hasSnapshot: streamMsgs !== null,
      streaming,
      streamEndedAt,
      historyUpdatedAt: history.dataUpdatedAt,
    }) === "history"
      ? (history.data?.messages ?? [])
      : (streamMsgs ?? []);

  // Draft survives tab switches — sessionStorage-backed.
  const [input, setInput] = useDraft("chat-input");
  // 「显示旁白」——长任务跑的时候持续解说（我们要打一发脉冲了/扫到 50% 了）。
  // 存 sessionStorage（和 useDraft 同一套）而不是设置项：settings_store 的
  // KNOWN_KEYS 是**双边动作**，加一个可编辑键必须同时改两处，本仓已经踩过四次。
  // 等这个开关稳定了再谈搬进设置（设计文档阶段 5）。
  const [showNarration, setShowNarration] = useDraft("chat-narration");
  const narrationOn = showNarration !== "off";
  const [voiceMsg, setVoiceMsg] = useState<string | null>(null);

  // HITL interrupts — poll continuously, badge the toolbar button, render the
  // waiting items INLINE (they used to take over the screen; see #38 below).
  const interrupts = useInterrupts(AGENT_ID);
  const hitlRef = useRef<PendingInterruptsHandle | null>(null);
  const interruptCount = interrupts.data?.count ?? 0;
  // The button used to always read 「人工审批」. An agent asking the operator to
  // DECIDE something is not an approval, and labelling it as one tells them to
  // expect a yes/no about an action the agent already chose.
  const interruptLabel = (interrupts.data?.interrupts ?? []).some(
    (i) => i.kind === "ask_user",
  )
    ? "智能体在问你"
    : "人工审批";

  // 已知问题：人工介入(HITL)的时候看不到后面的界面显示了。
  //
  // 这里原本有一个「一有中断就自动弹全屏 Modal」的 effect。那个 Modal 是
  // `fixed inset-0 bg-black/60 backdrop-blur-sm`，于是**恰恰在要判断该不该批准
  // 的那一刻**，偏压、电流、Z、扫描进度、告警全被审批框自己挡住了 —— 而做这个
  // 判断靠的正是那些读数。
  //
  // 自动弹的原始理由（写在这里，因为它没有失效，只是不再需要靠遮挡来实现）：
  // 发起中断的那一轮是**阻塞**的，把它藏在角标后面就是「对话安静十五分钟，然后
  // 报一个没人看见的超时」。所以「不可能错过」保留，只去掉「盖住屏幕」——
  // 内联卡片在文档流里，它把内容推开而不是盖住，两个性质本来就不必绑在一起。
  // 收起之后仍留一行「N 项等待处理」，见 PendingInterrupts。

  // Voice output (TTS) for the latest assistant reply.
  const voiceOut = useVoiceOutput((m) => setVoiceMsg(m));

  // ── conversation mutations ───────────────────────────────────────────
  const createConvo = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/agents/{agent_id}/conversations", {
        params: { path: { agent_id: AGENT_ID } },
        body: { kind: "private", title: null, experiment_id: null },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["chat", "conversations", AGENT_ID] });
      const cid = data?.conversation?.conversation_id;
      if (cid) {
        setActiveId(cid);
        setStreamMsgs(null);
        setStreamEndedAt(0);
      }
    },
  });

  const renameConvo = useMutation({
    mutationFn: async (vars: { conversation_id: string; title: string }) => {
      const { data, error } = await api.PATCH(
        "/api/agents/{agent_id}/conversations/{conversation_id}",
        {
          params: { path: { agent_id: AGENT_ID, conversation_id: vars.conversation_id } },
          body: { title: vars.title },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["chat", "conversations", AGENT_ID] }),
  });

  const deleteConvo = useMutation({
    mutationFn: async (conversation_id: string) => {
      const { data, error } = await api.DELETE(
        "/api/agents/{agent_id}/conversations/{conversation_id}",
        { params: { path: { agent_id: AGENT_ID, conversation_id } } },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (_data, conversation_id) => {
      qc.invalidateQueries({ queryKey: ["chat", "conversations", AGENT_ID] });
      if (activeId === conversation_id) {
        setActiveId(null);
        setStreamMsgs(null);
        setStreamEndedAt(0);
      }
    },
  });

  // ── send (SSE stream) ────────────────────────────────────────────────
  async function send(textArg?: string) {
    const text = (textArg ?? input).trim();
    if (!text || streaming || !activeId) return;

    setInput("");
    setStreamErr(null);
    setStreaming(true);
    setStreamEndedAt(0);

    // yield-first: show the user's message immediately on top of what is ON
    // SCREEN — which stops being the last snapshot as soon as anyone else has
    // written to this conversation.
    // ⚠️ `t`（消息发生时刻）**必须一起搬过来**。这里原本只挑 role/content ——
    // 而 `ChatMsg.t` 是可选字段，所以漏搬它 **TypeScript 一声不吭**，症状只是
    // 气泡上的时间偶尔缺失——生产方接上了、消费方却把它丢了，这类缺陷不是第一次出现。
    const base: ChatMsg[] = shownMessages.map((m) => ({
      role: m.role,
      content: m.content,
      t: m.t,
    }));
    // 乐观追加的这条给一个本地时刻：它确实**就是现在**发生的，而且下一帧快照
    // 会用服务端的戳把它换掉。不给的话，用户自己刚发的那句在回包之前没有时间。
    setStreamMsgs([
      ...base,
      { role: "user", content: escapeHtml(text), t: Date.now() / 1000 },
    ]);

    const ctrl = new AbortController();
    abortCtrl.current = ctrl;

    try {
      const resp = await fetch(`/api/agents/${AGENT_ID}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: activeId, user_text: text }),
        signal: ctrl.signal,
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const end = await readSseStream(resp, {
        controller: ctrl,
        onFrame: (frame: { kind?: string; messages?: ChatMsg[]; message?: string }) => {
          if (frame.kind === "snapshot" && Array.isArray(frame.messages)) {
            // REPLACE the whole list on each snapshot — the backend yields the
            // full rendered history each frame.
            // `t` 一并搬 —— 这条是**长任务跑的时候用户实际盯着的那一路**，
            // 而旁白正是按 `t` 排的。这里丢掉它，气泡和旁白就不在同一根时间轴上，
            // 而顺序看着乱正是这个缺陷的症状。
            setStreamMsgs(
              frame.messages.map((m) => ({ role: m.role, content: m.content, t: m.t })),
            );
          } else if (frame.kind === "error") {
            setStreamErr(frame.message ?? "对话出错");
          }
        },
      });
      // A body that ends without the server's `done` frame — or 75 s of silence
      // through a 15 s keep-alive — is a cut link, not a finished turn. The old
      // `if (done) break` could not tell them apart, so a network change left
      // this page looking like the model had simply stopped talking .
      // The turn itself keeps running server-side and lands in the checkpointer,
      // which the invalidate below re-reads.
      const cause = sseEndMessage(end);
      if (sseBroke(end) && cause) {
        setStreamErr(`${cause}本轮回答仍在后端继续，已自动重新载入历史；若还没出现，稍后再点一次该会话即可。`);
      }
    } catch (err) {
      if ((err as Error)?.name !== "AbortError") {
        setStreamErr(`连接出错：${String((err as Error)?.message ?? err)}`);
      }
    } finally {
      setStreaming(false);
      // 这一轮交还给服务端历史。在此之前这句「re-sync」只做到了一半：重取确实
      // 发生了，而画出来的仍是快照 —— 上面那句「已自动重新载入历史」曾经是假话。
      setStreamEndedAt(Date.now());
      abortCtrl.current = null;
      // Re-sync the canonical rendered history from the checkpointer.
      qc.invalidateQueries({ queryKey: ["chat", "messages", AGENT_ID, activeId] });
      qc.invalidateQueries({ queryKey: ["chat", "conversations", AGENT_ID] });
    }
  }

  async function stop() {
    // Signal the backend to abort the active turn, then drop the local reader.
    try {
      await api.POST("/api/agents/{agent_id}/chat/abort", {
        params: { path: { agent_id: AGENT_ID } },
      });
    } catch {
      /* best-effort */
    }
    abortCtrl.current?.abort();
    setStreaming(false);
  }

  const conversations: Conversation[] = (convos.data?.conversations ?? []) as Conversation[];

  // The latest assistant reply (plain-ish text) for the 朗读 (TTS) action.
  const latestAssistant = [...shownMessages].reverse().find((m) => m.role === "assistant");
  function speakLatest() {
    if (!latestAssistant) return;
    // strip HTML tags from the rendered content before sending to TTS.
    const txt = latestAssistant.content.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    voiceOut.speak(txt);
  }

  const activeConvo = conversations.find((c) => c.conversation_id === activeId) ?? null;
  const [renameDraft, setRenameDraft] = useState("");

  // 与「代理对话」同一个缺陷（报的是那一页，这一页形状一样）：
  // 消息历史只在本浏览器自己发完一条之后失效一次。这里更容易被忽略，因为主聊天
  // 的用户通常**就是**发消息的那个人，所以看起来一直是刷新的 —— 直到他在
  // 另一台机器上打开、或者一个后台跑的回合替他把话说完。
  useTranscriptRefresh({
    conversationId: activeId,
    updatedAt: activeConvo?.updated_at ?? null,
    streaming,
    onRefresh: () => {
      void qc.invalidateQueries({
        queryKey: ["chat", "messages", AGENT_ID, activeId],
      });
    },
  });

  return (
    <Section
      title="对话"
      actions={
        interruptCount > 0 ? (
          <button
            // 不再打开弹窗 —— 滚到那张内联卡片并展开它。转录很长的时候
            // 这个角标是唯一的锚点，但它的作用是**带你过去**，不是盖住你在看的东西。
            onClick={() => hitlRef.current?.reveal()}
            className="rounded border border-mast-warn-border bg-mast-warn-bg px-3 py-1 text-sm text-mast-warn"
          >
            {interruptLabel}
            <span className="ml-1.5 rounded bg-mast-warn-border px-1.5 text-xs">{interruptCount}</span>
          </button>
        ) : undefined
      }
    >
      {/* ── Lab Console strips above the transcript (old order: 数据条 → 视觉脉冲) ── */}
      <div className="mb-3 flex flex-col gap-2">
        <DataStrip />
        <VisionRibbon onOpenBuffer={onOpenBuffer} />
        {/* IC's live activity in a running 群聊 (multi-agent run) — this private
            chat and the group run are separate streams, so without this the
            operator couldn't see what IC was doing in the team run. Real-time via
            the run-task store; hidden entirely when there's no group activity. */}
        <IcGroupActivityStrip />
      </div>

      {/* 等待人工介入的事项 —— 就放在数据条**下面**，理由是要判断该不该批准，
          靠的正是数据条上那些读数（偏压 / 电流 / Z / 扫描状态）。从前那个全屏
          弹窗恰好把它们盖住了。没有待处理事项时整块不渲染。 */}
      <div className="mb-3">
        <PendingInterrupts ref={hitlRef} data={interrupts.data} agentId={AGENT_ID} />
      </div>

      {/* 归属徽章：这条对话记在实验级还是样品级。
          没选样品时开的对话归实验级（规划讨论），选了样品之后开的归样品级 ——
          这个分级在后端自动成立（sample_id 为 NULL 即实验级），这里只是把它
          显示出来，让用户知道这段讨论以后会出现在文件夹的哪个位置。 */}
      <ConversationScopeBadge
        convo={activeConvo as ConvScope | null}
      />

      {/* ── conversation picker bar (old mast-conv-bar: dropdown + 新对话 +
            重命名 + rename + delete) — a FLAT row, never nested tabs ── */}
      <div className="mb-3 flex items-center gap-2">
        <select
          value={activeId ?? ""}
          onChange={(e) => {
            setActiveId(e.target.value || null);
            setStreamMsgs(null);
            setStreamEndedAt(0);
            setStreamErr(null);
          }}
          className="min-w-0 flex-1 rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm text-mast-text outline-none focus:border-mast-accent"
        >
          {conversations.length === 0 && <option value="">新对话</option>}
          {conversations.map((c) => (
            <option key={c.conversation_id} value={c.conversation_id}>
              {c.title || "新对话"}
            </option>
          ))}
        </select>
        <button
          onClick={() => createConvo.mutate()}
          disabled={createConvo.isPending}
          className="shrink-0 rounded border border-mast-border bg-mast-panel px-3 py-1.5 text-sm hover:border-mast-accent disabled:opacity-50"
        >
          ➕ 新对话
        </button>
        <input
          value={renameDraft}
          onChange={(e) => setRenameDraft(e.target.value)}
          placeholder="重命名…"
          className="w-40 shrink-0 rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm outline-none focus:border-mast-accent"
        />
        <button
          onClick={() => {
            if (activeId && renameDraft.trim()) {
              renameConvo.mutate({ conversation_id: activeId, title: renameDraft.trim() });
              setRenameDraft("");
            }
          }}
          disabled={!activeId || !renameDraft.trim()}
          title="重命名当前对话"
          className="shrink-0 rounded border border-mast-border bg-mast-panel px-3 py-1.5 text-sm hover:border-mast-accent disabled:opacity-40"
        >
          ✎
        </button>
        <button
          onClick={() => setConfirmDelete(true)}
          disabled={!activeId}
          title="删除当前对话"
          className="shrink-0 rounded border border-mast-danger-border bg-mast-danger-bg px-3 py-1.5 text-sm text-mast-danger hover:bg-mast-danger-border disabled:opacity-40"
        >
          🗑
        </button>
        <ConfirmDialog
          open={confirmDelete}
          onClose={() => setConfirmDelete(false)}
          title="删除对话"
          tone="danger"
          confirmLabel="删除"
          busy={deleteConvo.isPending}
          busyLabel="删除中…"
          onConfirm={() => {
            if (activeId) deleteConvo.mutate(activeId);
            setConfirmDelete(false);
          }}
        >
          <p className="text-sm text-mast-text">
            删除会话「{activeConvo?.title || "新对话"}」？
          </p>
          <p className="text-xs text-mast-muted">
            这条对话的转录会一并删除。若它已被导出到实验文件夹，文件夹里的那份不受影响。
          </p>
        </ConfirmDialog>
      </div>

      {/* ── chatbot transcript ──────────────────────────────────────── */}
      <div
        className="flex flex-col overflow-hidden rounded-lg border border-mast-border bg-mast-panel"
        style={{ height: "calc(100vh - 420px)" }}
      >
        <div className="flex items-center justify-between border-b border-mast-border px-3 py-1.5 text-xs text-mast-muted">
          <span>MAST Assistant</span>
          <label className="flex cursor-pointer select-none items-center gap-1.5">
            <input
              type="checkbox"
              checked={narrationOn}
              onChange={(e) => setShowNarration(e.target.checked ? "" : "off")}
              className="h-3 w-3 accent-[var(--mast-accent)]"
            />
            <span title="长任务跑的时候，持续解说正在做什么（这些话智能体看不到）">
              显示旁白
            </span>
          </label>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {convos.data?.degraded && <DegradedNote what="对话列表" />}
          {!activeId && conversations.length === 0 && (
            <EmptyNote label="点击「➕ 新对话」开始与仪器控制智能体私聊" />
          )}
          {activeId && history.isPending && !streamMsgs && <Spinner />}
          {activeId && history.isError && !streamMsgs && <ErrorNote error={history.error} />}
          {activeId && history.data?.degraded && !streamMsgs && <DegradedNote what="对话引擎" />}
          {activeId && (shownMessages.length > 0 || streaming) && (
            <NarrationLane
              messages={shownMessages}
              conversationId={activeId}
              pending={streaming}
              show={narrationOn}
            />
          )}
          {activeId &&
            !streaming &&
            !history.isPending &&
            !history.data?.degraded &&
            shownMessages.length === 0 && (
              <EmptyNote label="还没有消息，在下方输入发送第一条" />
            )}
          {streamErr && <p className="mt-3 text-sm text-mast-danger">流式出错：{streamErr}</p>}
        </div>
      </div>

      {/* agent 的订阅推荐就地确认 —— 不用离开对话去点头，那一步最容易流失。
          放在输入框**上方**：它是对刚才那轮对话的回应，不是页尾的一个通知。 */}
      <div className="mt-3">
        <PendingRecommendations compact />
      </div>

      {/* ── input bar (old mast-chat-input-wrap): textbox + Send + Stop + ⟳ ── */}
      <div className="mt-3 flex items-center gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          placeholder="Ask MAST anything..."
          disabled={!activeId || streaming}
          className="min-w-0 flex-1 rounded border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent disabled:opacity-50"
        />
        {streaming ? (
          <button
            onClick={() => void stop()}
            className="shrink-0 rounded border border-mast-danger-border bg-mast-danger-bg px-5 py-2 text-sm text-mast-danger hover:bg-mast-danger-border"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={() => void send()}
            disabled={!activeId || !input.trim()}
            className="shrink-0 rounded border border-mast-accent bg-mast-accent px-5 py-2 text-sm font-medium text-mast-accent-ink hover:opacity-90 disabled:opacity-40"
          >
            Send
          </button>
        )}
        <button
          onClick={speakLatest}
          disabled={!latestAssistant || voiceOut.speaking}
          title="朗读最新回复"
          className="shrink-0 rounded border border-mast-border px-3 py-2 text-sm text-mast-muted hover:border-mast-accent hover:text-mast-accent disabled:opacity-40"
        >
          {voiceOut.speaking ? "…" : "⟳"}
        </button>
      </div>

      {/* ── voice dock (full-duplex 语音：连续听/说 + 流式 + 打断，深挂 agent) ── */}
      <div className="mt-2">
        <VoiceDock conversationId={activeId} />
        {voiceMsg && <p className="mt-1 text-xs italic text-mast-muted">{voiceMsg}</p>}
      </div>

      {/* ── quick-prompt row (single flex-wrap line, below input) ── */}
      <div className="mt-3">
        <QuickPrompts onPick={(p) => setInput(input ? input + " " + p : p)} />
      </div>

      {/* 人工介入不再是页尾的一个 modal —— 它现在内联在数据条下面。 */}
      {/* 对话反馈悬浮窗 now lives in AppLayout (visible on every tab). */}
    </Section>
  );
}

// ────────────────────────────────────────────────────────────────────────
//  Vision Buffer — scan map + vision pulse + recent frames + buffer events.
//  Reuses the same components + endpoints as the standalone VisionPage so the
//  operator can watch vision without leaving Chat (old "Vision Buffer" sub-tab).
// ────────────────────────────────────────────────────────────────────────
function fmtTime(epochS?: number | null): string {
  if (!epochS) return "—";
  try {
    return new Date(epochS * 1000).toLocaleTimeString();
  } catch {
    return "—";
  }
}

function usePulse() {
  return useQuery({
    queryKey: ["vision", "pulse"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/vision/pulse");
      if (error) throw error;
      return data;
    },
    refetchInterval: 2000,
  });
}
function useRecent() {
  return useQuery({
    queryKey: ["vision", "recent"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/vision/recent");
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });
}

function VisionBufferSurface() {
  return (
    <div className="space-y-4">
      {/* `/vision` 在顶栏里没有位置（`nav.ts` 的 INTENTIONALLY_UNLISTED，理由写着
          「embedded elsewhere」），而这个 surface 只嵌了它七个子页里的四个 ——
          信号捕获+FFT / 长期监控 / 拼图 从来没有任何入口。加上之前，全仓渲染出来的
          链接里一个指向 `/vision` 的都没有：那句「embedded elsewhere」是半句真话，
          而剩下半句的症状就是用户找不到东西（#44 / 仪器初始化那一轮同一个形状）。 */}
      <div className="text-right text-xs">
        <Link to="/vision" className="text-mast-accent hover:underline">
          完整视觉页（信号捕获+FFT · 长期监控 · 拼图）→
        </Link>
      </div>
      <BufferScanMapSection />
      <BufferPulseSection />
      <BufferRecentFramesSection />
      <Section title="视觉缓冲">
        <VisionBufferPanel />
      </Section>
    </div>
  );
}

function BufferPulseSection() {
  const q = usePulse();
  return (
    <Section title="视觉脉冲">
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data &&
        (q.data.degraded ? <DegradedNote what="视觉脉冲" /> : <PulseRibbon pulse={q.data} />)}
    </Section>
  );
}

function BufferScanMapSection() {
  const coarseQ = useCoarseMap();
  return (
    <Section title="扫描地图">
      <p className="mb-3 text-xs text-mast-muted">
        实验助手的可视化核心：实时扫描框 + 针尖位置 + 所有带位置的历史操作（扫图 / STS / 电脉冲 / 修针尖 /
        移动 / 手动），以及计划路线与候选序列。每 3 秒刷新；图层与显示范围与视觉页共用一套设置。
      </p>
      <ScanMapPanel compact />

      {/* 第二张地图：样品台尺度（「扫描地图的粗动大地图呢？」）。
          它一直存在，只是只画在 /vision 上 —— 而 /vision 既不在顶栏、全仓也没有
          任何一个渲染出来的链接指向它（VisionRibbon 里那个 `<Link to="/vision">`
          在 onOpenBuffer 恒被传入的情况下永远走不到）。也就是说：唯一进得去的那张
          「扫描地图」上没有粗动图。
          上面那张是压电量程内的 ±1.5 µm（单位米，只看当前代次），这张是整个样品
          （单位步，跨全部代次）。刻意分开画，但必须**同时**画：尺度差本身就是
          「换区不能靠压电」那句话的证据，只剩一张时它就消失了。 */}
      <div className="mt-4">
        {/* 对话页上默认**紧凑**（「粗动地图占用空间过大，搞一个小地图
            就行了」）。这里它和聊天抢地方，而 `/vision` 上它是主角，所以差别做在
            调用点、不做在组件里。能缩放之后小尺寸才不等于看不清 ——
            两条反馈是同一个修法的两半。面板上有「展开」，随时能变回大图。 */}
        <CoarseMapPanel
          data={coarseQ.data}
          isPending={coarseQ.isPending}
          isError={coarseQ.isError}
          error={coarseQ.error}
          isFetching={coarseQ.isFetching}
          onRefresh={() => void coarseQ.refetch()}
          defaultCompact
        />
      </div>
    </Section>
  );
}

function BufferRecentFramesSection() {
  const q = useRecent();
  const data = q.data;
  const frames = data?.frames ?? [];
  return (
    <Section title="近期标注帧">
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {data && data.degraded && <DegradedNote what="近期标注帧" />}
      {data && !data.degraded && frames.length === 0 && <EmptyNote label="暂无近期视觉事件。" />}
      {data && !data.degraded && frames.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {frames.map((f) => (
            <Card key={f.seqno} className="space-y-2 p-3">
              <RecentFrameThumb frame={f} alt={f.summary || f.kind} />
              <div className="space-y-1 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{f.kind || "事件"}</span>
                  <span
                    className={
                      f.severity?.toLowerCase() === "critical"
                        ? "text-mast-danger"
                        : f.severity?.toLowerCase() === "warn" ||
                            f.severity?.toLowerCase() === "warning"
                          ? "text-mast-warn"
                          : "text-mast-muted"
                    }
                  >
                    {f.severity || "info"}
                  </span>
                </div>
                <div className="text-mast-muted">
                  {fmtTime(f.t_wall)} · #{f.seqno}
                </div>
                {f.summary && <div className="line-clamp-2 text-mast-text">{f.summary}</div>}
                {f.file_path && (
                  <div className="truncate text-mast-muted" title={f.file_path}>
                    {f.file_path}
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </Section>
  );
}

/** 归属字段在 schema.d.ts 里可能尚未出现（同 GroupChatHistoryPane 的做法），
 *  这里按可选字段本地声明。 */
type ConvScope = { experiment_id?: string | null; sample_id?: string | null };

/** 对话的归属徽章 —— 实验级 / 样品级 / 未归属。 */
function ConversationScopeBadge({ convo }: { convo: ConvScope | null }) {
  const scope = useCurrentScope();
  if (!convo) return null;
  const expName =
    convo.experiment_id && convo.experiment_id === scope.data?.experiment?.id
      ? scope.data?.experiment?.name
      : null;
  const smpName =
    convo.sample_id && convo.sample_id === scope.data?.sample?.id
      ? scope.data?.sample?.name
      : null;

  if (!convo.experiment_id) {
    return (
      <div className="mb-2 text-[11px] text-mast-faint">
        未归属实验 —— 这段对话不会被导出到任何实验文件夹。
      </div>
    );
  }
  return (
    <div className="mb-2 flex items-center gap-1.5 text-[11px]">
      <span className="rounded-mast-badge border border-mast-border px-1.5 py-px text-mast-muted">
        {convo.sample_id ? "样品级" : "实验级"}
      </span>
      <span className="truncate text-mast-faint">
        {smpName ?? expName ?? "已归属"}
      </span>
    </div>
  );
}


// ── helpers ───────────────────────────────────────────────────────────

/** Minimal HTML escape for the optimistic user bubble only (the backend will
 *  re-render the canonical version on the next snapshot). */
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br/>");
}
