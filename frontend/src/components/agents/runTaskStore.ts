import { create } from "zustand";
import { persist } from "zustand/middleware";
import { api } from "@/api/client";
import { readSseStream, sseBroke, sseEndMessage } from "@/lib/sse";
import { AGENTS, SUPERVISOR, SUP_ID } from "./registry";

// ════════════════════════════════════════════════════════════════════════════
// runTaskStore — the 群聊 (multi-agent orchestrator) conversation, hoisted OUT
// of the React component into a module-scoped zustand store.
//
// WHY: the conversation used to live in RunTaskPanel's local useState, so any
// navigation / sub-tab switch UNMOUNTED the panel and wiped the whole thread —
// and the unmount cleanup even aborted the live SSE stream. The conversation was
// an ephemeral demo. Hoisting the state + driving the SSE loop from module scope
// means:
//   • switching tabs / pages no longer clears the conversation (the store holds
//     it for the whole SPA session) and no longer kills a running stream;
//   • the durable group conversation_id is persisted (localStorage), so a reload
//     reconnects: the panel re-fetches the server-side transcript (the source of
//     truth — the Python core persists every entry) and replays it.
//
// The server is authoritative for durable history; this store is the in-session
// UI mirror + the live-stream driver.
// ════════════════════════════════════════════════════════════════════════════

export type Role = "agent" | "tool" | "interject" | string;

/** The structured question behind an `interrupt_kind === "ask_user"` pause —
 *  an agent asking the OPERATOR to decide something, rather than asking them to
 *  approve something the agent already decided. */
export type AskQuestion = {
  question: string;
  header?: string;
  options: { label: string; description?: string }[];
  multi_select: boolean;
  allow_custom: boolean;
  timeout_action?: string;
};

export type Pending = {
  interrupt_id: string;
  agent: string;
  skill?: string | null;
  params?: Record<string, unknown> | null;
  rationale?: string | null;
  allowed_decisions: string[];
  interrupt_kind?: string | null;
  routes?: string[];
  ask?: AskQuestion | null;
  resolved?: boolean;
};

/** Narrow an untrusted `ask` payload, or give up and return null.
 *
 *  Returning null is a real outcome, not a failure path: the card then falls
 *  back to rendering `rationale` plus a free-text box, which still lets the
 *  operator answer. Rendering a half-parsed question would be worse — they
 *  would be choosing between options that may not be the ones the agent
 *  offered. */
export function normalizeAsk(raw: unknown): AskQuestion | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const question = typeof r.question === "string" ? r.question.trim() : "";
  if (!question) return null;
  const options = Array.isArray(r.options)
    ? r.options.flatMap((o) => {
        if (typeof o === "string") return o.trim() ? [{ label: o.trim() }] : [];
        if (!o || typeof o !== "object") return [];
        const label = String((o as Record<string, unknown>).label ?? "").trim();
        if (!label) return [];
        const description = String((o as Record<string, unknown>).description ?? "");
        return [description ? { label, description } : { label }];
      })
    : [];
  return {
    question,
    header: typeof r.header === "string" ? r.header : "",
    options,
    multi_select: r.multi_select === true && options.length > 0,
    // No options → the text box is the only way to answer, whatever the payload
    // claims (mirrors the same rule in the Python builder).
    allow_custom: r.allow_custom !== false || options.length === 0,
    timeout_action: typeof r.timeout_action === "string" ? r.timeout_action : "continue",
  };
}

export type Entry =
  | { kind: "operator"; text: string; target: string; t: number }
  // A `message` whose role is "tool" carries a one-line SUMMARY in `text`
  // (— the panel used to show `name(a=1, b=2, , )`, clipped
  // mid-argument with the closing paren pasted on after). What the summary
  // leaves out rides along so the operator can expand it:
  //   tool/args        — a tool CALL's name + its full arguments as pretty JSON
  //   detail           — a tool RETURN's raw payload, digested in `text`
  //   argsClipped      — the arguments were too big to send WHOLE; say so
  //                      rather than letting a partial payload read as complete
  | {
      kind: "message";
      agent: string;
      role: Role;
      text: string;
      t: number;
      tool?: string;
      args?: string;
      argsClipped?: boolean;
      detail?: string;
    }
  | { kind: "status"; text: string; backend?: string; t: number }
  // The context middleware replaced a stretch of this agent's history with a
  // summary. Every count is
  // OPTIONAL on purpose: the row renders as a bare "此处发生过一次上下文压缩"
  // divider when the backend didn't carry numbers, rather than showing a made-up
  // one. Above such a divider the transcript is NOT what was said.
  | {
      kind: "compaction";
      agent?: string;
      text: string;
      removed?: number;
      kept?: number;
      tokensBefore?: number;
      triggerTokens?: number;
      summary?: string;
      t: number;
    }
  | { kind: "interrupt"; pending: Pending; t: number }
  // `aborted` alone could not say WHY a run stopped, so a backend crash, an
  // unanswered approval and a real operator abort all rendered as 「已中止」 —
  // and a branch that died silently rendered as 「任务完成」. `failed` +
  // `stopReason` are the backend's own distinction (kind:"done" carries both);
  // consuming them is what makes the three outcomes three outcomes.
  | {
      kind: "done";
      finalText: string;
      aborted: boolean;
      failed?: boolean;
      stopReason?: string;
      t: number;
    };

export type KeyedEntry = Entry & { _k: string };
/** The approve/reject/edit verbs, plus the two verdicts that are NOT verbs: a
 *  `workflow_human` node takes one of its own route names, and an `ask_user`
 *  question takes the literal "answer" with the choice in `selected` /
 *  `custom_text`. Hence `string` rather than a closed union — the backend's
 *  ``allowed_decisions`` is the real gate. */
export type Decision = string;

/** Everything the resolve endpoint accepts beyond the verdict itself. */
export type ResolveExtras = {
  editedArgs?: Record<string, unknown> | null;
  selected?: string[] | null;
  customText?: string | null;
  comment?: string | null;
};

// @-mention roster: SUP + the 7 agents the operator can @-target.
const MENTIONABLE = [SUPERVISOR, ...AGENTS];

/** Resolve the FIRST "@<token>" in a message to an agent id (or null). */
export function mentionTarget(text: string): string | null {
  const m = text.match(/(^|\s)@(\S+)/);
  const tok = m?.[2]?.toLowerCase();
  if (!tok) return null;
  const hit = MENTIONABLE.find(
    (a) => a.id.toLowerCase() === tok || a.short.toLowerCase() === tok,
  );
  return hit ? hit.id : null;
}

// module-scoped (non-serializable) stream handles — survive component unmount so
// navigating away never aborts a running orchestrator stream.
let _abortCtrl: AbortController | null = null;
let _seq = 0;
function nextKey(): string {
  _seq += 1;
  return `e${_seq}`;
}

// result the panel uses to drive a toast (the store stays UI-framework-free).
export type SendResult =
  | { action: "start" }
  | { action: "interject"; ok: boolean; detail?: string; text: string }
  // server is still streaming this conversation (after a reload) — refuse to
  // open a second concurrent run on the same thread; the panel restores the text.
  | { action: "busy"; text: string }
  | { action: "noop" };

export type ResolveResult = {
  ok: boolean;
  applied?: boolean;
  detail?: string;
  // applied | no_pending_interrupt | route_not_allowed | answer_invalid | degraded
  status?: string;
};

interface RunTaskState {
  entries: KeyedEntry[];
  running: boolean;
  sending: boolean;
  degradedNote: string | null;
  streamErr: string | null;
  conversationId: string | null;
  // the conversationId we've already replayed the server transcript for, so a
  // remount does not re-fetch + duplicate.
  hydratedFor: string | null;
  // server says a run is still streaming into this conversation (after a reload
  // the client can't re-attach the live SSE, but can show the persisted tail).
  serverActive: boolean;
  // Turn-progress visibility: how far in / what the ceiling is, so the panel can
  // always show "N/限". The NAMES are historical (they were super-step counts);
  // `progressUnit` says what the numbers actually count, because the backend's
  // answer changes with the engine — "super_step" on the LangGraph path,
  // "model_call" on the v2 AgentLoop. Absent = the old path, which never said.
  step: number;
  stepLimit: number;
  progressUnit: string;
  // operator pause: a global __all__ hold — the
  // orchestrator parks at the next agent boundary until released.
  paused: boolean;
  // 目标终止判据（2026-08-28）。只有下发时带了 `done_when` 的 run 才会收到
  // `goal` 帧；没带的 run 这里永远是 null，面板上一个字都不多。
  //
  // 后端每次 supervisor 访问都会带上最新一次求值 —— 这个字段存在的理由是
  // 「代码认为还差什么」要**看得见**：Gate 1 会不问模型就结束、Gate 2 会拦下
  // 一次结束并转人问，而用户如果看不到判据，这两件事都像是系统在自作主张。
  goal: {
    verdict: string;
    met: number;
    total: number;
    reason: string;
    items: Array<{ kind?: string; state?: string; text?: string; reason?: string }>;
  } | null;

  send: (input: string) => Promise<SendResult>;
  abort: () => Promise<{ degraded: boolean }>;
  togglePause: () => Promise<{ ok: boolean; paused: boolean }>;
  resolveInterrupt: (
    it: Pending,
    decision: Decision,
    extras?: ResolveExtras,
  ) => Promise<ResolveResult>;
  markResolved: (interruptId: string) => void;
  hydrate: (force?: boolean) => Promise<void>;
  newConversation: () => void;
  switchConversation: (id: string) => Promise<void>;
}

function pushEntry(set: (fn: (s: RunTaskState) => Partial<RunTaskState>) => void, e: Entry) {
  set((s) => ({ entries: [...s.entries, { ...e, _k: nextKey() } as KeyedEntry] }));
}

// map a persisted transcript row → a render Entry (reconnect replay). Past HITL
// interrupts are dead (already resolved/expired), so they replay as a read-only
// status note, never an actionable card.
/** Parse a transcript row's JSON sidecar. Never throws: a row with an
 *  unreadable `meta` still renders its summary, just with nothing to expand. */
export function parseRowMeta(meta?: string | null): {
  tool?: string;
  args?: string;
  argsClipped?: boolean;
  detail?: string;
} {
  if (!meta) return {};
  try {
    const m = JSON.parse(meta) as Record<string, unknown>;
    if (!m || typeof m !== "object") return {};
    return {
      ...(typeof m.tool === "string" ? { tool: m.tool } : {}),
      ...(typeof m.args === "string" ? { args: m.args } : {}),
      ...(m.args_clipped === true ? { argsClipped: true } : {}),
      ...(typeof m.detail === "string" ? { detail: m.detail } : {}),
    };
  } catch {
    return {};
  }
}

/** Pull a compaction's facts out of a live frame's `compaction` object or a
 *  persisted row's JSON sidecar. Each field is carried over ONLY when it really
 *  arrived as a number/string — a missing count stays missing so the divider can
 *  say "压缩过" without inventing "压掉了 N 条". */
export function parseCompactionMeta(raw: unknown): {
  removed?: number;
  kept?: number;
  tokensBefore?: number;
  triggerTokens?: number;
  summary?: string;
} {
  let m: Record<string, unknown> | null = null;
  if (typeof raw === "string") {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === "object") m = parsed as Record<string, unknown>;
    } catch {
      return {};
    }
  } else if (raw && typeof raw === "object") {
    m = raw as Record<string, unknown>;
  }
  if (!m) return {};
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
  const removed = num(m.removed);
  const kept = num(m.kept);
  const tokensBefore = num(m.tokens_before_estimate);
  const triggerTokens = num(m.trigger_tokens);
  const summary = typeof m.summary === "string" && m.summary ? m.summary : undefined;
  return {
    ...(removed !== undefined ? { removed } : {}),
    ...(kept !== undefined ? { kept } : {}),
    ...(tokensBefore !== undefined ? { tokensBefore } : {}),
    ...(triggerTokens !== undefined ? { triggerTokens } : {}),
    ...(summary !== undefined ? { summary } : {}),
  };
}

function transcriptRowToEntry(r: {
  kind?: string;
  agent_id?: string;
  role?: string;
  text?: string;
  t?: number;
  meta?: string;
}): Entry {
  const t = r.t ?? 0;
  switch (r.kind) {
    case "operator":
      return { kind: "operator", text: r.text ?? "", target: SUP_ID, t };
    case "status":
      return { kind: "status", text: r.text ?? "", t };
    case "interrupt":
      return { kind: "status", text: `⚠ 审批 · ${r.text ?? ""}`, t };
    case "compaction":
      // Replays exactly like the live divider — this is the one row whose
      // ABSENCE on reload would put the operator back where // found them: reading a rewritten history that looks untouched.
      return {
        kind: "compaction",
        ...(r.agent_id ? { agent: r.agent_id } : {}),
        text: r.text ?? "",
        t,
        ...parseCompactionMeta(r.meta),
      };
    case "done": {
      // The persisted marker is one of exactly three shapes written by
      // `_terminal_label`: "完成", "已中止" (a genuine operator abort), or
      // "运行出错，已停止：<reason>". Matching only on "已中止" meant a CRASHED
      // conversation replayed as 「任务完成」 — the crash label does not contain
      // that word (2026-07-28 audit, 附带发现).
      const marker = (r.text ?? "").trim();
      const failed = marker.startsWith("运行出错");
      const abortedRow = marker.includes("已中止");
      return {
        kind: "done",
        finalText: r.text ?? "",
        aborted: failed || abortedRow,
        failed,
        ...(failed
          ? { stopReason: marker.replace(/^运行出错，已停止[:：]\s*/, "") }
          : {}),
        t,
      };
    }
    default:
      return {
        kind: "message",
        agent: r.agent_id || "agent",
        role: (r.role as Role) || "agent",
        text: r.text ?? "",
        t,
        // A replayed row expands exactly like a live one. Rows written before
        // 2026-07-28 have no sidecar and simply have nothing to expand — their
        // `text` is still the old raw call string, unchanged.
        ...parseRowMeta(r.meta),
      };
  }
}

export const useRunTaskStore = create<RunTaskState>()(
  persist(
    (set, get) => {
      // ── SSE frame → conversation entry (mirrors the live frame protocol) ──
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const handleFrame = (f: any) => {
        if (f?.degraded) {
          set(() => ({
            degradedNote:
              typeof f.message === "string" && f.message
                ? f.message
                : "编排器后端降级（独立开发模式或未接入核心）。",
          }));
        }
        const now = Date.now() / 1000;
        // Surface the turn's progress from any frame that carries it.
        //
        // Two spellings, on purpose. `progress` / `progress_limit` /
        // `progress_unit` are the runtime-agnostic pair (event protocol v2,
        // 2026-08-26); `step` / `step_limit` are the LangGraph super-step count
        // the backend has always sent. The super-step number is not a unit
        // anybody can reason about — its exchange rate with "how much work is
        // left in this turn" drifts with the middleware count (the same literal
        // 50 bought ~9 tool calls in June and 3 in August, with no diff ever
        // saying "limit lowered"). And once the LangGraph exit lands there are no
        // super-steps at all.
        //
        // So: prefer the new fields, fall back to the old ones. Both engines can
        // then talk to this client unchanged, and the day the old fields stop
        // being sent nothing here has to move.
        if (typeof f?.progress === "number") set(() => ({ step: f.progress }));
        else if (typeof f?.step === "number") set(() => ({ step: f.step }));
        if (typeof f?.progress_limit === "number") set(() => ({ stepLimit: f.progress_limit }));
        else if (typeof f?.step_limit === "number") set(() => ({ stepLimit: f.step_limit }));
        if (typeof f?.progress_unit === "string" && f.progress_unit) {
          set(() => ({ progressUnit: String(f.progress_unit) }));
        }
        switch (f?.kind) {
          case "start":
            // capture the durable group conversation id the server created/resumed
            if (f.conversation_id) {
              set(() => ({ conversationId: String(f.conversation_id), hydratedFor: String(f.conversation_id) }));
            }
            set(() => ({
              step: 0,
              stepLimit: Number(f.progress_limit ?? f.step_limit) || 0,
              progressUnit: typeof f.progress_unit === "string" ? f.progress_unit : "",
            }));
            pushEntry(set, { kind: "status", text: `任务已下发：${f.task ?? ""}`, t: now });
            break;
          case "goal":
            // 未知 kind 在 default 里是 no-op，所以这一条即使前端旧一版也不会
            // 出错；加它是因为「生产方接了、消费方缺席」在本仓是有名字的病。
            set(() => ({
              goal: {
                verdict: String(f.verdict ?? ""),
                met: Number(f.met ?? 0),
                total: Number(f.total ?? 0),
                reason: String(f.reason ?? ""),
                items: Array.isArray(f.items) ? f.items : [],
              },
            }));
            break;
          case "status":
            pushEntry(set, {
              kind: "status",
              text: String(f.text ?? ""),
              backend: f.backend,
              t: f.t ?? now,
            });
            break;
          case "message":
            pushEntry(set, {
              kind: "message",
              agent: String(f.agent ?? "agent"),
              role: String(f.role ?? "agent"),
              text: String(f.text ?? ""),
              t: f.t ?? now,
              ...(f.tool ? { tool: String(f.tool) } : {}),
              ...(f.args ? { args: String(f.args) } : {}),
              ...(f.args_clipped ? { argsClipped: true } : {}),
              ...(f.detail ? { detail: String(f.detail) } : {}),
            });
            break;
          case "compaction":
            pushEntry(set, {
              kind: "compaction",
              ...(f.agent ? { agent: String(f.agent) } : {}),
              text: String(f.text ?? "上下文压缩"),
              t: f.t ?? now,
              ...parseCompactionMeta(f.compaction),
            });
            break;
          case "interrupt":
            set((s) => {
              if (
                s.entries.some(
                  (e) => e.kind === "interrupt" && e.pending.interrupt_id === f.interrupt_id,
                )
              ) {
                return {};
              }
              const p: Pending = {
                interrupt_id: String(f.interrupt_id),
                agent: String(f.agent ?? SUP_ID),
                skill: f.skill,
                params: f.params ?? null,
                rationale: f.rationale,
                allowed_decisions: Array.isArray(f.allowed_decisions) ? f.allowed_decisions : [],
                interrupt_kind: f.interrupt_kind,
                ...(Array.isArray(f.routes) ? { routes: f.routes.map(String) } : {}),
                ask: normalizeAsk(f.ask),
              };
              return {
                entries: [...s.entries, { kind: "interrupt", pending: p, t: now, _k: nextKey() }],
              };
            });
            break;
          case "error":
            set(() => ({ streamErr: String(f.message ?? "编排器出错") }));
            break;
          case "done":
            pushEntry(set, {
              kind: "done",
              finalText: String(f.final_text ?? "task completed"),
              aborted: !!f.aborted,
              failed: !!f.failed,
              ...(f.stop_reason ? { stopReason: String(f.stop_reason) } : {}),
              t: now,
            });
            if (f.conversation_id) set(() => ({ conversationId: String(f.conversation_id) }));
            break;
          default:
            break;
        }
      };

      // ── START a task (first message, or any message while idle) ──────────
      const startTask = async (instruction: string) => {
        // clear a stale degraded banner: a fresh start that is itself still
        // degraded will re-emit the degraded frame before done.
        set(() => ({ running: true, streamErr: null, degradedNote: null, serverActive: true }));
        const ctrl = new AbortController();
        _abortCtrl = ctrl;
        // Set when the stream dies rather than finishing — decides whether
        // `serverActive` may be cleared in the finally below.
        let streamBroke = false;
        // Why it died, in operator language; the consequence sentence is added
        // after we've asked the server whether the run is still going.
        let breakCause: string | null = null;
        try {
          const resp = await fetch("/api/agents/run-task", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              task: instruction,
              conversation_id: get().conversationId || null,
            }),
            signal: ctrl.signal,
          });
          if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
          const end = await readSseStream(resp, { onFrame: handleFrame, controller: ctrl });
          if (sseBroke(end)) {
            // A body that ends WITHOUT the server's `done` frame, or 75 s of
            // total silence through the keep-alive, is a cut connection — not a
            // finished run. This used to be indistinguishable from success
            // (`if (done) break`), so a Tailscale roam left the transcript
            // stopping mid-sentence with no error, no spinner and no reconnect
            // banner: the conversation looked finished when it was not
            //.
            breakCause = sseEndMessage(end);
            set(() => ({ streamErr: breakCause }));
            streamBroke = true;
          }
        } catch (err) {
          if ((err as Error)?.name !== "AbortError") {
            breakCause = `连接出错：${String((err as Error)?.message ?? err)}`;
            set(() => ({ streamErr: breakCause }));
            // The stream died on us; the RUN did not. Keep serverActive so the
            // abort button stays reachable — see the finally below.
            streamBroke = true;
          }
        } finally {
          // Losing the stream is NOT the run ending.
          //
          // This used to clear `serverActive` unconditionally, which meant a
          // network blip / tab switch / refresh made the abort button vanish
          // (it is gated on `running || serverActive`) while the backend kept
          // going — and the next message then bounced off the busy guard with
          // "该群聊仍在后端运行". That left no way out and a
          // hint pointing at a button that does not exist. That is the "UI must never freeze" invariant being broken
          // not by a frozen screen but by a functional dead end.
          //
          // So: `running` (this tab is streaming) always clears, but
          // `serverActive` (the backend is working) is only cleared when the
          // stream ENDED NORMALLY — i.e. we saw the end of the response rather
          // than an error. hydrate() reconciles it against the server either way.
          set(() => ({
            running: false,
            ...(streamBroke ? {} : { serverActive: false }),
          }));
          _abortCtrl = null;
        }

        // ── auto-recover from a cut stream  ─────────────────────────────
        // Losing the connection must not cost the operator anything they'd have
        // to notice and repair by hand. `running` is false by now, so hydrate()
        // is allowed to run: it replays the persisted transcript (catching the
        // view up on everything that arrived after the cut) and reports whether
        // the backend run is still going. Only THEN can we say what the break
        // actually means — hydrate clears streamErr on success (that is what
        // 刷新进展 means), so the explanation is restated with its consequence.
        if (streamBroke) {
          try {
            await get().hydrate(true);
          } catch {
            /* offline / server unreachable — the banner below still stands */
          }
          const stillRunning = get().serverActive;
          set(() => ({
            streamErr:
              (breakCause ?? "连接中断。") +
              (stillRunning
                ? "后端任务仍在继续 —— 上方已自动补上最新进展；要停止请点「中止后端运行」。"
                : "后端任务已结束 —— 上方已是最终记录。"),
          }));
        }
      };

      // ── LIVE interject (steer the running run) ───────────────────────────
      const interject = async (text: string, to: string) => {
        try {
          const { data, error } = await api.POST("/api/agents/{agent_id}/interject", {
            params: { path: { agent_id: to } },
            body: { text },
          });
          if (error) throw error;
          if (data?.ok && !data.degraded) return { ok: true };
          return { ok: false, detail: data?.detail ?? "编排器未在运行 / 后端降级" };
        } catch (e) {
          return { ok: false, detail: String((e as Error)?.message ?? e) };
        }
      };

      return {
        entries: [],
        running: false,
        sending: false,
        degradedNote: null,
        streamErr: null,
        conversationId: null,
        hydratedFor: null,
        serverActive: false,
        step: 0,
        stepLimit: 0,
        // 空串 = 后端没说单位（旧引擎的老帧）。刻意不猜一个默认值：
        // 「读不到」与「是 super_step」是两件事。
        progressUnit: "",
        goal: null,
        paused: false,

        send: async (input: string): Promise<SendResult> => {
          const text = input.trim();
          if (!text || get().sending) return { action: "noop" };
          set(() => ({ sending: true }));
          const to = mentionTarget(text) ?? SUP_ID;
          try {
            if (get().running) {
              // running → inject live. Echo the operator turn ONLY on success so a
              // failed interject (orchestrator idle / degraded / network) doesn't
              // leave a misleading "sent" bubble with the text already lost.
              const r = await interject(text, to);
              if (r.ok) {
                pushEntry(set, { kind: "operator", text, target: to, t: Date.now() / 1000 });
              }
              return { action: "interject", ok: r.ok, detail: r.detail, text };
            }
            // idle BUT the server is still streaming this conversation (reload
            // case): refuse to open a second concurrent run on the same thread.
            if (get().serverActive) {
              return { action: "busy", text };
            }
            // idle → start (or continue) the durable group thread. NOT awaited to
            // completion here — startTask drives the stream in module scope so it
            // keeps running across navigation; we return immediately.
            pushEntry(set, { kind: "operator", text, target: to, t: Date.now() / 1000 });
            void startTask(text);
            return { action: "start" };
          } finally {
            set(() => ({ sending: false }));
          }
        },

        abort: async () => {
          let degraded = false;
          try {
            const { data } = await api.POST("/api/agents/run-task/abort");
            degraded = !!data?.degraded;
          } catch {
            degraded = true;
          }
          _abortCtrl?.abort();
          // An abort always lifts a standing pause — otherwise the NEXT run
          // would silently park at its first agent boundary.
          if (get().paused) {
            try {
              await api.POST("/api/agents/{agent_id}/release", {
                params: { path: { agent_id: "__all__" } },
              });
            } catch {
              /* best-effort */
            }
            set(() => ({ paused: false }));
          }
          return { degraded };
        },

        // Global pause/resume (群聊缺少暂停和继续功能).
        // Sets/clears the backend's __all__ hold; the run-task stream honours it
        // at the next agent boundary (_honor_holds) and yields held/resumed
        // status frames, so the transcript narrates the pause too.
        togglePause: async () => {
          const want = !get().paused;
          try {
            const { data, error } = await api.POST(
              want ? "/api/agents/{agent_id}/hold" : "/api/agents/{agent_id}/release",
              { params: { path: { agent_id: "__all__" } } },
            );
            if (error || !data?.ok) return { ok: false, paused: get().paused };
            set(() => ({ paused: want }));
            return { ok: true, paused: want };
          } catch {
            return { ok: false, paused: get().paused };
          }
        },

        resolveInterrupt: async (it, decision, extras) => {
          try {
            const { data, error } = await api.POST(
              "/api/agents/{agent_id}/interrupts/{interrupt_id}/resolve",
              {
                params: { path: { agent_id: it.agent, interrupt_id: it.interrupt_id } },
                body: {
                  decision,
                  edited_args: extras?.editedArgs ?? null,
                  comment: extras?.comment ?? null,
                  selected: extras?.selected ?? null,
                  custom_text: extras?.customText ?? null,
                },
              },
            );
            if (error) throw error;
            if (data?.ok && data?.applied) {
              get().markResolved(it.interrupt_id);
              return { ok: true, applied: true };
            }
            // NOT resolved. `answer_invalid` / `route_not_allowed` mean the
            // backend deliberately left the worker blocked so the operator can
            // correct and resubmit — greying the card out here would strand a
            // run that is still sitting there waiting for an answer.
            return {
              ok: false,
              detail: data?.detail ?? "后端降级或中断已处理",
              ...(data?.status ? { status: data.status } : {}),
            };
          } catch (e) {
            return { ok: false, detail: String((e as Error)?.message ?? e) };
          }
        },

        markResolved: (interruptId) =>
          set((s) => ({
            entries: s.entries.map((e) =>
              e.kind === "interrupt" && e.pending.interrupt_id === interruptId
                ? { ...e, pending: { ...e.pending, resolved: true } }
                : e,
            ),
          })),

        // Replay the durable server transcript for the current conversation. Used
        // on (re)mount / reload so a tab switch or refresh is non-destructive.
        hydrate: async (force = false) => {
          const cid = get().conversationId;
          if (!cid) return;
          if (get().running) return; // live stream already owns the entries
          if (!force && get().hydratedFor === cid && get().entries.length > 0) return;
          try {
            const { data, error } = await api.GET("/api/agents/group-transcript", {
              params: { query: { conversation_id: cid } },
            });
            if (error) throw error;
            if (!data || data.degraded) {
              // store unwired → keep whatever in-session entries we have.
              set(() => ({ hydratedFor: cid }));
              return;
            }
            const replay = (data.entries ?? []).map(
              (r) => ({ ...transcriptRowToEntry(r), _k: nextKey() }) as KeyedEntry,
            );
            set(() => ({
              entries: replay,
              hydratedFor: cid,
              serverActive: !!data.active,
              streamErr: null,
            }));
          } catch {
            set(() => ({ hydratedFor: cid }));
          }
        },

        newConversation: () => {
          _abortCtrl?.abort();
          set(() => ({
            entries: [],
            running: false,
            sending: false,
            degradedNote: null,
            streamErr: null,
            conversationId: null,
            hydratedFor: null,
            serverActive: false,
            step: 0,
            stepLimit: 0,
            progressUnit: "",
        goal: null,
          }));
        },

        switchConversation: async (id: string) => {
          if (get().running) _abortCtrl?.abort();
          set(() => ({
            entries: [],
            running: false,
            conversationId: id,
            hydratedFor: null,
            degradedNote: null,
            streamErr: null,
          }));
          await get().hydrate(true);
        },
      };
    },
    {
      name: "mast-runtask",
      // persist ONLY the durable group id — entries are re-fetched from the
      // server (the source of truth), never stuffed into localStorage.
      partialize: (s) => ({ conversationId: s.conversationId }),
    },
  ),
);
