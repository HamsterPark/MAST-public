"""Pydantic schemas for the multi-agent orchestrator run-task routes.

The run-task endpoint is the core MAST capability — it runs the real 6-agent
LangGraph orchestrator graph (supervisor → literature / experiment_design /
instrument_control / data_processing / paper_writing / paper_review) for a
single operator task and STREAMS its progress over SSE.

These endpoints are LIVE-ONLY: they relay onto the running orchestrator that
lives on the live app (``ctx.live_app`` — a ``CoreRuntime``):

  * ``_orchestrator``     — the compiled LangGraph 6-agent graph (or None until
                            a chat-model key is available / it has been built).
  * ``_orch_abort``       — the coarse abort ``threading.Event`` the stream loop
                            polls between super-steps (shared into every
                            ExecutionContext + the E_STOP hook — never replaced).
  * ``_orch_interrupts``  — the LangGraph HITL gate store (lock / pending /
                            resolved / events). When the IC subgraph pauses on a
                            DANGEROUS-skill approval we publish a pending entry
                            here and emit an ``interrupt`` SSE frame; the EXISTING
                            POST /agents/{id}/interrupts/{interrupt_id}/resolve
                            (agents_control.py) drains it and wakes the worker.

In standalone mode (no live app / no built orchestrator) the streaming handler
emits a SINGLE degraded SSE frame then done — never a 500. The abort handler
returns a typed degraded body (``degraded=True``). The authority (orchestration,
routing, SafetyGate, verdict translation) stays entirely in the live core / the
graph; the API only relays + bridges the sync stream to SSE.

The SSE frame shapes (``data: {json}\\n\\n``) are:

  {kind:"start",     task_id, task}
  {kind:"status",    text, backend}                     -- runtime info
  {kind:"message",   agent, role, text, t}              -- a per-agent message
  {kind:"handoff",   from_agent, to_agent, reason, t}   -- supervisor route / handoff
  {kind:"interrupt", interrupt_id, agent, skill, params, rationale,
                     allowed_decisions, interrupt_kind, thread_id, ask?} -- HITL gate
                     (interrupt_kind = dangerous | workflow_human | buffer_hitl |
                      ask_user; resolve via the EXISTING agents_control resolve
                      endpoint. ``ask`` is present ONLY for ask_user and carries
                      the structured question — question / header / options
                      [{label, description}] / multi_select / allow_custom /
                      timeout_action — which the choice card renders; that kind
                      resolves with decision="answer" + selected / custom_text.)
  {kind:"error",     message, degraded?}                -- recoverable / degraded
  {kind:"done",      final_text, aborted}               -- terminal
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── POST /agents/run-task (body) ─────────────────────────────────────────────
class RunTaskRequest(BaseModel):
    """Operator task for the 6-agent orchestrator.

    ``task`` is the natural-language instruction the supervisor decomposes and
    dispatches. ``conversation_id`` optionally continues an existing GROUP
    conversation's checkpointed thread (resumability); when omitted a fresh
    one-shot thread is used.
    """

    task: str = Field(..., description="Natural-language task for the supervisor.")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing GROUP conversation id to resume (reuses its durable "
        "checkpointed thread); omit for a fresh one-shot run.",
    )
    #: 「什么算做完了」——**代码求值**的终止判据（2026-08-27）。
    #:
    #: 省略 ⇒ 逐字节等于这道闸出现之前：模型自己判何时 __end__，代码层只有跳数/
    #: 预算/递归三个熔断兜底。给了 ⇒ 判据满足时确定性结束（不问模型），判据没
    #: 满足而模型想结束时转人问一次（每 run 一次）。
    #:
    #: 形状：一条谓词 ``{"kind": ..., ...}``，或 ``{"all"|"any"|"not": [...]}``，
    #: 或一个列表（= ``all`` 的简写）。谓词只能从 ``mast.goals.CATALOG`` 的闭集
    #: 里选；**非法一律 422 并逐条点名，不做部分丢弃** —— 丢掉 ``all`` 里一个
    #: 合取项会让目标被悄悄削弱，于是更早「达成」。
    done_when: Optional[Any] = Field(
        default=None,
        description=("目标终止判据（闭集，代码求值）。省略则行为与以前完全相同。"
                     "非法判据返回 422 并列出每一条的原因与可选目录。"),
    )
    goal_text: Optional[str] = Field(
        default=None,
        description=("目标的一句话描述；省略时取 task 本身。它只影响提示与提问"
                     "文案，不参与判定。"),
    )


# ── POST /agents/run-task/abort ──────────────────────────────────────────────
class AbortTaskResponse(BaseModel):
    """Result of signalling the orchestrator abort Event.

    LIVE-ONLY relay onto ``live_app._orch_abort`` (+ waking any worker blocked on
    a pending DANGEROUS interrupt). Degrades to a typed no-op when the live app /
    abort Event is absent — never a 500."""

    ok: bool = False
    aborted: list[str] = Field(default_factory=list)
    already_idle: bool = False
    detail: Optional[str] = None
    degraded: bool = True


# Streaming frames are emitted as raw SSE (see module docstring); they are not
# validated through a response_model (StreamingResponse). This typed shell exists
# only so callers / docs have a reference for the union of frame fields.
class RunTaskFrame(BaseModel):
    kind: str
    task_id: Optional[str] = None
    task: Optional[str] = None
    text: Optional[str] = None
    backend: Optional[str] = None
    agent: Optional[str] = None
    role: Optional[str] = None
    t: Optional[float] = None
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None
    reason: Optional[str] = None
    interrupt_id: Optional[str] = None
    skill: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    rationale: Optional[str] = None
    allowed_decisions: Optional[list[str]] = None
    interrupt_kind: Optional[str] = None  # dangerous | workflow_human
    thread_id: Optional[str] = None
    final_text: Optional[str] = None
    aborted: Optional[bool] = None
    message: Optional[str] = None
    degraded: Optional[bool] = None
    # 事件协议 v2（2026-08-26）：进度的**运行时无关**表示。``step`` / ``step_limit``
    # 说的是 LangGraph 的 super-step，退出之后没有对应物；这三个字段说的是「进度多少
    # / 上限多少 / 单位是什么」，旧引擎报 ``super_step``、新引擎报 ``model_call``。
    # 加字段不改字段 —— 老客户端照常工作。见 routes/orchestrator.py::_progress。
    progress: Optional[int] = None
    progress_limit: Optional[int] = None
    progress_unit: Optional[str] = None


# ── GET /agents/run-task/transcript (durable group transcript) ───────────────
class TranscriptEntry(BaseModel):
    """One persisted entry of a 群聊 (group) run, re-read on reconnect. ``kind``
    mirrors the live SSE entry kinds the TS client renders
    (operator | message | status | interrupt | compaction | done).

    ``compaction`` marks where the context middleware
    replaced earlier history with a summary; its ``meta`` carries the counts."""

    seq: int = 0
    kind: str = "message"
    agent_id: str = ""
    role: str = ""
    text: str = ""
    t: float = 0.0
    #: JSON sidecar for rows whose ``text`` is a SUMMARY, so the detail the
    #: summary leaves out is still there to expand (— a
    #: tool-call row carries ``{"tool", "args", "args_clipped"}``). Empty for
    #: every other row and for anything written before 2026-07-28.
    meta: str = ""


class TranscriptResponse(BaseModel):
    """The durable transcript for one group conversation. Lets the multi-agent
    conversation survive a tab switch / reload — the TS client fetches this on
    mount and replays it, then reconnects the live stream. Degrades to empty when
    no conversation store is wired (standalone dev)."""

    conversation_id: str
    entries: list[TranscriptEntry] = Field(default_factory=list)
    count: int = 0
    active: bool = False
    degraded: bool = True


# ── GET /agents/run-task/conversations (group history) ───────────────────────
class GroupConversation(BaseModel):
    conversation_id: str
    title: str = "新任务"
    thread_id: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_message_preview: str = ""
    # Where this 群聊 was started — its slot in 实验 → 样品 → 群聊 ().
    # NULL when nothing was active, or (sample_id) when the row predates the tag.
    # The chat is never scoped/filtered by these — it still spans experiments; the
    # UI groups by them so the operator sees the real three-level structure.
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    # Display names resolved server-side from ExperimentStorage so the UI can
    # render the tree without an N+1 fetch. None when the id is NULL or the
    # referenced row is gone (deleted experiment/sample → the chat still lists).
    experiment_name: Optional[str] = None
    sample_name: Optional[str] = None


class GroupConversationsResponse(BaseModel):
    """Durable group-conversation history (newest first) so the operator can
    resume a prior multi-agent run. ``active_conversation_id`` is the one a live
    run is currently streaming into (if any).

    Rows carry experiment/sample ids + names so the client can render the
    实验 → 样品 → 群聊 hierarchy the operator actually works in ."""

    conversations: list[GroupConversation] = Field(default_factory=list)
    active_conversation_id: Optional[str] = None
    count: int = 0
    degraded: bool = True


# ── Background runs (true parallelism — break the super-step barrier) ─────────
class BackgroundSpawnRequest(BaseModel):
    """Launch a long/independent task as a DETACHED orchestrator run so the
    foreground chat stays responsive. ``agents`` names the backgroundable agent(s)
    (instrument_control is rejected — the hardware agent stays foreground);
    ``conversation_id`` is the group conversation whose transcript the background
    results merge back into (omit → the run still executes but isn't merged)."""

    instruction: str = Field(..., description="Natural-language background task.")
    conversation_id: Optional[str] = Field(
        default=None, description="Group conversation to merge results into.")
    agents: list[str] = Field(
        default_factory=lambda: ["literature"],
        description="Backgroundable agent(s); instrument_control is rejected.")
    title: Optional[str] = Field(default=None, description="Short display title.")
    priority: Optional[str] = Field(
        default=None, description="Admission-queue priority: normal | high.")


class BackgroundRunInfo(BaseModel):
    run_id: str = ""
    conversation_id: str = ""
    instruction: str = ""
    agents: list[str] = Field(default_factory=list)
    title: str = ""
    priority: str = "normal"        # normal | high
    status: str = "queued"          # queued | running | done | failed | aborted
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    final_text: str = ""
    error: str = ""
    thread_id: str = ""
    # fine-grained progress (item ①) — surfaced to the UI progress bar
    steps: int = 0
    progress: Optional[int] = None  # 0..100
    last_activity: str = ""


class BackgroundSpawnResponse(BaseModel):
    ok: bool = False
    run: Optional[BackgroundRunInfo] = None
    detail: Optional[str] = None
    degraded: bool = True


class BackgroundRunsResponse(BaseModel):
    runs: list[BackgroundRunInfo] = Field(default_factory=list)
    count: int = 0
    degraded: bool = True


class BackgroundAbortResponse(BaseModel):
    ok: bool = False
    run_id: str = ""
    aborted: bool = False
    detail: Optional[str] = None
    degraded: bool = True


__all__ = [
    "RunTaskRequest",
    "AbortTaskResponse",
    "RunTaskFrame",
    "TranscriptEntry",
    "TranscriptResponse",
    "GroupConversation",
    "GroupConversationsResponse",
    "BackgroundSpawnRequest",
    "BackgroundRunInfo",
    "BackgroundSpawnResponse",
    "BackgroundRunsResponse",
    "BackgroundAbortResponse",
]
