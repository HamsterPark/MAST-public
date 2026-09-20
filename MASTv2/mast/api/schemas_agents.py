"""Pydantic request/response models for the Agents domain (TS-rewrite Phase 3).

Domain B (non-streaming): the per-agent tool catalog + model/thinking table, and
the conversation CRUD surface (list / create / rename / delete / messages /
chat-abort) the Agents tab drives. Chat token *streaming* is a separate seam.

Like ``mast.api.schemas`` these models are the single source of truth exported
via ``/openapi.json`` and consumed by the frontend type generators. Every
response carries a ``degraded`` boolean so the SPA can render an
empty-but-not-broken state when the live core (ConversationStore /
ConversationEngine / agents runtime) is not wired into this API process
(standalone dev). The API layer NEVER does business logic or safety checks —
it only mirrors the shapes the live handlers/backend already produce.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ── per-agent tool catalog ─────────────────────────────────────────────


class AgentToolEntry(BaseModel):
    """One tool a given agent exposes.

    Mirrors ``gui.agents_api._tool`` rows: ``name`` always present; ``safety``
    is the skill ``safety_level`` (AUTO for plain @tool tools); ``level`` is the
    composition level string (e.g. ``L0``) only for skill-registry skills."""

    name: str
    safety_level: str = "AUTO"
    composition_level: Optional[str] = None


class AgentToolCatalog(BaseModel):
    """The tool list for a single agent + a derived count (IC ~224)."""

    agent_id: str
    tools: list[AgentToolEntry] = Field(default_factory=list)
    count: int = 0


class AgentsToolsResponse(BaseModel):
    """Per-agent tool catalog for the Agents inspector. ``degraded`` is True when
    the static catalog has not been warmed / the agents runtime is unavailable —
    the frontend shows an empty-but-not-broken inspector."""

    agents: list[AgentToolCatalog] = Field(default_factory=list)
    degraded: bool = False


# ── per-agent model + thinking ─────────────────────────────────────────


class AgentModelInfo(BaseModel):
    """One agent's effective model alias + *honest* effective thinking level.

    ``thinking`` is what the model actually runs at (reasoning models pinned
    e.g. ``high (固定)``), not a requested override — mirrors
    ``gui.agents_api._resolve_agent_models``."""

    agent_id: str
    model: str
    thinking: Optional[str] = None


class AgentsModelsResponse(BaseModel):
    """Per-agent model+thinking table. ``degraded`` is True when the shared
    ``AGENT_MODEL`` registry is unavailable (standalone dev)."""

    agents: list[AgentModelInfo] = Field(default_factory=list)
    degraded: bool = False


# ── conversations ──────────────────────────────────────────────────────


class Conversation(BaseModel):
    """One conversation row (mirrors ``ConversationStore`` row dict).

    A conversation is a stable LangGraph ``thread_id`` + display metadata; the
    message history itself lives in the checkpointer keyed by ``thread_id``."""

    conversation_id: str
    agent_id: str
    thread_id: str
    title: str = "新对话"
    kind: str = "private"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_message_preview: str = ""
    # Where this chat was started, i.e. its place in 实验 → 样品 → 会话 .
    # Both may be NULL: nothing was active, or the row predates the sample tag.
    # Grouping metadata only — lists are never confined to one experiment/sample.
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    archived: bool = False


class ConversationsResponse(BaseModel):
    """The conversation list for one agent. ``degraded`` is True when no live
    ConversationStore is wired."""

    conversations: list[Conversation] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class CreateConversationRequest(BaseModel):
    """Body for POST .../conversations — all optional; the store fills defaults."""

    title: Optional[str] = None
    kind: str = "private"
    # Omit both and the server tags the chat with the experiment/sample that are
    # active right now — the usual case; an explicit value is for filing a chat
    # under a specific sample from the records UI .
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None


class CreateConversationResponse(BaseModel):
    """Result of a create. ``conversation`` is None on the degraded path."""

    ok: bool = False
    conversation: Optional[Conversation] = None
    degraded: bool = False


class RenameConversationRequest(BaseModel):
    """Body for PATCH .../conversations/{id} — the new title."""

    title: str


class MutationResult(BaseModel):
    """Generic write result for rename / delete / abort. ``ok`` reflects whether
    the live core applied the change; ``degraded`` True when no live core (so the
    frontend can distinguish 'not found / no-op' from 'backend absent')."""

    ok: bool = False
    degraded: bool = False


# ── chat abort ─────────────────────────────────────────────────────────


class ChatAbortRequest(BaseModel):
    """可选的请求体。不带就是「停这个 agent 名下所有正在跑的回合」。

    带 ``conversation_id`` 才是精确的那一种:私聊的多个会话都挂在同一个 agent 下,
    不点名就会把并发的另一条也停掉。
    """

    conversation_id: Optional[str] = None


class ChatAbortResult(BaseModel):
    """POST /chat/abort 的结果 —— **每个字段都必须是事实**。

    Event 对象可能在回合结束后继续保留；对象存在不代表回合仍在运行，
    更不代表停止信号已经送达。不能据此返回成功。

    现在 ``ok`` 的含义收紧成:**确实有正在跑的回合,并且它收到了停止信号**。

    ``ok`` 到此为止 —— 它**不**表示硬件已经停下。停止信号送达之后:

      * ``ExecutionContext.safe_call`` 立刻开始拒绝仪器写命令(只放行读取与
        停止/退针类动词);
      * 正在轮询的等待循环会在下一次 poll 退出(各家 poll 间隔不同,~0.02–2 s);
      * 但**已经下发、正阻塞在主 socket 上的那一条固件动作**(例如
        ``Motor_StartMove`` 的当前一段、``BiasSpectr_Start`` 的当前一次扫)不会被
        这条停止打断 —— 它只能由紧急停止从急停 socket 上用 ``Motor_StopMove`` /
        ``BiasSpectr_Stop`` 这类动词打断。``caveat`` 就是把这句话说给用户听,
        免得他以为「已停止」等于「针已经不动了」。
    """

    ok: bool = False
    degraded: bool = False
    signalled: int = 0
    conversation_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    caveat: str = ""


# ── messages ───────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """One rendered chat message — the shape ``mast.chat.render.render_history``
    produces."""

    role: str
    content: str
    #: 这条消息**发生的时刻**（epoch 秒），由 ``MessageClockMiddleware`` 盖上。
    #:
    #: ⚠️ ``None`` = **不知道它是什么时候说的**（重启前就在 checkpoint 里的
    #: 历史），不是「零时刻」。前端据此不显示时间 —— **绝不显示 1970**：
    #: 一个假时间比没有时间坏，因为它会被当成真的去推理。
    #:
    #: 要求 agent 的发言也带上时间戳。
    #: 它必须在这里显式声明：``render_history`` 已经产出 ``t``，而这个模型
    #: **只搬它声明过的字段** —— 少写一行的后果不是报错，是这个键在传输层被
    #: 静默丢掉，前端永远收不到（本仓「生产方接上了、消费方不存在」的同形第五次）。
    t: "float | None" = None


class MessagesResponse(BaseModel):
    """A conversation's full rendered history. ``degraded`` True when no live
    ConversationEngine is wired (empty-but-not-broken)."""

    conversation_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


# ── group activity (per-agent view of the 群聊 / multi-agent runs) ─────────


class GroupActivityEntry(BaseModel):
    """One message an agent contributed in a 群聊 (multi-agent orchestrator) run.

    The bridge that was missing in the rewrite: a 群聊 streamed each agent's
    messages ephemerally and indexed nothing, so an agent's individual view could
    never show what it said in the team run. These are read from the durable
    group transcript filtered by ``agent_id``."""

    conversation_id: str
    conversation_title: str = ""
    seq: int = 0  # stable per-conversation order key (for a stable React list key)
    role: str = ""
    text: str = ""
    t: float = 0.0
    #: JSON sidecar — a tool-call row's full arguments, which its one-line
    #: summary no longer spells out. Empty otherwise.
    meta: str = ""


class GroupActivityResponse(BaseModel):
    """An agent's 群聊 contributions across all multi-agent runs (newest first).
    Read-only mirror; degrades to empty when no ConversationStore is wired."""

    agent_id: str
    entries: list[GroupActivityEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


__all__ = [
    "AgentToolEntry",
    "AgentToolCatalog",
    "AgentsToolsResponse",
    "AgentModelInfo",
    "AgentsModelsResponse",
    "Conversation",
    "ConversationsResponse",
    "CreateConversationRequest",
    "CreateConversationResponse",
    "RenameConversationRequest",
    "MutationResult",
    "ChatMessage",
    "MessagesResponse",
    "GroupActivityEntry",
    "GroupActivityResponse",
]
