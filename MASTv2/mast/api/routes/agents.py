"""Agents domain routes (non-streaming) — TS-rewrite Phase 3.

Endpoints:
  GET    /api/agents/tools                                  per-agent tool catalog
  GET    /api/agents/models                                 per-agent model+thinking
  GET    /api/agents/{agent_id}/conversations              list conversations
  POST   /api/agents/{agent_id}/conversations              create a conversation
  PATCH  /api/agents/{agent_id}/conversations/{id}         rename a conversation
  DELETE /api/agents/{agent_id}/conversations/{id}         delete a conversation
  GET    /api/agents/{agent_id}/messages?conversation_id=  rendered chat history
  POST   /api/agents/{agent_id}/chat/abort                 abort the active turn

GRACEFUL DEGRADATION is mandatory: this API must boot STANDALONE next to the
live Gradio app without a wired core. Every endpoint that needs a live
subsystem (the conversation store / engine, the agents runtime, the shared
model registry) checks for it on ``ctx``, lazy-imports the heavy core module
INSIDE the handler wrapped in try/except, and on absence-or-error returns a
valid empty/degraded response (``degraded=True``) — NEVER 500, NEVER crashes on
import. NO business logic or safety checks live here — the API only calls into
the core. Real wiring to the live singletons (ConversationStore /
ConversationEngine + checkpointer, abort registry, safety passthrough) happens
at integration time at integration.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.schemas_agents import (
    AgentModelInfo,
    AgentsModelsResponse,
    AgentsToolsResponse,
    AgentToolCatalog,
    AgentToolEntry,
    ChatAbortRequest,
    ChatAbortResult,
    ChatMessage,
    Conversation,
    ConversationsResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    GroupActivityEntry,
    GroupActivityResponse,
    MessagesResponse,
    MutationResult,
    RenameConversationRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents"])

# Canonical agent roster (mirrors orchestrator._AGENT_NAMES + the two non-graph
# agents the inspector surfaces). Kept local so a missing agents runtime can
# never crash this read-only module on import.
_AGENT_IDS = (
    "instrument_control",
    # 2026-08-21：research_director 有私聊入口（runtime._CHAT_AGENT_MODULES 已接），
    # 这份名单不跟上，那条入口就是「接好了但这一层不认」。
    "research_director",
    "literature",
    "experiment_design",
    "data_processing",
    "paper_writing",
    "paper_review",
    "buffer_summarizer",
)


def _conversation_store(ctx):
    """Best-effort handle to the live ConversationStore (wired at integration at
    integration time). Returns None in standalone dev. Never raises."""
    return getattr(ctx, "conversation_store", None)


def _conversation_engine(ctx):
    """Best-effort handle to the live ConversationEngine. None in standalone dev."""
    return getattr(ctx, "conversation_engine", None)


def _current_experiment_id() -> "str | None":
    """Best-effort current experiment id — a PROVENANCE tag for a new conversation
    (which experiment it was born in). Conversations are NOT scoped to it: lists
    never filter by experiment_id, so a chat still spans experiments. Never raises."""
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        return getattr(log, "current_experiment_id", None) if log else None
    except Exception:
        return None


def _current_sample_id() -> "str | None":
    """Best-effort current sample id — the MIDDLE level of the hierarchy the
    operator actually works in: 一个实验若干样品，一个样品若干会话. Chats used to hang straight off the experiment, so the
    sample level did not exist in the data and could not be shown anywhere.

    Same contract as :func:`_current_experiment_id`: a where-it-started tag, not
    a lease — a chat outlives its sample and is never filtered away by it. Never
    raises (None when no sample is active, which is a perfectly valid chat)."""
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        return getattr(log, "current_sample_id", None) if log else None
    except Exception:
        return None


def _to_conversation(row: dict) -> Conversation:
    """Map a ConversationStore row dict → the typed Conversation model."""
    return Conversation(
        conversation_id=str(row.get("conversation_id")),
        agent_id=str(row.get("agent_id", "")),
        thread_id=str(row.get("thread_id", "")),
        title=row.get("title") or "新对话",
        kind=row.get("kind") or "private",
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
        last_message_preview=row.get("last_message_preview") or "",
        experiment_id=row.get("experiment_id"),
        sample_id=row.get("sample_id"),
        archived=bool(row.get("archived")),
    )


# ── tool catalog ───────────────────────────────────────────────────────


@router.get("/agents/tools", response_model=AgentsToolsResponse)
def get_agent_tools(request: Request) -> AgentsToolsResponse:
    """Per-agent tool catalog (name / safety_level / composition_level).

    Sourced from the cached static catalog computed off the live registry +
    domain tool lists (gui.agents_api). Degrades to empty when the catalog has
    not been warmed / the agents runtime is unavailable."""
    try:
        # gui.agents_api owns the cached catalog (warm_agent_tools). It is
        # metadata-only (no torch / LLM clients / pools), but the registry
        # discovery it does can be heavy/unavailable standalone — so it's
        # lazy-imported here and any failure degrades.
        from mast.webui.agents_api import get_agent_tools as _get_cached

        raw = _get_cached()  # {agent_id: [{name, safety, level?}, ...]} or {}
        if not raw:
            return AgentsToolsResponse(degraded=True)

        agents: list[AgentToolCatalog] = []
        for agent_id, rows in raw.items():
            tools = [
                AgentToolEntry(
                    name=str(r.get("name", "")),
                    safety_level=str(r.get("safety", "AUTO")),
                    composition_level=r.get("level"),
                )
                for r in (rows or [])
                if r.get("name")
            ]
            agents.append(
                AgentToolCatalog(agent_id=str(agent_id), tools=tools, count=len(tools))
            )
        return AgentsToolsResponse(agents=agents, degraded=False)
    except Exception as exc:  # any wiring/shape mismatch → degrade, never 500
        logger.warning("agent tools catalog build failed: %s", exc)
        return AgentsToolsResponse(degraded=True)


# ── model + thinking ───────────────────────────────────────────────────


@router.get("/agents/models", response_model=AgentsModelsResponse)
def get_agent_models(request: Request) -> AgentsModelsResponse:
    """Per-agent effective model alias + honest effective thinking level.

    Reads the shared ``AGENT_MODEL`` registry + ``effective_thinking`` (gui.
    agents_api._resolve_agent_models). Degrades to empty when the registry is
    unavailable (standalone dev)."""
    try:
        from mast.webui.agents_api import _resolve_agent_models

        models, thinking = _resolve_agent_models()
        if not models:
            return AgentsModelsResponse(degraded=True)
        agents = [
            AgentModelInfo(
                agent_id=str(aid),
                model=str(model_id),
                thinking=thinking.get(aid),
            )
            for aid, model_id in models.items()
        ]
        return AgentsModelsResponse(agents=agents, degraded=False)
    except Exception as exc:
        logger.warning("agent models table build failed: %s", exc)
        return AgentsModelsResponse(degraded=True)


# ── conversations: list / create ───────────────────────────────────────


@router.get("/agents/{agent_id}/conversations", response_model=ConversationsResponse)
def list_conversations(agent_id: str, request: Request) -> ConversationsResponse:
    """List an agent's (non-archived) conversations, newest first."""
    ctx = request.app.state.ctx
    store = _conversation_store(ctx)
    if store is None:
        return ConversationsResponse(degraded=True)
    try:
        # kind="private" ONLY: group (群聊) rows are agent_id="_supervisor" and must
        # not leak into a per-agent private-chat list (they'd pollute the 编排器
        # 私聊 list and, when clicked, read empty from a graph that has no
        # "_supervisor" private chat). Group history has its own /agents/
        # group-conversations endpoint.
        rows = store.list(agent_id=agent_id, kind="private")
        conversations = [_to_conversation(r) for r in (rows or [])]
        return ConversationsResponse(
            conversations=conversations,
            count=len(conversations),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("conversation list failed for %s: %s", agent_id, exc)
        return ConversationsResponse(degraded=True)


@router.post(
    "/agents/{agent_id}/conversations", response_model=CreateConversationResponse
)
def create_conversation(
    agent_id: str, request: Request, body: CreateConversationRequest | None = None
) -> CreateConversationResponse:
    """Create a new conversation for *agent_id*. Degrades to ``ok=False`` when no
    live store is wired (the contract still type-checks; integration wires the
    live singleton at integration time)."""
    ctx = request.app.state.ctx
    store = _conversation_store(ctx)
    if store is None:
        return CreateConversationResponse(ok=False, degraded=True)
    body = body or CreateConversationRequest()
    try:
        row = store.create(
            agent_id,
            kind=body.kind,
            title=body.title,
            # Default to whatever the operator is working in, so the chat lands in
            # the right 实验 → 样品 slot without them having to say so . Still
            # only a tag: the chat spans experiments and outlives the sample.
            experiment_id=body.experiment_id or _current_experiment_id(),
            sample_id=body.sample_id or _current_sample_id(),
        )
        return CreateConversationResponse(
            ok=True, conversation=_to_conversation(row), degraded=False
        )
    except Exception as exc:
        logger.warning("conversation create failed for %s: %s", agent_id, exc)
        return CreateConversationResponse(ok=False, degraded=True)


# ── conversations: rename / delete ─────────────────────────────────────


@router.patch(
    "/agents/{agent_id}/conversations/{conversation_id}",
    response_model=MutationResult,
)
def rename_conversation(
    agent_id: str,
    conversation_id: str,
    request: Request,
    body: RenameConversationRequest,
) -> MutationResult:
    """Rename a conversation. ``ok`` reflects whether the store applied it (False
    when the id is unknown); ``degraded`` True when no live store is wired."""
    ctx = request.app.state.ctx
    store = _conversation_store(ctx)
    if store is None:
        return MutationResult(ok=False, degraded=True)
    try:
        ok = bool(store.rename(conversation_id, body.title))
        return MutationResult(ok=ok, degraded=False)
    except Exception as exc:
        logger.warning("conversation rename failed for %s: %s", conversation_id, exc)
        return MutationResult(ok=False, degraded=True)


@router.delete(
    "/agents/{agent_id}/conversations/{conversation_id}",
    response_model=MutationResult,
)
def delete_conversation(
    agent_id: str, conversation_id: str, request: Request
) -> MutationResult:
    """Delete a conversation (and best-effort purge its checkpoint thread — the
    store handles that when a checkpointer is supplied at integration time).
    Degrades to ``ok=False`` when no live store is wired."""
    ctx = request.app.state.ctx
    store = _conversation_store(ctx)
    if store is None:
        return MutationResult(ok=False, degraded=True)
    try:
        # The checkpointer purge happens via the live engine's checkpointer at
        # integration time; standalone the store still removes the index row.
        checkpointer = getattr(_conversation_engine(ctx), "_checkpointer", None)
        ok = bool(store.delete(conversation_id, checkpointer=checkpointer))
        return MutationResult(ok=ok, degraded=False)
    except Exception as exc:
        logger.warning("conversation delete failed for %s: %s", conversation_id, exc)
        return MutationResult(ok=False, degraded=True)


# ── messages ───────────────────────────────────────────────────────────


def _epoch_or_none(value) -> "float | None":
    """一个**可信的 epoch 秒**，否则 ``None``。

    为什么不复用 ``_num_or_none``（仓里已有三份，``noble_tip_workflow`` /
    ``monitoring.store`` / ``_tip_policy``）：

    * 它们只挡 NaN。而这里 **0、负数、inf 同样是「不知道」** —— 一个 0 会被
      前端画成 1970 年，那比不显示时间坏得多（假时间会被当成真的去推理）。
      语义不同的东西共用一个名字，比抄一份更容易出错。
    * 而且那三份都住在领域模块里，从 API 路由 import 任何一个，都会为了五行
      算术把一整棵领域依赖拖进传输层。

    所以这里不是「第四份 ``_num_or_none``」，是**另一个判据**，名字也照此起。
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num or num <= 0.0 or num == float("inf"):
        return None
    return num


@router.get("/agents/{agent_id}/messages", response_model=MessagesResponse)
def get_messages(
    agent_id: str, request: Request, conversation_id: str
) -> MessagesResponse:
    """Rendered chat history for a conversation (read from the checkpointer via
    the engine, the source of truth). Degrades to empty when no live engine."""
    ctx = request.app.state.ctx
    engine = _conversation_engine(ctx)
    if engine is None:
        return MessagesResponse(conversation_id=conversation_id, degraded=True)
    try:
        rendered = engine.get_messages(conversation_id)  # [{role, content[, t]}, ...]
        # ``t`` 原样搬过来，**不给默认值**:缺席表示「不知道它是什么时候说的」
        # (重启前的历史),而 0 会被前端画成 1970 年。见 ChatMessage.t。
        messages = [
            ChatMessage(role=str(m.get("role", "")),
                        content=str(m.get("content", "")),
                        t=_epoch_or_none(m.get("t")))
            for m in (rendered or [])
            if isinstance(m, dict)
        ]
        return MessagesResponse(
            conversation_id=conversation_id,
            messages=messages,
            count=len(messages),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("get_messages failed for %s: %s", conversation_id, exc)
        return MessagesResponse(conversation_id=conversation_id, degraded=True)


# ── group activity (per-agent view of 群聊 / multi-agent runs) ───────────


@router.get("/agents/{agent_id}/group-activity", response_model=GroupActivityResponse)
def get_group_activity(agent_id: str, request: Request) -> GroupActivityResponse:
    """An agent's 群聊 contributions across all multi-agent orchestrator runs.

    Restores the bridge the rewrite dropped: the multi-agent conversation now
    persists each agent's messages into the durable group transcript, so an
    agent's individual view can surface what it said/did in the team run. READ-
    ONLY relay onto ``ConversationStore.agent_activity``; degrades to empty when
    no live store is wired (standalone dev)."""
    ctx = request.app.state.ctx
    store = _conversation_store(ctx)
    if store is None:
        return GroupActivityResponse(agent_id=agent_id, degraded=True)
    try:
        rows = store.agent_activity(agent_id)
        entries = [
            GroupActivityEntry(
                conversation_id=str(r.get("conversation_id", "")),
                conversation_title=str(r.get("conversation_title") or ""),
                seq=int(r.get("seq", 0) or 0),
                role=str(r.get("role", "")),
                text=str(r.get("text", "")),
                t=float(r.get("t", 0) or 0),
                meta=str(r.get("meta") or ""),
            )
            for r in (rows or [])
        ]
        return GroupActivityResponse(
            agent_id=agent_id, entries=entries, count=len(entries), degraded=False
        )
    except Exception as exc:
        logger.warning("group activity read failed for %s: %s", agent_id, exc)
        return GroupActivityResponse(agent_id=agent_id, degraded=True)


# ── chat abort ─────────────────────────────────────────────────────────


def _stop_caveat() -> str:
    """``ok`` 的边界，取自单一真源 ``core.abortability``。

    在这里**现取**而不是写成模块常量：软停哪天真的开始下发硬件停止动词
    （``soft_stop_sends_hardware_verbs()``），这句话必须跟着变。写死一份就会变成
    第二真源，然后在某次「顺手改一下」里静静变成假话。
    """
    try:
        from mast.core.abortability import SOFT_STOP_CAVEAT

        return SOFT_STOP_CAVEAT
    except Exception:  # noqa: BLE001 — 一句提示文案绝不能让停止端点 500
        return "停止信号已送达。仪器写命令即刻被拒；需要立刻让针停下请用紧急停止。"


@router.post("/agents/{agent_id}/chat/abort", response_model=ChatAbortResult)
def abort_chat(agent_id: str, request: Request,
               body: ChatAbortRequest | None = None) -> ChatAbortResult:
    """给 *agent_id*(可选:某个会话)**正在跑的**回合发停止信号。

    ## ``ok`` 说的是什么(2026-08-11 收紧)

    以前 ``ok:true`` 的含义是「这个 agent_id 名下有一个 Event 对象」—— 而那张表只增
    不减,所以进程发生过第一次对话之后它**永远**为真。实测:对一轮正在跑的仪器动作
    POST abort,回 ``{"ok":true,"degraded":false}``,那一轮跑完了整张 500 nm 图。

    现在:``ok`` ⇔ ``signalled >= 1`` ⇔ **确实有正在跑的回合,且它收到了停止信号**。
    没有正在跑的回合就 ``ok=false`` 并在 ``reason`` 里说明 —— 一个撒谎的停止按钮比
    一个明显坏掉的停止按钮更危险,用户会以为已经停了然后走开。

    ``ok`` 也**不**表示硬件已经停下,那不是这一层知道的事;``caveat`` 把边界说清楚。

    API 层不含业务逻辑:真正的登记表在 ``routes.chat_stream``(每轮一个专属事件,
    ``finally`` 里摘除),经由 ``ctx.chat_abort`` 调过去。没有 live runtime 时
    ``degraded=True`` —— 那是「问不到」,与「没有东西在跑」是两回事,别混。
    """
    ctx = request.app.state.ctx
    # The live app exposes a best-effort abort hook integration wires onto ctx;
    # absent it, this is a typed no-op (never 500, never hangs the UI).
    abort_hook = getattr(ctx, "chat_abort", None)
    if not callable(abort_hook):
        return ChatAbortResult(
            ok=False, degraded=True,
            reason="后端未接入停止通道（独立开发模式）—— **无法判断有没有东西在跑**。")
    cid = body.conversation_id if body is not None else None
    try:
        try:
            out = abort_hook(agent_id, cid)
        except TypeError:
            # 老式 hook（只收 agent_id，返回 bool）。不点名会话就退回全量停止；
            # 点了名却接不住，宁可拒绝也不假装停对了那一条。
            if cid:
                return ChatAbortResult(
                    ok=False, degraded=True,
                    reason="当前停止通道不支持按会话点名停止，已拒绝（避免连带停掉并发的另一条会话）。")
            out = abort_hook(agent_id)
    except Exception as exc:
        logger.warning("chat abort failed for %s: %s", agent_id, exc)
        return ChatAbortResult(ok=False, degraded=True,
                               reason=f"停止调用出错：{type(exc).__name__}: {exc}")
    if not isinstance(out, dict):
        # 老式 bool hook：它答不出「有没有一轮在跑」，所以这里也不许把它翻译成
        # 一个确定的 ok —— 说明它是个不带证据的回答。
        ok = bool(out)
        return ChatAbortResult(
            ok=ok, degraded=not ok,
            signalled=1 if ok else 0,
            reason="" if ok else "停止通道未报告任何正在进行的回合。",
            caveat=_stop_caveat() if ok else "")
    n = int(out.get("signalled", 0) or 0)
    return ChatAbortResult(
        ok=n >= 1,
        degraded=False,
        signalled=n,
        conversation_ids=list(out.get("conversation_ids") or []),
        reason=str(out.get("reason") or ""),
        caveat=_stop_caveat() if n >= 1 else "")
