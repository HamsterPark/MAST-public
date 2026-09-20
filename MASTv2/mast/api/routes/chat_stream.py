"""POST /api/agents/{agent_id}/chat — stream a chat turn over SSE.

Bridges the existing sync ``ConversationEngine.stream_turn`` generator (which
drives the real LangGraph private-chat graph and yields rendered history
snapshots) to a text/event-stream. Starlette runs the sync generator in a
threadpool, so the blocking graph.stream never stalls the event loop.

Frames: {kind:"snapshot", messages:[{role,content}]} … then {kind:"done"}.
Degrades to a single error frame when no ConversationEngine is wired (standalone
dev). Abort is a per-TURN threading.Event registered in the live-turn registry
below; the agents-slice /chat/abort endpoint fires it via the ctx.chat_abort hook
integration wires in app.py.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from mast.api import hitl_bridge as _bridge
from mast.api.schemas_chat import ChatTurnRequest
from mast.api.sse import SSE_HEADERS, with_heartbeat

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ── live-turn registry ─────────────────────────────────────────────────
#
# 这张表回答的是 /chat/abort 唯一该回答的问题:**现在有没有一轮在跑,它收到停止
# 信号了吗**。
#
# 它替换掉的旧实现是 ``_ABORTS: dict[agent_id, Event]`` —— 一个**只增不减**的表:
#
#   * Event 可能在回合结束后继续保留，因此仅检查对象存在会误报停止成功。
#     停止信号未送达时，仪器动作可能继续；响应必须区分信号送达与硬件已停止。
#   * 同一个 agent 的多个会话**共用一个 Event** ⇒ 停一个聊天会连带停掉并发的另一
#     个;更糟的是新回合开头那句 ``ev.clear()`` 会把另一条正在跑的回合刚收到的停止
#     信号清掉。engine 的 docstring 说隔离的单位是**会话**,这里没做到。
#
# 现在:每一轮开始时登记一个**全新的** Event(按 agent+会话),``finally`` 里摘除。
# 于是「表里有没有」正好等于「有没有一轮在跑」,``ok`` 才有资格说真话;而每轮一个
# 新事件也让 ``clear()`` 这个动作整个消失,连带那条竞态一起。
_LIVE_TURNS: dict[tuple[str, str], dict] = {}
_LIVE_LOCK = threading.Lock()


def _begin_turn(agent_id: str, conversation_id: str) -> tuple[tuple[str, str], threading.Event]:
    """登记「这一轮开始跑了」,返回 (key, 这一轮**专属**的停止事件)。"""
    key = (str(agent_id), str(conversation_id))
    ev = threading.Event()
    with _LIVE_LOCK:
        _LIVE_TURNS[key] = {"event": ev, "started_at": time.time()}
    return key, ev


def _end_turn(key: tuple[str, str]) -> None:
    """摘除登记 —— **必须在 finally 里调**。留着的话端点又会开始撒谎。"""
    with _LIVE_LOCK:
        _LIVE_TURNS.pop(key, None)


def _reset_turn_registry_for_tests() -> None:
    """只给测试用:清空进程级登记。上一条用例留下的活跃轮次会让
    「没有正在跑的轮次」那条断言凭空变绿。"""
    with _LIVE_LOCK:
        _LIVE_TURNS.clear()


def chat_abort_hook(agent_id: str, conversation_id: str | None = None) -> dict:
    """给这个 agent(可选:某个会话)**正在跑的**回合发停止信号。

    返回的是**事实**,不是「我试过了」:

      ``signalled``          真正被置位的回合数。0 就是 0 —— 端点据此拒绝报 ok。
      ``conversation_ids``   被停的是哪几轮。
      ``reason``             signalled==0 时,为什么(人话,给用户看)。

    ``ok`` 的含义到此为止是「**这一轮收到了停止信号**」,不是「硬件已经停下」。
    后者不是这个函数能知道的事,所以它不说。
    """
    want = str(conversation_id) if conversation_id else None
    with _LIVE_LOCK:
        hits = [(k, rec) for k, rec in _LIVE_TURNS.items()
                if k[0] == str(agent_id) and (want is None or k[1] == want)]
    for _k, rec in hits:
        rec["event"].set()
    if hits:
        logger.info("chat abort: signalled %d live turn(s) for %s: %s",
                    len(hits), agent_id, [k[1] for k, _ in hits])
        return {"signalled": len(hits),
                "conversation_ids": [k[1] for k, _ in hits],
                "reason": ""}
    if want is not None:
        reason = (f"会话 {want} 当前没有正在进行的回合 —— **没有东西被停止**"
                  "（这不等于「已经停了」）。")
    else:
        reason = (f"{agent_id} 当前没有正在进行的回合 —— **没有东西被停止**"
                  "（这不等于「已经停了」）。")
    logger.info("chat abort: no live turn for %s (conversation=%s)", agent_id, want)
    return {"signalled": 0, "conversation_ids": [], "reason": reason}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _make_hitl_resolver(ctx, agent_id: str, conversation_id: str, abort):
    """A resolver that publishes this turn's interrupt and blocks for the answer.

    ``ConversationEngine.stream_turn`` has always had the resume loop and this
    hook; nothing ever passed one, so a private chat that hit a DANGEROUS skill
    (or, now, asked the operator a question) printed "当前入口未接审批处理器" and
    stopped with the graph parked in its checkpoint — while the approval panel,
    polling the very store this writes into, showed nothing pending.

    Returns ``None`` when there is no live store (standalone dev), which keeps
    the old honest-notice behaviour instead of pretending a gate exists.
    """
    app = getattr(ctx, "live_app", None)
    store = getattr(app, "_orch_interrupts", None) if app is not None else None
    if not store:
        return None

    def _resolver(interrupted):
        pubs = _bridge.publish_interrupts(
            store, agent_id, interrupted, f"chat:{conversation_id}")
        if not pubs:
            logger.error("chat %s: unparseable __interrupt__: %r", agent_id, interrupted)
            return None
        values = [
            # The limits are read from the module HERE, not taken from
            # resolve_blocking's defaults: a default argument is bound at def
            # time, so a test that shortens MAX_APPROVAL_WAIT_S would be
            # silently ignored and its "timeout fails closed" assertion would
            # pass without ever timing out. Same reason run-task passes them.
            _bridge.resolve_blocking(store, p["event_id"], p["kind"], abort,
                                     max_wait=_bridge.MAX_APPROVAL_WAIT_S,
                                     beat=_bridge.APPROVAL_BEAT_S)
            for p in pubs
        ]
        # Any unanswered one (abort, or a timeout the asker said should stop)
        # means we do not resume: a partial answer set would feed the graph a
        # decision for one request and nothing for another.
        if any(v is None for v in values):
            return None
        # Address each paused branch BY ITS LangGraph interrupt id. A bare
        # resume value is BROADCAST to every pending interrupt, so under a
        # parallel fan-out one skill's approval would also answer another's
        # pending SetBias (spike-verified 2026-07-11 on the run-task path,
        # which assembles the resume exactly this way).
        by_lg: dict = {}
        legacy = False
        for p, v in zip(pubs, values):
            lg = p.get("lg_id")
            if not lg:
                legacy = True
                continue
            if p["kind"] in ("workflow_human", "ask_user"):
                by_lg[lg] = v                      # single-valued resume
            else:
                slot = by_lg.setdefault(lg, {"decisions": []})
                slot["decisions"].extend((v or {}).get("decisions", []))
        if by_lg and not legacy:
            return by_lg
        # Older LangGraph builds expose no interrupt id. The private-chat graph
        # is serial, so exactly one interrupt is pending and the broadcast is
        # equivalent — keep the legacy single value rather than send a mapping
        # whose keys would match nothing.
        if pubs[0]["kind"] in ("workflow_human", "ask_user"):
            return values[0]
        merged: list = []
        for v in values:
            merged.extend((v or {}).get("decisions", []))
        return {"decisions": merged}

    return _resolver


@router.post("/agents/{agent_id}/chat")
def chat_turn(agent_id: str, body: ChatTurnRequest, request: Request) -> StreamingResponse:
    ctx = request.app.state.ctx
    engine = getattr(ctx, "conversation_engine", None)

    def gen():
        if engine is None:
            yield _sse({"kind": "error", "message": "对话引擎未接入（独立开发模式）", "degraded": True})
            yield _sse({"kind": "done"})
            return
        key, ev = _begin_turn(agent_id, body.conversation_id)
        resolver = _make_hitl_resolver(ctx, agent_id, body.conversation_id, ev)
        try:
            for snapshot in engine.stream_turn(body.conversation_id, body.user_text,
                                               abort=ev, hitl_resolver=resolver):
                yield _sse({"kind": "snapshot", "messages": snapshot})
        except Exception as exc:  # surface the real error as a frame, never 500 mid-stream
            logger.warning("chat stream failed (%s): %s", agent_id, exc)
            yield _sse({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            # 摘登记必须在**发 done 之前**:客户端看到 done 就认为这一轮结束了,
            # 之后它再点停止,端点必须能如实说「没有正在跑的轮次」。
            _end_turn(key)
            yield _sse({"kind": "done"})

    # A single LLM turn can be minutes of total silence — which is precisely what
    # an idle-TCP reaper (Tailscale roam, NAT timeout, sleep) kills, leaving the
    # operator staring at a chat that never finishes (). with_heartbeat
    # keeps bytes on the wire and gives the client a trustworthy idle watchdog.
    return StreamingResponse(
        with_heartbeat(gen(), label=f"chat-{agent_id}"),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


__all__ = ["router", "chat_abort_hook", "_begin_turn", "_end_turn",
           "_reset_turn_registry_for_tests"]
