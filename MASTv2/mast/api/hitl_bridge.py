"""The HITL interrupt store: publish a pause, wait for the operator, resume.

Extracted from ``routes/orchestrator.py`` (2026-08-01) so BOTH entry points can
use it. Until then this machinery was reachable only from ``POST /run-task``,
with these consequences in private chat — which is the surface most operators
actually use:

  * ``ConversationEngine.stream_turn`` has always had a resume loop and a
    ``hitl_resolver`` hook, and **nothing ever passed one**. A private chat that
    hit a DANGEROUS skill printed "需要人工审批，但当前入口未接审批处理器" and
    stopped, leaving the graph parked in its checkpoint.
  * Nothing published those interrupts into the store, so the approval panel —
    which polls exactly this store — showed "当前无待处理中断" while a turn sat
    frozen waiting for an approval that had no way to arrive.

So the gate existed, the UI existed, and the two were not connected. Sharing the
store is what connects them; ``ask_user`` then works in private chat for free,
because it is the same pause.

WHY ``api/`` AND NOT ``core/``: :func:`await_resolution` BLOCKS its caller until
an operator answers. That is legitimate on a Starlette threadpool worker (which
is what run-task and the chat stream both are) and would be a bug inside the
graph, where the no-blocking invariant holds. Keeping it on the API side of the
seam keeps that distinction visible.

The store itself is ``CoreRuntime._orch_interrupts``::

    {"lock": threading.Lock(),
     "pending":  {event_id: <normalised dict>},   # what the UI lists
     "resolved": {event_id: <resume value>},      # what the worker consumes
     "events":   {event_id: threading.Event()}}   # how the worker is woken

It is process-local and deliberately so: a pending interrupt only means anything
while the worker blocked on it is alive. A restart loses the queue, not the work
— the graph's checkpoint still holds the paused thread, and the next turn
re-raises the interrupt with a fresh id.
"""
from __future__ import annotations

import logging
import threading
import time as _time
import uuid as _uuid
from typing import Any

logger = logging.getLogger(__name__)

# How long a published HITL approval waits for an operator verdict before the
# caller gives up. Nothing that blocks a stream worker may block it forever:
# this used to be ``while True``, and an approval nobody answered pinned a
# Starlette worker permanently while the conversation stayed "active", so every
# later message bounced off the busy guard.
#
# ⚠️ 这条通道**只服务对话轮之内的审批**。它的三条性质——900 s 上限、
# 进程本地的中断表、resume 会重放整个 tool call——对一次对话都是对的，
# 对「等人做事」则**每一条都是错的**：换样品要几个小时，降温要一夜，
# 而重放一次「退针 → 等人 → 进针」会**重复退针**。
#
# 所以框架级的分工是：
#   * 对话轮内的审批 → 这里；
#   * 等人做事 / 等物理条件 → 异步心愿单 + conduct 的等待闸
#     （`mast/conduct/spec.py` 的 WaitSpec：状态落在 conducts 行里，
#      判定发生在 tick 之间 —— **等人落在 run 之间，不落在 run 之内**）。
#
# `mast/conduct/**` 不得 import 本模块，有结构测试钉着
# （`tests/v2/unit/conduct/test_waiting_stays_out_of_the_run.py`）。
MAX_APPROVAL_WAIT_S = 900.0

# How often the wait emits a status frame. Doubles as the SSE keep-alive: a
# silent connection is what idle-TCP reapers (WireGuard keepalive, gateway idle
# timeouts, Wi-Fi roaming, sleep) kill — which is how a network blip turned into
# "对话卡住" . ws.py pings every 15 s for the same reason.
APPROVAL_BEAT_S = 15.0


def extract_hitl_request(interrupt_obj):
    """Pull the HITLRequest dict out of a LangGraph Interrupt (or raw value)."""
    val = getattr(interrupt_obj, "value", interrupt_obj)
    if isinstance(val, dict) and "action_requests" in val:
        return val
    return None



def publish_interrupts(store, owner, chunk_value, thread_id) -> list[dict]:
    """Register the pending interrupt(s) from a '__interrupt__' chunk into the
    LIVE ``_orch_interrupts`` store (the SAME store agents_control.resolve drains)
    and return the normalized pending dicts to emit as SSE frames.

    We publish exactly the shape the React UI + the resolve relay consume
    (event_id / agent_id / skill / params / rationale / allowed_decisions / kind /
    thread_id). Authority (verdict→Decision translation, SafetyGate re-check)
    stays in the core via ``_build_decision`` + the graph on resume."""
    if not store:
        return []
    interrupts = chunk_value if isinstance(chunk_value, (list, tuple)) else [chunk_value]
    published: list[dict] = []
    for intr in interrupts:
        _val = getattr(intr, "value", intr)
        # LangGraph's OWN interrupt id. Under parallel fan-out several branches
        # can be paused at once, and a resume must address each one BY ID
        # (Command(resume={id: value})) — a single bare resume value is broadcast
        # to every pending interrupt, so agent A's approval would also be fed to
        # agent B's pending SetBias (spike-verified 2026-07-11). None on the
        # serial path of older LangGraph builds → caller falls back to the legacy
        # single-value resume.
        _lg_id = getattr(intr, "id", None)
        # workflow_human node: interrupt(payload) inside a composite.
        if isinstance(_val, dict) and _val.get("kind") == "workflow_human":
            routes = [str(r) for r in (_val.get("routes") or ["resolved"])]
            pendings = [{
                "event_id": f"intr_{thread_id}_{_uuid.uuid4().hex[:8]}",
                "agent_id": owner,
                "skill": (f"工作流 {_val.get('workflow', '?')} · 节点 "
                          f"{_val.get('node_id', '?')}"),
                "params": dict(_val.get("inputs") or {}),
                "rationale": str(_val.get("message", "")),
                "allowed_decisions": list(routes),
                "routes": list(routes),
                "kind": "workflow_human",
                "thread_id": thread_id,
                "t": _time.time(),
            }]
        elif isinstance(_val, dict) and _val.get("kind") == "ask_user":
            # An agent (or the supervisor's ask_operator node) is asking the
            # OPERATOR a structured question and is blocked on the answer —
            # agents/_shared/ask_tools.py. Unlike every other kind here the
            # operator is not judging something the agent already decided; they
            # are making the decision.
            #
            # skill / rationale / params are filled with the question so a client
            # that does not know this kind still SHOWS it (the pre-2026-08 cards
            # render exactly those three) instead of a blank approval box. The
            # structured form the choice UI needs rides in ``ask``.
            _ask = dict(_val.get("ask") or {})
            if not _ask:
                _ask = {
                    "question": str(_val.get("question", "")),
                    "header": str(_val.get("header", "")),
                    "options": [o for o in (_val.get("options") or [])
                                if isinstance(o, dict)],
                    "multi_select": bool(_val.get("multi_select", False)),
                    "allow_custom": bool(_val.get("allow_custom", True)),
                    "timeout_action": str(_val.get("timeout_action") or "continue"),
                }
            _labels = [str(o.get("label")) for o in (_ask.get("options") or [])
                       if isinstance(o, dict)]
            pendings = [{
                "event_id": f"intr_{thread_id}_{_uuid.uuid4().hex[:8]}",
                # Self-reported owner wins: the supervisor's ask node is a PARENT
                # graph node, so its namespace is empty and `owner` would fall
                # back to instrument_control — attributing the supervisor's
                # question to an agent that never asked it.
                "agent_id": str(_val.get("agent_id") or "") or owner,
                "skill": "向用户提问",
                "params": {
                    "question": _ask.get("question", ""),
                    "options": _labels,
                    "multi_select": _ask.get("multi_select", False),
                    "allow_custom": _ask.get("allow_custom", True),
                },
                "rationale": str(_ask.get("question", "")),
                # Not approve/reject — the verdict IS the answer.
                "allowed_decisions": ["answer"],
                "kind": "ask_user",
                "ask": _ask,
                "thread_id": thread_id,
                "t": _time.time(),
            }]
        elif isinstance(_val, dict) and _val.get("kind") == "buffer_hitl":
            # BufferHITLMiddleware paused on critical hardware event(s) (tip
            # crash, e-stop, retract-needed). Surface as ONE operator approval so
            # the run does NOT silently abort as "unparseable" — the 2026-07-06
            # test dropped a critical tip_quality_drop here and the operator never
            # got a prompt.
            #
            # The verdict MATTERS (fixed 2026-07-28; it did not before). approve
            # reopens the middleware's tool gate and the run drives on; reject —
            # and the 900 s timeout that fails closed as reject — leaves the gate
            # SHUT, so the run may still read, stop and retract but cannot push
            # the experiment forward on data it has been told not to trust.
            events = _val.get("events") or []
            kinds = ", ".join(str(e.get("kind", "?")) for e in events) or "?"
            suggested = (events[0].get("suggested_action") if events else "") or "review"
            pendings = [{
                "event_id": f"intr_{thread_id}_{_uuid.uuid4().hex[:8]}",
                "agent_id": owner,
                "skill": f"缓冲区关键事件：{kinds}",
                "params": {"events": events, "suggested_action": suggested},
                "rationale": (f"检测到 {len(events)} 个关键硬件事件（{kinds}），"
                              f"建议动作：{suggested}。请在机台确认/处理后决定是否继续。"),
                "allowed_decisions": ["approve", "reject"],
                "kind": "buffer_hitl",
                "thread_id": thread_id,
                "t": _time.time(),
            }]
        else:
            req = extract_hitl_request(intr)
            if req is None:
                continue
            action_requests = req.get("action_requests") or []
            review_configs = req.get("review_configs") or []
            allowed_by_name = {
                rc.get("action_name"): rc.get("allowed_decisions")
                for rc in review_configs if isinstance(rc, dict)
            }
            # ONE pending per action_request. The old code took the first and
            # `break`-ed ("one pending per chunk"), so when the middleware asked
            # for several skills in a single interrupt the extra approvals were
            # silently dropped — and the resume then carried fewer decisions than
            # the middleware expected. All requests are surfaced now; they share
            # the interrupt's lg_id and their decisions are merged back into ONE
            # resume value for it (see _drive).
            pendings = []
            for ar in action_requests:
                if not isinstance(ar, dict):
                    continue
                skill = ar.get("name") or "未知 skill"
                args = ar.get("args") or {}
                pendings.append({
                    "event_id": f"intr_{thread_id}_{_uuid.uuid4().hex[:8]}",
                    "agent_id": owner,
                    "skill": skill,
                    "params": dict(args) if isinstance(args, dict) else {},
                    "rationale": ar.get("description") or "",
                    "allowed_decisions": list(
                        allowed_by_name.get(skill) or ["approve", "edit", "reject"]
                    ),
                    "kind": "dangerous",
                    "thread_id": thread_id,
                    "t": _time.time(),
                })
            if not pendings:
                continue
        for pending in pendings:
            # Tag every pending with the LangGraph interrupt it belongs to, so a
            # parallel run can resume each paused branch by id instead of
            # broadcasting one value to all of them.
            pending["lg_id"] = _lg_id
            eid = pending["event_id"]
            lock = store.get("lock")
            if lock is not None:
                with lock:
                    store["pending"][eid] = pending
                    store["events"][eid] = threading.Event()
            else:  # pragma: no cover - lock always present on a live store
                store["pending"][eid] = pending
                store["events"][eid] = threading.Event()
            published.append(pending)
            logger.warning(
                "run-task HITL: pending interrupt %s skill=%s agent=%s lg_id=%s",
                eid, pending["skill"], owner, _lg_id)
    return published


def await_resolution(store, event_id: str, kind: str, abort, *,
                     max_wait: float = MAX_APPROVAL_WAIT_S,
                     beat: float = APPROVAL_BEAT_S):
    """Wait for agents_control.resolve to produce a Decision for *event_id*.

    A GENERATOR: yields ``("beat", frame)`` while waiting, then exactly one
    ``("result", resume_value_or_None)`` as its final item. It has to be a
    generator because the caller IS the SSE generator — a plain callback could
    not push a frame out to the client, and a silent wait is exactly what made
    a stuck approval impossible to diagnose from the operator's chair.

    TIMEOUT — why it exists:
    this used to be ``while True``, i.e. an approval nobody answered pinned a
    Starlette threadpool worker FOREVER and left ``task.active=True`` on that
    conversation, so every later message bounced off the busy guard ("该群聊仍在
    后端运行"). The operator saw "编排器处理中… 由于某种原因卡住了，原因不详",
    and a mere network blip on the remote machine was enough to strand a run
    that nobody could then answer. The hold path below already had
    ``_MAX_HOLD_S`` auto-resume for exactly this reason; the approval path was
    the one place still able to hang indefinitely.

    On timeout we fail **CLOSED** — a forgotten approval resolves to *reject*,
    never approve. An unattended hardware action must not run just because the
    operator walked away.

    ``ask_user`` is the one exception, and it is not a weakening of that rule:
    the question does not execute anything, so there is no unattended action to
    protect against. Which way it goes is the ASKING AGENT's declaration
    (``timeout_action``), because only the agent knows whether its question has a
    safe default — ``continue`` returns "nobody answered, use the fallback you
    stated", ``halt`` stops the run with the checkpoint intact. Any hardware the
    agent then reaches for is still gated by SafetyGate and its own approval.
    """
    if not store:
        yield ("result", None)
        return
    ev = store.get("events", {}).get(event_id)
    if ev is None:
        yield ("result", None)
        return
    _pending = (store.get("pending") or {}).get(event_id) or {}
    _ask = _pending.get("ask") if isinstance(_pending.get("ask"), dict) else {}
    _timeout_action = str((_ask or {}).get("timeout_action") or "continue").lower()
    _is_ask = kind == "ask_user"
    t0 = _time.time()
    last_beat = t0
    while True:
        if abort is not None and abort.is_set():
            # Drop the entry: an aborted run's pending interrupt can never be
            # resolved (its worker is gone), and leaving it in the store makes
            # the approval panel offer a card whose resolve wakes nobody.
            lock = store.get("lock")
            if lock is not None:
                with lock:
                    store["resolved"].pop(event_id, None)
                    store["pending"].pop(event_id, None)
                    store["events"].pop(event_id, None)
            yield ("result", None)
            return
        if ev.wait(timeout=0.5):
            break
        # Heartbeat — two jobs, both learned the hard way:
        #  * #33 "由于某种原因卡住了，原因不详" — while an approval is pending the
        #    stream emitted NOTHING, so the UI could not say what it was waiting
        #    for. Now it says which interrupt, how long, and how much longer.
        #  * #34 "网络变化也会导致对话卡住" — a silent SSE connection gets reaped by
        #    anything that kills idle TCP (WireGuard keepalive, corporate gateway
        #    idle timeouts, Wi-Fi roaming, sleep). ws.py has pinged every 15 s for
        #    exactly this reason; run-task never did.
        now = _time.time()
        if now - last_beat >= beat:
            last_beat = now
            waited = now - t0
            left = max(0, int(max_wait - waited))
            if _is_ask:
                # Say what will ACTUALLY happen. "按拒绝处理" would be a lie here
                # — there is nothing to reject, and under `continue` the agent
                # carries on with its stated fallback.
                _beat_text = (
                    f"等待用户回答提问 — 已等 {waited:.0f}s，{left}s 后"
                    + ("本次运行将停下等待（进度已保存）"
                       if _timeout_action == "halt"
                       else "智能体将按其说明的保守默认自行继续")
                )
            else:
                _beat_text = (f"等待人工审批 — 已等 {waited:.0f}s，"
                              f"{left}s 后按拒绝处理")
            yield ("beat", {
                "kind": "status",
                "subkind": "awaiting_approval",
                "interrupt_id": event_id,
                "interrupt_kind": kind,
                "elapsed_s": round(waited, 1),
                "remaining_s": float(left),
                "text": _beat_text,
            })
        if _time.time() - t0 > max_wait:
            waited = _time.time() - t0
            logger.warning(
                "run-task HITL: %s %s (kind=%s) timed out after %.0fs — %s",
                "question" if _is_ask else "approval", event_id, kind, waited,
                (f"ask_user timeout_action={_timeout_action}" if _is_ask
                 else "failing CLOSED (reject)"))
            lock = store.get("lock")
            if lock is not None:
                with lock:
                    store["resolved"].pop(event_id, None)
                    store["pending"].pop(event_id, None)
                    store["events"].pop(event_id, None)
            if _is_ask and _timeout_action == "halt":
                msg = (f"提问超时({waited:.0f}s)未收到用户回答 — 本次运行就此停下"
                       "（进度已保存，回答后可继续）。")
            elif _is_ask:
                msg = (f"提问超时({waited:.0f}s)未收到用户回答 — 智能体将按其说明的"
                       "保守默认自行继续。")
            else:
                msg = (f"审批超时({waited:.0f}s)未收到用户裁决 — 已按拒绝处理，"
                       "未执行该动作。")
            # SAY SO (2026-07-28). The timeout emitted no frame at all: the
            # pending card simply vanished from the UI after a heartbeat that had
            # been promising 「Ns 后按拒绝处理」, and the operator was left
            # watching a stream that had quietly decided for them.
            yield ("beat", {
                "kind": "status",
                "subkind": "approval_timeout",
                "interrupt_id": event_id,
                "interrupt_kind": kind,
                "elapsed_s": round(waited, 1),
                "degraded": True,
                "text": msg,
            })
            if kind == "workflow_human":
                # Composite human nodes resume with {"route","note"}; there is no
                # universally-safe route name, so surface the timeout in the note
                # and let the interpreter's own out-of-set guard stop the run.
                yield ("result", {"route": "", "note": msg})
                return
            if _is_ask:
                if _timeout_action == "halt":
                    # No resume value → _drive's "等待人工审批未得到答复" path
                    # ends the run with the checkpoint intact, which is what the
                    # asking agent declared it wanted.
                    yield ("result", None)
                    return
                yield ("result", {"selected": [], "custom_text": "",
                                  "timeout": True, "note": msg})
                return
            yield ("result", {"decisions": [{"type": "reject", "message": msg}]})
            return
    lock = store.get("lock")
    if lock is not None:
        with lock:
            decision = store["resolved"].pop(event_id, None)
            store["pending"].pop(event_id, None)
            store["events"].pop(event_id, None)
    else:  # pragma: no cover
        decision = store["resolved"].pop(event_id, None)
        store["pending"].pop(event_id, None)
        store["events"].pop(event_id, None)
    if decision is None:
        yield ("result", None)
        return
    # workflow_human expects the decision dict ({"route","note"}) and ask_user the
    # answer dict ({"selected","custom_text","note"}) — both resume the SINGLE
    # value straight back into interrupt(). Only the DANGEROUS HITL middleware
    # wants the {"decisions": [...]} envelope.
    yield ("result", decision if kind in ("workflow_human", "ask_user")
           else {"decisions": [decision]})


def resolve_blocking(store, event_id: str, kind: str, abort, *,
                     max_wait: float = MAX_APPROVAL_WAIT_S,
                     beat: float = APPROVAL_BEAT_S):
    """:func:`await_resolution` for callers that cannot forward the heartbeats.

    The SSE run-task path yields every beat straight to the client, which is how
    the operator sees 「已等 Ns」 instead of a dead stream. A callback-shaped
    caller (the private-chat resolver) has nowhere to put them: it is invoked
    from inside the engine's generator, so no frame can leave until it returns.
    Those callers rely on ``api/sse.with_heartbeat`` — an independent pump
    thread — to keep the connection alive, and take only the final value here.

    Returns the resume value, or ``None`` (abort / no decision / a timeout the
    asker declared should stop the run).
    """
    result = None
    for what, payload in await_resolution(store, event_id, kind, abort,
                                          max_wait=max_wait, beat=beat):
        if what == "result":
            result = payload
        else:
            logger.debug("hitl wait beat (%s): %s", event_id, payload.get("text", ""))
    return result


__all__ = [
    "APPROVAL_BEAT_S",
    "MAX_APPROVAL_WAIT_S",
    "await_resolution",
    "extract_hitl_request",
    "publish_interrupts",
    "resolve_blocking",
]
