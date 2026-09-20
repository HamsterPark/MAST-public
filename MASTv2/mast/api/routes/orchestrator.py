"""POST /api/agents/run-task — run the 6-agent orchestrator over SSE.

This is the CORE MAST capability that was unreachable from the API: chat_stream
only does single-agent private chat, whereas this drives the real LangGraph
multi-agent orchestrator graph (supervisor → literature / experiment_design /
instrument_control / data_processing / paper_writing / paper_review) for one
operator task and streams its super-steps as SSE frames.

THIN RELAY + BRIDGE (house rule): orchestration / routing / SafetyGate / verdict
translation all live in the core (the compiled graph + ``_orch_interrupts`` +
``_build_decision``). This module only:

  1. relays onto the live app's already-built ``_orchestrator`` (building it once
     via ``_build_orchestrator`` if a chat-model key is available),
  2. drives ``orchestrator.stream(..., subgraphs=True, stream_mode="updates")``
     and buckets each chunk into per-agent SSE frames,
  3. publishes any ``__interrupt__`` chunk into the live ``_orch_interrupts``
     store and emits an ``interrupt`` SSE frame so the UI resolves it via the
     EXISTING POST /agents/{id}/interrupts/{interrupt_id}/resolve
     (agents_control.py) — we do NOT reimplement resolve, only await + resume,
  4. honours the shared ``_orch_abort`` Event between super-steps.

SSE bridging mirrors chat_stream.py: a SYNC generator (the graph stream blocks)
is handed to Starlette's StreamingResponse, which runs it on a threadpool so the
blocking ``.stream()`` never stalls the event loop. Blocking-in-the-worker is
allowed here (the no-sleep/no-block invariant constrains agents/**/graph.py
nodes, NOT this bridge), so awaiting a HITL resolution by polling the Event is
fine.

GRACEFUL DEGRADATION: boots STANDALONE. No live app, no built orchestrator, or
any relay raising ⇒ a SINGLE error/degraded SSE frame then ``done`` (streaming),
or a typed ``degraded=True`` body (abort) — NEVER a 500. Heavy backends
(langgraph / langchain_core) are lazy-imported inside ``try`` so the module
imports with nothing heavy installed.

Frames: see schemas_orchestrator.py module docstring.
"""

from __future__ import annotations

import contextlib as _contextlib
import json
import logging
import re
import threading
import time as _time
import uuid as _uuid
from typing import Any

# A held agent auto-resumes after this long so a forgotten hold (no release, no
# abort) can never hang the SSE stream / worker thread forever.
_MAX_HOLD_S = 600.0

# How long a published HITL approval waits for an operator verdict before the
# run gives up. Same order as _MAX_HOLD_S and for the same reason: nothing that
# blocks a stream worker may block it forever. Times out fail-CLOSED (reject) —
# see _await_resolution. .
_MAX_APPROVAL_WAIT_S = 900.0

# How often the approval wait emits a status frame. Doubles as the SSE keep-alive
# for this stream: a silent connection is what idle-TCP reapers (WireGuard
# keepalive, gateway idle timeouts, Wi-Fi roaming, sleep) kill — which is how a
# network blip turned into "对话卡住" . ws.py pings every 15 s for the same
# reason; matching it here.
_APPROVAL_BEAT_S = 15.0

# How long a new run on the SAME group conversation waits for an already-
# aborting previous stream to wind down before bouncing the operator (the old
# stream honours abort at super-step boundaries, so most LLM steps clear well
# inside this window; a long hardware wait may not — then we re-signal abort
# and ask the operator to retry shortly). Feedback 2026-07-10 #41.
_ABORT_HANDOVER_GRACE_S = 12.0

# LangGraph super-step budget for one orchestrator run (counts EVERY step incl.
# each agent's inner ReAct step). Surfaced to the UI (start frame `step_limit` +
# per-message `step`) so the operator always sees how close the run is to it.
#
# 500 → 250 (2026-07-30). This was the ONLY effective bound on one run's cost for
# months, because the USD gate below was seeded by nobody — and at the measured
# ~$0.20 per instrument turn, 500 super-steps is about $50. The ledger says a
# legitimate multi-stage pipeline uses ~50 steps and the worst run ever recorded
# (the $30.66 runaway) reached 304, so 250 sits above every legitimate run
# observed while cutting the theoretical worst case in half. It is a BACKSTOP, not
# a quota: the real ceiling is now the USD gate, which is finally wired.
_RECURSION_LIMIT = 250

#: Lowest value an operator may pin ``orchestrator_recursion_limit`` to. Not a
#: taste call: below roughly 11 × (a few rounds) an agent cannot finish a
#: multi-step instruction NOR reach the guards that would end its turn readably.
#: See ``_effective_recursion_limit`` and ``agents._shared.call_limits``.
_MIN_USEFUL_RECURSION_LIMIT = 150

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from mast.api.schemas_orchestrator import (
    AbortTaskResponse,
    BackgroundAbortResponse,
    BackgroundRunInfo,
    BackgroundRunsResponse,
    BackgroundSpawnRequest,
    BackgroundSpawnResponse,
    GroupConversation,
    GroupConversationsResponse,
    RunTaskRequest,
    TranscriptEntry,
    TranscriptResponse,
)
from mast.api import hitl_bridge as _bridge
from mast.api.sse import SSE_HEADERS, with_heartbeat
from mast.api.tool_narration import (
    narrate_tool_call,
    narrate_tool_result,
    summarize_args,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["orchestrator"])


class _AnyAbort:
    """A stop signal that fires when ANY of its member Events fires.

    A run's stream must stop for two different reasons that must NOT be the
    same Event: the GLOBAL emergency latch (E_STOP, watchdog, environment
    alarm — stops everything, everywhere) and THIS run's own stop button.
    Sharing one process-level Event for both meant aborting one of two
    concurrent runs stopped the other as well, and starting a run re-armed a
    composite that an E_STOP had just halted in a different chain
    (审计 致命一(b) + 严重级「并发 run 无互斥」).

    Duck-types ``threading.Event`` for the three methods this module uses.
    """

    __slots__ = ("_all", "_own")

    def __init__(self, *, global_event=None, run_event=None):
        self._all = [e for e in (global_event, run_event) if e is not None]
        # Setting the union means "stop THIS run" — it must never latch the
        # global emergency, or an ordinary abort would masquerade as an E-STOP
        # and freeze the other chains too. Degrades to the global one only when
        # there is no run-scoped Event at all.
        self._own = [run_event] if run_event is not None else list(self._all)

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._all)

    def set(self) -> None:
        for e in self._own:
            e.set()

    def clear(self) -> None:
        for e in self._own:
            e.clear()


def _live_app(ctx: Any):
    """Best-effort handle to the running app (None in standalone). Never raises."""
    return getattr(ctx, "live_app", None) or getattr(ctx, "app", None)


def _current_experiment_id() -> "str | None":
    """Best-effort current experiment id — a PROVENANCE tag for a new group chat
    (which experiment it was born in). Conversations are NOT scoped to it: the
    lists never filter by experiment_id, so a 群聊 still spans experiments. Never
    raises (None in standalone / no active experiment)."""
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        return getattr(log, "current_experiment_id", None) if log else None
    except Exception:
        return None


def _current_sample_id() -> "str | None":
    """Best-effort current sample id — the MIDDLE level of the hierarchy the
    operator actually works in: 一个实验若干样品，一个样品若干群聊. Without it every 群聊 hung straight off the experiment, so
    the sample level did not exist in the data and could not exist in the UI.

    Same contract as :func:`_current_experiment_id`: a where-it-started tag, not
    a lease — a 群聊 outlives its sample and is never filtered away by it. Never
    raises (None when no sample is active, which stays a perfectly valid chat)."""
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        return getattr(log, "current_sample_id", None) if log else None
    except Exception:
        return None


def _resume_lead_messages(app, instruction: str) -> list:
    """Leading messages to inject BEFORE the operator's instruction so the
    supervisor resumes with context instead of dead-ending on 「No prior context
    or ongoing task found」 (2026-07 analysis ⑥/⑦).

    Two independent, best-effort blocks:
      * ⑥ when the instruction reads as "继续" AND an unfinished experiment exists,
        a compact description of that experiment (id / name / goal / current
        sample / recent actions / active plan phase) — so the supervisor resumes
        the real run rather than re-introducing itself or spawning a duplicate.
      * ⑦ any operator answer to an agent→user 心愿单 request not yet read back —
        surfaced automatically so a BLOCKED agent that stopped gets the reply on
        this run without having to poll ``check_my_requests``. The answers are
        marked delivered here so each is handed over exactly once.

    Returns a (possibly empty) list of HumanMessages. Never raises — a
    context-injection glitch must not break the run."""
    lead: list = []
    try:
        from langchain_core.messages import HumanMessage
        from mast.agents._shared.resume_context import (
            build_experiment_resume_block,
            build_pending_reply_hint,
            is_resume_intent,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("resume-context import failed: %s", exc)
        return lead

    # ⑥ resume experiment context — only on an explicit continuation intent, so a
    # genuinely new task in a new conversation is not polluted with a stale run.
    try:
        if is_resume_intent(instruction):
            block = build_experiment_resume_block(
                getattr(app, "_experiment_log", None),
                getattr(app, "_storage", None),
                getattr(app, "_plan_store", None),
            )
            if block:
                lead.append(HumanMessage(content=block))
                logger.info("run-task: injected experiment resume context (继续 intent)")
    except Exception as exc:  # noqa: BLE001
        logger.debug("resume experiment block failed: %s", exc)

    # ⑦ operator replies — a SHORT, READ-ONLY hint so the supervisor routes to the
    # agent that will consume them. The full path/note is delivered to that agent
    # by RequestReplyReadbackMiddleware (before_model), which also marks them
    # delivered — so we do NOT mark anything here (no double-delivery / no race).
    try:
        from mast.wishlist import get_board

        hint = build_pending_reply_hint(get_board())
        if hint:
            lead.append(HumanMessage(content=hint))
            logger.info("run-task: injected pending operator-reply routing hint")
    except Exception as exc:  # noqa: BLE001
        logger.debug("pending reply hint failed: %s", exc)

    return lead


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


#: 进度的**运行时无关**表示（事件协议 v2，2026-08-26）。
#:
#: 今天前端读的是 ``step`` / ``step_limit``，语义是「super-step 数 / recursion
#: limit」——那是 LangGraph 的记账单位，而且它与「这一回合还能干几件事」之间的换算率
#: **随中间件数量漂移**：同一个字面量 50 在 2026-06 买到约 9 次工具调用，2026-08 只买
#: 到 3 次，而没有任何一次 diff 里出现过「限额被改小」（KNOWN_ISSUES §2.18）。
#:
#: 退出 LangGraph 之后不存在 super-step，那两个字段没有对应物。所以在它们**旁边**加
#: 一组不绑定实现的字段，并把单位显式写出来：旧引擎诚实地报 ``super_step``，新引擎
#: 报 ``model_call``。前端因此只改一次、两代同协议。
#:
#: 原则是**加字段不改字段** —— ``step`` / ``step_limit`` 一个字节都没动，老客户端
#: 照常工作。
_PROGRESS_UNIT_LEGACY = "super_step"


def _progress(step: int, limit: int) -> dict:
    """``{progress, progress_limit, progress_unit}``，供 SSE 帧展开。"""
    return {"progress": int(step or 0), "progress_limit": int(limit or 0),
            "progress_unit": _PROGRESS_UNIT_LEGACY}


def _v2_group_enabled(app) -> bool:
    """``engine_v2_group_chat`` —— 退出 LangGraph 的第四个切换面。

    DEFAULT OFF / 每次 run 读一次 / fail-safe OFF。这是最大的切换面，也是唯一一个
    会碰硬件的 —— 但 v2 路径**开跑前就拒绝**把 instrument_control 放进目标名单
    （它的安全中间件还没有 v2 装配路径，见 ``agentruntime/sse.drive_group_run``），
    所以打开它只影响不碰仪器的任务。
    """
    try:
        rt = getattr(app, "_settings", None) or getattr(
            getattr(app, "state", None), "settings", None)
        if rt is None:
            from mast.webui.settings_store import settings_store_for_runtime
            rt = settings_store_for_runtime()
        return bool(rt.get("engine_v2_group_chat"))
    except Exception:  # noqa: BLE001 — 读不到开关 = 旧路径
        return False


def _drive_v2(app, instruction: str, task_id: str, targets, persist, abort,
              pending=None):
    """v2 群聊：把 :func:`agentruntime.sse.drive_group_run` 的帧转出来。

    这一层刻意很薄 —— 帧的翻译住在 ``agentruntime/sse.py``，因为那段**本来就该是
    运行时的一部分**：旧桥那 900 行之所以那么长，是因为它在从 chunk 里**推断**
    agent 名字、去重、以及「分支死没死」。事件流自己说得清的东西，不该在消费端
    重新猜一遍。

    最后 yield 一个带下划线前缀的「内部帧」把结局交回去（``_final_text`` /
    ``_completed`` / ``_stop_reason``）——调用方按它自己的契约发收尾帧，因为
    `done` 帧的字段组合是那一层的事。

    ``pending`` 是 :class:`agentruntime.sse.PendingFrames`：阻塞期间帧的**第二条
    出口**。这个生成器自己排空它，``with_heartbeat`` 的 ``on_idle`` 也排空它 ——
    同一把锁，取走即清空，所以一条帧只会出去一次。不给的话行为退回从前（帧只在
    编排器步与步之间出得去，工具帧因此迟到一整次工具调用）。
    """
    from mast.agentruntime.sse import drive_group_run

    router = _orchestrator_router_model(app)
    # ★ 提问通道（2026-08-27）。不接的话 ``ask_user`` 会把「问不出去」转成一句提示，
    #   这一轮以 ``final`` 正常收场，而**用户永远看不到那个问题** —— 实测出来的，
    #   而且它不在任何一条闸门的视野里。用与旧桥**同一个** store。
    hitl_store = getattr(app, "_orch_interrupts", None)
    if not hitl_store:
        logger.warning("v2 群聊：核心没有 _orch_interrupts，本次运行问不了用户")
    gen = drive_group_run(instruction=instruction, agents=list(targets),
                          router_model=router, persist=persist, abort=abort,
                          build_loop=_v2_loop_factory(app),
                          hitl_store=hitl_store, thread_id=f"task:{task_id}",
                          pending=pending)
    while True:
        try:
            yield next(gen)
        except StopIteration as stop:
            result = stop.value
            break
    yield {"_final_text": result.final_text or "（本次运行没有产出文本）",
           "_completed": result.outcome == "completed",
           "_stop_reason": result.stop_reason}


def _v2_group_targets(app=None) -> list[str]:
    """v2 群聊可派发的 agent 名单。

    **从 ``_shared/roster.AGENT_NAMES`` 派生**，不在这里另抄一份 —— 「名单抄第二遍
    就会漂」是本仓记过的事故形状（后台可后台化名单漂过一次）。

    2026-08-27：名单的来源从 ``orchestrator/graph._AGENT_NAMES`` 换成了
    ``_shared/roster``。同一份数据，但**不再需要 import 一个即将被删的图模块** ——
    「有哪几个 agent」是系统的组成，不是某个执行引擎的实现细节。

    ``instrument_control`` **只有在核心能装配它时**才进名单。判据不是「有人觉得它
    可以了」，而是 ``CoreRuntime._build_instrument_loop_v2`` 在不在 —— 那个方法里
    少任何一件安全件都会抛。核心没接线（独立 API 进程、测试替身）就把它去掉。
    """
    from mast.agents._shared.roster import AGENT_NAMES
    from mast.chat.engine_v2 import HARDWARE_AGENTS

    can_build_ic = callable(getattr(app, "_build_instrument_loop_v2", None))
    return [a for a in AGENT_NAMES
            if a not in HARDWARE_AGENTS or can_build_ic]


def _v2_loop_factory(app):
    """``agent_id -> AgentLoop``：仪器 agent 走核心的专门装配，其余走通用装配。

    核心没接线时对仪器 agent 返回 None 是**不行**的 —— 那会变成一个装配不出来的
    循环在 run 中途炸。所以这里不做兜底：``_v2_group_targets`` 已经在名单层面把它
    去掉了，能走到这里就说明核心接得上。
    """
    from mast.agentruntime.assembly import build_agent_loop
    from mast.chat.engine_v2 import HARDWARE_AGENTS

    def _build(agent_id: str):
        if agent_id in HARDWARE_AGENTS:
            return app._build_instrument_loop_v2()
        return build_agent_loop(agent_id, buf=None)
    return _build


def _orchestrator_router_model(app):
    """路由用的 chat model —— **向核心要，不自己在核心身上挂缓存**。

    第一版把缓存写成 ``getattr(app, "_v2_router_model", None)``，
    ``test_api_layer_never_reaches_for_a_nonexistent_core_attribute`` 当场变红：
    那道闸门守的是「API 中继向核心要一个它没有的东西，会永远静默降级 —— UI 照渲染、
    请求照成功，只有**效果**不见了」（那正是 HITL 审批坏掉五周的形状）。

    缓存一个模型是核心的事，所以它现在是 ``CoreRuntime.v2_router_model()``。
    """
    fn = getattr(app, "v2_router_model", None)
    if callable(fn):
        model = fn()
        if model is not None:
            return model
    # 核心没接线（独立 API 进程、测试替身）→ 自己建一个，不缓存。
    from mast.agents._shared.models import make_chat_model

    return make_chat_model("orchestrator")


#: Longest terminal label persisted. The row is a marker, not a log line — the
#: full text also went to the client as an error frame and to the service log.
_TERMINAL_LABEL_MAX = 220

#: Message text limits. The persisted row is the only lasting record of a tool
#: return (see _clip's note); the SSE frame only has to render.
_PERSIST_TEXT_MAX = 200_000
_SSE_TEXT_MAX = 8_000


#: A transcript row's structured sidecar. Capped so one pathological tool call
#: cannot bloat the shared conversations DB; over the cap the row keeps its
#: summary and simply has nothing to expand, which is visible rather than silent.
_META_JSON_MAX = 60_000

#: How much of a digested tool RETURN is kept for the expander. Deliberately the
#: SAME 8000 that ``ConversationStore.append_message`` already clips ``text`` to:
#: the durable record must not hold LESS than it did before the row's text became
#: a summary. (``_PERSIST_TEXT_MAX`` is 200k, but the store has always trimmed to
#: 8000 on the way in, so 8000 is what "the full text" has actually meant here.)
#: ``_clip`` writes the truncation into the string itself, so an expander showing
#: a trimmed payload says that it is trimmed.
_META_DETAIL_MAX = 8_000


def _meta_json(meta: dict) -> str:
    """Serialise a transcript row's ``meta`` sidecar. Never raises — a sidecar
    that fails to serialise costs an expandable panel, not the run."""
    try:
        text = json.dumps(meta, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.debug("run-task: meta serialisation failed: %s", exc)
        return ""
    return text if len(text) <= _META_JSON_MAX else ""


def _clip(text: str, limit: int) -> str:
    """Truncate to ``limit``, and SAY SO when it happens.

    The old code did ``txt[:2000]`` in both the SSE frame and the persisted row,
    with nothing marking the cut. 2026-07-27 forensics: 68 of 340 stored
    messages sat exactly at that cap, including experiment_design's 28-step
    ExperimentPlan, which ends mid-JSON inside step 3::

        {"skill_name": "FullScan", "params": {"center_x_m": 0.0, "width_m": 5e-7, "he

    Nothing downstream could tell that from a complete plan. Combined with an
    empty ``action.data`` column and no plan store at all, the full text of
    every batch-scan return, STS trace and experiment plan existed nowhere but
    in those first 2000 characters.

    Raising the persisted cap alone would not be enough — a silent cut at
    200 000 characters is the same defect further out. The marker is the fix;
    the larger cap just means it almost never fires.
    """
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return (text[:limit] +
            f"\n…[内容过长，此处截断，省略 {dropped} 字符（完整 {len(text)} 字符）]")


def _terminal_label(completed: bool, stop_reason: "str | None") -> str:
    """Persist a terminal marker that distinguishes completion, error and abort.

    A backend exception or unanswered approval must not be replayed as an
    operator-requested abort. The stored label must reflect the actual cause.
    """
    if completed:
        return "完成"
    if not stop_reason:
        return "已中止"                       # genuine operator abort
    reason = " ".join(str(stop_reason).split())
    if len(reason) > _TERMINAL_LABEL_MAX:
        reason = reason[:_TERMINAL_LABEL_MAX - 1] + "…"
    return f"运行出错，已停止：{reason}"


# ── per-agent bucketing helpers (pure; mirror gui/app.py classifiers) ─────────
def _agent_from_namespace(ns) -> "str | None":
    """Resolve the agent id that owns a subgraph stream namespace.

    With ``subgraphs=True`` each chunk is ``(namespace, update)`` where namespace
    is a tuple like ``("instrument_control:<uuid>",)`` for a subgraph step, or
    ``()`` for the top-level orchestrator (supervisor). Returns the agent id, or
    None for top-level updates (caller falls back to the inner node name)."""
    if not ns:
        return None
    head = ns[0] if isinstance(ns, (tuple, list)) else str(ns)
    agent = str(head).split(":", 1)[0]
    return agent or None


def _normalize_node(node_name: str) -> str:
    """Map a LangGraph node name to an Agents-UI agent id."""
    if node_name in ("supervisor", "__start__", "__end__"):
        return "_supervisor"
    return node_name


# The supervisor narrates every dispatch as `[SUPERVISOR → a ‖ b] reason`; that
# note is the AUTHORITATIVE record of which agents were just set running (with
# parallel fan-out the interleaved subgraph chunks are not — they only say who
# happened to emit last). "‖" is the multi-target separator the supervisor uses.
_DISPATCH_RE = re.compile(r"\[SUPERVISOR\s*→\s*([^\]]+)\]")


def _parse_dispatch_targets(msgs) -> "list[str] | None":
    """Agent ids the supervisor just dispatched to, from its own route note.

    Returns None when these messages contain no dispatch note (so the caller can
    tell "no dispatch here" from "dispatched to nobody"), and [] for an END."""
    for m in reversed(list(msgs or [])):
        hit = _DISPATCH_RE.search(_msg_text(m))
        if not hit:
            continue
        raw = hit.group(1)
        targets = [t.strip() for t in raw.split("‖")]
        return [t for t in targets if t and t != "__end__"]
    return None


# Must MATCH mast.agents.orchestrator.graph._AUTO_BG_MARKER — the supervisor emits
# this marker line to signal which agents to run as DETACHED background runs
# (conservative auto-background). Kept as a literal (not an import) so this bridge
# stays free of a heavy graph import; a parity test pins the two together.
_AUTO_BG_MARKER = "[SUPERVISOR::AUTO_BACKGROUND]"


def _parse_auto_background(text: str) -> list[str]:
    """Agent ids named in an auto-background marker line, else []. Format:
    ``[SUPERVISOR::AUTO_BACKGROUND] literature,paper_writing :: <reason>``."""
    if not text.startswith(_AUTO_BG_MARKER):
        return []
    body = text[len(_AUTO_BG_MARKER):].split("::", 1)[0].strip()
    return [a.strip() for a in body.split(",") if a.strip()]


# Must MATCH mast.agents._shared.compaction_mw.COMPACTION_META_KEY — the context
# compaction middleware stamps its facts under this key on the summary message it
# substitutes for the compacted history. A literal (not an import) for the same
# reason as _AUTO_BG_MARKER above: this bridge stays free of heavy agent imports.
# A parity test pins the two spellings together.
_COMPACTION_META_KEY = "mast_compaction"


def _compaction_event(msgs) -> "dict | None":
    """The compaction facts carried by this state update, or None.

    (2026-07-27)「群聊内的压缩功能并未显性体现」. Compaction
    was always running in 群聊 — ``_chat_agent_middleware`` attaches it to every
    orchestrator agent — but the update it emits is
    ``[RemoveMessage(ALL), summary, *preserved]``, and this bridge classified the
    summary as a HumanMessage ("operator echo — already in `start`") and dropped
    it. So the operator's history was silently rewritten mid-conversation: scroll
    back far enough and the earlier turns are simply gone, with nothing saying a
    machine removed them.
    """
    for m in (msgs or []):
        ak = getattr(m, "additional_kwargs", None)
        if isinstance(ak, dict):
            event = ak.get(_COMPACTION_META_KEY)
            if isinstance(event, dict):
                return event
    return None


def _compaction_line(event: dict) -> str:
    """One operator-facing line for a compaction. Says only what the event
    actually carries: a compaction whose counts didn't come through is reported
    as a compaction with no counts, NEVER with a plausible-looking number.

    Two DIFFERENT things reach here and they must not be described alike:

      * an agent-internal *summarisation* — older turns replaced by an LLM-written
        summary, inside one agent's subgraph;
      * the supervisor's parent-channel *prune* (2026-07-29) — older routing and
        handoff notes DELETED outright, no summary, no model call.

    Calling the second one "已被摘要替代" would tell an operator a summary exists
    that they could go read. It does not.
    """
    removed = event.get("removed")
    kept = event.get("kept")
    if event.get("mode") == "parent_prune":
        head = ("群聊主线整理：较早的路由/交接消息已移除"
                if not isinstance(removed, int) or removed <= 0
                else f"群聊主线整理：较早的 {removed} 条路由/交接消息已移除")
        if isinstance(kept, int) and kept > 0:
            head += f"，最近 {kept} 条保留"
        return head + "（没有摘要，也没有影响各智能体的产物）"
    if isinstance(removed, int) and removed > 0 and isinstance(kept, int):
        return (f"上下文压缩：较早的 {removed} 条消息已被摘要替代，"
                f"最近 {kept} 条保留原文")
    if isinstance(removed, int) and removed > 0:
        return f"上下文压缩：较早的 {removed} 条消息已被摘要替代"
    return "上下文压缩：此处较早的对话已被摘要替代"


def _msg_text(m) -> str:
    txt = getattr(m, "content", "") or ""
    if isinstance(txt, list):
        txt = " ".join(str(x) for x in txt)
    return str(txt)


def _msg_signature(m) -> str:
    """Stable content signature for dedup across subgraph re-emits (LangGraph
    re-assigns ids to messages echoed up from a subgraph)."""
    cls_name = m.__class__.__name__
    text = _msg_text(m).strip()
    tcs = getattr(m, "tool_calls", None) or []
    parts = []
    for tc in tcs:
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("args", {})
        else:
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", {})
        try:
            args_repr = repr(sorted((args or {}).items()))[:160]
        except Exception:
            args_repr = str(args)[:160]
        parts.append(f"{name}({args_repr})")
    tcid = getattr(m, "tool_call_id", "") or ""
    return f"{cls_name}|{tcid}|{';'.join(parts)}|{text[:240]}"


def _extract_hitl_request(interrupt_obj):
    """Re-export — the implementation moved to :mod:`mast.api.hitl_bridge`."""
    return _bridge.extract_hitl_request(interrupt_obj)


def _publish_interrupt(app, ns, chunk_value, thread_id) -> list[dict]:
    """Publish this chunk's interrupt(s) into the live store, run-task flavour.

    Thin wrapper over :func:`mast.api.hitl_bridge.publish_interrupts`: resolves
    the store off the live app and maps the LangGraph namespace to an owning
    agent. The shared implementation takes both directly so the private-chat
    entry point (which has an agent id and no namespace) can use it too.
    """
    store = getattr(app, "_orch_interrupts", None)
    owner = _agent_from_namespace(ns) or "instrument_control"
    return _bridge.publish_interrupts(store, owner, chunk_value, thread_id)


def _await_resolution(app, event_id: str, kind: str, abort):
    """Wait for a verdict, yielding heartbeat frames — run-task flavour.

    The module-level timings are passed EXPLICITLY rather than left to the
    shared defaults so this stream's constants stay the ones that govern it (and
    stay patchable in tests) even though the loop itself now lives elsewhere.
    """
    store = getattr(app, "_orch_interrupts", None)
    yield from _bridge.await_resolution(
        store, event_id, kind, abort,
        max_wait=_MAX_APPROVAL_WAIT_S, beat=_APPROVAL_BEAT_S,
    )


def _ensure_orchestrator(app) -> "tuple[Any, str]":
    """Return (orchestrator, backend_note). Builds the graph once if a chat-model
    key is available. Returns (None, reason) when it can't be built. Never raises.
    """
    orch = getattr(app, "_orchestrator", None)
    if orch is not None:
        return orch, "orchestrator"
    builder = getattr(app, "_build_orchestrator", None)
    if not callable(builder):
        return None, "orchestrator not available (no builder)"
    try:
        builder()
    except Exception as exc:  # graph build needs a chat-model key etc.
        logger.warning("run-task: orchestrator build failed: %s", exc)
        return None, f"orchestrator build failed: {type(exc).__name__}"
    orch = getattr(app, "_orchestrator", None)
    if orch is None:
        return None, "orchestrator unavailable (no chat-model key?)"
    return orch, "orchestrator"



def _honor_holds(st, agent_id: str, abort):
    """Pause the run while *agent_id* (or ``__all__``) is held by the operator
    (POST /agents/{id}/hold). Yields a 'status' held frame, polls until released
    or aborted, then a 'resumed' frame — mirrors the old gui ``_wait_while_held``.
    Exploits ``.stream()`` laziness: pausing here (before the next chunk is pulled
    in the caller's loop) pauses the whole graph at the agent boundary. Bridge-
    thread blocking is allowed (the no-sleep invariant constrains agents/**/graph.py
    nodes, not this relay)."""
    if not st:
        return
    lock = st.get("lock")

    def _is_held() -> bool:
        # single-key dict reads are GIL-atomic; the lock makes the two-key read a
        # consistent snapshot vs a concurrent hold/release write.
        with (lock or _contextlib.nullcontext()):
            holds = st.get("holds") or {}
            return bool(holds.get(agent_id) or holds.get("__all__"))

    if not _is_held():
        return
    task = st.get("task") if isinstance(st.get("task"), dict) else None
    t0 = _time.time()
    if task is not None:
        task["active_agent_id"] = agent_id
        task.setdefault("events", []).append(
            {"t": t0, "kind": "held", "text": f"{agent_id} 已被用户暂停 — 等待恢复"})
    yield _sse({"kind": "status", "agent": agent_id, "held": True,
                "text": f"{agent_id} 已被用户暂停 — 等待恢复"})
    timed_out = False
    while _is_held():
        if abort is not None and abort.is_set():
            return
        if _time.time() - t0 > _MAX_HOLD_S:  # forgotten hold → auto-resume
            timed_out = True
            break
        _time.sleep(0.25)
    waited = _time.time() - t0
    if timed_out:
        # CLEAR THE FLAG (2026-07-28). The timeout used to just `break`, leaving
        # holds[agent_id] set — so the very next super-step hit this function
        # again and parked another 600 s. What looked like "auto-resume after a
        # forgotten hold" was really "let one super-step through every ten
        # minutes", for as long as the run lasted. The snapshot also kept
        # reporting held=True while this frame said the agent had resumed, so
        # the two halves of the UI contradicted each other.
        with (lock or _contextlib.nullcontext()):
            holds = st.get("holds")
            if isinstance(holds, dict):
                for key in (agent_id, "__all__"):
                    if holds.get(key):
                        holds[key] = False
        logger.warning(
            "run-task: hold on %s timed out after %.0fs — flag cleared and the "
            "run resumed (it was silently re-arming every super-step before)",
            agent_id, waited)
        try:
            from mast.core.diagnostics import record as _diag

            _diag("hold_timeout", agent_id,
                  "用户暂停超时，已自动恢复并清除暂停标志", waited_s=round(waited, 1))
        except Exception:  # noqa: BLE001
            pass
    note = (f"{agent_id} hold 超时自动恢复 ({waited:.0f}s)" if timed_out
            else f"{agent_id} 已恢复 (暂停 {waited:.1f}s)")
    if task is not None:
        task.setdefault("events", []).append(
            {"t": _time.time(), "kind": "resumed", "text": note})
    yield _sse({"kind": "status", "agent": agent_id, "held": False, "text": note})


def _effective_recursion_limit(app) -> int:
    """Per-run super-step budget for run-task, read live from settings.

    Counts EVERY super-step (supervisor hop + each agent inner ReAct step), so a
    full autonomous campaign (survey -> image -> assess -> STS across 6 agents)
    needs ample headroom. Default 250 (was 500 — see the constant); operator
    retunes via the ``orchestrator_recursion_limit`` setting.

    FLOOR (2026-08-04): 50 → ``_MIN_USEFUL_RECURSION_LIMIT``. A subgraph inherits
    this number BY VALUE and spends it on its own counter (verified: a child
    needing 20 inner steps fails under a parent limit of 10 and passes at 25), so
    the floor is the budget each agent gets to itself — and one model→tool round
    on the IC graph costs ELEVEN super-steps, not one, because every
    ``before_model``/``after_model`` middleware is its own node. A floor of 50
    therefore let an operator pin every agent to 3 tool calls per invocation,
    which is the identical trap that made the private chat fail a five-step
    request (see ``agents._shared.call_limits``). 150 keeps ~13 rounds, enough
    for StallGuard's nudge→nudge→stop ladder to complete."""
    try:
        st = getattr(app, "_settings", None)
        v = st.get("orchestrator_recursion_limit") if st is not None else None
        if v is not None:
            return max(_MIN_USEFUL_RECURSION_LIMIT, int(v))
    except Exception:
        pass
    return _RECURSION_LIMIT


#: Per-run USD ceiling when the operator has not set one.
#:
#: Raised 8 → 80 on operator instruction (2026-07-30). The original was 1.5x the
#: measured legitimate worst case ($5.19 for a 4-agent run with one revision cycle),
#: which is a tight fit around observed history — and history so far is six days of
#: mostly short runs. A ceiling tuned that closely turns a long autonomous campaign
#: into a mid-thought abort, and THAT failure is both more likely and more annoying
#: than an expensive run: the runaway this guards against was $30.66, so 80 still
#: catches it, just without second-guessing legitimate work.
#:
#: It is a BACKSTOP, not a quota. Retune from the 待唤醒 / 设置 page as real
#: campaign costs come in.
_RUN_BUDGET_USD = 80.0


def _effective_run_budget_usd(app) -> float:
    """Per-run USD ceiling, read live from settings. 0.0 = gate OFF.

    A NEGATIVE or unparseable setting is treated as OFF rather than clamped to
    something small: silently inventing a tiny ceiling would end every run
    instantly and look like a broken orchestrator, whereas OFF is the honest,
    pre-existing behaviour of this gate.
    """
    try:
        st = getattr(app, "_settings", None)
        v = st.get("orchestrator_run_budget_usd") if st is not None else None
        if v is not None:
            f = float(v)
            return f if f > 0 else 0.0
    except Exception:  # noqa: BLE001
        pass
    return _RUN_BUDGET_USD


def _park_board_seed() -> dict:
    """Open parks for the active experiment, as the run's ``pending_activations`` cache.

    Read-only and fail-safe to ``{}``: a board that cannot be read means this run
    re-asks a parked agent, which costs a model call. A board read that RAISED would
    mean the run does not start at all — a far worse trade for a cache.
    """
    try:
        from mast.core.park_board import board

        return board().as_state_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("park board seed unavailable: %s", exc)
        return {}


def _run_task_stream(app, instruction: str, task_id: str, conversation_id: str = "",
                     *, turn_holder: dict | None = None, pending=None,
                     goal: dict | None = None):
    """SYNC generator yielding SSE strings for one orchestrator run.

    Pure relay/bridge: drives ``app._orchestrator.stream(...)`` and buckets into
    SSE frames, publishing HITL ``__interrupt__`` chunks into ``app._orch_interrupts``
    (drained by the agents_control resolve endpoint) and honouring ``app._orch_abort``.
    Degrades to a single error frame + done — never raises. Exposed at module level
    so it can be driven directly off the request thread (Starlette wraps it in a
    threadpool via StreamingResponse). ``conversation_id`` resumes a GROUP thread."""
    def gen():
        if not instruction:
            yield _sse({"kind": "error", "message": "缺少 'task' 字段", "degraded": True})
            yield _sse({"kind": "done", "final_text": "", "aborted": False})
            return
        if app is None:
            yield _sse({"kind": "error", "message": "编排器未接入（独立开发模式）",
                        "degraded": True})
            yield _sse({"kind": "done", "final_text": "", "aborted": False})
            return

        orchestrator, backend = _ensure_orchestrator(app)
        if orchestrator is None:
            yield _sse({"kind": "error", "message": backend, "degraded": True})
            yield _sse({"kind": "done", "final_text": "", "aborted": False})
            return

        # Resolve a durable thread_id AND a durable GROUP conversation. Resume the
        # given group conversation's checkpointed thread when asked; else create a
        # fresh group row bound to this run's thread. The durable conversation is
        # what lets a 群聊 survive a tab switch / reload (the TS client reconnects
        # and re-reads the transcript) and what makes each agent's 群聊 messages
        # readable from its per-agent view — the persistence that the run-task path
        # had dropped (it streamed ephemerally and indexed nothing).
        thread_id = f"agents-{task_id}"
        conv_id_in = (conversation_id or "").strip()
        conv_store = getattr(app, "_conv_store", None)
        conv_id_out = ""
        if conv_store is not None:
            try:
                resolved = conv_store.get(conv_id_in) if conv_id_in else None
                if (resolved and resolved.get("kind") == "group"
                        and resolved.get("thread_id")):
                    thread_id = resolved["thread_id"]
                    conv_id_out = conv_id_in
                else:
                    # No (or unknown / non-group) id → fresh durable group row,
                    # filed under the experiment AND the sample it was started on
                    # (#17: 实验 → 样品 → 群聊). Tags only — the chat is never
                    # scoped/filtered by them, so it still spans experiments.
                    row = conv_store.create(
                        "_supervisor", kind="group",
                        title=(instruction[:40] or "新任务"), thread_id=thread_id,
                        experiment_id=_current_experiment_id(),
                        sample_id=_current_sample_id())
                    conv_id_out = row.get("conversation_id", "") or ""
            except Exception as exc:
                logger.debug("run-task: group conversation resolve/create failed: %s",
                             exc)

        def _persist(kind: str, *, agent: str = "", role: str = "",
                     text: str = "", t: "float | None" = None,
                     meta: str = "") -> None:
            """Append one rendered entry to the durable transcript. Best-effort:
            a transcript write must NEVER break the live stream (degrade-safe).

            ``meta`` is the JSON sidecar for rows whose ``text`` is a summary
            (tool calls and JSON tool returns, #12) — it holds what the summary
            left out, so a shorter transcript is never a lossier one."""
            if not conv_id_out or conv_store is None:
                return
            try:
                conv_store.append_message(conv_id_out, kind=kind, agent_id=agent,
                                          role=role, text=text, t=t, meta=meta)
            except Exception as exc:  # noqa: BLE001
                logger.debug("run-task: transcript append failed: %s", exc)

        # 阻塞期间帧的**第二条出口**（2026-08-27）。盒子由**端点**建好传进来 ——
        # 不能在这里建：这是个生成器函数，函数体要到第一次 next() 才跑，而
        # ``with_heartbeat(on_idle=…)`` 在那之前就要拿到排空器。兜底只为直接调用这个
        # 生成器的调用方（测试、以及将来别的驱动方）：它们没有时钟，帧照旧只在编排器
        # 步与步之间出得去 —— 能跑，但工具帧会迟到一整次工具调用。
        from mast.agentruntime.sse import PendingFrames as _PendingFrames

        _v2_pending = pending if pending is not None else _PendingFrames()

        # Concurrency guard: refuse to open a SECOND stream on a group conversation
        # a live run is ALREADY streaming into (reload-then-send, or two tabs).
        # Two streams driving one checkpointed thread would corrupt its state. A
        # fresh run can't trip this (its conv_id_out is a brand-new row). Degrade-
        # safe: a typed busy error frame + done, never a 500.
        #
        # Abort-handover: the abort endpoint only sets
        # an Event that the OLD stream honours at the next super-step boundary —
        # if that stream is inside a long LLM/tool call it stays `active` for a
        # while, and an immediate "继续" here used to dead-end on the busy error
        # with no recourse ("中止后群聊进入后端运行？控制不了了？"). When the old
        # run IS already aborting, wait a short grace window for it to wind down
        # instead of bouncing the operator.
        _st_now = getattr(app, "_agents_api_state", None)
        if conv_id_out and isinstance(_st_now, dict) and isinstance(_st_now.get("task"), dict):
            _live_task = _st_now["task"]
            if _live_task.get("active") and _live_task.get("conversation_id") == conv_id_out:
                # "Is the old run winding down?" must consult BOTH the global
                # emergency latch and the OLD run's own stop Event — since
                # 2026-07-28 the abort button sets the latter, not the former.
                _global_ab = getattr(app, "_orch_abort", None)
                _old_rid = str(_live_task.get("id") or "")
                _mk_ev = getattr(app, "run_abort_event", None)
                _old_run_ab = (_mk_ev(_old_rid)
                               if (callable(_mk_ev) and _old_rid) else None)
                _abort_now = _AnyAbort(global_event=_global_ab,
                                       run_event=_old_run_ab)
                _aborting = _abort_now.is_set()
                if _aborting:
                    yield _sse({"kind": "status",
                                "text": "上一次运行正在中止收尾——等待其停止后自动继续"})
                    _t_wait = _time.time()
                    while (_time.time() - _t_wait) < _ABORT_HANDOVER_GRACE_S:
                        _lt = _st_now.get("task")
                        if not (isinstance(_lt, dict) and _lt.get("active")
                                and _lt.get("conversation_id") == conv_id_out):
                            break  # old stream wound down — proceed with this run
                        _time.sleep(0.25)
                _lt = _st_now.get("task")
                if (isinstance(_lt, dict) and _lt.get("active")
                        and _lt.get("conversation_id") == conv_id_out):
                    msg = (
                        "该群聊的上一次运行仍在中止收尾（可能卡在一个长工具调用，"
                        "如等待扫描完成）——已再次发出中止信号，请稍候几秒再点继续。"
                        if _aborting else
                        "该群聊正在后端运行中——请先中止或等待完成"
                        "（已避免并发开第二条流，防止线程状态损坏）。"
                    )
                    if _aborting and _abort_now is not None:
                        try:
                            _abort_now.set()
                        except Exception:  # noqa: BLE001
                            pass
                    yield _sse({"kind": "error", "degraded": True, "message": msg})
                    yield _sse({"kind": "done", "final_text": "", "aborted": False,
                                "conversation_id": conv_id_out})
                    return

        # Reset the shared abort Event for this run (created eagerly in core; the
        # E_STOP hook + every ExecutionContext share its identity — NEVER replace
        # it, only clear()).
        #
        # …EXCEPT when it is a LATCHED EMERGENCY (2026-07-28). The clear used to
        # be unconditional, so an E_STOP that had just halted an in-flight
        # PRIVATE-chat composite was undone the instant anyone started a group
        # task — its write commands re-armed in place, with nobody informed. An
        # emergency needs an explicit release (MASTApp.clear_emergency_latch),
        # not a side effect of the next task.
        abort = getattr(app, "_orch_abort", None)
        if abort is not None:
            if getattr(app, "_orch_abort_emergency", False) and abort.is_set():
                yield _sse({"kind": "error", "degraded": True, "message": (
                    "急停仍处于生效状态——本次任务不会启动。急停会同时冻结群聊/私聊"
                    "/信号采集三条链路的仪器写命令；请先在界面上解除急停（确认针尖与"
                    "样品状态后），再重新下发任务。")})
                yield _sse({"kind": "done", "final_text": "", "aborted": True,
                            "failed": True,
                            "stop_reason": "急停生效中，任务未启动",
                            "conversation_id": conv_id_out})
                return
            try:
                abort.clear()
            except Exception:
                abort = None
        # Publish this run's id so every ExecutionContext built for it scopes its
        # composite step-progress sidecars to THIS run — a finished run's
        # progress can then never be resumed by the next one (the 2026-07-10
        # fake-进针: a completed AutoApproach sidecar short-circuited every later
        # approach into an instant success without touching the instrument).
        try:
            app._orch_run_id = task_id
        except Exception:  # noqa: BLE001 — degrade to the legacy name-only key
            pass
        # This run's OWN stop Event, fresh. `abort` below is the union the stream
        # polls: the global emergency latch OR this run's stop. Aborting run A no
        # longer stops run B, and starting B no longer un-stops A.
        run_abort = None
        try:
            _mk = getattr(app, "run_abort_event", None)
            if callable(_mk):
                run_abort = _mk(task_id)
                if run_abort is not None:
                    run_abort.clear()
        except Exception as exc:  # noqa: BLE001
            logger.debug("run-task: per-run abort Event unavailable: %s", exc)
        abort = _AnyAbort(global_event=abort, run_event=run_abort)
        try:
            from mast.skills.composite.graph_executor import _sweep_stale_sidecars
            _sweep_stale_sidecars()
        except Exception:  # noqa: BLE001 — housekeeping must never break a run
            pass
        # Tie every refusal in this run to this run, so a post-mortem can isolate
        # ONE task out of a day's worth (记录 → 诊断, filter by run).
        try:
            from mast.core.diagnostics import set_run_id
            set_run_id(task_id)
        except Exception:  # noqa: BLE001
            pass
        # 本回合的来历，供任何在这条链上运行的工具读取（文档保存据此填
        # conversation_id / run_id，回答「这份报告是哪次群聊产出的」）。
        # best-effort：provenance 缺一个字段是记账损失，不该影响这次运行。
        try:
            from mast.core.turn_context import set_turn
            set_turn(conversation_id=conv_id_out or None, run_id=task_id)
        except Exception:  # noqa: BLE001
            pass
        if turn_holder is not None:
            # 外层包装每次恢复执行前从这里重读并重设（并发流之间不会串档）。
            turn_holder["conversation_id"] = conv_id_out or None
            turn_holder["run_id"] = task_id

        # Recovery point: re-arm the tip-crash SafetyWatchdog. Its anomaly latch
        # is one-shot, so a retract that fired during a previous task would
        # otherwise leave the net disarmed for the rest of the session. Starting
        # a new task means the operator has taken control and is proceeding.
        try:
            reset_wd = getattr(app, "reset_watchdog", None)
            if callable(reset_wd):
                reset_wd()
        except Exception:
            pass

        rec_limit = _effective_recursion_limit(app)
        yield _sse({"kind": "start", "task_id": task_id, "task": instruction,
                    "conversation_id": conv_id_out, "step_limit": rec_limit,
                    **_progress(0, rec_limit)})
        # Open a training/usage trajectory for this task so the skill/agent-turn
        # recorders persist (#8 P1; the sink went un-wired in the TS rewrite —
        # 审查). Best-effort.
        try:
            begin_traj = getattr(app, "begin_trajectory", None)
            if callable(begin_traj):
                begin_traj(thread_id, operator_intent={"instruction": instruction})
        except Exception:
            pass
        # Persist the operator turn first so a reconnect shows what was asked.
        _persist("operator", role="user", text=instruction)
        yield _sse({"kind": "status", "text": "SUP 接收任务，准备分发", "backend": backend})

        # The live Agents-API task slot (GET /api/agents/snapshot overlay) is set up
        # AFTER the langgraph import guard below, so an import failure can't leave a
        # stale active=True task behind (the finally only covers the post-setup body).
        st = getattr(app, "_agents_api_state", None)
        st = st if isinstance(st, dict) else None
        st_lock = st.get("lock") if st else None

        seen_ids: set = set()
        last_text = ""
        last_node = "_supervisor"
        final_text = ""
        completed = True
        # Persist why a run did not complete, distinguishing a backend error,
        # unreadable interrupt and unanswered approval from an operator abort.
        # A reason shown only in the live error frame would be lost on replay.
        # None denotes an actual operator abort.
        stop_reason: str | None = None
        # Did the supervisor actually route to END? See the latch in _drive: a
        # subgraph branch that never hands back dies silently, and without this
        # the run reports success anyway.
        saw_end = False
        # 这一 run 属于哪条科研纲领（2026-08-28）。收尾时落一条 goal_check 事件 ——
        # 「它是什么时候变成满足的」需要一条带时间戳的序列，而 campaigns 行上只有
        # 最新状态。**只认流里真见过的那一个，不猜。**
        _campaign_seen: dict = {}
        running_flag_set = False
        step_count = 0  # super-steps consumed of _RECURSION_LIMIT (surfaced to UI)
        try:
            from langchain_core.messages import HumanMessage
            from langgraph.types import Command
        except Exception as exc:
            yield _sse({"kind": "error",
                        "message": f"langgraph 未安装: {type(exc).__name__}",
                        "degraded": True})
            yield _sse({"kind": "done", "final_text": "", "aborted": False})
            return

        # Idle-gate for background dreaming + the wake scheduler: mark busy while
        # streaming.
        try:
            app._orch_running = True
            running_flag_set = True
        except Exception:
            pass

        # ARM the per-run cost ceiling. Same lifecycle as the busy flag above and
        # disarmed in the same finally, because a meter that outlives its run makes
        # the NEXT run look like it has already spent its budget.
        try:
            _arm = getattr(app, "begin_run_budget", None)
            if callable(_arm):
                _arm(_effective_run_budget_usd(app))
        except Exception:  # noqa: BLE001 — billing never blocks a run
            pass

        # CLAIM the task slot ATOMICALLY (post import-guard) so the snapshot
        # reflects this run; cleared (active=False) in the finally on EVERY exit
        # path.
        #
        # The check and the write used to be far apart — read at the top of this
        # generator, written here, with seconds of graph building and DB writes
        # in between, and the read only ever compared conversation_id. Two
        # streams could therefore both pass and then fight over process-level
        # SINGLE slots: this `task` dict, `_orch_run_id` (which scopes composite
        # step-progress sidecars) and, before today, one shared abort Event.
        # Doing the check inside the same critical section as the write closes
        # the window; refusing a run from ANOTHER conversation closes the
        # cross-conversation hole (审计 严重级
        # 「并发 run 无互斥」). Genuine parallelism has its own door:
        # POST /agents/run-task/background.
        # A queued agent-tool rebuild must land BEFORE a new task claims the
        # slot . The invariant this buys is stronger than
        # "the drain eventually happens on some exit path": **no task ever
        # starts on a stale tool table.** It is also the backstop for the
        # finally-block drain below — a hard client disconnect or a crashed
        # stream can skip that one, and then nothing else would ever fire.
        # Synchronous on purpose: starting the run first would race the rebuild.
        try:
            _drain = getattr(app, "drain_pending_agent_rebuild", None)
            if callable(_drain):
                _drained = _drain(sync=True)
                if _drained:
                    logger.info("run-task: applied queued agent rebuild (%s)",
                                _drained)
        except Exception:  # noqa: BLE001 — never block a run on this
            logger.warning("run-task: queued rebuild drain failed",
                           exc_info=True)

        if st is not None:
            _t0 = _time.time()
            with (st_lock or _contextlib.nullcontext()):
                _held = st.get("task")
                if isinstance(_held, dict) and _held.get("active"):
                    _same = _held.get("conversation_id") == conv_id_out
                    _msg = (
                        "该群聊正在后端运行中——请先中止或等待完成"
                        "（已避免并发开第二条流，防止线程状态损坏）。"
                        if _same else
                        f"另一个群聊任务仍在运行（{_held.get('description', '')[:40]}）"
                        "——前台编排器一次只跑一个任务，因为它驱动的是同一台仪器。"
                        "请等它结束或先中止；需要真并行请用「后台运行」。"
                    )
                    _busy_claim = _msg
                else:
                    _busy_claim = ""
                    st["task"] = {
                        "active": True, "id": task_id,
                        "description": instruction,
                        "conversation_id": conv_id_out,
                        "started_at": _t0, "active_agent_id": "_supervisor",
                        # Plural since parallel fan-out: SEVERAL agents can be
                        # running at once, and the topology view must be able to
                        # light them all up. active_agent_id stays for back-compat
                        # (it carries the joined label when more than one is live).
                        "active_agents": [],
                        "handoffs": [], "threads": {},
                        "final_text": "", "error": None,
                        # NB: there is deliberately no "artifacts" slot any more.
                        # It was initialised to {} here and written by NOTHING,
                        # anywhere — so GET /api/artifacts could only ever report
                        # zero produced artifacts (a mock in the test suite hid
                        # that). The artifacts a run produces are FILES;
                        # /api/artifacts now enumerates them from the real stores
                        # (agents._shared.artifacts.list_existing).
                        "events": [{"t": _t0, "kind": "start",
                                    "text": "SUP 接收任务"}],
                    }
                    # Drop anything left in the interjection queue from an
                    # earlier task. Belt-and-braces with the task_id filter in
                    # `_orch_control_provider`: a run that ends in a window with
                    # no consumer (final LLM call / abort / exception) leaves
                    # entries behind, and the next task's FIRST supervisor hop
                    # used to drain them (审计 严重级
                    # 「插话队列跨任务泄漏」).
                    _left = st.get("interjects") or []
                    if _left:
                        logger.warning(
                            "run-task %s: discarding %d undelivered "
                            "interjection(s) from the previous task",
                            task_id, len(_left))
                    st["interjects"] = []
            if _busy_claim:
                yield _sse({"kind": "error", "degraded": True,
                            "message": _busy_claim})
                yield _sse({"kind": "done", "final_text": "", "aborted": False,
                            "conversation_id": conv_id_out})
                if running_flag_set:
                    try:
                        app._orch_running = False
                    except Exception:  # noqa: BLE001
                        pass
                return

        # Clear any critical-event gate state left on the CACHED middleware
        # instances. The compiled graph lives on ``app._orchestrator`` and is
        # reused run after run, so BufferHITLMiddleware state leaks both ways:
        # an aborted run used to leave ``_interrupted_ids`` behind and the next
        # run's first before_model reopened a gate nobody had answered, while a
        # broken approval channel left ``_unresolved`` behind with no reachable
        # reopen branch — wedging the instrument until the process restarted.
        # A new task IS the operator's acknowledgement; reset_all_gates logs +
        # records a diagnostic whenever it clears something that was holding.
        try:
            from mast.agents._shared.buffer_hitl import reset_all_gates

            _n_reset = reset_all_gates(f"new run {task_id}")
            if _n_reset:
                yield _sse({"kind": "status",
                            "text": (f"注意：本次运行清除了 {_n_reset} 个仍未解决的"
                                     f"关键事件闸门（上一次运行遗留）")})
        except Exception as exc:  # noqa: BLE001 — never block a run on this
            logger.debug("run-task: buffer HITL gate reset skipped: %s", exc)

        def _drive(stream_input):
            """Yield SSE strings for one stream pass; recurse via Command(resume)
            on a HITL interrupt. Returns (frames_generator) semantics via yield."""
            nonlocal last_text, last_node, final_text, completed, step_count
            nonlocal stop_reason, saw_end
            for ns, chunk in orchestrator.stream(
                stream_input, config=cfg, stream_mode="updates", subgraphs=True,
            ):
                step_count += 1  # one super-step consumed of _RECURSION_LIMIT
                if abort is not None and abort.is_set():
                    completed = False
                    return
                # Honour an operator hold on the currently-active agent: pause the
                # graph at this boundary (no further chunks pulled) until released
                # or aborted. POST /agents/{id}/hold sets the flag; /release clears.
                yield from _honor_holds(st, last_node, abort)
                if abort is not None and abort.is_set():
                    completed = False
                    return
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    published = _publish_interrupt(
                        app, ns, chunk["__interrupt__"], thread_id)
                    if not published:
                        logger.error("run-task: unparseable __interrupt__: %r", chunk)
                        completed = False
                        stop_reason = "内部错误：无法解析审批请求（__interrupt__）"
                        return
                    decisions = []
                    for p in published:
                        yield _sse({
                            "kind": "interrupt",
                            "interrupt_id": p["event_id"],
                            "agent": p["agent_id"],
                            "skill": p["skill"],
                            "params": p["params"],
                            "rationale": p["rationale"],
                            "allowed_decisions": p["allowed_decisions"],
                            "thread_id": thread_id,
                            "interrupt_kind": p["kind"],
                            # ask_user only: the structured question the choice
                            # card renders. Omitted entirely for every other kind
                            # so existing frame consumers see no new key.
                            **({"ask": p["ask"]} if p.get("ask") else {}),
                        })
                        _persist("interrupt", agent=p["agent_id"],
                                 text=f"{p['skill']} — {p.get('rationale', '')}".strip(" —"))
                    for p in published:
                        resume_val = None
                        # Relay the wait's heartbeats straight to the client so
                        # the UI can show "等待人工审批 · 已等 Ns" instead of a
                        # dead stream, and so idle-TCP reapers keep their hands
                        # off this connection .
                        for _what, _payload in _await_resolution(
                                app, p["event_id"], p["kind"], abort):
                            if _what == "beat":
                                yield _sse(_payload)
                            else:
                                resume_val = _payload
                        if resume_val is None:
                            completed = False
                            # Distinguish "nobody answered in time" from "the
                            # operator pressed abort" — an approval that timed
                            # out is a workflow problem, not an operator action.
                            if abort is None or not abort.is_set():
                                if p["kind"] == "ask_user":
                                    # halt-on-timeout: the agent declared this
                                    # question has no safe default, so stopping
                                    # IS the requested outcome — not a failure to
                                    # get an approval. The checkpoint stands.
                                    stop_reason = (
                                        f"提问未得到答复，已按提问方要求停下等待"
                                        f"（{p.get('rationale', '')}）".strip("（）"))
                                else:
                                    stop_reason = (
                                        f"等待人工审批未得到答复（{p.get('skill', '')} "
                                        f"— {p.get('rationale', '')}）".strip(" —"))
                            return
                        decisions.append(resume_val)
                        if p["kind"] == "buffer_hitl":
                            # A critical-event pause has a REAL verdict now.
                            # Until 2026-07-28 approve / reject / the 900 s
                            # timeout-as-reject all ended with the tool gate
                            # OPEN, because the middleware discarded the resume
                            # value. It reads it now; this frame is what tells
                            # the operator which of the two happened, and the
                            # out-of-band resolve covers the case where the
                            # graph never replays that node.
                            try:
                                from mast.agents._shared.buffer_hitl import (
                                    _verdict_is_approval,
                                    resolve_all_gates,
                                )
                                _ok = _verdict_is_approval(resume_val)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("gate verdict read skipped: %s", exc)
                                _ok = None
                                resolve_all_gates = None
                            if _ok is True:
                                try:
                                    if resolve_all_gates is not None:
                                        resolve_all_gates(
                                            f"operator approved {p['event_id']}")
                                except Exception as exc:  # noqa: BLE001
                                    logger.debug("gate resolve skipped: %s", exc)
                                yield _sse({"kind": "status",
                                            "text": "关键事件已确认处理 — 恢复运行"})
                            else:
                                yield _sse({
                                    "kind": "status",
                                    "text": ("关键事件未获批准 — 工具闸门保持关闭，"
                                             "本次运行只能读取/停止/退针"),
                                })
                    # Resume the SAME thread — ADDRESSED BY INTERRUPT ID.
                    #
                    # Under parallel fan-out several branches can be paused at
                    # once (e.g. instrument_control awaiting a DANGEROUS-skill
                    # approval while a composite's human node waits in another
                    # branch). A bare Command(resume=<value>) is BROADCAST to
                    # every pending interrupt, so one branch's decision would
                    # also be fed to the other's — approving B's SetBias because
                    # you approved A's. LangGraph accepts a {interrupt_id: value}
                    # map (spike-verified), which routes each decision to exactly
                    # the branch that asked. Decisions from several
                    # action_requests inside ONE interrupt are merged back into
                    # that interrupt's single expected {"decisions": [...]} value.
                    by_lg: dict[str, Any] = {}
                    legacy: list = []           # pendings with no lg_id (older graphs)
                    for p, d in zip(published, decisions):
                        lg = p.get("lg_id")
                        if not lg:
                            legacy.append((p, d))
                            continue
                        if p["kind"] in ("workflow_human", "ask_user"):
                            by_lg[lg] = d       # single-valued resume
                        else:
                            slot = by_lg.setdefault(lg, {"decisions": []})
                            slot["decisions"].extend((d or {}).get("decisions", []))
                    if by_lg and not legacy:
                        resume: Any = by_lg
                    else:
                        # Legacy / mixed: fall back to the original single-value
                        # shape (correct whenever there is exactly one pending
                        # interrupt, which is every serial run).
                        if published and published[0]["kind"] in (
                                "workflow_human", "ask_user"):
                            resume = decisions[0]
                        else:
                            merged = []
                            for d in decisions:
                                merged.extend((d or {}).get("decisions", []))
                            resume = {"decisions": merged}
                    yield from _drive(Command(resume=resume))
                    return
                # Normal chunk → bucket into per-agent message / handoff frames.
                owner = _agent_from_namespace(ns)
                for node_name, node_state in (chunk or {}).items():
                    resolved = owner or _normalize_node(node_name)
                    msgs = (node_state or {}).get("messages") or []
                    # ── 目标判据帧（2026-08-27） ─────────────────────────
                    # 只在 supervisor 真的写回了一次求值时才发。面板据此显示
                    # 「目标判据 k/n 已满足」。客户端对未知 kind 是
                    # ``default: break``，所以旧前端零影响；没给 done_when 的
                    # run 这里一帧都不会多。
                    _rc = (node_state or {}).get("research_campaign")
                    if _rc is not None:
                        _cid = (_rc.get("campaign_id") if isinstance(_rc, dict)
                                else getattr(_rc, "campaign_id", ""))
                        if _cid:
                            _campaign_seen["id"] = str(_cid)
                    _g = (node_state or {}).get("goal")
                    if isinstance(_g, dict) and _g.get("last_verdict"):
                        _lv = _g["last_verdict"]
                        yield _sse({"kind": "goal", "t": _time.time(),
                                    "verdict": _lv.get("verdict"),
                                    "met": _lv.get("satisfied"),
                                    "total": _lv.get("total"),
                                    "reason": _lv.get("reason"),
                                    "items": _lv.get("per_predicate") or []})
                    # ── DID THE GRAPH ACTUALLY TERMINATE? ────────────────────
                    # The parent graph has exactly one edge (START→supervisor)
                    # and the agent subgraphs are bare compiled graphs with no
                    # outgoing edge, so a branch that ends WITHOUT calling a
                    # handoff tool just… stops. `graph.stream()` runs dry with no
                    # exception, and every one of these puts an agent there:
                    # the model answering in prose, StallGuard's bare AIMessage,
                    # ModelCallLimit / ToolCallLimit with exit_behavior="end"
                    # (which jumps to the SUBGRAPH's end, not back to the
                    # supervisor — the comment claiming otherwise was wrong).
                    # The driver then reported `{"aborted": false, "failed":
                    # false}` and the UI showed 「任务完成」: a dead branch and a
                    # finished run were pixel-identical.
                    #
                    # The evidence was already in the stream and nobody read it:
                    # EVERY supervisor END branch writes active_agent="__end__"
                    # (loop guard, budget, routing error, no-model, normal end),
                    # and a dispatch writes the target names instead. So: latch
                    # it on END, clear it on dispatch, and check it once the
                    # stream is exhausted.
                    #
                    # Two independent signals because neither covers everything:
                    # the route note is absent on the loop-guard / budget /
                    # routing-error END branches, and `active_agent` is absent
                    # from any update that only carries messages.
                    _aa = (node_state or {}).get("active_agent")
                    if isinstance(_aa, str) and _aa:
                        saw_end = (_aa == "__end__")
                    elif resolved == "_supervisor":
                        _tg = _parse_dispatch_targets(msgs)
                        if _tg is not None:
                            saw_end = not _tg
                    # ── context compaction ──────────────
                    # The middleware just replaced a stretch of this agent's
                    # history with a summary. Render it as ONE explicit divider
                    # rather than as messages: the update re-emits the preserved
                    # tail verbatim, and on a RESUMED conversation (a fresh
                    # `seen_ids`) those would otherwise be re-rendered AND
                    # re-persisted — the compaction would silently DUPLICATE up
                    # to `keep` rows of history it was supposed to shorten.
                    _comp = _compaction_event(msgs)
                    if _comp is not None:
                        for m in msgs:
                            seen_ids.add(getattr(m, "id", None) or id(m))
                            seen_ids.add(_msg_signature(m))
                        _line = _compaction_line(_comp)
                        _cmeta = {k: v for k, v in _comp.items() if k != "summary"}
                        _csum = str(_comp.get("summary") or "")
                        if _csum:
                            _cmeta["summary"] = _clip(_csum, _META_DETAIL_MAX)
                        yield _sse({
                            "kind": "compaction", "agent": owner or "",
                            "text": _line, "t": _time.time(),
                            "compaction": _cmeta,
                            "step": step_count, "step_limit": rec_limit,
                            **_progress(step_count, rec_limit),
                        })
                        # `agent` is the SUBGRAPH owner (compaction is per-agent
                        # context, not the group's) — empty at top level, where
                        # the client shows the divider without attribution
                        # rather than guessing an agent.
                        _persist("compaction", agent=owner or "", text=_line,
                                 meta=_meta_json(_cmeta))
                        continue
                    for m in msgs:
                        mid = getattr(m, "id", None) or id(m)
                        sig = _msg_signature(m)
                        if mid in seen_ids or sig in seen_ids:
                            continue
                        seen_ids.add(mid)
                        seen_ids.add(sig)
                        cls = m.__class__.__name__
                        if cls == "HumanMessage":
                            continue  # operator echo — already in `start`
                        txt = _msg_text(m).strip()
                        # Auto-background signal (conservative whitelist): the
                        # supervisor peeled an independent agent (literature) off a
                        # fan-out that also had instrument_control, so it can run
                        # DETACHED instead of barrier-blocking the instrument turn.
                        # The graph node can't start threads, so the SPAWN happens
                        # here in the bridge. seen_ids (above) makes each marker
                        # message fire exactly once. Best-effort: a spawn failure
                        # degrades to a visible status, never breaks the stream.
                        bg_names = _parse_auto_background(txt)
                        if bg_names:
                            _mgr = _bg_manager(app)
                            for _ag in bg_names:
                                _ok = False
                                if _mgr is not None:
                                    try:
                                        _mgr.spawn(instruction=instruction,
                                                   conversation_id=conv_id_out,
                                                   agents=(_ag,),
                                                   title=f"[后台] {instruction[:30]}")
                                        _ok = True
                                    except Exception as _exc:  # noqa: BLE001
                                        logger.warning("auto-background spawn failed (%s): %s",
                                                       _ag, _exc)
                                _txt = (f"已自动将 {_ag} 转入后台运行——前台仪器不被阻塞，结果稍后带回本群聊"
                                        if _ok else
                                        f"尝试将 {_ag} 转入后台失败，已退回前台由编排器处理")
                                yield _sse({"kind": "status", "agent": _ag, "text": _txt,
                                            "t": _time.time(), "step": step_count,
                                            "step_limit": rec_limit,
                                            **_progress(step_count, rec_limit)})
                                _persist("status", agent=_ag, text=_txt)
                            continue
                        tool_calls = getattr(m, "tool_calls", None) or []
                        if not txt and tool_calls:
                            # ONE natural-language line per call; the arguments
                            # ride along as structured data for the collapsed
                            # 参数 panel.
                            #
                            # What this replaces:
                            #     f"{name}({', '.join(f'{k}={v}')[:120]})"
                            # — the join was clipped at 120 chars and the ")"
                            # appended AFTER, which is where the operator's
                            # "…'w_m': 5.0, , )" came from, and everything past
                            # the fourth argument was dropped with no trace. A
                            # reader could not tell that from a two-argument call.
                            for tc in tool_calls:
                                tname = str((tc.get("name") if isinstance(tc, dict)
                                             else getattr(tc, "name", "tool")) or "tool")
                                targs = (tc.get("args") if isinstance(tc, dict)
                                         else getattr(tc, "args", {})) or {}
                                if not isinstance(targs, dict):
                                    targs = {"args": targs}
                                summary = narrate_tool_call(tname, targs)
                                # Pretty JSON, capped SERVER-side where the size
                                # is known, with the cap reported rather than
                                # applied silently.
                                args_json, clipped = summarize_args(targs)
                                meta = {"tool": tname, "args": args_json,
                                        "args_clipped": clipped}
                                yield _sse({
                                    "kind": "message", "agent": resolved,
                                    "role": "tool",
                                    "text": summary, "t": _time.time(),
                                    "tool": tname, "args": args_json,
                                    "args_clipped": clipped,
                                    "step": step_count, "step_limit": rec_limit,
                            **_progress(step_count, rec_limit),
                                })
                                # The summary is the durable row's TEXT, and the
                                # arguments go to `meta` — summarising must not
                                # mean deleting them from the only lasting record
                                # of the run.
                                _persist("message", agent=resolved, role="tool",
                                         text=summary, meta=_meta_json(meta))
                            continue
                        if not txt:
                            continue
                        role = ("agent" if cls == "AIMessage"
                                else "tool" if cls == "ToolMessage" else "agent")
                        # A tool RETURN is written for the model, not for a person
                        # watching: {"seqno": 1036, "progress": {"seqno": 1036,
                        # "line_idx": 2, …}} glued into the feed says nothing at a
                        # glance . Lead with a MECHANICAL digest of the
                        # top-level scalars — every token in it appears verbatim in
                        # the payload — and keep the full text one click away.
                        digest = narrate_tool_result(txt) if role == "tool" else ""
                        if digest:
                            yield _sse({
                                "kind": "message", "agent": resolved, "role": role,
                                "text": digest, "t": _time.time(),
                                "detail": _clip(txt, _SSE_TEXT_MAX),
                                "step": step_count, "step_limit": rec_limit,
                            **_progress(step_count, rec_limit),
                            })
                            _persist("message", agent=resolved, role=role,
                                     text=digest,
                                     meta=_meta_json({"detail": _clip(
                                         txt, _META_DETAIL_MAX)}))
                            last_text = txt[:1000]
                            final_text = last_text
                            continue
                        # The persisted row is the ONLY lasting copy of a tool
                        # return — action.data is empty and there is no plan
                        # store — so it gets the generous limit; the SSE frame
                        # is for immediate rendering and stays small. Both mark
                        # a clip explicitly: silent truncation is why a 28-step
                        # ExperimentPlan looked complete while ending mid-JSON
                        # at step 3 ( 68 of 340
                        # stored messages sat exactly at the old 2000 cap).
                        yield _sse({
                            "kind": "message", "agent": resolved, "role": role,
                            "text": _clip(txt, _SSE_TEXT_MAX), "t": _time.time(),
                            "step": step_count, "step_limit": rec_limit,
                            **_progress(step_count, rec_limit),
                        })
                        _persist("message", agent=resolved, role=role,
                                 text=_clip(txt, _PERSIST_TEXT_MAX))
                        last_text = txt[:1000]
                        final_text = last_text
                    # Record agent transitions + a per-agent thread tick into the
                    # live task slot UNDER THE STATE LOCK so the snapshot (read on
                    # another thread) sees a consistent handoff timeline + active
                    # agent + thread counts.
                    #
                    # PARALLEL (2026-07-11): when the supervisor fans out, chunks
                    # from the concurrent branches interleave, so the old
                    # "resolved != last_node ⇒ a handoff happened" heuristic
                    # invented a A→B→A→B ping-pong that never occurred. The
                    # supervisor's own route note is the authoritative dispatch
                    # record (`[SUPERVISOR → a ‖ b]`), so: parse it for the true
                    # fan-out set, and only log a handoff for a genuine SERIAL
                    # transition (nothing is concurrently active).
                    if st is not None and isinstance(st.get("task"), dict):
                        task = st["task"]
                        with (st_lock or _contextlib.nullcontext()):
                            fanned = _parse_dispatch_targets(msgs) if resolved == "_supervisor" else None
                            if fanned is not None:
                                task["active_agents"] = list(fanned)
                                task["active_agent_id"] = (
                                    " ‖ ".join(fanned) if fanned else "")
                                # THREE cases, not two. `_parse_dispatch_targets`
                                # returns [] for an END (it absorbs "__end__"),
                                # and the old expression fell into the `else`
                                # branch and read `fanned[0]` on that empty list.
                                #
                                # So EVERY orchestrator run that the supervisor
                                # finished normally died on the very last chunk
                                # with "IndexError: list index out of range" —
                                # after the work was done, after the
                                # "[SUPERVISOR → __end__]" note was already on
                                # screen. The operator saw 「运行出错，已停止」on
                                # runs that had actually succeeded, three times in
                                # one night (2026-07-28 #40/#41/#42/#48), and the
                                # log held one line with no frame.
                                #
                                # Unit tests missed it because they run without a
                                # live `_agents_api_state`, so `st` is None and
                                # this whole block is skipped.
                                if not fanned:
                                    kind, text = "end", "_supervisor → 结束"
                                elif len(fanned) > 1:
                                    kind = "dispatch"
                                    text = f"并行下发 → {' ‖ '.join(fanned)}"
                                else:
                                    kind, text = "handoff", f"_supervisor → {fanned[0]}"
                                task["handoffs"].append({
                                    "t": _time.time(),
                                    "kind": kind,
                                    "from": "_supervisor",
                                    "to": " ‖ ".join(fanned),
                                    "targets": list(fanned),
                                    "text": text,
                                })
                                if len(task["handoffs"]) > 200:
                                    del task["handoffs"][:-200]
                            elif (resolved and resolved != last_node
                                  and len(task.get("active_agents") or []) <= 1):
                                task["handoffs"].append({
                                    "t": _time.time(), "kind": "handoff",
                                    "from": last_node, "to": resolved,
                                    "text": f"{last_node} → {resolved}",
                                })
                                if len(task["handoffs"]) > 200:
                                    del task["handoffs"][:-200]
                                task["active_agent_id"] = resolved
                            who = resolved or last_node
                            thr = task["threads"].setdefault(who, [])
                            thr.append({"t": _time.time()})
                            if len(thr) > 100:
                                del thr[:-100]
                    last_node = resolved if resolved else last_node

        # recursion_limit counts EVERY super-step (incl. each step of an agent's
        # inner ReAct loop), not just supervisor hops — so a real multi-agent task
        # over a 6-agent pipeline needs ample headroom. 50 was too tight: a single
        # agent looping on a failed precondition / SafetyGate block could exhaust
        # it before the per-agent run-limit / supervisor hop guards engaged. 150
        # gives real tasks room while still bounding a genuine runaway.
        cfg = {"configurable": {"thread_id": thread_id},
               "recursion_limit": rec_limit}
        try:
            # Prepend resume / operator-reply context so a "继续" on a fresh
            # supervisor thread resumes the real experiment instead of answering
            # 「No prior context」, and a blocked agent's pending 心愿单 answer is
            # handed over automatically (2026-07 analysis ⑥/⑦). Best-effort — the
            # helper degrades to [] and never raises.
            _lead = _resume_lead_messages(app, instruction)
            _budget_seed = _effective_run_budget_usd(app)
            _park_seed = _park_board_seed()
            initial_state = {
                "messages": _lead + [HumanMessage(content=instruction)],
                "executed_skills": [], "scan_paths": [], "scan_metadata": {},
                "error_log": [], "event_refs": [],
                # None, not {} — visit_count's reducer ADDS, so an empty dict is a
                # no-op and hops accumulated for the LIFETIME of the thread. Once
                # the running total passed the loop guard (40) every later task in
                # that conversation ended instantly with no explanation. None is
                # the reducer's explicit CLEAR (same escape hatch routing_hints
                # has). See docs/v2/design/agent_communication_context_redesign.md.
                "visit_count": None,
                "pending_approvals": {},
                # SEED the budget gate (2026-07-30). This key had a working hard
                # gate in the supervisor and no writer anywhere in the tree, so it
                # never fired once; the run's only real cost bound was
                # recursion_limit, i.e. ~$50. The supervisor REFRESHES this every
                # hop from the probe wired in runtime; seeding it here is what
                # makes the first hop's comparison meaningful and what shows the
                # operator the ceiling in the UI. 0.0 from settings = gate off, in
                # which case the key is omitted entirely rather than seeded with a
                # zero that would end the run immediately.
                **({"budget_remaining_usd": _budget_seed}
                   if _budget_seed > 0 else {}),
                # Parks raised BEFORE this run existed (2026-07-30). The board is the
                # authority; this seeds the run's read-only cache of it. Without this
                # every new run re-asks every parked agent — paying again for a
                # decision already recorded on disk, and losing the decline count that
                # the next question depends on.
                **({"pending_activations": _park_seed} if _park_seed else {}),
                # 目标终止判据（2026-08-27）。**无条件写**：给了就是那份目标，
                # 没给就是 ``None`` —— 而 ``None`` 是这个裸 LastValue 通道的
                # **显式清除**，与上面 ``visit_count: None`` 是同一件事、同一个
                # 理由（2026-08-28 修）。
                #
                # 第一版写成「只在给了时才出现这个键」，看起来更保守，实际上是
                # 一个高危缺陷：``run_task(conversation_id=C)`` 复用同一个
                # ``thread_id``，而 checkpoint 里的 ``goal`` 没有任何人清。于是
                # 同一个群聊里，任务 #1 带着 done_when 跑完并产出了 analysis，
                # 任务 #2 **不带** done_when —— 它继承 #1 的 done_when 与
                # baseline，第一跳 Gate 1 就命中「目标已全部满足」，直接 END：
                # 一个 agent 都不派，用户的第二个问题永远没被回答。
                #
                # 这正是紧邻的 visit_count 注释里记着的那个病（「每一个后续任务
                # 都瞬间结束且没有解释」）换了一个通道复发。
                "goal": dict(goal) if goal else None,
            }
            # ── v2 编排（退出 LangGraph 的第四个切换面，DEFAULT OFF） ─────────
            # 开关每次 run 读一次；任何失败**退回旧路径并留痕** —— 静默兜底会让
            # 「新引擎一直在挂」看起来和「工作正常」一模一样。
            if _v2_group_enabled(app):
                try:
                    for _frame in _drive_v2(app, instruction, task_id,
                                            _v2_group_targets(app), _persist,
                                            abort, pending=_v2_pending):
                        if isinstance(_frame, dict):
                            if _frame.get("_final_text"):
                                final_text = _frame["_final_text"]
                                completed = bool(_frame.get("_completed", True))
                                stop_reason = _frame.get("_stop_reason", "")
                                saw_end = True   # v2 的结局是显式的，不需要闩锁推断
                                continue
                            yield _sse(_frame)
                        else:
                            yield _frame
                    _v2_ran = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("v2 group orchestration failed (%s); falling "
                                   "back to the LangGraph path", exc)
                    yield _sse({"kind": "status",
                                "text": f"（v2 编排失败已回退：{type(exc).__name__}: {exc}）"})
                    _v2_ran = False
            else:
                _v2_ran = False

            if not _v2_ran:
                yield from _drive(initial_state)
            # ── FAIL-SILENT TERMINATION GUARD (2026-07-28) ───────────────────
            # The stream is exhausted. If the supervisor never routed to END,
            # this run did not finish — a subgraph branch died where it stood
            # and langgraph raised nothing. Measured on a topology copied from
            # production: with one target not handing back, the supervisor runs
            # ONCE and stream() runs dry silently; under parallel fan-out a live
            # branch still closes the run with 「任务完成」 while the dead
            # branch's work is lost without a trace.
            #
            # `completed` was initialised True and only four paths ever set it
            # False (abort, unparseable interrupt, approval timeout, stream
            # exception) — not one of them is about the hand-back. So the run
            # persisted "完成" and the client, which reads only `aborted`, agreed.
            # 收尾留痕：同值合并靠 dedup_key，所以判据没变的十次 run 只留一行。
            # **只有这一 run 真的带着判据时才写** —— 否则「没给 done_when ⇒
            # 逐字节不变」在这条路上不成立：一次普通的群聊也会给纲领落一行
            # verdict=unknown 的事件。
            if _campaign_seen.get("id") and _goal:
                try:
                    from mast.agents._shared.campaign_tools import publish_goal_check

                    publish_goal_check(_campaign_seen["id"])
                except Exception as exc:  # noqa: BLE001 — 留痕绝不影响一次 run 的结局
                    logger.debug("campaign goal_check 收尾留痕失败: %s", exc)

            if completed and not saw_end:
                completed = False
                stop_reason = (
                    "任务未正常结束：某个智能体没有交回控制权就停了（模型直接作答未调用"
                    " handoff、空转保护终止、或调用次数达上限）。这一分支的工作可能已"
                    "丢失——请查看上面各智能体的最后一条消息，必要时重新下发任务。")
                logger.warning(
                    "run-task %s: stream exhausted with no [SUPERVISOR → __end__] "
                    "— fail-silent branch termination", task_id)
                try:
                    from mast.core.diagnostics import record as _diag_record

                    _diag_record("fail_silent_end", task_id, stop_reason,
                                 last_node=last_node, steps=step_count)
                except Exception:  # noqa: BLE001
                    pass
                yield _sse({"kind": "status", "degraded": True,
                            "text": stop_reason, "t": _time.time()})
            # Cross-check the answer against what this run actually did. The
            # missing half of on 2026-07-27 the IC agent reported a
            # finished 5-point STS grid and named a per-point summary file, the
            # run ended 「完成」, and nothing in the system could contradict it —
            # no skill had run, no action row existed, the file was never
            # written. The operator believed it.
            #
            # Only ever ADDS a warning: audit_run_claim never raises, and
            # returns ok=True whenever it has no ledger to judge against, so a
            # cold process or a missing runtime cannot turn a good run into a
            # flagged one.
            try:
                _audit = getattr(app, "audit_run_claim", None)
                _res = _audit(final_text) if (callable(_audit) and final_text) else None
                if _res and not _res.get("ok", True):
                    _warn = "⚠ 本次回答与运行记录不一致：" + "；".join(
                        str(p) for p in (_res.get("problems") or []))
                    logger.warning("run-task claim audit: %s", _warn)
                    yield _sse({"kind": "status", "text": _warn,
                                "t": _time.time(), "degraded": True})
                    _persist("status", role="assistant", text=_warn)
            except Exception as exc:  # noqa: BLE001 — an audit must never break a run
                logger.debug("run-task claim audit skipped: %s", exc)
        except Exception as exc:
            # A recursion-limit hit means a step runaway (often an agent spinning on
            # a precondition/safety failure) — surface it as a READABLE, degraded
            # frame, not a raw "GraphRecursionError" the UI renders as "加载失败".
            is_recursion = type(exc).__name__ == "GraphRecursionError" or (
                "recursion limit" in str(exc).lower())
            if is_recursion:
                msg = ("任务步数达到上限——可能某个智能体在预条件或安全门上反复失败而空转。"
                       "已停止本次运行。建议把任务拆细，或先检查/满足仪器预条件后重试。")
                logger.warning("run-task hit recursion limit: %s", exc)
            else:
                msg = f"{type(exc).__name__}: {exc}"
                # logger.exception, NOT .warning — this branch catches ANY
                # exception out of the whole graph stream, and the message alone
                # is often useless. The field saw a run die three times with
                # "IndexError: list index out of range" (2026-07-28 #40/#41/#48)
                # and the log held exactly that one line: no frame, no file, no
                # agent. An unlocatable crash in the one path that drives the
                # instrument is not a diagnosable system.
                logger.exception("run-task orchestrator stream failed: %s", exc)
                # Also record it as a diagnostic so 记录 → 诊断 can show it
                # against THIS task (the SSE frame is gone once the page reloads).
                try:
                    import traceback as _tb

                    from mast.core.diagnostics import record as _diag_record

                    # A LIST of the last frames, not one long string: _safe caps
                    # a str at 400 chars (which would keep the useless top of the
                    # trace) but keeps 20 list items at 400 each.
                    _lines = [ln for ln in _tb.format_exc().splitlines() if ln.strip()]
                    _diag_record("run_error", task_id, msg, traceback=_lines[-20:])
                except Exception:  # noqa: BLE001 — diagnostics must never mask it
                    pass
            yield _sse({"kind": "error", "message": msg, "degraded": True})
            completed = False
            stop_reason = msg
            if st is not None and isinstance(st.get("task"), dict):
                with (st_lock or _contextlib.nullcontext()):
                    st["task"]["error"] = msg
        finally:
            # 线程池会复用线程 —— 不清掉的话下一个回合的产物会记上这一轮的来历。
            try:
                from mast.core.turn_context import clear_turn
                clear_turn()
            except Exception:  # noqa: BLE001
                pass
            if running_flag_set:
                try:
                    app._orch_running = False
                except Exception:
                    pass
            # DISARM the cost ceiling on every exit path — a live meter would keep
            # counting and the next run would start already "over budget".
            try:
                _disarm = getattr(app, "end_run_budget", None)
                if callable(_disarm):
                    _disarm()
            except Exception:  # noqa: BLE001
                pass
            # Close the training trajectory on EVERY exit path (done/abort/raise/
            # client-disconnect) so it doesn't leak open (#8 P1).
            try:
                end_traj = getattr(app, "end_trajectory", None)
                if callable(end_traj):
                    end_traj(exit_status=("completed" if completed else "aborted"))
            except Exception:
                pass
            # Mark the live task slot complete (under the lock) so the snapshot
            # stops showing it active — runs on EVERY exit path (done/abort/raise).
            # Ownership check: after an abort-handover a NEW run may already own
            # the slot (task.id differs) — a late-exiting old stream must not
            # flip the new run to inactive.
            _released_the_slot = False
            if st is not None and isinstance(st.get("task"), dict):
                with (st_lock or _contextlib.nullcontext()):
                    if st["task"].get("id") == task_id:
                        st["task"]["active"] = False
                        st["task"]["final_text"] = final_text or "task completed"
                        st["task"].setdefault("events", []).append(
                            {"t": _time.time(),
                             "kind": ("done" if completed
                                      else "failed" if stop_reason else "aborted"),
                             "text": _terminal_label(completed, stop_reason)})
                        _released_the_slot = True
            # Now that the slot is free, apply anything that was queued while
            # this task ran . INSIDE the ownership check on
            # purpose: after an abort-handover a NEW run may already own the
            # slot, and a late-exiting old stream draining here would rebuild
            # the graph out from under it — the same race the id check above
            # was added for.
            if _released_the_slot:
                try:
                    _drain = getattr(app, "drain_pending_agent_rebuild", None)
                    if callable(_drain):
                        _drained = _drain()
                        if _drained:
                            logger.info("task finished: applying queued agent "
                                        "rebuild (%s)", _drained)
                except Exception:  # noqa: BLE001
                    logger.warning("queued rebuild drain failed", exc_info=True)
            # Persist the terminal entry + bump the group conversation's preview/
            # title INSIDE the finally so they still run when the client hard-
            # disconnects mid-stream (GeneratorExit runs finally but skips code
            # after it) — otherwise a reconnect would miss the done marker/preview.
            # Persist the terminal marker as a SHORT status, NOT a copy of
            # final_text — the agent's last 'message' row (persisted above) already
            # carries the answer, so re-persisting it here duplicated the final
            # answer on transcript replay ("第二次刷屏", 2026-07-06 feedback). The
            # live 'done' SSE frame still carries final_text and the conversation
            # preview below still uses it; only this persisted row is a marker now.
            # The persisted terminal row is the ONLY record a replay sees — the
            # error frame above goes to the live client and is gone. Writing
            # "已中止" for a crash made nine of eleven runs indistinguishable
            # from an operator abort ().
            _persist("done", role="assistant",
                     text=_terminal_label(completed, stop_reason))
            if conv_id_out and conv_store is not None:
                try:
                    conv_store.touch(conv_id_out, preview=final_text or "task completed")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("run-task: group conversation touch failed: %s", exc)
        if not completed:
            yield _sse({"kind": "status",
                        "text": (f"运行出错，已停止：{stop_reason}" if stop_reason
                                 else "operator aborted — stream stopped")})
        yield _sse({
            "kind": "done",
            "final_text": final_text or "task completed",
            "aborted": not completed,
            # `aborted` alone cannot tell the client whether the operator stopped
            # this or the backend fell over; both used to arrive as aborted=True.
            "failed": bool(stop_reason),
            "stop_reason": stop_reason or "",
            "conversation_id": conv_id_out,
            "step": step_count, "step_limit": rec_limit,
                            **_progress(step_count, rec_limit),
        })

    return gen()


@router.post("/agents/run-task")
def run_task(body: RunTaskRequest, request: Request) -> StreamingResponse:
    """Run a task through the 6-agent orchestrator and stream SSE frames.

    LIVE-ONLY: relays onto ``ctx.live_app._orchestrator`` (built once if needed).
    Degrades to a single error frame + done when the live app / orchestrator is
    absent. HITL ``__interrupt__`` chunks are published into the live
    ``_orch_interrupts`` store and surfaced as ``interrupt`` frames; resolve them
    via the existing agents_control resolve endpoint. Honours ``_orch_abort``."""
    ctx = request.app.state.ctx
    app = _live_app(ctx)
    instruction = (body.task or "").strip()
    # ── 目标终止判据 (2026-08-27) ────────────────────────────────────
    # **非法一律 422，不夹紧。** ``_clamp_waiting_for`` 夹的是模型输出（拒绝了
    # 它无处可去）；这里的来源是用户 / UI，静默丢掉一条谓词等于把目标悄悄改
    # 小，而「更容易达成的目标」正是这道闸要防的东西。拒绝无害：没有 done_when
    # 就是今天的行为。报文里带上目录与示例 —— 让写错的人在出错的地方拿到闭集。
    _goal: dict | None = None
    if body.done_when is not None:
        try:
            from mast.goals import normalise_done_when
            from mast.goals.spec import EXAMPLE, catalog_json, done_when_to_json
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"目标判据模块不可用：{exc}") from exc
        _spec, _errs = normalise_done_when(body.done_when)
        if _errs:
            raise HTTPException(status_code=422, detail={
                "error": f"done_when 有 {len(_errs)} 条不合法 —— 没有开始运行",
                "problems": _errs,
                "done_when_catalog": catalog_json(),
                "example": EXAMPLE,
            })
        if _spec is not None:
            _goal = {"text": (body.goal_text or instruction or "").strip(),
                     "done_when": done_when_to_json(_spec)}
    task_id = _uuid.uuid4().hex[:12]
    # 包一层「每次恢复执行前重设回合标识」：这个 stream 是同步 generator，交给
    # StreamingResponse 后由 Starlette 的 iterate_in_threadpool 消费，**每次 next()
    # 单独走一趟 to_thread，不保证同一个 worker 线程**。只在流开头设一次
    # thread-local，两条并发流会互相读到对方的 conversation_id / run_id ——
    # 那会把一条错的因果链当事实写进文档的 provenance。
    # 详见 mast/core/turn_context.reassert_turn_each_resume。
    _turn_holder: dict = {}
    # 阻塞期间帧的**第二条出口**（2026-08-27）。v2 编排是同步生成器，它阻塞在
    # ``next()`` 里时 yield 不出东西 —— 而「原地阻塞等人」正是这次迁移用来替掉
    # interrupt 重放的东西。生成器阻塞期间，工具进度与 HITL 等待帧
    # 都可能无法及时到达面板，所以需要不经过该生成器的输出通道。
    #
    # 帧在产生它的那条线程上就地生成、存进这个盒子，由下面 ``on_idle`` 的时钟排空
    # —— 那条路不经过被阻塞的生成器。**盒子必须建在这里而不是流里面**：
    # ``_run_task_stream`` 是生成器函数，函数体要到第一次 next() 才跑，而 ``on_idle``
    # 在那之前就要拿到它。走旧引擎时盒子始终是空的，``on_idle`` 排出空列表，心跳退回
    # ``: ping``，与从前逐字节相同。
    #
    # ★ import 写在函数里是**刻意的**，不是随手。``mast.agentruntime`` 的全部引用都
    #   是惰性的，而 ``test_packaging_declares_the_runtime`` 把这条**当成前提**钉着：
    #   正因为 PyInstaller 静态分析看不见它，``mast2.spec`` 才必须显式点名它。此前
    #   在这里写了顶层 import，那条自检当场变红 —— 它就是为这一刻写的。
    #   更要紧的是迁移自己的总不变式：**新运行时不该影响旧路径**。顶层 import 会让每
    #   一个从没翻过 v2 开关的用户在 bootstrap 时就加载它。实测代价：首次 16 ms，
    #   之后 0.2 µs —— 惰性没有任何可测的坏处。
    from mast.agentruntime.sse import PendingFrames as _PendingFrames

    _v2_pending = _PendingFrames()

    def _idle_frames() -> "list[str]":
        """``with_heartbeat`` 空闲时的回调：把攒下的帧序列化交出去。

        跑在**消费者线程**上，所以只能做「取走一个加了锁的列表」这种量级的事：
        它卡住多久，真正的帧就晚多久。
        """
        return [_sse(f) for f in _v2_pending.drain()]

    stream = _run_task_stream(app, instruction, task_id, body.conversation_id or "",
                              turn_holder=_turn_holder, pending=_v2_pending,
                              goal=_goal)
    try:
        from mast.core.turn_context import reassert_turn_each_resume
        stream = reassert_turn_each_resume(stream, _turn_holder)
    except Exception:  # noqa: BLE001 — provenance 是 best-effort，绝不拦住运行
        pass
    # _APPROVAL_BEAT_S only keeps the HITL *wait* alive; everything else in this
    # stream (one long LLM step, one long instrument tool call, a 10-minute
    # operator hold) is total silence, which is what idle-TCP reapers kill on a
    # Tailscale link. with_heartbeat covers the whole run
    # and — see mast/api/sse.py — deliberately lets the run FINISH when the client
    # drops instead of aborting the experiment at the next super-step boundary.
    return StreamingResponse(
        with_heartbeat(stream, label=f"run-task-{task_id}",
                       on_idle=_idle_frames),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/agents/run-task/abort", response_model=AbortTaskResponse)
def abort_task(request: Request) -> AbortTaskResponse:
    """Signal the orchestrator abort Event (stops the run BETWEEN super-steps)
    and wake any worker blocked on a pending DANGEROUS interrupt.

    LIVE-ONLY relay onto ``live_app._orch_abort`` + ``_orch_interrupts['events']``
    (mirrors the old task_abort). Degrades to a typed no-op when the live app /
    abort Event is absent — never a 500."""
    ctx = request.app.state.ctx
    app = _live_app(ctx)
    if app is None:
        return AbortTaskResponse(detail="no live app (standalone)", degraded=True)
    aborted: list[str] = []
    try:
        # Stop THIS run, not everything. The global ``_orch_abort`` is the
        # emergency latch shared by the group chat, the private chat, the
        # signals routes and the executor; setting it here meant 「中止这个群聊
        # 任务」 also froze a composite the operator was running in the main chat
        # (and, symmetrically, could not stop just one of two concurrent runs).
        # The E-STOP button — which DOES mean everything — goes through
        # MASTApp.emergency_stop().
        rid = str(getattr(app, "_orch_run_id", "") or "")
        _mk = getattr(app, "run_abort_event", None)
        run_ev = _mk(rid) if (callable(_mk) and rid) else None
        if run_ev is not None:
            run_ev.set()
            aborted.append(f"run:{rid}")
        else:
            # No run-scoped Event (older app / no live run) — fall back to the
            # legacy global signal rather than silently doing nothing.
            ab = getattr(app, "_orch_abort", None)
            if ab is not None:
                ab.set()
                aborted.append("orchestrator")
    except Exception as exc:
        logger.debug("orchestrator abort signal failed: %s", exc)
    # Wake any worker blocked on a pending DANGEROUS interrupt so abort actually
    # stops it (it polls abort between waits; setting the events releases it now).
    store = getattr(app, "_orch_interrupts", None)
    if store:
        try:
            lock = store.get("lock")
            if lock is not None:
                with lock:
                    for ev in store.get("events", {}).values():
                        ev.set()
            else:  # pragma: no cover
                for ev in store.get("events", {}).values():
                    ev.set()
        except Exception as exc:
            logger.debug("interrupt wake on abort failed: %s", exc)
    if not aborted:
        return AbortTaskResponse(
            ok=True, already_idle=True,
            detail="no orchestrator abort Event wired", degraded=True,
        )
    logger.info("run-task: abort requested → %s", aborted)
    return AbortTaskResponse(ok=True, aborted=aborted, degraded=False)


# ── Background runs (true parallelism — break the super-step barrier) ─────────
# A long/independent task (a literature survey) runs as a SEPARATE orchestrator
# invocation on its own thread_id + checkpointer (core: CoreRuntime.
# _background_run_fn), so the FOREGROUND group chat + instrument_control stay
# responsive. Its transcript merges back into the same durable conversation,
# tagged 「后台」. These are thin relays onto ``live_app._background_runs``; all
# isolation / concurrency / merge authority stays in the core manager.
def _bg_manager(app):
    """The live BackgroundRunManager (built lazily by the core). None when no live
    app / it can't be built. Never raises."""
    if app is None:
        return None
    ensure = getattr(app, "_ensure_background_manager", None)
    try:
        return ensure() if callable(ensure) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("background manager unavailable: %s", exc)
        return None


@router.post("/agents/run-task/background", response_model=BackgroundSpawnResponse)
def run_task_background(body: BackgroundSpawnRequest, request: Request) -> BackgroundSpawnResponse:
    """Launch a task as a DETACHED background orchestrator run (true parallelism —
    the foreground chat is NOT blocked). Results merge into the given group
    conversation's durable transcript, tagged 「后台」. instrument_control cannot be
    backgrounded (it is the foreground hardware agent). LIVE-ONLY; degrades to a
    typed body, never a 500."""
    app = _live_app(request.app.state.ctx)
    mgr = _bg_manager(app)
    if mgr is None:
        return BackgroundSpawnResponse(detail="background runs unavailable (standalone / no key)",
                                       degraded=True)
    try:
        rec = mgr.spawn(
            instruction=(body.instruction or "").strip(),
            conversation_id=(body.conversation_id or ""),
            agents=tuple(body.agents or ("literature",)),
            title=(body.title or ""),
            priority=(body.priority or "normal"))
    except ValueError as exc:  # caller error (empty / no backgroundable agent)
        return BackgroundSpawnResponse(ok=False, detail=str(exc), degraded=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("background spawn failed: %s", exc)
        return BackgroundSpawnResponse(detail=f"spawn failed: {type(exc).__name__}", degraded=True)
    logger.info("run-task background spawned: %s", rec.get("run_id"))
    return BackgroundSpawnResponse(ok=True, run=BackgroundRunInfo(**rec), degraded=False)


@router.get("/agents/background-runs", response_model=BackgroundRunsResponse)
def background_runs(request: Request, conversation_id: str = "",
                    active_only: bool = False) -> BackgroundRunsResponse:
    """List background runs (newest first), optionally scoped to one conversation
    or to still-active runs. READ-ONLY relay; degrades to empty."""
    mgr = _bg_manager(_live_app(request.app.state.ctx))
    if mgr is None:
        return BackgroundRunsResponse(degraded=True)
    try:
        runs = mgr.list_runs(conversation_id=(conversation_id or None),
                             active_only=active_only)
        return BackgroundRunsResponse(
            runs=[BackgroundRunInfo(**r) for r in runs], count=len(runs), degraded=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("background runs list failed: %s", exc)
        return BackgroundRunsResponse(degraded=True)


@router.post("/agents/run-task/background/{run_id}/abort",
             response_model=BackgroundAbortResponse)
def background_abort(run_id: str, request: Request) -> BackgroundAbortResponse:
    """Signal a background run to stop (honoured between its super-steps).
    LIVE-ONLY relay; degrades to a typed no-op."""
    mgr = _bg_manager(_live_app(request.app.state.ctx))
    if mgr is None:
        return BackgroundAbortResponse(run_id=run_id,
                                       detail="background runs unavailable", degraded=True)
    ok = bool(mgr.abort(run_id))
    return BackgroundAbortResponse(ok=ok, run_id=run_id, aborted=ok, degraded=False)


def _conv_store(ctx: Any):
    """Best-effort handle to the live ConversationStore. None in standalone."""
    store = getattr(ctx, "conversation_store", None)
    if store is not None:
        return store
    app = _live_app(ctx)
    return getattr(app, "_conv_store", None) if app is not None else None


def _active_group_conversation_id(ctx: Any) -> "str | None":
    """The group conversation a live run is currently streaming into (if any),
    read from the live operator-control task slot. None when idle / standalone."""
    app = _live_app(ctx)
    st = getattr(app, "_agents_api_state", None) if app is not None else None
    if isinstance(st, dict) and isinstance(st.get("task"), dict):
        task = st["task"]
        if task.get("active"):
            cid = task.get("conversation_id")
            return str(cid) if cid else None
    return None


# ── GET /agents/group-transcript ─────────────────────────────────────────────
# NOTE: a LITERAL 2-segment path (sibling of /agents/tools), NOT /agents/run-task/
# transcript — a "/agents/<x>/transcript" shape would be captured by the
# path-param route /agents/{agent_id}/... registered earlier (agents.py before
# orchestrator in app.py), silently binding agent_id="run-task" and returning the
# WRONG handler. Group reads live under their own literal /agents/group-* paths.
@router.get("/agents/group-transcript", response_model=TranscriptResponse)
def run_task_transcript(conversation_id: str, request: Request) -> TranscriptResponse:
    """Durable transcript of a 群聊 (group) run — what the TS client replays on
    reconnect so a multi-agent conversation survives a tab switch / reload.

    READ-ONLY relay onto the live ``ConversationStore`` transcript table. Degrades
    to an empty (degraded=True) body when no store is wired (standalone dev) — the
    client falls back to its in-session buffer and never breaks."""
    ctx = request.app.state.ctx
    store = _conv_store(ctx)
    if store is None:
        return TranscriptResponse(conversation_id=conversation_id, degraded=True)
    try:
        rows = store.messages_for(conversation_id)
        entries = [
            TranscriptEntry(
                seq=int(r.get("seq", 0)),
                kind=str(r.get("kind", "message")),
                agent_id=str(r.get("agent_id", "")),
                role=str(r.get("role", "")),
                text=str(r.get("text", "")),
                t=float(r.get("t", 0) or 0),
                meta=str(r.get("meta") or ""),
            )
            for r in (rows or [])
        ]
        active = _active_group_conversation_id(ctx) == conversation_id
        return TranscriptResponse(
            conversation_id=conversation_id, entries=entries,
            count=len(entries), active=active, degraded=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("run-task transcript read failed (%s): %s", conversation_id, exc)
        return TranscriptResponse(conversation_id=conversation_id, degraded=True)


def _scope_names(ctx, rows: list) -> "tuple[dict[str, str], dict[str, str]]":
    """Resolve {experiment_id: name} and {sample_id: name} for a batch of
    conversation rows, so the client renders 实验 → 样品 → 群聊 with real names
    instead of UUID prefixes  and without an N+1 round-trip per row.

    Reads the EXPERIMENT DB (conversations live in their own file, so this is a
    join we have to do here). Degrade-safe on every axis: no storage wired, a
    deleted experiment/sample, or a raising storage all just yield fewer names —
    the conversation list itself must still come back."""
    exp_names: dict[str, str] = {}
    sample_names: dict[str, str] = {}
    storage = getattr(ctx, "experiment_storage", None)
    if storage is None:
        return exp_names, sample_names
    exp_ids = {str(r.get("experiment_id")) for r in rows if r.get("experiment_id")}
    want_samples = {str(r.get("sample_id")) for r in rows if r.get("sample_id")}
    for eid in exp_ids:
        try:
            exp = storage.get_experiment(eid)
            if exp and exp.get("name"):
                exp_names[eid] = str(exp["name"])
            # Sample names are only reachable per-experiment; pulling the whole
            # sample list once per experiment covers every conversation under it.
            for s in (storage.get_samples(eid) or []):
                sid = str(s.get("id") or "")
                if sid in want_samples and s.get("name"):
                    sample_names[sid] = str(s["name"])
        except Exception as exc:  # noqa: BLE001 — a missing name must not 500 the list
            logger.debug("scope name lookup failed for experiment %s: %s", eid, exc)
    return exp_names, sample_names


# ── GET /agents/group-conversations ──────────────────────────────────────────
# Literal path (see /agents/group-transcript note): /agents/run-task/conversations
# WOULD be shadowed by /agents/{agent_id}/conversations and silently return an
# empty list with degraded=False (a false-healthy). Keep it collision-free.
@router.get("/agents/group-conversations", response_model=GroupConversationsResponse)
def run_task_conversations(request: Request) -> GroupConversationsResponse:
    """Durable group-conversation history (newest first) so a prior multi-agent
    run can be resumed. READ-ONLY relay onto ``ConversationStore.list(kind=
    'group')``, enriched with experiment/sample NAMES so the client can render the
    实验 → 样品 → 群聊 tree ; degrades to empty when no store is wired."""
    ctx = request.app.state.ctx
    store = _conv_store(ctx)
    if store is None:
        return GroupConversationsResponse(degraded=True)
    try:
        rows = store.list(kind="group")
        exp_names, sample_names = _scope_names(ctx, rows or [])
        conversations = [
            GroupConversation(
                conversation_id=str(r.get("conversation_id", "")),
                title=str(r.get("title") or "新任务"),
                thread_id=str(r.get("thread_id", "")),
                created_at=(str(r["created_at"]) if r.get("created_at") else None),
                updated_at=(str(r["updated_at"]) if r.get("updated_at") else None),
                last_message_preview=str(r.get("last_message_preview") or ""),
                experiment_id=r.get("experiment_id"),
                sample_id=r.get("sample_id"),
                experiment_name=exp_names.get(r.get("experiment_id") or ""),
                sample_name=sample_names.get(r.get("sample_id") or ""),
            )
            for r in (rows or [])
        ]
        return GroupConversationsResponse(
            conversations=conversations,
            active_conversation_id=_active_group_conversation_id(ctx),
            count=len(conversations),
            degraded=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("run-task conversations list failed: %s", exc)
        return GroupConversationsResponse(degraded=True)


__all__ = ["router"]

