"""向用户提问，**原地阻塞**等答案 —— ``interrupt()`` 重放的替代品。

为什么这个文件到今天才出现（2026-08-27）
----------------------------------------
``RunContext.ask_human`` 从第一天就在，但它一直是一个**没有实现的挂点**：默认 ``None``，
而 ``agentruntime`` 里没有任何地方把它接到 ``api/hitl_bridge`` 上。后果是具体的 ——
**v2 路径上问不了用户任何问题**：``ask_user`` 走的是 langgraph 的 ``interrupt()``，
它靠抛 ``GraphInterrupt`` 做控制流，在新循环里那就是一个普通异常，这一轮以
``outcome="error"`` 收场，而用户那边什么也不会弹出来。

这是「翻开关」真正危险的地方之一，而它不在任何一条闸门的视野里：闸门查的是开关、
import 和打包，查不出「新引擎少一件能力」。

重放 vs 原地续行
----------------
langgraph 需要**重放**（暂停点之后整个节点重跑一遍）不是因为那样更好，而是因为
**图节点不能占着线程等人**。代价是三处幂等设计在伺候它，外加 ``GraphInterrupt`` 要在
``execution_context`` / ``graph_executor`` / ``interpreter`` / ``skill_adapter`` **四处**
被特判豁免 —— 少一处，一次暂停就会被读成「这一步失败了」，**HITL 静默取消**。

而这次迁移的四条驱动链**全是可阻塞的工作线程**（Starlette threadpool worker / 后台
run 线程 / CLI 的 ``asyncio.to_thread``）。既然可以阻塞，就不需要重放：

* 工具体**只跑一次**（不必幂等）；
* 暂停不是异常，不需要谁去豁免它；
* 答案直接作为函数返回值回到问的那一行。

``hitl_bridge`` 整体保留
------------------------
它的等待机制（``threading.Event`` + 900 s fail-closed + beat 帧）**本来就是阻塞式的**，
只是此前只有 langgraph 那条路能把 pending 登记进去（``publish_interrupts`` 吃的是
``__interrupt__`` chunk）。这里做的事只有一件：**用同一份形状、往同一个 store 里登记
一条 pending**，然后驱动同一个 ``await_resolution``。

所以审批面板、``agents_control.resolve`` 中继、超时 fail-closed、``ask_user`` 的
``timeout_action`` 语义 —— 一个字都不用改，两个引擎共用。

⚠️ 形状必须**字面一致**
-----------------------
React 的审批卡按 ``kind`` 分支，按 ``skill`` / ``params`` / ``rationale`` 渲染兜底。
这里的 pending 逐字段照抄 ``publish_interrupts`` 的 ``ask_user`` / ``dangerous`` 分支
（有测试对着那个函数逐键比对）—— 造一份「差不多」的形状，症状会是**卡片渲染成空白**，
而那看起来像「用户没收到通知」。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: 与 ``hitl_bridge`` 同一份默认值（那边是单一真源，这里只在拿不到时兜底）。
_FALLBACK_MAX_WAIT_S = 900.0


#: live store 必须有的三个桶。``await_resolution`` 直接下标取用它们
#: （``store["resolved"].pop(...)``），缺一个就是从 ``hitl_bridge`` 深处冒出来的
#: ``KeyError`` —— 而那个报错读起来像「审批桥坏了」，不像「调用方给的 store 不完整」。
_REQUIRED_BUCKETS = ("pending", "events", "resolved")


def _register(store: Any, pending: dict) -> str:
    """把一条 pending 登记进 live store —— 与 ``publish_interrupts`` 同一套写法。"""
    missing = [k for k in _REQUIRED_BUCKETS if not isinstance(store.get(k), dict)]
    if missing:
        raise RuntimeError(
            f"HITL store 缺少 {missing} —— 它至少要有 {list(_REQUIRED_BUCKETS)} "
            "三个字典（见 CoreRuntime._orch_interrupts）。在这里明说，是因为往下走"
            "会在 hitl_bridge 里以 KeyError 收场，而那个报错指向错的地方。")
    eid = pending["event_id"]
    lock = store.get("lock")
    if lock is not None:
        with lock:
            store["pending"][eid] = pending
            store["events"][eid] = threading.Event()
    else:  # pragma: no cover — live store 一定有 lock
        store["pending"][eid] = pending
        store["events"][eid] = threading.Event()
    return eid


def _ask_user_pending(payload: dict, *, owner: str, thread_id: str) -> dict:
    """``ask_user`` 的 pending —— 逐字段照抄 ``publish_interrupts`` 的同名分支。"""
    ask = dict(payload.get("ask") or {})
    if not ask:
        ask = {
            "question": str(payload.get("question", "")),
            "header": str(payload.get("header", "")),
            "options": [o for o in (payload.get("options") or [])
                        if isinstance(o, dict)],
            "multi_select": bool(payload.get("multi_select", False)),
            "allow_custom": bool(payload.get("allow_custom", True)),
            "timeout_action": str(payload.get("timeout_action") or "continue"),
        }
    labels = [str(o.get("label")) for o in (ask.get("options") or [])
              if isinstance(o, dict)]
    return {
        "event_id": f"intr_{thread_id}_{uuid.uuid4().hex[:8]}",
        "agent_id": str(payload.get("agent_id") or "") or owner,
        "skill": "向用户提问",
        "params": {
            "question": ask.get("question", ""),
            "options": labels,
            "multi_select": ask.get("multi_select", False),
            "allow_custom": ask.get("allow_custom", True),
        },
        "rationale": str(ask.get("question", "")),
        # 不是 approve/reject —— 裁决**本身就是**答案。
        "allowed_decisions": ["answer"],
        "kind": "ask_user",
        "ask": ask,
        "thread_id": thread_id,
        "t": time.time(),
        # ★ 新引擎没有 langgraph 的 interrupt id，也不需要：定址靠 event_id。
        #   显式写 None 而不是不写这个键 —— 消费方 `.get("lg_id")` 两种都得到 None，
        #   但「写了 None」说得出「这里本来有个东西，新路径不需要它」。
        "lg_id": None,
    }


def _generic_pending(payload: dict, *, owner: str, thread_id: str) -> dict:
    """其余 kind（``dangerous`` / ``workflow_human`` / ``buffer_hitl`` …）。

    刻意**不**为每种 kind 各写一个构造器：那些分支在 ``publish_interrupts`` 里是为了
    从 langgraph 的 interrupt 值里**解析**出这些字段，而这里调用方直接给的就是解析好
    的形状。再抄一遍解析逻辑 = 第二个真源。
    """
    kind = str(payload.get("kind") or "dangerous")
    return {
        "event_id": f"intr_{thread_id}_{uuid.uuid4().hex[:8]}",
        "agent_id": str(payload.get("agent_id") or "") or owner,
        "skill": str(payload.get("skill") or "未知 skill"),
        "params": dict(payload.get("params") or {}),
        "rationale": str(payload.get("rationale") or ""),
        "allowed_decisions": list(payload.get("allowed_decisions")
                                  or ["approve", "edit", "reject"]),
        "kind": kind,
        "thread_id": thread_id,
        "t": time.time(),
        "lg_id": None,
    }
    # 注：``ask`` 键只有 ask_user 有，这里不塞空的 —— 一个空 ``ask`` 会让选择型
    #     UI 渲染出一张没有选项的卡片。


def make_ask_human(
    store: Any, *, owner: str = "", thread_id: str = "",
    on_beat: Callable[[dict], None] | None = None,
    max_wait: float | None = None,
) -> Callable[[dict], dict | None]:
    """造一个可以塞进 ``RunContext.ask_human`` 的问答函数。

    返回的函数**阻塞**直到用户回答、超时、或 run 被中止；返回用户的裁决
    （``hitl_bridge`` 的 resume value）或 ``None``。

    ``on_beat`` 是等待期间的心跳回调（SSE 那条链用它推 beat 帧给前端）。**不给也
    能用** —— 私聊与 CLI 没有帧要推，而一个「必须提供心跳」的接口会逼那两条路
    编一个假的出来。

    ★ ``store`` 为空时返回 ``None``，**不抛**
    ----------------------------------------
    「问不出去」和「问了被拒」是两件事，而调用方（``ask_user``）已经能区分：拿到
    ``None`` 它说「当前入口没接审批处理器」。抛异常会让整轮以 error 收场 —— 那正是
    今天 v2 上的样子，也是这个文件要修的东西。
    """
    def _ask(payload: dict) -> dict | None:
        if not store:
            logger.warning("ask_human: 没有 HITL store，问题发不出去（owner=%s）", owner)
            return None

        kind = str((payload or {}).get("kind") or "ask_user")
        build = _ask_user_pending if kind == "ask_user" else _generic_pending
        pending = build(dict(payload or {}), owner=owner,
                        thread_id=thread_id or "v2")
        eid = _register(store, pending)
        logger.warning("v2 HITL: pending %s kind=%s agent=%s", eid, kind,
                       pending["agent_id"])

        abort = (payload or {}).get("_abort")
        wait = max_wait
        if wait is None:
            wait = float((payload or {}).get("_timeout_s")
                         or _FALLBACK_MAX_WAIT_S)

        from mast.api.hitl_bridge import await_resolution

        # ``await_resolution`` 是生成器（它的原调用方是 SSE 生成器，要把 beat 帧推
        # 出去）。这里我们**就在可阻塞的工作线程上**，所以直接抽干它。
        for item in await_resolution(store, eid, kind, abort, max_wait=wait):
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            tag, value = item
            if tag == "beat":
                if on_beat is not None:
                    try:
                        on_beat(value)
                    except Exception as exc:  # noqa: BLE001 — 心跳坏了不该毁掉等待
                        logger.debug("ask_human beat sink failed: %s", exc)
            elif tag == "result":
                return value
        return None

    return _ask


def attach(ctx: Any, store: Any, *, owner: str = "", thread_id: str = "",
           on_beat: Callable[[dict], None] | None = None) -> Any:
    """把问答能力接到一个 ``RunContext`` 上，返回同一个 ctx（便于链式）。

    分开成一个函数而不是让调用方自己赋值：``ask_human`` 的签名与 ``ctx`` 的字段名
    是两处，接错了的症状是**没有症状** —— 字段还是 ``None``，问题还是发不出去，
    而那和「这条路本来就没有 HITL」在日志里长得一样。
    """
    ctx.ask_human = make_ask_human(store, owner=owner,
                                   thread_id=thread_id or getattr(ctx, "run_id", ""),
                                   on_beat=on_beat)
    return ctx


__all__ = ["make_ask_human", "attach"]
