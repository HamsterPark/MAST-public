"""给 agent 的每条消息盖一个**发生时刻**，让转录和旁白排得到一起去。

## 旁白排序诊断

症状:旁白的输出顺序有时看起来混乱,有时候连续输出几个 pulse,有时候连续
输出扫图,不确定是输出本身有问题还是顺序有问题。

查下来:执行顺序没问题(``completed_steps`` 里 A(脉冲)→B(验证) 严格交替)、
旁白的存储顺序也没问题(53 条,``seq`` 序与 ``t`` 序零逆序)。错在**两个发出方
对同一时刻的观测有时差** —— 那一半在 ``chat/narration.py`` 的 ``event_t`` 修了。

另一半要求:**agent 的发言也带上时间**才更完整。

而这正是 ``frontend/src/lib/narration.ts`` 里写了很久的那句话:

    ⚠️ 已知近似……要让它变精确,**得给 render_history 的每条消息加稳定时间戳**,
       那是另一个改动。

这个模块就是那个改动。

## 为什么盖在 ``additional_kwargs`` 上

因为它是**唯一**能随消息一起进 checkpointer、并在重启后仍然跟着那条消息的
地方。本仓已经这么用过:``compaction_mw`` 把压缩事件盖在摘要消息上
(``COMPACTION_META_KEY``),跨 6 家 provider 跑过 —— 所以这不是一条新路。

## 「不知道」和「等于 0」是两句话

⚠️ **进程重启前就存在的那些消息,这里一律不盖。**

一条从 checkpoint 里读回来的旧消息,我们**不知道**它是什么时候说的。给它盖
``time.time()`` 会造出一段「所有历史消息都发生在重启那一秒」的假历史 —— 而
前端会把它当真的画出来。所以:

  · 每次模型调用**只盖它确定新增的那些** —— 判据是这个 thread 上一次看到的
    消息条数(``_seen``);
  · 第一次见到某个 thread ⇒ **只记基线,一条都不盖**;
  · 而这一轮模型**真正返回**的那条由 ``_stamp_result`` 盖 —— 它的时刻是确定的
    (handler 刚刚返回它),所以它**不受基线约束**;
  · 已经盖过的绝不重盖 —— 否则每次刷新都会把历史改写成「刚刚」。

## 它对 recursion 预算的成本是 0

钩子用的是 ``wrap_model_call`` 而**不是** ``after_model`` —— 后者会变成图里
**自己的一个节点**,而私聊的 ``recursion_limit`` 数的正是 super-step。
详见下面 hooks 那一段;这条由测试钉住,不是靠记得。

没盖到的消息在前端**不显示时间**(``fmtClock`` 对无效值返回空串),
不显示 1970。

## 它绝不能弄坏一次对话

全程 best-effort:任何异常只会让这条消息少一个时间戳,不会让回合失败。
与本仓其它 provenance 写入同一条纪律。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover — 上游缺席时这个中间件整个不挂
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # noqa: BLE001
    AgentMiddleware = object  # type: ignore[assignment,misc]

#: 时刻盖在 ``additional_kwargs`` 的这个键下。
#:
#: 字面量而不是从别处 import:``chat/render.py`` 要读它,而 render 那一侧刻意
#: 不依赖 agent 包(与 ``COMPACTION_META_KEY`` 同一条规矩)。两处拼写由
#: ``tests/v2/unit/agents/test_message_clock.py`` 的平价测试钉在一起。
MESSAGE_TIME_KEY = "mast_t"

#: 单次模型调用最多补盖多少条入站消息。
#:
#: 一次 model call 之后新增的消息 = 若干 ToolMessage + 一条 AIMessage,
#: 而工具并发上限远小于这个数。设上限是为了让「基线算错」这种错**有界**:
#: 最坏情况是给最近 64 条盖上略晚的时刻,而不是把整段历史改写成「刚刚」。
_MAX_STAMP_PER_CALL = 64


def message_time(msg: Any) -> "float | None":
    """这条消息上盖着的时刻,没有就是 ``None``。

    ``None`` = **不知道**,不是 0。调用方必须把这两种区分开 —— 一个 0 会被
    渲染成 1970,而一个假时间比没有时间坏(它会被当成真的去推理)。
    """
    kw = getattr(msg, "additional_kwargs", None)
    if not isinstance(kw, dict):
        return None
    try:
        val = float(kw.get(MESSAGE_TIME_KEY))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if val != val or val <= 0.0 or val == float("inf"):
        return None
    return val


def _stamp(msg: Any, t: float) -> bool:
    """盖一个时刻。已经有了就不动。返回「这次真的盖了吗」。"""
    kw = getattr(msg, "additional_kwargs", None)
    if not isinstance(kw, dict):
        return False
    if message_time(msg) is not None:
        return False
    kw[MESSAGE_TIME_KEY] = t
    return True


class MessageClockMiddleware(AgentMiddleware):  # type: ignore[misc,valid-type]
    """给这一轮新产生的消息盖上 wall-clock 时刻。

    挂在哪个 agent 上都行,状态按 thread 分开记(``_seen``)。
    """

    def __init__(self) -> None:
        try:
            super().__init__()
        except Exception:  # noqa: BLE001 — AgentMiddleware 缺席时的替身是 object
            pass
        #: thread_id → 上次见到的消息条数。**不是全局计数** —— 两个会话交替
        #: 跑的时候,共用一个计数会让其中一个的基线永远偏。
        self._seen: dict[str, int] = {}

    # 必须使用 wrap_model_call / awrap_model_call，不新增 before_model 或 after_model 节点。
    # recursion_limit 计数 super-step；独立节点会增加每轮预算消耗，而 wrap 钩子不增加节点。
    # 同步与异步钩子都需实现，避免基类异步默认实现拒绝调用。
    # 无额外步数的契约由 test_the_clock_middleware_costs_no_super_steps 验证。
    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        self._stamp_request(request)
        result = handler(request)
        self._stamp_result(result)
        return result

    async def awrap_model_call(self, request: Any,
                               handler: Callable[[Any], Any]) -> Any:
        self._stamp_request(request)
        result = await handler(request)
        self._stamp_result(result)
        return result

    # ── the work ─────────────────────────────────────────────────────

    @staticmethod
    def _thread_of(runtime: Any) -> str:
        """取得当前转录的计数键；无法取得时返回空串。
        
        优先兼容 runtime 上可能存在的 config.configurable.thread_id 或 configurable。
        真实 Runtime 未必提供这些属性；turn_context 的可用字段是 conversation_id，
        不能把不存在的 thread_id 当成有效来源。按 conversation_id 隔离 _seen 基线，
        避免新会话继承其他会话的消息计数而遗漏首条消息时间戳。
        时间戳还用于把上下文注入与对应模型请求关联，不能用全局计数替代会话键。"""
        for path in (("config", "configurable", "thread_id"),
                     ("configurable", "thread_id")):
            cur: Any = runtime
            for key in path:
                if cur is None:
                    break
                cur = (cur.get(key) if isinstance(cur, dict)
                       else getattr(cur, key, None))
            if cur:
                return str(cur)
        try:
            from mast.core.turn_context import current_turn

            turn = current_turn()
            return str(turn.get("thread_id") or turn.get("conversation_id") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _stamp_request(self, request: Any) -> None:
        """给**这一轮新增的**入站消息盖时刻(主要是刚返回的 ToolMessage)。

        ``request.messages`` 是即将发给模型的那一份,历史在前、新的在后。
        基线(``_seen``)保证不会往回盖到重启前的历史 —— 见模块自述。
        """
        try:
            msgs = list(getattr(request, "messages", None) or [])
        except Exception:  # noqa: BLE001 — request 不是预期形状
            return
        if not msgs:
            return
        try:
            thread = self._thread_of(getattr(request, "runtime", None))
            seen = self._seen.get(thread)
            # **末尾那条 HumanMessage 无条件盖**,不受基线约束。
            #
            # 模型此刻正被调用,而消息列表以一条用户消息结尾 ⇒ 那就是触发这一轮
            # 的那条,它**就是刚刚发生的**。这不是猜。
            #
            # 2026-08-23 从「只在首次见到这个 thread 时盖」提上来:基线键一旦
            # 算错(它整整算错了一段时间,见 ``_thread_of``),``start`` 会直接
            # 跳到列表末尾,``msgs[start:]`` 空,于是用户自己刚发的那句
            # **一个戳都拿不到** —— 而那正是 ``InjectedContext`` 用来对齐
            # 「这一轮往模型那儿塞了什么」的唯一钥匙。
            #
            # 提上来是安全的:``_stamp`` 从不覆盖已有的戳,而这一条的时刻是
            # 全流程里最确定的两个之一(另一个是模型刚返回的那条)。
            tail = msgs[-1]
            if type(tail).__name__ == "HumanMessage":
                _stamp(tail, time.time())
            if seen is None:
                # 第一次见到这条转录:前面那些是从 checkpoint 读回来的,
                # **我们不知道它们是什么时候说的** —— 一条都不盖。
                # (这一轮真正新产生的那条 AIMessage 由 _stamp_result 负责,
                #  那一条的时刻是**确定**的:模型刚刚返回它。)
                self._seen[thread] = len(msgs)
                return
            start = max(0, min(seen, len(msgs)))
            start = max(start, len(msgs) - _MAX_STAMP_PER_CALL)
            self._stamp_all(msgs[start:], thread)
            self._seen[thread] = len(msgs)
        except Exception as exc:  # noqa: BLE001 — 一个时间戳绝不许弄坏一个回合
            logger.debug("message clock (request) failed: %s", exc)

    def _stamp_result(self, result: Any) -> None:
        """给模型**刚刚返回的**那条(些)消息盖时刻。

        这一条的时刻是全流程里最确定的:handler 刚返回,现在就是它产生的时刻。
        所以它**不受基线约束** —— 基线防的是「把历史写成刚刚」,而这条本来就是刚刚。

        返回值的形状随上游版本变(ModelResponse / AIMessage / list / dict),
        所以这里按几种已知形状找,而不是硬取某个属性:取错了会是**静默**的
        (没有异常,只是永远盖不上),而那正是本仓最贵的一类失效。
        找不到就记一条 debug —— 让形状变更**可发现**。
        """
        try:
            found = self._messages_in(result)
            if not found:
                logger.debug("message clock: no message found in model result "
                             "(%s) — 上游返回形状可能变了", type(result).__name__)
                return
            self._stamp_all(found, "<model-result>")
        except Exception as exc:  # noqa: BLE001
            logger.debug("message clock (result) failed: %s", exc)

    @staticmethod
    def _messages_in(obj: Any, _depth: int = 0) -> list:
        """从模型返回值里挖出消息对象。已知的几种形状都试。"""
        if obj is None or _depth > 2:
            return []
        if isinstance(getattr(obj, "additional_kwargs", None), dict):
            return [obj]                                   # 直接就是一条消息
        if isinstance(obj, (list, tuple)):
            out: list = []
            for x in obj:
                out.extend(MessageClockMiddleware._messages_in(x, _depth + 1))
            return out
        for attr in ("result", "messages", "message", "output"):
            val = (obj.get(attr) if isinstance(obj, dict)
                   else getattr(obj, attr, None))
            if val is not None:
                found = MessageClockMiddleware._messages_in(val, _depth + 1)
                if found:
                    return found
        return []

    @staticmethod
    def _stamp_all(msgs, where: str) -> None:
        now = time.time()
        n = sum(1 for m in msgs if _stamp(m, now))
        if n:
            logger.debug("message clock: stamped %d message(s) @%s", n, where)


__all__ = ["MESSAGE_TIME_KEY", "MessageClockMiddleware", "message_time"]
