"""上下文压缩 —— 摆脱 ``SummarizationMiddleware`` 的私有 API。

为什么这是全案最难的单件
------------------------
今天的 ``compaction_mw._MemorySinkSummarization`` 继承 langchain 的
``SummarizationMiddleware``，并**匹配它的私有实现细节**：``_create_summary`` /
``_build_new_messages`` 的覆写、``additional_kwargs["lc_source"] ==
"summarization"`` 这个内部标记、以及上游那句英文前言的字面量。这是全仓对上游内部
实现最脆的一处依赖——上游改一次措辞，标记就认不出来，而症状是「压缩静默地不再可见」。

上游替我们做对的四件事，这里逐条复刻
------------------------------------
1. **绝对 token 阈值**。``("fraction", x)`` 触发器要读 ``model.profile
   ["max_input_tokens"]``，而四家 OpenAI-compat provider（Kimi/DeepSeek/Qwen/GLM）
   根本不暴露它——用 fraction 会在**构造期**就抛。所以阈值从
   ``config.model_input_context`` 的保守窗口表算，复用既有的
   :func:`~mast.agents._shared.compaction_mw.compaction_trigger_tokens`（那是个纯
   函数，与 SummarizationMiddleware 无关，没有理由重写一遍）。
2. **pair-safe 截断**。切点绝不能落在 ``AIMessage(tool_calls=…)`` 与它的
   ``ToolMessage`` 之间——留下孤儿会让 provider 400。见 :func:`pair_safe_cut`。
3. **失败哨兵**。摘要器出错时上游返回的是一个**错误字符串**而不是抛异常。拿它当摘要
   去替换历史，等于因为一次摘要器打嗝就把整段实验上下文（做过什么、哪些区域禁止、
   计划进行到哪）不可逆地抹掉。所以摘要失败 = **不压缩**。
4. **压缩必须可见**。上游压缩是静默的：它返回的 state 更新里没有任何东西说明刚刚
   替换掉了多少历史，于是群聊桥把摘要当成「用户回声」丢掉了，用户翻看长对话时
   读到的是一份被系统改写过、却没有任何标记的转录。

这一版**多做对的一件事**：压缩不再销毁历史
-------------------------------------------
上游靠 ``RemoveMessage(REMOVE_ALL_MESSAGES)`` 把历史从通道里真的删掉，而那条通道是
唯一副本——所以在旧世界里「压缩」约等于「销毁」。这里压缩只改**这一轮送给模型的
列表**；全量历史留在 ``chat_messages`` 表里（只追加、永不裁剪），随时读得回来。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from mast.agentruntime.middleware import Middleware, TurnView

logger = logging.getLogger(__name__)

#: 与 ``compaction_mw.COMPACTION_META_KEY`` 必须**同字面量**：群聊 SSE 桥读的是
#: 这个字符串而不是 import 这个模块（桥刻意不带重型 agent import）。新旧引擎产出的
#: 标记要长得一样，前端才不用认两种。
COMPACTION_META_KEY = "mast_compaction"

#: 摘要器失败时返回的哨兵前缀。**拿这些当摘要去替换历史 = 用一句报错抹掉整段实验
#: 上下文。** 沿用上游的字面量，因为一期的摘要器可能就是上游那一个。
_FAILURE_SENTINELS = (
    "Error generating summary",
    "Previous conversation was too long",
    "No previous",
)

#: 默认保留多少条最近消息不压缩。太少会把当前任务的上下文也摘掉；太多则压不动。
DEFAULT_KEEP_RECENT = 8


def estimate_tokens(messages: Iterable[Any]) -> int:
    """粗估这批消息的 token 数。

    **是估计，而且要一直被称为估计。** 这个数字唯一的用途是决定「要不要压缩」，
    在触发点上没有任何 provider 报告的 prompt-token 数可用。优先用 langchain 的近似
    计数器（与今天触发器用的是同一个，两代引擎的触发点因此可比），拿不到就退回
    「四个字符一个 token」——中文更接近 1.5 字符/token，所以这个退路**偏保守**
    （偏早压缩），而不是偏晚。
    """
    msgs = list(messages)
    try:
        from langchain_core.messages.utils import count_tokens_approximately

        return int(count_tokens_approximately(msgs))
    except Exception:  # noqa: BLE001 — 二期这个 import 会消失
        total = 0
        for m in msgs:
            content = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
            total += len(str(content or "")) // 4 + 8
        return total


def _is_tool_message(msg: Any) -> bool:
    return getattr(msg, "type", "") == "tool" or (
        isinstance(msg, dict) and msg.get("role") == "tool")


def _has_tool_calls(msg: Any) -> bool:
    if isinstance(msg, dict):
        return bool(msg.get("tool_calls"))
    return bool(getattr(msg, "tool_calls", None))


def pair_safe_cut(messages: list, desired: int) -> int:
    """把切点往**前**挪到不劈开 tool 配对的位置，返回安全切点。

    两条约束，都是为了不留孤儿：

    * 切点上不能是一条 ``ToolMessage``——那意味着它的 ``AIMessage`` 被摘要走了，
      而它自己留在了送给模型的列表里（孤儿 tool result → provider 400）；
    * 切点**前一条**不能是带 ``tool_calls`` 的 ``AIMessage``——那意味着调用被摘要
      走了、结果却留着（同一个孤儿，反过来看）。

    往前挪（而不是往后）保证收敛：最坏挪到 0，也就是这一轮不压缩。往后挪会越切越多，
    在一串长工具往返上可能把整个工作集吃掉。
    """
    cut = max(0, min(int(desired), len(messages)))
    while cut > 0:
        lands_on_tool_result = cut < len(messages) and _is_tool_message(messages[cut])
        splits_a_call_batch = _has_tool_calls(messages[cut - 1])
        if lands_on_tool_result or splits_a_call_batch:
            cut -= 1
            continue
        break
    return cut


def is_failed_summary(summary: str) -> bool:
    return bool(summary) and str(summary).startswith(_FAILURE_SENTINELS)


class CompactionMiddleware(Middleware):
    """按 token 阈值压缩上下文。

    ``summarizer`` 收一批消息、返回一段摘要文本。刻意是个**函数**而不是一个模型：
    这样测试可以脚本化它，而生产可以让它走 ``make_chat_model``（六家 provider 任意
    一家）——摘要器与主模型不必是同一个。
    """

    def __init__(self, *, summarizer: Callable[[list], str],
                 trigger_tokens: int, keep_recent: int = DEFAULT_KEEP_RECENT,
                 memory_sink: Callable[[str], None] | None = None,
                 agent_id: str = ""):
        self._summarize = summarizer
        self._trigger = max(1, int(trigger_tokens))
        self._keep = max(1, int(keep_recent))
        self._memory_sink = memory_sink
        self._agent_id = agent_id

    @property
    def name(self) -> str:
        return f"Compaction({self._agent_id})" if self._agent_id else "Compaction"

    def before_model(self, turn: TurnView) -> None:
        messages = list(turn.messages)
        if len(messages) <= self._keep:
            return
        before_tokens = estimate_tokens(messages)
        if before_tokens < self._trigger:
            return

        cut = pair_safe_cut(messages, len(messages) - self._keep)
        if cut <= 0:
            # 挪到 0 = 这一轮没有安全切点（比如整段都是一串未闭合的工具往返）。
            # 不压缩比留个孤儿好：后者会让下一次请求直接 400。
            logger.info("[%s] compaction skipped: no pair-safe cut point",
                        self._agent_id or "?")
            return

        to_summarize, preserved = messages[:cut], messages[cut:]
        try:
            summary = str(self._summarize(to_summarize) or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] summariser raised (%s); NOT compacting",
                           self._agent_id or "?", exc)
            return

        if not summary.strip() or is_failed_summary(summary):
            # ★ 摘要失败 = 不压缩。用一句报错替换历史，等于因为摘要器打嗝就把整段
            # 实验上下文不可逆地抹掉。
            logger.warning("[%s] summariser returned a failure sentinel; NOT compacting",
                           self._agent_id or "?")
            return

        self._tee_to_memory(summary)
        stamped = self._summary_message(summary, removed=len(to_summarize),
                                        kept=len(preserved),
                                        before=len(messages),
                                        tokens_before=before_tokens)
        turn.replace_messages([stamped, *preserved],
                              removed=len(to_summarize), kept=len(preserved))
        logger.info("[%s] compaction fired: %d message(s) → summary, %d kept "
                    "(≈%d tokens before)", self._agent_id or "?",
                    len(to_summarize), len(preserved), before_tokens)

    # ── 内部 ───────────────────────────────────────────────────────────
    def _summary_message(self, summary: str, *, removed: int, kept: int,
                         before: int, tokens_before: int):
        """带 ``mast_compaction`` 章的摘要消息。

        用 ``SystemMessage`` 而不是 ``HumanMessage``：上游用后者，于是群聊桥把它
        分类成「用户回声」丢掉了——摘要不是要求的话，把它放进 human 通道
        本来就是错的。
        """
        from langchain_core.messages import SystemMessage

        return SystemMessage(
            content=summary,
            additional_kwargs={COMPACTION_META_KEY: {
                "removed": removed, "kept": kept, "before": before,
                # 名字里带 estimate，因为它**是**估计——触发点上拿不到 provider
                # 报告的 prompt-token 数。
                "tokens_before_estimate": tokens_before,
                "trigger_tokens": self._trigger,
                "summary": summary,
                "engine": "agentruntime",
            }})

    def _tee_to_memory(self, summary: str) -> None:
        if not self._memory_sink:
            return
        try:
            self._memory_sink(summary)
        except Exception as exc:  # noqa: BLE001 — 记忆写入绝不许毁掉一轮对话
            logger.debug("compaction memory_sink failed: %s", exc)


def trigger_tokens_for(model_id: str) -> int:
    """该模型的压缩阈值。

    复用 ``compaction_mw.compaction_trigger_tokens`` —— 它是个**纯函数**（读
    ``config.model_input_context`` 的窗口表），与 ``SummarizationMiddleware`` 毫无
    关系。重写一遍只会制造第二个真源，而两个窗口表迟早会各说各话。
    """
    from mast.agents._shared.compaction_mw import compaction_trigger_tokens

    return compaction_trigger_tokens(model_id)


__all__ = [
    "CompactionMiddleware",
    "COMPACTION_META_KEY",
    "estimate_tokens",
    "pair_safe_cut",
    "is_failed_summary",
    "trigger_tokens_for",
    "DEFAULT_KEEP_RECENT",
]
