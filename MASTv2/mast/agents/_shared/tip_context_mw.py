"""TipContextMiddleware — 当前针尖 + 信号链约定,注入每一次 model call。

Agent 层的薄包装,数据与渲染在 :mod:`mast.core.tip_state`(与
``instrument_profile_mw`` 同样的分层理由:让 skill 层能依赖数据模块而不必拖进
langchain)。

**为什么挂共享中间件栈而不是逐个 graph**:``InstrumentProfileMiddleware`` 只接
在 instrument_control 与 experiment_design 两个 graph 上,而针尖块里最贵的一条
——偏压加在样品还是针尖——恰恰是 **data_processing** 解释 dI/dV 谱时最需要的。
挂进 ``runtime._chat_agent_middleware`` 一处即覆盖主群聊、后台群聊与私聊三条路。

**永远注入**(与 experiment_prefs 空则 no-op 相反):「针尖未登记」「偏压极性
未声明」本身就是模型需要知道的事实。不说,它就会按最常见的约定默认下去,而那
正是解释谱图时最贵的一类错 —— 而且在数据里完全看不出来。
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from mast.agents._shared.inject import append_system_block
from mast.core.instrument_profile import get_profile
from mast.core.tip_state import format_tip_block, get_current_tip

#: 登记表条目 id。
PROMPT_ID = "mw.tip_context"

#: 哪些 agent 挂它。**这是真源** —— 共享中间件栈与登记表都从这里派生。
#:
#: 2026-08-24 从「全部 7 个」收到这三个。原来的理由（偏压加在样品还是针尖，
#: 是 data_processing 解释 dI/dV 谱时最需要的一条）**仍然成立，而且正是
#: 这三个的名单来源**：IC 动针、XD 定参数、DP 读谱。paper_writing /
#: paper_review / research_director / literature 不碰针尖也不读谱，它们
#: 每一轮为这 483 个字符付钱，换来的信息一次也用不上。
#:
#: 「针尖未登记」这条 always-inject 的纪律没变，只是不再广播给全体。
AGENTS = ("instrument_control", "experiment_design", "data_processing")

logger = logging.getLogger(__name__)


class TipContextMiddleware(AgentMiddleware):
    """Append the current-tip + signal-chain block to the system message.

    Implements BOTH sync and async ``wrap_model_call`` — LangChain's async base
    raises NotImplementedError for a sync-only middleware, which would crash the
    ``python -m mast`` async dispatch path (same note as
    InstrumentProfileMiddleware / LiveStateMiddleware).
    """

    def __init__(self, get_tip_fn=None, get_profile_fn=None):
        super().__init__()
        self._get_tip = get_tip_fn or get_current_tip
        self._get_profile = get_profile_fn or get_profile

    def _apply(self, request):
        try:
            block = format_tip_block(self._get_tip(), self._get_profile())
        except Exception as exc:  # noqa: BLE001 — 读针尖失败绝不能弄坏一次 run
            logger.debug("TipContextMiddleware render failed: %s", exc)
            return request
        return append_system_block(request, PROMPT_ID, block)

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))


__all__ = ["TipContextMiddleware"]
