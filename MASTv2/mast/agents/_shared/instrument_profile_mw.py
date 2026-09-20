"""InstrumentProfileMiddleware — inject the per-rig instrument mechanism +
config + learned dI/dV calibration into every model call.

Thin agent-layer wrapper around :mod:`mast.core.instrument_profile` (the data +
render layer). Kept separate so the skill layer can depend on the data module
without pulling in langchain / the agent stack.

Unlike :class:`ExperimentPrefsMiddleware` (#147, which no-ops on an empty
holder), this ALWAYS injects a block: the *mechanism* knowledge (退针分级 Z 自检
防撞针; 进针可用 dI/dV 判距离) is a safety-relevant instrument fact the agent
should always have when reasoning about 进/退针 — the operator's co-design ask
"辅以适当的上下文注入，使得 AI 对仪器有深入了解".
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from mast.agents._shared.inject import append_system_block
from mast.core.instrument_profile import format_profile_block, get_profile

#: 登记表里的条目 id（mast/prompts/registry.py）。常量而不是字面量：
#: 登记表要从这里派生，两边才不会各写一份。
PROMPT_ID = "mw.instrument_profile"

#: 哪些 agent 挂它。**真源在这里**，登记表与共享栈都从这个常量派生 ——
#: 让登记表去决定挂不挂，等于把「清单」变成「调度器」，而清单恰恰是这次
#: 发现归属标错了四条的地方。
AGENTS = ("instrument_control", "experiment_design")

logger = logging.getLogger(__name__)


class InstrumentProfileMiddleware(AgentMiddleware):
    """Append the instrument mechanism/config/calibration block to the system
    message on every model call (live-read from the process holder).

    Implements BOTH the sync and async ``wrap_model_call`` hooks — LangChain's
    async base raises NotImplementedError for a sync-only middleware, which
    would crash the ``python -m mast`` async dispatch path (see
    LiveStateMiddleware / ExperimentPrefsMiddleware)."""

    def __init__(self, get_profile_fn: "Callable[[], dict[str, Any]] | None" = None):
        super().__init__()
        self._get_profile = get_profile_fn or get_profile

    def _apply(self, request):
        try:
            profile = self._get_profile()
            block = format_profile_block(profile)
        except Exception as exc:  # never break a run over a profile read
            logger.debug("InstrumentProfileMiddleware render failed: %s", exc)
            return request
        return append_system_block(request, PROMPT_ID, block)

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))


__all__ = ["InstrumentProfileMiddleware"]
