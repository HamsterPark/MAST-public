"""按需加载工具：这一次调用给模型看哪些工具的 schema。

## 它做的事

``wrap_model_call`` 里把 ``request.tools`` 收窄成「核心包 ＋ 本轮已取的包」，
并在 system 消息末尾附一段**目录**告诉模型还有什么、怎么取。

## 它不做的事（三条，缺一条这就成了权限机制）

1. **不动 ToolNode。** 全部工具照常注册 —— 模型只要报得出名字，调用照常执行。
   收窄的只是「这一次它看得见谁的 schema」。
2. **不动任何安全面。** SafetyGate、validator 数值界、自主度策略、op 状态机
   一个字节没改。2026-08-20 的裁决（工具面全开 + 服务端包络裁决）说的是
   **权限**，这里改的是**可见性**：不给工具防不住模型换条路做，而真正拦住过
   事故的是包络。目录不是门禁。
3. **不回收。** 一次对话里取过的包只增不减 —— 既是为了别让模型刚看见就丢，
   也是为了 prompt cache：可见集单调增长，工具前缀才稳定。

## 为什么是 wrap_model_call 而不是 before_model

每个实现 ``before_model``/``after_model`` 的中间件都会变成图里**自己的一个
节点**，而私聊的 ``recursion_limit`` 数的正是 super-step。六个中间件曾把一回合
从 5 步推到 11 步（见 ``message_clock_mw`` 的同款说明）。这里只读状态、只改请求，
``wrap_model_call`` 足够。

## 收窄合法吗

``langchain/agents/factory.py::_get_bound_model`` 只校验中间件**加进来**的工具
是不是 ToolNode 认识的；**缩小**到已注册的子集永远合法。（而且 IC 挂着
SafetyGate 的 ``wrap_tool_call``，那道校验本来就被跳过 —— 更要自律只做子集。）
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware

from mast.agents._shared import tool_packs as _tp
from mast.agents._shared.inject import append_system_block

logger = logging.getLogger(__name__)

#: 登记表条目 id。
PROMPT_ID = "mw.tool_index"

#: state 里记「本轮取了哪些包」的字段名（reducer 是 dedupe_append：只增不减）。
STATE_KEY = "loaded_tool_packs"

#: 工具数少于这个值就不值得分包 —— 目录本身要花几百字符，而省下来的还没它多。
#: IC 是 394，其余 agent 是 8–42，所以实际上只有 IC 走这条路。
MIN_TOOLS = 60


class ToolVisibilityMiddleware(AgentMiddleware):
    """Narrow the visible tool schemas to core + whatever this run has loaded."""

    def __init__(self, catalog: _tp.Catalog, *, enabled: bool = True):
        super().__init__()
        self._catalog = catalog
        self._enabled = bool(enabled) and len(catalog.packs_by_tool) >= MIN_TOOLS

    @property
    def name(self) -> str:
        return "ToolVisibilityMiddleware"

    @property
    def catalog(self) -> _tp.Catalog:
        return self._catalog

    def _loaded(self, request: Any) -> list[str]:
        try:
            state = getattr(request, "state", None) or {}
            got = state.get(STATE_KEY) if hasattr(state, "get") else None
            return [str(p) for p in (got or [])]
        except Exception:  # noqa: BLE001
            return []

    def _apply(self, request: Any) -> Any:
        if not self._enabled:
            return request
        try:
            tools = list(getattr(request, "tools", None) or [])
            if not tools:
                return request
            loaded = self._loaded(request)
            visible = self._catalog.visible(loaded)
            kept = [t for t in tools
                    if str(getattr(t, "name", "") or "") in visible
                    or not getattr(t, "name", None)]
            if not kept or len(kept) >= len(tools):
                # 全都留下 = 没省下什么，也就不必贴目录（比如某个 agent 的工具
                # 恰好全在核心里）。
                return request
            index = _tp.render_index(self._catalog, loaded)
            override = getattr(request, "override", None)
            if callable(override):
                try:
                    request = override(tools=kept)
                except Exception:  # noqa: BLE001
                    request.tools = kept
            else:
                request.tools = kept
            if index:
                request = append_system_block(request, PROMPT_ID, index)
            return request
        except Exception:  # noqa: BLE001 — 收窄失败就全都给，绝不弄坏一次 run
            logger.debug("ToolVisibilityMiddleware skipped", exc_info=True)
            return request

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._apply(request))


def make_tool_visibility_middleware(agent: str, tools: Any, registry: Any = None,
                                    *, enabled: bool = True):
    """``(middleware, catalog)``；``enabled=False`` 或工具太少时中间件是 no-op。"""
    catalog = _tp.build_catalog(agent, tools, registry)
    return ToolVisibilityMiddleware(catalog, enabled=enabled), catalog


__all__ = [
    "MIN_TOOLS", "PROMPT_ID", "STATE_KEY",
    "ToolVisibilityMiddleware", "make_tool_visibility_middleware",
]
