"""共享中间件栈：私聊 / 主群聊 / 后台群聊三条路共用的那一张表。

原来它是 ``CoreRuntime._chat_agent_middleware`` 的方法体。抽成纯函数有两个理由：

1. **测试建得出完整的表。** 方法体里 ``make_chat_model(who)`` 在没有 API key 时
   抛异常，被 ``except`` 吞掉 —— 于是单测里拿到的永远是**缺了 compaction 与
   tool_refine 的残表**，而断言照样绿。这是「替身太顺让负例恒绿」的镜像：替身
   太残，让正例恒假，闸门于是钉不住真实的挂载顺序。
2. **归属有个能被核对的地方。** ``mast/prompts/manifest.py`` 要回答「哪个 agent
   收到哪些块」，它需要一个不必先起一整个 CoreRuntime 就能问的函数。

## 定向注入（2026-08-24）

以前这张表上的每一项都发给全部七个 agent。``TipContextMiddleware`` 因此每轮给
paper_review / paper_writing / research_director 各塞 483 字符的针尖状态 —— 它们
不碰针尖也不读谱，这段信息一次也用不上。

现在按**中间件自己声明的** ``AGENTS`` 常量筛。真源在中间件里，不在登记表里：
让登记表决定挂不挂，等于把「清单」变成「调度器」，而清单恰恰是这次发现归属
标错了四条的地方。没有 ``AGENTS`` 属性 = 全员（compaction / tool_refine /
memory / readback 都是真正的全员块）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def declared_agents(mw: Any) -> tuple[str, ...] | None:
    """这个中间件声明它发给哪些 agent；``None`` = 没声明 = 全员。

    约定是**模块级常量 ``AGENTS``**（``tip_context_mw.AGENTS`` 等），不是类属性
    —— 登记表 ``mast/prompts/registry.py`` 也从同一个常量派生，一个真源两个读者。
    类属性也认，方便将来某个中间件需要按实例决定。
    """
    for holder in (mw, type(mw)):
        got = getattr(holder, "AGENTS", None)
        if got:
            return tuple(got)
    import sys as _sys
    mod = _sys.modules.get(type(mw).__module__)
    got = getattr(mod, "AGENTS", None)
    return tuple(got) if got else None


def applies_to(mw: Any, agent_id: str | None) -> bool:
    """这个中间件该发给 *agent_id* 吗？

    ``agent_id=None`` 出现在两处：按编排器尺寸建的主聊天表，以及离线构造。
    两处都按「全员块照给、定向块按 IC 处理」办 —— 主聊天实际跑的就是 IC。
    """
    agents = declared_agents(mw)
    if not agents:
        return True
    who = agent_id or "instrument_control"
    return who in agents


def build_shared_middleware(
    agent_id: str | None,
    *,
    model_id: str,
    summarizer: Any,
    cognition: Any = None,
    namespace_provider: Callable[[], Any] | None = None,
    memory_sink: Callable[[str], None] | None = None,
    settings: Any = None,
) -> list:
    """The one list shared by 私聊 + 主群聊 + 后台群聊，for *agent_id*.

    Args:
        agent_id: 真正的 agent id；None = 按编排器尺寸（主聊天）。
        model_id: 这个 agent **生效**的模型 id（``resolve_effective_model_id``，
            不是 ``get_model_id``）—— compaction 的触发点按它算。
        summarizer: 压缩 / 精炼用的小模型。
        cognition: CognitionContext；None = 不挂记忆召回。
        namespace_provider: 记忆命名空间（通常是当前实验 id）。
        memory_sink: 压缩摘要的落库回调。
        settings: SettingsStore-like（``.get(key)``）。
    """
    mw: list = []

    def _setting(key: str, default: Any = None) -> Any:
        if settings is None:
            return default
        try:
            v = settings.get(key)
        except Exception:  # noqa: BLE001
            return default
        return default if v is None else v

    # ── 压缩（before_model）────────────────────────────────────────────
    try:
        from mast.agents._shared.compaction_mw import make_compaction_middleware
        mw.append(make_compaction_middleware(
            model_id=model_id, summarizer_model=summarizer,
            memory_sink=memory_sink))
    except Exception as exc:  # noqa: BLE001
        logger.debug("compaction middleware unavailable: %s", exc)

    # ── 逐轮工具返回精炼（before_model）────────────────────────────────
    try:
        if bool(_setting("tool_refine_enabled", True)):
            try:
                min_chars = int(_setting("tool_refine_min_chars", 600) or 600)
            except (TypeError, ValueError):
                min_chars = 600
            from mast.agents._shared.tool_refine_mw import ToolRefinementMiddleware
            mw.append(ToolRefinementMiddleware(
                summarizer_model=summarizer, min_chars=max(1, min_chars)))
    except Exception as exc:  # noqa: BLE001
        logger.debug("tool refine middleware unavailable: %s", exc)

    # ── 记忆召回 ──────────────────────────────────────────────────────
    if cognition is not None:
        try:
            from mast.agents._shared.memory_mw import make_memory_recall_middleware
            mw.append(make_memory_recall_middleware(
                cognition, namespace_provider=namespace_provider))
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory recall middleware unavailable: %s", exc)

    # ── ⑦ 心愿单回程 ──────────────────────────────────────────────────
    # 取文板的回程也走这里，但只给 literature —— 一篇论文到货不该打断仪器对话。
    try:
        from mast.agents._shared.request_readback_mw import (
            make_request_readback_middleware,
        )
        mw.append(make_request_readback_middleware(
            fetch_for_agent=("literature" if (agent_id or "") == "literature" else "")))
    except Exception as exc:  # noqa: BLE001
        logger.debug("request-readback middleware unavailable: %s", exc)

    # ── 当前针尖 + 信号链（定向：IC / XD / DP）──────────────────────────
    try:
        from mast.agents._shared.tip_context_mw import TipContextMiddleware
        candidate = TipContextMiddleware()
        if applies_to(candidate, agent_id):
            mw.append(candidate)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tip context middleware unavailable: %s", exc)

    return mw


__all__ = ["applies_to", "build_shared_middleware", "declared_agents"]
