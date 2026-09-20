"""从各 agent 现有的零件装配一个 :class:`~mast.agentruntime.loop.AgentLoop`。

为什么装配住在这里，而不是各 agent 的 ``graph.py`` 里
----------------------------------------------------
strangler 迁移期间新旧两条路要**并存**。把 v2 装配写进六个 ``graph.py``，等于在六个
正在被日常工作修改的文件里各开一个分叉；写在这里，v2 只**读**它们的公共零件：

======================  ==========================================
``tools.py``            ``build_tools(buf, …)``
``prompts.py``          ``SYSTEM_PROMPT``
提示词覆写              ``resolve_prompt("agent.<id>.system", …)``
模型                    ``models.make_chat_model(agent_id, …)``
======================  ==========================================

六个可委托 agent 提供一致的公共接口，使用同一份通用装配；
``instrument_control`` **不在此列**——它不可委托，而且它的装配带着一整套硬件专属
中间件（SafetyGate / BufferHITL / AlertDelivery / ModeBelief），由 :mod:`mast.agentruntime.ic_assembly` 单独装配。

一期的边界
----------
工具仍然是 langchain 的 ``BaseTool``，经 ``spec_from_langchain_tool`` 包成
``ToolSpec``。它们因此拿不到显式的 ``RunContext``——这不是遗漏，是分期：156 个
``@tool`` 照旧工作，等 ``wrap_skill`` 移植过来时才换成 ``fn(args, ctx)``。

中间件走桥，不重抄
------------------
中间件应与既有实现共享行为，因此经 :mod:`mast.agentruntime.compat` 把既有的
``AgentMiddleware`` 实例桥进新栈——一行都不碰那些文件，逻辑一个字不改。

``with_middleware=True``（默认）会挂上与旧路径**同一份**通用中间件表；等工作树落定、
新运行时在生产上烤熟之后，再逐个把桥拆成原生实现，那时每拆一个都有对照测试兜着。
桥接过的中间件在栈里叫 ``lc:<原名>``，所以「还剩几条挂在桥上」是可读的。
"""
from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Iterable

from mast.agentruntime.loop import AgentLoop, CallLimits
from mast.agentruntime.middleware import Middleware
from mast.agentruntime.model import ChatModelPort, LangChainModelPort
from mast.agentruntime.tools import ToolSpec, spec_from_langchain_tool

logger = logging.getLogger(__name__)

#: 与 ``skills/composite/agent_node.DELEGATABLE_AGENTS`` 同源。
#: **刻意不 import 它**：那个常量表达的是「工作流可以委托谁」（一条能动性策略），
#: 这里表达的是「哪些 agent 的零件形状被核对过」（一条实现事实）。两者今天重合，
#: 但它们会因为不同的理由变化——把一个当成另一个的真源，是本仓记过的
#: 「名单抄第二遍就会漂」的另一面。
ASSEMBLABLE_AGENTS = ("literature", "data_processing", "experiment_design",
                      "paper_writing", "paper_review", "research_director")


def _system_prompt(agent_id: str) -> str:
    """agent 的系统提示词，**经过用户覆写层**。

    直接读 ``prompts.SYSTEM_PROMPT`` 会绕过 ``resolve_prompt`` —— 那样用户在界面上
    改的提示词对 v2 引擎无效，而界面上看起来一切正常。
    """
    mod = importlib.import_module(f"mast.agents.{agent_id}.prompts")
    base = getattr(mod, "SYSTEM_PROMPT", "")
    try:
        from mast.prompts.registry import resolve as resolve_prompt

        resolved = resolve_prompt(f"agent.{agent_id}.system", base)
    except Exception as exc:  # noqa: BLE001 — 覆写层坏了也要能跑
        logger.warning("prompt override unavailable for %s: %s", agent_id, exc)
        resolved = base
    return resolved + _extra_prompt_blocks(agent_id)


def _extra_prompt_blocks(agent_id: str) -> str:
    """基础提示词之外，旧路径还往后拼的那些块。

    ★ 为什么需要它（2026-08-27 实测发现）
    -----------------------------------
    拿两个引擎给同一个 agent 装出来的提示词逐个对字符数，六个里五个完全一致，
    **``literature`` 少 187 字符** —— 少的正是
    ``agent.literature.unavailable_note``：那段告诉模型「哪些工具当前不可用、
    为什么」。

    旧路径把它拼在 ``SYSTEM_PROMPT`` 后面（``literature/graph.py``：
    ``system_prompt = _base_prompt + _unavail_note``）。v2 这边**探测做了**
    （``_langchain_tools`` 把死工具换成了同名占位符），**话没说** —— 于是 agent
    拿着一批一调就失败的工具，而没有一句话告诉它为什么。它会一个个试过去，
    每次烧掉一轮，正是那条注释里写的「burning a turn per call rediscovering
    that web_search / search_papers are down」。

    ⚠️ 这个差异**双引擎对照看不见**：对照用的是脚本化模型，它不读提示词。
    「两边行为一致」是在一个看不见提示词的模型上验的 —— 而提示词恰恰是决定真模型
    行为的那个东西。

    按**签名**取而不是按 agent 名字写分支（与 ``_langchain_tools`` 同一条纪律）：
    哪个 agent 有附加块，问它自己的 ``tools`` 模块有没有那个函数。
    """
    try:
        mod = importlib.import_module(f"mast.agents.{agent_id}.tools")
    except Exception:  # noqa: BLE001
        return ""
    note_fn = getattr(mod, "unavailable_tools_note", None)
    probe = getattr(mod, "tool_availability", None)
    if note_fn is None or probe is None:
        return ""
    try:
        return str(note_fn(probe()) or "")
    except Exception as exc:  # noqa: BLE001 — 探测失败不该挡住装配
        logger.debug("unavailable-tools note failed for %s: %s", agent_id, exc)
        return ""


def _langchain_tools(agent_id: str, buf: Any) -> list:
    """调 agent 自己的 ``build_tools``，容忍各家不同的可选参数。

    literature 要 ``unavailable=``（死工具探测），别家不要。按签名给参数，而不是
    按 agent 名字写分支——后者是「名单抄第二遍」的又一个入口。
    """
    mod = importlib.import_module(f"mast.agents.{agent_id}.tools")
    build = getattr(mod, "build_tools", None)
    if build is None:
        raise RuntimeError(f"agent {agent_id} 没有 build_tools")

    kwargs: dict = {}
    params = inspect.signature(build).parameters

    if "unavailable" in params:
        # 死工具探测住在**该 agent 自己的** tools 模块里（今天只有 literature 有）。
        # 从那里取而不是从某个共享位置取：这个函数只承诺「按签名给参数」，
        # 不承诺知道每个 agent 的探测器叫什么、在哪。
        probe = getattr(mod, "tool_availability", None)
        if probe is not None:
            try:
                kwargs["unavailable"] = probe()
            except Exception as exc:  # noqa: BLE001 — 探测失败不该挡住装配
                logger.debug("tool availability probe failed for %s: %s",
                             agent_id, exc)

    if "registry" in params:
        # experiment_design 的工具表要一份技能目录（它读 metadata 生成
        # describe_skills）。目录由那个 agent 的 graph 模块负责发现，公共别名
        # ``discover_xd_catalog`` 就是为此存在的；这里用它而不是自己 discover
        # 一遍，免得两处的包清单各说各话。
        kwargs["registry"] = _skill_catalog(agent_id)

    missing = [n for n, p in params.items()
               if n not in kwargs and n != next(iter(params))
               and p.default is inspect.Parameter.empty
               and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if missing:
        # 宁可在这里明说，也不要让 TypeError 从 build_tools 深处冒出来 ——
        # 后者读起来像「这个 agent 坏了」，其实是「装配器还不认识它的新参数」。
        raise RuntimeError(
            f"{agent_id}.build_tools 需要装配器还不知道怎么提供的参数：{missing}。"
            "在 _langchain_tools 里为它加一条，并写清这个参数是干什么的。")

    return list(build(buf, **kwargs))


def _skill_catalog(agent_id: str):
    """该 agent 的技能目录（只要 metadata，不要实例）。

    ★ 从 ``.tools`` 取，**不是 ``.graph``**（2026-08-27 副本试验发现）
    ----------------------------------------------------------------
    前一版是 ``importlib.import_module(f"mast.agents.{agent_id}.graph")`` ——
    也就是**新的通用装配器 import 了它正要替代的那张图**。删掉图之后，
    ``build_agent_loop`` 对每一个 ``build_tools`` 需要 registry 的 agent 都
    ``ModuleNotFoundError``，而那正是 v2 路径的入口。

    这种耦合读代码看不出来：这一行长得像「去那个 agent 的模块里拿个东西」，
    只有真删一次、看谁炸了，才知道拿的那个模块是要死的。
    """
    tools_mod = importlib.import_module(f"mast.agents.{agent_id}.tools")
    for fname in ("discover_xd_catalog", "_discover_catalog",
                  "discover_instrument_skills"):
        fn = getattr(tools_mod, fname, None)
        if fn is not None:
            return fn()
    from mast.core.registry import SkillRegistry

    logger.warning("%s has no catalog discovery helper; using an empty registry",
                   agent_id)
    return SkillRegistry()


def _shared_middleware(agent_id: str) -> list:
    """与旧路径**同一份**通用中间件（未桥接，langchain 实例）。

    这张表照抄各 ``graph.py`` 里非 IC 专属的那几条，顺序也照抄——顺序在
    ``wrap_*`` 的洋葱里是有意义的（``ToolPairGuard`` 在 ``prefill_guard`` 外层，
    好让 prefill 保住「结尾必须 user 消息」的最终裁决）。

    **刻意不含的两条**，各有理由：

    * ``ToolPairGuardMiddleware`` —— 它唯一的存在理由是修 ``Command(graph=PARENT)``
      短路留下的孤儿 tool result。新循环里每个 tool_call 在同一次 ``run()`` 里
      得到配对回应，孤儿在构造上不可能出现（``test_agent_loop`` 钉着）。挂上去
      不会出错，但那是一道**防一件不会发生的事**的守卫，留着只会让人以为它还有用。
    * ``make_call_limit_middleware`` —— 限流是 ``AgentLoop`` 的内建
      （``CallLimits``，单位是模型/工具调用数而不是 super-step）。两套限流叠着，
      先撞上的那个说了算，而「哪个先撞」取决于换算率——正是要消灭的东西。
    """
    from mast.agents._shared.prefill_guard_mw import ClaudePrefillGuardMiddleware
    from mast.agents._shared.stall_guard_mw import StallGuardMiddleware
    from mast.agents._shared.upstream_mw import UpstreamArtifactMiddleware
    from mast.agents._shared.vision_mw import SkillImageMiddleware

    return [
        UpstreamArtifactMiddleware(agent_id),
        StallGuardMiddleware(agent_name=agent_id),
        SkillImageMiddleware(),
        ClaudePrefillGuardMiddleware(),
    ]


def build_agent_loop(
    agent_id: str, *, buf: Any = None, model: Any = None,
    max_model_calls: int = 25, max_tool_calls: int = 60,
    middleware: Iterable[Middleware] = (),
    extra_tools: Iterable[Any] = (),
    system_suffix: str = "",
    with_middleware: bool = True,
) -> AgentLoop:
    """装配一个 agent 的 v2 循环。

    ``model`` 可以是 ``ChatModelPort``、langchain 的 ``BaseChatModel``、或 None
    （按 agent 取默认）。三种都接，因为测试要塞脚本化端口，生产要用真模型，
    而 ``agent_node`` 这类调用方两种都可能。
    """
    if agent_id not in ASSEMBLABLE_AGENTS:
        raise ValueError(
            f"{agent_id!r} 的零件形状没有被核对过；可装配的是 {ASSEMBLABLE_AGENTS}。"
            "instrument_control 走专门的装配路径（它带一整套硬件中间件）。")

    port = _as_port(model) if model is not None else _default_port(agent_id)

    tools: list[ToolSpec] = []
    for t in list(_langchain_tools(agent_id, buf)) + list(extra_tools):
        name = str(getattr(t, "name", ""))
        # 独立委托里没有父编排器，交棒工具调了就是错。旧路径靠往提示词里写
        # 「不要 handoff」+ 事后从 checkpoint 抢救；这里直接不给这个工具 ——
        # **移除诱因，别在提示词里说服模型**。
        if name.startswith("handoff_to_"):
            continue
        try:
            tools.append(spec_from_langchain_tool(t))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping tool %s for %s: %s", name, agent_id, exc)

    prompt = _system_prompt(agent_id) + (system_suffix or "")

    stack: list[Middleware] = []
    if with_middleware:
        from mast.agentruntime.compat import bridge_all

        try:
            stack.extend(bridge_all(_shared_middleware(agent_id),
                                    agent_id=agent_id))
        except Exception as exc:  # noqa: BLE001
            # 通用中间件建不起来 ⇒ **说出来并继续**。这几条是上下文注入与停机
            # 判据，不是安全件（SafetyGate 只挂 IC，走单独的专门装配）——
            # 少了它们回合质量下降，而拦住整次装配会把一个可降级的问题变成
            # 一次彻底失败。
            logger.warning("shared middleware unavailable for %s (%s); "
                           "running the loop without them", agent_id, exc)
    stack.extend(middleware)

    return AgentLoop(
        name=agent_id, model=port, tools=tools, system_prompt=prompt,
        middleware=stack,
        limits=CallLimits(max_model_calls=max_model_calls,
                          max_tool_calls=max_tool_calls))


def _as_port(model: Any) -> ChatModelPort:
    if isinstance(model, ChatModelPort):
        return model
    return LangChainModelPort(model)


def _default_port(agent_id: str) -> ChatModelPort:
    from mast.agents._shared.models import make_chat_model

    return LangChainModelPort(make_chat_model(agent_id))


__all__ = ["build_agent_loop", "ASSEMBLABLE_AGENTS"]
