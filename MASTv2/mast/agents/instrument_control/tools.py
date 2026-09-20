"""Instrument-Control agent tool list builder.

Wraps every v2 builtin (and Phase 4 composite/paper) skill as a LangChain
StructuredTool via wrap_skill, then concatenates buffer tools + handoffs.

Auto-discovery: at agent build time, SkillRegistry.discover() walks
`mast.skills.builtins` (and later `mast.skills.composite`,
`mast.skills.paper`) to find all BaseSkill subclasses. Each gets one tool.

Test/dev: pass a context_provider that returns a FakeCtx; production: pass
a callable that returns ExecutionContext(pool, state, registry).
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, Any, Callable

from mast.agents._shared.buffer_tools import make_buffer_tools
from mast.agents._shared.handoff import make_handoff
from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.registry import SkillRegistry

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Skill discovery + wrap
# ─────────────────────────────────────────────────────────────────────

def discover_instrument_skills(
    discover_packages: tuple[str, ...] = (
        "mast.skills.builtins",
        "mast.skills.composite",  # Phase 4 Session 4 — composite skills
    ),
) -> SkillRegistry:
    """Build a SkillRegistry populated from the given packages.

    Default scope: builtins + composite. paper/ goes to data_processing
    agent's tool list, not IC.
    """
    registry = SkillRegistry()
    n = registry.discover(*discover_packages)
    logger.info("instrument_control: discovered %d skills from %s", n, discover_packages)
    return registry


def build_instrument_skill_tools(
    registry: SkillRegistry,
    context_provider: Callable[[], Any],
    *,
    post_hook=None,
    recorder=None,
) -> list:
    """Return [wrap_skill(SkillCls, context_provider) for each skill in registry].

    ``post_hook`` (optional) is forwarded to every wrapped skill — producer-side
    glue run after a successful execute (render/record/buffer-emit), keyed on
    skill_name inside the hook. See wrap_skill.

    **Three gates, same mechanism, different reasons.**

    * *Hardware modules* (高级 → 硬件模块) — hardware the operator may not OWN. A KPFM
      controller you do not have is off because calling it can only ever fail.
    * *Advanced capabilities* (高级 → 高级能力) — powers that can step around a
      protection: script-file I/O (the allow-list vets what is IN a slot), quitting
      Nanonis, multi-pass config files, the blocking wait.
    * *Subscription* (技能 → 市场) — 这位用户这段时间在做什么。**装载面偏好，
      既不是硬件事实也不是保护**：未订阅的技能照样能被手动执行、被 composite 子步
      调用、被 conduct 的 ``ctx.run`` 下发（那三条路都走 ExecutionContext，那里
      没有名单式判断）。出厂是「全订阅」，所以这道门装上去当天是恒真门。

    The first two ship OFF, and off means the skills are not wrapped at all — the
    agent cannot call what it cannot see. For hardware that is not there, a tool can
    only ever fail, and the model reads the whole tool list on every turn. For an
    advanced capability, "cannot see it" is the actual protection. All three leave
    the skill in the SkillRegistry, so the manual/GUI executor path still reaches it.

    三道门的并集只算一次，定义在 :mod:`mast.skills.tool_face` —— 在此之前它在本文件
    和 ``webui/agents_api.py`` 里被各算了一遍，而漏改任何一处都不报错。
    """
    from mast.skills import tool_face

    all_names = tool_face.names_from_registry(registry)
    skip_info = tool_face.compute(all_names)
    skip = skip_info.names
    tools = []
    for meta in registry.list_skills():
        if meta.name in skip:
            continue
        try:
            skill_cls = registry.get(meta.name)
            tool = wrap_skill(skill_cls, context_provider,
                              post_hook=post_hook, recorder=recorder)
            tools.append(tool)
        except Exception as e:
            logger.warning("Failed to wrap %s: %s", meta.name, e)
    logger.info("instrument_control: wrapped %d skill tools（%s 而未挂载）",
                len(tools), skip_info.describe())
    _record_wrap(registry, skip)
    return tools


#: 上一次**真的 wrap 出来**的工具表指纹。生效探针的 agent 侧。
#:
#: 注册表说「已覆盖」和模型**手上拿着覆盖版**是两件事，中间隔着一次图重建 ——
#: 而重建可能因任务占用被推迟、可能失败、也可能压根没接线。三种情况在日志里
#: 长得都不一样，而用户只想知道一件事：现在生效了吗。
#:
#: 两侧算**同一个**指纹（``skills.overlay.provenance.fingerprint``），不等就说
#: 「尚未跟上」。一个数、几毫秒，而且不会因为遗漏而说谎 —— 它覆盖全集。
LAST_WRAP: dict = {"at": 0.0, "n": 0, "fingerprint": "", "overlaid": []}


def _record_wrap(registry, skipped) -> None:
    try:
        from mast.skills.overlay.provenance import (
            OVERLAY_NAMESPACE, fingerprint, registry_triples,
        )

        triples = [t for t in registry_triples(registry) if t[0] not in skipped]
        LAST_WRAP.update({
            "at": _time.time(),
            "n": len(triples),
            "fingerprint": fingerprint(triples),
            "overlaid": sorted(n for n, mod, _v in triples
                               if OVERLAY_NAMESPACE in mod),
        })
    except Exception:  # noqa: BLE001 — 探针坏了不能让工具表建不起来
        logger.debug("wrap 指纹记录失败", exc_info=True)


def expected_wrap_fingerprint(registry) -> tuple[str, list[str]]:
    """注册表**此刻**应该 wrap 出什么。与 :data:`LAST_WRAP` 比。

    门的并集必须与 :func:`build_instrument_skill_tools` **逐字同源**（两处都调
    ``tool_face.compute``）—— 只给装配加一道门而忘了这里，探针会把「已生效」永远
    报成「没跟上」。这正是把并集搬进 ``tool_face`` 的原因。
    """
    from mast.skills import tool_face
    from mast.skills.overlay.provenance import (
        OVERLAY_NAMESPACE, fingerprint, registry_triples,
    )

    skip = tool_face.compute(tool_face.names_from_registry(registry)).names
    triples = [t for t in registry_triples(registry) if t[0] not in skip]
    return (fingerprint(triples),
            sorted(n for n, mod, _v in triples if OVERLAY_NAMESPACE in mod))


# ─────────────────────────────────────────────────────────────────────
# Nanonis software-manual lookup tool (Integration B — on-demand)
# ─────────────────────────────────────────────────────────────────────

def make_nanonis_manual_tool():
    """Read-only on-demand lookup into the Nanonis Mimea SOFTWARE manual.

    The full 128-topic manual is NOT stuffed into context; the agent calls this
    tool to pull a module/topic section when it needs depth (parameter ranges,
    procedures, gotchas). Short per-skill hints are already in the relevant tool
    descriptions (Integration A). Graceful: if the manual hasn't been extracted
    on this machine, the tool returns a clear note instead of erroring."""
    from langchain_core.tools import tool as _tool
    from mast.knowledge.nanonis_manual import search

    @_tool("nanonis_manual")
    def nanonis_manual(query: str) -> str:
        """查询 Nanonis 软件操作手册（GUI 在线帮助），获取某测量模块/功能的详细说明
        （参数含义与范围、操作流程、注意事项）。query 用模块或主题名，例如
        "Bias Spectroscopy"、"Z-Controller"、"Scan Control"、"Lock-In"、"PLL"、
        "Auto-Approach"、"Atom Tracking"。返回该主题手册正文（过长会截断）；
        无匹配时返回可用模块/主题清单。"""
        return search(query)

    return nanonis_manual


# ─────────────────────────────────────────────────────────────────────
# Top-level tool list assembly (skills + buffer + handoffs)
# ─────────────────────────────────────────────────────────────────────

def build_tools(
    buf: "BufferService | None",
    context_provider: Callable[[], Any],
    registry: SkillRegistry | None = None,
    targets: tuple[str, ...] = ("supervisor", "data_processing"),
    *,
    post_hook=None,
    recorder=None,
) -> list:
    """Top-level tool-list builder for the IC agent.

    Order matters slightly for LLM behaviour: skills first (most numerous),
    then buffer reads (queried often during tip-condition loops), then handoffs
    last (terminal actions). ``post_hook`` is forwarded to every wrapped skill.
    """
    if registry is None:
        registry = discover_instrument_skills()
    tools = build_instrument_skill_tools(registry, context_provider,
                                         post_hook=post_hook, recorder=recorder)
    # On-demand Nanonis software-manual lookup (Integration B).
    try:
        tools = tools + [make_nanonis_manual_tool()]
    except Exception as e:  # noqa: BLE001 - manual tool is optional
        logger.warning("nanonis_manual tool unavailable: %s", e)
    if buf is not None:
        tools = tools + make_buffer_tools(buf)
    # 技能工坊(2026-08-25):查目录 / 起草 / 保存热注册 / 执行 / 提议新原子技能。
    # 底层(声明式 IR、解释器、版本库、热注册、校验内核)早就齐了,只挂在 GUI 的
    # builder 页上 —— 这五个工具是 agent 侧唯一缺的那一环。
    #
    # 挂在这里而不是各入口分别挂:群聊(orchestrator)与私聊(runtime)都经过
    # build_tools,一处挂载两边同时到位。「每页各自记得」是本仓人肉找不齐的形状。
    try:
        from mast.agents._shared.skill_forge_tools import make_skill_forge_tools
        tools = tools + make_skill_forge_tools(
            registry, context_provider, recorder=recorder, post_hook=post_hook)
    except Exception as e:  # noqa: BLE001 — 工坊不可用绝不能让仪器 agent 起不来
        logger.warning("skill forge tools unavailable: %s", e)
    tools = tools + [
        make_handoff(target, _handoff_description(target))
        for target in targets
    ]
    return tools


def _handoff_description(target: str) -> str:
    """Descriptions the model reads on EVERY turn — they gate the handoff.

    Both used to describe the handoff as available "when you've completed your
    phase" / "for newly-acquired results", with nothing tying "completed" to the
    operator's actual request. (2026-07-27): asked for 5 STS
    points, the agent handed off to data_processing right away. The prompt's rule
    lives in instrument_control/prompts.py 「交接时机」; these two strings must say
    the same thing, because a tool description that reads "hand newly-acquired
    results over" is an invitation the checklist cannot outvote.
    """
    descriptions = {
        "supervisor": (
            "Return control to the orchestrator. Use when EVERY measurement in "
            "your current Instrument-Control phase is finished, or when you are "
            "blocked pending human input / a hardware precondition you cannot "
            "satisfy. Not for a mid-sequence pause: if points or parameters the "
            "operator asked for remain, keep acquiring instead."
        ),
        "data_processing": (
            "Hand acquired scan/STS results to the Data-Processing agent for "
            "OFFLINE analysis. Call this ONLY once every point/parameter the "
            "operator asked for has actually been acquired — data_processing "
            "cannot touch the instrument, so a spectrum you did not take is one "
            "nobody downstream can take for you. The one "
            "legitimate mid-sequence use is when the REMAINING steps need an "
            "analysis result first (e.g. picking STS sites from a scan); say so "
            "in `reason` and name what is still outstanding. "
            "Pass the file path(s) via the agent state.last_scan."
        ),
        "experiment_design": (
            "Hand back to Experiment-Design agent if the plan needs revision."
        ),
    }
    return descriptions.get(target, f"Hand off control to {target}.")


__all__ = [
    "discover_instrument_skills",
    "build_instrument_skill_tools",
    "make_nanonis_manual_tool",
    "build_tools",
]
