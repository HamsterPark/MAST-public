"""Agent-tool skills — 把 agent @tool **自动桥接**为 registry 技能（P3-A）。

只有 IC 是 registry 技能 agent（其 BaseSkill 经 wrap_skill 转成 @tool 喂给
create_agent）；其余 5 个 agent 的能力是 LangChain ``@tool`` 函数。本模块
**自动**遍历每个非 IC agent 已导出的 tool 列表（``WORKFLOW_TOOL_EXPORTS``，
即各 tools.py 里现成的 ``AGENT_TOOLS`` / ``LIBRARY_TOOLS``），把每个包成
AUTO BaseSkill 注册进 registry——于是「写一次 @tool 就自动出现在技能菜单 +
工作流构建器」，**不再手工维护白名单**（旧白名单只覆盖 11/≈30，菜单是残缺
镜像）。step 节点经 ``ExecutionContext.run`` 照常调用，过安全收口与参数校验，
并出现在构建器 palette（source=agent_tool）。

@tool 删不掉：它是 LangGraph ``create_agent`` 的唯一入口（IC 的硬件 skill 也
是先 wrap_skill 成 @tool）；本桥接统一的是「编写体验/菜单」，不是运行时。

明确排除（``WORKFLOW_TOOL_EXCLUDE`` + 结构性不在导出列表里）：
  * ``run_numpy_snippet`` —— 任意代码执行，违反「数据而非代码」主轨（R3）；
  * 依赖 agent 运行态的工具（buffer 队列 / handoff / MASTState 注入）——它们
    不在 ``AGENT_TOOLS`` 导出列表里，结构上就进不来。
  * experiment_design 的工具是工厂闭包（非模块级 @tool 对象），故不在
    ``WORKFLOW_TOOL_EXPORTS`` 中（describe_skills 也与菜单重复）。

工具失败语义：这些工具面向 LLM 设计，出错返回 "xxx failed: ..." 文本而不抛
异常——包装层做保守判定（错误前缀 → success=False），原文永远保留在
``data["text"]`` 供 llm/if 节点判断。
"""

from __future__ import annotations

import importlib
import logging
import re

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

# AUTO-BRIDGE (2026-06-15): every NON-IC agent's LangGraph @tool capabilities are
# surfaced as registry skills AUTOMATICALLY — author a @tool once and it appears in
# the skill menu + workflow builder, with NO hand-maintained allowlist (which had
# drifted to a partial mirror: only 11 of ~30 analysis tools showed). Each agent
# already exports its model-facing tools as module-level lists; we iterate those.
#   • instrument_control is absent — it authors BaseSkills directly (the registry IS
#     its vocabulary; wrap_skill converts them to tools for create_agent).
#   • experiment_design is absent — its 3 tools are FACTORY closures over the live
#     registry (describe_skills/lookup_sample/query_past_experiments), not
#     module-level @tool objects, so there is nothing to iterate (describe_skills is
#     also meta-over-the-registry, redundant in the menu).
# {agent 模块名: (tools.py 中导出的 tool 列表属性名, ...)}
WORKFLOW_TOOL_EXPORTS: dict[str, tuple[str, ...]] = {
    "data_processing":  ("AGENT_TOOLS",),
    "literature":       ("AGENT_TOOLS", "LIBRARY_TOOLS"),
    "paper_writing":    ("AGENT_TOOLS",),
    "paper_review":     ("AGENT_TOOLS",),
    # XD's tools are factory closures; tools.py materialises the no-arg safe ones
    # into WORKFLOW_EXPORT_TOOLS (describe_skills omitted — meta-over-registry).
    "experiment_design": ("WORKFLOW_EXPORT_TOOLS",),
}

# Bucket-C exclusion: tools present in an export list that must NEVER become a
# menu/workflow skill. run_numpy_snippet runs arbitrary LLM-authored Python in a
# RestrictedPython sandbox — it violates the「数据而非代码」main rail (R3) and stays
# barred. (handoff / buffer tools are control-flow / runtime-coupled and are NOT in
# these export lists at all, so they need no entry here.)
# py_run / py_stage_data 同理（2026-08-19）：py_run 跑任意 LLM 写的 Python；
# py_stage_data 的产物是「某个会话目录里出现了一个文件」——那不是可组合的步骤
# 输出，一个工作流步骤没法把它接给下一步。
WORKFLOW_TOOL_EXCLUDE: frozenset[str] = frozenset({
    "run_numpy_snippet", "py_run", "py_stage_data",
})

# Domain labels for the builder palette grouping (source=agent_tool).
AGENT_TOOL_DOMAINS = {
    "data_processing":   "数据处理工具",
    "literature":        "文献工具",
    "paper_writing":     "论文写作工具",
    "paper_review":      "论文评审工具",
    "experiment_design": "实验设计工具",
}

_ERROR_PREFIX = re.compile(
    r"^\s*\w[\w .]*?(failed|error)[:：]", re.IGNORECASE)


def _params_from_args_schema(schema) -> list[ParameterSpec]:
    """pydantic args_schema → ParameterSpec（int/float/bool/str 四类映射）。"""
    out: list[ParameterSpec] = []
    fields = getattr(schema, "model_fields", None) or {}
    for name, fi in fields.items():
        ann = getattr(fi, "annotation", str)
        if ann is int:
            ptype = "int"
        elif ann is float:
            ptype = "float"
        elif ann is bool:
            ptype = "bool"
        else:
            ptype = "str"
        required = bool(getattr(fi, "is_required", lambda: True)()
                        if callable(getattr(fi, "is_required", None))
                        else fi.default is None)
        default = None if required else fi.default
        out.append(ParameterSpec(
            name=name, type=ptype,
            description=str(getattr(fi, "description", "") or ""),
            required=required, default=default,
        ))
    return out


def wrap_agent_tool(tool_obj, source_agent: str) -> type:
    """LangChain StructuredTool → 可实例化注册的 BaseSkill 子类。"""
    desc = (getattr(tool_obj, "description", "") or "").strip()
    short = desc.split("\n", 1)[0][:200]
    params = _params_from_args_schema(getattr(tool_obj, "args_schema", None))
    tool_name = tool_obj.name

    class _AgentToolSkill(BaseSkill):
        _AGENT_TOOL_SOURCE = source_agent      # builder_api 据此标 source/domain

        def metadata(self) -> SkillMetadata:
            return SkillMetadata(
                name=tool_name, version="1.0.0",
                category=SkillCategory.ANALYSIS,
                safety_level=SafetyLevel.AUTO,   # 纯分析/检索，不触仪器
                description=short,
                parameters=params,
                tags=[source_agent, "agent_tool"],
                composition_level=0,
            )

        def execute(self, ctx, p: dict) -> SkillResult:
            try:
                out = tool_obj.invoke(dict(p or {}))
            except Exception as exc:  # noqa: BLE001 — 工具崩溃 → 干净失败
                return SkillResult(skill_name=tool_name, success=False,
                                   error=f"{type(exc).__name__}: {exc}")
            text = out if isinstance(out, str) else str(out)
            failed = bool(_ERROR_PREFIX.match(text))
            return SkillResult(
                skill_name=tool_name, success=not failed,
                error=(text[:300] if failed else ""),
                data={"text": text},
            )

    _AgentToolSkill.__name__ = f"Tool_{tool_name}"
    _AgentToolSkill.__qualname__ = _AgentToolSkill.__name__
    return _AgentToolSkill


def register_workflow_tool_skills(registry) -> list[str]:
    """Auto-bridge every non-IC agent's exported @tool capabilities into *registry*
    as AUTO BaseSkills (so they appear in the skill menu + workflow builder), by
    iterating each agent's WORKFLOW_TOOL_EXPORTS lists. ``WORKFLOW_TOOL_EXCLUDE``
    (code-exec) and name collisions with existing skills are skipped. Best-effort:
    a single failure only logs. Idempotent w.r.t. names already registered."""
    registered: list[str] = []
    seen: set[str] = set()
    for agent_mod, list_names in WORKFLOW_TOOL_EXPORTS.items():
        try:
            mod = importlib.import_module(f"mast.agents.{agent_mod}.tools")
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool-skills: import %s.tools failed: %s",
                           agent_mod, exc)
            continue
        for list_name in list_names:
            for t in (getattr(mod, list_name, None) or ()):
                nm = getattr(t, "name", None)
                if not nm or not hasattr(t, "invoke"):
                    continue              # not a LangChain tool object
                if nm in WORKFLOW_TOOL_EXCLUDE or nm in seen:
                    continue              # bucket-C plumbing, or already bridged
                if registry.has(nm):
                    logger.warning("tool-skills: %r collides with a registered "
                                   "skill — skipped", nm)
                    continue
                try:
                    registry.register(wrap_agent_tool(t, agent_mod))
                    registered.append(nm)
                    seen.add(nm)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("tool-skills: register %s failed: %s", nm, exc)
    if registered:
        logger.info("tool-skills: auto-registered %d agent tools as skills: %s",
                    len(registered), ", ".join(registered))
    return registered


__all__ = ["WORKFLOW_TOOL_EXPORTS", "WORKFLOW_TOOL_EXCLUDE", "AGENT_TOOL_DOMAINS",
           "wrap_agent_tool", "register_workflow_tool_skills"]
