"""`/agents/tools` 必须报**agent 真正拿到的**工具表。

2026-08-19 之前它自己 `SkillRegistry()` + `discover()`，从头到尾不碰进程里那个
活的注册表。后果是这张表**从来没有对过**：

* 声明式 composite（`config/composite_skills/*.json`）—— 一个都不在
* 用户 custom skill —— 一个都不在
* 桥接进来的 agent @tool（`register_workflow_tool_skills`）—— 一个都不在
* `mast.skills.paper` —— discover 列表里压根没写，**整包缺席**

而且这个缺口有个更坏的性质：**只加一个缓存失效函数是假修**。缓存清掉之后重算，
算出来的还是同一张错表。所以这里钉的是「读的是不是 live registry」，不是「缓存
会不会过期」。

第二件事：IC 实际拿到的是 registry **减去两个闸门**（`build_instrument_skill_tools`
`instrument_control/tools.py:83-85`）。列出 agent 看不见的技能，是换一种方式说同
一个谎，所以这里也要过同一对 skip-set。
"""

from __future__ import annotations

import pytest

from mast.core.registry import SkillRegistry
from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata
from mast.skills.base import BaseSkill
from mast.webui import agents_api


class _OnlyInTheLiveRegistry(BaseSkill):
    """一个 discover() 永远找不到的技能 —— 它不在任何被扫描的包里。

    这正是 composite spec / custom skill / 桥接 @tool 在真实运行时的处境。
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="_LiveRegistryOnlyProbe",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="probe",
            parameters=[],
        )

    def execute(self, context, params):  # pragma: no cover
        raise AssertionError("probe skill must never execute")


@pytest.fixture(autouse=True)
def _clean_module_state():
    """模块级状态在测试之间必须归零，否则一个测试的 registry 会跟着下一个跑。"""
    agents_api.set_live_registry(None)
    agents_api.invalidate_agent_tools()
    yield
    agents_api.set_live_registry(None)
    agents_api.invalidate_agent_tools()


def _ic_names(catalog) -> set[str]:
    return {t["name"] for t in catalog.get("instrument_control", [])}


def test_agent_tools_reads_the_live_registry():
    """塞进 live registry 的技能必须出现在列表里。

    这条在 2026-08-19 之前必然失败 —— 那时 `_compute_agent_tools` 造的是自己的
    注册表，外面塞什么它都看不见。
    """
    reg = SkillRegistry()
    reg.discover("mast.skills.builtins")
    reg.register(_OnlyInTheLiveRegistry)
    agents_api.set_live_registry(reg)

    names = _ic_names(agents_api._compute_agent_tools())
    assert "_LiveRegistryOnlyProbe" in names
    assert not agents_api.agent_tools_degraded()


def test_without_a_live_registry_it_says_so():
    """降级要报出来。不知道就报 True，绝不静默给一张错表。"""
    agents_api.set_live_registry(None)
    agents_api._compute_agent_tools()
    assert agents_api.agent_tools_degraded() is True


def test_gated_skills_are_not_listed(monkeypatch):
    """被硬件模块/高级能力闸门关掉的技能，agent 看不见，这里也不能出现。"""
    reg = SkillRegistry()
    reg.register(_OnlyInTheLiveRegistry)
    agents_api.set_live_registry(reg)

    assert "_LiveRegistryOnlyProbe" in _ic_names(agents_api._compute_agent_tools())

    monkeypatch.setattr(
        "mast.skills.hardware_modules.disabled_skill_names",
        lambda: frozenset({"_LiveRegistryOnlyProbe"}),
    )
    assert "_LiveRegistryOnlyProbe" not in _ic_names(agents_api._compute_agent_tools())


def test_invalidate_drops_the_cache():
    reg = SkillRegistry()
    reg.register(_OnlyInTheLiveRegistry)
    agents_api.set_live_registry(reg)
    agents_api.warm_agent_tools()
    assert agents_api._AGENT_TOOLS_CACHE is not None
    agents_api.invalidate_agent_tools()
    assert agents_api._AGENT_TOOLS_CACHE is None


def test_a_newly_registered_skill_needs_an_invalidation_to_show_up():
    """缓存和 live registry 是两件事 —— 两个都要，少一个就是「以为生效」。"""
    reg = SkillRegistry()
    agents_api.set_live_registry(reg)
    agents_api.warm_agent_tools()
    assert "_LiveRegistryOnlyProbe" not in _ic_names(agents_api.get_agent_tools())

    reg.register(_OnlyInTheLiveRegistry)
    assert "_LiveRegistryOnlyProbe" not in _ic_names(agents_api.get_agent_tools()), (
        "热注册本身不该让陈旧缓存自己更新 —— 那会掩盖『谁负责失效』这个问题"
    )

    agents_api.invalidate_agent_tools()
    agents_api.warm_agent_tools()
    assert "_LiveRegistryOnlyProbe" in _ic_names(agents_api.get_agent_tools())
