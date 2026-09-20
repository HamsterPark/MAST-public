"""技能工坊的**工具面**:有哪几个、治理在哪一层、以及它们**不**在哪里出现。

这一组钉的是三件事,每一件缺了都会让这个能力悄悄变形:

1. **名单与实物一致** —— 「清单说有、图上没有」是本仓踩过的形状。
2. **它们没有 skill_metadata** —— 这是**有意**的:safety_mw / auto_approval_mw
   看到 ``meta_obj is None`` 就放行,治理全部落在执行层。哪天有人给它们盖上
   metadata,这条测试要先红,因为那会把工坊接进一套它并不适用的闸门。
3. **它们进核心包** —— tool packs 把大多数工具藏起来了,而 ``skill_catalog``
   是「官方技能优先」那条阶梯的强制第一步。藏起来就把阶梯倒过来了。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from _forge_fixtures import StubCtx, registry, tools  # noqa: E402

from mast.agents._shared.skill_forge_tools import (  # noqa: E402
    FORGE_TOOL_NAMES,
    make_skill_forge_tools,
)


def test_the_name_list_matches_what_the_factory_actually_builds():
    assert tuple(tools(ctx=StubCtx())) == FORGE_TOOL_NAMES


def test_without_a_context_provider_there_is_no_run_composite():
    """没有硬件宿主时诚实地少一个工具,而不是给一个会炸的。"""
    names = tuple(tools())
    assert "run_composite" not in names
    assert set(FORGE_TOOL_NAMES) - set(names) == {"run_composite"}


@pytest.mark.parametrize("name", FORGE_TOOL_NAMES)
def test_forge_tools_carry_no_skill_metadata(name):
    """治理在执行层,不在工具面 —— 这些是普通 @tool。

    ``safety_mw._safety_block`` 只门控带 ``skill_metadata`` 的工具。给工坊盖上
    metadata 会让 SafetyGate 拿一个「组合技能的元数据」去判一个**创建动作**,
    那是两件事。真正的闸门在 ``run_composite`` 内部与每个子步的 ctx.run 上。
    """
    tool = tools(ctx=StubCtx())[name]
    meta = getattr(tool, "metadata", None) or {}
    assert "skill_metadata" not in meta


def test_forge_tools_are_in_the_core_pack_not_hidden_behind_a_load():
    """`skill_catalog` 藏进包里 = 把「先查有没有现成的」这一步藏起来。

    tool_packs.py 自己记着这个教训:「藏一步的代价不是多一轮,是模型压根想不起来
    去查」。对工坊尤其致命 —— 想不起来查就直接开始造,而官方优先那条阶梯的第一级
    就是查。``run_composite`` 还有个结构性理由:它要用到的那一轮,恰恰是包还没取
    的那一轮(刚 save 完、工具表还没刷新)。
    """
    from mast.agents._shared.tool_packs import CORE_PREFIXES
    for name in FORGE_TOOL_NAMES:
        assert any(name.startswith(p) for p in CORE_PREFIXES), (
            f"{name} 不在核心包里 —— 模型要先 load_tool_pack 才看得见它")


def test_forge_tools_do_not_leak_into_the_workflow_builder_menu():
    """工坊工具不该变成可以被 composite 步骤调用的「技能」。

    ``register_workflow_tool_skills`` 把各 agent 的 ``AGENT_TOOLS`` 桥接进
    builder palette。工坊是**工厂闭包**,不是模块级导出列表,所以结构上进不去 ——
    这条测试钉住那个结构性事实。让它进去会造出递归(一个 composite 步骤保存并
    执行另一个 composite),而那不是这次要开的口子。
    """
    import mast.agents._shared.skill_forge_tools as mod
    from mast.skills.composite.tool_skills import WORKFLOW_TOOL_EXPORTS

    for attrs in WORKFLOW_TOOL_EXPORTS.values():
        for attr in attrs:
            assert getattr(mod, attr, None) is None, (
                f"skill_forge_tools 里出现了 {attr} —— 桥接器会把它扫进菜单")
    # 工具只在工厂调用之后才存在:模块里没有任何现成的 tool 对象。
    exported = [v for v in vars(mod).values()
                if hasattr(v, "name") and getattr(v, "name", "") in FORGE_TOOL_NAMES]
    assert exported == []


def test_the_factory_refuses_nothing_when_the_registry_is_empty():
    """空注册表下工具仍然建得起来(诚实作答,而不是建图时炸)。"""
    from mast.core.registry import SkillRegistry
    built = make_skill_forge_tools(SkillRegistry(), lambda: StubCtx())
    assert tuple(t.name for t in built) == FORGE_TOOL_NAMES


def test_catalog_marks_official_skills_as_official():
    """来源标签是「官方优先」在目录里的那一半 —— 分不出官方就无从优先。"""
    from _forge_fixtures import call
    out = call(tools(ctx=StubCtx())["skill_catalog"], query="ScanAt")
    rows = {r["name"]: r for r in out["skills"]}
    assert "ScanAt" in rows
    assert rows["ScanAt"]["official"] is True
    assert rows["ScanAt"]["origin"] in ("官方原子", "官方组合")


def test_catalog_puts_official_skills_first():
    """排序也承载优先序:同名/近名的多份实现里,官方那份先被看见。"""
    from _forge_fixtures import call
    out = call(tools(ctx=StubCtx())["skill_catalog"])
    flags = [r["official"] for r in out["skills"]]
    assert flags == sorted(flags, reverse=True), "官方技能没有排在前面"
    assert registry() is not None
