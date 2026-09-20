"""声明式工作流的安全级继承 + 完整节点树遍历。

两个洞:
  1. ``safety_level`` 纯手填、无继承 —— 一个内含 DANGEROUS 步骤的工作流可以把
     自己声明成 ``auto``,而 HITL 门控完全由 safety_level 驱动,于是整个绕过去。
     构建器上那个下拉框看起来完全正常,这是静默的。
  2. 遍历 spec 树的工具只认 ``if`` / ``loop`` —— 藏在 ``try`` 体、``llm`` 路由、
     ``human`` 路由里的 step 是隐形的,「引用了不存在的技能」这类检查漏掉它们,
     于是那个 spec 照常注册,运行时先打出前面的硬件动作再崩。
"""

from __future__ import annotations

import pytest

from mast.core.registry import SkillRegistry
from mast.core.types import (
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.composite.interpreter import make_spec_skill
from mast.skills.composite.loader import register_spec
from mast.skills.composite.spec import CompositeSpec, collect_step_skills


def _mk_skill(name: str, level: SafetyLevel):
    class _S(BaseSkill):
        def metadata(self) -> SkillMetadata:
            return SkillMetadata(name=name, version="1.0.0",
                                 category=SkillCategory.WRITE,
                                 safety_level=level)

        def execute(self, context, params: dict) -> SkillResult:
            return SkillResult(skill_name=name, success=True)

    _S.__name__ = f"Skill_{name}"
    return _S


@pytest.fixture
def registry():
    reg = SkillRegistry()
    reg.register(_mk_skill("AutoStep", SafetyLevel.AUTO))
    reg.register(_mk_skill("ConfirmStep", SafetyLevel.CONFIRM))
    reg.register(_mk_skill("DangerStep", SafetyLevel.DANGEROUS))
    return reg


def _spec(nodes, declared="auto", name="T"):
    return CompositeSpec(name=name, safety_level=declared, nodes=nodes)


def _level(spec, registry):
    return make_spec_skill(spec, registry)().metadata().safety_level


# ── 继承:声明只能收紧,不能放松 ─────────────────────────────────────────────

def test_declared_auto_is_raised_by_a_dangerous_step(registry):
    spec = _spec([{"type": "step", "id": "s", "skill": "DangerStep"}],
                 declared="auto")
    assert _level(spec, registry) == SafetyLevel.DANGEROUS


def test_declared_auto_is_raised_by_a_confirm_step(registry):
    spec = _spec([{"type": "step", "id": "s", "skill": "ConfirmStep"}],
                 declared="auto")
    assert _level(spec, registry) == SafetyLevel.CONFIRM


def test_declared_dangerous_is_never_lowered_by_harmless_steps(registry):
    """用户认为它危险,那它就是危险 —— 继承只提级。"""
    spec = _spec([{"type": "step", "id": "s", "skill": "AutoStep"}],
                 declared="dangerous")
    assert _level(spec, registry) == SafetyLevel.DANGEROUS


def test_all_auto_steps_keep_the_declared_auto(registry):
    spec = _spec([{"type": "step", "id": "s", "skill": "AutoStep"}],
                 declared="auto")
    assert _level(spec, registry) == SafetyLevel.AUTO


def test_highest_of_several_steps_wins(registry):
    spec = _spec([
        {"type": "step", "id": "a", "skill": "AutoStep"},
        {"type": "step", "id": "b", "skill": "DangerStep"},
        {"type": "step", "id": "c", "skill": "ConfirmStep"},
    ], declared="auto")
    assert _level(spec, registry) == SafetyLevel.DANGEROUS


def test_unknown_step_skill_does_not_crash_metadata(registry):
    """技能不在册交给 loader 的缺失检查处理,不能让 metadata() 抛。"""
    spec = _spec([{"type": "step", "id": "s", "skill": "NoSuchSkill"}],
                 declared="confirm")
    assert _level(spec, registry) == SafetyLevel.CONFIRM


def test_no_registry_keeps_the_declared_level(registry):
    """独立构造(测试 / 预览)没有 registry,保持声明值。"""
    spec = _spec([{"type": "step", "id": "s", "skill": "DangerStep"}],
                 declared="auto")
    assert make_spec_skill(spec)().metadata().safety_level == SafetyLevel.AUTO


def test_register_spec_wires_the_registry_so_inheritance_applies(registry):
    """安全性真正依赖的是注册路径 —— HITL map 是从 registry 里推导的。"""
    spec = _spec([{"type": "step", "id": "s", "skill": "DangerStep"}],
                 declared="auto", name="SneakyWorkflow")
    register_spec(registry, spec)
    meta = registry._get_metadata(registry.get("SneakyWorkflow"))
    assert meta.safety_level == SafetyLevel.DANGEROUS


# ── 继承要能穿透每一种嵌套容器 ───────────────────────────────────────────────

@pytest.mark.parametrize("nodes,where", [
    ([{"type": "if", "id": "i", "cond": "True",
       "then": [{"type": "step", "id": "s", "skill": "DangerStep"}]}], "if.then"),
    ([{"type": "if", "id": "i", "cond": "True", "then": [],
       "else": [{"type": "step", "id": "s", "skill": "DangerStep"}]}], "if.else"),
    ([{"type": "loop", "id": "l", "mode": "repeat", "count": "1", "var": "i",
       "body": [{"type": "step", "id": "s", "skill": "DangerStep"}]}], "loop.body"),
    ([{"type": "try", "id": "t",
       "body": [{"type": "step", "id": "s", "skill": "DangerStep"}]}], "try.body"),
    ([{"type": "try", "id": "t", "body": [],
       "finally": [{"type": "step", "id": "s", "skill": "DangerStep"}]}],
     "try.finally"),
    ([{"type": "llm", "id": "d", "responsibility": "r", "mode": "route",
       "escape": "no", "routes": {
           "no": [],
           "yes": [{"type": "step", "id": "s", "skill": "DangerStep"}]}}],
     "llm.routes"),
    ([{"type": "human", "id": "h", "message": "?", "routes": {
        "go": [{"type": "step", "id": "s", "skill": "DangerStep"}]}}],
     "human.routes"),
    ([{"type": "llm", "id": "d", "responsibility": "r", "mode": "data",
       "output_schema": {"x": "float"},
       "on_error": [{"type": "step", "id": "s", "skill": "DangerStep"}]}],
     "llm.on_error"),
])
def test_inheritance_reaches_every_nesting_container(registry, nodes, where):
    """少数一个容器,藏在那一支里的危险步骤就能静默绕过门控。"""
    assert _level(_spec(nodes, declared="auto"), registry) == \
        SafetyLevel.DANGEROUS, f"{where} 里的步骤没有被继承规则看到"


# ── 树遍历本身 ───────────────────────────────────────────────────────────────

def test_collect_step_skills_finds_steps_in_try_bodies():
    """这是既有 bug:藏在 try 里的缺失技能能通过注册校验,运行时才崩。"""
    nodes = [{"type": "try", "id": "t",
              "body": [{"type": "step", "id": "a", "skill": "InBody"}],
              "finally": [{"type": "step", "id": "b", "skill": "InFinally"}]}]
    assert collect_step_skills(nodes) == {"InBody", "InFinally"}


def test_collect_step_skills_finds_steps_in_llm_and_human_routes():
    nodes = [
        {"type": "llm", "id": "d", "responsibility": "r", "mode": "route",
         "escape": "safe", "routes": {
             "safe": [{"type": "step", "id": "a", "skill": "SafeBranch"}],
             "risky": [{"type": "step", "id": "b", "skill": "RiskyBranch"}]}},
        {"type": "human", "id": "h", "message": "?", "routes": {
            "ok": [{"type": "step", "id": "c", "skill": "HumanBranch"}]}},
    ]
    assert collect_step_skills(nodes) == {
        "SafeBranch", "RiskyBranch", "HumanBranch"}


def test_collect_step_skills_recurses_deeply():
    nodes = [{"type": "loop", "id": "l", "mode": "repeat", "count": "1",
              "var": "i", "body": [
                  {"type": "if", "id": "f", "cond": "True", "then": [
                      {"type": "try", "id": "t", "body": [
                          {"type": "step", "id": "s", "skill": "DeepSkill"}]}]}]}]
    assert collect_step_skills(nodes) == {"DeepSkill"}


def test_collect_step_skills_tolerates_garbage():
    assert collect_step_skills(None) == set()
    assert collect_step_skills("not a list") == set()
    assert collect_step_skills([None, 42, {"type": "step"}]) == set()


def test_loader_missing_skill_check_now_sees_nested_steps(registry):
    """注册前的缺失技能检查必须看得见嵌套分支里的 step。"""
    from mast.skills.composite.loader import _missing_step_skills

    spec = _spec([{"type": "try", "id": "t", "body": [
        {"type": "step", "id": "s", "skill": "GhostSkill"}]}])
    assert _missing_step_skills(registry, spec) == ["GhostSkill"]
