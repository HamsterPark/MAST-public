"""SAFE/SEMI 在 ``ExecutionContext.run`` 上有闸 —— 三条旁路的唯一收口。

2026-08-27 之前，SAFE 的「不修针」契约只在 **agent 的门口**执行
（``safety_mw._mode_block`` 看 agent 自己的 tool call；``skill_forge_tools``
为 ``run_composite`` 又写了一遍同样的判断）。而 ``ExecutionContext.run`` 里
**一份都没有** —— 于是这三条路照常打脉冲：

* composite 子步（``interpreter`` → ``ctx.run``）
* conduct 的每一步（``conduct/adapters.py`` → ``ctx.run``）
* ``POST /api/skills/{name}/execute``（``routes/skill_exec.py`` → ``ec.run``）

一个含 TipPulse 的 conduct 模板在 SAFE 下真的会把脉冲打出去，而且看着很合规。

本文件钉两件事：**行为**（闸在，且方向正确）与**单源**（判据只有一份）。
"""
from __future__ import annotations

import ast
import sys
import threading
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core import safety as safety_mod  # noqa: E402
from mast.core.execution_context import ExecutionContext  # noqa: E402
from mast.core.operating_mode import bind_mode_source  # noqa: E402
from mast.core.types import (  # noqa: E402
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill  # noqa: E402
from mast.core.registry import SkillRegistry  # noqa: E402

RAN: list[str] = []


class _PulseProbe(BaseSkill):
    """带 ``bias_pulse`` 标签的探针 —— 与 BiasPulse 同一个标签，不碰硬件。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ModeGatePulseProbe",
            description="测试用探针（声明电脉冲能力，实际什么都不做）",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            capabilities=frozenset({"bias_pulse"}),
            parameters=[],
        )

    def execute(self, context, params) -> SkillResult:
        RAN.append("pulse")
        return SkillResult(skill_name="ModeGatePulseProbe", success=True,
                           data={"ran": True})


class _ShapeProbe(BaseSkill):
    """带 ``tip_shaping`` 标签 + 一个米单位深度参数的探针。

    ``tip_lift_m`` 的名字与单位都要对得上 —— ``semi_tip_depth_violations``
    是按「ParameterSpec 的 unit 是 m」且「名字含 tip_lift/lift_height/depth」
    来找深度参数的，名字起错了这个测试会绿在一个假的理由上。
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ModeGateShapeProbe",
            description="测试用探针（声明机械修针能力，实际什么都不做）",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            capabilities=frozenset({"tip_shaping"}),
            parameters=[
                ParameterSpec(name="tip_lift_m", type="float", unit="m",
                              required=False, default=0.0),
                # 显式 0 V：否则 is_electrical_pulse 对 tip_shaping 会把「没说」
                # 读成「带电」（fail-closed），这个探针就变成脉冲了，SEMI 用例
                # 便测不到深度闸这一条。
                ParameterSpec(name="bias_lift_v", type="float", unit="V",
                              required=False, default=0.0),
            ],
        )

    def execute(self, context, params) -> SkillResult:
        RAN.append("shape")
        return SkillResult(skill_name="ModeGateShapeProbe", success=True,
                           data={"ran": True})


class _Pool:
    def safe_call(self, method, *args, role="main"):
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", [0]))


@pytest.fixture(autouse=True)
def _clean():
    RAN.clear()
    yield
    bind_mode_source(None)      # 泄漏一个绑定会翻掉后面每个测试的针尖判定
    RAN.clear()


def _ctx(approval_source: str = "auto") -> ExecutionContext:
    reg = SkillRegistry()
    reg.register(_PulseProbe)
    reg.register(_ShapeProbe)
    return ExecutionContext(pool=_Pool(), state=None, registry=reg,
                            abort_event=threading.Event(),
                            approval_source=approval_source)


# ── 行为 ────────────────────────────────────────────────────────────────

def test_safe_refuses_a_pulse_and_the_skill_never_runs():
    """最重要的一条：SAFE 下经这扇门的脉冲被拒，而且 execute 没被调到。

    只断言 ``success is False`` 是不够的 —— 一个「跑完了再报失败」的实现同样
    满足它，而脉冲已经打出去了。RAN 才是「针尖没被碰」的证据。
    """
    bind_mode_source(lambda: "safe")
    res = _ctx().run("ModeGatePulseProbe", {})
    assert res.success is False
    assert "safe_mode_tip_processing_blocked" in (res.error or "")
    assert RAN == [], "SAFE 下 execute 竟然跑到了 —— 闸在报失败，却没有拦住动作"


def test_auto_lets_the_same_pulse_through():
    bind_mode_source(lambda: "auto")
    res = _ctx().run("ModeGatePulseProbe", {})
    assert res.success is True and RAN == ["pulse"]


def test_unbound_mode_behaves_exactly_as_before():
    """未绑定 = 不知道 = 放行。测试进程 / headless / 离线工具零影响。"""
    bind_mode_source(None)
    res = _ctx().run("ModeGatePulseProbe", {})
    assert res.success is True and RAN == ["pulse"]


def test_human_approval_source_is_not_gated():
    """手动路径本身就是人工授权；模式管的是「系统自己干什么」。"""
    bind_mode_source(lambda: "safe")
    res = _ctx(approval_source="human").run("ModeGatePulseProbe", {})
    assert res.success is True and RAN == ["pulse"]


def test_semi_refuses_a_deep_plunge_but_allows_a_shallow_one():
    bind_mode_source(lambda: "semi")
    deep = _ctx().run("ModeGateShapeProbe",
                      {"tip_lift_m": 30e-9, "bias_lift_v": 0.0})
    assert deep.success is False
    assert "semi_mode_shallow_only" in (deep.error or "")
    assert RAN == []

    shallow = _ctx().run("ModeGateShapeProbe",
                         {"tip_lift_m": 0.5e-9, "bias_lift_v": 0.0})
    assert shallow.success is True, shallow.error
    assert RAN == ["shape"]


def test_safe_refuses_mechanical_shaping_too():
    """SAFE 拒的是「针尖处理」，不只是电脉冲。"""
    bind_mode_source(lambda: "safe")
    res = _ctx().run("ModeGateShapeProbe",
                     {"tip_lift_m": 0.5e-9, "bias_lift_v": 0.0})
    assert res.success is False
    assert "safe_mode_tip_processing_blocked" in (res.error or "")
    assert RAN == []


def test_a_read_only_skill_is_untouched_in_safe():
    """闸只认能力标签。没有标签的技能在 SAFE 下照跑 —— 否则 SAFE 会变成
    「什么都不许做」，而它的契约是「专注实验、不修针」。"""
    bind_mode_source(lambda: "safe")

    class _Read(BaseSkill):
        def metadata(self):
            return SkillMetadata(name="ModeGateReadProbe", description="只读",
                                 category=SkillCategory.READ,
                                 safety_level=SafetyLevel.AUTO, parameters=[])

        def execute(self, context, params):
            RAN.append("read")
            return SkillResult(skill_name="ModeGateReadProbe", success=True)

    reg = SkillRegistry()
    reg.register(_Read)
    ctx = ExecutionContext(pool=_Pool(), state=None, registry=reg,
                           abort_event=threading.Event())
    assert ctx.run("ModeGateReadProbe", {}).success is True
    assert RAN == ["read"]


# ── 变异：拆掉判据，拦截必须消失 ──────────────────────────────────────

def test_mutation_without_the_gate_the_pulse_goes_through(monkeypatch):
    """把 ``mode_refusal`` 打成恒放行 ⇒ SAFE 下脉冲真的跑了。

    这条与 ``test_safe_refuses_a_pulse_and_the_skill_never_runs`` 是一对：
    没有它，那条测试可能绿在别的原因上（比如探针本来就跑不起来），而不是
    「这道闸在起作用」。
    """
    bind_mode_source(lambda: "safe")
    monkeypatch.setattr(safety_mod, "mode_refusal",
                        lambda *a, **k: None, raising=True)
    res = _ctx().run("ModeGatePulseProbe", {})
    assert res.success is True and RAN == ["pulse"], (
        "拆掉判据之后脉冲仍被拦住 —— 那说明拦住它的是别的东西，"
        "上面那条测试并没有在测这道闸")


# ── 单源：判据只有一份 ──────────────────────────────────────────────

def _imports_mode_refusal(rel: str) -> bool:
    tree = ast.parse((Path(_MASTV2_ROOT) / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("core.safety"):
                if any(a.name == "mode_refusal" for a in node.names):
                    return True
    return False


@pytest.mark.parametrize("rel", [
    "mast/agents/_shared/safety_mw.py",
    "mast/agents/_shared/skill_forge_tools.py",
    "mast/core/execution_context.py",
])
def test_every_entry_point_delegates_to_the_one_judgement(rel):
    assert _imports_mode_refusal(rel), (
        f"{rel} 没有从 mast.core.safety 取 mode_refusal —— "
        "它要么自己又判了一遍（第二真源），要么根本没有闸")


def test_the_refusal_keys_live_in_exactly_one_module():
    """两个拒绝键各自只许在一个模块里出现字面量。

    第二处字面量 = 第二份判断已经长出来了。扫描器自己也要有正例：
    必须真的在 ``core/safety.py`` 里看见它们，否则「只有一处」可能只是
    扫描器坏了。
    """
    root = Path(_MASTV2_ROOT) / "mast"
    for key in ("safe_mode_tip_processing_blocked", "semi_mode_shallow_only"):
        hits = sorted(p.relative_to(root).as_posix()
                      for p in root.rglob("*.py")
                      if key in p.read_text(encoding="utf-8", errors="ignore"))
        assert hits == ["core/safety.py"], (
            f"'{key}' 出现在 {hits} —— 应当只有 core/safety.py 一处")
