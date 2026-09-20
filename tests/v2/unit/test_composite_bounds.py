"""ExecutionContext.run enforces global SafetyGate bounds on composite sub-skills.

审查: composites invoked sub-skills with per-step params
that bypassed the global numeric caps. ExecutionContext.run now applies them.

修复项 2026-06-11: ExecutionContext.run also enforces the APPROVAL GATE at this
leaf call point — a composite sub-step calling MotorMove direction='z-approach'
(the one physically-dangerous action) or any DANGEROUS skill is rejected unless
the context carries approval_source='human'. Previously this path bypassed the
gate entirely (SafetyGateMiddleware sees only the composite's top-level tool
call; SkillExecutor guards only the manual path).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_composite_bounds.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.core.execution_context import ExecutionContext
from mast.core.registry import SkillRegistry
from mast.core.types import (
    ParameterSpec, SafetyLevel, SkillCategory, SkillMetadata, SkillResult,
)
from mast.skills.base import BaseSkill


class FakeSetBias(BaseSkill):
    """bias_v param with NO per-skill cap — only the GLOBAL bound can catch it."""
    def metadata(self):
        return SkillMetadata(
            name="FakeSetBias", version="1.0.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM, description="",
            parameters=[ParameterSpec(name="bias_v", type="float", unit="V",
                                      required=True)],  # no min/max here
        )
    def execute(self, ctx, params):
        return SkillResult(skill_name="FakeSetBias", success=True, data=params)


def _ctx():
    reg = SkillRegistry()
    reg.register(FakeSetBias)
    return ExecutionContext(pool=None, state=None, registry=reg)


def test_out_of_bounds_subskill_rejected():
    res = _ctx().run("FakeSetBias", {"bias_v": 1_000_000.0})  # 1 MV — absurd
    assert res.success is False
    assert "global safety bounds" in res.error


def test_in_bounds_subskill_passes():
    res = _ctx().run("FakeSetBias", {"bias_v": 0.5})
    assert res.success is True
    assert res.data["bias_v"] == 0.5


# ── 修复项 approval gate at the composite sub-step leaf point ─────────────────

class FakeDangerous(BaseSkill):
    def metadata(self):
        return SkillMetadata(
            name="FakeDangerous", version="1.0.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS, description="", parameters=[],
        )
    def execute(self, ctx, params):
        return SkillResult(skill_name="FakeDangerous", success=True, data={})


def _gate_ctx(approval_source: str = "auto"):
    from mast.skills.builtins.motor import MotorMove
    reg = SkillRegistry()
    reg.register(MotorMove)
    reg.register(FakeDangerous)
    return ExecutionContext(
        pool=None, state=None, registry=reg, approval_source=approval_source,
    )


# ── ⑰(2026-08-08):这一节里**两种「要人」被拆开了** ─────────────────────────
#
# 原来五条测试都对着同一句 "Human approval required",因为代码里也确实是同一个
# ``required = "human"``:参数条件硬闸(z-approach)和基线 ``safety_level=DANGEROUS``
# 混在一起,一律拒绝并让人「去审批面板」。
#
# 审批面板对 DANGEROUS 技能已经不再产生任何东西(确认框整条链路割掉了),所以那条
# 拒绝语从此指向一扇不存在的门 —— 而 agent 会照着它反复重试(「一个错的路标比没有
# 路标更坏」)。拆开之后:
#
#   * **硬闸**(z-approach / 粗动驱动电压 / 裸横向粗动)照旧拒绝,而且拒绝语明说
#     「这不是等待审批,没有任何批准会到来」;出路是用户走手动路径;
#   * **基线 DANGEROUS** 转成「执行 + 留痕 + 通知」,与 agent 顶层路径同口径 ——
#     同一个技能不该因为「是不是被复合技能调用的」而有两种安全语义。
#
# 断言因此分成两组,而不是继续共用一个字符串。

_HARD_GATE_MARK = "硬闸拒绝"          # 「硬闸拒绝」


def test_coarse_z_approach_subskill_blocked():
    """The one physically-dangerous action must NOT run as a composite sub-step.

    ⑰ 点名保留:这是拒绝型防护,不弹框、不等人。"""
    res = _gate_ctx().run("MotorMove", {"direction": "z-approach", "steps": 10})
    assert res.success is False
    assert _HARD_GATE_MARK in res.error
    assert "没有任何批准会到来" in res.error, (
        "拒绝语必须说清「这不是在等审批」——审批面板已经不会为它产生任何东西，"
        "指向一扇不存在的门比不给提示更坏")


def test_coarse_z_approach_blocked_even_with_llm_source():
    res = _gate_ctx("llm").run("MotorMove", {"direction": "z-approach", "steps": 10})
    assert res.success is False
    assert _HARD_GATE_MARK in res.error


def test_z_retract_subskill_passes_gate():
    """Retract (away from sample) is safe — gate must not fire. The call then
    fails later at execute (pool=None), which proves it got past the gate."""
    res = _gate_ctx().run("MotorMove", {"direction": "z-retract", "steps": 10})
    assert _HARD_GATE_MARK not in (res.error or "")


def test_dangerous_subskill_now_runs_and_is_noticed():
    """**语义反转的正主。** 原名 ``test_dangerous_subskill_blocked_without_human``。

    基线 DANGEROUS 的子步骤不再被拒 —— 它跑,并且在诊断台账里留一行。
    通知那一半必须一起钉:少了它,这条改动就只是「悄悄放开」。"""
    from mast.core import diagnostics as diag

    before = len([r for r in diag.recent(200, kinds=("notice_only",))
                  if r.get("subject") == "FakeDangerous"])
    res = _gate_ctx().run("FakeDangerous", {})
    assert res.success is True, "基线 DANGEROUS 子步骤又被拒了"
    rows = [r for r in diag.recent(200, kinds=("notice_only",))
            if r.get("subject") == "FakeDangerous"]
    assert len(rows) == before + 1, "跑了却没留下通知"
    assert rows[0]["composite_substep"] is True
    assert "DANGEROUS" in rows[0]["reason"]


def test_dangerous_subskill_runs_with_human_source():
    res = _gate_ctx("human").run("FakeDangerous", {})
    assert res.success is True


def test_a_human_context_does_not_get_a_pointless_notice():
    """approval_source=human ⇒ 本来就有人授权,不该再报「本来会等人批准」。

    这条是通知的**另一侧**:只钉「该通知时通知了」的话,一个恒真的判据也会绿。"""
    from mast.core import diagnostics as diag

    before = len([r for r in diag.recent(200, kinds=("notice_only",))
                  if r.get("subject") == "FakeDangerous"])
    _gate_ctx("human").run("FakeDangerous", {})
    after = len([r for r in diag.recent(200, kinds=("notice_only",))
                 if r.get("subject") == "FakeDangerous"])
    assert after == before


def test_coarse_z_approach_passes_gate_with_human_source():
    res = _gate_ctx("human").run("MotorMove", {"direction": "z-approach", "steps": 10})
    assert _HARD_GATE_MARK not in (res.error or "")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
