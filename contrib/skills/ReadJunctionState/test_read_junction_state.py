"""ReadJunctionState（只读组合 spec）的测试。

两件事：它过合规判据（子步全是官方只读技能、足迹只读）；以及**真解释器**按这份 spec
走一遍 —— 子技能由替身上下文回放，验证步骤顺序与 ``outputs`` 的表达式真的取得到值。
spec 是数据，不跑一遍，``bias['bias_v']`` 这种键名写错了只会在仪器前现形。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from mast.core.types import SkillResult

_HERE = Path(__file__).resolve().parent
SPEC = json.loads((_HERE / "spec.json").read_text(encoding="utf-8"))

#: 各子技能真实返回的 data 键（见各自的 execute）；值是替身。
_CANNED = {
    "GetBias": {"bias_v": 0.5},
    "GetSetpoint": {"setpoint_a": 1e-10},
    "GetCurrent": {"current_a": 9.8e-11},
    "GetZControllerState": {"controller_on": True, "module_status_name": "On"},
}


class _ReplayCtx:
    """只回放子技能结果的执行上下文。组合技能自己不该发任何 Nanonis 命令。"""

    def __init__(self) -> None:
        # 每次执行一个唯一 run_id：进度旁车文件按 (技能名, run_id) 分开，不会串到别的运行
        self.run_id = f"contrib-test-{uuid.uuid4().hex[:12]}"
        self.calls: list[str] = []

    def run(self, skill_name, params=None, **kwargs):
        self.calls.append(skill_name)
        return SkillResult(skill_name=skill_name, success=True, data=dict(_CANNED[skill_name]))

    def check_abort(self) -> bool:
        return False

    def safe_call(self, *args, **kwargs):
        raise AssertionError("组合技能不该直接发 Nanonis 命令")


@pytest.fixture(scope="module")
def registry():
    from mast.skills.compliance import default_registry

    reg, env = default_registry()
    assert reg is not None, [f.message for f in env]
    return reg


def test_spec_passes_the_compliance_checker(registry) -> None:
    from mast.skills.compliance import check_spec

    rep = check_spec(SPEC, registry=registry)
    assert rep.ok, rep.render(verbose=True)
    assert rep.footprint == "hardware-read-only"
    assert rep.extra.get("effective_safety_level") == "auto"


def test_every_step_is_an_official_read(registry) -> None:
    for node in SPEC["nodes"]:
        meta = registry._get_metadata(registry.get(node["skill"]))
        assert meta.category.value == "read", node["skill"]
        assert meta.safety_level.value == "auto", node["skill"]


def test_the_interpreter_runs_it_in_order_and_fills_the_outputs(registry, tmp_path, monkeypatch) -> None:
    # 组合执行会写一个进度旁车文件（项目根下的 experiments/）；指到临时目录，不碰真实数据
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    from mast.skills.composite.interpreter import make_spec_skill
    from mast.skills.composite.spec import CompositeSpec

    skill = make_spec_skill(CompositeSpec.from_dict(SPEC), registry)()
    meta = skill.metadata()
    assert meta.safety_level.value == "auto"
    assert not meta.capabilities

    ctx = _ReplayCtx()
    res = skill.execute(ctx, {})
    assert res.success, res.error
    assert ctx.calls == ["GetBias", "GetSetpoint", "GetCurrent", "GetZControllerState"]
    assert res.data["outputs"] == {
        "bias_v": 0.5,
        "setpoint_a": 1e-10,
        "current_a": 9.8e-11,
        "z_controller_on": True,
    }
    assert not res.data.get("output_errors")
