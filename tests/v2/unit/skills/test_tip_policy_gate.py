"""针尖类型门与参数方案接线的合成测试。

qPlus 门默认关闭，环境开关开启后必须拒绝未显式放行的下压操作；
未登记类型时沿用默认行为。参数包络与方案溯源分别验证。
"""

from __future__ import annotations

import pytest

from mast.core import instrument_profile as iprof
from mast.core import tip_state
from mast.skills.builtins._tip_policy import (
    apply_tip_policy,
    policy_fields_for_result,
    qplus_gate,
)

_QPLUS = {"id": "q1", "name": "qPlus #1", "material": "PtIr",
          "fabrication": "cut", "form": "qplus"}
_W = {"id": "w1", "name": "W-etched #1", "material": "W",
      "fabrication": "etched", "form": "stm_wire"}


@pytest.fixture(autouse=True)
def _clean():
    tip_state.set_current_tip(None)
    iprof.set_persist_sink(None)
    iprof.set_profile({})
    yield
    tip_state.set_current_tip(None)
    iprof.set_persist_sink(None)
    iprof.set_profile({})


# ── qPlus 门 ────────────────────────────────────────────────────────────────

def test_the_gate_is_off_by_default(monkeypatch) -> None:
    """默认关闭类型门；显式启用时的拒绝行为另行验证。"""
    monkeypatch.delenv("MAST_QPLUS_POKE_GUARD", raising=False)
    tip_state.set_current_tip(_QPLUS)
    assert qplus_gate("TipShape", {}) is None


def test_the_switch_is_read_every_call_not_at_import(monkeypatch) -> None:
    """开关**每次调用都读**,不是 import 时读一次。

    只在 import 时生效的开关,在冻结的应用里等于没有 —— 进程一起来就定死了,
    再也改不动(``narration.enabled()`` 同形)。这里在**同一个进程里**开关各来
    一次:两次结果必须不同,否则那个开关是个摆设。
    """
    tip_state.set_current_tip(_QPLUS)
    monkeypatch.delenv("MAST_QPLUS_POKE_GUARD", raising=False)
    assert qplus_gate("TipShape", {}) is None
    monkeypatch.setenv("MAST_QPLUS_POKE_GUARD", "1")
    assert qplus_gate("TipShape", {}) is not None


def test_the_gate_still_works_when_it_is_turned_back_on(monkeypatch) -> None:
    """开回来之后**拦得住,而且话说得全**。

    默认值改了 ≠ 门可以烂掉。一道关着的门还得能用,否则哪天有人开回来,
    等着他的是一道**看着在防护、其实早就坏了**的门。
    """
    monkeypatch.setenv("MAST_QPLUS_POKE_GUARD", "1")
    tip_state.set_current_tip(_QPLUS)
    res = qplus_gate("TipShape", {})
    assert res is not None and res.success is False
    assert "qPlus" in res.error
    assert "不可逆" in res.error
    assert "allow_on_qplus" in res.error


def test_gate_allows_with_the_explicit_override() -> None:
    tip_state.set_current_tip(_QPLUS)
    assert qplus_gate("TipShape", {"allow_on_qplus": True}) is None


def test_gate_allows_a_normal_wire_tip() -> None:
    tip_state.set_current_tip(_W)
    assert qplus_gate("TipShape", {}) is None


def test_gate_is_fail_open_when_nothing_is_registered() -> None:
    """未登记 ≠ 没有针。系统不知道装的是什么,不该拦住一个可能完全正常的操作。"""
    tip_state.set_current_tip(None)
    assert qplus_gate("TipShape", {}) is None


def test_gate_refusal_names_the_skill_and_the_tip(monkeypatch) -> None:
    monkeypatch.setenv("MAST_QPLUS_POKE_GUARD", "1")
    tip_state.set_current_tip(_QPLUS)
    res = qplus_gate("ShapeTipOnSurface", {})
    assert "ShapeTipOnSurface" in res.error
    assert "qPlus #1" in res.error


# ── 方案接线 ────────────────────────────────────────────────────────────────

def test_policy_fills_a_missing_parameter() -> None:
    tip_state.set_current_tip(_W)
    params, plan = apply_tip_policy({}, ("pulse_v",))
    assert params["pulse_v"] == 5.0            # 钨那一档
    assert plan is not None and plan.ok


def test_policy_respects_an_explicit_parameter() -> None:
    tip_state.set_current_tip(_W)
    params, plan = apply_tip_policy({"pulse_v": 4.0}, ("pulse_v",))
    assert params["pulse_v"] == 4.0
    assert plan.trace["pulse_v"] == "explicit"


def test_policy_renames_fields_to_skill_parameter_names() -> None:
    """方案表说 shaper_bias_v,TipShape 的参数叫 bias_v —— 映射不能错位。"""
    tip_state.set_current_tip(_W)
    params, _ = apply_tip_policy({}, ("shaper_bias_v",), {"shaper_bias_v": "bias_v"})
    assert "bias_v" in params
    assert "shaper_bias_v" not in params


def test_policy_refuses_an_over_envelope_explicit_value() -> None:
    """显式参数超过当前方案包络时必须拒绝，不能静默截断。"""
    tip_state.set_current_tip(_QPLUS)
    _, plan = apply_tip_policy({"pulse_v": 11.0}, ("pulse_v",))
    assert not plan.ok


def test_result_fields_carry_the_source_trace() -> None:
    """一个数字从哪来,事后必须查得到 —— 结果里就带着。"""
    tip_state.set_current_tip(_W)
    _, plan = apply_tip_policy({}, ("pulse_v",))
    fields = policy_fields_for_result(plan)
    assert "tip_policy" in fields
    assert fields["tip_registered"] is True
    assert fields["tip_name"] == "W-etched #1"


def test_result_fields_say_when_no_tip_is_registered() -> None:
    tip_state.set_current_tip(None)
    _, plan = apply_tip_policy({}, ("pulse_v",))
    assert policy_fields_for_result(plan)["tip_registered"] is False


def test_policy_never_breaks_the_skill_when_the_resolver_is_broken(monkeypatch) -> None:
    """方案表读不到绝不能让修针技能失败 —— 技能自带的默认值仍在。"""
    import mast.core.tip_conditioning_resolver as res

    def boom(*a, **k):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(res, "resolve_conditioning", boom)
    params, plan = apply_tip_policy({"pulse_v": 3.0}, ("pulse_v",))
    assert params["pulse_v"] == 3.0
    assert plan is None


# ── 技能 metadata 上的旁路参数 ──────────────────────────────────────────────

def test_poke_skills_declare_the_override_parameter() -> None:
    """门是软的,模型必须**看得见**怎么签字,否则它只会重试同一个被拒的调用。"""
    from mast.skills.builtins.tip_shaper import TipShape
    from mast.skills.composite.shape_tip_on_surface import ShapeTipOnSurface

    for cls in (TipShape, ShapeTipOnSurface):
        names = {p.name for p in cls().metadata().parameters}
        assert "allow_on_qplus" in names, f"{cls.__name__} 没有暴露旁路参数"


def test_conditioning_skills_no_longer_require_a_pulse_voltage() -> None:
    """既定的那条:别让不懂 STM 的模型自己想参数。

    pulse_v 从 required 改成可选,正是「移除发明数字的诱因」—— 留空就查方案表。"""
    from mast.skills.composite.condition_tip import ConditionTip
    from mast.skills.composite.tip_pulse import TipPulse

    for cls in (TipPulse, ConditionTip):
        spec = {p.name: p for p in cls().metadata().parameters}["pulse_v"]
        assert spec.required is False, f"{cls.__name__}.pulse_v 仍是必填"
        # 2026-08-25 中文化：原来钉的是英文串 "policy table"。
        # **拿措辞当规则的代理，措辞一换就红、而规则被删了又可能不红。**
        # 改成钉实质：这段话必须告诉模型「别自己填，留空就去查针尖策略表」。
        d = spec.description
        assert "策略表" in d, (
            f"{cls.__name__}.pulse_v 没说清「留空就查针尖策略表」：{d}")
        assert "别填" in d or "留空" in d or "不要填" in d, (
            f"{cls.__name__}.pulse_v 没说清「没理由就别自己填」：{d}")


def test_conditioning_validate_refuses_over_envelope() -> None:
    """包络检查在 validate 层 —— 必须在任何硬件调用之前拦住。

    9 V → 11 V 的理由同上一条:qPlus 包络已是 10 V。
    """
    from mast.skills.composite.tip_pulse import TipPulse

    tip_state.set_current_tip(_QPLUS)
    errors = TipPulse().validate_params({"pulse_v": 11.0})
    assert any("超出" in e for e in errors)


def test_conditioning_validate_passes_when_within_envelope() -> None:
    from mast.skills.composite.tip_pulse import TipPulse

    tip_state.set_current_tip(_QPLUS)
    assert TipPulse().validate_params({"pulse_v": 2.0}) == []


def test_conditioning_validate_passes_with_no_pulse_voltage_at_all() -> None:
    """留空是被鼓励的用法,不能因此报「缺必填参数」。"""
    from mast.skills.composite.tip_pulse import TipPulse

    tip_state.set_current_tip(_W)
    assert TipPulse().validate_params({}) == []
