"""等待条件的数必须**真的**来自用户填的那个参数。

## 这个文件为什么存在

``ConditionSpec`` 是 frozen dataclass,写在模板里 —— 于是「等到 5 K」这个阈值
在模板作者手上;而 ``params_schema`` 里又有一个 ``target_temperature_k`` 让用户
填。两边都不知道对方存在,结果是**填的人以为它生效了**,机器等的是模板里那个数,
而且这件事在任何日志、任何面板上都对不出来。

``_smoke`` 模板的注释把这一步明确留给了 M1-c(「这两个数在 approve 冻结参数时
渲染进来」)。这里钉的就是它,连同它的反面:

* **取不到就是步失败**,不许退回模板里的占位值(退回 = 上面那句话的另一种写法);
* 解出来的数**存进等待记录**,判定/面板/审计从此看同一组数字;
* 拼错的引用在 **approve 之前**就被校验器拦下,而不是凌晨三点进等待那一刻。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.conduct.spec import ConditionSpec, ParamSpec  # noqa: E402
from mast.conduct.validator import validate_spec  # noqa: E402

from tests.v2.unit.conduct._harness import (  # noqa: E402
    FakeReading, FakeTemperature, build, retract_step, spec as make_spec,
    stage, wait_step,
)

TARGET = ParamSpec(name="target_k", type="float", unit="K", min_value=0.0,
                   max_value=400.0, help="等到多少 K 以下")
STALE = ParamSpec(name="stale_s", type="float", unit="s", min_value=1.0,
                  max_value=86400.0, help="多旧算读不到")


def _bound_condition(**kw):
    return ConditionSpec(signal="temperature_k", op="<=",
                         value=400.0, stale_after_s=120.0,
                         value_ref="params.target_k",
                         stale_after_ref="params.stale_s", **kw)


def _spec_with_bound_wait():
    return make_spec([
        stage("W", [retract_step("W.00"),
                    wait_step("W.01", kind="condition",
                              condition=_bound_condition())]),
    ], params=(TARGET, STALE))


# ── 1. 纯函数:解出来的是参数的数,不是占位值 ─────────────────────

def test_resolve_prefers_the_operator_parameter_over_the_placeholder():
    got = _bound_condition().resolve({"target_k": 5.0, "stale_s": 600.0})
    assert got["value"] == 5.0 and got["stale_after_s"] == 600.0


def test_unbound_fields_keep_the_template_literal():
    """没绑定的那一项,模板里的字面值**就是**它的真值,不该被抹掉。"""
    got = _bound_condition(hold_s=1800.0).resolve({"target_k": 5.0,
                                                   "stale_s": 600.0})
    assert got["hold_s"] == 1800.0


def test_a_missing_parameter_raises_instead_of_falling_back():
    with pytest.raises(KeyError) as exc:
        _bound_condition().resolve({"stale_s": 600.0})
    assert "不兜底" in str(exc.value)


def test_a_non_numeric_parameter_raises():
    with pytest.raises(KeyError):
        _bound_condition().resolve({"target_k": "5", "stale_s": 600.0})
    with pytest.raises(KeyError):
        _bound_condition().resolve({"target_k": True, "stale_s": 600.0})


def test_a_ref_outside_the_params_namespace_is_refused_at_import_time():
    """模板 import 的那一刻就炸 —— 最早、最便宜的那道闸。"""
    with pytest.raises(ValueError) as exc:
        ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                      stale_after_s=60.0, value_ref="steps.S1.02.temp_k")
    assert "params." in str(exc.value)


def test_a_resolved_stale_window_of_zero_is_refused():
    """0 会让「读不到」重新变成「还没到」—— 那正是 stale_after_s 存在的理由。"""
    with pytest.raises(KeyError):
        _bound_condition().resolve({"target_k": 5.0, "stale_s": 0.0})


# ── 2. 校验器:拼错的引用在 approve 之前就红 ──────────────────────

def test_validator_catches_a_ref_to_an_undeclared_parameter():
    bad = make_spec([
        stage("W", [retract_step("W.00"),
                    wait_step("W.01", kind="condition",
                              condition=ConditionSpec(
                                  signal="temperature_k", op="<=", value=5.0,
                                  stale_after_s=60.0,
                                  value_ref="params.typo_k"))]),
    ], params=(TARGET,))
    rep = validate_spec(bad, skills={}, analyses=[])
    codes = {f.code for f in rep.errors}
    assert "binding_not_produced" in codes
    assert any("typo_k" in f.message for f in rep.errors)


def test_validator_catches_a_ref_to_a_non_numeric_parameter():
    text_param = ParamSpec(name="note", type="str", help="随便写点什么")
    bad = make_spec([
        stage("W", [retract_step("W.00"),
                    wait_step("W.01", kind="condition",
                              condition=ConditionSpec(
                                  signal="temperature_k", op="<=", value=5.0,
                                  stale_after_s=60.0,
                                  value_ref="params.note"))]),
    ], params=(text_param,))
    rep = validate_spec(bad, skills={}, analyses=[])
    assert "binding_unparseable" in {f.code for f in rep.errors}


def test_a_correctly_bound_condition_passes_the_validator():
    rep = validate_spec(_spec_with_bound_wait(), skills={}, analyses=[])
    # 只看绑定这一族:这份 spec 用的是 harness 的假技能名,注册表检查必然报
    # unknown_skill,那是另一条规则的事。
    binding_codes = {f.code for f in rep.errors
                     if f.code.startswith("binding_")}
    assert not binding_codes, rep.describe()


# ── 3. Director:判定用的是解出来的那组数 ────────────────────────

def test_the_director_waits_on_the_operator_threshold(tmp_path):
    """填 5 K ⇒ 读到 300 K 时**不能**放行(模板占位值是 400 K)。"""
    temp = FakeTemperature(FakeReading(value_k=300.0, age_s=1.0))
    rig = build(tmp_path, _spec_with_bound_wait(),
                params={"target_k": 5.0, "stale_s": 600.0}, temperature=temp)
    rig.run_until("waiting_condition", limit=10)
    assert rig.row()["status"] == "waiting_condition"
    wait = rig.row()["active_wait"]
    assert wait["cond_value"] == 5.0, (
        "等待记录里存的还是模板占位值 —— 用户填的 5 K 一路没到达判定")
    rig.tick(2)
    assert rig.row()["status"] == "waiting_condition", (
        "300 K 放行了 —— 判定用的是模板里那个 400 K,而不是人填的 5 K")

    temp.reading = FakeReading(value_k=4.2, age_s=1.0)
    rig.tick(2)
    # 等待解除后这份 spec 没有别的步了,所以 running 一拍就走到 completed ——
    # 要断言的是「不再等了」,不是某一个具体的终点。
    assert rig.row()["status"] in ("running", "completed")
    assert rig.row()["active_wait"] in (None, {})


def test_the_director_uses_the_operator_stale_window(tmp_path):
    """读数 300 s 旧、人填的窗口是 60 s ⇒ 判**读不到**(占位窗口 120 s 也一样,
    但这里要证明用的是人填的那个)。"""
    temp = FakeTemperature(FakeReading(value_k=4.2, age_s=300.0))
    rig = build(tmp_path, _spec_with_bound_wait(),
                params={"target_k": 5.0, "stale_s": 60.0}, temperature=temp)
    rig.run_until("waiting_condition", limit=10)
    assert rig.row()["active_wait"]["cond_stale_after_s"] == 60.0
    rig.tick(2)
    row = rig.row()
    assert row["status"] == "waiting_operator", (
        "读数太旧却当成「还没到」干等下去 —— 读不到 ≠ 没到")
    assert "读不到" in row["status_reason"]


def test_a_missing_parameter_fails_the_step_and_never_enters_the_wait(tmp_path):
    """参数没填齐 ⇒ 这一步失败停下来问人,**不进等待**。

    进了等待才是最坏的:它会安安静静地按模板里那个占位阈值等下去,而没有任何
    地方显示「你填的那个数没生效」。
    """
    rig = build(tmp_path, _spec_with_bound_wait(),
                params={"stale_s": 600.0})     # 少了 target_k
    rig.tick(6)
    row = rig.row()
    assert row["status"] == "waiting_operator"
    assert row["active_wait"] in (None, {}), "不该进等待"
    assert "target_k" in row["status_reason"]
