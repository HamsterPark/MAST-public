"""贵金属修针流程参数表 —— 出厂值忠于用户口述，覆写越界就丢弃。"""
from __future__ import annotations

import logging

import pytest

from mast.core.noble_tip_workflow import (
    NOBLE_METAL_BASELINE,
    NobleTipWorkflow,
    descend_sequence,
    resolve,
)


# ── 出厂值必须是要求的那些数 ────────────────────────────────────────────

def test_factory_values_match_what_the_operator_described():
    w = NOBLE_METAL_BASELINE
    assert 3.0 <= w.approach_bias_v <= 6.0          # 进针偏压 3~6V
    assert 0.02 <= w.junction_bias_v <= 0.1         # 结电压 20~100mV
    assert w.junction_setpoint_a == pytest.approx(1e-9)   # 结电流上限 1nA
    assert abs(w.pulse_v) == 10.0                   # 脉冲电压 ±10V
    assert w.pulse_width_s == pytest.approx(0.5)    # 脉冲宽度 500ms
    # 判定阈值下界从 10 nm 改为 5 nm。下界钉在**噪声之上**:``step_tol_nm=0.5``
    # 是判定的分辨力,判据低于它就是在把噪声读成成功。上界留着,防有人手滑写成 500。
    assert 1.0 <= w.pulse_success_dz_nm <= 50.0
    assert 0.5 <= w.verify_bias_v <= 2.0            # 验证偏压 500mV~2V
    assert 5e-11 <= w.verify_setpoint_a <= 2e-10    # 验证电流 50~200pA
    assert 100.0 <= w.step_scan_nm <= 200.0         # 台阶扫描尺寸 100nm 或 200nm
    assert 20.0 <= w.flat_region_nm <= 100.0        # 调平区域 20~100nm
    # 起手扎针深度 **500 pm**。
    #
    # 这一行原来对应 ``1.0 <= poke_depth_nm <= 2.0`` 这一档更保守的深度范围。
    # 现行范本起手 **−500 pm**,收敛到 200 pm(序列 500→500→500→200)。
    #
    # 方向选择往浅里走的物理依据:扎针尖的破坏性远小于脉冲的破坏性,大的扎针尖不代表
    # 破坏性更大。本仓此前一直把扎针当成比脉冲更危险的动作,方向反了。
    assert w.poke_depth_nm == pytest.approx(0.5)    # 起手 −500 pm
    assert w.poke_dwell_s == pytest.approx(0.5)     # 停留 500ms,可调
    # 验证当前配方的可配置默认，不携带真实试验的成功率或深度记录。
    assert w.critical_start_pm == pytest.approx(300.0)
    assert w.critical_step_pm == pytest.approx(50.0)     # 每级 +50 pm
    assert w.descend_pulse_v == (7.0, 5.0, 3.0)     # 脉冲电压逐级降低:7V,5V,3V


def test_every_stage_has_an_action_budget():
    """预算不是重试次数：打不动就该报告，不是无限打下去。"""
    w = NOBLE_METAL_BASELINE
    assert w.pulse_budget > 0 and w.poke_budget > 0 and w.max_verify_rounds > 0


# ── 覆写 ────────────────────────────────────────────────────────────────────

def test_none_means_not_given_not_zero():
    """技能的可选参数不填时就是 None —— 那是「按流程表来」，不是「设成 0」。"""
    assert resolve({"pulse_v": None, "poke_depth_nm": None}) is NOBLE_METAL_BASELINE


def test_explicit_override_wins():
    w = resolve({"pulse_v": 7.0, "pulse_budget": 5})
    assert w.pulse_v == 7.0 and w.pulse_budget == 5
    assert w.poke_depth_nm == NOBLE_METAL_BASELINE.poke_depth_nm


def test_out_of_range_is_dropped_not_clamped(caplog):
    """夹紧会让调用方以为自己设的是 X 而实际跑的是 Y —— 这里每个数字都会变成
    一次真实的硬件动作。"""
    with caplog.at_level(logging.WARNING):
        w = resolve({"pulse_v": 25.0})
    assert w.pulse_v == NOBLE_METAL_BASELINE.pulse_v, "越界值不许夹到边界"
    # 丢弃必须留下痕迹：静默忽略和夹紧一样会让人以为跑的是自己填的值。
    assert any("忽略" in r.getMessage() for r in caplog.records)


def test_unknown_keys_and_junk_are_ignored():
    w = resolve({"not_a_field": 1, "pulse_v": "十伏"})
    assert w == NOBLE_METAL_BASELINE


def test_int_fields_stay_ints():
    w = resolve({"pulse_budget": 7.0, "critical_repeat_n": 3.4})
    assert isinstance(w.pulse_budget, int) and w.pulse_budget == 7
    assert isinstance(w.critical_repeat_n, int) and w.critical_repeat_n == 3


def test_descend_sequence_drops_out_of_range_entries(caplog):
    with caplog.at_level(logging.WARNING):
        w = resolve({"descend_pulse_v": [7.0, 50.0, 3.0]})
    assert w.descend_pulse_v == (7.0, 3.0)


def test_workflow_is_frozen():
    """策略是数据，不是可变全局状态。"""
    with pytest.raises(Exception):
        NOBLE_METAL_BASELINE.pulse_v = 1.0      # type: ignore[misc]


# ── qPlus 分叉 ──────────────────────────────────────────────────────────────

def test_descend_sequence_is_empty_on_qplus(monkeypatch):
    """音叉上那几伏可能把叉臂也一起修了。"""
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: {
        "id": 1, "name": "qp", "material": "PtIr", "form": "qplus"})
    assert descend_sequence(NOBLE_METAL_BASELINE) == ()


def test_descend_sequence_is_present_on_a_metal_tip(monkeypatch):
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: {
        "id": 2, "name": "w", "material": "W", "form": "stm_wire"})
    assert descend_sequence(NOBLE_METAL_BASELINE) == (7.0, 5.0, 3.0)


def test_tip_type_has_exactly_one_source_of_truth():
    """针尖是不是 qPlus 只有针尖登记说了算 —— 不许在 instrument_profile 里
    再开一个设置项。两个真源迟早各自漂移。"""
    from mast.core.instrument_profile import ALL_KEYS
    assert "tip_type" not in ALL_KEYS


def test_unregistered_tip_is_not_treated_as_qplus(monkeypatch):
    """读不到就当不是 qPlus（真正会戳到音叉的下压动作另有一道软门拦着）。"""
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: None)
    assert descend_sequence(NOBLE_METAL_BASELINE) == (7.0, 5.0, 3.0)


# ── 与针尖方案表的分工 ──────────────────────────────────────────────────────

def test_workflow_does_not_duplicate_the_tip_envelope():
    """安全包络（max_abs_pulse_v / max_poke_depth_m）属于按针尖类型的方案表。
    在这里复制一份，两处迟早各自漂移。"""
    names = {f for f in NobleTipWorkflow.__dataclass_fields__}
    assert not {"max_abs_pulse_v", "max_poke_depth_m", "max_pulse_count"} & names


def test_workflow_does_not_carry_avoid_radii():
    """污染半径已经在 instrument_profile 里，扫描地图的避让圆用的就是它们。"""
    names = {f for f in NobleTipWorkflow.__dataclass_fields__}
    assert not any("avoid" in n or "radius" in n for n in names)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
