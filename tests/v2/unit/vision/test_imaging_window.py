# -*- coding: utf-8 -*-
"""成像条件闸门的参数边界回归。

以窗口内部、外部和缺失条件的合成输入验证：条件不支持原子判读时应输出
无法判断，而不是推断针尖不合格；内置窗口不代表任何站点实验结果。
"""
from __future__ import annotations

import pytest

from mast.vision.imaging_window import (
    ATOMIC_BIAS_MAX_V,
    ATOMIC_SETPOINT_MIN_A,
    check_atomic_window,
    window_for,
)


# ── 纯函数 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bias_v", [0.01, 0.04, 0.08, -0.08, 0.12])
def test_synthetic_conditions_inside_the_window_pass(bias_v):
    """在窗口内部选择多组正负偏压合成输入，验证条件闸门放行。
    """
    v = check_atomic_window(bias_v, 500e-12, "Au(111)")
    assert v.ok, "%.3f V 本该放行：%s" % (bias_v, v.reason)


def test_bias_above_the_configured_window_is_rejected():
    """窗口外的合成偏压应拒判。"""
    v = check_atomic_window(ATOMIC_BIAS_MAX_V * 3, 500e-12, "Au(111)")
    assert not v.ok
    assert v.reason == "bias_out_of_atomic_window"
    assert "不要读成针尖不好" in v.detail_zh, \
        "文案必须明确否掉「针尖不好」这个读法 —— 那正是造成损失的那一步"


def test_setpoint_floor():
    v = check_atomic_window(0.02, 20e-12, "Au(111)")
    assert not v.ok and v.reason == "setpoint_below_atomic_window"
    assert check_atomic_window(0.02, 500e-12, "Au(111)").ok


def test_missing_conditions_do_not_block():
    """读不到条件是「不知道」，不是「不合格」。

    把缺字段当不合格，会让所有缺头信息的旧帧凭空变成判不了 ——
    那是把一道防误判的闸门变成新的误判来源。
    """
    assert check_atomic_window(None, None, "Au(111)").ok
    assert check_atomic_window(None, 500e-12, "Au(111)").ok
    assert check_atomic_window(0.02, None, "Au(111)").ok


def test_unknown_surface_falls_back_to_generic_window():
    assert window_for("NoSuchSurface(999)") == (ATOMIC_BIAS_MAX_V, ATOMIC_SETPOINT_MIN_A)
    assert window_for(None) == (ATOMIC_BIAS_MAX_V, ATOMIC_SETPOINT_MIN_A)
    # 未知衬底不该因此拒绝作答
    assert check_atomic_window(0.02, 500e-12, "NoSuchSurface(999)").ok


# ── 接进判读路径之后的行为（这条才是真正防事故的那一条）────────────────

def _fake_frame(monkeypatch, bias_v, setpoint_a):
    """让判读 skill 读到指定的成像条件，图像本身固定为「没有晶格」的噪声。"""
    import numpy as np

    from mast.skills.builtins import atomic_lattice as AL

    rng = np.random.default_rng(0)
    img = rng.normal(0, 5e-12, (256, 256))
    monkeypatch.setattr(AL, "_load_frame",
                        lambda path, ch: (img, img, 0.0195, ""))
    monkeypatch.setattr(AL, "_frame_conditions",
                        lambda path, ch="Z": (bias_v, setpoint_a))
    # 不用管文件存不存在：_load_frame 已经被换掉，它才是唯一碰磁盘的那一步
    return AL


def test_wrong_bias_gives_undetermined_not_absent(monkeypatch):
    """**核心**：同一张「没有晶格」的图，

    * 条件正常 ⇒ ``absent``（确实没有）
    * 偏压 1.0 V ⇒ ``undetermined``（回答不了）

    两者绝不能混为一谈：前者该升级修针，后者该先把工作点改回来。
    """
    from mast.skills.builtins.atomic_lattice import AssessAtomicResolution

    AL = _fake_frame(monkeypatch, 0.02, 500e-12)
    ok = AssessAtomicResolution().execute(None, {"scan_path": "x.sxm",
                                                 "surface": "Au(111)"})
    assert ok.data["verdict"] == "absent", ok.data

    AL = _fake_frame(monkeypatch, 1.0, 500e-12)
    bad = AssessAtomicResolution().execute(None, {"scan_path": "x.sxm",
                                                  "surface": "Au(111)"})
    assert bad.data["verdict"] == "undetermined", bad.data
    # 人能读的说明
    assert any("偏压" in str(n) for n in (bad.data.get("warnings") or [])), \
        "必须说清是偏压的问题，否则调用方仍会读成针尖不好"
    # 机器能读的码 —— 上层不该靠匹配中文散文来分诊
    win = bad.data.get("imaging_window") or {}
    assert win.get("ok") is False and win.get("reason") == "bias_out_of_atomic_window", win
    assert win.get("bias_v") == 1.0 and win.get("bias_max_v") == ATOMIC_BIAS_MAX_V, win
    # 条件正常那一支必须报 ok，否则这个字段没有区分力
    assert (ok.data.get("imaging_window") or {}).get("ok") is True, ok.data
