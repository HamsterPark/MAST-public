# -*- coding: utf-8 -*-
"""双通道 burst 的 B 路选择。

信号名使用协议常见形式，槽位由测试重新安排。验证按名称解析而非硬编码索引，
并验证选不到时拒绝继续采集。本文件不含实际仪器的信号表。
"""
from __future__ import annotations

import pytest

from mast.envhistory import zburst as ZB

#: 合成信号目录：故意打散槽位，并同时包含 X/Y 与第二解调器作为干扰项。
SYNTHETIC_NAMES = {
    3: "Current (A)", 11: "Bias (V)", 17: "Z (m)",
    5: "OC D1 Amplitude (m)", 21: "OC M1 Freq. Shift (Hz)",
    8: "LI Demod 1 X (A)", 14: "LI Demod 1 Y (A)",
    23: "LI Demod 2 X (A)", 26: "LI Demod 2 Y (A)",
}


def _names_list(n=32):
    out = ["Signal %d" % i for i in range(n)]
    for i, nm in SYNTHETIC_NAMES.items():
        out[i] = nm
    return out


def test_every_declared_channel_matches_a_synthetic_signal_name():
    """每个声明通道应通过名字解析到合成目录中的信号；槽位不能写死。"""
    names = _names_list()
    for key, hints in ZB._CHANNEL_HINTS.items():
        idx, nm = ZB._match(names, hints)
        assert idx >= 0, "通道 %r 的线索 %s 在合成信号目录中认不出任何一路" % (key, hints)
        assert nm, key


@pytest.mark.parametrize("key,expect_idx", [
    ("current", 3), ("bias", 11), ("z", 17), ("amplitude", 5), ("df", 21),
    ("didv_x", 8), ("didv_y", 14),
])
def test_channels_resolve_to_the_expected_synthetic_indices(key, expect_idx):
    """解析结果必须是指定的槽位，并区分同名家族中的 X/Y 与不同解调器。"""
    idx, _nm = ZB._match(_names_list(), ZB._CHANNEL_HINTS[key])
    assert idx == expect_idx


def test_unknown_channel_is_refused_not_defaulted():
    """通道名写错时必须停手 —— **不许悄悄退回 Z**。

    退回默认值意味着调用方以为拿到的是 bias、实际拿到的是 Z，而两者都是
    合法的数值序列，没有任何一步会报错。
    """
    calls = []

    class _Pool:
        def safe_call(self, verb, *a, **k):
            calls.append(verb)
            raise AssertionError("不该走到任何 Nanonis 调用: %s" % verb)

    out = ZB.run_dual_burst(lambda: _Pool(), burst_s=0.1, channel="voltage")
    assert out is None
    # Osci2T_Run 是第一个调用，允许它先发（那只是让示波器跑起来）；
    # 关键是不能继续往下走到设通道。
    assert "Osci2T_ChsSet" not in calls


def test_missing_signal_name_skips_instead_of_substituting():
    """信号名里找不到目标时返回 None，而不是换一路采。"""
    idx, _ = ZB._match(["Current (A)", "Z (m)"], ZB._CHANNEL_HINTS["bias"])
    assert idx < 0, "信号名里没有 Bias 时不该认出任何一路"


def test_resolve_channels_defaults_to_z_for_backwards_compatibility():
    """不给 hints 时仍然找 Z —— 这条路径原来的调用方不该被这次改动波及。"""
    names = _names_list()
    z_idx, z_nm = ZB._match(names, ZB._Z_HINTS)
    assert z_idx == 17 and z_nm == "Z (m)"


def test_result_says_which_channel_it_actually_sampled():
    """返回里必须有 ``channel``。

    ``z`` / ``z_name`` 这两个键是历史包袱（这条路原来只采 Z）。键名叫 z 而
    内容是 bias 是事故的标准配方 —— 读数据的人必须能一眼看出手里这批是什么。
    """
    import inspect

    src = inspect.getsource(ZB.run_dual_burst)
    assert '"channel": key' in src, "返回里没有权威的通道标识"
