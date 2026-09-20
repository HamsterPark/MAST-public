# -*- coding: utf-8 -*-
"""ConfigureScan must verify both readback and geometric reachability.

Synthetic frame coordinates and independently selected calibration factors
exercise exact matches, clamping, missing readings and voltage-limit changes.
The returned coordinates must come from readback rather than echoing the request."""
from __future__ import annotations

import math

import pytest

from mast.skills.builtins.imaging import (
    ConfigureScan,
    frame_exceeds,
    frame_extent,
    frame_readback_mismatch,
)


# ── 包络几何 ───────────────────────────────────────────────────────────
def test_extent_of_an_unrotated_frame():
    x0, x1, y0, y1 = frame_extent(600e-9, -550e-9, 1.8e-6, 1.8e-6, 0.0)
    assert x0 == pytest.approx(-300e-9)
    assert x1 == pytest.approx(1500e-9)
    assert y0 == pytest.approx(-1450e-9)
    assert y1 == pytest.approx(350e-9)


def test_rotation_grows_the_extent():
    """转 45° 的框对角线伸出 √2 倍 —— 只比 center ± size/2 会系统性少算。"""
    _, x1a, _, _ = frame_extent(0, 0, 1e-6, 1e-6, 0.0)
    _, x1b, _, _ = frame_extent(0, 0, 1e-6, 1e-6, 45.0)
    assert x1a == pytest.approx(0.5e-6)
    assert x1b == pytest.approx(0.5e-6 * math.sqrt(2), rel=1e-6)


def test_exceeds_reports_which_edge_and_by_how_much():
    over = frame_exceeds(600e-9, -550e-9, 1.8e-6, 1.8e-6, 0.0, 1200e-9, 1100e-9)
    assert over is not None
    assert over["x_high_m"] == pytest.approx(300e-9, abs=1e-10)
    assert over["y_low_m"] == pytest.approx(350e-9, abs=1e-10)
    assert "x_low_m" not in over and "y_high_m" not in over


def test_frame_inside_range_is_not_flagged():
    assert frame_exceeds(0, 0, 1e-6, 1e-6, 0.0, 1200e-9, 1100e-9) is None


def test_unknown_half_range_is_not_a_pass():
    """半程读不到时不作断言（返回 None = 不注解），**不是**「没超」。"""
    assert frame_exceeds(5e-6, 0, 1e-6, 1e-6, 0.0, None, None) is None


# ── ★ 读回比对 ─────────────────────────────────────────────────────────
REQ = (600e-9, -550e-9, 1.8e-6, 1.8e-6, 0.0)


def test_exact_readback_matches():
    assert frame_readback_mismatch(REQ, list(REQ)) is None


def test_float32_rounding_is_tolerated():
    """Nanonis 内部是 float32 —— 拿等号比会把每一次正常配置都判成被夹。"""
    got = [float(f"{v:.7g}") for v in REQ]
    assert frame_readback_mismatch(REQ, got) is None


def test_clamped_center_is_caught():
    """仪器把中心夹回量程内 —— 这正是要抓的那件事。"""
    got = [1200e-9 - 0.9e-6, -550e-9, 1.8e-6, 1.8e-6, 0.0]
    m = frame_readback_mismatch(REQ, got)
    assert m is not None and "center_x_m" in m
    assert m["center_x_m"]["requested"] == pytest.approx(600e-9)


def test_shrunk_size_is_caught():
    m = frame_readback_mismatch(REQ, [600e-9, -550e-9, 1.4e-6, 1.8e-6, 0.0])
    assert m is not None and "width_m" in m


def test_rotated_frame_is_caught():
    m = frame_readback_mismatch(REQ, [600e-9, -550e-9, 1.8e-6, 1.8e-6, 30.0])
    assert m is not None and "angle_deg" in m


def test_unreadable_is_not_a_match():
    """**读不到不是对上了。** 这一条是整个闸门的成败所在。"""
    assert frame_readback_mismatch(REQ, None) == {"unreadable": True}
    assert frame_readback_mismatch(REQ, [1, 2]) == {"unreadable": True}
    assert frame_readback_mismatch(REQ, ["x"] * 5) == {"unreadable": True}
    assert frame_readback_mismatch(REQ, [float("nan")] * 5) == {"unreadable": True}


# ── execute：桩 ────────────────────────────────────────────────────────
class _Rec:
    def __init__(self, rv=None, error=""):
        self.return_value = rv
        self.error = error


class _Ctx:
    """readback=None → Scan_FrameGet 读不回；否则给这五个值。"""

    def __init__(self, readback="echo", frameget_error="", limits=None,
                 calib_error=""):
        self.readback = readback
        self.frameget_error = frameget_error
        # (enabled, x_low, x_high, y_low, y_high) —— None = 合成的未启用限位（±10 V）
        self.limits = limits if limits is not None else (False, -10.0, 10.0,
                                                         -10.0, 10.0)
        self.calib_error = calib_error
        self.frame_set = None
        self.calls: list[str] = []

    def safe_call(self, method, *args):
        self.calls.append(method)
        if method == "Scan_FrameSet":
            self.frame_set = list(args)
            return _Rec()
        if method == "Scan_FrameGet":
            if self.frameget_error:
                return _Rec(error=self.frameget_error)
            if self.readback == "echo":
                v = self.frame_set or [0.0, 0.0, 1e-6, 1e-6, 0.0]
            elif self.readback is None:
                return _Rec(rv=("", b"", None))
            else:
                v = list(self.readback)
            return _Rec(rv=("", b"", v))
        if method == "Piezo_CalibrGet":
            if self.calib_error:
                return _Rec(error=self.calib_error)
            return _Rec(rv=("", b"", [1.2e-7, 1.1e-7, 3e-8]))
        if method == "Piezo_XYZLimitsGet":
            en, xl, xh, yl, yh = self.limits
            return _Rec(rv=("", b"", [1 if en else 0, xl, xh, yl, yh, -10.0, 10.0]))
        if method == "Signals_NamesGet":
            return _Rec(rv=("", b"", [1, 1, ["Z (m)", "Current (A)"]]))
        return _Rec()


P = {"center_x_m": 600e-9, "center_y_m": -550e-9,
     "width_m": 1.8e-6, "height_m": 1.8e-6, "angle_deg": 0.0, "line_time_s": 0.04}


def _run(ctx, **over):
    p = dict(P)
    p.update(over)
    return ConfigureScan().execute(ctx, p)


def test_matching_readback_succeeds_and_returns_the_readback_not_the_echo():
    # 量程内的框 —— 默认的 P 是独立构造的越界帧，会先撞越界闸门
    ctx = _Ctx(readback="echo")
    res = _run(ctx, center_x_m=100e-9, center_y_m=0.0, width_m=1e-6, height_m=1e-6)
    assert res.success is True
    assert res.data["frame_verified"] is True
    assert "Scan_FrameGet" in ctx.calls
    # 返回的是读回值
    assert res.data["center_x_m"] == pytest.approx(100e-9)
    assert res.data["requested_frame"][0] == pytest.approx(100e-9)


def test_clamped_frame_is_refused_not_silently_used():
    """**拒绝，不夹紧。** 按夹紧后的框扫出来的图坐标是假的，而图看着完全正常。"""
    ctx = _Ctx(readback=[300e-9, -550e-9, 1.8e-6, 1.8e-6, 0.0])
    res = _run(ctx)
    assert res.success is False
    assert "没有接受" in (res.error or "")
    assert res.data["frame_mismatch"]["center_x_m"]["readback"] == pytest.approx(300e-9)


def test_unreadable_frame_is_refused():
    ctx = _Ctx(readback=None)
    res = _run(ctx)
    assert res.success is False
    assert "读不回" in (res.error or "")


def test_frameget_error_is_refused_not_ignored():
    ctx = _Ctx(frameget_error="link down")
    res = _run(ctx)
    assert res.success is False
    assert "读不回" in (res.error or "")


def test_out_of_range_frame_is_refused():
    """读回与请求一致也不能证明几何可达。
    合成框同时越过正 X 与负 Y 边界时，必须拒绝并报告对应超出量。"""
    ctx = _Ctx(readback="echo")
    res = _run(ctx)
    assert res.success is False
    assert "伸出压电量程" in (res.error or "")
    over = res.data["frame_exceeds_piezo_range"]
    assert "x_high_m" in over and "y_low_m" in over
    assert over["x_high_m"] == pytest.approx(300e-9, abs=1e-9)


def test_in_range_frame_passes():
    ctx = _Ctx(readback="echo")
    res = _run(ctx, center_x_m=0.0, center_y_m=0.0, width_m=1e-6, height_m=1e-6)
    assert res.success is True
    assert res.data["frame_exceeds_piezo_range"] is None
    assert res.data["piezo_half_x_m"] == pytest.approx(1200e-9, abs=1e-9)


def test_narrowed_piezo_limits_shrink_the_usable_range():
    """用户在 Piezo Calibration 里收窄了电压限位 ⇒ 可达范围跟着变小。
    只看 calibration 会放过一个其实会被夹的框。"""
    ctx = _Ctx(readback="echo", limits=(True, -5.0, 5.0, -5.0, 5.0))
    res = _run(ctx, center_x_m=0.0, center_y_m=0.0, width_m=1.4e-6, height_m=1.4e-6)
    assert res.success is False, "合成的 ±600/550 nm 半程装不下 ±700 nm 的框"
    ctx2 = _Ctx(readback="echo", limits=(False, -5.0, 5.0, -5.0, 5.0))
    res2 = _run(ctx2, center_x_m=0.0, center_y_m=0.0, width_m=1.4e-6, height_m=1.4e-6)
    assert res2.success is True, "限位没启用时不该拿它去卡"


def test_calibration_already_includes_the_hva_gain():
    """DAC 端的校准系数已经包含放大增益，不应重复乘算。
    用独立选定的压电灵敏度与增益构造该系数，再验证电压半程到位移半程的换算。"""
    from mast.skills.builtins.imaging import piezo_half_range_m
    sens_at_piezo, hva_gain = 6e-9, 20.0
    calibration = 1.2e-7
    assert sens_at_piezo * hva_gain == pytest.approx(calibration, rel=2e-5)
    assert piezo_half_range_m(calibration) == pytest.approx(1200e-9, abs=1e-9)


def test_unreadable_half_range_does_not_pass_as_in_range():
    """**「读不到」不是「没超」。** 半程读不到时 frame_exceeds 不作断言，
    调用侧必须能看出这一格没验成（piezo_half_x_m is None），而不是当它通过了。"""
    ctx = _Ctx(readback="echo", calib_error="TCP timeout")
    res = _run(ctx)
    assert res.success is True, "读不到半程不阻断扫描（否则一次通信抖动就停机）"
    assert res.data["piezo_half_x_m"] is None, "这一格没验成，要看得出来"
    assert res.data["frame_exceeds_piezo_range"] is None


# ── 变异抓到的两个洞─────────────────────────────────────────────
def test_data_carries_the_readback_not_the_request():
    """★ 桩原来把请求原样回声 ⇒ 请求值与读回值长得一样 ⇒ 断言分不开它们，
    把 `data` 改回返回请求值也照样绿。**替身太顺让断言空转。**

    这里让读回值在容差**之内**但确实不等于请求值，逼出区别。
    """
    # 用量程内的框，免得先撞上越界闸门（那是另一条用例的事）
    cx = 100e-9
    rb = [cx + 1e-13, 0.0, 1e-6, 1e-6, 0.0]
    ctx = _Ctx(readback=rb)
    res = _run(ctx, center_x_m=cx, center_y_m=0.0, width_m=1e-6, height_m=1e-6)
    assert res.success is True, "1e-13 的差在容差内，不该被判成夹紧"
    assert res.data["center_x_m"] == rb[0], "data 必须是读回值"
    assert res.data["center_x_m"] != cx, "不能是请求值的回声"
    assert res.data["requested_frame"][0] == cx


def test_a_small_clamp_is_caught_too():
    """小幅夹紧也必须被识别；用请求中心的百分之一偏差验证读回容差。"""
    m = frame_readback_mismatch(REQ, [594e-9, -550e-9, 1.8e-6, 1.8e-6, 0.0])
    assert m is not None and "center_x_m" in m, "1% 的夹紧也必须抓到"
