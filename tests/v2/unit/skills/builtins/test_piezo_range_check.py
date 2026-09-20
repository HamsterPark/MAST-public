"""压电范围对账自检：一致、不一致和未知三种结果均应可区分。

协议返回值采用嵌套的 (header, raw, values) 形状，数值在第三项中；
使用独立合成范围检验全程到半程的换算及与配置边界的比较。
"""
from __future__ import annotations

import pytest


class _Rec:
    """Nanonis 线格式:``('', b'…', [值...])``。"""

    def __init__(self, values=None, error=""):
        self.return_value = ("", b"", list(values)) if values else ("", b"", [])
        self.error = error
        self.method = ""
        self.args = ()


class _Ctx:
    def __init__(self, *, ranges=None, error=""):
        self.ranges = ranges
        self.error = error
        self.state = None
        self.asked: list = []

    def safe_call(self, method, *args, role="main"):
        self.asked.append(method)
        if method == "Piezo_RangeGet":
            return _Rec(error=self.error) if self.error else _Rec(values=self.ranges)
        return _Rec()


def _run(ctx, **p):
    from mast.skills.builtins.piezo_range_check import CheckPiezoRange

    return CheckPiezoRange().execute(ctx, dict(p))


def test_it_decodes_the_real_wire_shape():
    """数值在第三个元素里面 —— 只扫顶层的解码会 100% 判不了。"""
    ctx = _Ctx(ranges=[2.4e-6, 2.4e-6, 4e-7])
    res = _run(ctx)
    assert res.data["instrument_half_x_m"] == pytest.approx(1.2e-6)
    assert res.data["instrument_half_y_m"] == pytest.approx(1.2e-6)
    assert res.data["verdict"] != "unknown", (
        "读到了却报判不了 —— 解码又只扫了顶层?")


def test_the_accident_is_reported_as_a_mismatch():
    """合成仪器半程小于配置时，应报告 mismatch 和缩小的方向。"""
    ctx = _Ctx(ranges=[2.4e-6, 2.4e-6, 4e-7])
    res = _run(ctx)
    assert res.data["verdict"] == "mismatch"
    assert res.data["configured_exceeds_instrument"] is True
    assert res.data["relative_difference"] == pytest.approx(0.25, abs=1e-10)
    assert "配置比仪器大" in res.summary
    # 处置必须给得出:光说「不一致」没法行动
    assert "1200" in res.summary


def test_an_unreadable_scanner_is_undecidable_not_ok():
    """**读不到 ≠ 一致。** 折成 ok 会让这个自检变成永远为真的安慰话。"""
    ctx = _Ctx(error="Piezo module not available")
    res = _run(ctx)
    assert res.data["verdict"] == "unknown"
    assert "判不了" in res.data["undecidable"]
    assert res.data["instrument_half_x_m"] is None


def test_a_matching_range_says_ok():
    """配置正好等于仪器 ⇒ ok。"""
    ctx = _Ctx(ranges=[3.0e-6, 3.0e-6, 4e-7])   # 半程 1500 nm = 配置值
    res = _run(ctx)
    assert res.data["verdict"] == "ok"


def test_it_never_changes_anything():
    """只报数。一个自检不该顺手改全仪器的设置。"""
    ctx = _Ctx(ranges=[2.4e-6, 2.4e-6, 4e-7])
    _run(ctx)
    assert ctx.asked == ["Piezo_RangeGet"], (
        f"除了读范围还调了别的:{ctx.asked}")


def test_the_configured_side_goes_through_the_effective_limits():
    """配置那一侧要走 ``_get_effective_limits``,**不是**直接读类默认值。

    直接读 ``SafetyLimits().xy_max_m`` 会绕过管理员覆写,报出来的数就不是实际
    生效的那个 —— 一个对账工具报错数字,比不对账更坏。
    """
    import inspect

    from mast.skills.builtins import piezo_range_check

    src = inspect.getsource(piezo_range_check)
    assert "_get_effective_limits" in src
