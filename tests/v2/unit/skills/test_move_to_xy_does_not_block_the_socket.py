"""移动指令非阻塞下发，再通过位置轮询核验到位。

若硬件等待到达后才回包，移动耗时可能超过连接层读超时。测试验证 wait=0、
位置与错误状态的真实传递、以及基于距离和合成速度输入派生等待预算。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.skills.builtins.navigation import MoveToXY  # noqa: E402

_TARGET = (1.0e-7, -2.0e-7)


class _Ctx:
    """记下每一次调用;位置按剧本逐次回。"""

    def __init__(self, positions, set_error: str = "",
                 speed: float | None = 8.0e-9) -> None:
        self.calls: list = []
        self._positions = list(positions)
        self._set_error = set_error
        self._speed = speed
        self._reads = 0

    def safe_call(self, method, *args, **kw):
        self.calls.append((method, args))
        if method == "FolMe_XYPosSet":
            return SimpleNamespace(error=self._set_error, return_value=("", b"", []))
        if method == "FolMe_SpeedGet":
            # 真机形状:(speed_m_s, custom)。speed=None ⇒ 模拟读不到。
            if self._speed is None:
                return SimpleNamespace(error="TimeoutError: timed out",
                                       return_value=None)
            return SimpleNamespace(error="",
                                   return_value=("", b"", [self._speed, 0.0]))
        if method == "FolMe_XYPosGet":
            i = min(self._reads, len(self._positions) - 1)
            self._reads += 1
            pos = self._positions[i]
            if pos is None:                      # 这一拍读不到
                return SimpleNamespace(error="TimeoutError: timed out",
                                       return_value=None)
            return SimpleNamespace(error="", return_value=("", b"", list(pos)))
        return SimpleNamespace(error="", return_value=("", b"", []))

    def sets(self):
        return [c for c in self.calls if c[0] == "FolMe_XYPosSet"]


def test_the_move_command_never_asks_nanonis_to_block():
    """``wait`` 参数必须传 0 —— 传 1 就是让 Nanonis 扣着我们的 socket。"""
    ctx = _Ctx([_TARGET])
    MoveToXY().execute(ctx, {"x_m": _TARGET[0], "y_m": _TARGET[1]})
    sets = ctx.sets()
    assert len(sets) == 1, f"下发次数不对:{sets}"
    method, args = sets[0]
    assert args[2] == 0, (
        f"FolMe_XYPosSet 的 wait 参数是 {args[2]},不是 0 —— "
        "那会让 Nanonis 端阻塞到针尖走到为止,而我们的 socket 只等 5 s。"
        "移动耗时必须由位置轮询单独管理。")


def test_it_polls_until_it_actually_arrives():
    """到没到自己看 —— 而且**真的**看,不是发完就宣布成功。"""
    away = (_TARGET[0] - 50e-9, _TARGET[1])
    ctx = _Ctx([away, away, _TARGET])
    r = MoveToXY().execute(ctx, {"x_m": _TARGET[0], "y_m": _TARGET[1]})
    assert r.success is True
    assert r.data["arrived"] is True
    gets = [c for c in ctx.calls if c[0] == "FolMe_XYPosGet"]
    assert len(gets) >= 3, f"没有轮询到位就返回了(只读了 {len(gets)} 次)"


def test_arriving_reports_the_measured_position_not_the_request():
    """报回来的要是**读到的**位置,不是我们请求的那个数。"""
    landed = (_TARGET[0] + 0.2e-9, _TARGET[1] - 0.1e-9)   # 容差内
    ctx = _Ctx([landed])
    r = MoveToXY().execute(ctx, {"x_m": _TARGET[0], "y_m": _TARGET[1]})
    assert r.success is True
    assert r.data["x_m"] == pytest.approx(landed[0])
    assert r.data["requested_x_m"] == pytest.approx(_TARGET[0])


def test_a_failed_set_returns_immediately():
    """下发就失败 ⇒ 不进轮询循环。"""
    ctx = _Ctx([_TARGET], set_error="ConnectionResetError: 10054")
    r = MoveToXY().execute(ctx, {"x_m": _TARGET[0], "y_m": _TARGET[1]})
    assert r.success is False
    assert "10054" in r.error
    assert not [c for c in ctx.calls if c[0] == "FolMe_XYPosGet"], (
        "下发都失败了还去轮询位置")


def test_wait_false_does_not_poll():
    """``wait=false`` ⇒ 发完就走,并且**说清楚**没等到位。"""
    ctx = _Ctx([_TARGET])
    r = MoveToXY().execute(ctx, {"x_m": _TARGET[0], "y_m": _TARGET[1],
                                 "wait": False})
    assert r.success is True
    assert r.data["arrived"] is None, "没等就不许声称到了 —— 也不许声称没到"
    assert not [c for c in ctx.calls if c[0] == "FolMe_XYPosGet"]


def test_a_position_read_that_fails_is_not_taken_as_not_arrived():
    """位置读不到 = **不知道到没到**,不是「没到」。

    读不到就当没到,会让一次成功的移动被判成失败;而这一天已经有四个
    「读不到被折叠成一个具体值」的缺陷了。
    """
    ctx = _Ctx([None, None, _TARGET])
    r = MoveToXY().execute(ctx, {"x_m": _TARGET[0], "y_m": _TARGET[1]})
    assert r.success is True, "中间几拍读不到,不该让整次移动失败"
    assert r.data["arrived"] is True


def test_a_timeout_says_the_command_did_go_out():
    """超时的失败原文必须说清「指令发出去了」——**不是「没动」**。

    用户读到「移动失败」会以为针还在原地;而真相是它可能正在路上,
    那两件事的下一步完全不同。
    """
    import mast.skills.builtins.navigation as nav

    ctx = _Ctx([(0.0, 0.0)])          # 永远到不了
    real_mono = nav.__dict__.get("time")
    # 直接把 deadline 逼到过期:轮询一拍就超时
    import time as _t
    t0 = _t.monotonic()
    orig = _t.monotonic
    seq = iter([t0, t0 + 999.0, t0 + 999.0, t0 + 999.0])
    _t.monotonic = lambda: next(seq, t0 + 999.0)          # type: ignore[assignment]
    try:
        r = MoveToXY().execute(ctx, {"x_m": _TARGET[0], "y_m": _TARGET[1]})
    finally:
        _t.monotonic = orig                                # type: ignore[assignment]
        if real_mono is not None:
            nav.__dict__["time"] = real_mono
    assert r.success is False
    assert "已下发" in r.error and "不是「没动」" in r.error
    assert r.data["arrived"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── 等待预算必须派生,不能是常数 ────────────────────────────────────────

def test_the_budget_is_derived_from_distance_and_speed():
    """等待预算必须由移动距离和设备回读速度派生。
    以合成速度输入核对预算数值，不使用固定经验超时替代距离/速度计算。
    """
    far = (0.0, 0.0)
    ctx = _Ctx([far], speed=8.0e-9)          # 合成速度 8 nm/s
    r = MoveToXY().execute(ctx, {"x_m": 1.0e-7, "y_m": 0.0})   # 走 100 nm
    assert r.success is False                 # 位置永远不动 ⇒ 超时
    # 100 nm ÷ 8 nm/s = 12.5 s,×1.5 + 10 = 28.75 s
    assert r.data["budget_derived"] is True
    assert r.data["budget_s"] == pytest.approx(28.75, rel=0.01)
    assert r.data["folme_speed_m_s"] == pytest.approx(8.0e-9)
    assert "派生" in r.error


def test_an_unreadable_speed_says_the_budget_is_a_fallback():
    """读不到速度 ⇒ 用兜底值,并且**说清它是兜底不是算出来的**。

    「不知道」不许伪装成「算过了」—— 下一个人会照着那个数去推断仪器的速度。
    """
    ctx = _Ctx([(0.0, 0.0)], speed=None)
    import mast.skills.builtins.navigation as nav

    orig = nav._FALLBACK_MOVE_TIMEOUT_S
    nav._FALLBACK_MOVE_TIMEOUT_S = 1.0        # 测试里不真等 300 s
    try:
        r = MoveToXY().execute(ctx, {"x_m": 1.0e-7, "y_m": 0.0})
    finally:
        nav._FALLBACK_MOVE_TIMEOUT_S = orig
    assert r.success is False
    assert r.data["budget_derived"] is False
    assert r.data["folme_speed_m_s"] is None
    assert "兜底值不是算出来的" in r.error


def test_a_long_move_gets_a_long_budget():
    """较长的合成移动必须得到超过短固定等待值的预算。"""
    ctx = _Ctx([(0.0, 0.0)], speed=8.0e-9)
    import mast.skills.builtins.navigation as nav

    orig = nav._FALLBACK_MOVE_TIMEOUT_S
    nav._FALLBACK_MOVE_TIMEOUT_S = 1.0
    try:
        import time as _t

        t0 = _t.monotonic()
        seq = iter([t0, t0 + 1e6])            # 立刻超时,只看预算算得对不对
        orig_mono = _t.monotonic
        _t.monotonic = lambda: next(seq, t0 + 1e6)   # type: ignore[assignment]
        try:
            r = MoveToXY().execute(ctx, {"x_m": 1.8e-6, "y_m": 0.0})
        finally:
            _t.monotonic = orig_mono           # type: ignore[assignment]
    finally:
        nav._FALLBACK_MOVE_TIMEOUT_S = orig
    assert r.data["budget_s"] == pytest.approx(1.8e-6 / 8.0e-9 * 1.5 + 10.0, rel=0.02)
    assert r.data["budget_s"] > 300.0, "1.8 µm 的预算不该短于 300 s"
