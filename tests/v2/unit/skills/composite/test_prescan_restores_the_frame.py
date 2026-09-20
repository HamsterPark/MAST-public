"""预扫描结束后恢复借用的原框；读取失败时不虚构坐标或尺寸。"""
from __future__ import annotations

import sys

import pytest
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.types import NanonisCallRecord  # noqa: E402
from mast.skills.composite.prescan_check import PreScanCheck  # noqa: E402


class _Ctx:
    """只记下发过什么；`Scan_FrameGet` 按构造参数回答。"""

    def __init__(self, frame=None, get_error=""):
        self._frame = frame
        self._get_error = get_error
        self.calls: list[tuple[str, tuple]] = []

    def safe_call(self, method, *args, role="main", allow_on_abort=False):
        self.calls.append((method, args))
        if method == "Scan_FrameGet":
            return NanonisCallRecord(method=method, args=args,
                                     error=self._get_error,
                                     return_value=self._frame)
        return NanonisCallRecord(method=method, args=args)

    def sets(self):
        return [a for m, a in self.calls if m == "Scan_FrameSet"]


def test_the_frame_is_put_back_after_the_prescan():
    """借了就还 —— 还回去的必须是**读到的那一份**，一个数都不许变。"""
    original = (1.0e-7, -2.0e-7, 6.0e-8, 6.0e-8, 0.0)
    ctx = _Ctx(frame=original)
    PreScanCheck._restore_frame(ctx, original)
    sets = ctx.sets()
    assert sets, ("扫描框没有被还原 —— 之后每一张图都会继承预扫描的细长条"
                  "（用户看到的 50nm×2.5nm 就是它）。")
    assert sets[-1][:4] == original[:4], (
        f"还回去的框和读到的不是同一个: {sets[-1][:4]} vs {original[:4]}")


def test_an_unreadable_frame_is_not_guessed():
    """读不到原框 ⇒ **什么都不做**，不许放回一个编出来的框。

    放回一个猜的框会把下一次扫描送到一个没人要求的地方 —— 那比留着细长条更坏，
    因为细长条至少是看得见的。
    """
    ctx = _Ctx(frame=None)
    PreScanCheck._restore_frame(ctx, None)
    assert not ctx.sets(), "读不到原框却还是下发了一个框 —— 那个框是编的"


def test_a_reply_shaped_wrong_is_not_forced_into_a_frame():
    """帧回包解析失败时保留原有几何。"""
    for junk in ((), (1.0,), "not a frame", 42, ((1.0,), (2.0,))):
        ctx = _Ctx()
        PreScanCheck._restore_frame(ctx, junk)
        assert not ctx.sets(), f"回包 {junk!r} 看不懂，却硬凑出了一个框"


def test_tuple_wrapped_numbers_are_accepted():
    """单元素元组包装的数值回包也必须正确恢复。"""
    wrapped = ((1.0e-7,), (-2.0e-7,), (6.0e-8,), (6.0e-8,), (0.0,))
    ctx = _Ctx()
    PreScanCheck._restore_frame(ctx, wrapped)
    sets = ctx.sets()
    assert sets, "单元素元组回包未能正确恢复扫描框"
    assert abs(sets[-1][2] - 6.0e-8) < 1e-18


# 同时覆盖扁平数值、元组数值及三段信封；识别信封时检查 body 长度，避免误解单元素包装。

_FRAME = (-1.1055e-07, -1.7592e-07, 1.0e-07, 1.0e-07, 90.0)
_WRAPPED = tuple((v,) for v in _FRAME)


@pytest.mark.parametrize("shape,why", [
    (("", b"\x00" * 20, list(_FRAME)), "真机:(error, raw, body) 三段信封"),
    (("", b"", list(_WRAPPED)), "协议信封中的单元素元组"),
    (_FRAME, "扁平裸数字(仓里替身的老形状)"),
    (_WRAPPED, "扁平 + 1-元组"),
])
def test_every_reply_shape_that_can_actually_arrive_is_restored(shape, why):
    """所有支持的回包形状都必须还原同一个扫描框。"""
    ctx = _Ctx()
    PreScanCheck._restore_frame(ctx, shape)
    sets = ctx.sets()
    assert sets, f"{why}:没有下发 Scan_FrameSet —— 这一份形状不会被还原"
    got = sets[-1]
    for i, want in enumerate(_FRAME):
        assert abs(float(got[i]) - want) < 1e-18, (
            f"{why}:第 {i} 个数还错了({got[i]} vs {want})")


def test_the_envelope_test_looks_at_the_body_length_not_just_its_type():
    """判别信封要看 **body 有几个元素**,不能只看第三项是不是序列。

    `((cx,),(cy,),(w,),(h,),(ang,))` 的第三项 `(w,)` 也是序列 ——
    只看类型会把它当 body 剥出来,只剩一个数,于是又变成「不还原」。
    第一版正是这么写的,这条把它钉住。
    """
    ctx = _Ctx()
    PreScanCheck._restore_frame(ctx, _WRAPPED)
    assert ctx.sets(), "扁平 1-元组形状被误当成信封剥掉了"
    assert abs(ctx.sets()[-1][2] - 1.0e-07) < 1e-18


@pytest.mark.parametrize("junk", [
    ("", b"", [1.0, 2.0]),              # body 太短
    ("", b"", ["a", "b", "c", "d"]),    # body 不是数
    ("", b""),                          # 根本没有 body
])
def test_an_unusable_body_is_still_not_guessed(junk):
    """信封在、body 用不了 ⇒ **仍然什么都不做**。不猜。"""
    ctx = _Ctx()
    PreScanCheck._restore_frame(ctx, junk)
    assert not ctx.sets(), f"回包 {junk!r} 用不了,却硬凑出了一个框"
