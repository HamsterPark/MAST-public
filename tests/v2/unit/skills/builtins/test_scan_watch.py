"""WatchScanLines：扫描进行中检查新增数据、覆盖范围与可读性。

未取得新行时应明确报告；尚无有效数据不能被判作平坦表面。
通道和尺度缺失时拒绝推测。
"""
from __future__ import annotations

import numpy as np
import pytest

from mast.core.types import NanonisCallRecord
from mast.skills.builtins.scan_watch import WatchScanLines

PIX = 64


def _frame_reply(img: np.ndarray) -> tuple:
    """``Scan_FrameDataGrab`` 的**真机形态**：body 是异质列表
    ``[name_len, name(str), rows, cols, data_2D(ndarray), dir]``，
    而且第 5 个元素是**真正的 ndarray** —— 解析器找的就是它。
    第一版这里传了 ``img.tolist()``，解析器返回 None，于是整组用例都在
    「取不到帧」上失败：**假件不像真机，测的就不是真机会走的那条路**。"""
    rows, cols = img.shape
    return ("", b"", [1, "Z (m)", rows, cols, np.asarray(img, dtype=float), 1])


class _Ctx:
    """按需回四种 Nanonis 调用的最小上下文。"""

    def __init__(self, img, *, channels=(0, 30), pixels=PIX, width_m=5e-9,
                 names=("Current (A)",) + ("x",) * 29 + ("Z (m)",),
                 buffer_error="", frame_error=""):
        self.img = img
        self.channels = list(channels)
        self.pixels = pixels
        self.width_m = width_m
        self.names = list(names)
        self.buffer_error = buffer_error
        self.frame_error = frame_error
        self.calls: list[str] = []

    def safe_call(self, method, *args, **kwargs):
        self.calls.append(method)
        if method == "Scan_BufferGet":
            if self.buffer_error:
                return NanonisCallRecord(method=method, args=args,
                                         error=self.buffer_error)
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [len(self.channels), self.channels,
                                        self.pixels, self.pixels]))
        if method == "Signals_NamesGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [self.names]))
        if method == "Scan_FrameGet":
            if self.width_m is None:
                return NanonisCallRecord(method=method, args=args, error="no frame")
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [0.0, 0.0, self.width_m, self.width_m, 0.0]))
        if method == "Scan_FrameDataGrab":
            if self.frame_error:
                return NanonisCallRecord(method=method, args=args,
                                         error=self.frame_error)
            return NanonisCallRecord(method=method, args=args,
                                     return_value=_frame_reply(self.img))
        return NanonisCallRecord(method=method, args=args)

    def check_abort(self):
        return False


def _partial(n_done: int, seed: int = 0) -> np.ndarray:
    """合成部分帧：已扫行有数据，未扫行填零。"""
    rs = np.random.RandomState(seed)
    img = np.zeros((PIX, PIX))
    img[PIX - n_done:] = rs.randn(n_done, PIX) * 1e-11 - 1.5e-7
    return img


def _run(ctx, **params):
    return WatchScanLines().execute(ctx, params)


# ── 游标：0 条新行必须变成一条要有人接的信号 ────────────────────────
def test_no_new_lines_since_the_cursor_is_reported_as_not_advancing():
    img = _partial(20)
    ctx = _Ctx(img)
    first = _run(ctx)
    assert first.success
    cursor = first.data["last_line_index"]

    again = _run(ctx, since_line=cursor)          # 同一张图，没推进
    assert again.success
    assert again.data["n_lines_new"] == 0
    assert again.data["advancing"] is False
    text = " ".join(again.data["observations"])
    assert "一行新的都没有" in text, text
    assert "旧" in text, "没有说清「接着打分看到的都是旧的」——那正是会被忽略的那半句"


def test_new_lines_after_the_cursor_are_counted():
    ctx = _Ctx(_partial(20))
    first = _run(ctx)
    cursor = first.data["last_line_index"]
    ctx.img = _partial(30)                         # 又扫了 10 行
    again = _run(ctx, since_line=cursor)
    assert again.data["n_lines_new"] == 10
    assert again.data["advancing"] is True


# ── 「还没扫到」不是「表面是平的」 ────────────────────────────────
def test_an_empty_buffer_is_not_reported_as_a_flat_surface():
    res = _run(_Ctx(np.zeros((PIX, PIX))))
    assert res.success
    assert res.data["n_lines_done"] == 0
    assert res.data["advancing"] is False
    text = " ".join(res.data["observations"])
    assert "一行都还没有" in text
    assert "不等于" in text and "平" in text, (
        "空缓冲的措辞没有把「没扫到」和「是平的」分开：%s" % text)
    # 没有量可报时不许给出粗糙度 —— 一个 0 会被读成「测了，很平」
    assert "rms_roughness_m" not in res.data


# ── 读不到就拒绝，不猜 ────────────────────────────────────────────
def test_it_refuses_when_the_scan_buffer_cannot_be_read():
    res = _run(_Ctx(_partial(10), buffer_error="TCP timeout"))
    assert res.success is False
    assert "扫描缓冲" in res.error


def test_it_refuses_when_the_scale_cannot_be_computed():
    res = _run(_Ctx(_partial(10), width_m=None))
    assert res.success is False
    assert "nm/px" in res.error
    assert "猜" in res.error, "拒绝的理由没说清「宁可拒绝也不猜尺度」"


def test_it_names_the_available_channels_when_the_grab_fails():
    """通道号填错时的报错要指向**通道**，不是指向解析器。"""
    res = _run(_Ctx(_partial(10), frame_error="response layout mismatch at offset 16"))
    assert res.success is False
    assert "信号索引" in res.error and "缓冲位" in res.error
    assert "[0, 30]" in res.error


def test_a_z_channel_absent_from_the_buffer_is_refused_with_the_list():
    res = _run(_Ctx(_partial(10), channels=(0,), names=("Current (A)",)))
    assert res.success is False
    assert "找不到 Z" in res.error
    assert "[0]" in res.error


# ── 量：行间相关是提醒不是判定 ────────────────────────────────────
def test_uncorrelated_lines_raise_a_hint_that_says_it_is_a_hint():
    rs = np.random.RandomState(3)
    img = np.zeros((PIX, PIX))
    img[PIX - 30:] = rs.randn(30, PIX) * 1e-11        # 行与行之间毫不相关
    res = _run(_Ctx(img))
    assert res.success
    assert res.data["line_to_line_corr"] < 0.5
    text = " ".join(res.data["observations"])
    assert "行间相关" in text
    assert "提醒" in text and "判定" in text, (
        "行间相关的措辞滑成了判定口吻：%s" % text)


def test_correlated_lines_do_not_raise_the_hint():
    base = np.sin(np.linspace(0, 8 * np.pi, PIX)) * 1e-11
    img = np.zeros((PIX, PIX))
    img[PIX - 30:] = base + np.random.RandomState(4).randn(30, PIX) * 1e-13
    res = _run(_Ctx(img))
    assert res.data["line_to_line_corr"] > 0.8
    assert "行间相关" not in " ".join(res.data["observations"])


def test_progress_is_reported_against_the_real_line_count():
    res = _run(_Ctx(_partial(16)))
    assert res.data["n_lines_done"] == 16
    assert res.data["n_lines_total"] == PIX
    assert res.data["fraction_done"] == pytest.approx(16 / PIX)


def test_the_cursor_it_returns_can_be_fed_straight_back():
    """last_line_index 必须是**下一次能直接当 since_line 用**的那个数。"""
    ctx = _Ctx(_partial(20))
    a = _run(ctx)
    b = _run(ctx, since_line=a.data["last_line_index"])
    assert b.data["n_lines_new"] == 0        # 图没变 ⇒ 一条新的都不该有
    ctx.img = _partial(21)
    c = _run(ctx, since_line=a.data["last_line_index"])
    assert c.data["n_lines_new"] == 1
