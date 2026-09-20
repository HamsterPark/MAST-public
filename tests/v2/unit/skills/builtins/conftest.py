"""共享确定性时钟供读回采集测试使用。墙钟延迟会改变窗口点数与时间戳，因此这些测试验证采集逻辑，不验证仪器吞吐量。insufficient_data 与 none 仍须严格区分；需要时由测试文件显式请求 fixture。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if _MASTV2_ROOT not in sys.path:
    sys.path.insert(0, _MASTV2_ROOT)


class FakeClock:
    """``time`` 的确定性替身,只提供采集循环用到的两个函数。

    ``perf_counter()`` **每被调用一次就前进一个 tick** —— 这保证循环一定终止
    (时间严格单调增),而且采到的点数只由参数决定,与机器负载无关。

    ``advance()`` 给需要模拟「这一次调用阻塞了很久」的测试用:那种测试原本靠真的
    ``time.sleep()`` 烧掉几十毫秒,再去断言一个由 ``perf_counter`` 算出来的量 ——
    换成显式推进假时钟之后,它既确定又直接表达了意图。
    """

    def __init__(self, tick: float = 1e-4) -> None:
        self.t = 0.0
        self.tick = float(tick)
        self.sleeps = 0

    def perf_counter(self) -> float:
        self.t += self.tick
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.t += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        """显式推进,模拟一次阻塞的硬件调用。"""
        self.t += float(seconds)


@pytest.fixture
def readback_clock(monkeypatch):
    """把 ``_readback_stream`` 模块里的 ``time`` 换成 :class:`FakeClock`。

    换的是**模块属性**而不是全局 ``time`` 模块 —— 后者会波及 pytest 自己和一切
    第三方库。请求这个夹具即生效,返回值可以用来 ``advance()``。
    """
    from mast.skills.builtins import _readback_stream

    clock = FakeClock()
    monkeypatch.setattr(_readback_stream, "time", clock)
    return clock
