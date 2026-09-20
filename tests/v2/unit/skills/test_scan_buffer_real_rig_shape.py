"""每一个 ``Scan_BufferGet`` 通道列表的消费方,都要吃得下真机形态。

**背景(2026-08-04, v6.1.1 真机阻断项)。** 真机上 nanonis_spm 把 ``Scan_BufferGet``
的 ``*i`` 通道数组解析成一串 **1-元组**::

    channel_indexes = [(0,), (30,)]     # 不是 [0, 30]

仓里当时有**五处**在读这个列表:两处解对了(scan_monitor / full_scan),三处没有
—— `SetScanBuffer` 直接 ``TypeError`` 崩掉(整条 ScanAt 扫图路径断),`GetScanBuffer`
把元组原样报给模型看,`ScanIntelSelfCheck` 把元组送进 ``Scan_BufferSet`` 这条**写
命令**。三处都是抄写漂移的产物。

判断现在只有**一份**实现(``mast.io.nanonis_files.channel_ids_from_buffer``),
这个文件是它的**接线测试**:逐个消费方喂真机那一包,断言各自的下游拿到的是裸
int。合并之后光测那一份原语是不够的 —— 那只证明原语对,不证明有人用了它。

裸 int 那一路同样要过:模拟器和桩就是那个形态,两种形态都真实存在。
"""

from __future__ import annotations

import pytest

from mast.core.types import NanonisCallRecord

# 协议适配器与模拟器可能分别返回单元素元组和裸整数，两种形态均应覆盖。
@pytest.fixture(params=["real_rig_one_tuple", "stub_bare_int"])
def wrap_ch(request):
    return (lambda c: (c,)) if request.param == "real_rig_one_tuple" else (lambda c: c)


def _reply(channels, pixels=512, lines=512, wrap=lambda c: c):
    """``Scan_BufferGet`` 的完整 return_value。"""
    return ("", b"", [len(channels), [wrap(c) for c in channels], pixels, lines])


class _Ctx:
    """只回 Scan_BufferGet 的最小上下文,记录所有调用。"""

    def __init__(self, reply):
        self._reply = reply
        self.calls: list[tuple[str, tuple]] = []

    def safe_call(self, method, *args, **kwargs):
        self.calls.append((method, args))
        if method == "Scan_BufferGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=self._reply)
        if method == "Scan_FrameDataGrab":
            # 非空、非全 NaN,免得崩溃检查把它当撞针
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [1, "Z (m)", 2, 2, [0.0, 1.0, 2.0, 3.0], 1]))
        return NanonisCallRecord(method=method, args=args)

    def check_abort(self):
        return False


# ── GetScanBuffer(scan_extra)——「不崩,但报错数据」那一类 ──────────────────

def test_get_scan_buffer_reports_plain_ints(wrap_ch):
    """报给模型看的通道号必须是 int。原先 ``list(vals[1])`` 不崩,只是让模型看到
    「通道 (0,)」—— 静默错比 TypeError 更难发现。"""
    from mast.skills.builtins.scan_extra import GetScanBuffer

    res = GetScanBuffer().execute(_Ctx(_reply([0, 30], 512, 256, wrap=wrap_ch)), {})
    assert res.success
    assert res.data["channel_indexes"] == [0, 30]
    assert all(type(c) is int for c in res.data["channel_indexes"])
    assert res.data["pixels"] == 512 and res.data["lines"] == 256


def test_get_scan_buffer_keeps_the_instruments_declared_count():
    """num_channels 报仪器声明的值,不是 len(解析结果)。"""
    from mast.skills.builtins.scan_extra import GetScanBuffer

    res = GetScanBuffer().execute(_Ctx(("", b"", [5, [(0,), (30,)], 512, 512])), {})
    assert res.data["num_channels"] == 5
    assert res.data["channel_indexes"] == [0, 30]


# ── FullScan ────────────────────────────────────────────────────────────────

def test_full_scan_resolves_crash_channels(wrap_ch):
    """崩溃检查探的必须是**真实采集的**通道。解析失败会退回硬编码的 (0, 14),
    而本机的 Z 是 30 —— 那正是 2026-06-29 「每次扫描都 skipped」的老毛病。"""
    from mast.skills.composite.full_scan import FullScan

    skill = FullScan()
    skill._all_calls = []
    assert skill._resolve_crash_channels(
        _Ctx(_reply([0, 30], wrap=wrap_ch))) == [(0, "ch0"), (30, "ch30")]


def test_full_scan_crash_channels_fall_back_when_unreadable():
    """读不出来 → 退回静态探针列表(而不是空表:空表 = 一个通道都不查 =
    「没撞针」的假报告)。"""
    from mast.skills.composite.full_scan import FullScan

    skill = FullScan()
    skill._all_calls = []
    assert skill._resolve_crash_channels(_Ctx("garbage")) == list(
        FullScan._CRASH_CHECK_CHANNELS)


def test_full_scan_reads_line_count(wrap_ch):
    from mast.skills.composite.full_scan import FullScan

    assert FullScan._read_scan_lines(_Ctx(_reply([0, 30], 256, 1024, wrap=wrap_ch))) == 1024


# ── ScanVisionMonitor —— 唯一一处**写**回硬件之外的真机既有正确实现 ──────────

def test_scan_monitor_resolves_topography_channel(wrap_ch):
    """M12 要的是形貌图。通道 id 解析错 → 抓不到 Z → 视觉脉冲整条哑掉。"""
    from mast.vision.scan_monitor import ScanVisionMonitor

    mon = ScanVisionMonitor(pool=None, scan_id="t")
    names = {0: "Current (A)", 30: "Z (m)"}

    def fake_safe_call(method, *args, role="monitor"):
        if method == "Scan_BufferGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=_reply([0, 30], wrap=wrap_ch))
        if method == "Scan_FrameDataGrab":
            nm = names.get(args[0], "?")
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [len(nm), nm, 2, 2, [0.0, 1.0], 1]))
        return NanonisCallRecord(method=method, args=args)

    mon._safe_call = fake_safe_call
    assert mon._resolve_channel() == 30      # Z (m),不是第一个通道


def test_scan_monitor_falls_back_to_first_channel_without_topography(wrap_ch):
    from mast.vision.scan_monitor import ScanVisionMonitor

    mon = ScanVisionMonitor(pool=None, scan_id="t")

    def fake_safe_call(method, *args, role="monitor"):
        if method == "Scan_BufferGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=_reply([5, 9], wrap=wrap_ch))
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", [1, "Bias (V)", 2, 2, [0.0], 1]))

    mon._safe_call = fake_safe_call
    assert mon._resolve_channel() == 5       # 第一个采集通道


# ── ScanIntelSelfCheck —— 元组会被送进一条**写**命令 ────────────────────────

def test_selfcheck_probe_writes_back_plain_ints(wrap_ch):
    """``_probe_buffer_semantics`` 把读到的通道**原样写回**。元组送进
    Scan_BufferSet 是这一族 bug 里唯一会碰硬件的那个。"""
    from mast.skills.builtins.scan_intel_selfcheck import ScanIntelSelfCheck

    ctx = _Ctx(_reply([0, 30], 512, 512, wrap=wrap_ch))
    out = ScanIntelSelfCheck()._probe_buffer_semantics(ctx, [])
    assert out["ok"], out
    sets = [args for m, args in ctx.calls if m == "Scan_BufferSet"]
    assert sets, "探测没有发出 Scan_BufferSet"
    for args in sets:
        assert args[0] == [0, 30]
        assert all(type(c) is int for c in args[0])


def test_selfcheck_probe_refuses_when_channels_unreadable():
    """读不到通道就**不发写入** —— 拿猜的通道列表去写会清掉用户的采集配置。"""
    from mast.skills.builtins.scan_intel_selfcheck import ScanIntelSelfCheck

    ctx = _Ctx(("", b"", [0, [], 512, 512]))
    out = ScanIntelSelfCheck()._probe_buffer_semantics(ctx, [])
    assert out["ok"] is False
    assert "未发出任何写入" in out["error"]
    assert not any(m == "Scan_BufferSet" for m, _ in ctx.calls)
