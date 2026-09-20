"""Osci2T Z-noise burst — the nanonis_spm command-name fixes, and the burst
loop's refusal to misbehave on a shared oscilloscope.

The command names get their own assertions because the upstream bug is silent
in the worst possible way: ``Osci2T_ChSet`` sends ``Osci1T.ChSet``, so a caller
that believes it is configuring the dual-channel scope actually re-points the
oscilloscope the tunnelling-current monitor is pumping — and that monitor then
stores Bias as if it were Current.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import threading

import numpy as np
import pytest

from mast.envhistory.zburst import run_z_burst

FS = 20000.0
DT = 1.0 / FS
NAMES = ["Current (A)", "Bias (V)", "Z (m)", "Amplitude"]
Z_IDX, I_IDX = 2, 0


class _Rec:
    def __init__(self, parsed=None, error=""):
        self.error = error
        self.method = "x"
        self.return_value = ["", b"", parsed or []]


class _Pool:
    """A scriptable Osci2T. Records every verb so the test can assert the
    protocol, not just the outcome."""

    def __init__(self, *, healthy=True, channels=(I_IDX, Z_IDX),
                 n_traces=6, err_on=None, drift_channels_after=None):
        self._healthy = healthy
        self.channels = list(channels)
        self.n_traces = n_traces
        self.err_on = err_on or {}
        #: 逐元素裹成 1-元组 —— `decodeArray` 在 nanonis_patch 之前的形态。
        #: 替身默认给裸值(= 打完补丁后的生产形态),于是**元组那一侧从来没被测过**。
        self.tuple_wrapped = False
        self._drift_after = drift_channels_after
        self.calls: list[tuple] = []
        self._t0 = 100.0
        self._served = 0

    def comms_healthy(self):
        return self._healthy

    def safe_call(self, verb, *args, **kw):
        self.calls.append((verb, args, kw))
        if verb in self.err_on:
            return _Rec(error=self.err_on[verb])
        if verb == "Signals_NamesGet":
            return _Rec([[*NAMES]])
        if verb in ("Osci2T_ChGet", "Osci2T_ChsGet"):
            return _Rec([self.channels[0], self.channels[1]])
        if verb == "Osci2T_ChSet":
            self.channels = [int(args[0]), int(args[1])]
            return _Rec()
        if verb == "Osci2T_TimebaseGet":
            return _Rec([0, 3, [DT, DT * 2, DT * 4]])
        if verb == "Osci2T_DataGet":
            self._served += 1
            if self._served > self.n_traces:
                return _Rec(error="no more")
            if (self._drift_after is not None
                    and self._served > self._drift_after):
                self.channels = [1, 1]      # someone re-pointed the scope
            n = 4096
            rng = np.random.default_rng(self._served)
            a = rng.normal(0, 2e-11, n).tolist()
            b = rng.normal(0, 3e-12, n).tolist()
            self._t0 += n * DT
            if self.tuple_wrapped:
                a = [(v,) for v in a]
                b = [(v,) for v in b]
            return _Rec([self._t0, DT, n, a, n, b])
        return _Rec()

    def verbs(self):
        return [c[0] for c in self.calls]


def _run(pool, **kw):
    kw.setdefault("burst_s", 0.05)
    kw.setdefault("bins", 60)
    return run_z_burst(lambda: pool, stop=threading.Event(), **kw)


# ── the happy path ──────────────────────────────────────────────────────────

def test_burst_produces_a_z_spectrum():
    pool = _Pool()
    snap = _run(pool)
    assert snap is not None
    assert snap.channel == "z"
    assert snap.unit == "m^2/Hz"
    assert snap.n_segments >= 1
    assert len(snap.freqs) >= 2
    assert abs(snap.fs_hz - FS) < 1.0


def test_every_call_uses_the_data_role_without_feeding_the_breaker():
    pool = _Pool()
    _run(pool)
    assert pool.calls
    for verb, _args, kw in pool.calls:
        assert kw.get("role") == "data", verb
        assert kw.get("count_health") is False, verb


def test_the_z_channel_is_read_not_the_current_channel():
    """Data comes back as [t0, dt, sizeA, A, sizeB, B]; Z is channel B."""
    pool = _Pool()
    snap = _run(pool)
    assert snap is not None
    # A Z trace has ~3 pV RMS in this fixture and the current trace ~20 pA;
    # picking the wrong column would put the PSD seven decades higher.
    assert max(snap.psd) < 1e-20


def test_channels_are_left_alone_when_z_is_already_displayed():
    pool = _Pool(channels=(I_IDX, Z_IDX))
    _run(pool)
    assert "Osci2T_ChSet" not in pool.verbs()


def test_channels_are_set_and_restored_when_z_is_absent():
    pool = _Pool(channels=(I_IDX, 3))
    _run(pool)
    sets = [c for c in pool.calls if c[0] == "Osci2T_ChSet"]
    assert len(sets) == 2                       # configure, then put it back
    assert sets[0][1] == (I_IDX, Z_IDX)
    assert sets[-1][1] == (I_IDX, 3)            # operator's view restored
    assert pool.channels == [I_IDX, 3]


def test_trigger_is_forced_to_immediate():
    """Left on Level with nothing crossing it, the scope stops re-arming and
    every DataGet returns the same stale buffer."""
    pool = _Pool()
    _run(pool)
    trig = [c for c in pool.calls if c[0] == "Osci2T_TrigSet"]
    assert trig and trig[0][1][0] == 0          # mode 0 = Immediate


def test_the_fastest_timebase_is_selected():
    pool = _Pool()
    _run(pool)
    sets = [c for c in pool.calls if c[0] == "Osci2T_TimebaseSet"]
    assert sets and sets[0][1] == (0,)          # index of the smallest dt


# ── refusals ────────────────────────────────────────────────────────────────

def test_no_pool_is_a_skip_not_a_crash():
    assert run_z_burst(lambda: None, stop=threading.Event(), burst_s=0.01) is None


def test_module_not_loaded_is_a_skip():
    """The bundled simulator does not load Osci2T; that is normal, not an error."""
    pool = _Pool(err_on={"Osci2T_Run": "NeedModule: Osci2T"})
    assert _run(pool) is None


def test_open_breaker_mid_burst_is_a_skip():
    pool = _Pool(err_on={"Osci2T_DataGet": "comms_circuit_open"})
    assert _run(pool) is None


def test_role_busy_throughout_gives_up_instead_of_spinning():
    pool = _Pool(err_on={"Osci2T_DataGet": "RoleBusy: data held by ScanFrame"})
    assert _run(pool) is None


def test_missing_z_signal_is_a_skip():
    pool = _Pool()
    pool.safe_call_names_override = True

    class _NoZ(_Pool):
        def safe_call(self, verb, *args, **kw):
            if verb == "Signals_NamesGet":
                self.calls.append((verb, args, kw))
                return _Rec([["Current (A)", "Bias (V)"]])
            return super().safe_call(verb, *args, **kw)

    assert _run(_NoZ()) is None


def test_a_scope_re_pointed_mid_burst_discards_the_whole_round():
    """Osci2T is shared with the optional-scope skills, which take no
    instrument token. Half a trace of somebody else's signal is not salvageable."""
    pool = _Pool(drift_channels_after=2)
    assert _run(pool) is None


def test_a_set_stop_event_ends_the_burst_promptly():
    pool = _Pool(n_traces=10_000)
    stop = threading.Event()
    stop.set()
    assert run_z_burst(lambda: pool, stop=stop, burst_s=30.0, bins=60) is None


def test_burst_never_raises_on_a_pool_that_explodes():
    class _Boom:
        def comms_healthy(self):
            return True

        def safe_call(self, *a, **kw):
            raise RuntimeError("socket gone")

    assert _run(_Boom()) is None


# ── the nanonis_spm patch ───────────────────────────────────────────────────

class _Client:
    """Captures the command name and specs quickSend is called with."""

    def __init__(self, fail_names=()):
        self.sent: list[tuple] = []
        self._fail = set(fail_names)

    def quickSend(self, command, body, body_types, response_types):
        self.sent.append((command, tuple(body), tuple(body_types),
                          tuple(response_types)))
        if command in self._fail:
            return ["unknown command", b"", []]
        return ["", b"", [0, 0]]


@pytest.fixture()
def patched():
    from mast.core import nanonis_patch
    nanonis_patch.apply()
    return nanonis_patch


def test_osci2t_timebase_get_targets_the_right_scope_and_verb(patched):
    """Osci2T 的时基字段规格应由实际协议类型探测，不能照搬 Osci1T 的类型。
    测试保持 getter 动词正确，并核验首选候选类型与探测路径。"""
    c = _Client()
    patched.Nanonis.Osci2T_TimebaseGet(c)
    assert c.sent[0][0] == "Osci2T.TimebaseGet"      # not Osci1T, not …Set
    assert c.sent[0][1] == ()                        # a GET takes no arguments
    # 库自己声明的那条排第一（证据最强的先试）。
    assert c.sent[0][3] == ("H", "i", "*f")
    # 而且它是候选之一，不是唯一答案：这个假客户端回的 [0, 0] 不自洽
    # （count=0），所以探测必须继续往下试，而不是就此定案。
    assert [s[3] for s in c.sent] == [("H", "i", "*f"), ("i", "i", "*f")]
    assert getattr(c, "_mast_osci2t_tb_spec", None) is None, "不自洽却被缓存了"


def test_osci2t_timebase_set_targets_the_right_scope(patched):
    c = _Client()
    patched.Nanonis.Osci2T_TimebaseSet(c, 3)
    assert c.sent[0][0] == "Osci2T.TimebaseSet"
    assert c.sent[0][1] == (3,)


def test_osci2t_oversampl_get_is_a_getter(patched):
    c = _Client()
    patched.Nanonis.Osci2T_OversamplGet(c)
    assert c.sent[0][0] == "Osci2T.OversamplGet"     # upstream sent …Set
    assert c.sent[0][1] == ()


def test_channel_verb_is_probed_with_the_getter_first(patched):
    """The plural/singular question is resolved by a READ, so a wrong guess
    costs one rejected query rather than a misconfigured oscilloscope."""
    c = _Client()
    patched.Nanonis.Osci2T_ChsSet(c, 1, 2)
    assert c.sent[0][0].endswith("Get")              # probe precedes any write
    assert c.sent[0][1] == ()
    assert c.sent[-1][0] in ("Osci2T.ChsSet", "Osci2T.ChSet")
    assert c.sent[-1][1] == (1, 2)


def test_channel_verb_falls_back_when_the_plural_name_is_rejected(patched):
    c = _Client(fail_names={"Osci2T.ChsGet"})
    patched.Nanonis.Osci2T_ChsGet(c)
    assert [s[0] for s in c.sent][:2] == ["Osci2T.ChsGet", "Osci2T.ChGet"]


def test_the_resolved_channel_verb_is_cached(patched):
    c = _Client()
    patched.Nanonis.Osci2T_ChsGet(c)
    n = len(c.sent)
    patched.Nanonis.Osci2T_ChsGet(c)
    assert len(c.sent) == n + 1                      # one probe, not two


def test_singular_spellings_are_aliases_of_the_fixed_pair(patched):
    """skills/builtins/optional_scopes.py calls Osci2T_ChSet; leaving that
    pointed at Osci1T would keep it reconfiguring the wrong oscilloscope."""
    c = _Client()
    patched.Nanonis.Osci2T_ChSet(c, 4, 5)
    assert all(not s[0].startswith("Osci1T") for s in c.sent)
    assert c.sent[-1][1] == (4, 5)


def test_the_patch_does_not_disturb_the_osci1t_path(patched):
    """The current monitor pumps Osci1T; its verbs must be untouched."""
    c = _Client()
    patched.Nanonis.Osci1T_TimebaseGet(c)
    assert c.sent[0][0] == "Osci1T.TimebaseGet"
    assert c.sent[0][3] == ("i", "i", "*f")


# ── 元素被裹成 1-元组时不许静默变形（v6.1.3, KNOWN_ISSUES）──────────────
#
# `Osci2T_DataGet` 的 `*d` 数组走 decodeArray，而那个函数**曾经**把每个元素裹成
# 1-元组（`nanonis_patch` 现在在解码层拆掉了）。这一行代码不该依赖那个补丁还在。
#
# 关键在于它**不会报错**：`np.asarray([(1.0,), (2.0,)])` 给出 **(N, 1)**，
# 而外面那个 `except (TypeError, ValueError, IndexError)` 接不住形状改变，
# `y.size` 也仍然读作 N，所以空检查照过。一次静默的降级。
#
# 这是本轮 33 个「会产生元组的 Nanonis 方法」分诊里，**唯一**一处没有元素级守卫的
# 数组读取 —— 其余 22 个文件全部走 isinstance / _scalar / parse_* 。


def test_tuple_wrapped_samples_still_yield_a_1d_trace():
    pool = _Pool()
    pool.tuple_wrapped = True
    snap = _run(pool)
    assert snap is not None, "元组形态把整条 burst 弄没了"
    assert snap.channel == "z"
    assert len(snap.freqs) >= 2
    assert abs(snap.fs_hz - FS) < 1.0


def test_both_element_shapes_give_the_same_spectrum():
    """裸标量数组与单元素元组数组必须解析为相同结果。"""
    bare = _run(_Pool())
    wrapped_pool = _Pool()
    wrapped_pool.tuple_wrapped = True
    wrapped = _run(wrapped_pool)
    assert bare is not None and wrapped is not None
    assert len(bare.freqs) == len(wrapped.freqs)
    assert abs(bare.fs_hz - wrapped.fs_hz) < 1e-9
    assert bare.unit == wrapped.unit
    assert len(bare.psd) == len(wrapped.psd)
    # ⚠️ 刻意**不**比 `n_segments`:采几段由这条 burst 在墙钟窗口里收到几条 trace
    # 决定,两次运行本来就可能不同(实测 6 vs 5)。拿它当等价判据,量的是机器忙不忙
    # 而不是元素形状 —— 今晚刚在读回采集那条上踩过同一个坑。
