"""AcquirePLLFreqSweep 把测到的 f₀/Q 写回 instrument_profile。

在此之前这两个数只活在那一次的 SkillResult 里 —— 想知道当前音叉的共振就得重扫
一次,而它们是判断「这支传感器还好不好」的基本量。补上写回后与
TiltCalibrate → set_tilt_calibration 同一形状。

关键:**记账失败绝不能弄坏一次成功的扫描**,而扫描失败时回的 0 / NaN 也绝不能
被存成真值。
"""

from __future__ import annotations

import pytest

from mast.core import instrument_profile as iprof
from mast.skills.builtins.pll import AcquirePLLFreqSweep


class _Rec:
    def __init__(self, return_value=None, error=""):
        self.return_value = return_value
        self.error = error


class _Ctx:
    """最小执行上下文:按调用名回放预置结果。"""

    def __init__(self, sweep_return):
        self._sweep_return = sweep_return
        self.calls: list[str] = []

    def safe_call(self, name, *args):
        self.calls.append(name)
        if name == "PLLFreqSwp_Start":
            return _Rec(self._sweep_return)
        return _Rec(None)


#: AcquirePLLFreqSweep 的必填参数(与写回无关,但不给就到不了写回那一步)。
_PARAMS = {"num_points": 256, "period_s": 0.01}


def _sweep_payload(f0, q):
    """PLLFreqSwp_Start 的返回形状:parsed[2] 里第 6/7 位是 f₀ 和 Q。"""
    return [None, None, [0, 0, 0, 0, 0, 0, f0, q]]


@pytest.fixture(autouse=True)
def _clean():
    iprof.set_persist_sink(None)
    iprof.set_profile({})
    yield
    iprof.set_persist_sink(None)
    iprof.set_profile({})


def test_successful_sweep_persists_f0_and_q() -> None:
    ctx = _Ctx(_sweep_payload(32701.5, 21000.0))
    res = AcquirePLLFreqSweep().execute(ctx, _PARAMS)

    assert res.success is True
    assert res.data["resonance_freq_hz"] == pytest.approx(32701.5)
    prof = iprof.get_profile()
    assert prof["qplus_f0_measured_hz"] == pytest.approx(32701.5)
    assert prof["qplus_q_measured"] == pytest.approx(21000.0)
    assert prof["qplus_fq_updated_at"] > 0


def test_writeback_fires_the_persist_sink() -> None:
    """不落盘的话下次启动就丢了 —— 写回的意义正是跨会话活下来。"""
    seen: dict = {}
    iprof.set_persist_sink(lambda p: seen.update(p))
    AcquirePLLFreqSweep().execute(_Ctx(_sweep_payload(32768.0, 30000.0)), _PARAMS)
    assert seen.get("qplus_f0_measured_hz") == pytest.approx(32768.0)


def test_a_failed_sweep_does_not_store_junk() -> None:
    """扫描失败时 Nanonis 会回 0 / NaN。存进去比不存更糟 —— 之后每次都当真值用。"""
    for f0, q in ((0.0, 100.0), (32768.0, 0.0), (float("nan"), 100.0)):
        iprof.set_profile({})
        AcquirePLLFreqSweep().execute(_Ctx(_sweep_payload(f0, q)), _PARAMS)
        assert "qplus_f0_measured_hz" not in iprof.get_profile(), (f0, q)


def test_a_short_payload_is_ignored() -> None:
    """真机上遇到过返回位数不足的形状 —— 不能因此抛。"""
    res = AcquirePLLFreqSweep().execute(_Ctx([None, None, [1, 2, 3]]), _PARAMS)
    assert res.success is True
    assert "resonance_freq_hz" not in res.data
    assert "qplus_f0_measured_hz" not in iprof.get_profile()


def test_writeback_failure_never_breaks_the_sweep(monkeypatch) -> None:
    """记账失败不该让一次成功的扫描变成失败。"""
    def boom(*a, **k):
        raise RuntimeError("profile exploded")

    monkeypatch.setattr(iprof, "set_qplus_resonance", boom)
    res = AcquirePLLFreqSweep().execute(_Ctx(_sweep_payload(32768.0, 30000.0)), _PARAMS)
    assert res.success is True
    assert res.data["q_factor"] == pytest.approx(30000.0)


def test_measured_values_are_cleared_when_a_new_tip_is_registered() -> None:
    """旧那支音叉的共振不属于新装的这一支。"""
    AcquirePLLFreqSweep().execute(_Ctx(_sweep_payload(32768.0, 30000.0)), _PARAMS)
    assert "qplus_f0_measured_hz" in iprof.get_profile()

    dropped = iprof.clear_tip_bound_state()
    assert "qplus_f0_measured_hz" in dropped
    assert "qplus_f0_measured_hz" not in iprof.get_profile()
