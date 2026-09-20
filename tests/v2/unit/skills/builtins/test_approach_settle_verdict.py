"""进针判定必须来自连续一致的读数。

模块停止与反馈接管之间可能存在瞬态，不能立即用一次电流读取下结论。
多次采样的最大值也可能被尖峰误导，测试要求连续读数支持同一判断。
AutoApproach 与 ApproachTip 均须保留 True / False / None 三态：
已进入、稳定低于判据、预算内无法判断必须可区分。测试输入均为独立合成。"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field

import pytest

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.builtins.approach import (
    ApproachTip,
    AutoApproach,
    EngageVerdict,
    settle_engagement,
)

SETPOINT = 5e-10


def _reader(sequence, setpoint=SETPOINT, *, cycle=False):
    """A read_pair() serving ``sequence``.

    ``cycle=False`` holds the last value forever (a junction that reached a
    steady state); ``cycle=True`` repeats the whole sequence (a junction that
    keeps oscillating). The difference matters: the first shape can never
    produce an "unknown" verdict, so a test about unknown MUST use the second
    or it silently tests the settled case instead."""
    seq = list(sequence)
    box = {"i": 0}

    def _read():
        i = box["i"]
        box["i"] += 1
        return (seq[i % len(seq)] if cycle else seq[min(i, len(seq) - 1)],
                setpoint)

    return _read, box


# ══════════════════════════════════════════════════════════════════════
# settle_engagement — the judgement itself
# ══════════════════════════════════════════════════════════════════════

def test_settle_transient_then_engaged_is_success():
    """合成电流先低于相对门限，随后连续稳定在门限以上，应确认已进入隧穿态。"""
    read, _ = _reader([0.45 * SETPOINT, 0.90 * SETPOINT, 0.90 * SETPOINT])
    v = settle_engagement(read, interval_s=0.0, budget_s=5.0)
    assert v.engaged is True
    assert v.agreed_n >= 2
    assert v.current_a == pytest.approx(0.90 * SETPOINT)


def test_settle_persistently_low_still_fails():
    """持续低电流必须判 False，而不是无法判断；修复瞬态处理不能绕过低电流保护。"""
    read, _ = _reader([2.0e-13] * 8)
    v = settle_engagement(read, interval_s=0.0, budget_s=5.0)
    assert v.engaged is False           # 测出来是零,不是没测出来
    assert v.agreed_n >= 2


def test_transient_spike_is_not_engagement():
    """变异测试:把判据换成「窗口内出现过一次达标」,这条就会挂。

    序列 = 噪声、一次尖峰、噪声、噪声。峰值判据会说「进针了」;
    连续一致判据看到尖峰两侧都不一致,继续等,最后稳定在低位 → False。
    (尖峰必须出现在头两次读数之内,否则前两次噪声就已经一致、判定提前返回,
    尖峰根本没被读到 —— 那样这条测试就变成了另一条测试。)"""
    noise = 2.0e-13
    read, _ = _reader([noise, 2.0 * SETPOINT, noise, noise, noise])
    v = settle_engagement(read, interval_s=0.0, budget_s=5.0)
    assert v.engaged is False
    assert 2.0 * SETPOINT in v.reads_a, "证据里必须留着那次尖峰"


def test_never_agreeing_reads_are_unknown_not_failure():
    """读数一直在两侧跳 → engaged is None(判不了),不是 False(判定没进针)。

    「没测出来」与「测出来是零」必须是两句话 —— 把它们合成一句正是让坏判据
    看起来和正常结果一模一样的那个错误。"""
    read, _ = _reader([0.2 * SETPOINT, 2.0 * SETPOINT], cycle=True)
    v = settle_engagement(read, interval_s=0.0, budget_s=0.05)
    assert v.engaged is None
    assert v.reads_a, "判不了也要留下读到的数"
    assert v.total_reads_n >= len(v.reads_a)


def test_unreadable_pair_breaks_the_run():
    """读不到的采样会打断连续一致的判据，不能沿用前一次结果。"""
    seq = [0.90 * SETPOINT, None, 0.90 * SETPOINT, 0.90 * SETPOINT]
    read, _ = _reader(seq)
    v = settle_engagement(read, interval_s=0.0, budget_s=5.0)
    assert v.engaged is True
    assert v.unreadable_n == 1
    # 达标的读数一共 3 次,但中间断了 —— 结论只能来自最后那两次。
    assert v.agreed_n == 2


def test_unreadable_setpoint_never_yields_a_verdict():
    """设定点读不到 → 判据算不出来 → 只能是 None,绝不能是 False。

    (读不到设定点时把 bar 兜底成 0 会让噪声「达标」;兜底成 +∞ 会让成功进针
    被判失败。两个方向都错,所以不兜底。)"""
    read, _ = _reader([0.90 * SETPOINT] * 10, setpoint=None)
    v = settle_engagement(read, interval_s=0.0, budget_s=0.05)
    assert v.engaged is None
    assert v.unreadable_n >= 2


def test_dead_readback_gives_up_early_instead_of_burning_the_budget():
    """一次都读不到 → 不等满预算就返回。

    等下去不会把坏掉的读回链路等好,而「电流读不到」和「结还在稳定」是两种
    不同的修法,调用方现在就要拿到诊断。**读得到但不一致不能走这条早退** ——
    那正是要继续等的那种情况。"""
    import time

    read, _ = _reader([None] * 50)
    t0 = time.monotonic()
    v = settle_engagement(read, interval_s=0.05, budget_s=30.0)
    assert time.monotonic() - t0 < 5.0, "读不到也把预算烧完了"
    assert v.engaged is None
    assert v.total_reads_n == 0

    # 对照:读得到但一直不一致 —— 必须一直等到预算耗尽,不许走早退。
    read2, _ = _reader([0.2 * SETPOINT, 2.0 * SETPOINT], cycle=True)
    t1 = time.monotonic()
    v2 = settle_engagement(read2, interval_s=0.05, budget_s=0.4)
    assert time.monotonic() - t1 >= 0.35, "读数不一致时提前放弃了"
    assert v2.engaged is None
    assert v2.total_reads_n > 0


def test_abort_stops_the_window_without_a_verdict():
    read, _ = _reader([2.0e-13] * 10)
    v = settle_engagement(read, interval_s=0.0, budget_s=5.0,
                          check_abort=lambda: True)
    assert v.aborted is True
    assert v.engaged is None


def test_evidence_has_measurements_and_no_guesses():
    """结论只报告读数，不添加未经测量的解释。"""
    read, _ = _reader([2.0e-13] * 4)
    v = settle_engagement(read, interval_s=0.0, budget_s=5.0)
    text = v.evidence()
    assert "0.20 pA" in text and "500.00 pA" in text
    for guess in ("coarse range", "Z at limit", "stale state", "量程耗尽",
                  "外部停止", "motor range"):
        assert guess not in text, f"猜测词回来了:{guess}"


# ══════════════════════════════════════════════════════════════════════
# AutoApproach._phase_wait_complete — 第一处
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _WaitCtx:
    """模块跑 1 次轮询后停止;电流按 ``currents`` 逐次给出。"""

    currents: list = field(default_factory=list)
    setpoint_a: float = SETPOINT
    running_polls: int = 1
    calls: list = field(default_factory=list)
    _oog: int = 0
    _cur: int = 0

    def safe_call(self, method: str, *args, **kw) -> NanonisCallRecord:
        self.calls.append(method)
        if method == "AutoApproach_OnOffGet":
            self._oog += 1
            still = self._oog <= self.running_polls
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [1 if still else 0]))
        if method == "Current_Get":
            i = min(self._cur, len(self.currents) - 1)
            self._cur += 1
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [self.currents[i]]))
        if method == "ZCtrl_SetpntGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [self.setpoint_a]))
        return NanonisCallRecord(method=method, args=args, return_value=("", b"", []))

    def check_abort(self) -> bool:
        return False

    def emit_progress(self, progress) -> None:
        pass

    def get_progress(self, name):
        return None

    def checkpoint_flush(self) -> None:
        pass


def _fast(skill):
    skill._poll_interval_s = 0.01
    skill._engage_interval_s = 0.0
    skill._engage_budget_s = 2.0
    return skill


def test_wait_complete_survives_the_handover_transient():
    """交接后的暂态电流不应让已经完成的等待被误判失败。"""
    skill = _fast(AutoApproach())
    ctx = _WaitCtx(currents=[0.45 * SETPOINT, 0.90 * SETPOINT, 0.90 * SETPOINT])
    res = skill.execute(ctx, {})
    assert res.success, res.error


def test_wait_complete_still_fails_on_a_dead_junction():
    skill = _fast(AutoApproach())
    ctx = _WaitCtx(currents=[2.0e-13] * 6)
    res = skill.execute(ctx, {})
    assert not res.success
    assert "稳定地" in (res.error or "")


def test_wait_complete_failure_text_prints_measurements_not_guesses():
    skill = _fast(AutoApproach())
    ctx = _WaitCtx(currents=[2.0e-13] * 6)
    res = skill.execute(ctx, {})
    err = res.error or ""
    assert "0.20 pA" in err and "500.00 pA" in err
    for guess in ("coarse range", "Z piezo", "Z at limit", "stopped externally",
                  "量程耗尽", "motor range"):
        assert guess not in err, f"猜测词回来了:{guess}"


def test_wait_complete_stops_the_module_on_failure():
    """失败路径仍然停模块 —— 这条不是判据,是安全动作,不许被改动顺手丢掉。"""
    skill = _fast(AutoApproach())
    ctx = _WaitCtx(currents=[2.0e-13] * 6)
    skill.execute(ctx, {})
    assert "AutoApproach_OnOffSet" in ctx.calls


# ══════════════════════════════════════════════════════════════════════
# ApproachTip post-approach 复核 — 第二处(同形状,同一次修)
# ══════════════════════════════════════════════════════════════════════

class _TipCtx:
    def __init__(self, currents, setpoint=SETPOINT, *, cycle=False):
        self.currents = list(currents)
        self.setpoint = setpoint
        self.cycle = cycle
        self._i = 0
        self.names: list[str] = []

    def run(self, name: str, params: dict) -> SkillResult:
        self.names.append(name)
        if name == "TryEngageController":
            return SkillResult(skill_name=name, success=True,
                               data={"engaged": False, "needs_auto_approach": True})
        if name == "AutoApproach":
            return SkillResult(skill_name=name, success=True, data={})
        if name == "GetCurrent":
            i = (self._i % len(self.currents) if self.cycle
                 else min(self._i, len(self.currents) - 1))
            self._i += 1
            return SkillResult(skill_name=name, success=True,
                               data={"current_a": self.currents[i]})
        if name == "GetSetpoint":
            return SkillResult(skill_name=name, success=True,
                               data={"setpoint_a": self.setpoint})
        return SkillResult(skill_name=name, success=True, data={})

    def safe_call(self, method: str, *args, **kw) -> NanonisCallRecord:
        return NanonisCallRecord(method=method, args=args, error="not wired")


def _fast_tip():
    skill = ApproachTip()
    skill._engage_interval_s = 0.0
    skill._engage_budget_s = 2.0
    return skill


def test_approach_tip_survives_the_handover_transient():
    ctx = _TipCtx([0.45 * SETPOINT, 0.90 * SETPOINT, 0.90 * SETPOINT])
    res = _fast_tip().execute(ctx, {})
    assert res.success, res.error
    assert res.data["engagement"]["agreed_n"] >= 2


def test_approach_tip_still_rejects_a_lying_composite():
    """#42:AutoApproach 报 success 而电流在噪声底 —— 复核仍要拆穿它。"""
    ctx = _TipCtx([2.0e-13] * 6)
    res = _fast_tip().execute(ctx, {})
    assert not res.success
    assert res.data["engaged"] is False
    assert "稳定地" in (res.error or "")


def test_approach_tip_unknown_is_worded_differently_from_not_engaged():
    """三值判定必须落到三种文案,否则 None 会被读成 False。"""
    ctx = _TipCtx([0.2 * SETPOINT, 2.0 * SETPOINT], cycle=True)
    skill = _fast_tip()
    skill._engage_budget_s = 0.05
    res = skill.execute(ctx, {})
    assert not res.success
    err = res.error or ""
    assert "判不出" in err
    assert "这不等于没进针" in err
    assert "稳定地" not in err, "把「判不了」印成「测出来是零」"


def test_approach_tip_failure_text_has_no_guesses():
    ctx = _TipCtx([2.0e-13] * 6)
    res = _fast_tip().execute(ctx, {})
    err = res.error or ""
    for guess in ("coarse range exhausted", "Z at limit", "stale state",
                  "check Z position", "motor range"):
        assert guess not in err, f"猜测词回来了:{guess}"


def test_engage_verdict_three_values_are_distinguishable():
    """EngageVerdict 的 as_dict 必须区分三种状态 —— 下游读的是这个 dict。"""
    assert EngageVerdict(engaged=None).as_dict()["engaged"] is None
    assert EngageVerdict(engaged=False).as_dict()["engaged"] is False
    assert EngageVerdict(engaged=True).as_dict()["engaged"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
