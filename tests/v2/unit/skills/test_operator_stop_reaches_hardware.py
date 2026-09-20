"""用户喊停 → 循环退出 → 硬件收势 → 语义如实(缺陷⑬ 的下半截)。

上半截(信号到不到得了技能)钉在 ``tests/v2/unit/chat/test_abort_reaches_the_skill.py``:
根因是 `active_abort_event()` 的 thread-local 在 generator 换 worker 线程后读不到。

这里钉的是**信号到了之后**该发生什么,团队交底的四条验收:

1. abort 置位后 **≤2 个 poll 周期**内循环退出;
2. **硬件收势的调用已发**(AutoApproach → 停模块);
3. SkillResult 语义 = **operator_abort**(不是失败,不是超时);
4. E_STOP 路径不受影响 —— 它仍是更高层的核弹。

还有第五条,是要求四的另一半:**软停只停不动**。收势只做安全方向的动作(停模块),
**不做写类「恢复」**(不自作主张把压电搬回去、不 withdraw)—— 那是 E_STOP 和用户的事。
停在哪儿要说清楚,因为下一个动作会在这个状态上做。
"""
from __future__ import annotations

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

import pytest  # noqa: E402

from mast.skills.builtins.approach import AutoApproach  # noqa: E402
from mast.skills.composite.graph_executor import (  # noqa: E402
    CompositeProgress,
    abort_facts,
)


class _Rig:
    """一台一直在跑进针的假机器;``stop_after`` 个 poll 之后用户按停止。"""

    def __init__(self, *, stop_after=3):
        self.stop_after = stop_after
        self.calls: list[str] = []
        self.polls = 0
        self.polls_after_stop = 0
        self._stopped = False

    def safe_call(self, method, *args, **kw):
        self.calls.append(method)
        if method == "AutoApproach_OnOffGet":
            self.polls += 1
            if self._stopped:
                self.polls_after_stop += 1
        rv = ("", b"", [1])          # 一直「在跑」
        if method == "AutoApproach_OnOffSet":
            rv = ("", b"", [])

        class _R:
            error = ""
            return_value = rv
            method = ""
            args = ()
        return _R()

    def check_abort(self):
        if self.polls >= self.stop_after:
            self._stopped = True
        return self._stopped

    def run(self, name, params):
        from mast.core.types import SkillResult
        return SkillResult(skill_name=name, success=True, data={})


def _wait_phase(rig):
    skill = AutoApproach()
    skill._poll_interval_s = 0.0
    skill._crosstalk_every_s = 0          # 报告不参与这条钉子
    skill._wait_timeout_s = 60.0

    class _Exec:
        def set_partial(self, *a, **k):
            pass

    skill._executor = _Exec()
    skill._call_log = []
    return skill._phase_wait_complete(rig)


def test_the_wait_loop_exits_within_two_polls_of_the_stop():
    """**≤2 个 poll 周期。** 一个「最终会停」的循环不是软停通道。

    用户按下停止之后每多转一圈,都是仪器在做他已经叫停的事。
    """
    rig = _Rig(stop_after=3)
    _wait_phase(rig)
    assert rig.polls_after_stop <= 2, f"停止后又轮询了 {rig.polls_after_stop} 圈"


def test_the_module_is_actually_stopped():
    """收势要**到硬件**:光返回一个「已中止」而模块还在走,针还在往下扎。"""
    rig = _Rig(stop_after=2)
    _wait_phase(rig)
    assert "AutoApproach_OnOffSet" in rig.calls, (
        f"没有发出停模块的命令:{rig.calls}")


def test_the_result_is_an_operator_stop_not_a_failure():
    """「用户喊停」和「跑失败了」是两句话 —— 下游要能分得开,而且不靠猜措辞。"""
    rig = _Rig(stop_after=2)
    res = _wait_phase(rig)
    assert res.success is False          # 它确实没完成
    text = (res.error or "")
    assert "abort" in text.lower() or "中止" in text


def test_abort_facts_separates_the_operator_from_everything_else():
    """只有**中止闩**算「用户喊停」。

    CRITICAL 针尖停机、必要步骤失败也会让 aborted=True —— 但没有人喊过停。
    把它们混成一个位,就会出现 #46 那种「我可没有 aborted,谁在冒充我」。
    """
    p = CompositeProgress("X")
    p.aborted = True
    p.aborted_reason = "aborted by user"
    assert abort_facts(p)["aborted_by_operator"] is True

    p2 = CompositeProgress("X")
    p2.aborted = True
    p2.aborted_reason = "tip quality CRITICAL — halted"
    facts = abort_facts(p2)
    assert facts["aborted"] is True
    assert facts["aborted_by_operator"] is False
    assert "CRITICAL" in facts["abort_reason"]

    p3 = CompositeProgress("X")
    assert abort_facts(p3) == {"aborted": False, "aborted_by_operator": False,
                               "abort_reason": ""}


def test_estop_is_not_weakened_by_any_of_this():
    """E_STOP 仍是更高层的核弹 —— 它走的是同一个 check_abort,只是设的人不同。

    这条钉的是「没被绕开」:任何一个让 check_abort 为真的来源,循环都必须退出。
    分不出来源是**对的** —— 软停和急停在「要不要立刻停下」上没有分歧。
    """
    class _EStop(_Rig):
        def check_abort(self):
            return True                  # 第一圈就被扳

    rig = _EStop()
    _wait_phase(rig)
    assert rig.polls <= 2
    assert "AutoApproach_OnOffSet" in rig.calls


# ══════════════════════════════════════════════════════════════════════
# 软停只停不动:不做写类「恢复」
# ══════════════════════════════════════════════════════════════════════

def test_a_soft_stop_does_not_move_the_piezo_back():
    """倾斜循环被停 ⇒ **不回滚**,但要说清楚停在哪儿。

    把压电搬回去是一次写类恢复 —— 软停的语义是「停手」,不是「替我收拾」。
    要求四点名了这条(例子是自作主张 withdraw)。停在哪儿必须有人说,否则下一个
    动作是在一个没人描述过的状态上做的。
    """
    from mast.skills.composite import auto_tilt

    calls: list = []
    res = auto_tilt._stopped_by_operator("AutoTilt", calls, "倾斜停在 0.1/0.2°")
    assert res.data["aborted_by_operator"] is True
    assert res.data["aborted"] is True
    assert "停在" in res.data["stopped_where"]
    assert calls == [], "软停返回里不该夹带任何新的硬件调用"


def test_the_stop_check_fails_open():
    """查停止本身坏了,不该把技能带走 —— 那会把一个能跑的流程变成不能跑的。"""
    from mast.skills.composite import auto_tilt

    class _Broken:
        def check_abort(self):
            raise RuntimeError("boom")

    assert auto_tilt._operator_stopped(_Broken()) is False

    class _NoChannel:
        pass

    assert auto_tilt._operator_stopped(_NoChannel()) is False


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
