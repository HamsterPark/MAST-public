"""2026-07-27 field (后半) — stopping a composite that is
ALREADY running when vision says the tip has gone bad.

Five CRITICAL ``tip_quality_drop`` interrupts fired that day, every one of them
AFTER the hardware action had finished:

    14:20:40  BufferHITLMiddleware raising interrupt … ['tip_quality_drop']
              ← BatchRegionsScan had JUST finished its 4 regions (107 s)
    14:20:54  tip_quality_drop · critical · #469 — "扫描中途针尖变化（已采集
              71 行中第 ~9 行）…其后行不可信，建议中止扫描并修针尖"

…and two more multi-region batches ran afterwards. The only consumer was a
middleware that runs BETWEEN LLM calls, and nothing runs during a tool call.

The fix is a stop signal with deliberately different physics from
``_orch_abort``. Reusing the abort latch here would be actively harmful:
``skill_adapter``'s abort gate refuses EVERY new instrument action, so a bad
tip would block 修针 / 退针 / 停扫 — the remedies — and wedge the instrument
against its own recovery. These tests pin both halves: the halt really stops a
running plan, and it really does not latch anything.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_forensics_20260727_tip_halt.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite.graph_executor import (  # noqa: E402
    CompositeStep,
    GraphExecutor,
)

RUN = "task-abc"
ADVICE = "扫描中途针尖变化（已采集 71 行中第 ~9 行）——其后行不可信，建议中止扫描并修针尖"


def _rt():
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._orch_run_id = RUN
    rt._tip_halt = None
    return rt


class _Ctx:
    """Minimal ExecutionContext stand-in for the executor."""

    def __init__(self, halt_check=None, abort=False):
        self.ran: list[str] = []
        self._abort = abort
        if halt_check is not None:
            self.check_halt = halt_check

    def check_abort(self) -> bool:
        return self._abort

    def run(self, skill_name, params, version=None):
        self.ran.append(skill_name)
        return SkillResult(skill_name=skill_name, success=True, data={})


def _plan(n=5):
    return [CompositeStep(step_id=f"region_{i}:scan", skill_name="StartScan",
                          params={}) for i in range(n)]


def _exec(ctx):
    return GraphExecutor(composite_name="BatchRegionsScan", context=ctx)


# ── the halt state machine ──

def test_halt_is_scoped_to_the_run_that_raised_it():
    rt = _rt()
    rt.raise_tip_halt(ADVICE, event_id="469")
    assert CoreRuntime.consume_tip_halt(rt, "some-other-run") == ""
    assert "tip_quality_drop" in CoreRuntime.consume_tip_halt(rt, RUN)


def test_halt_is_one_shot():
    """Consumed once, gone — a resolved event must not keep stopping every
    later plan in the same run."""
    rt = _rt()
    rt.raise_tip_halt(ADVICE, event_id="469")
    assert CoreRuntime.consume_tip_halt(rt, RUN)
    assert CoreRuntime.consume_tip_halt(rt, RUN) == ""


def test_halt_carries_the_advice_and_the_event_id():
    rt = _rt()
    rt.raise_tip_halt(ADVICE, event_id="469")
    msg = CoreRuntime.consume_tip_halt(rt, RUN)
    assert "建议中止扫描并修针尖" in msg and "469" in msg
    assert "未回滚" in msg          # the operator needs to know nothing was undone


def test_status_does_not_consume():
    rt = _rt()
    rt.raise_tip_halt(ADVICE, event_id="469")
    assert rt.tip_halt_status()["run_id"] == RUN
    assert CoreRuntime.consume_tip_halt(rt, RUN)      # still there


def test_clear_drops_it():
    rt = _rt()
    rt.raise_tip_halt(ADVICE, event_id="469")
    rt.clear_tip_halt()
    assert rt.tip_halt_status() is None
    assert CoreRuntime.consume_tip_halt(rt, RUN) == ""


def test_raise_never_raises_it_runs_on_the_publisher_thread():
    class _Hostile:
        _orch_run_id = RUN

        def __setattr__(self, k, v):
            raise RuntimeError("nope")
    CoreRuntime.raise_tip_halt(_Hostile(), ADVICE)   # must not propagate


# ── the executor actually stops ──

def test_a_running_plan_stops_at_the_next_step_boundary():
    """The 107-second BatchRegionsScan: the drop is detected after region 1 and
    regions 2-5 must not run."""
    rt = _rt()
    ctx = _Ctx(halt_check=rt._make_halt_check(RUN))
    ex = _exec(ctx)

    steps = iter(_plan(5))
    first = next(steps)
    ctx.run(first.skill_name, {})          # region 1 completed before the verdict
    rt.raise_tip_halt(ADVICE, event_id="469")

    assert ex.run_plan(steps) is False
    assert ctx.ran == ["StartScan"], "no further region may run"
    assert ex.progress.aborted is True
    assert "tip_quality_drop" in ex.progress.aborted_reason
    assert "修针尖" in ex.progress.aborted_reason


def test_without_a_halt_the_whole_plan_runs():
    rt = _rt()
    ctx = _Ctx(halt_check=rt._make_halt_check(RUN))
    assert _exec(ctx).run_plan(iter(_plan(5))) is True
    assert len(ctx.ran) == 5


def test_a_halt_from_another_run_does_not_stop_this_one():
    rt = _rt()
    rt.raise_tip_halt(ADVICE, run_id="an-older-task", event_id="469")
    ctx = _Ctx(halt_check=rt._make_halt_check(RUN))
    assert _exec(ctx).run_plan(iter(_plan(3))) is True
    assert len(ctx.ran) == 3


def test_a_context_without_check_halt_still_works():
    """Legacy contexts / test fakes have no check_halt — they simply never
    halt, rather than breaking the run."""
    ctx = _Ctx()
    assert not hasattr(ctx, "check_halt")
    assert _exec(ctx).run_plan(iter(_plan(3))) is True
    assert len(ctx.ran) == 3


def test_a_broken_check_halt_cannot_stop_work():
    def _boom():
        raise RuntimeError("halt check exploded")
    ctx = _Ctx(halt_check=_boom)
    assert _exec(ctx).run_plan(iter(_plan(2))) is True
    assert len(ctx.ran) == 2


def test_a_magicmock_context_does_not_halt_everything():
    """Most composite tests build their ExecutionContext from a MagicMock, and
    `mock.check_halt()` returns a truthy sentinel — which would stop every plan
    at its first step. Only a real reason STRING is a halt."""
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.check_abort.return_value = False
    ctx.run.side_effect = lambda name, params, version=None: SkillResult(
        skill_name=name, success=True, data={})
    ex = _exec(ctx)
    assert ex.run_plan(iter(_plan(3))) is True
    assert ex.progress.aborted is False
    assert ctx.run.call_count == 3


def test_a_non_string_halt_reason_is_ignored():
    for bogus in (True, 1, object(), ["stop"]):
        ctx = _Ctx(halt_check=lambda b=bogus: b)
        assert _exec(ctx).run_plan(iter(_plan(2))) is True, bogus
        assert len(ctx.ran) == 2


def test_the_halt_reason_is_distinguishable_from_an_operator_abort():
    """Both land in progress.aborted_reason; a post-mortem must be able to tell
    「用户中止」 from 「针尖坏了」."""
    rt = _rt()
    ctx = _Ctx(halt_check=rt._make_halt_check(RUN))
    rt.raise_tip_halt(ADVICE, event_id="469")
    ex = _exec(ctx)
    ex.run_plan(iter(_plan(2)))
    halt_reason = ex.progress.aborted_reason

    ctx2 = _Ctx(abort=True)
    ex2 = _exec(ctx2)
    ex2.run_plan(iter(_plan(2)))
    abort_reason = ex2.progress.aborted_reason

    assert halt_reason.startswith("tip_quality_drop:")
    assert not abort_reason.startswith("tip_quality_drop:")


# ── the buffer hook that arms it ──

def _tip_event(severity="critical", kind=None, advice=ADVICE, seq=469):
    from mast.buffer.schemas import (
        Severity, VisionEvent, VisionEventType,
    )
    return VisionEvent(
        event_id=f"ev-{seq}",
        kind=kind or VisionEventType.TIP_QUALITY_DROP,
        severity=Severity(severity), seqno=seq, ts=0.0,
        payload={"advice": advice})


def test_a_critical_physical_drop_arms_the_halt():
    """原名 ``test_a_critical_tip_drop_arms_the_halt``,喂的是**视觉**事件。

    ⑰-C1(2026-08-09)之后视觉判定在任何场景下都不再 arm 这个 halt,所以这条改喂
    电流监控来源 —— 它要证明的性质没变(**一条 CRITICAL 能 arm halt,并把判定原文
    带到用户面前**),只是唯一还能触发它的来源变了。
    视觉那一侧由 ``test_a_vision_verdict_never_halts_in_any_scenario`` 反向钉住。
    """
    from mast.core.runtime import make_tip_halt_hook
    rt = _rt()
    make_tip_halt_hook(rt)(_current_event("current_saturation", seq=469))
    h = rt.tip_halt_status()
    assert h and h["run_id"] == RUN and h["event_id"] == "ev-469"
    assert "current_saturation 摘要" in h["reason"]


def test_a_warn_level_tip_drop_does_not_arm_the_halt():
    """A WARN drop is an observation, not grounds for stopping a plan."""
    from mast.core.runtime import make_tip_halt_hook
    rt = _rt()
    make_tip_halt_hook(rt)(_tip_event(severity="warn"))
    assert rt.tip_halt_status() is None


def test_other_event_kinds_do_not_arm_the_halt():
    from mast.buffer.schemas import VisionEventType
    from mast.core.runtime import make_tip_halt_hook
    rt = _rt()
    hook = make_tip_halt_hook(rt)
    for kind in (VisionEventType.E_STOP, VisionEventType.SCAN_COMPLETE):
        hook(_tip_event(kind=kind))
    assert rt.tip_halt_status() is None


def test_a_malformed_event_cannot_break_event_fanout():
    from mast.core.runtime import make_tip_halt_hook
    make_tip_halt_hook(_rt())(object())          # must not raise


# ── the thing it must NOT do ──

def test_the_halt_does_not_touch_the_abort_latch():
    """The whole point. `_orch_abort` makes skill_adapter refuse every new
    instrument action; if a bad tip set it, 修针 / 退针 / 停扫 would all be
    blocked and the instrument would be wedged against its own recovery."""
    import threading

    rt = _rt()
    rt._orch_abort = threading.Event()
    rt._executor = None
    rt.raise_tip_halt(ADVICE, event_id="469")
    assert rt._orch_abort.is_set() is False


def test_a_skill_still_runs_while_a_halt_is_pending():
    """skill_adapter's gate reads check_abort(), which the halt never sets — so
    the tip-recovery skills the verdict itself recommends stay available."""
    from types import SimpleNamespace

    from mast.agents._shared.skill_adapter import wrap_skill
    from mast.core.types import ParameterSpec, SafetyLevel, SkillMetadata

    class _TipPrep:
        def metadata(self):
            return SkillMetadata(name="TipShapeWithReadback", version="1.0.0",
                                 safety_level=SafetyLevel.CONFIRM,
                                 description="stub", parameters=[
                                     ParameterSpec(name="depth_m", type="float",
                                                   required=False, default=1e-9,
                                                   description="")])

        def validate_params(self, params):
            return []

        def execute(self, ctx, params):
            return SkillResult(skill_name="TipShapeWithReadback", success=True,
                               data={"indent": {"verdict": "improved"}})

    rt = _rt()
    rt.raise_tip_halt(ADVICE, event_id="469")

    ctx = SimpleNamespace(check_abort=lambda: False,
                          check_halt=rt._make_halt_check(RUN))
    tool = wrap_skill(_TipPrep, lambda: ctx)
    out = tool.func(depth_m=1e-9, tool_call_id="tc1")
    assert "aborted" not in str(out), "the remedy must not be refused"
    assert out.update["executed_skills"] == ["TipShapeWithReadback"]



# ══════════════════════════════════════════════════════════════════════
# ⑰-C2(2026-08-09):修针期间,**瞬变类**物理事件不再 arm 这个 halt
# ══════════════════════════════════════════════════════════════════════
#
# ⑭(2026-08-06 ForgeAuTip 首演)已经判定「修针期间的瞬变类物理事件零真阳性、不该
# 打断」,但当时只改了**确认框**那一层(``buffer_hitl``)。halt 这一层漏掉了 ——
# ``_tip_work_suppresses_tip_halt()`` 第一行就是「``source=current_monitor`` 返回
# 空」,把整个电流监控来源一刀切掉。
#
# 后果正是用户在首演里挨的那一下:弹框不弹了,**外环照样被从中间掐断**。
# ⑰-C2 补上另一半 —— 不是新决定,是同一个决定的另一半。
#
# 边界照 ⑭:**瞬变豁免、持续不豁免**。下面四条把两侧都钉住,外加「没归类的默认
# 不豁免」和「非修针场景行为不变」——只钉豁免那一侧的话,一个「全放行」的实现
# 也会全绿。

_TIP_WORK = "mast.core.tip_intent.active_tip_work"


def _current_event(signal: str, *, source: str = "current_monitor", seq: int = 900):
    """一条电流监控来源的 CRITICAL ``tip_quality_drop``。

    与视觉来源**同一个 kind** —— 这正是判据必须读 payload 而不是读 kind 的原因。
    """
    from mast.buffer.schemas import Severity, VisionEvent, VisionEventType

    return VisionEvent(
        event_id=f"ev-{seq}",
        kind=VisionEventType.TIP_QUALITY_DROP,
        severity=Severity("critical"), seqno=seq, ts=0.0,
        payload={"signal": signal, "source": source,
                 "summary_zh": f"{signal} 摘要"})


def _armed(monkeypatch, event, *, tip_work: str) -> bool:
    """在指定的修针令牌状态下投一条事件,返回 halt 有没有被 arm。"""
    from mast.core import tip_intent
    from mast.core.runtime import make_tip_halt_hook

    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: tip_work,
                        raising=True)
    rt = _rt()
    make_tip_halt_hook(rt)(event)
    return rt.tip_halt_status() is not None


def test_a_transient_spike_no_longer_halts_during_tip_work(monkeypatch):
    """**⑰-C2 的主判据。** 修针期间的 giant_spike 不再中止外环。

    这一条以前是红的(⑭ 只改了弹框那一层),它就是用户首演里挨的那一下。
    """
    assert _armed(monkeypatch, _current_event("current_giant_spike"),
                  tip_work="ForgeAuTip") is False


@pytest.mark.parametrize("signal", ["current_saturation", "current_freeze"])
def test_sustained_physical_still_halts_during_tip_work(monkeypatch, signal):
    """**豁免的边界。** 持续类不豁免 —— 修针不该造成持续贴轨。

    把它们一起放行等于在最需要拦的时候把拦截关掉:一支压进表面出不来的针,恰恰
    最可能出现在修针流程里。
    """
    assert _armed(monkeypatch, _current_event(signal),
                  tip_work="ForgeAuTip") is True


def test_an_unclassified_current_signal_still_halts_during_tip_work(monkeypatch):
    """白名单不是黑名单:明天新加一条判据,没被归进瞬变类就照旧 halt。

    没有这一条,「豁免全部电流来源」的实现会和正确实现一样绿 —— 那正是 ⑰-C2 之前
    的那个 bug 的镜像。
    """
    assert _armed(monkeypatch, _current_event("some_new_rule_2027"),
                  tip_work="ForgeAuTip") is True


def test_the_same_spike_still_halts_outside_tip_work(monkeypatch):
    """非修针场景**行为不变** —— 豁免的条件是「正在蓄意修针」,不是「是个瞬变」。"""
    assert _armed(monkeypatch, _current_event("current_giant_spike"),
                  tip_work="") is True


def test_the_vision_exemption_widened_from_tip_work_to_everything(monkeypatch):
    """原名 ``test_the_vision_exemption_is_unchanged``,断言的后半句被 C1 反转。

    ⑰-C2 那一轮它写的是「修针期间不 arm、非修针期间 arm」—— ⑬ 的边界。
    ⑰-C1 把视觉豁免从「修针场景」扩到**全场景**,所以后半句现在也是 False。
    这条留在原地,是为了让「⑬ 的边界后来怎么了」有一个落点。
    """
    ev = _tip_event()                                   # payload 只有 advice
    assert _armed(monkeypatch, ev, tip_work="ForgeAuTip") is False
    assert _armed(monkeypatch, ev, tip_work="") is False   # ⑰-C1:曾经是 True


def test_an_unreadable_tip_token_halts(monkeypatch):
    """读不到令牌 ⇒ **照常 halt**。

    豁免的失败模式必须是「保护照常生效」,而不是「因为看不清所以放行」。
    """
    from mast.core import tip_intent

    def _boom():
        raise RuntimeError("lock unavailable")

    monkeypatch.setattr(tip_intent, "active_tip_work", _boom, raising=True)
    from mast.core.runtime import make_tip_halt_hook
    rt = _rt()
    make_tip_halt_hook(rt)(_current_event("current_giant_spike"))
    assert rt.tip_halt_status() is not None


def test_the_two_buckets_are_the_single_source_and_cover_everything():
    """判据只有一份,而且每个物理信号都被归了类。

    ⑰-C2 把两张表从 ``agents/_shared/buffer_hitl`` 搬到 ``core/tip_intent`` ——
    「会停仪器的那一方」不该去「只写通知措辞的模块」里 import 判据。这里钉住搬完
    之后**只有一份**(re-export 指向同一个对象),以及「忘了归类」不会悄悄变成豁免。
    """
    from mast.agents._shared import buffer_hitl as bh
    from mast.core import tip_intent as ti

    assert bh.TRANSIENT_PHYSICAL_SIGNALS is ti.TRANSIENT_PHYSICAL_SIGNALS
    assert bh.SUSTAINED_PHYSICAL_SIGNALS is ti.SUSTAINED_PHYSICAL_SIGNALS
    assert bh.exempt_during_tip_work is ti.exempt_during_tip_work

    assert not (ti.TRANSIENT_PHYSICAL_SIGNALS & ti.SUSTAINED_PHYSICAL_SIGNALS)
    unclassified = ti._PHYSICAL_SIGNAL_FALLBACK - (
        ti.TRANSIENT_PHYSICAL_SIGNALS | ti.SUSTAINED_PHYSICAL_SIGNALS)
    assert not unclassified, f"没归类的物理信号:{sorted(unclassified)}"


def test_the_estop_and_crash_paths_are_untouched_by_this_exemption(monkeypatch):
    """C2 只碰 ``tip_quality_drop`` 的 halt。E_STOP 与撞针另有其路,一个字没动。

    ``make_tip_halt_hook`` 第一行就按 kind 过滤,所以修针期间的 E_STOP 根本不进这条
    分支 —— 它走的是 abort 闩锁。这条断言证明豁免碰不到它。
    """
    import threading

    from mast.buffer.schemas import VisionEventType
    from mast.core import tip_intent
    from mast.core.runtime import make_tip_halt_hook

    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: "ForgeAuTip",
                        raising=True)
    rt = _rt()
    rt._orch_abort = threading.Event()
    hook = make_tip_halt_hook(rt)
    hook(_tip_event(kind=VisionEventType.E_STOP))
    # 这个 hook 本来就不管 E_STOP(它只 arm composite halt),所以没被 arm 是对的;
    # E_STOP 的停止力在 runtime._estop_sets_abort → abort 闩锁上,而修针豁免连
    # 那条分支的门都进不去。
    assert rt.tip_halt_status() is None
    assert rt._orch_abort.is_set() is False



# ══════════════════════════════════════════════════════════════════════
# ⑰-C1(2026-08-09):**视觉针尖判定在任何场景下都不再中止 composite**
# ══════════════════════════════════════════════════════════════════════
#
# 判据:「也割掉」。理由与 ⑰ 整条线同源 —— 它是「猜针尖好坏然后停掉工作」,
# 两天实机零真阳性;而它**一次都没有**在「拒绝型防护接不住」的场合救过场。
#
# ## 这一节里两条断言反了向,而且是本文件最核心的那两条
#
# 本文件开头那段 2026-07-27 取证的结论是:「五个 CRITICAL 全部在动作**之后**才发,
# 而 run 照扫不误 —— 需要一个能在步边界真正停住 composite 的停止信号」。那个信号
# 建对了(下面 `test_a_running_plan_stops_at_the_next_step_boundary` 一字未改),
# **但它的视觉那一路触发器现在被拿掉了**。
#
# 换句话说:2026-07-27 促成这套机制的那条事件(视觉判「其后行不可信,建议中止
# 扫描」),今天不会再中止任何东西。这是**刻意**的,不是回归。三次收窄的终点:
#
#   2026-08-01  SAFE 模式下豁免视觉来源
#   2026-08-05(⑬)  蓄意修针期间豁免视觉来源
#   2026-08-09(⑰-C1)  **任何模式、任何场景**都不再中止
#
# ## 要把它改回来,需要观测到什么
#
# 举出一次实例 —— 视觉针尖判定中止了一条 composite,而那次中止**避免了一次真实
# 损害**,并且撞针状态机、电流监控的物理三类、针尖包络、SafetyGate 都没有接住它。
# 截至 2026-08-09 **零例**(「至今没被观测到」,不是「不可能存在」)。
#
# 停止机制本身**没有删**:``raise_tip_halt`` / ``check_halt`` / 步边界消费全部原样,
# 电流监控的物理越界仍然经它中止。割掉的只是「视觉」这一路触发器。


@pytest.mark.parametrize("payload", [
    {"advice": ADVICE},                                   # 历史生产者(无 signal)
    {"signal": "tip_change", "source": "vision"},
    {"signal": "learned_quality", "source": "vision"},
    {"signal": "some_new_vision_rule_2027"},              # 明天新加的判定
])
@pytest.mark.parametrize("tip_work,safe", [
    ("", False),            # 非修针、AUTO —— ⑬/⑭ 都没覆盖过的那一格
    ("ForgeAuTip", False),  # 修针期间(⑬ 已豁免)
    ("", True),             # SAFE 模式(2026-08-01 已豁免)
])
def test_a_vision_verdict_never_halts_in_any_scenario(monkeypatch, payload,
                                                      tip_work, safe):
    """**⑰-C1 的主判据。** 12 格全部「不中止」。

    第一格(非修针 + AUTO)是这次真正新增的;另外两格是 ⑬ 与 2026-08-01 已经豁免的,
    一并钉住,因为把它们写在一个参数化里,才能保证下一个人不会「只把某一格接回去」。
    """
    from mast.buffer.schemas import Severity, VisionEvent, VisionEventType
    from mast.core import tip_intent
    from mast.core.runtime import make_tip_halt_hook

    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: tip_work,
                        raising=True)
    monkeypatch.setattr("mast.core.runtime.safe_mode_active", lambda: safe)
    rt = _rt()
    make_tip_halt_hook(rt)(VisionEvent(
        event_id="ev-c1", kind=VisionEventType.TIP_QUALITY_DROP,
        severity=Severity("critical"), seqno=469, ts=0.0, payload=payload))
    assert rt.tip_halt_status() is None


def test_the_2026_07_27_event_itself_no_longer_stops_the_run(monkeypatch):
    """把那条促成整套机制的事件原样投进去 —— 它现在不中止任何东西。

    单独写一条(而不是只靠上面的参数化),是因为这是**本文件的立身之本被反转**,
    值得有一个能被 grep 到的名字。措辞照抄当天的 payload。
    """
    from mast.core import tip_intent
    from mast.core.runtime import make_tip_halt_hook

    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: "", raising=True)
    monkeypatch.setattr("mast.core.runtime.safe_mode_active", lambda: False)
    rt = _rt()
    make_tip_halt_hook(rt)(_tip_event())          # payload = {"advice": ADVICE}
    assert rt.tip_halt_status() is None, (
        "视觉针尖判定又开始中止 composite 了 —— 若这是刻意的，请先回答 ⑰-C1 的"
        "翻盘观测：哪一次中止避免了真实损害，而拒绝型防护都没接住？")


def test_the_vision_verdict_still_leaves_a_ledger_line(monkeypatch):
    """**「记录」那一半。** 不中止 ≠ 不记账。

    验收时该问的是「本来会拦我几次」,而这一行就是那个数的落点。
    按 ``subject`` 过滤而不是读最新一条:``diagnostics`` 是进程级环形缓冲,同一次
    pytest 里别的文件写进去的行会让「读最新」读到不属于本测试的东西(本轮改造中
    真的栽过一次)。
    """
    from mast.core import diagnostics as diag, tip_intent
    from mast.core.runtime import make_tip_halt_hook

    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: "", raising=True)
    monkeypatch.setattr("mast.core.runtime.safe_mode_active", lambda: False)

    def _n() -> int:
        return len([r for r in diag.recent(500, kinds=("notice_only",))
                    if r.get("subject") == "tip_halt:vision"])

    before = _n()
    make_tip_halt_hook(_rt())(_tip_event())
    rows = [r for r in diag.recent(500, kinds=("notice_only",))
            if r.get("subject") == "tip_halt:vision"]
    assert len(rows) == before + 1, "视觉判定被放行了,却没有留下任何账"
    assert "只记录不中止流程" in rows[0]["reason"]
    assert "建议中止扫描并修针尖" in rows[0]["reason"], "判定原文必须带出来"
    assert rows[0]["event_id"] == "ev-469"


@pytest.mark.parametrize("signal", ["current_saturation", "current_freeze",
                                    "current_giant_spike"])
def test_physical_halts_are_untouched_outside_tip_work(monkeypatch, signal):
    """**C1 的边界。** 非修针场景下,电流监控的物理三类照旧 halt —— 一格没动。

    只钉「视觉不中止」的话,一个「谁都不中止」的实现会全绿,而那正是这次**不该**
    做的事(要求保留物理来源)。
    """
    from mast.core import tip_intent
    from mast.core.runtime import make_tip_halt_hook

    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: "", raising=True)
    monkeypatch.setattr("mast.core.runtime.safe_mode_active", lambda: False)
    rt = _rt()
    make_tip_halt_hook(rt)(_current_event(signal))
    assert rt.tip_halt_status() is not None


def test_safe_mode_does_not_switch_off_the_physical_halt(monkeypatch):
    """SAFE 是「不修针」,从来不是「关掉物理保护」—— 这条 2026-08-01 的分界线没动。"""
    from mast.core import tip_intent
    from mast.core.runtime import make_tip_halt_hook

    monkeypatch.setattr(tip_intent, "active_tip_work", lambda: "", raising=True)
    monkeypatch.setattr("mast.core.runtime.safe_mode_active", lambda: True)
    rt = _rt()
    make_tip_halt_hook(rt)(_current_event("current_saturation"))
    assert rt.tip_halt_status() is not None


def test_the_source_discrimination_has_exactly_one_definition():
    """⑰-C1 之前这段判别写了**三遍**:SAFE 抑制、修针抑制、``raise_tip_halt`` 的
    ``source=`` 实参。三份必须保持相等的判别正是本仓反复踩过的形状,现在只剩
    :func:`tip_halt_source` 一份。

    连同「读不出来算 vision」这个默认一起钉 —— 它决定了明天新加的判定在有人明确
    归类之前**不会**获得中止实验的权力。
    """
    from mast.core import runtime as rt_mod

    assert not hasattr(rt_mod, "_safe_mode_suppresses_tip_halt"), (
        "SAFE 专用的视觉抑制又回来了 —— 它已被「视觉在任何模式都不中止」包含")

    src = rt_mod.tip_halt_source
    assert src({"source": "current_monitor"}) == "current_monitor"
    assert src({"signal": "current_saturation"}) == "current_monitor"
    assert src({"signal": "current_giant_spike"}) == "current_monitor"
    assert src({"signal": "tip_change", "source": "vision"}) == "vision"
    assert src({}) == "vision"                       # 读不出来 → 安静那一侧
    assert src({"signal": "brand_new_2027"}) == "vision"


def test_the_halt_machinery_itself_is_still_there():
    """割的是**触发器**,不是**机制**。

    ``raise_tip_halt`` / ``check_halt`` / 步边界消费全部原样 —— 电流监控的物理越界
    仍然经它中止一条正在跑的 composite(上面那两条 physical 测试证明这条路活着)。
    这条断言防的是「顺手把整套 halt 删了」的过头修法。
    """
    from mast.core.runtime import CoreRuntime

    for attr in ("raise_tip_halt", "consume_tip_halt", "tip_halt_status",
                 "_make_halt_check"):
        assert hasattr(CoreRuntime, attr), attr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
