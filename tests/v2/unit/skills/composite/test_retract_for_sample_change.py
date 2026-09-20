"""RetractForSampleChange (换样品/关机 laddered coarse retract) tests.

Pins the anti-crash contract:
  1. Ladder = 1 → 10 → 100 → (rest chunked by step_max).
  2. Normal (receding) retract completes; coarse steps use the configured
     Nanonis direction code; total steps retracted == requested.
  3. A BACKWARDS direction is caught on the FIRST (1-step) rung: the Z-piezo
     retracts (approaching) → abort + StopMotor + withdraw, only 1 step spent.
  4. A current spike during the self-check also trips (approaching) even when
     Z is inconclusive.
  5. stop_powered halts AutoApproach + motor first.
  6. Abort mid-check stops the motor and withdraws.
  7. dI/dV is read when a lock-in signal index is configured.
  8. Metadata is CONFIRM + tagged dangerous/retract.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/composite/test_retract_for_sample_change.py -x -v
"""
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

from mast.core import instrument_profile as ip
from mast.core.types import NanonisCallRecord, SafetyLevel
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.retract_for_sample_change import RetractForSampleChange


@pytest.fixture(autouse=True)
def _clean_profile():
    ip.set_persist_sink(None)
    ip.set_profile({})
    yield
    ip.set_persist_sink(None)
    ip.set_profile({})


# ── FakeCtx ───────────────────────────────────────────────────────────────
@dataclass
class FakeCtx:
    """Synthetic coarse-retract model with a Z loop that ramps over successive polls.

The mode independently specifies receding, approaching or ambiguous motion.
The configuration does not determine that response. A bounded travel per poll
prevents a fixed sleep from passing merely because the fake teleports to its
final reading. verify_current can inject a separate current spike."""
    baseline_z: float = 0.0
    setpoint_a: float = 1e-10
    mode: str = "receding"
    z_delta: float = 5e-9
    verify_current: float = 1e-13
    lockin_val: float = 2e-4
    abort_after: int | None = None
    #: How far the piezo travels per poll. Small enough that reaching the
    #: surface from the withdraw position takes many polls — a settle that
    #: concludes in one or two has not observed the ramp at all.
    travel_per_poll: float = 200e-9
    withdraw_span: float = 1e-6
    calls: list = field(default_factory=list)
    _z: float | None = None
    _target: float | None = None
    _loop_on: bool = False
    _abort_count: int = 0

    # ── physics ───────────────────────────────────────────────────────────
    def _init_state(self) -> None:
        if self._target is None:
            self._target = self.baseline_z
            self._z = self.baseline_z - self.withdraw_span

    def _advance(self) -> None:
        if not self._loop_on:
            return
        if abs(self._z - self._target) <= self.travel_per_poll:
            self._z = self._target
        else:
            self._z += (self.travel_per_poll if self._target > self._z
                        else -self.travel_per_poll)

    def _engaged(self) -> bool:
        return self._loop_on and self._z == self._target

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        self._init_state()

        def _rec(vals):
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", vals))

        if method == "ZCtrl_Withdraw":
            self._z = self.baseline_z - self.withdraw_span
            self._loop_on = False
            return _rec([])
        if method == "ZCtrl_OnOffSet":
            self._loop_on = bool(args[0]) if args else False
            return _rec([])
        if method == "ZCtrl_OnOffGet":
            return _rec([1 if self._loop_on else 0])
        if method == "ZCtrl_ZPosGet":
            self._advance()
            return _rec([self._z])
        if method == "ZCtrl_SetpntGet":
            return _rec([self.setpoint_a])
        if method == "Current_Get":
            # Tunnelling only once the loop is holding the surface; the test's
            # verify_current rides on top so a spike can be modelled.
            physics = self.setpoint_a if self._engaged() else 0.0
            return _rec([max(physics, self.verify_current)])
        if method == "Motor_StartMove":
            code = int(args[0]) if args else 0
            if code in (4, 5):      # coarse Z
                if self.mode == "receding":
                    self._target += self.z_delta
                elif self.mode == "approaching":
                    self._target -= self.z_delta
            return _rec([])
        if method == "Signals_ValGet":
            return _rec([self.lockin_val])
        return _rec([])

    def check_abort(self) -> bool:
        self._abort_count += 1
        if self.abort_after is not None and self._abort_count >= self.abort_after:
            return True
        return False


def _mk(**profile):
    # ``FakeCtx`` 的物理是「退针 ⇒ target 升 ⇒ Z 升」,而 ``ZCtrl_Withdraw`` 把 Z 拉到
    # ``baseline_z - withdraw_span``(下方)—— 也就是**伸长时 Z 增大**,``+1``。
    #
    # 必须显式写:从 2026-08-11 起 ``_judge_recede`` 用 ``z_extend_sign_or_none()``,
    # 没声明过的机器一律 ``no_sign`` 拒判。在此之前这些用例全都在吃出厂默认 ——
    # 也就是说它们钉住的行为**取决于一个没人回答过的值**。
    # 想测「没声明」那一支的用例传 ``z_extend_sign=None`` 覆盖掉。
    profile.setdefault("z_extend_sign", "+1")
    ip.set_profile({k: v for k, v in profile.items() if v is not None})
    skill = RetractForSampleChange()
    # Poll as fast as the fake answers: the ramp here is driven by POLL COUNT,
    # not wall clock, so the settle runs in full without sleeping. There is no
    # "settle window" left to zero out — that knob was the bug.
    skill._poll_interval_s = 0.0
    skill._settle_timeout_s = 5.0     # wall-clock backstop; a healthy rig never hits it
    return skill


def _methods(ctx):
    return [c[0] for c in ctx.calls]


# ── shape ──────────────────────────────────────────────────────────────────
def test_is_composite_graph():
    assert issubclass(RetractForSampleChange, CompositeSkillGraph)


def test_metadata_confirm_and_dangerous():
    meta = RetractForSampleChange().metadata()
    assert meta.name == "RetractForSampleChange"
    assert meta.safety_level == SafetyLevel.CONFIRM
    assert "dangerous" in meta.tags
    assert "retract" in meta.tags


def test_ladder_shape():
    assert RetractForSampleChange._ladder(3000, 1000) == [1, 10, 100, 1000, 1000, 889]
    assert RetractForSampleChange._ladder(50, 1000) == [1, 10, 39]
    assert RetractForSampleChange._ladder(5, 1000) == [1, 4]


def test_plan_structure():
    skill = _mk(retract_total_steps=111)
    plan = skill.plan({})
    ids = [s.step_id for s in plan]
    assert ids[:3] == ["stop_powered", "baseline_z", "withdraw_initial"]
    # 111 → rungs [1,10,100] → 3 retract steps
    assert ids[3:] == ["retract_0_1", "retract_1_10", "retract_2_100"]
    assert all(not s.optional for s in plan)


# ── normal receding retract ─────────────────────────────────────────────────
def test_normal_retract_succeeds():
    skill = _mk(retract_total_steps=111, retract_motor_dir="z+")
    ctx = FakeCtx(mode="receding")
    result = skill.execute(ctx, {})
    assert result.success, f"unexpected failure: {result.error}"
    assert result.data["retracted"] is True
    assert result.data["total_steps_retracted"] == 111
    assert result.data["recede_confirmed_rungs"] == 3
    # coarse steps used direction code 4 (z+)
    moves = [c for c in ctx.calls if c[0] == "Motor_StartMove"]
    assert moves and all(m[1][0] == 4 for m in moves)
    assert [m[1][1] for m in moves] == [1, 10, 100]   # ladder step counts


def test_direction_code_follows_config():
    skill = _mk(retract_total_steps=11, retract_motor_dir="z-")
    ctx = FakeCtx(mode="receding")
    skill.execute(ctx, {})
    moves = [c for c in ctx.calls if c[0] == "Motor_StartMove"]
    assert moves and all(m[1][0] == 5 for m in moves)   # z- → 5


def test_total_steps_param_overrides_config():
    skill = _mk(retract_total_steps=99999)
    ctx = FakeCtx(mode="receding")
    result = skill.execute(ctx, {"total_steps": 11})   # → [1, 10]
    assert result.success
    assert result.data["total_steps_retracted"] == 11


# ── backwards direction caught on the first rung ────────────────────────────
def test_backwards_direction_caught_on_first_step():
    skill = _mk(retract_total_steps=3000)
    ctx = FakeCtx(mode="approaching")     # Z retracts on the verify → approaching
    result = skill.execute(ctx, {})
    assert not result.success
    assert "逼近" in (result.error or "")
    # only the FIRST rung (1 step) was attempted before aborting
    moves = [c for c in ctx.calls if c[0] == "Motor_StartMove"]
    assert len(moves) == 1 and moves[0][1][1] == 1
    # emergency cleanup ran: motor stopped + withdrawn
    assert "Motor_StopMove" in _methods(ctx)
    assert "ZCtrl_Withdraw" in _methods(ctx)
    assert result.data["total_steps_retracted"] == 0


def test_reversing_the_physical_direction_reverses_the_verdict():
    """With configuration and ladder unchanged, reversing the synthetic physical response must reverse the direction verdict."""
    verdicts = {}
    for mode in ("receding", "approaching"):
        skill = _mk(retract_total_steps=11, retract_motor_dir="z+")
        ctx = FakeCtx(mode=mode)
        result = skill.execute(ctx, {})
        rungs = result.data.get("rungs") or []
        assert rungs, f"no rung was judged with mode={mode}"
        verdicts[mode] = rungs[0]["verdict"]
        assert verdicts[mode] == mode, (
            f"rig moving {mode} judged {verdicts[mode]!r} ({rungs[0]['reason']})")
    assert verdicts["receding"] != verdicts["approaching"]


def test_a_loop_that_never_settles_refuses_instead_of_retracting_blind():
    """No settled baseline ⇒ no direction check ⇒ no 3000 coarse steps.

    This used to degrade to "|I| ≈ 0, call it receding" — an answer equally true
    of a receding tip, an approaching tip still out of range, and a dead preamp.
    """
    skill = _mk(retract_total_steps=3000)
    skill._poll_interval_s = 0.005
    skill._settle_timeout_s = 0.08
    ctx = FakeCtx(mode="receding", travel_per_poll=0.0)   # loop never arrives
    result = skill.execute(ctx, {})
    assert not result.success
    assert "无法建立基线" in (result.error or "")
    assert "z_settle_timeout_s" in (result.error or "")
    assert not [c for c in ctx.calls if c[0] == "Motor_StartMove"], (
        "not one coarse step may be taken without a baseline to judge against"
    )


def test_current_spike_trips_the_check():
    skill = _mk(retract_total_steps=3000)
    # Z inconclusive but a big current during the self-check ⇒ approaching.
    ctx = FakeCtx(mode="ambiguous", verify_current=1e-8)
    result = skill.execute(ctx, {})
    assert not result.success
    assert "逼近" in (result.error or "")
    assert "Motor_StopMove" in _methods(ctx)


# ── stop powered first ──────────────────────────────────────────────────────
def test_stop_powered_halts_approach_and_motor():
    skill = _mk(retract_total_steps=1)
    ctx = FakeCtx(mode="receding")
    skill.execute(ctx, {})
    # first two calls are the powered-stop pair
    assert ("AutoApproach_OnOffSet", (0,)) in ctx.calls
    assert ("Motor_StopMove", ()) in ctx.calls


# ── abort mid-check ─────────────────────────────────────────────────────────
def test_abort_stops_and_withdraws():
    skill = _mk(retract_total_steps=3000)
    ctx = FakeCtx(mode="receding", abort_after=1)
    result = skill.execute(ctx, {})
    assert not result.success


def test_abort_during_the_settle_is_not_a_verdict():
    """The settle polls for seconds; an abort inside it must not become an answer.

    Landing the abort mid-ramp is the one moment a careless implementation would
    hand back whatever Z it was holding. It has to come out as an abort, with no
    rung verdict recorded and the motor stopped."""
    skill = _mk(retract_total_steps=3000)
    # Past the executor's own pre-step abort checks, into the baseline poll loop.
    ctx = FakeCtx(mode="receding", abort_after=6)
    result = skill.execute(ctx, {})
    assert not result.success
    assert not (result.data.get("rungs") or []), (
        "an aborted settle must not leave a judged rung behind"
    )


# ── dI/dV read when configured ──────────────────────────────────────────────
def test_didv_read_when_signal_index_configured():
    skill = _mk(retract_total_steps=1, lockin_signal_index=14)
    ctx = FakeCtx(mode="receding")
    result = skill.execute(ctx, {})
    assert result.success
    assert "Signals_ValGet" in _methods(ctx)
    assert result.data["rungs"][0]["didv_v"] == pytest.approx(2e-4)


def test_no_didv_read_when_not_configured():
    skill = _mk(retract_total_steps=1)   # no lockin_signal_index
    ctx = FakeCtx(mode="receding")
    skill.execute(ctx, {})
    assert "Signals_ValGet" not in _methods(ctx)


def test_registered_via_discover():
    """Auto-discovered by registry.discover() (via composite/__init__ import) so
    the agent can actually call it."""
    from mast.core.registry import SkillRegistry

    r = SkillRegistry()
    r.discover("mast.skills.composite")     # scan just the composite package
    assert r.has("RetractForSampleChange")
    meta = r.get("RetractForSampleChange")().metadata()
    assert meta.safety_level == SafetyLevel.CONFIRM


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
