"""The coarse stepper's drive voltage is the operator's parameter. Four locks.

Why this file is long: the thing being protected cannot be un-broken. A bias that
is too high makes a bad scan; a drive amplitude that is too high delaminates the
piezo stack, and no software layer above it gets another try. There is also no
reading that says which rig you are on — the controller will output 400 V on a
stack that dies at 300 — so the ONLY protection is a number a human wrote down.

Every test here is about a failure mode that would otherwise look like success:
a silently clamped value, a default standing in for an undeclared limit, an
unreadable drive treated as a fine one, a second code path reaching the setter.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mast.core import coarse_drive
from mast.core.safety import (
    is_coarse_drive_change,
    is_unguarded_lateral_coarse_move,
)


@pytest.fixture(autouse=True)
def _clean_declaration():
    """Every test starts undeclared, and leaves nothing behind."""
    before = coarse_drive.get_declaration()
    coarse_drive.set_declaration({})
    coarse_drive.set_persist_sink(None)
    yield
    coarse_drive.set_declaration(before)


# ── Lock 3: the rig ceiling ─────────────────────────────────────────────────

def test_undeclared_refuses_everything():
    """Silence is not consent.

    Not knowing what this stack tolerates is a reason to do nothing — not a
    reason to fall back to a number someone chose for a different instrument."""
    for amp in (0.0, 1.0, 50.0, 150.0, 399.0):
        ok, reason = coarse_drive.authorize(amp, 200.0)
        assert not ok, f"{amp} V was authorised with no declaration on file"
        assert "未声明" in reason


def test_within_the_declared_ceiling_is_allowed():
    coarse_drive.declare(300.0)
    ok, reason = coarse_drive.authorize(250.0, 200.0)
    assert ok, reason
    ok, _ = coarse_drive.authorize(300.0, 200.0)
    assert ok, "the declared ceiling itself is allowed"


def test_above_the_ceiling_is_REFUSED_not_clamped():
    """The single most important behaviour in this module.

    Clamping would turn "300 V would destroy this stack" into "ran at 300 V"
    while the caller believes it asked for 400 — a wrong action reported as a
    success, and the operator never learns the declaration was in the way."""
    coarse_drive.declare(300.0)
    ok, reason = coarse_drive.authorize(380.0, 200.0)
    assert not ok
    assert "拒绝" in reason
    assert "300" in reason and "380" in reason, (
        "the refusal must name both numbers so the operator can act on it"
    )


def test_a_declaration_does_not_mutate_the_request():
    """No API here returns a 'corrected' amplitude — there is nothing to clamp with."""
    coarse_drive.declare(150.0)
    ok, _ = coarse_drive.authorize(400.0)
    assert ok is False
    assert coarse_drive.max_amplitude_v() == 150.0, "the ceiling is not moved by a request"


# ── Lock 4: the absolute ceiling ────────────────────────────────────────────

def test_absolute_ceiling_beats_any_declaration():
    """No configuration widens this one — same character as _PHYSICAL_ABSURD."""
    coarse_drive.set_declaration({"max_amplitude_v": 1e6})   # sanitize drops it
    assert coarse_drive.max_amplitude_v() is None, (
        "an out-of-range declaration is DROPPED, not clamped into range — a claim "
        "we had to correct is not a claim to act on"
    )
    coarse_drive.declare(coarse_drive.ABSOLUTE_MAX_AMPLITUDE_V)
    ok, reason = coarse_drive.authorize(coarse_drive.ABSOLUTE_MAX_AMPLITUDE_V + 1.0)
    assert not ok and "绝对上限" in reason


def test_absurd_magnitudes_are_named_as_such():
    """4000 V is a units error, not an aggressive setting; say so."""
    coarse_drive.declare(300.0)
    ok, reason = coarse_drive.authorize(4000.0)
    assert not ok
    assert "数量级" in reason or "单位" in reason


@pytest.mark.parametrize("bad", [None, "", "abc", float("nan"), float("inf"), -1.0])
def test_non_numeric_and_negative_are_refused(bad):
    coarse_drive.declare(300.0)
    ok, _ = coarse_drive.authorize(bad)
    assert not ok


def test_frequency_is_bounded_too():
    coarse_drive.declare(300.0)
    ok, _ = coarse_drive.authorize(100.0, coarse_drive.ABSOLUTE_MAX_FREQUENCY_HZ + 1)
    assert not ok


# ── The pre-move readback check ─────────────────────────────────────────────

def test_unreadable_drive_refuses_the_move():
    """A check whose failure mode is 'pass' is not a check.

    The declaration constrains what MAST writes; it says nothing about what the
    drive is set to right now. Somebody may have raised it in the Nanonis UI."""
    coarse_drive.declare(300.0)
    for unreadable in (None, "", "n/a", float("nan")):
        ok, reason = coarse_drive.readback_matches(unreadable)
        assert not ok, f"{unreadable!r} was treated as an acceptable readback"
        assert "读不到" in reason or "不是有限数值" in reason


def test_readback_above_the_ceiling_refuses_the_move():
    coarse_drive.declare(200.0)
    ok, reason = coarse_drive.readback_matches(280.0)
    assert not ok
    assert "拒绝粗动" in reason


def test_readback_within_tolerance_passes():
    coarse_drive.declare(200.0)
    ok, _ = coarse_drive.readback_matches(200.0 * (1.0 + coarse_drive.READBACK_REL_TOL / 2))
    assert ok, "a DAC-quantisation difference must not block work"


def test_readback_with_no_declaration_refuses():
    ok, reason = coarse_drive.readback_matches(100.0)
    assert not ok and "未声明" in reason


def test_frequency_mismatch_warns_but_does_not_block():
    """A wrong frequency changes how far a step travels — it does not kill the
    stack. Blocking on it would trade a real capability for a bookkeeping worry,
    so it rides back as a warning attached to a pass."""
    coarse_drive.declare(200.0, expected_frequency_hz=1000.0)
    ok, reason = coarse_drive.readback_matches(150.0, 300.0)
    assert ok
    assert "里程表" in reason or "不符" in reason


# ── Lock 2: the Layer-0 predicates ──────────────────────────────────────────

def test_predicate_catches_the_setter_regardless_of_arguments():
    """Value-independent on purpose: there is no amplitude that is safe to set
    without knowing the rig, and knowing the rig is not something the caller
    can demonstrate through an argument."""
    assert is_coarse_drive_change("SetMotorFreqAmp", {})
    assert is_coarse_drive_change("SetMotorFreqAmp", {"amplitude_v": 0.0})
    assert not is_coarse_drive_change("GetMotorFreqAmp", {"axis": "all"})
    assert not is_coarse_drive_change("MotorMove", {"direction": "x+", "steps": 1})


@pytest.mark.parametrize("direction", ["x+", "x-", "y+", "y-", "X+", " y- "])
def test_bare_lateral_moves_are_flagged(direction):
    assert is_unguarded_lateral_coarse_move("MotorMove", {"direction": direction})


@pytest.mark.parametrize("direction", ["z-retract", "z-approach"])
def test_z_moves_are_not_flagged_by_the_lateral_predicate(direction):
    """Retract is safe; approach has its own, older gate. Double-gating approach
    here would only make the refusal message less specific."""
    assert not is_unguarded_lateral_coarse_move("MotorMove", {"direction": direction})


def test_the_guarded_composite_is_not_flagged():
    """RelocateCoarseXY must stay autonomous — it is the whole point.

    It also never reaches this predicate in practice: it issues Motor_StartMove
    through safe_call, which carries no skill name."""
    assert not is_unguarded_lateral_coarse_move("RelocateCoarseXY",
                                                {"axis": "x", "direction": "+"})


# ── Lock 1 + the static guarantee ───────────────────────────────────────────

def test_the_capability_gate_hides_the_setter_by_default():
    from mast.skills.advanced_capabilities import (
        CAPABILITY_BY_ID,
        disabled_skill_names,
        set_enabled,
    )

    set_enabled([])   # default state
    try:
        cap = CAPABILITY_BY_ID["coarse_drive"]
        assert cap.default_on is False
        assert "SetMotorFreqAmp" in cap.skills
        assert "SetMotorFreqAmp" in disabled_skill_names(), (
            "with the capability off the tool must not be wrapped at all — the "
            "agent cannot call what it cannot see"
        )
        assert "GetMotorFreqAmp" not in disabled_skill_names(), (
            "reading the drive is always allowed; it is how a move verifies itself"
        )
    finally:
        set_enabled([])


def test_the_capability_switch_is_admin_pin_guarded():
    from mast.api.admin_pin import GUARDED_KEYS

    assert "advanced_capabilities" in GUARDED_KEYS
    assert "coarse_drive" in GUARDED_KEYS, (
        "the declaration itself must not be writable without the PIN"
    )


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2" / "mast").is_dir():
            return p
        p = p.parent
    pytest.skip("repo root not found")


def test_only_one_place_in_the_whole_skill_tree_writes_the_drive():
    """A second call site would be a fourth lock that nobody installed.

    This is checkable at all because ``safe_call`` verbs must be string literals
    (test_safe_call_verbs_are_literal) — so a static walk really does see every
    call. If this fails, do not add the verb to the allow-list: route the new
    caller through SetMotorFreqAmp so it inherits authorize()."""
    root = _repo_root() / "MASTv2" / "mast"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name not in ("safe_call", "run"):
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "Motor_FreqAmpSet":
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    outside = [o for o in offenders
               if not o.replace("\\", "/").startswith("skills/builtins/motor.py")]
    assert not outside, (
        f"Motor_FreqAmpSet is written from {outside} — it must be reachable only "
        f"from SetMotorFreqAmp.execute in skills/builtins/motor.py, which is the "
        f"only place coarse_drive.authorize() runs. Route the new caller through "
        f"the skill instead of adding it here."
    )
    assert offenders, (
        "no call site found at all — either the verb was renamed (update this "
        "test with it) or the setter lost its implementation"
    )


def test_setter_refuses_before_touching_the_hardware():
    """The refusal must happen BEFORE the write, not be caught after it."""
    from mast.skills.builtins.motor import SetMotorFreqAmp

    calls: list[tuple] = []

    class _Ctx:
        def safe_call(self, verb, *args):
            calls.append((verb, args))
            raise AssertionError("hardware was touched despite an unauthorised request")

    res = SetMotorFreqAmp().execute(_Ctx(), {"frequency_hz": 1000.0,
                                             "amplitude_v": 380.0,
                                             "axis": "all"})
    assert res.success is False
    assert not calls, "no Nanonis call may be issued for a refused amplitude"
    assert "未声明" in (res.error or "")
