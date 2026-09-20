"""The post-abort hardware policy, checked against the REAL Nanonis API.

``test_abort_gate.py`` pins the policy on hand-picked examples. Examples are how
the policy got two of its entries WRONG: the first draft of ``_ABORT_SAFE_WRITES``
listed ``BiasSpectrMLS_Stop`` and ``GenSweep_Stop``, and **neither verb exists**
(the MLS mode is stopped by ``BiasSpectr_Stop``; the generic sweep is
``GenSwp_Stop``). Two stop paths I believed were open were silently closed, and
no example-based test could see it — the entries simply never matched anything.

These are property tests over the ground truth instead:

  * the LIVE ``nanonis_spm`` API (684 verbs) — every allow-list entry must be a
    real method, and the read heuristic must agree with Nanonis's own naming
    convention on ALL of them;
  * the verbs MAST's skills ACTUALLY call (grepped from the source) — an entry
    no skill can emit cannot help an abort, and the skills' own abort-cleanup
    handlers must not be refused by the very gate that fired them;
  * the real signature of each overloaded verb — a rule keyed to the wrong
    argument index is a hardware-safety bug in the worst direction.

The verbs MAST calls are grepped rather than imported because the gate keys on
the STRING passed to ``safe_call`` — a typo there is exactly the failure mode
this file exists to catch, so reading the strings back is the honest check.
"""
from __future__ import annotations

import inspect
import re
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

import importlib  # noqa: E402

import pytest  # noqa: E402

from mast.core.execution_context import (  # noqa: E402
    _ABORT_SAFE_WRITES,
    _is_abort_safe,
    _is_read,
)

_SKILLS_DIR = Path(_MASTV2_ROOT) / "mast"
# BOTH entry points onto the link: ``safe_call`` and the emergency-path
# ``urgent_call`` (bounded role-lock wait + force-unstick, added 2026-07-28).
# The E-STOP's four verbs go through the latter, and an abort-policy checker
# that could not see them would be checking the wrong set — those are exactly
# the writes that MUST stay permitted once an abort is latched.
_SAFE_CALL_RE = re.compile(
    r'(?:safe_call|urgent_call|_urgent)\(\s*["\']([A-Za-z0-9_]+)["\']')


def _import_real_nanonis():
    """Import the REAL nanonis_spm, around the suite-wide mock.

    ``tests/conftest.py`` replaces ``nanonis_spm`` with a MagicMock so the suite
    runs with no instrument attached — and a MagicMock answers ``hasattr`` for
    *every* name, which would make every property here vacuously true. Evict the
    mock, import the package for real, then put the mock back so the rest of the
    session is unaffected.
    """
    saved = {k: v for k, v in sys.modules.items()
             if k == "nanonis_spm" or k.startswith("nanonis_spm.")}
    for k in saved:
        del sys.modules[k]
    try:
        return importlib.import_module("nanonis_spm")
    finally:
        for k in [k for k in sys.modules
                  if k == "nanonis_spm" or k.startswith("nanonis_spm.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _nanonis_api() -> dict[str, inspect.Signature]:
    """Every public method of the real nanonis_spm client, with its signature."""
    real = _import_real_nanonis()
    out: dict[str, inspect.Signature] = {}
    for name in dir(real.Nanonis):
        if name.startswith("_"):
            continue
        fn = getattr(real.Nanonis, name)
        if not callable(fn):
            continue
        try:
            out[name] = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover — C-level callables
            continue
    return out


def _verbs_mast_calls() -> set[str]:
    """Every Nanonis verb string that appears in a ``safe_call(...)`` in MAST."""
    verbs: set[str] = set()
    for p in _SKILLS_DIR.rglob("*.py"):
        verbs |= set(_SAFE_CALL_RE.findall(
            p.read_text(encoding="utf-8", errors="replace")))
    return verbs


API = _nanonis_api()
MAST_VERBS = _verbs_mast_calls()


def test_the_ground_truth_actually_loaded():
    """A vacuous property test is worse than none — pin the domains."""
    assert len(API) > 600, f"nanonis_spm API failed to load ({len(API)} verbs)"
    assert len(MAST_VERBS) > 200, f"only {len(MAST_VERBS)} safe_call verbs found"


# ════════════════════════════════════════════════════════════════════════
# The read heuristic vs. Nanonis's own naming convention — over all 684
# ════════════════════════════════════════════════════════════════════════

def _is_read_by_convention(method: str) -> bool:
    """Nanonis names every read with a ``Get`` suffix (or ``*Status`` / ``Read*``)."""
    return method.endswith("Get") or method.endswith("Status") or "Read" in method


def test_read_heuristic_agrees_with_the_naming_convention_on_every_verb():
    """``_is_read`` is a lowercase SUBSTRING test ("get" in name). That is a trap
    waiting for a verb like ``XY_TargetSet`` — "tar-GET" — which would be waved
    through as a read and could then WRITE while an abort is latched.

    Today no such verb exists (296 reads by either rule, exactly the same 296).
    This test is what makes that a fact rather than an assumption: the day
    nanonis_spm ships one, a test fails instead of the tip moving.
    """
    heuristic = {m for m in API if _is_read(m)}
    convention = {m for m in API if _is_read_by_convention(m)}
    wrongly_allowed = sorted(heuristic - convention)
    wrongly_blocked = sorted(convention - heuristic)
    assert not wrongly_allowed, (
        "these WRITE verbs would be treated as reads and allowed to run after an "
        f"abort: {wrongly_allowed}")
    assert not wrongly_blocked, (
        f"these reads would be refused during a stop sequence: {wrongly_blocked}")


# ════════════════════════════════════════════════════════════════════════
# The allow-list must be REAL — the bug that started this file
# ════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("verb", sorted(_ABORT_SAFE_WRITES))
def test_every_allow_listed_verb_is_a_real_nanonis_method(verb: str):
    """A typo here is invisible: the entry simply never matches, so the stop it
    was meant to permit is refused — and the operator's abort leaves the hardware
    running. ``GenSweep_Stop`` (real name: ``GenSwp_Stop``) did exactly this."""
    assert verb in API, (
        f"{verb!r} is not a nanonis_spm method — the stop path it claims to open "
        f"does not exist")


@pytest.mark.parametrize("verb", sorted(_ABORT_SAFE_WRITES))
def test_every_allow_listed_verb_is_actually_called_by_a_skill(verb: str):
    """An allow-list entry no skill can emit cannot help an abort — it only makes
    the policy look more complete than it is (``BiasSpectrMLS_Stop``)."""
    assert verb in MAST_VERBS, (
        f"no MAST skill ever calls {verb!r} — either wire the stop or drop the "
        f"entry; a phantom entry is a stop path that does not exist")


@pytest.mark.parametrize("verb,rule", sorted(
    (v, r) for v, r in _ABORT_SAFE_WRITES.items() if r is not None))
def test_conditional_rules_point_at_a_real_argument(verb: str, rule):
    """``Scan_Action(action, direction)``: a rule keyed to the DIRECTION argument
    would have let a post-abort ``Scan_Action(0, …)`` start a fresh scan. Pin each
    index against the verb's real signature."""
    idx, _values = rule
    params = [p for p in API[verb].parameters if p != "self"]
    assert 0 <= idx < len(params), (
        f"{verb} rule pins arg #{idx}, but its signature is {tuple(params)}")


# ════════════════════════════════════════════════════════════════════════
# Nothing that STARTS may survive an abort — fuzzed over the real API
# ════════════════════════════════════════════════════════════════════════

_START_WORDS = ("Start", "Run", "Open", "Pulse", "Sweep", "Move", "Approach")
_ARG_FUZZ = [(), (0,), (1,), (2,), (0, 0), (1, 0), (0, 1), (1, 1), (2, 1),
             (-1,), (1e-9,), ("x",), (0, 0, 0), (1, 1, 1)]


def _starts_something(verb: str) -> bool:
    if verb in _ABORT_SAFE_WRITES or _is_read(verb):
        return False
    return any(w in verb for w in _START_WORDS) and "Stop" not in verb


def test_no_start_verb_is_abort_safe_under_any_arguments():
    """Fuzz every start-shaped verb in the real API against every plausible arg
    tuple. Not one may pass the gate — a single leak is a scan, a sweep, or an
    approach restarting after the operator said stop."""
    offenders = []
    for verb in API:
        if not _starts_something(verb):
            continue
        for args in _ARG_FUZZ:
            if _is_abort_safe(verb, args):
                offenders.append((verb, args))
    assert not offenders, f"these would run after an abort: {offenders[:10]}"


def test_every_non_read_verb_outside_the_allow_list_is_refused():
    """Fail-closed over the WHOLE API, not just the start-shaped names: the gate's
    contract is an allow-list, so anything unlisted must be refused whatever it
    is called."""
    leaks = [v for v in API
             if not _is_read(v) and v not in _ABORT_SAFE_WRITES
             and any(_is_abort_safe(v, a) for a in _ARG_FUZZ)]
    assert not leaks, f"unlisted verbs slipped through the gate: {leaks}"


def test_every_read_verb_survives_an_abort():
    """"read anything" — the other half of the policy. Blocking a read would
    break the stop sequences themselves (they poll status to know they worked)."""
    blocked = [v for v in API if _is_read_by_convention(v)
               and not _is_abort_safe(v, ())]
    assert not blocked, f"reads refused after abort: {blocked}"


# ════════════════════════════════════════════════════════════════════════
# The overloaded verbs — exhaustive, because the meaning flips on one int
# ════════════════════════════════════════════════════════════════════════

class TestOverloadedVerbs:
    @pytest.mark.parametrize("direction", [0, 1])
    def test_scan_action_start_blocked_stop_and_pause_allowed(self, direction):
        # Scan_Action(Scan_action, Scan_direction): 0=START 1=STOP 2=PAUSE
        assert _is_abort_safe("Scan_Action", (0, direction)) is False
        assert _is_abort_safe("Scan_Action", (1, direction)) is True
        assert _is_abort_safe("Scan_Action", (2, direction)) is True

    def test_auto_approach_off_allowed_on_blocked(self):
        assert _is_abort_safe("AutoApproach_OnOffSet", (0,)) is True
        assert _is_abort_safe("AutoApproach_OnOffSet", (1,)) is False

    def test_pattern_pause_allowed_resume_blocked(self):
        # Pattern_ExpPause(Pause_Resume): 1 = Pause, 0 = RESUME.
        # Resuming the grid experiment the operator just aborted is the exact
        # inverse of what they asked for.
        assert _is_abort_safe("Pattern_ExpPause", (1,)) is True
        assert _is_abort_safe("Pattern_ExpPause", (0,)) is False

    @pytest.mark.parametrize("at_control", [0, 1, 2])
    def test_atom_track_off_allowed_on_blocked(self, at_control):
        # AtomTrack_CtrlSet(AT_control, Status): the tracker DRIVES the tip, so
        # Status=0 is a motion stop and Status=1 would (re)start it. The rule must
        # key on arg #1, not arg #0 (which only selects WHICH controller).
        assert _is_abort_safe("AtomTrack_CtrlSet", (at_control, 0)) is True
        assert _is_abort_safe("AtomTrack_CtrlSet", (at_control, 1)) is False


# ════════════════════════════════════════════════════════════════════════
# The skills' OWN abort-cleanup paths must pass the gate that fired them
# ════════════════════════════════════════════════════════════════════════

class TestSkillCleanupPathsSurviveTheirOwnAbort:
    """Several skills detect the abort themselves and then STOP the hardware:

        approach.py  _stop_module            → AutoApproach_OnOffSet(0)
        scan_utils.py WaitScanComplete       → Scan_Action(1, 0)
        pattern.py   _stop_pattern_experiment→ Pattern_ExpStop, else Pattern_ExpPause(1)
        folme.py     StopFolMe               → FolMe_Stop

    Every one of those calls goes through the SAME ``safe_call`` gate that the
    abort just armed. Refuse them and the abort does the opposite of its job: the
    grid experiment, the sweep, or the follow-me motion keeps running on the
    controller with nobody watching. (Pattern_ExpStop / Pattern_ExpPause / FolMe_Stop
    were all refused until 2026-07-11.)
    """

    @pytest.mark.parametrize("verb,args", [
        ("AutoApproach_OnOffSet", (0,)),      # approach.py::_stop_module
        ("Scan_Action", (1, 0)),              # scan_utils.py::WaitScanComplete
        ("Pattern_ExpStop", ()),              # pattern.py::_stop_pattern_experiment
        ("Pattern_ExpPause", (1,)),           # …its fallback
        ("FolMe_Stop", ()),                   # folme.py::StopFolMe
        ("Motor_StopMove", ()),
        ("ZCtrl_Withdraw", (1, -1)),          # EmergencyRetract
        ("BiasSpectr_Stop", ()),
        ("ZSpectr_Stop", ()),
        ("GenSwp_Stop", ()),
        ("PLLFreqSwp_Stop", (1,)),
    ])
    def test_cleanup_call_is_allowed(self, verb, args):
        assert _is_abort_safe(verb, args) is True, (
            f"{verb}{args} is a STOP a skill issues *because* of the abort — "
            f"refusing it leaves the hardware running")

    # Verbs whose NAME contains Stop/Withdraw but which CONFIGURE the stop rather
    # than performing it. These must stay OFF the allow-list, and the reason is not
    # pedantry:
    #
    #   ZCtrl_WithdrawRateSet sets how FAST a withdraw happens. An agent that could
    #   write it while an abort is latched could set the rate to 1e-9 m/s and the
    #   emergency retract would take a week — the abort would look honoured and the
    #   tip would still be in the surface. That is the same shape as the field bug
    #   where a skill "met" a current threshold by lowering the setpoint: gaming the
    #   check instead of meeting it. A stop must not be reconfigurable by the thing
    #   it is stopping.
    _CONFIGURES_THE_STOP_DOES_NOT_PERFORM_IT = {
        "ZCtrl_WithdrawRateSet",     # how fast to retract
        "ZCtrl_SwitchOffDelaySet",   # how long to average before switching off
    }

    def test_every_stop_verb_mast_calls_is_allow_listed(self):
        """Sweep MAST's whole call set for stop-shaped verbs and demand each one
        be reachable after an abort. This is the check that would have caught the
        missing FolMe_Stop / GenSwp_Stop / Pattern_ExpStop / PLLFreqSwp_Stop the
        day they were written, instead of on the instrument."""
        stops = {v for v in MAST_VERBS
                 if ("Stop" in v or "Withdraw" in v) and not _is_read(v)
                 and v not in self._CONFIGURES_THE_STOP_DOES_NOT_PERFORM_IT}
        assert stops, "no stop verbs found — the grep broke"
        missing = sorted(v for v in stops if v not in _ABORT_SAFE_WRITES)
        assert not missing, (
            f"these stop verbs are refused while an abort is latched: {missing}")

    def test_the_stop_configurators_are_NOT_abort_safe(self):
        """The inverse, pinned. Somebody reading the failure above could 'fix' it by
        adding ZCtrl_WithdrawRateSet to the allow-list — which would hand the agent a
        way to neuter the emergency retract from inside an abort. This makes that fix
        fail instead."""
        for verb in self._CONFIGURES_THE_STOP_DOES_NOT_PERFORM_IT:
            assert verb not in _ABORT_SAFE_WRITES, (
                f"{verb} 进了中止白名单。它不执行停止——它配置停止。"
                "中止期间能改退针速率，就等于中止期间能把紧急退针废掉。"
            )
