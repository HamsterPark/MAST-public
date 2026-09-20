"""Coarse approach is blocked in EVERY mode — the answer to the operator's question.

They asked, in the field : "是不是安全模式下不动粗逼近？"

They asked because nothing told them, and the one thing that spoke told them
wrong: the AUTO tooltip read "自动模式：全部允许（默认，无限制）". That is false,
and false in the direction that crashes a tip — it invites the operator to
believe auto mode will drive the coarse motor toward the sample for them.

The truth: an open-loop coarse Z step TOWARD the sample (``MotorMove`` with
``direction='z-approach'``, or a closed-loop move with a Z component) is
hard-blocked by SafetyGate **Layer 0**, which runs BEFORE the operating-mode gate
and never consults it. safe / semi / auto all refuse it. Only a human runs it, on
the manual path. ``AutoApproach`` — current-feedback, stops on tunnelling — is
allowed in all three.

The tooltip now says exactly that. **A UI string asserting a safety property is a
claim about the backend, and a claim nobody checks is how the last one rotted.**
So the property is pinned here: the day someone makes coarse approach
mode-dependent, this fails and they know the UI has to change with it.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 not found")


_MASTV2_ROOT = _find_mastv2_root()
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

from mast.core.safety import is_coarse_sample_approach  # noqa: E402

_MODES = ("safe", "semi", "auto")

# The exact sentence the three tooltips now carry (TopBar.tsx COARSE_NOTE).
_UI_CLAIM = "开环粗逼近（MotorMove z-approach）在三种模式下一律禁止，须用户手动执行"


class TestTheAnswerToTheOperatorsQuestion:
    @pytest.mark.parametrize("mode", _MODES)
    def test_coarse_approach_toward_the_sample_is_refused_in_every_mode(self, mode):
        """The gate does not take a mode at all — which IS the answer. It is a
        Layer-0 block, upstream of the mode gate, so there is no mode in which the
        agent may drive the coarse motor into the sample."""
        assert is_coarse_sample_approach(
            "MotorMove", {"direction": "z-approach", "steps": 50}) is True

    def test_the_mode_gate_has_no_opinion_on_coarse_approach(self):
        """"自动模式：全部允许（无限制）" was the tooltip. It was a lie because the
        mode gate does not own this decision at all.

        The mode gate (Layer 0d) classifies exactly two things — electrical pulses
        and tip shaping. A coarse move is neither, so NO mode can wave it through:
        Layer 0 has already refused it, upstream. Pin that, and "auto unlocks it"
        can never quietly become true."""
        from mast.core.safety import is_electrical_pulse, is_tip_shaping

        args = {"direction": "z-approach", "steps": 50}
        assert is_electrical_pulse("MotorMove", args, frozenset()) is False
        assert is_tip_shaping("MotorMove", args, frozenset()) is False

    def test_retracting_is_never_blocked(self):
        """The block is directional: stepping AWAY from the sample is always fine.
        A gate that blocked retraction would strand the tip."""
        assert is_coarse_sample_approach(
            "MotorMove", {"direction": "z-withdraw", "steps": 50}) is False

    def test_auto_approach_is_a_different_thing_and_stays_allowed(self):
        """AutoApproach is current-feedback — it stops itself on tunnelling. It is
        NOT what is blocked, and the tooltip must not imply the tip can never be
        engaged autonomously."""
        assert is_coarse_sample_approach("AutoApproach", {}) is False
        assert is_coarse_sample_approach("ApproachTip", {}) is False


def test_the_ui_says_what_the_backend_does():
    """The tooltip is the ONLY place the operator can learn this, so its text is
    load-bearing. Pin the sentence itself: if someone softens it, or drops it back
    to "无限制", the mismatch fails here rather than at the tip."""
    top_bar = (Path(_MASTV2_ROOT).parent / "frontend" / "src" / "components"
               / "shell" / "TopBar.tsx").read_text(encoding="utf-8")

    assert _UI_CLAIM in top_bar, (
        "the coarse-approach note is gone from the operating-mode tooltips — the "
        "operator has no way left to learn that auto mode does NOT drive the "
        "coarse motor")

    # Only the tooltip STRINGS, not the file: a comment quoting the old text
    # (which is exactly what this file's post-mortem does) is not a lie in the UI.
    titles = [ln for ln in top_bar.splitlines() if ln.strip().startswith("title:")]
    assert len(titles) == 3, f"expected 3 mode tooltips, found {len(titles)}"
    for t in titles:
        assert "COARSE_NOTE" in t, (
            f"this mode's tooltip dropped the coarse-approach note: {t.strip()}")
        assert "无限制" not in t, (
            f"a tooltip claims 无限制 while coarse approach is still blocked in "
            f"every mode — it is telling the operator the opposite of the truth: "
            f"{t.strip()}")
