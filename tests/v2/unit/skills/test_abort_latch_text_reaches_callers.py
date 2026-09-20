"""2026-07-28 — the #46 fix landed on 2 of 27 readers of ``aborted_reason``.

#46 was an abort wrongly attributed to the operator, and the fix added
``abort_error_text(progress)``, which translates the abort-latch sentinel into
"aborted by user" and passes every other cause through verbatim. It was wired
into WaitScanComplete and RunGridExperiment.

But ``aborted_reason`` is read DIRECTLY — as ``progress.aborted_reason or
"<skill> aborted"`` — in 25 other places, including
``CompositeSkill._decide_outcome``, which is the DEFAULT outcome path every
composite that does not hand-roll ``run_composite`` goes through. Those readers
never call the translator, so pressing 中止 reported the raw sentinel:

    error = "external abort flag before step"

That is executor jargon naming an internal flag, handed to the operator and to
the agent. It is also the precise ingredient of the #46 aftermath the original
docstring describes — the agent "spent the run trying to get the operator to
release the abort latch that did not exist". A string that names an *external
abort flag* is an invitation to exactly that hallucination.

The two sibling latch paths in graph_executor already wrote the human string
"aborted by operator"; only the pre-step check wrote the sentinel. So the fix is
at the write site: store text that is safe to render, because 25 call sites
render it directly. ``abort_error_text`` still translates the LEGACY sentinel so
a sidecar written by an older build (a run resumed across an upgrade) does not
regress.

## The existing test was checking the door that was already shut

``test_abort_reason_attribution.py::test_the_internal_latch_wording_never_reaches
_the_caller`` asserts the sentinel does not survive ``abort_error_text`` — true,
and true before this change. It says the sentinel "must be translated" while 25
readers never translate it. This file checks the other 25 doors.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_abort_latch_text_reaches_callers.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
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

import threading  # noqa: E402

import pytest  # noqa: E402

from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite.graph_executor import (  # noqa: E402
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    _ABORT_LATCH_REASON,
    _USER_ABORT_TEXT,
    abort_error_text,
)

#: The jargon that must never reach a caller through ANY door.
_JARGON = "external abort flag before step"


class _AbortedCtx:
    """Context whose abort latch is already set — i.e. the operator pressed 中止
    (or E_STOP fired) before the executor reached its next step."""

    def __init__(self, run_id: str = "r1"):
        self.run_id = run_id
        self._ev = threading.Event()
        self._ev.set()
        self.calls: list[str] = []

    def check_abort(self) -> bool:
        return self._ev.is_set()

    def run(self, skill_name, params=None, **kw):  # pragma: no cover - never reached
        self.calls.append(skill_name)
        return SkillResult(skill_name=skill_name, success=True, data={})


def _plan():
    return iter([CompositeStep(step_id="s1", skill_name="GetBias", params={})])


@pytest.fixture()
def aborted_progress() -> CompositeProgress:
    """Run a real executor into the latch and hand back the progress it wrote."""
    ctx = _AbortedCtx()
    ex = GraphExecutor(composite_name="ProbeComposite", context=ctx)
    ok = ex.run_plan(_plan())
    assert ok is False, "the latch must stop the plan"
    assert ex.progress.aborted is True
    assert ctx.calls == [], "no step may run after the latch"
    return ex.progress


# ── the 25 doors: readers that never call the translator ─────────────────────


def test_the_stored_reason_is_operator_facing(aborted_progress) -> None:
    """THE REGRESSION. Whatever the latch stores is rendered verbatim by 25 call
    sites, so it has to be readable on its own."""
    assert _JARGON not in aborted_progress.aborted_reason, (
        "the abort latch still stores executor jargon; every "
        "`aborted_reason or \"<skill> aborted\"` reader will print it")
    assert aborted_progress.aborted_reason == _USER_ABORT_TEXT


def test_the_direct_reader_idiom_reports_a_user_abort(aborted_progress) -> None:
    """The literal expression used at full_scan.py:345, condition_tip.py:447,
    prescan_check.py:211, survey_surface.py:490 … (25 sites)."""
    rendered = aborted_progress.aborted_reason or "scan aborted"
    assert rendered == "aborted by user"
    assert _JARGON not in rendered


def test_the_default_composite_outcome_path_is_clean(aborted_progress) -> None:
    """``CompositeSkill._decide_outcome`` is what EVERY composite that does not
    hand-roll run_composite ends up in — the single highest-traffic door."""
    from mast.skills.composite._base import CompositeSkillGraph

    ok, err = CompositeSkillGraph._decide_outcome(
        None, False, aborted_progress, {})  # type: ignore[arg-type]
    assert ok is False
    assert err == "aborted by user"
    assert _JARGON not in err


def test_the_translator_still_agrees(aborted_progress) -> None:
    """The 2 call sites that DO translate must not change behaviour."""
    assert abort_error_text(aborted_progress) == "aborted by user"


# ── the properties the fix must not break ────────────────────────────────────


def test_legacy_sidecar_sentinel_is_still_translated() -> None:
    """A run resumed across an upgrade carries the OLD sentinel in its sidecar.
    ``abort_error_text`` must keep translating it rather than leaking jargon."""
    p = CompositeProgress(composite_name="X")
    p.aborted, p.aborted_reason = True, _ABORT_LATCH_REASON
    assert abort_error_text(p) == "aborted by user"
    assert _JARGON not in abort_error_text(p)


def test_a_non_latch_cause_still_keeps_its_own_words() -> None:
    """The whole point of #46: only the latch may be called a user abort. A tip
    halt / failed mandatory step must survive verbatim through BOTH doors."""
    halt = ("tip_quality_drop: 扫描中途针尖变化（已采集 71 行中第 ~9 行）"
            "—— 已在当前步骤边界停止本流程，未回滚；修针尖后可重跑。")
    p = CompositeProgress(composite_name="WaitScanComplete")
    p.aborted, p.aborted_reason = True, halt

    assert abort_error_text(p) == halt                      # translator door
    assert (p.aborted_reason or "scan aborted") == halt      # direct door
    assert "aborted by user" not in p.aborted_reason, "又在冒充用户"


def test_an_empty_reason_is_not_invented_by_the_direct_reader() -> None:
    """An empty reason keeps each skill's own fallback on the direct path (the
    translator's blanket "aborted by user" is deliberate only for ITS callers)."""
    p = CompositeProgress(composite_name="X")
    p.aborted, p.aborted_reason = True, ""
    assert (p.aborted_reason or "scan aborted") == "scan aborted"
    assert abort_error_text(p) == "aborted by user"
