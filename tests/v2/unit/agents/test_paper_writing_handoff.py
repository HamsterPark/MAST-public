"""Who decides whether a report gets reviewed — and where that decision lives.

Measured 2026-07-28, three real end-to-end runs. The goal asked only for
「写一份简短的实验报告并导出」, and paper_review ran every time. Two rounds of
edits to the ORCHESTRATOR's router prompt changed nothing:

  round 1  prose: 「审稿是可选的，只有用户明确要了才派 paper_review」  → PR ran
  round 2  the section's default-shape diagram (it still listed 审稿)      → PR ran

Both edits were to a prompt that is **not consulted for this hop**. paper_writing
hands off directly (``handoff_to_paper_review``), and a handoff hint dispatches
deterministically in ``supervisor_node`` — the LLM router only runs when there is
no hint. Its own checklist said, unconditionally:

    6. Hand off to paper_review, quoting the exact file path save_draft returned

and its Handoff section said "hand off to paper_review again" after a revision,
directly contradicting the router's 「改完不再复审」.

So the rule has to live where the decision is made. This pins that:

* review is CONDITIONAL on the operator asking, and the alternative
  (handoff_to_supervisor) is named — a rule with no stated alternative gets
  ignored under the pull of the checklist;
* a revision is NOT re-reviewed;
* the file says WHY it, and not the orchestrator, owns this — otherwise the next
  person to see a stray review round edits the router prompt again.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_paper_writing_handoff.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.agents.paper_writing.prompts import SYSTEM_PROMPT as PW  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# The decision itself
# ════════════════════════════════════════════════════════════════════════════

def test_review_is_conditional_on_the_operator_asking():
    assert "用户要了审稿" in PW or "用户没要审稿" in PW
    assert "handoff_to_supervisor" in PW, (
        "只说了「别送审」而没说该送给谁 —— 没有替代路径的规则会被忽略")


def test_the_unconditional_hand_off_instruction_is_gone():
    """The line that actually drove three runs' worth of unwanted reviews."""
    joined = " ".join(PW.split())
    assert "Hand off to paper_review, quoting the exact file path" not in joined
    assert "After all sections are drafted: hand off to paper_review" not in joined


def test_a_revision_is_not_re_reviewed():
    assert "不要把改完的稿子再送回 paper_review" in PW
    # And the instruction that said the opposite must be gone.
    assert "hand off to paper_review again" not in " ".join(PW.split())


def test_the_draft_path_is_still_quoted_on_handoff():
    """The fix must not lose what the old line got right: whoever receives the
    handoff needs the real file path, not a name to reconstruct."""
    assert "save_draft 返回的真实文件路径" in PW or "路径是什么" in PW


# ════════════════════════════════════════════════════════════════════════════
# Where the rule lives — and why it is here
# ════════════════════════════════════════════════════════════════════════════

def test_the_file_explains_why_this_decision_is_not_the_orchestrators():
    """Without this note the next stray review round gets 'fixed' in the router
    prompt again — which is exactly what happened twice."""
    assert "编排器" in PW
    assert "handoff" in PW.lower()


def test_the_orchestrator_still_states_the_same_rule():
    """The two prompts must not disagree. The router does not decide this hop,
    but it does decide whether to dispatch paper_review on its own."""
    from mast.agents.orchestrator.graph import _ROUTER_PROMPT

    assert "审稿是可选的" in _ROUTER_PROMPT
    assert "改完不再复审" in _ROUTER_PROMPT


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
