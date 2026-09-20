"""Backend failures must not be recorded as operator aborts.

The persistent terminal label must retain the distinction between completion,
exceptions, unanswered approval and an actual stop request. A live error frame
alone cannot make that cause recoverable when the conversation is replayed.
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path


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

from mast.api.routes.orchestrator import _terminal_label  # noqa: E402


def test_completed_run_says_done():
    assert _terminal_label(True, None) == "完成"


def test_completed_wins_even_if_a_reason_lingers():
    """A stale stop_reason must not turn a finished run into a failure."""
    assert _terminal_label(True, "ValueError: x") == "完成"


def test_genuine_operator_abort_still_says_aborted():
    """The one case where "已中止" is the truth must keep saying exactly that —
    otherwise this fix just moves the lie."""
    assert _terminal_label(False, None) == "已中止"
    assert _terminal_label(False, "") == "已中止"


@pytest.mark.parametrize("reason", [
    "ValueError: zero-size array to reduction operation fmin which has no identity",
    "IndexError: list index out of range",
    "等待人工审批未得到答复（BiasPulse — 针尖修复）",
    "内部错误：无法解析审批请求（__interrupt__）",
])
def test_a_crash_is_not_reported_as_an_abort(reason):
    label = _terminal_label(False, reason)
    assert "已中止" not in label, f"crash still labelled as an abort: {label!r}"
    assert "运行出错" in label


def test_the_actual_reason_survives_into_the_label():
    """The point of the fix: a replay must be able to recover WHY."""
    label = _terminal_label(
        False, "ValueError: zero-size array to reduction operation fmin")
    assert "fmin" in label, "the real cause did not survive into the record"


def test_label_is_bounded_and_single_line():
    """The row is a marker, not a log dump — a 10 KB traceback must not land in
    the transcript, and a newline would break single-row rendering."""
    label = _terminal_label(False, "Traceback:\n" + "x" * 5000)
    assert len(label) < 300
    assert "\n" not in label
    assert label.endswith("…")


def test_multiline_reason_is_flattened():
    label = _terminal_label(False, "line one\n  line two\tand three")
    assert "\n" not in label and "\t" not in label
    assert "line one line two and three" in label


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
