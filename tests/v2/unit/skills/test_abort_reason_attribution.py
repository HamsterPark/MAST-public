"""A stop signal must not lie about where it came from.

``CompositeProgress.aborted`` is set by four different things — the operator's
中止 button, E_STOP, a CRITICAL tip-quality halt, and a mandatory step failing —
and the hand-rolled ``run_composite`` overrides collapsed all four into the
literal string ``"aborted by user"``. In the field that produced three
consecutive scans reported as operator aborts that the operator never made,
while a bad tip was the real cause. The agent, reading the same string, spent
the run asking the operator to "release the abort latch" instead of dealing
with the tip; the operator, reading it, could only conclude that something was
impersonating them.

What is pinned here:

* the ABORT LATCH — and only it — may be rendered as "aborted by user";
* a tip-quality halt keeps its own text, all the way out to ``SkillResult.error``
  the agent reads;
* a failed mandatory step keeps its own text too (it is not an abort at all).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_abort_reason_attribution.py -q
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

from mast.core.types import NanonisCallRecord, SkillResult  # noqa: E402
from mast.skills.composite.graph_executor import (  # noqa: E402
    CompositeProgress,
    _ABORT_LATCH_REASON,
    abort_error_text,
)

_TIP_HALT = (
    "tip_quality_drop: 扫描中途针尖变化（已采集 71 行中第 ~9 行）"
    "（视觉事件 2e1839e2）—— 已在当前步骤边界停止本流程，未回滚；修针尖后可重跑。"
)


# ════════════════════════════════════════════════════════════════════════════
# abort_error_text — the attribution rule itself
# ════════════════════════════════════════════════════════════════════════════

def test_only_the_latch_is_reported_as_a_user_abort():
    """The abort Event has exactly two setters: 中止 and E_STOP."""
    p = CompositeProgress(composite_name="X")
    p.aborted, p.aborted_reason = True, _ABORT_LATCH_REASON
    assert abort_error_text(p) == "aborted by user"


def test_an_empty_reason_falls_back_to_the_user_abort():
    p = CompositeProgress(composite_name="X")
    p.aborted, p.aborted_reason = True, ""
    assert abort_error_text(p) == "aborted by user"


def test_a_tip_halt_keeps_its_own_words():
    """The failure that made #46 undiagnosable from both ends."""
    p = CompositeProgress(composite_name="WaitScanComplete")
    p.aborted, p.aborted_reason = True, _TIP_HALT
    out = abort_error_text(p)
    assert out == _TIP_HALT
    assert "aborted by user" not in out, "又在冒充用户"
    assert "tip_quality_drop" in out, "agent 看不出该修针尖还是该重试"


def test_a_failed_mandatory_step_is_not_a_user_abort():
    p = CompositeProgress(composite_name="BatchRegionsScan")
    p.aborted = True
    p.aborted_reason = "FullScan raised ValueError: setpoint out of range"
    out = abort_error_text(p)
    assert "aborted by user" not in out
    assert "FullScan" in out and "ValueError" in out


def test_the_internal_latch_wording_never_reaches_the_caller():
    """`external abort flag before step` is executor jargon, not an operator
    message — it is a sentinel, and the sentinel must be translated."""
    p = CompositeProgress(composite_name="X")
    p.aborted, p.aborted_reason = True, _ABORT_LATCH_REASON
    assert _ABORT_LATCH_REASON not in abort_error_text(p)


# ════════════════════════════════════════════════════════════════════════════
# End-to-end through WaitScanComplete — the skill that actually did the lying
# ════════════════════════════════════════════════════════════════════════════

class _HaltCtx:
    """Context whose tip-quality halt is armed, like the field run."""

    def __init__(self, halt: str = _TIP_HALT):
        self._halt = halt
        self.calls: list[tuple] = []

    def check_abort(self) -> bool:
        return False                      # the operator did NOT abort

    def check_halt(self) -> str:
        h, self._halt = self._halt, ""    # one-shot, exactly like the real one
        return h

    def safe_call(self, method: str, *args, **kw) -> NanonisCallRecord:
        self.calls.append((method, args))
        return NanonisCallRecord(method=method, args=args, kwargs={},
                                 return_value=("", b"", []), error="")

    def run(self, skill_name: str, params: dict) -> SkillResult:
        return SkillResult(skill_name=skill_name, success=True)


def test_wait_scan_complete_reports_the_tip_halt_not_the_operator(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    from mast.skills.builtins.scan_utils import WaitScanComplete

    ctx = _HaltCtx()
    result = WaitScanComplete().execute(ctx, {"timeout_ms": 60000})

    assert not result.success
    assert "tip_quality_drop" in (result.error or ""), (
        f"WaitScanComplete 仍在冒充用户: {result.error!r}")
    assert "aborted by user" not in (result.error or "")
    # The scan must still be STOPPED — attribution changed, behaviour did not.
    assert any(m == "Scan_Action" and a[:1] == (1,) for m, a in ctx.calls), (
        "停扫调用丢了 — 针尖已坏却让扫描继续跑")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
