"""Sidecar resume must not fire when the step count is unknown (2026-07-27).

Field forensics, service-20260727.log. Inside ONE BatchRegionsScan:

    14:18:52  region 0 starts
    ...       region 0 runs 105.6 s   (a real 100 nm scan)
    14:20:37  [WaitScanComplete] resuming from step-level sidecar (211 completed > 0)
    14:20:37  [diag:step_skip] WaitScanComplete.finalize — 已完成（从 sidecar 断点恢复）
    ...       regions 1..4 take 0.37 / 0.38 / 0.38 / 0.22 s each

and the composite returned::

    {'success_count': 5, 'fail_count': 0,
     'scanned_paths': [..._0008.sxm, ..._0009.sxm, ..._0010.sxm, ..._0011.sxm, ..._0012.sxm],
     'recommended_region': <the 0.38 s one>}

A 100 nm scan at line_time_s=0.1 cannot complete in 0.37 s. The five .sxm paths
went to the operator as five distinct measurements, and the "best" region
recommended was one of the fake ones.

Root cause: the terminal-sidecar guard reads

    terminal = side.total_steps > 0 and len(side.completed_steps) >= side.total_steps

but run_plan() only assigns total_steps after the plan fully drains, while the
sidecar is flushed at every checkpoint. A STREAMING composite (WaitScanComplete
emits one synthetic _phase_poll_<i> step per poll) therefore always persists
total_steps == 0, so `terminal` could never fire and a finished run's sidecar
looked interrupted forever.

This is the second shipment of this exact failure — the guard's own comment
records the 2026-07-10 AutoApproach case , where a 9-day-old sidecar
declared 进针成功 at 0.17 pA against a 500 pA setpoint.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_sidecar_unmeasurable.py -q
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

import json  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from mast.skills.composite import graph_executor as _ge  # noqa: E402
from mast.skills.composite.graph_executor import GraphExecutor  # noqa: E402


def _sidecar_path(name: str, run_id: str) -> Path:
    """Always go through the MODULE attribute.

    conftest's _isolate_composite_sidecar monkeypatches _sidecar_path to redirect
    sidecars into tmp. A module-level `from ... import _sidecar_path` would bind
    the original at import time, so the test would WRITE to the real project dir
    while the executor READ from tmp — the test would then "pass" by finding
    nothing to resume, which is exactly the assertion it is supposed to prove.
    """
    return _ge._sidecar_path(name, run_id)


class _Ctx:
    """Minimal context: the run_id the sidecar is scoped to, plus a run() that
    succeeds at everything. No hardware, no registry."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.calls: list[str] = []

    def run(self, skill_name, params=None, **kw):
        from mast.core.types import SkillResult
        self.calls.append(skill_name)
        return SkillResult(skill_name=skill_name, success=True, data={})

    def checkpoint_flush(self):
        pass


def _write_sidecar(name: str, run_id: str, *, completed: int, total: int,
                   age_s: float = 0.0) -> Path:
    p = _sidecar_path(name, run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "completed_steps": [f"_phase_poll_{i}" for i in range(completed)],
        "total_steps": total,
        "last_update_at": time.time() - age_s,
        "partial_data": {"marker": "from-sidecar"},
        "aborted": False,
    }), encoding="utf-8")
    return p


@pytest.fixture
def run_id(tmp_path, monkeypatch):
    """A unique run_id per test so sidecars never collide across tests."""
    rid = "test-run-" + tmp_path.name
    yield rid
    try:
        _sidecar_path("StreamingComposite", rid).unlink(missing_ok=True)
        _sidecar_path("FixedComposite", rid).unlink(missing_ok=True)
    except Exception:
        pass


def _make(name: str, ctx) -> GraphExecutor:
    return GraphExecutor(composite_name=name, context=ctx)


# ════════════════════════════════════════════════════════════════════════════
# The regression: a FINISHED streaming run must not be resumed by the next caller
# ════════════════════════════════════════════════════════════════════════════

def _drained_plan(n_steps: int):
    """A plan of n steps, mimicking a streaming composite: the executor discovers
    the count as it goes, exactly like WaitScanComplete's one _phase_poll_<i>
    per poll."""
    from mast.skills.composite.graph_executor import CompositeStep
    for i in range(n_steps):
        yield CompositeStep(step_id=f"_phase_poll_{i}",
                            skill_name="NoOp", params={})


def test_a_completed_run_leaves_no_sidecar_at_all(run_id):
    """跑完 ⇒ **把自己的记录删掉**,而不是留一份标着「已完成」的。

    ## 这一条替换了什么(2026-08-12)

    原来钉的是「跑完要把 total_steps 落盘」——那是给读端的 terminal 判据准备分母,
    也就是**留一份文件、指望下一个调用者认出它已完成**。而 `run_plan` 收尾处的注释
    白纸黑字写着「**Normal completion clears the sidecar right after this**」——
    查实:``clear_sidecar()`` 全文件只在构造时的丢弃路径被调用过,**那个清理不存在**。
    整套 terminal/stale 守卫建立在一个不存在的清理之上,而注释让每个读它的人相信它在。

    这个失败模式已经出货**三次**:2026-07-10 AutoApproach 假进针、
    2026-07-27 BatchRegionsScan 五个 region 跳过 WaitScanComplete 报 success_count=5、
    2026-08-12 ForgeAuTip 同一次运行里 WaitScanComplete 从前一次调用的 sidecar 恢复。

    **没有文件比一份标着「已完成」的文件更强** —— 后者仍然依赖读端每一条判据都
    正确(而流式 composite 的 total_steps==0 恰好让 terminal 判据永假)。
    删掉之后,读端判据是**多余的保险**,不是唯一的防线。

    被打断的调用走不到 `run_plan` 收尾,它的 sidecar 照常留着续跑 ——
    `test_interrupted_streaming_run_still_resumes` 钉的正是那一条,两条不冲突。
    """
    ex = _make("StreamingComposite", _Ctx(run_id))
    ex.run_plan(_drained_plan(7))

    p = _sidecar_path("StreamingComposite", run_id)
    assert not p.exists(), (
        "跑完之后 sidecar 还在 —— 同一次运行里的下一个调用会把它当成自己的进度,"
        "跳过本该执行的步骤。这个形状已经出货三次了。")
    # 进度本身仍在内存里如实记着(报告要用),只是不再留在盘上给别人捡。
    assert len(ex.progress.completed_steps) == 7
    assert ex.progress.total_steps == 7


def test_next_caller_discards_the_finished_sidecar(run_id):
    """THE FIX, read side: given the step count now on disk, the terminal guard
    recognises the finished run and the next caller starts fresh.

    This is the field scenario — inside one BatchRegionsScan every region shares
    a run_id and therefore one sidecar file. Region 0 must not hand its finished
    progress to region 1.
    """
    ctx = _Ctx(run_id)
    first = _make("StreamingComposite", ctx)
    first.run_plan(_drained_plan(7))

    second = _make("StreamingComposite", ctx)
    assert len(second.progress.completed_steps) == 0, (
        "region 1 resumed region 0's finished sidecar — this is the 0.37-second "
        "fake scan that still reported success and a .sxm path"
    )


def test_five_sequential_streaming_runs_all_execute(run_id):
    """Five regions in one batch: every one actually runs its plan."""
    ctx = _Ctx(run_id)
    for region in range(5):
        ex = _make("StreamingComposite", ctx)
        assert len(ex.progress.completed_steps) == 0, (
            f"region {region} resumed a previous region's sidecar")
        ex.run_plan(_drained_plan(7))


# ════════════════════════════════════════════════════════════════════════════
# The behaviour that must NOT regress: genuine interrupt resume still works
# ════════════════════════════════════════════════════════════════════════════

def test_genuine_interrupted_sidecar_still_resumes(run_id):
    """A partially-completed run with a KNOWN step count is what resume exists
    for. Discarding this one would make the fix a functionality removal."""
    _write_sidecar("FixedComposite", run_id, completed=3, total=6)
    ex = _make("FixedComposite", _Ctx(run_id))
    assert len(ex.progress.completed_steps) == 3, "genuine resume was broken"
    assert ex.progress.partial_data.get("marker") == "from-sidecar"


def test_terminal_sidecar_still_discarded(run_id):
    """Rule 1 (all steps completed) must keep working."""
    _write_sidecar("FixedComposite", run_id, completed=6, total=6)
    ex = _make("FixedComposite", _Ctx(run_id))
    assert len(ex.progress.completed_steps) == 0


def test_stale_sidecar_still_discarded(run_id):
    """Rule 2 (older than the resume window) must keep working."""
    _write_sidecar("FixedComposite", run_id, completed=3, total=6, age_s=9 * 86400)
    ex = _make("FixedComposite", _Ctx(run_id))
    assert len(ex.progress.completed_steps) == 0


def test_interrupted_streaming_run_still_resumes(run_id):
    """The behaviour a read-side heuristic would break: a streaming composite
    INTERRUPTED mid-plan (never drained, so total_steps is still 0) must still
    resume. Discarding it would replay instrument actions and re-ask the
    operator questions already answered — see
    tests/v2/skills/composite/test_human_node_sidecar.py, which pins exactly
    that for the human-decision cache."""
    _write_sidecar("StreamingComposite", run_id, completed=3, total=0)
    ex = _make("StreamingComposite", _Ctx(run_id))
    assert len(ex.progress.completed_steps) == 3, (
        "an interrupted streaming run was discarded — resume exists precisely "
        "so a crash does not replay hardware actions")
    assert ex.progress.partial_data.get("marker") == "from-sidecar"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
