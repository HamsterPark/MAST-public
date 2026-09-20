"""GraphExecutor + SetBiasRamp regression tests (Phase 7 migration).

Pins the following contracts:

  1. ``SetBiasRamp`` subclasses :class:`CompositeSkillGraph` and exposes
     each ramp step as a synthetic ``_phase_set_step_<i>`` step.
  2. When ``bias_v_start`` is given explicitly, the plan contains exactly
     N=max(1, |Δv| / (slew * step_interval)) steps (no get_current phase).
  3. When ``bias_v_start`` is omitted, a leading ``get_current`` phase
     issues Bias_Get to seed the starting voltage.
  4. Each ramp step issues exactly one Bias_Set, and the final aggregate
     reports ``steps``, ``bias_v``, and ``slew_rate_v_per_s``.
  5. Resume skips already-completed ramp steps (no duplicate Bias_Set).
  6. Per-step ``emit_progress`` fires; checkpoint flush only on the last
     ramp step.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_set_bias_ramp_graph.py -x -v
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
from typing import Any

import pytest

from mast.core.types import NanonisCallRecord, SafetyLevel, SkillResult
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeProgress
from mast.skills.builtins.bias import SetBiasRamp


# ── FakeCtx ───────────────────────────────────────────────────────────────


@dataclass
class FakeCtx:
    bias_get_return: Any = ("", b"", [0.0])
    bias_set_error: str | None = None
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    emitted: list[CompositeProgress] = field(default_factory=list)
    prior_progress: dict | None = None
    flushes: int = 0
    _abort: bool = False

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method == "Bias_Get":
            return NanonisCallRecord(
                method=method, args=args, return_value=self.bias_get_return,
            )
        if method == "Bias_Set":
            if self.bias_set_error:
                return NanonisCallRecord(
                    method=method, args=args, error=self.bias_set_error,
                )
            return NanonisCallRecord(
                method=method, args=args, return_value=("", b"", []),
            )
        return NanonisCallRecord(method=method, args=args,
                                 error=f"unmocked: {method}")

    def check_abort(self) -> bool:
        return self._abort

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Shape tests ────────────────────────────────────────────────────────────


def test_set_bias_ramp_is_composite_graph():
    assert issubclass(SetBiasRamp, CompositeSkillGraph)


def test_set_bias_ramp_metadata():
    meta = SetBiasRamp().metadata()
    assert meta.name == "SetBiasRamp"
    assert meta.safety_level == SafetyLevel.CONFIRM


# ── Plan tests ─────────────────────────────────────────────────────────────


def test_plan_step_count_5_steps():
    """5V change @ 10V/s, 0.1s/step → max(1, 5 / (10*0.1)) = 5 steps."""
    skill = SetBiasRamp()
    plan = skill.plan({
        "bias_v_start": 0.0,
        "bias_v_end": 5.0,
        "slew_rate_v_per_s": 10.0,
        "step_interval_s": 0.1,
    })
    assert len(plan) == 5
    # Step ids consecutive
    assert [s.step_id for s in plan] == [
        "set_step_0", "set_step_1", "set_step_2",
        "set_step_3", "set_step_4",
    ]
    # Skill names are _phase_set_step_<i>
    assert all(s.skill_name.startswith("_phase_set_step_") for s in plan)
    # All mandatory (one bad set aborts the ramp)
    assert all(not s.optional for s in plan)
    # Only the last step flushes checkpoint
    assert [s.checkpoint_after for s in plan] == [False, False, False, False, True]


def test_plan_single_shot_below_1mv():
    """|Δv| < 1mV → single shot (no real ramp)."""
    skill = SetBiasRamp()
    plan = skill.plan({
        "bias_v_start": 1.0,
        "bias_v_end": 1.0005,
        "slew_rate_v_per_s": 1.0,
        "step_interval_s": 0.1,
    })
    assert len(plan) == 1
    # Single step targets the requested end voltage directly
    assert plan[0].params["target_v"] == pytest.approx(1.0005)


def test_target_voltages_linear():
    """Targets are evenly spaced (np.linspace from start to end, no start)."""
    skill = SetBiasRamp()
    plan = skill.plan({
        "bias_v_start": 0.0,
        "bias_v_end": 1.0,
        "slew_rate_v_per_s": 10.0,
        "step_interval_s": 0.1,
    })
    # 1V / (10*0.1) = 1 step → linspace(0, 1, 2)[1:] = [1.0]
    assert len(plan) == 1
    assert plan[0].params["target_v"] == pytest.approx(1.0)

    plan_4 = skill.plan({
        "bias_v_start": 0.0,
        "bias_v_end": 0.4,
        "slew_rate_v_per_s": 1.0,
        "step_interval_s": 0.1,
    })
    # 0.4V / (1 * 0.1) = 4 steps → linspace(0, 0.4, 5)[1:] = [0.1, 0.2, 0.3, 0.4]
    targets = [s.params["target_v"] for s in plan_4]
    assert targets == pytest.approx([0.1, 0.2, 0.3, 0.4])


# ── Execution tests ───────────────────────────────────────────────────────


def test_execution_5_steps_issues_5_bias_set(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = SetBiasRamp()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "bias_v_start": 0.0,
        "bias_v_end": 5.0,
        "slew_rate_v_per_s": 10.0,
        "step_interval_s": 0.1,
    })
    assert result.success
    bias_sets = [c for c in ctx.calls if c[0] == "Bias_Set"]
    assert len(bias_sets) == 5
    # Last Bias_Set targets the end voltage exactly
    assert bias_sets[-1][1][0] == pytest.approx(5.0)
    # No leading Bias_Get (bias_v_start was provided)
    assert not any(c[0] == "Bias_Get" for c in ctx.calls)
    # Aggregate carries the contract fields
    assert result.data["bias_v"] == pytest.approx(5.0)
    assert result.data["slew_rate_v_per_s"] == 10.0
    assert result.data["steps"] == 5


def test_execution_uses_bias_get_when_start_omitted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = SetBiasRamp()
    # Bias_Get returns the current bias 0.0 → ramp 0 → 1V
    ctx = FakeCtx(bias_get_return=("", b"", [0.0]))
    result = skill.execute(ctx, {
        "bias_v_end": 1.0,
        "slew_rate_v_per_s": 1.0,
        "step_interval_s": 0.1,
    })
    assert result.success
    # Bias_Get fires exactly once (the get_current phase)
    bias_gets = [c for c in ctx.calls if c[0] == "Bias_Get"]
    assert len(bias_gets) == 1
    # Then 10 ramp steps (1V / 0.1 = 10)
    bias_sets = [c for c in ctx.calls if c[0] == "Bias_Set"]
    assert len(bias_sets) == 10


def test_failure_aborts_remaining_steps(monkeypatch):
    """Bias_Set error on the 2nd step → ramp aborts, no more Bias_Set calls."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = SetBiasRamp()
    # Make Bias_Set always fail
    ctx = FakeCtx(bias_set_error="hardware lockout")
    result = skill.execute(ctx, {
        "bias_v_start": 0.0,
        "bias_v_end": 5.0,
        "slew_rate_v_per_s": 10.0,
        "step_interval_s": 0.1,
    })
    assert not result.success
    assert "Slew failed" in (result.error or "")
    # Only ONE Bias_Set was attempted (first step fails → mandatory abort)
    bias_sets = [c for c in ctx.calls if c[0] == "Bias_Set"]
    assert len(bias_sets) == 1


# ── Progress emission ─────────────────────────────────────────────────────


def test_emit_progress_fires_per_step(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = SetBiasRamp()
    ctx = FakeCtx()
    skill.execute(ctx, {
        "bias_v_start": 0.0,
        "bias_v_end": 5.0,
        "slew_rate_v_per_s": 10.0,
        "step_interval_s": 0.1,
    })
    # 5 set_step + 1 final summary → at least 5 emits
    assert len(ctx.emitted) >= 5
    # Final emit clears current_step
    assert ctx.emitted[-1].current_step is None


# ── Resume ────────────────────────────────────────────────────────────────


def test_resume_skips_completed_steps(monkeypatch):
    """Prior progress has steps 0+1 done → only 3 fresh Bias_Sets."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = SetBiasRamp()
    prior = CompositeProgress(
        composite_name="SetBiasRamp",
        total_steps=5,
        completed_steps=["set_step_0", "set_step_1"],
        partial_data={"bias_v_start": 0.0, "n_steps": 5, "bias_v_end": 5.0,
                      "slew_rate_v_per_s": 10.0},
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    skill.execute(ctx, {
        "bias_v_start": 0.0,
        "bias_v_end": 5.0,
        "slew_rate_v_per_s": 10.0,
        "step_interval_s": 0.1,
    })
    bias_sets = [c for c in ctx.calls if c[0] == "Bias_Set"]
    # Only 3 fresh ramp steps (steps 2, 3, 4)
    assert len(bias_sets) == 3


# ── Checkpoint flush ──────────────────────────────────────────────────────


def test_checkpoint_flushes_only_on_last(monkeypatch):
    """Only the last ramp step has checkpoint_after=True."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = SetBiasRamp()
    ctx = FakeCtx()
    skill.execute(ctx, {
        "bias_v_start": 0.0,
        "bias_v_end": 5.0,
        "slew_rate_v_per_s": 10.0,
        "step_interval_s": 0.1,
    })
    # 5 steps → only 1 flush (on the last step)
    assert ctx.flushes == 1


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ── 读不到当前偏压 ⇒ 拒绝斜坡（v6.1.3, KNOWN_ISSUES）────────────────────
#
# 修前:`_phase_get_current` 解析失败兜底成 `0.0`,而这个值是 `_compute_steps`
# 的**起点**。真实偏压 1 V 而起点被当成 0 时,第一步就把硬件从 1 V 拽到接近 0 ——
# **那正是 `slew_rate_v_per_s` 存在的意义所要防止的突变**。
# 一个失败的读取变成一个假的测量值,而那个假值废掉了一条安全保护。
#
# 同一个文件里的 `SetBias` 早就是拒绝的("Refusing to ramp from an assumed 0.0V
# start, which would defeat slew protection")—— 那句话一直写在那里,
# 而 `SetBiasRamp` 一直在做它拒绝的事。


@pytest.mark.parametrize("bad", [
    ("", b"", []),                 # 空 Variables:parsed[2][0] IndexError
    ("", b"", ["not a number"]),   # 非数值:float() ValueError
    ("", b"", [None]),             # None:TypeError
    ("", b"", [float("nan")]),     # NaN —— 是个 float,却不是一个可用的起点
    None,                          # 整个回包缺失
])
def test_unreadable_bias_refuses_to_ramp(monkeypatch, bad):
    """拒绝,而不是挑一个更好的默认值。"""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(bias_get_return=bad)
    result = SetBiasRamp().execute(ctx, {
        "bias_v_end": 1.0, "slew_rate_v_per_s": 1.0, "step_interval_s": 0.1,
    })
    assert not result.success
    assert "0.0V" in (result.error or "") or "0.0 V" in (result.error or "")
    # 关键断言:**一次硬件写都没发生**。报错但仍然斜坡过,等于没修。
    assert not [c for c in ctx.calls if c[0] == "Bias_Set"], (
        "读不到起点却仍然下发了 Bias_Set —— 限斜率保护已被绕过")


def test_the_refusal_names_slew_protection():
    """错误文案要说清**为什么**拒绝,否则下一个人会把它当成过度严格删掉。
    照抄兄弟 `SetBias` 的措辞,两处说同一件事。"""
    ctx = FakeCtx(bias_get_return=("", b"", []))
    result = SetBiasRamp().execute(ctx, {
        "bias_v_end": 1.0, "slew_rate_v_per_s": 1.0,
    })
    assert "slew" in (result.error or "").lower()


def test_an_explicit_start_still_works_without_reading():
    """对照组:调用方显式给起点时不需要读,也不该被这条拒绝挡住。"""
    ctx = FakeCtx(bias_get_return=None)          # 读了就会失败
    result = SetBiasRamp().execute(ctx, {
        "bias_v_start": 0.0, "bias_v_end": 1.0,
        "slew_rate_v_per_s": 1.0, "step_interval_s": 0.1,
    })
    assert result.success
    assert not [c for c in ctx.calls if c[0] == "Bias_Get"]
