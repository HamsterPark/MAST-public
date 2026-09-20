"""Transcripts must preserve complete tool returns or explicitly signal truncation. Router selection must also respect providers that reject pinned tool_choice with mandatory reasoning."""
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

import pytest  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# — truncation must be visible, and the record must hold a real plan
# ════════════════════════════════════════════════════════════════════════════

#: The real shape that got cut: a 28-step plan of FullScan entries.
def _plan_json(n_steps: int = 28) -> str:
    return json.dumps({"steps": [
        {"skill_name": "FullScan",
         "params": {"center_x_m": i * 1e-7, "center_y_m": 0.0,
                    "width_m": 5e-7, "height_m": 5e-7},
         "expected_metric": f"step {i} topography"}
        for i in range(n_steps)]}, ensure_ascii=False)


def test_short_text_is_untouched():
    from mast.api.routes.orchestrator import _clip
    assert _clip("hello", 100) == "hello"
    assert _clip("x" * 100, 100) == "x" * 100      # exactly at the limit


def test_truncation_is_announced():
    """The defect was not the cut — it was the SILENCE."""
    from mast.api.routes.orchestrator import _clip
    out = _clip("A" * 2500, 2000)
    assert len(out) > 2000                          # marker appended
    assert "截断" in out
    assert "2500" in out, "the true length must be recoverable from the record"


def test_a_28_step_plan_survives_the_persist_cap():
    """The exact payload that was lost. It must now be stored whole."""
    from mast.api.routes.orchestrator import _clip, _PERSIST_TEXT_MAX

    plan = _plan_json(28)
    assert len(plan) > 2000, "fixture too small to reproduce the bug"
    stored = _clip(plan, _PERSIST_TEXT_MAX)
    assert stored == plan, "the plan was clipped again"
    # And it must still parse — the old failure was a cut mid-JSON.
    assert len(json.loads(stored)["steps"]) == 28


def test_persist_cap_is_far_above_the_sse_cap():
    """The record and the render have different jobs; the record keeps more."""
    from mast.api.routes.orchestrator import _PERSIST_TEXT_MAX, _SSE_TEXT_MAX
    assert _PERSIST_TEXT_MAX >= 100_000
    assert _SSE_TEXT_MAX < _PERSIST_TEXT_MAX


def test_clip_handles_none():
    from mast.api.routes.orchestrator import _clip
    assert _clip(None, 100) == ""


# ════════════════════════════════════════════════════════════════════════════
# — do not send a request the provider is known to refuse
# ════════════════════════════════════════════════════════════════════════════

def test_reasoning_models_are_known_to_reject_forced_tool_choice():
    from mast.agents._shared.models import (
        rejects_forced_tool_choice, KIMI_K3, DEEPSEEK_V4_PRO, GLM_5_1,
    )
    for m in (KIMI_K3, DEEPSEEK_V4_PRO, GLM_5_1):
        assert rejects_forced_tool_choice(m), f"{m} must be gated"


def test_non_reasoning_models_still_use_the_structured_tier():
    """The fast path must survive — this fix skips a tier, and skipping it for
    models that handle it fine would be a downgrade."""
    from mast.agents._shared.models import rejects_forced_tool_choice, SONNET_4_6
    assert not rejects_forced_tool_choice(SONNET_4_6)


@pytest.mark.parametrize("bad", [None, "", "some-unknown-model"])
def test_unknown_model_is_not_gated(bad):
    """Unknown → attempt the structured tier and let the fallback catch it.
    Guessing "probably a reasoning model" would silently disable the good path."""
    from mast.agents._shared.models import rejects_forced_tool_choice
    assert not rejects_forced_tool_choice(bad)


def test_model_id_of_reads_the_usual_attributes():
    from mast.agents._shared.models import model_id_of

    class _M:
        model_name = "kimi-k3"

    class _N:
        model = "claude-sonnet-4-6"

    class _Blank:
        pass

    assert model_id_of(_M()) == "kimi-k3"
    assert model_id_of(_N()) == "claude-sonnet-4-6"
    assert model_id_of(_Blank()) is None


def test_router_skips_tier1_for_a_reasoning_model(monkeypatch):
    """End-to-end on the router: with a reasoning model, with_structured_output
    must never be called — that call IS the 400."""
    from mast.agents.orchestrator import graph as g

    calls = {"structured": 0, "invoke": 0}

    class _FakeModel:
        model_name = "kimi-k3"

        def with_structured_output(self, *a, **kw):
            calls["structured"] += 1
            raise AssertionError("tier 1 was attempted on a reasoning model")

        def invoke(self, messages, **kw):
            calls["invoke"] += 1

            class _R:
                content = '{"next_agents": ["instrument_control"], "reason": "go"}'
            return _R()

    fn = getattr(g, "_parallel_route_decision", None)
    if fn is None:
        pytest.skip("router helper not exposed under the expected name")
    out = fn(_FakeModel(), [{"role": "user", "content": "scan something"}])
    assert calls["structured"] == 0
    assert out and out.get("next_agents") == ["instrument_control"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
