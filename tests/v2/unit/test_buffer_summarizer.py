"""Buffer summarizer tests — covers fallback path (no network) + cache."""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.agents.buffer_summarizer import node as _node
from mast.agents.buffer_summarizer.node import (
    _fallback_summary,
    _CACHE,
    _CACHE_LOCK,
    summarize_tip_status,
    summarize_segmentation,
    summarize_partial,
    summarize_tip_fine,
)


def _reset_cache():
    with _CACHE_LOCK:
        _CACHE.clear()


def test_fallback_tip_good():
    s = _fallback_summary("tip_coarse", {"label": "good", "confidence": 0.94})
    assert "良好" in s
    assert "0.94" in s


def test_fallback_tip_bad():
    s = _fallback_summary("tip_coarse", {"label": "bad", "confidence": 0.81})
    assert "较差" in s


def test_fallback_segmentation():
    s = _fallback_summary("segmentation", {
        "class_counts": {"TERRACE": 35000, "POINT_DEFECT_BRIGHT": 280, "STEP_EDGE": 1450}
    })
    # TERRACE is rendered in Chinese (台面) by the enriched deterministic templates.
    assert "台面" in s
    # Percentages should be sane (TERRACE ~95%)
    assert "95%" in s


def test_fallback_partial():
    s = _fallback_summary("partial", {"coarse_label": "good", "frac_acquired": 0.5})
    assert "50%" in s
    # coarse_label is rendered in Chinese (良好) by the enriched templates.
    assert "良好" in s


def test_fallback_unknown_kind_doesnt_crash():
    s = _fallback_summary("unknown_kind", {"foo": "bar"})
    assert isinstance(s, str)
    assert "unknown_kind" in s


def test_cache_hit_skips_llm(monkeypatch):
    """Second call with identical payload returns cached value, no LLM."""
    _reset_cache()
    called = {"n": 0}

    class FakeLLM:
        def invoke(self, msgs):
            called["n"] += 1
            class R: content = "针尖良好。"
            return R()

    monkeypatch.setattr(_node, "_get_llm", lambda model_id=None: FakeLLM())

    payload = {"label": "good", "confidence": 0.99}
    s1 = summarize_tip_status(payload)
    s2 = summarize_tip_status(payload)
    assert s1 == s2 == "针尖良好。"
    assert called["n"] == 1, f"LLM called {called['n']} times (cache miss)"


def test_llm_failure_falls_back(monkeypatch):
    """If the LLM raises, return the deterministic fallback summary."""
    _reset_cache()

    class BrokenLLM:
        def invoke(self, msgs):
            raise RuntimeError("network out")

    monkeypatch.setattr(_node, "_get_llm", lambda model_id=None: BrokenLLM())

    s = summarize_tip_status({"label": "bad", "confidence": 0.7})
    # fallback template kicks in
    assert "较差" in s
    assert "0.70" in s


def test_segmentation_strips_mask_rle(monkeypatch):
    """`mask_rle` should NOT be sent to the LLM (too large + binary)."""
    _reset_cache()
    captured = {}

    class CapturingLLM:
        def invoke(self, msgs):
            captured["user_content"] = msgs[1].content
            class R: content = "表面以平台为主。"
            return R()

    monkeypatch.setattr(_node, "_get_llm", lambda model_id=None: CapturingLLM())

    seg_payload = {
        "shape": (256, 256),
        "class_counts": {"TERRACE": 65000, "POINT_DEFECT_BRIGHT": 1000},
        "mask_rle": b"\x00" * 100,  # bytes — should be stripped
    }
    s = summarize_segmentation(seg_payload)
    assert "平台" in s
    # Crucial: mask_rle absent from prompt
    assert "mask_rle" not in captured["user_content"]


def test_oversized_response_uses_fallback(monkeypatch):
    """If LLM returns >200 char text (probably a runaway), use fallback."""
    _reset_cache()

    class RamblingLLM:
        def invoke(self, msgs):
            class R: content = "针尖" + "状态描述太长了。" * 50
            return R()

    monkeypatch.setattr(_node, "_get_llm", lambda model_id=None: RamblingLLM())

    s = summarize_tip_status({"label": "good", "confidence": 0.9})
    assert "良好" in s  # fallback content
    assert len(s) < 200
