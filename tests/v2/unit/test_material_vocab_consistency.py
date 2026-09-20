"""Controlled-vocabulary consistency for material / sample_type lookups.

The three enum paths used to disagree: start_sample rejected free text with NO
candidates; another path silently accepted anything; a third listed candidates on
error. Unify to ONE rule — a miss ALWAYS returns the candidate list — while
start_sample ACCEPTS free text (never blocks the experiment) and merely attaches
the candidates so a canonical type can be chosen.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_material_vocab_consistency.py -q
"""
from __future__ import annotations

import json
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

from mast.agents._shared.meta_tools import make_meta_tools
from mast.knowledge import list_material_candidates, material_candidates_hint

_MISS = "zzz_nonexistent_material_qwerty"


class _Log:
    """Minimal experiment log — accepts the idempotent start_sample signature."""
    _last_sample_reused = False

    def start_sample(self, name, description="", sample_type="",
                     sample_subtype="", reuse_active=False):
        return "sample-1"


def _tools():
    return make_meta_tools(lambda: {"experiment_log": _Log()})


def _tool(name):
    return next(t for t in _tools() if t.name == name)


# ── the shared helper ────────────────────────────────────────────────────

def test_public_snapshot_material_candidates_are_explicitly_empty():
    assert list_material_candidates() == []


def test_empty_material_vocabulary_has_no_misleading_hint():
    assert material_candidates_hint() == ""


# ── every material tool returns candidates on a miss ─────────────────────

def test_workflow_advice_miss_returns_explicitly_empty_candidates():
    out = json.loads(_tool("get_workflow_advice").invoke({"query": _MISS}))
    assert out["success"] is False
    assert out["candidates"] == []


def test_literature_parameters_miss_returns_explicitly_empty_candidates():
    out = json.loads(_tool("get_literature_parameters").invoke({"material": _MISS}))
    assert out["success"] is False
    assert out["candidates"] == []


def test_measurement_template_miss_returns_candidates():
    out = json.loads(
        _tool("get_measurement_template").invoke({"measurement_type": _MISS}))
    assert out["success"] is False
    assert out.get("candidates")  # its own vocabulary (template keys)


def test_all_material_tools_agree_on_the_empty_candidate_list():
    a = json.loads(_tool("get_workflow_advice").invoke({"query": _MISS}))
    b = json.loads(_tool("get_literature_parameters").invoke({"material": _MISS}))
    assert a["candidates"] == b["candidates"] == []


# ── start_sample: accept free text, attach candidates ────────────────────

def test_start_sample_accepts_free_text_without_material_knowledge():
    out = json.loads(_tool("start_sample").invoke({"name": _MISS}))
    assert out["success"] is True
    assert out["sample_id"] == "sample-1"
    assert out["name"] == _MISS
    assert out["sample_type_candidates"] == []
    assert "sample_type" not in out


def test_start_sample_does_not_invent_a_type_without_material_knowledge():
    out = json.loads(_tool("start_sample").invoke({"name": "Au(111)"}))
    assert out["success"] is True
    assert out["sample_id"] == "sample-1"
    assert out["name"] == "Au(111)"
    assert out["sample_type_candidates"] == []
    assert "sample_type" not in out


def test_start_sample_explicit_type_is_respected():
    out = json.loads(_tool("start_sample").invoke(
        {"name": _MISS, "sample_type": "superconductor"}))
    assert out["success"] is True
    assert out["sample_type"] == "superconductor"
    assert "sample_type_candidates" not in out  # explicit type → no nudge
