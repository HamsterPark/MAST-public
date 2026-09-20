"""Dreaming — background consolidation of experiment records into memory.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/memory/test_dreaming.py -x -v
"""
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

from mast.logging.storage import ExperimentStorage
from mast.memory.dreaming import (
    DREAM_TAG, DreamingService, gather_context, rule_based_consolidate,
)
from mast.memory.store import MemoryStore


def _setup(tmp_path):
    db = str(tmp_path / "exp.db")
    st = ExperimentStorage(db)
    exp = st.create_experiment("Run A", "study Si(111)")
    # log a few actions
    from mast.core.types import ActionRecord, SkillResult
    for sk in ["GetBias", "GetBias", "Scan"]:
        st.log_action(ActionRecord(
            experiment_id=exp, skill_name=sk,
            result=SkillResult(skill_name=sk, success=True)))
    st.log_conversation("user", "扫一张图", experiment_id=exp)
    mem = MemoryStore(db)
    return db, exp, mem


def test_gather_context(tmp_path):
    db, exp, mem = _setup(tmp_path)
    ctx = gather_context(db)
    assert len(ctx["experiments"]) == 1
    assert ctx["skill_counts"].get("GetBias") == 2
    assert ctx["statuses"].get("running") == 1


def test_rule_based_consolidate(tmp_path):
    db, exp, mem = _setup(tmp_path)
    entries = rule_based_consolidate(gather_context(db))
    assert entries and DREAM_TAG in entries[0]["content"]
    assert "GetBias" in entries[0]["content"]   # surfaced the top skill


def test_dream_once_writes_tagged_memory(tmp_path):
    db, exp, mem = _setup(tmp_path)
    svc = DreamingService(db, mem)
    written = svc.dream_once()
    assert written
    m = mem.read("global", "dreams/consolidation.md")
    assert m is not None and m["kind"] == "dream"
    assert DREAM_TAG in m["content"]            # honest, non-instrumental label
    assert m["author"] == "dream"


def test_dream_dedup(tmp_path):
    db, exp, mem = _setup(tmp_path)
    svc = DreamingService(db, mem)
    svc.dream_once()
    # an identical second cycle writes nothing new
    assert svc.dream_once() == []


def test_no_experiments_graceful(tmp_path):
    db = str(tmp_path / "empty.db")
    ExperimentStorage(db)  # creates schema, no experiments
    mem = MemoryStore(db)
    svc = DreamingService(db, mem)
    assert svc.dream_once() == []               # nothing to consolidate, no crash


def test_custom_consolidator(tmp_path):
    db, exp, mem = _setup(tmp_path)
    def _c(ctx):
        return [{"path": "dreams/custom.md", "title": "C", "kind": "insight",
                 "content": "a hypothesis"}]
    svc = DreamingService(db, mem, consolidator=_c)
    svc.dream_once()
    m = mem.read("global", "dreams/custom.md")
    assert m["kind"] == "insight" and DREAM_TAG in m["content"]  # tag auto-prepended


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
