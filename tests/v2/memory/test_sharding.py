"""Conversation phase-sharding: start/end/summary→memory/auto-shard.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/memory/test_sharding.py -x -v
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
from mast.memory.sharding import PhaseManager, rule_based_summary
from mast.memory.store import MemoryStore


def _setup(tmp_path):
    db = str(tmp_path / "exp.db")
    st = ExperimentStorage(db)
    exp = st.create_experiment("Run")
    mem = MemoryStore(db)
    pm = PhaseManager(db, memory_store=mem)
    return st, exp, mem, pm


def test_rule_based_summary():
    msgs = [{"role": "user", "content": "扫一张图"},
            {"role": "assistant", "content": "完成,Si(111) 7x7 清晰"}]
    s = rule_based_summary(msgs)
    assert "扫一张图" in s and "Si(111)" in s and "2 条" in s


def test_phase_lifecycle_and_summary_to_memory(tmp_path):
    st, exp, mem, pm = _setup(tmp_path)
    pm.start_phase("针尖准备", experiment_id=exp)
    st.log_conversation("user", "脉冲修复针尖", experiment_id=exp)
    st.log_conversation("assistant", "针尖已合格", experiment_id=exp)
    closed = pm.end_phase(exp, summarize=True)
    assert closed is not None and closed["summary"]
    assert "脉冲修复针尖" in closed["summary"]
    # summary persisted as a memory entry
    m = mem.read(f"experiment:{exp}", "phases/phase-0.md")
    assert m is not None and m["kind"] == "summary"
    # phase recorded
    phases = pm.list_phases(exp)
    assert len(phases) == 1 and phases[0]["ended_msg_id"] is not None


def test_current_phase(tmp_path):
    st, exp, mem, pm = _setup(tmp_path)
    assert pm.current_phase(exp) is None
    pm.start_phase("p", experiment_id=exp)
    assert pm.current_phase(exp)["title"] == "p"
    pm.end_phase(exp)
    assert pm.current_phase(exp) is None   # closed


def test_auto_shard(tmp_path):
    st, exp, mem, pm = _setup(tmp_path)
    pm.start_phase("p0", experiment_id=exp)
    for i in range(5):
        st.log_conversation("user", f"msg {i}", experiment_id=exp)
    # threshold 3 → should shard (end p0, start p1)
    new = pm.maybe_auto_shard(exp, threshold=3)
    assert new is not None and new["phase_index"] == 1
    phases = pm.list_phases(exp)
    assert len(phases) == 2
    assert phases[0]["ended_msg_id"] is not None   # p0 closed + summarised
    assert phases[0]["summary"]


def test_no_conversation_log_graceful(tmp_path):
    # PhaseManager on a db without conversation_log yet must not crash
    pm = PhaseManager(str(tmp_path / "empty.db"))
    assert pm.current_phase() is None
    p = pm.start_phase("x")
    assert p["phase_index"] == 0
    assert pm.end_phase() is not None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
