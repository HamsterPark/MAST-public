"""CognitionContext — assembly of memory/sharding/dreaming + agent tools.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/memory/test_cognition_service.py -q -p no:randomly
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.agents._shared.cognition import CognitionContext, from_storage
from mast.logging.storage import ExperimentStorage
from mast.memory.dreaming import DREAM_TAG, DreamingService
from mast.memory.sharding import PhaseManager
from mast.memory.store import MemoryStore


# ── helpers ───────────────────────────────────────────────────────────
def _call_tool(t, **kwargs) -> str:
    """Invoke a LangChain StructuredTool by name regardless of API version."""
    return t.invoke(kwargs)


def _tool_by_name(tools, name):
    return next(t for t in tools if t.name == name)


# ── tests ─────────────────────────────────────────────────────────────
def test_builds_three_backends(tmp_path):
    cog = CognitionContext(tmp_path / "exp.db", author="planner")
    assert isinstance(cog.store, MemoryStore)
    assert isinstance(cog.phases, PhaseManager)
    assert isinstance(cog.dreaming, DreamingService)
    # the memory store + phase manager share the same DB file as cognition
    assert cog.phases._memory is cog.store


def test_tools_returns_four_memory_tools(tmp_path):
    cog = CognitionContext(tmp_path / "exp.db")
    tools = cog.tools(experiment_id="exp-1")
    names = {t.name for t in tools}
    assert len(tools) == 4
    assert names == {"memory_write", "memory_read", "memory_list", "memory_search"}


def test_write_read_persists_across_new_context_same_db(tmp_path):
    db = tmp_path / "exp.db"
    cog = CognitionContext(db, author="agent-A")
    tools = cog.tools(experiment_id="exp-7")
    write = _tool_by_name(tools, "memory_write")
    out = _call_tool(write, path="insights/tip.md",
                     content="脉冲 3V 可整针", kind="insight")
    assert "已保存记忆" in out and "experiment:exp-7" in out

    # a BRAND NEW context over the SAME db must read it back (true persistence)
    cog2 = CognitionContext(db, author="agent-B")
    read = _tool_by_name(cog2.tools(experiment_id="exp-7"), "memory_read")
    got = _call_tool(read, path="insights/tip.md")
    assert "脉冲 3V 可整针" in got
    # and the raw store agrees on the namespace it landed in
    assert cog2.store.read("experiment:exp-7", "insights/tip.md") is not None


def test_memory_provider_namespace_rule(tmp_path):
    cog = CognitionContext(tmp_path / "exp.db", author="default-author")
    # experiment_id present -> experiment:{id}
    p_exp = cog.memory_provider(experiment_id="run-42")()
    assert p_exp["namespace"] == "experiment:run-42"
    assert p_exp["experiment_id"] == "run-42"
    assert p_exp["store"] is cog.store
    assert p_exp["author"] == "default-author"
    # no experiment_id -> global
    p_glob = cog.memory_provider()()
    assert p_glob["namespace"] == "global"
    assert p_glob["experiment_id"] is None
    # per-call author override wins
    p_over = cog.memory_provider(experiment_id="x", author="vision")()
    assert p_over["author"] == "vision"


def test_start_stop_dreaming_idempotent(tmp_path):
    cog = CognitionContext(tmp_path / "exp.db")
    calls = {"n": 0}
    cog.start_dreaming(should_dream=lambda: calls.__setitem__("n", calls["n"] + 1) or True)
    assert cog._dreaming_on is True
    assert cog.dreaming._running is True
    # repeated start is a no-op (still running, no second thread)
    t1 = cog.dreaming._thread
    cog.start_dreaming()
    assert cog.dreaming._thread is t1
    # stop joins cleanly; repeated stop is harmless
    cog.stop_dreaming()
    assert cog._dreaming_on is False
    assert cog.dreaming._running is False
    cog.stop_dreaming()  # idempotent, must not raise
    # the idle predicate we injected is the one the service holds
    assert cog.dreaming._should_dream() is True


def test_custom_consolidator_is_used_by_dreaming(tmp_path):
    db = tmp_path / "exp.db"
    cog = CognitionContext(db, author="dreamer")
    seen = {"called": False}

    def fake_consolidator(context: dict) -> list:
        seen["called"] = True
        return [{"path": "dreams/custom.md", "title": "自定义做梦",
                 "content": "整合结论 X", "kind": "dream"}]

    cog.set_consolidator(fake_consolidator)
    assert cog.dreaming._consolidate is fake_consolidator

    written = cog.dream_once()
    assert seen["called"] is True
    assert any(w["path"] == "dreams/custom.md" for w in written)
    # the dream landed in the store, honesty-tagged, under the global namespace
    rec = cog.store.read("global", "dreams/custom.md")
    assert rec is not None
    assert "整合结论 X" in rec["content"]
    assert DREAM_TAG in rec["content"]
    assert rec["kind"] == "dream"


def test_set_summarizer_injects_into_phase_manager(tmp_path):
    cog = CognitionContext(tmp_path / "exp.db")

    def fake_summary(messages: list) -> str:
        return f"LLM摘要({len(messages)}条)"

    cog.set_summarizer(fake_summary)
    assert cog.phases._summarizer is fake_summary
    # revert to rule-based default
    cog.set_summarizer(None)
    from mast.memory.sharding import rule_based_summary
    assert cog.phases._summarizer is rule_based_summary


def test_from_storage_factory_shares_db(tmp_path):
    db = str(tmp_path / "exp.db")
    st = ExperimentStorage(db)
    exp = st.create_experiment("Run A", "study Si(111)")

    # classmethod + module-level factory both work and share the storage DB
    cog = CognitionContext.from_storage(st, author="orc")
    cog2 = from_storage(st)
    assert str(cog.store._db_path) == db
    assert str(cog2.store._db_path) == db

    # round-trip a memory through the factory-built context
    tools = cog.tools(experiment_id=exp)
    _call_tool(_tool_by_name(tools, "memory_write"),
               path="notes/a.md", content="hello", kind="note")
    read = _tool_by_name(cog2.tools(experiment_id=exp), "memory_read")
    assert "hello" in _call_tool(read, path="notes/a.md")


def test_offline_dream_once_rule_based(tmp_path):
    """No LLM, no network: default rule-based consolidation still produces value."""
    db = str(tmp_path / "exp.db")
    st = ExperimentStorage(db)
    exp = st.create_experiment("Run A", "study Si(111)")
    from mast.core.types import ActionRecord, SkillResult
    for sk in ["GetBias", "GetBias", "Scan"]:
        st.log_action(ActionRecord(
            experiment_id=exp, skill_name=sk,
            result=SkillResult(skill_name=sk, success=True)))

    cog = CognitionContext.from_storage(st)
    written = cog.dream_once()  # default consolidator == rule_based_consolidate
    assert written  # produced at least one dream entry
    rec = cog.store.read("global", written[0]["path"])
    assert rec is not None and DREAM_TAG in rec["content"]
