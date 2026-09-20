"""Context compaction + cross-conversation memory + ported meta-tools.

Pins the shared-layer modernity: compaction fires on a token threshold (persistent
RemoveMessage + memory_sink), semantic recall is namespace-scoped (no cross-
experiment leak) with reindex-on-update, the recall middleware injects a block,
and the ported planner meta-tools build.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 not found")


_ROOT = _find_mastv2_root()
if sys.path[0] != _ROOT:
    while _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        if "MASTv2" not in (getattr(sys.modules[_n], "__file__", "") or "").replace("\\", "/"):
            del sys.modules[_n]

import types

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


class _FM(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


# ── compaction ─────────────────────────────────────────────────────────
def test_compaction_trigger_from_window():
    from mast.agents._shared.compaction_mw import compaction_trigger_tokens
    assert compaction_trigger_tokens("kimi-k2.6") == int(0.75 * (240000 - 16000))
    assert compaction_trigger_tokens("unknown-model") == int(0.75 * (120000 - 16000))


def test_compaction_fires_and_persists_and_sinks():
    from mast.agents._shared.compaction_mw import _MemorySinkSummarization
    sink = []
    mw = _MemorySinkSummarization(
        model=_FM(messages=iter([AIMessage("【摘要】要点。") for _ in range(8)])),
        trigger=("tokens", 40), keep=("messages", 2),
        memory_sink=lambda s: sink.append(s))
    agent = create_agent(
        model=_FM(messages=iter([AIMessage("回复 " + "词 " * 30) for _ in range(8)])),
        tools=[], system_prompt="s", middleware=[mw], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "c"}}
    for i in range(5):
        agent.invoke({"messages": [("user", f"问题{i} " + "词 " * 30)]}, config=cfg)
    msgs = agent.get_state(cfg).values["messages"]
    joined = " ".join(str(getattr(m, "content", "")) for m in msgs)
    assert "summary of the conversation" in joined  # summary persisted
    assert sink  # memory_sink received the summary


# ── semantic memory (cognition) ─────────────────────────────────────────
def _force_hash_embedder(monkeypatch):
    import mast.memory.embeddings as emb
    from mast.logging.v2.vector_search import HashEmbedder
    monkeypatch.setattr(emb, "make_embedder",
                        lambda prefer="auto": (HashEmbedder(dim=64), 64, "hashtest"))


def test_recall_is_namespace_scoped_and_reindexes(tmp_path, monkeypatch):
    _force_hash_embedder(monkeypatch)
    from mast.agents._shared.cognition import CognitionContext
    cog = CognitionContext(str(tmp_path / "exp.db"), author="user")
    cog.remember("global", "p/a.md", "tip conditioning pulse atomic", title="tip", kind="protocol")
    cog.remember("experiment:E1", "n/s.md", "sample TaS2 cdw", title="s", kind="note",
                 experiment_id="E1")
    cog.remember("experiment:E2", "n/o.md", "graphene moire twist", title="o", kind="note",
                 experiment_id="E2")
    rows = cog.recall("tip conditioning pulse", experiment_id="E1", k=5)
    ns = {r["namespace"] for r in rows}
    assert "experiment:E2" not in ns  # no cross-experiment leak
    # update same path → no stale duplicate vector
    cog.remember("global", "p/a.md", "UPDATED tip conditioning pulse", title="tip2",
                 kind="protocol")
    vec = cog._ensure_vec()
    hits = [h for h in vec.knn("tip conditioning", k=20)
            if h["entity_id"] == "global/p/a.md"]
    assert len(hits) == 1


def test_recall_middleware_injects_block(tmp_path, monkeypatch):
    _force_hash_embedder(monkeypatch)
    from mast.agents._shared.cognition import CognitionContext
    from mast.agents._shared.memory_mw import MemoryRecallMiddleware
    cog = CognitionContext(str(tmp_path / "exp.db"), author="user")
    cog.remember("experiment:E1", "n/s.md", "样品 TaS2 在 180K 相变", title="样品笔记",
                 kind="note", experiment_id="E1")
    mw = MemoryRecallMiddleware(cog, namespace_provider=lambda: "E1", k=5)
    req = types.SimpleNamespace(messages=[HumanMessage("样品信息")],
                               system_message=SystemMessage("你是 IC"))
    out = mw.wrap_model_call(req, lambda r: r)
    # 2026-08-24：召回块挂**最后一条 human 消息**，不再动 system —— 它逐轮不同
    # （检索键 = 本轮提问），而 Anthropic 的 cache 断点打在 system 末尾。
    human = out.messages[-1].content
    assert "MEMORY" in human or "记忆" in human
    # 而 system 逐字节没变，这才是搬家的目的。
    assert out.system_message.content == "你是 IC"


# ── ported meta-tools ───────────────────────────────────────────────────
def test_meta_tools_build():
    from mast.agents._shared.meta_tools import make_meta_tools
    tools = make_meta_tools(lambda: {})
    names = {t.name for t in tools}
    assert {"start_experiment", "query_knowledge", "get_next_scan_position",
            "create_plan", "load_scan_file",
            "rename_experiment", "rename_sample"} <= names
    # Plan-EXECUTION tools added 2026-07-03 (get_plan_progress / advance_plan /
    # pause_plan / resume_plan) so overnight multi-phase plans can be tracked +
    # resumed after an interruption.
    assert {"get_plan_progress", "advance_plan", "pause_plan", "resume_plan"} <= names
    # spawn_background_task added 2026-07-20 (true-parallel offload — the foreground
    # agent detaches a long/independent sub-task to a background run).
    assert "spawn_background_task" in names
    # Scope switching added 2026-07-28. Experiments and samples are permanent and
    # switchable in both directions — 「做了一个月这个又回去做那个」 — so the model
    # must be able to go back to an older experiment itself rather than minting a
    # duplicate (the field saw 5 identically-named experiments two minutes apart).
    assert {"switch_experiment", "switch_sample", "clear_sample",
            "list_experiments", "list_samples"} <= names
    # Scan-map analysis added 2026-07-30. The agent must not decide where it has
    # scanned, what it has ruined, or whether to relocate by looking at a picture
    # — these read programmatic conclusions computed from the recorded markers.
    assert {"get_map_analysis", "record_coarse_move", "get_markers_near"} <= names
    # Multi-frame planning added 2026-07-31. "Scan me several images" must not be
    # the model picking positions and parameters — the planner reads the coverage
    # map and the operator's per-scale policy table and returns a deterministic
    # frame list (or a typed rejection). See docs/v2/design/scan_intelligence_scripted_rfc.md
    assert "plan_scan_batch" in names
    # Coarse map added 2026-07-31. A SECOND map at a different scale: the
    # scan-map analysis covers ±1.5 µm of piezo range in metres and one
    # generation at a time, while this one covers the whole sample in coarse
    # STEPS across every generation. "Where on the sample have we already
    # worked" cannot be answered from the first — after a lateral coarse move
    # those metre coordinates address a different patch entirely.
    assert "get_coarse_map" in names
    # Tip registry added 2026-07-31. Instrument-scoped, not experiment-scoped: a
    # tip outlives both the sample and the experiment, and which tip is in there
    # decides how conditioning must be done (a tungsten etched tip and a qPlus
    # sensor take nothing alike). See docs/v2/design/tip_registry_and_hardware_profile.md
    assert {"register_tip", "get_current_tip", "list_tips",
            "update_tip", "remove_current_tip"} <= names
    assert len(tools) == 44


def test_embedder_factory_substring_when_offline(monkeypatch):
    # no DashScope key + no local model → substring tier (embedder None)
    import mast.agents._shared.models as models
    monkeypatch.setattr(models, "load_provider_key", lambda p: "")
    from mast.memory import embeddings
    monkeypatch.setattr(embeddings, "_local_available", lambda: False)
    emb, dim, backend = embeddings.make_embedder()
    assert backend == "substring" and emb is None and dim == 0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
