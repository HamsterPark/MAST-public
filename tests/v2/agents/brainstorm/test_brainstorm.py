"""Brainstorm (P4) — facilitated multi-viewpoint discussion backend.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/brainstorm/test_brainstorm.py -q -p no:randomly
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── bootstrap: ensure the v2 MASTv2/ tree shadows the repo-root v1 mast/ ──
_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest

from mast.agents.brainstorm import (
    BRAINSTORM_TAG,
    gather_grounding,
    run_brainstorm,
    run_discussion,
)
from mast.agents.brainstorm.state import BrainstormState
from mast.core.types import ActionRecord, SkillResult
from mast.logging.storage import ExperimentStorage
from mast.memory.store import MemoryStore


# ── fixtures ───────────────────────────────────────────────────────────

def _make_experiment(db_path: str) -> str:
    """Synthesise an experiment with a few actions + a sample."""
    st = ExperimentStorage(db_path)
    exp = st.create_experiment("Au(111) 表征", "研究 Au(111) 表面台阶与重构")
    st.create_sample(exp, "Au-sample-1", "单晶 Au(111)", "clean_metal", "Au(111)")
    for sk in ["SetBias", "SetBias", "Scan", "Scan", "Scan", "BiasSpectr"]:
        st.log_action(ActionRecord(
            experiment_id=exp, skill_name=sk,
            result=SkillResult(skill_name=sk, success=True)))
    return exp


def _make_memory(db_path: str) -> MemoryStore:
    mem = MemoryStore(db_path)
    mem.write("global", "notes/au111.md",
              "Au(111) 的 herringbone 重构典型周期约 6 nm。",
              title="Au(111) 重构", kind="note")
    mem.write("global", "insights/drift.md",
              "低温下漂移主要来自热平衡未达。", title="漂移来源", kind="insight")
    return mem


class _FakeLLM:
    """Minimal langchain-style chat model: .invoke([...]) -> object with .content.

    Returns a deterministic, viewpoint-agnostic line so we can exercise the LLM
    branch without a network/API key.
    """

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        # last message is the human prompt
        prompt = ""
        for m in messages:
            prompt = getattr(m, "content", "") or prompt
        text = "这是一条头脑风暴设想发言:基于现有记录提出下一步并指出一个分歧点。"

        class _Msg:
            content = text
        return _Msg()


# ── tests ──────────────────────────────────────────────────────────────

def test_grounding_reads_real_record(tmp_path):
    """gather_grounding reports REAL skill counts / statuses (no fabrication)."""
    db = str(tmp_path / "exp.db")
    exp = _make_experiment(db)
    mem = _make_memory(db)
    g = gather_grounding(db, exp, memory_store=mem, topic="重构")
    assert g["experiment"]["name"] == "Au(111) 表征"
    assert g["actions_total"] == 6
    assert g["skill_counts"]["Scan"] == 3
    assert g["skill_counts"]["SetBias"] == 2
    # memory pulled in
    paths = {m["path"] for m in g["memory"]}
    assert "notes/au111.md" in paths
    # samples surfaced
    assert any(s["name"] == "Au-sample-1" for s in g["samples"])


def test_run_brainstorm_rule_based_multi_viewpoint(tmp_path):
    """llm=None → non-empty transcript with multiple distinct viewpoints + a summary."""
    db = str(tmp_path / "exp.db")
    exp = _make_experiment(db)
    out = run_brainstorm(db, exp, topic="下一步该测什么", max_rounds=2, llm=None)
    transcript = out["transcript"]
    summary = out["summary"]
    assert transcript, "transcript must be non-empty"
    assert summary.strip(), "summary must be non-empty"
    # facilitator present
    roles = {t["role"] for t in transcript}
    assert "facilitator" in roles
    # at least 5 distinct viewpoint roles spoke
    viewpoint_roles = roles - {"facilitator", "user"}
    assert len(viewpoint_roles) >= 5, f"expected >=5 viewpoints, got {viewpoint_roles}"
    # every viewpoint line is honesty-tagged (非实测)
    for t in transcript:
        if t["role"] not in ("facilitator", "user"):
            assert BRAINSTORM_TAG in t["content"]


def test_user_viewpoints_injected_into_transcript(tmp_path):
    """User opinions appear as 'user' turns and are referenced in the summary."""
    db = str(tmp_path / "exp.db")
    exp = _make_experiment(db)
    uv = ["我怀疑针尖状态不好", "应该先做一次 STS 谱"]
    out = run_brainstorm(db, exp, topic="排查问题", user_viewpoints=uv,
                         max_rounds=1, llm=None)
    transcript = out["transcript"]
    user_turns = [t for t in transcript if t["role"] == "user"]
    assert len(user_turns) == 2
    assert any("针尖" in t["content"] for t in user_turns)
    # summary acknowledges user viewpoints
    assert "用户观点" in out["summary"]
    assert "STS" in out["summary"] or "针尖" in out["summary"]


def test_memory_store_writes_brainstorm_kind_with_label(tmp_path):
    """Passing memory_store writes a kind='brainstorm' entry carrying 非实测."""
    db = str(tmp_path / "exp.db")
    exp = _make_experiment(db)
    mem = _make_memory(db)
    before = len(mem.list("global", kind="brainstorm"))
    out = run_brainstorm(db, exp, topic="台阶密度", max_rounds=1,
                         llm=None, memory_store=mem)
    after = mem.list(kind="brainstorm")
    assert len(after) == before + 1
    entry = after[0]
    assert entry["kind"] == "brainstorm"
    assert "非实测" in entry["content"]
    assert BRAINSTORM_TAG in entry["content"]
    # the written content matches the returned summary core
    assert "台阶密度" in entry["content"] or "台阶密度" in out["summary"]
    # tagged for filtering
    assert "brainstorm" in entry["tags"]


def test_empty_experiment_does_not_crash(tmp_path):
    """No experiment / empty DB still produces a valid (thin) discussion."""
    db = str(tmp_path / "empty.db")
    ExperimentStorage(db)  # create empty schema only
    out = run_brainstorm(db, experiment_id=None, topic="泛泛而谈",
                         max_rounds=1, llm=None)
    assert out["transcript"], "even empty grounding yields a transcript"
    assert out["summary"].strip()
    # grounding reflects zero actions truthfully (no fabricated numbers)
    g = gather_grounding(db, None)
    assert g["actions_total"] == 0
    assert g["skill_counts"] == {}


def test_max_rounds_is_respected(tmp_path):
    """The number of facilitator round-summaries equals max_rounds."""
    db = str(tmp_path / "exp.db")
    exp = _make_experiment(db)

    def _count_round_summaries(transcript):
        # facilitator turns with round>=1 are the per-round summaries
        # (round 0 facilitator turn is the opening agenda).
        return len([t for t in transcript
                    if t["role"] == "facilitator" and t["round"] >= 1])

    out1 = run_brainstorm(db, exp, topic="t", max_rounds=1, llm=None)
    out3 = run_brainstorm(db, exp, topic="t", max_rounds=3, llm=None)
    assert _count_round_summaries(out1["transcript"]) == 1
    assert _count_round_summaries(out3["transcript"]) == 3
    # more rounds => strictly more viewpoint turns
    vp1 = len([t for t in out1["transcript"] if t["role"] not in ("facilitator", "user")])
    vp3 = len([t for t in out3["transcript"] if t["role"] not in ("facilitator", "user")])
    assert vp3 > vp1


def test_llm_branch_used_when_model_given(tmp_path):
    """A fake chat model is actually invoked and its text feeds the transcript."""
    db = str(tmp_path / "exp.db")
    exp = _make_experiment(db)
    llm = _FakeLLM()
    out = run_brainstorm(db, exp, topic="用 LLM 跑", max_rounds=1, llm=llm)
    assert llm.calls > 0, "the model must be invoked"
    # the fake model's text shows up in at least one viewpoint line
    assert any("头脑风暴设想发言" in t["content"]
               for t in out["transcript"] if t["role"] not in ("facilitator", "user"))
    # honesty banner still enforced on viewpoint lines
    for t in out["transcript"]:
        if t["role"] not in ("facilitator", "user"):
            assert BRAINSTORM_TAG in t["content"]


def test_state_is_json_serialisable(tmp_path):
    """The full graph output is plain JSON (checkpoint-safe: no objects)."""
    import json
    db = str(tmp_path / "exp.db")
    exp = _make_experiment(db)
    out = run_brainstorm(db, exp, topic="序列化", max_rounds=2, llm=None)
    # round-trips through JSON without custom encoders
    blob = json.dumps(out, ensure_ascii=False)
    again = json.loads(blob)
    assert again["transcript"] and again["summary"]


def test_the_discussion_runs_without_any_graph():
    """讨论是一个普通的循环，不需要编译、不需要 checkpointer。

    2026-08-27 取代 ``test_build_compiles_without_checkpointer``：那条钉的是
    「``StateGraph`` 编译得过」，而那张图已经被展开成 :func:`run_discussion`
    （四个步骤本来就是纯变换，图除了一个 while 循环之外没表达任何东西）。
    这条钉的是同一件事的**行为**：给它一个初始 state，它跑得完并给出结论。
    """
    from mast.agents.brainstorm.graph import run_discussion

    state = {
        "topic": "不用图也能讨论", "experiment_id": "", "grounding": {},
        "user_viewpoints": [], "max_rounds": 1, "round": 0,
        "agenda": [], "transcript": [], "summary": "", "done": False,
    }
    out = run_discussion(state, llm=None)

    assert out["transcript"], "要有讨论内容"
    assert out["summary"], "要有综述"
    assert out.get("done") is True, "主持人应当在 max_rounds 之后收敛"


def test_a_facilitator_that_never_converges_still_terminates(caplog):
    """★ 兜底上限要**说话**，不能悄悄截断。

    这条替代的是原来的 ``recursion_limit`` —— 同一个作用，但单位是**轮数**而不是
    图的超步数，所以它不会因为将来给讨论加一个环节而悄悄变小。

    一个悄悄被截断的讨论，读的人无从知道它被截断了 —— 所以除了「会停」，
    还要断言「它说了自己为什么停」。
    """
    import logging

    from mast.agents.brainstorm import graph as g

    state = {
        "topic": "永不收敛", "experiment_id": "", "grounding": {},
        "user_viewpoints": [], "max_rounds": 10_000,   # 主持人永远不置 done
        "round": 0, "agenda": [], "transcript": [], "summary": "", "done": False,
    }
    with caplog.at_level(logging.WARNING, logger=g.__name__):
        out = g.run_discussion(state, llm=None)

    assert out["summary"], "撞了上限也要给出综述，而不是空手而归"
    assert out["round"] <= g._MAX_ROUNDS_HARD_CAP + 1, (
        f"轮数必须被 {g._MAX_ROUNDS_HARD_CAP} 的硬上限拦住")
    assert any("hard cap" in r.message for r in caplog.records), (
        "撞上限必须留一条 warning —— 悄悄截断的讨论看起来和正常结束一模一样")


def test_no_hardware_or_skill_tools_imported():
    """The brainstorm graph module must not pull in any instrument/skill tool."""
    import mast.agents.brainstorm.graph as g
    src = Path(g.__file__).read_text(encoding="utf-8")
    # no nanonis / connection / skill-wrapping imports
    for forbidden in ("nanonis", "wrap_skill", "ConnectionPool", "instrument_control"):
        assert forbidden not in src, f"brainstorm graph must not reference {forbidden!r}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
