"""v2 unit tests for mast.llm.quickask.QuickAskAgent.

Verifies:
  * READ + ANALYSIS skills are surfaced; WRITE skills are filtered out
  * State-mutating meta-tools are excluded from the offered tool list
  * Tool-use loop dispatches read meta-tools without touching real services
  * Defensive guard rejects WRITE skill names that the LLM hallucinates

Ported from tests/unit/test_quickask.py (v1). v2 mast.llm.quickask is a
mirror copy of v1 with identical API surface; this test exercises the
v2 import path under MASTv2/mast/ to guard against silent regressions.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/llm/ -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
_REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
# Remove any sys.path entry whose resolved Path equals the repo root,
# regardless of separator style (Windows: D:\... vs D:/...).
sys.path[:] = [
    p for p in sys.path
    if not p or Path(p).resolve() != _REPO_ROOT_PATH
]
while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)
# Purge any already-loaded v1 mast.* modules
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.core.types import (
    SkillCategory, SkillMetadata, ParameterSpec, SkillResult,
)
from mast.llm.quickask import QuickAskAgent, _READ_META_TOOLS


# ── Fakes ────────────────────────────────────────────────────────────

class FakeRegistry:
    def __init__(self, metas: list[SkillMetadata]):
        self._metas = metas

    def list_skills(self):
        return list(self._metas)

    def to_tool_definitions(self):
        out = []
        for m in self._metas:
            props = {}
            req = []
            for p in m.parameters:
                props[p.name] = {"type": "number"}
                if p.required:
                    req.append(p.name)
            out.append({
                "name": m.name,
                "description": m.description,
                "input_schema": {"type": "object", "properties": props, "required": req},
            })
        return out


class FakeExecutor:
    def __init__(self, response_data=None):
        self.calls: list[tuple[str, dict]] = []
        self._response_data = response_data or {"ok": True}

    def run(self, skill_name, params, approval_source=""):
        self.calls.append((skill_name, params))
        return SkillResult(
            skill_name=skill_name, success=True,
            data=self._response_data, elapsed_s=0.01,
        )


class FakeState:
    def snapshot(self):
        return None


class ScriptedClient:
    """Returns a pre-scripted sequence of LLM responses."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.calls: list[dict] = []

    def chat(self, messages, system="", tools=None):
        self.calls.append({
            "messages": list(messages),
            "tools": [t["name"] for t in (tools or [])],
        })
        if not self._responses:
            return {"content": [{"type": "text", "text": "(no more)"}]}
        return self._responses.pop(0)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_skills() -> list[SkillMetadata]:
    return [
        SkillMetadata(
            name="GetBias", description="read bias",
            category=SkillCategory.READ, parameters=[],
        ),
        SkillMetadata(
            name="GetCurrent", description="read current",
            category=SkillCategory.READ, parameters=[],
        ),
        SkillMetadata(
            name="SetBias", description="set bias",
            category=SkillCategory.WRITE,
            parameters=[ParameterSpec(name="bias_v", type="float", required=True)],
        ),
        SkillMetadata(
            name="StartScan", description="start scan",
            category=SkillCategory.WRITE, parameters=[],
        ),
        SkillMetadata(
            name="Smooth", description="numpy smoothing",
            category=SkillCategory.ANALYSIS, parameters=[],
        ),
        SkillMetadata(
            name="FullScan", description="composite",
            category=SkillCategory.COMPOSITE, parameters=[],
        ),
    ]


def _agent(client) -> QuickAskAgent:
    return QuickAskAgent(
        client=client,
        executor=FakeExecutor(),
        registry=FakeRegistry(_make_skills()),
        state=FakeState(),
    )


# ── Tests: tool filtering ────────────────────────────────────────────

def test_only_read_and_analysis_skills_surfaced():
    ag = _agent(ScriptedClient([]))
    tools, allowed = ag._build_read_only_skill_tools()
    names = {t["name"] for t in tools}
    assert names == {"GetBias", "GetCurrent", "Smooth"}
    assert allowed == names


def test_write_and_composite_skills_excluded():
    ag = _agent(ScriptedClient([]))
    _, allowed = ag._build_read_only_skill_tools()
    assert "SetBias" not in allowed
    assert "StartScan" not in allowed
    # Composite skills are excluded too — they may chain WRITE actions.
    assert "FullScan" not in allowed


def test_meta_tool_whitelist_excludes_state_mutators():
    ag = _agent(ScriptedClient([]))
    meta_names = {t["name"] for t in ag._build_meta_tools()}
    forbidden = {
        "start_experiment", "end_experiment",
        "start_sample", "end_sample",
        "create_plan", "execute_plan",
        "resume_plan", "pause_plan", "abort_plan",
        "mark_area_used",
    }
    for name in forbidden:
        assert name not in meta_names, f"{name} should not be in QuickAsk meta tools"


def test_meta_tool_whitelist_has_essential_lookups():
    ag = _agent(ScriptedClient([]))
    meta_names = {t["name"] for t in ag._build_meta_tools()}
    essential = {
        "get_workflow_advice",
        "get_skill_guidance",
        "query_knowledge",
        "search_local_corpus",
        "lookup_glossary",
        "get_material_coverage",
        "get_fault_diagnosis",
        "get_noise_reference",
    }
    missing = essential - meta_names
    assert not missing, f"missing essential read meta-tools: {missing}"


def test_read_meta_tools_constant_matches_builder():
    ag = _agent(ScriptedClient([]))
    builder_names = {t["name"] for t in ag._build_meta_tools()}
    # The frozenset constant should be a subset of what the builder emits.
    # (Future builder additions are allowed; constant is a "must-include" set.)
    assert _READ_META_TOOLS.issubset(builder_names)


# ── Tests: tool-use dispatch ─────────────────────────────────────────

def test_text_only_response_returned_as_is():
    client = ScriptedClient([
        {"content": [{"type": "text", "text": "你好，我是查询助手。"}]},
    ])
    ag = _agent(client)
    out = ag.one_shot("hi")
    assert out == "你好，我是查询助手。"
    assert len(client.calls) == 1


def test_read_skill_dispatched_through_executor():
    executor = FakeExecutor(response_data={"bias_v": 0.42})
    client = ScriptedClient([
        {"content": [{"type": "tool_use", "id": "t1", "name": "GetBias", "input": {}}]},
        {"content": [{"type": "text", "text": "当前偏压 0.42 V。"}]},
    ])
    ag = QuickAskAgent(
        client=client, executor=executor,
        registry=FakeRegistry(_make_skills()), state=FakeState(),
    )
    out = ag.one_shot("查偏压")
    assert "0.42" in out
    assert executor.calls == [("GetBias", {})]


def test_hallucinated_write_skill_is_rejected():
    # The LLM tries to call SetBias even though it's not in the tool list.
    executor = FakeExecutor()
    client = ScriptedClient([
        {"content": [{"type": "tool_use", "id": "t1", "name": "SetBias",
                      "input": {"bias_v": 0.5}}]},
        {"content": [{"type": "text", "text": "已拒绝。"}]},
    ])
    ag = QuickAskAgent(
        client=client, executor=executor,
        registry=FakeRegistry(_make_skills()), state=FakeState(),
    )
    ag.one_shot("set bias to 0.5 V")
    # Executor must NOT have run SetBias.
    assert executor.calls == []
    # The second LLM call should have received an error tool_result for SetBias.
    second_msgs = client.calls[1]["messages"]
    last = second_msgs[-1]
    assert last["role"] == "user"
    block = last["content"][0]
    assert block["type"] == "tool_result"
    assert "read-only" in block["content"]


def test_max_steps_terminates_loop():
    # LLM keeps calling tools forever.
    client = ScriptedClient([
        {"content": [{"type": "tool_use", "id": f"t{i}",
                      "name": "GetBias", "input": {}}]}
        for i in range(20)
    ])
    ag = _agent(client)
    out = ag.one_shot("loop forever", max_steps=3)
    assert "查询助手" in out or "最大步数" in out
    # Only 3 LLM round-trips allowed.
    assert len(client.calls) == 3


def test_empty_query_returns_placeholder():
    ag = _agent(ScriptedClient([]))
    assert "(empty query)" in ag.one_shot("")
    assert "(empty query)" in ag.one_shot("   ")


# ── Tests: meta tool dispatch (read-only) ───────────────────────────

def test_lookup_glossary_dispatch(monkeypatch):
    """Verify lookup_glossary calls the glossary module without side effects."""
    captured: dict = {}

    def fake_lookup(term, k=8):
        captured["term"] = term
        captured["k"] = k
        return [{"canonical_en": "STM", "abbrev": "STM", "zh": ["扫描隧道显微镜"],
                 "domain": "instrumentation"}]

    def fake_format(matches):
        return "fake formatted glossary"

    monkeypatch.setattr("mast.knowledge.glossary.lookup", fake_lookup)
    monkeypatch.setattr("mast.knowledge.glossary.format_for_prompt", fake_format)

    ag = _agent(ScriptedClient([]))
    out = ag._handle_meta_tool("lookup_glossary", {"term": "STM"})
    assert out["success"] is True
    assert out["matches"] == "fake formatted glossary"
    assert captured["term"] == "STM"


def test_unknown_tool_returns_error():
    ag = _agent(ScriptedClient([]))
    out = ag._handle_meta_tool("nonexistent_tool", {})
    assert out["success"] is False
    assert "unknown tool" in out["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
