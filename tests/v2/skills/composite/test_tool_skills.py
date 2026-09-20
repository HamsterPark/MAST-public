"""P3-A：策展 agent @tool → registry 技能包装。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_tool_skills.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np
import pytest
from langchain_core.tools import tool

from mast.core.registry import SkillRegistry
from mast.skills.composite.tool_skills import (
    WORKFLOW_TOOL_EXCLUDE,
    WORKFLOW_TOOL_EXPORTS,
    register_workflow_tool_skills,
    wrap_agent_tool,
)


@tool("toy_sum")
def toy_sum(a: float, b: float, label: str = "sum") -> str:
    """Add two numbers.

    Args:
        a: first number.
        b: second number.
        label: result label.
    """
    return f"{label}: {a + b}"


@tool("toy_fail")
def toy_fail(x: str) -> str:
    """Always fails like agent tools do (error string, no raise)."""
    return f"toy_fail failed: bad {x}"


class TestWrap:
    def test_metadata_mapping(self):
        cls = wrap_agent_tool(toy_sum, "data_processing")
        m = cls().metadata()
        assert m.name == "toy_sum"
        assert m.safety_level.name == "AUTO"
        assert "agent_tool" in m.tags and "data_processing" in m.tags
        p = {x.name: x for x in m.parameters}
        assert p["a"].type == "float" and p["a"].required
        assert p["label"].type == "str" and not p["label"].required
        assert p["label"].default == "sum"

    def test_execute_success(self):
        cls = wrap_agent_tool(toy_sum, "data_processing")
        res = cls().execute(None, {"a": 1.5, "b": 2.5})
        assert res.success and res.data["text"] == "sum: 4.0"

    def test_error_string_detected_as_failure(self):
        cls = wrap_agent_tool(toy_fail, "data_processing")
        res = cls().execute(None, {"x": "input"})
        assert res.success is False
        assert "toy_fail failed" in res.error
        assert res.data["text"].startswith("toy_fail failed")

    def test_run_numpy_snippet_is_banned(self):
        # data_processing exports run_numpy_snippet in AGENT_TOOLS, but it must be
        # hard-excluded from the auto-bridge (arbitrary code exec).
        assert "run_numpy_snippet" in WORKFLOW_TOOL_EXCLUDE
        assert "data_processing" in WORKFLOW_TOOL_EXPORTS

    def test_pyexec_tools_are_banned_too(self):
        """py_run / py_stage_data 同样不能进工作流菜单（2026-08-19）。

        py_run 跑任意 LLM 写的 Python（同 run_numpy_snippet 的理由）；
        py_stage_data 的「输出」是某个会话目录里多了个文件 —— 一个工作流步骤
        没法把那个接给下一步，放进菜单只会让人以为它可组合。
        """
        assert "py_run" in WORKFLOW_TOOL_EXCLUDE
        assert "py_stage_data" in WORKFLOW_TOOL_EXCLUDE


class TestRegistration:
    def test_register_auto_bridges_all_agents(self):
        reg = SkillRegistry()
        names = register_workflow_tool_skills(reg)
        # data_processing + literature (incl. LIBRARY_TOOLS) as before
        assert "load_scan" in names and "fft_2d" in names
        assert "search_papers" in names
        assert "lib_search" in names and "propose_citations" in names  # LIBRARY_TOOLS
        # NEW: paper_writing + paper_review tools now auto-bridged (were invisible)
        assert "draft_section" in names and "embed_figure" in names      # paper_writing
        assert "produce_review" in names and "check_methodology" in names  # paper_review
        # NEW: experiment_design's no-arg factory tools bridged (describe_skills omitted)
        assert "lookup_sample" in names and "query_past_experiments" in names
        assert "describe_skills" not in names      # meta-over-registry, excluded
        # code-exec stays barred
        assert not reg.has("run_numpy_snippet")
        assert "run_numpy_snippet" not in names
        # 经 ExecutionContext.run（安全收口）真实执行一个 DP 分析工具
        from mast.core.execution_context import ExecutionContext
        arr = np.random.default_rng(0).normal(size=(32, 32))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scan.npy"
            np.save(p, arr)
            ectx = ExecutionContext(pool=None, state=None, registry=reg)
            res = ectx.run("load_scan", {"path": str(p)})
            assert res.success, res.error
            assert "shape: (32, 32)" in res.data["text"]

    def test_collision_skipped(self):
        from mast.core.types import SkillCategory, SkillMetadata, SkillResult
        from mast.skills.base import BaseSkill

        class Existing(BaseSkill):
            def metadata(self):
                return SkillMetadata(name="load_scan", version="9.9.9",
                                     category=SkillCategory.READ,
                                     description="")
            def execute(self, ctx, params):
                return SkillResult(skill_name="load_scan", success=True)
        reg = SkillRegistry()
        reg.register(Existing)
        register_workflow_tool_skills(reg)
        assert reg.get("load_scan") is Existing   # 不覆盖既有技能


class TestCatalogIntegration:
    def test_source_and_domain(self, monkeypatch):
        from mast.webui import builder_api
        reg = SkillRegistry()
        register_workflow_tool_skills(reg)
        monkeypatch.setattr(builder_api, "_registry", reg)
        monkeypatch.setattr(builder_api, "_catalog_cache", None)
        cat = builder_api.get_catalog()
        e = next(x for x in cat["index"] if x["name"] == "fft_2d")
        assert e["source"] == "agent_tool"
        assert e["domain"] == "数据处理工具"
        lit = next(x for x in cat["index"] if x["name"] == "search_papers")
        assert lit["domain"] == "文献工具"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
