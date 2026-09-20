"""DP agent smoke tests — Phase 5 real implementations.

Builds the agent with a fake LLM, verifies compilation + tool list shape,
and exercises each real tool against a synthetic .npy / .dat fixture
created in tmp_path. End-to-end agent invocation drives a load_scan
tool_call through the graph.
"""
from __future__ import annotations

# ── path bootstrap (robust walk-up, mirrors XD pattern) ───────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found above " + str(Path(__file__).resolve()))


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
# Evict stale mast.* modules that may have resolved to v1 outside MASTv2.
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from mast.agents.data_processing.graph import build  # noqa: E402
from mast.agents.data_processing.tools import (  # noqa: E402
    AGENT_TOOLS,
    build_tools,
    detect_defects,
    fit_sts_peaks,
    fft_2d,
    load_scan,
    plane_subtract,
    run_numpy_snippet,
)


# Public capability contract, written independently of the runtime registry.
EXPECTED_PUBLIC_DOMAIN_NAMES = {
    "load_scan", "record_analysis", "plot_scan", "plot_spectrum",
    "fft_2d", "plane_subtract", "detect_defects", "fit_sts_peaks",
    "run_numpy_snippet", "py_stage_data", "py_run", "mosaic_scans",
    "find_flat_region", "assess_cluster_roundness", "analyze_scan_image",
    "auto_process_scan_batch", "get_latest_scan_file", "list_scan_dir", "glob_scans",
}


# ─────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────

class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel with no-op bind_tools so create_agent() won't crash."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _FakeBuf:
    """Minimal BufferService-like object satisfying make_buffer_tools' API."""

    def get_latest_tip_status(self):
        return (None, 0)

    def get_latest_progress(self):
        return (None, 0)

    def get_tip_history(self, since_seq: int):
        return []


@pytest.fixture
def fake_topo_npy(tmp_path):
    """Synthetic 64×64 STM topo: tilt + period-8 lattice + injected defects."""
    rng = np.random.default_rng(42)
    h, w = 64, 64
    yy, xx = np.indices((h, w))
    tilt = 0.001 * xx + 0.0005 * yy
    lattice = 0.05 * np.sin(2 * np.pi * xx / 8) * np.sin(2 * np.pi * yy / 8)
    rough = rng.normal(0, 0.005, (h, w))
    img = tilt + lattice + rough
    # Inject one bright + one dark defect, larger than min_size_px
    img[20:25, 20:25] += 0.5
    img[40:43, 40:43] -= 0.5
    p = tmp_path / "topo.npy"
    np.save(p, img.astype(np.float64))
    return str(p)


@pytest.fixture
def fake_sts_dat(tmp_path):
    """Synthetic STS curve with three Gaussian peaks at -0.5, 0.0, +0.5 V."""
    bias = np.linspace(-1.0, 1.0, 401)
    didv = (
        np.exp(-((bias + 0.5) / 0.05) ** 2) * 1.0
        + np.exp(-(bias / 0.05) ** 2) * 0.6
        + np.exp(-((bias - 0.5) / 0.05) ** 2) * 0.8
        + 0.01
    )
    arr = np.column_stack([bias, didv])
    p = tmp_path / "sts.dat"
    np.savetxt(p, arr)
    return str(p)


# ─────────────────────────────────────────────────────────────────────
# Test 1 — graph compiles
# ─────────────────────────────────────────────────────────────────────

class TestAgentCompiles:
    def test_build_returns_compiled_state_graph(self):
        """build() should return a LangGraph CompiledStateGraph, not a stub fn."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Analysis complete."),
        ]))
        agent = build(
            buf=None,
            model=fake_llm,
            checkpointer=InMemorySaver(),
        )
        assert agent is not None
        assert callable(getattr(agent, "invoke", None)), (
            "build() must return a CompiledStateGraph with .invoke; "
            "got a plain function (stub not replaced?)"
        )

    def test_build_without_checkpointer(self):
        """build() must succeed when no checkpointer is provided."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Analysis complete."),
        ]))
        agent = build(buf=None, model=fake_llm)
        assert agent is not None
        assert callable(getattr(agent, "invoke", None))


# ─────────────────────────────────────────────────────────────────────
# Test 2 — build_tools shape
# ─────────────────────────────────────────────────────────────────────

class TestBuildTools:
    def test_count_without_buf(self):
        """The public snapshot exposes 19 domain tools and two handoffs."""
        tools = build_tools(buf=None)
        assert len(tools) == 21, f"Unexpected public tool count: {[t.name for t in tools]}"
        assert {t.name for t in tools} == EXPECTED_PUBLIC_DOMAIN_NAMES | {
            "handoff_to_supervisor", "handoff_to_paper_writing"}

    def test_count_with_buf(self):
        """Adding a buffer contributes exactly the three shared read tools."""
        tools = build_tools(buf=_FakeBuf())
        assert len(tools) == 24, f"Unexpected buffered public tool count: {[t.name for t in tools]}"
        assert {t.name for t in tools} == EXPECTED_PUBLIC_DOMAIN_NAMES | {
            "handoff_to_supervisor", "handoff_to_paper_writing",
            "read_latest_tip_status", "get_scan_progress", "get_tip_history_since"}

    def test_domain_tool_names_present(self):
        """Every explicitly declared public capability must reach the built graph."""
        names = {t.name for t in build_tools(buf=None)}
        assert EXPECTED_PUBLIC_DOMAIN_NAMES <= names

    def test_handoff_names_present(self):
        """Both handoffs must be present."""
        tools = build_tools(buf=None)
        names = {t.name for t in tools}
        assert "handoff_to_supervisor" in names
        assert "handoff_to_paper_writing" in names

    def test_buffer_tool_names_when_buf_provided(self):
        """When buf is provided, the three buffer tools must appear."""
        tools = build_tools(buf=_FakeBuf())
        names = {t.name for t in tools}
        assert "read_latest_tip_status" in names
        assert "get_scan_progress" in names
        assert "get_tip_history_since" in names

    def test_agent_tools_constant_has_nineteen_entries(self):
        """No excluded tool or duplicate registration may enter the public domain list."""
        assert len(AGENT_TOOLS) == 19, [t.name for t in AGENT_TOOLS]

    def test_agent_tools_names(self):
        """The runtime names must exactly match the independent public contract."""
        names = {t.name for t in AGENT_TOOLS}
        assert names == EXPECTED_PUBLIC_DOMAIN_NAMES, (
            f"Public domain tools changed: {names ^ EXPECTED_PUBLIC_DOMAIN_NAMES}")


# ─────────────────────────────────────────────────────────────────────
# Test 3 — real tools on synthetic fixtures
# ─────────────────────────────────────────────────────────────────────

class TestRealTools:
    """Each tool runs on a tmp_path fixture and returns real numbers."""

    # ── load_scan ────────────────────────────────────────────────────

    def test_load_scan_reports_shape(self, fake_topo_npy):
        result = load_scan.invoke({"path": fake_topo_npy})
        assert "(64, 64)" in result, f"shape missing: {result!r}"
        assert "shape" in result
        assert "min" in result and "max" in result and "mean" in result

    def test_load_scan_reports_format(self, fake_topo_npy):
        result = load_scan.invoke({"path": fake_topo_npy})
        assert "npy" in result.lower()

    def test_load_scan_handles_missing_file(self):
        result = load_scan.invoke({"path": "/nonexistent/not_a_real_file.npy"})
        assert "failed" in result.lower() or "not found" in result.lower()

    # ── fft_2d ───────────────────────────────────────────────────────

    def test_fft_2d_reports_peaks(self, fake_topo_npy):
        result = fft_2d.invoke({"path": fake_topo_npy})
        assert "FFT 2D" in result
        assert "peak" in result.lower()
        assert "kx" in result and "ky" in result

    def test_fft_2d_default_top_n_lists_four(self, fake_topo_npy):
        """Default top_n_peaks=4 should print 4 peaks."""
        result = fft_2d.invoke({"path": fake_topo_npy})
        assert result.count("peak ") >= 4

    def test_fft_2d_explicit_top_n(self, fake_topo_npy):
        """top_n_peaks=2 should print exactly 2 peaks."""
        result = fft_2d.invoke({"path": fake_topo_npy, "top_n_peaks": 2})
        assert result.count("peak ") == 2

    # ── plane_subtract ───────────────────────────────────────────────

    def test_plane_subtract_reports_roughness(self, fake_topo_npy):
        result = plane_subtract.invoke({"path": fake_topo_npy})
        assert "RMS roughness" in result
        assert "order=1" in result

    def test_plane_subtract_default_order(self, fake_topo_npy):
        """plane_subtract must accept path alone (order has default 1)."""
        result = plane_subtract.invoke({"path": fake_topo_npy})
        assert isinstance(result, str) and len(result) > 0

    def test_plane_subtract_explicit_order_2(self, fake_topo_npy):
        result = plane_subtract.invoke({"path": fake_topo_npy, "order": 2})
        assert "order=2" in result
        assert "RMS roughness" in result

    def test_plane_subtract_invalid_order(self, fake_topo_npy):
        result = plane_subtract.invoke({"path": fake_topo_npy, "order": 5})
        assert "must be 1 or 2" in result

    # ── detect_defects ───────────────────────────────────────────────

    def test_detect_defects_reports_counts(self, fake_topo_npy):
        result = detect_defects.invoke({"path": fake_topo_npy})
        assert "bright protrusions" in result
        assert "dark spots" in result
        assert "px" in result

    def test_detect_defects_default_min_size(self, fake_topo_npy):
        result = detect_defects.invoke({"path": fake_topo_npy})
        assert isinstance(result, str) and len(result) > 0

    def test_detect_defects_high_min_size_filters_out(self, fake_topo_npy):
        """Unrealistically high min_size_px should yield zero defects."""
        result = detect_defects.invoke({"path": fake_topo_npy, "min_size_px": 100000})
        assert "0 (≥100000px)" in result

    def test_detect_defects_explicit_sigma(self, fake_topo_npy):
        result = detect_defects.invoke(
            {"path": fake_topo_npy, "sigma_threshold": 1.5}
        )
        assert "σ-threshold 1.5" in result

    # ── fit_sts_peaks ────────────────────────────────────────────────

    def test_fit_sts_peaks_finds_peaks(self, fake_sts_dat):
        result = fit_sts_peaks.invoke({"path": fake_sts_dat})
        assert "STS peaks" in result
        # Synthetic peaks at -0.5, 0.0, +0.5 — at least one bias entry must show
        assert "bias" in result.lower()

    def test_fit_sts_peaks_default_n_peaks(self, fake_sts_dat):
        result = fit_sts_peaks.invoke({"path": fake_sts_dat})
        # Default n_peaks=3 → at most 3 bias lines (find_peaks may give fewer)
        bias_lines = [l for l in result.splitlines() if "bias" in l.lower() and "=" in l]
        assert 1 <= len(bias_lines) <= 3

    def test_fit_sts_peaks_explicit_n_one(self, fake_sts_dat):
        result = fit_sts_peaks.invoke({"path": fake_sts_dat, "n_peaks": 1})
        bias_lines = [l for l in result.splitlines() if "bias" in l.lower() and "=" in l]
        assert len(bias_lines) == 1

    def test_fit_sts_peaks_handles_missing_file(self):
        result = fit_sts_peaks.invoke({"path": "/nonexistent/sts.dat"})
        assert "failed" in result.lower() or "not found" in result.lower()

    # ── run_numpy_snippet ────────────────────────────────────────────

    def test_run_numpy_snippet_basic_arithmetic(self):
        result = run_numpy_snippet.invoke({"code": "x = 1 + 2"})
        assert "sandbox ok" in result.lower()
        assert "x = 3" in result

    def test_run_numpy_snippet_compile_error(self):
        result = run_numpy_snippet.invoke({"code": "this is not valid python"})
        assert "error" in result.lower()

    def test_run_numpy_snippet_returns_string(self):
        result = run_numpy_snippet.invoke({"code": ""})
        assert isinstance(result, str) and len(result) > 0

    def test_run_numpy_snippet_blocks_dunder_attr(self):
        """_safe_getattr must reject attributes starting with underscore."""
        result = run_numpy_snippet.invoke({"code": "y = (1).__class__"})
        assert "error" in result.lower() or "AttributeError" in result


# ─────────────────────────────────────────────────────────────────────
# Test 4 — end-to-end agent invocation with fake LLM
# ─────────────────────────────────────────────────────────────────────

class TestAgentInvocation:
    """Compile DP with a fake LLM that emits a load_scan tool_call,
    then verify the tool executes against a real .npy fixture."""

    def _make_agent(self, messages_iter):
        fake_llm = _FakeChatModel(messages=messages_iter)
        return build(
            buf=None,
            model=fake_llm,
            checkpointer=InMemorySaver(),
        )

    def test_agent_invokes_without_error(self, fake_topo_npy):
        """Agent should complete without raising an exception."""
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_scan",
                    "args": {"path": fake_topo_npy},
                    "id": "tc-dp-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Scan loaded. Proceeding with FFT."),
        ]))
        result = agent.invoke(
            {"messages": [("user", f"Analyse scan at {fake_topo_npy}.")]},
            config={"configurable": {"thread_id": "smoke-dp-1"}},
        )
        assert result is not None
        assert "messages" in result

    def test_load_scan_tool_result_in_messages(self, fake_topo_npy):
        """After load_scan, the ToolMessage content should echo the path or shape."""
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_scan",
                    "args": {"path": fake_topo_npy},
                    "id": "tc-dp-2",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Loaded successfully."),
        ]))
        result = agent.invoke(
            {"messages": [("user", f"Load {fake_topo_npy}.")]},
            config={"configurable": {"thread_id": "smoke-dp-2"}},
        )
        all_content = "\n".join(
            str(m.content) for m in result["messages"] if hasattr(m, "content")
        )
        assert fake_topo_npy in all_content or "(64, 64)" in all_content, (
            f"Expected path or shape from load_scan in messages. Got:\n{all_content}"
        )

    def test_final_ai_message_present(self, fake_topo_npy):
        """Agent output must include at least one final AIMessage with text."""
        final_text = "RMS roughness analysis complete."
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_scan",
                    "args": {"path": fake_topo_npy},
                    "id": "tc-dp-3",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content=final_text),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Analyse.")]},
            config={"configurable": {"thread_id": "smoke-dp-3"}},
        )
        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert ai_messages, "No non-empty AIMessage found in result"
        all_ai_text = " ".join(m.content for m in ai_messages)
        assert final_text in all_ai_text or "Analysis" in all_ai_text

    def test_no_stub_fn_in_graph(self):
        """Agent must not be a bare function (indicates Phase-3 stub not replaced)."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Done."),
        ]))
        agent = build(buf=None, model=fake_llm)
        assert hasattr(agent, "get_graph"), (
            "build() returned a plain function — Phase-3 stub was not replaced"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
