"""pipeline.main smoke — environment build + orchestrator invocation w/ fake hardware."""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path
def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 not found")

_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import asyncio
import pytest

from mast.pipeline.main import _build_minimal_environment, _make_fake_pool, _make_fake_state, main


class TestEnvironmentBuild:
    def test_no_hardware_no_buffer_builds(self):
        buf, ctx_provider, registry = _build_minimal_environment(no_hardware=True, no_buffer=True)
        assert buf is None
        assert callable(ctx_provider)
        # Test the context_provider produces a fresh ExecutionContext
        ctx = ctx_provider()
        assert hasattr(ctx, "safe_call")
        assert hasattr(ctx, "run")
        # Registry has skills (200 ported)
        assert len(registry.list_skills()) >= 100

    def test_fake_pool_safe_call(self):
        pool = _make_fake_pool()
        rec = pool.safe_call("Bias_Get")
        assert rec.method == "Bias_Get"
        assert rec.error == ""
        assert rec.return_value == ("", b"", [0.0])

    def test_fake_state_snapshot(self):
        state = _make_fake_state()
        snap = state.snapshot()
        assert snap.bias_v == 0.1
        assert snap.z_controller_on is True


class TestCliMain:
    def test_help_exits_cleanly(self, capsys):
        # argparse exits with SystemExit(0) on --help
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestPipelineMain:
    """End-to-end pipeline test using fake hardware. Skipped unless ANTHROPIC_API_KEY
    is configured because supervisor tries to instantiate ChatAnthropic."""

    def test_no_instruction_no_gui_help_path(self):
        # No instruction, no GUI → just print help, return 0
        rc = main([])
        assert rc == 0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
