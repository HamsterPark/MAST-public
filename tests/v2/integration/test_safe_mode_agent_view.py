"""What the agent actually SEES in SAFE mode — the end-to-end claim.

The individual overrides are pinned in their own unit tests. This file asserts
the thing they exist for: with a backend that calls every tip bad, an agent
running in SAFE mode finds **no evidence of a bad tip anywhere it can look** —
tool results, buffer state, scan results — and nothing halts or interrupts it.

That claim is what makes SAFE coherent. Before 2026-08-01 the belief block told
the model "the tip is fine" while every one of these surfaces said otherwise,
and the model was left arguing with its own tools (and getting halted mid-scan
by the very verdict it was told to ignore).
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
    raise RuntimeError("MASTv2 dir not found")


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

import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from mast.core.operating_mode import bind_mode_source  # noqa: E402
from mast.vision.module import VisionModule  # noqa: E402
from mast.vision.scan_monitor import ScanVisionMonitor  # noqa: E402


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    bind_mode_source(None)
    VisionModule._instance = None


@pytest_asyncio.fixture
async def buf(tmp_path):
    from mast.buffer.service import BufferService
    b = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


def _mock_vision(monkeypatch):
    """The mock backend calls every tip bad, unconditionally — the worst case
    for SAFE, and the state a degraded (no-weights) install is really in."""
    monkeypatch.setenv("MAST_VISION_BACKEND", "mock")
    VisionModule._instance = None
    return VisionModule.get()


def _publish_one_assessment(vm, buf, scan_id="e2e"):
    mon = ScanVisionMonitor(None, scan_id=scan_id, buffer=buf)
    coarse = vm.assess_tip_coarse(
        np.random.RandomState(0).randn(64, 64).astype(np.float32))
    mon._publish_coarse(buf, coarse, ordinal=1)
    return coarse


@pytest.mark.asyncio
async def test_agent_tool_view_has_no_bad_tip_in_safe(monkeypatch, buf):
    """`read_latest_tip_status` is the agent's own window onto the tip. In SAFE
    it must not hand back the word the belief block just told it to disbelieve."""
    from mast.agents._shared.buffer_tools import make_buffer_tools

    bind_mode_source(lambda: "safe")
    vm = _mock_vision(monkeypatch)
    _publish_one_assessment(vm, buf)

    tools = {t.name: t for t in make_buffer_tools(buf)}
    out = tools["read_latest_tip_status"].invoke({})

    assert out["tip"]["quality"] == "good"
    # Nothing anywhere in the serialised tool result may re-introduce the verdict
    # (this is what a raw field on TipStatus would have done — see the design doc).
    assert "bad" not in json.dumps(out, default=str).lower()


@pytest.mark.asyncio
async def test_same_view_is_honest_outside_safe(monkeypatch, buf):
    """The mirror: without SAFE the agent sees the real verdict. If this ever
    goes green while the test above does too, the override is unconditional."""
    from mast.agents._shared.buffer_tools import make_buffer_tools

    bind_mode_source(lambda: "auto")
    vm = _mock_vision(monkeypatch)
    _publish_one_assessment(vm, buf)

    tools = {t.name: t for t in make_buffer_tools(buf)}
    assert tools["read_latest_tip_status"].invoke({})["tip"]["quality"] == "bad"


@pytest.mark.asyncio
async def test_no_critical_event_and_no_halt_in_safe(monkeypatch, buf):
    """A full assessment in SAFE produces no CRITICAL anywhere in the buffer, so
    neither the composite halt hook nor the HITL gate (both CRITICAL-only) can
    fire. This is the "SAFE stops interrupting the experiment" claim itself."""
    from mast.buffer.schemas import Severity

    bind_mode_source(lambda: "safe")
    vm = _mock_vision(monkeypatch)
    _publish_one_assessment(vm, buf)

    events = buf.get_event_history(since_seqno=-1, limit=200)
    assert not [e for e in events if e.severity is Severity.CRITICAL]


@pytest.mark.asyncio
async def test_full_scan_verdict_carries_no_repair_note_in_safe(monkeypatch, buf):
    """`full_scan` attaches a `vision_note` telling the agent NOT to call the
    image normal and to consider ConditionTip. It reads the TipStatus, so the
    facade override should already have removed the trigger — pin that, because
    this note lands directly in the scan result the agent reads."""
    from mast.buffer.active import set_active_buffer
    from mast.skills.composite.full_scan import FullScan

    bind_mode_source(lambda: "safe")
    vm = _mock_vision(monkeypatch)
    _publish_one_assessment(vm, buf)

    set_active_buffer(buf)
    try:
        verdict = FullScan._vision_verdict()
    finally:
        set_active_buffer(None)

    assert verdict.get("vision_tip_quality") == "good"
    assert "vision_note" not in verdict
