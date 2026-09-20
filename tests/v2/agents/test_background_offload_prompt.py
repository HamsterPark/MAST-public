"""The prompt guidance that makes agents PROACTIVELY use background offload.

The `spawn_background_task` tool + BackgroundRunManager give the CAPABILITY, but a
tool the model is never told about stays dormant. These pin the guidance so a
prompt refactor can't silently drop it and quietly kill 「边跑文献边跟 IC 对话」.
"""
from __future__ import annotations

from mast.agents.instrument_control.prompts import SYSTEM_PROMPT as IC_PROMPT
from mast.agents.orchestrator.graph import _ROUTER_PROMPT


def test_ic_prompt_teaches_background_offload():
    # names the actual tool the model must call
    assert "spawn_background_task" in IC_PROMPT
    # tells it WHEN (independent long side-task) and WHY (don't block instrument)
    assert "后台" in IC_PROMPT
    assert any(k in IC_PROMPT for k in ("独立", "不依赖"))
    # and the hard rule: IC itself is never backgroundable
    assert "instrument_control" in IC_PROMPT


def test_ic_prompt_coexists_with_existing_sections():
    """The offload section must be ADDED, not replace the units/safety guidance."""
    for anchor in ("METERS", "SafetyGateMiddleware", "ApproachTip",
                   "handoff_to_supervisor"):
        assert anchor in IC_PROMPT, f"offload edit clobbered the '{anchor}' section"


def test_router_prompt_mentions_backgrounding_without_dropping_pipeline():
    # the new background note is present …
    assert "spawn_background_task" in _ROUTER_PROMPT
    assert "BACKGROUND" in _ROUTER_PROMPT
    # … and the context-injection "drive the pipeline autonomously" section it must
    # coexist with is still intact (not overwritten)
    assert "autonomously" in _ROUTER_PROMPT
    assert "Parallel dispatch" in _ROUTER_PROMPT
