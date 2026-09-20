"""brainstorm — P4 cognitive layer: facilitated multi-viewpoint discussion.

A GUI-callable orchestration where a facilitator and ~6 viewpoint agents discuss
the current experiment + progress (the user may inject opinions), producing a
discussion transcript plus a synthesised summary that can be written to memory.

Isolated from the 6-agent orchestrator: its own ``BrainstormState`` (NOT
MASTState), no hardware/skill tools, all-JSON state, non-blocking steps, and an
offline rule-based path when ``llm=None``. Honesty: every line and the summary
carry a visible 非实测 banner.

Public surface::

    from mast.agents.brainstorm import run_brainstorm, BRAINSTORM_TAG

2026-08-27：``build`` 已移除。它返回的是一张编译好的 ``StateGraph``，而那张图除了
「一个 while 循环」之外没有表达任何东西（见 ``graph.py`` 模块文档）。生产代码从来
只调 ``run_brainstorm``；要直接驱动讨论用 ``run_discussion(state, llm=…)``。
"""

from __future__ import annotations

from .graph import (
    BRAINSTORM_TAG,
    gather_grounding,
    grounding_to_text,
    run_brainstorm,
    run_discussion,
)
from .state import BrainstormState, Turn

__all__ = [
    "run_brainstorm",
    "run_discussion",
    "gather_grounding",
    "grounding_to_text",
    "BrainstormState",
    "Turn",
    "BRAINSTORM_TAG",
]
