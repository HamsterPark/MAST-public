"""BrainstormState — self-contained TypedDict for the P4 brainstorm subgraph.

Design: docs/v2/design/agentic-cognition.md §4 (头脑风暴 / 预热).

This graph deliberately does NOT reuse MASTState — a facilitated multi-viewpoint
discussion has nothing to do with the hardware/skill orchestration state, and
keeping it isolated means the brainstorm graph never leaks into (or out of) the
6-agent orchestrator's checkpoint. Every field here is plain JSON
(str / int / list / dict of scalars) so the whole state could be serialised into
a LangGraph checkpoint without ever holding a tensor / ndarray / handle / socket
/ Nanonis client.

Honesty (hard rule, mirrors dreaming.py): a brainstorm is **inferred discussion,
not measured fact** — the prompts and the final summary carry a visible "非实测"
label, and the summary lands ONLY in the memory store (kind="brainstorm"), never
the experiment record or the knowledge base.
"""

from __future__ import annotations

from typing import TypedDict


class Turn(TypedDict):
    """One line of the discussion transcript (all-JSON, no objects)."""
    speaker: str   # display name, e.g. "主持人" / "实验设计" / "用户"
    role: str      # logical role key: "facilitator" / a viewpoint key / "user"
    content: str
    round: int     # which round this line belongs to (facilitator agenda = 0)


class BrainstormState(TypedDict, total=False):
    """State for one facilitated brainstorm session.

    All fields are JSON-serialisable. ``grounding`` is the read-only context the
    viewpoint agents may consult — experiment skill counts / statuses + memory
    excerpts — but it is *text/scalars only*; viewpoint agents get NO hardware
    or skill-write tools, so there is nothing in here that could touch the
    instrument.
    """

    # ── inputs ────────────────────────────────────────────────────────
    topic: str                      # discussion topic / question
    experiment_id: str              # the experiment under discussion ("" = none)
    grounding: dict                 # {experiment, skill_counts, statuses,
                                    #  memory[], actions_total, ...} — all JSON
    user_viewpoints: list           # list[str] — opinions the user injected
    max_rounds: int                 # convergence cap (facilitator decides earlier)

    # ── working / outputs ─────────────────────────────────────────────
    round: int                      # current round index (1-based once speaking)
    agenda: list                    # list[str] — facilitator's per-round agenda
    transcript: list                # list[Turn]
    summary: str                    # final synthesised summary (carries 非实测 tag)
    done: bool                      # facilitator declared convergence


__all__ = ["BrainstormState", "Turn"]
