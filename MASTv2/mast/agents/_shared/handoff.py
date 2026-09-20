"""Handoff tool factory — canonical way for agents to transfer control.

Usage inside an agent's tool list:
    from mast.agents._shared.handoff import make_handoff
    tools = [
        ...domain_tools,
        make_handoff("data_processing", "Hand scan results to analysis."),
        make_handoff("supervisor", "Return control to orchestrator."),
    ]

The returned tool is a LangChain @tool that, when called by the model, returns
a `Command(goto="supervisor", graph=Command.PARENT, update={...})`. EVERY
handoff routes back through the parent ``supervisor`` node — never directly to a
sibling subgraph.

Why route through the supervisor:
  The orchestrator's loop / budget guard (visit_count cap, budget hard-gate)
  lives ONLY in ``supervisor_node``. The previous implementation returned
  ``Command(goto=<sibling>, graph=Command.PARENT)`` for sibling targets, which
  LangGraph routes straight from one agent node to another in the parent graph —
  completely bypassing the supervisor and therefore its guards. An A→B→A→B
  sibling ping-pong (e.g. paper_writing↔paper_review revision loops) was
  unbounded by anything except the graph-level recursion_limit. By always
  going through ``supervisor``, the guard runs on EVERY inter-agent hop. The
  intended next agent is preserved as a ``routing_hint`` in state so the
  supervisor honours the agent's choice (after its guards) without re-asking the
  LLM router — no routing behaviour is lost, only the unguarded shortcut.

Handoff is the ONLY inter-agent communication primitive. Direct imports between
agent packages are forbidden and enforced by agent_boundary 钩子（不随仓）.

THE CUSTOMS DESK (2026-07-29)
-----------------------------
``graph=Command.PARENT`` short-circuits out of the agent subgraph: only
``command.update`` reaches the parent. Everything else the agent did — its prose,
every tool call, every tool result — stays in the subgraph namespace and is gone.
That is correct LangGraph behaviour and we are not fighting it; it just means
this tool is the ONLY place an artifact can cross the boundary.

So the handoff now copies the agent's non-empty artifact fields into that update
(``artifact_channel.CARRIED_FIELDS``). ``reason`` stays what it always was — a
sentence for the human reading the transcript — and stops being the sole carrier
of the work, which it was never shaped to be (no schema, no length floor, and
compaction is allowed to summarise it away; ``paper_writing``'s prompt had to
explicitly tell the model not to trust it).

Requires the agent subgraph to be compiled with ``state_schema=MASTState`` —
without it those channels do not exist in the subgraph, the writing tool's
``Command(update=...)`` is silently dropped, and this desk finds nothing to
carry. (Verified empirically 2026-07-29: with no ``state_schema`` a tool update
to an undeclared key vanishes without an error. That is also why the skill
adapter's ``executed_skills`` / ``composite_progress`` writes never landed.)

NOT done here, deliberately: synthesising the paired ``AIMessage`` so the parent
never sees an orphan tool result. ``tool_pair_guard_mw`` already repairs those
losslessly at the last boundary before the model AND rescues orphans sitting in
checkpoints written before it existed; adding a second, partial fix here would
put an extra assistant message into every operator-visible transcript to solve a
problem that is already solved.
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from mast.agents._shared.artifact_channel import carried_from


def make_handoff(target: str, description: str):
    """Return a @tool that hands off toward `target` VIA the supervisor.

    `target` is the *intended* next node in the parent StateGraph (one of:
    literature, experiment_design, instrument_control, data_processing,
    paper_writing, paper_review, supervisor).

    Routing semantics: the emitted Command ALWAYS goes
    to ``supervisor`` (``goto="supervisor"``). When ``target`` is a real agent
    (not the supervisor itself), the agent's intended next hop is appended to
    ``state["routing_hints"]`` so the supervisor re-dispatches there AFTER running
    its loop / budget guards. When ``target == "supervisor"`` it is a plain
    "I'm done, decide what's next" return and no hint is set (the supervisor
    routes via its normal LLM classifier).

    Parallel-safe (2026-07-11): every key this tool writes has a state reducer,
    because under fan-out several agents run this SAME tool in one super-step —
    ``messages`` (add_messages), ``visit_count`` (sum_int_dicts),
    ``active_agent`` (last_wins), ``routing_hints`` (merge_routing_hints). A key
    added here without a reducer would crash every parallel run.
    """
    name = f"handoff_to_{target}"
    # Handing off "to the supervisor" is just a return — no routing hint.
    _is_return = target == "supervisor"

    @tool(name, description=description)
    def _handoff(
        reason: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState] = None,  # type: ignore[assignment]
    ) -> Command:
        """Issue a handoff. `reason` is logged to the message stream for trace.

        ``state`` is injected by the framework (never by the model — it is not in
        the tool schema the LLM sees). Defaulted so the ~dozen direct-call unit
        tests that invoke ``tool.func(reason=..., tool_call_id=...)`` keep
        working; in the graph it is always supplied.
        """
        update: dict[str, Any] = {
            "active_agent": target,
            "messages": [
                ToolMessage(
                    content=f"[HANDOFF → {target}] {reason}",
                    tool_call_id=tool_call_id,
                    name=name,
                ),
            ],
        }
        # ── customs desk: carry this agent's products across the PARENT boundary.
        # Best-effort by construction — a handoff must never fail because an
        # artifact looked odd. Losing the control transfer is far worse than
        # losing a pointer, and the bodies are on disk either way.
        try:
            update.update(carried_from(state))
        except Exception:  # noqa: BLE001 — see above
            pass
        if not _is_return:
            # +1 DELTA for the target SIBLING agent — the visit_count reducer
            # (sum_int_dicts) ACCUMULATES these so the supervisor's per-agent loop
            # guard sees the real running total even across many hops ().
            # The supervisor honours the hint WITHOUT re-counting the target
            # (graph.py only writes {"supervisor": 1} on the hint-dispatch branch),
            # so counting it here is the agent's single contribution for this hop.
            update["visit_count"] = {target: 1}
            # Record the intended next agent as a LIST element: under parallel
            # fan-out several agents hand back in the same super-step and each
            # contributes its own hint. The merge_routing_hints reducer collects
            # them (append-dedupe); the supervisor consumes ALL of them — several
            # agents asking for a next hop IS a fan-out request — and then clears
            # the channel by writing None. (Was a scalar `routing_hint`, which
            # both lost the other branches' intent and crashed on the concurrent
            # write: a bare channel accepts only one value per step.)
            update["routing_hints"] = [target]
        # NB (double-count fix): a RETURN handoff (`target == "supervisor"`) does
        # NOT write a visit_count delta. The supervisor NODE counts its OWN visit
        # ({"supervisor": 1}) on every branch when it runs next; if the handoff
        # ALSO wrote {"supervisor": 1} the supervisor would be counted twice for a
        # single logical visit, so the per-agent guard (cap 8) — for which the
        # supervisor key is the binding constraint — would trip at ~half the
        # intended hop budget and cut the six-phase pipeline short.
        # ALWAYS land on the supervisor so its loop/budget guard runs every hop
        # (). The supervisor then honours routing_hint (if set) or
        # falls back to LLM routing.
        return Command(goto="supervisor", graph=Command.PARENT, update=update)

    return _handoff


# REMOVED 2026-07-29: ``make_supervisor_notice(from_agent, summary)``, which
# built an ``[X DONE] {summary}`` narration and claimed in its docstring that
# "every agent's return path uses the same template". It had ZERO call sites, for
# its whole life. The supervisor never summarised anything — its narration on the
# common path is the literal string "(per agent request)".
#
# It is not being revived, because the thing it was reaching for now exists and
# is better: what the next agent needs is not prose ABOUT the work but a pointer
# TO it, and that travels in the artifact channel above. A dead function that
# describes an intention is worse than no function — it reads, to the next person,
# like a mechanism that is already in place.


__all__ = ["make_handoff"]
