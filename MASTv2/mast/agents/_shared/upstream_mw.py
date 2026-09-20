"""UpstreamArtifactMiddleware — tell an agent what the agents before it produced.

This is the READ half of the inter-agent product channel; the write half is the
handoff customs desk in ``handoff.py``. Which agent is shown which product is
declared once in ``artifact_channel.CONSUMES``.

Why a middleware and not a tool: the whole failure this fixes is that an agent
did not KNOW there was anything to fetch. A tool only helps a model that already
suspects the document exists — and the prompts that told it to suspect
(``experiment_design``: "use any LIT summary"; ``paper_writing``: "read the saved
review") were describing a channel that did not carry anything. The block is
pushed, and it names the exact tool call that fetches the body.

Shape copied from ``MemoryRecallMiddleware`` (same package): implement
``wrap_model_call`` / ``awrap_model_call``, append to ``request.system_message``,
and never let a failure escape — a context glitch must not break a turn.

Two properties worth keeping when editing this:

  * **Empty renders nothing.** No "（无）" placeholders and never an illustrative
    example. "There is no upstream product" and "we could not read it" must not
    look alike, and neither may look like a real product.
  * **Stable within a turn.** The block is derived from state, which only changes
    between super-steps, so it is naturally constant across the tool sub-steps of
    one model turn — that keeps the Anthropic prompt-cache prefix intact. Do not
    make it depend on the message list.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from mast.agents._shared import artifact_channel as _ch
from mast.agents._shared.inject import append_system_block

#: 登记表条目 id（抬头是另一条 ``mw.upstream_artifacts.header``）。
PROMPT_ID = "mw.upstream_artifacts.block"

logger = logging.getLogger(__name__)


class UpstreamArtifactMiddleware(AgentMiddleware):
    """Append the 'upstream products' block to this agent's system message."""

    def __init__(self, agent_id: str):
        super().__init__()
        self._agent_id = agent_id or ""

    # ── block construction ─────────────────────────────────────────────
    @staticmethod
    def _tool_names(request) -> "set[str] | None":
        """The names this agent really holds, or None if they cannot be read.

        Used to gate the readback instructions in the rendered block. None and
        ``set()`` mean different things and must stay distinguishable: None is
        "could not determine" (fall back to the canonical hint), an empty set is
        "this agent holds no tools" (render no instructions). Collapsing them
        would either suppress every hint on an API change, or advertise tools an
        agent does not have — the latter is exactly the 2026-07-30 defect.
        """
        tools = getattr(request, "tools", None)
        if tools is None:
            return None
        try:
            names = set()
            for t in tools:
                n = getattr(t, "name", None)
                if n is None and isinstance(t, dict):
                    n = t.get("name") or (t.get("function") or {}).get("name")
                if n:
                    names.add(str(n))
            return names
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not read tool names: %s", exc)
            return None

    def _block(self, request) -> str:
        state = getattr(request, "state", None)
        if state is None:
            return ""
        header = _resolve_header()
        return _ch.render_upstream_block(
            state, self._agent_id, header=header,
            available_tools=self._tool_names(request))

    def _apply(self, request):
        try:
            block = self._block(request)
        except Exception as exc:  # noqa: BLE001 — never break a turn over context
            logger.debug("UpstreamArtifactMiddleware failed: %s", exc)
            return request
        return append_system_block(request, PROMPT_ID, block)

    def wrap_model_call(self, request, handler: Callable[[Any], Any]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._apply(request))


def _resolve_header() -> str:
    """The block header, honouring an operator override when one is set.

    Read per call (not captured at build time) so an edit in
    高级管理 → 上下文注入 takes effect without a restart — same contract the
    other overridable middleware texts advertise.
    """
    # 2026-08-24 修：``resolve`` 的签名是 ``resolve(prompt_id, default)`` ——
    # **default 是必填的**。在此之前这里少传了它，于是每一次调用都 TypeError，
    # 被下面那个 except 吞掉，覆写永远退回常量：登记表标它 overridable、UI 存得
    # 进去、存了没有任何效果。而「overridable 的 id 必须被源码引用」那道闸门是
    # 字符串级 grep —— 它看得见这个名字，看不见它被调错了。现在 AST 闸门核实参
    # 个数（tests/v2/unit/prompts/test_prompt_wiring_gate.py）。
    try:
        from mast.prompts.registry import resolve
        return resolve("mw.upstream_artifacts.header", _ch.UPSTREAM_BLOCK_HEADER)
    except Exception:  # noqa: BLE001 — registry is optional at this layer
        return _ch.UPSTREAM_BLOCK_HEADER


def make_upstream_artifact_middleware(agent_id: str) -> UpstreamArtifactMiddleware:
    return UpstreamArtifactMiddleware(agent_id)


__all__ = ["UpstreamArtifactMiddleware", "make_upstream_artifact_middleware"]
