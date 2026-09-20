"""Provider-agnostic closed-set LLM routing primitives (P2-A).

Extracted verbatim-in-spirit from ``orchestrator/graph.py`` (the 2026-06-08 F5
provider-portability fix) and PARAMETERIZED so one hardened implementation
serves every router in MAST:

  * the orchestrator supervisor (7-agent dispatch),
  * the workflow ``llm`` node (P2-B: closed enum routes + escape),
  * any future bounded decision point.

Battle-tested constraints baked in (do NOT "simplify" these away):
  - ``with_structured_output`` MUST use ``method="function_calling"`` on a
    TEXT-flattened message list — response_format / prefill / replayed tool
    ids each break a different provider (deepseek/qwen/glm/sonnet/minimax).
  - The text fallback prompt must contain the literal token "json" (qwen) and
    end on a user turn (sonnet).
  - The bare-name fallback only fires on EXACTLY ONE word-boundary target
    mention — ambiguous prose must return None, never a guessed agent.

The decision is returned together with a ``parse_path`` tag
(structured | text_json | call_re | bare_name) — the workflow decision log
records it from day one because bare_name hits are LABEL NOISE for the future
router-graduation dataset (X_safety review, 2026-06-11).
"""

from __future__ import annotations

import json as _json
import logging
import re as _re

logger = logging.getLogger(__name__)


def flatten_messages_for_router(messages) -> list[dict]:
    """Collapse mixed LangChain / dict messages into plain {role, content} text.

    Strips tool-call ids (minimax/sonnet reject replays), keeps ToolMessage
    TEXT (relabeled assistant), drops tool_use blocks inside list content."""
    out: list[dict] = []
    for m in messages or []:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            cls = m.__class__.__name__.lower()
            role = ("system" if "system" in cls
                    else "user" if "human" in cls else "assistant")
            content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        content = str(content).strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def parse_route_text(text: str, valid_targets, *, field: str = "next_agent"
                     ) -> tuple[dict | None, str | None]:
    """Tolerant parse of a routing decision from free text.

    Returns ``(decision, parse_path)`` — decision is
    ``{field: <target>, "reason": str}`` or None. Paths: text_json / call_re /
    bare_name. Hardening history: brace-balanced JSON decode (a '}' inside the
    reason string must not truncate), word-boundary single-mention bare-name
    fallback (0 or >1 mentions → None, see 2026-06-08 adversarial review)."""
    if not text:
        return None, None
    t = text.strip()
    targets = frozenset(valid_targets)
    # 1) Brace-balanced JSON from every '{'.
    dec = _json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        try:
            obj, _end = dec.raw_decode(t[i:])
        except _json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get(field) in targets:
            return ({field: obj[field], "reason": str(obj.get("reason", ""))},
                    "text_json")
        # llm-node style: {"route": ...} when field differs but a unique key
        # matches a target value — strictness first: only the declared field.
    # 2) Call-style: Route(literature, "reason") / route(continue, ...)
    m = _re.search(r"\w+\(\s*['\"]?([A-Za-z_][\w\-]*)['\"]?\s*,\s*['\"](.*?)['\"]\s*\)",
                   t, _re.DOTALL)
    if m and m.group(1) in targets:
        return {field: m.group(1), "reason": m.group(2)}, "call_re"
    # 3) Bare target mention — only when exactly ONE distinct target is named.
    named = {
        name for name in targets
        if _re.search(rf"(?<![\w]){_re.escape(name)}(?![\w])", t)
    }
    if len(named) == 1:
        return {field: next(iter(named)), "reason": t[:200]}, "bare_name"
    return None, None


def route_decision(model, routing_messages, *, schema, valid_targets,
                   json_hint: str, field: str = "next_agent"
                   ) -> tuple[dict, str]:
    """Provider-agnostic enum routing → ``({field, reason}, parse_path)``.

    Two tiers: (1) function_calling structured output on flattened text;
    (2) plain-text JSON + tolerant parse. Raises ValueError only when both
    yield nothing parseable (callers escape/END gracefully)."""
    flat = flatten_messages_for_router(routing_messages)
    try:
        decision = model.with_structured_output(
            schema, method="function_calling"
        ).invoke(flat)
        if hasattr(decision, "model_dump"):
            decision = decision.model_dump()
        elif hasattr(decision, "dict") and not isinstance(decision, dict):
            decision = decision.dict()
        if isinstance(decision, dict) and decision.get(field):
            if decision[field] in frozenset(valid_targets):
                return decision, "structured"
            logger.info("structured route %r off-enum; falling back",
                        decision.get(field))
    except Exception as e:  # noqa: BLE001
        logger.info("function_calling route failed (%s); trying text JSON", e)
    resp = model.invoke(flat + [{"role": "user", "content": json_hint}])
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content
                           if isinstance(b, dict) and b.get("type") == "text")
    parsed, path = parse_route_text(str(content), valid_targets, field=field)
    if parsed and parsed.get(field):
        return parsed, (path or "text_json")
    raise ValueError(f"unparseable routing response: {str(content)[:200]}")


__all__ = ["flatten_messages_for_router", "parse_route_text", "route_decision"]

