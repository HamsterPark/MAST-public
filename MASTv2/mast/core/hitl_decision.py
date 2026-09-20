"""Operator verdict → LangGraph HITL Decision translation.

WHY THIS MODULE EXISTS (the rig)
-----------------------------------------------------------
This logic used to live as ``MASTApp._build_decision`` in ``mast/gui/app.py``.
Commit ``7aa1996`` ("全量转向 TypeScript SPA，彻底删除 Gradio", 2026-06-21)
deleted that file **without migrating these functions**, while
``api/routes/agents_control.py`` kept reaching for them via
``getattr(app, "_build_decision", None)``. ``CoreRuntime`` never had them, so
that lookup returned ``None`` on every single call and the endpoint answered::

    {"status": "degraded", "detail": "live decision builder unavailable"}

Consequence: for ~5 weeks **no human approval could be granted at all** — not
DANGEROUS-skill gates, not workflow human nodes, and not ``buffer_hitl`` for
CRITICAL hardware events (tip_quality_drop / E_STOP / retract_needed). The
operator's click reached the backend and got a well-formed reply; it just never
woke the blocked worker. The run stayed stuck, which in turn kept a stream
worker pinned and made the whole group chat look "blocked" ().

The unit tests did not catch it because the fake app in
``tests/v2/unit/api/test_agents_control.py`` **defined its own**
``_build_decision``, i.e. it asserted a contract that no real object satisfied.

These are PURE functions — no ``self``, no live state. Treating them as a
"live-only capability" behind ``getattr`` was the design error; they belong in
core where both the API layer and any future front-end can import them.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Operator-facing synonyms → canonical verdict. Shared with the
# ``allowed_decisions`` enforcement so both speak the same vocabulary.
VERDICT_ALIASES: dict[str, str] = {
    "accept": "approve",
    "approved": "approve",
    "rejected": "reject",
    "deny": "reject",
    "edit_approve": "edit",
    "edited": "edit",
}


def canonical_verdict(verdict: str | None) -> str:
    """Normalise an operator verdict to one of approve / reject / edit."""
    v = (verdict or "").strip().lower()
    return VERDICT_ALIASES.get(v, v)


def coerce_arg_value(original: Any, raw: Any) -> Any:
    """Best-effort coerce an edited (often string) value toward original's type.

    The param editor sends every field as a string. SafetyGate and the skill
    expect the original numeric/bool types — e.g. an edited ``bias_v`` of
    ``"50"`` must become ``50.0`` so ``check_global_bounds`` (which only
    inspects int/float values) actually re-evaluates it. **Dropping this
    coercion silently disables the post-edit safety re-check**, which is why it
    is migrated verbatim alongside :func:`build_decision`.

    Falls back to the raw value if coercion fails or there's no original to
    mirror.
    """
    if isinstance(raw, (int, float, bool)) or raw is None:
        return raw
    if isinstance(original, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("true", "1", "yes", "on")
        return bool(raw)
    if isinstance(original, int) and not isinstance(original, bool):
        try:
            return int(float(str(raw)))
        except (TypeError, ValueError):
            return raw
    if isinstance(original, float):
        try:
            return float(str(raw))
        except (TypeError, ValueError):
            return raw
    # No original type to mirror — try a numeric parse so SafetyGate can see
    # it; keep the string otherwise.
    if isinstance(raw, str):
        s = raw.strip()
        try:
            return int(s) if s.lstrip("-").isdigit() else float(s)
        except (TypeError, ValueError):
            return raw
    return raw


def build_decision(verdict: str, skill: str | None, base_args: dict | None,
                   payload_params: Any, reason: str) -> dict:
    """Translate an operator verdict into a LangGraph HITL Decision dict.

    approve → ``{"type":"approve"}``
    reject  → ``{"type":"reject","message":reason}``
    edit    → ``{"type":"edit","edited_action":{"name":skill,"args":<merged>}}``

    For edit, HITL replaces the ToolCall.args WHOLESALE, so we merge the
    operator-supplied params over the original args to form the complete new
    arg set (never a partial patch). Param values arriving as strings are
    coerced back toward the original arg's type so the skill + SafetyGate see
    real numbers, not ``"50"``.

    An unrecognised verdict collapses to *reject* — the safe direction, so
    nothing dangerous auto-runs on a typo or a future UI sending a new verb.
    """
    v = canonical_verdict(verdict)
    if v == "approve":
        return {"type": "approve"}
    if v == "reject":
        msg = reason.strip() if isinstance(reason, str) else ""
        return {"type": "reject", "message": msg or "Operator rejected the action."}
    if v == "edit":
        merged = dict(base_args or {})
        if isinstance(payload_params, dict):
            for k, raw in payload_params.items():
                merged[k] = coerce_arg_value(merged.get(k), raw)
        return {"type": "edit", "edited_action": {"name": skill, "args": merged}}
    # Unknown verdict — safest is to reject so nothing dangerous auto-runs.
    return {"type": "reject", "message": f"Unknown verdict '{verdict}'."}


def enforce_allowed(verdict: str, allowed_decisions: list | None,
                    skill: str | None, base_args: dict | None,
                    payload_params: Any, reason: str) -> dict:
    """:func:`build_decision`, but first enforce this interrupt's allow-list.

    Each published interrupt advertises ``allowed_decisions``. Without this
    gate an operator could, e.g., send ``edit`` to an ``EmergencyRetract`` that
    only permits ``approve`` and thereby rewrite its parameters — a real safety
    bypass. Disallowed verdicts collapse to *reject* rather than erroring, so
    the blocked worker always gets a well-formed decision and never hangs.
    """
    allowed = [str(d).lower() for d in (allowed_decisions or [])]
    v_canon = canonical_verdict(verdict)
    if allowed and v_canon not in allowed:
        logger.warning(
            "HITL verdict %r not in allowed=%s — collapsing to reject",
            verdict, allowed,
        )
        return {
            "type": "reject",
            "message": (f"Operator verdict '{verdict}' not permitted "
                        f"(allowed: {allowed})."),
        }
    return build_decision(verdict, skill, base_args, payload_params, reason)


def build_workflow_route(verdict: str, routes: list | None,
                         note: str = "") -> tuple[dict | None, str | None]:
    """Resolve a ``workflow_human`` node verdict into a route decision.

    Composite workflow human nodes resume with ``{"route", "note"}`` rather
    than an approve/reject Decision — ``skills/composite/interpreter.py`` reads
    ``decision["route"]`` and raises if it isn't one of the node's options.

    Returns ``(decision, error)``: exactly one is non-None. An unmatched route
    yields ``(None, "<message>")`` so the caller can answer ``route_not_allowed``
    WITHOUT writing anything into the resolved store.
    """
    opts = [str(r) for r in (routes or [])]
    target = (verdict or "").strip().lower()
    match = next((r for r in opts if r.lower() == target), None)
    if opts and match is None:
        return None, f"route {verdict!r} 不在选项 {opts} 内"
    return {"route": match or (verdict or "resolved"), "note": note or ""}, None


def build_ask_answer(selected: list | None, custom_text: str | None,
                     note: str, ask: dict | None) -> tuple[dict | None, str | None]:
    """Resolve an ``ask_user`` interrupt into the answer the tool resumes with.

    Same ``(decision, error)`` contract as :func:`build_workflow_route`: exactly
    one is non-None, and an error means the caller answers *without* writing
    anything into the resolved store, so the blocked worker keeps waiting and the
    operator can correct and resubmit instead of the run dying on a bad click.

    The checks live here rather than in the API layer for the same reason
    :func:`enforce_allowed` does: what the operator is permitted to answer is
    defined by the question the agent asked, and that must hold no matter which
    client posts the resolve.
    """
    meta = ask if isinstance(ask, dict) else {}
    labels = [str(o.get("label")) for o in (meta.get("options") or [])
              if isinstance(o, dict) and str(o.get("label", "")).strip()]
    sel = [str(s).strip() for s in (selected or []) if str(s).strip()]
    txt = (custom_text or "").strip()
    unknown = [s for s in sel if s not in labels]
    if unknown:
        return None, f"选项 {unknown} 不在这个问题给出的选项内"
    if not meta.get("multi_select", False) and len(sel) > 1:
        return None, "这个问题是单选，只能选择一项"
    if txt and not meta.get("allow_custom", True):
        return None, "这个问题不接受自定义回答，请从给出的选项中选择"
    if not sel and not txt:
        return None, "回答为空：请至少选择一项，或填写自定义回答"
    return {"selected": sel, "custom_text": txt, "note": note or ""}, None


__all__ = [
    "VERDICT_ALIASES",
    "build_ask_answer",
    "build_decision",
    "build_workflow_route",
    "canonical_verdict",
    "coerce_arg_value",
    "enforce_allowed",
]
