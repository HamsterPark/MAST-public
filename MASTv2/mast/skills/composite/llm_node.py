"""Workflow ``llm`` node — bounded LLM decision inside a declarative composite.

P2-B (RFC docs/v2/design/skill_builder_p1_rfc.md §9). Two modes:

  * ``route`` — the LLM picks EXACTLY ONE of the node's named routes (closed
    enum + a reserved structural ``uncertain`` abstention). Parse failure /
    off-enum / uncertain / exception all fall to the node's ``escape`` route —
    "uncertain → safe exit" is enforced by structure, never by model goodwill.
  * ``data`` — the LLM fills a small typed ``output_schema``; failure walks the
    ``on_error`` slot instead of binding data.

Confidence policy (D5, logprob is dead on all 6 cloud providers): structural
``uncertain`` option + escape; the two-tier provider-portable decision call is
shared with the orchestrator supervisor (mast.agents._shared.llm_route).

Every decision is appended to the JSONL decision log from day one — the
fields (inputs snapshot, ACTUAL model, parse_path, escape reason) are the
training corpus for the future router-graduation classifier and cannot be
reconstructed after the fact (X_safety review, 2026-06-11).

NEVER raises out of decide_*(): an instrument workflow must degrade to its
escape slot, not crash mid-run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_LOG_LOCK = threading.Lock()


# ── decision log (JSONL, append-only) ───────────────────────────────────────

def decision_log_path():
    from mast._runtime_paths import project_root
    d = project_root() / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d / "decision_log.jsonl"


def log_decision(record: dict) -> None:
    """Best-effort JSONL append — logging must never break the workflow."""
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOG_LOCK:
            with open(decision_log_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("decision log append failed: %s", exc)


# ── model acquisition (patchable seam for tests) ────────────────────────────

def _model_factory(node: dict):
    """Return the chat model for one llm-node decision. Lazy heavy import.

    Defaults to the orchestrator's configured routing model (low temperature,
    bounded tokens). ``node['model']`` is reserved for a future per-node
    override; tests monkeypatch this function."""
    from mast.agents._shared.models import make_chat_model
    return make_chat_model("orchestrator", max_tokens=2048, temperature=0.1)


def describe_model(model) -> str:
    """Best-effort 'which model actually answered' (silent-fallback audit)."""
    for attr in ("model_name", "model", "model_id"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(model).__name__


# ── prompt assembly ─────────────────────────────────────────────────────────

def _persona_text(node: dict) -> str:
    """Injectable agent-context slices (P2-C). Best-effort: absent → ''."""
    persona = node.get("persona")
    if not persona:
        return ""
    try:
        from mast.skills.composite.persona import render_sections
        return render_sections(persona, node.get("context_sections"))
    except Exception as exc:  # noqa: BLE001 — persona must never block a decision
        logger.warning("persona %r injection failed: %s", persona, exc)
        return ""


def _route_messages(node: dict, inputs: dict) -> list[dict]:
    routes = list((node.get("routes") or {}).keys())
    descs = node.get("route_descriptions") or {}
    options = "\n".join(
        [f"- {r}: {descs.get(r, '')}".rstrip(": ") for r in routes]
        + ["- uncertain: 信息不足或无法判断（将走 escape 安全出口）"])
    persona = _persona_text(node)
    sys = (
        "你是自动化 STM 工作流中的一个受限决策节点（不是对话助手）。\n"
        f"你的职责：{node.get('responsibility', '')}\n"
        "你只能在给定选项中选择恰好一个，不执行任何动作、不输出参数。"
        + (f"\n\n[领域上下文]\n{persona}" if persona else ""))
    user = (
        f"当前输入（JSON）：\n{json.dumps(inputs, ensure_ascii=False, default=str)}\n\n"
        f"可选项：\n{options}")
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


def _data_messages(node: dict, inputs: dict) -> list[dict]:
    schema = node.get("output_schema") or {}
    persona = _persona_text(node)
    sys = (
        "你是自动化 STM 工作流中的一个受限结构化输出节点（不是对话助手）。\n"
        f"你的职责：{node.get('responsibility', '')}\n"
        "只输出要求的字段，不执行任何动作。"
        + (f"\n\n[领域上下文]\n{persona}" if persona else ""))
    user = (
        f"当前输入（JSON）：\n{json.dumps(inputs, ensure_ascii=False, default=str)}\n\n"
        f"需要的输出字段：{json.dumps(schema, ensure_ascii=False)}")
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


# ── route mode ──────────────────────────────────────────────────────────────

def decide_route(node: dict, inputs: dict, *, model=None) -> dict:
    """→ {route, reason, escaped, escape_reason, parse_path, model, duration_ms}

    ``route`` is ALWAYS one of the node's routes (escape on any failure).
    JSON-serializable by construction (cached into CompositeProgress.partial_data
    for resume determinism)."""
    routes = list((node.get("routes") or {}).keys())
    escape = node.get("escape") or (routes[-1] if routes else "")
    targets = routes + ["uncertain"]
    t0 = time.perf_counter()

    def _out(route, reason, escaped, esc_reason, path, mdl):
        return {"route": route, "reason": str(reason)[:500],
                "escaped": escaped, "escape_reason": esc_reason,
                "parse_path": path, "model": mdl,
                "duration_ms": int((time.perf_counter() - t0) * 1000)}

    try:
        from typing import Literal

        from pydantic import create_model

        from mast.agents._shared.llm_route import route_decision
        mdl = model if model is not None else _model_factory(node)
        schema = create_model(
            "WorkflowRoute",
            route=(Literal[tuple(targets)], ...),
            reason=(str, ""),
        )
        hint = ('Respond with ONLY a single json object: {"route": "<one of '
                + "|".join(targets) + '>", "reason": "<one sentence>"}')
        decision, path = route_decision(
            mdl, _route_messages(node, inputs), schema=schema,
            valid_targets=targets, json_hint=hint, field="route")
        chosen = decision.get("route")
        mname = describe_model(mdl)
        if chosen == "uncertain":
            return _out(escape, decision.get("reason", ""), True,
                        "uncertain", path, mname)
        if chosen not in routes:  # pragma: no cover — route_decision validates
            return _out(escape, decision.get("reason", ""), True,
                        "off_enum", path, mname)
        return _out(chosen, decision.get("reason", ""), False, "", path, mname)
    except Exception as exc:  # noqa: BLE001 — escape, never crash the workflow
        logger.warning("llm route node %r failed → escape %r: %s",
                       node.get("id"), escape, exc)
        return _out(escape, "", True, f"error: {exc}", "none",
                    describe_model(model) if model is not None else "?")


# ── data mode ───────────────────────────────────────────────────────────────

_PY_TYPES = {"str": str, "float": float, "int": int, "bool": bool}


def _coerce(value, type_name: str):
    t = _PY_TYPES.get(type_name, str)
    if t is bool and isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "是")
    return t(value)


def decide_data(node: dict, inputs: dict, *, model=None) -> dict:
    """→ {ok, data, escaped, escape_reason, parse_path, model, duration_ms}"""
    schema_def: dict = node.get("output_schema") or {}
    t0 = time.perf_counter()

    def _out(ok, data, esc_reason, path, mdl):
        return {"ok": ok, "data": data, "escaped": not ok,
                "escape_reason": esc_reason, "parse_path": path, "model": mdl,
                "duration_ms": int((time.perf_counter() - t0) * 1000)}

    try:
        from pydantic import create_model

        from mast.agents._shared.llm_route import flatten_messages_for_router
        mdl = model if model is not None else _model_factory(node)
        mname = describe_model(mdl)
        fields = {k: (_PY_TYPES.get(v, str), ...) for k, v in schema_def.items()}
        schema = create_model("WorkflowData", **fields)
        msgs = flatten_messages_for_router(_data_messages(node, inputs))
        # tier 1: function_calling structured output（provider 可移植）
        try:
            res = mdl.with_structured_output(
                schema, method="function_calling").invoke(msgs)
            if hasattr(res, "model_dump"):
                res = res.model_dump()
            if isinstance(res, dict) and set(schema_def) <= set(res):
                data = {k: _coerce(res[k], schema_def[k]) for k in schema_def}
                return _out(True, data, "", "structured", mname)
        except Exception as e:  # noqa: BLE001
            logger.info("llm data node structured tier failed (%s); text tier", e)
        # tier 2: 纯文本 JSON + 括号平衡容错解析
        hint = ("Respond with ONLY a single json object with exactly these "
                f"fields: {json.dumps(schema_def, ensure_ascii=False)}")
        resp = mdl.invoke(msgs + [{"role": "user", "content": hint}])
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
        dec = json.JSONDecoder()
        for i, ch in enumerate(str(content)):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(str(content)[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and set(schema_def) <= set(obj):
                try:
                    data = {k: _coerce(obj[k], schema_def[k]) for k in schema_def}
                except (TypeError, ValueError) as e:
                    return _out(False, {}, f"type_coerce: {e}", "text_json", mname)
                return _out(True, data, "", "text_json", mname)
        return _out(False, {}, "parse_error", "none", mname)
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm data node %r failed: %s", node.get("id"), exc)
        return _out(False, {}, f"error: {exc}", "none",
                    describe_model(model) if model is not None else "?")


__all__ = ["decide_route", "decide_data", "log_decision", "decision_log_path",
           "describe_model"]
