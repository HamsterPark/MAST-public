"""QA (查询助手) domain — single-turn READ-ONLY question helper.

Re-exposes the 查询助手 whose LOGIC lives in the kept core
(``mast.llm.quickask.QuickAskAgent``) but whose UI was lost in the Gradio→TS
rewrite. This handler is a THIN relay onto ``QuickAskAgent.one_shot`` — no
LLM / tool-loop / read-only-gating logic lives here. The old handler it mirrors
is ``gui/app.py:_qa_handle`` (the 查询助手 submit handler that called
``self._quickask.one_shot(text, scope=scope)``).

Contract preserved verbatim (the old "单次只读" contract):
  * single-turn, no persisted history;
  * READ-ONLY — never touches the instrument, never writes logs / plan store;
  * independent ``qa_model`` alias (resolved from the persisted ``qa_model``
    setting or the config default), so it doesn't disturb the main chat model.

GRACEFUL DEGRADATION is mandatory (house rule 2): this router must boot
STANDALONE with no live core wired. The QuickAsk backend (a live
``QuickAskAgent`` built from the executor / registry / state) is reached via
``ctx.quickask``; with nothing wired — or if ``one_shot`` raises — the handler
returns a valid degraded answer (``degraded=True``) with an explanatory string,
never a 500. Heavy backends are LAZY-imported INSIDE the handler in try/except.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_qa import QaRequest, QaResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])

# Fallback alias when neither the persisted ``qa_model`` setting nor the config
# default resolves. 2026-07-18: bumped kimi-k2.6 → kimi-k3 with the global default
# switch (was the old GUI's ``_s.get("qa_model") or "kimi-k2.6"``).
_DEFAULT_QA_MODEL = "kimi-k3"


def _quickask(ctx: Any):
    """Best-effort handle to a live QuickAskAgent.

    The leader wires one onto the context at integration time (built from the
    live executor / registry / state — the read-only subset). In standalone dev
    nothing is wired; we do NOT construct one here (it would need the live
    instrument-backed executor) → absent ⇒ degrade."""
    return getattr(ctx, "quickask", None)


def _resolve_qa_model(ctx: Any, qa) -> str:
    """Resolve the qa_model alias the SAME way the old handler did.

    Priority (mirrors gui/app.py): the live QuickAsk client's current
    ``model_alias`` (it was restored from the persisted ``qa_model`` at build
    time) → the persisted ``qa_model`` setting → the config default model alias →
    the legacy fallback. Never raises."""
    # 1) The live client's current alias (already reflects the restored qa_model).
    client = getattr(qa, "_client", None) if qa is not None else None
    if client is not None:
        try:
            alias = getattr(client, "model_alias", None) or getattr(
                getattr(client, "_config", None), "model_alias", None
            )
            if alias:
                return str(alias)
        except Exception:  # pragma: no cover - defensive
            pass
    # 2) The persisted qa_model setting.
    try:
        store = ctx.settings_store
        if store is not None:
            persisted = store.get("qa_model")
            if persisted:
                return str(persisted)
    except Exception:  # pragma: no cover - defensive
        logger.debug("qa_model settings lookup failed", exc_info=True)
    # 3) The config default model alias.
    try:
        cfg = getattr(ctx, "config", None)
        llm = getattr(cfg, "llm", None)
        alias = getattr(llm, "model_alias", None) or getattr(llm, "model", None)
        if alias:
            return str(alias)
    except Exception:  # pragma: no cover - defensive
        pass
    # 4) Legacy fallback.
    return _DEFAULT_QA_MODEL


@router.post("/qa", response_model=QaResponse)
def ask(request: Request, body: QaRequest) -> QaResponse:
    """Single-turn, READ-ONLY question → one final answer (degrade-safe).

    Relays straight to ``QuickAskAgent.one_shot(question, scope=scope)`` — the
    core owns the read-only tool gating, the tool-use loop, and the answer. No
    history is persisted; the instrument is never touched. With no QuickAsk
    backend wired (standalone / no API key) — or if ``one_shot`` raises — returns
    ``degraded=True`` with an explanatory answer string, never a 500.

    Backing fn: ``mast.llm.quickask.QuickAskAgent.one_shot``."""
    ctx = request.app.state.ctx
    question = (body.question or "").strip()
    scope = body.scope or "all"

    qa = _quickask(ctx)
    model = _resolve_qa_model(ctx, qa)

    if not question:
        # Empty input is a benign no-op, not a backend failure — mirror the old
        # handler's "_请输入查询_" early return (not degraded).
        return QaResponse(
            answer="（请输入查询内容。）", model=model, scope=scope, degraded=False
        )

    if qa is None:
        # No read-only helper wired (standalone, no LLM key). Mirror the old
        # short-circuit message — explanatory, never a 500.
        return QaResponse(
            answer=(
                "查询助手未初始化（需要 LLM API key，且仅在已连接核心的运行模式下可用）。"
            ),
            model=model,
            scope=scope,
            degraded=True,
        )

    try:
        answer = qa.one_shot(question, scope=scope)
    except Exception as exc:  # noqa: BLE001 — any backend failure degrades, never 500
        logger.warning("QA one_shot failed (scope=%r): %s", scope, exc)
        return QaResponse(
            answer=f"查询失败：{type(exc).__name__}: {exc}",
            model=model,
            scope=scope,
            degraded=True,
        )

    return QaResponse(
        answer=answer or "(没有文本回复)", model=model, scope=scope, degraded=False
    )
