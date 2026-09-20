"""Pydantic request/response models for the QA (查询助手) slice.

The 查询助手 is MAST's single-turn, READ-ONLY question helper: the operator can
ask questions (instrument state, literature, knowledge base, …) while the main
对话 tab is busy executing something. Its contract — preserved verbatim from the
old Gradio handler (``gui/app.py:_qa_handle`` → ``QuickAskAgent.one_shot``):

  * **Single turn**: one question in, one final text answer out. No conversation
    history is persisted between calls.
  * **READ-ONLY**: the backend agent's tool list is filtered to READ + ANALYSIS
    skills + a knowledge-only meta-tool subset — it never touches the instrument
    and never writes to ExperimentLog / chat-history / plan store.
  * **Independent model**: the helper uses its own ``qa_model`` alias (resolved
    from the persisted ``qa_model`` setting or the config default) so switching
    the main chat's model / thinking level never disturbs it.

Per the house rules this file is the SINGLE SOURCE OF TYPES for this slice. The
handler carries ``response_model`` + every response carries ``degraded`` so the
frontend renders an explanatory-but-not-broken answer when the QuickAsk backend
(LLM client / read-only agent) is absent (standalone, no API key).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QaRequest(BaseModel):
    """One read-only question for the 查询助手.

    ``scope`` is the agent-domain hint passed straight through to
    ``QuickAskAgent.one_shot(scope=...)`` (one of ``all`` / ``literature`` /
    ``experiment_design`` / ``instrument_control`` / ``data_processing`` /
    ``paper_writing`` / ``paper_review`` / ``buffer_summarizer``). Unknown values
    degrade to ``all`` inside the core — the API does not gate them. ``all`` is
    the legacy default."""

    question: str = Field(..., description="The operator's read-only question.")
    scope: str = Field(
        default="all",
        description="Agent-domain hint forwarded to QuickAskAgent.one_shot(scope=...).",
    )


class QaResponse(BaseModel):
    """The single-turn answer.

    ``degraded`` is ``True`` when the QuickAsk backend was unavailable (no LLM
    client / no API key / the one-shot loop raised). In that case ``answer`` is
    an explanatory string rather than a model answer — never a 500. ``model`` is
    the resolved ``qa_model`` alias used (or the configured/persisted default when
    degraded). ``scope`` echoes the effective scope."""

    answer: str = Field(..., description="The final single-turn answer text.")
    model: str = Field(default="", description="Resolved qa_model alias used.")
    scope: str = Field(default="all", description="Effective scope for this query.")
    degraded: bool = Field(
        default=False,
        description="True when the QuickAsk backend was absent / failed.",
    )


__all__ = ["QaRequest", "QaResponse"]
