"""Automatic context compaction for MAST agents (private 私聊 + group 群聊).

Thin wrapper over langchain's stock ``SummarizationMiddleware`` (which already
handles the hard parts: persistent ``RemoveMessage`` writeback, AI/Tool-pair-safe
cutoffs so a provider never 400s on an orphaned ``tool_use``, provider-aware
token counting, and fail-safe summary generation that returns an error STRING
rather than raising).

Three MAST-specific changes:

  1. **Absolute ``("tokens", N)`` trigger computed from our own window table.**
     The stock ``("fraction", x)`` trigger reads ``model.profile["max_input_tokens"]``,
     which the OpenAI-compatible models (Kimi/DeepSeek/Qwen/GLM, custom base_url +
     non-OpenAI ids) do NOT expose → it raises at ``__init__``. So we derive an
     absolute threshold from ``config.model_input_context`` (conservative
     per-model windows) minus an output reserve.
  2. **Optional ``memory_sink``** — each summary is also written to long-term
     memory (``MemoryStore(kind="summary")``) so a compacted-away stretch of the
     conversation stays recallable later, cross-conversation.
  3. **The compaction is STAMPED onto the summary message** (,
     2026-07-27「群聊内的压缩功能并未显性体现」). Upstream compacts silently: the
     state update it returns is ``[RemoveMessage(ALL), summary, *preserved]`` and
     nothing in it says how much history was just replaced. The 群聊 SSE bridge
     saw that update, classified the summary as a HumanMessage ("operator echo")
     and dropped it — so an operator scrolling back through a long group chat was
     reading a transcript the system had rewritten, with no marker anywhere.

     ``_stamp_compaction`` attaches ``additional_kwargs[COMPACTION_META_KEY]``
     with the counts it can actually derive (messages removed / kept, and an
     ESTIMATE of the pre-compaction token count — the same approximate counter
     the trigger itself uses, never a provider-reported number). It travels with
     the state update through the graph stream, so the bridge needs no side
     channel and it survives checkpointing.

Provider-portable: the summariser runs through ``make_chat_model`` (any of the 6
providers). Attach to EVERY agent's middleware stack so both chat modes compact.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware.summarization import SummarizationMiddleware

from mast.config import model_input_context

logger = logging.getLogger(__name__)

# Reserve this many tokens of headroom for the model's OWN reply + thinking when
# deciding the compaction trigger. Reasoning models fold the chain-of-thought into
# the output budget and are floored to 16 000 (models.py), so reserve at least
# that. Conservative: a bigger reserve only compacts slightly earlier.
_DEFAULT_OUTPUT_RESERVE = 16000
_DEFAULT_RATIO = 0.75
_MIN_TRIGGER = 8000

#: Key under which a compaction's facts ride on the summary message's
#: ``additional_kwargs``. The 群聊 SSE bridge (``api/routes/orchestrator.py``)
#: reads this LITERAL rather than importing this module — the bridge is kept free
#: of heavy agent imports (same rule as ``_AUTO_BG_MARKER``); a parity test pins
#: the two spellings together.
COMPACTION_META_KEY = "mast_compaction"

#: langchain's marker on the message it substitutes for the compacted history
#: (``SummarizationMiddleware._build_new_messages``). Matching on it — rather
#: than on "is a HumanMessage whose text starts with…" — means an operator
#: message that happens to quote the prefix is never mistaken for a compaction.
_LC_SUMMARY_SOURCE = "summarization"

#: The English preamble upstream prepends to the summary body. Stripped so the
#: panel shows the summary itself; if upstream ever changes the wording the
#: whole content is kept verbatim instead (never a silent partial strip).
_LC_SUMMARY_PREFIX = "Here is a summary of the conversation to date:"


def compaction_trigger_tokens(
    model_id: str,
    *,
    ratio: float = _DEFAULT_RATIO,
    output_reserve: int = _DEFAULT_OUTPUT_RESERVE,
) -> int:
    """Absolute prompt-token threshold at which to compact for ``model_id``.

    ``ratio * (input_window - output_reserve)`` floored at ``_MIN_TRIGGER`` so a
    tiny/unknown window never produces a degenerate (≤0) threshold.
    """
    window = model_input_context(model_id)
    budget = max(window - max(output_reserve, 0), window // 2)
    return max(_MIN_TRIGGER, int(ratio * budget))


class _MemorySinkSummarization(SummarizationMiddleware):
    """SummarizationMiddleware that also tees each summary into long-term memory."""

    def __init__(self, *args, memory_sink: Callable[[str], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_sink = memory_sink

    def _emit(self, summary: str) -> None:
        if not self._memory_sink or not summary:
            return
        if summary.startswith(("Error generating summary", "No previous", "Previous conversation was too long")):
            return  # don't persist the failure/empty sentinels
        try:
            self._memory_sink(summary)
        except Exception as exc:  # noqa: BLE001 — memory write must never break a turn
            logger.debug("compaction memory_sink failed: %s", exc)

    # Sentinels the base SummarizationMiddleware returns instead of a real
    # summary. Compacting with one of these as the "summary" would REPLACE the
    # entire deleted history with an error string — irreversibly wiping the
    # experiment context (what's been done, forbidden zones, plan progress) on a
    # single summarizer hiccup.
    _FAILURE_SENTINELS = (
        "Error generating summary",
        "Previous conversation was too long",
    )

    def _is_failed_summary(self, summary: str) -> bool:
        return bool(summary) and summary.startswith(self._FAILURE_SENTINELS)

    def _create_summary(self, messages_to_summarize):  # type: ignore[override]
        summary = super()._create_summary(messages_to_summarize)
        self._emit(summary)
        return summary

    async def _acreate_summary(self, messages_to_summarize):  # type: ignore[override]
        summary = await super()._acreate_summary(messages_to_summarize)
        self._emit(summary)
        return summary

    def before_model(self, state, runtime):  # type: ignore[override]
        result = super().before_model(state, runtime)
        return self._stamp_compaction(state, self._guard_compaction(result))

    async def abefore_model(self, state, runtime):  # type: ignore[override]
        result = await super().abefore_model(state, runtime)
        return self._stamp_compaction(state, self._guard_compaction(result))

    # ── make the compaction VISIBLE downstream ──────────
    def _stamp_compaction(self, state, result):
        """Attach the compaction's facts to the summary message.

        Everything here is DERIVED from the two lists this call already has —
        the pre-compaction ``state["messages"]`` and the update being returned.
        Nothing is estimated except the token count, which is labelled as an
        estimate because it IS one (``count_tokens_approximately``, the same
        approximate counter the trigger fires on — there is no provider-reported
        prompt-token number available at this point in the turn).

        Derived here rather than in the summariser hooks because a single
        middleware INSTANCE is shared by every agent in a 群聊 build: stashing
        per-compaction scratch on ``self`` would race under a parallel fan-out.
        ``state`` and ``result`` are call-local, so this is race-free.

        Best-effort: a failure here costs the marker, never the compaction.
        """
        if not result:
            return result
        try:
            msgs = list(result.get("messages") or [])
            idx = next(
                (i for i, m in enumerate(msgs)
                 if isinstance(getattr(m, "additional_kwargs", None), dict)
                 and m.additional_kwargs.get("lc_source") == _LC_SUMMARY_SOURCE),
                None)
            if idx is None:
                return result
            before = list((state or {}).get("messages") or [])
            kept = max(0, len(msgs) - idx - 1)   # everything after the summary
            removed = max(0, len(before) - kept)
            event: dict[str, Any] = {
                "removed": removed,
                "kept": kept,
                "before": len(before),
                "summary": self._summary_body(msgs[idx]),
            }
            trigger = self._trigger_token_threshold()
            if trigger is not None:
                event["trigger_tokens"] = trigger
            estimate = self._token_estimate(before)
            if estimate is not None:
                event["tokens_before_estimate"] = estimate
            msgs[idx].additional_kwargs[COMPACTION_META_KEY] = event
            logger.info("compaction fired: %d message(s) → summary, %d kept "
                        "(≈%s tokens before)", removed, kept,
                        event.get("tokens_before_estimate", "?"))
        except Exception as exc:  # noqa: BLE001 — a marker must not break a turn
            logger.debug("compaction stamp failed: %s", exc)
        return result

    @staticmethod
    def _summary_body(msg) -> str:
        """The summary text with upstream's English preamble removed."""
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        head, sep, tail = content.partition(_LC_SUMMARY_PREFIX)
        return tail.strip() if (sep and not head.strip()) else content.strip()

    def _trigger_token_threshold(self) -> "int | None":
        """The absolute token threshold this instance fires at, when it has one.
        ``None`` for a message-count or fraction trigger — reporting a token
        number there would be inventing one."""
        for kind, value in (self._trigger_conditions or []):
            if kind == "tokens":
                return int(value)
        return None

    def _token_estimate(self, messages) -> "int | None":
        """Approximate prompt tokens of the pre-compaction history, or None."""
        try:
            return int(self.token_counter(messages))
        except Exception as exc:  # noqa: BLE001
            logger.debug("compaction token estimate failed: %s", exc)
            return None

    def _guard_compaction(self, result):
        """If the base produced a compaction whose summary is a failure sentinel,
        DROP it (return None → keep the full history this turn, retry next turn)
        rather than let the error string overwrite the conversation."""
        if not result:
            return result
        try:
            for msg in result.get("messages", []) or []:
                content = getattr(msg, "content", "")
                if isinstance(content, str) and any(
                        s in content for s in self._FAILURE_SENTINELS):
                    logger.warning(
                        "compaction summarizer failed — keeping full history this "
                        "turn instead of overwriting it with the error sentinel")
                    return None
        except Exception:  # pragma: no cover - defensive
            return result
        return result


def make_compaction_middleware(
    *,
    model_id: str,
    summarizer_model: Any,
    keep_messages: int = 20,
    ratio: float = _DEFAULT_RATIO,
    output_reserve: int = _DEFAULT_OUTPUT_RESERVE,
    memory_sink: Callable[[str], None] | None = None,
    trim_tokens_to_summarize: int = 6000,
) -> SummarizationMiddleware:
    """Build the compaction middleware for an agent.

    Args:
        model_id: the AGENT's model id — sizes the trigger from its input window.
        summarizer_model: a ChatModel used to write the summary (provider-portable;
            pass a cheap fixed model, e.g. ``make_chat_model("orchestrator")``, to
            avoid burning the agent's premium model on summaries).
        keep_messages: recent messages kept verbatim after compaction.
        memory_sink: optional ``str -> None`` to persist each summary long-term.
    """
    trigger_tokens = compaction_trigger_tokens(
        model_id, ratio=ratio, output_reserve=output_reserve)
    logger.info("compaction: model=%s window=%d trigger=%d tokens keep=%d msgs",
                model_id, model_input_context(model_id), trigger_tokens, keep_messages)
    return _MemorySinkSummarization(
        model=summarizer_model,
        trigger=("tokens", trigger_tokens),
        keep=("messages", keep_messages),
        trim_tokens_to_summarize=trim_tokens_to_summarize,
        memory_sink=memory_sink,
    )


__all__ = ["make_compaction_middleware", "compaction_trigger_tokens",
           "COMPACTION_META_KEY"]
