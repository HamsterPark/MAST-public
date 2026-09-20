"""Operator overrides for context-injection texts — persisted, restart-proof.

Storage: ``config/overrides/prompt_overrides.json`` via ``ConfigOverrideRegistry``
(NOT ``SettingsStore``). Three reasons for the split:

* A system prompt is 5–40 KB of text. ``ui_settings.json`` holds single-value UI
  preferences (font scale, voice, ports); dropping prompt bodies in there would
  bloat a file that is rewritten on every UI toggle.
* ``ConfigOverrideRegistry`` already gives timestamped history + restore, and the
  高级管理 → 覆盖历史 panel already reads it. Prompt edits are exactly the kind of
  change you want to be able to walk back.
* It is the same layer safety limits / guidance / knowledge use. This is config
  governance, not a UI preference.

The registry file name is registered in ``override_store._ALL_FILES`` — without
that, ``save()`` writes it and the next boot silently ignores it (the same
failure shape as ``SettingsStore.KNOWN_KEYS``).

Every function here is fail-safe: a missing/corrupt store degrades to "no
override", never to an exception. A prompt lookup runs on the model-call path;
it must not be able to break a run.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FILENAME = "prompt_overrides.json"

# Hard cap on a stored override. Generous (the largest shipped prompt is ~40 KB)
# but finite — an unbounded field on an admin form is how a config file becomes
# unopenable.
MAX_OVERRIDE_CHARS = 200_000


def _registry() -> Any | None:
    """The live ConfigOverrideRegistry singleton, or None if unavailable."""
    try:
        from mast.admin.override_store import ConfigOverrideRegistry

        return ConfigOverrideRegistry.get()
    except Exception as exc:  # noqa: BLE001 — never break a model call
        logger.debug("prompt overrides: registry unavailable (%s)", exc)
        return None


def load_all() -> dict[str, str]:
    """Every stored override as ``{prompt_id: text}``. ``{}`` when none/broken."""
    reg = _registry()
    if reg is None:
        return {}
    try:
        raw = reg.get_raw(FILENAME) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("prompt overrides: read failed (%s)", exc)
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def get(prompt_id: str) -> str | None:
    """The stored override text for *prompt_id*, or None."""
    return load_all().get(prompt_id)


def set_override(prompt_id: str, text: str) -> bool:
    """Store *text* as the override for *prompt_id*. True when persisted.

    An empty/whitespace-only body is treated as "clear this override" rather
    than "inject nothing" — a blank system prompt is never what someone meant by
    saving an empty box, and silently shipping one would be a live-fire change.
    """
    if not isinstance(text, str) or not text.strip():
        return clear(prompt_id)
    if len(text) > MAX_OVERRIDE_CHARS:
        raise ValueError(
            f"覆写文本过长（{len(text)} 字符，上限 {MAX_OVERRIDE_CHARS}）"
        )
    reg = _registry()
    if reg is None:
        return False
    try:
        data = reg.get_raw(FILENAME) or {}
        data[prompt_id] = text
        reg.save_and_reload(FILENAME, data)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("prompt override save failed for %s", prompt_id)
        return False


def clear(prompt_id: str) -> bool:
    """Drop the override for *prompt_id* (revert to the code default)."""
    reg = _registry()
    if reg is None:
        return False
    try:
        data = reg.get_raw(FILENAME) or {}
        if prompt_id not in data:
            return True  # already at default — idempotent, still a success
        data.pop(prompt_id, None)
        # Empty payload deletes the file rather than persisting an empty stub.
        reg.save_or_delete_and_reload(FILENAME, data)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("prompt override clear failed for %s", prompt_id)
        return False


def resolve(prompt_id: str, default: str) -> str:
    """*prompt_id*'s effective text: the operator override, else *default*.

    THE hot path — called on every model call for the per-request injectors, and
    at graph-build time for the static agent prompts. Any failure returns
    *default*, so a broken override file degrades to shipped behaviour.
    """
    try:
        text = get(prompt_id)
    except Exception:  # noqa: BLE001
        return default
    return text if isinstance(text, str) and text.strip() else default


__all__ = [
    "FILENAME",
    "MAX_OVERRIDE_CHARS",
    "clear",
    "get",
    "load_all",
    "resolve",
    "set_override",
]
