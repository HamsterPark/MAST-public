"""Context-injection inventory, operator overrides, and real-request capture.

WHY THIS PACKAGE EXISTS
=======================
Everything an agent "knows" before it acts is assembled from two piles: a static
``SYSTEM_PROMPT`` per agent, plus whatever the middlewares append at run time.
Nothing showed both piles in one place. That opacity had a cost — the 2026-07-27
coordinate incident (the model wrote ``1.2531`` for ``1.2531e-6`` m) traced back
to OUR OWN injection: ``live_state_mw`` printed ``(= 1253.1 nm)`` next to the
metre value. Nobody caught it because nobody could read the assembled text.

``MASTv2/scripts/dump_prompts_html.py`` solved the offline half (a static HTML
snapshot from source). This package is the running-application half:

* :mod:`mast.prompts.registry`  — the inventory: what gets injected, from where,
  what it currently says, and — crucially — an honest marker when a block simply
  cannot be rendered outside a live request.
* :mod:`mast.prompts.overrides` — operator overrides, persisted through
  ``ConfigOverrideRegistry`` (``config/overrides/prompt_overrides.json``), so an
  edit survives a restart and carries the usual timestamped history.
* :mod:`mast.prompts.capture`   — a bounded ring of the last REAL model requests.
  Not a dry run, not a reconstruction: the exact message list the provider was
  handed, timestamped.

HONESTY RULE (non-negotiable)
=============================
A block that needs live hardware or per-request state is reported as
unavailable, with the reason. It is never replaced by a plausible-looking
sample. A fake would make this page worse than nothing: someone would debug a
real incident against invented text.
"""

from mast.prompts.registry import (
    PromptEntry,
    entries,
    get_entry,
    render_default,
    resolve,
)

__all__ = ["PromptEntry", "entries", "get_entry", "render_default", "resolve"]
