"""Pydantic request/response models for domain ``guidance_ext``.

These re-expose the THREE guidance sub-surfaces the TS SPA lost when the old
``技能指导`` admin tab (4 sub-tabs) was rewritten Gradio→TS — only the per-skill
annotation sub-tab (``skill_extra``) was re-wired (in ``settings_admin_write``).
The remaining three sub-tabs were:

* ``decision_trees``  → guidance_overrides.json ``decision_trees`` key,
  code defaults in ``mast.knowledge.skill_guidance.DECISION_TREES`` (per-id merge).
* ``recipes``         → guidance_overrides.json ``workflow_recipes`` key,
  code defaults in ``WORKFLOW_RECIPES`` (whole-list REPLACE, not merge).
* ``templates``       → guidance_overrides.json ``measurement_templates`` key,
  code defaults in ``MEASUREMENT_TEMPLATES`` (per-key merge).

All three live in the SAME override file (``guidance_overrides.json``) under a
distinct top-level key, edited via the kept ``ConfigOverrideRegistry`` exactly
the way ``settings_admin_write.py`` relays the per-skill (``skill_extra``) and
knowledge / encyclopedia surfaces — the API only forwards bytes; the merge /
hot-reload authority stays in core (R6).

EVERY response carries a ``degraded`` boolean so a standalone API process (no
live ``ConfigOverrideRegistry`` wired) returns an empty-but-valid body rather
than a 500.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# The three editable guidance "kinds" beyond per-skill annotations. Kept as a
# Literal for the frontend type generators; the route param is a free ``str`` so
# an unknown kind degrades cleanly instead of 404-ing.
GuidanceKind = Literal["decision_trees", "recipes", "templates"]


class GuidanceExtraResponse(BaseModel):
    """The effective guidance config for one ``kind`` (code defaults merged with
    the persisted override) plus the raw override layer.

    ``data`` is the effective value the editor renders (default ⊕ override);
    ``override`` is only the diff the core persists; ``has_override`` flags
    whether a saved override exists for this kind; ``writable`` is False for a
    read-only kind (none currently — all three have a write seam). ``degraded``
    is True when the live registry / code defaults are unavailable (the body is
    still valid, just empty)."""

    kind: str
    data: Any = None
    override: Any = None
    has_override: bool = False
    writable: bool = True
    degraded: bool = False


class GuidanceExtraWriteRequest(BaseModel):
    """A whole-section guidance write for one ``kind``.

    ``data`` is the override payload to persist under this kind's key in
    ``guidance_overrides.json`` (shape is kind-specific — a dict of trees/
    templates, or a list of recipes — forwarded verbatim to the core). An empty
    ``data`` (None / {} / []) means 'reset to code defaults' (the core drops the
    key; the file is deleted when no overrides remain)."""

    data: Any = None


class GuidanceExtraWriteResponse(BaseModel):
    """Result of a guidance-section write. The core persists + triggers an
    in-process hot-reload (``save_or_delete_and_reload``). ``reloaded`` reports
    whether the hot-reload fired; ``override`` echoes the persisted override for
    this kind. ``degraded`` is True when no registry is wired (no-op) or the kind
    is unknown / read-only."""

    ok: bool = False
    kind: str
    reloaded: bool = False
    override: Any = None
    degraded: bool = False


__all__ = [
    "GuidanceKind",
    "GuidanceExtraResponse",
    "GuidanceExtraWriteRequest",
    "GuidanceExtraWriteResponse",
]
