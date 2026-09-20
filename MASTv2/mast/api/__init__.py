"""MAST typed HTTP API (TS-rewrite Phase 2).

This package is the thin, typed service seam between the new 100% TypeScript
frontend and the irreducible Python core (nanonis hardware, torch/DINOv3
vision, LangGraph agents). It is **additive**: in Phase 2 it runs ALONGSIDE the
existing Gradio app and never imports or mutates ``mast.webui.app``.

Design contract: ``docs/v2/design/ts_rewrite_rfc.md`` (§2.10, §3).

Boundaries (Phase 2):
  - READ-ONLY endpoints only, zero side effects.
  - Reuses existing backend functions (``mast.config``, registries, stores);
    business logic + safety gates stay in the core, never in this layer.
  - Pydantic models are the single source of types → exported as OpenAPI →
    generated into the frontend (``openapi-typescript`` + ``orval``).

Run standalone (dev): ``.venv-v2-py313/Scripts/python.exe -m mast.api``
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazy re-export so ``import mast.api`` stays cheap (no FastAPI import cost
    until the app is actually built)."""
    from mast.api.app import create_app as _create_app

    return _create_app(*args, **kwargs)
