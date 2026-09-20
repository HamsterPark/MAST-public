"""Agentic cognition layer: persistent memory, phase-sharding, dreaming.

See docs/v2/design/agentic-cognition.md. All state lives in the experiment
SQLite DB so a project's cognitive context is part of its record.
"""

from __future__ import annotations

from mast.memory.store import KINDS, MemoryStore, sanitize_path

__all__ = ["MemoryStore", "sanitize_path", "KINDS"]
