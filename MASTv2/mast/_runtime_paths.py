"""Runtime path resolution that works in both dev and PyInstaller-frozen environments.

In a normal dev install of MAST2, ``Path(__file__).resolve().parents[2]`` is the
repo root (the same one that holds ``mast/`` for v1). For PyInstaller --onedir
the ``mast/`` package lives inside ``dist/MAST2/_internal/`` and user data
must sit *next to* MAST2.exe — see ``project_root`` for the resolution.

This module is the single source of truth so individual call sites don't each
have to know about ``sys.frozen``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Return the directory that holds user-facing data folders.

    Resolution order:
      1. ``MAST2_PROJECT_ROOT`` env var (manual override; useful for tests)
      2. ``MAST_PROJECT_ROOT`` env var (legacy compat)
      3. Frozen build → directory containing the executable.
      4. Dev install → MASTv2's parent (= MAST repo root, sibling of v1 ``mast/``).
    """
    for var in ("MAST2_PROJECT_ROOT", "MAST_PROJECT_ROOT"):
        override = os.environ.get(var, "").strip()
        if override:
            return Path(override).resolve()

    if _is_frozen():
        return Path(sys.executable).resolve().parent

    # __file__ is .../MAST/MASTv2/mast/_runtime_paths.py — three .parents up to repo root
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """Return the ``mast/`` package directory (always next to source files)."""
    return Path(__file__).resolve().parent
