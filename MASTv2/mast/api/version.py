"""Resolve the running version, robust to dev vs frozen builds.

``_buildinfo.py`` is gitignored and generated at build time (installer/
mast2_build.ps1); in a dev checkout it may carry the last local build's version.
Fall back to a dev sentinel if it's missing.
"""

from __future__ import annotations


def get_version() -> str:
    try:
        from mast._buildinfo import VERSION  # type: ignore

        return str(VERSION)
    except Exception:
        return "0.0.0-dev"


__all__ = ["get_version"]
