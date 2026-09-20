"""Intranet push-update system for MAST2.

The push server (admin's machine) hosts new MAST setup.exe + manifest;
clients poll periodically, download in background, and prompt the user to
install on next launcher start.
"""

from __future__ import annotations

from mast.update.manifest import (
    Manifest,
    compute_sha256,
    load_manifest,
    make_manifest,
    write_manifest,
)

__all__ = [
    "Manifest",
    "compute_sha256",
    "load_manifest",
    "make_manifest",
    "write_manifest",
]
