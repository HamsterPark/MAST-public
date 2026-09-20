"""Content-Addressable Storage (CAS) for scan files.

Files are stored under ``<root>/cas/<sha256[:2]>/<sha256[2:4]>/<sha256>`` (Git/IPFS
style 2-byte bucketing). The original path is retained as metadata; the sha256
is the truth.

Reference: compass §2.2.6 (scan_files table, OCFL-inspired fixity).
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB read chunk


@dataclass(frozen=True)
class CASEntry:
    sha256: str
    size_bytes: int
    cas_path: Path
    original_path: Path


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Stream-hash a file. Returns (sha256_hex, size_bytes)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(_CHUNK)
        if not chunk:
            break
        h.update(chunk)
        size += len(chunk)
    return h.hexdigest(), size


class CASStore:
    """Content-addressable file store rooted at ``root``.

    The root is expected to live next to the v2 db file. If you keep the v2 db
    at ``experiments/mast_experiments_v2.db`` then CAS is at ``experiments/cas/``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ─────────────────────────────────────────────────

    def cas_path(self, sha256: str) -> Path:
        if len(sha256) != 64:
            raise ValueError(f"sha256 must be 64 hex chars, got {len(sha256)}")
        return self.root / sha256[:2] / sha256[2:4] / sha256

    def has(self, sha256: str) -> bool:
        return self.cas_path(sha256).is_file()

    # ── Ingest paths ─────────────────────────────────────────────────

    def ingest(
        self,
        source: str | Path,
        *,
        mode: str = "copy",
    ) -> CASEntry:
        """Add *source* to the CAS. Returns the CASEntry.

        ``mode``:
          - ``"copy"`` (default): copy file into CAS (safe across volumes)
          - ``"hardlink"``: hardlink into CAS (no extra disk; same volume only)
          - ``"move"``: shutil.move into CAS (destructive on source)
          - ``"verify"``: do not store; just compute the digest and return entry
            with ``cas_path == source``.
        """
        src = Path(source).resolve()
        sha, size = sha256_file(src)
        target = self.cas_path(sha)
        if mode == "verify":
            return CASEntry(sha256=sha, size_bytes=size, cas_path=src, original_path=src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # Already present; trust prior digest. Verify size sanity.
            if target.stat().st_size != size:
                raise RuntimeError(
                    f"sha256 collision or corruption: {sha} size mismatch "
                    f"({target.stat().st_size} != {size})"
                )
            logger.debug("CAS hit: %s already at %s", sha, target)
            return CASEntry(sha256=sha, size_bytes=size, cas_path=target, original_path=src)
        if mode == "copy":
            shutil.copy2(src, target)
        elif mode == "hardlink":
            try:
                os.link(src, target)
            except OSError as exc:
                logger.warning("hardlink failed (%s), falling back to copy", exc)
                shutil.copy2(src, target)
        elif mode == "move":
            shutil.move(str(src), str(target))
        else:
            raise ValueError(f"Unknown CAS mode: {mode!r}")
        return CASEntry(sha256=sha, size_bytes=size, cas_path=target, original_path=src)

    # ── Fixity ────────────────────────────────────────────────────────

    def verify(self, sha256: str) -> bool:
        """Recompute sha256 of CAS file and compare. False if missing or mismatched."""
        p = self.cas_path(sha256)
        if not p.is_file():
            return False
        got, _ = sha256_file(p)
        return got == sha256

    def iter_all(self):
        """Iterate over all (sha256, path) pairs currently in the CAS."""
        for bucket1 in self.root.iterdir():
            if not bucket1.is_dir() or len(bucket1.name) != 2:
                continue
            for bucket2 in bucket1.iterdir():
                if not bucket2.is_dir() or len(bucket2.name) != 2:
                    continue
                for f in bucket2.iterdir():
                    if f.is_file() and len(f.name) == 64:
                        yield f.name, f
