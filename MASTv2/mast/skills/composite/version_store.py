"""Version-controlled store for declarative composite specs.

Each composite spec is a JSON document under ``<project_root>/config/
composite_skills/`` (the same ``project_root()`` resolution as the API-key and
sensor configs, so it follows the user's data dir). Every :meth:`save` writes a
timestamped immutable copy into ``_history/`` and bumps the spec's integer
``version`` — so a composite can be rolled back, diffed across versions, and its
full edit history inspected.

Layout::

    config/composite_skills/
        <name>.json                 # current version
        _history/
            <name>.v1.json          # immutable snapshots
            <name>.v2.json
            ...

Restoring an old version writes it forward as a NEW version (non-destructive),
so the history is append-only and you can never lose work by rolling back.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from mast.skills.composite.spec import CompositeSpec

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_一-鿿][A-Za-z0-9_\-一-鿿]{0,80}$")

# 修复项 (2026-06-11): one RLock per store ROOT (not per instance) — the GUI
# panel, the admin tab and ad-hoc CompositeVersionStore() constructions all
# point at the same directory, so an instance-level lock would not serialise
# their read-modify-write save cycles. Single-process app → threading is the
# right scope.
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


def _root_lock(root: Path) -> threading.RLock:
    try:
        # normcase: Windows paths are case-insensitive — two spellings of the
        # same root must map to the SAME lock.
        key = os.path.normcase(str(root.resolve()))
    except OSError:  # pragma: no cover — exotic paths
        key = os.path.normcase(str(root))
    with _ROOT_LOCKS_GUARD:
        lk = _ROOT_LOCKS.get(key)
        if lk is None:
            lk = _ROOT_LOCKS[key] = threading.RLock()
        return lk


class VersionStoreError(ValueError):
    pass


class VersionConflictError(VersionStoreError):
    """Optimistic-concurrency (CAS) check failed: the stored composite changed
    since the caller loaded it. Reload, re-apply edits, then save again."""


def store_dir() -> Path:
    from mast._runtime_paths import project_root
    return project_root() / "config" / "composite_skills"


class CompositeVersionStore:
    """JSON-file version store for :class:`CompositeSpec`. Stateless façade."""

    def __init__(self, root: Path | None = None):
        self._root = Path(root) if root is not None else store_dir()
        self._history = self._root / "_history"
        self._lock = _root_lock(self._root)

    # -- helpers --
    def _ensure(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._history.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_name(name: str) -> None:
        if not name or not _SAFE_NAME.match(name):
            raise VersionStoreError(
                f"invalid composite name {name!r} (letters/digits/_/-/CJK, ≤81 chars, "
                "no path separators)")

    def _path(self, name: str) -> Path:
        return self._root / f"{name}.json"

    def _history_path(self, name: str, version: int) -> Path:
        return self._history / f"{name}.v{version}.json"

    @staticmethod
    def _atomic_write(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp",
                                   prefix=path.name + ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- public API --
    def exists(self, name: str) -> bool:
        # P1 review F2: exists() was the ONE entry without _check_name — a
        # traversal-shaped name (..%5C..) became a filesystem existence
        # oracle. Invalid names simply don't exist.
        try:
            self._check_name(name)
        except VersionStoreError:
            return False
        return self._path(name).exists()

    def list_specs(self) -> list[dict]:
        """Return a summary of every stored composite (name, version, desc...)."""
        out: list[dict] = []
        if not self._root.exists():
            return out
        for p in sorted(self._root.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append({
                    "name": d.get("name", p.stem),
                    "version": d.get("version", 1),
                    "description": d.get("description", ""),
                    "safety_level": d.get("safety_level", "confirm"),
                    "n_nodes": len(d.get("nodes", [])),
                    "tags": d.get("tags", []),
                })
            except Exception as exc:
                logger.warning("composite store: bad file %s: %s", p, exc)
        return out

    def load(self, name: str) -> CompositeSpec:
        self._check_name(name)
        # Readers share the per-root lock too (修复项 review): on Windows a
        # reader holding the file open (no FILE_SHARE_DELETE) makes a
        # concurrent save's os.replace fail with a sharing violation — cheap
        # reads under the RLock close that window.
        with self._lock:
            p = self._path(name)
            if not p.exists():
                raise VersionStoreError(f"composite {name!r} not found")
            return CompositeSpec.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def load_meta(self, name: str) -> dict:
        """The persisted ``_``-prefixed metadata of *name* (``{}`` if absent).

        ``load()`` returns a :class:`CompositeSpec`, and ``from_dict`` drops every
        underscore key — so the sidecar facts written by :meth:`save`
        (``_saved_at`` / ``_content_sha256`` / caller-supplied ``_``-keys) had no
        reader at all. The seeding guard needs exactly those: it must tell an
        untouched factory copy from one the operator edited, and that question is
        answered by metadata, not by the spec body.

        Returns ``{}`` for a missing/unreadable entry — **an empty dict means
        "nothing recorded", never "unchanged"**; callers must not read absence as
        a verdict (the caller here treats it as "unknown provenance ⇒ don't
        touch").
        """
        try:
            self._check_name(name)
        except VersionStoreError:
            return {}
        with self._lock:
            p = self._path(name)
            if not p.exists():
                return {}
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 — 读不动就是「没记录」
                logger.warning("composite store: cannot read meta of %s: %s",
                               name, exc)
                return {}
        return {k: v for k, v in d.items()
                if isinstance(k, str) and k.startswith("_")}

    def save(self, spec: CompositeSpec, *,
             base_version: int | None = None,
             extra_meta: dict | None = None) -> CompositeSpec:
        """Validate, bump version (if a prior exists), archive, and persist.

        ``base_version``  enables optimistic concurrency: pass the
        version the caller LOADED, and the save is rejected with
        :class:`VersionConflictError` if someone saved a newer version in
        between (two browser tabs / LAN clients editing the same composite).
        ``None`` keeps the legacy last-write-wins behaviour.

        ``extra_meta`` (P1, sync-ready record RFC §7): extra underscore-
        prefixed metadata merged into the persisted JSON (e.g.
        ``_resolved_skills``). Only ``_``-keys are accepted — they ride next
        to ``_saved_at`` and are ignored by ``CompositeSpec.from_dict``.

        Returns the spec as written (with the new version number). Raises
        :class:`VersionStoreError` if the spec is invalid."""
        self._check_name(spec.name)
        problems = spec.validate()
        if problems:
            raise VersionStoreError("invalid spec:\n  - " + "\n  - ".join(problems))
        # The whole read-modify-write cycle is serialised per store root —
        # without this, two concurrent saves both read version N and both
        # write v(N+1), the second os.replace silently OVERWRITING the first's
        # "immutable" _history snapshot .
        with self._lock:
            self._ensure()
            prior = None
            if self._path(spec.name).exists():
                try:
                    prior = json.loads(self._path(spec.name).read_text(encoding="utf-8"))
                except Exception:
                    prior = None
            current_version = int(prior.get("version", 0)) if prior else 0
            if base_version is not None and current_version != base_version:
                raise VersionConflictError(
                    f"composite {spec.name!r} changed since you loaded it "
                    f"(stored v{current_version}, your base v{base_version}) — "
                    "reload the latest version and re-apply your edits, or "
                    "save under a new name")
            spec.version = current_version + 1 if prior else max(1, spec.version)
            # Belt-and-braces: NEVER clobber an existing history snapshot, even
            # if files were touched outside this process — bump until free.
            while self._history_path(spec.name, spec.version).exists():
                spec.version += 1
            data = spec.to_dict()
            data["_saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # sync-ready record (P1, RFC §7): content hash always; caller-
            # provided _-prefixed metadata (author/machine/_resolved_skills).
            # Hash excludes the version counter (review nit) so identical
            # content re-saved under a new version keeps the same fingerprint
            # — that's what makes it usable for drift detection.
            try:
                import hashlib
                hashable = spec.to_dict()
                hashable.pop("version", None)
                canon = json.dumps(hashable, ensure_ascii=False, sort_keys=True)
                data["_content_sha256"] = hashlib.sha256(
                    canon.encode("utf-8")).hexdigest()
            except Exception:  # pragma: no cover — never block a save
                pass
            for k, v in (extra_meta or {}).items():
                if isinstance(k, str) and k.startswith("_"):
                    data[k] = v
            self._atomic_write(self._history_path(spec.name, spec.version), data)
            self._atomic_write(self._path(spec.name), data)
        logger.info("composite store: saved %s v%d", spec.name, spec.version)
        return spec

    def list_versions(self, name: str) -> list[dict]:
        """Version history (newest first): [{version, saved_at, n_nodes}]."""
        self._check_name(name)
        out: list[dict] = []
        with self._lock:
            return self._list_versions_locked(name, out)

    def _list_versions_locked(self, name: str, out: list[dict]) -> list[dict]:
        if not self._history.exists():
            return out
        for p in self._history.glob(f"{name}.v*.json"):
            m = re.match(rf"{re.escape(name)}\.v(\d+)\.json$", p.name)
            if not m:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({"version": int(m.group(1)),
                        "saved_at": d.get("_saved_at", ""),
                        "n_nodes": len(d.get("nodes", [])),
                        "description": d.get("description", "")})
        out.sort(key=lambda x: x["version"], reverse=True)
        return out

    def load_version(self, name: str, version: int) -> CompositeSpec:
        self._check_name(name)
        with self._lock:
            p = self._history_path(name, version)
            if not p.exists():
                raise VersionStoreError(f"{name} v{version} not found")
            return CompositeSpec.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def restore(self, name: str, version: int) -> CompositeSpec:
        """Roll back to *version* by writing it forward as a new version
        (non-destructive — history is append-only)."""
        with self._lock:  # RLock — save() re-enters
            old = self.load_version(name, version)
            old.notes = (f"Restored from v{version}. " + (old.notes or "")).strip()
            return self.save(old)

    def clone(self, src_name: str, new_name: str, *, author: str = "") -> CompositeSpec:
        """Template primitive: copy *src_name* to a fresh *new_name* (v1)."""
        self._check_name(new_name)
        with self._lock:  # exists-check + save must be one atomic step
            if self.exists(new_name):
                raise VersionStoreError(f"{new_name!r} already exists")
            src = self.load(src_name)
            clone = src.clone(new_name, author=author)
            clone.version = 0  # save() will bump to 1
            return self.save(clone)

    def delete(self, name: str) -> None:
        """Delete the current spec (history snapshots are retained)."""
        self._check_name(name)
        with self._lock:
            p = self._path(name)
            if p.exists():
                p.unlink()
                logger.info("composite store: deleted current %s (history kept)", name)

    def diff(self, name: str, v1: int, v2: int) -> dict:
        """Structured diff between two versions for the UI.

        Returns {added, removed, changed, fields} where node lists are keyed by
        the node ``id`` (or positional index)."""
        a = self.load_version(name, v1).to_dict()
        b = self.load_version(name, v2).to_dict()

        def _index(nodes):
            idx = {}
            for i, n in enumerate(nodes):
                idx[n.get("id") or f"#{i}"] = n
            return idx
        ia, ib = _index(a.get("nodes", [])), _index(b.get("nodes", []))
        added = [k for k in ib if k not in ia]
        removed = [k for k in ia if k not in ib]
        changed = [k for k in ib if k in ia and ia[k] != ib[k]]
        field_changes = {f: [a.get(f), b.get(f)]
                         for f in ("description", "safety_level", "version", "notes")
                         if a.get(f) != b.get(f)}
        return {"added": added, "removed": removed, "changed": changed,
                "fields": field_changes}


__all__ = ["CompositeVersionStore", "VersionConflictError", "VersionStoreError",
           "store_dir"]
