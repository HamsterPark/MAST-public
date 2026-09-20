"""User-definable literature library registry (Decision 3, §2.2/§4.3/§5.3/§10.1).

This is the single, agent-free backend shared by the GUI 文献库 tab and the
literature agent's library-management tools (`create_/rename_/delete_/list_`
`libraries`, `set_active_library`, `add_/remove_members`).  Keeping one
implementation here means the operator and the agent never diverge on what a
library is or which papers it holds.

A *library* is a first-class, user-named object — independent of any single
experiment (Decision 3):

  * one persistent ``scope="global"`` "reading library", auto-created on first
    launch and **never deletable**;
  * any number of ``scope="custom"`` libraries;
  * ``scope="experiment"`` libraries — **one per experiment**, id
    ``exp_<experiment id8>``, carrying an ``experiment_id`` field.

Where the authority lives (2026-07-29, design §4.2)
---------------------------------------------------

``scope="experiment"`` used to be a decorative label: nothing consumed it and no
binding existed anywhere (the old design's ``experiment_library_binding`` was
specified but never written). It now has a behavioural consumer:

* an **experiment** library's member set is authoritative in the experiment
  folder (``<exp_dir>/library/members.jsonl``, an append-only event log owned by
  :mod:`mast.knowledge.experiment_library`). This file is why "copy the
  experiment folder ⇒ the bibliography travels with it" is true.
* every OTHER library (global / custom) keeps its members right here in
  ``registry.json``.

So for experiment libraries **this file is an index + cache**: writes go
folder-first, and on any disagreement **the folder wins** (``reindex`` can
rebuild these entries from the folders at any time). The reverse — rewriting a
folder from the registry — is forbidden; the single narrow exception is seeding a
folder that has no file at all (see ``experiment_library._adopt_once``).

The default target of a member add is therefore no longer the single global
``active_library_id`` scalar but the **effective library**:
``resolve_effective_library()`` = current experiment's library → manual pointer →
``reading``. ``active_library_id`` is demoted to "the manual pointer used when no
experiment is active"; two experiments running in parallel no longer trample it.

Persistence (§5.3): when a LangGraph checkpointer is configured the registry
lives on a dedicated checkpoint thread; **this module implements the offline /
test fallback** — a JSON file at ``artifacts/literature_libs/registry.json``.
The on-disk shape mirrors the in-state record shape (§2.2) so the two faces
stay interchangeable: IDs / names / paths / small provenance only, never
vectors or handles (checkpointer-safety, §2.4).

Scope of THIS module (P1 of the rollout, §10.2):
  * registry CRUD + membership management + JSON persistence;
  * ``apply_merge_map`` — rewrite ``local:<hash>`` member ids to real
    ``work_id``s after re-provisioning auto-merge (Decision 1, §3.2).

Deliberately **not** here yet: the bind-time write to the v2 logging /
RO-Crate provenance record (§5.4) — that imports ``mast.logging.v2.*`` and is
left to a later commit so this module has no logging dependency.

No cross-agent imports (it lives in ``mast.knowledge``, importable by both the
GUI and the agent tools); no ``time.sleep`` / blocking graph calls; path /
input validation throughout.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────
GLOBAL_LIBRARY_ID = "reading"
GLOBAL_LIBRARY_NAME = "reading library"
_VALID_SCOPES = ("global", "experiment", "custom")
MAX_MEMBERS = 500  # bounded set (§4.3): curation is human/agent-scale, tens not thousands
_MAX_NAME_LEN = 200
_MAX_SLUG_LEN = 64
_SLUG_FALLBACK = "lib"


# ── Paths ────────────────────────────────────────────────────────────
# Resolved LAZILY through the one shared resolver (``knowledge/paths.py``).
# The private ``_find_repo_root()`` that used to live here was one of five
# near-identical copies across ``knowledge/*``; freezing any of them into a
# module constant at import time is exactly how pointed every data
# path at a non-existent directory in the frozen build. ``paths.libs_dir()``
# resolves to the same directory the old walk did (verified byte-for-byte on
# 2026-07-29), and honours ``MAST_LITERATURE_LIBS_DIR``.


# ── Time helper ──────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Slug generation (stable, collision-suffixed) ─────────────────────
def _slugify(name: str) -> str:
    """Derive a stable filesystem-/key-safe slug from a display name.

    ASCII-fold, lowercase, collapse non-alphanumerics to single underscores.
    Non-ASCII (e.g. CJK) names that fold to nothing fall back to ``lib``;
    the caller adds a numeric suffix on collision.
    """
    if not isinstance(name, str):
        name = str(name)
    # Normalise + drop combining marks, keep ASCII letters/digits.
    folded = unicodedata.normalize("NFKD", name)
    folded = folded.encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", folded).strip("_")
    if not slug:
        slug = _SLUG_FALLBACK
    return slug[:_MAX_SLUG_LEN].strip("_") or _SLUG_FALLBACK


def _unique_slug(base: str, existing: set[str]) -> str:
    """Append ``_2``, ``_3``… until the slug is free."""
    if base not in existing:
        return base
    n = 2
    while True:
        cand = f"{base}_{n}"
        if cand not in existing:
            return cand
        n += 1


# ── work_id normalisation / validation ───────────────────────────────
def _normalize_doi(doi: str) -> str:
    """Lowercase + strip the ``https://doi.org/`` prefix (§3.2 match key)."""
    if not doi:
        return ""
    d = str(doi).strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d.strip()


def _is_valid_work_id(work_id: Any) -> bool:
    """A member id is an OpenAlex W… id, an OpenAlex URL, or ``local:<…>``.

    Guards against path traversal / injection: no control chars, bounded
    length, no whitespace-only ids.
    """
    if not isinstance(work_id, str):
        return False
    wid = work_id.strip()
    if not wid or len(wid) > 512:
        return False
    if any(ord(c) < 0x20 for c in wid):  # control chars
        return False
    return True


# An OpenAlex id embedded in either the bare ("W123…") or URL
# ("https://openalex.org/W123…") form. Mirrors literature_index._OPENALEX_ID_RE.
_OPENALEX_ID_RE = re.compile(r"[Ww]\d{2,}")
#: 整串校验用的形态。``\d+``（而不是 ``\d{2,}``）是刻意的：配 ``fullmatch`` 用，
#: 不可能截断，所以下限放宽只会让**短 id 的双形去重也正确**（``W3`` 与
#: ``https://openalex.org/W3`` 此前会被存成两个成员，而「删除报成功却什么都没删」
#: 是很坏的形状）。真实 OpenAlex id 是 8–10 位，短 id 只出现在测试里 —— 但测试 id
#: 确实泄漏进过实机数据。
_OPENALEX_ID_FULL_RE = re.compile(r"[Ww]\d+")


def _canonical_work_id(work_id: str) -> str:
    """Collapse an OpenAlex id to its bare canonical ``W…`` form for the dedup
    key, tolerating the full URL form; ``local:<…>`` and non-OpenAlex ids pass
    through stripped-but-unchanged.

    Delegates to the SINGLE SOURCE OF TRUTH
    ``literature_index.canonical_work_id`` via a LAZY import —
    that module pulls numpy/pandas/httpx, which this lightweight registry
    deliberately avoids at import time. Falls back to an inline copy of the same
    regex when the heavy module can't load (offline / test env with no numpy), so
    membership dedup normalisation never depends on the vector stack being present.

    Without this, ``add_members`` stored ``W123`` and
    ``https://openalex.org/W123`` as two DISTINCT pointers to the SAME paper —
    the duplicate-on-disk bug lit-abstracts flagged (2026-07-10).
    """
    try:
        from mast.knowledge.literature_index import canonical_work_id
        return canonical_work_id(work_id)
    except Exception:  # heavy deps absent → inline the identical rule
        # **必须与真源逐字同义**：剥已知 URL 前缀 → 整串 fullmatch → 认不出就原样
        # 返回。原来这里是 ``search`` + 取匹配段，会把 ``W1998Barth`` 静默截断成
        # ``W1998``（一个不同的 id）；截断值会成为文件夹书目的 key 并传播进 registry，
        # 而 registry 无备份、不可逆（2026-07-29 架构审查 M4）。
        wid = (work_id or "").strip()
        if not wid or wid.lower().startswith("local:"):
            return wid
        bare = wid
        for prefix in ("https://openalex.org/", "http://openalex.org/",
                       "https://api.openalex.org/works/", "openalex.org/"):
            if bare.lower().startswith(prefix):
                bare = bare[len(prefix):]
                break
        bare = bare.strip().strip("/")
        if _OPENALEX_ID_FULL_RE.fullmatch(bare):
            return "W" + bare[1:]
        return wid


# ─────────────────────────────────────────────────────────────────────
# Registry backend
# ─────────────────────────────────────────────────────────────────────
class LibraryError(ValueError):
    """Raised for invalid library operations (bad scope, missing id, …)."""


class LibraryRegistry:
    """JSON-file-backed registry of user-definable libraries (§5.3 fallback).

    Thread-safe (a single ``RLock`` guards all mutations + the file write).
    Every mutating call persists immediately via atomic temp-file rename so a
    crash never leaves a half-written ``registry.json``.

    The in-memory model is plain JSON-compatible dicts (no pydantic / no
    ``agents.state`` import) so this module is independent of the concurrently
    edited ``state.py`` and trivially serialisable.
    """

    def __init__(self, libs_dir: str | os.PathLike[str] | None = None) -> None:
        from mast.knowledge.paths import libs_dir as _resolve_libs_dir

        self._dir = Path(libs_dir) if libs_dir is not None else _resolve_libs_dir()
        self._registry_path = self._dir / "registry.json"
        self._lock = threading.RLock()
        # in-memory store: {library_id: record}
        self._libs: dict[str, dict[str, Any]] = {}
        self._active_library_id: str = GLOBAL_LIBRARY_ID
        self._migrated = False
        self._load()
        self._ensure_global()
        # Persist the canonical-migration ONCE if loading normalised/collapsed
        # anything on disk (legacy URL-form ids or dual-form duplicates). The
        # in-memory state is already correct; this just writes it back so the
        # duplicates are gone from registry.json, not re-collapsed every load.
        if self._migrated:
            # **先留一份 .bak,再写回。** 这个文件存的是手工策展的阅读清单(每篇为什么
            # 留着),没有任何东西能重新生成它,而且它 gitignored —— 一次错误的归一
            # (2026-07-29 架构审查 M4:``canonical_work_id`` 曾把 ``W1998Barth`` 静默
            # 截断成 ``W1998``)在写回后就不可逆、事后也无法诊断。备份让「迁移做错了
            # 什么」永远可查。
            self._backup_before_migration()
            with self._lock:
                self._save()
            logger.info(
                "library registry migrated to canonical work_ids: %s",
                self._registry_path,
            )

    def _backup_before_migration(self) -> None:
        """把迁移前的 ``registry.json`` 原样复制成 ``registry.pre-migration.bak``。

        只在**还没有**备份时写 —— 第一份备份才是「任何迁移之前」的状态,后来的迁移
        不该把它覆盖掉。永不抛:备份失败不能拦住加载。
        """
        try:
            if not self._registry_path.is_file():
                return
            bak = self._registry_path.with_suffix(".pre-migration.bak")
            if bak.exists():
                return
            bak.write_bytes(self._registry_path.read_bytes())
            logger.info("registry 迁移前备份已写出:%s", bak)
        except OSError as exc:
            logger.warning("registry 迁移前备份失败(不影响加载):%r", exc)

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        """Read ``registry.json`` if present; degrade gracefully on errors."""
        try:
            if self._registry_path.exists():
                raw = self._registry_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                libs = data.get("libraries", {})
                if isinstance(libs, dict):
                    self._libs = {
                        k: self._coerce_record(v)
                        for k, v in libs.items()
                        if isinstance(v, dict)
                    }
                    # Migration flag: did canonicalisation rewrite an id or
                    # collapse a dual-form duplicate anywhere? Compare the VALID
                    # raw member ids on disk to the coerced (canonical) ids — any
                    # difference (a rewrite or a shorter list) means the disk form
                    # was legacy and __init__ should persist the normalised form.
                    self._migrated = self._detect_migration(libs)
                active = data.get("active_library_id")
                if isinstance(active, str) and active:
                    self._active_library_id = active
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # Corrupt / unreadable registry must not crash the GUI or agent;
            # start from an empty registry (the global lib is re-created).
            logger.warning("Could not read library registry %s: %s", self._registry_path, exc)
            self._libs = {}

    def _save(self) -> None:
        """Atomically persist the registry to disk (temp file + rename)."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "active_library_id": self._active_library_id,
                "libraries": self._libs,
                "saved_at": _now_iso(),
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            # atomic: write to a temp file in the same dir, then os.replace
            fd, tmp_path = tempfile.mkstemp(
                prefix=".registry.", suffix=".tmp", dir=str(self._dir)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(text)
                os.replace(tmp_path, self._registry_path)
            except BaseException:
                # clean up the temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            # Persistence is best-effort in offline/test mode; never fatal.
            logger.warning("Could not persist library registry %s: %s", self._registry_path, exc)

    def _detect_migration(self, raw_libs: dict[str, Any]) -> bool:
        """True when ``_coerce_record`` normalised or collapsed the on-disk
        members of ANY library — i.e. the disk holds legacy URL-form ids or
        dual-form duplicates that the just-built ``self._libs`` has canonicalised.

        Compares the VALID raw member id list (disk order) against the coerced
        canonical id list. Only reached during ``_load`` with a live registry
        file, so it is cheap (curation is tens of members, not thousands).
        Defensive: any error just returns False (no migration save) rather than
        blocking the load."""
        try:
            for lib_id, rec_raw in raw_libs.items():
                if not isinstance(rec_raw, dict):
                    continue
                raw_ids = [
                    m.get("work_id")
                    for m in (rec_raw.get("members") or [])
                    if isinstance(m, dict) and _is_valid_work_id(m.get("work_id"))
                ]
                coerced_ids = [
                    m["work_id"] for m in self._libs.get(lib_id, {}).get("members", [])
                ]
                # A canonical rewrite (URL→bare) makes the strings differ; a
                # dual-form collapse makes the lists differ in length. Either
                # ⇒ migrate.
                if raw_ids != coerced_ids:
                    return True
        except Exception as exc:  # pragma: no cover - never block a load
            logger.debug("migration detection skipped: %s", exc)
        return False

    def migrate_canonical(self) -> dict[str, Any]:
        """Re-normalise every library's members to canonical bare work_ids and
        collapse any dual-form duplicates in place, persisting the result.

        Idempotent: after a load (which already canonicalises + merges) this is a
        no-op that returns ``changed=False``. Exposed for an explicit / on-demand
        migration trigger (the automatic one runs in ``__init__`` after load).
        Returns ``{"changed", "members_before", "members_after", "collapsed",
        "libraries_touched"}``."""
        with self._lock:
            before = sum(len(r.get("members", [])) for r in self._libs.values())
            changed = False
            touched: list[str] = []
            for lib_id, rec in self._libs.items():
                new_members: list[dict[str, Any]] = []
                by_id: dict[str, dict[str, Any]] = {}
                lib_changed = False
                for m in rec.get("members", []) or []:
                    key = _canonical_work_id(m.get("work_id", ""))
                    if not key:
                        continue
                    if key != m.get("work_id"):
                        m = dict(m)
                        m["work_id"] = key
                        lib_changed = True
                    existing = by_id.get(key)
                    if existing is not None:
                        self._merge_member_into(existing, m)
                        lib_changed = True
                        continue
                    by_id[key] = m
                    new_members.append(m)
                if lib_changed or len(new_members) != len(rec.get("members", [])):
                    rec["members"] = new_members
                    lib_changed = True
                    touched.append(lib_id)
                changed = changed or lib_changed
            after = sum(len(r.get("members", [])) for r in self._libs.values())
            if changed:
                self._save()
            return {
                "changed": changed,
                "members_before": before,
                "members_after": after,
                "collapsed": before - after,
                "libraries_touched": touched,
            }

    # ── record coercion / construction ───────────────────────────
    @staticmethod
    def _coerce_member(m: dict[str, Any]) -> dict[str, Any] | None:
        raw = m.get("work_id", "")
        if not _is_valid_work_id(raw):
            return None
        # Store the CANONICAL bare form on disk : "W123" and
        # "https://openalex.org/W123" collapse to one id, so a load→save round
        # trip can never keep two rows for the same paper. local:<…> and any
        # non-OpenAlex id pass through unchanged.
        wid = _canonical_work_id(raw)
        return {
            "work_id": wid,
            "doi": str(m.get("doi", "") or ""),
            "added_by": m.get("added_by", "user") if m.get("added_by") in ("agent", "user") else "user",
            "added_at": str(m.get("added_at", "") or ""),
            "reason": str(m.get("reason", "") or ""),
            "source": m.get("source", "openalex")
            if m.get("source") in ("openalex", "user_pdf", "user_url", "user_manual")
            else "openalex",
            "fulltext_status": m.get("fulltext_status", "none")
            if m.get("fulltext_status") in ("none", "requested", "ingested")
            else "none",
            "fulltext_ref": m.get("fulltext_ref") if isinstance(m.get("fulltext_ref"), str) else None,
        }

    @classmethod
    def _coerce_record(cls, rec: dict[str, Any]) -> dict[str, Any]:
        """Sanitise a record read from disk into the canonical shape (§2.2).

        Members are canonicalised (``_coerce_member``) and de-duplicated by the
        CANONICAL work_id. A legacy dual-form duplicate — a registry that stored
        BOTH ``W123`` and its ``https://openalex.org/W123`` URL as two rows —
        is MERGED, not dropped, so the richer reason / earliest added_at /
        upgraded full-text status of the second row survive the collapse
        (mirrors ``apply_merge_map``'s §3.2 merge). This is the on-load half of
        the migration; ``_load`` persists it when the disk form actually changed.

        An ``experiment``-scope record with **no** ``experiment_id`` is demoted to
        ``custom`` with a warning: without that field nothing can resolve which
        folder owns the member list, so the scope would be exactly the decorative
        label this design set out to remove. (Zero such records exist on disk
        today — this guards the future.)
        """
        scope = rec.get("scope", "custom")
        if scope not in _VALID_SCOPES:
            scope = "custom"
        raw_eid = rec.get("experiment_id")
        experiment_id = str(raw_eid) if isinstance(raw_eid, str) and raw_eid.strip() else None
        if scope == "experiment" and not experiment_id:
            logger.warning(
                "library %r has scope=experiment but no experiment_id — treating it "
                "as custom (its members stay in registry.json, no folder authority)",
                rec.get("library_id", "?"),
            )
            scope = "custom"
        members: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for m in rec.get("members", []) or []:
            if not isinstance(m, dict):
                continue
            cm = cls._coerce_member(m)
            if cm is None:
                continue
            existing = by_id.get(cm["work_id"])  # key already canonical
            if existing is not None:
                # legacy dual form → fold the duplicate in (keep the richer meta)
                cls._merge_member_into(existing, cm)
                continue
            if len(members) >= MAX_MEMBERS:
                continue  # bounded set; keep scanning for merges, add no new rows
            by_id[cm["work_id"]] = cm
            members.append(cm)
        return {
            "library_id": str(rec.get("library_id", "")),
            "name": str(rec.get("name", "")),
            "scope": scope,
            "experiment_id": experiment_id,
            "created_at": str(rec.get("created_at", "") or ""),
            "index_dir": rec.get("index_dir") if isinstance(rec.get("index_dir"), str) else None,
            "members": members,
        }

    def _ensure_global(self) -> None:
        """Auto-create the undeletable global reading library on first launch."""
        with self._lock:
            if GLOBAL_LIBRARY_ID not in self._libs:
                self._libs[GLOBAL_LIBRARY_ID] = {
                    "library_id": GLOBAL_LIBRARY_ID,
                    "name": GLOBAL_LIBRARY_NAME,
                    "scope": "global",
                    "experiment_id": None,
                    "created_at": _now_iso(),
                    "index_dir": None,
                    "members": [],
                }
                self._save()
            # if active points at a now-absent library, fall back to global
            if self._active_library_id not in self._libs:
                self._active_library_id = GLOBAL_LIBRARY_ID

    # ── CRUD ─────────────────────────────────────────────────────
    def create_library(self, name: str, scope: str = "custom") -> dict[str, Any]:
        """Mint a stable slug from *name*, append a record. Returns the record.

        ``scope="global"`` is rejected here — the single global library is
        auto-created and cannot be minted by the user (§4.3).

        ``scope="experiment"`` is **silently downgraded to custom** (with a
        ``downgraded_from`` marker on the returned record so the caller can say
        so). An experiment library is not a user-named thing: its id is derived
        from the experiment (``exp_<id8>``) and its members live in that
        experiment's folder, so it can only be created through
        ``experiment_library.ensure_experiment_library``. Minting one here would
        produce a record with a name-derived id and no ``experiment_id`` — i.e.
        another decorative label, which is the bug this design removes.
        """
        name = (name or "").strip()
        if not name:
            raise LibraryError("library name must be non-empty")
        if len(name) > _MAX_NAME_LEN:
            raise LibraryError(f"library name too long (>{_MAX_NAME_LEN} chars)")
        if scope == "global":
            raise LibraryError("the global reading library is reserved and auto-created")
        if scope not in _VALID_SCOPES:
            raise LibraryError(f"invalid scope {scope!r}; must be one of {_VALID_SCOPES}")
        downgraded_from = ""
        if scope == "experiment":
            downgraded_from = "experiment"
            scope = "custom"
        with self._lock:
            base = _slugify(name)
            library_id = _unique_slug(base, set(self._libs.keys()))
            rec = {
                "library_id": library_id,
                "name": name,
                "scope": scope,
                "experiment_id": None,
                "created_at": _now_iso(),
                "index_dir": None,
                "members": [],
            }
            self._libs[library_id] = rec
            self._save()
            out = dict(rec)
            if downgraded_from:
                out["downgraded_from"] = downgraded_from
            return out

    def ensure_experiment_record(self, library_id: str, *, experiment_id: str,
                                 name: str = "") -> dict[str, Any]:
        """Idempotently register the registry-side record of an experiment library.

        Called by ``experiment_library.ensure_experiment_library`` — the ONLY
        creator of ``scope="experiment"`` records. ``library_id`` is supplied (it
        is derived from the experiment id, never slugified from a name), so this
        deliberately bypasses ``create_library``'s slug minting.

        Reminder on authority: the record created here is an **index entry**. The
        member list it caches is refreshed from the experiment folder by
        ``replace_members_cache``; on disagreement the folder wins.
        """
        lib_id = str(library_id or "").strip()
        eid = str(experiment_id or "").strip()
        if not lib_id or not eid:
            raise LibraryError("ensure_experiment_record needs both library_id and experiment_id")
        with self._lock:
            rec = self._libs.get(lib_id)
            if rec is None:
                rec = {
                    "library_id": lib_id,
                    "name": (name or lib_id).strip()[:_MAX_NAME_LEN],
                    "scope": "experiment",
                    "experiment_id": eid,
                    "created_at": _now_iso(),
                    "index_dir": None,
                    "members": [],
                }
                self._libs[lib_id] = rec
                self._save()
                return self._copy_record(rec)
            # Existing record: repair scope/experiment_id if a previous version
            # (or a hand-edited registry) lost them. Name is NOT overwritten —
            # it is frozen at creation, same rule as experiment folder names.
            changed = False
            if rec.get("scope") != "experiment":
                rec["scope"] = "experiment"
                changed = True
            if rec.get("experiment_id") != eid:
                rec["experiment_id"] = eid
                changed = True
            if changed:
                self._save()
            return self._copy_record(rec)

    def replace_members_cache(self, library_id: str,
                              members: list[dict[str, Any]]) -> int:
        """Overwrite an experiment library's cached member list from the folder.

        **This is the folder→registry direction and the only one allowed to
        replace wholesale.** The caller
        (``experiment_library._sync_registry``) has already folded
        ``members.jsonl``; refusing to run for a non-experiment library keeps a
        stray call from wiping a custom library whose members are authoritative
        HERE. Returns the cached member count.
        """
        lib_id = str(library_id or "").strip()
        with self._lock:
            rec = self._libs.get(lib_id)
            if rec is None:
                raise LibraryError(f"no such library: {lib_id!r}")
            if rec.get("scope") != "experiment":
                raise LibraryError(
                    f"{lib_id!r} is not an experiment library — its members are "
                    "authoritative in registry.json and must not be replaced "
                    "from a folder")
            coerced: list[dict[str, Any]] = []
            seen: set[str] = set()
            for m in members or []:
                cm = self._coerce_member(m) if isinstance(m, dict) else None
                if cm is None or cm["work_id"] in seen:
                    continue
                if len(coerced) >= MAX_MEMBERS:
                    break
                seen.add(cm["work_id"])
                coerced.append(cm)
            rec["members"] = coerced
            self._save()
            return len(coerced)

    def set_member_fulltext(self, library_id: str, work_id: str, status: str,
                            ref: str | None) -> bool:
        """Write ``fulltext_status`` / ``fulltext_ref`` on ONE registry member.

        For non-experiment (global / custom) libraries this is where the two
        fields — schema slots that no production code had ever written until
        2026-07-29 — actually get set. Experiment libraries go through
        ``experiment_library.set_fulltext`` instead (folder first).
        Returns False when the library or member is unknown; never raises.
        """
        if status not in ("none", "requested", "ingested"):
            status = "none"
        key = _canonical_work_id(work_id)
        with self._lock:
            rec = self._libs.get(str(library_id or "").strip())
            if rec is None or not key:
                return False
            for m in rec.get("members", []) or []:
                if _canonical_work_id(m.get("work_id", "")) == key:
                    m["fulltext_status"] = status
                    m["fulltext_ref"] = ref if isinstance(ref, str) and ref else None
                    self._save()
                    return True
            return False

    def rename_library(self, library_id: str, new_name: str) -> dict[str, Any]:
        """Update the display *name* only; ``library_id`` never changes (§4.3)."""
        new_name = (new_name or "").strip()
        if not new_name:
            raise LibraryError("new name must be non-empty")
        if len(new_name) > _MAX_NAME_LEN:
            raise LibraryError(f"library name too long (>{_MAX_NAME_LEN} chars)")
        with self._lock:
            rec = self._libs.get(library_id)
            if rec is None:
                raise LibraryError(f"no such library: {library_id!r}")
            rec["name"] = new_name
            self._save()
            return dict(rec)

    def delete_library(self, library_id: str) -> bool:
        """Remove a library. **Refuses to delete the global reading library.**

        Returns True on deletion. Does not touch shared ``data/papers/`` PDFs;
        only drops the registry record (per-library index dir cleanup is the
        ingestion module's concern, §4.3).
        """
        with self._lock:
            rec = self._libs.get(library_id)
            if rec is None:
                raise LibraryError(f"no such library: {library_id!r}")
            if rec.get("scope") == "global" or library_id == GLOBAL_LIBRARY_ID:
                raise LibraryError("the global reading library cannot be deleted")
            del self._libs[library_id]
            if self._active_library_id == library_id:
                self._active_library_id = GLOBAL_LIBRARY_ID
            self._save()
            return True

    def list_libraries(self) -> list[dict[str, Any]]:
        """Return a snapshot list of every library (with member counts).

        Read-only. Each entry: library_id, name, scope, experiment_id,
        created_at, index_dir, member_count, is_active.

        ``member_count`` for an experiment library is the **cached** count. It can
        lag the folder by one write if a crash landed between the jsonl append and
        the cache refresh; that is the accepted cost of writing the authority
        first. The library detail / member endpoints fold the folder, so nothing
        downstream reads a stale member *set* — only this count.
        """
        with self._lock:
            out: list[dict[str, Any]] = []
            for lib_id, rec in self._libs.items():
                out.append(
                    {
                        "library_id": lib_id,
                        "name": rec.get("name", ""),
                        "scope": rec.get("scope", "custom"),
                        "experiment_id": rec.get("experiment_id"),
                        "created_at": rec.get("created_at", ""),
                        "index_dir": rec.get("index_dir"),
                        "member_count": len(rec.get("members", [])),
                        "is_active": lib_id == self._active_library_id,
                    }
                )
            return out

    def get_library(self, library_id: str) -> dict[str, Any]:
        """Return a deep-ish copy of one library record (incl. members).

        For an experiment library the members come from the **experiment folder**
        when its ``members.jsonl`` exists — the folder is the authority, and this
        is the read side of that rule (a cache that lagged one write would
        otherwise show a paper the agent just added as missing). Falls back to the
        cached list when there is no folder file to read.
        """
        with self._lock:
            rec = self._libs.get(library_id)
            if rec is None:
                raise LibraryError(f"no such library: {library_id!r}")
            out = self._copy_record(rec)
        eid = out.get("experiment_id")
        if out.get("scope") == "experiment" and eid:
            folded = self._folder_members(str(eid))
            if folded is not None:
                out["members"] = folded
        return out

    @staticmethod
    def _folder_members(experiment_id: str) -> list[dict[str, Any]] | None:
        """Folded ``members.jsonl`` for an experiment, or ``None`` when the file is
        absent / unreadable. Lazy import — ``experiment_library`` imports THIS
        module at its top, so the dependency may only ever point this way at call
        time."""
        try:
            from mast.knowledge.experiment_library import folder_members
            return folder_members(experiment_id)
        except Exception as exc:  # noqa: BLE001 — a read hiccup must not fail get
            logger.warning("folder member read failed (%s): %r", experiment_id, exc)
            return None

    def get_active(self) -> dict[str, Any]:
        """Return the currently active library record (defaults to global)."""
        with self._lock:
            lib_id = self._active_library_id
            if lib_id not in self._libs:
                lib_id = GLOBAL_LIBRARY_ID
                self._active_library_id = lib_id
            return self._copy_record(self._libs[lib_id])

    @property
    def active_library_id(self) -> str:
        with self._lock:
            return self._active_library_id

    def set_active_library(self, library_id: str) -> dict[str, Any]:
        """Set the default library for subsequent add/search calls (§4.3)."""
        with self._lock:
            rec = self._libs.get(library_id)
            if rec is None:
                raise LibraryError(f"no such library: {library_id!r}")
            self._active_library_id = library_id
            self._save()
            return self._copy_record(rec)

    # ── membership ───────────────────────────────────────────────
    def _resolve_library_id(self, library_id: str | None) -> str:
        """Empty/None ⇒ the **effective** library, not the bare active pointer.

        Effective = current experiment's own library → manual ``active_library_id``
        → ``reading`` (``experiment_library.resolve_effective_library``, resolved
        fresh on every call so it follows a scope switch with no subscription).
        With no experiment active this returns exactly what it always did, so the
        offline / test behaviour is unchanged.

        Lazy import: ``experiment_library`` imports this module at its top.
        """
        if library_id:
            return library_id
        try:
            from mast.knowledge.experiment_library import resolve_effective_library
            lib_id, _source = resolve_effective_library(registry=self)
            if lib_id:
                return lib_id
        except Exception as exc:  # noqa: BLE001 — scope resolution must never block
            logger.warning("effective-library resolution failed: %r", exc)
        return self._active_library_id

    def _experiment_id_of(self, library_id: str) -> str:
        """The ``experiment_id`` of an experiment library, else ``""``."""
        with self._lock:
            rec = self._libs.get(library_id) or {}
            if rec.get("scope") != "experiment":
                return ""
            return str(rec.get("experiment_id") or "")

    def _resolve_write_target(self, library_id: str | None) -> tuple[str, str]:
        """``(library_id, experiment_id)`` for a MEMBER WRITE. Creates nothing.

        Two ways to learn that the target is an experiment library, and both are
        needed. An explicit id is looked up in the registry. An omitted id resolves
        through the effective library, where the answer may be an experiment
        library **that has never been created** — its id is derived, so the record
        can legitimately not exist yet. Reading ``experiment_id`` off a missing
        record would come back empty and the write would take the registry-only
        path and fail with "no such library", instead of lazily creating the
        experiment's library the way it is supposed to.
        """
        if library_id:
            return library_id, self._experiment_id_of(library_id)
        try:
            from mast.knowledge.experiment_library import (
                active_experiment_id,
                resolve_effective_library,
            )
            lib_id, source = resolve_effective_library(registry=self)
            if source == "experiment":
                return lib_id, active_experiment_id()
            if lib_id:
                return lib_id, self._experiment_id_of(lib_id)
        except Exception as exc:  # noqa: BLE001 — scope resolution must never block
            logger.warning("write-target resolution failed: %r", exc)
        return self._active_library_id, self._experiment_id_of(self._active_library_id)

    def add_members(
        self,
        work_ids: list[str] | tuple[str, ...] | str,
        library_id: str | None = None,
        *,
        reason: str = "",
        source: str = "openalex",
        added_by: str = "agent",
        doi_by_work_id: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Upsert members into a library. Dedup by ``work_id``, bounded.

        Returns ``{"library_id", "added": [...], "skipped": [...],
        "rejected": [...], "member_count", "at_cap"}``. Idempotent: a
        ``work_id`` already present is upgraded in place (status/source/reason)
        rather than duplicated.

        **Experiment libraries are delegated** to
        ``experiment_library.add_members``, which appends to the experiment
        folder's ``members.jsonl`` first and refreshes this registry after. Every
        caller — module-level helper, GUI, agent tool — lands on the folder-first
        path through here, so there is no way to write half of the pair.
        """
        target, eid = self._resolve_write_target(library_id)
        if eid:
            from mast.knowledge.experiment_library import add_members as _folder_add
            return _folder_add(
                eid, work_ids, reason=reason, source=source, added_by=added_by,
                doi_by_work_id=doi_by_work_id, registry=self,
            )
        return self._add_members_registry_only(
            work_ids, target, reason=reason, source=source, added_by=added_by,
            doi_by_work_id=doi_by_work_id,
        )

    def _add_members_registry_only(
        self,
        work_ids: list[str] | tuple[str, ...] | str,
        library_id: str | None = None,
        *,
        reason: str = "",
        source: str = "openalex",
        added_by: str = "agent",
        doi_by_work_id: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """The registry-only core of :meth:`add_members` — **no folder write**.

        Two callers only: :meth:`add_members` for non-experiment libraries, and
        ``experiment_library``'s degraded path (experiment row gone / folder not
        creatable) where members must still land somewhere rather than be lost.
        Keeping it separate is what stops the delegation from recursing.
        """
        if isinstance(work_ids, str):
            work_ids = [work_ids]
        if source not in ("openalex", "user_pdf", "user_url", "user_manual"):
            source = "openalex"
        if added_by not in ("agent", "user"):
            added_by = "agent"
        doi_by_work_id = doi_by_work_id or {}
        now = _now_iso()
        added: list[str] = []
        skipped: list[str] = []
        rejected: list[str] = []
        with self._lock:
            lib_id = self._resolve_library_id(library_id)
            rec = self._libs.get(lib_id)
            if rec is None:
                raise LibraryError(f"no such library: {lib_id!r}")
            members: list[dict[str, Any]] = rec.setdefault("members", [])
            # Dedup key = CANONICAL work_id, so an existing "https://openalex.org/W1"
            # member and an incoming bare "W1" collapse onto the same row instead of
            # duplicating (the lit-abstracts 2026-07-10 finding). Existing members
            # keep their stored id string; only the dedup index is canonicalised.
            index = {_canonical_work_id(m["work_id"]): m for m in members}
            for raw in work_ids:
                if not _is_valid_work_id(raw):
                    rejected.append(str(raw))
                    continue
                wid = _canonical_work_id(raw)
                if wid in index:
                    # in-place upgrade (idempotent upsert)
                    m = index[wid]
                    if reason:
                        m["reason"] = reason
                    if source != "openalex":
                        m["source"] = source
                    if wid in doi_by_work_id and doi_by_work_id[wid]:
                        m["doi"] = str(doi_by_work_id[wid])
                    skipped.append(wid)
                    continue
                if len(members) >= MAX_MEMBERS:
                    rejected.append(wid)
                    continue
                member = {
                    "work_id": wid,
                    "doi": str(doi_by_work_id.get(wid, "") or ""),
                    "added_by": added_by,
                    "added_at": now,
                    "reason": reason,
                    "source": source,
                    "fulltext_status": "none",
                    "fulltext_ref": None,
                }
                members.append(member)
                index[wid] = member
                added.append(wid)
            self._save()
            return {
                "library_id": lib_id,
                "added": added,
                "skipped": skipped,
                "rejected": rejected,
                "member_count": len(members),
                "at_cap": len(members) >= MAX_MEMBERS,
            }

    def remove_members(
        self,
        work_ids: list[str] | tuple[str, ...] | str,
        library_id: str | None = None,
    ) -> dict[str, Any]:
        """Drop the named members from a library (defaulting to the effective one).

        Experiment libraries are delegated to ``experiment_library.remove_members``
        where the removal is recorded as an **event**, not a rewrite — the
        experiment folder is append-only (INCREMENTAL-ONLY: pull the plug at any
        moment and what is on disk is still self-consistent).
        """
        target_lib, eid = self._resolve_write_target(library_id)
        if eid:
            from mast.knowledge.experiment_library import remove_members as _folder_remove
            return _folder_remove(eid, work_ids, registry=self)
        return self._remove_members_registry_only(work_ids, target_lib)

    def _remove_members_registry_only(
        self,
        work_ids: list[str] | tuple[str, ...] | str,
        library_id: str | None = None,
    ) -> dict[str, Any]:
        """Registry-only core of :meth:`remove_members` — **no folder write**
        (see :meth:`_add_members_registry_only` for why it is split out)."""
        if isinstance(work_ids, str):
            work_ids = [work_ids]
        # Compare on the CANONICAL form so a removal by bare "W1" drops a member
        # stored as "https://openalex.org/W1" (and vice versa) — symmetric with
        # add_members' canonical dedup.
        target = {_canonical_work_id(w) for w in work_ids if _is_valid_work_id(w)}
        with self._lock:
            lib_id = self._resolve_library_id(library_id)
            rec = self._libs.get(lib_id)
            if rec is None:
                raise LibraryError(f"no such library: {lib_id!r}")
            members = rec.get("members", [])
            before = len(members)
            removed = [m["work_id"] for m in members
                       if _canonical_work_id(m["work_id"]) in target]
            rec["members"] = [m for m in members
                              if _canonical_work_id(m["work_id"]) not in target]
            self._save()
            return {
                "library_id": lib_id,
                "removed": removed,
                "member_count": len(rec["members"]),
                "n_removed": before - len(rec["members"]),
            }

    # ── auto-merge (Decision 1, §3.2) ────────────────────────────
    def apply_merge_map(self, merge_map: dict[str, str]) -> dict[str, Any]:
        """Rewrite ``local:<hash>`` member ids to real ``work_id``s everywhere.

        Called by the re-provisioning step (or first post-provision load) after
        a fresh OpenAlex snapshot auto-merges previously-synthetic papers into
        their canonical ``work_id`` (Decision 1, §3.2/§7.5).  For each library:

          * every member whose ``work_id`` is a key in *merge_map* has its id
            rewritten to the mapped ``work_id``;
          * if that target ``work_id`` is *already* a member (many-locals →
            one collapse), the two members are merged into a single row —
            full-text refs unioned, the richer ``reason`` / earliest
            ``added_at`` kept, status/source upgraded — and the duplicate is
            dropped (§3.2 collapse).

        Provenance preserved: ``source`` and the ``original_local_id`` of the
        merged member(s) survive. Returns a per-library summary of rewrites.
        No-op for ids not in the map (idempotent — a second run finds nothing
        to rewrite).
        """
        if not merge_map:
            return {"rewritten": 0, "collapsed": 0, "libraries_touched": []}
        # sanitise the map: only rewrite valid id → valid id pairs
        clean: dict[str, str] = {}
        for old, new in merge_map.items():
            if _is_valid_work_id(old) and _is_valid_work_id(new) and old != new:
                clean[old.strip()] = new.strip()
        if not clean:
            return {"rewritten": 0, "collapsed": 0, "libraries_touched": []}

        rewritten = 0
        collapsed = 0
        touched: list[str] = []
        with self._lock:
            for lib_id, rec in self._libs.items():
                members = rec.get("members", [])
                new_members: list[dict[str, Any]] = []
                by_id: dict[str, dict[str, Any]] = {}
                lib_changed = False
                for m in members:
                    wid = m["work_id"]
                    if wid in clean:
                        new_id = clean[wid]
                        lib_changed = True
                        rewritten += 1
                        # record provenance of the original synthetic id
                        orig = m.get("original_local_id")
                        if isinstance(orig, list):
                            orig_list = list(orig)
                        elif orig:
                            orig_list = [orig]
                        else:
                            orig_list = []
                        if wid not in orig_list:
                            orig_list.append(wid)
                        m = dict(m)
                        m["work_id"] = new_id
                        m["original_local_id"] = orig_list
                        wid = new_id
                    if wid in by_id:
                        # collapse: merge m into the existing target row
                        collapsed += 1
                        lib_changed = True
                        self._merge_member_into(by_id[wid], m)
                    else:
                        by_id[wid] = m
                        new_members.append(m)
                if lib_changed:
                    rec["members"] = new_members
                    touched.append(lib_id)
            if touched:
                self._save()
        return {
            "rewritten": rewritten,
            "collapsed": collapsed,
            "libraries_touched": touched,
        }

    @staticmethod
    def _merge_member_into(keep: dict[str, Any], drop: dict[str, Any]) -> None:
        """Fold *drop* into *keep* (many-locals → one collapse, §3.2)."""
        # earliest added_at wins
        ka, da = keep.get("added_at", ""), drop.get("added_at", "")
        if da and (not ka or da < ka):
            keep["added_at"] = da
        # richest reason wins
        if len(str(drop.get("reason", ""))) > len(str(keep.get("reason", ""))):
            keep["reason"] = drop.get("reason", "")
        # upgrade fulltext status (ingested > requested > none)
        rank = {"none": 0, "requested": 1, "ingested": 2}
        if rank.get(drop.get("fulltext_status", "none"), 0) > rank.get(
            keep.get("fulltext_status", "none"), 0
        ):
            keep["fulltext_status"] = drop.get("fulltext_status")
            keep["fulltext_ref"] = drop.get("fulltext_ref")
        elif keep.get("fulltext_ref") is None and drop.get("fulltext_ref"):
            keep["fulltext_ref"] = drop.get("fulltext_ref")
        # union doi (prefer a present one)
        if not keep.get("doi") and drop.get("doi"):
            keep["doi"] = drop["doi"]
        # union the recorded original_local_id provenance
        ko = keep.get("original_local_id")
        do = drop.get("original_local_id")
        merged: list[str] = []
        for o in (ko, do):
            if isinstance(o, list):
                merged.extend(o)
            elif o:
                merged.append(o)
        if merged:
            # de-dup preserving order
            seen: set[str] = set()
            keep["original_local_id"] = [x for x in merged if not (x in seen or seen.add(x))]

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _copy_record(rec: dict[str, Any]) -> dict[str, Any]:
        out = dict(rec)
        out["members"] = [dict(m) for m in rec.get("members", [])]
        return out


# ─────────────────────────────────────────────────────────────────────
# Module-level singleton + thin functional API (the names the GUI / tools
# call, §4.3). The singleton is the JSON-fallback face; an explicit
# `registry=` arg lets tests / a checkpointer-backed caller inject another.
# ─────────────────────────────────────────────────────────────────────
_default_registry: LibraryRegistry | None = None
_default_lock = threading.Lock()


def get_registry() -> LibraryRegistry:
    """Return the process-wide JSON-fallback registry (lazy, thread-safe)."""
    global _default_registry
    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = LibraryRegistry()
    return _default_registry


def _reg(registry: LibraryRegistry | None) -> LibraryRegistry:
    return registry if registry is not None else get_registry()


def create_library(
    name: str, scope: str = "custom", *, registry: LibraryRegistry | None = None
) -> dict[str, Any]:
    return _reg(registry).create_library(name, scope)


def rename_library(
    library_id: str, new_name: str, *, registry: LibraryRegistry | None = None
) -> dict[str, Any]:
    return _reg(registry).rename_library(library_id, new_name)


def delete_library(library_id: str, *, registry: LibraryRegistry | None = None) -> bool:
    return _reg(registry).delete_library(library_id)


def list_libraries(*, registry: LibraryRegistry | None = None) -> list[dict[str, Any]]:
    return _reg(registry).list_libraries()


def get_library(library_id: str, *, registry: LibraryRegistry | None = None) -> dict[str, Any]:
    return _reg(registry).get_library(library_id)


def get_active(*, registry: LibraryRegistry | None = None) -> dict[str, Any]:
    return _reg(registry).get_active()


def set_active_library(
    library_id: str, *, registry: LibraryRegistry | None = None
) -> dict[str, Any]:
    return _reg(registry).set_active_library(library_id)


def add_members(
    work_ids: list[str] | tuple[str, ...] | str,
    library_id: str | None = None,
    *,
    reason: str = "",
    source: str = "openalex",
    added_by: str = "agent",
    doi_by_work_id: dict[str, str] | None = None,
    registry: LibraryRegistry | None = None,
) -> dict[str, Any]:
    return _reg(registry).add_members(
        work_ids,
        library_id,
        reason=reason,
        source=source,
        added_by=added_by,
        doi_by_work_id=doi_by_work_id,
    )


def remove_members(
    work_ids: list[str] | tuple[str, ...] | str,
    library_id: str | None = None,
    *,
    registry: LibraryRegistry | None = None,
) -> dict[str, Any]:
    return _reg(registry).remove_members(work_ids, library_id)


def apply_merge_map(
    merge_map: dict[str, str], *, registry: LibraryRegistry | None = None
) -> dict[str, Any]:
    return _reg(registry).apply_merge_map(merge_map)


def migrate_canonical(*, registry: LibraryRegistry | None = None) -> dict[str, Any]:
    """On-demand: re-normalise every library's members to canonical bare
    work_ids + collapse dual-form duplicates + persist (idempotent). Runs
    automatically on registry load; this is the explicit trigger ."""
    return _reg(registry).migrate_canonical()


def reset_default_registry() -> None:
    """Drop the process-wide singleton (test helper)."""
    global _default_registry
    with _default_lock:
        _default_registry = None
