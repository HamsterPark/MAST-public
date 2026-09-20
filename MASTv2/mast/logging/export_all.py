"""One-click full export of all MAST historical data (lab-internal tool).

Bundles EVERYTHING the lab might want to archive or hand off into a single
timestamped ZIP: experiment records (v1 + v2 SQLite DBs), agent orchestrator
checkpoints, the vision buffer, conversations, operator ratings/feedback,
training trajectories, chat/client logs, plans, GUI snapshots, and on-disk
data artifacts (captured traces, mosaics, analysis products). This is an
internal tool — there is deliberately NO privacy/sensitivity filtering.

What it does NOT bundle by default: large *regenerable* model weights and
caches (DINOv3 backbone, legacy vision checkpoints, the OpenAlex literature
index, HF hub caches, decompiled manuals). Those are gigabytes of redownloadable
artifacts, not experiment history. Pass ``include_heavy=True`` to fold them in
anyway (the operator can tick a box in the GUI).

Freeze-safety (the hard project rule): this module does pure *blocking* I/O —
sqlite backup + a recursive file walk + zip writes. It MUST run on a daemon
thread (the GUI wires it through ``app._offload`` so the click yields a
placeholder first and the queue worker never blocks). It takes no lock the GUI
holds and never touches the event loop, so it can slow down but can never
freeze the UI.

DB consistency: each SQLite file is snapshotted via the online backup API
(``sqlite3.Connection.backup``) rather than copied byte-for-byte. A live GUI
process keeps WAL-mode write connections open; a raw copy of the bare ``.db``
would miss everything still sitting in the ``-wal`` file. The backup API takes a
consistent read transaction that includes un-checkpointed WAL pages. If the
backup fails for any reason we fall back to copying the ``.db`` + ``-wal`` +
``-shm`` triplet so nothing is ever silently lost.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Known SQLite databases under experiments/ — snapshotted via the backup API.
# Paths are relative to experiments/ and MAY name a subdirectory — the ones that
# do are flattened for the temp/arc name below.
_DB_FILES = (
    "mast_experiments_v2.db",          # v2 records: campaigns…trajectories
    "mast_experiments.db",             # v1 records + feedback + memory
    "mast_conversations.db",           # chat/群聊 conversations + transcript (split out)
    "orchestrator_checkpoints.sqlite", # LangGraph thread state (chat/HITL)
    "vision_buffer.wal.sqlite",        # persisted vision/scan event stream
    # Both of these live in a subdirectory and were therefore missing from every
    # export: _add_tree runs with skip_db=True and drops the whole db family
    # wherever it finds it, so a file not named here was not backed up at all.
    "current_monitor/monitor.sqlite",  # segment index + features corpus + alerts
    "env_history/env_history.sqlite",  # environment statistic buckets + spectra
)


def _flat_db_name(name: str) -> str:
    """``a/b.sqlite`` → ``a__b.sqlite``. Used for the temp file and the zip entry.

    The archive keeps every database in one flat ``databases/`` folder, and the
    scratch file must not inherit a directory component that does not exist next
    to the destination zip.
    """
    return name.replace("/", "__").replace("\\", "__")

# Top-level artifacts names that are large, REGENERABLE model weights / caches /
# manuals — not historical experiment data. Pruned in light mode (the default).
_HEAVY_ARTIFACT_NAMES = frozenset({
    "vision_backbone",     # ~1.2 GB DINOv3 HF cache
    "legacy_models",       # ~337 MB v1 vision checkpoints
    "literature_index",    # ~269 MB OpenAlex corpus index
    "literature_libs",     # HF library cache
    "openalex_pipeline",   # ~48 MB literature pipeline products
    "nanonis_manual_raw",  # ~12 MB decompiled manual
    "nanonis_manual",      # rendered manual pages
    "tts_cache",           # regenerable speech audio
    "hub",                 # any HuggingFace hub cache dir
})
# File suffixes that are model/dataset blobs, skipped in light mode.
_HEAVY_SUFFIXES = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin",
                   ".chm", ".npy", ".parquet")
# In light mode, also skip any single file above this size (a stray blob).
_LIGHT_FILE_MAX_BYTES = 64 * 1024 * 1024  # 64 MB


def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _is_db_family(name: str) -> bool:
    """True for a SQLite db / -wal / -shm sidecar (handled by backup, not copy)."""
    n = name.lower()
    return (n.endswith((".db", ".sqlite", "-wal", "-shm"))
            or ".db-" in n or ".sqlite-" in n)


def _backup_db(src: Path, tmp: Path) -> bool:
    """Snapshot SQLite *src* to *tmp* via the online backup API. Returns ok.

    Uses a plain (read-only-in-practice) connection — ``backup`` only reads the
    source — to stay portable across platforms (the ``file:…?mode=ro`` URI form
    is fiddly with Windows drive letters). Never raises: a False return tells the
    caller to fall back to a raw triplet copy.
    """
    sconn = dconn = None
    try:
        sconn = sqlite3.connect(str(src), timeout=5.0)
        dconn = sqlite3.connect(str(tmp))
        sconn.backup(dconn)
        return True
    except Exception as exc:  # noqa: BLE001 — degrade to raw copy, never crash
        logger.warning("export: backup of %s failed (%s); will copy raw", src.name, exc)
        return False
    finally:
        for c in (dconn, sconn):
            try:
                if c is not None:
                    c.close()
            except Exception:  # noqa: BLE001
                pass


def _add_tree(
    zf: zipfile.ZipFile,
    base: Path,
    root: Path,
    manifest: dict,
    *,
    skip_db: bool,
    include_heavy: bool,
    dest_zip: Path,
) -> None:
    """Walk *base* and add every file to *zf* under its path relative to *root*.

    Prunes heavy/cache dirs and (light mode) heavy-suffix or oversized files,
    recording each skip in ``manifest['skipped']``. Lock files and — when
    *skip_db* — db sidecars are always skipped. Never descends into a skipped
    dir (pruned in-place on ``os.walk``'s dirnames).
    """
    base = Path(base)
    if not base.is_dir():
        return
    dest_resolved = dest_zip.resolve()
    for dirpath, dirnames, filenames in os.walk(base):
        dp = Path(dirpath)
        # Prune heavy/cache subdirectories so we never descend into gigabytes.
        if not include_heavy:
            dirnames[:] = sorted(d for d in dirnames
                                 if d.lower() not in _HEAVY_ARTIFACT_NAMES)
        else:
            dirnames.sort()
        for fn in sorted(filenames):
            f = dp / fn
            n = fn.lower()
            if n.endswith(".lock") or fn == ".mast_gui.lock":
                continue
            if skip_db and _is_db_family(fn):
                continue
            try:
                if f.resolve() == dest_resolved:  # don't zip the zip we're writing
                    continue
            except OSError:
                pass
            if not include_heavy:
                if n.endswith(_HEAVY_SUFFIXES):
                    manifest["skipped"].append(
                        {"path": _rel(f, root), "reason": "heavy-suffix",
                         "bytes": _safe_size(f)})
                    continue
                if _safe_size(f) > _LIGHT_FILE_MAX_BYTES:
                    manifest["skipped"].append(
                        {"path": _rel(f, root), "reason": "oversize",
                         "bytes": _safe_size(f)})
                    continue
            arc = _rel(f, root)
            try:
                zf.write(f, arc)
            except (OSError, ValueError) as exc:
                manifest["errors"].append({"path": str(f), "error": str(exc)})
                continue
            sz = _safe_size(f)
            manifest["files"].append({"path": arc, "bytes": sz})
            manifest["total_bytes"] += sz


def _rel(f: Path, root: Path) -> str:
    """Path of *f* relative to *root* as a posix zip arcname (falls back to name)."""
    try:
        return f.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return f.name


def export_all_history(
    dest_zip: str | Path,
    *,
    include_heavy: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Bundle all historical data into *dest_zip*; return a manifest dict.

    Args:
        dest_zip: Output ``.zip`` path (parent dirs created as needed).
        include_heavy: Also bundle large regenerable model weights / caches.
        progress: Optional cheap, non-blocking callback fed short status
            strings as each section completes (the GUI surfaces them).

    Returns:
        Manifest dict (also written into the zip as ``MANIFEST.json``) with
        keys: created_at, project_root, include_heavy, databases, files,
        skipped, errors, total_bytes, file_count, dest.
    """
    from mast._runtime_paths import project_root

    def _note(msg: str) -> None:
        logger.info("export: %s", msg)
        if progress is not None:
            try:
                progress(msg)
            except Exception:  # noqa: BLE001 — progress must never break export
                pass

    root = project_root()
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "tool": "MAST full history export",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "include_heavy": include_heavy,
        "databases": [],
        "files": [],
        "skipped": [],
        "errors": [],
        "total_bytes": 0,
    }

    exp_dir = root / "experiments"
    artifact_roots = [root / "artifacts", root / "MASTv2" / "artifacts"]

    tmp_dbs: list[Path] = []
    try:
        with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED,
                             allowZip64=True) as zf:
            # ── 1. Databases: consistent snapshots (incl. un-checkpointed WAL) ──
            for name in _DB_FILES:
                src = exp_dir / name
                if not src.exists():
                    continue
                _note(f"快照数据库 {name} …")
                flat = _flat_db_name(name)
                tmp = dest_zip.parent / f".{flat}.export-tmp"
                if _backup_db(src, tmp):
                    tmp_dbs.append(tmp)
                    zf.write(tmp, f"databases/{flat}")
                    sz = _safe_size(tmp)
                    manifest["databases"].append(
                        {"name": name, "bytes": sz, "method": "backup"})
                    manifest["total_bytes"] += sz
                else:
                    # Fallback: raw triplet copy so no WAL data is lost.
                    for suffix in ("", "-wal", "-shm"):
                        sf = exp_dir / (name + suffix)
                        if sf.exists():
                            arc = _flat_db_name(name + suffix)
                            zf.write(sf, f"databases/{arc}")
                            sz = _safe_size(sf)
                            manifest["databases"].append(
                                {"name": arc, "bytes": sz, "method": "raw"})
                            manifest["total_bytes"] += sz

            # ── 2. experiments/ — jsonl logs, plans, snapshots (db handled above) ──
            _note("打包 experiments/ 会话与日志 …")
            _add_tree(zf, exp_dir, root, manifest,
                      skip_db=True, include_heavy=include_heavy, dest_zip=dest_zip)

            # ── 3. artifacts/ — captured traces, mosaics, analysis products ──
            for aroot in artifact_roots:
                if aroot.is_dir():
                    _note(f"打包 {aroot.relative_to(root).as_posix()} 数据产物 …")
                    _add_tree(zf, aroot, root, manifest,
                              skip_db=False, include_heavy=include_heavy,
                              dest_zip=dest_zip)

            # ── 4. Manifest ──
            manifest["file_count"] = len(manifest["files"])
            manifest["dest"] = str(dest_zip)
            zf.writestr("MANIFEST.json",
                        json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        for tmp in tmp_dbs:
            try:
                tmp.unlink()
            except OSError:
                pass

    _note("打包完成。")
    return manifest
