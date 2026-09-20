"""Process-level registry of where scan files actually live.

Why: instrument_control saves .sxm files into
Nanonis's session directory (e.g. ``D:\\Data\\fsSTM\\STM\\2026\\20260629``),
but the data_processing agent's file tools run with ``context=None`` — no
pool, no ``Util_SessionPathGet`` — so their directory search only covered
``<data_root>/working-sessions`` and came back empty. The DP agent then
guessed a dozen non-existent paths (…``working-sessions\\…npy``) and blocked
the whole analyse→plot→report pipeline on "file not found".

This tiny holder closes the loop: whoever LEARNS a real location records it
(SaveScan records each saved file; any session-path resolve records the
directory), and whoever SEARCHES reads it first. Plain paths/strings only,
thread-safe, no hardware access, importable from anywhere (core layer).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_recent_files: list[str] = []   # newest LAST; bounded
_session_dirs: list[str] = []   # newest LAST; bounded
# scan_id → structured record, so a caller can look a scan up by IDENTITY
# (its Nanonis basename) instead of guessing an absolute path. This is the
# single-source-of-truth half of feedback #①: the IC side that SAVES a scan
# records it here; the DP side that ANALYSES it resolves scan_id→path from
# here rather than盲猜 a directory + filename (72e18bf1: 64× FileNotFoundError).
# A record is ``{scan_id, path, mtime, channels, frame, recorded_at}`` — plain
# JSON-friendly values only (core layer, no ndarray/handle), never the pixels.
_records: dict[str, dict] = {}
_id_order: list[str] = []        # scan_ids, newest LAST; bounded
_MAX_KEEP = 50


def _derive_scan_id(path: str) -> str:
    """A scan's stable identity = its file stem (the Nanonis basename), with the
    trailing ``_`` Nanonis appends stripped. ``…/Au111_mica_001_.sxm`` →
    ``Au111_mica_001``. Falls back to the full name when there is no stem."""
    try:
        p = Path(path)
    except Exception:  # noqa: BLE001
        return ""
    stem = p.stem.rstrip("_")
    return stem or p.name


# Public alias — discovery tools derive display ids with the SAME rule the
# registry keys by, so a listed scan_id always round-trips through get_scan.
derive_scan_id = _derive_scan_id


def _mtime_or_none(s: str) -> "float | None":
    try:
        return Path(s).stat().st_mtime
    except OSError:
        return None


def _upsert_path_locked(s: str) -> None:
    """Recency + session-dir bookkeeping for a saved file. Caller holds _lock."""
    if s in _recent_files:
        _recent_files.remove(s)
    _recent_files.append(s)
    del _recent_files[:-_MAX_KEEP]
    try:
        parent = str(Path(s).parent)
    except Exception:  # noqa: BLE001
        return
    if parent and parent not in _session_dirs:
        _session_dirs.append(parent)
        del _session_dirs[:-_MAX_KEEP]


def _current_scope() -> tuple[str | None, str | None]:
    """当前 (experiment_id, sample_id)。**永不抛** —— 这个模块必须无依赖可用。"""
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        if log is None:
            return None, None
        return log.current_experiment_id, log.current_sample_id
    except Exception:  # noqa: BLE001
        return None, None


def _upsert_record_locked(
    s: str,
    scan_id: "str | None",
    channels: "list[str] | None",
    frame: "dict | None",
) -> None:
    """Create/refresh the scan_id→record entry. Caller holds _lock. Enrichment
    (channels/frame) is only ever ADDED — a later bare ``record_scan_path`` for
    the same id must not wipe metadata an earlier rich ``record_scan`` supplied."""
    sid = scan_id or _derive_scan_id(s)
    if not sid:
        return
    rec = _records.get(sid)
    if rec is None:
        rec = {"scan_id": sid, "path": s, "channels": None, "frame": None}
        _records[sid] = rec
    else:
        rec["path"] = s  # newest path for this identity wins
    rec["mtime"] = _mtime_or_none(s)
    rec["recorded_at"] = time.time()
    # 归属：这条扫描是在哪个实验/样品下记的（2026-07-28）。此前这张表完全没有
    # scope，于是"这个 scan_id 属于哪块样品"只能靠猜。只在为空时填 —— 一条记录
    # 的归属应当是它**第一次被登记时**的作用域，而不是后来谁碰过它。
    if not rec.get("experiment_id"):
        eid, sid_scope = _current_scope()
        if eid:
            rec["experiment_id"] = eid
        if sid_scope:
            rec["sample_id"] = sid_scope
    if channels is not None:
        rec["channels"] = list(channels)
    if frame is not None:
        rec["frame"] = dict(frame)
    if sid in _id_order:
        _id_order.remove(sid)
    _id_order.append(sid)
    if len(_id_order) > _MAX_KEEP:
        for old in _id_order[:-_MAX_KEEP]:
            _records.pop(old, None)
        del _id_order[:-_MAX_KEEP]


def _publish_scan_complete(path: str, *, was_new: bool) -> None:
    """Announce a newly-recorded scan on the EventBus. Must be called OUTSIDE ``_lock``.

    ``EventBus.publish_scan_complete`` has existed for a long time with **zero
    callers anywhere in the tree** — so a saved scan announced nothing, and any
    consumer that wanted to react to "new measurement data exists" had no choice but
    to poll. That gap matters now: ``last_scan`` is one of only two HARD dependencies
    in the activation table, which means "wait until there is data to analyse" is
    half of what the wake scheduler is for, and it was the half with no trigger.

    Two boundaries:

    * only when the path is genuinely NEW — this function is also reached by ordinary
      bookkeeping (a re-record of an already-known file), and re-announcing it would
      spam a bus that only replays its last 100 events;
    * outside the registry lock, because subscribers run synchronously on the calling
      thread and a subscriber that touched the registry would deadlock it.
    """
    if not was_new:
        return
    try:
        from mast.core.events import EventBus

        EventBus.get().publish_scan_complete(path)
    except Exception as exc:  # noqa: BLE001 — bookkeeping never fails over a notice
        logger.debug("scan_complete publish skipped: %s", exc)


def record_scan_path(path: "str | Path | None") -> None:
    """Record a freshly-saved scan file (call after a successful save).

    Also creates/refreshes a scan_id→path record (id derived from the file
    stem) so the scan is immediately resolvable by identity via
    :func:`resolve_scan_id` / :func:`get_scan` — no channel/frame metadata yet
    (that is filled lazily on first load, or eagerly via :func:`record_scan`)."""
    if not path:
        return
    s = str(path)
    with _lock:
        was_new = s not in _recent_files
        _upsert_path_locked(s)
        _upsert_record_locked(s, None, None, None)
    _publish_scan_complete(s, was_new=was_new)


def record_scan(
    path: "str | Path | None",
    *,
    scan_id: "str | None" = None,
    channels: "list[str] | None" = None,
    frame: "dict | None" = None,
) -> "str | None":
    """Record a scan with its structured metadata and return its scan_id.

    Superset of :func:`record_scan_path`: does the same recency/session-dir
    bookkeeping AND stores ``channels`` (e.g. ``["Z", "Current"]``) + ``frame``
    (e.g. ``{"scan_offset": [...], "scan_range": [...], "scan_pixels": [...]}``)
    so a later ``load_scan(scan_id=…)`` / ``list_scans()`` carries real geometry
    instead of a bare path. Metadata is merged, never dropped."""
    if not path:
        return None
    s = str(path)
    sid = scan_id or _derive_scan_id(s)
    with _lock:
        was_new = s not in _recent_files
        _upsert_path_locked(s)
        _upsert_record_locked(s, sid, channels, frame)
    _publish_scan_complete(s, was_new=was_new)
    return sid


def get_scan(scan_id: "str | None") -> "dict | None":
    """Structured record for a scan_id, or None. Match is exact first, then
    case-insensitive, then stem-normalised (so ``Au111_001_`` finds
    ``Au111_001``). Returns a shallow copy so callers can't mutate the table."""
    if not scan_id:
        return None
    want = str(scan_id).strip()
    with _lock:
        rec = _records.get(want)
        if rec is None:
            norm = want.rstrip("_").lower()
            for sid, r in _records.items():
                if sid.lower() == want.lower() or sid.rstrip("_").lower() == norm:
                    rec = r
                    break
        return dict(rec) if rec is not None else None


def resolve_scan_id(scan_id: "str | None") -> "str | None":
    """The absolute path recorded for a scan_id, or None. Existence is NOT
    checked here (the file may live on a temporarily-unmounted share) — callers
    resolve/verify."""
    rec = get_scan(scan_id)
    return rec.get("path") if rec else None


def list_scans(n: int = 20) -> list[dict]:
    """The most recently recorded scan records, newest first (shallow copies)."""
    with _lock:
        ids = list(reversed(_id_order[-max(0, n):]))
        return [dict(_records[i]) for i in ids if i in _records]


def record_session_dir(dir_path: "str | Path | None") -> None:
    """Record a resolved Nanonis session directory."""
    if not dir_path:
        return
    s = str(dir_path)
    with _lock:
        if s in _session_dirs:
            _session_dirs.remove(s)
        _session_dirs.append(s)
        del _session_dirs[:-_MAX_KEEP]


def recent_scan_paths(n: int = 10) -> list[str]:
    """Most recently recorded scan files, newest first."""
    with _lock:
        return list(reversed(_recent_files[-n:]))


def known_scan_dirs() -> list[Path]:
    """Directories scans are known to land in, newest first. Existence is NOT
    checked here (callers filter) so a temporarily-unmounted share isn't
    silently forgotten."""
    with _lock:
        dirs = list(reversed(_session_dirs))
    return [Path(d) for d in dirs]


def clear() -> None:
    """Test helper."""
    with _lock:
        _recent_files.clear()
        _session_dirs.clear()
        _records.clear()
        _id_order.clear()


# ── 持久化（2026-07-28）────────────────────────────────────────────
#
# 这张表原本是纯内存的：上限 50 条、进程退出即失。于是一次重启之后，
# "上次那张图的 scan_id 对应哪个文件" 就没人知道了 —— 而 DP agent 的
# load_scan(scan_id) 正是靠它解析。落进实验文件夹的 .mast/ 里，重启能捡回来。

def _state_path() -> "Path | None":
    try:
        from mast.core.experiment_paths import experiment_root
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        if log is None or not log.current_experiment_id:
            return None
        # 目录名从 DB 拿；拿不到就不持久化（没有实验文件夹可写）。
        dir_name = (log._storage.get_experiment(log.current_experiment_id) or {}
                    ).get("dir_name")
        if not dir_name:
            return None
        return experiment_root() / dir_name / ".mast" / "scan_registry.json"
    except Exception:  # noqa: BLE001
        return None


def save_state() -> bool:
    """把注册表落盘到当前实验的 ``.mast/``。永不抛。"""
    p = _state_path()
    if p is None:
        return False
    try:
        import json
        with _lock:
            payload = {
                "records": [_records[k] for k in _id_order if k in _records],
                "session_dirs": list(_session_dirs),
                "recent_files": list(_recent_files)[-_MAX_KEEP:],
            }
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        import os
        os.replace(tmp, p)
        return True
    except Exception:  # noqa: BLE001
        return False


def load_state() -> int:
    """从当前实验的 ``.mast/`` 恢复注册表。返回恢复的记录条数。永不抛。

    **不覆盖**已经在内存里的记录 —— 本次会话新记的东西比磁盘上的旧快照新。
    """
    p = _state_path()
    if p is None or not p.is_file():
        return 0
    try:
        import json
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    with _lock:
        for rec in (payload.get("records") or []):
            sid = rec.get("scan_id")
            if not sid or sid in _records:
                continue
            _records[sid] = dict(rec)
            _id_order.append(sid)
            n += 1
        for d in (payload.get("session_dirs") or []):
            if d not in _session_dirs:
                _session_dirs.append(d)
        for f in (payload.get("recent_files") or []):
            if f not in _recent_files:
                _recent_files.append(f)
        if len(_id_order) > _MAX_KEEP:
            for old in _id_order[:-_MAX_KEEP]:
                _records.pop(old, None)
            del _id_order[:-_MAX_KEEP]
    return n


__all__ = [
    "record_scan_path",
    "record_scan",
    "record_session_dir",
    "recent_scan_paths",
    "known_scan_dirs",
    "get_scan",
    "resolve_scan_id",
    "list_scans",
    "derive_scan_id",
    "save_state",
    "load_state",
    "clear",
]
