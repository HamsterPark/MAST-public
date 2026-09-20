"""Shared user-data path resolution for agent tools.

Why this exists: paper_writing and
paper_review each carried a private ``_repo_root()`` that walked up from
``__file__`` looking for a checkout layout (``mast/`` + ``MASTv2/`` siblings).
In the PyInstaller build there is no checkout, so the walk fell through to
``parents[4]`` — the *install* directory (e.g. ``C:\\MAST``) — and every
data path derived from it (``data/drafts``, ``data/figures``,
``data/experiments.db``) pointed at a directory that does not exist. Result:
drafts could never be loaded ("drafts directory not found at
C:\\MAST\\data\\drafts"), the experiment DB was "missing", and review reports
had nowhere to live.

The runtime already has ONE correct answer — :func:`mast._runtime_paths.
project_root` (env override → frozen exe dir → dev repo root); the launcher
points it at the user-data root (e.g. ``D:\\MAST-data``). Everything here
derives from it lazily so a late env change (tests, frozen launcher) is
honoured.

Cross-agent import rule: agents must not import from each other — putting the
helpers in ``_shared`` keeps PW / PR / anyone else on the same directories.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mast._runtime_paths import project_root

__all__ = [
    "project_root",
    "drafts_dir",
    "reviews_dir",
    # reports_dir and version_sort_key were defined and imported elsewhere but
    # missing from this list — an incomplete __all__ is a silent trap for
    # ``from … import *`` and for anyone reading it as the module's inventory.
    "reports_dir",
    "figures_dir",
    "experiment_db_path",
    "v2_experiment_db_path",
    "next_version_path",
    "latest_versions",
    "version_sort_key",
]


def _env_dir(var: str) -> Path | None:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else None


def drafts_dir(*, create: bool = False) -> Path:
    """Manuscript drafts. ``MAST_DRAFTS_DIR`` env > <project_root>/data/drafts.

    **SUPERSEDED (2026-07-29).** Drafts and reports now live inside the owning
    experiment's folder (``<experiment>/reports/<doc-dir>/vNNN.md``, written by
    ``mast.documents.store``). This resolver is kept for migration and for
    reading pre-existing files; nothing new should be written here.
    """
    d = _env_dir("MAST_DRAFTS_DIR") or project_root() / "data" / "drafts"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def reviews_dir(*, create: bool = False) -> Path:
    """Review reports. ``MAST_REVIEWS_DIR`` env > <project_root>/data/reviews.

    **SUPERSEDED (2026-07-29).** Reviews are now documents in the experiment
    folder (``<experiment>/reports/<doc-dir>/``) with ``target_doc_id`` recording
    what they reviewed. Kept for migration and compatibility reads only.
    """
    d = _env_dir("MAST_REVIEWS_DIR") or project_root() / "data" / "reviews"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir(*, create: bool = False) -> Path:
    """Exported self-contained HTML reports. ``MAST_REPORTS_DIR`` env >
    <project_root>/data/reports.

    Separate from drafts/: a draft is the editable working copy, an export is
    the thing you hand to someone. Keeping them apart means an export can be
    regenerated or deleted without touching the versioned draft history.

    **SUPERSEDED (2026-07-29).** Exports now land in ``<experiment>/exports/``
    with a timestamp in the filename (``DocumentStore.export_path``), so
    re-exporting no longer destroys the previous deliverable. Kept for migration
    and compatibility reads only.
    """
    d = _env_dir("MAST_REPORTS_DIR") or project_root() / "data" / "reports"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(*, create: bool = True) -> Path:
    """Where data_processing renders figures, and what ``list_figures`` lists.
    ``MAST_FIGURES_DIR`` env > <project_root>/data/figures.

    **No longer embed_figure's destination (2026-07-29):** it now copies the
    figure into the experiment's own pool ``<experiment>/reports/_assets/`` and
    emits ``../_assets/x.png``, so the link keeps working when the experiment
    folder is moved to another machine. This directory remains the SOURCE pool
    (and the read path for figures embedded by older builds).
    """
    d = _env_dir("MAST_FIGURES_DIR") or project_root() / "data" / "figures"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def experiment_db_path() -> Path:
    """The v1 experiment-log DB (the one Records/feedback write to).

    ``MAST_EXPERIMENT_DB`` env > <project_root>/experiments/mast_experiments.db
    — the same canonical location ``mast.config`` uses, NOT the old
    ``data/experiments.db`` guess that never matched a real deployment.
    """
    env = os.environ.get("MAST_EXPERIMENT_DB", "").strip()
    if env:
        return Path(env).expanduser()
    return project_root() / "experiments" / "mast_experiments.db"


def v2_experiment_db_path() -> Path:
    """The **v2** records DB (``logging/v2`` — campaigns / plans / claims …).

    ``MAST_DATA_DIR`` env > ``<project_root>/experiments/mast_experiments_v2.db``.

    ⚠️ This is NOT byte-for-byte ``logging.v2.storage.open_store()``'s default.
    That function falls back to ``Path(".")`` — the **current working directory**
    — when ``MAST_DATA_DIR`` is unset, i.e. the store it opens depends on where
    the process happened to be started from. Every other data path in this module
    resolves through ``project_root()`` (which honours ``MAST2_PROJECT_ROOT``),
    and two of them are the same physical directory in every real deployment:
    dev runs from the repo root, and the frozen build's CWD is the exe directory.

    The divergence matters in exactly two places and both are why this exists:
      * a **test** can redirect ``MAST2_PROJECT_ROOT`` but has no way to redirect a
        CWD-relative default without chdir'ing the whole process — so an artifact
        probe built on ``open_store()``'s default would read the developer's real
        campaigns and report 「已产出」 in a fixture that redirected everything else;
      * a caller that resolves the path ITSELF (as the campaign tools do) can open
        the store **read-only-safe**, since ``open_store`` creates the file and the
        full schema as a side effect of merely asking where it is.

    Callers that must agree with the live v2 writers should pass this path INTO
    ``open_store(path)`` rather than calling ``open_store()`` bare.
    """
    env = os.environ.get("MAST_DATA_DIR", "").strip()
    root = Path(env).expanduser() if env else project_root()
    return root / "experiments" / "mast_experiments_v2.db"


_SLUG_RE = re.compile(r"[^\w一-鿿-]+")
_VER_RE = re.compile(r"_v(\d+)$")


def _slug(stem: str, fallback: str) -> str:
    s = _SLUG_RE.sub("_", stem.strip()).strip("_")
    return s or fallback


def next_version_path(dir_: Path, stem: str, suffix: str = ".md",
                      fallback_stem: str = "untitled") -> Path:
    """Next free ``<stem>_vNNN<suffix>`` inside ``dir_``.

    **NEW CODE MUST NOT CALL THIS.** It is a check-then-act (TOCTOU) race: it
    globs for the highest existing version, returns a path, and the caller writes
    it later — with no lock, no ``O_EXCL``, no atomic create anywhere in between.
    There are at least four concurrent writers of these files (save_draft,
    save_review, ``PUT /api/documents``, ``POST /api/artifacts/{id}/edit``); any
    two that overlap compute the SAME ``_v003`` and the second write **silently
    overwrites the first**. That is precisely the hole in the "never overwrites"
    promise this function was introduced to keep.

    Version allocation now belongs to ``mast.documents.store``, which claims the
    number with ``open(vNNN.md, 'x')`` (atomic across processes, retrying on
    ``FileExistsError``) before writing through a ``.part`` + ``os.replace``.

    Kept only for migrating the old ``data/{drafts,reviews}`` files and for
    reading them. Versioning was and remains the point: every save lands as a NEW
    file so the full history of a manuscript and its reviews survives (feedback
    #108 — "评审记录也应该保存，论文的历史版本也应该保存").
    """
    stem = _slug(stem, fallback_stem)
    # Strip a caller-supplied _vNN so "report_v2" doesn't become report_v2_v001.
    stem = _VER_RE.sub("", stem)
    existing = 0
    for p in dir_.glob(f"{stem}_v*{suffix}"):
        m = _VER_RE.search(p.stem)
        if m:
            existing = max(existing, int(m.group(1)))
    return dir_ / f"{stem}_v{existing + 1:03d}{suffix}"


def latest_versions(dir_: Path, suffix: str = ".md") -> list[Path]:
    """All ``suffix`` files in ``dir_``, most recently modified first.

    Ties on mtime are broken by the ``_vNNN`` version number. Sorting on mtime
    alone is not enough: the filesystem timestamp granularity here is about a
    millisecond, and two saves in the same millisecond are common — measured on
    this machine, **70.5% of back-to-back writes land on an identical mtime**.
    ``sorted()`` is stable, so a tie fell through to ``iterdir()`` order, which
    on NTFS is filename order, which puts ``_v001`` ahead of ``_v002``.

    The consequence was that ``load_draft("current")`` handed the reviewer the
    version BEFORE the operator's edit — an edit that had in fact been saved
    correctly. It reproduced on roughly one run in five, which is exactly the
    frequency that gets written off as flakiness (2026-07-27: it was misread as
    leftover state from a preceding test, twice).
    """
    if not dir_.is_dir():
        return []
    return sorted(
        (p for p in dir_.iterdir() if p.suffix.lower() == suffix),
        key=version_sort_key, reverse=True,
    )


def version_sort_key(p: Path) -> tuple:
    """Newest-first sort key for versioned files: ``(mtime, version)``.

    Use this ANYWHERE "the latest draft/review" is resolved. Four call sites
    each wrote ``max(candidates, key=lambda p: p.stat().st_mtime)`` by hand and
    each carried the same tie bug — see :func:`latest_versions` for the
    measurements. A shared key is also the only way this stays fixed: the next
    "give me the newest one" will be written by copying one of these lines.
    """
    m = _VER_RE.search(p.stem)
    return (p.stat().st_mtime, int(m.group(1)) if m else -1)
