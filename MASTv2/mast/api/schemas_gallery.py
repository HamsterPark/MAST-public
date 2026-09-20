"""Pydantic models for the 数据图库 slice.


* **incremental preprocessing** (``mast.gallery.build``): inventory the configured
  data roots → run the calibrated judges on new frames → render thumbnails →
  write ``index.json``;
* **a marks store** (``mast.gallery.marks``): ``marks.json`` is the single source
  of truth for ratings / tags / notes / series / spectrum→frame anchors, and every
  write also refreshes ``marks.md`` / ``marks.csv`` / ``marks_series.csv``;
* **a single-page UI** (``frontend/src/components/gallery``).

This file is the single source of types for the slice. Two responses are
DOCUMENTED here but served straight from disk (``GalleryIndex`` and
``GalleryMarksDoc``): validating 5000+ index items through Pydantic on every page
load buys nothing, and re-serialising a marks document would pad every stored mark
with default keys it never had. The models still land in ``openapi.json`` so the
frontend gets real types.

``GalleryItem`` keys are SHORT on purpose. They are the keys of the standalone
gallery's ``data.js`` (plus a few additions marked below), so the two can be
diffed field by field when checking the port against the original.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ── config ─────────────────────────────────────────────────────────────


class GalleryRoot(BaseModel):
    """One data root the gallery indexes.

    ``name`` is the stable half of every item id (``<name>/<relative path>``):
    when the data moves, edit ``path`` and keep ``name`` and every mark follows."""

    name: str = Field(..., description="stable root name; the first segment of every item id")
    path: str = Field(..., description="absolute directory path")
    enabled: bool = True
    exists: bool = Field(
        False, description="filled on responses: whether the directory exists right now"
    )


class GalleryRootSuggestion(BaseModel):
    """A directory worth offering as a root. Offered, never added implicitly."""

    path: str
    why: str


class GalleryConfigResponse(BaseModel):
    ok: bool = True
    state_dir: str = Field("", description="where caches, thumbnails, index and marks live")
    roots: list[GalleryRoot] = Field(default_factory=list)
    suggestions: list[GalleryRootSuggestion] = Field(default_factory=list)
    workers: int = 2
    degraded: bool = False
    detail: Optional[str] = None


class GalleryRootIn(BaseModel):
    name: Optional[str] = Field(
        None, description="omitted ⇒ derived from the directory name (made unique)"
    )
    path: str
    enabled: bool = True


class GalleryConfigUpdate(BaseModel):
    """Replace the root list (and optionally the worker count). Validation errors
    come back as ``ok=false`` + ``detail`` and nothing is written."""

    roots: list[GalleryRootIn] = Field(default_factory=list)
    workers: Optional[int] = Field(default=None, ge=1, le=16)


# ── build ──────────────────────────────────────────────────────────────


class GalleryBuildRequest(BaseModel):
    force: bool = Field(
        False,
        description="re-render and re-judge everything; the inventory cache is still "
                    "keyed on (size, mtime_ns), so unchanged headers are not re-read",
    )


class GalleryBuildError(BaseModel):
    id: str
    why: str


GalleryPhase = Literal["idle", "inventory", "render", "dups", "index", "done", "cancelled", "error"]


class GalleryBuildStatus(BaseModel):
    """State of the (single, process-wide) background build."""

    running: bool = False
    phase: GalleryPhase = "idle"
    started: Optional[str] = None
    finished: Optional[str] = None
    done: int = Field(0, description="render/analysis tasks finished in this build")
    total: int = Field(0, description="render/analysis tasks queued in this build")
    n_files: int = Field(0, description="index entries after copy-collapse")
    n_new: int = 0
    n_changed: int = 0
    n_render: int = 0
    n_analysis: int = 0
    n_failed: int = 0
    errors: list[GalleryBuildError] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list, description="most recent log lines")
    message: str = ""
    degraded: bool = False
    detail: Optional[str] = None


# ── index ──────────────────────────────────────────────────────────────


class GalleryItem(BaseModel):
    """One frame (``k='f'``), point spectrum (``'s'``) or grid spectrum (``'g'``).

    Field meanings are tabulated in the design doc §4.1. Additions over the
    standalone gallery's ``data.js``: ``p pf cp cpl r0 r1 ar hl seg`` (and ``hn``
    on grids). ``sm`` (a sample kind hard-coded from file names) is deliberately
    absent — see the doc, trap T10."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="<root name>/<path relative to the root>")
    k: Literal["f", "s", "g"]
    d: str = Field(..., description="directory key: <root name>/<relative parent dir>")
    fn: str
    p: str = Field("", description="absolute path of the representative file")
    pf: str = Field("", description="file-name prefix (stem minus the trailing counter)")
    t: Optional[float] = Field(None, description="start, epoch s (frame: REC time; spectrum/grid: Start time)")
    mt: Optional[float] = Field(None, description="file mtime, epoch s (= when the instrument saved it)")
    ad: str = Field("", description="batch in which the file was first inventoried")
    th: str = Field("", description="thumbnail URL")
    cp: Optional[int] = Field(None, description="byte-identical copies folded into this entry (>1 only)")
    cpl: Optional[list[str]] = Field(None, description="absolute paths of every copy (>1 only)")

    # frame (and grid) geometry / working point
    w: Optional[float] = Field(None, description="frame width, nm")
    hn: Optional[float] = Field(None, description="frame height, nm")
    b: Optional[float] = Field(None, description="bias, V")
    sp: Optional[float] = Field(None, description="setpoint, pA")
    nx: Optional[int] = None
    ny: Optional[int] = None
    ang: Optional[float] = Field(None, description="scan angle, deg; positive = frame rotated clockwise")
    cx: Optional[float] = Field(None, description="centre x, nm")
    cy: Optional[float] = Field(None, description="centre y, nm")
    sd: Optional[Literal["u", "d"]] = Field(None, description="scan direction")
    acq: Optional[float] = Field(None, description="ACQ_TIME, s")
    rows: Optional[int] = Field(None, description="rows that are entirely finite")
    rall: Optional[int] = Field(None, description="rows in the frame")
    r0: Optional[int] = Field(None, description="thumbnail shows oriented rows [r0, r1)")
    r1: Optional[int] = None
    li: Optional[str] = Field(None, description="lock-in channel thumbnail URL, '' when none")
    at: Optional[float] = Field(None, description="atomic-phase angular concentration; 0 = not passed")
    ar: Optional[str] = Field(None, description="why the atomic judge could not decide, '' otherwise")
    hf: Optional[float] = Field(None, description="superstructure candidate/control ratio; 0 = none")
    hl: Optional[str] = Field(None, description="superstructure fraction label, e.g. '(0.5,0)'")
    dup: Optional[str] = Field(None, description="id of the frame this one is a repeat save of")
    seg: Optional[int] = Field(None, description="distinct files from the same acquisition (>=2 only)")

    # spectrum (and grid)
    n: Optional[int] = Field(None, description="points per sweep")
    v0: Optional[float] = None
    v1: Optional[float] = None
    zo: Optional[float] = Field(None, description="Z offset, pm")
    x: Optional[float] = Field(None, description="spectrum position x, nm")
    y: Optional[float] = Field(None, description="spectrum position y, nm")
    sw: Optional[int] = Field(None, description="sweeps")
    lic: Optional[int] = Field(None, description="1 = lower panel is a lock-in channel")
    dn: Optional[int] = Field(None, description="1 = lower panel is a numerical derivative of I(V)")
    ex: Optional[str] = Field(None, description="Experiment name when this is not a bias spectrum")

    # grid
    t1: Optional[float] = Field(None, description="end, epoch s")
    gx: Optional[int] = None
    gy: Optional[int] = None
    have: Optional[int] = Field(None, description="grid points completed")


class GalleryIndex(BaseModel):
    version: int = 1
    generated: str = ""
    built: bool = Field(False, description="false until the first build has written an index")
    roots: list[GalleryRoot] = Field(default_factory=list)
    n_frames: int = 0
    n_spectra: int = 0
    n_grids: int = 0
    last_batch: str = ""
    items: list[GalleryItem] = Field(default_factory=list)
    degraded: bool = False
    detail: Optional[str] = None


# ── marks ──────────────────────────────────────────────────────────────


class GalleryAnchor(BaseModel):
    """Where a spectrum (or a series of them) was taken, pinned to a frame."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="frame item id")
    fn: str = ""
    rel: str = Field("", description="'prev' or 'next' relative to the spectrum")
    dt: Optional[float] = Field(None, description="seconds between spectrum and frame")
    desc: str = ""
    u: Optional[float] = Field(None, description="fraction across the frame from the left")
    v: Optional[float] = Field(None, description="fraction down the frame from the top")
    inside: Optional[bool] = None


class GalleryMark(BaseModel):
    """One item's mark. In a patch, ``{"del": true, "ts": …}`` removes it."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    r: int = Field(0, description="2 = 重点, 1 = 可用, -1 = 排除, 0 = none")
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    t: str = Field("", description="local time of the last edit")
    ts: int = Field(0, description="ms stamp; the server drops edits older than what it holds")
    tt: Optional[float] = Field(None, description="the item's own start time (for sorting)")
    k: str = ""
    meta: str = ""
    anchor: Optional[GalleryAnchor] = None
    deleted: bool = Field(False, alias="del")


class GallerySeries(BaseModel):
    """A named run of consecutive items. ``{"del": true}`` or empty ``ids`` removes it."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = ""
    k: str = Field("", description="f / s / g / mix")
    ids: list[str] = Field(default_factory=list)
    r: int = 0
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    t: str = ""
    ts: int = 0
    anchor: Optional[GalleryAnchor] = None
    deleted: bool = Field(False, alias="del")


class GalleryDirMark(BaseModel):
    """Per-directory state. Stored under ``days`` for file compatibility with the
    standalone gallery, whose directories were dates."""

    model_config = ConfigDict(extra="allow")

    done: bool = False
    note: str = ""
    ts: int = 0


class GalleryMarksDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: int = 1
    rev: int = 0
    updated: str = ""
    tags: list[str] = Field(default_factory=list)
    items: dict[str, GalleryMark] = Field(default_factory=dict)
    series: dict[str, GallerySeries] = Field(default_factory=dict)
    days: dict[str, GalleryDirMark] = Field(default_factory=dict)
    tomb: dict[str, int] = Field(default_factory=dict)
    stomb: dict[str, int] = Field(default_factory=dict)
    degraded: bool = False
    detail: Optional[str] = None


class GalleryMarksPatch(BaseModel):
    items: dict[str, GalleryMark] = Field(default_factory=dict)
    series: dict[str, GallerySeries] = Field(default_factory=dict)
    days: dict[str, GalleryDirMark] = Field(default_factory=dict)
    tags: Optional[list[str]] = Field(None, description="the whole tag table, when it changed")


class GalleryMarksPatchResult(BaseModel):
    ok: bool = False
    rev: int = 0
    updated: str = ""
    degraded: bool = False
    detail: Optional[str] = None


class GalleryMarksImportRequest(BaseModel):
    """Merge a ``marks.json`` (e.g. the standalone gallery's) into this one."""

    doc: dict[str, Any]
    key_prefix: str = Field(
        "",
        description="prepended to item keys, series members and anchor ids — the "
                    "standalone gallery keyed paths relative to its RAW root, so "
                    "importing it needs the root name here (e.g. 'SPM/')",
    )


class GalleryMarksImportResult(BaseModel):
    ok: bool = False
    items: int = 0
    series: int = 0
    days: int = 0
    tags_added: int = 0
    unmatched: int = Field(
        0, description="imported keys with no item in the current index (imported anyway)"
    )
    rev: int = 0
    updated: str = ""
    degraded: bool = False
    detail: Optional[str] = None
