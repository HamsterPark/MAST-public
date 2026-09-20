"""Pydantic models for 数据图库 → 出图 (figures from marks and series).

gallery workflow — preprocess → display → human marking → **figures** — and are
made by ``mast.gallery.figures`` from the marks document, the index and the raw
files. Products live under ``<gallery state dir>/figures/<category>/``; every
figure writes a ``<base>.figure.json`` next to its files (inputs, options, key
numbers), which is what this API lists.

Kept in its own module (not ``schemas_gallery``) so the figures slice can grow
without touching the base gallery contract.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from mast.api.schemas_gallery import GalleryBuildError

FigureKind = Literal[
    "marked_frames",   # batch: marked frames + members of series with <= N frames (+ grids)
    "frame_sheet",     # frames in ``ids``: 原版 | 逐行调平 side by side
    "grid_sheets",     # grids in ``ids`` (empty = every grid in the index): every bias layer
    "sts_lines",       # spectra series in ``series`` = blocks of ONE line: 4 figures
    "sts_stitch",      # spectra in ``ids`` (empty = marked singles, grouped by directory)
    "series_slides",   # one frame series: 16:9 pages, column = scan angle, row = round
    "series_stack",    # one frame series: rigid + lattice-corrected stack, npy, CSV
]

FigureCategory = Literal["frames", "grids", "sts_lines", "sts_stitch", "series"]


class GalleryFigureFile(BaseModel):
    name: str
    url: str = Field(..., description="download URL (/api/gallery/figures/file/...?v=)")
    preview_url: Optional[str] = Field(
        None, description="480 px JPEG preview URL; images only"
    )
    ext: str = Field(..., description="png / jpg / csv / npy / json")
    size: int = 0
    mtime: float = 0.0


class GalleryFigureEntry(BaseModel):
    """One generated figure: a base name and every file that belongs to it."""

    key: str = Field(..., description="<category>/<base>")
    category: FigureCategory
    kind: FigureKind
    title: str = ""
    base: str
    files: list[GalleryFigureFile] = Field(default_factory=list)
    created: str = ""
    ids: list[str] = Field(default_factory=list, description="item ids the figure was made from")
    series: list[str] = Field(default_factory=list, description="series ids the figure was made from")
    options: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="key numbers written by the maker (colour spans, r, factors, "
                    "station assignment counts, singular values, ...)",
    )


class GalleryFigureCategory(BaseModel):
    key: FigureCategory
    title: str
    figures: list[GalleryFigureEntry] = Field(default_factory=list)


class GalleryFiguresList(BaseModel):
    categories: list[GalleryFigureCategory] = Field(default_factory=list)
    degraded: bool = False
    detail: Optional[str] = None


class GalleryFigureRequest(BaseModel):
    """Start one figure job.

    ``options`` by kind (all optional; defaults in brackets):

    * ``marked_frames``: ``max_series_frames`` [100], ``include_grids`` [true]
    * ``sts_lines``: ``line_name`` [common part of the series names],
      ``station_marks`` [{}] e.g. ``{"5": "V"}``, ``bad_from`` [{}] ``{series_id: k}``
      (spectra from the k-th, 0-based acquisition order, on are drawn red and left
      out of the means), ``exclude_rejected`` [true] (series rated 排除 are left
      out of the means)
    * ``sts_stitch``: ``kappa_per_nm`` [omitted: no theoretical multiplier], ``group_by_dir`` [true when ``ids`` is empty]
    * ``series_slides``: ``page_w`` [3840], ``page_h`` [2160]
    * ``series_stack``: ``anchor`` ["darkest" | "brightest" | "center"], ``fov_nm``
      [first input frame width], ``min_cover`` [0.25]
    """

    kind: FigureKind
    ids: list[str] = Field(default_factory=list)
    series: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


FigureJobPhase = Literal["idle", "running", "done", "cancelled", "error"]


class GalleryFigureJobStatus(BaseModel):
    """The single figure job slot (one figure job at a time, process-wide)."""

    running: bool = False
    phase: FigureJobPhase = "idle"
    kind: Optional[FigureKind] = None
    done: int = 0
    total: int = 0
    made: list[str] = Field(default_factory=list, description="figure keys written by this job")
    errors: list[GalleryBuildError] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    started: Optional[str] = None
    finished: Optional[str] = None
    message: str = ""
    degraded: bool = False
    detail: Optional[str] = None


class GalleryStsLinePlanRequest(BaseModel):
    series: list[str] = Field(..., description="spectra series = blocks of one line")
    options: dict[str, Any] = Field(default_factory=dict, description="same keys as sts_lines")


class GalleryStsAssigned(BaseModel):
    id: str
    station: int
    residual_nm: float
    order: int = Field(..., description="acquisition order within its series, 0-based")


class GalleryStsBlockPlan(BaseModel):
    series: str
    label: str
    n: int
    offset_nm: float = Field(0.0, description="drift offset along the line vs the reference block")
    assigned: list[GalleryStsAssigned] = Field(default_factory=list)
    unmatched: list[str] = Field(default_factory=list, description="spectra farther than 0.35 step from any station")


class GalleryStsLinePlan(BaseModel):
    """Dry run of the station inference (design D17) — positions come from the
    index, so this reads no raw file and is cheap enough to call while a dialog
    is open."""

    ok: bool = False
    line_name: str = ""
    direction: list[float] = Field(default_factory=list, description="unit vector (x, y) of the line")
    step_nm: Optional[float] = None
    n_stations: int = 0
    reference: str = Field("", description="series id used as the station reference")
    blocks: list[GalleryStsBlockPlan] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False
    detail: Optional[str] = None
