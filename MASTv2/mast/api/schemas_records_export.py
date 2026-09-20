"""Pydantic request/response models for the records_export slice (Parity Wave A).

These shapes expose functionality whose LOGIC still lives in the Python core but
lost its UI in the Gradio→TS rewrite:

* trajectory training-dataset export (``logging.v2.trajectory_export``):
  jsonl / sft / dpo / failure_mining views over the v2 trajectories store.
* one-click full-history ZIP archive (``logging.export_all``).
* scan-file preview / latest discovery (``webui.scan_preview`` + ``mast.io``):
  most-recent .sxm/.dat/.3ds across the data dirs + a base64-PNG preview.
* experiment detail TIMELINE incl. per-action TCP calls + state-diff, mirroring
  ``webui.experiment_viewer._build_action_timeline`` over the v1 ``ActionRecord``
  (``mast.core.types``).

Per the house rules this file is the SINGLE SOURCE OF TYPES for this slice. Every
endpoint has a ``response_model``; every response carries ``degraded`` so the
frontend can render an empty-but-not-broken state when the live core is absent.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── POST /api/trajectories/export ──────────────────────────────────────


class TrajectoryExportRequest(BaseModel):
    """Export agent training/usage trajectories to a training dataset.

    ``format`` selects the view shaped by ``logging.v2.trajectory_export``:
      * ``jsonl``         — one self-contained trajectory per line (iter_full).
      * ``sft``           — SFT samples (intent+context → ordered step seq).
      * ``dpo``           — DPO/preference pairs from edited HITL resolutions.
      * ``failure_mining``— failed/aborted/rolled-back trajectories only.

    ``thread_id`` (jsonl only) restricts to one conversation. ``limit`` caps the
    number of source trajectories scanned. ``inline`` returns the rows in the
    response body (capped by ``max_inline``); the count is always returned."""

    format: str = "jsonl"
    thread_id: Optional[str] = None
    limit: int = Field(default=10000, ge=1, le=100000)
    inline: bool = True
    max_inline: int = Field(default=500, ge=0, le=5000)


class TrajectoryExportResult(BaseModel):
    """Result of a trajectory export. ``count`` is the full number of rows the
    selected view produced; ``rows`` holds up to ``max_inline`` of them when
    ``inline`` is set (the frontend can download/stream the rest separately)."""

    ok: bool = False
    format: str = "jsonl"
    count: int = 0
    rows: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    degraded: bool = False
    detail: Optional[str] = None


# ── POST /api/experiments/export ───────────────────────────────────────


class ExperimentsExportRequest(BaseModel):
    """One-click full-history ZIP archive (``logging.export_all``).

    ``include_heavy`` also bundles large regenerable model weights / caches /
    manuals (the operator's GUI checkbox). ``dest`` optionally overrides the
    output path; default is a timestamped zip under the project ``exports/``."""

    include_heavy: bool = False
    dest: Optional[str] = None


class ExperimentsExportResult(BaseModel):
    """Result of the full-history export. Mirrors the ``export_all`` manifest:
    output path + roll-up counts (file_count / total_bytes / db + skip + error
    tallies). The heavy blocking work runs in the live app's offload worker;
    here we relay the manifest the core returns."""

    ok: bool = False
    dest: Optional[str] = None
    file_count: int = 0
    total_bytes: int = 0
    database_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    include_heavy: bool = False
    created_at: Optional[str] = None
    degraded: bool = False
    detail: Optional[str] = None


# ── GET /api/scans/latest ──────────────────────────────────────────────


class ScanFileLocation(BaseModel):
    """One on-disk copy of a scan file.

    ``kind`` uses the ``file_locations.root_kind`` vocabulary (``origin`` /
    ``experiment`` / ``quarantine``) so the filesystem view and the DB view of
    the same file can be read side by side."""

    path: str
    kind: str = "origin"


class ScanFileEntry(BaseModel):
    """One discovered Nanonis scan/spectroscopy file (``webui.scan_preview``
    ``collect_scan_stats`` shape): path + name + extension + mtime + size.

    ``copies``/``locations`` describe the automatic ingest into the experiment
    folder: one measurement exists as several byte-identical files, and this
    listing used to show one card per copy — identical name, size and timestamp
    on each, so the operator could not tell them apart .
    Copies are folded into ONE entry that says how many there are and where they
    live. Nothing is dropped: ``locations`` names every member, so a wrong fold is
    visible on screen rather than silently hiding a file."""

    path: str
    name: str
    ext: str
    mtime: Optional[float] = None
    size_bytes: Optional[int] = None
    kind: str = Field(
        "origin",
        description="root_kind of THIS entry's representative copy (origin/experiment/quarantine)",
    )
    copies: int = Field(1, description="how many on-disk copies were folded into this entry")
    locations: list[ScanFileLocation] = Field(
        default_factory=list,
        description="every copy, when copies>1; empty for a single-copy file (the path is above)",
    )


class LatestScansResponse(BaseModel):
    """Most-recent .sxm/.dat/.3ds files across the data search dirs (mtime-desc).
    Empty-but-not-broken when no dirs exist / nothing is found."""

    scans: list[ScanFileEntry] = Field(default_factory=list)
    count: int = 0
    search_dirs: list[str] = Field(default_factory=list)
    degraded: bool = False
    # ── type filtering (2026-08-04, 的残余) ──────────────────────
    # ``scans`` is the newest ``n`` of the REQUESTED extensions. The two fields
    # below describe the whole discovery result, not that slice, so a client can
    # show honest chip counts without asking for every file on disk.
    #
    # This exists because filtering client-side on a server-truncated list is a
    # lie: ask for 60 files, get 60 .dat, filter to .sxm, show "SXM (0)" while
    # 26 sit right there. That is exactly, and it comes back wherever
    # the truncation and the filter live on opposite sides of the wire.
    counts_by_ext: dict[str, int] = Field(
        default_factory=dict,
        description="files per extension across the FULL discovery result, before the n-slice",
    )
    total_matched: int = Field(
        0, description="files matching the requested ext across the full result (>= len(scans))"
    )
    # ── paging + copy collapse (2026-08-21) ──────────────────────────────────
    # The client used to ask for a fixed 60 and show "只显示最近 60 个" forever —
    # there was no way to reach file 61. ``offset`` pages the COLLAPSED list, so
    # a page boundary never falls in the middle of one measurement's copies.
    offset: int = Field(0, description="echo of the requested offset into the collapsed list")
    total_collapsed: int = Field(
        0, description="entries after copy-collapse matching the requested ext (<= total_matched)"
    )
    has_more: bool = Field(False, description="offset+len(scans) < total_collapsed")
    counts_by_ext_collapsed: dict[str, int] = Field(
        default_factory=dict,
        description="per-extension counts AFTER copy-collapse; chip labels use these so the "
                    "number on the chip matches the number of cards the filter produces",
    )


# ── GET /api/scans/preview ─────────────────────────────────────────────


class ScanPreviewResponse(BaseModel):
    """Base64-PNG preview of one scan file (``webui.scan_preview``).

    ``image`` is a ``data:image/png;base64,…`` URI (thumbnail-rendered). When
    the file is missing / unsupported / the matplotlib stack is unavailable the
    response degrades with ``found``/``rendered`` flags — never a 500."""

    path: str
    found: bool = False
    rendered: bool = False
    ext: Optional[str] = None
    image: Optional[str] = None
    degraded: bool = False
    detail: Optional[str] = None
    # ── 去衬底 (2026-08-21) ──────────────────────────────────────────────────
    # A raw topograph is mostly sample tilt; the surface is picometres on top of
    # nanometres of ramp. ``flatten`` reports what was ACTUALLY subtracted, which
    # is not always what was asked: ``auto`` resolves to whichever method
    # ``scan_prep.plan_for`` picked (possibly poly2 / masked_line), and any mode
    # falls back to ``raw`` rather than returning no picture at all. The client
    # shows this string, so a fallback is never mistaken for a successful flatten.
    flatten: Optional[str] = Field(
        None, description="flatten method actually applied (raw/plane/line/poly2/masked_line)"
    )
    flatten_why: list[str] = Field(
        default_factory=list,
        description="why auto chose that method (Chinese, from scan_prep.FlattenPlan.why)",
    )
    channel: Optional[str] = Field(None, description=".sxm channel actually rendered")
    width_nm: Optional[float] = None
    height_nm: Optional[float] = None
    bias_v: Optional[float] = None


# ── GET /api/scans/spectrum ────────────────────────────────────────────


class SpectrumSeries(BaseModel):
    """One plottable trace against the sweep axis.

    ``source`` is load-bearing: ``file`` is a channel that was measured, while
    ``numeric`` is dI/dV obtained by differentiating I(V) because the file has no
    lock-in column. Those are not the same measurement and the client labels them
    differently — a numeric derivative of a noisy current drawn under a bare
    "dI/dV" label reads as spectroscopy that was never taken."""

    id: str
    name: str
    values: list[Optional[float]] = Field(
        default_factory=list,
        description="NaN/Inf arrive as null — JSON has no NaN, and a null also "
                    "breaks the line rather than drawing across a gap",
    )
    source: str = "file"


class SpectrumDataResponse(BaseModel):
    """Numeric contents of one .dat/.txt point spectrum.

    Exists because the only way a spectrum reached the frontend was a small PNG
    of the first two columns — no axes, no units, no zoom, every other channel
    (backward sweep, the lock-in that IS the dI/dV) discarded before the wire.

    ``columns`` always lists EVERY column name in the file even when only some
    became series, so what is on screen can be checked against what is in the
    file."""

    path: str
    found: bool = False
    kind: str = Field("", description='"iv" | "iz" | "" — which column is sweeping')
    kind_evidence: str = Field("", description="human-readable basis for kind (Chinese)")
    sweep_name: str = ""
    sweep: list[Optional[float]] = Field(default_factory=list)
    series: list[SpectrumSeries] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    didv_source: Optional[str] = Field(
        None, description='"lockin" | "numeric" | null — three states, never conflated'
    )
    n_points: int = 0
    decimated: bool = Field(False, description="downsampled for transport")
    degraded: bool = False
    detail: Optional[str] = None


# ── GET /api/scans/attribution ─────────────────────────────────────────


class FileAttributionEntry(BaseModel):
    """One row of ``file_locations`` — which experiment/sample a file belongs to.

    ``abs_path`` is reconstructed from the experiment folder layout and can be
    null when the folder cannot be located (renamed root, moved data); the
    ``origin_path`` the ingest recorded is then the only usable path."""

    sha256: str = ""
    rel_path: str = ""
    abs_path: Optional[str] = None
    origin_path: Optional[str] = None
    root_kind: str = "experiment_folder"
    sample_id: Optional[str] = None
    source: str = ""
    size_bytes: int = 0
    status: str = "ok"
    ingested_at: Optional[str] = None


class ScanAttributionResponse(BaseModel):
    """Files belonging to one experiment, from the v2 ``file_locations`` table.

    Separate from ``/api/scans/latest`` on purpose: that endpoint is a pure
    filesystem walk with a clean degradation story, and folding a SQLite query
    into it would put DB availability in the path of the file listing's first
    paint. This one may degrade on its own without taking the listing with it."""

    experiment_id: str
    files: list[FileAttributionEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False
    detail: Optional[str] = None


# ── GET /api/experiments/{id}/timeline ─────────────────────────────────


class TimelineTcpCall(BaseModel):
    """One Nanonis TCP call inside an action's detail (``NanonisCallRecord``)."""

    method: str = ""
    args: str = ""
    error: Optional[str] = None
    elapsed_s: float = 0.0


class TimelineStateDiff(BaseModel):
    """Before/after instrument-state diff for one action (``HardwareState``).

    Only the fields the viewer surfaces (bias / current / Z); each side is None
    when the action did not capture that snapshot."""

    before: Optional[dict[str, Any]] = None
    after: Optional[dict[str, Any]] = None


class TimelineEntry(BaseModel):
    """One action on the experiment timeline (mirrors
    ``experiment_viewer._build_action_timeline``): the skill call + outcome +
    duration, plus the expandable detail (TCP calls + state-diff + context)."""

    id: str
    sample_id: Optional[str] = None
    timestamp: Optional[str] = None
    skill_name: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    success: Optional[bool] = None
    error: Optional[str] = None
    duration_s: float = 0.0
    context: Optional[str] = None
    tcp_calls: list[TimelineTcpCall] = Field(default_factory=list)
    state_diff: TimelineStateDiff = Field(default_factory=TimelineStateDiff)


class ExperimentTimelineResponse(BaseModel):
    """Full experiment-detail timeline: header + ordered action entries with
    TCP calls + state-diff. ``found``/``degraded`` let the frontend show an
    empty-but-not-broken state when storage is unwired or the id is unknown."""

    id: str
    name: Optional[str] = None
    status: Optional[str] = None
    goal_text: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    entries: list[TimelineEntry] = Field(default_factory=list)
    action_count: int = 0
    succeeded: int = 0
    failed: int = 0
    total_duration_s: float = 0.0
    found: bool = False
    degraded: bool = False


__all__ = [
    "TrajectoryExportRequest",
    "TrajectoryExportResult",
    "ExperimentsExportRequest",
    "ExperimentsExportResult",
    "ScanFileLocation",
    "ScanFileEntry",
    "LatestScansResponse",
    "ScanPreviewResponse",
    "SpectrumSeries",
    "SpectrumDataResponse",
    "FileAttributionEntry",
    "ScanAttributionResponse",
    "TimelineTcpCall",
    "TimelineStateDiff",
    "TimelineEntry",
    "ExperimentTimelineResponse",
]
