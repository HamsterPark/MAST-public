"""Pydantic request/response models for the Vision + scan-map + experimental
one-shot slice of the typed API (TS-rewrite Phase 3, domain I/J read + one-shot).

CRITICAL CONTRACT: responses NEVER carry ndarrays / tensors / file handles. The
only image bytes that ever cross the wire are base64 PNG *thumbnails* (the
``image_b64`` fields), everything else is scalar metadata. This keeps the seam
serializable, cheap, and free of the heavy-array poisoning the core forbids in
checkpoints.

Every response model carries a ``degraded: bool``. When the live core (the
BufferService / vision module / experiment storage / IC skills) is not wired
into this API process — standalone dev — the handlers return an empty-but-valid
payload with ``degraded=True`` instead of 500-ing.

Re-uses core Pydantic models where they already exist (none in this slice are a
1:1 fit — the buffer schemas are frozen producer-side models that would leak
``mask_rle`` bytes / monotonic ns — so the API exposes flattened, b64-only view
models here).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Alert level for the vision pulse ribbon (derived, never authored by the API).
AlertLevel = Literal["idle", "info", "warn", "critical"]
# Trend of recent vision activity (events/s rising / steady / falling / idle).
Trend = Literal["idle", "rising", "steady", "falling"]


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/recent
# ─────────────────────────────────────────────────────────────────────

class RecentFrame(BaseModel):
    """One recent annotated vision frame / event for the recent strip.

    ``image_b64`` is an OPTIONAL base64 PNG thumbnail (data-URI-less; the client
    prefixes ``data:image/png;base64,``). It is None when the originating event
    carried only a file path we did not decode (decoding .sxm is over the recent
    strip's latency budget — the heavy preview is fetched on demand elsewhere).
    NEVER a raw array."""

    seqno: int = 0
    kind: str = ""
    severity: str = "info"
    cause_ref: Optional[str] = None
    t_wall: float = Field(0.0, description="approx wall-clock seconds (epoch) of the frame")
    file_path: Optional[str] = Field(None, description="source .sxm/PNG path if the event carried one")
    image_b64: Optional[str] = Field(None, description="base64 PNG thumbnail; None if not rendered")
    summary: str = ""


class VisionRecentResponse(BaseModel):
    frames: list[RecentFrame] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/pulse
# ─────────────────────────────────────────────────────────────────────

class VisionPulseResponse(BaseModel):
    """Concise rollup of recent Vision-Buffer activity for the Chat-tab ribbon.

    ``dino_score`` is the latest tip-quality confidence from the vision model
    (0..1, the DINOv3 / legacy head's score) — None when nothing has been
    assessed yet. ``alert_level`` / ``trend`` are derived from the recent event
    window + buffer stats; the API performs no business logic beyond rollup."""

    dino_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="latest tip-quality confidence")
    tip_quality: Optional[str] = Field(None, description="latest tip quality label (good/degraded/bad/unknown)")
    alert_level: AlertLevel = "idle"
    trend: Trend = "idle"
    recent_count: int = Field(0, description="events in the recent window (default 5 min)")
    critical_count: int = 0
    dropped_oldest: int = Field(0, description="buffer events_dropped_oldest counter")
    events_published: int = 0
    rate_per_s: float = Field(0.0, description="recent events / second over the window")
    degraded: bool = False
    safe_mode: bool = Field(
        False,
        description=("SAFE 模式生效中：tip_quality/dino_score 是被覆写后的值（一律 good），"
                     "真实判定见里程碑事件 payload 的 safe_mode_raw。UI 应标注，"
                     "以免把覆写值当成真实针尖状态。"))


# ─────────────────────────────────────────────────────────────────────
# GET /api/scan-map
# ─────────────────────────────────────────────────────────────────────

class XYZ(BaseModel):
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    z_m: Optional[float] = None


class ScanFrame(BaseModel):
    """The live current scan frame footprint (metres, stage frame)."""

    center_x_m: Optional[float] = None
    center_y_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    angle_deg: float = 0.0


class ScanImage(BaseModel):
    """A saved .sxm scan placed on the map by its REAL stage footprint — a live
    surface mosaic underlay. ``image_b64`` is a base64 PNG thumbnail (never a raw
    array); the centre/size/angle are metres/degrees in the Nanonis stage frame so
    the client can draw the image exactly where it was scanned."""

    image_b64: str = ""
    center_x_m: Optional[float] = None
    center_y_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    angle_deg: float = 0.0
    name: str = ""
    path: str = ""
    stale_epoch: bool = Field(
        default=False,
        description=(
            "True when this .sxm was written BEFORE the most recent lateral "
            "coarse move — its header coordinates belong to a dead frame, so it "
            "is drawn over surface it never imaged. Render it faded, exactly "
            "like a marker whose coord_epoch is behind current_epoch. A saved "
            "scan has no coord_epoch of its own; the file's mtime against the "
            "last coarse_move timestamp is the only link there is."),
    )


class MapMarkerView(BaseModel):
    """One spatial marker on the experiment map — a flat, JSON-safe view of
    ``mast.io.exp_map.MapMarker`` (no numpy, no figure)."""

    kind: str = "move"
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    w_m: Optional[float] = None
    h_m: Optional[float] = None
    angle_deg: float = 0.0
    label: str = ""
    skill_name: str = ""
    status: str = "done"
    source: str = "skill"
    timestamp: str = ""
    coord_epoch: Optional[int] = Field(
        default=None,
        description=("coordinate-system generation this marker's xy belongs to; "
                     "compare with ScanMapResponse.current_epoch — an older "
                     "generation means a lateral coarse move has since made this "
                     "coordinate meaningless. None on live/planned overlays."))


class ScanMapImportRequest(BaseModel):
    """Import the operator's OWN scans/spectra into the map by path — a FILE
    (.sxm/.dat/.3ds) or a DIRECTORY (imported recursively). Complements the
    auto-capture daemons for files saved outside the searched dirs, older than
    the recent window, or that the operator simply wants to curate onto the map."""

    path: str = Field(description="a .sxm/.dat/.3ds file, or a folder of them")
    recursive: bool = Field(True, description="recurse into subfolders when path is a directory")


class ScanMapImportResponse(BaseModel):
    ok: bool = False
    imported: int = 0
    scans: int = Field(0, description="how many .sxm placed as scan footprints")
    spectra: int = Field(0, description="how many .dat/.3ds placed as spectrum markers")
    skipped: int = Field(0, description="files with no usable stage xy in their header")
    message: str = ""
    degraded: bool = False


class ScanMapResponse(BaseModel):
    """The live experiment map: current frame + tip xyz + history markers.

    ``image_b64`` is an optional base64 PNG render of the whole map (None when
    no live state/markers are available or rendering is skipped). The structured
    ``frame`` / ``tip_xyz`` / ``markers`` are always present so the TS client can
    re-render natively without decoding the PNG."""

    frame: Optional[ScanFrame] = None
    tip_xyz: XYZ = Field(default_factory=XYZ)
    markers: list[MapMarkerView] = Field(default_factory=list)
    marker_count: int = 0
    sample_label: str = ""
    image_b64: Optional[str] = None
    scan_images: list[ScanImage] = Field(
        default_factory=list,
        description="saved .sxm scans placed by real stage footprint (surface mosaic underlay)")
    current_epoch: int = Field(
        default=0,
        description=("live coordinate-system generation (count of lateral coarse "
                     "moves in this scope); markers with a lower coord_epoch are "
                     "drawn faded because their coordinates are no longer valid"))
    plan_title: str = Field(
        default="",
        description=("title of the published route, if any. The steps themselves "
                     "ride in ``markers`` with status='planned', in execution "
                     "order — a client numbers them from that order and needs no "
                     "further fields. Only the title cannot be derived."))
    piezo_half_range_m: Optional[float] = Field(
        default=None,
        description=("half-width of the reachable piezo area, metres. Carried on "
                     "the map poll (not only on the analysis call) so a client can "
                     "offer a whole-range view without first running the ~1 s "
                     "analysis. None when the config is unavailable."))
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# GET /api/scan-map/analysis  — what the markers MEAN for where to go next
# ─────────────────────────────────────────────────────────────────────

class AvoidZoneView(BaseModel):
    """A keep-out disc: somewhere the surface is damaged or contaminated."""

    x_m: float
    y_m: float
    radius_m: float
    kind: str = ""
    label: str = ""


class NextPositionView(BaseModel):
    """A recommended scan-frame centre, with the reasoning that chose it."""

    x_m: Optional[float] = None
    y_m: Optional[float] = None
    strategy: str = ""
    reason: str = ""
    ring_index: Optional[int] = None
    candidates_left: int = 0


class CoarseAdviceView(BaseModel):
    """Whether to give up on this patch of surface and relocate."""

    suggest: bool = False
    reasons: list[str] = Field(default_factory=list)


class ScanMapAnalysisResponse(BaseModel):
    """Programmatic read of the scan map: coverage, damage, and where to go next.

    Computed by ``mast.io.map_analysis`` from the recorded markers — the SAME
    function the agent's tools call, so the panel next to the map and the agent
    can never be looking at different conclusions. That is the point of the
    button: these decisions are made in code, and the operator gets to inspect
    them.

    All percentages are of the whole reachable piezo area. Note that
    ``usable_pct`` does NOT subtract already-scanned area — re-imaging a clean
    spot, or going back to take spectra there, is ordinary work. Only damage
    consumes surface. ``usable_unscanned_pct`` is the survey figure."""

    current_epoch: int = 0
    markers_total: int = 0
    markers_current_epoch: int = 0
    coverage_pct: float = 0.0
    usable_pct: float = 0.0
    usable_unscanned_pct: float = 0.0
    strategy: str = ""
    frame_size_m: Optional[float] = None
    piezo_half_range_m: float = 1.5e-6
    avoid_zones: list[AvoidZoneView] = Field(default_factory=list)
    damage_counts: dict[str, int] = Field(default_factory=dict)
    next_position: Optional[NextPositionView] = None
    upcoming: list[NextPositionView] = Field(
        default_factory=list,
        description=("the next few positions this strategy would walk, first "
                     "one first. ``upcoming[0]`` IS ``next_position``; the rest "
                     "additionally avoid each other, so the list is a route that "
                     "could be executed rather than N restatements of one "
                     "recommendation. Shorter than the endpoint asked for means "
                     "the surface really has that few positions left."))
    coarse_advice: CoarseAdviceView = Field(default_factory=CoarseAdviceView)
    sts_points: list[XYZ] = Field(
        default_factory=list,
        description="spectroscopy positions in this generation (z unused)")
    sts_total: int = 0
    pending_plan_steps: int = 0
    pending_already_scanned: int = 0
    route_truncated: bool = Field(
        default=False,
        description=("the candidate route hit its length cap; the recommended "
                     "position is still the first acceptable one, but "
                     "candidates_left is a floor rather than a count"))
    degraded: bool = False
    detail: str = ""


class RecordCoarseMoveRequest(BaseModel):
    """Backfill a coarse move the operator made by hand in Nanonis.

    MAST cannot observe those: ``HardwareState`` does not poll the coarse motor,
    and the move leaves no lasting trace in any status it does poll. Recording it
    starts a new coordinate generation FROM NOW — it does not rewrite history,
    because there is no way to know when during the record the move happened."""

    direction: str = ""
    steps: int = 0
    note: str = ""


class RecordCoarseMoveResponse(BaseModel):
    ok: bool = False
    new_coord_epoch: int = 0
    message: str = ""
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# GET /api/coarse-map  — the stage-scale map (steps, not metres)
# ─────────────────────────────────────────────────────────────────────

class CoarseSiteView(BaseModel):
    """One patch of the SAMPLE the tip has worked on — one coordinate generation.

    Positions are in coarse-motor STEPS, from an open-loop odometer, and
    ``uncertainty_steps`` is not decoration: draw the site as a blob of that
    radius, never as a point. ``approx_*_um`` is an annotation for humans and
    exists only when the operator has entered a step calibration; nothing
    navigates by it, because step size drifts with drive amplitude, load and
    (strongly) temperature."""

    index: int = 0
    x_steps: int = 0
    y_steps: int = 0
    position_known: bool = True
    uncertainty_steps: float = 0.0
    is_current: bool = False
    temperature_k: Optional[float] = None
    first_ts: str = ""
    last_ts: str = ""
    summary: dict[str, int] = Field(default_factory=dict)
    approx_x_um: Optional[float] = None
    approx_y_um: Optional[float] = None


class RelocationSuggestionView(BaseModel):
    axis: str = ""
    direction: str = ""
    steps: int = 0
    lands_at_steps: list[int] = Field(default_factory=list)
    clearance_steps: float = 0.0
    reason: str = ""


class CoarseMapResponse(BaseModel):
    """The same map and the same suggestion the agent's ``get_coarse_map`` returns.

    One computation, two audiences — the reason the operator can check what the
    agent is acting on rather than a parallel implementation of it."""

    sites: list[CoarseSiteView] = Field(default_factory=list)
    current_index: int = 0
    position_known: bool = True
    suggestion: Optional[RelocationSuggestionView] = None
    note: str = ""
    budget_used_steps: dict[str, int] = Field(default_factory=dict)
    axis_step_budget: int = 0
    site_spacing_steps: int = 0
    piezo_half_range_m: float = 1.5e-6
    step_m: Optional[float] = None
    vacuum_allow: bool = False
    vacuum_reason: str = ""
    coarse_drive_declared: bool = False
    coarse_drive_note: str = ""
    degraded: bool = False
    detail: str = ""


class VacuumAttestRequest(BaseModel):
    """The operator signing that the chamber pressure is safe for coarse motion.

    Needed only when the gauge cannot answer — no gauge fitted, gauge offline,
    reading over range, reading stale. The signature is time-limited and
    process-local: a restart is a change of scene, and re-signing costs ten
    seconds while an authorisation that outlives its conditions costs a stack."""

    reason: str = Field(
        description="vented_to_atmosphere | high_vacuum_gauge_unavailable")
    signed_by: str = ""
    ttl_hours: float = 8.0
    note: str = ""


class VacuumInterlockResponse(BaseModel):
    allow: bool = False
    reason: str = ""
    source: str = "none"
    pressure_pa: Optional[float] = None
    age_s: Optional[float] = None
    mode: str = ""
    over_range: bool = False
    attested: bool = False
    attestation_remaining_h: Optional[float] = None
    degraded: bool = False
    detail: str = ""


# ─────────────────────────────────────────────────────────────────────
# POST /api/experimental/fft
# ─────────────────────────────────────────────────────────────────────

class FFTRequest(BaseModel):
    """One-shot rfft on an already-captured trace.

    The caller supplies the raw ``samples`` (a 1-D float list — NOT an ndarray)
    plus the sampling rate; the API computes the one-sided spectrum via the core
    ``mast.webui.exp_capture.compute_fft`` helper. Either ``fs_hz`` or
    ``duration_s`` must be given so a frequency axis can be derived."""

    samples: list[float] = Field(default_factory=list, description="time-domain samples (1-D)")
    fs_hz: Optional[float] = Field(None, gt=0.0, description="sample rate (Hz); preferred")
    duration_s: Optional[float] = Field(None, gt=0.0, description="trace duration (Hz derived) if fs_hz absent")
    window: Literal["hann", "hamming", "rect"] = "hann"
    output: Literal["magnitude", "power"] = "magnitude"
    detrend: bool = True
    unit: str = "A"
    channel_name: str = "signal"


class FFTResponse(BaseModel):
    """One-sided spectrum. ``freqs_hz`` / ``spectrum`` are plain float lists
    (never ndarrays). ``ok`` is False / ``degraded`` True when the transform
    could not run (too few samples, or numpy unavailable in standalone)."""

    ok: bool = False
    freqs_hz: list[float] = Field(default_factory=list)
    spectrum: list[float] = Field(default_factory=list)
    n_samples: int = 0
    fs_hz: float = 0.0
    nyquist_hz: float = 0.0
    df_hz: float = 0.0
    window: str = "hann"
    output: str = "magnitude"
    unit: str = "A"
    channel_name: str = "signal"
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# POST /api/experimental/mosaic
# ─────────────────────────────────────────────────────────────────────

class MosaicRequest(BaseModel):
    """Stitch every .sxm in ``directory`` into one big-canvas overview, placed
    by each scan's real stage xy. Returns a b64 PNG of the canvas — the canvas
    ndarray itself NEVER crosses the wire."""

    directory: str = Field(description="folder of .sxm scans to stitch")
    channel: str = "Z"
    recursive: bool = False
    line_normalize: bool = False
    cmap: str = "viridis"
    label: str = ""


class MosaicScanMeta(BaseModel):
    path: str = ""
    cx: Optional[float] = None
    cy: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None
    channel: str = ""


class MosaicResponse(BaseModel):
    """Mosaic result: scalar geometry + a b64 PNG canvas thumbnail. ``ok`` is
    False / ``degraded`` True when no valid scans were found or the core mosaic
    module is unavailable (standalone)."""

    ok: bool = False
    image_b64: Optional[str] = None
    placed: int = 0
    n_input: int = 0
    extent_m: Optional[list[float]] = Field(None, description="[x_min, x_max, y_min, y_max] metres")
    res_m_per_px: float = 0.0
    canvas_px: Optional[list[int]] = Field(None, description="[W, H] pixels")
    angle_warning: bool = False
    scans_meta: list[MosaicScanMeta] = Field(default_factory=list)
    error: str = ""
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# POST /api/experimental/monitor/start  &  /stop
# ─────────────────────────────────────────────────────────────────────

class MonitorStartRequest(BaseModel):
    """Start a long-term single-channel monitor (periodic reads → CSV on a
    daemon thread inside the live core). The API only relays the request to the
    live core; it owns no monitor thread itself."""

    channel: str = Field(description="signal channel index (as string) or 'current'")
    interval_s: float = Field(5.0, ge=0.5, le=600.0, description="seconds per sample")


class MonitorStatus(BaseModel):
    running: bool = False
    channel: str = ""
    interval_s: float = 0.0
    count: int = 0
    last_value: Optional[float] = None
    unit: str = ""
    last_t: str = ""
    csv_path: str = ""
    error: str = ""


class MonitorActionResponse(BaseModel):
    """Result of a monitor start/stop. ``ok`` False + ``degraded`` True when no
    live core (no app / IC skills) is wired — the API never spins hardware."""

    ok: bool = False
    status: MonitorStatus = Field(default_factory=MonitorStatus)
    message: str = ""
    degraded: bool = False


__all__ = [
    "AlertLevel",
    "Trend",
    "RecentFrame",
    "VisionRecentResponse",
    "VisionPulseResponse",
    "XYZ",
    "ScanFrame",
    "ScanImage",
    "MapMarkerView",
    "ScanMapResponse",
    "ScanMapImportRequest",
    "ScanMapImportResponse",
    "FFTRequest",
    "FFTResponse",
    "MosaicRequest",
    "MosaicScanMeta",
    "MosaicResponse",
    "MonitorStartRequest",
    "MonitorStatus",
    "MonitorActionResponse",
]
