"""Parity Wave A — records_export domain.

Re-exposes functionality whose LOGIC still lives in the Python core but whose UI
was lost in the Gradio→TS rewrite. Each handler is a THIN relay onto a kept
backend; no business/safety logic lives here.

Endpoints:
  * POST /api/trajectories/export   — training-dataset export (jsonl/sft/dpo/
                                       failure_mining) via logging.v2.trajectory_export.
  * POST /api/experiments/export    — one-click full-history ZIP via logging.export_all.
  * GET  /api/scans/latest          — most-recent .sxm/.dat/.3ds discovery via
                                       webui.scan_preview._collect_scans.
  * GET  /api/scans/preview?path=   — base64-PNG scan preview via webui.scan_preview.
  * GET  /api/experiments/{id}/timeline — experiment detail timeline incl. TCP
                                       calls + state-diff, mirroring
                                       webui.experiment_viewer._build_action_timeline
                                       over the v1 ExperimentStorage / ActionRecord.

GRACEFUL DEGRADATION is mandatory (house rule 2): this router must boot
STANDALONE with no live core wired. Every endpoint that needs a live subsystem /
heavy backend LAZY-imports it INSIDE the handler in try/except and returns a
valid empty/degraded body (``degraded=True``) on any absence or failure — never a
500. Heavy/blocking work (the ZIP export) is owned by the live app's offload
worker; here we only relay.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query, Request

from mast.api.schemas_records_export import (
    ExperimentsExportRequest,
    ExperimentsExportResult,
    ExperimentTimelineResponse,
    FileAttributionEntry,
    LatestScansResponse,
    ScanAttributionResponse,
    ScanFileEntry,
    ScanFileLocation,
    ScanPreviewResponse,
    SpectrumDataResponse,
    TimelineEntry,
    TimelineStateDiff,
    TimelineTcpCall,
    TrajectoryExportRequest,
    TrajectoryExportResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["records_export"])

# Recognised Nanonis scan/spectroscopy extensions (mirrors scan_preview._SCAN_EXTS).
_SCAN_EXTS = (".sxm", ".sm4", ".dat", ".3ds", ".txt", ".csv", ".asc", ".tsv")

# Ceiling on the discovery pass behind ``counts_by_ext``. _collect_scans globs
# every dir regardless, so this only bounds the list we sort and count — but it
# has to be far above any plausible ``n`` or the counts become a truncated tally,
# i.e. the very lie this endpoint's ext filter exists to stop telling.
_DISCOVERY_CAP = 20000


def _parse_ext_filter(ext: str | None) -> set[str] | None:
    """``'.sxm,dat'`` → ``{'.sxm', '.dat'}``. None when nothing was asked for.

    A filter naming only extensions we do not recognise returns an EMPTY set,
    which the caller honours as "match nothing". Falling back to "match
    everything" would answer a question the client did not ask and hand back a
    screen of .dat files to someone who asked for .sxm — the failure #53/#60 are
    about, one layer up."""
    if ext is None or not ext.strip():
        # A blank value is not a request for a type, so it cannot mean "match
        # nothing"; it means the caller left the filter off.
        return None
    raw = [e.strip().lower() for e in ext.split(",")]
    out = {e if e.startswith(".") else f".{e}" for e in raw if e}
    return {e for e in out if e in _SCAN_EXTS}


# ── helpers ────────────────────────────────────────────────────────────


def _scan_search_dirs(ctx, extra: str | None = None) -> list[str]:
    """Best-effort list of dirs to scan for recent files.

    Mirrors the live app's ``_scan_search_dirs`` (config.experiments_dir +
    session paths) when wired; otherwise falls back to the project experiments
    dir. An explicit ``extra`` (query override) always wins. Never raises."""
    # An explicit dir override is EXCLUSIVE — the caller (Data-tab manual browse,
    # tests) wants exactly that dir, not the default fallback set folded in.
    if extra:
        return [extra]
    dirs: list[str] = []
    # A wired live app exposes the very tuple the GUI built.
    app_handle = getattr(ctx, "app", None) or getattr(ctx, "live_app", None)
    live = getattr(app_handle, "_scan_search_dirs", None)
    if live:
        try:
            dirs.extend(str(d) for d in live if d)
        except Exception:  # pragma: no cover - defensive
            pass
    # The Nanonis session dir (Util_SessionPathGet) is where scans ACTUALLY save —
    # frequently OUTSIDE <data_root>/working-sessions. It was never captured before
    # (`_session_path` only ever read, always None) so discovery missed every real
    # .sxm and the Records/Data tab showed no thumbnails (2026-06-29). Resolve it
    # live off the wired app (cached ~15 s); degrades to nothing when no hardware.
    try:
        resolver = getattr(app_handle, "_resolve_session_dir", None)
        sess = resolver() if callable(resolver) else None
        if sess:
            dirs.append(str(sess))
    except Exception:  # pragma: no cover - defensive
        pass
    # Config-derived experiments dir (works standalone too).
    cfg = getattr(ctx, "config", None)
    exp_dir = getattr(cfg, "experiments_dir", None)
    if exp_dir:
        dirs.append(str(exp_dir))
    # Final fallback: the REAL scan output dirs. Scans save to
    # <project_root>/working-sessions/ (skills/builtins/scan_extra.py), NOT
    # <project_root>/experiments — and the live app never sets _scan_search_dirs,
    # so without working-sessions here the Records / Data tab discovered ZERO
    # scans and showed no thumbnails despite valid .sxm files (2026-06-29). Keep
    # experiments too for any legacy layout.
    try:
        from mast._runtime_paths import project_root

        root = project_root()
        dirs.append(str(root / "working-sessions"))
        dirs.append(str(root / "experiments"))
    except Exception:  # pragma: no cover - defensive
        pass
    # De-dup, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _state_to_dict(state) -> dict | None:
    """Surface the bias/current/Z fields the viewer renders off a HardwareState."""
    if state is None:
        return None
    return {
        "bias_v": getattr(state, "bias_v", None),
        "current_a": getattr(state, "current_a", None),
        "z_pos_m": getattr(state, "z_pos_m", None),
        "z_controller_on": getattr(state, "z_controller_on", None),
        "scan_running": getattr(state, "scan_running", None),
    }


# ── POST /api/trajectories/export ──────────────────────────────────────


@router.post("/trajectories/export", response_model=TrajectoryExportResult)
def export_trajectories(
    body: TrajectoryExportRequest, request: Request
) -> TrajectoryExportResult:
    """Export agent training/usage trajectories to a training dataset.

    Relays the read-only views in ``logging.v2.trajectory_export`` over the v2
    store (jsonl / sft / dpo / failure_mining). The v2 store is reached on disk
    independent of ctx; absent/empty ⇒ a degraded empty result. Nothing here
    writes to the store."""
    fmt = (body.format or "jsonl").strip().lower()
    valid = {"jsonl", "sft", "dpo", "failure_mining"}
    if fmt not in valid:
        return TrajectoryExportResult(
            ok=False, format=fmt, degraded=True,
            detail=f"unknown format (expected one of {sorted(valid)})",
        )

    try:
        from mast.logging.v2 import trajectory_export as tx
        from mast.logging.v2.repos import build_repos
        from mast.logging.v2.storage import open_store

        store = open_store()
        repos = build_repos(store)
    except Exception as exc:
        logger.warning("trajectory export: cannot open v2 store: %s", exc)
        return TrajectoryExportResult(ok=False, format=fmt, degraded=True, detail=str(exc))

    try:
        if fmt == "jsonl":
            it = tx.iter_full(repos, limit=body.limit, thread_id=body.thread_id)
        elif fmt == "sft":
            it = tx.to_sft_samples(repos, limit=body.limit)
        elif fmt == "dpo":
            it = tx.to_preference_pairs(repos, limit=body.limit)
        else:  # failure_mining
            it = tx.failure_view(repos, limit=body.limit)

        rows: list[dict] = []
        count = 0
        cap = body.max_inline if body.inline else 0
        truncated = False
        for rec in it:
            count += 1
            if body.inline:
                if len(rows) < cap:
                    rows.append(rec)
                else:
                    truncated = True
        return TrajectoryExportResult(
            ok=True, format=fmt, count=count, rows=rows,
            truncated=truncated, degraded=False,
        )
    except Exception as exc:
        logger.warning("trajectory export (%s) failed: %s", fmt, exc)
        return TrajectoryExportResult(ok=False, format=fmt, degraded=True, detail=str(exc))


# ── POST /api/experiments/export ───────────────────────────────────────


@router.post("/experiments/export", response_model=ExperimentsExportResult)
def export_experiments(
    body: ExperimentsExportRequest, request: Request
) -> ExperimentsExportResult:
    """One-click full-history ZIP archive (``logging.export_all``).

    Bundles all experiment records / checkpoints / artifacts into a timestamped
    zip and returns the manifest roll-up. This is pure blocking I/O the live app
    runs on its offload worker; the relay degrades safely if the core / paths are
    unavailable. The privacy/heavy-filter policy stays entirely in the core."""
    try:
        from datetime import datetime

        from mast._runtime_paths import project_root
        from mast.logging.export_all import export_all_history

        if body.dest:
            dest = Path(body.dest)
        else:
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            dest = project_root() / "exports" / f"mast_full_export_{stamp}.zip"

        manifest = export_all_history(dest, include_heavy=body.include_heavy)
    except Exception as exc:
        logger.warning("experiments export failed: %s", exc)
        return ExperimentsExportResult(ok=False, degraded=True, detail=str(exc))

    m = manifest or {}
    return ExperimentsExportResult(
        ok=True,
        dest=m.get("dest"),
        file_count=int(m.get("file_count", len(m.get("files") or [])) or 0),
        total_bytes=int(m.get("total_bytes", 0) or 0),
        database_count=len(m.get("databases") or []),
        skipped_count=len(m.get("skipped") or []),
        error_count=len(m.get("errors") or []),
        include_heavy=bool(m.get("include_heavy", body.include_heavy)),
        created_at=m.get("created_at"),
        degraded=False,
    )


# ── GET /api/scans/latest ──────────────────────────────────────────────


@router.get("/scans/latest", response_model=LatestScansResponse)
def latest_scans(
    request: Request,
    n: int = Query(default=12, ge=1, le=200),
    dir: str | None = Query(default=None),
    ext: str | None = Query(
        default=None,
        description="comma-separated extensions to keep, e.g. '.sxm' or 'sxm,dat'. "
                    "Unrecognised entries are ignored; an all-unrecognised filter "
                    "returns nothing rather than silently returning everything.",
    ),
    offset: int = Query(
        default=0, ge=0,
        description="how many COLLAPSED entries to skip; with n it pages the whole result "
                    "instead of only ever showing the newest window.",
    ),
) -> LatestScansResponse:
    """Most-recent .sxm/.dat/.3ds files across the data search dirs.

    Relays ``webui.scan_preview._collect_scans`` (mtime-desc, deduped). Degrades
    to an empty list when no dirs exist / nothing is found — never a 500.

    ``ext`` filters BEFORE the ``n``-slice. That ordering is the whole point: a
    caller that takes the newest ``n`` of everything and filters afterwards gets
    however many survive, which on a rig writing spectroscopy is zero .sxm while
    the disk is full of them (, and its 数据面板 twin #60). The same
    trap is documented on ``_collect_scans``.

    The discovery glob is ext-independent work, so the response also carries
    ``counts_by_ext`` over the FULL result — that is what lets a client label its
    filter chips truthfully instead of counting the truncated page.

    Byte-identical COPIES (the automatic ingest into the experiment folder keeps
    the original in place, so one measurement exists several times) are folded
    into one entry carrying ``copies``/``locations``. The fold is visible, never
    silent: every member path ships with the entry. ``total_matched`` still counts
    raw files, ``total_collapsed`` counts entries — the difference is exactly how
    many duplicates the operator is no longer looking at."""
    ctx = request.app.state.ctx
    search_dirs = _scan_search_dirs(ctx, dir)
    wanted = _parse_ext_filter(ext)
    try:
        from mast.webui.scan_preview import (
            _experiment_root_key,
            classify_scan_path,
            collapse_scan_groups,
            collect_scan_stats,
        )

        # No ``exts=`` here on purpose: the counts have to describe every kind
        # present, and narrowing the SEARCH would hide the ones being filtered
        # out — which is the chip reading 0 for a type that exists. The glob
        # cost is the same either way; the ext-narrowing that collect_scan_stats
        # offers only helps a caller that does not need the other counts.
        all_stats = collect_scan_stats(*search_dirs)[:_DISCOVERY_CAP]
        # Resolve the experiment root ONCE for the whole request rather than per
        # file: it is the same answer every time and it touches config.
        exp_root_key = _experiment_root_key()
        all_groups = collapse_scan_groups(all_stats, exp_root_key)
    except Exception as exc:
        logger.warning("latest scans discovery failed: %s", exc)
        return LatestScansResponse(search_dirs=search_dirs, degraded=True)

    counts: dict[str, int] = {}
    for s in all_stats:
        e = s.path.suffix.lower()
        counts[e] = counts.get(e, 0) + 1
    counts_collapsed: dict[str, int] = {}
    for g in all_groups:
        e = g.rep.path.suffix.lower()
        counts_collapsed[e] = counts_collapsed.get(e, 0) + 1

    # Filtering the FULL mtime-desc list and slicing after is the correct order;
    # see the docstring. `wanted` is None when no filter was asked for. Every
    # member of a group shares one basename and therefore one extension, so
    # filtering on the representative decides the whole group.
    matched = [g for g in all_groups if wanted is None or g.rep.path.suffix.lower() in wanted]
    total_matched = sum(len(g.members) for g in matched)
    page = matched[offset : offset + n]

    scans: list[ScanFileEntry] = []
    for g in page:
        pp = g.rep.path
        scans.append(
            ScanFileEntry(
                path=str(pp),
                name=pp.name,
                ext=pp.suffix.lower(),
                mtime=g.rep.mtime_ns / 1e9,
                size_bytes=g.rep.size_bytes,
                kind=classify_scan_path(pp, exp_root_key),
                copies=len(g.members),
                # Single-copy files carry no location list: the one path is
                # already the entry's `path`, and shipping it twice for every
                # card is bytes on the wire for nothing.
                locations=(
                    [
                        ScanFileLocation(
                            path=str(m.path), kind=classify_scan_path(m.path, exp_root_key)
                        )
                        for m in g.members
                    ]
                    if len(g.members) > 1
                    else []
                ),
            )
        )
    return LatestScansResponse(
        scans=scans, count=len(scans), search_dirs=search_dirs, degraded=False,
        counts_by_ext=counts, total_matched=total_matched,
        offset=offset, total_collapsed=len(matched),
        has_more=offset + len(scans) < len(matched),
        counts_by_ext_collapsed=counts_collapsed,
    )


# ── GET /api/scans/preview ─────────────────────────────────────────────


@router.get("/scans/preview", response_model=ScanPreviewResponse)
def scan_preview(
    request: Request,
    path: str = Query(...),
    size: int = Query(default=256, ge=16, le=1024),
    flatten: str | None = Query(
        default=None,
        pattern="^(raw|plane|line|auto)$",
        description="background removal: raw | plane | line | auto. Omitted ⇒ per-extension "
                    "default (.sxm → line). 'auto' measures the frame (~3 s the first time "
                    "per file) and may answer poly2 / masked_line.",
    ),
    channel: str | None = Query(
        default=None, description=".sxm channel to render; default Z → Current → Bias"
    ),
) -> ScanPreviewResponse:
    """Base64-PNG preview of one Nanonis scan/spectroscopy file.

    Relays ``webui.scan_preview.render_scan_thumbnail_v2`` (cached, no pyplot).
    Missing file / unsupported ext / no matplotlib ⇒ degraded with found/rendered
    flags — never a 500.

    The response reports the flatten mode that was ACTUALLY applied, which can
    differ from the request: ``auto`` resolves to a concrete method, and a method
    that raises degrades to ``raw`` with a picture rather than to no picture. The
    client renders that string, so the operator is never told a frame was
    flattened when it was not."""
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        return ScanPreviewResponse(path=path, found=False, ext=ext, degraded=True,
                                   detail="file not found")
    if ext not in _SCAN_EXTS:
        return ScanPreviewResponse(path=path, found=True, ext=ext, degraded=True,
                                   detail=f"unsupported extension {ext}")
    try:
        from mast.webui.scan_preview import render_scan_thumbnail_v2

        res = render_scan_thumbnail_v2(str(p), size=size, flatten=flatten, channel=channel)
    except Exception as exc:
        logger.warning("scan preview render failed (%s): %s", path, exc)
        return ScanPreviewResponse(path=path, found=True, ext=ext, degraded=True,
                                   detail=str(exc))

    uri = res.get("uri")
    if not uri:
        # File present but unrenderable (no channels / no matplotlib stack).
        return ScanPreviewResponse(path=path, found=True, rendered=False, ext=ext,
                                   degraded=True, detail=res.get("detail") or "could not render")
    return ScanPreviewResponse(
        path=path, found=True, rendered=True, ext=ext, image=uri, degraded=False,
        flatten=res.get("flatten"), flatten_why=list(res.get("why") or []),
        channel=res.get("channel"), width_nm=res.get("width_nm"),
        height_nm=res.get("height_nm"), bias_v=res.get("bias_v"),
    )


# ── GET /api/scans/spectrum ────────────────────────────────────────────


#: Extensions whose bytes are a point spectrum rather than an image.
_SPECTRUM_EXTS = (".dat", ".txt", ".csv", ".asc", ".tsv")


@router.get("/scans/spectrum", response_model=SpectrumDataResponse)
def scan_spectrum(
    request: Request,
    path: str = Query(...),
) -> SpectrumDataResponse:
    """Numeric contents of one point spectrum, for an interactive curve.

    Relays ``webui.spectrum_data.extract_spectrum``. The Data tab previously had
    only a thumbnail of the first two columns; this hands over the numbers so the
    client can draw I-V / dI/dV with axes and zoom.

    Degrades rather than 500s, and degrades HONESTLY: a file whose channel roles
    we cannot name still returns its columns (``degraded=False``) rather than
    pretending there is nothing to show."""
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        return SpectrumDataResponse(path=path, found=False, degraded=True,
                                    detail="file not found")
    if ext not in _SPECTRUM_EXTS:
        return SpectrumDataResponse(path=path, found=True, degraded=True,
                                    detail=f"not a spectrum file ({ext})")
    try:
        from mast.webui.spectrum_data import extract_spectrum

        return SpectrumDataResponse(**extract_spectrum(str(p)))
    except Exception as exc:
        logger.warning("spectrum extraction failed (%s): %s", path, exc)
        return SpectrumDataResponse(path=path, found=True, degraded=True, detail=str(exc))


# ── GET /api/scans/attribution ─────────────────────────────────────────


def _resolve_experiment(experiment_id: str) -> tuple[object | None, list[str]]:
    """``(folder | None, ids to query)`` for one experiment.

    **One experiment has two ids.** The folder is named from the v1 row id
    (``…__627c3dab``) while ``file_locations.experiment_id`` holds the v2 ULID
    (``01KYP9DB…``) — and ``GET /api/experiments``, which is where a client gets
    an id to pass here, returns the v1 one. Matching on the id8 suffix alone
    therefore finds nothing for the id the client actually has, and querying the
    table with the client's id returns zero rows for an experiment that has 64 of
    them: an empty screen that looks exactly like "this experiment produced no
    data".

    ``experiment.json`` carries both under ``provenance`` (``v1_row_id`` /
    ``v2_experiment_id``), which is the only place the two are tied together.
    Read it — a shallow scan of the root, a handful of small JSON files — and
    hand back every spelling so the caller can ask for all of them.

    Never raises: with no readable root the caller falls back to the id it was
    given and to ``origin_path`` for paths."""
    ids = [experiment_id]
    try:
        from mast.core.experiment_paths import experiment_root
        from mast.logging.v2.manifest import read_json

        root = experiment_root()
        if not root.is_dir():
            return None, ids
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            meta = read_json(d / "experiment.json") or {}
            prov = meta.get("provenance") or {}
            known = {
                str(meta.get("id") or ""),
                str(prov.get("v1_row_id") or ""),
                str(prov.get("v2_experiment_id") or ""),
            } - {""}
            if experiment_id in known:
                # Query every spelling: rows written at different times may carry
                # either, and asking for all of them cannot return a wrong row —
                # these ids identify the same experiment.
                return d, [experiment_id] + sorted(known - {experiment_id})
    except Exception:  # noqa: BLE001 — resolution is best-effort by design
        return None, ids
    return None, ids


@router.get("/scans/attribution", response_model=ScanAttributionResponse)
def scan_attribution(
    request: Request,
    experiment_id: str = Query(...),
    limit: int = Query(default=2000, ge=1, le=10000),
) -> ScanAttributionResponse:
    """Which files belong to one experiment (and to which sample).

    Reads the v2 ``file_locations`` table — the only mutable record of where a
    given set of bytes currently lives. ``/api/scans/latest`` cannot answer this:
    it walks the disk and a path alone does not say which experiment claimed it.

    Absent store / unknown id ⇒ empty + degraded, never a 500."""
    exp_dir, candidate_ids = _resolve_experiment(experiment_id)
    try:
        from mast.logging.v2.repos import build_repos
        from mast.logging.v2.storage import open_store

        store = open_store()
        repos = build_repos(store)
        rows: list = []
        for eid in candidate_ids:
            rows = repos.file_locations.for_experiment(eid, limit=limit)
            if rows:
                break
    except Exception as exc:
        logger.warning("scan attribution: cannot read file_locations: %s", exc)
        return ScanAttributionResponse(experiment_id=experiment_id, degraded=True,
                                       detail=str(exc))

    files: list[FileAttributionEntry] = []
    for row in rows:
        rel = str(row.get("rel_path") or "")
        abs_path = None
        if exp_dir is not None and rel:
            try:
                abs_path = str(exp_dir / rel)
            except Exception:  # noqa: BLE001 — a weird rel_path loses the path, not the row
                abs_path = None
        files.append(
            FileAttributionEntry(
                sha256=str(row.get("sha256") or ""),
                rel_path=rel,
                abs_path=abs_path,
                origin_path=row.get("origin_path"),
                root_kind=str(row.get("root_kind") or "experiment_folder"),
                sample_id=row.get("sample_id"),
                source=str(row.get("source") or ""),
                size_bytes=int(row.get("size_bytes") or 0),
                status=str(row.get("status") or "ok"),
                ingested_at=row.get("ingested_at"),
            )
        )
    return ScanAttributionResponse(experiment_id=experiment_id, files=files,
                                   count=len(files), degraded=False)


# ── GET /api/experiments/{id}/timeline ─────────────────────────────────


@router.get("/experiments/{experiment_id}/timeline", response_model=ExperimentTimelineResponse)
def experiment_timeline(experiment_id: str, request: Request) -> ExperimentTimelineResponse:
    """Experiment detail TIMELINE incl. per-action TCP calls + state-diff.

    Mirrors ``webui.experiment_viewer._build_action_timeline`` over the v1
    ``ExperimentStorage`` / ``ActionRecord`` (mast.core.types): each action's
    skill call + outcome + duration plus the expandable detail (Nanonis TCP
    calls, before/after instrument state, context). Degrades to an empty-but-not
    -broken timeline when storage is unwired or the id is unknown."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return ExperimentTimelineResponse(id=experiment_id, degraded=True)

    try:
        exp = storage.get_experiment(experiment_id)
    except Exception as exc:
        logger.warning("timeline experiment lookup failed: %s", exc)
        return ExperimentTimelineResponse(id=experiment_id, degraded=True)

    if not exp:
        return ExperimentTimelineResponse(id=experiment_id, found=False, degraded=False)

    try:
        actions = storage.get_actions(experiment_id) or []
    except Exception as exc:
        logger.warning("timeline actions lookup failed: %s", exc)
        actions = []

    entries: list[TimelineEntry] = []
    succeeded = 0
    total_dur = 0.0
    for a in actions:
        result = getattr(a, "result", None)
        success = bool(getattr(result, "success", False)) if result is not None else None
        if success:
            succeeded += 1
        error = (getattr(result, "error", "") or "") if result is not None else ""
        dur = float(getattr(a, "duration_s", 0.0) or 0.0)
        total_dur += dur

        # TCP calls — prefer the action-level list, falling back to the result's.
        raw_calls = getattr(a, "nanonis_calls", None)
        if not raw_calls and result is not None:
            raw_calls = getattr(result, "nanonis_calls", None)
        tcp_calls: list[TimelineTcpCall] = []
        for call in raw_calls or []:
            tcp_calls.append(
                TimelineTcpCall(
                    method=str(getattr(call, "method", "") or ""),
                    args=str(getattr(call, "args", "") or ""),
                    error=(getattr(call, "error", "") or "") or None,
                    elapsed_s=float(getattr(call, "elapsed_s", 0.0) or 0.0),
                )
            )

        before = getattr(a, "state_before", None)
        after = getattr(a, "state_after", None)
        if before is None and result is not None:
            before = getattr(result, "state_before", None)
        if after is None and result is not None:
            after = getattr(result, "state_after", None)

        entries.append(
            TimelineEntry(
                id=str(getattr(a, "id", "")),
                sample_id=getattr(a, "sample_id", None) or None,
                timestamp=getattr(a, "timestamp", None),
                skill_name=getattr(a, "skill_name", "") or "",
                parameters=getattr(a, "parameters", None) or {},
                success=success,
                error=error or None,
                duration_s=dur,
                context=getattr(a, "context", None) or None,
                tcp_calls=tcp_calls,
                state_diff=TimelineStateDiff(
                    before=_state_to_dict(before),
                    after=_state_to_dict(after),
                ),
            )
        )

    return ExperimentTimelineResponse(
        id=experiment_id,
        name=exp.get("name"),
        status=exp.get("status"),
        goal_text=exp.get("goal_text"),
        start_time=exp.get("start_time"),
        end_time=exp.get("end_time"),
        entries=entries,
        action_count=len(entries),
        succeeded=succeeded,
        failed=len(entries) - succeeded,
        total_duration_s=total_dur,
        found=True,
        degraded=False,
    )
