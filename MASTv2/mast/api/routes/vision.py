"""Vision recent / pulse + scan-map + experimental one-shot (FFT / mosaic /
monitor) — domain I/J read + one-shot of the typed API (TS-rewrite Phase 3).

NON-STREAMING only: the WS live streams (frame push, tip-status push) are wired
separately at integration. Everything here is a one-shot read or POST.

GRACEFUL DEGRADATION is the contract. This module must import with NOTHING heavy
present and every handler must boot STANDALONE: it reads optional live
subsystems off ``request.app.state.ctx`` (the BufferService, the live MASTApp,
the experiment storage), and if any is absent OR any call raises, it returns a
valid empty/degraded response with ``degraded=True`` — never a 500.

Heavy core modules (numpy, matplotlib, mast.io.mosaic, mast.webui.exp_capture,
mast.io.exp_map) are LAZY-imported INSIDE the handlers wrapped in try/except,
exactly like routes/skills.py lazy-imports builder_api. We NEVER import gradio.

CRITICAL: no ndarray / tensor ever enters a response — only b64 PNG thumbnails
and scalar metadata.

The live core wiring (sharing the BufferService / app singletons) and the
safety passthrough for the monitor write endpoints happen at integration time;
the API layer contains NO business logic and NO safety checks — it only relays
into the core.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request, Response

from mast.api.schemas_vision import (
    AvoidZoneView,
    CoarseAdviceView,
    CoarseMapResponse,
    CoarseSiteView,
    FFTRequest,
    FFTResponse,
    MapMarkerView,
    MonitorActionResponse,
    MonitorStartRequest,
    MonitorStatus,
    MosaicRequest,
    MosaicResponse,
    MosaicScanMeta,
    NextPositionView,
    RecentFrame,
    RecordCoarseMoveRequest,
    RecordCoarseMoveResponse,
    RelocationSuggestionView,
    ScanFrame,
    ScanImage,
    ScanMapAnalysisResponse,
    ScanMapImportRequest,
    ScanMapImportResponse,
    ScanMapResponse,
    VacuumAttestRequest,
    VacuumInterlockResponse,
    VisionPulseResponse,
    VisionRecentResponse,
    XYZ,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision"])

# Recent-window for the pulse rollup (mirrors gui.vision_pulse._PULSE_WINDOW_S).
_PULSE_WINDOW_S = 300.0
_RECENT_LIMIT = 24


# ─────────────────────────────────────────────────────────────────────
# Small helpers — all defensive, never raise.
# ─────────────────────────────────────────────────────────────────────

def _get_buffer(ctx: Any) -> Any:
    """Best-effort fetch of a live BufferService from the context.

    Standalone dev wires nothing → returns None and the caller degrades. The
    leader may expose it via ``ctx.buffer`` or via a live ``ctx.app._buffer``;
    we try both without importing gradio or constructing anything."""
    if ctx is None:
        return None
    buf = getattr(ctx, "buffer", None)
    if buf is not None:
        return buf
    app = getattr(ctx, "app", None)
    if app is not None:
        # Prefer the lazy GUI helper if present, else the cached attribute.
        try:
            from mast.core.runtime import _ensure_buffer_for_gui  # lazy, may be absent

            b = _ensure_buffer_for_gui(app)
            if b is not None:
                return b
        except Exception as exc:  # noqa: BLE001 — never crash on a missing helper
            logger.debug("vision: _ensure_buffer_for_gui unavailable: %s", exc)
        return getattr(app, "_buffer", None)
    return None


def _get_app(ctx: Any) -> Any:
    """The live MASTApp, if integration wires one; else None (standalone)."""
    if ctx is None:
        return None
    return getattr(ctx, "app", None)


def _resolve_scan_path(app: Any, file_path: str | None) -> Optional[str]:
    """Return a path that EXISTS for render_scan_thumbnail, else best-effort.

    A vision event may carry a path that no longer resolves (the scan moved, or was
    saved under a Nanonis session dir the event didn't capture) — every such frame
    then rendered '未解码缩略图' with no image (2026-06-29). Fall back to a basename
    match across the LIVE scan dirs: the Nanonis session dir (Util_SessionPathGet,
    now captured) + <data_root>/working-sessions. Pure read; never raises."""
    if not file_path:
        return None
    try:
        if Path(file_path).exists():
            return file_path
    except OSError:
        pass
    try:
        name = Path(file_path).name
    except Exception:  # noqa: BLE001
        return file_path
    if not name:
        return file_path
    dirs: list[Path] = []
    try:
        resolver = getattr(app, "_resolve_session_dir", None)
        sess = resolver() if callable(resolver) else None
        if sess:
            dirs.append(Path(sess))
    except Exception:  # noqa: BLE001 — degrade, never break the strip
        pass
    try:
        from mast._runtime_paths import project_root

        dirs.append(project_root() / "working-sessions")
    except Exception:  # noqa: BLE001
        pass
    for d in dirs:
        try:
            cand = d / name
            if cand.exists():
                return str(cand)
        except OSError:
            continue
    return file_path  # unresolved → render_scan_thumbnail returns None (placeholder)


# Cache the (heavy) scan-image assembly on the (path, mtime) set so the 3 s
# scan-map poll doesn't re-read + re-render every .sxm each tick.
_SCAN_IMG_CACHE: dict = {"key": None, "value": []}


def _safe_mtime(p) -> float:
    try:
        import os
        return os.path.getmtime(str(p))
    except OSError:
        return 0.0


def _epoch_boundary_ts(app: Any, exp_id, sample_id) -> float:
    """Unix time of the last lateral coarse move, or 0.0 if there was none.

    Everything the map draws by METRE coordinate needs to know this instant.
    Markers carry ``coord_epoch``; a saved ``.sxm`` carries nothing but its
    header and its mtime, so this is the only handle on "was this imaged before
    or after the stage slid". 0.0 means "no boundary" — nothing is stale."""
    try:
        storage = getattr(app, "_storage", None)
        fn = getattr(storage, "last_coarse_move_timestamp", None)
        raw = fn(exp_id, sample_id) if callable(fn) else None
        if not raw:
            return 0.0
        import datetime as _dt
        return _dt.datetime.fromisoformat(str(raw)).timestamp()
    except Exception as exc:  # noqa: BLE001 — no boundary beats no map
        logger.debug("scan-map: epoch boundary lookup failed: %s", exc)
        return 0.0


def _recent_scan_images(app: Any, limit: int = 12, extra_paths: list | None = None,
                        epoch_boundary_ts: float = 0.0) -> list:
    """Recent saved .sxm scans as ScanImage underlays — a live surface mosaic:
    each scan's real stage footprint (centre/size/angle from its header) + a b64
    thumbnail, so the map shows the ACTUAL surface, not just outlines. ``extra_paths``
    (operator-IMPORTED or marker-referenced .sxm) are ALWAYS included, even outside
    the searched dirs or older than the recent window. Cached on the (path, mtime,
    boundary) set; degrades to [] on any failure. Never raises.

    ``epoch_boundary_ts`` — unix time of the last lateral coarse move. Files older
    than it get ``stale_epoch=True``: their header centre is a coordinate in a
    frame that no longer exists, so painting them at full strength claims we
    imaged surface we have never seen. This underlay was the last piece of the
    map still drawing dead coordinates as if they were live (2026-07-31)."""
    try:
        import os

        dirs: list[str] = []
        resolver = getattr(app, "_resolve_session_dir", None)
        sess = resolver() if callable(resolver) else None
        if sess:
            dirs.append(sess)
        try:
            from mast._runtime_paths import project_root
            dirs.append(str(project_root() / "working-sessions"))
        except Exception:  # noqa: BLE001
            pass
        cfg = getattr(app, "config", None)
        exp_dir = getattr(cfg, "experiments_dir", None)
        if exp_dir:
            dirs.append(str(exp_dir))

        from mast.webui.scan_preview import get_latest_scans, render_scan_thumbnail
        paths: list[str] = []
        if dirs:
            # exts goes to the SEARCH, not to the result — see _recent_scan_paths.
            # Filtering after the slice means the underlay holds however many
            # .sxm happen to be among the newest `limit` files of any kind, which
            # on a session doing spectroscopy is none: the .dat files are all
            # newer. The map then silently loses its surface images and shows
            # bare outlines, with nothing anywhere saying why.
            paths = [str(p) for p in get_latest_scans(limit, *dirs, exts=(".sxm",))]
        # Imported and marker-linked scans are shown regardless of age or folder.
        # Deduplicate resolved paths because markers and directory discovery may
        # refer to the same file with different slash or letter-case spellings.
        def _real(s: str) -> str:
            try:
                return os.path.normcase(os.path.realpath(s))
            except OSError:
                return os.path.normcase(s)

        seen = {_real(p) for p in paths}
        for xp in (extra_paths or []):
            sp = str(xp)
            if not sp.lower().endswith(".sxm") or _real(sp) in seen:
                continue
            try:
                if os.path.exists(sp):
                    paths.append(sp)
                    seen.add(_real(sp))
            except OSError:
                pass
        if not paths:
            return []
        # Paint oldest first so newer frames of the same footprint appear on top.
        # Combine imported and discovered files before sorting; append order and
        # newest-first discovery alone do not give the canvas the required order.
        paths.sort(key=_safe_mtime)
        # The boundary is part of the key: a coarse move changes what these very
        # same files MEAN without touching a single mtime, so a cache keyed only
        # on (path, mtime) would keep serving them as current forever.
        key = (tuple(sorted((str(p), _safe_mtime(p)) for p in paths)),
               float(epoch_boundary_ts or 0.0))
        if _SCAN_IMG_CACHE.get("key") == key:
            return _SCAN_IMG_CACHE["value"]

        from mast.io.mosaic import parse_xy_meta
        from mast.io.nanonis_files import read_sxm
        out: list[ScanImage] = []
        for p in paths:
            try:
                header = (read_sxm(str(p)) or {}).get("header", {}) or {}
                meta = parse_xy_meta(header)
                if not meta:
                    continue  # no stage footprint in header → can't place it
                uri = render_scan_thumbnail(str(p), size=200)
                b64 = uri.split(",", 1)[1] if uri and "," in uri else ""
                if not b64:
                    continue
                out.append(ScanImage(
                    image_b64=b64, center_x_m=meta["cx"], center_y_m=meta["cy"],
                    width_m=meta["w"], height_m=meta["h"],
                    angle_deg=float(meta.get("angle", 0.0) or 0.0),
                    name=os.path.basename(str(p)), path=str(p),
                    stale_epoch=bool(epoch_boundary_ts
                                     and _safe_mtime(p) < epoch_boundary_ts)))
            except Exception:  # noqa: BLE001 — one bad file must not break the map
                continue
        _SCAN_IMG_CACHE["key"] = key
        _SCAN_IMG_CACHE["value"] = out
        return out
    except Exception as exc:  # noqa: BLE001 — degrade, never 500 the map
        logger.debug("scan images assembly failed: %s", exc)
        return []


#: How fresh a milestone PNG must be to be shown as THE LIVE FRAME. The vision
#: monitor writes one per 1/8 of the scan, so on a slow scan the newest can be
#: minutes old and still current; past this it is stale enough that showing it
#: over the live footprint would be a claim we cannot support.
_LIVE_FRAME_MAX_AGE_S = 600.0


def _live_scan_image(app: Any, frame_view) -> "ScanImage | None":
    """The scan IN PROGRESS, as an underlay — or None.

    「扫描地图中得到的STM图没有实时更新」. The mosaic underlay
    is built from saved ``.sxm`` files, and Nanonis writes the .sxm when the scan
    FINISHES. So for the entire duration of a scan — the exact window an operator
    watches the map — the map had nothing new to show.

    The live picture does exist: ``ScanVisionMonitor`` persists a PNG of the real
    partial frame at every 1/8 milestone (``payload.frame_path``). This places
    the newest one at the live frame's footprint, so the map fills in as the scan
    rasters.

    Three conditions, all necessary to keep it honest:

    * the scan must be RUNNING — otherwise the finished .sxm is the truth and
      this would compete with it;
    * the frame footprint must be known — an image with no placement is not a map
      element;
    * the PNG must be recent (mtime). A milestone from a previous scan drawn on
      the current footprint is the falsified history of #76/#78.
    """
    try:
        import os
        import time as _t
        from pathlib import Path

        st = getattr(app, "_state", None)
        snap = st.snapshot() if st is not None else None
        if not getattr(snap, "scan_running", False):
            return None
        if frame_view is None or frame_view.center_x_m is None \
                or frame_view.width_m is None:
            return None
        buf = getattr(app, "_buffer", None)
        if buf is None:
            return None
        newest: Path | None = None
        for ev in reversed(list(buf.get_event_history(since_seqno=-1,
                                                      limit=200) or [])):
            fp = (dict(getattr(ev, "payload", {}) or {})).get("frame_path")
            if not fp:
                continue
            p = Path(str(fp))
            if p.is_file():
                newest = p
                break
        if newest is None:
            return None
        if (_t.time() - _safe_mtime(newest)) > _LIVE_FRAME_MAX_AGE_S:
            return None
        import base64
        return ScanImage(
            image_b64=base64.b64encode(newest.read_bytes()).decode("ascii"),
            center_x_m=frame_view.center_x_m,
            center_y_m=frame_view.center_y_m,
            width_m=frame_view.width_m,
            height_m=frame_view.height_m if frame_view.height_m is not None
            else frame_view.width_m,
            angle_deg=float(frame_view.angle_deg or 0.0),
            name=os.path.basename(str(newest)), path=str(newest))
    except Exception as exc:  # noqa: BLE001 — an underlay must never 500 the map
        logger.debug("live scan image assembly failed: %s", exc)
        return None


def _event_t_wall(ev: Any) -> float:
    """Approx wall-clock seconds for a VisionEvent (mirrors gui _event_to_dict)."""
    ns = int(getattr(ev, "t_mono_ns", 0) or 0)
    if not ns:
        return 0.0
    delta_s = (time.monotonic_ns() - ns) / 1e9
    return time.time() - delta_s


def _enum_value(v: Any, default: str = "") -> str:
    return str(getattr(v, "value", v) or default)


def _scan_search_dirs(app: Any) -> list[str]:
    """Live dirs where saved .sxm scans land: the Nanonis session dir + the data
    root's working-sessions + experiments_dir. Shared by the scan-map underlay
    and the 近期帧 thumbnail fallback."""
    dirs: list[str] = []
    try:
        resolver = getattr(app, "_resolve_session_dir", None)
        sess = resolver() if callable(resolver) else None
        if sess:
            dirs.append(str(sess))
    except Exception:  # noqa: BLE001
        pass
    try:
        from mast._runtime_paths import project_root

        dirs.append(str(project_root() / "working-sessions"))
    except Exception:  # noqa: BLE001
        pass
    cfg = getattr(app, "config", None)
    exp_dir = getattr(cfg, "experiments_dir", None)
    if exp_dir:
        dirs.append(str(exp_dir))
    return dirs


def _recent_scan_paths(app: Any, limit: int) -> list[str]:
    """Newest-first .sxm files for events without an explicit file path.

    Apply the extension filter during discovery, before limiting the result.
    Filtering after the limit can produce an empty pool when newer telemetry
    or spectroscopy files occupy the window.
    """
    try:
        from mast.webui.scan_preview import get_latest_scans

        dirs = _scan_search_dirs(app)
        if not dirs:
            return []
        return [str(p) for p in get_latest_scans(limit, *dirs, exts=(".sxm",))]
    except Exception:  # noqa: BLE001
        return []


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/recent
# ─────────────────────────────────────────────────────────────────────

def _segment_of(cause_ref) -> Optional[int]:
    """``"current_monitor#42"`` → ``42``. Anything else → ``None``.

    Only the current-monitor prefix is accepted. A bare ``"current_monitor"``
    (an alert that had no segment) must NOT resolve to some other event's
    picture, and neither must ``"scan#42"`` — the id spaces are unrelated and a
    numeric collision would show a current waveform under a scan event.
    """
    s = str(cause_ref or "")
    prefix = "current_monitor#"
    if not s.startswith(prefix):
        return None
    try:
        return int(s[len(prefix):])
    except ValueError:
        return None


def _evidence_alert_by_segment(limit: int) -> dict[int, int]:
    """``{segment_id: alert_id}`` for recent alerts that HAVE an evidence PNG.

    Built once per request, not per event: this endpoint polls every 5 s, and a
    query per image-less event would be two dozen round trips a tick.

    Empty for every ordinary reason — monitoring not installed, no alerts, or
    only WARNs (which deliberately render nothing). The caller then falls
    through to the honest no-image placeholder.

    自 2026-08-15 起 ``alerts_query`` 查询失败会**抛** ``StoreQueryFailed``
    (在此之前它自吞异常回 ``([], 0)``)。这里照旧吞成 ``{}`` —— 这个函数产出的
    是**缩略图索引**，空的后果是「这条事件没有配图」，而下游本来就为此准备了
    一句诚实的占位。它不参与任何判决,所以「读不到 ⇒ 没有图」在这一处是对的
    方向;同一个空值在 conduct 闸门那里就不是了(见 ``conduct/adapters.py``)。
    """
    try:
        from mast.monitoring.store import get_store

        rows, _ = get_store().alerts_query(limit=int(limit))
    except Exception:  # noqa: BLE001 — monitoring is an optional install
        return {}
    out: dict[int, int] = {}
    for r in rows:
        seg = r.get("segment_id")
        if seg is None or not r.get("evidence_available"):
            continue
        # alerts_query is ts DESC, so the first hit on a segment is the newest.
        out.setdefault(int(seg), int(r.get("id") or 0))
    return out


def _alert_evidence_bytes(alert_id: int) -> Optional[bytes]:
    try:
        from mast.monitoring.store import get_store

        return get_store().alert_evidence_png(int(alert_id))
    except Exception:  # noqa: BLE001
        return None


#: How big the click-to-enlarge render of a .sxm is, in pixels. The strip's own
#: thumbnail is 160 px — blowing THAT up in the browser is not 「放大」, it is the
#: same 160 px with bigger squares.
_ENLARGED_PX = 900


def _recent_frame_sources(app: Any, events: list, *,
                          limit: int = _RECENT_LIMIT) -> dict[int, tuple[str, str]]:
    """``{seqno: (source_kind, ref)}`` — WHICH picture belongs to each recent event.

    ``source_kind`` is ``"png"`` (a persisted milestone frame), ``"alert"`` (a
    current-monitor evidence PNG) or ``"sxm"`` (a saved scan file, rendered).
    An event with no picture is simply absent.

    Extracted so the 近期标注帧 strip and the click-to-enlarge endpoint below
    can never disagree about which picture a tile is showing. They would have
    disagreed by construction: the ``scan_complete`` fallback hands out the
    recent .sxm files POSITIONALLY (``scan_ptr``), so "which file is event #N's"
    is a property of the whole batch, not of #N. A second, independent
    re-derivation is exactly how you get an enlarge button that opens a
    different frame than the tile — the falsified history of #76/#78 with an
    extra click.

    One deliberate difference from the loop this replaces: a ``frame_path`` that
    ``is_file()`` but then fails to READ used to fall through to the alert/scan
    fallbacks. Here it is chosen and the read fails later, so the tile says it
    has no picture instead of showing a different event's. That is the direction
    #76/#78 already settled; it is written down because it is a change.
    """
    recent_scan_paths = _recent_scan_paths(app, limit)
    scan_ptr = 0
    # One lookup for the whole batch — see _evidence_alert_by_segment.
    evidence_by_segment = _evidence_alert_by_segment(limit * 4)
    out: dict[int, tuple[str, str]] = {}
    for ev in reversed(list(events or [])):  # newest first — the borrow order
        seqno = int(getattr(ev, "seqno", 0) or 0)
        payload = dict(getattr(ev, "payload", {}) or {})
        # FIRST choice: the persisted PNG of the EXACT (partial) frame this pulse
        # analysed (scan_monitor writes one per milestone since 2026-07-10).
        frame_png = payload.get("frame_path")
        if frame_png:
            try:
                if Path(str(frame_png)).is_file():
                    out[seqno] = ("png", str(frame_png))
                    continue
            except OSError:
                pass
        # SECOND: for current-monitor events, the evidence PNG the monitor renders
        # for a CRITICAL — looked up by the segment id in this event's OWN
        # cause_ref, never borrowed .
        seg_id = _segment_of(getattr(ev, "cause_ref", None))
        alert_id = evidence_by_segment.get(seg_id) if seg_id is not None else None
        if alert_id:
            out[seqno] = ("alert", str(alert_id))
            continue
        # THIRD: the saved scan file. In-scan pulses (feature_of_interest) are
        # NEVER given a borrowed image; a COMPLETED scan may take the next-newest
        # real .sxm, which genuinely is its product.
        file_path = payload.get("file_path") or payload.get("path")
        resolved = _resolve_scan_path(app, str(file_path) if file_path else None)
        have_file = False
        try:
            have_file = bool(resolved) and Path(resolved).exists()
        except OSError:
            have_file = False
        if not have_file and _enum_value(getattr(ev, "kind", None)) == "scan_complete":
            if scan_ptr < len(recent_scan_paths):
                resolved = recent_scan_paths[scan_ptr]
                have_file = True
                scan_ptr += 1
        if have_file and resolved:
            out[seqno] = ("sxm", str(resolved))
    return out


def _frame_png_bytes(source: tuple[str, str], *, px: int) -> Optional[bytes]:
    """Render one ``_recent_frame_sources`` entry to PNG bytes, or None.

    ``px`` only reaches the ``sxm`` branch: the other two are stored PNGs and are
    handed back at whatever size they were written."""
    kind, ref = source
    if kind == "png":
        try:
            return Path(ref).read_bytes()
        except OSError:
            return None
    if kind == "alert":
        try:
            return _alert_evidence_bytes(int(ref))
        except (TypeError, ValueError):
            return None
    if kind == "sxm":
        try:
            from mast.webui.scan_preview import render_scan_thumbnail

            uri = render_scan_thumbnail(ref, size=px)
            if uri and "," in uri:
                return base64.b64decode(uri.split(",", 1)[1])
        except Exception:  # noqa: BLE001 — degrade, never break the strip
            return None
    return None


@router.get("/vision/recent", response_model=VisionRecentResponse)
def get_vision_recent(request: Request) -> VisionRecentResponse:
    """Recent annotated vision frames/events (newest first).

    Each frame mirrors a BufferService VisionEvent: scalar metadata + an
    OPTIONAL b64 thumbnail (only the file path is surfaced here; raw .sxm decode
    is out of budget for the strip). Never raw tensors."""
    ctx = request.app.state.ctx
    buf = _get_buffer(ctx)
    if buf is None:
        return VisionRecentResponse(degraded=True)
    app = _get_app(ctx)  # for scan-path resolution (session dir + working-sessions)

    try:
        events = buf.get_event_history(since_seqno=-1, limit=_RECENT_LIMIT)
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("vision/recent get_event_history failed: %s", exc)
        return VisionRecentResponse(degraded=True)

    frames: list[RecentFrame] = []
    try:
        # WHICH picture each event gets is decided once, in _recent_frame_sources
        # — see there for the three fallbacks and why the choice cannot be made
        # twice independently. Here we only RENDER the choice, small (160 px);
        # /vision/recent-frame/{seqno} renders the same choice large.
        sources = _recent_frame_sources(app, list(events or []))
        for ev in reversed(list(events or [])):  # newest first
            payload = dict(getattr(ev, "payload", {}) or {})
            file_path = payload.get("file_path") or payload.get("path")
            kind = _enum_value(getattr(ev, "kind", None))
            sev = _enum_value(getattr(ev, "severity", None), "info")
            seqno = int(getattr(ev, "seqno", 0) or 0)
            source = sources.get(seqno)
            image_b64 = None
            resolved = None
            if source is not None:
                # render_scan_thumbnail is cached by path+mtime+size, so the 5 s
                # poll is a cache hit after the first load. The frontend re-adds
                # the data-URI prefix, so hand over raw base64.
                png = _frame_png_bytes(source, px=160)
                if png:
                    image_b64 = base64.b64encode(png).decode("ascii")
                # The alert branch has no file to name; the other two do.
                if source[0] in ("png", "sxm"):
                    resolved = source[1]
            frames.append(
                RecentFrame(
                    seqno=seqno,
                    kind=kind,
                    severity=sev,
                    cause_ref=getattr(ev, "cause_ref", None),
                    t_wall=_event_t_wall(ev),
                    file_path=resolved or (str(file_path) if file_path else None),
                    image_b64=image_b64,
                    # scan_monitor writes the milestone narration to 'summary_zh';
                    # reading only 'summary'/'detail' left every recent-frame
                    # narration blank in the TS UI.
                    summary=str(payload.get("summary_zh")
                                or payload.get("summary")
                                or payload.get("detail") or ""),
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision/recent shaping failed: %s", exc)
        return VisionRecentResponse(degraded=True)

    return VisionRecentResponse(frames=frames, count=len(frames), degraded=False)


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/recent-frame/{seqno}  — the tile's picture, full size
# ─────────────────────────────────────────────────────────────────────


@router.get("/vision/recent-frame/{seqno}", response_class=Response,
            responses={200: {"content": {"image/png": {}}}, 404: {}})
def get_vision_recent_frame(seqno: int, request: Request) -> Response:
    """The picture of 近期标注帧 tile ``seqno``, big enough to look at, or 404.

    -1 (DB #93):「近期标注帧不能点开放大？」— they could not. The
    grid rendered a bare ``<img>`` with no handler anywhere, on both the 视觉 page
    and the 对话 page. The current-monitor tiles were the exception: those already
    linked through to the full waveform, which is why only the picture tiles were
    dead ends.

    A separate endpoint rather than a bigger base64 in the list: the strip polls
    every 5 s with up to 24 tiles, and a 900 px render each would be megabytes a
    tick for pictures nobody opened. This is fetched on click.

    The picture is the one the TILE shows — same ``_recent_frame_sources``, so an
    enlarge can never open a different frame than the thumbnail it came from.
    ``no-store``, deliberately: for a ``scan_complete`` with no path of its own,
    which .sxm it borrows is POSITIONAL, so the answer for a given seqno legally
    changes as newer scans land. A cached response would freeze one moment's
    answer under a URL that means something else tomorrow.
    """
    ctx = request.app.state.ctx
    buf = _get_buffer(ctx)
    if buf is None:
        return Response(status_code=404)
    try:
        events = list(buf.get_event_history(since_seqno=-1,
                                            limit=_RECENT_LIMIT) or [])
        source = _recent_frame_sources(_get_app(ctx), events).get(int(seqno))
        if source is None:
            return Response(status_code=404)
        png = _frame_png_bytes(source, px=_ENLARGED_PX)
        if not png:
            return Response(status_code=404)
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("vision recent-frame %s failed: %s", seqno, exc)
        return Response(status_code=404)


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/pulse
# ─────────────────────────────────────────────────────────────────────

def _derive_alert_level(critical: int, dropped: int, recent: int) -> str:
    if critical > 0:
        return "critical"
    if dropped > 0:
        return "warn"
    if recent > 0:
        return "info"
    return "idle"


def _derive_trend(events: list[Any], window_s: float) -> str:
    """rising / steady / falling / idle from the first-half vs second-half
    event rate over the recent window (mirrors the pulse sparkline intent)."""
    if not events:
        return "idle"
    now = time.time()
    half = window_s / 2.0
    older = newer = 0
    for ev in events:
        age = now - _event_t_wall(ev)
        if age < 0 or age > window_s:
            continue
        if age <= half:
            newer += 1
        else:
            older += 1
    if newer == 0 and older == 0:
        return "idle"
    if newer > older:
        return "rising"
    if newer < older:
        return "falling"
    return "steady"


def _safe_mode_flag() -> bool:
    """Whether tip verdicts are currently being overridden by SAFE mode."""
    try:
        from mast.core.operating_mode import safe_mode_active
        return safe_mode_active()
    except Exception:  # noqa: BLE001 — a rollup endpoint never fails on this
        return False


@router.get("/vision/pulse", response_model=VisionPulseResponse)
def get_vision_pulse(request: Request) -> VisionPulseResponse:
    """Concise rollup of recent vision activity: dino_score (latest tip-quality
    confidence) + alert_level + trend + counters. Degrades to an idle pulse when
    no BufferService is wired."""
    ctx = request.app.state.ctx
    buf = _get_buffer(ctx)
    if buf is None:
        return VisionPulseResponse(degraded=True)

    try:
        raw = buf.get_event_history(since_seqno=-1, limit=200)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision/pulse get_event_history failed: %s", exc)
        return VisionPulseResponse(degraded=True)

    try:
        stats = dict(buf.get_stats() or {})
    except Exception:  # noqa: BLE001 — stats are best-effort
        stats = {}

    dino_score: Optional[float] = None
    tip_quality: Optional[str] = None
    try:
        ts, _seqno = buf.get_latest_tip_status()
        if ts is not None:
            conf = getattr(ts, "confidence", None)
            dino_score = float(conf) if conf is not None else None
            tip_quality = _enum_value(getattr(ts, "quality", None)) or None
    except Exception as exc:  # noqa: BLE001 — tip status is optional
        logger.debug("vision/pulse latest tip status unavailable: %s", exc)

    now = time.time()
    recent = [ev for ev in (raw or []) if (now - _event_t_wall(ev)) <= _PULSE_WINDOW_S]
    critical = sum(1 for ev in recent if _enum_value(getattr(ev, "severity", None)).lower() == "critical")
    dropped = int(stats.get("events_dropped_oldest", 0) or 0)
    published = int(stats.get("events_published", 0) or 0)
    rate = (len(recent) / _PULSE_WINDOW_S) if recent else 0.0

    return VisionPulseResponse(
        dino_score=dino_score,
        tip_quality=tip_quality,
        alert_level=_derive_alert_level(critical, dropped, len(recent)),  # type: ignore[arg-type]
        trend=_derive_trend(recent, _PULSE_WINDOW_S),  # type: ignore[arg-type]
        recent_count=len(recent),
        critical_count=critical,
        dropped_oldest=dropped,
        events_published=published,
        rate_per_s=rate,
        degraded=False,
        # In SAFE the tip verdicts above are overridden to "good" at their
        # producer. Tell the UI so a human is never shown a manufactured "good"
        # without knowing it is one — the agent's view is uniform, the
        # operator's stays honest.
        safe_mode=_safe_mode_flag(),
    )


# ─────────────────────────────────────────────────────────────────────
# GET /api/scan-map
# ─────────────────────────────────────────────────────────────────────

def _fig_to_b64_png(fig: Any) -> Optional[str]:
    """Render a matplotlib Figure to a base64 PNG string, or None on failure.

    Never lets a render error escape — the structured map data is the contract;
    the PNG is a best-effort thumbnail."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan-map figure->png failed: %s", exc)
        return None


def _current_epoch_or_derived(app: Any, exp_id, sample_id, rows: list) -> int:
    """The live coordinate generation — authoritative query, derived as fallback.

    ``storage.current_epoch()`` COUNTs ``kind='coarse_move'`` over the WHOLE scope
    through ``idx_map_markers_scope``. Counting them inside ``rows`` instead looks
    equivalent and is not: ``get_markers()`` returns at most ``limit`` (2000)
    newest rows, so a long-lived sample under-reports its generation, and the
    frontend then compares a marker's true ``coord_epoch`` against a too-small
    ``current_epoch`` and paints stale coordinates at full opacity (2026-07-31).

    Falls back to the derived count only when the query is unavailable — a
    slightly-wrong number beats a 500 on the map poll."""
    try:
        storage = getattr(app, "_storage", None)
        fn = getattr(storage, "current_epoch", None)
        if callable(fn):
            return int(fn(exp_id, sample_id))
    except Exception as exc:  # noqa: BLE001 — never break the map for this
        logger.debug("scan-map: current_epoch query failed, deriving: %s", exc)
    return sum(1 for r in (rows or []) if (r or {}).get("kind") == "coarse_move")


def _plan_overlay_title() -> str:
    """Title of the published route, or "" — the one thing a client cannot
    derive from the ordered ``planned`` markers themselves."""
    try:
        from mast.io.plan_overlay import get_plan_overlay

        return str(getattr(get_plan_overlay(), "title", "") or "")
    except Exception as exc:  # noqa: BLE001 — a missing title is not a broken map
        logger.debug("scan-map: plan title unavailable: %s", exc)
        return ""


def _piezo_half_range(app: Any) -> Optional[float]:
    """Half-width of the reachable piezo area, metres, or None.

    Comes off the same ``build_map_analysis_config()`` the analysis endpoint and
    the agent tools use, so a whole-range view and the analysis can never
    disagree about where the edge is. Carried on the map poll because the view
    must work with the analysis layers switched off — one float, no work."""
    try:
        cfg = app.build_map_analysis_config()
        v = float(getattr(cfg, "piezo_half_range_m", 0.0) or 0.0)
        return v if v > 0.0 else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan-map: piezo half range unavailable: %s", exc)
        return None


def _planned_marker_views() -> list[MapMarkerView]:
    """The route published by ``show_plan_on_map``, as map markers.

    that tool writes into the process-local ``PlanOverlay``
    singleton and answers "计划路线已显示在扫描地图上". It was not — this endpoint
    never read that singleton, so the plan landed in a store with NO READER while
    the tool reported success. The frontend was ready all along: ScanMapCanvas
    filters ``status === "planned"`` and draws the dashed route; nothing ever
    sent it any.

    Deliberately independent of the live app: a plan is process-local state, so
    it stays visible even when the map degrades for want of a wired core.
    Best-effort — a missing route is a missing route, never a broken map.
    """
    try:
        from mast.io.plan_overlay import get_plan_overlay

        return [
            MapMarkerView(
                kind=getattr(m, "kind", "plan"),
                x_m=getattr(m, "x_m", None),
                y_m=getattr(m, "y_m", None),
                w_m=getattr(m, "w_m", None),
                h_m=getattr(m, "h_m", None),
                angle_deg=float(getattr(m, "angle_deg", 0.0) or 0.0),
                label=getattr(m, "label", "") or "",
                skill_name="",
                status="planned",
                source="plan",
                timestamp="",
            )
            for m in get_plan_overlay().snapshot()
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan-map: plan overlay unavailable: %s", exc)
        return []


@router.get("/scan-map", response_model=ScanMapResponse)
def get_scan_map(request: Request) -> ScanMapResponse:
    """The live experiment map: current scan frame + tip xyz + history markers,
    plus any planned route. Degrades to a plan-only map when no live app is
    wired (the plan is process-local and does not need one)."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return ScanMapResponse(degraded=True, markers=_planned_marker_views(),
                               plan_title=_plan_overlay_title())

    # Lazy-import the pure map layer (numpy lives behind it). Any failure →
    # degraded, never 500.
    try:
        from mast.io.exp_map import (
            frame_marker_from_state,
            markers_from_rows,
            tip_xy_from_state,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scan-map: exp_map import failed: %s", exc)
        return ScanMapResponse(degraded=True, markers=_planned_marker_views(),
                               plan_title=_plan_overlay_title())

    # ── live hardware-state snapshot ──
    state = None
    try:
        st = getattr(app, "_state", None)
        if st is not None:
            state = st.snapshot()
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan-map: state snapshot failed: %s", exc)

    # ── markers from the experiment record ──
    rows: list[dict] = []
    exp_id = None
    sample_id = None
    try:
        el = getattr(app, "_experiment_log", None)
        if el is not None:
            exp_id = getattr(el, "current_experiment_id", None)
            sample_id = getattr(el, "current_sample_id", None)
        storage = getattr(app, "_storage", None)
        if storage is not None and exp_id is not None:
            rows = storage.get_markers(exp_id, sample_id) or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan-map: get_markers failed: %s", exc)

    try:
        markers = markers_from_rows(rows)
        frame_m = frame_marker_from_state(state)
        tip = tip_xy_from_state(state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scan-map: marker assembly failed: %s", exc)
        return ScanMapResponse(degraded=True)

    # ── flat, JSON-safe views ──
    marker_views: list[MapMarkerView] = []
    for m in markers:
        marker_views.append(
            MapMarkerView(
                kind=getattr(m, "kind", "move"),
                x_m=getattr(m, "x_m", None),
                y_m=getattr(m, "y_m", None),
                w_m=getattr(m, "w_m", None),
                h_m=getattr(m, "h_m", None),
                angle_deg=float(getattr(m, "angle_deg", 0.0) or 0.0),
                label=getattr(m, "label", "") or "",
                skill_name=getattr(m, "skill_name", "") or "",
                status=getattr(m, "status", "done") or "done",
                source=getattr(m, "source", "skill") or "skill",
                timestamp=getattr(m, "timestamp", "") or "",
                coord_epoch=getattr(m, "coord_epoch", None),
            )
        )

    # Number of REAL (executed) markers — planned steps are appended below but
    # must not inflate this, it means "operations recorded", not "shapes drawn".
    done_marker_count = len(marker_views)

    # Planned route rides in ``markers`` — see _planned_marker_views .
    marker_views.extend(_planned_marker_views())

    frame_view: Optional[ScanFrame] = None
    if frame_m is not None:
        frame_view = ScanFrame(
            center_x_m=getattr(frame_m, "x_m", None),
            center_y_m=getattr(frame_m, "y_m", None),
            width_m=getattr(frame_m, "w_m", None),
            height_m=getattr(frame_m, "h_m", None),
            angle_deg=float(getattr(frame_m, "angle_deg", 0.0) or 0.0),
        )

    z_m = getattr(state, "z_pos_m", None) if state is not None else None
    tip_xyz = XYZ(
        x_m=tip[0] if tip else None,
        y_m=tip[1] if tip else None,
        z_m=float(z_m) if z_m is not None else None,
    )

    # ── b64 PNG: DELIBERATELY NOT RENDERED ────────────────────────────────
    # The field stays in the schema (clients may still read it) but nothing does:
    # the map is drawn client-side by ScanMapCanvas (react-konva) from the
    # structured data above. Grepped the whole frontend — every ``image_b64``
    # consumer reads a DIFFERENT payload (ScanImage per-.sxm thumbnails, RecentFrame,
    # FFT, mosaic); none reads ScanMapResponse.image_b64.
    # Rendering it cost ~110 ms of matplotlib plus ~30 KB of base64 on EVERY poll
    # (every 3 s, per client) and was thrown away every time.
    image_b64: Optional[str] = None

    return ScanMapResponse(
        frame=frame_view,
        tip_xyz=tip_xyz,
        markers=marker_views,
        # planned steps ride in ``markers`` (frontend already filters them) but
        # are excluded here — this counts operations actually performed.
        marker_count=done_marker_count,
        sample_label="",
        image_b64=image_b64,
        # surface-mosaic underlay: recent .sxm in the searched dirs PLUS any scan
        # explicitly imported / linked via a marker's meta.file (shown regardless
        # of dir or age), PLUS the scan currently in progress (#23 — the .sxm
        # only lands when a scan finishes, so the map was frozen for exactly the
        # window the operator watches it). The live frame goes LAST so it paints
        # over the older saved scans it overlaps.
        scan_images=_recent_scan_images(
            app,
            extra_paths=[str((r.get("meta") or {}).get("file"))
                         for r in rows if (r.get("meta") or {}).get("file")],
            epoch_boundary_ts=_epoch_boundary_ts(app, exp_id, sample_id),
        ) + [im for im in (_live_scan_image(app, frame_view),) if im is not None],
        # A SECOND query, deliberately. Deriving this from ``rows`` was wrong:
        # get_markers() returns at most `limit` newest rows, so once a scope
        # outgrows the window the count under-reports the generation — and the
        # frontend then compares a marker's real coord_epoch (say 5) against a
        # low current_epoch (say 2) and renders STALE markers at full opacity.
        # current_epoch() COUNTs the whole scope through idx_map_markers_scope.
        current_epoch=_current_epoch_or_derived(app, exp_id, sample_id, rows),
        plan_title=_plan_overlay_title(),
        piezo_half_range_m=_piezo_half_range(app),
        degraded=False,
    )


# ─────────────────────────────────────────────────────────────────────
# GET /api/scan-map/analysis  — the same conclusions the agent acts on
# ─────────────────────────────────────────────────────────────────────
#
# Registered BEFORE /scan-map/import purely for readability; both are literal
# paths so ordering is not load-bearing here.

#: How far ahead the route is walked for the map. Enough to read as a direction
#: of travel, short enough that one extra pass over the candidate list stays
#: invisible next to the rasterisation this call already does. Not a query
#: parameter: this API family takes none, and the number is a rendering choice,
#: not something a client should be able to turn into an unbounded search.
_UPCOMING_COUNT = 8


@router.get("/scan-map/analysis", response_model=ScanMapAnalysisResponse)
def get_scan_map_analysis(request: Request) -> ScanMapAnalysisResponse:
    """Coverage, keep-out zones, the next scan position and whether to relocate.

    Calls the SAME ``mast.io.map_analysis.analyze_map`` the agent's
    ``get_map_analysis`` tool calls, so the operator's panel and the agent cannot
    reach different conclusions — the whole reason the button exists is to make
    those programmatic decisions inspectable.

    Not on the 3 s map poll: this is button-driven. Degrades rather than 500s."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return ScanMapAnalysisResponse(degraded=True, detail="核心未就绪")
    try:
        from mast.io.map_analysis import analyze_map

        el = getattr(app, "_experiment_log", None)
        exp_id = getattr(el, "current_experiment_id", None) if el else None
        sample_id = getattr(el, "current_sample_id", None) if el else None
        storage = getattr(app, "_storage", None)
        if storage is None:
            return ScanMapAnalysisResponse(degraded=True, detail="实验记录存储不可用")
        rows = storage.get_markers(exp_id, sample_id) or []

        cfg = app.build_map_analysis_config()
        plan_steps = None
        try:
            from mast.io.plan_overlay import get_plan_overlay
            plan_steps = list(get_plan_overlay().snapshot() or [])
        except Exception as exc:  # noqa: BLE001 — overlay is optional
            logger.debug("scan-map analysis: plan overlay unavailable: %s", exc)

        res = analyze_map(
            rows, cfg, plan_markers=plan_steps,
            current_epoch=_current_epoch_or_derived(app, exp_id, sample_id, rows),
            upcoming_count=_UPCOMING_COUNT)
        return ScanMapAnalysisResponse(
            current_epoch=res.current_epoch,
            markers_total=res.markers_total,
            markers_current_epoch=res.markers_current_epoch,
            coverage_pct=round(res.coverage_frac * 100.0, 3),
            usable_pct=round(res.usable_frac * 100.0, 2),
            usable_unscanned_pct=round(res.usable_unscanned_frac * 100.0, 2),
            strategy=res.strategy,
            frame_size_m=cfg.frame_size_m,
            piezo_half_range_m=cfg.piezo_half_range_m,
            avoid_zones=[
                AvoidZoneView(x_m=c.x_m, y_m=c.y_m, radius_m=c.radius_m,
                              kind=c.kind, label=c.label)
                for c in res.avoid_circles
            ],
            damage_counts=res.damage_counts,
            next_position=(
                None if res.next_position is None else NextPositionView(
                    x_m=res.next_position.x_m, y_m=res.next_position.y_m,
                    strategy=res.next_position.strategy,
                    reason=res.next_position.reason,
                    ring_index=res.next_position.ring_index,
                    candidates_left=res.next_position.candidates_left)
            ),
            upcoming=[
                NextPositionView(x_m=p.x_m, y_m=p.y_m, strategy=p.strategy,
                                 reason=p.reason, ring_index=p.ring_index,
                                 candidates_left=p.candidates_left)
                for p in res.upcoming
            ],
            coarse_advice=CoarseAdviceView(
                suggest=res.coarse_advice.suggest,
                reasons=list(res.coarse_advice.reasons)),
            sts_points=[XYZ(x_m=x, y_m=y) for x, y in res.sts_points],
            sts_total=res.sts_total,
            pending_plan_steps=int(res.survey.get("pending_plan_steps", 0)),
            pending_already_scanned=int(res.survey.get("pending_already_scanned", 0)),
            route_truncated=res.route_truncated,
            degraded=False,
        )
    except Exception as exc:  # noqa: BLE001 — analysis never breaks the page
        logger.warning("scan-map analysis failed: %s", exc, exc_info=True)
        return ScanMapAnalysisResponse(degraded=True, detail=str(exc)[:200])


# ─────────────────────────────────────────────────────────────────────
# POST /api/scan-map/coarse-move  — backfill a coarse move done by hand
# ─────────────────────────────────────────────────────────────────────

@router.post("/scan-map/coarse-move", response_model=RecordCoarseMoveResponse)
def post_scan_map_coarse_move(
    request: Request, body: RecordCoarseMoveRequest
) -> RecordCoarseMoveResponse:
    """Record a lateral coarse move the operator made directly in Nanonis.

    Moves through MAST are recorded automatically; this covers the blind spot —
    ``HardwareState`` does not poll the coarse motor, so a move made by hand is
    invisible, and every marker on the map silently keeps a coordinate that no
    longer points at the same surface. Writes the same ``coarse_move`` marker the
    recorder and the agent's ``record_coarse_move`` tool write, so all three
    paths produce one kind of row.

    Starts a new generation FROM NOW and does not rewrite history: there is no
    way to know when during the record the move actually happened, and guessing
    an insertion point would be less honest than dating it from the report."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return RecordCoarseMoveResponse(ok=False, degraded=True,
                                        message="核心未就绪")
    try:
        storage = getattr(app, "_storage", None)
        if storage is None:
            return RecordCoarseMoveResponse(ok=False, degraded=True,
                                            message="实验记录存储不可用")
        el = getattr(app, "_experiment_log", None)
        exp_id = getattr(el, "current_experiment_id", None) if el else None
        sample_id = getattr(el, "current_sample_id", None) if el else None
        direction = (body.direction or "").strip()
        meta = {"manual_backfill": True, "source_ui": True,
                "direction": direction or None,
                "steps": int(body.steps) or None,
                "note": (body.note or "").strip() or None}
        storage.log_marker(
            kind="coarse_move", x_m=None, y_m=None,
            label=f"补记手动粗动{(' ' + direction) if direction else ''}",
            skill_name="", status="done", source="manual",
            experiment_id=exp_id, sample_id=sample_id,
            meta={k: v for k, v in meta.items() if v is not None})
        epoch = storage.current_epoch(exp_id, sample_id)
        # Same invalidation the automatic path does — the plan route and the
        # tip-crash block-list both cache piezo-frame positions that this move
        # just made meaningless.
        from mast.core.coarse_move_effects import on_coarse_move_recorded
        on_coarse_move_recorded(source="api:scan-map/coarse-move")
        return RecordCoarseMoveResponse(
            ok=True, new_coord_epoch=epoch,
            message=(f"已补记手动粗动，扫描地图进入第 {epoch} 代坐标系；"
                     "此前的标记将淡显且不再参与选点分析，"
                     "计划路线与撞针封锁也已清空。"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("scan-map coarse-move backfill failed: %s", exc)
        return RecordCoarseMoveResponse(ok=False, degraded=True,
                                        message=str(exc)[:200])


# ─────────────────────────────────────────────────────────────────────
# POST /api/scan-map/import  — operator imports their OWN scans/spectra
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# GET /api/coarse-map  — the stage-scale map, plus the two gates
# ─────────────────────────────────────────────────────────────────────
#
# Same derivation the agent's `get_coarse_map` tool calls, for the same reason
# the scan-map analysis endpoint exists: the operator has to be able to inspect
# what the agent is acting on, and a second implementation would defeat that.
#
# The vacuum verdict and the drive declaration ride along because they are what
# turns a suggestion into something that can actually be executed — a map with a
# perfectly good destination on it is useless information while the interlock is
# refusing, and the operator needs to see WHY in the same place.

@router.get("/coarse-map", response_model=CoarseMapResponse)
def get_coarse_map(request: Request) -> CoarseMapResponse:
    """Sites visited on the sample (in coarse-motor steps), and where to go next."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return CoarseMapResponse(degraded=True, detail="核心未就绪")
    try:
        from mast.core import coarse_drive, vacuum_interlock
        from mast.io.coarse_map import CoarseMapConfig, build_coarse_map

        storage = getattr(app, "_storage", None)
        if storage is None:
            return CoarseMapResponse(degraded=True, detail="实验记录存储不可用")
        el = getattr(app, "_experiment_log", None)
        exp_id = getattr(el, "current_experiment_id", None) if el else None
        sample_id = getattr(el, "current_sample_id", None) if el else None
        rows = storage.get_markers(exp_id, sample_id) or []

        try:
            cfg = app.build_coarse_map_config()
        except Exception:  # noqa: BLE001
            cfg = CoarseMapConfig()
        cmap = build_coarse_map(rows, cfg)
        raw = cmap.as_dict()

        verdict = vacuum_interlock.check()
        half = 1.5e-6
        try:
            limits = getattr(app.config, "safety", None)
            if limits is not None and getattr(limits, "xy_max_m", None):
                half = abs(float(limits.xy_max_m))
        except Exception:  # noqa: BLE001
            pass

        return CoarseMapResponse(
            sites=[CoarseSiteView(**s) for s in raw["sites"]],
            current_index=raw["current_index"],
            position_known=raw["position_known"],
            suggestion=(RelocationSuggestionView(**raw["suggestion"])
                        if raw.get("suggestion") else None),
            note=raw.get("note", ""),
            budget_used_steps=raw.get("budget_used_steps", {}),
            axis_step_budget=raw.get("axis_step_budget", 0),
            site_spacing_steps=raw.get("site_spacing_steps", 0),
            piezo_half_range_m=half,
            step_m=cfg.step_m,
            vacuum_allow=verdict.allow,
            vacuum_reason=verdict.reason,
            coarse_drive_declared=coarse_drive.is_declared(),
            coarse_drive_note=coarse_drive.format_block(),
            degraded=False,
        )
    except Exception as exc:  # noqa: BLE001 — a map must never 500 the page
        logger.warning("coarse map failed: %s", exc, exc_info=True)
        return CoarseMapResponse(degraded=True, detail=str(exc)[:200])


@router.get("/coarse-map/selfcheck")
def get_coarse_selfcheck(request: Request, probe_step_counter: bool = True) -> dict:
    """Read-only survey of everything coarse motion depends on. Commissioning aid.

    A dedicated endpoint rather than "ask the agent to run the skill", for two
    reasons that both matter over a slow remote link during commissioning:
    it does not depend on the chat/agent loop being healthy, and it is one
    request whose whole output can be saved verbatim.

    STRICTLY READ-ONLY — every call it makes is a getter MAST already issues in
    ordinary operation. Safe with the tip engaged, mid-scan, at any pressure.

    Builds its own one-shot ExecutionContext the same way ``signals.py`` does;
    with the singletons unwired it degrades rather than touching hardware from a
    half-built process."""
    ctx = request.app.state.ctx
    try:
        from mast.api.routes.signals import _execution_context

        exec_ctx = _execution_context(ctx)
        if exec_ctx is None:
            return {"degraded": True,
                    "detail": "内核未就绪(pool/state/registry 未接线)",
                    "ready": False}
        from mast.skills.builtins.coarse_selfcheck import CoarseMotionSelfCheck

        res = CoarseMotionSelfCheck().execute(
            exec_ctx, {"probe_step_counter": bool(probe_step_counter)})
        out = dict(res.data or {})
        out["degraded"] = False
        return out
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not 500
        logger.warning("coarse selfcheck failed: %s", exc, exc_info=True)
        return {"degraded": True, "detail": str(exc)[:300], "ready": False}


@router.get("/coarse-map/vacuum", response_model=VacuumInterlockResponse)
def get_vacuum_interlock(request: Request) -> VacuumInterlockResponse:
    """The coarse-motion vacuum verdict on its own, for a status strip."""
    try:
        from mast.core import vacuum_interlock

        v = vacuum_interlock.check()
        att = vacuum_interlock.get_attestation()
        return VacuumInterlockResponse(
            allow=v.allow, reason=v.reason, source=v.source,
            pressure_pa=v.pressure_pa, age_s=v.age_s, mode=v.mode,
            over_range=v.over_range, attested=v.attested,
            attestation_remaining_h=(
                round(att.remaining_s() / 3600.0, 2)
                if att is not None and not att.expired() else None),
            degraded=False)
    except Exception as exc:  # noqa: BLE001
        # Degrade to "not allowed": a status strip that cannot compute the
        # verdict must not render a green light.
        return VacuumInterlockResponse(
            allow=False, degraded=True, detail=str(exc)[:200],
            reason="真空互锁状态不可用 —— 按拒绝处理")


@router.post("/coarse-map/vacuum/attest", response_model=VacuumInterlockResponse)
def post_vacuum_attest(request: Request,
                       body: VacuumAttestRequest) -> VacuumInterlockResponse:
    """Sign that the pressure is safe, when the gauge cannot say so.

    Not exposed to the agent, and there is no agent tool for it: attesting to a
    physical condition MAST cannot observe is the operator's act by definition —
    a model signing it would just be laundering the same ignorance into a
    permission."""
    try:
        from mast.core import vacuum_interlock

        vacuum_interlock.attest(
            body.reason, signed_by=body.signed_by,
            ttl_s=max(0.1, float(body.ttl_hours)) * 3600.0, note=body.note)
        return get_vacuum_interlock(request)
    except ValueError as exc:
        return VacuumInterlockResponse(allow=False, degraded=True,
                                       detail=str(exc)[:200],
                                       reason="签署原因无效")
    except Exception as exc:  # noqa: BLE001
        return VacuumInterlockResponse(allow=False, degraded=True,
                                       detail=str(exc)[:200],
                                       reason="签署失败")


@router.delete("/coarse-map/vacuum/attest", response_model=VacuumInterlockResponse)
def delete_vacuum_attest(request: Request) -> VacuumInterlockResponse:
    """Revoke the signature (e.g. after starting a pump-down)."""
    try:
        from mast.core import vacuum_interlock

        vacuum_interlock.revoke_attestation()
        return get_vacuum_interlock(request)
    except Exception as exc:  # noqa: BLE001
        return VacuumInterlockResponse(allow=False, degraded=True,
                                       detail=str(exc)[:200], reason="撤销失败")


@router.post("/scan-map/import", response_model=ScanMapImportResponse)
def post_scan_map_import(request: Request, body: ScanMapImportRequest) -> ScanMapImportResponse:
    """Import the operator's own scans/spectra onto the map by path (a file or a
    folder). .sxm → a scan footprint placed by its header stage-xy (+ thumbnail);
    .dat/.3ds → a spectrum marker at its header xy. Scoped to the active
    experiment/sample. Degrades when no live core/storage is wired; asks the
    operator to start an experiment first when none is active."""
    import os

    ctx = request.app.state.ctx
    app = _get_app(ctx)
    storage = getattr(app, "_storage", None) if app else None
    if storage is None:
        return ScanMapImportResponse(ok=False, degraded=True,
                                     message="导入不可用（无实时核心 / 存储）")
    el = getattr(app, "_experiment_log", None)
    exp_id = getattr(el, "current_experiment_id", None) if el else None
    sample_id = getattr(el, "current_sample_id", None) if el else None
    if exp_id is None:
        return ScanMapImportResponse(
            ok=False, degraded=False,
            message="请先开始一个实验——导入的记录要归属到当前实验/样品。")

    p = (body.path or "").strip().strip('"')
    if not p or not os.path.exists(p):
        return ScanMapImportResponse(ok=False, message=f"路径不存在：{p}")

    exts = (".sxm", ".dat", ".3ds")
    files: list[str] = []
    if os.path.isdir(p):
        if body.recursive:
            for root, _dirs, names in os.walk(p):
                files += [os.path.join(root, n) for n in names if n.lower().endswith(exts)]
        else:
            files += [os.path.join(p, n) for n in os.listdir(p)
                      if n.lower().endswith(exts)]
    elif p.lower().endswith(exts):
        files.append(p)
    if not files:
        return ScanMapImportResponse(ok=False, message="未找到 .sxm/.dat/.3ds 文件。")

    from mast.io.exp_map import manual_marker_from_spectrum_file
    scans = spectra = skipped = 0
    for f in sorted(files):
        try:
            if f.lower().endswith(".sxm"):
                from mast.io.mosaic import parse_xy_meta
                from mast.io.nanonis_files import read_sxm
                header = (read_sxm(f) or {}).get("header", {}) or {}
                meta = parse_xy_meta(header)
                if not meta:
                    skipped += 1
                    continue
                storage.log_marker(
                    kind="scan", x_m=meta["cx"], y_m=meta["cy"], w_m=meta["w"],
                    h_m=meta["h"], angle_deg=float(meta.get("angle", 0.0) or 0.0),
                    label=os.path.basename(f), skill_name="", status="done",
                    source="import", experiment_id=exp_id, sample_id=sample_id,
                    meta={"file": f})
                scans += 1
            else:  # .dat / .3ds spectrum
                m = manual_marker_from_spectrum_file(f)
                if m is None:
                    skipped += 1
                    continue
                storage.log_marker(
                    kind="sts", x_m=m.x_m, y_m=m.y_m, w_m=m.w_m, h_m=m.h_m,
                    angle_deg=m.angle_deg, label=os.path.basename(f), skill_name="",
                    status="done", source="import", experiment_id=exp_id,
                    sample_id=sample_id, meta={"file": f})
                spectra += 1
        except Exception as exc:  # noqa: BLE001 — one bad file must not fail the batch
            logger.debug("scan-map import %s failed: %s", f, exc)
            skipped += 1

    _SCAN_IMG_CACHE["key"] = None  # so imported .sxm thumbnails appear immediately
    n = scans + spectra
    return ScanMapImportResponse(
        ok=n > 0, imported=n, scans=scans, spectra=spectra, skipped=skipped,
        message=(f"已导入 {scans} 张扫描 + {spectra} 条谱到地图"
                 + (f"；{skipped} 个文件无可用坐标已跳过" if skipped else "")),
        degraded=False)


# ─────────────────────────────────────────────────────────────────────
# POST /api/experimental/fft
# ─────────────────────────────────────────────────────────────────────

@router.post("/experimental/fft", response_model=FFTResponse)
def post_fft(request: Request, body: FFTRequest) -> FFTResponse:
    """One-shot one-sided rfft on a supplied trace. Pure compute via the core
    ``compute_fft`` helper — degrades (ok=False, degraded=True) when the helper /
    numpy is unavailable or the trace has too few samples."""
    if not body.samples or len(body.samples) < 4:
        # Faithful to compute_fft's n<4 guard — empty result, not an error.
        return FFTResponse(
            ok=False,
            n_samples=len(body.samples or []),
            window=body.window,
            output=body.output,
            unit=body.unit,
            channel_name=body.channel_name,
            degraded=False,
        )

    try:
        from mast.io.signal_fft import compute_fft  # lazy: pulls numpy
    except Exception as exc:  # noqa: BLE001
        logger.warning("fft: compute_fft import failed: %s", exc)
        return FFTResponse(ok=False, degraded=True, window=body.window,
                           output=body.output, unit=body.unit,
                           channel_name=body.channel_name)

    trace = {
        "samples": list(body.samples),
        "fs_hz": body.fs_hz or 0.0,
        "duration_s": body.duration_s or 0.0,
        "unit": body.unit,
        "channel_name": body.channel_name,
    }
    try:
        fft = compute_fft(trace, window=body.window, detrend=body.detrend,
                          output=body.output)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fft compute failed: %s", exc)
        return FFTResponse(ok=False, degraded=True, window=body.window,
                           output=body.output, unit=body.unit,
                           channel_name=body.channel_name)

    if not fft:  # too few samples / fs unresolvable
        return FFTResponse(ok=False, n_samples=len(body.samples),
                           window=body.window, output=body.output,
                           unit=body.unit, channel_name=body.channel_name)

    return FFTResponse(
        ok=True,
        freqs_hz=[float(f) for f in (fft.get("freqs_hz") or [])],
        spectrum=[float(s) for s in (fft.get("spectrum") or [])],
        n_samples=int(fft.get("n_samples", 0)),
        fs_hz=float(fft.get("fs_hz", 0.0)),
        nyquist_hz=float(fft.get("nyquist_hz", 0.0)),
        df_hz=float(fft.get("df_hz", 0.0)),
        window=str(fft.get("window", body.window)),
        output=str(fft.get("output", body.output)),
        unit=str(fft.get("unit", body.unit)),
        channel_name=str(fft.get("channel_name", body.channel_name)),
        degraded=False,
    )


# ─────────────────────────────────────────────────────────────────────
# POST /api/experimental/mosaic
# ─────────────────────────────────────────────────────────────────────

@router.post("/experimental/mosaic", response_model=MosaicResponse)
def post_mosaic(request: Request, body: MosaicRequest) -> MosaicResponse:
    """Stitch a directory of .sxm scans into one big-canvas overview and return
    a b64 PNG of the canvas. The canvas ndarray is rendered to PNG and dropped —
    it NEVER crosses the wire. Degrades (ok=False, degraded=True) when the mosaic
    core / numpy / matplotlib is unavailable."""
    try:
        from mast.io.mosaic import mosaic_from_dir, render_mosaic_figure  # lazy
    except Exception as exc:  # noqa: BLE001
        logger.warning("mosaic: import failed: %s", exc)
        return MosaicResponse(ok=False, degraded=True)

    try:
        result = mosaic_from_dir(
            body.directory,
            channel=body.channel,
            recursive=body.recursive,
            line_normalize=body.line_normalize,
            cmap=body.cmap,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("mosaic build failed: %s", exc)
        return MosaicResponse(ok=False, degraded=True,
                              error=f"{type(exc).__name__}: {exc}")

    extent = result.get("extent_m")
    canvas_px = result.get("canvas_px")
    scans_meta = [
        MosaicScanMeta(
            path=str(s.get("path", "")),
            cx=s.get("cx"),
            cy=s.get("cy"),
            w=s.get("w"),
            h=s.get("h"),
            channel=str(s.get("channel", "")),
        )
        for s in (result.get("scans_meta") or [])
    ]

    image_b64: Optional[str] = None
    placed = int(result.get("placed", 0) or 0)
    if result.get("image") is not None:
        try:
            fig = render_mosaic_figure(result, title=body.label)
            image_b64 = _fig_to_b64_png(fig)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mosaic render failed: %s", exc)

    return MosaicResponse(
        ok=placed > 0,
        image_b64=image_b64,
        placed=placed,
        n_input=int(result.get("n_input", 0) or 0),
        extent_m=[float(v) for v in extent] if extent else None,
        res_m_per_px=float(result.get("res_m_per_px", 0.0) or 0.0),
        canvas_px=[int(v) for v in canvas_px] if canvas_px else None,
        angle_warning=bool(result.get("angle_warning", False)),
        scans_meta=scans_meta,
        error=str(result.get("error", "") or ""),
        degraded=False,
    )


# ─────────────────────────────────────────────────────────────────────
# POST /api/experimental/monitor/start  &  /stop
#
# WRITE endpoints. Defined for contract completeness; they ONLY relay into the
# live core (which owns the daemon thread + the IC skill calls + safety floor).
# The API layer holds no monitor state and performs no hardware I/O. When no
# live app is wired (standalone) they degrade to {ok: false, degraded: true}.
# ─────────────────────────────────────────────────────────────────────

def _monitor_status_view(st: Optional[dict]) -> MonitorStatus:
    st = st or {}
    return MonitorStatus(
        running=bool(st.get("running", False)),
        channel=str(st.get("channel", "") or ""),
        interval_s=float(st.get("interval_s", 0.0) or 0.0),
        count=int(st.get("count", 0) or 0),
        last_value=st.get("last_value"),
        unit=str(st.get("unit", "") or ""),
        last_t=str(st.get("last_t", "") or ""),
        csv_path=str(st.get("csv_path", "") or ""),
        error=str(st.get("error", "") or ""),
    )


@router.post("/experimental/monitor/start", response_model=MonitorActionResponse)
def post_monitor_start(request: Request, body: MonitorStartRequest) -> MonitorActionResponse:
    """Relay a long-term-monitor START to the live core. The core owns the
    daemon thread, CSV writes, and the IC skill safety floor — the API only
    forwards. Degrades when no live app is wired."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return MonitorActionResponse(ok=False, degraded=True,
                                     message="monitor unavailable (no live core wired)")

    # The live wiring (a thin core entry point that starts the monitor thread)
    # is integrated at integration. Until then we relay defensively: if the core
    # exposes a start hook we call it; otherwise we degrade rather than spin a
    # thread from the API layer (no business logic / hardware here).
    try:
        start_fn = getattr(app, "start_experimental_monitor", None)
        if not callable(start_fn):
            return MonitorActionResponse(
                ok=False, degraded=True,
                status=_monitor_status_view(getattr(app, "_exp_monitor", None)),
                message="monitor start hook not wired",
            )
        st = start_fn(channel=body.channel, interval_s=body.interval_s)
        status = _monitor_status_view(st if isinstance(st, dict)
                                      else getattr(app, "_exp_monitor", None))
        return MonitorActionResponse(ok=status.running, status=status,
                                     message="monitor started", degraded=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("monitor start failed: %s", exc)
        return MonitorActionResponse(ok=False, degraded=True,
                                     message=f"{type(exc).__name__}: {exc}")


@router.post("/experimental/monitor/stop", response_model=MonitorActionResponse)
def post_monitor_stop(request: Request) -> MonitorActionResponse:
    """Relay a long-term-monitor STOP to the live core. Degrades when no live
    app is wired. Idempotent — stopping an absent monitor is a benign no-op."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return MonitorActionResponse(ok=False, degraded=True,
                                     message="monitor unavailable (no live core wired)")

    try:
        stop_fn = getattr(app, "stop_experimental_monitor", None)
        if callable(stop_fn):
            st = stop_fn()
            status = _monitor_status_view(st if isinstance(st, dict)
                                          else getattr(app, "_exp_monitor", None))
            return MonitorActionResponse(ok=True, status=status,
                                         message="monitor stopped", degraded=False)
        # Fallback: signal the existing in-core monitor state directly (the same
        # stop Event the GUI uses). Still no business logic here — just a relay.
        st = getattr(app, "_exp_monitor", None)
        if isinstance(st, dict) and st.get("running"):
            ev = st.get("stop")
            if ev is not None:
                ev.set()
            st["running"] = False
            return MonitorActionResponse(ok=True, status=_monitor_status_view(st),
                                         message="monitor stopped", degraded=False)
        return MonitorActionResponse(
            ok=True, status=_monitor_status_view(st if isinstance(st, dict) else None),
            message="no monitor running", degraded=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("monitor stop failed: %s", exc)
        return MonitorActionResponse(ok=False, degraded=True,
                                     message=f"{type(exc).__name__}: {exc}")
