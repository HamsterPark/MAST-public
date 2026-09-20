"""Scan image preview for the MAST GUI.

Parses Nanonis scan/spectroscopy files (.sxm / .dat / .3ds) plus generic numeric
text (.txt / .csv / .asc / .tsv) and renders them to base64 PNG thumbnails via
matplotlib (Agg backend). Provides helpers to discover the most recent scans
across search directories.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# Extensions the Data tab knows how to render.
_SCAN_EXTS: tuple[str, ...] = (".sxm", ".sm4", ".dat", ".3ds", ".txt", ".csv", ".asc", ".tsv")

#: Immediate parent directories containing telemetry rather than scan images.
#: Environment sidecars receive frequent append writes, so an mtime-sorted scan
#: listing would otherwise be dominated by telemetry. signals and monitors are
#: separate immediate-parent names and require the same exclusion as env.
_NON_SCAN_DIR_NAMES: frozenset[str] = frozenset({"env", "signals", "monitors"})

#: How deep to walk from a search directory.
#:
#: FIVE, not three, and this is the other half of 「看不到 sxm 了」 —
#: the scans were not merely buried, they were never listed at all. From
#: ``experiments_dir`` the MAST-managed layout puts a scan at
#: ``<exp>/samples/<sample>/raw/sxm/<file>`` (``experiment_paths.fits_path_budget``
#: calls that the deepest path there is), which is depth 5. The telemetry CSV at
#: ``<exp>/env/signals/`` is depth 3. So the old limit could see the noise and
#: could not see the signal — the exact inversion the operator reported.
_MAX_SEARCH_DEPTH = 5


def _is_scan_file(p: Path) -> bool:
    """False for files that live in a known non-scan sidecar directory."""
    try:
        return p.parent.name.lower() not in _NON_SCAN_DIR_NAMES
    except Exception:  # noqa: BLE001 — a weird path must not take discovery down
        return True


class ScanStat(NamedTuple):
    """One discovered file plus the stat fields the Data tab needs.

    ``mtime_ns`` is the INTEGER nanosecond mtime, not the float-seconds
    ``st_mtime``. The copy-collapse below compares two files' mtimes for
    EQUALITY, and float seconds cannot carry an NTFS timestamp exactly — two
    genuine byte-for-byte copies would compare unequal (or, worse, two unrelated
    files land on the same rounded value). ``shutil.copystat`` propagates the ns
    value verbatim, so the integer is what makes "this is the same file" decidable
    without hashing megabytes on every request."""

    path: Path
    mtime_ns: int
    size_bytes: int


def _norm_key(p: Path | str) -> str:
    """Case-folded real path — the identity of a path ON DISK.

    ``Path.resolve()`` alone was the old dedup key and it is not enough on
    Windows: it follows junctions but leaves case alone, so ``D:\\Data\\a.sxm``
    and ``d:\\data\\A.SXM`` (both of which the search dirs really do produce, one
    from config and one from the Nanonis session path) counted as two files. The
    scan-map underlay hit exactly this and fixed it with normcase+realpath on
    2026-08-09 (``api/routes/vision.py``); discovery never got the same fix."""
    try:
        return os.path.normcase(os.path.realpath(str(p)))
    except OSError:
        return os.path.normcase(str(p))


def collect_scan_stats(
    *search_dirs: str | None,
    max_depth: int = _MAX_SEARCH_DEPTH,
    exts: tuple[str, ...] | None = None,
) -> list[ScanStat]:
    """Discover scan files under the given dirs, mtime-desc, deduped by real path.

    Same walk as ``_collect_scans`` (which is now a thin wrapper) but it KEEPS
    the stat it already paid for. Callers that need size/mtime — the Data tab
    listing and the copy-collapse — would otherwise stat every file a second
    time, and a second stat is also a second chance to disagree with the first
    while Nanonis is writing."""
    wanted = tuple(e.lower() for e in exts) if exts else _SCAN_EXTS
    files: list[Path] = []
    for d in search_dirs:
        if not d:
            continue
        p = Path(d)
        if not p.exists():
            continue
        for depth in range(max_depth + 1):
            for ext in wanted:
                pattern = "/".join(["*"] * depth) + f"/*{ext}" if depth > 0 else f"*{ext}"
                # depth 0 = files sitting directly in a dir the CALLER named.
                # Never second-guess that one: if someone points discovery at a
                # folder called `env`, they meant that folder. The sidecar rule
                # only applies to directories we walked into ourselves.
                if depth == 0:
                    files.extend(p.glob(pattern))
                else:
                    files.extend(f for f in p.glob(pattern) if _is_scan_file(f))
    seen: set[str] = set()
    # Stat up front so a file that vanishes between glob() and the sort (common
    # while Nanonis is actively writing / rotating scan files) cannot crash the
    # Data tab. p.glob() returns paths that existed a moment ago; by the time we
    # stat() them one may be gone, raising FileNotFoundError / OSError. Skip any
    # path whose stat() fails rather than propagating up through get_latest_scans
    # -> build_data_strip_html into the GUI render.
    out: list[ScanStat] = []
    for f in files:
        key = _norm_key(f)
        if key in seen:
            continue
        seen.add(key)
        try:
            st = f.stat()
        except OSError:
            # File deleted/moved/inaccessible since glob() — drop it.
            continue
        out.append(ScanStat(path=f, mtime_ns=int(st.st_mtime_ns), size_bytes=int(st.st_size)))
    out.sort(key=lambda s: s.mtime_ns, reverse=True)
    return out


def _collect_scans(
    *search_dirs: str | None,
    max_depth: int = _MAX_SEARCH_DEPTH,
    exts: tuple[str, ...] | None = None,
) -> list[Path]:
    """Return all recognised Nanonis scan/spectroscopy files under the given
    dirs (mtime-desc, deduped). Recognises .sxm, .dat, .3ds.

    ``exts`` narrows the search to those extensions. Pass it rather than
    filtering the result: this listing is mtime-desc and callers slice the head
    off it, so a caller that wants only ``.sxm`` and filters AFTER the slice gets
    however many survive — which on a rig doing spectroscopy is close to none,
    because the .dat files are all newer. See ``_recent_scan_paths``.

    Files under a ``_NON_SCAN_DIR_NAMES`` directory are skipped — see there."""
    return [s.path for s in collect_scan_stats(*search_dirs, max_depth=max_depth, exts=exts)]


# ── copy collapse (自动拷贝 → 同一份数据在列表里出现多次) ────────────────────
#
# MAST ingests every scan Nanonis writes into the experiment folder
# (``logging.v2.filestore``), keeping the original where it was. That is
# deliberate and stays. What it does to THIS listing is that one measurement
# shows up once per copy, with an identical name, size and timestamp on every
# card — the operator sees "the same file" three times and cannot tell which
# one to click.
#
# Collapse them into one entry that says how many copies exist and where. Note
# what is NOT done here: nothing is dropped. A collapsed row still carries every
# member path, because a listing that silently hides files is the failure this
# codebase has already recorded twice in the other direction.

#: Path segment the ingest sink uses for files it could not place.
_QUARANTINE_SEG = "_quarantine"


def _experiment_root_key() -> str | None:
    """normcase'd experiment-folder root, or None when it cannot be resolved.

    Resolving it is best-effort on purpose: this module renders thumbnails for a
    GUI and must not acquire a hard dependency on the experiment-folder layout.
    When it is unavailable every file simply classifies as ``origin``, which is
    the honest answer — we cannot prove otherwise."""
    try:
        from mast.core.experiment_paths import experiment_root

        return _norm_key(experiment_root())
    except Exception:  # noqa: BLE001 — classification must never take discovery down
        return None


def classify_scan_path(path: Path | str, exp_root_key: str | None = None) -> str:
    """``"origin"`` | ``"experiment"`` | ``"quarantine"`` for one path.

    Vocabulary matches ``file_locations.root_kind`` (logging/v2/schema.py) so the
    two halves of the story — what is on disk, what the DB recorded — can be read
    side by side without a translation table."""
    key = _norm_key(path)
    parts = key.replace("\\", "/").split("/")
    if _QUARANTINE_SEG in parts:
        return "quarantine"
    root = exp_root_key if exp_root_key is not None else _experiment_root_key()
    if root and (key == root or key.startswith(root.rstrip("\\/") + os.sep)):
        return "experiment"
    return "origin"


class CollapsedScan(NamedTuple):
    """One measurement plus every on-disk copy of it.

    ``rep`` is the copy to show and to render previews from; ``members`` is all
    of them including ``rep``, so ``len(members)`` is the copy count and nothing
    is lost."""

    rep: ScanStat
    members: list[ScanStat]


def collapse_scan_groups(
    stats: list[ScanStat], exp_root_key: str | None = None
) -> list[CollapsedScan]:
    """Fold byte-identical copies of one measurement into a single entry.

    **The key is ``(lowercased name, size, mtime_ns)``** — deliberately not a
    content hash. Hashing would be the rigorous answer and is not affordable
    here: this runs on every listing request over hundreds of multi-megabyte
    files. The triple is sound for the case that actually produces duplicates,
    because ``filestore`` copies with ``shutil.copystat`` and therefore
    reproduces name, size and nanosecond mtime exactly.

    Files with the same basename in distinct session folders can contain different
    measurements. Different mtime_ns values keep them separate: a shared name alone
    is not evidence of identical content.

    If the key ever over-collapses, the failure is visible rather than silent:
    the entry names every member path, so an operator sees two unrelated files
    listed under one card instead of one of them vanishing.

    Representative: prefer a copy OUTSIDE the experiment folder. The originals
    keep their identity, while managed copies get re-ingested into whichever
    experiment is current — pointing previews at the original means the card does
    not change identity when that happens."""
    root = exp_root_key if exp_root_key is not None else _experiment_root_key()
    order: list[tuple[str, int, int]] = []
    groups: dict[tuple[str, int, int], list[ScanStat]] = {}
    for s in stats:
        key = (s.path.name.lower(), s.size_bytes, s.mtime_ns)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)

    out: list[CollapsedScan] = []
    for key in order:
        members = groups[key]
        rep = next(
            (m for m in members if classify_scan_path(m.path, root) == "origin"),
            members[0],
        )
        out.append(CollapsedScan(rep=rep, members=members))
    return out


# Legacy alias — kept so external callers / tests that reference the old name
# don't break. New code should use _collect_scans.
_collect_sxm = _collect_scans


def get_latest_scan(*search_dirs: str | None,
                    max_depth: int = _MAX_SEARCH_DEPTH) -> str | None:
    """Find the most recent .sxm / .dat / .3ds file across the given dirs."""
    unique = _collect_scans(*search_dirs, max_depth=max_depth)
    return str(unique[0]) if unique else None


def get_latest_scans(
    n: int,
    *search_dirs: str | None,
    max_depth: int = _MAX_SEARCH_DEPTH,
    exts: tuple[str, ...] | None = None,
) -> list[Path]:
    """Return up to *n* most-recent .sxm / .dat / .3ds files.

    ``exts`` restricts the *search*, so the n returned are the n newest of that
    kind — not the survivors of filtering n mixed results."""
    return _collect_scans(*search_dirs, max_depth=max_depth, exts=exts)[:n]


# ── 去衬底 (background / tilt removal) ──────────────────────────────────────
#
# A raw STM topograph is dominated by the sample tilt, not by the surface: over a
# 100 nm frame a fraction of a degree of slope is nanometres of z, while the
# atomic corrugation the operator wants to see is picometres. Rendering the raw
# array — which is what this preview did until 2026-08-21 — spends the entire
# colour range on the ramp and shows a smooth gradient with the structure buried
# in one shade of it.
#
# The four modes are the ones ``mast.vision.scan_prep`` already implements, so
# the Data tab and the analysis skills subtract backgrounds the SAME way:
#   raw    — no processing (what a raw file looks like; kept so the operator can
#            always see what is actually on disk).
#   plane  — one least-squares plane over the frame (conservative; keeps genuine
#            row-to-row differences).
#   line   — per-row first-order fit; also removes scan-line drift/offset banding.
#            The DEFAULT for .sxm.
#   auto   — measure the frame and let ``plan_for`` choose (may answer poly2 or
#            masked_line, which no fixed mode offers) and say why in Chinese.
#            ~3 s per frame: single previews only, never a grid of thumbnails.
FLATTEN_MODES: tuple[str, ...] = ("raw", "plane", "line", "auto")

#: Per-extension default. Only topography gets flattened by default: a .3ds bias
#: slice is spectroscopy, and a plane fit through it removes signal rather than
#: background.
_DEFAULT_FLATTEN_BY_EXT: dict[str, str] = {".sxm": "line"}


def default_flatten_for(ext: str) -> str:
    """Which mode applies when the caller did not name one."""
    return _DEFAULT_FLATTEN_BY_EXT.get(str(ext).lower(), "raw")


#: ``(norm_path, mtime_ns, size, flatten, channel)`` → data-URI.
#:
#: 512, not the old 128: the Data tab now renders a GRID of thumbnails and the
#: same file is legitimately cached under several flatten modes, so the working
#: set is (files on screen) × (modes the operator toggled through). At ~30-80 KB
#: per 256 px URI the ceiling is tens of MB.
_THUMB_CACHE: dict[tuple[str, int, int, str, str], str] = {}
_THUMB_CACHE_CAP = 512

#: Per-thumbnail metadata, keyed exactly like ``_THUMB_CACHE`` and evicted with
#: it — a cache hit has to answer "which channel / how was it flattened" too, or
#: the second request for a cached picture would report defaults.
_THUMB_META: dict[tuple[str, int, int, str, str], dict] = {}

#: ``(norm_path, mtime_ns)`` → ``(FlattenPlan, FrameMetrics)`` for ``auto``.
#: The measurement is the expensive half (~3 s); the answer does not change until
#: the file does, so the second look at the same frame is as fast as any other
#: mode. Metrics are kept alongside the plan because ``masked_line`` needs them
#: to re-apply.
_PLAN_CACHE: dict[tuple[str, int], tuple[object, object]] = {}
_PLAN_CACHE_CAP = 256

#: NaN pixels (an unfinished scan is normal) — dark grey, matching
#: ``data.visualization.plot_flattened_scan`` so an unscanned band reads the same
#: way in the Data tab as it does in a report figure.
_NAN_RGB: tuple[int, int, int] = (0x20, 0x20, 0x20)


def _flatten_frame(arr, mode: str, cache_key: tuple[str, int] | None = None,
                   nm_per_px: float | None = None):
    """Apply one flatten mode. Returns ``(array, effective_mode, clip, why)``.

    ``effective_mode`` can differ from ``mode``: ``auto`` resolves to whatever
    ``plan_for`` chose, and any mode degrades to ``raw`` if the processing raises
    — a preview that shows the unprocessed frame is far better than a card with
    no picture, as long as it SAYS which one it is (the caller reports the
    effective mode back to the client)."""
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    clip = (1.0, 99.0)
    if mode == "raw" or a.ndim != 2:
        return a, "raw", clip, []
    try:
        from mast.vision.scan_prep import line_subtract, poly_subtract

        if mode == "plane":
            return poly_subtract(a, 1), "plane", clip, []
        if mode == "line":
            return line_subtract(a, 1), "line", clip, []
        if mode == "auto":
            from mast.vision.scan_prep import apply_flatten, measure_frame, plan_for

            cached = _PLAN_CACHE.get(cache_key) if cache_key else None
            if cached is None:
                m = measure_frame(a, nm_per_px=nm_per_px)
                plan = plan_for(m)
                if cache_key:
                    _PLAN_CACHE[cache_key] = (plan, m)
                    if len(_PLAN_CACHE) > _PLAN_CACHE_CAP:
                        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
            else:
                plan, m = cached
            return (apply_flatten(a, plan.method, m), plan.method,
                    tuple(plan.clip), list(plan.why))
    except Exception as exc:  # noqa: BLE001 — a picture beats a blank card
        logger.info("flatten %s failed, falling back to raw: %s", mode, exc)
        return a, "raw", clip, [f"{mode} 处理失败，显示原始数据：{exc}"]
    return a, "raw", clip, []


def _image_to_png_uri(arr, size: int, clip: tuple[float, float] = (1.0, 99.0),
                      cmap: str = "afmhot") -> str | None:
    """2D array → base64 PNG data-URI, WITHOUT touching pyplot.

    pyplot keeps every figure it makes in a process-global registry (``Gcf``)
    that is not thread-safe, and these renders run on FastAPI's thread pool —
    ``vision/cluster_panel.py`` documents the same hazard. Going straight from
    array to pixels avoids the whole question and is faster besides.

    Row 0 of ``arr`` becomes the TOP row of the picture — one orientation
    convention for every branch, the one ``io.nanonis_files.rows_top_first``
    establishes. Callers hand over an array that is already the right way up
    (for .sxm that is what ``sxm_oriented_frames`` is for); this function makes
    no claim about acquisition order and holds no second copy of the #94 flip
    rule."""
    import base64
    import io as _io

    import numpy as np

    from mast.io.mosaic import _get_cmap, _normalize01, _resample_nn

    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        return None

    h, w = a.shape
    longest = max(h, w)
    size = max(int(size), 16)
    if longest > size:
        scale = size / float(longest)
        a = _resample_nn(a, max(1, int(round(h * scale))), max(1, int(round(w * scale))))

    normed = _normalize01(a, clip)
    rgb = (np.asarray(_get_cmap(cmap)(normed))[..., :3] * 255.0).astype(np.uint8)
    # NaN normalises to 0, which in afmhot is black and indistinguishable from a
    # genuinely low pixel. Paint it explicitly so "not scanned" cannot be misread
    # as "flat and dark".
    holes = ~np.isfinite(a)
    if holes.any():
        rgb[holes] = _NAN_RGB

    buf = _io.BytesIO()
    try:
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(buf, format="png")
    except Exception:  # noqa: BLE001 — Pillow absent/odd build
        import matplotlib.image as mpimg

        mpimg.imsave(buf, rgb, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_scan_thumbnail_v2(
    path: str,
    size: int = 256,
    flatten: str | None = None,
    channel: str | None = None,
) -> dict:
    """Render one Nanonis file to a PNG data-URI plus what was done to it.

    Returns a dict — ``uri`` (None when nothing could be drawn), ``flatten`` (the
    mode ACTUALLY applied), ``why`` (Chinese rationale, ``auto`` only),
    ``channel``, ``width_nm`` / ``height_nm`` / ``bias_v`` (so a card can label
    the picture without a second round trip), ``detail``.

    ``render_scan_thumbnail`` is the old string-only entry point and delegates
    here."""
    out: dict = {"uri": None, "flatten": "raw", "why": [], "channel": None,
                 "width_nm": None, "height_nm": None, "bias_v": None, "detail": None}
    p = Path(path)
    ext = p.suffix.lower()
    try:
        mtime_ns = int(p.stat().st_mtime_ns)
    except OSError:
        out["detail"] = "file not found"
        return out

    mode = (flatten or default_flatten_for(ext)).strip().lower()
    if mode not in FLATTEN_MODES:
        mode = default_flatten_for(ext)

    cache_key = (_norm_key(p), mtime_ns, int(size), mode, channel or "")
    cached = _THUMB_CACHE.get(cache_key)
    if cached is not None:
        meta = _THUMB_META.get(cache_key) or {}
        out.update(meta)
        out["uri"] = cached
        return out

    try:
        import numpy as np  # noqa: F401 — the render paths below all need it
    except ImportError:
        out["detail"] = "numpy unavailable"
        return out

    plan_key = (_norm_key(p), mtime_ns)
    try:
        if ext == ".sxm":
            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

            scan = read_sxm(path)
            channels = list((scan.get("channels") or {}).keys())
            # Same preference order the Data tab has always used; an explicit
            # channel wins. #94: sxm_oriented_frames is what puts row 0 at the
            # top of the picture for BOTH scan directions and un-mirrors the
            # backward block — the thumbnail is also the scan-map underlay, so a
            # frame drawn upside down lands upside down on the map.
            wanted = ([channel] if channel else []) + ["Z", "z", "Current", "current",
                                                       "Bias", "bias"] + channels
            frame = None
            for name in wanted:
                if not name:
                    continue
                f = sxm_oriented_frames(scan, name)
                if f.get("forward") is not None:
                    frame = f
                    break
            if frame is None:
                out["detail"] = "no renderable channel"
                return out
            out["channel"] = frame.get("channel")
            out["width_nm"] = frame.get("width_nm")
            out["height_nm"] = frame.get("height_nm")
            out["bias_v"] = frame.get("bias_v")
            data, eff, clip, why = _flatten_frame(
                frame["forward"], mode, plan_key, frame.get("nm_per_px"))
            out["flatten"], out["why"] = eff, why
            out["uri"] = _image_to_png_uri(data, size, clip)
        elif ext == ".3ds":
            from mast.io.nanonis_files import read_3ds

            grid = read_3ds(path).get("grid")
            if grid is None or not grid.size:
                out["detail"] = "no grid data"
                return out
            # Middle bias slice, as before — but drawn row-0-at-top like every
            # other branch, where the old pyplot path left it row-0-at-bottom.
            # That was the `origin` variable's INITIAL value, not a decision
            # about .3ds row order: only the .sxm branch ever set it, and #94 is
            # the record of that default being wrong there. Whether a grid comes
            # off Nanonis in the same order is not known — there is no .3ds here
            # with a known feature at a known edge to check against, the same
            # position `_render_sm4` documents below. Sharing one convention is
            # the honest default; it is not a claim to have verified this one.
            data, eff, clip, why = _flatten_frame(
                grid[:, :, grid.shape[2] // 2], mode, plan_key)
            out["flatten"], out["why"] = eff, why
            out["uri"] = _image_to_png_uri(data, size, clip)
        elif ext == ".dat":
            # A spectrum is a curve, not an image; flattening does not apply.
            # The interactive version lives at GET /api/scans/spectrum — this is
            # the thumbnail for the card.
            out["flatten"] = "raw"
            out["uri"] = _render_curve_thumbnail(path, size)
        else:
            out["detail"] = f"no thumbnail renderer for {ext}"
            return out
    except Exception as exc:
        logger.info("Thumbnail parse failed for %s: %s", path, exc)
        out["detail"] = str(exc)
        return out

    if out["uri"]:
        _THUMB_CACHE[cache_key] = out["uri"]
        _THUMB_META[cache_key] = {k: out[k] for k in
                                  ("flatten", "why", "channel", "width_nm",
                                   "height_nm", "bias_v")}
        if len(_THUMB_CACHE) > _THUMB_CACHE_CAP:
            old = next(iter(_THUMB_CACHE))
            _THUMB_CACHE.pop(old)
            _THUMB_META.pop(old, None)
    return out


def _render_curve_thumbnail(path: str, size: int) -> str | None:
    """First two columns of a .dat as a tiny line plot.

    Uses ``_new_figure`` (a bare Figure + Agg canvas) rather than pyplot for the
    thread-safety reason in ``_image_to_png_uri``."""
    import base64
    import io as _io

    from mast.io.nanonis_files import read_dat

    cols = read_dat(path).get("columns") or {}
    if len(cols) < 2:
        return None
    names = list(cols.keys())
    dpi = 100
    in_size = max(int(size), 16) / dpi
    # _new_figure mirrors plt.subplots: it returns (fig, axes) and forwards **kw
    # to fig.subplots(), NOT to Figure() — so dpi belongs on savefig.
    fig, ax = _new_figure((in_size, in_size))
    ax.plot(cols[names[0]], cols[names[1]], linewidth=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_scan_thumbnail(path: str, size: int = 96,
                          flatten: str | None = None,
                          channel: str | None = None) -> str | None:
    """Render a Nanonis file (.sxm/.dat/.3ds) as a small base64 PNG thumbnail.

    Thin wrapper over :func:`render_scan_thumbnail_v2` kept for the callers that
    only want the picture (scan-map underlay, agent vision middleware, the old
    GUI data strip). Note that .sxm thumbnails are now LINE-FLATTENED by default
    — the same picture these callers wanted all along, with the sample tilt taken
    out."""
    return render_scan_thumbnail_v2(path, size=size, flatten=flatten,
                                    channel=channel).get("uri")


# Theme-neutral palette — readable on both light (#fafafa) and dark (#09090b).
_TEXT_COLOR = "#52525b"   # zinc-600
_SPINE_COLOR = "#a1a1aa"  # zinc-400


def _style_axis(ax) -> None:
    ax.tick_params(colors=_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)


def _new_figure(figsize, **kw):
    """Create a standalone matplotlib Figure WITHOUT registering it in pyplot's
    global ``Gcf`` figure manager.

    HANDLE-LEAK FIX: the render helpers used ``plt.subplots()``,
    which stashes every figure in ``matplotlib._pylab_helpers.Gcf`` and keeps
    it alive (and its Agg canvas / file handles) until an explicit
    ``plt.close()``. ``render_scan_preview`` returns the Figure to Gradio and
    never closed it, so each Data-tab refresh leaked one figure + canvas. A
    figure built directly via ``Figure()`` + an attached ``FigureCanvasAgg`` is
    NOT tracked by Gcf, so it is freed by GC once Gradio is done serialising it
    — no global registry, nothing to close. Returns ``(fig, axes)`` to mirror
    ``plt.subplots``.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)  # attach a canvas so savefig/draw work, no Gcf entry
    nrows = kw.pop("nrows", 1)
    ncols = kw.pop("ncols", 1)
    axes = fig.subplots(nrows, ncols, **kw)
    return fig, axes


def _render_sxm(filepath: str, plt):
    """2D heatmap of preferred channel from a .sxm scan."""
    from mast.io.nanonis_files import read_sxm
    return _render_channels(read_sxm(filepath), filepath, plt, honour_scan_dir=True)


def _render_sm4(filepath: str, plt):
    """2D heatmap of preferred topographic channel from an RHK .sm4 scan.

    ``honour_scan_dir`` stays False here — NOT because RHK row order is known to
    match, but because it is **not known at all**: there is no .sm4 anywhere in
    this tree to check against and ``read_sm4`` records no direction field. The
    #94 flip is a claim about acquisition order; making it for a format we have
    never inspected would be inventing one. Leaving this branch alone is not a
    statement that it is right. To settle it, get one real .sm4 with a known
    feature at a known edge.
    """
    from mast.io.nanonis_files import read_sm4
    return _render_channels(read_sm4(filepath), filepath, plt, honour_scan_dir=False)


def _render_channels(result, filepath: str, plt, *, honour_scan_dir: bool = False):
    """2D heatmap of the preferred channel from a read_sxm/read_sm4 result dict."""
    channels = result.get("channels", {})

    scan_data = None
    channel_name = "Z"
    for ch_name in ["Z", "z", "Current", "current", "Bias", "bias"]:
        if ch_name in channels:
            ch = channels[ch_name]
            scan_data = ch.get("forward", ch.get("backward"))
            if scan_data is not None:
                channel_name = ch_name
                break
    if scan_data is None and channels:
        first_name = next(iter(channels))
        ch = channels[first_name]
        scan_data = ch.get("forward", ch.get("backward"))
        if scan_data is not None:
            channel_name = first_name
    if scan_data is None:
        return None

    # #94 — same flip as the thumbnail, same single rule. See rows_top_first.
    img_origin = "lower"
    if honour_scan_dir:
        from mast.io.nanonis_files import rows_top_first
        scan_data = rows_top_first(scan_data,
                                   (result.get("header") or {}).get("scan_dir"))
        img_origin = "upper"

    fig, ax = _new_figure((6, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    im = ax.imshow(scan_data, cmap="afmhot", origin=img_origin, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    cbar.outline.set_edgecolor(_SPINE_COLOR)
    for label in cbar.ax.get_yticklabels():
        label.set_color(_TEXT_COLOR)
    ax.set_title(f"{channel_name} — {Path(filepath).name}", color=_TEXT_COLOR, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def _render_dat(filepath: str, plt):
    """Line plot of every numeric column vs the first column (typically bias)."""
    from mast.io.nanonis_files import read_dat
    result = read_dat(filepath)
    cols = result.get("columns") or {}
    if not cols:
        return None
    names = list(cols.keys())
    x_name = names[0]
    x = cols[x_name]
    y_names = names[1:] if len(names) > 1 else names
    if not y_names:
        return None

    fig, ax = _new_figure((6, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for yn in y_names:
        ax.plot(x, cols[yn], linewidth=1.2, label=yn)
    ax.set_xlabel(x_name, color=_TEXT_COLOR, fontsize=9)
    ax.set_ylabel(", ".join(y_names) if len(y_names) <= 2 else "value", color=_TEXT_COLOR, fontsize=9)
    leg = ax.legend(loc="best", fontsize=8, frameon=False)
    for text in leg.get_texts():
        text.set_color(_TEXT_COLOR)
    ax.set_title(f"Spectrum — {Path(filepath).name}", color=_TEXT_COLOR, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def _render_txt(filepath: str, plt):
    """Render a generic numeric text file (.txt/.csv/.asc/.tsv).

    A wide 2-D matrix (e.g. an exported height map) is shown as a heatmap; a
    narrow columnar file (e.g. exported spectra) is shown as a line plot of its
    columns vs the first. Returns None if nothing numeric parses.
    """
    from mast.io.nanonis_files import read_txt
    result = read_txt(filepath)
    matrix = result.get("matrix")
    cols = result.get("columns") or {}
    if matrix is None or getattr(matrix, "size", 0) == 0:
        return None

    # Wide 2-D matrix → image-like height map. Narrow / columnar → line plot.
    if matrix.ndim == 2 and matrix.shape[0] > 1 and matrix.shape[1] > 8:
        fig, ax = _new_figure((6, 5))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        im = ax.imshow(matrix, cmap="afmhot", origin="lower", aspect="auto")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
        cbar.outline.set_edgecolor(_SPINE_COLOR)
        for label in cbar.ax.get_yticklabels():
            label.set_color(_TEXT_COLOR)
        ax.set_title(f"matrix — {Path(filepath).name}", color=_TEXT_COLOR, fontsize=10)
        _style_axis(ax)
        fig.tight_layout()
        return fig

    names = list(cols.keys())
    if not names:
        return None
    x = cols[names[0]]
    y_names = names[1:] if len(names) > 1 else names
    fig, ax = _new_figure((6, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for yn in y_names:
        ax.plot(x, cols[yn], linewidth=1.2, label=yn)
    ax.set_xlabel(names[0], color=_TEXT_COLOR, fontsize=9)
    ax.set_ylabel(", ".join(y_names) if len(y_names) <= 2 else "value", color=_TEXT_COLOR, fontsize=9)
    leg = ax.legend(loc="best", fontsize=8, frameon=False)
    for text in leg.get_texts():
        text.set_color(_TEXT_COLOR)
    ax.set_title(f"data — {Path(filepath).name}", color=_TEXT_COLOR, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def _render_3ds(filepath: str, plt):
    """Grid-spectroscopy preview: 2D map at mid-bias slice + averaged spectrum."""
    import numpy as np
    from mast.io.nanonis_files import read_3ds
    result = read_3ds(filepath)
    grid = result.get("grid")
    bias = result.get("bias")
    if grid is None or grid.size == 0:
        return None
    ny, nx, n_points = grid.shape
    slice_idx = n_points // 2  # midpoint of the bias sweep
    slice_2d = grid[:, :, slice_idx]
    mean_spec = grid.reshape(-1, n_points).mean(axis=0)

    fig, axes = _new_figure((9, 4.2), ncols=2, gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_alpha(0.0)

    ax0 = axes[0]
    ax0.set_facecolor("none")
    im = ax0.imshow(slice_2d, cmap="afmhot", origin="lower", aspect="equal")
    cbar = fig.colorbar(im, ax=ax0, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
    cbar.outline.set_edgecolor(_SPINE_COLOR)
    for label in cbar.ax.get_yticklabels():
        label.set_color(_TEXT_COLOR)
    bias_val = float(bias[slice_idx]) if bias is not None and len(bias) > slice_idx else slice_idx
    ax0.set_title(f"Slice @ bias≈{bias_val:.3g} V", color=_TEXT_COLOR, fontsize=10)
    _style_axis(ax0)

    ax1 = axes[1]
    ax1.set_facecolor("none")
    x_axis = bias if bias is not None and len(bias) == n_points else np.arange(n_points)
    ax1.plot(x_axis, mean_spec, linewidth=1.2)
    ax1.set_xlabel("Bias (V)" if bias is not None else "point index", color=_TEXT_COLOR, fontsize=9)
    ax1.set_ylabel("⟨channel⟩ over grid", color=_TEXT_COLOR, fontsize=9)
    ax1.set_title(f"Grid mean — {Path(filepath).name}", color=_TEXT_COLOR, fontsize=10)
    _style_axis(ax1)

    fig.tight_layout()
    return fig


def _render_placeholder(filepath: str, plt, msg: str):
    fig, ax = _new_figure((6, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.text(
        0.5, 0.5, f"Could not render:\n{Path(filepath).name}\n\n({msg})",
        transform=ax.transAxes, ha="center", va="center",
        color=_TEXT_COLOR, fontsize=10,
    )
    ax.set_title(f"Preview — {Path(filepath).name}", color=_TEXT_COLOR, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def render_scan_preview(filepath: str) -> Figure | None:
    """Read and render a Nanonis scan/spectroscopy file as a matplotlib Figure.

    Dispatches by file extension:
        .sxm  → 2D heatmap of preferred topography channel
        .dat  → line plot of all numeric columns vs the first column (bias)
        .3ds  → mid-bias 2D slice + grid-averaged spectrum
        .txt/.csv/.asc/.tsv → matrix heatmap (wide) or columnar line plot

    Returns None if the file is missing or the matplotlib/numpy stack is
    unavailable. Returns a placeholder figure if parsing fails for a
    supported extension (so the GUI doesn't show an empty box).
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning("Scan file not found: %s", filepath)
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for scan preview")
        return None

    ext = path.suffix.lower()
    try:
        if ext == ".sxm":
            fig = _render_sxm(filepath, plt)
        elif ext == ".sm4":
            fig = _render_sm4(filepath, plt)
        elif ext == ".dat":
            fig = _render_dat(filepath, plt)
        elif ext == ".3ds":
            fig = _render_3ds(filepath, plt)
        elif ext in (".txt", ".csv", ".asc", ".tsv"):
            fig = _render_txt(filepath, plt)
        else:
            return _render_placeholder(filepath, plt, f"unsupported extension {ext}")
    except Exception as exc:
        logger.info("Could not parse %s: %s", ext, exc)
        return _render_placeholder(filepath, plt, f"{ext} parse error: {exc}")

    if fig is None:
        return _render_placeholder(filepath, plt, "no data channels / columns")
    return fig
