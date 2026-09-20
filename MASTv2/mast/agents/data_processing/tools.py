"""Data-Processing agent tool list — Phase 5 real implementations.

Real numpy/scipy analysis on STM image files. Supports `.npy` directly (most
universal) and `.sxm` via v1's parser when available (`mast.io.nanonis_files`).
RestrictedPython sandbox enables ad-hoc analysis snippets safely.

Tools:
  1. load_scan(path)                  — load .npy / .sxm; return shape + stats
  2. fft_2d(path, channel)            — 2D FFT; Bragg peaks + lattice constant
  3. plane_subtract(path, order)      — polynomial plane removal + RMS roughness
  4. detect_defects(path, min_size_px) — connected-component defect counting
  5. fit_sts_peaks(path, n_peaks)     — Gaussian fits to dI/dV
  6. run_numpy_snippet(code)          — RestrictedPython sandbox
  7. buffer tools                     — read live tip/scan state
  8/9. handoffs                       — supervisor / paper_writing
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
from langchain_core.messages import ToolMessage
# RunnableConfig 必须是**运行时**可解析的名字，不能藏在 TYPE_CHECKING 里：
# LangChain 建 args_schema 时会 eval 类型注解（`from __future__ import annotations`
# 让它们变成字符串），拿不到这个名字就直接 NameError。同一个形状咬过一次
# （`NotRequired` 在 PEP 563 下静默失效）。
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from mast.agents._shared.artifact_channel import AnalysisResult, ArtifactToolReturn
from mast.agents._shared.buffer_tools import make_buffer_tools
from mast.agents._shared.handoff import make_handoff
# 米量纲参数的类型 + 「解析 SI 前缀 → 判量级」的那一份实现。刻意从 _shared 导入
# 公开名而不是在这里再写一遍 —— 同一个物理量两套解析迟早各自漂移。
from mast.agents._shared.meta_tools import MetreSize, resolve_metre_args

if TYPE_CHECKING:
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)

# ── run_numpy_snippet resource bounds ──
# The snippet is LLM-authored, so it MUST run under a wall-clock timeout and a
# best-effort address-space cap: an unbounded `while True:` loop or a giant
# allocation (`np.zeros((1e6, 1e6))`) would otherwise hang or OOM the whole
# agent process. The RestrictedPython sandbox controls WHAT names a snippet may
# touch; these bounds control HOW LONG / HOW MUCH it may consume.
_SNIPPET_TIMEOUT_S = 5.0           # wall-clock ceiling for one snippet run
_SNIPPET_MEM_LIMIT_BYTES = 1 << 30  # 1 GiB address-space cap (POSIX best-effort)


# ─────────────────────────────────────────────────────────────────────
# File-loading helper — .npy / .sxm (v2 parser) / .dat
# ─────────────────────────────────────────────────────────────────────

_SCAN_SUFFIXES = (".sxm", ".npy", ".npz", ".dat")


def _resolve_missing(p: Path) -> "Path | None":
    """A path the model guessed → the file that actually exists, or None.

    THE BUG THIS EXISTS FOR:

        load_scan failed: FileNotFoundError: file not found:
        D:\\MAST-data\\working-sessions\\Au111_mica_STM_STS_Au111_mica_001_.npy

    The scan was on disk. The model simply reconstructed a plausible path — right
    stem, wrong directory, wrong extension (``.npy`` for what Nanonis saved as
    ``.sxm``) — and ``_load_array`` did a bare ``p.exists()`` and gave up. An LLM
    WILL guess paths; refusing to look is a choice, and it strands the operator's
    data behind a typo.

    So look, in the places the run actually wrote to: the directories
    ``core.scan_registry`` recorded (the IC skills feed it on every SaveScan /
    session resolve), plus the scans it recorded by name. Match on the STEM, so a
    wrong extension resolves too — that is the exact miss above. Never invent a
    file: return None and let the caller say where it DID look.
    """
    try:
        from mast.core.scan_registry import known_scan_dirs, recent_scan_paths
    except Exception:  # noqa: BLE001 — registry unavailable → no resolution
        return None

    stem = p.stem.rstrip("_")
    if not stem:
        return None

    # 1. a scan this run really saved, matched by stem
    for s in recent_scan_paths(50):
        sp = Path(s)
        if sp.stem.rstrip("_") == stem and sp.is_file():
            return sp

    # 2. the same basename in a directory the run really used
    dirs = [d for d in known_scan_dirs() if d.is_dir()]
    if p.parent.is_dir():
        dirs.insert(0, p.parent)
    for d in dirs:
        exact = d / p.name
        if exact.is_file():
            return exact
        for suf in _SCAN_SUFFIXES:      # right stem, wrong extension
            cand = d / f"{stem}{suf}"
            if cand.is_file():
                return cand
    return None


def _not_found_message(p: Path) -> str:
    """Say where we looked. "file not found" alone sent the agent round in
    circles guessing new paths ."""
    try:
        from mast.core.scan_registry import known_scan_dirs, recent_scan_paths

        dirs = [str(d) for d in known_scan_dirs()]
        recent = [Path(s).name for s in recent_scan_paths(5)]
    except Exception:  # noqa: BLE001
        dirs, recent = [], []
    msg = f"file not found: {p}"
    if dirs:
        msg += f"；已在这些目录按同名/同前缀查找未果：{', '.join(dirs)}"
    else:
        msg += "；本进程尚无已记录的扫描目录"
    if recent:
        msg += f"；本次运行已保存的扫描文件：{', '.join(recent)}"
    # ALWAYS point at the tool that knows. Without this the agent's next move is
    # to guess another path — which is exactly the loop the operator watched it
    # spin in on 2026-07-10 (#92: "load_scan then failed on a dozen guessed paths").
    msg += "。不要自行拼接路径——调用 get_latest_scan_file 取真实路径。"
    return msg


def _load_array(path: str) -> tuple[np.ndarray, dict]:
    """Load an STM image as a 2D numpy array. Returns (array, metadata).

    Supported formats:
      .npy       — direct numpy load
      .sxm       — via v2's mast.io.nanonis_files.read_sxm (topography
                   channel preferred; channel name + header geometry in meta)
      .dat       — STS dI/dV column-pair (bias, didv)

    A path that does not exist is RESOLVED against the scan registry before it is
    refused (see :func:`_resolve_missing`) — the model guessing a plausible-but-
    wrong path is the normal case, not an exceptional one.
    """
    p = Path(path)
    if not p.exists():
        resolved = _resolve_missing(p)
        if resolved is None:
            raise FileNotFoundError(_not_found_message(p))
        logger.info("load: %s not found → resolved to %s", path, resolved)
        p = resolved
    suffix = p.suffix.lower()
    meta: dict = {"path": str(p), "size_bytes": p.stat().st_size}

    if suffix == ".npy":
        arr = np.load(p)
        meta["format"] = "npy"
        return arr, meta

    if suffix == ".sxm":
        # ONE .sxm parser for the whole system: mast.io.nanonis_files.read_sxm —
        # the same bytes the data viewer, load_image_2d, and the mosaic tool
        # read. The topography-channel choice reuses mast.data.loaders'
        # _pick_image_channel so "which channel is the image" is decided in
        # exactly one place (feedback ②: .sxm handling had drifted per-tool).
        from mast.data.loaders import _pick_image_channel
        from mast.io.nanonis_files import read_sxm, sxm_frame_meta
        parsed = read_sxm(str(p))
        header = parsed.get("header", {}) or {}
        channels = parsed.get("channels", {}) or {}
        if not channels:
            raise ValueError(f"no channel data in {path} (header-only sxm?)")
        ch_name = _pick_image_channel(channels, None)
        ch = channels[ch_name] or {}
        arr = ch.get("forward")
        if arr is None:
            arr = next(iter(ch.values()), None)
        if arr is None:
            raise ValueError(f"channel {ch_name!r} in {path} has no frames")
        meta["format"] = "sxm"
        meta["channel"] = ch_name
        meta["channels_available"] = list(channels)
        for k in ("scan_offset", "scan_range", "scan_pixels", "bias", "comment"):
            if k in header:
                meta[k] = (list(header[k])
                           if isinstance(header[k], (list, tuple)) else header[k])
        # Enrich the scan registry so a later load_scan(scan_id=…) / list_scans()
        # carries real channels + geometry, not just a path (feedback ①).
        try:
            from mast.core.scan_registry import record_scan
            record_scan(str(p), channels=list(channels),
                        frame=sxm_frame_meta(header))
        except Exception:  # noqa: BLE001 — registry is best-effort
            pass
        return np.asarray(arr), meta

    if suffix in (".dat", ".txt"):
        # Canonical readers (mast.data.load_spectrum → read_dat / read_txt), NOT a
        # bare np.loadtxt — the viewer + paper skills read these files this way
        # (feedback ②: text parsing had diverged per-tool). A .dat is ambiguous:
        # a Nanonis point spectrum HAS a [DATA] block (read_dat), but a plain
        # 2-column bias/dIdV export does not — fit_sts_peaks documents both. Try
        # the Nanonis reader first, then fall back to the canonical plain-text
        # reader (read_txt) so a headerless .dat still loads.
        from mast.data import load_spectrum_named
        try:
            arr, col_names = load_spectrum_named(p)
            arr = np.asarray(arr)
        except Exception:  # noqa: BLE001 — headerless .dat → plain numeric matrix
            from mast.io.nanonis_files import read_txt
            arr = np.asarray(read_txt(str(p))["matrix"])
            col_names = []
        meta["format"] = "dat"
        # Carried so load_scan can report PER-COLUMN statistics. A spectrum is a
        # table of unlike quantities (volts beside amps), not an image, and
        # pooling them produces numbers with no physical meaning at all.
        if col_names:
            meta["columns"] = col_names
        return arr, meta

    raise ValueError(f"unsupported file format: {suffix} (only .npy / .sxm / .dat)")


def _truncate(s: str, n: int = 1500) -> str:
    return s if len(s) <= n else s[:n] + "...[truncated]"


# ─────────────────────────────────────────────────────────────────────
# 1. load_scan
# ─────────────────────────────────────────────────────────────────────

@tool("load_scan")
def load_scan(path: str = "", scan_id: str = "") -> str:
    """Load an STM image / STS file and report shape + basic statistics.

    Args:
        path: Absolute path to a .npy / .sxm / .dat file. A plausible-but-wrong
            path (right name, wrong dir/extension) is resolved against the scan
            registry before being refused.
        scan_id: Alternative to path — a scan's identity (its Nanonis basename,
            e.g. "Au111_mica_001"). Resolved to the real path via the scan
            registry, so you do NOT have to know where Nanonis saved it. Use
            list_scan_dir / glob_scans first to see the known scan_ids.

    Returns shape, dtype, min/max/mean of the array. Give path OR scan_id.
    """
    src = path or ""
    if not src and scan_id:
        from mast.core.scan_registry import resolve_scan_id
        src = resolve_scan_id(scan_id) or ""
        if not src:
            return (
                f"load_scan failed: 未知 scan_id {scan_id!r}。"
                "先调用 list_scan_dir / glob_scans 查看已记录的扫描（含 scan_id），"
                "或用 get_latest_scan_file 取最新扫描路径。"
            )
    if not src:
        return "load_scan failed: 需要提供 path 或 scan_id 之一。"
    # A skill's tool-return sidecar is a SUMMARY OVERFLOW file (repr/JSON of a
    # result dict), not measurement data — nothing is meant to read it back.
    # Refuse before parsing: a GridSTS summary can contain one all-numeric line
    # ("1400 1400 5"), which parses as a 1x3 "scan" and yields plausible-looking
    # statistics that are pure garbage. A wrong number an agent believes is worse
    # than an error it must handle.
    if "tool_returns" in str(src).replace("\\", "/").split("/"):
        return (
            f"load_scan failed: {src} 是工具返回记录(摘要溢出件)，不是测量数据。"
            "谱数据由 Nanonis 自动保存为 .dat、图像为 .sxm —— "
            "请用 list_scan_dir / glob_scans 在扫描目录里找它们。"
        )
    try:
        arr, meta = _load_array(src)
    except Exception as e:
        return f"load_scan failed: {type(e).__name__}: {e}"
    # `ndim < 1` never fired for the shapes that actually break things: an empty
    # array from a .npy is (0, 0) — ndim 2 — and sailed straight into np.nanmin,
    # which raises "zero-size array to reduction operation fmin which has no
    # identity" OUTSIDE the try above, taking down the whole run with an error
    # naming neither the file nor the cause. Same for a string-dtype .npy
    # (UFuncTypeError). Check what actually matters: is there numeric data.
    if arr.size == 0:
        return (f"load_scan failed: {meta.get('path', src)} 解析出空数组 "
                f"(shape {arr.shape}) — 该文件不含可用的数值数据。")
    if arr.dtype.kind not in "fiub":
        return (f"load_scan failed: {meta.get('path', src)} 的数据类型是 "
                f"{arr.dtype} (非数值) — 无法作为扫描数据处理。")
    # A SPECTRUM is a table of unlike quantities — a bias column in volts next to
    # currents in amps — so whole-array statistics are meaningless, and worse,
    # they LOOK fine. Measured end-to-end on 2026-07-28: a real 201×3 STS .dat
    # whose currents are ~1e-9 A reported "min: -1, max: 1, mean: 3.072e-10,
    # std: 0.335". Every one of those four numbers is dominated by, or is purely,
    # the bias axis — and the run's report published "std ≈ 0.335" as a
    # repeatability statistic for three spectra. Report each column separately,
    # under its real name, so there is nothing left to misread.
    col_names = meta.get("columns") or []
    if col_names and arr.ndim == 2 and arr.shape[1] == len(col_names):
        lines = [
            f"Loaded {meta.get('format', '?')} spectrum from {meta.get('path', src)}",
            f"  shape: {arr.shape}, dtype: {arr.dtype}  "
            f"({arr.shape[0]} points × {arr.shape[1]} columns)",
            "  逐列统计（不同物理量不可混算）:",
        ]
        for j, nm in enumerate(col_names):
            c = arr[:, j]
            lines.append(
                f"    {nm}: min {np.nanmin(c):.4g}, max {np.nanmax(c):.4g}, "
                f"mean {np.nanmean(c):.4g}, std {np.nanstd(c):.4g}")
        lines.append("  第 1 列通常是扫描轴（偏压/Z），不是信号。"
                     "画图用 plot_spectrum，找峰用 fit_sts_peaks。")
        return "\n".join(lines)

    info = (
        f"Loaded {meta.get('format', '?')} from {meta.get('path', src)}\n"
        f"  shape: {arr.shape}, dtype: {arr.dtype}\n"
        f"  min: {np.nanmin(arr):.4g}, max: {np.nanmax(arr):.4g}, "
        f"mean: {np.nanmean(arr):.4g}\n"
        f"  std: {np.nanstd(arr):.4g}"
    )
    if meta.get("channel"):
        info += (f"\n  channel: {meta['channel']} "
                 f"(available: {', '.join(meta.get('channels_available', []))})")
    for k in ("scan_offset", "scan_range", "scan_pixels"):
        if k in meta:
            info += f"\n  {k}: {meta[k]}"
    return info


# ─────────────────────────────────────────────────────────────────────
# 2. fft_2d
# ─────────────────────────────────────────────────────────────────────

@tool("fft_2d")
def fft_2d(path: str, top_n_peaks: int = 4) -> str:
    """Compute the 2D FFT magnitude and report top-N Bragg-like peaks.

    Args:
        path: Absolute path to the scan file (.npy / .sxm).
        top_n_peaks: Number of strongest *physical* peaks (excluding DC) to
            report. Default 4.

    Returns peak (kx, ky) positions in pixel-frequency units (cycles per pixel)
    and the implied lattice constant if pixel-size metadata is available.

    Mirror handling: an STM topograph is real-valued, so its FFT magnitude is
    centro-symmetric — every physical Bragg vector ``k`` appears twice, at
    ``+k`` and ``-k``. Reporting both would double-count the lattice (e.g. a
    single 1D grating would look like two distinct Bragg vectors). We therefore
    restrict the peak search to a single half-plane (ky > 0, or ky == 0 and
    kx >= 0) so each ``±k`` pair contributes exactly one peak.
    """
    try:
        arr, meta = _load_array(path)
    except Exception as e:
        return f"fft_2d load failed: {type(e).__name__}: {e}"
    if arr.ndim != 2:
        return f"fft_2d requires 2D image, got shape {arr.shape}"

    # Subtract mean to suppress DC; compute FFT magnitude
    arr = arr - arr.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    h, w = F.shape
    cy, cx = h // 2, w // 2
    # Frequency offsets from DC (centre), per axis.
    yy, xx = np.ogrid[:h, :w]
    dy = yy - cy
    dx = xx - cx
    # Mask DC neighborhood (|k| within a few px of centre is the DC blob).
    dc_mask = (dy ** 2 + dx ** 2) > 9
    # Half-plane mask: keep only one member of each (+k, -k) mirror pair so a
    # single physical Bragg vector is reported once, not twice. The retained
    # half is {ky > 0} ∪ {ky == 0 and kx >= 0}; its mirror fills the other half.
    half_plane = (dy > 0) | ((dy == 0) & (dx >= 0))
    mask = dc_mask & half_plane
    F_masked = np.where(mask, F, 0.0)
    # Cap the request at the number of available (non-zero) candidates so
    # argpartition never indexes past the valid pool.
    n_candidates = int(np.count_nonzero(F_masked))
    n_report = max(1, min(top_n_peaks, n_candidates)) if n_candidates else 0
    peaks = []
    if n_report:
        flat = F_masked.flatten()
        idx = np.argpartition(flat, -n_report)[-n_report:]
        idx_sorted = idx[np.argsort(-flat[idx])]
        for i in idx_sorted:
            py, px = i // w, i % w
            kx = (px - cx) / w  # cycles per pixel
            ky = (py - cy) / h
            peaks.append((float(kx), float(ky), float(F.flat[i])))

    lines = [f"FFT 2D analysis of {path} (shape {arr.shape})"]
    if not peaks:
        lines.append("  no non-DC peaks found (image may be featureless / flat)")
    for n, (kx, ky, mag) in enumerate(peaks, 1):
        lines.append(f"  peak {n}: (kx={kx:+.4f}, ky={ky:+.4f}) mag={mag:.4g}")
    # Try lattice constant if metadata has nm_per_pixel
    nmpp = meta.get("nm_per_pixel")
    if nmpp and peaks:
        # Use the strongest non-DC peak's |k|
        kx, ky, _ = peaks[0]
        k_mag = max((kx * kx + ky * ky) ** 0.5, 1e-9)
        a_nm = nmpp / k_mag
        lines.append(f"  estimated lattice constant ≈ {a_nm:.3f} nm "
                     f"(from peak 1, nm_per_pixel={nmpp})")
    elif not nmpp:
        lines.append("  (no nm_per_pixel metadata; lattice constant not computed)")
    return _truncate("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# 3. plane_subtract
# ─────────────────────────────────────────────────────────────────────

@tool("plane_subtract")
def plane_subtract(path: str, order: int = 1) -> str:
    """Fit and subtract a polynomial background plane; report RMS roughness.

    Args:
        path: Absolute path to the scan file (.npy / .sxm).
        order: Polynomial order. 1 = linear tilt removal (default), 2 = quadratic.

    Returns the RMS deviation of the corrected image. If 2D-array units are
    metres (default convention for STM topography), the return is in metres.
    """
    try:
        arr, meta = _load_array(path)
    except Exception as e:
        return f"plane_subtract load failed: {type(e).__name__}: {e}"
    if arr.ndim != 2:
        return f"plane_subtract requires 2D image, got shape {arr.shape}"

    h, w = arr.shape
    yy, xx = np.indices(arr.shape)
    if order == 1:
        # z = a*x + b*y + c
        A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(arr.size)])
    elif order == 2:
        # z = a*x + b*y + c + d*x^2 + e*y^2 + f*xy
        A = np.column_stack([
            xx.ravel(), yy.ravel(), np.ones(arr.size),
            xx.ravel() ** 2, yy.ravel() ** 2, (xx * yy).ravel(),
        ])
    else:
        return f"plane_subtract order must be 1 or 2, got {order}"

    coefs, *_ = np.linalg.lstsq(A, arr.ravel(), rcond=None)
    bg = (A @ coefs).reshape(arr.shape)
    corrected = arr - bg
    rms = float(corrected.std())
    out = (
        f"plane_subtract order={order} on {path} (shape {arr.shape})\n"
        f"  coefficients: {[f'{c:.4g}' for c in coefs]}\n"
        f"  RMS roughness after subtraction: {rms:.4g} (units of input array)"
    )
    # A step is not roughness. Without this the headline number of a report can
    # be 4× the real surface roughness and read as a property of the material
    # (2026-07-28 audit — see _structure_dominance).
    out += _structure_note(
        corrected,
        what=("上面这个 RMS 主要是台阶高度，不是表面粗糙度——"
              f"该表面的局部粗糙度约 {_structure_dominance(corrected)[1]:.4g}。"))
    # The fitted slopes are equally affected: a step biases the least-squares
    # plane along the direction it runs. On the audited frame the x slope was
    # right to 0.1 % while the y slope came out 13× the true tilt, because the
    # 240 pm step forced the plane to lean in y.
    if _structure_dominance(corrected)[2] >= _STRUCTURE_RATIO:
        out += ("\n    同理，上面的拟合斜率也被台阶带偏了——台阶跨越的方向上，"
                "这个斜率不是样品倾斜。")
    return out


# ─────────────────────────────────────────────────────────────────────
# Shared: is this residual dominated by LARGE-SCALE STRUCTURE?
# ─────────────────────────────────────────────────────────────────────

#: Above this ratio the plane-subtracted residual is dominated by structure a
#: first-order plane cannot remove (a step / terraces), not by local texture.
#: Measured separation on synthetic ground truth (2026-07-28 audit): flat
#: surfaces, gentle curvature and real defect populations all sit at 1.00–1.19;
#: one step gives 3.19 (any orientation), two steps 2.43. 1.8 is the gap.
_STRUCTURE_RATIO = 1.8
_TILE_PX = 32


def _structure_dominance(flat: "np.ndarray", tile: int = _TILE_PX):
    """(global_sigma, local_sigma, ratio) for a plane-subtracted image.

    ``local_sigma`` is the MEDIAN of per-tile standard deviations: most tiles see
    only local texture (lattice + noise), so the median is immune to the few
    tiles a step crosses. ``global_sigma`` is the whole-frame std. Their ratio
    says how much of the height spread is large-scale structure.

    Why this exists — the 2026-07-28 artefact audit. On a synthetic surface whose
    ground truth is *lattice + noise + ONE 240 pm step and ZERO defects*:

      * ``plane_subtract`` reported "RMS roughness 63.8 pm" (arithmetically
        exact) and the report published "a moderately rough surface (RMS
        63.8 pm)". The real surface roughness is 15.4 pm — the other 4× is the
        step, i.e. a structural feature reported as roughness.
      * ``detect_defects`` thresholded at ±2σ of that same step-inflated σ and
        found "27 bright protrusions, 82 dark spots". Every one of the 109 lies
        in y = 128…157, a 30-px band straddling the step at y = 150. They are one
        step edge, chopped up by the lattice crossing the threshold. The report
        published them as "dark spots outnumber bright protrusions 3:1 …
        consistent with depressions (e.g. vacancies or pits)".

    Neither tool was arithmetically wrong. Both answered a question whose
    precondition (the residual is local texture) had silently failed, and nothing
    in either output said so. Note NOT a robust-statistics problem: MAD gives
    68.8 pm here, essentially the same as σ, because two terraces make the height
    distribution bimodal rather than heavy-tailed.
    """
    import numpy as np

    g = float(flat.std())
    h, w = flat.shape
    ny, nx = h // tile, w // tile
    if ny < 2 or nx < 2:
        return g, g, 1.0
    tiles = [flat[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile].std()
             for i in range(ny) for j in range(nx)]
    local = float(np.median(tiles))
    ratio = g / local if local > 0 else 1.0
    return g, local, ratio


def _structure_note(flat, *, what: str) -> str:
    """One operator-facing line when large-scale structure dominates, else ''."""
    g, local, ratio = _structure_dominance(flat)
    if ratio < _STRUCTURE_RATIO:
        return ""
    return (
        f"\n  ⚠ 这幅图的高度起伏由**大尺度结构**主导（整幅 σ={g:.4g}，"
        f"局部纹理 σ={local:.4g}，比值 {ratio:.1f}×）——一阶平面扣不掉台阶/多层露台。"
        f"\n    {what}"
        f"\n    先按露台分别展平（或逐行/逐列中值展平）再算，才是这个表面的真实"
        f"局部粗糙度；台阶高度应当作为**结构特征单独报告**，不要计进粗糙度或缺陷。"
    )


# ─────────────────────────────────────────────────────────────────────
# 4. detect_defects
# ─────────────────────────────────────────────────────────────────────

@tool("detect_defects")
def detect_defects(path: str, min_size_px: int = 5, sigma_threshold: float = 2.0) -> str:
    """Detect dark / bright defects via threshold + connected-component labeling.

    Args:
        path: Absolute path to the scan file (.npy / .sxm).
        min_size_px: Minimum connected-component size to count. Default 5.
        sigma_threshold: Pixels with value > mean+σ*std are bright; < mean-σ*std are dark.
                         Default σ=2.0.

    Returns counts of dark + bright defects, mean component size, and largest size.
    """
    try:
        arr, meta = _load_array(path)
    except Exception as e:
        return f"detect_defects load failed: {type(e).__name__}: {e}"
    if arr.ndim != 2:
        return f"detect_defects requires 2D image, got shape {arr.shape}"

    try:
        from scipy import ndimage
    except ImportError:
        return "detect_defects: scipy.ndimage unavailable"

    # Plane-subtract first to avoid tilt-induced false positives
    yy, xx = np.indices(arr.shape)
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(arr.size)])
    coefs, *_ = np.linalg.lstsq(A, arr.ravel(), rcond=None)
    flat = arr - (A @ coefs).reshape(arr.shape)
    sigma = flat.std()
    mu = flat.mean()
    high = flat > mu + sigma_threshold * sigma
    low = flat < mu - sigma_threshold * sigma

    bright_lbl, n_bright = ndimage.label(high)
    dark_lbl, n_dark = ndimage.label(low)

    # Filter by size
    bright_sizes = ndimage.sum(high, bright_lbl, range(1, n_bright + 1)) if n_bright > 0 else []
    dark_sizes = ndimage.sum(low, dark_lbl, range(1, n_dark + 1)) if n_dark > 0 else []
    bright_kept = [s for s in bright_sizes if s >= min_size_px]
    dark_kept = [s for s in dark_sizes if s >= min_size_px]

    mean_size = float(np.mean(bright_kept + dark_kept)) if (bright_kept or dark_kept) else 0.0
    max_size = float(max(bright_kept + dark_kept)) if (bright_kept or dark_kept) else 0.0

    out = (
        f"detect_defects on {path} (shape {arr.shape}, σ-threshold {sigma_threshold})\n"
        f"  bright protrusions: {len(bright_kept)} (≥{min_size_px}px), "
        f"dark spots: {len(dark_kept)} (≥{min_size_px}px)\n"
        f"  mean component size: {mean_size:.1f} px, "
        f"largest: {max_size:.0f} px"
    )
    # WHERE they are decides WHAT they are. A real defect population is spread
    # over the frame; one step edge chopped up by the lattice is a narrow band.
    # On the audited frame all 109 "defects" sat in 30 of 256 rows and were
    # published as a 3:1 vacancy excess (see _structure_dominance).
    hit = high | low
    if hit.any() and (bright_kept or dark_kept):
        rows = np.where(np.any(hit, axis=1))[0]
        cols = np.where(np.any(hit, axis=0))[0]
        row_span = int(rows.max() - rows.min() + 1)
        col_span = int(cols.max() - cols.min() + 1)
        frac = hit.mean() * 100.0
        out += (f"\n  空间分布：行 {int(rows.min())}–{int(rows.max())} "
                f"（{row_span}/{arr.shape[0]}），列 {int(cols.min())}–{int(cols.max())} "
                f"（{col_span}/{arr.shape[1]}），覆盖全图 {frac:.1f}% 像素")
        # min(row_span, col_span) — a band is narrow in ONE direction whichever
        # way it runs; a real defect population is wide in both. Measured on
        # ground truth: steps (horizontal / vertical) give 41–46 of 256; 6, 20
        # and 40 scattered vacancies all give 253–254. The gap is enormous, so
        # n//4 is a safe cut that stays direction-agnostic.
        if min(row_span, col_span) <= max(8, min(arr.shape) // 4):
            out += ("\n  ⚠ 全部检出都挤在一条窄带里——这是**一条边缘/台阶**被阈值切碎的"
                    "样子，不是分布在表面上的点缺陷。不要按缺陷密度或明暗比解读。")
    out += _structure_note(
        flat,
        what=("σ 是被台阶撑大的，±nσ 等值线画出来的是台阶边缘，"
              "不是点缺陷——上面这些计数不能当缺陷用。"))
    return out


# ─────────────────────────────────────────────────────────────────────
# 5. fit_sts_peaks
# ─────────────────────────────────────────────────────────────────────

@tool("fit_sts_peaks")
def fit_sts_peaks(path: str, n_peaks: int = 3) -> str:
    """Fit Gaussian peaks to a dI/dV curve.

    Args:
        path: Absolute path to .dat / .npy / .txt with two columns (bias, dIdV).
        n_peaks: Number of Gaussians to fit. Default 3.

    Returns fitted peak centres (V) and amplitudes.
    """
    try:
        arr, meta = _load_array(path)
    except Exception as e:
        return f"fit_sts_peaks load failed: {type(e).__name__}: {e}"
    if arr.ndim != 2 or arr.shape[1] < 2:
        return f"fit_sts_peaks expects (N, 2+) bias/didv columns, got shape {arr.shape}"
    bias = arr[:, 0]
    didv = arr[:, 1]

    try:
        from scipy.signal import find_peaks
    except ImportError:
        return "fit_sts_peaks: scipy.signal unavailable"

    # Quick peak finding (no curve_fit — fast + robust enough for top-N)
    peak_idx, props = find_peaks(didv, height=didv.max() * 0.05)
    if len(peak_idx) == 0:
        return f"fit_sts_peaks on {path}: no peaks above 5% of max"
    # Top-N by height
    heights = props["peak_heights"]
    if len(peak_idx) > n_peaks:
        order = np.argsort(-heights)[:n_peaks]
        peak_idx = peak_idx[order]
        heights = heights[order]
    # Sort by bias for readability
    order = np.argsort(bias[peak_idx])
    peak_idx = peak_idx[order]
    heights = heights[order]

    lines = [f"STS peaks in {path} (top {n_peaks}):"]
    for v, h in zip(bias[peak_idx], heights):
        lines.append(f"  bias = {v:+.3f} V, dI/dV = {h:.4g}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# 6. run_numpy_snippet (RestrictedPython sandbox — ALLOW-LIST model)
# ─────────────────────────────────────────────────────────────────────
#
# SECURITY MODEL
# ----------------------------------------------------------------------
# This snippet runner executes LLM-authored Python in-process. The previous
# implementation used a *deny-list* (`_NP_IO_DENY`) over the full `np` module,
# which left a large residual attack surface:
#   * `scipy.fft` was exposed as a whole module (→ `scipy.fft.os`-style reach).
#   * `np.linalg` / `np.random` / `np.core` / `np.fft` … were all reachable as
#     full submodules; the deny-set only blocked a handful of top-level names,
#     so e.g. `np.core` (→ `np.core.multiarray`, `numpy._core`) stayed open.
#   * `_safe_getattr` filtered a single hop by `name in deny_set`, never
#     recursing, so once you reached a submodule its attrs were unguarded by the
#     name set (RestrictedPython's `_getattr_` does re-fire per hop, but the
#     deny-set didn't list submodule-internal escape names like `__loader__`).
#   * RestrictedPython's `safe_builtins` still ships `setattr` / `delattr` /
#     `__build_class__`, none of which a math snippet needs.
#
# The model below is a strict ALLOW-LIST:
#   1. `np` is NOT the real module — it is a frozen proxy (`_NumpyProxy`) that
#      exposes ONLY an explicit set of pure-math functions/constants, plus two
#      sub-proxies `np.fft` and `np.linalg` (again whitelisted function-by-
#      function). No file IO, no `ctypes`, no `lib`, no `core`, no `random`-RNG
#      reach to `os`, no `np.load`/`np.save`/`fromfile`/`memmap`/`genfromtxt`.
#   2. The guarded `_getattr_` (`_sandbox_getattr`) denies EVERY underscore
#      name (kills `__class__` / `__globals__` / `__builtins__` / `__reduce__`
#      dunder chains) and, for ndarray instances, allows only an explicit set of
#      safe array methods — `.tofile` / `.dump` / `.dumps` / `.tobytes` /
#      `.ctypes` / `.view` / `.setflags` / `.base` / `.data` are all denied.
#   3. `__builtins__` is a hardened copy of `safe_builtins` with `setattr`,
#      `delattr`, and `__build_class__` removed and a few safe math/collection
#      builtins (`min`/`max`/`sum`/`list`/`dict`/`set`/`enumerate`/`all`/`any`)
#      added. There is no `getattr`, `open`, `eval`, `exec`, `compile`, or
#      `__import__` anywhere in scope.
#   4. `npy_load(path)` stays the only IO primitive: `.npy` only,
#      `allow_pickle=False`, confined to the project data-dir allow-list.
#
# Net effect: the snippet cannot import modules, touch the filesystem (read or
# write), escape via dunder attribute chains, or call arbitrary code. Anything
# not on the allow-list raises rather than silently exposing a new surface.

# Pure-math top-level numpy functions/constants safe to expose. Deliberately
# EXCLUDES every IO / pickle / ctypes / module-loader name.
_NP_SAFE_FUNCS = (
    # array creation / shape
    "array", "asarray", "ascontiguousarray", "zeros", "zeros_like", "ones",
    "ones_like", "empty", "empty_like", "full", "full_like", "arange",
    "linspace", "logspace", "geomspace", "eye", "identity", "diag", "diagflat",
    "indices", "meshgrid", "fromfunction",
    "reshape", "ravel", "transpose", "moveaxis", "swapaxes", "concatenate",
    "stack", "column_stack", "row_stack", "vstack", "hstack", "dstack",
    "split", "array_split", "squeeze", "expand_dims", "atleast_1d",
    "atleast_2d", "atleast_3d", "broadcast_to", "broadcast_arrays",
    "where", "clip", "sort", "argsort", "lexsort", "partition", "argpartition",
    "unique", "flip", "fliplr", "flipud", "roll", "rot90", "tile", "repeat",
    "take", "put", "compress", "extract", "nonzero", "flatnonzero", "searchsorted",
    "pad",
    # reductions / stats
    "sum", "mean", "std", "var", "min", "max", "amin", "amax", "ptp",
    "argmin", "argmax", "median", "average", "percentile", "quantile",
    "cumsum", "cumprod", "prod", "nanmean", "nanstd", "nanvar", "nansum",
    "nanmin", "nanmax", "nanmedian", "nanpercentile", "histogram", "histogram2d",
    "histogramdd", "bincount", "digitize", "diff", "ediff1d", "gradient",
    "trapezoid", "corrcoef", "cov", "count_nonzero",
    # elementwise math
    "abs", "absolute", "fabs", "sqrt", "cbrt", "square", "exp", "exp2",
    "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan", "arcsin",
    "arccos", "arctan", "arctan2", "hypot", "sinh", "cosh", "tanh", "arcsinh",
    "arccosh", "arctanh", "floor", "ceil", "trunc", "rint", "round", "around",
    "sign", "copysign", "power", "float_power", "mod", "fmod", "remainder",
    "reciprocal", "real", "imag", "conj", "conjugate", "angle", "deg2rad",
    "rad2deg", "degrees", "radians", "real_if_close", "unwrap",
    # linear-algebra-ish top level
    "add", "subtract", "multiply", "divide", "true_divide", "floor_divide",
    "dot", "vdot", "inner", "outer", "cross", "matmul", "tensordot",
    "kron", "trace", "einsum",
    # logic / comparison
    "isnan", "isinf", "isfinite", "isnat", "nan_to_num", "allclose", "isclose",
    "array_equal", "array_equiv", "maximum", "minimum", "fmax", "fmin",
    "logical_and", "logical_or", "logical_not", "logical_xor", "greater",
    "greater_equal", "less", "less_equal", "equal", "not_equal", "any", "all",
    "isin", "in1d", "intersect1d", "union1d", "setdiff1d",
    # dtypes (constructors only — harmless)
    "float64", "float32", "float16", "int64", "int32", "int16", "int8",
    "uint64", "uint32", "uint16", "uint8", "complex128", "complex64",
    "bool_", "intp", "dtype",
    # constants
    "pi", "e", "euler_gamma", "inf", "nan", "newaxis", "ndarray",
)

# numpy.fft sub-namespace — pure transforms, no IO.
_NP_FFT_SAFE = (
    "fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "rfft", "irfft",
    "rfft2", "irfft2", "rfftn", "irfftn", "hfft", "ihfft",
    "fftshift", "ifftshift", "fftfreq", "rfftfreq",
)

# numpy.linalg sub-namespace — pure linear algebra, no IO.
_NP_LINALG_SAFE = (
    "inv", "pinv", "solve", "lstsq", "svd", "eig", "eigh", "eigvals",
    "eigvalsh", "det", "slogdet", "norm", "qr", "cholesky", "matrix_rank",
    "matrix_power", "cond", "tensorsolve", "tensorinv", "multi_dot",
)

# Safe ndarray instance methods. Anything touching IO/buffer/casting-escape
# (tofile, dump, dumps, tobytes, ctypes, view, setflags, base, data, getfield,
# setfield, resize, byteswap, item/itemset on raw memory…) is intentionally
# absent → denied by the guarded getattr.
_NDARRAY_SAFE_ATTRS = frozenset({
    # read-only descriptors
    "shape", "dtype", "ndim", "size", "T", "real", "imag", "flat", "nbytes",
    "itemsize", "strides", "mT",
    # value-returning math methods
    "mean", "std", "var", "sum", "prod", "min", "max", "ptp", "argmin",
    "argmax", "cumsum", "cumprod", "trace", "dot",
    "reshape", "ravel", "flatten", "transpose", "swapaxes", "squeeze",
    "astype", "copy", "clip", "round", "conj", "conjugate", "sort", "argsort",
    "take", "repeat", "diagonal", "nonzero", "all", "any", "tolist",
    "fill", "item",
})


def _build_numpy_proxy():
    """Construct the frozen allow-list `np` proxy and its sub-namespaces.

    Returns a `_NumpyProxy` whose attribute access is restricted to the
    whitelisted names. Building it once at import time avoids per-call cost.
    """

    class _FrozenNamespace:
        """A read-only namespace exposing only an explicit dict of names."""

        __slots__ = ("_d", "_label")

        def __init__(self, label: str, d: dict):
            object.__setattr__(self, "_label", label)
            object.__setattr__(self, "_d", d)

        def __getattr__(self, name):  # only called for names not in __slots__
            d = object.__getattribute__(self, "_d")
            if name in d:
                return d[name]
            raise AttributeError(
                f"{object.__getattribute__(self, '_label')} has no allowed "
                f"attribute {name!r}"
            )

        def __setattr__(self, name, value):
            raise AttributeError("sandbox numpy namespace is read-only")

        def __delattr__(self, name):
            raise AttributeError("sandbox numpy namespace is read-only")

        def __repr__(self):
            return f"<sandbox {object.__getattribute__(self, '_label')}>"

    def _collect(module, names):
        out = {}
        for n in names:
            if hasattr(module, n):
                out[n] = getattr(module, n)
        return out

    np_names = _collect(np, _NP_SAFE_FUNCS)
    np_names["fft"] = _FrozenNamespace("numpy.fft", _collect(np.fft, _NP_FFT_SAFE))
    np_names["linalg"] = _FrozenNamespace(
        "numpy.linalg", _collect(np.linalg, _NP_LINALG_SAFE)
    )
    return _FrozenNamespace("numpy", np_names)


# Built once; the proxy is immutable so sharing it across calls is safe.
_NP_PROXY = _build_numpy_proxy()


# Modules the sandbox's guarded __import__ may resolve. numpy 2.x lazily imports
# its own internal helpers the first time a math method runs (e.g. ndarray.sum()
# pulls in numpy._core._methods); with a bare-dict __builtins__ that omits
# __import__ this surfaces as `KeyError: '__import__'`. We therefore install a
# *guarded* __import__ that resolves ONLY numpy/scipy submodules and the handful
# of pure stdlib helpers numpy reaches for — and refuses everything else (os,
# sys, subprocess, builtins, …). The snippet itself can never reference the
# bare name `__import__` (RestrictedPython makes it a SyntaxError) nor write an
# `import os` statement (it routes through this guard and is rejected), so this
# only unblocks numpy's internal machinery, not LLM-authored imports.
_IMPORT_OK_PREFIXES = ("numpy", "scipy")
_IMPORT_OK_EXACT = frozenset({
    "functools", "warnings", "operator", "math", "itertools", "collections",
    "collections.abc", "contextlib", "numbers", "re", "copy", "weakref",
    "types", "_io", "_pocketfft", "_helper", "pickle",
})


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = (name or "").split(".")[0]
    allowed = (
        name in _IMPORT_OK_EXACT
        or top in _IMPORT_OK_EXACT
        or any(name == p or name.startswith(p + ".") for p in _IMPORT_OK_PREFIXES)
        or top in _IMPORT_OK_PREFIXES
    )
    if not allowed:
        raise ImportError(f"import of {name!r} is not allowed in the sandbox")

    # `from numpy import save` / `from scipy import ...` (a fromlist on a numpy/
    # scipy module) would bind the REAL function straight into the snippet,
    # bypassing the restricted proxy. numpy's own internals also use this form
    # (e.g. numpy.ma → `from numpy import array`), so we only block it when the
    # *calling frame* is NOT itself a numpy/scipy module. The snippet's frame
    # always carries the sandbox __name__ (never "numpy*"/"scipy*").
    if fromlist:
        caller = ""
        if isinstance(globals, dict):
            caller = str(globals.get("__name__", ""))
        caller_top = caller.split(".")[0]
        if caller_top not in _IMPORT_OK_PREFIXES:
            raise ImportError(
                "`from <module> import name` is not allowed in the sandbox"
            )

    import importlib
    return importlib.import_module(name)


def _warm_numpy_paths() -> None:
    """Pre-import numpy's internal math submodules so the guarded import never
    has to cold-import them inside an active snippet frame."""
    try:
        _a = np.arange(4.0)
        _a.sum(); _a.mean(); _a.std(); _a.argsort()
        np.fft.fft2(np.zeros((2, 2)))
        np.linalg.norm(np.ones(2))
        np.histogram(_a, bins=2)
    except Exception:  # pragma: no cover - warm-up is best-effort
        pass


_warm_numpy_paths()


def _build_sandbox_builtins():
    """Hardened copy of RestrictedPython.safe_builtins.

    Removes write/escape primitives (`setattr`, `delattr`, `__build_class__`,
    `open`, `eval`, `exec`, `compile`, `getattr`) and adds back safe
    math/collection builtins the deny-list `safe_builtins` happens to omit.

    `__import__` is installed as the *guarded* `_guarded_import` — present so
    numpy's internal lazy imports don't `KeyError`, but it only resolves
    numpy/scipy/stdlib-math helpers and refuses everything else. The snippet
    cannot reference the bare name (SyntaxError) nor `import os` (rejected).
    """
    from RestrictedPython import safe_builtins

    b = dict(safe_builtins)
    for bad in ("setattr", "delattr", "__build_class__", "open", "eval",
                "exec", "compile", "getattr", "__import__"):
        b.pop(bad, None)
    # numpy 2.x internals need __import__ present in the frame builtins.
    b["__import__"] = _guarded_import
    # Add safe extras (all pure / no IO / no module reach).
    import builtins as _py
    for extra in ("min", "max", "sum", "list", "dict", "set", "frozenset",
                  "enumerate", "all", "any", "map", "filter", "reversed",
                  "abs", "divmod", "format"):
        if hasattr(_py, extra):
            b[extra] = getattr(_py, extra)
    return b


def _sandbox_getattr(obj, name, *default):
    """Guarded attribute access for the sandbox.

    Hard rules (allow-list):
      * Any underscore-prefixed name is denied (kills every dunder chain:
        __class__, __globals__, __builtins__, __reduce__, __subclasses__, …).
      * For numpy ndarray instances, only `_NDARRAY_SAFE_ATTRS` is allowed.
      * For the frozen numpy proxy / sub-namespaces, their own `__getattr__`
        already enforces the function allow-list — we just forward.
      * Everything else (arbitrary module/object attribute reads) is denied.
    """
    if not isinstance(name, str) or name.startswith("_"):
        raise AttributeError(f"access to {name!r} is denied in the sandbox")

    # ndarray: explicit method/descriptor allow-list.
    if isinstance(obj, np.ndarray):
        if name in _NDARRAY_SAFE_ATTRS:
            return getattr(obj, name)
        raise AttributeError(
            f"ndarray attribute {name!r} is not allowed in the sandbox"
        )

    # Frozen numpy namespaces enforce their own allow-list via __getattr__.
    # `type(obj).__name__` avoids importing the inner class.
    if type(obj).__name__ == "_FrozenNamespace":
        return getattr(obj, name)  # raises AttributeError if not whitelisted

    # numpy scalar / generic results of math ops: allow the same safe descriptor
    # set so chained expressions like `arr.mean().item()` keep working.
    if isinstance(obj, np.generic):
        if name in _NDARRAY_SAFE_ATTRS:
            return getattr(obj, name)
        raise AttributeError(
            f"numpy scalar attribute {name!r} is not allowed in the sandbox"
        )

    # Plain Python containers / numbers / strings: allow normal (non-underscore)
    # method access — these cannot reach the filesystem or import machinery.
    if isinstance(obj, (list, dict, tuple, set, frozenset, str, bytes,
                        int, float, complex, bool, type(None))):
        return getattr(obj, name)

    # Anything else (e.g. a stray module if one ever leaked in) is denied.
    raise AttributeError(
        f"attribute access on {type(obj).__name__} is denied in the sandbox"
    )


def _allowed_npy_roots() -> list:
    from mast._runtime_paths import project_root
    root = project_root()
    return [root / "experiments", root / "session", root / "data",
            root / "stm-datasets", Path.home() / "experiments"]


def _npy_load(p: str) -> np.ndarray:
    """Load a `.npy` file: `.npy` only, no pickle, project data dirs only."""
    rp = Path(p).resolve()
    if rp.suffix.lower() != ".npy":
        raise ValueError("npy_load only loads .npy files")
    roots = [r.resolve() for r in _allowed_npy_roots()]
    if not any(str(rp).startswith(str(r)) for r in roots):
        raise ValueError("npy_load path outside the allowed data directories")
    return np.load(rp, allow_pickle=False)


def _maybe_set_memory_rlimit() -> None:
    """Best-effort address-space cap for the current process (POSIX only).

    On POSIX a soft RLIMIT_AS makes a runaway allocation fail fast with a
    catchable MemoryError instead of swapping the host to death. We never RAISE
    the limit (only lower the soft cap, and only if it is currently unlimited or
    higher than our target), and we never touch the hard limit — so we cannot
    relax an existing tighter sandbox. The `resource` module does not exist on
    Windows; there the wall-clock timeout is the sole guard (a 1e18-element
    allocation either MemoryErrors immediately or is killed by the timeout).
    """
    try:
        import resource  # POSIX-only; ImportError on Windows
    except Exception:  # pragma: no cover - Windows path
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = _SNIPPET_MEM_LIMIT_BYTES
        if hard != resource.RLIM_INFINITY:
            target = min(target, hard)
        if soft == resource.RLIM_INFINITY or soft > target:
            resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except Exception:  # pragma: no cover - defensive; never block on rlimit
        pass


def _exec_snippet_bounded(compiled, safe_globals: dict, local: dict) -> str | None:
    """Run the compiled snippet under a wall-clock timeout in a worker thread.

    Returns:
        None             on success (results land in ``local``)
        an error string  on a sandbox runtime error or a timeout.

    A worker thread cannot be force-killed in CPython, so on timeout we leave it
    as a DAEMON thread (it cannot block process exit) and return a timeout error
    to the caller — the agent is freed immediately rather than hanging. The
    POSIX address-space rlimit set inside the worker bounds a runaway allocation
    so a timed-out thread cannot keep growing memory unbounded after we return.
    NB: this lives in tools.py (allowed a controlled timeout), NOT in any
    agents/**/graph.py (which the hooks forbid from blocking) — the agent's
    graph node simply gets the bounded string result back.
    """
    result: dict = {"error": None, "done": False}

    def _worker() -> None:
        _maybe_set_memory_rlimit()
        try:
            exec(compiled, safe_globals, local)  # noqa: S102 - RestrictedPython sandbox
        except Exception as e:  # noqa: BLE001
            result["error"] = f"sandbox runtime error: {type(e).__name__}: {e}"
        finally:
            result["done"] = True

    t = threading.Thread(target=_worker, name="numpy-snippet", daemon=True)
    t.start()
    t.join(_SNIPPET_TIMEOUT_S)
    if t.is_alive():
        # Timed out: the daemon thread is abandoned (cannot be killed) but cannot
        # block process exit; the rlimit caps its memory growth. Free the agent.
        return (
            f"sandbox timeout: snippet exceeded {_SNIPPET_TIMEOUT_S:g}s wall-clock "
            "limit (possible infinite loop or oversized computation)"
        )
    return result["error"]


@tool("run_numpy_snippet")
def run_numpy_snippet(code: str) -> str:
    """Run a short numpy snippet in a RestrictedPython allow-list sandbox.

    Args:
        code: Python source. Allowed names: `np` — a restricted numpy proxy
              exposing only pure-math functions (array creation/shape/stats/
              elementwise math) plus `np.fft.*` and `np.linalg.*`; and
              `npy_load(path)` to load a `.npy` file from the project data dirs.
              No file I/O (no np.save/np.load/loadtxt/fromfile/memmap),
              no imports, no subprocess, no dunder/attribute escapes.
              Bounded resources: a 5 s wall-clock timeout (kills infinite loops)
              and a best-effort 1 GiB address-space cap on POSIX (caps runaway
              allocations) — see _exec_snippet_bounded.

    Returns: a repr of the snippet's last top-level binding, or an error.

    Example::
        a = npy_load('/data/scan.npy')
        m = a.mean()
        m
    """
    try:
        from RestrictedPython import compile_restricted
        from RestrictedPython.Eval import default_guarded_getitem
        from RestrictedPython.Guards import (
            guarded_iter_unpack_sequence,
            guarded_unpack_sequence,
        )
    except ImportError:
        return "RestrictedPython not installed"

    try:
        compiled = compile_restricted(code, filename="<sandbox>", mode="exec")
    except Exception as e:
        return f"sandbox compile error: {type(e).__name__}: {e}"

    safe_globals: dict = {
        "__builtins__": _build_sandbox_builtins(),
        "_getitem_": default_guarded_getitem,
        "_getiter_": iter,
        "_getattr_": _sandbox_getattr,
        "_write_": lambda obj: obj,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "np": _NP_PROXY,
        "numpy": _NP_PROXY,
        "npy_load": _npy_load,
    }

    local: dict = {}
    # Run under a wall-clock timeout + best-effort memory rlimit so an LLM-authored
    # infinite loop / oversized allocation cannot hang or OOM the agent process.
    err = _exec_snippet_bounded(compiled, safe_globals, local)
    if err is not None:
        return err

    # Capture last expression result if any
    result_repr = "(no return value)"
    for k, v in local.items():
        if not k.startswith("_"):
            try:
                result_repr = f"{k} = {v!r}"
            except Exception:
                pass
    return _truncate(f"sandbox ok. Last binding: {result_repr}")


# ─────────────────────────────────────────────────────────────────────
# 7. mosaic_scans — spatial stitch of many .sxm into one overview canvas
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 7b. Publication-ready single figures — the BASELINE deliverable.
#
#     The operator's minimum bar for this agent (2026-07-27): "把图以正确的
#     坐标轴、标签画出来，然后提供给论文 agent 写实验报告。用户或者编排 agent
#     不要求分析的时候可以不分析。" Rendering a figure is MANDATORY; analysis
#     is on request.
#
#     Before this, the agent had no way to plot a single scan and NO way at all
#     to plot a spectrum. The only illustration tool (compose_montage) draws no
#     axes, has an unlabelled colorbar, and takes its field of view from a
#     hand-typed argument — which produced a 20 cm scale bar on a 100 nm image
#     and still answered "ok". Here the FOV is PARSED from the file header, so
#     it cannot be typed wrong.
#
#     The plotting itself reuses mast.data.visualization, which was written
#     correctly and then never called by anything.
# ─────────────────────────────────────────────────────────────────────

def _figure_out_dir():
    """Where rendered figures go — the ONE directory paper_writing looks in."""
    from mast.agents._shared.data_paths import figures_dir
    return figures_dir()


def _cjk_font():
    """Per-artist CJK font, or None. NEVER touches global rcParams — doing that
    turns every English figure elsewhere into tofu (io/exp_map.py:583 note)."""
    try:
        from mast.io.exp_map import _cjk_fontprops
        return _cjk_fontprops()
    except Exception:  # noqa: BLE001
        return None


def _resolve_src(path: str, scan_id: str, tool: str) -> str:
    """`path` or `scan_id` → a real path, or an "<tool> failed: …" message.

    Mirrors load_scan's resolution so every figure tool accepts the same two
    ways of naming a measurement.
    """
    src = path or ""
    if not src and scan_id:
        from mast.core.scan_registry import resolve_scan_id
        src = resolve_scan_id(scan_id) or ""
        if not src:
            return (f"{tool} failed: 未知 scan_id {scan_id!r}。"
                    "先调用 list_scan_dir / glob_scans 查看已记录的扫描。")
    if not src:
        return f"{tool} failed: 需要提供 path 或 scan_id 之一。"
    return src


def _scan_fov_m(meta: dict) -> "tuple[float, float] | None":
    """Physical scan size (w, h) in metres from a .sxm header, or None.

    Uses the same parser the mosaic builder trusts. Returning None is a real
    answer — the caller then labels the axes in pixels AND says so, rather than
    inventing a scale.
    """
    try:
        from mast.io.mosaic import parse_xy_meta
        xy = parse_xy_meta(meta.get("header") or meta or {})
        if xy and xy.get("w") and xy.get("h"):
            return float(xy["w"]), float(xy["h"])
    except Exception:  # noqa: BLE001
        pass
    return None


# ── 出图工具的图像回路（2026-08-19）────────────────────────────────────
#
# 在此之前，**这个 agent 从来没有看见过它自己画的任何一张图**。
# `SkillImageMiddleware` 挂在 DP 上，但 IMAGES_KEY 全仓只有一处写点
# （`_shared/skill_adapter.py:1024-1026`），而 DP 的工具一个都不走 skill
# adapter —— 它们返回裸字符串。中间件在那儿等了很久，一张图都没收到。
#
# 这不只是「模型少看了点东西」：出图是这个 agent 的**基线交付**，而它无法判断
# 自己刚交付的东西对不对。现在它能看见，也就能发现「色标压死了」「画错通道了」
# 这类只有看一眼才知道的问题。

_IMG_SUFFIX = (".png", ".jpg", ".jpeg")


def _figures_in(text) -> list[str]:
    """从工具返回里挖出**确实存在的**图片路径。

    判据是「文件真的在盘上」而不是「看起来像路径」。后者会把报错信息里提到的
    不存在的路径也挂上去，那时模型收到的是一个空图像通道 —— 生产方在记、消费方
    读不到，正是这个仓踩过的静默失败形状。

    两种返回形态都覆盖：``Saved figure: <path>``（plot_scan/plot_spectrum）和
    ``xxx ok: {json}``（montage/diff/mosaic）。不去记每个工具的字段名 —— 那份
    对应关系没人维护，而「文件存在」这条判据永远不会过期。
    """
    import json as _json
    import re as _re

    raw = str(text or "")
    cand: list[str] = []
    for line in raw.splitlines():
        m = _re.match(r"\s*(?:Saved figure|Saved|图已保存)\s*[:：]\s*(.+?)\s*$", line)
        if m:
            cand.append(m.group(1))
    brace = raw.find("{")
    if brace >= 0:
        try:
            payload = _json.loads(raw[brace:])
        except Exception:  # noqa: BLE001 — 不是 JSON 就算了
            payload = None

        def _walk(v):
            if isinstance(v, str):
                cand.append(v)
            elif isinstance(v, dict):
                for x in v.values():
                    _walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    _walk(x)

        _walk(payload)

    seen: set[str] = set()
    keep: list[str] = []
    for c in cand:
        c = c.strip().strip('"').strip("'")
        if (c.lower().endswith(_IMG_SUFFIX) and c not in seen
                and Path(c).is_file()):
            seen.add(c)
            keep.append(c)
    return keep


def _with_figures(summary, tool_call_id: str, name: str):
    """把返回里提到的图挂上图像通道。

    ``tool_call_id`` 为空时**原样返回字符串** —— 几十个直调单测
    （``tool.invoke({...})`` 然后对文本断言）因此行为完全不变，而在真实图里
    tool_call_id 总是被框架注入的。把行为改变面收窄到「只有跑在 agent 里时」。
    """
    text = str(summary)
    if not tool_call_id:
        return summary
    figs = _figures_in(text)
    if not figs:
        return summary
    from mast.agents._shared.vision_mw import IMAGES_KEY, MAX_IMAGES_PER_REQUEST

    msg = ToolMessage(content=text, tool_call_id=tool_call_id, name=name,
                      additional_kwargs={IMAGES_KEY: figs[:MAX_IMAGES_PER_REQUEST]})
    return ArtifactToolReturn(text, {"messages": [msg]},
                              tool_call_id=tool_call_id, name=name)


@tool("plot_scan")
def plot_scan(path: str = "", scan_id: str = "", channel: str = "",
              label: str = "",
              tool_call_id: Annotated[str, InjectedToolCallId] = "") -> "ArtifactToolReturn | str":
    """Render ONE scan as a publication-ready figure: real nm axes, axis labels,
    and a colorbar WITH units. This is the baseline deliverable — call it for
    every measurement, no approval needed, before any analysis.

    The field of view is read from the file header; never pass it yourself.
    Give `path` OR `scan_id` (from list_scan_dir / glob_scans /
    get_latest_scan_file). `channel` picks the data channel (default: the
    file's first). `label` names the output file and titles the plot.
    Saves a PNG next to the other figures and returns its path.
    """
    from pathlib import Path

    src = _resolve_src(path, scan_id, "plot_scan")
    if src.startswith("plot_scan failed:"):
        return src
    try:
        arr, meta = _load_array(str(src))
    except Exception as e:  # noqa: BLE001
        return f"plot_scan failed: {type(e).__name__}: {e}"
    if arr.size == 0:
        return f"plot_scan failed: {src} 解析出空数组，没有可画的数据。"
    if arr.ndim != 2:
        return (f"plot_scan failed: 数据是 {arr.ndim} 维 (shape {arr.shape})，"
                "不是二维图像。谱数据请用 plot_spectrum。")
    # A spectrum table is 2-D too — (n_points, n_channels) — so `ndim == 2` is
    # NOT enough. Rendering one as an image produces a few-pixel-wide grey
    # smear that looks like a successful figure; that is exactly the invented
    # output this tool exists to stop. Reject on BOTH signals: a spectroscopy
    # extension, and an aspect ratio no real scan has.
    _ext = Path(str(src)).suffix.lower()
    _lo, _hi = min(arr.shape), max(arr.shape)
    if _ext in (".dat", ".txt", ".csv", ".asc", ".tsv") or _lo <= 8 < _hi:
        return (f"plot_scan failed: {Path(str(src)).name} 看着是谱数据不是图像 "
                f"(shape {arr.shape})。请用 plot_spectrum 画它。"
                "（若这确实是一张极窄的扫描，请先转成 .npy 再传。）")

    fov = _scan_fov_m(meta)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fp = _cjk_font()
        fig, ax = plt.subplots(figsize=(6, 5))
        ch = channel or str(meta.get("channel", "") or "")
        # Colorbar units follow the channel: Z is a height, Current is a current.
        is_current = "curr" in ch.lower() or "i" == ch.lower()
        if fov is not None:
            w_nm, h_nm = fov[0] * 1e9, fov[1] * 1e9
            im = ax.imshow(arr, origin="lower", cmap="afmhot",
                           extent=[0, w_nm, 0, h_nm])
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            scale_note = f"{w_nm:.1f} x {h_nm:.1f} nm"
        else:
            im = ax.imshow(arr, origin="lower", cmap="afmhot")
            # Say it on the figure rather than passing pixels off as nm. Axis
            # text stays ASCII — it goes into a paper, and a CJK glyph here
            # would also need the per-artist font on every label.
            ax.set_xlabel("x (px — no SCAN_RANGE in header)")
            ax.set_ylabel("y (px)")
            scale_note = f"{arr.shape[1]} x {arr.shape[0]} px (物理尺度未知)"
        cbar_label = "I (A)" if is_current else "z (m)"
        fig.colorbar(im, ax=ax, label=cbar_label)
        title = label or Path(str(src)).stem
        if fp is not None:
            ax.set_title(title, fontproperties=fp)
        else:
            ax.set_title(title)
        fig.tight_layout()

        out_dir = _figure_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "".join(c for c in (label or Path(str(src)).stem)
                       if c.isalnum() or c in "-_") or "scan"
        out = out_dir / f"{stem}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.exception("plot_scan render failed")
        return f"plot_scan failed: {type(exc).__name__}: {exc}"
    return _with_figures(
        f"Saved figure: {out}\n"
        f"  channel: {ch or '(default)'}, {scale_note}, colorbar: {cbar_label}\n"
        f"  paper_writing 可用 list_figures() 找到它。",
        tool_call_id, "plot_scan")


@tool("plot_spectrum")
def plot_spectrum(path: str = "", scan_id: str = "", label: str = "",
                  tool_call_id: Annotated[str, InjectedToolCallId] = "") -> "ArtifactToolReturn | str":
    """Render an STS spectrum as a publication-ready figure: bias on x with
    units, current/dI-dV on y with units. This is the baseline deliverable for
    spectroscopy — call it for every measurement, no approval needed.

    Give `path` OR `scan_id` (a Nanonis .dat from AcquireSTS / GridSTS).
    `label` names the output file and titles the plot. Saves a PNG next to the
    other figures and returns its path.
    """
    from pathlib import Path

    src = _resolve_src(path, scan_id, "plot_spectrum")
    if src.startswith("plot_spectrum failed:"):
        return src
    # Reject images up front. Loading a .sxm as a spectrum surfaces numpy's
    # "This file contains pickled (object) data" — true, but useless to an
    # agent, which cannot tell from it that it simply picked the wrong tool.
    _sext = Path(str(src)).suffix.lower()
    if _sext in (".sxm", ".sm4"):
        return (f"plot_spectrum failed: {Path(str(src)).name} 是扫描图像不是谱。"
                "请用 plot_scan 画它。")
    try:
        from mast.data.loaders import load_spectrum_named
        raw, col_names = load_spectrum_named(str(src))
        raw = np.asarray(raw, dtype=float)
    except Exception as e:  # noqa: BLE001
        return (f"plot_spectrum failed: {type(e).__name__}: {e}\n"
                "（形貌图请用 plot_scan；这里要的是谱文件 .dat/.txt/.npy）")
    if raw.size == 0:
        return f"plot_spectrum failed: {src} 没有解析出可画的谱数据。"
    if raw.ndim == 3:
        return ("plot_spectrum failed: 这是网格谱立方 (.3ds)，不是单点谱。"
                "先用 grid_to_spectra 展开，或指定要画的点。")
    # load_spectrum_named returns (n_points, n_cols) + the REAL channel names
    # ("Bias calc (V)", "Current (A)", "LIX 1 omega (A)"). Until 2026-07-28 this
    # used the nameless loader and invented "ch1", "ch2", … — which made the
    # legend useless AND silently disabled the y-axis logic below, because that
    # logic pattern-matches the channel names and the only names it could ever
    # see were the placeholders this function had just made up.
    if raw.ndim == 1:
        volt = np.arange(raw.size, dtype=float)
        arr = raw[np.newaxis, :]
        names: list[str] = [col_names[0] if col_names else "signal"]
        x_label = "点序号"
    else:
        volt = raw[:, 0]
        arr = raw[:, 1:].T
        names = ([str(c) for c in col_names[1:]] if len(col_names) > 1
                 else [f"ch{j + 1}" for j in range(arr.shape[0])])
        # The sweep axis names itself too — a Z-spectroscopy .dat sweeps Z (m),
        # not bias, and labelling that "Bias (V)" is a wrong axis on a figure
        # headed for a manuscript.
        x_label = str(col_names[0]) if col_names else "Bias (V)"
    if arr.size == 0:
        return (f"plot_spectrum failed: {src} 只有一列（{raw.shape}），"
                "没有可对偏压作图的信号列。")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fp = _cjk_font()
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, curve in enumerate(arr):
            nm = names[i] if names and i < len(names) else f"curve {i}"
            ax.plot(np.asarray(volt, dtype=float), curve, label=str(nm))
        # Axis labels now carry text from the FILE (channel names), so they can
        # hold non-ASCII. The CJK font was previously applied to the title only,
        # which was fine while every axis label was a hard-coded ASCII string —
        # the moment one was not, it rendered as tofu boxes.
        ax.set_xlabel(x_label, **({"fontproperties": fp} if fp is not None else {}))
        # Name what is actually on y. Each curve is now checked SEPARATELY: the
        # old test joined every channel name into one string, so a file holding
        # both Current and a lock-in dI/dV (the ordinary STS .dat — that is what
        # the lock-in is FOR) got a single label, and whichever it picked was a
        # wrong axis for the other curve. Mixed channels get the shared unit and
        # no claim about which quantity it is.
        def _kind(n: str) -> str:
            low = str(n).lower()
            if any(k in low for k in ("di/dv", "di-dv", "didv", "lix", "liy",
                                      "lock", "conduct")):
                return "didv"
            return "current" if "current" in low or low.strip() in ("i", "i (a)") else "?"

        kinds = {_kind(n) for n in (names or [])}
        if kinds == {"didv"}:
            y_label = "dI/dV (a.u.)"
        elif kinds == {"current"}:
            y_label = "I (A)"
        else:
            units = {n[n.rfind("(") + 1:n.rfind(")")].strip()
                     for n in (names or []) if "(" in n and ")" in n}
            y_label = f"Signal ({units.pop()})" if len(units) == 1 else "Signal (a.u.)"
        ax.set_ylabel(y_label, **({"fontproperties": fp} if fp is not None else {}))
        ax.axhline(0, lw=0.6, color="0.7")
        ax.axvline(0, lw=0.6, color="0.7")
        if names:
            # Same reason as the axis labels: the legend now shows the file's own
            # channel names, which are no longer guaranteed ASCII.
            if fp is not None:
                ax.legend(prop=fp, fontsize=8)
            else:
                ax.legend(fontsize=8)
        title = label or Path(str(src)).stem
        if fp is not None:
            ax.set_title(title, fontproperties=fp)
        else:
            ax.set_title(title)
        fig.tight_layout()

        out_dir = _figure_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "".join(c for c in (label or Path(str(src)).stem)
                       if c.isalnum() or c in "-_") or "spectrum"
        out = out_dir / f"{stem}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.exception("plot_spectrum render failed")
        return f"plot_spectrum failed: {type(exc).__name__}: {exc}"
    # Describe the figure that was actually drawn — ONE axes, all curves
    # overlaid, and which curve is which. The old return said only
    # "N curve(s), M points, x=Bias (V)", which left the writing agent to guess
    # the layout; in an end-to-end run on 2026-07-28 it guessed a two-panel
    # figure and captioned all three figures 「Top: I(V); bottom: dI/dV」 for
    # single-panel plots. A caption that does not match its figure is the same
    # failure family as a report whose numbers are not real.
    return _with_figures(
        f"Saved figure: {out}\n"
        f"  单幅坐标系，{arr.shape[0]} 条曲线叠加（不是上下分栏）："
            f"{'、'.join(str(n) for n in names)}\n"
            f"  {arr.shape[1]} 点，x={x_label}，y={y_label}\n"
        f"  写图注时按此描述，不要臆造上下分栏。"
        f" paper_writing 可用 list_figures() 找到它。",
        tool_call_id, "plot_spectrum")


@tool("mosaic_scans")
def mosaic_scans(directory: str, channel: str = "Z", recursive: bool = False,
                 line_normalize: bool = True, label: str = "",
                 tool_call_id: Annotated[str, InjectedToolCallId] = "") -> "ArtifactToolReturn | str":
    """Stitch every .sxm scan in a directory into ONE big-canvas overview,
    placing each image by its real xy stage coordinates (scan_offset /
    scan_range). Use to get a single-look map of everything scanned on one
    sample — e.g. after an unattended overnight survey. `channel` picks the data
    channel (default 'Z' topography); `recursive` walks subfolders;
    `line_normalize` flattens per-row banding (recommended for Z); `label` tags
    the output files (use the sample name — a new sample = a new mosaic). Saves a
    PNG + NPY + JSON to artifacts/mosaics/ and returns a summary with paths.
    """
    from pathlib import Path

    from mast.io.mosaic import mosaic_from_dir, save_mosaic

    if not directory or not Path(directory).is_dir():
        return f"error: directory not found: {directory!r}"
    try:
        mosaic = mosaic_from_dir(directory, channel=channel or "Z",
                                 recursive=bool(recursive),
                                 line_normalize=bool(line_normalize))
    except Exception as exc:  # noqa: BLE001
        logger.exception("mosaic_scans failed")
        return f"error: mosaic build failed: {type(exc).__name__}: {exc}"
    if mosaic.get("error"):
        return f"mosaic failed: {mosaic['error']}"
    if not mosaic.get("placed"):
        return "no scans placed (no .sxm with SCAN_OFFSET/SCAN_RANGE found)"
    paths = save_mosaic(mosaic, label=label or "")
    ext = mosaic["extent_m"]
    extra = f" (+npy {paths['npy']})" if paths.get("npy") else ""
    return _with_figures(
        f"Mosaic of {mosaic['placed']}/{mosaic['n_input']} scans (channel "
        f"'{channel or 'Z'}'): canvas {mosaic['canvas_px'][0]}x"
        f"{mosaic['canvas_px'][1]} px covering "
        f"{(ext[1] - ext[0]) * 1e9:.0f}x{(ext[3] - ext[2]) * 1e9:.0f} nm at "
        f"{mosaic['res_m_per_px'] * 1e9:.2f} nm/px. Saved: {paths['png']}{extra}",
        tool_call_id, "mosaic_scans")


# ─────────────────────────────────────────────────────────────────────
# 8. Ported analysis skills (AutoCrop / DiffScans / ComposePanelMontage)
#    Thin @tool bridges so the data_processing agent can call these paper
#    BaseSkills directly — they were registry/workflow-only (orphaned from this
#    agent's toolset, per the 2026-06-26 classification audit). Pure analysis:
#    execute() uses load_image_2d, not a hardware context, so a None context is OK.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 9. Dual-homed morphology skills (FindFlatRegion / AssessClusterRoundness)
#    These analyse a SAVED .sxm FILE (execute() only reads params["scan_path"],
#    never a hardware context) so they belong in data_processing too — not just
#    instrument_control. Routing audit (Haiku+deepseek, 2026-07-01) showed the LLM
#    naturally sends "find a flat region / assess cluster roundness in this scan"
#    to data_processing (it IS file analysis); before this bridge DP lacked them
#    → request_user_action / getter-loop dead-end. Thin @tool, None context OK.
# ─────────────────────────────────────────────────────────────────────

@tool("find_flat_region")
def find_flat_region(scan_path: str, window_fraction: float = 0.2,
                     channel: str = "Z", min_separation_m: MetreSize = 0.0) -> str:
    """Find the flattest (lowest plane-subtracted RMS) sub-window of a SAVED .sxm scan
    and return its centre in instrument coords (metres) — e.g. to pick a clean spot for
    the next scan or a tip-shaper plunge. window_fraction: window side as a fraction of
    the scan extent (0.2 = 20%). min_separation_m: 离已排除点的最小距离,写成带 SI
    前缀的字符串,如 '100n';留 0 = 取窗口边长。
    Offline file analysis; scan_path = .sxm on disk."""
    import json
    from mast.skills.builtins.flat_region import FindFlatRegion
    # 这个 @tool 直接调 `FindFlatRegion().execute()`,**绕过了 wrap_skill** ——
    # 于是 skill 那一侧给米量纲参数的全部防护(声明成字符串、`_coerce_si_params`
    # 解析、bounds 进 schema)在这条路上一条都不生效。同一个 skill 两条注册路径,
    # 保护强度相反:ParameterSpec 里 min_separation_m 明明写着 unit="m"
    # (skills/builtins/flat_region.py:102-113),skill 路径上它是强制前缀的字符串,
    # 这里却是裸 float 且无人判量级。
    #
    # 只读工具,量级错了不会写坏数据 —— 但 min_separation_m=0.1(十厘米)会排除掉
    # 整张图的每一个候选窗口,然后如实回答「找不到平坦区域」,又是一句完全正常的话。
    _q, _errs = resolve_metre_args("find_flat_region",
                                   {"min_separation_m": min_separation_m})
    if _errs:
        return "find_flat_region failed: " + " ".join(_errs)
    min_separation_m = _q["min_separation_m"]
    params: dict = {"scan_path": scan_path, "window_fraction": window_fraction,
                    "channel": channel}
    if min_separation_m:
        params["min_separation_m"] = min_separation_m
    res = FindFlatRegion().execute(None, params)
    if not res.success:
        return f"find_flat_region failed: {res.error}"
    return "find_flat_region ok: " + json.dumps(res.data)


@tool("assess_cluster_roundness")
def assess_cluster_roundness(scan_path: str, threshold_sigma: float = 1.5,
                             polarity: str = "auto", channel: str = "Z") -> str:
    """Segment the largest bright (or dark) protrusion in a SAVED .sxm scan and score its
    roundness/circularity — for morphology QC of clusters/adsorbates/islands. polarity:
    'auto'|'bright'|'dark'. threshold_sigma: segmentation cut in std-devs above the plane.
    Offline file analysis; scan_path = .sxm on disk."""
    import json
    from mast.skills.builtins.cluster_roundness import AssessClusterRoundness
    params: dict = {"scan_path": scan_path, "threshold_sigma": threshold_sigma,
                    "polarity": polarity, "channel": channel}
    res = AssessClusterRoundness().execute(None, params)
    if not res.success:
        return f"assess_cluster_roundness failed: {res.error}"
    return "assess_cluster_roundness ok: " + json.dumps(res.data)


@tool("analyze_scan_image")
def analyze_scan_image(scan_path: str, channel: str = "Z", flatten: str = "auto",
                       save_png: bool = False, threshold_profile: str = "") -> str:
    """Measure a SAVED .sxm frame and decide HOW IT SHOULD BE PROCESSED — which
    flattening (plane / 2nd-order surface / line-by-line / line-fitted-on-the-
    dominant-terrace) and how tight the colour scale should be — with the measured
    reason for every choice. Call this BEFORE level_lines / subtract_plane_ransac /
    subtract_poly2d when you do not already know which one this frame needs; those
    tools apply a method, this one picks it.

    It also forwards the existing verdicts for the frame: atomic phase, mid-scan
    tip change, forward/backward agreement, bad scan lines. It never claims a
    lattice on its own — and 'cannot judge at this pixel size' is a different
    answer from 'no lattice'. flatten='auto' (default) lets the measurements
    decide. Offline file analysis; scan_path = .sxm on disk."""
    from mast.skills.builtins.scan_prep import AnalyzeScanImage
    res = AnalyzeScanImage().execute(None, {
        "scan_path": scan_path, "channel": channel, "flatten": flatten,
        "save_png": bool(save_png), "threshold_profile": threshold_profile})
    if not res.success:
        return f"analyze_scan_image failed: {res.error}"
    return _truncate("analyze_scan_image ok:\n" + (res.summary or ""), 3000)


@tool("auto_process_scan_batch")
def auto_process_scan_batch(folder: str, channel: str = "Z", render: bool = True,
                            threshold_profile: str = "") -> str:
    """Do the same for EVERY .sxm in a folder, then HARMONISE frames that share a
    scan size and bias so a contrast difference between two of them can only come
    from the sample, never from the processing (frames with a real step keep their
    protective treatment). Renders the PNGs and writes _report.md listing every
    measured number and the reason for every decision — point the user at that file
    rather than repeating it. Use this when asked to 'process/plot all the scans in
    this folder'. Offline; touches no hardware."""
    from mast.skills.builtins.scan_prep import AutoProcessScanBatch
    res = AutoProcessScanBatch().execute(None, {
        "folder": folder, "channel": channel, "render": bool(render),
        "threshold_profile": threshold_profile})
    if not res.success:
        return f"auto_process_scan_batch failed: {res.error}"
    return _truncate("auto_process_scan_batch ok:\n" + (res.summary or ""), 3000)


@tool("get_latest_scan_file")
def get_latest_scan_file(max_age_s: int = 300) -> str:
    """Locate the path of the most recently written .sxm scan on disk (session dir →
    data working-sessions/ → legacy). Use this FIRST when asked to analyse "the latest /
    current scan" but no explicit file path was given — then feed the returned path into
    find_flat_region / assess_cluster_roundness / detect_defects / fft_2d. Returns
    {path, age_s, ...} or {path: null} if nothing recent. Pure filesystem read."""
    import json
    from mast.skills.builtins.scan_extra import GetLatestScanFile
    res = GetLatestScanFile().execute(None, {"max_age_s": max_age_s})
    if not res.success:
        return f"get_latest_scan_file failed: {res.error}"
    return "get_latest_scan_file ok: " + json.dumps(res.data)


# ─────────────────────────────────────────────────────────────────────
# 10/11. Read-only scan DISCOVERY — list_scan_dir / glob_scans
#    The 2026 field run had the LLM guess a dozen directory+filename combos and
#    load_scan-fail 64× (72e18bf1) because it had no way to SEE what was on
#    disk — only a "load this exact path" primitive. These two tools give it a
#    directory listing / name search in ONE call, and record every hit into the
#    scan registry so the follow-up is load_scan(scan_id=…), not another guess.
# ─────────────────────────────────────────────────────────────────────

# Extensions a discovery listing surfaces (superset of the loaders' image set —
# a directory scan should show .3ds grids + .sm4 too, even if load_scan is
# image/STS only, so the agent knows they exist).
_DISCOVER_SUFFIXES = (".sxm", ".npy", ".npz", ".dat", ".3ds", ".sm4", ".txt", ".csv")


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _register_discovered(p: Path) -> str:
    """Record a discovered file into the scan registry and return its scan_id.

    For .sxm we enrich with channels + frame geometry via a CHEAP header-only
    read (no full frame load — a directory listing must stay fast). Other
    formats record path-only so their scan_id still resolves."""
    try:
        from mast.core.scan_registry import derive_scan_id
        sid = derive_scan_id(str(p))
    except Exception:  # noqa: BLE001
        sid = p.stem.rstrip("_")
    if p.suffix.lower() == ".sxm":
        try:
            from mast.core.scan_registry import record_scan
            from mast.io.nanonis_files import read_sxm_header, sxm_frame_meta
            hdr = read_sxm_header(str(p))
            record_scan(str(p), scan_id=sid,
                        channels=hdr.get("channel_names") or None,
                        frame=(sxm_frame_meta(hdr) if hdr else None))
            return sid
        except Exception:  # noqa: BLE001 — fall through to path-only record
            pass
    try:
        from mast.core.scan_registry import record_scan_path
        record_scan_path(str(p))
    except Exception:  # noqa: BLE001
        pass
    return sid


def _render_listing(files: list[Path], max_items: int, where: str) -> str:
    """Sort newest-first, register each, and render a compact LLM-friendly table
    plus a JSON array (scan_id/name/path/size/age) for machine use."""
    import json
    import time as _time
    files = sorted(set(files), key=_safe_mtime, reverse=True)
    if not files:
        return f"no scan files found ({where})"
    now = _time.time()
    rows: list[dict] = []
    for p in files[:max_items]:
        sid = _register_discovered(p)
        try:
            st = p.stat()
            size = st.st_size
            age = round(now - st.st_mtime, 1)
        except OSError:
            size, age = None, None
        rows.append({"scan_id": sid, "name": p.name, "path": str(p),
                     "ext": p.suffix.lower(), "size_bytes": size, "age_s": age})
    lines = [f"{len(files)} scan file(s) {where}; showing {len(rows)} newest:"]
    for r in rows:
        age = f"{r['age_s']}s ago" if r["age_s"] is not None else "?"
        kb = f"{r['size_bytes'] / 1024:.0f}KB" if r["size_bytes"] else "?"
        lines.append(f"  [{r['scan_id']}] {r['name']}  ({kb}, {age})")
    lines.append("→ load with load_scan(scan_id=\"<id>\") — no need to retype the path.")
    lines.append("json=" + json.dumps(rows, ensure_ascii=False))
    return _truncate("\n".join(lines), 3000)


@tool("list_scan_dir")
def list_scan_dir(directory: str = "", max_items: int = 50) -> str:
    """List scan/data files in a directory, newest first — ONE call to SEE what
    is actually on disk instead of guessing filenames and load_scan-ing them one
    at a time. Leave `directory` empty to list the folders the instrument is
    known to have saved into this session (the Nanonis session dir, recorded on
    every SaveScan). Each row is a scan_id you can feed straight into
    load_scan(scan_id=…). Read-only; shows .sxm/.npy/.npz/.dat/.3ds/.sm4/.txt/.csv."""
    from mast.core.scan_registry import known_scan_dirs
    if directory:
        d = Path(directory)
        if not d.is_dir():
            return f"list_scan_dir failed: 目录不存在或不可读: {directory!r}"
        dirs = [d]
        where = f"in {directory}"
    else:
        dirs = [d for d in known_scan_dirs() if d.is_dir()]
        if not dirs:
            return ("list_scan_dir: 本进程尚无已记录的扫描目录（还没保存过扫描？）。"
                    "请指定 directory=<绝对路径>，或先调用 get_latest_scan_file。")
        where = f"across {len(dirs)} known scan dir(s)"
    files: list[Path] = []
    for d in dirs:
        try:
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in _DISCOVER_SUFFIXES:
                    files.append(p)
        except OSError:
            continue
    return _render_listing(files, max_items, where)


@tool("glob_scans")
def glob_scans(pattern: str, directory: str = "", recursive: bool = True,
               max_items: int = 50) -> str:
    """Find scan files by NAME across the known scan directories (plus an
    optional `directory`). `pattern` is a glob ("Au111*", "*_001.sxm") or a plain
    substring ("Au111" → *Au111*). Use when you know PART of a scan's name but
    not its folder — instead of guessing full paths. Records every hit into the
    scan registry, so the follow-up is load_scan(scan_id=…). Read-only discovery."""
    from mast.core.scan_registry import known_scan_dirs
    pat = (pattern or "").strip()
    if not pat:
        return "glob_scans failed: 需要一个名字模式（glob 或子串）。"
    # A plain substring (no glob metacharacter) → contains-match.
    glob_pat = pat if any(c in pat for c in "*?[]") else f"*{pat}*"
    dirs = [d for d in known_scan_dirs() if d.is_dir()]
    if directory:
        d = Path(directory)
        if d.is_dir():
            dirs.insert(0, d)
        else:
            return f"glob_scans failed: 目录不存在或不可读: {directory!r}"
    if not dirs:
        return ("glob_scans: 本进程尚无已记录的扫描目录。请指定 directory=<绝对路径>，"
                "或先 get_latest_scan_file 让扫描目录被记录。")
    files: list[Path] = []
    seen_dirs = set()
    for d in dirs:
        try:
            rd = d.resolve()
        except OSError:
            continue
        if rd in seen_dirs:
            continue
        seen_dirs.add(rd)
        try:
            it = d.rglob(glob_pat) if recursive else d.glob(glob_pat)
            for p in it:
                if p.is_file() and p.suffix.lower() in _DISCOVER_SUFFIXES:
                    files.append(p)
        except OSError:
            continue
    return _render_listing(files, max_items, f"matching {glob_pat!r}")


# ─────────────────────────────────────────────────────────────────────
# Top-level assembly
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 10. Ported paper/ analysis skills — the ON-REQUEST analysis library.
#
#     Same thin-@tool pattern as §8: pure ANALYSIS BaseSkills whose execute()
#     reads a file path and never touches a hardware context, so a None context
#     is correct rather than a shortcut.
#
#     These ANALYSE, they never render. plot_scan / plot_spectrum own the
#     figures; a tool here returns numbers, or writes a new .npy for plot_scan
#     to draw. Nothing here parses a field of view or draws an axis.
#
#     DELIBERATELY NOT BRIDGED (do not "fix" this by adding them) — all verified
#     by execution, 2026-07-27:
#       * PredictSpectrumFromTopo / PredictStructure_ASD / IdentifyTopology_CARP
#         — model_path is REQUIRED and no such weights ship with MAST. All three
#           return success=False on valid input. A tool that can only fail makes
#           the agent retry-loop.
#       * the 9 paper/ COMPOSITE skills (AutonomousSurvey_Scanbot, AtomManip_SAC,
#         ConditionTip_DQN, …) — they DRIVE THE TIP, are safety_level=CONFIRM,
#         and belong to instrument_control. With a None context FOUR of them
#         return success=True while every internal step logged
#         "MoveToXY raised AttributeError: 'NoneType'". Bridging them here would
#         hand this agent a tool that reports a finished survey having moved
#         nothing, AND bypass the HITL gate their CONFIRM level exists to fire.
#       * FindEmptySpot — duplicates find_flat_region.
#       * AssessTip_ResNet / AssessTip_VGG — with no weights both return a
#         BIT-IDENTICAL heuristic score; tip quality is IC's artifact anyway
#         (read_latest_tip_status).
# ─────────────────────────────────────────────────────────────────────

#: skill class name → the @tool name exposing it. Read by the orphan-guard test
#: so a bridge cannot rot and a new paper/ skill cannot silently go unowned.
PAPER_SKILL_BRIDGES: dict[str, str] = {}   # 公开版：skills/paper 不随仓发布


def _finite(o):
    """Recursively replace NaN/Infinity with None.

    curve_fit reports a non-converged fit as r_squared=NaN and *_err=Infinity
    (verified: FitFano_Kondo model='hurwitz_fano' does exactly this). json.dumps
    then emits bare ``NaN`` / ``Infinity``, which is NOT valid JSON — the
    browser's JSON.parse throws on it. Nulling them is also the honest
    rendering: "this number does not exist" beats a token that reads like one.
    """
    import math
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _fit_verdict(r2, value, err) -> str:
    """Turn a curve_fit result into an explicit trust verdict.

    A BAD FIT DOES NOT COME BACK AS success=False — it comes back as a
    plausible-looking number with a broken R². Verified: FitGap_BCS pointed at a
    Nanonis .dat's Current column (which is what didv_column=1 selects!) returns
    delta_eV = 1.39 meV — perfectly reasonable to the eye — with r_squared =
    4.1e-06 and an error bar 590% of the value. Reporting that number without
    the verdict is how a fabricated gap reaches a manuscript.
    """
    if r2 is None or not isinstance(r2, (int, float)) or r2 != r2:
        return ("拟合未收敛（R² 不存在／协方差无法估计）——参数等于初始猜测，"
                "不可采信。请换 model= 或检查 didv_column 是否指向真正的 dI/dV 列。")
    if r2 < 0.5:
        return (f"拟合失败（R²={r2:.3g}）。最常见原因：didv_column 指错列 —— "
                "原始 Nanonis .dat 的 column 1 通常是 Current (A)，dI/dV 在 "
                "LI Demod 列（多为 column 2）。不要引用这些数值。")
    rel = (abs(err / value) if (isinstance(value, (int, float)) and value
                                and isinstance(err, (int, float))) else None)
    if rel is not None and rel > 0.2:
        return (f"拟合勉强（R²={r2:.3g}，相对误差 {rel:.0%}）——可报告但必须带"
                "误差棒，不可作为结论性数值。")
    return f"拟合良好（R²={r2:.3g}）。"


def _run_skill(skill_cls, params: dict, label: str) -> str:
    """Execute a pure-analysis paper skill and render its result for the model."""
    import json
    try:
        res = skill_cls().execute(None, {k: v for k, v in params.items()
                                         if v is not None and v != ""})
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", label)
        return f"{label} failed: {type(exc).__name__}: {exc}"
    if not res.success:
        return f"{label} failed: {res.error}"
    return _truncate(f"{label} ok: "
                     + json.dumps(_finite(res.data or {}), ensure_ascii=False))


# ── image preprocessing — each writes a NEW .npy, feed that to plot_scan ──────

# ── drift / registration ─────────────────────────────────────────────────────

# ── feature detection / quality ──────────────────────────────────────────────

@tool("record_analysis")
def record_analysis(summary: str, metrics_json: str = "", figure_paths: str = "",
                    anomalies: str = "", metrics_path: str = "",
                    tool_call_id: Annotated[str, InjectedToolCallId] = "",
                    ) -> "ArtifactToolReturn | str":
    """把你这一轮的**分析结论**登记为交付产物，让下游（写报告的智能体）直接看到。

    在这个工具存在之前，你算出的指标和渲染出的图只活在对话流里：交接时只剩一句
    "分析完成"，写报告的智能体既不知道有哪些图、也不知道关键数值，只能重算或漏写。

    什么时候调：**在交接之前**，把已经得出的结论登记一次。可以多次调用，后一次整体
    覆盖前一次（所以请给完整的一份，不要只给增量）。

    Args:
        summary:      一两句话的结论（例如「Au(111) 人字纹清晰，缺陷密度 0.8/100nm²」）。
        metrics_json: 数值指标的 JSON 对象，例如 {"lattice_nm": 0.288, "defects": 12}。
                      只放**数值**；文字说明写进 summary。
        figure_paths: 图片路径，多个用换行或逗号分隔（就是 plot_scan 等返回给你的路径）。
        anomalies:    观察到的异常，多条用换行分隔。没有就留空 —— **不要编**。
        metrics_path: 一个 JSON 文件的路径（通常是 py_run 里
                      `mastdata.save_result(...)` 写出来的 `out/result.json`）。
                      **给了它就不要再手打 metrics_json** —— 里面的数值会被直接读
                      进来，且**覆盖** metrics_json 里的同名键。

    为什么有 metrics_path：数值从 `curve_fit` 到这里，整条路径上**没有任何一步需要
    你把它重打一遍**。你复述一个 `3.2e-12` 时丢掉指数，结果看起来会完全合理 ——
    那种错没人查得出来。能让文件自己走完这段路，就别经手。
    """
    text = (summary or "").strip()
    if not text and not (metrics_json or figure_paths or anomalies):
        return "record_analysis: 没有任何内容可登记。"

    metrics: dict = {}
    if (metrics_json or "").strip():
        try:
            import json as _json
            raw = _json.loads(metrics_json)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        metrics[str(k)] = float(v)
                    except (TypeError, ValueError):
                        # A non-numeric "metric" is a note; keep it visible in the
                        # summary rather than silently dropping it.
                        text = f"{text}｜{k}={v}" if text else f"{k}={v}"
            else:
                return ("record_analysis: metrics_json 必须是一个 JSON **对象**"
                        f"（形如 {{\"key\": 1.23}}），收到的是 {type(raw).__name__}。")
        except Exception as exc:  # noqa: BLE001 — the model's JSON, not ours
            return f"record_analysis: metrics_json 不是合法 JSON（{exc}）。"

    # metrics_path 后合并 ⇒ 文件里的值覆盖手打的值。顺序是刻意的：文件是数值的
    # 出处，手打的那份只是模型的转述，两者冲突时该信出处。
    if (metrics_path or "").strip():
        try:
            import json as _json
            from pathlib import Path as _P

            raw = _json.loads(_P(metrics_path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return (f"record_analysis: {metrics_path} 里不是一个 JSON 对象"
                        f"（是 {type(raw).__name__}）。")
            for k, v in raw.items():
                try:
                    metrics[str(k)] = float(v)
                except (TypeError, ValueError):
                    text = f"{text}｜{k}={v}" if text else f"{k}={v}"
        except FileNotFoundError:
            return (f"record_analysis: 找不到 {metrics_path}。"
                    "它通常是 py_run 里 mastdata.save_result(...) 写的 "
                    "out/result.json —— 先跑一次再来登记。")
        except Exception as exc:  # noqa: BLE001
            return f"record_analysis: 读不了 {metrics_path}（{exc}）。"

    def _split(s: str) -> list[str]:
        out: list[str] = []
        for chunk in (s or "").replace(",", "\n").replace("，", "\n").splitlines():
            c = chunk.strip()
            if c:
                out.append(c)
        return out

    figures = _split(figure_paths)
    anomaly_list = _split(anomalies)

    try:
        result = AnalysisResult(metrics=metrics, figures=figures,
                                anomalies=anomaly_list, summary=text)
    except Exception as exc:  # noqa: BLE001
        return f"record_analysis: 登记失败（{type(exc).__name__}: {exc}）。"

    lines = ["已登记本轮分析结论，写报告的智能体会在自己的上下文里看到它。"]
    if metrics:
        lines.append(f"  指标 {len(metrics)} 项")
    if figures:
        lines.append(f"  图 {len(figures)} 张")
    if anomaly_list:
        lines.append(f"  异常 {len(anomaly_list)} 条")
    lines.append("再次调用会整体覆盖，请给完整的一份。")
    summary_text = "\n".join(lines)
    return ArtifactToolReturn(summary_text, {"analysis": result},
                              tool_call_id=tool_call_id, name="record_analysis")


# ═════════════════════════════════════════════════════════════════════
# Python 执行环境（mast.pyexec）
#
# 一个**独立解释器进程**里的会话式工作目录：完整科学栈、可读遍测量数据、
# 可并行、可跑很久。三条底线由 mast.pyexec 保证 —— 进不了仪器（那个解释器里
# 没有 nanonis_spm）、毁不掉已存在的测量文件、随时停得下来。
#
# 这两个 tool 同 run_numpy_snippet 一样被 WORKFLOW_TOOL_EXCLUDE 挡在工作流菜单
# 之外（「数据而非代码」主轨，tool_skills.py:66-71）。
# ═════════════════════════════════════════════════════════════════════

#: 同一会话连续多少次非零退出就停手。StallGuard 归一化错误签名靠的是文本相似，
#: 而 traceback 带着行号和临时路径 —— 它未必认得出「同一个错」。这个计数器是
#: 兜底：不让模型在一个改不好的脚本上把 80 次 tool 预算烧光。
_PY_MAX_CONSECUTIVE_FAILURES = 5
_py_failures: dict = {}


def _py_session(state, config, *, reset: bool = False):
    """从框架注入的东西派生会话 —— **不让模型自己起名字**。

    让模型记住会话名，就会有「上次那个叫什么来着」，而它答错的代价是丢掉全部中间
    结果。experiment_id 来自 state，thread_id 来自 config，两个它都碰不到。
    """
    from mast.pyexec import get_session

    st = state if isinstance(state, dict) else {}
    cfg = config if isinstance(config, dict) else {}
    thread = str((cfg.get("configurable") or {}).get("thread_id") or "")
    return get_session(experiment_id=str(st.get("experiment_id") or ""),
                       thread_id=thread or "dp-default", reset=reset)


def _py_runtime():
    """``(runtime, 说明)``。没有运行时就返回一句**点名查了哪些地方**的话。"""
    from mast.pyexec.runtime import probe_runtime

    rep = probe_runtime()
    if rep.runtime is None:
        return None, rep.why_not()
    return rep.runtime, ""


_PY_HINTS = (
    ("plot_scan", ("imshow", "savefig"),
     "这一步的出图部分 plot_scan 也能做，而且会自动带真实 nm 坐标轴和带单位的色标"),
    ("fft_2d", ("fft2",), "只做二维 FFT 的话 fft_2d 一步就有"),
    ("fit_sts_peaks", ("find_peaks",), "谱峰拟合可以直接用 fit_sts_peaks（带 R² 和误差棒）"),
    ("detect_defects", ("label(",), "数缺陷可以直接用 detect_defects"),
    ("plane_subtract", ("lstsq",), "单纯的平面扣除 plane_subtract 一步就有"),
)


def _py_hint(code: str) -> str:
    """跑完之后附一句「这件事有现成的」—— **提示，不拦截**。

    第一版设计成命中就拒绝执行，撤销了：一个更复杂的分析只要恰好用了
    imshow+savefig 就被打回，那一轮白费。而这条的真正价值不是省一次执行 ——
    是 plot_scan 的图**客观更好**（真 nm 坐标轴、带单位色标）。是路由到更好的
    输出，不是阻拦，所以放在结果后面、用建议的语气。
    """
    low = (code or "").lower()
    for name, needles, msg in _PY_HINTS:
        if all(n.lower() in low for n in needles):
            return f"\n提示：{msg}（工具名 {name}）。这次的结果不受影响。"
    return ""


@tool("py_stage_data")
def py_stage_data(scan_id: str = "", path: str = "", name: str = "",
                  state: Annotated[dict, InjectedState] = None,  # type: ignore[assignment]
                  config: RunnableConfig = None) -> str:
    """把一份测量数据放进本次分析的工作目录，**连同算好的物理尺度**。

    这是**便利入口，不是唯一入口**：批量分析请在 py_run 里用
    `from mast.io.nanonis_files import read_sxm` 自己循环，不要调我 200 次。

    那它多给了什么？—— nm/像素、偏压、扫描范围，由主程序从文件表头算好写进
    `inputs/manifest.json`。脚本读一个数就行，不用自己换算，你也不用把这些数字
    抄进对话（抄错一个指数就是三个数量级，而结果看起来完全合理）。

    Args:
        scan_id: 扫描 id（推荐；来自 list_scan_dir / glob_scans / get_latest_scan_file）。
        path:    或者直接给文件路径。支持 .sxm / .dat / .txt / .npy。
        name:    脚本里引用它的名字（默认用文件名）。

    脚本里这样取：``arrays, meta = mastdata.load("<name>")``。
    """
    if not (scan_id or path):
        return "py_stage_data: 要么给 scan_id，要么给 path。"
    try:
        from mast.pyexec import stage

        session = _py_session(state, config)
        st = stage(session, path=path, scan_id=scan_id, name=name)
    except FileNotFoundError as exc:
        return f"py_stage_data: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"py_stage_data 失败（{type(exc).__name__}: {exc}）。"

    lines = [f"已放入 `{st.name}`（{st.kind}）。脚本里：`arrays, meta = mastdata.load(\"{st.name}\")`"]
    for k, d in st.arrays.items():
        unit = f" [{d['unit']}]" if d.get("unit") else ""
        lines.append(f"  {k}: shape={tuple(d['shape'])} {d['dtype']}{unit}")
    nm = st.meta.get("nm_per_px")
    if nm:
        lines.append(f"  nm/像素 = {nm[0]:.5g}（meta['nm_per_px']，别自己换算）")
    if st.meta.get("bias_V") is not None:
        lines.append(f"  偏压 = {st.meta['bias_V']} V（meta['bias_V']）")
    return "\n".join(lines)


@tool("py_run")
def py_run(code: str, filename: str = "", timeout_s: int = 300,
           reset: bool = False,
           tool_call_id: Annotated[str, InjectedToolCallId] = "",
           state: Annotated[dict, InjectedState] = None,  # type: ignore[assignment]
           config: RunnableConfig = None) -> "ArtifactToolReturn | str":
    """**上面那张「问题 → 工具」表匹配不到你要回答的问题时**，写 Python 自己算。

    表里有的就用表里的：那些工具的输出更好（真坐标轴、带单位、带 R² 和误差棒），
    而且不用你写代码。这里是给表答不了的那些问题的。

    工作目录在多次调用之间**保留**：上一步存的中间结果、写出的 .npz/.png 下一步
    还在。所以请把长分析**拆成几个小步骤**，而不是写一个几百行的脚本。

    可用：numpy、scipy、matplotlib(Agg)、pandas、scikit-image、scikit-learn。
    `import mastdata` 提供：
      · `names()` / `load(name)` —— 取 py_stage_data 放进来的数据（含 nm/像素、偏压）
      · `scan_dirs()` —— 测量数据都在哪（可以自己 glob，不必经 py_stage_data）
      · `out(f)` / `savefig(fig, "x.png")` / `save_result(**metrics)`
    `from mast.io.nanonis_files import read_sxm, read_dat, read_3ds` 直接读原始文件；
    `mast.io.map_analysis` / `mast.core.si_quantity` 也在。
    `source/` 目录里是**整个 MAST 的源码**（读，不是 import）—— 那些 skill 里有大量
    STM 数据处理的现成写法，比从零推导快。

    **数值结论请用 `mastdata.save_result(...)` 写进 `out/result.json`**：它会原样
    回传给你，还能直接 `record_analysis(metrics_path=...)`。不要把数字抄进回答，
    也不要凭记忆重述量级。
    **图请用 `mastdata.savefig(fig, "xxx.png")`** —— 会自动进图库（下游写报告的
    智能体只看得见那里），而且**你下一轮会看见这张图**，可以据此判断要不要重画。

    环境边界：进不了仪器；不能覆盖或删除已存在的 .sxm/.dat/.3ds（新建文件、写派生
    结果都可以）；超时会被整棵进程树杀掉。出错会返回完整 traceback，据此改代码重跑。

    Args:
        code:      Python 源码。
        filename:  存成哪个文件名（默认 stepNN.py）。历次脚本全部留档。
        timeout_s: 墙钟上限，默认 300 秒，最大 3600。
        reset:     True = 归档当前工作目录、从空的开始（**归档不是删除**）。
    """
    if not (code or "").strip():
        return "py_run: code 是空的。"
    if len(code) > 50_000:
        return ("py_run: 代码超过 50000 字符 —— 请拆成几步。工作目录在调用之间保留，"
                "中间结果存 .npz 下一步接着用。")

    runtime, why = _py_runtime()
    if runtime is None:
        return f"py_run 不可用。\n{why}"

    from mast.pyexec import harvest as _harvest, images_for_toolmessage, run, snapshot

    try:
        session = _py_session(state, config, reset=reset)
    except Exception as exc:  # noqa: BLE001
        return f"py_run: 建不起工作目录（{type(exc).__name__}: {exc}）。"

    fails = _py_failures.get(session.sid, 0)
    if fails >= _PY_MAX_CONSECUTIVE_FAILURES:
        _py_failures[session.sid] = 0
        return (f"py_run: 这个会话已经连续 {fails} 次执行失败。先停下来 —— "
                "报告目前已经得到的结果，或者说明卡在哪里，别继续改这个脚本。")

    name = (filename or session.next_step_name()).strip()
    if not name.endswith(".py"):
        name += ".py"
    name = name.replace("/", "_").replace("\\", "_")
    (session.code_dir / name).write_text(code, encoding="utf-8")

    before = snapshot(session)
    res = run(session, f"code/{name}", runtime=runtime, timeout_s=float(timeout_s))
    h = _harvest(session, before)
    session.record_step({"script": name, "rc": res.returncode,
                         "timed_out": res.timed_out,
                         "figures": len(h.figures),
                         "duration_s": round(res.duration_s, 2)})

    if res.ok:
        _py_failures[session.sid] = 0
    else:
        _py_failures[session.sid] = fails + 1

    parts: list[str] = []
    if res.timed_out:
        parts.append(f"⏱ 超时：{res.killed_reason}。整棵进程树已被终止。")
    elif not res.ok:
        parts.append(f"✗ 退出码 {res.returncode}。")
        if res.killed_reason:
            parts.append(res.killed_reason)
    else:
        parts.append(f"✓ {res.duration_s:.1f}s")

    if res.stdout.strip():
        parts.append(f"\n--- stdout ---\n{res.stdout.rstrip()}")
    if res.stderr.strip():
        parts.append(f"\n--- stderr ---\n{res.stderr.rstrip()}")
    if res.truncated:
        parts.append(f"（输出已截断，完整内容在 {Path(res.stdout_path).name} / "
                     f"{Path(res.stderr_path).name}，可以在下一步 py_run 里读它）")

    if h.result_json:
        parts.append(f"\n--- out/result.json（原样）---\n{h.result_json.rstrip()}")
        parts.append("（这些数值可以直接 record_analysis(metrics_path=...) 登记，"
                     "不用重打）")
    if h.new_files:
        parts.append(f"\nout/ 新增：{'、'.join(h.new_files)}")
    if h.figures:
        parts.append(f"已进图库 {len(h.figures)} 张图。")
    for note in h.notes:
        parts.append(f"⚠ {note}")
    if res.ok:
        hint = _py_hint(code)
        if hint:
            parts.append(hint)

    summary = "\n".join(parts)
    images = images_for_toolmessage(h)
    if not images:
        return summary

    # 图挂在 ToolMessage 上 —— 模型下一轮就看得见自己画的图。
    # 键名从 vision_mw import，绝不重打字面量（生产方在记、消费方读不到，
    # 是这个仓踩过的静默失败形状）。
    from mast.agents._shared.vision_mw import IMAGES_KEY

    msg = ToolMessage(content=summary, tool_call_id=tool_call_id, name="py_run",
                      additional_kwargs={IMAGES_KEY: images})
    return ArtifactToolReturn(summary, {"messages": [msg]},
                              tool_call_id=tool_call_id, name="py_run")


AGENT_TOOLS: list = [
    load_scan,
    record_analysis,
    # Figures first: rendering one is the baseline deliverable for this agent,
    # analysis is on request.
    plot_scan,
    plot_spectrum,
    fft_2d,
    plane_subtract,
    detect_defects,
    fit_sts_peaks,
    run_numpy_snippet,
    # 2026-08-19: 独立解释器进程里的会话式 Python（mast.pyexec）。与
    # run_numpy_snippet 并存 —— 后者进程内 ~50 ms，适合「这个数组的均值是多少」；
    # 需要 numpy 以外的东西、或者要产出文件，就用 py_run。
    py_stage_data,
    py_run,
    mosaic_scans,
    find_flat_region,
    assess_cluster_roundness,
    # Which flattening does THIS frame need, and why (mast.vision.scan_prep).
    # These pick a method; level_lines / subtract_* below apply one.
    analyze_scan_image,
    auto_process_scan_batch,
    get_latest_scan_file,
    list_scan_dir,
    glob_scans,
    # §10 on-request analysis library — called only when an analysis was asked
    # for. Figures stay with plot_scan / plot_spectrum; these return numbers.
    # Image preprocessing. Each writes a NEW .npy — chain into plot_scan to see
    # the result; none of them overwrite the measurement.
    # Drift / registration.
    # Feature detection / scan quality.
]


def build_tools(buf: "BufferService | None") -> list:
    tools: list = list(AGENT_TOOLS)
    if buf is not None:
        tools = tools + make_buffer_tools(buf)
    tools = tools + [
        make_handoff(
            "supervisor",
            "Return control to the orchestrator. Use when analysis is complete.",
        ),
        make_handoff(
            "paper_writing",
            "Hand analysis results to the Paper-Writing agent for draft authoring.",
        ),
    ]
    logger.info("data_processing: built %d tools (buf=%s)", len(tools), buf is not None)
    return tools


__all__ = [
    "load_scan", "fft_2d", "plane_subtract", "detect_defects",
    "fit_sts_peaks", "run_numpy_snippet", "py_stage_data", "py_run",
    "mosaic_scans",
    "list_scan_dir", "glob_scans",
    "analyze_scan_image", "auto_process_scan_batch",
    "AGENT_TOOLS", "build_tools",
]
