"""Spatial mosaic of multiple .sxm scans — place each by its stage coordinates.

Builds one big-canvas overview from many STM scans of a single sample: every
scan is positioned by its real center (``scan_offset``) and size
(``scan_range``) in a common reference frame, so the operator sees *where on the
surface* each image sits. (Contrast with sxm_preview-style grid mosaics, which
just tile images by file order; this one honours physical xy, placing every scan
by its real xy coordinates in a single shared reference frame.)

Pure / offline: reads .sxm via ``mast.io.nanonis_files.read_sxm`` and returns a
numpy RGB canvas + helpers to render/save it. No hardware, no agent state — the
big array only ever lands on disk (PNG/NPY), never in a checkpoint.

Display pipeline borrows the proven choices from the lab's offline preview tool
(HamsterPark/Nanonis-RHK-SPM-PyTools): optional per-row median flattening for Z,
percentile clipping, then a matplotlib colormap.

Coordinate frame: Nanonis stage frame, meters. Canvas spans the bounding box of
all scan footprints. Scan angle is assumed ~0 (axis-aligned footprints); a
rotated frame is placed by its bounding box and flagged in ``angle_warning``
(full affine warping is a future enhancement; most overview scans are angle 0).

"Switched samples → switch canvas" is just a fresh call with the new file set:
each mosaic is independent (nothing accumulates across calls), and ``label`` /
the output filename carry the sample name.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

_MAX_CANVAS_PX = 4096   # hard ceiling per side — guards against runaway canvases
_DEFAULT_CMAP = "viridis"


# ── Header → physical footprint ────────────────────────────────────────────────

def parse_xy_meta(header: dict) -> dict | None:
    """Extract {cx, cy, w, h, angle} in METERS from an sxm header, or None.

    ``read_sxm``'s generic header parser stores ``:SCAN_OFFSET:`` /
    ``:SCAN_RANGE:`` / ``:SCAN_ANGLE:`` as raw whitespace-separated strings, so
    we split + float them here.
    """
    def _two(key: str):
        v = header.get(key)
        if not v:
            return None
        parts = str(v).split()
        try:
            return float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            return None

    off = _two("scan_offset")
    rng = _two("scan_range")
    if off is None or rng is None:
        return None
    w, h = abs(rng[0]), abs(rng[1])
    if w <= 0.0 or h <= 0.0:
        return None
    # ⚠️ 角度解析失败时,``angle`` 仍然给 0.0(摆放要一个数,而轴对齐是合理的
    # 近似),但**必须把「这是猜的」传下去** —— 否则一帧真的转过的图会被当成
    # 轴对齐悄悄摆上画布,而 ``angle_warning`` 判的是 ``abs(angle) > 1.0``,
    # 0.0 恰好让那道提示**不触发**。兜底值又一次落在了「没什么可担心的」那一侧。
    #
    # 这里没有把 ``angle`` 本身改成 ``None``:六个调用方都当它是数
    # (`vision.py` / `flat_region.py` / `frame_tilt.py` / `condition_tip.py`),
    # 改类型是一次波及面大得多的改动,而这条只是**弱实例**(离线 header 解析,
    # 后果是摆放近似,不是硬件动作)。加一个旗标,让判提示的那一侧自己决定。
    ang = 0.0
    ang_known = True
    a = header.get("scan_angle")
    if a:
        try:
            ang = float(str(a).split()[0])
        except (IndexError, ValueError):
            ang, ang_known = 0.0, False
    else:
        # header 里根本没有 SCAN_ANGLE —— 同样是「不知道」,不是「等于 0」。
        ang_known = False
    return {"cx": off[0], "cy": off[1], "w": w, "h": h,
            "angle": ang, "angle_known": ang_known}


def px_to_m(px_x: float, px_y: float, *, nx: int, ny: int,
            cx_m: float, cy_m: float, w_m: float, h_m: float,
            angle_deg: float = 0.0) -> "tuple[float, float]":
    """像素坐标 → 仪器坐标(米)。**全仓唯一一份。**

    一行就能用,这是有意的::

        from mast.io.mosaic import parse_xy_meta, px_to_m
        meta = parse_xy_meta(header)
        x_m, y_m = px_to_m(col, row, nx=nx, ny=ny,
                           cx_m=meta["cx"], cy_m=meta["cy"],
                           w_m=meta["w"], h_m=meta["h"],
                           angle_deg=meta["angle"])

    ## 为什么它必须好用到没人想自己写

    2026-08-10:这套换算一度有**三份**实现 —— ``flat_region`` 里一份、
    ``cluster_extract`` 里一份,以及一次性分析脚本里内联的第三份。
    第三份是**唯一没被测过的那份**,而写它的人事后说:
    「**我甚至没意识到自己在造第三份。**」

    追问「为什么内联」,答案是「**import 那份比自己写四行麻烦**」。
    ⇒ 所以修法不是提醒大家别复制,是**把复制的诱因去掉**:
    一个一行能导入、带例子、在文档里搜得到的函数,没有人会去内联。
    (拦截那一半由 ``tests/v2/unit/skills/builtins/test_px_to_m_single_source.py``
    负责 —— 但它有射程,见那个文件的说明。)

    ## 约定(``.sxm``)

    * 原点在扫描框**左下**,帧中心 = ``scan_offset``;
    * **y 翻转**:``scan_dir='down'`` 时顶行(row 0)是**最大** y;
    * 先算相对帧中心的偏移,再绕中心转 ``scan_angle``,最后加回中心。
      少了旋转这一步,一张转过 30° 的图上算出的坐标会落在别处 ——
      而调用方拿着它去移动针尖。

    ⚠️ ``angle_deg`` 请从 :func:`parse_xy_meta` 取,并且**把它的
    ``angle_known`` 一起传下去**:角度解析失败时它返回 0.0,而下游
    「是不是转过」的判据通常是 ``abs(angle) > 1`` —— 0.0 恰好让那道提示不触发。
    """
    ca = math.cos(math.radians(angle_deg))
    sa = math.sin(math.radians(angle_deg))
    dx = (px_x + 0.5) / nx * w_m - w_m * 0.5
    dy = h_m * 0.5 - (px_y + 0.5) / ny * h_m
    return (cx_m + dx * ca - dy * sa, cy_m + dx * sa + dy * ca)


def _pick_channel(channels: dict, channel: str, direction: str):
    """Return (name, 2D array) for the requested channel, or None.

    Match priority: exact (case-insensitive) → substring → first channel.
    Direction falls back forward→backward→any.
    """
    if not channels:
        return None
    want = (channel or "").strip().lower()
    name = None
    for k in channels:
        if k.strip().lower() == want:
            name = k
            break
    if name is None and want:
        for k in channels:
            if want in k.strip().lower():
                name = k
                break
    if name is None:
        name = next(iter(channels))
    ch = channels.get(name) or {}
    arr = None
    if isinstance(ch, dict):
        arr = ch.get(direction)
        if arr is None:
            arr = ch.get("forward")
        if arr is None:
            arr = ch.get("backward")
        if arr is None and ch:
            arr = next(iter(ch.values()))
    if arr is None:
        return None
    return name, arr


def load_scan_for_mosaic(path, channel: str = "Z",
                         direction: str = "forward") -> dict | None:
    """Read one .sxm → {data, cx, cy, w, h, angle, nx, ny, channel, path} or None."""
    from mast.io.nanonis_files import read_sxm, rows_top_first

    try:
        res = read_sxm(str(path))
    except Exception as exc:  # noqa: BLE001 — one bad file shouldn't kill a batch
        logger.warning("mosaic: cannot read %s: %s", path, exc)
        return None
    header = res.get("header", {})
    meta = parse_xy_meta(header)
    if meta is None:
        logger.info("mosaic: %s has no SCAN_OFFSET/RANGE; skipped", path)
        return None
    picked = _pick_channel(res.get("channels", {}), channel, direction)
    if picked is None:
        return None
    chan_name, arr = picked
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return None
    # scan_dir correction: Nanonis "up" scans have row 0 at the bottom; flip so
    # row 0 is the high-y (top) edge, matching how we place onto the canvas.
    # The rule itself lives in nanonis_files.rows_top_first — it is the same
    # question the scan-map underlay and sxm_oriented_frames ask, and three
    # private spellings of it is how #94 got in.
    arr = np.asarray(rows_top_first(arr, header.get("scan_dir")), dtype=np.float64)
    ny, nx = arr.shape
    return {**meta, "data": arr, "nx": nx, "ny": ny,
            "channel": chan_name, "path": str(path)}


# ── Display helpers (borrowed from the offline preview tool) ────────────────────

def _line_normalize(data: np.ndarray) -> np.ndarray:
    """Subtract each row's median (kills STM scan-line offset banding)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(data, axis=1, keepdims=True)
    med = np.where(np.isnan(med), 0.0, med)
    return data - med


def _normalize01(data: np.ndarray, percentiles) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, percentiles)
    if lo == hi:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _resample_nn(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resample a 2D array to (out_h, out_w). Good enough for
    an overview canvas, and dependency-free (no PIL/scipy)."""
    out_h = max(1, int(out_h))
    out_w = max(1, int(out_w))
    ys = np.clip((np.arange(out_h) * arr.shape[0] // out_h), 0, arr.shape[0] - 1)
    xs = np.clip((np.arange(out_w) * arr.shape[1] // out_w), 0, arr.shape[1] - 1)
    return arr[ys][:, xs]


def _get_cmap(name: str):
    try:
        from matplotlib import colormaps
        return colormaps.get_cmap(name)
    except Exception:  # noqa: BLE001 — old matplotlib fallback
        from matplotlib import cm
        return cm.get_cmap(name)


# ── Canvas assembly ─────────────────────────────────────────────────────────────

def build_mosaic(scans: list[dict], *, cmap: str = _DEFAULT_CMAP,
                 percentiles=(1.0, 99.0), max_canvas_px: int = _MAX_CANVAS_PX,
                 line_normalize: bool = False, bg=(12, 12, 16)) -> dict:
    """Assemble *scans* (from load_scan_for_mosaic) into one RGB canvas.

    Returns a dict: image (H,W,3 uint8 or None), placed, n_input, extent_m
    (x_min,x_max,y_min,y_max), res_m_per_px, canvas_px (W,H), angle_warning,
    error.
    """
    valid = [s for s in scans if s and s.get("data") is not None
             and s.get("nx", 0) > 0 and s.get("ny", 0) > 0]
    if not valid:
        return {"image": None, "placed": 0, "n_input": len(scans),
                "error": "no valid scans (need .sxm with SCAN_OFFSET/RANGE)"}

    x_min = min(s["cx"] - s["w"] / 2 for s in valid)
    x_max = max(s["cx"] + s["w"] / 2 for s in valid)
    y_min = min(s["cy"] - s["h"] / 2 for s in valid)
    y_max = max(s["cy"] + s["h"] / 2 for s in valid)
    span_x = max(x_max - x_min, 1e-12)
    span_y = max(y_max - y_min, 1e-12)

    # Resolution = median per-scan pixel size, but coarsened if needed so the
    # canvas stays within max_canvas_px on each side.
    px_sizes = [s["w"] / s["nx"] for s in valid if s["nx"] > 0]
    res = float(np.median(px_sizes)) if px_sizes else span_x / 512.0
    res = max(res, span_x / max_canvas_px, span_y / max_canvas_px)

    W = max(1, min(int(np.ceil(span_x / res)), max_canvas_px))
    H = max(1, min(int(np.ceil(span_y / res)), max_canvas_px))
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:, :] = np.array(bg, dtype=np.uint8)

    cmap_fn = _get_cmap(cmap)
    angle_warn = False
    placed = 0
    for s in valid:
        # 角度未知也要提示:这道提示的全部意义是「这一帧的摆放可能不对」,
        # 而「读不到角度」正是最该提示的情形之一。默认 True 是为了兼容
        # 不带该键的旧调用方(它们的角度是真读到的)。
        if abs(s.get("angle", 0.0)) > 1.0 or not s.get("angle_known", True):
            angle_warn = True
        data = s["data"]
        if line_normalize:
            data = _line_normalize(data)
        normed = _normalize01(data, percentiles)
        pw = max(1, int(round(s["w"] / res)))
        ph = max(1, int(round(s["h"] / res)))
        rgb = (cmap_fn(_resample_nn(normed, ph, pw))[:, :, :3] * 255).astype(np.uint8)
        # Top-left pixel of this footprint on the canvas. Canvas row 0 = top =
        # y_max; the scan's top edge is at cy + h/2.
        c0 = int(round((s["cx"] - s["w"] / 2 - x_min) / res))
        r0 = int(round((y_max - (s["cy"] + s["h"] / 2)) / res))
        # Keep edge footprints inside the (possibly clamped) canvas — otherwise
        # a far-right/bottom scan rounds one pixel past the edge and drops out.
        c0 = max(0, min(c0, W - pw))
        r0 = max(0, min(r0, H - ph))
        r1, c1 = r0 + ph, c0 + pw
        cr0, cc0 = max(0, r0), max(0, c0)
        cr1, cc1 = min(H, r1), min(W, c1)
        if cr1 <= cr0 or cc1 <= cc0:
            continue
        canvas[cr0:cr1, cc0:cc1] = rgb[cr0 - r0:cr0 - r0 + (cr1 - cr0),
                                       cc0 - c0:cc0 - c0 + (cc1 - cc0)]
        placed += 1

    return {
        "image": canvas, "placed": placed, "n_input": len(scans),
        "extent_m": (x_min, x_max, y_min, y_max),
        "res_m_per_px": res, "canvas_px": (W, H),
        "angle_warning": angle_warn, "error": "",
    }


def mosaic_from_paths(paths, *, channel: str = "Z", **kw) -> dict:
    """Load each path and build a mosaic. Adds ``scans_meta`` (per-placed info)."""
    scans = [load_scan_for_mosaic(p, channel) for p in paths]
    scans = [s for s in scans if s]
    out = build_mosaic(scans, **kw)
    out["scans_meta"] = [
        {k: s[k] for k in ("path", "cx", "cy", "w", "h", "channel")}
        for s in scans
    ]
    return out


def mosaic_from_dir(directory, *, channel: str = "Z", recursive: bool = False,
                    **kw) -> dict:
    """Build a mosaic from every .sxm in *directory* (optionally recursive)."""
    d = Path(directory)
    paths = sorted(d.rglob("*.sxm") if recursive else d.glob("*.sxm"))
    return mosaic_from_paths(paths, channel=channel, **kw)


# ── Render + persist ────────────────────────────────────────────────────────────

def render_mosaic_figure(mosaic: dict, *, title: str = "") -> "Figure":
    """matplotlib Figure of the canvas with axes in nm (Agg, no pyplot)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.5, 7.0), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    img = mosaic.get("image")
    if img is None:
        ax.text(0.5, 0.5, mosaic.get("error") or "no mosaic",
                ha="center", va="center")
        ax.set_axis_off()
        return fig
    ext = mosaic.get("extent_m") or (0.0, 1.0, 0.0, 1.0)
    # imshow extent = (left, right, bottom, top); origin upper → row 0 at top.
    ax.imshow(img, extent=[ext[0] * 1e9, ext[1] * 1e9, ext[2] * 1e9, ext[3] * 1e9],
              origin="upper", aspect="equal", interpolation="nearest")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    res_nm = mosaic.get("res_m_per_px", 0.0) * 1e9
    ax.set_title(
        title or (f"{mosaic.get('placed', 0)}/{mosaic.get('n_input', 0)} scans · "
                  f"{res_nm:.2f} nm/px"),
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def _mosaic_dir():
    from mast._runtime_paths import project_root
    d = project_root() / "artifacts" / "mosaics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_mosaic(mosaic: dict, *, label: str = "", now=None,
                save_npy: bool = True) -> dict:
    """Write the mosaic to disk: PNG (always) + NPY (canvas) + JSON (metadata).

    Returns {png, [npy], json}. Lands in artifacts/mosaics/.
    """
    import json
    from datetime import datetime

    now = now or datetime.now()
    stamp = now.strftime("%Y%m%dT%H%M%S")
    safe = "".join(c for c in (label or "") if c.isalnum() or c in "-_") or "mosaic"
    base = _mosaic_dir() / f"mosaic_{safe}_{stamp}"

    png_path = base.with_suffix(".png")
    fig = render_mosaic_figure(mosaic, title=label)
    fig.savefig(str(png_path), dpi=120, bbox_inches="tight")
    out = {"png": str(png_path)}

    if save_npy and mosaic.get("image") is not None:
        npy_path = base.with_suffix(".npy")
        np.save(str(npy_path), mosaic["image"])
        out["npy"] = str(npy_path)

    meta: dict[str, Any] = {
        k: mosaic.get(k) for k in
        ("placed", "n_input", "extent_m", "res_m_per_px", "canvas_px",
         "angle_warning")
    }
    meta["label"] = label
    meta["scans_meta"] = mosaic.get("scans_meta", [])
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    out["json"] = str(json_path)
    return out
