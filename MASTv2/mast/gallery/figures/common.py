# -*- coding: utf-8 -*-
"""出图的共用件（设计文档 §10）。

**复用本体、不另写第二份**：

* matplotlib 绘图锁 ``render._MPL_LOCK``。对数 / symlog 刻度要走 mathtext，它的解析器是进程内
  共享的 pyparsing 对象、不是线程安全的。构建线程出谱图时持同一把锁，
  出图与构建在 matplotlib 绘制阶段串行。
* 中文字体：Pillow 用本体的候选文件表 ``render._CJK_FONT_FILES``；matplotlib 的字体名取自
  ``io.exp_map._cjk_fontprops``（本体 ``render._fp`` 用的同一个来源），这里只补一个 DejaVu 回退，
  免得 ``D₋`` 这类字符在单一字族里缺字（T28）。雅黑里没有 ★ ✓ 字形，评级写成字（:data:`RWORD`）。
* 色表 ``render._lut``；平面 ``vision.scan_prep.poly_subtract(a, 1)`` —— NaN 安全、只用有限像素
  拟合，与 是同一个最小二乘平面。

**不许 pyplot、``rcParams``、``rc_context``**（T7 / T27）。绘图实现在模块级 ``plt.rcParams.update``
改纸色、墨色、字体；这里把颜色写在 ``Figure`` / ``Axes`` 的参数上，中文逐个带 ``fontproperties``。
``rc_context`` 也不行：它改的是进程级字典，不是线程局部的。

**原子写入**一律走 ``mast.gallery.paths``（临时后缀 ``.tmp-``，绝不是 ``.part-``，T2）。
"""
from __future__ import annotations

import bisect
import io
import json
import math
import re
import threading
import time
import warnings
from pathlib import Path

import numpy as np

from mast.gallery import paths as _paths
from mast.gallery import render as _render
from mast.gallery.inventory import dir_label, dir_short

#: 类别：目录名 → 标题。顺序就是列表页的顺序（D16）。
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("frames", "标记帧"), ("grids", "网格谱"), ("sts_lines", "拉线谱"),
    ("sts_stitch", "单根谱拼接"), ("series", "旋转系列"),
)
CATEGORY_TITLE = dict(CATEGORIES)

KIND_CATEGORY = {
    "marked_frames": "frames", "frame_sheet": "frames", "grid_sheets": "grids",
    "sts_lines": "sts_lines", "sts_stitch": "sts_stitch",
    "series_slides": "series", "series_stack": "series",
}

FIGURE_JSON = ".figure.json"
FIGURE_JSON_VERSION = 1
PREVIEW_DIR = ".previews"

#: 与本体构建共用的那一把 matplotlib 锁。
MPL_LOCK = _render._MPL_LOCK

# Pillow 版式的颜色（绘图实现 draw_sheets.py）
BG, INK, DIM = (22, 20, 24), (238, 234, 226), (165, 160, 152)
#: 雅黑里没有 ★ ✓ 字形，用字（绘图实现 draw_sheets.py:48）。
RWORD = {2: "重点", 1: "可用", -1: "排除"}

# matplotlib 版式的颜色（绘图实现 draw_sts.py）
PAPER, INK_HEX, EDGE = "#faf7f2", "#241f1b", "#cfc5b6"
TEAL, AMBER, WARN, ROSE, DIM_HEX = "#0f6b62", "#96650b", "#a8362a", "#b8375f", "#6b6357"
BLUE, GREEN, PURPLE = "#3d7fb5", "#4a7c35", "#6b4c9a"
#: 区组颜色：原版青 / 蓝 / 绿，更多区组时接着循环（D17）。
BLOCK_COLORS = (TEAL, BLUE, GREEN, AMBER, PURPLE, ROSE)


class JobError(Exception):
    """出图任务里「这一项做不了」的原因（中文，给操作员看）。"""


# ── 目录与 figure.json ─────────────────────────────────────────────────


def figures_dir(lay: _paths.Layout) -> Path:
    return lay.state / "figures"


def category_dir(lay: _paths.Layout, cat: str) -> Path:
    return figures_dir(lay) / cat


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_name(text, fallback: str = "figure") -> str:
    """能当 Windows 文件名的基名（保留中文与 ``·`` ``–``）。"""
    s = _BAD_CHARS.sub("_", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip().rstrip(". ")
    return s[:120] or fallback


def clean_json(obj):
    """figure.json 里不许有 NaN / inf，也不许有 numpy 标量。"""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    return obj


def write_bytes(lay: _paths.Layout, cat: str, name: str, data: bytes) -> Path:
    p = category_dir(lay, cat) / name
    _paths.atomic_write_bytes(p, data)
    return p


def write_figure_json(lay: _paths.Layout, cat: str, base: str, *, kind: str, title: str,
                      files: list[str], ids=(), series=(), options=None, summary=None,
                      detail=None, maker_version: int = 1) -> str:
    """写 ``<base>.figure.json``，返回条目 key ``<类别>/<base>``（D16）。

    ``summary`` 的键名原样显示成界面上的小标签：键短、中文，值是数字或短字符串。
    机器要读的完整数字放 ``detail``（列表接口不带它；对账脚本直接读文件）。
    ``files`` 的第一个是主图。"""
    meta = {
        "version": FIGURE_JSON_VERSION, "kind": kind, "category": cat, "base": base,
        "title": title, "created": now_str(), "maker_version": int(maker_version),
        "ids": list(ids), "series": list(series), "options": dict(options or {}),
        "summary": dict(summary or {}), "detail": detail or {}, "files": list(files),
    }
    data = json.dumps(clean_json(meta), ensure_ascii=False, indent=1, allow_nan=False)
    _paths.atomic_write_bytes(category_dir(lay, cat) / (base + FIGURE_JSON), data.encode("utf-8"))
    return f"{cat}/{base}"


# ── Pillow ─────────────────────────────────────────────────────────────

_TL = threading.local()


def pil_font(size: int):
    """任意字号的中文字体（按线程缓存；FreeType face 不宜跨线程共用）。候选文件表是本体那一份。"""
    cache = getattr(_TL, "fonts", None)
    if cache is None:
        cache = _TL.fonts = {}
    f = cache.get(size)
    if f is None:
        from PIL import ImageFont

        for fp in _render._CJK_FONT_FILES:
            try:
                f = ImageFont.truetype(fp, size)
                break
            except OSError:
                continue
        if f is None:
            try:
                f = ImageFont.load_default(size)
            except TypeError:                     # 旧 Pillow 不收字号
                f = ImageFont.load_default()
        cache[size] = f
    return f


def lut(name: str) -> np.ndarray:
    return _render._lut(name)


def plane(z) -> np.ndarray:
    from mast.vision.scan_prep import poly_subtract

    return poly_subtract(np.asarray(z, dtype=np.float64), 1)


def row_level(z) -> np.ndarray:
    """每一行减去这一行的中位数（NaN 忽略）。"""
    a = np.asarray(z, dtype=np.float64)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return a - np.nanmedian(a, axis=1, keepdims=True)


def to_img(a, table: np.ndarray, p=(1, 99)):
    """``(PIL 图, lo, hi)``：按百分位拉伸，NaN 画成底色。照"""
    from PIL import Image

    a = np.asarray(a, dtype=np.float64)
    v = a[np.isfinite(a)]
    if v.size:
        lo, hi = (float(x) for x in np.percentile(v, p))
    else:
        lo, hi = 0.0, 1.0
    hi = hi if hi > lo else lo + 1e-12
    with np.errstate(all="ignore"):
        u = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    rgb = table[u]
    rgb[~np.isfinite(a)] = BG
    return Image.fromarray(rgb), lo, hi


def png_bytes(img, *, optimize: bool = True) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=optimize)
    return buf.getvalue()


def jpeg_bytes(img, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def tag(d, xy, text: str, font=None) -> None:
    font = font or pil_font(12)
    x, y = xy
    w = d.textlength(text, font=font)
    size = getattr(font, "size", 12)
    d.rectangle([x - 2, y - 1, x + w + 3, y + size + 3], fill=(0, 0, 0))
    d.text((x, y - 1), text, fill=INK, font=font)


def caption(d, W: int, y: int, left: str, right: str) -> int:
    """底部说明：左边大字，右边小字；放不下就换到第二行。返回占用高度（照绘图实现）。"""
    f_cap, f_sub = pil_font(17), pil_font(13)
    lw = d.textlength(left, font=f_cap)
    rw = d.textlength(right, font=f_sub)
    d.text((6, y + 4), left, fill=INK, font=f_cap)
    if lw + rw + 24 <= W:
        d.text((W - 6 - rw, y + 8), right, fill=DIM, font=f_sub)
        return 30
    d.text((6, y + 28), right, fill=DIM, font=f_sub)
    return 50


# ── 文字格式（照绘图实现）───────────────────────────────────────────────


def fmt_bias(b) -> str:
    if b is None:
        return "?"
    b = float(b)
    sign = "−" if b < 0 else "+"
    return "%s%.3g mV" % (sign, abs(b) * 1e3) if abs(b) < 0.1 else "%s%.2f V" % (sign, abs(b))


def fmt_cur(pa) -> str:
    if pa is None:
        return "?"
    pa = float(pa)
    return "%.3g nA" % (pa / 1e3) if abs(pa) >= 1000 else ("%.0f pA" % pa if abs(pa) >= 1 else "%.2f pA" % pa)


def fmt_size(w, h) -> str:
    w = float(w or 0.0)
    return "%g×%g nm" % (round(w, 3), round(float(h) if h else w, 3))


def num4(fn: str) -> str:
    """文件名扩展名前的最后 4 个字符（帧编号）。"""
    stem = fn.rsplit(".", 1)[0] if "." in fn else fn
    return stem[-4:]


def stem(fn: str) -> str:
    return fn.rsplit(".", 1)[0] if "." in fn else fn


def dur(s) -> str:
    s = abs(float(s))
    return "%.0f 秒" % s if s < 90 else ("%.0f 分" % (s / 60) if s < 5400 else "%.1f 小时" % (s / 3600))


def local(t, fmt: str) -> str:
    return time.strftime(fmt, time.localtime(t)) if t else "?"


# ── matplotlib（Figure + Agg，无 pyplot）────────────────────────────────


def new_figure(figsize, facecolor: str = PAPER):
    """不经 pyplot 的 Figure（不进 ``Gcf``，与本体 ``scan_preview._new_figure`` 同一做法）。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, facecolor=facecolor)
    FigureCanvasAgg(fig)
    return fig


_FAMILY: list = []


def _cjk_family() -> list[str]:
    if not _FAMILY:
        names: list[str] = []
        try:
            from mast.io.exp_map import _cjk_fontprops

            base = _cjk_fontprops()
            if base is not None:
                names = [str(n) for n in base.get_family()]
        except Exception:  # noqa: BLE001 — 字体缺失不该让出图失败
            names = []
        _FAMILY.append(names)
    return list(_FAMILY[0])


def fp(size: float, *, mono: bool = False):
    """按 artist 给的字体（中文字族 + DejaVu 回退）。"""
    from matplotlib.font_manager import FontProperties

    fam = (["Consolas"] if mono else []) + _cjk_family() + (
        ["DejaVu Sans Mono"] if mono else ["DejaVu Sans"])
    return FontProperties(family=fam, size=size)


def style_axes(ax, *, labelsize: float = 11) -> None:
    """绘图实现 rcParams 的那几项，写在这一个 Axes 上。"""
    ax.set_facecolor(PAPER)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
    ax.tick_params(colors=INK_HEX, labelsize=labelsize)


def savefig_bytes(fig, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return buf.getvalue()


# ── 索引、帧时间线、坐标 ───────────────────────────────────────────────


def by_id(doc: dict) -> dict:
    return {it["id"]: it for it in (doc.get("items") or [])}


class Timeline:
    """有保存时刻、且不是重复保存的帧，按保存时刻排。"""

    def __init__(self, items) -> None:
        self.frames = sorted((it for it in items if it.get("k") == "f" and it.get("mt")
                              and not it.get("dup")), key=lambda it: it["mt"])
        self._keys = [f["mt"] for f in self.frames]

    def last_before(self, t):
        """谱开始 ``t`` 之前（+1 s 容差）最后保存的那张帧。"""
        if t is None:
            return None
        i = bisect.bisect_right(self._keys, float(t) + 1) - 1
        return self.frames[i] if i >= 0 else None


def to_frame(f: dict, x: float, y: float) -> tuple[float, float]:
    """压电坐标 (nm) → 该帧的帧内坐标 (nm，相对帧心，x′ 右 y′ 上)。
    SCAN_ANGLE 为正 = 扫描框相对压电坐标顺时针转（T6）。照"""
    th = math.radians(float(f.get("ang") or 0.0))
    dx, dy = float(x) - float(f.get("cx") or 0.0), float(y) - float(f.get("cy") or 0.0)
    return dx * math.cos(th) - dy * math.sin(th), dx * math.sin(th) + dy * math.cos(th)


def load_z(path: str) -> tuple[np.ndarray, bool]:
    """``(定向后的 Z 帧 float64（行 0 = 上沿）, 是否只有反扫)``。

    方向只有 ``sxm_oriented_frames`` 一个落点（T4 / T29）；没有正扫时它把去镜像后的反扫当
    ``forward`` 交回来，这里另外看一眼原始通道判断是不是反扫。"""
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

    scan = read_sxm(path)
    chans = scan.get("channels") or {}
    zc = chans.get("Z")
    if zc is None:
        low = {str(k).lower(): k for k in chans}
        zc = chans.get(low.get("z")) if "z" in low else None
    fr = sxm_oriented_frames(scan, "Z")
    z = fr.get("forward")
    if z is None:
        raise JobError("文件里没有 Z 通道")
    bwd_only = bool(zc) and zc.get("forward") is None and zc.get("backward") is not None
    return np.asarray(z, dtype=np.float64), bwd_only


def valid_rows(z: np.ndarray) -> tuple[int, int, int]:
    """``(i0, i1, 整行有限的行数)``：第一个到最后一个整行有限的行（绘图实现同一定义）。"""
    ok = np.isfinite(z).all(axis=1)
    if not ok.any():
        raise JobError("一整行都没扫到")
    i0 = int(np.argmax(ok))
    i1 = int(len(ok) - np.argmax(ok[::-1]))
    return i0, i1, int(ok.sum())


# ── 数值导数（T31：出图按电压定窗口，与缩略图按点数定窗口是两个函数）────────


def deriv_by_volts(V, I, window_v: float = 0.08) -> np.ndarray:
    """I(V) 的数值导数，Savitzky–Golay 窗口按**电压**定（≈ ``window_v``，二次多项式）。

    缩略图那一份是 ``render.num_deriv``（窗口 ≈ 点数/25）——
    两条规则都照原版，**不合并**：合并必然改掉其中一个的输出。"""
    V = np.asarray(V, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    n = len(V)
    if n < 2:
        return np.zeros(n)
    dV = np.diff(V)
    if n >= 7 and np.allclose(dV, dV.mean(), rtol=1e-2, atol=1e-9):
        from scipy.signal import savgol_filter

        w = max(5, int(round(window_v / abs(float(dV.mean())))) | 1)
        w = min(w, n - 1 if (n - 1) % 2 else n - 2)
        return savgol_filter(I, w, 2, deriv=1, delta=float(dV.mean()))
    return np.gradient(I, V)


def dir_title(d: str) -> str:
    return dir_label(d)


__all__ = [
    "CATEGORIES", "CATEGORY_TITLE", "KIND_CATEGORY", "FIGURE_JSON", "PREVIEW_DIR", "MPL_LOCK",
    "JobError", "figures_dir", "category_dir", "safe_name", "write_bytes", "write_figure_json",
    "pil_font", "lut", "plane", "row_level", "to_img", "png_bytes", "jpeg_bytes", "tag", "caption",
    "fmt_bias", "fmt_cur", "fmt_size", "num4", "stem", "dur", "local", "new_figure", "fp",
    "style_axes", "savefig_bytes", "by_id", "Timeline", "to_frame", "load_z", "valid_rows",
    "deriv_by_volts", "dir_short", "dir_title",
]
