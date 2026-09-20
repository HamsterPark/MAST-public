# -*- coding: utf-8 -*-
"""缩略图：帧 JPG、lock-in 帧 JPG、谱 PNG、网格谱 PNG（设计文档 D7）。

**版式照旧版兼容格式**，前端的几何换算依赖它（陷阱 T5）：

* 帧图宽 :data:`THUMB_W` px，图像区高 = ``max(int(THUMB_W × 显示行数 / nx), 8)``，下面接
  :data:`STRIP_PX` px 的两行信息条；图像区画的是**定向后**第 ``[r0, r1)`` 行（第一个到最后一个
  整行有限的行）。前端把谱的 (u, v) 画到这张图上时用的就是这三个数 —— 改这里的版式必须
  同步改 ``frontend/src/lib/gallery/context.ts`` 的 ``STRIP_PX``（有测试对账两边的字面量）。
* 帧的方向一律走 ``io.nanonis_files.sxm_oriented_frames``（行 0 = 帧的上沿、bwd 已去镜像），
  这里**不写第二份翻转**（陷阱 T4）。

线程安全（陷阱 T7）：帧图走 Pillow；谱与网格图走 ``Figure`` + ``FigureCanvasAgg``
（``webui.scan_preview._new_figure``），**不碰 pyplot**，不改全局 rcParams；中文字体按 artist 给。
PIL 的字体对象按线程各建一份（FreeType face 不宜跨线程共用）。

版本号：改了渲染方式就把对应的 ``VER_*`` 加 1，下次构建自动重出。
"""
from __future__ import annotations

import io
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from mast.gallery import paths as _paths
from mast.gallery.inventory import dir_key, dir_short

THUMB_W = 400
STRIP_PX = 34
#: 有效行少于这个数的帧不出图。
EMPTY_ROWS = 8
#: 视野不大于这个宽度（nm）时逐行去中位偏置；更大的视野只减平面。
LINE_LEVEL_MAX_NM = 30.0

VER_FRAME = 1
VER_LI = 1
VER_STS = 1
VER_GRID = 1


def thumb_rel(item_id: str, suffix: str) -> str:
    """缩略图相对 ``thumbs/`` 的路径：``<id><suffix>``（id 本身就是 ``<根名>/<相对路径>``）。

    保留原文件的扩展名（``x.sxm.jpg`` / ``x.dat.png``），同目录里同名不同类型的文件不会撞。"""
    return item_id + suffix


def _save_bytes(thumbs: Path, rel: str, data: bytes) -> None:
    _paths.atomic_write_bytes(Path(thumbs, *rel.split("/")), data)


# ── 帧图（Pillow）──────────────────────────────────────────────────────

_TL = threading.local()
_LUTS: dict[str, np.ndarray] = {}
_LUT_LOCK = threading.Lock()

_FONT_FILES = (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf",
               "DejaVuSansMono.ttf", "DejaVuSans.ttf")
_CJK_FONT_FILES = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                   "NotoSansCJK-Regular.ttc")


def _fonts():
    f = getattr(_TL, "fonts", None)
    if f is None:
        from PIL import ImageFont

        def load(files, size):
            for fp in files:
                try:
                    return ImageFont.truetype(fp, size)
                except OSError:
                    continue
            return None

        big, small = load(_FONT_FILES, 13), load(_FONT_FILES, 11)
        if big is None or small is None:
            big = small = ImageFont.load_default()
        cbig, csmall = load(_CJK_FONT_FILES, 13), load(_CJK_FONT_FILES, 11)
        f = _TL.fonts = (big, small, cbig or big, csmall or small)
    return f


def _font(text: str, size: str):
    big, small, cbig, csmall = _fonts()
    ascii_only = all(ord(c) < 128 for c in text)
    if size == "big":
        return big if ascii_only else cbig
    return small if ascii_only else csmall


def _lut(name: str) -> np.ndarray:
    with _LUT_LOCK:
        lut = _LUTS.get(name)
        if lut is None:
            from mast.io.mosaic import _get_cmap

            lut = (np.asarray(_get_cmap(name)(np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)
            _LUTS[name] = lut
    return lut


def _stretch(a: np.ndarray, lo_pct: float, hi_pct: float, eps: float):
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    hi = hi if hi > lo else lo + eps
    u8 = np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return u8, float(lo), float(hi)


def _canvas_jpeg(u8: np.ndarray, lut: np.ndarray, w_nm: float, line1: str, line2: str) -> bytes:
    """图像区 + 34 px 信息条 + 比例尺。照旧版兼容格式 ``_canvas``。"""
    from PIL import Image, ImageDraw

    ny, nx = u8.shape
    img = Image.fromarray(lut[u8])
    hh = max(int(THUMB_W * ny / nx), 8)
    resample = Image.Resampling.BILINEAR if nx >= THUMB_W else Image.Resampling.NEAREST
    img = img.resize((THUMB_W, hh), resample)
    canvas = Image.new("RGB", (THUMB_W, hh + STRIP_PX), (24, 22, 20))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((5, hh + 2), line1, fill=(240, 236, 228), font=_font(line1, "big"))
    d.text((5, hh + 17), line2, fill=(170, 165, 155), font=_font(line2, "small"))
    if w_nm and w_nm > 0:
        sb = 10 ** math.floor(math.log10(w_nm / 5))
        ratio = w_nm / 5 / sb
        sb = sb * (1 if ratio < 2 else (2 if ratio < 5 else 5))
        px = int(sb / w_nm * THUMB_W)
        d.rectangle([THUMB_W - px - 10, hh - 12, THUMB_W - 10, hh - 8], fill=(255, 255, 255))
        label = "%g nm" % sb
        d.text((THUMB_W - px - 10, hh - 28), label, fill=(255, 255, 255), font=_font(label, "small"))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=86)
    return buf.getvalue()


def _stamp(t0) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(t0)) if t0 else "?"


def _num4(fn: str) -> str:
    """旧版兼容格式信息条第一段：扩展名前的最后 4 个字符（帧编号）。"""
    stem = fn.rsplit(".", 1)[0] if "." in fn else fn
    return stem[-4:]


def render_frame(path: str, item_id: str, summ: dict, thumbs: Path, *,
                 want_z: bool = True, want_li: bool = True, scan: dict | None = None) -> dict:
    """一帧的 Z 图（``<id>.jpg``）与 lock-in 图（``<id>.li.jpg``）。

    返回 ``ok why rows rows_all r0 r1 th v``，以及出了 LI 图时的 ``li li_ch vli``。
    ``want_z=False`` 时只重出 LI（Z 图记录由调用方保留），但仍从 Z 算显示行。
    """
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.scan_prep import poly_subtract

    if scan is None:
        scan = read_sxm(path)
    fr = sxm_oriented_frames(scan, "Z")
    z = fr.get("forward")
    if z is None:
        return {"ok": False, "why": "no Z", "v": VER_FRAME}
    z = np.asarray(z, dtype=np.float64)
    ok_rows = np.isfinite(z).all(axis=1)
    n_ok = int(ok_rows.sum())
    if n_ok < EMPTY_ROWS:
        return {"ok": False, "why": "empty", "rows": n_ok, "rows_all": int(len(ok_rows)),
                "v": VER_FRAME}
    i0 = int(np.argmax(ok_rows))
    i1 = int(len(ok_rows) - np.argmax(ok_rows[::-1]))
    w_nm = float(summ.get("w_nm") or 1.0)
    fn = item_id.rsplit("/", 1)[-1]
    tstr = _stamp(summ.get("t0"))
    short = dir_short(dir_key(item_id))
    res: dict = {"ok": True, "why": "", "rows": n_ok, "rows_all": int(len(ok_rows)),
                 "r0": i0, "r1": i1, "th": thumb_rel(item_id, ".jpg"), "v": VER_FRAME}

    if want_z:
        zz = z[i0:i1].copy()
        zz = np.where(np.isfinite(zz), zz, np.nanmedian(zz))
        if w_nm <= LINE_LEVEL_MAX_NM:
            zz = zz - np.median(zz, axis=1, keepdims=True)
        zz = poly_subtract(zz, 1)
        u8, lo, hi = _stretch(zz, 1, 99, 1e-12)
        line1 = "%s  %s  %.4gnm  %+.2fV  %.0fpA" % (
            _num4(fn), tstr, w_nm, summ.get("bias_V") or 0, summ.get("setpoint_pA") or 0)
        line2 = "%s  %dpx  rows %d/%d  z %.0f..%.0f pm  ang %.0f" % (
            short, summ.get("nx") or 0, n_ok, len(ok_rows), lo * 1e12, hi * 1e12,
            summ.get("angle") or 0)
        _save_bytes(thumbs, res["th"], _canvas_jpeg(u8, _lut("afmhot"), w_nm, line1, line2))

    if want_li:
        # 挑行内起伏大的那一路：相位对的那一路有衬度，另一路只剩噪声与直流偏置。
        best = None
        for c in (scan.get("channels") or {}):
            if not str(c).upper().startswith("LI_DEMOD_1"):
                continue
            a = sxm_oriented_frames(scan, c).get("forward")
            if a is None:
                continue
            a = np.asarray(a, dtype=np.float64)[i0:i1]
            if not np.isfinite(a).any():
                continue
            a = np.where(np.isfinite(a), a, np.nanmedian(a)) * 1e15
            s = float(np.median(np.std(a - np.median(a, axis=1, keepdims=True), axis=1)))
            if best is None or s > best[0]:
                best = (s, str(c), a)
        res["vli"] = VER_LI
        res["li"] = ""
        if best is not None:
            _s, c, a = best
            if w_nm <= LINE_LEVEL_MAX_NM:
                a = a - np.median(a, axis=1, keepdims=True)
            u8, lo, hi = _stretch(a, 1, 99, 1e-9)
            line1 = "%s  %s  %s  %+.2fV" % (_num4(fn), tstr, c.replace("_", " "),
                                            summ.get("bias_V") or 0)
            line2 = "%s  dI/dV lock-in%s  %.3g..%.3g fA  rows %d/%d" % (
                short, " line-lev" if w_nm <= LINE_LEVEL_MAX_NM else "", lo, hi, n_ok, len(ok_rows))
            res["li"] = thumb_rel(item_id, ".li.jpg")
            res["li_ch"] = c
            _save_bytes(thumbs, res["li"], _canvas_jpeg(u8, _lut("viridis"), w_nm, line1, line2))
    return res


# ── 数值导数 ───────────────────────────────────────────────────────────


def num_deriv(V, I):
    """I(V) 的数值导数（陷阱 T17）。``(dI/dV, 窗口点数)``，窗口 0 表示没有平滑。

    Savitzky–Golay：窗口约为扫描点数的 1/25、至少 5 点、奇数；二或三次多项式；
    ``mode="interp"``（SciPy 默认）在两端各拟合一个多项式 —— **不补零**，端点不会造出假峰。
    点太少或偏压不等间隔时退回 ``np.gradient``。"""
    V = np.asarray(V, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    n = len(V)
    if n < 2:
        return np.zeros(n), 0
    dV = np.diff(V)
    w = max(5, int(round(n / 25)) | 1)
    if w >= n:
        w = n - 1 if (n - 1) % 2 else n - 2
    if w >= 5 and np.allclose(dV, dV.mean(), rtol=1e-2, atol=1e-9):
        from scipy.signal import savgol_filter

        return savgol_filter(I, w, 3 if w >= 7 else 2, deriv=1, delta=float(dV.mean())), w
    return np.gradient(I, V), 0


# ── 谱与网格（Figure + Agg，无 pyplot）───────────────────────────────────


#: matplotlib 的 mathtext 解析器（对数坐标的刻度标签 ``$\mathdefault{10^{-3}}$`` 要用它）是进程内
#: 共享的 pyparsing 对象，**不是线程安全的**。谱图与网格图从读数到 savefig
#: 整段持同一把锁串行，防止并发 mathtext 解析错误。帧图（Pillow）不受影响。
_MPL_LOCK = threading.Lock()


def _mpl_serialised(fn):
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _MPL_LOCK:
            return fn(*args, **kwargs)

    return wrapper


def _figure(figsize, **kw):
    from mast.webui.scan_preview import _new_figure

    return _new_figure(figsize, **kw)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)          # dpi 写在 savefig 上（T7）
    return buf.getvalue()


def _fp(size: float) -> dict:
    """中文文字的字体参数：按 artist 给，不动全局 rcParams（T7）。"""
    try:
        from mast.io.exp_map import _cjk_fontprops

        fp = _cjk_fontprops()
    except Exception:  # noqa: BLE001
        fp = None
    if fp is None:
        return {"fontsize": size}
    p = fp.copy()
    p.set_size(size)
    return {"fontproperties": p}


def _pick(names: list[str], cols: dict, prefix: str):
    """旧版兼容格式 ``pick``：前缀匹配的列里，正扫取单次（没有就取 AVG），反扫取单次；多列取平均。"""
    one = [c for c in names if c.startswith(prefix) and "bwd" not in c and "AVG" not in c]
    avg = [c for c in names if c.startswith(prefix) and "bwd" not in c and "AVG" in c]
    bw = [c for c in names if c.startswith(prefix) and "bwd" in c and "AVG" not in c]
    f = one or avg
    fwd = np.mean([np.asarray(cols[c], dtype=np.float64) for c in f], axis=0) if f else None
    bwd = np.mean([np.asarray(cols[c], dtype=np.float64) for c in bw], axis=0) if bw else None
    return fwd, bwd


@_mpl_serialised
def render_spectrum(path: str, item_id: str, summ: dict, thumbs: Path) -> dict:
    """点谱 PNG。返回 ``ok why th li didv v``，出了 lock-in 下半图时另有 ``li_ch``。

    dI/dV 三态（陷阱 T16）：lock-in 的 X、Y 两路里挑与 I(V) 数值导数 |r| 最大的一路，
    两路都 < 0.3 视为没开；没有可用 lock-in 时画 I(V) 的数值导数并在图里注明；
    没有电流列（Z 噪声谱之类）画各列对第一列。"""
    from mast.io.nanonis_files import read_dat

    res = read_dat(path)
    cols = res.get("columns") or {}
    names = list(cols)
    h = res.get("header") or {}
    rel = thumb_rel(item_id, ".png")
    if not names or len(np.asarray(cols[names[0]])) == 0:
        return {"ok": False, "why": "empty", "v": VER_STS}
    V = np.asarray(cols[names[0]], dtype=np.float64)
    stem = item_id.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    tstr = _stamp(summ.get("t0"))
    I, Ib = _pick(names, cols, "Current")

    if I is None:
        fig, ax = _figure((4.2, 3.3))
        logx = "Frequency" in names[0]
        for j in range(1, min(len(names), 5)):
            y = np.asarray(cols[names[j]], dtype=np.float64)
            use_log = logx and bool((V > 0).all()) and bool((y > 0).all())
            (ax.loglog if use_log else ax.plot)(V, y, lw=1.0, label=names[j][:30])
        ax.set_xlabel(names[0], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.set_title("%s  %s\n%s  n=%d  (not a bias spectrum)" % (
            stem, tstr, h.get("Experiment", ""), summ.get("n") or len(V)), fontsize=7.5)
        fig.tight_layout(pad=0.4)
        _save_bytes(thumbs, rel, _png(fig))
        return {"ok": True, "why": "", "th": rel, "li": False, "didv": "", "v": VER_STS}

    li = None
    with np.errstate(all="ignore"):
        didv = np.gradient(I, V) if len(V) > 2 else None
        ok_d = didv is not None and bool(np.isfinite(didv).all()) and float(np.std(didv)) > 0
        for pre in ("LI Demod 1 X", "LI Demod 1 Y"):
            a, ab = _pick(names, cols, pre)
            if a is None or not ok_d or not np.isfinite(a).all() or np.std(a) == 0:
                continue
            c = abs(float(np.corrcoef(a, didv)[0, 1]))
            if li is None or c > li[0]:
                li = (c, pre, a, ab)
    if li is not None and not (li[0] >= 0.3):
        li = None
    nd = None
    if li is None and len(V) >= 3 and np.isfinite(I).all():
        dI, w = num_deriv(V, I * 1e12)
        dIb = num_deriv(V, Ib * 1e12)[0] if Ib is not None and np.isfinite(Ib).all() else None
        nd = (dI, dIb, w)

    two = li is not None or nd is not None
    if two:
        fig, axs = _figure((4.2, 4.0), nrows=2, sharex=True,
                           gridspec_kw=dict(height_ratios=[1.15, 1]))
        ax = axs[0]
    else:
        fig, ax = _figure((4.2, 3.3))
        axs = None
    ax.plot(V, I * 1e12, color="#0f6b62", lw=1.2, label="fwd")
    if Ib is not None:
        ax.plot(V, Ib * 1e12, color="#a8362a", lw=0.9, alpha=0.8, label="bwd")
    with np.errstate(all="ignore"):
        ax2 = ax.twinx()
        ax2.semilogy(V, np.abs(I * 1e12 - np.median(I * 1e12)) + 1e-3, color="#3d7fb5",
                     lw=0.7, alpha=0.6)
    ax2.tick_params(labelsize=6, colors="#3d7fb5")
    ax.set_ylabel("I (pA)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    ax.tick_params(labelsize=7)
    out = {"ok": True, "why": "", "th": rel, "li": li is not None,
           "didv": "li" if li is not None else ("num" if nd is not None else ""), "v": VER_STS}
    if li is not None:
        cr, pre, a, ab = li
        axs[1].plot(V, a * 1e15, color="#8a5c07", lw=1.1)
        if ab is not None:
            axs[1].plot(V, ab * 1e15, color="#8a5c07", lw=0.8, ls="--", alpha=0.7)
        axs[1].set_ylabel(pre.replace("LI Demod 1 ", "LI ") + " (fA)", fontsize=8)
        axs[1].tick_params(labelsize=7)
        axs[1].text(0.02, 0.88, "r(LI, dI/dV num) = %.2f" % cr, transform=axs[1].transAxes,
                    fontsize=6.5, color="#555")
        out["li_ch"] = pre
        out["li_r"] = round(cr, 3)
    elif nd is not None:
        dI, dIb, w = nd
        axs[1].plot(V, dI, color="#6d28d9", lw=1.1)
        if dIb is not None:
            axs[1].plot(V, dIb, color="#6d28d9", lw=0.8, ls="--", alpha=0.7)
        axs[1].set_ylabel("dI/dV numerical (pA/V)", fontsize=8)
        axs[1].tick_params(labelsize=7)
        note = ("no lock-in: d(I)/dV, Savitzky-Golay %d pts = %.2f V"
                % (w, abs(V[-1] - V[0]) * (w - 1) / max(len(V) - 1, 1))) if w \
            else "no lock-in: d(I)/dV, np.gradient"
        axs[1].text(0.02, 0.88, note, transform=axs[1].transAxes, fontsize=6.5, color="#555")
        out["sg_window"] = int(w)
    (axs[1] if two else ax).set_xlabel("V", fontsize=8)
    ax.set_title("%s  %s\nn=%d  %.2f..%.2f V  Zoff %.0f pm  %s sweep  pos (%.2f, %.2f)" % (
        stem, tstr, summ.get("n") or len(V), summ.get("Vmin") or 0, summ.get("Vmax") or 0,
        summ.get("zoff_pm") or 0, summ.get("sweeps") or "?", summ.get("x_nm") or 0,
        summ.get("y_nm") or 0), fontsize=7.5)
    fig.tight_layout(pad=0.4)
    _save_bytes(thumbs, rel, _png(fig))
    return out


def _first_index(ch: list[str], *candidates: str) -> int | None:
    for name in candidates:
        if name in ch:
            return ch.index(name)
    return None


@_mpl_serialised
def render_grid(path: str, item_id: str, summ: dict, thumbs: Path) -> dict:
    """网格谱 PNG：扫满的行够多时 2×5（形貌 + 8 个偏压的 dI/dV 图 + 平均谱），否则只画已有点的谱。"""
    from mast.gallery.grid3ds import read_3ds

    G = read_3ds(path)
    ch = G["chans"]
    have, nx, ny, n = G["have"], G["nx"], G["ny"], G["npts"]
    if have == 0:
        return {"ok": False, "why": "empty", "v": VER_GRID}
    D, P = G["D"], G["P"]
    rel = thumb_rel(item_id, ".png")
    v0, v1 = summ.get("v0") or 0.0, summ.get("v1") or 0.0

    with np.errstate(all="ignore"):
        jb = [j for j, c in enumerate(ch) if c.startswith("Bias [AVG]") and "bwd" not in c]
        if not jb:
            jb = [j for j, c in enumerate(ch) if c == "Bias (V)"]
        if jb:
            V = np.nanmedian(D[:, :, jb[0], :].reshape(-1, n).astype(np.float64), axis=0)
        else:
            V = np.linspace(v0, v1, n)
        if not np.isfinite(V).all():
            V = np.linspace(v0, v1, n)
        jI = _first_index(ch, "Current [AVG] (A)", "Current (A)")
        didv = None
        if jI is not None and n > 2:
            Iav = np.nanmean(D[:, :, jI, :].reshape(-1, n).astype(np.float64), axis=0)
            if np.isfinite(Iav).all() and np.std(Iav) > 0:
                didv = np.gradient(Iav, V)
        best = None
        for pair in (("LI Demod 1 X [AVG] (A)", "LI Demod 1 X (A)"),
                     ("LI Demod 1 Y [AVG] (A)", "LI Demod 1 Y (A)")):
            j = _first_index(ch, *pair)
            if j is None:
                continue
            M = D[:, :, j, :].astype(np.float64)
            sp = np.nanmean(M.reshape(-1, n), axis=0)
            c = (abs(float(np.corrcoef(sp, didv)[0, 1]))
                 if didv is not None and np.isfinite(sp).all() and np.std(sp) > 0 else 0.0)
            if best is None or c > best[0]:
                best = (c, ch[j], M * 1e15)
        num_used = False
        if (best is None or not (best[0] >= 0.3)) and jI is not None and n >= 3:
            best = (1.0, "数值 dI/dV",
                    np.gradient(D[:, :, jI, :].astype(np.float64) * 1e12, V, axis=2))
            num_used = True

    zi = G["pars"].index("Z (m)") if "Z (m)" in G["pars"] else None
    stem = item_id.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    t = _stamp(summ.get("t0"))
    t1 = _stamp(summ.get("t1")) if summ.get("t1") else "?"
    title = "%s · %d×%d · %.3g nm · %d 点 %+.2f→%+.2f V · %s sweep · Zoff %.0f pm · %+.2f V/%.0f pA · %s→%s · 完成 %d/%d" % (
        stem, nx, ny, G["w_nm"], n, V[0], V[-1], summ.get("sweeps") or "?",
        summ.get("zoff_pm") or 0, summ.get("bias_V") or 0, summ.get("setpoint_pA") or 0,
        t, t1, have, nx * ny)
    full_rows = have // nx
    if full_rows >= max(4, ny // 4) and best is not None:
        fig, AX = _figure((8.4, 4.3), nrows=2, ncols=5,
                          gridspec_kw=dict(hspace=0.28, wspace=0.08))
        rows = slice(0, full_rows)
        h_nm = G["h_nm"] or G["w_nm"]
        ext = [0, G["w_nm"], 0, h_nm * full_rows / ny]
        with np.errstate(all="ignore"):
            if zi is not None:
                Z = P[rows, :, zi].astype(np.float64) * 1e12
                Z = Z - np.nanmedian(Z, axis=1, keepdims=True)     # 一张网格扫十几小时：按行去偏
                Z = np.where(np.isfinite(Z), Z, 0.0)
                lo, hi = np.percentile(Z, [2, 98])
                AX[0, 0].imshow(Z, origin="lower", cmap="afmhot", vmin=lo, vmax=hi, extent=ext)
                AX[0, 0].set_title("形貌 Z（按行去偏）%.0f pm" % (hi - lo), **_fp(8))
            idx = np.unique(np.round(np.linspace(0, n - 1, 8)).astype(int))
            cells = [AX[0, 1], AX[0, 2], AX[0, 3], AX[0, 4], AX[1, 1], AX[1, 2], AX[1, 3], AX[1, 4]]
            cr, name, M = best
            for ax, j in zip(cells, idx):
                A = M[rows, :, j]
                lo, hi = np.nanpercentile(A, [2, 98])
                ax.imshow(A, origin="lower", cmap="viridis", vmin=lo, vmax=hi, extent=ext)
                ax.set_title("%+.2f V" % V[j], fontsize=8,
                             color="#0f6b62" if V[j] > 0 else "#a8362a")
            for ax in AX.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            ax = AX[1, 0]
            spec = np.nanmean(M[rows].reshape(-1, n), axis=0)
        ax.plot(V, spec, color="#8a5c07", lw=1.2)
        ax.set_xticks([float(V.min()), 0, float(V.max())])
        ax.tick_params(labelsize=6)
        short = name.replace(" [AVG] (A)", "").replace(" (A)", "").replace("LI Demod 1 ", "LI ")
        ax.set_title(("平均 %s（pA/V）" % name) if num_used else ("平均 %s  r=%.2f" % (short, cr)),
                     **_fp(8))
        fig.suptitle(title, **_fp(7.6))
        fig.subplots_adjust(top=0.86, bottom=0.07, left=0.02, right=0.99)
    else:
        fig, ax = _figure((8.4, 3.2))
        if best is not None:
            Mi = best[2]
        elif jI is not None:
            Mi = D[:, :, jI, :].astype(np.float64) * 1e12
        else:
            Mi = None
        if Mi is not None:
            for s in Mi.reshape(-1, n)[:have]:
                ax.plot(V, s, lw=0.6, alpha=0.5)
        ax.set_xlabel("V", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title + "\n（未扫完：只画已有点的谱）", **_fp(7.6))
        fig.tight_layout()
    _save_bytes(thumbs, rel, _png(fig))
    return {"ok": True, "why": "", "th": rel, "v": VER_GRID}


def thumb_exists(thumbs: Path, rel: str | None) -> bool:
    return bool(rel) and os.path.isfile(Path(thumbs, *str(rel).split("/")))
