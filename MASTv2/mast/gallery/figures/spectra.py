# -*- coding: utf-8 -*-
"""谱的读取、STS 图共用的色标小工具，与左栏的 STM 面板（``frame_img`` / ``stm_panel``）。

dI/dV 三态（T16）：先看头里 ``Lock-in>Lock-in status``（不是 OFF 才考虑 lock-in；没有这一项当 ON），
Y、X 两路里挑与 I(V) 数值导数（按电压定窗口，T31）|r| 最大、且 ≥ 0.3 的一路；都不够就用电流
数值求导并在名字里写明。**不看** ``Bias Spectroscopy>Lock-In run``：它 FALSE 只表示谱模块不去开关锁相，
锁相通道可能独立启用。解析同时识别 ``[AVG]`` 与单次扫描列名
（``LI Demod 1 Y (A)``），带 ``[AVG]`` 的文件结果不变。
"""
from __future__ import annotations

import math
import time

import numpy as np

from mast.gallery.figures import common as C

ROBUST = 1.4826


def dshort(dname: str) -> str:
    """图上用的 dI/dV 短名。"""
    return "dI/dV（LI %s）" % dname.split()[-1] if "LI Demod" in dname else "dI/dV（数值求导）"


def nice(x: float, fallback: float = 1.0) -> float:
    """1-2-5 序列里对数距离最近的值。``x`` 不是正的有限数 ⇒ ``fallback``（原版会抛）。"""
    x = float(x)
    if not (math.isfinite(x) and x > 0):
        return fallback
    e = math.floor(math.log10(x))
    return min((m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1, 2, 5)),
               key=lambda v: abs(math.log10(v / x)))


def noise_sigma(M) -> float:
    """高频噪声：每条谱减 Savitzky–Golay(9, 2) 平滑后的稳健标准差。"""
    from scipy.signal import savgol_filter

    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    M = M[np.isfinite(M).all(axis=1)]
    if not M.size or M.shape[-1] < 9:
        return 0.0
    return ROBUST * float(np.median(np.abs(M - savgol_filter(M, 9, 2, axis=-1))))


def noise_x0(M) -> float:
    """对数色标 / symlog 的 x₀ = 噪声 5σ 就近的 1-2-5 值。

    没有噪声（合成数据）时退到 |x| 99% 分位的千分之一，再退到 1 —— 原版在这两种情形直接抛，真实数据走不到。"""
    sig = noise_sigma(M)
    if sig > 0:
        return nice(5 * sig)
    a = np.abs(np.asarray(M, dtype=np.float64))
    a = a[np.isfinite(a)]
    return nice(1e-3 * float(np.percentile(a, 99))) if a.size else 1.0


def slog(A, x0: float):
    """sign(x)·log₁₀(1+|x|/x₀)。"""
    return np.sign(A) * np.log10(1 + np.abs(A) / x0)


def _first(names, *cands):
    for c in cands:
        if c in names:
            return c
    return None


def load_spectrum(it: dict, cache: dict | None = None) -> dict:
    """``{id V If I Df D dunit dname li_r x y t mt zoff sweeps fn}``；I/D 是正反扫平均，If/Df 是正扫。"""
    key = it["id"]
    if cache is not None and key in cache:
        return cache[key]
    if it.get("ex"):
        raise C.JobError("不是偏压谱（%s）" % it["ex"])
    from mast.io.nanonis_files import read_dat

    res = read_dat(it["p"])
    h = res.get("header") or {}
    cols = res.get("columns") or {}
    names = list(cols)
    if not names:
        raise C.JobError("谱文件里没有数据列")
    vname = (_first(names, "Bias calc (V)", "Bias (V)")
             or next((c for c in names if c.lower().startswith("bias")), names[0]))
    V = np.asarray(cols[vname], dtype=np.float64)

    def fb(name, scale):
        if name not in cols:
            return None, None
        f = np.asarray(cols[name], dtype=np.float64) * scale
        bn = name.replace(" (", " [bwd] (", 1)
        b = np.asarray(cols[bn], dtype=np.float64) * scale if bn in cols else None
        return f, ((f + b) / 2 if b is not None else f)

    If, I = fb("Current [AVG] (A)", 1e12)
    if If is None:
        If, I = fb("Current (A)", 1e12)
    if If is None:
        raise C.JobError("谱里没有电流列")
    ref = C.deriv_by_volts(V, If)
    best = None
    if str(h.get("Lock-in>Lock-in status", "ON")).strip().upper() != "OFF":
        for base in ("LI Demod 1 Y", "LI Demod 1 X"):
            name = _first(names, base + " [AVG] (A)", base + " (A)")
            if name is None:
                continue
            f, avg = fb(name, 1e15)
            with np.errstate(all="ignore"):
                if f is None or not (np.std(f) > 0):
                    continue
                r = abs(float(np.corrcoef(f, ref)[0, 1]))
            if math.isfinite(r) and r >= 0.3 and (best is None or r > best[0]):
                best = (r, name, f, avg)
    if best:
        Df, D, dunit = best[2], best[3], "fA"
        dname = "dI/dV · %s" % best[1].replace(" [AVG] (A)", "").replace(" (A)", "")
    else:
        Df, D, dunit, dname = C.deriv_by_volts(V, If), C.deriv_by_volts(V, I), "pA/V", "dI/dV · 电流数值求导"
    sweeps = h.get("Bias Spectroscopy>Number of sweeps") or it.get("sw") or "?"
    out = dict(id=key, V=V, If=If, I=I, Df=Df, D=D, dunit=dunit, dname=dname,
               li_r=float(best[0]) if best else None,
               x=float(it.get("x") or 0.0), y=float(it.get("y") or 0.0),
               t=it.get("t"), mt=it.get("mt"), zoff=float(it.get("zo") or 0.0),
               sweeps=str(sweeps).strip() if not isinstance(sweeps, (int, float)) else str(int(sweeps)),
               fn=it["fn"])
    if cache is not None:
        cache[key] = out
    return out


def frame_image(f: dict, cache: dict | None = None):
    """STM 面板用的图：逐行调平 + 平面，只取扫到的行；``(z, extent[左,右,下,上] nm 相对帧心)``。"""
    key = ("frame_img", f["id"])
    if cache is not None and key in cache:
        return cache[key]
    z, _bwd = C.load_z(f["p"])
    i0, i1, _n = C.valid_rows(z)
    rall = z.shape[0]
    zz = C.plane(C.row_level(z[i0:i1] * 1e12))
    w = float(f.get("w") or 0.0) or 1.0
    H = float(f.get("hn") or 0.0) or w
    top = H / 2 - i0 / rall * H
    out = (zz, [-w / 2, w / 2, top - H * (i1 - i0) / rall, top])
    if cache is not None:
        cache[key] = out
    return out


def stm_panel(ax, f: dict | None, pts, labels, rings, t_ref, head: str = "", cache=None) -> None:
    """帧 + 谱位置（青点）+ 标注站位（黄圈）+ 标题。"""
    ax.set_facecolor("#17151a")
    if f is None:
        ax.text(0.5, 0.5, "之前没有 STM", ha="center", va="center", color="#dddddd",
                transform=ax.transAxes, fontproperties=C.fp(10))
        ax.set_xticks([])
        ax.set_yticks([])
        return
    z, ext = frame_image(f, cache)
    v = z[np.isfinite(z)]
    lo, hi = (np.percentile(v, [1, 99]) if v.size else (0.0, 1.0))
    ax.imshow(z, cmap="afmhot", extent=ext, vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    P = np.array([C.to_frame(f, x, y) for x, y in pts]) if len(pts) else np.zeros((0, 2))
    if len(P) > 1:
        ax.plot(P[:, 0], P[:, 1], "-", color="#19e0c8", lw=0.9, alpha=0.55)
    if len(P):
        ax.plot(P[:, 0], P[:, 1], "o", ms=4.2, mfc="#19e0c8", mec="#0a0a0a", mew=0.5)
    spots: list[list] = []                          # 同一位置（< 20 pm）的几条谱合成一个标签
    for (u, vv), lab, ring in zip(P, labels, rings):
        for sp in spots:
            if math.hypot(sp[0] - u, sp[1] - vv) < 0.02:
                sp[2].append(lab)
                sp[3] = sp[3] or ring
                break
        else:
            spots.append([u, vv, [lab], ring])
    for u, vv, labs, ring in spots:
        if ring:
            ax.plot(u, vv, "o", ms=13, mfc="none", mec="#ffd24a", mew=1.8)
        labs = [x for x in labs if x]
        if labs:
            ax.text(u, vv, " " + " / ".join(labs), color="#ffffff", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#000000", ec="none", alpha=0.55),
                    fontproperties=C.fp(8))
    xs = [ext[0], ext[1]] + list(P[:, 0])
    ys = [ext[2], ext[3]] + list(P[:, 1])
    pad = 0.15
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    t = f.get("t")
    ax.set_title("%s%s/%s · %s · %s · %s · %g×%g nm\n%s 开扫，比谱早 %s 存盘" % (
        head, C.dir_short(f["d"]), C.num4(f["fn"]), C.fmt_bias(f.get("b")), C.fmt_cur(f.get("sp")),
        ("转角 %g°" % float(f["ang"])) if f.get("ang") else "0°",
        float(f.get("w") or 0.0), float(f.get("hn") or f.get("w") or 0.0),
        time.strftime("%m-%d %H:%M", time.localtime(t)) if t else "?",
        C.dur((t_ref or 0) - (f.get("mt") or 0))),
        color=C.DIM_HEX, linespacing=1.2, fontproperties=C.fp(8.5))


def on_grid(Vref: np.ndarray, V: np.ndarray, y: np.ndarray) -> np.ndarray:
    """把一条谱放到参照偏压格点上：同一套格点原样返回，不同就插值（偏压可能递减）。"""
    if len(V) == len(Vref) and np.allclose(V, Vref, rtol=0, atol=1e-6):
        return y
    o = np.argsort(V)
    return np.interp(Vref, V[o], y[o], left=np.nan, right=np.nan)
