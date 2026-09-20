# -*- coding: utf-8 -*-
"""网格谱逐层页。

一个 .3ds 一张图：左边形貌 Z（网格点上记录的 Z，按行减中位数 —— 一张网格扫十几小时，行间漂移大），
右边上半 dI/dV 各层、下半电流各层，**每一个**偏压层一格（每行最多 11 格），每层色标各自 2–98%
（看的是每层的空间分布，层与层之间的亮度不可比）；偏压文字正绿负红。

dI/dV：与平均 I(V) 数值导数最相关的那一路 LI Demod 1（图里注明 r）；|r| < 0.3 时改画 I 对 V 的
数值导数（T16）。没采完的网格只画采到的行。

方向（T29）：.3ds 数组第 0 行是视野**下沿**（``grid3ds`` 的约定），出图时翻成上沿在上，与 .sxm 帧同向。
"""
from __future__ import annotations

import math
import time

import numpy as np

from mast.gallery import paths as _paths
from mast.gallery.figures import common as C
from mast.gallery.inventory import num

VER = 1
CAT = "grids"


def _first(ch: list[str], *names: str) -> int | None:
    for n in names:
        if n in ch:
            return ch.index(n)
    return None


def _tparse(g: dict, key: str):
    try:
        return time.strptime(str(g.get(key, "")).strip(), "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return None


def make_grid_sheet(lay: _paths.Layout, it: dict, options: dict | None = None) -> str:
    from PIL import Image, ImageDraw

    from mast.gallery.grid3ds import read_3ds

    G = read_3ds(it["p"])
    g, ch, n, nx, ny, have = G["hdr"], G["chans"], G["npts"], G["nx"], G["ny"], G["have"]
    if have <= 0:
        raise C.JobError("网格里一个点都还没有")
    rows = max(1, math.ceil(have / nx))
    D = G["D"][:rows].astype(np.float64)
    P = G["P"][:rows].astype(np.float64)

    with np.errstate(all="ignore"):
        jb = [j for j, c in enumerate(ch) if c.startswith("Bias [AVG]") and "bwd" not in c]
        V = (np.nanmedian(D[:, :, jb[0], :].reshape(-1, n), axis=0) if jb
             else np.full(n, np.nan))
        if not np.isfinite(V).all():
            v0 = num(g.get("Bias Spectroscopy>Sweep Start (V)"), it.get("v0"))
            v1 = num(g.get("Bias Spectroscopy>Sweep End (V)"), it.get("v1"))
            V = (np.linspace(float(v0), float(v1), n) if v0 is not None and v1 is not None
                 else np.arange(n, dtype=np.float64))
        jI = _first(ch, "Current [AVG] (A)", "Current (A)")
        if jI is None:
            raise C.JobError("网格里没有电流通道")
        I = D[:, :, jI, :] * 1e12
        Iav = np.nanmean(I.reshape(-1, n), axis=0)
        didv = np.gradient(Iav, V) if n > 2 and np.isfinite(Iav).all() else None
        best = None
        for pair in (("LI Demod 1 X [AVG] (A)", "LI Demod 1 X (A)"),
                     ("LI Demod 1 Y [AVG] (A)", "LI Demod 1 Y (A)")):
            j = _first(ch, *pair)
            if j is None:
                continue
            M = D[:, :, j, :] * 1e15
            sp = np.nanmean(M.reshape(-1, n), axis=0)
            r = (abs(float(np.corrcoef(sp, didv)[0, 1]))
                 if didv is not None and np.isfinite(sp).all() and np.std(sp) > 0 else 0.0)
            if not math.isfinite(r):
                r = 0.0
            short = ch[j].replace(" [AVG] (A)", "").replace(" (A)", "")
            if best is None or r > best[0]:
                best = (r, short, M)
        if best is None or not (best[0] >= 0.3):
            dl, dunit, dmaps, src = "数值 dI/dV（I 对 V 求导）", "pA/V", np.gradient(I, V, axis=2), "数值导数"
        else:
            dl, dunit, dmaps, src = "dI/dV · %s（r=%.2f）" % (best[1], best[0]), "fA", best[2], best[1]
        zi = G["pars"].index("Z (m)") if "Z (m)" in G["pars"] else None
        Z = C.row_level(P[:, :, zi] * 1e12) if zi is not None else None

    def flip(a):                                     # 第 0 行在下沿 → 翻成上沿在上（T29）
        return a[::-1]

    k = max(1, math.ceil(144 / nx))
    pw, ph = nx * k, rows * k
    cols = min(n, 11)
    nrow = math.ceil(n / cols)
    gap, lab_h, head_h = 3, 15, 20
    zw = max(pw, 144)
    grid_w = cols * (pw + gap) - gap
    W = zw + 10 + grid_w
    block_h = head_h + nrow * (lab_h + ph + gap)
    Hc = max(2 * block_h, head_h + lab_h + round(zw * rows / nx) + 4)
    canvas = Image.new("RGB", (W, Hc + 52), C.BG)
    d = ImageDraw.Draw(canvas)
    f_tag = C.pil_font(12)
    nearest = Image.Resampling.NEAREST
    if Z is not None:
        zimg, zlo, zhi = C.to_img(flip(Z), C.lut("afmhot"), (2, 98))
        zimg = zimg.resize((zw, max(1, round(zw * rows / nx))), nearest)
        d.text((0, 2), "形貌 Z（按行去偏）", fill=C.INK, font=f_tag)
        d.text((0, head_h), "%.0f pm" % (zhi - zlo), fill=C.DIM, font=f_tag)
        canvas.paste(zimg, (0, head_h + lab_h))
    else:
        d.text((0, 2), "形貌 Z：网格里没有 Z 参数", fill=C.INK, font=f_tag)
    x0 = zw + 10
    for bi, (title, maps, table, unit) in enumerate(((dl, dmaps, C.lut("viridis"), dunit),
                                                     ("电流 I", I, C.lut("magma"), "pA"))):
        y0 = bi * block_h
        d.text((x0, y0 + 2), "%s　每层色标各自 2–98%%，单位 %s" % (title, unit), fill=C.INK, font=f_tag)
        for j in range(n):
            cx = x0 + (j % cols) * (pw + gap)
            cy = y0 + head_h + (j // cols) * (lab_h + ph + gap)
            img, _lo, _hi = C.to_img(flip(maps[:, :, j]), table, (2, 98))
            canvas.paste(img.resize((pw, ph), nearest), (cx, cy + lab_h))
            d.text((cx, cy), C.fmt_bias(float(V[j])),
                   fill=(120, 220, 190) if V[j] > 0 else (240, 150, 140), font=f_tag)

    t0, t1 = _tparse(g, "Start time"), _tparse(g, "End time")
    bias = num(g.get("Bias>Bias (V)"))
    setp = num(g.get("Z-Controller>Setpoint"))
    left = "%s → %s    %s    %s    %s" % (
        time.strftime("%Y-%m-%d %H:%M:%S", t0) if t0 else "?",
        time.strftime("%m-%d %H:%M:%S", t1) if t1 else "?",
        C.fmt_bias(bias) if bias is not None else "偏压未记录",
        C.fmt_cur(setp * 1e12) if setp is not None else "电流未记录",
        C.fmt_size(G["w_nm"], G["h_nm"]))
    stem = C.stem(it["fn"])
    right = "%s · %d×%d 点，完成 %d/%d · %d 层 %s→%s · Zoff %.0f pm · %s sweeps · lock-in %s" % (
        stem, nx, ny, have, nx * ny, n, C.fmt_bias(float(V[0])), C.fmt_bias(float(V[-1])),
        (num(g.get("Bias Spectroscopy>Z offset (m)"), 0.0) or 0.0) * 1e12,
        g.get("Bias Spectroscopy>Number of sweeps") or "未记录",
        g.get("Lock-in>Lock-in status") or "未记录")
    used = C.caption(d, W, Hc, left, right)
    canvas = canvas.crop((0, 0, W, Hc + used))

    base = "%s_%s" % (time.strftime("%Y%m%d-%H%M%S", t0) if t0 else "unknown", stem.replace(" ", "_"))
    name = base + ".png"
    C.write_bytes(lay, CAT, name, C.png_bytes(canvas))
    r_best = float(best[0]) if best is not None else None
    summary = {"dI/dV": src, "完成": f"{have}/{nx * ny}", "层数": int(n)}
    if src != "数值导数" and r_best is not None:
        summary["r"] = round(r_best, 2)
    detail = {"didv": src, "r": r_best, "rows": rows, "have": int(have), "n_layers": int(n),
              "V0": float(V[0]), "V1": float(V[-1]), "name": name}
    return C.write_figure_json(lay, CAT, base, kind="grid_sheets", title="%s · 每层一格" % stem,
                               files=[name], ids=[it["id"]], options=options, summary=summary,
                               detail=detail, maker_version=VER)
