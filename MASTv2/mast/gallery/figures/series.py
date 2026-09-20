# -*- coding: utf-8 -*-
"""旋转系列：16:9 幻灯片拼图 + 刚性 / 晶格校正叠加。

① 拼图：一列一个扫描转角（从第 1 帧的转角起按采集顺序排），一行一轮；每帧逐行调平后转到第 1 帧的朝向，
   以锚点为中心取帧宽大小的视野。只出现一次的转角放进角度最近的列并标注；不从第 0 列开始的一行标「补扫」。
② 叠加：
   锚点   刚性转到第 1 帧朝向后高斯平滑（σ 0.3 nm，抹掉晶格），中心 1.6 nm 内最暗处（``anchor`` 可选
          darkest / brightest / center）+ 抛物线亚像素
   种子   **首帧**逐行调平后在 (x 右, y 上) 坐标下原子带功率谱里两个最强、非共线（25°–155°）的峰，
          与后面的投影振幅细化采用同一套坐标，避免跨坐标约定引入符号错误
   仿射   参考基矢转到每帧坐标为起点，四级 5×5 投影振幅细化两个峰；2×2 矩阵把每帧晶格拉成参考晶格，
          奇异值出 0.9–1.1 的帧只做刚性；参考换成全体实测平均再迭代
   平移   对全体平均做 FFT 互相关精修（中心 3.2 nm、搜索 ±0.35 nm、抛物线亚像素），3 轮
   叠加   每帧减中位数、除以稳健标准差后逐像素平均 / 中位数，乘回全体帧稳健标准差的中位数换成 pm；
          覆盖不到 ``min_cover``（默认 25%）帧的像素留黑；按两个基矢峰振幅之和选主图
   配准表 每帧锚点的帧内（第 1 帧朝向）与压电坐标（按时间看就是漂移轨迹）、仿射奇异值、残余转角、平移精修量

方向约定（T29 / T6）：SCAN_ANGLE 为正 = 扫描框相对压电坐标顺时针转；帧内 (x′, y′) = Rot(θk − θ1)·(u, v)，
(u, v) 是第 1 帧朝向坐标；采样 ``row = (0.5 − y′/h)·ny − 0.5``（行 0 在上沿）。
"""
from __future__ import annotations

import collections
import csv
import io
import math
import time

import numpy as np

from mast.gallery import paths as _paths
from mast.gallery.figures import common as C

VER = 1
CAT = "series"
ROBUST = 1.4826


# ── 读数据 ─────────────────────────────────────────────────────────────


def load_series(doc: dict, marks: dict, sid: str) -> tuple[dict, list[dict]]:
    sv = (marks.get("series") or {}).get(sid)
    if not isinstance(sv, dict):
        raise C.JobError(f"系列 {sid} 不存在")
    by = C.by_id(doc)
    ids = [i for i in (sv.get("ids") or []) if (by.get(i) or {}).get("k") == "f"]
    if not ids:
        raise C.JobError("系列里没有在索引里的帧")
    ids.sort(key=lambda i: (by[i].get("t") or 0, i))
    frames = []
    for i in ids:
        it = by[i]
        z, _bwd = C.load_z(it["p"])
        z = C.plane(C.row_level(z * 1e12))
        med = np.nanmedian(z) if np.isfinite(z).any() else 0.0
        z = np.where(np.isfinite(z), z, med)
        frames.append(dict(id=i, it=it, z=z, th=float(it.get("ang") or 0.0)))
    return sv, frames


def _wh(it: dict) -> tuple[float, float]:
    w = float(it.get("w") or 0.0) or 1.0
    return w, float(it.get("hn") or 0.0) or w


def sample(fr: dict, uv: np.ndarray, th1: float) -> np.ndarray:
    """``uv[..., 2]``：第 1 帧朝向、相对本帧中心的坐标（nm）→ 本帧 Z（双线性；出界 NaN）。"""
    from scipy.ndimage import map_coordinates

    w, h = _wh(fr["it"])
    ny, nx = fr["z"].shape
    D = math.radians(fr["th"] - th1)
    c, s = math.cos(D), math.sin(D)
    xp = uv[..., 0] * c - uv[..., 1] * s
    yp = uv[..., 0] * s + uv[..., 1] * c
    col = (xp / w + 0.5) * nx - 0.5
    row = (0.5 - yp / h) * ny - 0.5
    return map_coordinates(fr["z"], [row, col], order=1, mode="constant", cval=np.nan)


def grid(L: float, n: int) -> np.ndarray:
    ax = (np.arange(n) - (n - 1) / 2) * (L / n)
    U, V = np.meshgrid(ax, -ax)                     # 行 0 在上（v 最大）
    return np.stack([U, V], -1)


def normz(a: np.ndarray):
    med = np.nanmedian(a)
    sd = ROBUST * np.nanmedian(np.abs(a - med))
    return (a - med) / (sd if sd > 0 else 1.0), float(sd)


def rotvec(v, deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def _nmpp(fr: dict) -> float:
    w, _h = _wh(fr["it"])
    return w / fr["z"].shape[1]


# ── ① 16:9 拼图 ────────────────────────────────────────────────────────


def column_layout(thetas: list[float], th1: float):
    """``(列转角, 行 [{列: 帧序号}], 每帧的转角档)``。"""
    keys = [round(t * 2) / 2 for t in thetas]
    cnt = collections.Counter(keys)
    cols = sorted((k for k, c in cnt.items() if c >= 2), key=lambda ang: (ang - th1) % 360)
    if not cols:
        cols = sorted(set(keys), key=lambda ang: (ang - th1) % 360)

    def col_of(ang):
        return min(range(len(cols)), key=lambda i: abs((ang - cols[i] + 180) % 360 - 180))

    rows, cur, last = [], {}, -1
    for k, ang in enumerate(keys):
        c = col_of(ang)
        if c in cur or c <= last:                  # 这一轮这一列已经有了，或转角回头 → 新一轮
            rows.append(cur)
            cur = {}
        cur[c] = k
        last = c
    rows.append(cur)
    return cols, rows, keys


def anchors(frames: list[dict], th1: float, mode: str = "darkest") -> list[np.ndarray]:
    from scipy.ndimage import gaussian_filter

    nmpp = _nmpp(frames[0])
    L = 4.0
    n = max(8, int(round(L / nmpp)))
    G = grid(L, n)
    rr = np.hypot(G[..., 0], G[..., 1])
    out = []
    for fr in frames:
        if mode == "center":
            out.append(np.zeros(2))
            continue
        A = sample(fr, G, th1)
        if not np.isfinite(A).any():
            out.append(np.zeros(2))
            continue
        A = gaussian_filter(np.where(np.isfinite(A), A, np.nanmedian(A)), 0.3 / nmpp)
        if mode == "brightest":
            A = -A
        A = np.where(rr <= 1.6, A, np.inf)
        i, j = np.unravel_index(np.argmin(A), A.shape)
        di = dj = 0.0
        if 0 < i < n - 1 and np.isfinite(A[i - 1, j] + A[i + 1, j]):
            den = A[i - 1, j] - 2 * A[i, j] + A[i + 1, j]
            di = 0.5 * (A[i - 1, j] - A[i + 1, j]) / den if den > 0 else 0.0
        if 0 < j < n - 1 and np.isfinite(A[i, j - 1] + A[i, j + 1]):
            den = A[i, j - 1] - 2 * A[i, j] + A[i, j + 1]
            dj = 0.5 * (A[i, j - 1] - A[i, j + 1]) / den if den > 0 else 0.0
        u = G[i, j, 0] + dj * (L / n)
        v = G[i, j, 1] - di * (L / n)
        out.append(np.array([u, v]))
    return out


def _mtime(fr: dict):
    return fr["it"].get("mt") or fr["it"].get("t")


def make_slides(lay: _paths.Layout, doc: dict, marks: dict, sid: str, options: dict | None = None) -> str:
    from PIL import Image, ImageDraw

    options = dict(options or {})
    W = int(options.get("page_w") or 3840)
    H = int(options.get("page_h") or 2160)
    sv, frames = load_series(doc, marks, sid)
    th1 = frames[0]["th"]
    anc = anchors(frames, th1, str(options.get("anchor") or "darkest"))
    cols, rows, keys = column_layout([fr["th"] for fr in frames], th1)
    nc = len(cols)
    per = math.ceil(len(rows) / 2)
    pages = [rows[:per], rows[per:]] if len(rows) > per else [rows]
    side, gap, top, head, lab, cap, rgap, bottom = 12, 4, 96, 46, 28, 58, 8, 40
    cell = int(min((W - 2 * side - (nc - 1) * gap) / nc,
                   (H - top - head - bottom - per * (lab + cap + rgap)) / per))
    cell = max(cell, 16)
    fov, _h = _wh(frames[0]["it"])
    G = grid(fov, cell)
    grid_w = nc * cell + (nc - 1) * gap
    x0 = (W - grid_w) // 2
    F_T, F_S, F_H, F_C, F_B, F_L, F_c = (C.pil_font(z) for z in (40, 26, 30, 24, 22, 21, 18))
    table = C.lut("afmhot")
    name0 = str(sv.get("name") or sid)
    base = C.safe_name(name0 + "_16比9")
    files: list[str] = []
    for pi, page in enumerate(pages):
        canvas = Image.new("RGB", (W, H), C.BG)
        d = ImageDraw.Draw(canvas)
        ks = [k for r in page for k in r.values()]
        r0 = sum(len(pg) for pg in pages[:pi]) + 1
        rounds = [r0 + ri for ri, r in enumerate(page) if r and min(r) == 0] or [r0, r0 + len(page) - 1]
        extra = any(r and min(r) != 0 for r in page)
        d.text((x0, 4), "旋转系列「%s」%d 帧 · 第 %d/%d 页 · 第 %d–%d 轮%s · %s → %s" % (
            name0, len(frames), pi + 1, len(pages), rounds[0], rounds[-1], " + 补扫" if extra else "",
            C.local(frames[min(ks)]["it"].get("t"), "%m-%d %H:%M"), C.local(_mtime(frames[max(ks)]), "%m-%d %H:%M")),
            fill=C.INK, font=F_T)
        d.text((x0, 56), "每列一个扫描转角；图像已全部转到第 1 帧（%s，%g°）的朝向，以中心锚点为中心，每格 %g×%g nm；逐行调平；每格色标各自 1–99%%；黑角是转到帧外的部分" % (
            C.num4(frames[0]["it"]["fn"]), th1, fov, fov), fill=C.DIM, font=F_S)
        y = top                                     # 两页都顶部对齐：翻页时列与行的位置不动
        for ci, ang in enumerate(cols):
            txt = "%g°" % ang
            cx = x0 + ci * (cell + gap) + cell / 2
            d.text((cx - d.textlength(txt, font=F_H) / 2, y + 6), txt, fill=C.INK, font=F_H)
        y += head
        for ri, r in enumerate(page):
            if not r:
                continue
            kr = sorted(r.values())
            if min(r) != 0:
                rl = "补扫"
            else:
                rl = "第 %d 轮" % (r0 + ri) + ("（只扫了 %d 个转角）" % len(r) if len(r) < nc else "")
            d.text((x0, y + 2), "%s · %s → %s" % (rl, C.local(frames[kr[0]]["it"].get("t"), "%m-%d %H:%M"),
                                                 C.local(_mtime(frames[kr[-1]]), "%H:%M")), fill=C.DIM, font=F_L)
            yc = y + lab
            for ci, k in r.items():
                fr = frames[k]
                it = fr["it"]
                x = x0 + ci * (cell + gap)
                img, _lo, _hi = C.to_img(sample(fr, G + anc[k], th1), table)
                canvas.paste(img, (x, yc))
                d.text((x + 2, yc + cell + 1), C.local(it.get("t"), "%H:%M:%S"), fill=C.INK, font=F_C)
                rt = "#%d · %s" % (k + 1, C.num4(it["fn"]))
                d.text((x + cell - 2 - d.textlength(rt, font=F_c), yc + cell + 7), rt, fill=C.DIM, font=F_c)
                d.text((x + 2, yc + cell + 30), "%s · %s" % (C.fmt_bias(it.get("b")), C.fmt_cur(it.get("sp"))),
                       fill=C.INK, font=F_B)
                if keys[k] not in cols:             # 不在转角档里的帧：图左上角橙字注明
                    wt = "扫描角 %g°" % fr["th"]
                    tw = d.textlength(wt, font=F_c)
                    d.rectangle([x + 3, yc + 3, x + 9 + tw, yc + 27], fill=(0, 0, 0))
                    d.text((x + 6, yc + 4), wt, fill=(245, 165, 90), font=F_c)
            y += lab + cell + cap + rgap
        px = int(round(cell / fov))
        bx = x0 + grid_w - px - 70
        d.rectangle([bx, H - 26, bx + px, H - 18], fill=(255, 255, 255))
        d.text((bx + px + 10, H - 36), "1 nm", fill=C.INK, font=F_L)
        stem = "%s_第%d页" % (base, pi + 1)
        C.write_bytes(lay, CAT, stem + ".png", C.png_bytes(canvas))
        C.write_bytes(lay, CAT, stem + ".jpg", C.jpeg_bytes(canvas, 92))
        files += [stem + ".png", stem + ".jpg"]
    summary = {"帧数": len(frames), "转角列": nc, "轮数": len(rows), "格子 px": cell}
    detail = {"cols": cols, "rounds": [{str(c): frames[k]["id"] for c, k in r.items()} for r in rows],
              "pages": len(pages), "cell": cell, "anchor": str(options.get("anchor") or "darkest"),
              "anchors_uv": [a.tolist() for a in anc]}
    return C.write_figure_json(lay, CAT, base, kind="series_slides", title="%s · 16:9 拼图" % name0,
                               files=files, ids=[fr["id"] for fr in frames], series=[sid], options=options,
                               summary=summary, detail=detail, maker_version=VER)


# ── ② 叠加 ─────────────────────────────────────────────────────────────


def _pgrid(ny: int, nx: int, nmpp: float):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return ((xx - nx / 2) * nmpp).ravel(), ((ny / 2 - yy) * nmpp).ravel()


def _proj_many(zw: np.ndarray, wsum: float, XY, ks: np.ndarray) -> np.ndarray:
    """一次算多个波矢（nm⁻¹，x 右 y 上）处的投影振幅。zw = (z − 均值)·Hann 窗。"""
    X, Y = XY
    ph = np.exp(-2j * np.pi * (ks[:, 0:1] * X[None, :] + ks[:, 1:2] * Y[None, :]))
    return np.abs(ph @ zw.ravel()) / wsum * 2


def _windowed(fr: dict):
    z = fr["z"]
    ny, nx = z.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    return (z - z.mean()) * win, float(win.sum()), _pgrid(ny, nx, _nmpp(fr))


def _refine(zw, wsum, XY, k) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    span = 0.12 * float(np.linalg.norm(k))
    for _lev in range(4):
        g = np.linspace(-span, span, 5)
        cand = np.array([[k[0] + dx, k[1] + dy] for dx in g for dy in g])
        k = cand[int(np.argmax(_proj_many(zw, wsum, XY, cand)))]
        span /= 3
    return k


def seed_basis(z: np.ndarray, nmpp: float) -> np.ndarray | None:
    """首帧功率谱里两个最强、非共线（25°–155°）的原子带峰，(x 右, y 上) 坐标，nm⁻¹。

    坐标与 :func:`_proj_many` 同一约定：列 ↔ x = (col − nx/2)·nmpp ⇒ kx = fftfreq(nx)；
    行 ↔ y = (ny/2 − row)·nmpp ⇒ ky = −fftfreq(ny)（行向下，y 向上取负）。"""
    from scipy.ndimage import maximum_filter

    from mast.vision.atomic_phase import ATOMIC_BAND_NM

    ny, nx = z.shape
    zw = (z - z.mean()) * np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fftshift(np.fft.fft2(zw))) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=nmpp))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=nmpp))
    KX, KYrow = np.meshgrid(fx, fy)
    KY = -KYrow
    K = np.hypot(KX, KY)
    width = min(nx, ny) * nmpp
    kmin = 1.0 / min(ATOMIC_BAND_NM[1], width / 5.0)
    kmax = 1.0 / ATOMIC_BAND_NM[0]
    band = (K >= kmin) & (K <= kmax)
    upper = (KY > 1e-12) | ((np.abs(KY) <= 1e-12) & (KX > 0))
    if not (band & upper).any():
        return None
    bg = float(np.median(P[band]))
    loc = maximum_filter(P, size=3, mode="wrap") == P
    cand = np.argwhere(band & upper & loc & (P > 8 * bg))
    peaks = sorted(((float(P[r, c]), np.array([KX[r, c], KY[r, c]])) for r, c in cand), key=lambda t: -t[0])
    if len(peaks) < 2:
        return None
    k1 = peaks[0][1]
    for _pw, k in peaks[1:]:
        cosang = abs(float(np.dot(k1, k)) / (np.linalg.norm(k1) * np.linalg.norm(k)))
        if cosang < math.cos(math.radians(25)):
            return np.array([k1, k])
    return None


def lattice_maps(frames: list[dict], th1: float):
    """``(每帧 M, 是否用上仿射, 参考基矢 [g1; g2]（第 1 帧朝向）或 None, 说明)``。"""
    N = len(frames)
    zw0, wsum0, XY0 = _windowed(frames[0])
    seed = seed_basis(frames[0]["z"], _nmpp(frames[0]))
    if seed is None:
        return [np.eye(2)] * N, [False] * N, None, "首帧功率谱里找不到两个非共线的晶格峰 —— 只做刚性叠加"
    Gref = np.array([_refine(zw0, wsum0, XY0, seed[q]) for q in (0, 1)])
    meas = []
    for fr in frames:
        zw, wsum, XY = _windowed(fr)
        D = fr["th"] - th1
        rows = []
        for q in (0, 1):
            k = _refine(zw, wsum, XY, rotvec(Gref[q], D))       # 参考基矢 → 该帧坐标
            rows.append(rotvec(k, -D))                          # 回到第 1 帧朝向
        meas.append(np.array(rows))
    Ms, used = [np.eye(2)] * N, [False] * N
    for _it in range(2):                                        # 参考换成全体实测的平均，再算一遍矩阵
        Ms, used = [], []
        for Gm in meas:
            try:
                M = np.linalg.solve(Gm, Gref)
                sv = np.linalg.svd(M, compute_uv=False)
                ok = bool(0.9 < sv.min() and sv.max() < 1.1)
            except np.linalg.LinAlgError:
                M, ok = np.eye(2), False
            Ms.append(M if ok else np.eye(2))
            used.append(ok)
        if any(used):
            Gref = np.mean([Gm for Gm, u in zip(meas, used) if u], axis=0)
    return Ms, used, Gref, ""


def stack(frames: list[dict], th1: float, Ms, anc, L: float, n: int, passes: int = 3):
    """按锚点 + 矩阵重采样，FFT 互相关对全体平均精修平移。``(图堆, 精修后锚点, 每帧稳健标准差, 平移量)``。"""
    R = grid(L, n)
    anc = [np.array(a, dtype=np.float64) for a in anc]
    step = L / n
    c0 = n // 2
    # Keep the correlation window inside the sampled field;
    # a two-pixel Hann window is all zero, so use at least four pixels.
    hw = min(c0, max(2, int(round(1.6 / step))))
    win = np.outer(np.hanning(2 * hw), np.hanning(2 * hw))
    srch = min(int(round(0.35 / step)), max(0, hw - 1))
    shifts = [np.zeros(2) for _ in frames]
    Js, sds = [], []
    for it in range(passes + 1):
        Js, sds = [], []
        for fr, M, a in zip(frames, Ms, anc):
            J, sd = normz(sample(fr, R @ M.T + a, th1))
            Js.append(J)
            sds.append(sd)
        if it == passes or hw < 2:
            break
        with np.errstate(all="ignore"):
            ref = np.nanmean(np.stack(Js), 0)
        Rp = np.nan_to_num(ref[c0 - hw:c0 + hw, c0 - hw:c0 + hw]) * win
        FR = np.conj(np.fft.fft2(Rp))
        for k, (J, M) in enumerate(zip(Js, Ms)):
            Jp = np.nan_to_num(J[c0 - hw:c0 + hw, c0 - hw:c0 + hw]) * win
            Cc = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(Jp) * FR)))
            sub = Cc[hw - srch:hw + srch + 1, hw - srch:hw + srch + 1]
            i, j = np.unravel_index(np.argmax(sub), sub.shape)
            di = dj = 0.0
            if 0 < i < sub.shape[0] - 1:
                den = sub[i - 1, j] - 2 * sub[i, j] + sub[i + 1, j]
                di = 0.5 * (sub[i - 1, j] - sub[i + 1, j]) / den if den < 0 else 0.0
            if 0 < j < sub.shape[1] - 1:
                den = sub[i, j - 1] - 2 * sub[i, j] + sub[i, j + 1]
                dj = 0.5 * (sub[i, j - 1] - sub[i, j + 1]) / den if den < 0 else 0.0
            dr, dc = (i - srch) + di, (j - srch) + dj
            delta = np.array([dc * step, -dr * step])        # 图内容相对参考偏移了 δ（nm，u 右 v 上）
            anc[k] = anc[k] + M @ delta
            shifts[k] = shifts[k] + delta
    return np.stack(Js), anc, np.array(sds), shifts


def _panel(a, table, mask, p=(1, 99.5), scale: int = 1):
    from PIL import Image

    img, lo, hi = C.to_img(np.where(mask, a, np.nan), table, p)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.BICUBIC)
    return img, lo, hi


def _scalebar(d, x: int, y: int, px_per_nm: int, label: str = "1 nm") -> None:
    d.rectangle([x, y, x + px_per_nm, y + 4], fill=(255, 255, 255))
    d.text((x, y - 16), label, fill=(255, 255, 255), font=C.pil_font(12))


def _npy_bytes(a) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(a))
    return buf.getvalue()


def make_stack(lay: _paths.Layout, doc: dict, marks: dict, sid: str, options: dict | None = None,
               cancel=None) -> str:
    from PIL import Image, ImageDraw

    options = dict(options or {})
    min_cover = float(options.get("min_cover") or 0.25)
    mode = str(options.get("anchor") or "darkest")
    sv, frames = load_series(doc, marks, sid)
    if cancel is not None and cancel.is_set():
        raise C.JobError("已取消")
    name0 = str(sv.get("name") or sid)
    th1 = frames[0]["th"]
    nmpp = _nmpp(frames[0])
    L = float(options.get("fov_nm") or _wh(frames[0]["it"])[0])
    N = len(frames)
    anc = anchors(frames, th1, mode)
    Ms, good, Gref, note = lattice_maps(frames, th1)
    n = int(round(L / nmpp))
    I2 = [np.eye(2)] * N
    S_rig, _anc_rig, _sd_rig, _sh_rig = stack(frames, th1, I2, anc, L, n)
    if cancel is not None and cancel.is_set():
        raise C.JobError("已取消")
    S_aff, anc_aff, sd_aff, sh_aff = stack(frames, th1, Ms, anc, L, n)
    pm = float(np.median(sd_aff))                    # 归一化单位 → pm（全体帧稳健标准差的中位数）
    cover = np.sum(np.isfinite(S_aff), 0)
    mask = cover >= min_cover * N
    with np.errstate(all="ignore"):
        rig_mean = np.nanmean(S_rig, 0) * pm
        aff_mean = np.nanmean(S_aff, 0) * pm
        aff_med = np.nanmedian(S_aff, 0) * pm
    Rg = grid(L, n)
    rr = np.hypot(Rg[..., 0], Rg[..., 1])
    rad = float(rr[cover == N].max()) if (cover == N).any() else 2.0
    wdisc = np.where(rr <= rad, 0.5 * (1 + np.cos(np.pi * rr / rad)), 0.0)

    def bragg(a, g):
        with np.errstate(all="ignore"):
            b = np.nan_to_num(a - np.nanmean(np.where(rr <= rad, a, np.nan))) * wdisc
        return float(abs(np.sum(b * np.exp(-2j * np.pi * (g[0] * Rg[..., 0] + g[1] * Rg[..., 1]))))
                     / np.sum(wdisc) * 2)

    amp: dict = {}
    if Gref is not None:
        peaks = {"基矢 1": Gref[0], "基矢 2": Gref[1], "基矢1+基矢2": Gref[0] + Gref[1],
                 "基矢1−基矢2": Gref[0] - Gref[1]}
        amp = {k: (bragg(rig_mean, g), bragg(aff_mean, g)) for k, g in peaks.items()}
        use_aff = bool(amp["基矢 1"][1] + amp["基矢 2"][1] >= amp["基矢 1"][0] + amp["基矢 2"][0])
    else:
        use_aff = False
    amp_txt = "，".join("%s %.2f→%.2f" % (k, v[0], v[1]) for k, v in amp.items()) or "没有晶格种子"

    table = C.lut("afmhot")
    gap = 4
    pans = [_panel(rig_mean, table, mask), _panel(aff_mean, table, mask), _panel(aff_med, table, mask)]
    cimg = Image.fromarray((np.clip(cover / N, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    pw = pans[0][0].width
    Wc = 4 * pw + 3 * gap
    canvas = Image.new("RGB", (Wc, pw + 72), C.BG)
    d = ImageDraw.Draw(canvas)
    labels = ["刚性叠加（只旋转+平移）平均" + ("" if use_aff else " · 主图"),
              "晶格校正叠加 · 平均" + (" · 主图" if use_aff else ""), "晶格校正叠加 · 中位数",
              "每像素覆盖帧数（黑 0 → 白 %d）" % N]
    for k, (img, lab) in enumerate(zip([p[0] for p in pans] + [cimg], labels)):
        x = k * (pw + gap)
        canvas.paste(img, (x, 0))
        C.tag(d, (x + 5, 5), lab + ("  %.0f pm" % (pans[k][2] - pans[k][1]) if k < 3 else ""))
    _scalebar(d, 12, pw - 16, int(round(1 / (L / n))))
    t0 = frames[0]["it"].get("t")
    t1 = max((_mtime(fr) or 0) for fr in frames)
    cur = sorted({round(float(fr["it"].get("sp") or 0.0)) for fr in frames})
    ncur = {c: sum(1 for fr in frames if round(float(fr["it"].get("sp") or 0.0)) == c) for c in cur}
    cur_txt = " / ".join("%s（%d 帧）" % (C.fmt_cur(c), ncur[c]) for c in sorted(cur, key=lambda c: -ncur[c]))
    left = "%s → %s    %s    %s    视野 %.1f×%.1f nm" % (
        C.local(t0, "%Y-%m-%d %H:%M:%S"), C.local(t1, "%m-%d %H:%M:%S"),
        " / ".join(sorted({C.fmt_bias(fr["it"].get("b")) for fr in frames})), cur_txt, L, L)
    right = ("%d 帧叠加 · 朝向＝第 1 帧（%s，%g°）· 锚点＝%s · 平移按全体平均互相关精修 3 轮 · 仿射按各帧实测晶格（%d 帧可用）"
             " · 每帧按稳健标准差归一 · 覆盖 <%d%% 的像素留黑 · 色标 1–99.5%%" % (
                 N, C.num4(frames[0]["it"]["fn"]), th1,
                 {"darkest": "视野中心的暗点", "brightest": "视野中心的亮点", "center": "帧心"}.get(mode, mode),
                 sum(good), round(min_cover * 100)))
    f16, f12, f14 = C.pil_font(16), C.pil_font(12), C.pil_font(14)
    d.text((6, pw + 4), left, fill=C.INK, font=f16)
    d.text((6, pw + 28), right, fill=C.DIM, font=f12)
    d.text((6, pw + 46), ("晶格峰振幅（⌀%.1f nm 全覆盖区，pm）刚性 → 晶格校正：%s" % (2 * rad, amp_txt)) + (" · " + note if note else ""),
           fill=C.DIM, font=f12)
    base = C.safe_name(name0 + "_叠加")
    C.write_bytes(lay, CAT, base + ".png", C.png_bytes(canvas))

    big, lo, hi = _panel(aff_mean if use_aff else rig_mean, table, mask, scale=2)
    bc = Image.new("RGB", (big.width, big.height + 52), C.BG)
    bc.paste(big, (0, 0))
    bd = ImageDraw.Draw(bc)
    C.tag(bd, (6, 6), "%s · %d 帧平均  %.0f pm" % ("晶格校正叠加" if use_aff else "刚性叠加", N, hi - lo), font=f14)
    _scalebar(bd, 16, big.height - 20, int(round(2 / (L / n))))
    bd.text((6, big.height + 4), left, fill=C.INK, font=f16)
    bd.text((6, big.height + 28), "朝向＝第 1 帧（%s，%g°）· 平移精修%s · 覆盖 <%d%% 留黑 · 晶格峰振幅 刚性→校正 %s" % (
        C.num4(frames[0]["it"]["fn"]), th1, " + 晶格仿射校正" if use_aff else "（晶格校正没有更锐，主图用刚性）",
        round(min_cover * 100), amp_txt), fill=C.DIM, font=f12)
    C.write_bytes(lay, CAT, base + "_主图_2x.png", C.png_bytes(bc))
    C.write_bytes(lay, CAT, base + "_晶格校正平均.npy", _npy_bytes(aff_mean))
    C.write_bytes(lay, CAT, base + "_刚性平均.npy", _npy_bytes(rig_mean))
    C.write_bytes(lay, CAT, base + "_覆盖帧数.npy", _npy_bytes(cover))

    th1r = math.radians(th1)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["序号", "文件", "开始时刻", "SCAN_ANGLE", "设定电流 pA", "帧中心 X nm", "帧中心 Y nm",
                "锚点帧内 u nm（第1帧朝向）", "锚点帧内 v nm", "锚点压电 X nm", "锚点压电 Y nm",
                "仿射奇异值大", "仿射奇异值小", "仿射残余转角°", "仿射已用", "平移精修 du nm", "平移精修 dv nm", "帧稳健标准差 pm"])
    for k, (fr, a, M, g, sh, sd) in enumerate(zip(frames, anc_aff, Ms, good, sh_aff, sd_aff)):
        it = fr["it"]
        cx, cy = float(it.get("cx") or 0.0), float(it.get("cy") or 0.0)
        X = cx + a[0] * math.cos(th1r) + a[1] * math.sin(th1r)
        Y = cy - a[0] * math.sin(th1r) + a[1] * math.cos(th1r)
        U, svs, Vt = np.linalg.svd(M)
        Rm = U @ Vt
        w.writerow([k + 1, it["fn"], C.local(it.get("t"), "%d.%m.%Y %H:%M:%S"), fr["th"],
                    round(float(it.get("sp") or 0.0)), round(cx, 4), round(cy, 4), round(float(a[0]), 4),
                    round(float(a[1]), 4), round(X, 4), round(Y, 4), round(float(svs[0]), 4), round(float(svs[1]), 4),
                    round(math.degrees(math.atan2(Rm[1, 0], Rm[0, 0])), 3), int(g), round(float(sh[0]), 4),
                    round(float(sh[1]), 4), round(float(sd), 2)])
    C.write_bytes(lay, CAT, base + "_配准表.csv", buf.getvalue().encode("utf-8-sig"))

    sv_all = np.array([np.linalg.svd(M, compute_uv=False) for M in Ms])
    files = [base + ".png", base + "_主图_2x.png", base + "_配准表.csv", base + "_晶格校正平均.npy",
             base + "_刚性平均.npy", base + "_覆盖帧数.npy"]
    summary = {"帧数": N, "主图": "晶格校正" if use_aff else "刚性", "仿射可用帧": int(sum(good)),
               "奇异值": "%.3f–%.3f" % (float(sv_all.min()), float(sv_all.max())),
               "最大平移 nm": round(float(max(np.linalg.norm(s) for s in sh_aff)), 3)}
    detail = {"use_aff": use_aff, "amp": {k: list(v) for k, v in amp.items()}, "rad": rad, "good": int(sum(good)),
              "sv_min": float(sv_all.min()), "sv_max": float(sv_all.max()),
              "shift_max": float(max(np.linalg.norm(s) for s in sh_aff)),
              "anc_spread": float(np.ptp(np.array(anc), 0).max()) if N > 1 else 0.0,
              "sharp_rig": float(np.nanstd(np.where(mask, rig_mean, np.nan))),
              "sharp_aff": float(np.nanstd(np.where(mask, aff_mean, np.nan))),
              "gref": Gref.tolist() if Gref is not None else None, "note": note, "anchor": mode, "fov_nm": L,
              "anchors_final_uv": [a.tolist() for a in anc_aff]}
    return C.write_figure_json(lay, CAT, base, kind="series_stack", title="%s · 叠加" % name0, files=files,
                               ids=[fr["id"] for fr in frames], series=[sid], options=options, summary=summary,
                               detail=detail, maker_version=VER)
