# -*- coding: utf-8 -*-
"""拉线谱：站位推断（纯函数）+ 比较 / 均值的热图与瀑布。

**一条线 = 若干谱系列，每个系列一个区组**。站位由输入谱的位置推断：

1. 全体谱位置做 PCA 定线方向 û；符号取「参照系列采集顺序第一条谱的投影在中位数之下」；
2. 每条谱投影 ``s = (p − p̄)·û``；
3. 参照系列 = 谱最多的那个（并列取最早）：由相邻投影差估步长（差值 > 0.3×P75 的那些取中位数，
   同站位的重复测量不会把步长拉成 0），再对「整数站位号 k」拟合 ``s = s0 + k·step``；
4. 其它系列的常数偏移 δ（漂移）分两段定：**小数部分**在 ±半个步长内网格搜索使 ``Σ |到最近格点的距离|`` 最小、
   再用残差中位数精修；**整步部分**在整条线长度内挑「落在参照站位范围内的谱最多 → 其中对得上的最多 → |δ| 最小」。
   区组之间的漂移可能超过一个步长，只看 ±半个步长会整组错格；
   覆盖参照全部站位的区组，整步偏移由「两头对齐」唯一确定；只覆盖一部分站位的区组定不下来 ——
   取漂移最小的那个并写进 warnings，操作员可用选项 ``station_shift {系列: 整数}`` 平移；
   每条谱归最近的格点；残差 > :data:`MATCH_FRAC` × 步长的记「对不上」；
5. 站位号 = k − 全体最小 k（缺了的站位留空行，不压缩）。

区组标签 = 系列名去掉公共部分（末尾中英文括号注释归到标签里，如「（中断）」）；线名默认 = 公共部分。

作图约定：

* 左栏 = 每个区组这组谱之前的最后一张 STM，青点 = 实测位置，黄圈 = 标注站位；
* 比较_热图：Current、dI/dV 两行；左半线性色标（截在 |x| 99% 分位），右半对数色标 sign(x)·log₁₀(1+|x|/x₀)
  （x₀ = 这组谱高频噪声 5σ 就近的 1-2-5 值），各区组并排，同一行同一种色标共用范围；
* 比较_瀑布：每个站位把各区组的谱（正扫）叠在一起，每条抬高 P90×0.30；
* 均值_热图 / 均值_瀑布：每条正反扫平均后按站位逐点平均（纵轴 ×n），热图线性、对数各一排，瀑布带标准误；
* 剔除 = 系列的 ``bad_from`` 起（针尖在变）+ 评为排除的系列：**整条删去不画**，灰行 = 删去或没测，
  标题第二行写删了哪些。

均值由同一站位所有有效谱直接计算。
**不用 pyplot**，每张图的绘制段持本体的锁（T27）。
"""
from __future__ import annotations

import math
import re

import numpy as np

from mast.gallery import paths as _paths
from mast.gallery.figures import common as C
from mast.gallery.figures import spectra as SP

VER = 1
CAT = "sts_lines"
FIG, DPI = (16, 9), 200
MATCH_FRAC = 0.35
NODATA = "#cfc8bd"                                  # 热图里删去 / 没测的站位

_TRAIL_NOTE = re.compile(r"\s*([（(][^（）()]*[)）])\s*$")


def _tokens(name) -> list[str]:
    return [t.strip() for t in re.split(r"\s*·\s*", str(name or "")) if t.strip()]


def names_common(names: list[str]) -> tuple[str, list[str]]:
    """``(线名, 每个系列的区组标签)``。比较时去掉末尾括号注释，注释归到标签里。"""
    toks = [_tokens(n) for n in names]
    if not toks:
        return "", []
    bases = [[_TRAIL_NOTE.sub("", t) for t in ts] for ts in toks]
    if len(toks) == 1:
        return " · ".join(bases[0]), [""]
    common = [b for b in bases[0] if all(b in bs for bs in bases[1:])]
    labels = []
    for ts, bs in zip(toks, bases):
        lab = []
        for t, b in zip(ts, bs):
            if b in common:
                note = t[len(b):].strip()
                if note:
                    lab.append(note)
            else:
                lab.append(t)
        labels.append(" ".join(lab))
    return " · ".join(common), labels


# ── 站位推断（纯函数）──────────────────────────────────────────────────


def _empty_block(b: dict) -> dict:
    return {"series": str(b.get("sid", "")), "label": str(b.get("label", "")), "n": 0,
            "offset_nm": 0.0, "assigned": [], "unmatched": []}


def infer_stations(blocks: list[dict], *, match_frac: float = MATCH_FRAC, shifts: dict | None = None) -> dict:
    """``blocks = [{"sid", "label", "spectra": [{"id", "x", "y", "order", "t"}]}]``（位置 nm）。

    ``shifts {sid: 整数}``：把该区组的站位号整体加上这个数（几何定不下整步偏移时由操作员给）。
    返回 ``ok direction step_nm n_stations reference blocks warnings``（``GalleryStsLinePlan`` 去掉
    ``line_name degraded detail``）。纯函数：不读文件。"""
    warnings: list[str] = []
    out = {"ok": False, "direction": [], "step_nm": None, "n_stations": 0, "reference": "",
           "blocks": [_empty_block(b) for b in blocks], "warnings": warnings}
    usable = [(bi, b) for bi, b in enumerate(blocks) if b.get("spectra")]
    if not usable:
        warnings.append("这些系列里没有带位置的偏压谱")
        return out
    P = np.array([[float(s["x"]), float(s["y"])] for _bi, b in usable for s in b["spectra"]])
    center = P.mean(axis=0)
    u = np.array([1.0, 0.0])
    if len(P) >= 2:
        w, vecs = np.linalg.eigh(np.cov((P - center).T))
        if float(np.max(w)) > 0:
            u = vecs[:, int(np.argmax(w))]
    u = u / (float(np.linalg.norm(u)) or 1.0)

    def tmin(b):
        return min(float(s.get("t") or 0.0) for s in b["spectra"])

    ref_bi, ref = min(usable, key=lambda p: (-len(p[1]["spectra"]), tmin(p[1]), p[0]))
    ref_sp = sorted(ref["spectra"], key=lambda s: s["order"])

    def projs(sp) -> np.ndarray:
        return (np.array([[float(s["x"]), float(s["y"])] for s in sp]) - center) @ u

    s_ref = projs(ref_sp)
    if len(s_ref) > 1 and s_ref[0] > float(np.median(s_ref)):
        u = -u
        s_ref = -s_ref

    ss = np.sort(s_ref)
    dd = np.diff(ss)
    step = None
    if dd.size and float(dd.max()) > 1e-6:
        thr = 0.3 * float(np.percentile(dd, 75))
        big = dd[dd > max(thr, 1e-6)]
        if big.size:
            step = float(np.median(big))

    if not step or step <= 0:
        # 只有一个位置：全部归站位 0。
        warnings.append("参照系列只有一个位置 —— 当成单站位")
        m = float(np.mean(s_ref))
        blocks_out = []
        for b in blocks:
            sp = sorted(b.get("spectra") or [], key=lambda s: s["order"])
            s = projs(sp) if sp else np.zeros(0)
            blocks_out.append({"series": str(b.get("sid", "")), "label": str(b.get("label", "")),
                               "n": len(sp), "offset_nm": 0.0, "unmatched": [],
                               "assigned": [{"id": x["id"], "station": 0, "residual_nm": round(float(v - m), 4),
                                             "order": int(x["order"])} for x, v in zip(sp, s)]})
        out.update(ok=True, direction=[round(float(u[0]), 6), round(float(u[1]), 6)], step_nm=None,
                   n_stations=1, reference=str(ref.get("sid", "")), blocks=blocks_out)
        return out

    k_ref = np.round((s_ref - ss[0]) / step)
    s0 = float(np.mean(s_ref - k_ref * step))
    if len(np.unique(k_ref)) >= 2:
        A = np.column_stack([k_ref, np.ones_like(k_ref)])
        (step_fit, s0_fit), *_ = np.linalg.lstsq(A, s_ref, rcond=None)
        if 0.5 * step < float(step_fit) < 1.5 * step:
            step, s0 = float(step_fit), float(s0_fit)

    k_fit = np.round((s_ref - s0) / step).astype(int)
    k_lo, k_hi = int(k_fit.min()), int(k_fit.max())

    def wrap(x):
        return x - step * np.round(x / step)

    def frac_offset(s: np.ndarray) -> float:
        """偏移的小数部分：±半个步长内使到最近格点的距离和最小，再用残差中位数精修。"""
        grid = np.linspace(-0.5, 0.5, 201) * step
        costs = [float(np.abs(wrap(s - d - s0)).sum()) + 1e-9 * abs(float(d)) for d in grid]
        d = float(grid[int(np.argmin(costs))])
        d2 = d + float(np.median(wrap(s - d - s0)))
        return d2 if abs(d2) <= 0.5 * step else d

    def place(s: np.ndarray) -> tuple[float, bool]:
        """整步部分：落在参照站位范围内的谱最多 → 其中对得上的最多 → |δ| 最小。``(δ, 是否唯一)``。"""
        d0 = frac_offset(s)
        span = (k_hi - k_lo) + len(s) + 2
        cands = []
        for m in range(-span, span + 1):
            d = d0 + m * step
            kk = np.round((s - d - s0) / step).astype(int)
            inside = (kk >= k_lo) & (kk <= k_hi)
            ok = inside & (np.abs(s - d - (s0 + kk * step)) <= match_frac * step)
            cands.append((-int(inside.sum()), -int(ok.sum()), abs(d), d))
        cands.sort()
        return cands[0][3], sum(1 for c in cands if c[:2] == cands[0][:2]) == 1

    blocks_out = []
    for bi, b in enumerate(blocks):
        sp = sorted(b.get("spectra") or [], key=lambda s: s["order"])
        if not sp:
            blocks_out.append(_empty_block(b))
            continue
        s = projs(sp)
        delta, unique = (0.0, True) if bi == ref_bi else place(s)
        shift = int((shifts or {}).get(str(b.get("sid", "")), 0) or 0)
        delta -= shift * step
        kk = np.round((s - delta - s0) / step).astype(int)
        res = s - delta - (s0 + kk * step)
        assigned, unmatched = [], []
        for x, k, r in zip(sp, kk, res):
            if abs(float(r)) > match_frac * step:
                unmatched.append(str(x["id"]))
            else:
                assigned.append({"id": str(x["id"]), "k": int(k), "residual_nm": float(r), "order": int(x["order"])})
        if not unique and not shift:
            warnings.append("「%s」只落在参照的一部分站位上，整步偏移靠位置定不下来（取漂移最小的 %+.3f nm）"
                            "—— 站位号不对就用 station_shift 平移" % (b.get("label") or b.get("sid"), delta))
        blocks_out.append({"series": str(b.get("sid", "")), "label": str(b.get("label", "")), "n": len(sp),
                           "offset_nm": round(float(delta), 4), "_assigned": assigned, "unmatched": unmatched})
    ks = [a["k"] for bo in blocks_out for a in bo.get("_assigned", [])]
    kmin, kmax = (min(ks), max(ks)) if ks else (0, -1)
    for bo in blocks_out:
        bo["assigned"] = [{"id": a["id"], "station": a["k"] - kmin, "residual_nm": round(a["residual_nm"], 4),
                           "order": a["order"]} for a in bo.pop("_assigned", [])]
    n_unmatched = sum(len(bo["unmatched"]) for bo in blocks_out)
    if n_unmatched:
        warnings.append("%d 条谱离最近的站位超过 %.2f 个步长，没有归入任何站位" % (n_unmatched, match_frac))
    out.update(ok=bool(ks), direction=[round(float(u[0]), 6), round(float(u[1]), 6)],
               step_nm=round(float(step), 5), n_stations=int(kmax - kmin + 1) if ks else 0,
               reference=str(ref.get("sid", "")), blocks=blocks_out)
    return out


# ── 从索引与标记取数 ───────────────────────────────────────────────────


def blocks_from_store(doc: dict, marks: dict, series_ids: list[str]) -> tuple[list[dict], str, list[str]]:
    """``(blocks, 默认线名, warnings)``。区组按首条谱的时刻排；每个系列里谱按开始时刻编采集顺序。"""
    by = C.by_id(doc)
    ser = marks.get("series") or {}
    blocks: list[dict] = []
    warnings: list[str] = []
    for sid in series_ids:
        sv = ser.get(sid)
        if not isinstance(sv, dict):
            warnings.append(f"系列 {sid} 不存在")
            continue
        sp, skipped = [], 0
        for i in sv.get("ids") or []:
            it = by.get(i)
            if not it or it.get("k") != "s":
                continue
            if it.get("ex") or it.get("x") is None or it.get("y") is None:
                skipped += 1
                continue
            sp.append(it)
        if skipped:
            warnings.append("「%s」里 %d 条不是带位置的偏压谱，跳过" % (sv.get("name") or sid, skipped))
        sp.sort(key=lambda it: (it.get("t") or 0, it["id"]))
        blocks.append({"sid": sid, "name": str(sv.get("name") or sid), "rating": sv.get("r") or 0,
                       "spectra": [{"id": it["id"], "x": float(it["x"]), "y": float(it["y"]), "order": o,
                                    "t": float(it.get("t") or 0.0)} for o, it in enumerate(sp)]})
    blocks.sort(key=lambda b: min([s["t"] for s in b["spectra"]] or [float("inf")]))
    line, labels = names_common([b["name"] for b in blocks])
    for b, lab in zip(blocks, labels):
        b["label"] = lab or b["name"]
    return blocks, line, warnings


def station_shifts(options: dict | None) -> dict[str, int]:
    """选项 ``station_shift {系列 id: 整数}``；读不懂的项跳过。"""
    out: dict[str, int] = {}
    for k, v in ((options or {}).get("station_shift") or {}).items():
        try:
            if v not in (None, "") and int(v):
                out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def plan_from_store(series: list[str], options: dict | None = None, lay: _paths.Layout | None = None) -> dict:
    """``GalleryStsLinePlan``：只读索引与标记，不读原始文件。"""
    from mast.gallery import index as _index
    from mast.gallery import marks as _marks

    options = dict(options or {})
    lay = lay or _paths.layout()
    base = {"ok": False, "line_name": "", "direction": [], "step_nm": None, "n_stations": 0,
            "reference": "", "blocks": [], "warnings": [], "degraded": False, "detail": None}
    doc = _index.read_index(lay)
    if not doc or not doc.get("items"):
        return dict(base, detail="图库还没有索引 —— 先构建一次")
    marks = _marks.load_marks(lay)
    blocks, line, warns = blocks_from_store(doc, marks, list(series))
    plan = infer_stations(blocks, shifts=station_shifts(options))
    plan["warnings"] = warns + plan["warnings"]
    plan["line_name"] = str(options.get("line_name") or line or "")
    plan["degraded"] = False
    plan["detail"] = None
    return plan


# ── 色标（绘图实现 color_scales / log_cbar）──────────────────────────────


def color_scales(M, kind: str) -> dict:
    """一组谱（行×偏压）的两种色标：lin = 截在 |x| 99% 分位；log = sign·log₁₀(1+|x|/x₀) 到 99.9% 分位。"""
    M = np.asarray(M, dtype=np.float64)
    a = np.abs(M[np.isfinite(M)])
    v = float(np.percentile(a, 99)) if a.size else 1.0
    x0 = SP.noise_x0(M)
    t = float(SP.slog(np.percentile(a, 99.9), x0)) if a.size else 1.0
    v = v if v > 0 else 1.0                          # 全零（合成数据）时色标不能退化成一个点
    t = t if t > 0 else 1.0
    if kind == "I":
        return dict(cmap="RdBu_r", lin=(-v, v), log=(-t, t), x0=x0)
    return dict(cmap="magma", lin=(-0.05 * v, v), log=(0.0, t), x0=x0)


def log_cbar(cb, x0: float, tlo: float, thi: float) -> None:
    """对数色条：刻度放在 0、±1/2/5×10^k 上，标实际数值。"""
    cand = [(0, 0.0)] + [(p, s * m * 10.0 ** e) for e in range(-4, 8) for p, m in ((1, 1), (2, 5), (3, 2))
                         for s in (1, -1)]
    keep: list[tuple[float, float]] = []
    for _p, v in sorted(cand, key=lambda c: (c[0], -abs(c[1]))):
        t = float(SP.slog(v, x0))
        if tlo - 1e-9 <= t <= thi + 1e-9 and all(abs(t - k) >= 0.075 * (thi - tlo) for k, _ in keep):
            keep.append((t, v))
    keep.sort()
    cb.set_ticks([t for t, _ in keep])
    cb.set_ticklabels(["%g" % v for _, v in keep])


def means(G: list[dict], A: dict, n: int, key: str, nV: int):
    """每个站位：不剔除的谱逐点平均，≥ 2 条时给标准误。"""
    M = np.full((n, nV), np.nan)
    E = np.full_like(M, np.nan)
    cnt = [0] * n
    for i in range(n):
        cs = [A[r["id"]][key] for r in G if r["idx"] == i and not r["excluded"]]
        cnt[i] = len(cs)
        if cs:
            M[i] = np.mean(cs, axis=0)
            if len(cs) >= 2:
                E[i] = np.std(cs, axis=0, ddof=1) / math.sqrt(len(cs))
    return M, E, cnt


def _p90_step(arr) -> float:
    a = np.abs(np.asarray(arr, dtype=np.float64))
    v = float(np.nanpercentile(a, 90)) * 0.30 if np.isfinite(a).any() else 0.0
    return v if math.isfinite(v) and v > 0 else 1.0


# ── 作图 ───────────────────────────────────────────────────────────────


def _bold(size: float):
    f = C.fp(size)
    f.set_weight("bold")
    return f


def _heat(ax, M, V, n, cmap, vmin, vmax, labels=None):
    ax.set_facecolor(NODATA)                         # 删去 / 没测的站位留灰
    im = ax.imshow(M, aspect="auto", origin="upper", cmap=cmap, extent=[V[0], V[-1], n - 0.5, -0.5],
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    for sp in ax.spines.values():
        sp.set_edgecolor(C.EDGE)
    ax.tick_params(colors=C.INK_HEX, labelsize=11)
    ax.set_yticks(range(n))
    if labels is None:
        ax.tick_params(labelleft=False)
    else:
        ax.set_yticklabels(labels, fontproperties=C.fp(8))
    ax.set_xlabel("偏压 (V)", color=C.INK_HEX, fontproperties=C.fp(9.5))
    return im


def _cbar(cb, label: str) -> None:
    cb.ax.tick_params(labelsize=8, colors=C.INK_HEX)
    cb.set_label(label, color=C.INK_HEX, fontproperties=C.fp(9))
    cb.outline.set_edgecolor(C.EDGE)


def _title(ax, text: str, size: float = 11) -> None:
    ax.set_title(text, color=C.INK_HEX, fontproperties=C.fp(size))


def _suptitle(fig, text: str, size: float, y: float) -> None:
    fig.suptitle(text, color=C.INK_HEX, fontproperties=C.fp(size), y=y)


class _Ctx:
    """一条线的全部作图材料（在锁外准备好，锁里只画）。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def yt_labels(self, extra=None) -> list[str]:
        return ["#%d%s%s" % (i, (" " + self.st_marks[i]) if i in self.st_marks else "",
                             extra[i] if extra else "") for i in range(self.n)]

    def color(self, blk: int) -> str:
        return C.BLOCK_COLORS[blk % len(C.BLOCK_COLORS)]

    def left_stms(self, fig, spec_left, only=None) -> None:
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        use = self.blks if only is None else [only]
        gl = GridSpecFromSubplotSpec(len(use), 1, subplot_spec=spec_left, hspace=0.32)
        for c, b in enumerate(use):
            g = sorted((r for r in self.G if r["blk"] == b), key=lambda r: r["idx"])
            ax = fig.add_subplot(gl[c])
            labels = ["#%d" % r["idx"] if (r["idx"] in self.st_marks or r is g[0] or r is g[-1]) else "" for r in g]
            SP.stm_panel(ax, self.stm[b], [(self.S[r["id"]]["x"], self.S[r["id"]]["y"]) for r in g], labels,
                         [r["idx"] in self.st_marks for r in g], self.t0[b],
                         head=("%s · " % self.blabel[b]) if only is None else "", cache=self.cache)

    def drop_note(self) -> str:
        """「已删去 区组2 的 #6–#10（针尖在变）」；没删就空串。"""
        if not self.drop:
            return ""
        parts = []
        for b in sorted({r["blk"] for r in self.drop}):
            rows = [r for r in self.drop if r["blk"] == b]
            why = "针尖在变" if all(r["bad"] for r in rows) else "系列评为排除"
            if not any(r["blk"] == b for r in self.G):
                parts.append("%s 整组 %d 条（%s）" % (self.blabel[b], len(rows), why))
                continue
            ids = sorted({r["idx"] for r in rows})
            seg, a = [], ids[0]
            for p, q in zip(ids, ids[1:] + [None]):
                if q != p + 1:
                    seg.append("#%d" % a + ("–#%d" % p if p != a else ""))
                    a = q
            parts.append("%s 的 %s（%s）" % (self.blabel[b], "、".join(seg), why))
        return "已删去 " + "；".join(parts)

    def head2(self) -> str:
        note = self.drop_note()
        return "%s · %s%s" % (self.when, self.spec_desc, (" · " + note) if note else "")


def _fig_compare_heat(x: _Ctx):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    nb = len(x.blks)
    fig = C.new_figure(FIG)
    gs = GridSpec(1, 2, width_ratios=[0.185, 0.815], wspace=0.075, left=0.022, right=0.955, top=0.845,
                  bottom=0.065, figure=fig)
    x.left_stms(fig, gs[0])
    wr = [1] * nb + [0.055, 0.62] + [1] * nb + [0.055]           # 线性组 | 色条 | 空 | 对数组 | 色条
    gr = GridSpecFromSubplotSpec(2, len(wr), subplot_spec=gs[1], width_ratios=wr, hspace=0.27, wspace=0.06)
    labels = x.yt_labels()
    x0s, heads = [], []
    for row, (kind, lab, unit) in enumerate((("I", "Current", "pA"), ("D", SP.dshort(x.dname), x.dunit))):
        sc = x.sc_cmp[kind]
        x0s.append("%s x₀ = %g %s" % ("Current" if kind == "I" else "dI/dV", sc["x0"], unit))
        for mode in ("lin", "log"):
            c0 = 0 if mode == "lin" else nb + 2
            axs = []
            im = None
            for c, b in enumerate(x.blks):
                ax = fig.add_subplot(gr[row, c0 + c])
                M = x.mats[b][row] if mode == "lin" else SP.slog(x.mats[b][row], sc["x0"])
                im = _heat(ax, M, x.V, x.n, sc["cmap"], *sc[mode], labels=labels if c == 0 else None)
                _title(ax, "%s · %s" % (x.blabel[b], lab), 10)
                axs.append(ax)
            cb = fig.colorbar(im, cax=fig.add_subplot(gr[row, c0 + nb]))
            _cbar(cb, "%s (%s)" % ("Current" if kind == "I" else "dI/dV", unit))
            if mode == "log":
                log_cbar(cb, sc["x0"], *sc["log"])
            if row == 0:
                heads.append((axs[0].get_position().x0, axs[-1].get_position().x1, axs[0].get_position().y1))
    for (xa, xb, y), txt in zip(heads, ("线性色标（同一行各区组共用，截在 99% 分位）",
                                     "对数色标 sign(x)·log₁₀(1+|x|/x₀) · " + "，".join(x0s))):
        fig.text((xa + xb) / 2, y + 0.034, txt, ha="center", va="bottom", color=C.TEAL, fontproperties=_bold(10.5))
    _suptitle(fig, "%s · 热图（各区组并排比较；灰行＝删去或没测）\n%s" % (x.line_name, x.head2()), 13, 0.978)
    return fig


def _fig_compare_fall(x: _Ctx):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    fig = C.new_figure(FIG)
    gs = GridSpec(1, 2, width_ratios=[0.19, 0.81], wspace=0.08, left=0.02, right=0.985, top=0.885, bottom=0.11,
                  figure=fig)
    x.left_stms(fig, gs[0])
    gr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.16)
    labels = x.yt_labels()
    for c, (key, lab, unit) in enumerate((("If", "Current", "pA"), ("Df", SP.dshort(x.dname), x.dunit))):
        ax = fig.add_subplot(gr[0, c])
        C.style_axes(ax)
        step = _p90_step([x.A[r["id"]][key] for r in x.G])
        for r in sorted(x.G, key=lambda r: (r["idx"], r["blk"])):
            ax.plot(x.V, x.A[r["id"]][key] + r["idx"] * step, "-", lw=1.25, color=x.color(r["blk"]), alpha=0.9)
        for i in range(x.n):
            ax.axhline(i * step, color=C.DIM_HEX, lw=0.45, alpha=0.45)
        ax.set_yticks([i * step for i in range(x.n)])
        ax.set_yticklabels(labels, fontproperties=C.fp(9))
        ax.set_ylim(-2.5 * step, (x.n + 1.5) * step)          # 冲出范围的大峰看热图
        ax.invert_xaxis()
        ax.grid(alpha=0.2, axis="x")
        ax.set_xlabel("偏压 (V)", color=C.INK_HEX, fontproperties=C.fp(10))
        _title(ax, "%s · 每条按站位抬高 %.3g %s（灰线＝各自零点）" % (lab, step, unit))
    hs = [Line2D([], [], color=x.color(b), lw=2, label=x.blabel[b]) for b in x.blks]
    leg = fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, bbox_to_anchor=(0.6, 0.005),
                     prop=C.fp(10.5))
    for t in leg.get_texts():
        t.set_color(C.INK_HEX)
    _suptitle(fig, "%s · 瀑布（各区组叠在一起比较，正扫）\n%s" % (x.line_name, x.head2()), 13.5, 0.975)
    return fig


def _fig_mean(x: _Ctx, kind: str):
    from matplotlib import colormaps
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = C.new_figure(FIG)
    gs = GridSpec(1, 2, width_ratios=[0.22, 0.78], wspace=0.14, left=0.02, right=0.965, top=0.885, bottom=0.07,
                  figure=fig)
    x.left_stms(fig, gs[0], only=x.blks[0])
    labels = x.yt_labels(extra=["  ×%d" % c for c in x.cnt])
    quants = ((x.MI, x.EI, "I", "Current", "pA"), (x.MD, x.ED, "D", SP.dshort(x.dname), x.dunit))
    if kind == "heat":
        gr = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], hspace=0.3, wspace=0.26)
        for c, (M, _E, k, lab, unit) in enumerate(quants):
            sc = x.sc_mean[k]
            for row, mode in enumerate(("lin", "log")):
                ax = fig.add_subplot(gr[row, c])
                im = _heat(ax, M if mode == "lin" else SP.slog(M, sc["x0"]), x.V, x.n, sc["cmap"], *sc[mode],
                           labels=labels)
                cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
                _cbar(cb, "%s (%s)" % ("Current" if k == "I" else "dI/dV", unit))
                if mode == "log":
                    log_cbar(cb, sc["x0"], *sc["log"])
                _title(ax, "%s · 站位均值 · %s" % (lab, "线性色标" if mode == "lin" else
                                                  "对数色标 sign·log₁₀(1+|x|/x₀)，x₀ = %g %s" % (sc["x0"], unit)), 10.5)
    else:
        viridis = colormaps["viridis"]
        gr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.16)
        for c, (M, E, _k, lab, unit) in enumerate(quants):
            ax = fig.add_subplot(gr[0, c])
            C.style_axes(ax)
            step = _p90_step(M)
            for i in range(x.n):
                if not np.isfinite(M[i]).all():
                    continue
                col = viridis(i / max(x.n - 1, 1))
                y = M[i] + i * step
                if np.isfinite(E[i]).all():
                    ax.fill_between(x.V, y - E[i], y + E[i], color=col, alpha=0.25, lw=0)
                ax.plot(x.V, y, "-", lw=1.3, color=col)
                ax.axhline(i * step, color=C.DIM_HEX, lw=0.45, alpha=0.45)
            ax.set_yticks([i * step for i in range(x.n)])
            ax.set_yticklabels(labels, fontproperties=C.fp(9))
            ax.set_ylim(-2.5 * step, (x.n + 1.5) * step)
            ax.invert_xaxis()
            ax.grid(alpha=0.2, axis="x")
            ax.set_xlabel("偏压 (V)", color=C.INK_HEX, fontproperties=C.fp(10))
            _title(ax, "%s · 均值 ± 标准误，每条抬高 %.3g %s" % (lab, step, unit))
    _suptitle(fig, "%s · %s（重复测量取均值：每条正反扫平均，再在区组间等权平均；×n＝该站位用了几条）\n%s · 左图＝%s 这组谱之前的最后一张 STM"
              % (x.line_name, "均值热图" if kind == "heat" else "均值瀑布", x.head2(), x.blabel[x.blks[0]]),
              12.5, 0.975)
    return fig


def _span_text(t0, t1) -> str:
    if not t0:
        return "?"
    same_day = t1 and C.local(t0, "%Y%m%d") == C.local(t1, "%Y%m%d")
    return "%s → %s" % (C.local(t0, "%m-%d %H:%M"), C.local(t1, "%H:%M" if same_day else "%m-%d %H:%M"))


def make_lines(lay: _paths.Layout, doc: dict, marks: dict, series: list[str], options: dict | None = None,
               cache: dict | None = None) -> str:
    cache = {} if cache is None else cache
    options = dict(options or {})
    blocks, line_default, warns = blocks_from_store(doc, marks, list(series))
    plan = infer_stations(blocks, shifts=station_shifts(options))
    warns = warns + list(plan["warnings"])
    if not plan["ok"]:
        raise C.JobError("；".join(warns) or "推断不出站位")
    line_name = str(options.get("line_name") or line_default or "拉线谱")
    by = C.by_id(doc)
    st_marks: dict[int, str] = {}
    for k, v in (options.get("station_marks") or {}).items():
        try:
            if str(v).strip():
                st_marks[int(k)] = str(v).strip()
        except (TypeError, ValueError):
            continue
    bad_from: dict[str, int] = {}
    for k, v in (options.get("bad_from") or {}).items():
        try:
            if v not in (None, ""):
                bad_from[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    exclude_rejected = bool(options.get("exclude_rejected", True))
    rating = {b["sid"]: b.get("rating") or 0 for b in blocks}
    n = int(plan["n_stations"])

    rows: list[dict] = []
    for bi, bp in enumerate(plan["blocks"]):
        sid = bp["series"]
        for a in bp["assigned"]:
            bad = sid in bad_from and a["order"] >= bad_from[sid]
            rejected = exclude_rejected and rating.get(sid) == -1
            rows.append(dict(blk=bi, sid=sid, idx=int(a["station"]), id=a["id"], order=int(a["order"]), bad=bool(bad),
                             excluded=bool(bad or rejected)))
    drop = [r for r in rows if r["excluded"]]
    G: list[dict] = []
    S: dict = {}
    load_errors: list[dict] = []
    for r in rows:
        if r["excluded"]:
            continue
        try:
            S[r["id"]] = SP.load_spectrum(by[r["id"]], cache)
        except Exception as exc:  # noqa: BLE001 — 一条读不出的谱不拖垮整条线
            load_errors.append({"id": r["id"], "why": str(exc)[:200]})
            continue
        G.append(r)
    if not G:
        raise C.JobError("这条线上没有可画的谱（%d 条全部剔除%s）" % (
            len(rows), "，%d 条读不出来" % len(load_errors) if load_errors else ""))
    blks = sorted({r["blk"] for r in G})
    blabel = {bi: bp["label"] or bp["series"] for bi, bp in enumerate(plan["blocks"])}
    first = S[G[0]["id"]]
    V = first["V"]
    A = {i: {key: SP.on_grid(V, s["V"], s[key]) for key in ("If", "I", "Df", "D")} for i, s in S.items()}
    dnames = {s["dname"] for s in S.values()}
    if len(dnames) > 1:
        warns.append("这一组谱的 dI/dV 来源不一致（%s），图里用第一条的名字与单位" % "、".join(sorted(dnames)))

    mats = {b: [np.full((n, len(V)), np.nan), np.full((n, len(V)), np.nan)] for b in blks}
    for r in G:
        mats[r["blk"]][0][r["idx"]] = A[r["id"]]["If"]
        mats[r["blk"]][1][r["idx"]] = A[r["id"]]["Df"]
    sc_cmp = {kind: color_scales(np.concatenate([mats[b][row] for b in blks]), kind)
              for row, kind in enumerate(("I", "D"))}
    MI, EI, cnt = means(G, A, n, "I", len(V))
    MD, ED, _ = means(G, A, n, "D", len(V))
    sc_mean = {"I": color_scales(MI, "I"), "D": color_scales(MD, "D")}

    tl = C.Timeline(doc.get("items") or [])
    t0 = {b: min(float(S[r["id"]]["t"] or 0.0) for r in G if r["blk"] == b) for b in blks}
    stm = {b: tl.last_before(t0[b]) for b in blks}
    for b, f in list(stm.items()):                  # 读帧放在锁外
        if f is None:
            continue
        try:
            SP.frame_image(f, cache)
        except Exception as exc:  # noqa: BLE001
            warns.append("帧 %s 读不出来：%s" % (f.get("fn"), exc))
            stm[b] = None
    t_all = [float(S[r["id"]]["t"]) for r in G if S[r["id"]]["t"]]
    mt_all = [float(S[r["id"]]["mt"]) for r in G if S[r["id"]]["mt"]]
    when = _span_text(min(t_all) if t_all else None, max(mt_all or t_all) if (mt_all or t_all) else None)
    spec_desc = ("%+.1f→%+.1f V · %d 点 · %s sweeps · Zoff %.0f pm" % (
        float(V[0]), float(V[-1]), len(V), first["sweeps"], first["zoff"])).replace("-", "−")
    x = _Ctx(G=G, drop=drop, S=S, A=A, V=V, n=n, blks=blks, blabel=blabel, st_marks=st_marks, dname=first["dname"],
             dunit=first["dunit"], line_name=line_name, when=when, spec_desc=spec_desc, t0=t0, stm=stm, cache=cache,
             mats=mats, sc_cmp=sc_cmp, MI=MI, EI=EI, MD=MD, ED=ED, cnt=cnt, sc_mean=sc_mean)

    base = C.safe_name(line_name.replace(" · ", "_"), "拉线谱")
    files: list[str] = []
    for suffix, make in (("_比较_热图", lambda: _fig_compare_heat(x)), ("_比较_瀑布", lambda: _fig_compare_fall(x)),
                         ("_均值_热图", lambda: _fig_mean(x, "heat")), ("_均值_瀑布", lambda: _fig_mean(x, "fall"))):
        with C.MPL_LOCK:
            data = C.savefig_bytes(make(), DPI)
        name = base + suffix + ".png"
        C.write_bytes(lay, CAT, name, data)
        files.append(name)

    n_unmatched = sum(len(b["unmatched"]) for b in plan["blocks"])
    summary = {"区组": len(blks), "站位数": n, "谱": len(G), "剔除": len(drop), "对不上": n_unmatched,
               "dI/dV": first["dname"].replace("dI/dV · ", "").replace("电流", "")}
    if plan.get("step_nm"):
        summary["步长 pm"] = round(float(plan["step_nm"]) * 1000, 1)
    if load_errors:
        summary["读不出"] = len(load_errors)

    def _sc(sc):
        return {"x0": sc["x0"], "lin": list(sc["lin"]), "log": list(sc["log"])}

    detail = {"plan": dict(plan, line_name=line_name), "per_station_n": cnt,
              "rows": [{k: r[k] for k in ("id", "sid", "blk", "idx", "order", "bad", "excluded")} for r in rows],
              "drop_note": x.drop_note(), "didv": first["dname"], "dunit": first["dunit"],
              "scales": {"compare": {k: _sc(v) for k, v in sc_cmp.items()},
                         "mean": {k: _sc(v) for k, v in sc_mean.items()}},
              "load_errors": load_errors, "warnings": warns}
    return C.write_figure_json(lay, CAT, base, kind="sts_lines", title="%s · 拉线谱" % line_name, files=files,
                               ids=[r["id"] for r in G], series=[b["series"] for b in plan["blocks"]],
                               options=options, summary=summary, detail=detail, maker_version=VER)
