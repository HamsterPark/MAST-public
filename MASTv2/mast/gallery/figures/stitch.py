# -*- coding: utf-8 -*-
"""单根谱宽范围拼接（ / ``factor_to_core``；设计文档 §10.1，T30 / T31）。

拼接约定：

* 核心 = 覆盖 0 V、sweep × 点数最多的一段（同范围同 Zoff 的几条先平均）；
* 每一段都有有效锁相（dI/dV 单位 fA）才用 LI，否则全部用电流数值求导 —— 免得单位混；
* 其余各段按 **dI/dV** 在接缝处对齐求一个系数 ×k，电流乘同一个 k：
  与核心在接缝内侧（离接缝 ≤ :data:`SEAM_W`）重叠 ≥ :data:`MIN_INSIDE` 点 ⇒ 这些点上两者均值之比；
  否则两边各取离接缝最近 5 点拟直线，比接缝处的值（点数 < 3 取均值）；
* 表格并列「电流在接缝的比」（同一算法作用在电流上）与按 Zoff 差估的 exp(2κΔz)（仅在调用方显式指定 κ 时计算）；
* 边缘段 LI / 电流数值导数之比（等分 5 份）最大/最小 > :data:`LI_VARY` 时，另画「电流数值求导 × 锁相响应」
  虚线作参照（原版 docstring 写「变化 > 20%」，代码是 1.35 —— 照代码）；
* 上排 Current、下排 dI/dV；左列线性纵轴、右列 symlog（linthresh = 噪声 5σ 就近的 1-2-5 值）；
  左栏 = 各段系定的 STM（没系定就用谱之前最后一张）。

T30：原版更早的 docstring 写「重叠 ≥ 5 点」，代码一直是 ≥ 3 点 —— 以代码为准。
T31：数值导数用出图那条规则（窗口按电压），与缩略图的按点数分开。
输入里的非偏压谱（索引带 ``ex`` 的）跳过，记进 figure.json（``detail.skipped``、``summary.跳过``）与任务 errors。
"""
from __future__ import annotations

import math

import numpy as np

from mast.gallery import paths as _paths
from mast.gallery.figures import common as C
from mast.gallery.figures import spectra as SP

VER = 1
CAT = "sts_stitch"
FIG, DPI = (16, 9), 200
SEAM_W = 0.15
MIN_INSIDE = 3
EDGE_K = 5
LI_VARY = 1.35


def single_groups(doc: dict, marks: dict) -> list[tuple[str, list[str]]]:
    """带标记、是谱、不在任何系列里的单根谱，按目录分组（组内按开始时刻排）。

    非偏压谱（``ex``）**留在组里**，由 :func:`make_stitch` 跳过并记下 —— 在这里悄悄滤掉，
    操作员就不知道自己标的那条为什么没进图（原版按 ``.dat`` 后缀收，同样不看实验类型）。"""
    by = C.by_id(doc)
    in_series = {i for sv in (marks.get("series") or {}).values() for i in (sv.get("ids") or [])}
    groups: dict[str, list[str]] = {}
    for k in (marks.get("items") or {}):
        it = by.get(k)
        if not it or it.get("k") != "s" or k in in_series:
            continue
        groups.setdefault(it["d"], []).append(k)
    return [(d, sorted(ids, key=lambda i: (by[i].get("t") or 0, i))) for d, ids in sorted(groups.items())]


def edge_val(V, D, Vj: float, side: int, k: int = EDGE_K) -> float:
    """接缝 Vj 一侧（side=+1 取 V ≥ Vj，−1 取 V ≤ Vj）离接缝最近 k 点拟直线，取它在 Vj 处的值。"""
    m = np.where((V - Vj) * side >= -1e-6)[0]
    m = m[np.argsort(np.abs(V[m] - Vj))][:k]
    if len(m) >= 3:
        return float(np.polyval(np.polyfit(V[m] - Vj, D[m], 1), 0.0))
    return float(np.mean(D[m]))


def factor_to_core(core: dict, piece: dict, key: str, W: float = SEAM_W,
                   min_inside: int = MIN_INSIDE) -> tuple[float, str]:
    """``(系数, 说明)``：``piece[key]`` 乘上系数后在接缝处接上 ``core[key]``（V 都已升序）。"""
    Vc, Dc, Vp, Dp = core["V"], core[key], piece["V"], piece[key]
    up = Vp.max() > Vc.max() + 1e-6
    Vj = Vc.max() if up else Vc.min()
    near = (Vp >= Vc.min() - 1e-6) & (Vp <= Vc.max() + 1e-6) & (np.abs(Vp - Vj) <= W)
    with np.errstate(all="ignore"):
        if near.sum() >= min_inside:
            return (float(np.mean(np.interp(Vp[near], Vc, Dc)) / np.mean(Dp[near])),
                    "%+.2f V 内侧 %d 点均值比" % (Vj, near.sum()))
        side = 1 if up else -1
        return edge_val(Vc, Dc, Vj, -side) / edge_val(Vp, Dp, Vj, side), "%+.2f V 两侧各 5 点外推到接缝" % Vj


def li_ratio_bins(D, Dn, nb: int = 5) -> list[float]:
    """把一段谱等分 nb 份，各份 LI / 电流数值导数 的比（只算数值导数够大的份）。"""
    big = np.abs(Dn).max()
    return [float(np.mean(D[s]) / np.mean(Dn[s])) for s in np.array_split(np.arange(len(D)), nb)
            if len(s) >= 3 and np.mean(Dn[s]) > 0.1 * big]


def _sweeps_num(s) -> float:
    txt = str(s)
    return float(txt) if txt.replace(".", "").isdigit() else 1.0


def _bold(size: float):
    f = C.fp(size)
    f.set_weight("bold")
    return f


def make_stitch(lay: _paths.Layout, doc: dict, marks: dict, ids: list[str], options: dict | None = None,
                cache: dict | None = None, report=None) -> str:
    cache = {} if cache is None else cache
    options = dict(options or {})
    raw_kappa = options.get("kappa_per_nm")
    kappa = None if raw_kappa in (None, "") else float(raw_kappa)
    if kappa is not None and (not math.isfinite(kappa) or kappa <= 0):
        raise C.JobError("κ 必须是有限正数（nm⁻¹），或留空以省略理论倍率")
    by = C.by_id(doc)
    skipped: list[dict] = []

    def skip(k, why):
        skipped.append({"id": k, "why": why})
        if report is not None:
            report(k, "跳过：" + why)

    S = []
    for k in ids:
        it = by.get(k)
        if not it:
            skip(k, "索引里没有这一项")
        elif it.get("k") != "s":
            skip(k, "不是谱")
        elif it.get("ex"):
            skip(k, "不是偏压谱（%s）" % it["ex"])
        else:
            try:
                S.append(SP.load_spectrum(it, cache))
            except Exception as exc:  # noqa: BLE001
                skip(k, str(exc)[:200])
    if not S:
        raise C.JobError("没有能拼的偏压谱" + ("（%d 条全部跳过）" % len(skipped) if skipped else ""))
    S.sort(key=lambda s: (s["t"] or 0, s["id"]))
    keys = [s["id"] for s in S]

    li = all(s["dunit"] == "fA" for s in S)       # 每段都有有效锁相才用 LI，否则全用电流数值求导
    P = []
    for s in S:
        o = np.argsort(s["V"])
        Dn = C.deriv_by_volts(s["V"][o], s["I"][o])
        P.append(dict(V=s["V"][o], I=s["I"][o], D=s["D"][o] if li else Dn, Dn=Dn))
    dname, dunit = (S[0]["dname"], "fA") if li else ("dI/dV · 电流数值求导", "pA/V")
    ranges = [(round(np.min(s["V"]), 2), round(np.max(s["V"]), 2), round(s["zoff"])) for s in S]
    covers0 = [i for i, (a, b, _z) in enumerate(ranges) if a < 0 < b]

    def sw(i):
        return _sweeps_num(S[i]["sweeps"]) * len(S[i]["V"])

    ci = max(covers0, key=sw) if covers0 else 0
    core_ids = [i for i in range(len(S)) if ranges[i] == ranges[ci]]
    core = dict(V=P[ci]["V"])
    for key in ("I", "D", "Dn"):
        core[key] = np.mean([np.interp(core["V"], P[i]["V"], P[i][key]) for i in core_ids], axis=0)
    big = np.abs(core["Dn"]) > 0.2 * np.abs(core["Dn"]).max()
    with np.errstate(all="ignore"):
        resp = float(np.median(core["D"][big] / core["Dn"][big])) if big.any() else float("nan")

    tl = C.Timeline(doc.get("items") or [])
    items = marks.get("items") or {}
    anchors = []
    for s in S:
        a = (items.get(s["id"]) or {}).get("anchor") or {}
        f = by.get(a.get("id")) if a.get("id") else None
        anchors.append(f if (f and f.get("k") == "f" and not f.get("dup")) else tl.last_before(s["t"]))
    uniq: list[dict] = []
    for f in anchors:
        if f is not None and f["id"] not in [u["id"] for u in uniq]:
            uniq.append(f)
    for f in list(uniq):                            # 读帧放在锁外
        try:
            SP.frame_image(f, cache)
        except Exception:  # noqa: BLE001
            uniq.remove(f)
            anchors = [None if (g is not None and g["id"] == f["id"]) else g for g in anchors]

    cols = [C.INK_HEX, C.TEAL, C.ROSE, C.AMBER, C.PURPLE, C.BLUE, C.GREEN]
    core_short = "+".join(S[i]["fn"][-9:-4] for i in core_ids)
    table: list[str] = []
    draws: list[dict] = []
    segs: list[dict] = []
    lo_all, hi_all = float(core["V"][0]), float(core["V"][-1])
    for n_i, s in enumerate(S):
        p = P[n_i]
        seg = {"id": s["id"], "fn": s["fn"], "V0": float(s["V"][0]), "V1": float(s["V"][-1]), "zoff": s["zoff"],
               "sweeps": s["sweeps"], "dname": s["dname"]}
        if n_i in core_ids:
            if n_i != core_ids[0]:
                segs.append(dict(seg, role="core_member", factor=1.0))
                continue
            col, fac = C.INK_HEX, 1.0
            Vuse, Iuse, Duse = core["V"], core["I"], core["D"]
            table.append("%s  %+.2f→%+.2f V · Zoff %.0f pm · %s sweeps   核心 ×1%s" % (
                "+".join(S[i]["fn"][:-4] if i == core_ids[0] else S[i]["fn"][-9:-4] for i in core_ids),
                s["V"][0], s["V"][-1], s["zoff"], s["sweeps"],
                "（%d 条平均）" % len(core_ids) if len(core_ids) > 1 else ""))
            draw = dict(col=col, dotted=None, dashed=None, leg="%s 核心 ×1" % core_short)
            segs.append(dict(seg, role="core", factor=1.0))
        else:
            col = cols[1 + (n_i % (len(cols) - 1))]
            fac, how = factor_to_core(core, p, "D")
            fac_i, _ = factor_to_core(core, p, "I")
            use = (p["V"] > core["V"][-1] + 1e-6) | (p["V"] < core["V"][0] - 1e-6)
            Vuse, Iuse, Duse = p["V"][use], p["I"][use] * fac, p["D"][use] * fac
            th = (math.exp(2 * kappa * (s["zoff"] - S[ci]["zoff"]) / 1000.0)
                  if kappa is not None else None)
            warn = ""
            rb = li_ratio_bins(p["D"][use], p["Dn"][use]) if li and use.sum() >= 15 else []
            dashed = None
            if len(rb) >= 2 and max(rb) / min(rb) > LI_VARY:
                dashed = (Vuse, p["Dn"][use] * resp * fac, "%s 电流数值求导×%.1f（参照）" % (s["fn"][-9:-4], resp))
                warn = ("\n      注意：这段 LI/数值导数 在 %.1f–%.1f 之间变（核心 %.1f），LI 形状可能失真；"
                        "虚线＝该段电流数值求导×%.1f×%.3g" % (min(rb), max(rb), resp, resp, fac))
            table.append("%s  %+.2f→%+.2f V · Zoff %.0f pm · %s sweeps   ×%.3g  [%s]   电流在接缝的比 ×%.3g%s%s" % (
                s["fn"][:-4], s["V"][0], s["V"][-1], s["zoff"], s["sweeps"], fac, how, fac_i,
                " · Zoff 差估 ×%.3g" % th if th is not None and abs(s["zoff"] - S[ci]["zoff"]) > 1 else "", warn))
            draw = dict(col=col, dotted=(p["V"], p["I"], p["D"]), dashed=dashed, leg="%s ×%.3g" % (s["fn"][-9:-4], fac))
            segs.append(dict(seg, role="piece", factor=fac, how=how, factor_I=fac_i, theory=th, li_bins=rb,
                             dashed=dashed is not None))
        draw["solid"] = (Vuse, Iuse, Duse) if len(Vuse) >= 3 else None
        if draw["solid"] is not None:
            lo_all, hi_all = min(lo_all, float(Vuse[0])), max(hi_all, float(Vuse[-1]))
        draws.append(draw)
    x0I, x0D = SP.noise_x0(core["I"]), SP.noise_x0(core["D"])

    pos = np.array([(s["x"], s["y"]) for s in S])
    spread = float(np.max(np.hypot(pos[:, 0] - pos[:, 0].mean(), pos[:, 1] - pos[:, 1].mean()))) * 2
    d = str(by[keys[0]].get("d") or "")
    dlast = d.rsplit("/", 1)[-1] if d else "?"
    t0 = min(float(s["t"] or 0.0) for s in S)
    t1 = max(float(by[k].get("mt") or by[k].get("t") or 0.0) for k in keys)

    with C.MPL_LOCK:
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        from matplotlib.ticker import FuncFormatter, NullLocator

        fig = C.new_figure(FIG)
        gs = GridSpec(1, 2, width_ratios=[0.2, 0.8], wspace=0.1, left=0.015, right=0.985, top=0.885, bottom=0.025,
                      figure=fig)
        gl = GridSpecFromSubplotSpec(max(1, len(uniq)), 1, subplot_spec=gs[0], hspace=0.3)
        for c, f in enumerate(uniq):
            own = [i for i in range(len(S)) if anchors[i] is not None and anchors[i]["id"] == f["id"]]
            SP.stm_panel(fig.add_subplot(gl[c]), f, [(S[i]["x"], S[i]["y"]) for i in own],
                         [S[i]["fn"][-9:-4] for i in own], [False] * len(own),
                         min(float(S[i]["t"] or 0.0) for i in own), cache=cache)
        gr = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1], height_ratios=[1, 1, 0.40], hspace=0.22, wspace=0.13)
        axIl = fig.add_subplot(gr[0, 0])
        axIs = fig.add_subplot(gr[0, 1], sharex=axIl)
        axDl = fig.add_subplot(gr[1, 0], sharex=axIl)
        axDs = fig.add_subplot(gr[1, 1], sharex=axIl)
        axT = fig.add_subplot(gr[2, :])
        axT.axis("off")
        for dr in draws:
            col = dr["col"]
            if dr["dotted"] is not None:
                Vr, Ir, Dr = dr["dotted"]
                axIs.plot(Vr, Ir, ":", color=col, lw=1.0, alpha=0.55)
                axDs.plot(Vr, Dr, ":", color=col, lw=0.9, alpha=0.5)
            if dr["dashed"] is not None:
                Vd, Dd, lab = dr["dashed"]
                for ax in (axDl, axDs):
                    ax.plot(Vd, Dd, "--", color=col, lw=1.1, alpha=0.8, label=lab)
            if dr["solid"] is not None:
                Vuse, Iuse, Duse = dr["solid"]
                for axI, axD in ((axIl, axDl), (axIs, axDs)):
                    axI.plot(Vuse, Iuse, "-", color=col, lw=1.6, label=dr["leg"])
                    axD.plot(Vuse, Duse, "-", color=col, lw=1.5, label=dr["leg"])
        dlab = "%s (%s)" % (SP.dshort(dname), dunit)
        for ax, lab, x0 in ((axIl, "Current (pA)", None), (axIs, "Current (pA)", x0I), (axDl, dlab, None),
                            (axDs, dlab, x0D)):
            C.style_axes(ax)
            if x0:
                ax.set_yscale("symlog", linthresh=x0, linscale=0.6)
                lo, hi = ax.get_ylim()                    # 刻度只放 0 和线性段以外的 ±10^k
                ax.set_yticks([0.0] + [sg * 10.0 ** e for e in range(-3, 7) for sg in (1, -1)
                                       if 10.0 ** e >= 1.5 * x0 and lo <= sg * 10.0 ** e <= hi])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
                ax.yaxis.set_minor_locator(NullLocator())
            ax.axhline(0, color=C.DIM_HEX, lw=0.6)
            for vb in (core["V"][0], core["V"][-1]):
                ax.axvline(vb, color=C.DIM_HEX, lw=0.8, ls="--", alpha=0.6)
            ax.grid(alpha=0.22)
            ax.set_ylabel(lab, color=C.INK_HEX, fontproperties=C.fp(10))
            if ax.get_legend_handles_labels()[0]:
                leg = ax.legend(loc="best", prop=_bold(10.5), framealpha=0.85, edgecolor=C.EDGE)
                for t in leg.get_texts():
                    t.set_color(C.INK_HEX)
        axIl.set_title("线性纵轴", color=C.TEAL, fontproperties=_bold(11.5))
        axIs.set_title("symlog 纵轴（|y| < x₀ 段线性；x₀：Current %g pA，dI/dV %g %s）" % (x0I, x0D, dunit),
                       color=C.TEAL, fontproperties=_bold(11.5))
        for ax in (axIl, axIs):
            ax.tick_params(labelbottom=False)
        for ax in (axDl, axDs):
            ax.set_xlabel("偏压 (V)", color=C.INK_HEX, fontproperties=C.fp(10))
        axT.text(0.0, 1.0, "\n".join(table), transform=axT.transAxes, va="top", ha="left", color=C.INK_HEX,
                 fontproperties=C.fp(9.5, mono=True))
        axT.text(0.0, 0.0, "拼接：每段在接缝处按 %s 对齐求系数 ×k（写在图例里），电流乘同一个 k；symlog 图里点线＝边缘段原始值，灰虚线＝核心段两端。\n"
                           "「电流在接缝的比」和 k 差得多，说明两段在接缝处的谱形不同（如峰位随针尖高度移动），电流图接缝处就有台阶，"
                           "这一侧的相对高低要谨慎看；%s" % (SP.dshort(dname),
                           ("Zoff 差估按指定的 κ = %g nm⁻¹ 算，仅作参照。" % kappa
                            if kappa is not None else "未指定 κ，不计算 Zoff 理论倍率。")),
                 transform=axT.transAxes, va="bottom", ha="left", color=C.DIM_HEX, fontproperties=C.fp(9.5))
        fig.suptitle("单根标记谱拼接 · 目录 %s · %s → %s · 以 %s 为核心、按 %s 拼到 %+.2f…%+.2f V\n各段压电坐标最大相距 %.2f nm%s · 左图＝各段系定的 STM" % (
            dlast, C.local(t0, "%m-%d %H:%M"), C.local(t1, "%m-%d %H:%M"), core_short, SP.dshort(dname), lo_all, hi_all,
            spread, "（不是同一点，中间隔了几小时、有漂移，拼接只作参考）" if spread > 0.3 else "（同一点）"),
            color=C.INK_HEX, fontproperties=C.fp(13), y=0.975)
        data = C.savefig_bytes(fig, DPI)

    whole_dir = any(gd == d and set(gids) == set(keys) for gd, gids in single_groups(doc, marks))
    if options.get("group_dir") or whole_dir:
        base = C.safe_name("单根谱拼接_%s" % dlast)
    else:
        base = C.safe_name("单根谱拼接_%s_%s-%s" % (dlast, S[0]["fn"][-9:-4], S[-1]["fn"][-9:-4]))
    name = base + ".png"
    C.write_bytes(lay, CAT, name, data)
    summary = {"段数": len(S), "核心": core_short, "范围 V": "%+.2f…%+.2f" % (lo_all, hi_all),
               "dI/dV": SP.dshort(dname).replace("dI/dV", "").strip("（）"), "相距 nm": round(spread, 2)}
    for sg in segs:
        if sg["role"] == "piece":
            summary["×k " + sg["fn"][-9:-4]] = float("%.3g" % sg["factor"])
    if skipped:
        summary["跳过"] = len(skipped)
    detail = {"segments": segs, "core": [S[i]["id"] for i in core_ids], "li": li, "resp": resp,
              "x0": {"I": x0I, "D": x0D}, "span_nm": spread, "kappa": kappa, "table": table, "dir": d,
              "skipped": skipped}
    return C.write_figure_json(lay, CAT, base, kind="sts_stitch", title="单根谱拼接 · %s" % C.dir_title(d),
                               files=[name], ids=keys, options=options, summary=summary, detail=detail,
                               maker_version=VER)
