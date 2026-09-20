# -*- coding: utf-8 -*-
"""将原子分辨评估渲染为三联图与判据读数。

形貌面板给出去平面图、pm 色标和比例尺；二维功率谱标注候选布拉格峰
及周期；快轴剖面展示高度起伏。读数同时呈现正反扫、帧内两半与成像条件，
使读者能核对结论依据，不能仅凭单个分数宣称通过。

由旁白或 post-skill 钩子在执行后调用，不在受限硬件动作中导入绘图库或
写盘。使用 Agg 后端。失败返回空串并记录日志，由调用方处理缺失产物。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 功率谱面板显示到几倍晶格波矢。3 倍够看到一阶峰和它们的对称，再大就只有噪声。
_K_VIEW_FACTOR = 3.0

#: 快轴剖面取中间多少行做平均。单行太吵，太多行会把倾斜/漂移平均进来。
_PROFILE_ROWS = 24


def _plt():
    """与 ``tip_shape_plot._plt`` 同一套家法：Agg + 中文字体，失败静默降级。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from matplotlib import font_manager
        avail = {fp.name for fp in font_manager.fontManager.ttflist}
        for f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                  "Source Han Sans SC", "Microsoft JhengHei"):
            if f in avail:
                plt.rcParams["font.sans-serif"] = [f] + list(
                    plt.rcParams.get("font.sans-serif", []))
                break
    except Exception:  # noqa: BLE001
        pass
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _fmt(v: Any, spec: str, dash: str = "—") -> str:
    """数字不存在时给一个破折号，**不给 0** —— 0 是一个读数，缺席不是。"""
    try:
        if v is None:
            return dash
        f = float(v)
        if f != f:  # NaN
            return dash
        return format(f, spec)
    except (TypeError, ValueError):
        return dash


def render_atomic_report(scan_path: str | Path, save_path: str | Path, *,
                         channel: str = "Z",
                         stats: "dict | None" = None) -> str:
    """渲染三联图，回保存路径；失败回空串（**不抛**）。

    ``stats`` 给了就**就地填**上这一帧量到的数（正/反扫角向集中度、周期、
    帧内两半）。这不是多量一次 —— 画数字块本来就要量，只是从前没往外交。
    反扫的角向集中度**全仓只有这里有生产方**：``AssessAtomicPhase`` 只取正扫，
    而「只在一个方向上出现的周期是针尖产物」是验收里少不了的一条。
    """
    try:
        return _render(Path(scan_path), Path(save_path), channel,
                       stats if stats is not None else {})
    except Exception as exc:  # noqa: BLE001 — 调用方是钩子，不能被画图搞崩
        logger.warning("原子分辨报告图渲染失败 %s: %s", scan_path, exc)
        return ""


def _render(scan_path: Path, save_path: Path, channel: str, stats: dict) -> str:
    import numpy as np

    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.atomic_phase import assess_atomic_phase
    from mast.vision.lattice_calibration import find_lattice_peaks
    from mast.vision.seg_scale_adaptive import flatten_robust

    # 用 sxm_oriented_frames 而不是裸 read_sxm:反扫块是沿 −x 采的、``SCAN_DIR: up``
    # 的第一行是画面**底部** —— 不做这两步归一,正反扫比的是一张图和它自己的镜像。
    fr = sxm_oriented_frames(read_sxm(str(scan_path)), channel=channel)
    fwd = fr.get("forward")
    bwd = fr.get("backward")
    nmpp = fr.get("nm_per_px")
    if fwd is None or not nmpp:
        logger.warning("原子分辨报告图:%s 里没有可用的 %s 通道或读不到像素尺度",
                       scan_path, channel)
        return ""
    fwd = np.asarray(fwd, dtype=float)
    nmpp = float(nmpp)
    width_nm = float(fr.get("width_nm") or (fwd.shape[1] * nmpp))
    height_nm = float(fr.get("height_nm") or (fwd.shape[0] * nmpp))

    res_f = assess_atomic_phase(fwd, nm_per_px=nmpp)
    res_b = (assess_atomic_phase(np.asarray(bwd, dtype=float), nm_per_px=nmpp)
             if bwd is not None else None)
    lat = find_lattice_peaks(fwd, nmpp)

    flat = np.asarray(flatten_robust(fwd), dtype=float)
    flat_pm = flat * 1e12 if np.nanmax(np.abs(flat)) < 1e-6 else flat

    # 量到的数交出去（调用方给了 dict 才填）。反扫那一项这里是**唯一**的生产方。
    stats.update({
        "angular_concentration": res_f.angular_concentration,
        "angular_concentration_backward": (
            res_b.angular_concentration if res_b is not None else None),
        "period_nm": (lat.period_mean_nm if lat.ok and lat.period_mean_nm
                      else res_f.period_nm),
        "half_concentrations": (list(res_f.half_concentrations)
                                if res_f.half_concentrations else None),
        "passed": bool(res_f.passed),
    })

    plt = _plt()
    fig = plt.figure(figsize=(13.0, 5.6), dpi=140)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.05), wspace=0.28,
                          left=0.05, right=0.985, top=0.90, bottom=0.24)

    # ── 面板 1：形貌 ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.nanpercentile(flat_pm, [1.0, 99.0])
    im = ax.imshow(flat_pm, cmap="afmhot", origin="upper",
                   extent=(0.0, width_nm, 0.0, height_nm),
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title("形貌（已扣平面）", fontsize=11)
    ax.set_xlabel("nm", fontsize=9)
    ax.set_ylabel("nm", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("pm", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # 比例尺：取视野的 1/4 并取整到 0.5 nm，画在左下角
    bar_nm = max(0.5, round(width_nm / 4.0 * 2.0) / 2.0)
    # 比例尺画在半透明深色底板上：``afmhot`` 的亮端接近纯白，白字压上去看不见
    # ；同时添加底板与描边以支持两种色标。
    import matplotlib.patheffects as _pe

    ax.add_patch(plt.Rectangle(
        (width_nm * 0.035, height_nm * 0.035), bar_nm + width_nm * 0.05,
        height_nm * 0.115, facecolor="black", alpha=0.45, edgecolor="none"))
    ax.plot([width_nm * 0.06, width_nm * 0.06 + bar_nm],
            [height_nm * 0.07] * 2, lw=3.0, color="w", solid_capstyle="butt")
    ax.text(width_nm * 0.06 + bar_nm / 2.0, height_nm * 0.088,
            "%g nm" % bar_nm, color="w", ha="center", va="bottom", fontsize=8,
            path_effects=[_pe.withStroke(linewidth=2.0, foreground="black")])

    # ── 面板 2：二维功率谱 + 布拉格峰 ──────────────────────────────────
    #
    # 这个面板是**判据的可视化**:角向集中度量的就是「能量集中在几个离散点上,
    # 还是摊成一个环」。真晶格 = 离散点;针尖抖动 = 弥散环。让人直接看见。
    ax2 = fig.add_subplot(gs[0, 1])
    win = np.outer(np.hanning(flat.shape[0]), np.hanning(flat.shape[1]))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(flat * win)))
    spec = np.log1p(spec / (np.nanmax(spec) or 1.0) * 1e3)
    cy, cx = spec.shape[0] // 2, spec.shape[1] // 2

    period_nm = (lat.period_mean_nm if lat.ok and lat.period_mean_nm
                 else res_f.period_nm)
    if period_nm and period_nm > 0:
        k_px = flat.shape[1] * nmpp / float(period_nm)   # 该周期对应的 |k|（像素）
        half = max(8.0, k_px * _K_VIEW_FACTOR)
    else:
        half = min(cy, cx) * 0.6
    y0, y1 = int(max(0, cy - half)), int(min(spec.shape[0], cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(spec.shape[1], cx + half))
    ax2.imshow(spec[y0:y1, x0:x1], cmap="magma", origin="lower",
               extent=(x0 - cx, x1 - cx, y0 - cy, y1 - cy),
               interpolation="nearest")
    ax2.set_title("二维功率谱（对数）", fontsize=11)
    ax2.set_xlabel("k$_x$（像素$^{-1}$）", fontsize=9)
    ax2.set_ylabel("k$_y$（像素$^{-1}$）", fontsize=9)
    ax2.tick_params(labelsize=8)
    if lat.ok and lat.peaks:
        for pk in lat.peaks:
            ax2.add_patch(plt.Circle((pk.kx, pk.ky), max(3.0, half * 0.055),
                                     fill=False, lw=1.4, color="#38bdf8"))
        # 只给最强的三个标注周期，六个全标会糊成一片
        for pk in sorted(lat.peaks, key=lambda p: -p.power)[:3]:
            ax2.annotate("%.3f nm" % pk.period_nm, (pk.kx, pk.ky),
                         textcoords="offset points", xytext=(6, 6),
                         color="#38bdf8", fontsize=7.5)
        ax2.text(0.02, 0.97, "一阶峰 %d 个%s" % (
            lat.n_peaks, "（六重）" if lat.hexagonal else ""),
            transform=ax2.transAxes, va="top", fontsize=8, color="#e2e8f0")
    else:
        ax2.text(0.5, 0.5, "没找到一阶峰\n（%s）" % (lat.reason or "—"),
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=9, color="#e2e8f0")

    # ── 面板 3：快轴剖面 ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    mid = flat_pm.shape[0] // 2
    lo = max(0, mid - _PROFILE_ROWS // 2)
    prof = np.nanmean(flat_pm[lo:lo + _PROFILE_ROWS], axis=0)
    xs = np.arange(prof.size) * nmpp
    ax3.plot(xs, prof, lw=1.0, color="#0f172a")
    ax3.set_title("快轴剖面（中间 %d 行平均）" % _PROFILE_ROWS, fontsize=11)
    ax3.set_xlabel("nm", fontsize=9)
    ax3.set_ylabel("起伏 pm", fontsize=9)
    ax3.tick_params(labelsize=8)
    ax3.grid(alpha=0.25, lw=0.5)
    if period_nm and period_nm > 0:
        # 在剖面上画出量到的周期,让人自己数一数对不对得上
        n_mark = int(min(12, xs[-1] / float(period_nm))) if xs.size else 0
        for i in range(1, n_mark + 1):
            ax3.axvline(i * float(period_nm), color="#38bdf8", lw=0.6, alpha=0.55)
        ax3.text(0.98, 0.03, "竖线间隔 = 量到的周期 %.3f nm" % period_nm,
                 transform=ax3.transAxes, ha="right", va="bottom", fontsize=7.5,
                 color="#0369a1")

    # ── 数字块 ──────────────────────────────────────────────────────────
    halves = res_f.half_concentrations
    lines = [
        "角向集中度  正扫 %s   反扫 %s        （真晶格 97–7645；针尖抖动 1.8–3.3）"
        % (_fmt(res_f.angular_concentration, ".1f"),
           _fmt(getattr(res_b, "angular_concentration", None), ".1f")),
        # ⚠️ 百分号要自己拼，别写进格式串：``format(f, ".0f%%")`` 不是合法格式符，
        # 会抛 ValueError 被 _fmt 吞成破折号 —— 一个「读不到」冒充「没有」的小例子。
        "帧内两半    %s / %s        周期 %s nm（三向散布 %s）"
        % (_fmt(halves[0] if halves else None, ".1f"),
           _fmt(halves[1] if halves else None, ".1f"),
           _fmt(period_nm, ".4f"),
           (_fmt(lat.period_spread * 100.0, ".0f") + "%")
           if lat.period_spread is not None else "—"),
        "成像条件    视野 %.2f × %.2f nm   %d px   %.5f nm/px   偏压 %s V   setpoint %s pA"
        % (width_nm, height_nm, flat.shape[1], nmpp,
           _fmt(fr.get("bias_v"), ".4f"),
           _fmt((fr.get("setpoint_a") or 0) * 1e12, ".0f")),
    ]
    if not res_f.passed:
        lines.append("判据未通过：%s" % ", ".join(res_f.reasons or ("—",)))
    # ⚠️ 这里**不能**指定 ``family="monospace"``：它解析到 DejaVu Sans Mono，
    # 而那个字体没有中文字形，可能使中文标签显示为缺字方框。
    # 用 rcParams 里选好的那个中文字体 —— 中文本来就是等宽的，数字对齐够用。
    fig.text(0.05, 0.145, "\n".join(lines), fontsize=8.6,
             va="top", linespacing=1.7)

    title = "原子分辨 · %s" % Path(scan_path).name
    if fr.get("rec_time"):
        title += "    %s" % fr.get("rec_time")
    fig.suptitle(title, fontsize=12.5, y=0.975)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), facecolor="white")
    plt.close(fig)
    return str(save_path)
