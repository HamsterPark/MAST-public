"""Quick plotting utilities for STM data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .processors import fft2d

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_plt = None


def _get_plt():
    """Lazy-load matplotlib on first use (~1-2s startup cost avoided)."""
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def plot_topography(
    image: np.ndarray,
    scan_size_m: tuple[float, float] | None = None,
    title: str = "",
    cmap: str = "copper",
    save_path: str | None = None,
) -> "Figure":
    """Plot 2D topography with scale bar and colorbar.

    If scan_size_m provided, axes labeled in nm. Otherwise in pixels.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 5))

    if scan_size_m is not None:
        sx_nm = scan_size_m[0] * 1e9
        sy_nm = scan_size_m[1] * 1e9
        extent = [0, sx_nm, 0, sy_nm]
        im = ax.imshow(image, origin="lower", cmap=cmap, extent=extent)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
    else:
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="z (m)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sts(
    voltage: np.ndarray,
    didv: np.ndarray,
    title: str = "dI/dV",
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> "Figure":
    """Plot STS dI/dV vs voltage curve(s). Can overlay multiple curves."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4))

    if didv.ndim == 1:
        didv = didv[np.newaxis, :]

    for i, curve in enumerate(didv):
        label = labels[i] if labels and i < len(labels) else None
        ax.plot(voltage, curve, label=label)

    ax.set_xlabel("Bias (V)")
    ax.set_ylabel("dI/dV (a.u.)")
    ax.set_title(title)
    if labels:
        ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_fft(
    image: np.ndarray,
    title: str = "FFT",
    save_path: str | None = None,
) -> "Figure":
    """Plot 2D FFT magnitude (log scale)."""
    fft_mag = fft2d(image)

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(fft_mag, cmap="inferno", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_topo_overlay(
    topo: np.ndarray,
    overlay: np.ndarray,
    scan_size_m: tuple[float, float] | None = None,
    title: str = "",
    alpha: float = 0.5,
    save_path: str | None = None,
) -> "Figure":
    """Overlay dI/dV map on topography."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 5))

    if scan_size_m is not None:
        sx_nm = scan_size_m[0] * 1e9
        sy_nm = scan_size_m[1] * 1e9
        extent = [0, sx_nm, 0, sy_nm]
    else:
        extent = None

    ax.imshow(topo, origin="lower", cmap="copper", extent=extent)
    im = ax.imshow(overlay, origin="lower", cmap="viridis", alpha=alpha, extent=extent)

    if scan_size_m is not None:
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
    else:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="overlay")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


#: 通道单位 → (显示倍率, 显示单位)。
_UNIT_DISPLAY: dict[str, tuple[float, str]] = {
    "m": (1e12, "pm"), "A": (1e12, "pA"), "V": (1e3, "mV"), "Hz": (1.0, "Hz"),
}


def _nice_bar_nm(width_nm: float) -> float:
    """1-2-5 序列里不小于视野 1/5 的那一档,用作比例尺长度。"""
    target = width_nm / 5.0
    for v in (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500):
        if v >= target:
            return float(v)
    return 500.0


def plot_flattened_scan(
    image: np.ndarray,
    *,
    width_nm: float | None = None,
    height_nm: float | None = None,
    clip_percentile: tuple[float, float] = (1.0, 99.0),
    title: str = "",
    subtitle: str = "",
    cmap: str = "afmhot",
    unit: str = "m",
    save_path: str | None = None,
    bare: bool = False,
) -> "Figure | None":
    """把一帧**已经平场过**的图渲染成 PNG,按给定百分位裁色阶。

    与 :func:`plot_topography` 的三点区别,都是这条通路需要的:

    * **百分位色阶**。色阶宽窄是 :mod:`mast.vision.scan_prep` 的一项决策产物
      (有精细周期结构时收紧,有台阶时放宽);``plot_topography`` 没有这个入口,
      而默认的 min-max 会让一个热点把整幅图压成一片黑。
    * ``origin="upper"``。喂进来的帧已经过
      :func:`mast.io.nanonis_files.sxm_oriented_frames` 归位(row 0 = 画面上边),
      画的时候必须跟着,否则报告里的「第 241 行」指到反方向去了。
    * **NaN 画成深灰**而不是透明 —— 没扫完的部分要看得见是没扫完。

    图上文字一律 ASCII:CJK 需要 per-artist 的 ``FontProperties``,而按本仓纪律
    绝不能去动全局 ``rcParams``(会污染同进程里别人的图)。中文说明在报告里,不在图上。

    ``bare=True`` 输出没有任何装饰的纯像素图(喂给视觉模型/拼图用)。
    图上没有可画的有限像素时返回 ``None``。
    """
    plt = _get_plt()
    import matplotlib

    arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    fac, ulab = _UNIT_DISPLAY.get(unit, (1.0, unit or ""))
    lo, hi = np.percentile(arr[finite], list(clip_percentile))
    if not (hi > lo):
        lo, hi = float(arr[finite].min()), float(arr[finite].max()) + 1e-15
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("#202020")
    data = np.ma.masked_invalid(arr) * fac
    lo, hi = lo * fac, hi * fac

    if bare:
        fig = plt.figure(figsize=(arr.shape[1] / 100, arr.shape[0] / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(data, cmap=cm, vmin=lo, vmax=hi, origin="upper",
                  interpolation="nearest")
        ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=100, pad_inches=0)
        return fig

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    extent = ([0, width_nm, 0, height_nm or width_nm]
              if width_nm else None)
    im = ax.imshow(data, cmap=cm, vmin=lo, vmax=hi, origin="upper",
                   interpolation="nearest", extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if width_nm:
        h_nm = float(height_nm or width_nm)
        bar = _nice_bar_nm(float(width_nm))
        x0, y0 = 0.06 * float(width_nm), 0.06 * h_nm
        ax.add_patch(plt.Rectangle((x0, y0), bar, 0.016 * h_nm, fc="white",
                                   ec="black", lw=0.6, zorder=5))
        import matplotlib.patheffects as pe
        ax.text(x0 + bar / 2, y0 + 0.032 * h_nm, f"{bar:g} nm", color="white",
                ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(ulab or "", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if subtitle:
        ax.text(0.5, -0.035, subtitle, transform=ax.transAxes, ha="center",
                va="top", fontsize=7.5, color="0.35")
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    return fig
