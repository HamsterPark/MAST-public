"""把「找台面 / 多针尖判读」这一步做了什么画成一张图,落盘,给旁白当配图。

## 出处

「找台面的分析图和计算结果也应该输出到旁白」「就像扎针团簇
的分析一样」。

四格 = 这一步真正走过的四步:

    ① 去斜后的大图            判据看到的形貌
    ② 高度直方图 + 台面能级    ← 多针尖判据的全部依据就在这张图里
    ③ 相邻台面差 ÷ 单台阶      单针尖贴着整数,劈裂冒出半整数
    ④ 选中的台面窗            扎针要落在这块里

## 与 cluster_panel 同一套纪律

* **不用 pyplot**(Gcf 非线程安全,而这里在 composite 的工作线程上画);
* 在**判读发生的那一刻**画好落盘,走 ``origin="milestone_png"``;
* 图上标的数**就是报文里那几个数**,不重算;
* 中文字体只设在这张图上,装不到就退回英文标题;
* **永不抛** —— 一张配图坏掉绝不许弄坏正在跑的实验。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_FIGSIZE = (12.0, 2.7)
_DPI = 96


def _cjk_font():
    """能显示中文的字体,找不到返回 None(标题退回英文)。"""
    try:
        from matplotlib.font_manager import FontProperties, findfont

        for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"):
            fp = FontProperties(family=name)
            if findfont(fp, fallback_to_default=False):
                return fp
    except Exception:  # noqa: BLE001
        pass
    return None


def render_terrace_panel(scan_path: str, split: dict,
                         flat: "dict | None" = None,
                         *, out_dir: "Path | None" = None) -> "str | None":
    """画四格并落盘,返回路径(任何失败返回 None)。

    ``split`` 是 ``double_tip.step_splitting`` 的返回,``flat`` 是
    ``FindFlatRegion`` 的返回(可以没有 —— 没找到台面时正是最该看这张图的时候)。
    """
    try:
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from scipy.ndimage import gaussian_filter1d

        from mast._runtime_paths import project_root
        from mast.data.processors import plane_subtract
        from mast.io.nanonis_files import read_sxm

        p = Path(str(scan_path))
        if not p.exists():
            return None
        ch = (read_sxm(str(p)).get("channels") or {}).get("Z")
        if not ch:
            return None
        img = np.asarray(ch.get("forward"), dtype=float)
        if img.ndim != 2 or img.size == 0:
            return None
        lv = plane_subtract(img) * 1e12          # pm

        step_pm = float(split.get("step_height_pm") or 235.5)
        levels = [float(x) for x in (split.get("levels_pm") or [])]
        gaps = [float(x) for x in (split.get("gaps_au") or [])]
        devs = [float(x) for x in (split.get("deviations") or [])]
        verdict = str(split.get("verdict") or "undecidable")

        fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
        FigureCanvasAgg(fig)
        cjk = _cjk_font()
        fp = {"fontproperties": cjk} if cjk else {}

        def t(zh: str, en: str) -> str:
            return zh if cjk else en

        fig.patch.set_facecolor("#fcfcfb")
        axes = fig.subplots(1, 4)

        # ① 形貌
        lo, hi = np.nanpercentile(lv, [2, 98])
        axes[0].imshow(lv, cmap="afmhot", vmin=lo, vmax=hi, interpolation="nearest")
        axes[0].set_title(t("① 去斜后的大图", "(1) leveled"), fontsize=8, **fp)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # ② 高度直方图 + 能级
        v = lv[np.isfinite(lv)]
        if v.size:
            a, b = np.percentile(v, [0.3, 99.7])
            nb = max(16, int((b - a) / (step_pm / 16.0)))
            h, e = np.histogram(v, bins=nb, range=(a, b))
            c = 0.5 * (e[1:] + e[:-1])
            hs = gaussian_filter1d(h.astype(float), sigma=(step_pm / 6.0) / ((b - a) / nb))
            axes[1].plot(c, hs, "-", color="#2a78d6", lw=1.3)
            for L in levels:
                axes[1].axvline(L, color="#d64545", ls="--", lw=0.9)
            # 只画分析用的那一段 —— 一条离群尾巴会把台面那几个峰压成一条线。
            axes[1].set_xlim(a, b)
            axes[1].set_xlabel("pm", fontsize=6)
        axes[1].set_title(
            t(f"② 高度直方图 · {len(levels)} 个台面",
              f"(2) height histogram - {len(levels)} terraces"), fontsize=8, **fp)
        axes[1].tick_params(labelsize=6)

        # ③ 台阶差 ÷ 单台阶 —— 判据本身
        if gaps:
            xs = np.arange(len(gaps))
            axes[2].bar(xs, gaps, color=["#d64545" if d > 0.25 else "#4a9d5f"
                                         for d in (devs or [0] * len(gaps))])
            for k in range(1, int(max(gaps)) + 2):
                axes[2].axhline(k, color="#52514e", ls=":", lw=0.8)
            axes[2].set_xticks(xs)
            axes[2].tick_params(labelsize=6)
        # 标题是**用户看到的字**,跟旁白同一套术语(2026-08-18 起改用「多针尖」这个说法,更专业)。原来写的是「劈裂」。
        zh3 = {"split": "③ 台阶差/单台阶 · **多针尖**",
               "single": "③ 台阶差/单台阶 · 单针尖"}.get(verdict, "③ 台阶差/单台阶 · 判不了")
        en3 = {"split": "(3) gap/step - MULTI-TIP",
               "single": "(3) gap/step - single"}.get(verdict, "(3) gap/step - n/a")
        axes[2].set_title(t(zh3, en3), fontsize=8, **fp)

        # ④ 选中的台面窗
        axes[3].imshow(lv, cmap="afmhot", vmin=lo, vmax=hi, interpolation="nearest")
        axes[3].set_xticks([]); axes[3].set_yticks([])
        win_txt = t("④ 没选到台面", "(4) no terrace")
        if flat:
            side = flat.get("window_side_m")
            rms = flat.get("rms_m")
            if side:
                win_txt = t(f"④ 选中的窗 · {float(side) * 1e9:.0f} nm",
                            f"(4) window - {float(side) * 1e9:.0f} nm")
                if rms:
                    win_txt += t(f" · {float(rms) * 1e12:.0f} pm",
                                 f" - {float(rms) * 1e12:.0f} pm")
        axes[3].set_title(win_txt, fontsize=8, **fp)

        for a in (axes[0], axes[3]):
            for sp in a.spines.values():
                sp.set_visible(False)
        fig.tight_layout(pad=0.4)

        d = out_dir or (project_root() / "artifacts" / "terrace_panels")
        d.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.-]", "_", p.stem)[:60] or "terrace"
        hh = hashlib.blake2b(digest_size=4)
        hh.update(stem.encode("utf-8", "replace"))
        hh.update(repr((float(np.nansum(lv)), len(levels), verdict)).encode())
        out = d / f"{stem}_{hh.hexdigest()}.png"
        fig.savefig(out, format="png", facecolor="#fcfcfb")
        return str(out)
    except Exception as exc:  # noqa: BLE001 — 一张配图绝不许弄坏实验
        logger.debug("terrace panel render failed: %s", exc, exc_info=True)
        return None


__all__ = ["render_terrace_panel"]
