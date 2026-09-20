"""把「判据对这个团簇做了什么」画成一张图,落盘,给旁白当配图。

## 出处

「另外旁白会不会返回对团簇的各种分析图?」

在此之前(08-15)他要求过一份离线版本,原话是:
「我想看的是,一个团簇图,这个团簇的**识别图**,这个团簇被做了什么计算,结果如何?」
那一版做成了 HTML 报告(``artifacts/cluster_criterion_explained.html``)。
这里是它的**在线版**:每扎一针,旁白右边就挂着这一张。

## 为什么必须落盘,而不是让端点现渲染

``/api/chat/narration-image/{seq}`` 的自述写得很清楚:

    **原样读盘,不重渲染。** …… 重新渲染一张(比如从最新的 .sxm)就正好复刻了
    #76/#78 那个事故:扫描途中每一次判读都配着上一张图,而新整帧存下来之后,
    历史里所有缩略图会静默变成那张整图。

    顺带:这条路径上**一行 matplotlib 都没有**。``render_scan_thumbnail`` 用的是
    pyplot 的全局 figure manager(Gcf,非线程安全),而 ``/api/vision/recent``
    已经在 API 线程上调它 —— 再加一个调用方就是两个线程共用 Gcf。
    ``origin="sxm"`` 那一档还没有生产方,所以那里连门都不开。

⇒ 两条结论,都照做:
  1. 图在**判读发生的那一刻**画好落盘,``origin="milestone_png"``(那一档能取到);
  2. 这里**不用 pyplot** —— 用 ``Figure`` + ``FigureCanvasAgg`` 的面向对象接口,
     它完全不碰 Gcf,所以在 composite 的工作线程上画是安全的。
     (``scan_monitor._persist_frame_png`` 用的是 pyplot;那是既有的另一个线程,
      不去动它,但也不跟着它错。)

## 永不抛

一张配图坏掉绝不许弄坏正在跑的实验 —— 与 ``narrate()`` 同一条纪律。
任何一步出事都返回 None,旁白照发,只是没有图。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: 面板宽高(英寸)与 dpi —— 一张给聊天泳道看的图,不是给论文用的。
_FIGSIZE = (11.0, 2.5)
_DPI = 96


def _cjk_font():
    """一个能显示中文的字体,找不到返回 None(标题退回英文)。

    冻结的应用里 ``DejaVu Sans`` 没有 CJK 字形,直接写中文会画出一排方块 ——
    而方块比英文更坏:它让人以为数据坏了。
    """
    try:
        from matplotlib.font_manager import FontProperties, findfont

        for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"):
            fp = FontProperties(family=name)
            got = findfont(fp, fallback_to_default=False)
            if got:
                return fp
    except Exception:  # noqa: BLE001
        pass
    return None


def render_cluster_panel(scan_path: str, result: dict,
                         *, out_dir: "Path | None" = None) -> "str | None":
    """画一张四格分解图并落盘,返回路径(任何失败返回 None)。

    四格 = 判据真正走过的四步:

        ① 去斜后的簇图        判据看到的原始形貌
        ② 阈值分割            背景众数 + ½ Au 台阶(或 1.5σ),连通域
        ③ 选中的那一块        select="center" 挑的是哪个
        ④ 边界 → 质心的 r     等效轴比就是从这个分布来的

    ``result`` 是 ``AssessClusterRoundness`` 的返回 data —— 图上标的数**就是
    报文里那几个数**,不重新算一遍。重算会长出第二个真源,而两处一旦不一致,
    看图的人无从知道该信哪个。
    """
    try:
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from scipy.ndimage import binary_erosion, label

        from mast._runtime_paths import project_root
        from mast.data.processors import plane_subtract
        from mast.io.nanonis_files import read_sxm
        from mast.vision.roundness import background_level

        p = Path(str(scan_path))
        if not p.exists():
            return None
        ch = (read_sxm(str(p)).get("channels") or {}).get("Z")
        if not ch:
            return None
        img = np.asarray(ch.get("forward"), dtype=float)
        if img.ndim != 2 or img.size == 0:
            return None
        lv = plane_subtract(img)

        # 阈值:照报文说的那一档来,别自己另选一套。
        base = background_level(lv)
        thr_mode = str(result.get("threshold_mode") or "sigma")
        if thr_mode == "physical" and base is not None:
            delta = float(result.get("threshold_above_background_m") or 117.7e-12)
            mask = lv > base + delta
        else:
            mu, sd = float(np.nanmean(lv)), float(np.nanstd(lv))
            mask = lv > mu + 1.5 * sd
        lab, n = label(mask)

        # 选中的那一块:报文说 select=center 就找离画面中心最近的。
        blob = None
        if n:
            cy0, cx0 = (lv.shape[0] - 1) / 2.0, (lv.shape[1] - 1) / 2.0
            best, bd = 0, float("inf")
            for k in range(1, n + 1):
                ys, xs = np.where(lab == k)
                d2 = (ys.mean() - cy0) ** 2 + (xs.mean() - cx0) ** 2
                if d2 < bd:
                    best, bd = k, d2
            blob = lab == best

        fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
        FigureCanvasAgg(fig)
        # ⚠️ 中文字体只设在**这张图上**,不碰 ``matplotlib.rcParams`` 全局。
        #
        # 全局改 rcParams 会波及同进程里所有别的绘图(视觉监视器那条线也在画图),
        # 而且那是一个**跨线程共享的可变状态** —— 正是这个文件开头躲开 pyplot
        # 的同一条理由。
        #
        # 装不到中文字体时**退回英文标题**,而不是画一排方块:
        # 一张看不懂的图和没有图一样,但方块还会让人以为是数据坏了。
        _cjk = _cjk_font()
        def _t(zh: str, en: str) -> str:
            return zh if _cjk else en
        _fp = {"fontproperties": _cjk} if _cjk else {}
        fig.patch.set_facecolor("#fcfcfb")
        axes = fig.subplots(1, 4)

        lo, hi = np.nanpercentile(lv, [2, 98])
        axes[0].imshow(lv, cmap="afmhot", vmin=lo, vmax=hi, interpolation="nearest")
        axes[0].set_title(_t("① 去斜后的簇图", "(1) leveled"), fontsize=8, **_fp)

        axes[1].imshow(mask, cmap="gray", interpolation="nearest")
        axes[1].set_title(_t(f"② 阈值分割 · {n} 块", f"(2) threshold - {n} blobs"), fontsize=8, **_fp)

        if blob is not None:
            show = np.zeros(blob.shape + (3,))
            show[mask] = [0.45, 0.45, 0.45]
            show[blob] = [0.16, 0.47, 0.84]
            axes[2].imshow(show, interpolation="nearest")
            axes[2].set_title(_t(f"③ 选中的那块 · {int(blob.sum())} px", f"(3) selected - {int(blob.sum())} px"), fontsize=8, **_fp)

            edge = blob & ~binary_erosion(blob)
            ey, ex = np.where(edge)
            ys, xs = np.nonzero(blob)
            cy, cx = ys.mean(), xs.mean()
            if ey.size:
                r = np.hypot(ey - cy, ex - cx)
                th = np.degrees(np.arctan2(ey - cy, ex - cx))
                o = np.argsort(th)
                axes[3].plot(th[o], r[o], "-", color="#2a78d6", lw=1.4)
                axes[3].axhline(r.mean(), color="#52514e", ls="--", lw=1.0)
                axes[3].tick_params(labelsize=6, colors="#52514e")
                for sp in ("top", "right"):
                    axes[3].spines[sp].set_visible(False)
                ax_r = result.get("equivalent_axis_ratio")
                axes[3].set_title(
                    _t("④ 边界→质心的 r(θ)", "(4) r(theta) of the boundary")
                    + (_t(f" · 轴比 {ax_r:.3f}", f" - axis ratio {ax_r:.3f}")
                       if isinstance(ax_r, (int, float)) else ""),
                    fontsize=8, **_fp)
        else:
            axes[2].set_title(_t("③ 没有选中的块", "(3) nothing selected"), fontsize=8, **_fp)
            axes[3].set_title(_t("④ 算不出 r(θ)", "(4) r(theta) unavailable"), fontsize=8, **_fp)

        for a in axes[:3]:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        fig.tight_layout(pad=0.4)

        d = (out_dir or (project_root() / "artifacts" / "cluster_panels"))
        d.mkdir(parents=True, exist_ok=True)
        # 文件名带源帧名 + 内容指纹:同一张帧重判不会互相覆盖,
        # 而历史里每一条旁白仍然指向**它自己当时那一张**(#76/#78 的教训)。
        stem = re.sub(r"[^\w.-]", "_", p.stem)[:60] or "cluster"
        # ⚠️ 用 hashlib 而**不是内置 ``hash()``**。
        #
        # 内置 hash 对 str 是**每个进程随机加盐的**(PYTHONHASHSEED),所以
        # ``hash((stem, ...))`` 根本不是内容指纹,是一个每次重启就变的随机数:
        #   · 上面那句「同一张帧重判不会互相覆盖」照样成立(碰巧);
        #   · 但「**内容**指纹」这半句是假的 —— 同一张图同一份读数,重启一次就
        #     多落一个文件,artifacts 会无声地攒重复。
        # 一句在说假话的注释比没有注释更贵,所以这里改成真的按内容取。
        h = hashlib.blake2b(digest_size=4)
        h.update(stem.encode("utf-8", "replace"))
        h.update(repr((float(np.nansum(lv)), int(mask.sum()), thr_mode)).encode())
        out = d / f"{stem}_{h.hexdigest()}.png"
        fig.savefig(out, format="png", facecolor="#fcfcfb")
        return str(out)
    except Exception as exc:  # noqa: BLE001 — 一张配图绝不许弄坏实验
        logger.debug("cluster panel render failed: %s", exc, exc_info=True)
        return None


__all__ = ["render_cluster_panel"]
