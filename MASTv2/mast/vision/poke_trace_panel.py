# -*- coding: utf-8 -*-
"""把一次扎针的 Z 与电流曲线分段绘制，供旁白显示与事后核对。

四段分别为：反馈开启的基线、反馈关闭后压入、Z 回到命令基线、反馈恢复后的
新平衡。第三段的命令位置本身不能证明针尖或表面是否发生变化；只有恢复反馈
后的稳定高度能回答这一问题。

段界采用电流回到 setpoint 的数据标志物，不把按参数推算的时间直接当作
硬件已恢复反馈的证据。前置放大器从饱和恢复可能滞后，因此该标志物用于
保守选取稳态采样区间，不应解读成精确的反馈接管时刻。

绘图失败返回 None，不中断正在执行的技能或旁白。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: 面板宽高（英寸）与 dpi —— 一张给聊天泳道看的图。
_FIGSIZE = (11.0, 3.2)
_DPI = 96

#: 四段的底色。③ 用灰是有意的：它是**没有信息**的那一段。
_SEG_FACE = {
    1: "#eef4fb",   # 基线
    2: "#fdeeee",   # 压入
    3: "#f0f0f0",   # 抬回原位（反馈还没开）
    4: "#eaf7ee",   # 反馈恢复 ← 判定读这一段
}
_SEG_NAME = {
    1: "① 基线\n反馈 ON",
    2: "② 压入\n反馈 OFF",
    # ③ 的右界是**电流回到 setpoint**，而那是保守下界（前置放大器出饱和要时间）。
    # 所以这一段的实际末尾可能已经属于④ —— 标题里写明「≤」。
    3: "③ 抬回原位\n反馈仍 OFF（≤）",
    4: "④ 反馈恢复\n判定读这里",
}


def _cjk_font():
    """能显示中文的字体。**不新写一份** —— 仓里已经有四份拷贝了。

    ``mast.io.exp_map._cjk_fontprops`` 是 ``agents/data_processing/tools.py``
    已经在用的那一份，跟着它走；找不到就返回 None，标题退回英文。
    冻结的应用里 ``DejaVu Sans`` 没有 CJK 字形，直接写中文会画出一排方块，
    而方块比英文更坏：它让人以为数据坏了。
    """
    try:
        from mast.io.exp_map import _cjk_fontprops

        return _cjk_fontprops()
    except Exception:  # noqa: BLE001
        return None


def _segments(z_t, current_s, current_t, event_t, t4, deep_idx):
    """四段的时间边界。取自数据，不取自声明。

    返回 ``[(seg_no, t_start, t_end), ...]``，取不到的段直接不出现 ——
    **不猜一个边界画上去**：一条画在错地方的段界比没有段界更贵。
    """
    out = []
    cap = float(z_t[-1])
    t_ev = float(event_t)
    out.append((1, 0.0, t_ev))
    if deep_idx:
        t_in, t_out = float(z_t[deep_idx[0]]), float(z_t[deep_idx[-1]])
        out.append((2, t_ev, t_out))
        if t4 is not None:
            out.append((3, t_out, float(t4)))
            out.append((4, float(t4), cap))
        else:
            # ③④ 分不开 —— 合成一段，并在图上明说分不开
            out.append((3, t_out, cap))
    else:
        out.append((2, t_ev, cap))
    return out


def render_poke_trace_panel(z_s, z_t, current_s, current_t, *,
                            event_t: float, indent: dict,
                            meta: "dict | None" = None,
                            label: str = "",
                            out_dir: "Path | None" = None) -> "str | None":
    """画一次扎针的 Z/电流曲线并落盘，返回路径（任何失败返回 None）。

    ``indent`` 是 ``step_verdict`` 的返回 —— 图上标的数**就是报文里那几个数**，
    不重新算一遍。重算会长出第二个真源，而两处一旦不一致，看图的人无从知道
    该信哪个（``cluster_panel`` 同一条纪律）。
    """
    try:
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from mast._runtime_paths import project_root

        z = np.asarray(list(z_s), dtype=float)
        t = np.asarray(list(z_t), dtype=float)
        if z.size < 8 or t.size != z.size:
            return None
        ci = np.asarray(list(current_s or []), dtype=float)
        ct = np.asarray(list(current_t or []), dtype=float)

        ev = float(event_t)
        pre = z[t < ev]
        if pre.size < 2:
            return None
        z1 = float(np.median(pre[pre.size // 2:]))
        dz = (z - z1) * 1e12                       # pm
        cap = float(t[-1])

        t4 = indent.get("feedback_restored_t")
        src = str(indent.get("feedback_segment_source") or "")
        verdict = str(indent.get("verdict") or indent.get("direction") or "")
        delta_pm = float(indent.get("delta_m") or 0.0) * 1e12

        depth_pm = abs(float((meta or {}).get("tip_lift_m") or 0.0)) * 1e12
        thr = max(60.0, depth_pm * 0.5)
        deep_idx = list(np.where(dz < -thr)[0]) if depth_pm else []

        fp = _cjk_font()
        kw = {"fontproperties": fp} if fp is not None else {}

        fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        for seg, a, b in _segments(t, ci, ct, ev, t4, deep_idx):
            if b <= a:
                continue
            ax.axvspan(a, b, facecolor=_SEG_FACE.get(seg, "#ffffff"),
                       edgecolor="none", zorder=0)
            name = _SEG_NAME.get(seg, str(seg))
            if seg == 3 and t4 is None:
                name = "③+④ 分不开\n(电流没回到 setpoint)"
            ax.annotate(name.replace("**", ""), xy=((a + b) / 2.0, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5, color="#555",
                        **kw)

        ax.plot(t, dz, lw=1.1, color="#1f4e79", zorder=3, label="Z")
        ax.axhline(0.0, lw=0.8, color="#888", ls=":", zorder=1)

        # 判定实际取的两个窗口 —— **画出来**，这是这张图最该回答的问题
        ax.axvspan(0.0, ev, facecolor="none", edgecolor="#1f4e79",
                   ls="--", lw=0.8, zorder=2)
        if t4 is not None:
            ax.axvspan(float(t4), cap, facecolor="none", edgecolor="#2e7d32",
                       ls="--", lw=0.9, zorder=2)
            ax.annotate("电流回到 setpoint\n（反馈已接管的**下界**）"
                        if fp is not None else "I back to setpoint (lower bound)",
                        xy=(float(t4), 0.06), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        ha="left", va="bottom", fontsize=6.8, color="#2e7d32",
                        **kw)
        if verdict and verdict != "insufficient_data":
            ax.annotate("", xy=(cap * 0.985, delta_pm), xytext=(cap * 0.985, 0.0),
                        arrowprops={"arrowstyle": "<->", "color": "#2e7d32",
                                    "lw": 1.2}, zorder=4)
            ax.annotate("Δz = %+.0f pm" % delta_pm,
                        xy=(cap * 0.98, delta_pm / 2.0), ha="right", va="center",
                        fontsize=9, color="#2e7d32", zorder=4, **kw)

        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("Z − 基线 (pm)" if fp is not None else "Z - baseline (pm)",
                      fontsize=8, **kw)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0.0, cap)

        # 电流放在右轴，对数 —— 它是段界的**唯一**可信标志物
        if ci.size >= 8 and ct.size == ci.size:
            ax2 = ax.twinx()
            ax2.plot(ct, np.abs(ci) * 1e12, lw=0.9, color="#c62828",
                     alpha=0.75, zorder=2)
            ax2.set_yscale("log")
            ax2.set_ylabel("|I| (pA)", fontsize=8, color="#c62828")
            ax2.tick_params(labelsize=7, colors="#c62828")

        why = {
            "current": "段界由电流定位（保守下界）",
            "no_press": "电流没升上去 —— 可能根本没压到表面",
            "no_return": "电流升上去后**再没回到 setpoint** —— 采集在反馈接管前就结束了",
            "too_short": "第④段太短，判不了",
            "no_current": "没有电流通道 —— 段界只能靠猜，未标",
            "too_few": "电流样本太少",
        }.get(src, src)
        head = "扎针 Z 曲线"
        if depth_pm:
            head += "（下压 %.0f pm）" % depth_pm
        head += "：%s" % (verdict or "—")
        if why:
            head += "  ·  %s" % why.replace("**", "")
        ax.set_title(head, fontsize=9.5, loc="left", **kw)

        fig.tight_layout(pad=0.5)

        d = (out_dir or (project_root() / "artifacts" / "poke_traces"))
        d.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.-]", "_", str(label or "poke"))[:60] or "poke"
        # 内容指纹用 hashlib，**不用内置 hash()** —— 后者对 str 每进程随机加盐，
        # 「同一条曲线重画不多落文件」这句话会变成假话（cluster_panel 的教训）。
        h = hashlib.blake2b(digest_size=4)
        h.update(stem.encode("utf-8", "replace"))
        h.update(repr((round(float(np.nansum(dz)), 3), round(delta_pm, 3),
                       verdict, src)).encode())
        out = d / f"{stem}_{h.hexdigest()}.png"
        fig.savefig(out, format="png", facecolor="#fcfcfb")
        return str(out)
    except Exception as exc:  # noqa: BLE001 — 一张配图绝不许弄坏实验
        logger.debug("poke trace panel render failed: %s", exc, exc_info=True)
        return None


__all__ = ["render_poke_trace_panel"]
