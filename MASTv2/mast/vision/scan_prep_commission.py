"""扫描图预处理阈值的分布检查：读取一批输入文件并给出建议，不写配置。

公开版内建 profile 未标定，数值仅供合成示例。工具区分可从数据分布估计的旋钮
和模型选择参数：前者只有出现两个非空且有足够间隙的群组时才建议间隙中点，
后者只报告当前阈值两侧的样本数。没有分离证据时不给新的数值。

正反扫不稳定度可以额外报告当前输入批次的 p85，供使用者检查；它不与任何预存
实验语料比较。所有建议都须由使用者验证，不能当成自动标定或仪器安全保证。

    python -m mast.vision.scan_prep_commission <folder>
    python -m mast.vision.scan_prep_commission <folder> --profile custom-example
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any

#: 参与标定报告的指标:``(FrameMetrics 字段, 它驱动的 knob | None, 中文名, 方向)``。
#: 方向 ``"above"`` = 指标**大于**阈值时判据触发;``"below"`` = 小于时触发。
_METRICS: tuple[tuple[str, str | None, str, str], ...] = (
    ("line_gain", "line_gain", "逐行平场增益", "above"),
    ("bow_gain", "bow_gain", "二阶曲面增益", "above"),
    ("sep_over_rough", "step_sep", "峰间距/粗糙度", "above"),
    ("row_purity", "step_purity", "行纯度", "above"),
    ("fine_periodic_snr", "fine_periodic_snr", "精细周期 SNR", "above"),
    ("rowcorr_median", "rowcorr_poor", "相邻行相关", "below"),
    ("fb_instability", "fb_instability_max", "正反扫不稳定度", "above"),
    # nan_frac 只作描述量:``nan_annotate`` 是一条**显示策略**(「缺多少像素值得
    # 在报告里提一句」),不是一个需要按样品标定的判据,放进任何一张标定表都会误导。
    ("nan_frac", None, "NaN 占比", ""),
    ("roughness_pm", None, "像素级粗糙度(pm)", ""),
    ("step_sep_pm", None, "峰间距(pm)", ""),
    ("plane_rms_pm", None, "扣平面后 RMS(pm)", ""),
)

#: 最大间隙要占全距多少才算「这批数分成了两团」。
_GAP_MIN_FRAC = 0.25


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def _describe(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "p10": _pct(xs, 10), "p50": _pct(xs, 50),
            "p90": _pct(xs, 90), "max": max(xs)}


def _largest_gap(xs: list[float]) -> dict:
    """最大间隙的中点,以及它两边各有几个。分不成两团时 ``suggested=None``。"""
    s = sorted(xs)
    if len(s) < 4:
        return {"suggested": None,
                "note": f"只有 {len(s)} 个样本,太少 —— 保留当前值"}
    span = s[-1] - s[0]
    if span <= 0:
        return {"suggested": None,
                "note": "这批图上这个量没有变化,无从标定 —— 保留当前值"}
    gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)]
    gap, i = max(gaps)
    if gap < _GAP_MIN_FRAC * span:
        return {"suggested": None,
                "note": (f"这批图上没有分成两团(最大间隙只占全距的 "
                         f"{gap / span:.0%},需要 ≥{_GAP_MIN_FRAC:.0%}) —— 保留当前值")}
    return {"suggested": 0.5 * (s[i] + s[i + 1]), "gap": gap, "span": span,
            "n_below": i + 1, "n_above": len(s) - i - 1}


# ── 取数 ────────────────────────────────────────────────────────────────


def collect(paths: list[str], channel: str = "Z",
            profile: "str | None" = None) -> dict:
    """量一批 ``.sxm``,返回 ``{profile, thresholds, frames:[…], errors:[…]}``。

    每个 frame 是一行扁平指标 + 选出来的处理方式。不渲染、不写盘。
    """
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.scan_prep import measure_frame, plan_for
    from mast.vision.scan_prep_thresholds import resolve

    th = resolve(profile)
    frames: list[dict] = []
    errors: list[str] = []
    for path in paths:
        try:
            scan = read_sxm(path)
            fr = sxm_oriented_frames(scan, channel)
            if fr["forward"] is None:
                errors.append(f"{os.path.basename(path)}: 没有 {channel} 通道")
                continue
            m = measure_frame(fr["forward"], bwd=fr["backward"],
                              nm_per_px=fr["nm_per_px"], thresholds=th)
            p = plan_for(m, th)
        except Exception as exc:  # noqa: BLE001 — 一个坏文件不该让整批标定停下
            errors.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
            continue
        row = m.to_dict()
        row["file"] = os.path.basename(path)
        row["method"] = p.method
        row["step_like"] = p.step_like
        row["fine_structure"] = p.fine_structure
        row["width_nm"] = fr["width_nm"]
        row["bias_v"] = fr["bias_v"]
        frames.append(row)
    return {"profile": th.name, "provenance": th.provenance,
            "thresholds": th.numeric_mapping(), "channel": channel,
            "frames": frames, "errors": errors}


def analyse(batch: dict) -> dict:
    """分布 + 每个 knob 的现状与建议。"""
    from mast.vision.scan_prep_thresholds import CALIBRATABLE_FROM_DISTRIBUTION

    th = batch.get("thresholds") or {}
    frames = batch.get("frames") or []
    out_metrics: list[dict] = []
    for key, knob, label, direction in _METRICS:
        vals = [v for v in (_finite(f.get(key)) for f in frames) if v is not None]
        entry: dict = {"metric": key, "knob": knob, "label": label,
                       "direction": direction, "dist": _describe(vals)}
        if knob is not None:
            cur = _finite(th.get(knob))
            entry["current"] = cur
            if cur is not None and vals:
                fires = (sum(1 for v in vals if v > cur) if direction == "above"
                         else sum(1 for v in vals if v < cur))
                entry["n_fires"] = fires
                entry["n_total"] = len(vals)
            entry["calibratable"] = knob in CALIBRATABLE_FROM_DISTRIBUTION
            if entry["calibratable"]:
                entry.update(_largest_gap(vals))
                if knob == "fb_instability_max":
                    entry["reference_p85"] = _pct(vals, 85)
        out_metrics.append(entry)

    methods: dict[str, int] = {}
    for f in frames:
        methods[str(f.get("method"))] = methods.get(str(f.get("method")), 0) + 1
    return {"n_frames": len(frames), "metrics": out_metrics,
            "method_counts": methods, "errors": batch.get("errors") or []}


# ── 报告 ────────────────────────────────────────────────────────────────


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4g}"


def render(batch: dict, report: dict) -> str:
    L: list[str] = []
    L.append("═══ 扫描图预处理 · 阈值标定报告 ═══\n")
    L.append(f"通道 {batch.get('channel')},共 {report['n_frames']} 帧,"
             f"当前 profile `{batch.get('profile')}`")
    L.append(f"  出身:{batch.get('provenance')}")
    if report["errors"]:
        L.append(f"  读不动的文件 {len(report['errors'])} 个:")
        for e in report["errors"][:5]:
            L.append(f"    {e}")
    L.append("")

    L.append("【处理方式分布】")
    for k, n in sorted(report["method_counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"  {k:14s} {n:4d}")
    L.append("")

    L.append("【指标分布】")
    L.append(f"  {'指标':18s} {'n':>4s} {'min':>10s} {'p10':>10s} {'p50':>10s} "
             f"{'p90':>10s} {'max':>10s}")
    for e in report["metrics"]:
        d = e["dist"]
        if not d.get("n"):
            continue
        L.append(f"  {e['label']:18s} {d['n']:4d} {_fmt(d['min']):>10s} "
                 f"{_fmt(d['p10']):>10s} {_fmt(d['p50']):>10s} "
                 f"{_fmt(d['p90']):>10s} {_fmt(d['max']):>10s}")
    L.append("")

    L.append("【能从分布标定的阈值】(最大间隙中点;绝不自动应用)")
    for e in report["metrics"]:
        if not e.get("knob") or not e.get("calibratable"):
            continue
        cur, sug = e.get("current"), e.get("suggested")
        head = f"  {e['label']:18s} 当前 {_fmt(cur):>10s}"
        if not e["dist"].get("n"):
            L.append(f"{head}   这批图里量不到这个值")
            continue
        fires = f"这批图里 {e.get('n_fires', 0)}/{e.get('n_total', 0)} 帧触发"
        if sug is None:
            L.append(f"{head}                  {fires};{e.get('note', '')}")
        else:
            L.append(f"{head} → 建议 {_fmt(sug):>10s}   {fires};"
                     f"两团 {e['n_below']}/{e['n_above']}")
        if e.get("reference_p85") is not None:
            L.append(f"    本批 p85 = {_fmt(e['reference_p85'])}；"
                     "仅描述本批分布，不代表已标定参考值")
    L.append("")

    L.append("【不从分布标定的阈值】(只报这批图被切成了几张几张)")
    for e in report["metrics"]:
        if not e.get("knob") or e.get("calibratable"):
            continue
        if not e["dist"].get("n"):
            continue
        L.append(f"  {e['label']:18s} 当前 {_fmt(e.get('current')):>10s}   "
                 f"这批图里 {e.get('n_fires', 0)}/{e.get('n_total', 0)} 帧触发")
    L.append("    ↑ 这几条是关于*模型选择*与*几何*的陈述,不随样品的噪声水平漂移。")
    L.append("      「逐行平场把残差压掉 30% 以上才值得用」、「真台阶横跨画面所以大多数")
    L.append("      行同时含两个高度」在任何样品上都成立。真要改,拿论证来,不是拿分布。")
    L.append("      两边都是 0/N 或 N/N 时才值得怀疑 —— 那说明这个判据在这批图上没在工作。")
    L.append("")

    sugg = {e["knob"]: e["suggested"] for e in report["metrics"]
            if e.get("calibratable") and e.get("suggested") is not None}
    if sugg:
        from mast.vision.scan_prep_thresholds import DEFAULT_PROFILE
        L.append("【可粘进 config/scan_prep_profiles.json 的 profile】")
        L.append("(先看上面的数字再决定要不要用;本工具不会自己写这个文件)")
        body = {"<自定义profile名称>": {
            "provenance": f"{report['n_frames']} 张 <样品名> 帧, <日期>, "
                          f"由 mast.vision.scan_prep_commission 建议",
            "base": DEFAULT_PROFILE,
            "thresholds": {k: round(v, 4) for k, v in sugg.items()},
        }}
        L.append(json.dumps(body, ensure_ascii=False, indent=2))
    else:
        L.append("【没有可建议的阈值】这批图在每个可标定的量上都没有分成两团。")
        L.append("  这本身是个结论:要么这批图太同质(全好或全坏),要么样本太少。")
    return "\n".join(L)


# ── CLI ─────────────────────────────────────────────────────────────────


def _expand(inputs: list[str]) -> list[str]:
    files: list[str] = []
    for p in inputs:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*.sxm"))
        elif p.lower().endswith(".sxm"):
            files.append(p)
    return sorted(set(files))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m mast.vision.scan_prep_commission",
        description="扫描图预处理阈值标定:看这批图的指标分布 → 建议阈值(只读)")
    ap.add_argument("inputs", nargs="+", help="文件夹或若干 .sxm")
    ap.add_argument("--channel", default="Z", help="通道名(默认 Z)")
    ap.add_argument("--profile", default=None, help="以哪个 profile 为对照(默认当前激活的)")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非报告")
    args = ap.parse_args(argv)

    files = _expand(args.inputs)
    if not files:
        print("没找到 .sxm 文件", file=sys.stderr)
        return 1
    batch = collect(files, channel=args.channel, profile=args.profile)
    report = analyse(batch)
    if args.json:
        print(json.dumps({"batch": {k: v for k, v in batch.items() if k != "frames"},
                          "frames": batch["frames"], "report": report},
                         ensure_ascii=False, indent=2, default=float))
    else:
        print(render(batch, report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["analyse", "collect", "main", "render"]
