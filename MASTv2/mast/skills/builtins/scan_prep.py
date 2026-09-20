"""扫描图自动预处理的技能外壳 —— 判据本体在 :mod:`mast.vision.scan_prep`。

* :class:`AnalyzeScanImage` —— 一张 ``.sxm``:量指标、选平场方式与色阶、说清楚为什么。
* :class:`AutoProcessScanBatch` —— 一个文件夹:同上 + 批次一致性 + PNG + ``_report.md``。

设计文档:``docs/v2/design/scan_prep_auto_flatten.md``。

## 这一层只做 IO,一个阈值都不判

判据是 :func:`mast.vision.scan_prep.measure_frame` / :func:`~mast.vision.scan_prep.plan_for`
两个纯函数(零 IO、阈值全参数化,所以合成数据测得动)。这里只负责:读文件、解析像素
尺度、挑 profile、渲染、写报告。

## 结论从哪来

「这一帧上有什么」的结论**不在这里产生**,而是转发既有判据:原子相来自
:mod:`mast.vision.atomic_phase`,针尖突变来自 :mod:`mast.vision.tip_change`,
正反扫来自 :mod:`mast.vision.tip_metrics`,坏行/振荡来自 :mod:`mast.vision.scan_artifacts`。
本技能自己只回答一个问题:**这一帧该怎么处理**。理由见设计文档 §2 —— 那四条判据
在仓里的版本都实测推翻过 sxm_auto 用的那一版,两套并存等于往模型上下文里塞两句
互相打架的结论。

## 三态,不是两态

只要文件读得动就 ``success=True``;判据的结论在 ``data`` 里。技能失败保留给「这件事
没做成」—— 文件不存在、通道缺失、依赖缺席。把「这一帧质量不好」表达成技能失败,
会让 composite 里 ``optional=False`` 的步骤直接中止整条流程,而「这一帧不好」恰恰是
流程要处理的正常情况。
"""

from __future__ import annotations

import logging
from pathlib import Path

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

_FLATTEN_CHOICES = ["auto", "plane", "poly2", "line", "masked_line"]

#: 一次批处理最多看多少个文件 —— 防止有人把整个数据盘指过来。
_MAX_BATCH = 200


def _profile_params() -> list[ParameterSpec]:
    """两个技能共用的 profile 参数。"""
    return [
        ParameterSpec(
            name="threshold_profile", type="str",
            description=(
                "阈值 profile = 这套阈值是在**哪个样品体系**上标定的。"
                "留空则用当前生效的那个。公开版默认 generic-uncommissioned 未标定，"
                "仅供合成示例；用于实际数据前，请先跑 "
                "`python -m mast.vision.scan_prep_commission <folder>`，"
                "再根据观察到的分布定义一个新的 profile。"),
            required=False, default=""),
        ParameterSpec(
            name="channel", type="str",
            description="通道名（默认 'Z'）。",
            required=False, default="Z"),
        ParameterSpec(
            name="flatten", type="str",
            description=("覆盖自动的 flattening 选择。'auto'"
                         "（默认）让测量结果自己决定。"),
            required=False, default="auto", allowed_values=list(_FLATTEN_CHOICES)),
    ]


def _load_frame(scan_path: str, channel: str):
    """``(frames dict, error)``。error 非空即读不动。"""
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

    p = Path(str(scan_path))
    if not p.exists():
        return None, f"文件不存在: {p}"
    try:
        scan = read_sxm(str(p))
    except Exception as exc:  # noqa: BLE001
        return None, f"读不动 .sxm: {type(exc).__name__}: {exc}"
    fr = sxm_oriented_frames(scan, channel)
    if fr.get("forward") is None:
        have = list((scan.get("channels") or {}).keys())
        return None, f"没有 {channel!r} 通道(这个文件里有: {have})"
    try:
        from mast.core.scan_registry import record_scan_path

        record_scan_path(str(p))
    except Exception:  # noqa: BLE001 — 登记失败不该让分析失败
        pass
    return fr, ""


def _output_dir(explicit: str, tag: str) -> Path:
    if explicit:
        d = Path(explicit)
    else:
        from mast.agents._shared.data_paths import figures_dir

        d = figures_dir() / f"scan_prep_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _render(flat, fr, out_png: Path, plan, title: str) -> str:
    """渲染一张 PNG,返回路径(失败返回空串 —— 画不出来不该让分析失败)。"""
    try:
        from mast.data.visualization import plot_flattened_scan

        fig = plot_flattened_scan(
            flat, width_nm=fr.get("width_nm"), height_nm=fr.get("height_nm"),
            clip_percentile=plan.clip, title=title,
            subtitle=(f"{plan.method}  |  clip {plan.clip[0]}-{plan.clip[1]} pct"
                      f"  |  V = {fr.get('bias_v')}"),
            unit=fr.get("unit") or "m", save_path=str(out_png))
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
            return str(out_png)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scan_prep 渲染失败 %s: %s", out_png, exc)
    return ""


class AnalyzeScanImage(BaseSkill):
    """量一张 .sxm,选出平场方式与色阶,并说清楚每一步的依据。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AnalyzeScanImage",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "测量**一帧**已保存的 .sxm，并决定它该怎么处理："
                "用哪种 flattening（平面 / 2nd-order 曲面 / "
                "逐行 / 在主导 terrace 上拟合行）、色标该收多紧 —— "
                "每一个选择都附上实测的理由。它只读，不碰硬件。它还会"
                "**转述**已有的那些判语：原子相位（mast.vision.atomic_phase）、"
                "扫描中途的针尖变化（mast.vision.tip_change）、正反扫一致性、"
                "坏扫描行。它绝不自己宣称看到了晶格 —— 那个断言来自 "
                "atomic_phase，而后者还能回答「在这个像素尺寸下判不了」，"
                "这跟「没有晶格」是两回事。"
            ),
            parameters=[
                ParameterSpec(name="scan_path", type="str",
                              description="已保存的那一帧 .sxm 的路径。",
                              required=True),
                *_profile_params(),
                ParameterSpec(
                    name="save_png", type="bool",
                    description=("顺便把 flatten 之后的帧，按所选色标"
                                 "渲染成一张 PNG。"),
                    required=False, default=False),
                ParameterSpec(name="output_dir", type="str",
                              description="PNG 放到哪里（默认：figures 目录）。",
                              required=False, default=""),
            ],
            estimated_duration_s=3.0,
            composition_level=2,
            tags=["scan", "analysis", "flatten", "preprocessing", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "AnalyzeScanImage"
        scan_path = str(params["scan_path"])
        channel = params.get("channel") or "Z"
        flatten = (params.get("flatten") or "auto").lower()
        profile = (params.get("threshold_profile") or "").strip()
        save_png = bool(params.get("save_png", False))

        try:
            from mast.vision.scan_prep import METHOD_LABEL, apply_flatten, measure_frame, plan_for
            from mast.vision.scan_prep_thresholds import resolve
        except ImportError as exc:
            return SkillResult(skill_name=name, success=False,
                               error=f"依赖缺席: {exc}")

        fr, err = _load_frame(scan_path, channel)
        if err:
            return SkillResult(skill_name=name, success=False, error=err)

        th = resolve(profile or None)
        try:
            m = measure_frame(fr["forward"], bwd=fr["backward"],
                              nm_per_px=fr["nm_per_px"], thresholds=th)
            plan = plan_for(m, th, override=flatten)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f"分析失败: {type(exc).__name__}: {exc}")

        png = ""
        if save_png:
            flat = apply_flatten(fr["forward"], plan.method, m)
            out = _output_dir(str(params.get("output_dir") or ""),
                              Path(scan_path).stem)
            png = _render(flat, fr, out / f"{Path(scan_path).stem}_{channel}_auto.png",
                          plan, Path(scan_path).stem)

        data = {
            "scan_path": scan_path,
            "channel": channel,
            "nm_per_px": fr["nm_per_px"],
            "width_nm": fr["width_nm"],
            "bias_v": fr["bias_v"],
            "metrics": m.to_dict(),
            "plan": plan.to_dict(),
            "png_path": png,
        }
        summary = (
            f"{Path(scan_path).name} [{channel}] → {plan.method}"
            f"({METHOD_LABEL.get(plan.method, plan.method)}), 色阶 "
            f"{plan.clip[0]}–{plan.clip[1]} 百分位 | line_gain {m.line_gain:.2f} "
            f"bow_gain {m.bow_gain:.2f} 行相关 {m.rowcorr_median:.2f}"
            f"\n依据: " + " / ".join(plan.why)
            + ("\n注意: " + " / ".join(plan.notes) if plan.notes else "")
            + f"\n阈值 profile `{plan.profile}` — {plan.provenance}")
        # 图像通道:这张 PNG 已经带了自动平场 + 选好的色阶,正是本技能算出来的那张
        # 「该怎么看」的图 —— 比原始 .sxm 缩略图更有信息量。给模型看这一张。
        # 只放路径(SkillResult.images 的契约);像素由 vision_mw 在出站时 materialize。
        return SkillResult(skill_name=name, success=True, data=data, summary=summary,
                           images=[png] if png else [])


class AutoProcessScanBatch(BaseSkill):
    """一整个文件夹的 .sxm:批次一致的处理方案 + PNG + 逐张说明的报告。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AutoProcessScanBatch",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "测量一个文件夹里的**每一个** .sxm，逐帧挑好 flattening 与色标，"
                "然后把扫描尺寸与偏压相同的那些帧**统一**起来 —— 好让其中两帧"
                "之间的对比度差异只可能来自样品，绝不可能来自处理过程"
                "（含有真实台阶的帧保留各自的保护性处理，不参与统一）。"
                "它会渲染出这些 PNG，并写一份 _report.md，里面有每一个实测"
                "数字、以及每一个决定的理由。它只读，不碰硬件。"
            ),
            parameters=[
                ParameterSpec(name="folder", type="str",
                              description="存放这些 .sxm 文件的文件夹。",
                              required=True),
                *_profile_params(),
                ParameterSpec(name="render", type="bool",
                              description="每帧渲染一张 PNG（默认 true）。",
                              required=False, default=True),
                ParameterSpec(name="write_report", type="bool",
                              description="写出 _report.md（默认 true）。",
                              required=False, default=True),
                ParameterSpec(name="output_dir", type="str",
                              description=("PNG 和这份报告放到哪里"
                                           "（默认：figures 目录）。"),
                              required=False, default=""),
                ParameterSpec(name="max_files", type="int",
                              description=f"处理文件数的上限（最多 {_MAX_BATCH}）。",
                              required=False, default=_MAX_BATCH,
                              min_value=1, max_value=_MAX_BATCH),
            ],
            estimated_duration_s=60.0,
            composition_level=2,
            tags=["scan", "analysis", "flatten", "preprocessing", "batch", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "AutoProcessScanBatch"
        folder = Path(str(params["folder"]))
        channel = params.get("channel") or "Z"
        flatten = (params.get("flatten") or "auto").lower()
        profile = (params.get("threshold_profile") or "").strip()
        do_render = bool(params.get("render", True))
        do_report = bool(params.get("write_report", True))
        max_files = int(params.get("max_files") or _MAX_BATCH)

        try:
            from mast.vision.scan_prep import (
                apply_flatten, harmonise_batch, measure_frame, plan_for,
            )
            from mast.vision.scan_prep_thresholds import resolve
        except ImportError as exc:
            return SkillResult(skill_name=name, success=False,
                               error=f"依赖缺席: {exc}")

        if not folder.is_dir():
            return SkillResult(skill_name=name, success=False,
                               error=f"不是一个目录: {folder}")
        files = sorted(folder.glob("*.sxm"))[:max(1, min(max_files, _MAX_BATCH))]
        if not files:
            return SkillResult(skill_name=name, success=False,
                               error=f"{folder} 里没有 .sxm 文件")

        th = resolve(profile or None)
        items: list[dict] = []
        skipped: list[str] = []
        for path in files:
            fr, err = _load_frame(str(path), channel)
            if err:
                skipped.append(f"{path.name}: {err}")
                continue
            try:
                m = measure_frame(fr["forward"], bwd=fr["backward"],
                                  nm_per_px=fr["nm_per_px"], thresholds=th)
                plan = plan_for(m, th, override=flatten)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            items.append({"path": path, "fr": fr, "m": m, "plan": plan})

        if not items:
            return SkillResult(skill_name=name, success=False,
                               error=("这个目录里没有一张读得动的 .sxm。"
                                      + " | ".join(skipped[:5])))

        # ── 批次一致性(只在 auto 模式下;调用方指定了方式就没有票可投) ──
        if flatten == "auto":
            keyed = [((round(float(it["fr"]["width_nm"] or 0), 3),
                       round(float(it["fr"]["bias_v"] or 0), 4)), it["plan"])
                     for it in items]
            for it, plan in zip(items, harmonise_batch(keyed, th)):
                it["plan"] = plan

        out = _output_dir(str(params.get("output_dir") or ""), folder.name)
        rows: list[dict] = []
        for it in items:
            path, fr, m, plan = it["path"], it["fr"], it["m"], it["plan"]
            png = ""
            if do_render:
                flat = apply_flatten(fr["forward"], plan.method, m)
                png = _render(flat, fr, out / f"{path.stem}_{channel}_auto.png",
                              plan, path.stem)
            rows.append({
                "file": path.name,
                "width_nm": fr["width_nm"],
                "bias_v": fr["bias_v"],
                "nm_per_px": fr["nm_per_px"],
                "method": plan.method,
                "clip_percentile": list(plan.clip),
                "step_like": plan.step_like,
                "line_gain": round(m.line_gain, 3),
                "bow_gain": round(m.bow_gain, 3),
                "row_purity": m.row_purity,
                "sep_over_rough": round(m.sep_over_rough, 2),
                "rowcorr_median": m.rowcorr_median,
                "fb_instability": m.fb_instability,
                "nan_frac": round(m.nan_frac, 4),
                "atomic_passed": (m.atomic or {}).get("passed"),
                "atomic_reasons": (m.atomic or {}).get("reasons"),
                "tip_changed": (m.tip_change or {}).get("changed"),
                "png_path": png,
            })

        report_path = ""
        if do_report:
            report_path = _write_report(out, channel, th, items, skipped)

        counts: dict[str, int] = {}
        for r in rows:
            counts[r["method"]] = counts.get(r["method"], 0) + 1
        data = {"folder": str(folder), "channel": channel, "output_dir": str(out),
                "report_path": report_path, "n_files": len(rows),
                "method_counts": counts, "skipped": skipped, "frames": rows,
                "threshold_profile": th.name}
        summary = (
            f"{len(rows)} 张 [{channel}] → "
            + ", ".join(f"{k} × {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            + (f";{len(skipped)} 张读不动" if skipped else "")
            + (f"\n报告(每张图测了什么、为什么这么处理): {report_path}" if report_path else "")
            + (f"\nPNG: {out}" if do_render else "")
            + f"\n阈值 profile `{th.name}` — {th.provenance}")
        # 同 AnalyzeScanImage:批次里每张都渲染了 PNG,路径一并交出去。带几张由
        # vision_mw 的 MAX_IMAGES_PER_REQUEST 封顶(它取最近的几张),所以这里不必
        # 自己截断 —— 截断规则只该有一处。
        return SkillResult(skill_name=name, success=True, data=data, summary=summary,
                           images=[r["png_path"] for r in rows if r["png_path"]])


def _g(v, unit: str = "") -> str:
    """格式化一个可能缺席的表头数值。头里没有 ``SCAN_RANGE`` / ``BIAS`` 的文件是存在的
    (截断的、别家软件导出的),不能让一个 ``None`` 把整份报告的格式化炸掉。"""
    if v is None:
        return "?"
    try:
        return f"{float(v):g}{unit}"
    except (TypeError, ValueError):
        return "?"


def _write_report(out: Path, channel: str, th, items: list[dict],
                  skipped: list[str]) -> str:
    """把每张图测到的数与每一步依据写成 Markdown。写不成返回空串。"""
    from mast.vision.scan_prep import METHOD_LABEL

    path = out / "_report.md"
    try:
        L: list[str] = []
        L.append("# 扫描图自动预处理报告\n")
        L.append(f"通道 `{channel}`,共 {len(items)} 个文件。"
                 f"下表是每张图测到的量,以及据此选择的处理方式。\n")
        L.append(f"**阈值 profile**:`{th.name}` —— {th.provenance}\n")
        L.append("| 文件 | 视野 | 偏压 | NaN | line_gain | bow_gain | "
                 "峰数/间距比/行纯度 | 精细SNR/周期 | 行相关 | 正反扫不稳 | "
                 "原子相 | 针尖突变 | 处理 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for it in items:
            m, plan, fr = it["m"], it["plan"], it["fr"]
            pur = f"{m.row_purity:.2f}" if m.row_purity == m.row_purity else "–"
            fine = (f"**{m.fine_periodic_snr:.0f} / {m.fine_period_nm:.3f}nm**"
                    if plan.fine_structure else f"{m.fine_periodic_snr:.0f} / –")
            a = m.atomic or {}
            atom = ("**通过**" if a.get("passed")
                    else ("判不了" if "scale_gate" in (a.get("reasons") or [])
                          else "未通过"))
            tc = m.tip_change or {}
            tcs = (f"**是**@{tc.get('change_row')}" if tc.get("changed")
                   else (f"否(z {tc.get('score', float('nan')):.1f})" if tc else "–"))
            fb = (f"{m.fb_instability:.2f}" if m.fb_instability is not None else "–")
            L.append(
                f"| {it['path'].stem[-4:]} | {_g(fr['width_nm'], ' nm')} | "
                f"{_g(fr['bias_v'], ' V')} | "
                f"{m.nan_frac * 100:.0f}% | {m.line_gain:.2f} | {m.bow_gain:.2f} | "
                f"{m.n_peaks} / {m.sep_over_rough:.1f} / {pur} | {fine} | "
                f"{m.rowcorr_median:+.2f} | {fb} | {atom} | {tcs} | "
                f"`{plan.method}` |")
        L.append("")
        L.append("判据阈值:" + ", ".join(
            f"`{k}`={v:g}" for k, v in sorted(th.numeric_mapping().items())))
        L.append("")
        L.append("> 「原子相」「针尖突变」「正反扫不稳」三列**不是本模块自己判的**,"
                 "分别转发 `mast.vision.atomic_phase` / `mast.vision.tip_change` / "
                 "`mast.vision.tip_metrics`。"
                 "「精细SNR」只用来决定色阶,不是「有没有晶格」的结论 —— "
                 "峰强度分不开针尖抖动造出的准周期条纹。详见 "
                 "`docs/v2/design/scan_prep_auto_flatten.md`。")
        L.append("")
        if skipped:
            L.append("## 读不动的文件\n")
            for s in skipped:
                L.append(f"- {s}")
            L.append("")
        L.append("---\n")
        L.append("## 逐张说明\n")
        for it in items:
            m, plan, fr = it["m"], it["plan"], it["fr"]
            L.append(f"### {it['path'].stem}\n")
            L.append(f"{_g(fr['width_nm'])} × "
                     f"{_g(fr['height_nm'] or fr['width_nm'])} nm, "
                     f"V = {_g(fr['bias_v'])} V, {fr['rec_time']}, "
                     f"{m.shape[0]}×{m.shape[1]} px ({(m.nm_per_px or 0):.4f} nm/px)\n")
            L.append(f"**处理**:{METHOD_LABEL.get(plan.method, plan.method)};"
                     f"色阶 {plan.clip[0]}–{plan.clip[1]} 百分位\n")
            L.append("**依据**:")
            for w in plan.why:
                L.append(f"- {w}")
            if plan.notes:
                L.append("\n**注意**:")
                for n in plan.notes:
                    L.append(f"- {n}")
            L.append("")
        path.write_text("\n".join(L), encoding="utf-8")
        return str(path)
    except Exception as exc:  # noqa: BLE001 — 写不成报告不该让整个批处理失败
        logger.warning("scan_prep 报告写不成 %s: %s", path, exc)
        return ""


__all__ = ["AnalyzeScanImage", "AutoProcessScanBatch"]
