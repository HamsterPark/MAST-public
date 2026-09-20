"""畴指纹判据的技能外壳 —— 判据本体在 ``mast.vision.domain_phase`` 里。

:class:`AssessDomainPhase` —— 一帧原子分辨图上的原子相是**哪一种**(多畴/孪晶/
多相晶体里的哪个畴)。这是「找畴界」那条流程的读数端:铺网格采样、二分定位都靠
它逐帧给判定。

**这一层只做 IO 与像素尺度/帧角解析,一个阈值都不判**。判据是
:func:`~mast.vision.domain_phase.extract_fingerprint` 与
:func:`~mast.vision.domain_phase.classify` 两个纯函数(零 IO、阈值全参数化,合成
数据测得动:反例误报率、四检验、分离度都在 ``tests/v2/unit/vision/`` 里)。

## 参照系没建立之前,它只出指纹

「这是 A 相」需要一个**人确认过**的参照系(``config/domain_references/*.json``,
见 :mod:`mast.vision.domain_reference`)。没有参照系时本技能返回
``verdict="undetermined"`` / ``reason="no_reference"`` **加上指纹本身** ——
**这是正确行为,不是失败**:第一轮普查的产物就是一批指纹,拿去聚类、给人看、
确认成参照系。代码不能自己发明「这是 A 相」。

## 帧角:从 header 直读

``sxm_frame_meta`` 只转发 ``channels``/``scan_offset``/``scan_range``/``scan_pixels``,
**角度在那一层被丢掉** —— 走 registry / checkpoint 那条路的帧拿不到角度。所以这里
直接从 ``.sxm`` header 解析(``mast.io.mosaic.parse_xy_meta`` 的 ``angle_known``),
**读不到就 ``undetermined(unknown_frame_angle)``,不按 0° 处理**:折叠成 0.0 会让
同一个畴在两种帧角下报成两个畴,而且零报错。

## 三态,不是两态

技能**永远返回 ``success=True``**(只要文件读得动):判定在 ``data.verdict`` 与
``data.verdict_reason`` 里。技能失败保留给「这件事没做成」—— 文件不存在、通道缺失、
依赖缺席。把「判不了」表达成技能失败会让 composite 的 ``optional=False`` 步骤直接
中止整条搜索,而「这一帧判不了」恰恰是搜索要处理的正常情况。
"""

from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: ``undetermined`` 细分码 → 给用户/规划器的**可执行下一步**。
#: 「判不了」不给下一步等于死路,而且每一条的下一步都不一样 —— 尤其
#: ``scale_gate``(换更小的视野)与 ``too_few_periods``(换**更大**的视野)是相反的。
NEXT_STEP: dict[str, str] = {
    "unknown_pixel_size": "帧的几何信息缺失,补 header/扫描范围来源再判。",
    "scale_gate": "这一帧太粗,分辨不了晶格 —— 换更小的视野或更多像素。"
                  "这不等于「这里没有畴」。",
    "scale_reduced": "落在过渡带,测量本身是降级的 —— 换更小的视野,"
                     "或显式接受过渡带(allow_reduced_scale)。",
    "too_few_periods": "帧里的晶格周期太少,判据静默失效 —— 换**更大**的视野"
                       "(注意与 scale_gate 的方向相反)。",
    "insufficient_data": "这一帧不是一张可用的二维图。",
    "resampled_input": "输入被重采样过 —— 拿原生采集帧来(尺度门在重采样帧上等于没有)。",
    "unknown_frame_angle": "帧角读不到 —— 补角度来源。**不按 0° 处理**:"
                           "那会让同一个畴在两种帧角下报成两个畴。",
    "no_atomic_phase": "这一帧根本没有原子分辨 —— 换点或修针。**不是新畴**。",
    "no_peaks": "原子周期带里没有可用谱峰 —— 换点或修针。",
    "slow_axis_degenerate": "主峰落在慢轴缺口里,而不做行对齐时它不存在 —— "
                            "把扫描框转 ~30° 重扫一张再判。",
    "shear_suspect": "径向周期与快扫方向的周期对不上 —— 降扫描速度/等漂移稳再扫。",
    "no_reference": "还没有标定过的参照系 —— 先采一批指纹、聚类、人确认,"
                    "固化成 config/domain_references/*.json。这是正常流程的第一步。",
    "reference_unusable": "参照系里没有可比的原型 —— 检查那个 JSON。",
    "ambiguous_match": "到每个原型的距离都差不多,而且哪个的峰都不全在 —— "
                       "同点重采一帧,仍不可判就换个偏移点。",
    "no_match": "指纹像谁都不像(可能是第三个畴)—— **这是证据矛盾,交给人看**,"
                "不要自动加一簇。",
}


class AssessDomainPhase(BaseSkill):
    """从一帧原子分辨图认出它属于哪个畴(需要标定过的参照系;没有就只出指纹)。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessDomainPhase",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "认出一帧已保存的 .sxm 显示的是**哪一个**原子相 / 畴 —— 面向多畴、"
                "孪晶或多相晶体。只读文件,不碰硬件。它抽出一张**方向-周期**峰表"
                "(也就是指纹),再拿它去比对 config/domain_references/*.json 里"
                "**经用户确认过**的原型。**没有**这样一个参照系文件时,它返回 "
                "verdict='undetermined'、reason='no_reference',外加指纹本身 —— "
                "这是正确的第一轮产出,不是失败:代码绝不能自己发明「这是 A 相」。"
                "verdict='mixed' 表示两个畴的峰**都在**,也就是有一道畴界正穿过"
                "**这一帧** —— 找畴界的时候,这是最有价值的信号。帧角读不出来的帧"
                "一律拒判(绝不按 0 deg 处理);比 0.05 nm/px 还粗的帧宁可拒绝判定,"
                "也不报「没有」。"
            ),
            parameters=[
                ParameterSpec(name="scan_path", type="str",
                              description=".sxm 帧的路径(要原子分辨的那种)。",
                              required=True),
                ParameterSpec(name="channel", type="str",
                              description="形貌通道('Z' 是标准选择)。",
                              required=False, default="Z"),
                ParameterSpec(
                    name="reference_version", type="str",
                    description=("畴参照系的版本(例如 'v001')。留空就取最新的"
                                 "那一版。不存在的版本一律拒绝,绝不悄悄换成别的。"),
                    required=False, default=""),
                ParameterSpec(
                    name="sample", type="str",
                    description=("样品名。只有当多个样品的参照系并存时才需要"
                                 "(那种情况下它宁可拒绝,也不去猜)。"),
                    required=False, default=""),
                ParameterSpec(
                    name="max_peaks", type="int",
                    description=("保留几个方向-周期峰。六方晶格给出 3 个不同"
                                 "方向(mod 180),矩形晶格是 2 个。"),
                    required=False, default=6, min_value=1, max_value=24),
                ParameterSpec(
                    name="allow_reduced_scale", type="bool",
                    description=("接受 0.02-0.05 nm/px 这条过渡带。"
                                 "在那里,测量本身就是降级的。"),
                    required=False, default=False),
            ],
            estimated_duration_s=4.0,
            composition_level=1,
            tags=["domain", "lattice", "fingerprint", "fft", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        name = "AssessDomainPhase"
        path = str(params["scan_path"])
        channel_name = params.get("channel") or "Z"
        if not Path(path).exists():
            return SkillResult(skill_name=name, success=False,
                               error=f"文件不存在: {path}")

        try:
            import numpy as np

            from mast.io.mosaic import parse_xy_meta
            from mast.io.nanonis_files import read_sxm
            from mast.vision.domain_phase import classify, extract_fingerprint
            from mast.vision.domain_reference import load_reference
        except ImportError as exc:
            return SkillResult(skill_name=name, success=False, error=f"缺依赖: {exc}")

        try:
            scan = read_sxm(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f".sxm 读取失败: {exc}")

        channels = scan.get("channels", {}) or {}
        ch = channels.get(channel_name) or next(iter(channels.values()), None)
        if ch is None:
            return SkillResult(
                skill_name=name, success=False,
                error=f"文件里没有可用通道(要的是 {channel_name!r})")
        arr = ch.get("forward")
        if arr is None:
            arr = ch.get("backward")
        if arr is None:
            return SkillResult(skill_name=name, success=False,
                               error="通道里没有正扫/反扫数据")
        arr = np.asarray(arr, dtype=np.float64)

        # 像素尺度与帧角:两个都从 header 来。**帧角读不到就是 None**,
        # ``parse_xy_meta`` 为了别的调用方把它兜底成 0.0,但同时给了 angle_known ——
        # 这里认那个旗标,不认那个 0.0。
        nm_per_px = None
        scan_angle_deg = None
        try:
            meta = parse_xy_meta(scan.get("header", {}) or {})
        except Exception:  # noqa: BLE001
            meta = None
        if meta:
            if arr.ndim == 2 and arr.shape[1]:
                nm_per_px = (float(meta["w"]) / arr.shape[1]) * 1e9
            if meta.get("angle_known"):
                scan_angle_deg = float(meta["angle"])

        fp = extract_fingerprint(
            arr,
            nm_per_px=nm_per_px,
            scan_angle_deg=scan_angle_deg,
            max_peaks=int(params.get("max_peaks") or 6),
            allow_reduced_scale=bool(params.get("allow_reduced_scale", False)),
            native_sampling=True,      # .sxm 是原生采集帧
        )

        version = str(params.get("reference_version") or "").strip()
        sample = str(params.get("sample") or "").strip()
        try:
            ref = load_reference(version or None, sample=sample or None)
        except Exception as exc:  # noqa: BLE001 — 参照系坏了 = 「没有参照系」
            logger.warning("畴参照系加载失败,按「没有参照系」处理: %s", exc)
            ref = None
        verdict = classify(fp, ref)

        data = {
            "scan_path": path,
            "verdict": verdict.verdict,
            "label": verdict.label,
            "verdict_reason": verdict.reason,
            "next_step": NEXT_STEP.get(verdict.reason, ""),
            "distances": {k: round(v, 6) for k, v in verdict.distances.items()},
            "coverage": {k: round(v, 4) for k, v in verdict.coverage.items()},
            "margin": (None if verdict.margin == float("inf") else verdict.margin),
            "reference_version": verdict.reference_version,
            "reference_requested": version or None,
            # ↓ 落 marker meta 的那几个键(设计 §3.4)。
            "fingerprint": [list(t) for t in fp.triples()],
            "scan_angle_deg": fp.scan_angle_deg,      # **null 表示读不到,不是 0**
            "n_peaks": fp.n_peaks,
            "nm_per_px": fp.nm_per_px,
            "scale": fp.scale,
            "periods_in_frame": fp.periods_in_frame,
            "period_fast_axis_nm": fp.period_fast_axis_nm,
            "angular_concentration": fp.angular_concentration,
            "atomic_passed": fp.atomic_passed,
            "peaks_frame_deg": [round(p.k_angle_frame_deg, 4) for p in fp.peaks],
            "reasons": list(fp.reasons),
            "warnings": list(fp.warnings),
            "notes": {k: v for k, v in fp.notes.items()
                      if isinstance(v, (int, float, str, list))},
        }

        if verdict.verdict == "mixed":
            summary = (
                f"两个畴的峰**都在这一帧里**(覆盖率 "
                f"{', '.join(f'{k} {v:.0%}' for k, v in verdict.coverage.items())})"
                f" —— 畴界就在这一帧的视野内。")
        elif verdict.verdict == "undetermined":
            why = verdict.reason or "unknown"
            summary = f"判不了({why})。{NEXT_STEP.get(why, '')}".strip()
            if fp.n_peaks:
                summary += (f" 指纹照报:{fp.n_peaks} 个方向-周期峰"
                            + (f",样品系 "
                               f"{', '.join(f'{p.k_angle_sample_deg:.1f}°/{p.period_nm:.3f}nm' for p in fp.peaks)}"
                               if fp.scan_angle_deg is not None else "(帧角未知,不可比)")
                            + "。")
        else:
            summary = (f"畴 {verdict.label}(距离 {verdict.distances[verdict.label]:.4f}"
                       f" ≤ 容差,与次近拉开 {verdict.margin:.4f})。")
        if fp.warnings:
            summary += f" 警告:{', '.join(fp.warnings)}。"

        return SkillResult(skill_name=name, success=True, data=data, summary=summary)


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return [wrap_skill(AssessDomainPhase, context_provider)]


__all__ = ["NEXT_STEP", "AssessDomainPhase"]
