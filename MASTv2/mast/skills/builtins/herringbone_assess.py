"""``AssessHerringbone`` —— 大平台上的针尖验收:Au(111) 的 22×√3 重构。

判据本体在 :func:`mast.vision.herringbone.assess_herringbone`(纯函数,零 IO,
阈值全参数化)。**这一层只做 IO 与通道/像素尺度解析,一个阈值都不判** ——
与 ``tip_spectro_assess`` 的两个技能同一个分工。

## 适用输入

台阶锐度需要图中存在台阶；大平台可改用重构条纹等适合该输入的读数。
输入不包含所需特征时，不应把无法测量当成针尖不合格。

## 三态,不是两态

技能**永远返回 success=True**(只要文件读得动):结论在 ``data.verdict`` 里,
取值 ``herringbone`` / ``absent`` / ``undetermined``。技能失败保留给「这件事没做
成」—— 文件不存在、通道缺失、依赖缺席。

``undetermined`` 与 ``absent`` 必须分开读:

* ``absent`` = 门都过了,这张图上确实没有条纹;
* ``undetermined`` = **这一帧答不了这个问题**(视野太小/像素太粗、死平帧、
  条纹平行于快扫方向)。下游该换视野或转角度重扫,**不是**去修针尖。

## 阈值:没有标定数据,所以只报数

设计前提是：终点暂时没有数据来标定,但不要紧,先验证循环可行。
所以:

* ``period_prior_nm`` 是**搜索窗的中心**,默认从当前衬底的知识库常数取
  (Au(111) 6.3 nm)。知识库的值是近似,随仪器/样品变 —— 它不是判定门槛。
* ``stripe_corrugation_pm`` / ``fft_sharpness`` / ``double_tip_score`` 一律**报数**,
  **不给 verdict**。要判「够不够好」,传 ``corrugation_min_pm``;不传就只有数。
  这一档的默认值是 ``None`` = **未标定**,不是「0 也算过」。

## 起伏这个数要跟偏压/设定点一起读

herringbone 的对比度强烈依赖偏压与设定点(知识库:300-600 mV / 0.5-1 nA)。所以
结果里带上 ``bias_v`` / ``setpoint_a`` —— 跨一次锻造循环比较起伏时,**先核对这两个
数一样**,否则比的是成像条件不是针尖。

## 双针尖:给得出数,给不出无阈值的判定

见判据模块的模块注释(那里有推导):对**纯周期**信号,双针尖卷积只是把那一个
Bragg 系数乘上一个复数 —— 谱上不产生任何新峰,任何只吃周期分量的统计量都分不开
单针尖与双针尖。带信息的是**非周期内容**(elbow / 缺陷 / 台阶),所以这里报的是
``mast.vision.double_tip.detect_double_tip`` 的原始读数,**外加它能不能说话的前提**
``aperiodic_fraction``。后者小的时候,``double_tip_detected=False`` 的意思是
**「没有证据」**而不是「针尖没问题」。
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

_NAME = "AssessHerringbone"


def _resolve_prior(explicit, substrate_param):
    """``(周期先验 nm, 衬底事实, 来源, 错误信息)``。

    显式值 > 当前样品的衬底常数。两条都不中就**拒绝执行并说清楚** —— 与
    ``tip_spectro_assess._resolve_expected`` 同一条纪律:绝不默认按 Au(111) 处理。
    """
    from mast.core.sample_facts import resolve_substrate

    facts = resolve_substrate(substrate_param or None)
    if explicit is not None and float(explicit) > 0:
        return float(explicit), facts, "explicit", ""
    if not facts.available:
        return None, facts, "unknown", (
            f"没给周期先验，也推断不出衬底：{facts.reason}")
    if facts.reconstruction_period_nm is None:
        return None, facts, "unknown", (
            f"{facts.material} 在知识库里没有表面重构的条纹周期 —— "
            f"这个面本来就没有 herringbone 这类重构，不是数据缺失。"
            f"用别的验收判据（台阶锐度 AssessTipSharpness / 原子相 "
            f"AssessAtomicPhase），或显式给一个 period_prior_nm。")
    return float(facts.reconstruction_period_nm), facts, "substrate", ""


class AssessHerringbone(BaseSkill):
    """一帧扫描图上有没有 Au(111) 的 herringbone 重构，以及针尖把它画得多好。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "判定一帧已保存的 .sxm 上有没有 Au(111) 的 22x-sqrt3 "
                "**herringbone** 重构,并报出与之配套的针尖质量读数(条纹起伏,"
                "单位 pm;FFT 锐度;正反扫不稳定度;双针尖读数)。只读。"
                "这是在**大片平坦台面**上锻出来的针尖的验收判据 —— 在那种表面上,"
                "台阶边缘判据(AssessTipSharpness)根本没有输入。verdict 取 "
                "'herringbone' / 'absent' / 'undetermined' 之一。'undetermined' "
                "的意思是**这一帧答不了**(视野太小或像素太粗、死平帧、或者条纹"
                "平行于快扫方向 —— 那时它与扫描行伪迹分不开)—— 换一个视野、或者"
                "把扫描框转个角度重扫;**不要**把它读成「针尖不行」。"
                "阈值是**未标定的(UNCALIBRATED)**:除非你传了 corrugation_min_pm,"
                "否则针尖质量读数只报数、不给判定。"
            ),
            parameters=[
                ParameterSpec(name="scan_path", type="str",
                              description=".sxm 帧的路径。",
                              required=True),
                ParameterSpec(name="channel", type="str",
                              description="形貌通道('Z' 是标准选择)。",
                              required=False, default="Z"),
                ParameterSpec(
                    name="substrate", type="str",
                    description=("衬底名(例如 'Au(111)')。"
                                 "留空则从当前样品记录里推断。"),
                    required=False, default=""),
                ParameterSpec(
                    name="period_prior_nm", type="float", unit="nm",
                    description=("预期的孤子对条纹周期。这是**搜索窗的中心**,"
                                 "不是一条及格线。留空则取自衬底"
                                 "(Au(111):6.3 nm)。"),
                    required=False, min_value=0.5, max_value=200.0),
                ParameterSpec(
                    name="snr_min", type="float",
                    description=("带内谱峰信噪比的下限。沿用 seg_scale_adaptive "
                                 "调好的 lat_snr。"),
                    required=False, default=4.0, min_value=1.0, max_value=1e6),
                ParameterSpec(
                    name="concentration_min", type="float",
                    description=("角向集中度的下限 —— 分的是「离散的条纹峰」还是"
                                 "「一圈弥散的环」。在合成物理数据上实测到的分离度:"
                                 "1:1 信噪比下的 herringbone 是 142+,而经过带通的"
                                 "针尖振铃最高只到 13.6。"),
                    required=False, default=20.0, min_value=1.0, max_value=1e9),
                ParameterSpec(
                    name="corrugation_min_pm", type="float", unit="pm",
                    description=("条纹起伏(peak-to-peak)达到或超过这个值,针尖"
                                 "才算好。**未标定(UNCALIBRATED)** —— 留空则"
                                 "只报数、不给判定;在拿到真实数据把它标出来之前,"
                                 "这才是诚实的缺省。"),
                    required=False, min_value=0.0, max_value=10000.0),
                ParameterSpec(
                    name="allow_reduced_scale", type="bool",
                    description=("在降级尺度带里也接受正判定(每个条纹周期 "
                                 "3-6 px,或者整帧里只有 4-6 个周期)。"),
                    required=False, default=False),
            ],
            estimated_duration_s=4.0,
            composition_level=1,
            tags=["tip", "herringbone", "reconstruction", "au111", "fft",
                  "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        path = str(params["scan_path"])
        channel_name = params.get("channel") or "Z"
        if not Path(path).exists():
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"文件不存在: {path}")

        prior, facts, prior_source, why = _resolve_prior(
            params.get("period_prior_nm"), params.get("substrate"))
        sub = {
            "substrate": facts.material,
            "substrate_source": facts.source,
            "substrate_available": facts.available,
            "period_prior_source": prior_source,
        }
        if prior is None:
            return SkillResult(skill_name=_NAME, success=False, error=why,
                               data=sub)

        try:
            import numpy as np

            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
            from mast.vision.herringbone import assess_herringbone
            from mast.vision.tip_metrics import _fwd_bwd_instability
        except ImportError as exc:
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"缺依赖: {exc}")

        try:
            scan = read_sxm(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=_NAME, success=False,
                               error=f".sxm 读取失败: {exc}")

        # 走 sxm_oriented_frames 而不是自己扒 channels:它把 backward 的镜像
        # (fliplr)与 :SCAN_DIR: up 的上下翻转都处理掉了。**正反扫对比缺了这一步
        # 就是拿一张图跟它自己的镜像比**,算出来的数没有意义。
        fr = sxm_oriented_frames(scan, channel_name)
        arr = fr.get("forward")
        if arr is None:
            return SkillResult(
                skill_name=_NAME, success=False,
                error=(f"文件里没有可用通道/正反扫数据"
                       f"(要的是 {channel_name!r})"))
        arr = np.asarray(arr, dtype=np.float64)
        bwd = fr.get("backward")
        bwd_arr = (np.asarray(bwd, dtype=np.float64)
                   if bwd is not None and np.asarray(bwd).shape == arr.shape
                   else None)

        nm_per_px = fr.get("nm_per_px")
        warn_extra: list[str] = []
        # 非方像素:判据里的环(angular_concentration)按方帧算半径,x/y 标度不同
        # 会把环拉成椭圆。Nanonis 默认是方的;不是的话要说出来,不要闷头算。
        w_nm, h_nm = fr.get("width_nm"), fr.get("height_nm")
        if w_nm and h_nm and arr.ndim == 2 and arr.shape[0] and arr.shape[1]:
            nmpp_x = float(w_nm) / arr.shape[1]
            nmpp_y = float(h_nm) / arr.shape[0]
            if nmpp_x > 0 and abs(nmpp_y / nmpp_x - 1.0) > 0.05:
                warn_extra.append("non_square_pixels")

        res = assess_herringbone(
            arr,
            nm_per_px=(float(nm_per_px) if nm_per_px else None),
            period_prior_nm=float(prior),
            snr_min=float(params.get("snr_min", 4.0)),
            concentration_min=float(params.get("concentration_min", 20.0)),
            allow_reduced_scale=bool(params.get("allow_reduced_scale", False)),
        )

        instab = None
        if bwd_arr is not None:
            try:
                instab = float(_fwd_bwd_instability(arr, bwd_arr))
            except Exception as exc:  # noqa: BLE001 — 附加读数坏了不该让判据失败
                logger.debug("正反扫稳定度算不出来: %s", exc)

        data = {
            "scan_path": path,
            "channel": fr.get("channel"),
            "verdict": res.verdict,
            "passed": res.passed,
            # ── 尺度与几何 ──
            "scale": res.scale,
            "nm_per_px": res.nm_per_px,
            "px_per_period": res.px_per_period,
            "periods_in_frame": res.periods_in_frame,
            "frame_nm": res.frame_nm,
            "rows": int(arr.shape[0]) if arr.ndim == 2 else None,
            "cols": int(arr.shape[1]) if arr.ndim == 2 else None,
            # ── 条纹 ──
            "period_nm": res.period_nm,
            "period_prior_nm": res.period_prior_nm,
            "period_band_nm": (list(res.period_band_nm)
                               if res.period_band_nm else None),
            "snr": res.snr,
            "angular_concentration": res.angular_concentration,
            "k_angle_deg": res.k_angle_deg,
            "stripe_angle_deg": res.stripe_angle_deg,
            # ── 针尖质量读数(报数,不判) ──
            "stripe_corrugation_pm": res.stripe_corrugation_pm,
            # 读取起伏前同时检查象限间离散度；局部大特征可能主导整帧统计，不能据此当成均匀条纹起伏。
            "corrugation_quadrant_spread": res.corrugation_quadrant_spread,
            "corrugation_rms_m": res.corrugation_rms_m,
            "fft_sharpness": res.fft_sharpness,
            "second_harmonic_ratio": res.second_harmonic_ratio,
            "fwd_bwd_instability": instab,
            # ── 双针尖(条件成立才有意义) ──
            "double_tip_detected": res.double_tip_detected,
            "double_tip_score": res.double_tip_score,
            "double_tip_threshold": res.double_tip_threshold,
            "double_tip_separation_nm": res.double_tip_separation_nm,
            "aperiodic_fraction": res.aperiodic_fraction,
            # ── 成像条件:起伏要跟它们一起读 ──
            "bias_v": fr.get("bias_v"),
            "setpoint_a": fr.get("setpoint_a"),
            "rec_time": fr.get("rec_time"),
            # ── zigzag 没查,以及要多大的帧才查得动 ──
            "chevron_prior_nm": res.chevron_prior_nm,
            "chevron_min_frame_nm": res.chevron_min_frame_nm,
            # ⚠️ row_free_band_snr **不带方向**（高既可能是划痕也可能只是鱼骨强）。
            # 要问「有没有慢轴划痕」看 slow_axis_power_ratio。
            "row_free_band_snr": res.row_free_band_snr,
            "slow_axis_power_ratio": res.slow_axis_power_ratio,
            "reasons": list(res.reasons),
            "warnings": list(res.warnings) + warn_extra,
            "notes": dict(res.notes),
            **sub,
        }

        thr = params.get("corrugation_min_pm")
        corr = res.stripe_corrugation_pm
        if res.verdict != "herringbone" or corr is None:
            data["tip_verdict"] = "not_measurable"
        elif thr is None:
            # 阈值未标定 —— 报数,不下判定。编一个数会让流程以为自己验收过了。
            data["tip_verdict"] = "measured"
        else:
            data["corrugation_min_pm"] = float(thr)
            data["tip_verdict"] = ("good" if float(corr) >= float(thr)
                                   else "weak")

        data["summary_verdict"] = res.verdict
        return SkillResult(skill_name=_NAME, success=True, data=data,
                           summary=_summarize(res, instab, data))


def _summarize(res, instab, data) -> str:
    """给用户/模型读的一句话。**「判不了」与「没有」必须是两句不同的话。**"""
    if res.verdict == "undetermined":
        why = ", ".join(res.reasons) or "判不了"
        head = f"这一帧判不了 herringbone（{why}）—— 这不等于「没有 herringbone」，"
        if "scale_gate" in res.reasons:
            head += (f"也不是针尖的问题：视野 {res.frame_nm:.0f} nm / "
                     f"{res.nm_per_px:.4f} nm/px 下，每个周期只有 "
                     f"{res.px_per_period:.1f} px、帧里只有 "
                     f"{res.periods_in_frame:.1f} 个周期。"
                     f"要 6 px/周期且 6 个周期以上才判得动。")
        elif "stripes_along_fast_axis" in res.reasons:
            head += res.notes.get("slow_axis", "条纹平行于快扫方向。")
        elif "frame_unusable" in res.reasons:
            head += res.notes.get("frame_unusable", "")
        else:
            head += "换一张再判；不要据此去修针尖。"
        return head

    if res.verdict == "absent":
        why = ", ".join(res.reasons)
        s = f"没有找到 herringbone 条纹（{why}）"
        if "row_alignment_artifact" in res.reasons:
            s += "；" + res.notes.get("row_alignment_artifact", "")
        return s + "。注意：这是「这张图上没有」，不直接等于「针尖不行」。"

    corr = res.stripe_corrugation_pm
    s = (f"检出 herringbone：周期 {res.period_nm:.2f} nm"
         f"（先验 {res.period_prior_nm:.2f} nm），条纹走向 "
         f"{res.stripe_angle_deg:.0f}°，角向集中度 "
         f"{res.angular_concentration:.0f}。")
    if corr is not None:
        s += f"条纹起伏（峰峰）{corr:.1f} pm"
        tv = data.get("tip_verdict")
        if tv == "measured":
            s += "（未给判定阈值 —— 阈值尚未标定，这里只报数）。"
        else:
            s += (f"，阈值 {data.get('corrugation_min_pm'):.1f} pm ⇒ "
                  f"{'够好' if tv == 'good' else '偏弱'}。")
    if instab is not None:
        s += f" 正反扫不稳定度 {instab:.2f}。"
    if res.double_tip_score is not None:
        s += (f" 双针尖读数 {res.double_tip_score:.2f}"
              f"（阈值 {res.double_tip_threshold:.2f}）"
              f"，本帧非周期内容占比 {res.aperiodic_fraction:.2f} —— "
              f"这个占比越小，「没检出重影」越接近「没有证据」而不是"
              f"「没有双针尖」。")
    if "chevron_not_resolvable" in res.warnings:
        s += (f" 注意：zigzag 本身没查 —— 要看清全周期 "
              f"{res.chevron_prior_nm:.0f} nm 的折线得有 "
              f"≥{res.chevron_min_frame_nm:.0f} nm 的帧。")
    return s


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(AssessHerringbone, context_provider)


__all__ = ["AssessHerringbone"]
