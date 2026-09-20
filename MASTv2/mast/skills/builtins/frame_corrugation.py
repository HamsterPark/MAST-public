"""``AssessFrameCorrugation`` —— 判据②的技能壳:一帧的起伏算不算「极大」。

设计文档:``docs/v2/design/`` 的「S1 修针循环 conduct 化」设计 §3.2
(文件名按「S1 修针循环」检索;通用层注释不写样品名前缀)。判据本体在
:func:`mast.vision.corrugation_gate.judge_corrugation`(纯函数、零 IO)。
**这一层只做 IO 与阈值取用,一个判据逻辑都不写** —— 与 ``AssessHerringbone`` /
``tip_spectro_assess`` 同一个分工。

## 它出的是观察,不是结论

返回的 ``verdict`` 只有 ``high`` / ``normal`` / ``low`` / ``undecidable``,
**没有 `bad_tip`**:一片台阶簇会给出与坏针团簇同样高的起伏,单帧分不开。
「是针还是表面」由换位置复测的跨点聚合
(:func:`mast.conduct.cross_check.aggregate_cross_points`)回答。

## 阈值与它的视野是一组,而且**不许混着来**

``threshold_pm`` 与 ``ref_scan_nm`` 要么**都不给**(走 profile 那一对),要么
**一起给**。只给一个 ⇒ ``undecidable``,而**不会**拿 profile 的另一半来补 ——
一个显式阈值配上 profile 声明的视野,量的就不是同一件事了(设计 §4 陷阱 6)。

## ``threshold_pm`` 没有 default,这一条是硬的

``ParameterSpec`` 若给它一个 default,pydantic 会在调用方**没传**时替它填上那个
数,于是调用方未配置时无法到达 profile 查询。测试必须经过 schema、参数转换和
execute 的生产入口，不能只传手工 dict 而绕过默认值注入。
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

_NAME = "AssessFrameCorrugation"


def resolve_threshold_pair(explicit_pm, explicit_ref_nm, thresholds):
    """``(threshold_pm, ref_scan_nm, source)`` —— 阈值和它的视野**同源**。

    * 两个都没显式给 ⇒ 走 profile 的那一对;
    * 给了任何一个 ⇒ 用显式的那一对,**缺的那半留 None**(⇒ 判不了)。
      绝不用 profile 去补另一半:那会把两次不同标定的数拼成一个判据。

    ⚠️ 这个函数是「没传 threshold_pm 时到底查不查 profile」的那一行。
    ``threshold_pm`` 一旦有了 ParameterSpec default,``explicit_pm`` 就永远不是
    ``None``,下面这个分支永远到不了 —— 测试从生产入口进就是为了盯住它。
    """
    if explicit_pm is None and explicit_ref_nm is None:
        return (getattr(thresholds, "corrugation_high_pm", None),
                getattr(thresholds, "corrugation_ref_scan_nm", None),
                f"profile:{getattr(thresholds, 'name', '') or '?'}")
    pm = None if explicit_pm is None else float(explicit_pm)
    ref = None if explicit_ref_nm is None else float(explicit_ref_nm)
    return pm, ref, "explicit"


class AssessFrameCorrugation(BaseSkill):
    """一帧扫描图的起伏有多大 —— 判据②的观察,不是针尖结论。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "量一帧已保存的 .sxm 里表面起伏有多大,再拿它跟样品 profile 的上限"
                "比。只读文件,不碰硬件。verdict 取 'high' / 'normal' / 'low' / "
                "'undecidable' 之一,它是一个**观察**,**不是**针尖结论:一簇台阶给出"
                "的起伏和一根坏针尖一样大,所以单帧分不开这两者。到底是**针尖**还是"
                "**表面**,要靠换到别的位置复测再聚合来回答(aggregate_cross_points)。"
                "'undecidable' 的意思是**这一帧答不了** —— 帧不可用、它的视野与阈值"
                "标定时的视野对不上(**从来不做**跨尺度换算)、或者根本还没有标定过"
                "阈值。'low' 的意思是这一帧太平,连正反扫判据也承载不了,于是判据"
                "**弃权**。阈值与它标定时所用的视野是**一组**:要么都传,要么都不传。"
            ),
            parameters=[
                ParameterSpec(name="scan_path", type="str",
                              description=".sxm 帧的路径。",
                              required=True),
                ParameterSpec(name="channel", type="str",
                              description="形貌通道('Z' 是标准选择)。",
                              required=False, default="Z"),
                ParameterSpec(
                    name="profile", type="str",
                    description=("样品阈值 profile 的名字。"
                                 "留空就用当前生效的那一个。"),
                    required=False, default=""),
                # ⚠️ 这两个**刻意不写 `unit=`**。带 unit 的 float 会被
                # `_si_params` 当成「模型用文本写的有量纲量」,schema 里变成 str,
                # 而 `parse_quantity` 会把 "40p" 读成 4e-11 —— 对一个**已经以 pm
                # 计**的参数来说那是差 1e12 的静默误读,而它直接改判决。
                # 单位写在名字里(_pm / _nm)与描述里,值就是普通浮点数。
                ParameterSpec(
                    name="threshold_pm", type="float",
                    description=(
                        "起伏上限,单位**皮米**(就写一个普通数字,例如 40 表示 "
                        "40 pm)。留空则从样品 profile 里取 —— 这里**刻意没有 "
                        "default**:没传的阈值必须保持「没传」,查 profile 那一行"
                        "才到得了。你要是传了这个数,就**必须(MUST)**同时把 "
                        "ref_scan_nm 也传上;一个不带标定视野的阈值毫无意义,"
                        "只会得到 'undecidable'。"),
                    required=False, min_value=0.1, max_value=1e6),
                ParameterSpec(
                    name="ref_scan_nm", type="float",
                    description=(
                        "threshold_pm 是在多大的扫描尺寸上标定的,单位**纳米**"
                        "(就写一个普通数字,例如 100 表示 100 nm)。只有和 "
                        "threshold_pm 一起给才有意义。这一帧自己的扫描尺寸与它"
                        "相差超过容差时,判定就是 'undecidable' —— 起伏"
                        "**绝不(NEVER)**跨视野换算。"),
                    required=False, min_value=0.1, max_value=1e5),
                ParameterSpec(
                    name="rel_tol", type="float",
                    description=("拿这一帧的扫描尺寸去对 ref_scan_nm 时的相对"
                                 "容差。0.05 足以覆盖用户输入 100 nm 而实际"
                                 "得到 99.98 的情况。"),
                    required=False, default=0.05,
                    min_value=0.0, max_value=0.5),
            ],
            estimated_duration_s=3.0,
            composition_level=2,
            tags=["tip", "corrugation", "surface", "analysis", "read", "scan"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        path = str(params.get("scan_path") or "")
        channel_name = params.get("channel") or "Z"
        if not path or not Path(path).exists():
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"文件不存在: {path}")

        try:
            import numpy as np

            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
            from mast.vision.corrugation_gate import judge_corrugation
            from mast.vision.frame_validity import judge_frame
            from mast.vision.scan_prep_thresholds import resolve
        except ImportError as exc:
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"缺依赖: {exc}")

        try:
            scan = read_sxm(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=_NAME, success=False,
                               error=f".sxm 读取失败: {exc}")

        # 几何归位的单一真源:backward 的 fliplr 与 :SCAN_DIR: up 的上下翻转都在
        # 这里处理。自己扒 channels 就是在拿一张图跟它自己的镜像比。
        fr = sxm_oriented_frames(scan, channel_name)
        arr = fr.get("forward")
        if arr is None:
            return SkillResult(
                skill_name=_NAME, success=False,
                error=f"文件里没有可用通道/正反扫数据(要的是 {channel_name!r})")
        arr = np.asarray(arr, dtype=np.float64)

        th = resolve(params.get("profile") or None)
        thr_pm, ref_nm, thr_source = resolve_threshold_pair(
            params.get("threshold_pm"), params.get("ref_scan_nm"), th)

        width_nm = fr.get("width_nm")
        height_nm = fr.get("height_nm")
        warnings: list[str] = []
        if (width_nm and height_nm
                and abs(float(height_nm) / float(width_nm) - 1.0) > 0.05):
            # 非方帧:尺度对账用的是宽度,而起伏是整帧的。说出来,不闷头算。
            warnings.append("non_square_frame")

        # `or 0.05` 会把一个合法的 0.0(要求精确同视野)悄悄换成 0.05 —— 那是
        # 「假值当缺席」那类 bug,这里只认 None 为「没给」。
        rel_tol = params.get("rel_tol")
        rel_tol = 0.05 if rel_tol is None else float(rel_tol)

        verdict = judge_frame(arr)
        res = judge_corrugation(
            verdict,
            threshold_pm=thr_pm,
            ref_scan_nm=ref_nm,
            this_scan_nm=width_nm,
            rel_tol=rel_tol,
            profile_name=str(getattr(th, "name", "") or ""),
            provenance=str(getattr(th, "provenance", "") or ""))

        # ── 旁证:不参与判决,但读者要能看见这一帧长什么样 ────────────────
        finite = np.isfinite(arr)
        nan_frac = float(1.0 - (finite.sum() / arr.size)) if arr.size else None
        bad_row_frac = None
        rows_used = None
        try:
            from mast.vision.frame_validity import acquired_row_mask
            from mast.vision.scan_artifacts import detect_scan_artifacts

            bwd = fr.get("backward")
            bwd_arr = (np.asarray(bwd, dtype=np.float64)
                       if bwd is not None and np.asarray(bwd).shape == arr.shape
                       else None)
            # 只拿**扫完的行**:Nanonis 把没扫到的行填 NaN,而任何平面/直线拟合
            # 碰上一个 NaN 就整幅返回 NaN(``detect_scan_artifacts`` 会直接抛)。
            # 半张图是输入,不是错误 —— 裁行用全仓那一份判据,不自己写第二份。
            mask = (acquired_row_mask(arr) if bwd_arr is None
                    else acquired_row_mask(arr, bwd_arr))
            rows_used = int(mask.sum())
            if rows_used >= 2:
                sub = arr[:mask.size][mask]
                sub_bwd = (bwd_arr[:mask.size][mask]
                           if bwd_arr is not None else None)
                bad_row_frac = float(
                    detect_scan_artifacts(sub, sub_bwd).bad_row_frac)
        except Exception as exc:  # noqa: BLE001 — 旁证坏了不该让判据失败
            logger.debug("坏行占比算不出来: %s", exc)

        data = dict(res.as_dict())
        data.update({
            "scan_path": path,
            "channel": fr.get("channel"),
            "threshold_source": thr_source,
            "frame_usable": bool(verdict.usable),
            "unusable_reason": ("" if verdict.usable else str(verdict.reason)),
            "nan_frac": nan_frac,
            "bad_row_frac": bad_row_frac,
            # 坏行占比是在**哪些行**上算的 —— 裁行是一种预处理,它会改这个数。
            "bad_row_rows_used": rows_used,
            "rows": int(arr.shape[0]) if arr.ndim == 2 else None,
            "cols": int(arr.shape[1]) if arr.ndim == 2 else None,
            "nm_per_px": fr.get("nm_per_px"),
            "width_nm": width_nm,
            "height_nm": height_nm,
            # 起伏要跟成像条件一起读:偏压/设定点变了,比的就不是针尖。
            "bias_v": fr.get("bias_v"),
            "setpoint_a": fr.get("setpoint_a"),
            "rec_time": fr.get("rec_time"),
            "warnings": warnings,
        })
        return SkillResult(skill_name=_NAME, success=True, data=data,
                           summary=_summarize(res, data))


def _summarize(res, data) -> str:
    """给用户/模型读的一句话。**「判不了」与「判出来不好」是两句不同的话。**"""
    head = {
        "undecidable": "这一帧的起伏判不了",
        "low": "这一帧起伏太小,判据弃权",
        "high": "这一帧起伏偏大",
        "normal": "这一帧起伏正常",
    }.get(res.verdict, res.verdict)
    s = f"{head} —— {res.reason}"
    if res.verdict == "high":
        s += ("这不是「针坏了」的结论:要判针还是表面,得换位置复测再聚合"
              "(判据③)。")
    if data.get("threshold_source", "").startswith("profile"):
        s += f" 阈值来源:{data['threshold_source']}。"
    elif res.threshold_pm is not None:
        s += " 阈值来源:调用方显式给的。"
    s += f" 口径 {res.detrend}+{res.statistic};{res.blind_to}"
    return s


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(AssessFrameCorrugation, context_provider)


__all__ = ["AssessFrameCorrugation", "resolve_threshold_pair"]
