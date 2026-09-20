"""用独立的形状特征联合判断团簇圆度。

像素边计数的 4πA/P² 存在方向性偏差：轴对齐正方形为 π/4，
而数字化圆盘受栅格边界影响，旧标度不能直接用作几何圆度。
当前 equivalent_axis_ratio 表示径向离散等效的椭圆轴比，
aspect_ratio 检查整体拉长；两项必须同时满足阈值，一项高分不能补另一项低分。
实现与几何推导见 vision.roundness。

小团簇受像素化限制，is_round 可为 None，并附 roundness_undecidable 理由；
无法判定不等于不圆，也不能当成通过。

分割仍独立于 ExtractClusters：此处使用平面处理后的 mean/std，
后者默认 RAW 下的 median/scaled-MAD，因此同一帧可能得到不同连通域。
比较判据结果时必须先确认分割与预处理口径一致。
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


#: 多针尖判据的两个数。**都不是拟合出来的**:
#: 25% —— 一个顶点扎出 500 px、另一个 5 px,后者是碎屑不是顶点;
#: 6 nm —— 针尖顶点间距只有几 nm。放宽到不限:误杀 3→4 而只多抓 1 张;
#:         收到 4 nm:只抓到 7/18。**6 nm 是这个权衡的拐点,不是最优值**。
#: 出处与成绩见 ``execute`` 里 multi_tip 那一段。
MULTI_TIP_SIZE_FRAC = 0.25
MULTI_TIP_SPAN_NM = 6.0


class AssessClusterRoundness(BaseSkill):
    """Compute roundness of the largest protrusion in a small .sxm scan."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessClusterRoundness",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "把一张小 .sxm 里扎出来的那个团簇坑分割出来，说出它有多圆。"
                "主输出是 equivalent_axis_ratio ∈ (0,1]：「这个团簇的不规则"
                "程度，相当于一个短轴/长轴等于该比值的椭圆」—— 1.0 = 完美圆盘；"
                "由于像素化本底已经被减掉，它在任何团簇尺寸上含义都一样。"
                "is_round 是一个**合取**（CONJUNCTION：equivalent_axis_ratio "
                "与二阶矩长宽比两条都要过），团簇小到判不了时它是 null —— "
                "**不是 false**。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path",
                    type="str",
                    description="那张团簇坑 .sxm 扫描图的路径。",
                    required=True,
                ),
                ParameterSpec(
                    name="threshold_sigma",
                    type="float",
                    description=(
                        "一个像素要高出局部均值多少个标准差，才算进团簇。"
                        "越大越严。"
                    ),
                    required=False,
                    default=1.5,
                    min_value=0.5,
                    max_value=5.0,
                ),
                ParameterSpec(
                    name="polarity",
                    type="str",
                    description=(
                        "'bright' 对应向上的凸包，'dark' 对应凹坑，'auto' 选择两个极端中更大的一侧。团簇占据多数像素时，auto 可能选中背景；评估向上凸起的目标时应显式选择 'bright'。"
                    ),
                    required=False,
                    default="auto",
                ),
                ParameterSpec(
                    name="threshold_mode",
                    type="str",
                    description=(
                        "'sigma'(出厂,向后兼容)= mean + threshold_sigma×std ——"
                        "**自指**:簇越大 std 越大、阈值越高、切掉的簇越多。"
                        "'physical' = 背景**众数** + physical_threshold_pm,"
                        "用外部高度标尺；sigma 模式的阈值则会随图像方差变化。"
                        "应根据目标特征和背景验证阈值，不能仅比较切出的面积。"
                    ),
                    required=False,
                    default="sigma",
                ),
                ParameterSpec(
                    name="physical_threshold_pm",
                    type="float",
                    description=(
                        "threshold_mode='physical' 时,高出背景多少算团簇(pm)。"
                        "出厂 117.7 = **半个 Au(111) 单原子台阶**(235.4 pm = a/√3)。"
                        "这是外部标尺不是拟合值 —— 换材料要换这个数。"
                    ),
                    required=False,
                    default=117.7,
                    min_value=1.0,
                    max_value=5000.0,
                ),
                ParameterSpec(
                    name="shape_mode",
                    type="str",
                    description=(
                        "'boundary'(出厂)= 二值化后量边界像素到质心的径向离散 ——"
                        "**阈值以上每一点的高度全被丢掉**,结果由阈值切在哪儿决定。"
                        "'weighted' = 高度加权二阶矩,每点按高出背景多少计权。"
                        "阈值挪 ±20% 时轴比变化:boundary 0.0532 / weighted 0.0125;"
                        "在用户标「阈值不够低」的 13 张上 0.2451 / 0.0084(稳 29 倍)。"
                        "两个数**永远都报**(weighted_axis_ratio / boundary_axis_ratio),"
                        "这个开关只决定谁驱动 is_round。"
                    ),
                    required=False,
                    default="boundary",
                ),
                ParameterSpec(
                    name="channel",
                    type="str",
                    description="通道名（默认 'Z'）。",
                    required=False,
                    default="Z",
                ),
                ParameterSpec(
                    name="select",
                    type="str",
                    description=(
                        "一帧里有好几坨时，评的是哪一坨："
                        "'largest'（默认）或 'center' —— 后者是离帧中心最近的"
                        "那一坨，也就是调用方刚刚在它把这次扫描对准的那个点上"
                        "扎出来的坑。一轮修针会留下好几个坑，"
                        "而最大的那个往往是更早留下的。"
                    ),
                    required=False,
                    default="largest",
                    allowed_values=["largest", "center"],
                ),
                ParameterSpec(
                    name="min_axis_ratio",
                    type="float",
                    description=(
                        "圆到什么程度才算够圆，用**等效轴比**"
                        "（EQUIVALENT AXIS RATIO）表述：0.75 的意思是"
                        "「不比一个长短轴相差 25% 的椭圆更不规则」。它是从边界的 "
                        "r(theta) 相对离散读出来的（已减掉像素化本底），"
                        "所以在任何团簇尺寸上含义都一样。"
                        "它**取代** round_threshold=0.65 —— 那个阈值所在的标度上，"
                        "任何圆都够不着（见模块 docstring）；"
                        "两个数**不可比**。"
                    ),
                    required=False,
                    default=0.75,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ParameterSpec(
                    name="min_aspect",
                    type="float",
                    description=(
                        "二阶矩长宽比下限，用于检查整体拉长；径向离散补充检查瓣状和缺口等形状。两条判据必须同时通过，一项高分不能补偿另一项低分。"
                    ),
                    required=False,
                    default=0.6,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=2,
            tags=["scan", "analysis", "cluster", "roundness", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        scan_path = params["scan_path"]
        threshold_sigma = float(params.get("threshold_sigma", 1.5))
        polarity = (params.get("polarity") or "auto").lower()
        channel_name = params.get("channel", "Z")
        threshold_mode = (params.get("threshold_mode") or "sigma").lower()
        shape_mode = (params.get("shape_mode") or "boundary").lower()
        #: 半个 Au(111) 单原子台阶(235.4 pm = a/√3)。**外部标尺**,不是拟合出来的。
        physical_threshold_m = float(
            params.get("physical_threshold_pm", 117.7)) * 1e-12
        if threshold_mode not in ("sigma", "physical"):
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=f"threshold_mode 只能是 sigma / physical,收到 {threshold_mode!r}")
        if shape_mode not in ("boundary", "weighted"):
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=f"shape_mode 只能是 boundary / weighted,收到 {shape_mode!r}")

        # ── `round_threshold` **作废**,而且不做静默别名 ──────────────────
        #
        # 旧参数比的是 `0.6*circularity + 0.4*aspect`,那个标度上**完美圆盘最高只有
        # 0.770、轴对齐正方形能拿 0.871**,而阈值是 0.65。新参数比的是等效轴比,
        # 完美圆盘 = 1.0。两个数**方向相同、量纲不同** —— 正因为方向相同,
        # 一个漏改的 0.65 会**照跑不误**并悄悄把闸门放宽到「长短轴差 35%」。
        # 所以这里当场报错,把处方一起给,而不是默默接受。
        if params.get("round_threshold") is not None:
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=("`round_threshold` 已作废：旧圆度组合受像素边界方向性偏差影响。请使用 min_axis_ratio 表示等效椭圆轴比；它与旧 circularity 的标度不同，不能直接复制旧阈值。"
                       ),
                data={"retired_parameter": "round_threshold",
                      "use_instead": "min_axis_ratio",
                      "suggested": {"min_axis_ratio": 0.75, "min_aspect": 0.6},
                      "why_not_a_silent_alias":
                          "两者方向相同(都是越大越圆),所以一个漏改的旧值会照跑不误"})

        min_axis_ratio = float(params.get("min_axis_ratio", 0.75))
        min_aspect = float(params.get("min_aspect", 0.6))

        if not Path(scan_path).exists():
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=f"scan_path not found: {scan_path}",
            )

        try:
            import math

            import numpy as np
            from scipy.ndimage import label, find_objects
            from mast.io.nanonis_files import read_sxm
            from mast.data.processors import plane_subtract
            from mast.vision.frame_validity import judge_frame
        except ImportError as e:
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=f"missing dependency: {e}",
            )

        try:
            scan = read_sxm(scan_path)
        except Exception as e:
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=f"failed to read .sxm: {e}",
            )

        channels = scan.get("channels", {})
        ch = channels.get(channel_name) or next(iter(channels.values()), None)
        if ch is None:
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error="no usable channel",
            )
        img = ch.get("forward")
        if img is None:
            img = ch.get("backward")
        if img is None:
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error="no forward/backward frame",
            )
        img = np.asarray(img, dtype=np.float64)

        # 标准差非零不足以证明帧具有有效形貌信息，舍入图样也可能产生伪连通域。
        # 统一使用 mast.vision.frame_validity 的前置判据。
        verdict = judge_frame(img)
        if not verdict.usable:
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=verdict.reason,
                data=verdict.meta(scan_path=scan_path,
                                  rows=int(img.shape[0]), cols=int(img.shape[1])),
            )

        try:
            leveled = plane_subtract(img)
        except Exception:
            leveled = img - np.nanmean(img)

        mean = float(np.nanmean(leveled))
        std = float(np.nanstd(leveled))
        if std == 0 or np.isnan(std):
            # 留着:上面的前置管「整帧没信息」,这一条管「plane_subtract 这条路
            # 自己退化了」。两者判的不是同一件事,而这条便宜。
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error="zero/NaN std — flat or unreadable scan",
                data=verdict.meta(scan_path=scan_path),
            )

        # 若阈值由整帧 σ 推导，强目标本身也会抬高阈值，产生自我影响。
        # 应区分背景估计与显式物理阈值，并报告实际采用的分割口径；默认参数未作样品标定。
        base = None
        if threshold_mode == "physical":
            from mast.vision.roundness import background_level

            base = background_level(leveled)
            if base is None:
                return SkillResult(
                    skill_name="AssessClusterRoundness", success=False,
                    error="physical threshold needs a background mode — frame too small",
                    data=verdict.meta(scan_path=scan_path),
                )
            delta = physical_threshold_m
            up_mask = leveled > base + delta
            dn_mask = leveled < base - delta
        else:
            up_mask = leveled > mean + threshold_sigma * std
            dn_mask = leveled < mean - threshold_sigma * std

        # polarity='auto' picks whichever extreme is bigger.
        if polarity == "auto":
            mask = up_mask if up_mask.sum() >= dn_mask.sum() else dn_mask
            chosen = "bright" if up_mask.sum() >= dn_mask.sum() else "dark"
        elif polarity == "dark":
            mask, chosen = dn_mask, "dark"
        else:
            mask, chosen = up_mask, "bright"

        if not mask.any():
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error=(
                    "no pixels exceeded threshold — lower threshold_sigma or "
                    "check the scan window covers the crater"
                ),
                data={"polarity_used": chosen, "mean": mean, "std": std},
            )

        labelled, n_components = label(mask)
        if n_components == 0:
            return SkillResult(
                skill_name="AssessClusterRoundness", success=False,
                error="segmentation produced 0 components",
            )
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0  # background
        # Select the intended component explicitly. A centered scan can use the
        # nearest-center component to avoid grading an older, larger feature.
        # Keep largest as the compatibility default; callers that know the target
        # location opt into center selection.
        select = str(params.get("select", "largest") or "largest").lower()
        if select == "center" and n_components > 1:
            cy0, cx0 = (mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0
            best, best_d = 0, float("inf")
            for lab in range(1, n_components + 1):
                if sizes[lab] <= 0:
                    continue
                ys_l, xs_l = np.where(labelled == lab)
                d = float((ys_l.mean() - cy0) ** 2 + (xs_l.mean() - cx0) ** 2)
                if d < best_d:
                    best, best_d = lab, d
            chosen_label = best or int(sizes.argmax())
        else:
            chosen_label = int(sizes.argmax())
        blob = labelled == chosen_label
        area_px = int(blob.sum())

        # 圆度:**共用的那一份**(`mast.vision.roundness`),与 ExtractClusters
        # 同一个表达式、同一个标度。2026-08-11 之前这里有第二份 4πA/P²,
        # 而 ExtractClusters 里那份的口径在 08-10 已经改过 —— **同一个词、两个数**。
        from mast.vision.roundness import assess_mask

        rnd = assess_mask(blob)

        # Major/minor axes = **全长**(等效椭圆的 4σ),经**共用的那一份**。
        #
        # 2026-08-11:这里原来有第二份拷贝,和 `ExtractClusters._axes` 犯同一个错
        # (报 σ 当轴长,尺寸低报 4 倍)。两份同样的错误要修两次,而其中一份总会
        # 被漏掉 —— 所以顺手合并成一份,别只把数字改对。
        from mast.skills.builtins.cluster_extract import ExtractClusters

        ys, xs = np.where(blob)
        major, minor, aspect_ratio = ExtractClusters._axes(xs, ys, np)

        # 拉长与边界不规则是互补特征，使用合取避免加权平均掩盖其中一项失败。
        # 加权矩用于降低分割阈值对二阶矩的影响，但不能替代对分割结果的检查。
        from mast.vision.roundness import background_level, weighted_axis_ratio

        w_base = base if base is not None else background_level(leveled)
        w_axis = (weighted_axis_ratio(leveled, blob, w_base)
                  if w_base is not None else None)
        if chosen == "dark" and w_base is not None:
            # 暗侧:把高度翻过来再加权,否则权重全被 clip 成 0。
            w_axis = weighted_axis_ratio(2.0 * w_base - leveled, blob, w_base)

        undecidable = None if rnd.ok else rnd.reason
        axis_ratio = rnd.axis_ratio
        if shape_mode == "weighted":
            # 判决改由加权轴比驱动。它**判不了就是判不了** —— 不回退到边界离散,
            # 那会让「用了哪个算法」取决于数据,而报文只写一个 shape_mode。
            axis_ratio = w_axis
            undecidable = (None if w_axis is not None else
                           "高度加权二阶矩算不出来(有效像素不足)")
        if undecidable:
            is_round = None
        else:
            is_round = bool(axis_ratio >= min_axis_ratio
                            and aspect_ratio >= min_aspect)

        # 连通域数量或形状异常本身不能证明双针尖，既有结构和分割误差也会造成相似结果。
        # 缺少适合比较的同状态多方向图像时保留 None。面积单位需要按像素尺度换算。
        header = scan.get("header", {})
        try:
            w_m, h_m = (float(v) for v in str(header.get("scan_range", "1e-7 1e-7")).split()[:2])
            ny, nx = img.shape
            px_size_x = w_m / nx
            px_size_y = h_m / ny
        except Exception:
            px_size_x = px_size_y = float("nan")

        # 附近结构可能来自先前动作，不能仅凭目标周围出现多个相似团簇就归因于双针尖。
        # 上游应提供已用位置，避免把旧结构当成当前动作的证据。
        # 排除邻近结构也需要对照，不能只靠距离规则删去不符合预期的候选。
        # 需要同状态的多方向图像或可靠差分证据，才能进一步区分针尖效应与表面结构。
        multi_tip = None
        multi_tip_reason = None
        px_nm = float(px_size_x) * 1e9 if np.isfinite(px_size_x) else None
        if px_nm and px_nm > 0 and n_components >= 1:
            sizes = np.bincount(labelled.ravel())
            sizes[0] = 0
            biggest = int(sizes.argmax())
            keep = [k for k in range(1, n_components + 1)
                    if sizes[k] >= MULTI_TIP_SIZE_FRAC * sizes[biggest]]
            cent = {k: (float(np.where(labelled == k)[0].mean()),
                        float(np.where(labelled == k)[1].mean())) for k in keep}
            by, bx = cent[biggest]
            near = [k for k in keep
                    if math.hypot(cent[k][0] - by, cent[k][1] - bx) * px_nm
                    <= MULTI_TIP_SPAN_NM]
            multi_tip = bool(len(near) >= 2)
            if multi_tip:
                far = max(math.hypot(cent[k][0] - by, cent[k][1] - bx) * px_nm
                          for k in near)
                multi_tip_reason = None
                multi_tip_detail = (
                    f"{len(near)} 坨体量相当(≥{MULTI_TIP_SIZE_FRAC:.0%} 最大块)"
                    f"的东西,最远 {far:.1f} nm(≤{MULTI_TIP_SPAN_NM:g} nm)")
            else:
                multi_tip_detail = (
                    f"只有 1 坨够格的东西(共 {int(n_components)} 个连通域,"
                    f"其余不足最大块的 {MULTI_TIP_SIZE_FRAC:.0%} 或超出 "
                    f"{MULTI_TIP_SPAN_NM:g} nm)")
        else:
            # 像素尺寸读不到 ⇒ 间距算不出 ⇒ **判不了**,不是「没有多针尖」。
            multi_tip_detail = None
            multi_tip_reason = (
                "判不了:读不到像素物理尺寸(scan_range),算不出连通域之间的间距。")

        return SkillResult(
            skill_name="AssessClusterRoundness", success=True,
            data={
                "scan_path": scan_path,
                # 判决:三态。None = **判不了**(团簇太小),不是「不圆」。
                "is_round": is_round,
                "roundness_undecidable": undecidable,
                # 主输出:能直接读的物理量 ——「这个团簇的不规则程度相当于一个
                # 短轴/长轴 = q 的椭圆」。用户的用法是**比较相继几次哪次更圆**,
                # 所以这个连续量比 is_round 更重要。
                "equivalent_axis_ratio": axis_ratio,
                # 两个算法**都报**,并说清这一次是谁在判 —— 只报一个数而不说
                # 它从哪来,下一个人就没法对账(89 张对不上的教训)。
                "weighted_axis_ratio": w_axis,
                "boundary_axis_ratio": rnd.axis_ratio,
                "shape_mode": shape_mode,
                "threshold_mode": threshold_mode,
                "background_level_m": w_base,
                "threshold_above_background_m": (
                    physical_threshold_m if threshold_mode == "physical"
                    else (mean + threshold_sigma * std - w_base)
                    if w_base is not None else None),
                # 多针尖:**三态**,出厂就是判不了。见上面那段论证。
                "multi_tip": multi_tip,
                "multi_tip_undecidable": multi_tip_reason,
                "multi_tip_detail": multi_tip_detail,
                "multi_tip_size_frac": MULTI_TIP_SIZE_FRAC,
                "multi_tip_span_nm": MULTI_TIP_SPAN_NM,
                "radial_dispersion": rnd.dispersion,
                "radial_dispersion_floor": rnd.floor,
                "radial_dispersion_excess": rnd.excess,
                "aspect_ratio": aspect_ratio,
                "min_axis_ratio": min_axis_ratio,
                "min_aspect": min_aspect,
                "area_px": area_px,
                "major_axis_px": major,
                "minor_axis_px": minor,
                "polarity_used": chosen,
                "n_components": int(n_components),
                "pixel_size_x_m": px_size_x,
                "pixel_size_y_m": px_size_y,
            },
        )
