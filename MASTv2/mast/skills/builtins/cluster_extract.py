"""ExtractClusters：先提取全部连通域，再由独立判定层选择目标。

未完成扫描需要先裁出有效行；NaN 行不能直接进入平面拟合，否则会污染全部结果。
一帧可能包含多个团簇、既有结构与线状伪影，面积最大不等于目标。

默认保留 RAW 数据，可选减去单个全局平面；不提供逐行平场。
逐行拟合会被同一行内的延展团簇抬高，减去该拟合会削弱团簇并引入伪影。
不同预处理得到的峰高、面积与分割结果不可直接比较，输出需说明所用口径。

提取层不因形状或边缘接触而丢弃连通域。touches_edge 与 touches_unscanned
提示截断，判定层需据此解释面积与长宽比，而不是把截断值当作完整形状。

圆度使用 vision.roundness 的 radial_dispersion、floor、excess 与
equivalent_axis_ratio。像素边计数的 4πA/P² 存在方向性偏差，不能沿用其旧阈值。
equivalent_axis_ratio 表示与当前径向离散等效的椭圆轴比。

目标选择结合形状、面积、峰高与空间锚点；不能因某条判据在一批数据上不承重
就放宽它。阈值与容差均须用相同预处理下的候选及反例验证。
这里的默认工作点未标定，不代表任何具体样品或仪器的测量结果。
"""
from __future__ import annotations

import logging
import math
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

# scaled-MAD 使用标准正态一致性系数，将稳健离散估计换算到 σ 的尺度。
_MAD_TO_SIGMA = 1.4826

#: 分割阈值(中位数 ± k×σ_MAD)。**未标定。**
_DEFAULT_THRESHOLD_MAD = 3.0

# 倾斜与纹理的比值用于诊断预处理是否需要调整。
# 该工作点未标定，不能把比值阈值当成已验证的样品分类边界。
_DEFAULT_TILT_WARN_RATIO = 20.0

# 最小连通域像素数是未标定的列表过滤参数。
# 它不替代上层的形状、面积、峰高与锚点联合判定。
_DEFAULT_MIN_AREA_PX = 4

#: 返回多少个(按面积降序截断)。截断了会在 `truncated` 里说出来。
_DEFAULT_MAX_CLUSTERS = 50

# 空间锚点容差用于匹配候选，应按当前定位与分割误差验证。
# 始终报告实际匹配距离；距离过远时弃权，不能把最近的候选自动认作目标。
_DEFAULT_ANCHOR_TOLERANCE_M = 3e-9

# 尺寸合理性仅作提示。此工作点未经样品标定，不能作为硬上界拒绝候选。
_PLAUSIBLE_MAX_DIAMETER_M = 3e-9


class ExtractClusters(BaseSkill):
    """分割一帧 .sxm 里的所有团簇,如实返回列表(不挑、不判、不过滤)。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ExtractClusters",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "把一张已保存的 .sxm 里**所有**团簇分割出来，以**列表**返回，"
                "每一个带几何量 + 真实 xy 坐标 + 峰高。**半张图也照样能用**"
                "（只吃已经扫完整的那些行）。找到什么就返回什么，线状伪影也一并返回 "
                "—— 挑哪一个、怎么判，是调用方的事。配合 "
                "AssessClusterRoundness 用 —— 注意**它是另一套分割，不是这一套的"
                "封装**：它按 `mean ± threshold_sigma × std`（默认 1.5σ，"
                "plane_subtract 之后）切，这里按 `median ± 3×σ_MAD`（默认 RAW）切，"
                "**同一帧会得到不同的 blob**；或者自己写"
                "「圆 且 大 且 高」的合取。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path", type="str", required=True,
                    description="已保存的那张 .sxm 的路径。",
                ),
                ParameterSpec(
                    name="channel", type="str", required=False, default="Z",
                    description=".sxm 头里的通道名（默认 Z）。",
                ),
                ParameterSpec(
                    name="polarity", type="str", required=False, default="auto",
                    description=(
                        "bright | dark | auto。auto 两侧都试，留下"
                        "**更紧凑**的那一侧，并把两侧的证据都报出来 —— "
                        "图像可呈现不同极性，因此保留显式选择及双侧证据。"
                    ),
                    allowed_values=["auto", "bright", "dark"],
                ),
                ParameterSpec(
                    name="threshold_mad", type="float", required=False,
                    default=_DEFAULT_THRESHOLD_MAD, min_value=0.5, max_value=20.0,
                    description="分割阈值，单位是 MAD-sigma。UNCALIBRATED。",
                ),
                ParameterSpec(
                    name="level", type="str", required=False, default="none",
                    allowed_values=["none", "plane"],
                    description=(
                        "分割**之前**的平场。'none'（默认）= RAW，"
                        "用户要的就是这个。'plane' = 减掉**一个**全局"
                        "平面。**逐行**平场不提供，传进来直接拒绝："
                        "它吃掉的正是我们要找的那一种团簇。"
                    ),
                ),
                ParameterSpec(
                    name="tilt_warn_ratio", type="float", required=False,
                    default=_DEFAULT_TILT_WARN_RATIO, min_value=0.0,
                    max_value=100000.0,
                    description=(
                        "残余平面的峰谷值超过这么多倍噪声 sigma 时告警"
                        "（「这一帧没调平，RAW 阈值可能不可靠」）。"
                        "NOT CALIBRATED（未标定）—— 到目前为止"
                        "还没有观察到任何一帧因此失败。"
                    ),
                ),
                ParameterSpec(
                    name="min_area_px", type="int", required=False,
                    default=_DEFAULT_MIN_AREA_PX, min_value=1, max_value=100000,
                    description=(
                        "过滤比此值小的连通域，使列表可读。此默认值未标定，过滤前后数量均会报告；目标判别应由上层使用完整特征完成。"
                    ),
                ),
                ParameterSpec(
                    name="max_clusters", type="int", required=False,
                    default=_DEFAULT_MAX_CLUSTERS, min_value=1, max_value=10000,
                    description="最多返回这么多个，按面积从大到小。",
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["scan", "analysis", "cluster", "extract", "read"],
        )

    # ── 主流程 ─────────────────────────────────────────────────────────
    def execute(self, context, params: dict) -> SkillResult:
        name = "ExtractClusters"
        scan_path = str(params.get("scan_path") or "").strip()
        channel = str(params.get("channel") or "Z").strip()
        polarity = str(params.get("polarity") or "auto").strip().lower()
        k_mad = float(params.get("threshold_mad", _DEFAULT_THRESHOLD_MAD))
        min_area = int(params.get("min_area_px", _DEFAULT_MIN_AREA_PX))
        max_clusters = int(params.get("max_clusters", _DEFAULT_MAX_CLUSTERS))

        if not scan_path or not Path(scan_path).exists():
            return SkillResult(skill_name=name, success=False,
                               error=f"scan_path 不存在: {scan_path!r}")
        try:
            import numpy as np
            from scipy.ndimage import label
            from mast.data.processors import plane_subtract
            from mast.io.mosaic import parse_xy_meta
            from mast.io.nanonis_files import read_sxm
        except ImportError as exc:
            return SkillResult(skill_name=name, success=False,
                               error=f"缺依赖: {exc}")

        try:
            scan = read_sxm(scan_path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f"读不了 .sxm: {type(exc).__name__}: {exc}")

        channels = scan.get("channels") or {}
        ch = channels.get(channel)
        if not isinstance(ch, dict):
            return SkillResult(
                skill_name=name, success=False,
                error=(f"这个 .sxm 里没有通道 {channel!r}。它有:{sorted(channels)}"
                       " —— 请指名要哪一个(挑错通道量出来的数看起来完全正常,"
                       "只是量的是另一个物理量)。"),
                data={"available_channels": sorted(channels)})
        # 先用统一几何入口处理快轴反扫镜像与慢轴方向，再计算仪器坐标。
        # 不能将裸数组中的行列索引直接解释为扫描框的物理位置。
        from mast.io.nanonis_files import sxm_oriented_frames
        img = sxm_oriented_frames(scan, channel).get("forward")
        if img is None:
            return SkillResult(skill_name=name, success=False,
                               error=f"通道 {channel!r} 没有正扫也没有反扫数据")
        img = np.asarray(img, dtype=np.float64)
        if img.ndim != 2 or img.size == 0:
            return SkillResult(skill_name=name, success=False,
                               error=f"不是二维图像: shape={img.shape}")

        ny_full, nx_full = img.shape

        # 只将完整且有限的扫描行交给平面拟合，避免 NaN 行污染整个结果。
        # 有效区域判定复用 frame_validity，不另建扫描前沿推断。
        from mast.vision.frame_validity import acquired_row_mask
        from mast.vision.roundness import assess_mask

        full_rows = acquired_row_mask(img)
        n_scanned = int(full_rows.sum())
        if n_scanned < 2:
            return SkillResult(
                skill_name=name, success=False,
                error=(f"只有 {n_scanned} 行是扫完整的(共 {ny_full} 行)——"
                       "这一帧还没有可分析的区域。这是**输入还没准备好**,"
                       "不是这一帧坏了。"),
                data={"rows_total": ny_full, "rows_scanned": n_scanned})
        crop = img[full_rows]

        # RAW 默认保留采集数据，全局平面可作为显式预处理并在输出中报告。
        # 逐行拟合会受到延展团簇自身的影响，从而削弱目标并产生伪影，因此不提供此选项。
        level = str(params.get("level") or "none").strip().lower()
        if level in ("line", "line_by_line", "poly1", "poly2", "median"):
            return SkillResult(
                skill_name=name, success=False,
                error=("逐行平场会被延展团簇抬高，减去拟合后削弱目标并可能产生伪影，本技能不提供。请使用 level='none'（RAW，默认）或 'plane'（单个全局平面）。"
                       ))
        if level not in ("none", "plane"):
            return SkillResult(skill_name=name, success=False,
                               error=f"level 只能是 none | plane,收到 {level!r}")

        # 残余斜率:「在调平的情况下」这个前提要能判,所以量出来报出去。
        tilt_pp_m = self._plane_pp(crop, np)
        leveled = plane_subtract(crop) if level == "plane" else crop
        if not np.isfinite(leveled).any():
            return SkillResult(skill_name=name, success=False,
                               error="平场之后没有有限值 —— 这一帧无法分析",
                               data={"rows_total": ny_full,
                                     "rows_scanned": n_scanned})

        med = float(np.nanmedian(leveled))
        sigma = float(np.nanmedian(np.abs(leveled - med))) * _MAD_TO_SIGMA
        if not (sigma > 0):
            return SkillResult(
                skill_name=name, success=False,
                error=("已扫区域的 MAD 是 0 —— 这一段真的是死平的"
                       "(注意:这条只对**已扫区域**成立,不是被 NaN 行拖累的)"),
                data={"rows_total": ny_full, "rows_scanned": n_scanned})

        tilt_ratio = params.get("tilt_warn_ratio", _DEFAULT_TILT_WARN_RATIO)
        tilt_warning = None
        if tilt_ratio is not None and sigma > 0 and tilt_pp_m / sigma > float(tilt_ratio):
            tilt_warning = (
                f"整帧残余斜面峰谷 {tilt_pp_m * 1e12:.0f} pm = {tilt_pp_m / sigma:.1f}×噪声,"
                f"超过 {float(tilt_ratio):g}× —— **这一帧可能没调平**,RAW 阈值分割"
                "可能被斜坡主导。可以试 level='plane',但请注意那会同时抬高噪声连通域数。"
                "(此提示阈值尚未按目标仪器标定，请结合原图核验。)")

        # ── 极性 ────────────────────────────────────────────────────────
        sides = {}
        for side in ("bright", "dark"):
            mask = (leveled > med + k_mad * sigma) if side == "bright" \
                else (leveled < med - k_mad * sigma)
            sides[side] = self._describe_side(mask, leveled, med, side, min_area, np, label)

        if polarity in ("bright", "dark"):
            chosen = polarity
            rule = "caller"
        else:
            # 按最大连通域的长宽比比较两种极性，同时报告双方证据。
            # 自动规则未经样品标定，调用方可显式指定 polarity。
            b, d = sides["bright"], sides["dark"]
            chosen = "bright" if (b["largest_aspect"], b["largest_peak_pm"]) >= \
                                 (d["largest_aspect"], d["largest_peak_pm"]) else "dark"
            rule = "auto:更紧凑(最大连通域长宽比,并列时比峰高)"

        labelled = sides[chosen]["labelled"]
        n_raw = sides[chosen]["n_raw"]

        # ── 坐标 ────────────────────────────────────────────────────────
        header = scan.get("header") or {}
        meta = parse_xy_meta(header)
        angle_known = True
        if meta:
            cx_m, cy_m = float(meta["cx"]), float(meta["cy"])
            w_m, h_m = float(meta["w"]), float(meta["h"])
            angle_deg = float(meta.get("angle") or 0.0)
            # ⚠️ `parse_xy_meta` 在角度**解析失败**时仍然给 0.0,而「是不是转过」
            # 的判据是 `abs(angle) > 1`,0.0 恰好让它不触发 —— 兜底值落在
            # 「没什么可担心的」那一侧。它为此带了一个旗标,这里必须透出去,
            # 否则一帧真的转过的图会被当成轴对齐,而调用方拿坐标去移动针尖。
            angle_known = bool(meta.get("angle_known", True))
        else:
            cx_m = cy_m = 0.0
            w_m = h_m = float("nan")
            angle_deg = 0.0
            angle_known = False

        # 像素 → 米:**用共享的那一份**(2026-08-10 合并,见 mosaic.px_to_m 的说明)。
        # 曾经有三份实现,第三份是一次性脚本里内联的、唯一没被测过的那份。
        #
        # 注意传的是 **ny_full / nx_full**,不是裁剪后的行数 —— 裁剪只砍掉了
        # 末尾未扫的行,保留行的行号与原帧一致,所以 y 换算必须按原帧算。
        from mast.io.mosaic import px_to_m as _px_to_m

        def px_to_m(px_x: float, px_y: float):
            if not (w_m == w_m and h_m == h_m):      # NaN → 没有几何信息
                return None, None
            return _px_to_m(px_x, px_y, nx=nx_full, ny=ny_full,
                            cx_m=cx_m, cy_m=cy_m, w_m=w_m, h_m=h_m,
                            angle_deg=angle_deg)

        px_area_m2 = (w_m / nx_full) * (h_m / ny_full) if w_m == w_m else float("nan")

        # ── 逐个连通域 ──────────────────────────────────────────────────
        clusters = []
        for lab_id in range(1, n_raw + 1):
            blob = labelled == lab_id
            area_px = int(blob.sum())
            if area_px < min_area:
                continue
            ys, xs = np.where(blob)
            px_m_x = (w_m / nx_full) if w_m == w_m else float("nan")
            px_m_y = (h_m / ny_full) if h_m == h_m else float("nan")
            have_geom = (px_m_x == px_m_x and px_m_y == px_m_y)
            # 没有几何信息时退回像素单位算(aspect 照样有意义),但 *_nm 报 None ——
            # 不要拿「像素当米」凑一个数出来。
            major_len, minor_len, aspect = self._axes(
                xs, ys, np, px_m_x if have_geom else 1.0, px_m_y if have_geom else 1.0)
            peak_m = float(np.nanmax(leveled[blob]) - med) if area_px else 0.0
            if chosen == "dark":
                peak_m = float(med - np.nanmin(leveled[blob]))
            cy_px, cx_px = float(ys.mean()), float(xs.mean())
            x_m, y_m = px_to_m(cx_px, cy_px)
            area_m2 = area_px * px_area_m2 if px_area_m2 == px_area_m2 else float("nan")
            equiv_d_m = (2.0 * math.sqrt(area_m2 / math.pi)
                         if area_m2 == area_m2 and area_m2 > 0 else float("nan"))
            rnd = assess_mask(blob)
            clusters.append({
                "id": int(lab_id),
                "x_m": x_m, "y_m": y_m,
                "x_px": cx_px, "y_px": cy_px,
                "area_px": area_px,
                "area_nm2": area_m2 * 1e18 if area_m2 == area_m2 else None,
                "equiv_diameter_nm": equiv_d_m * 1e9 if equiv_d_m == equiv_d_m else None,
                # **全长**(等效椭圆的 4σ),不是标准差 —— 见 `_axes` 的说明。
                # 均匀圆盘上 major_nm ≈ equiv_diameter_nm,这两个数现在**可以互相核对**;
                # 2026-08-11 之前 major_nm 恒等于等效直径的 1/4。
                "major_nm": major_len * 1e9 if have_geom else None,
                "minor_nm": minor_len * 1e9 if have_geom else None,
                "aspect": aspect,
                # 圆度:**从质心量边界 r(θ) 的相对离散**,减掉同面积完美圆盘那一份。
                # `equivalent_axis_ratio` 是能直接读的那个数:「这个团簇的不规则
                # 程度相当于一个短轴/长轴 = q 的椭圆」。判不了时 q 是 None 且
                # `roundness_undecidable` 带理由 —— 不给凑出来的数。
                #
                # ⚠️ 2026-08-11 起**不再有 `circularity` 字段**。旧的 4πA/P² 的
                # 上确界:圆盘 0.617、轴对齐正方形 0.785,而阈值是 0.65 ——
                # 圆的一律不合格、方的一律合格。它与这里的数**不可比、不可换算**,
                # 留着一个同名字段只会让人拿旧阈值去比新数。
                **rnd.as_dict(),
                # 峰高是目标判定的一个特征，应与形状、面积和位置联合解释。
                # 不能把单个数据批次上的可分性推广为所有样品上的保证。
                "peak_height_pm": peak_m * 1e12,
                "touches_edge": bool(xs.min() == 0 or ys.min() == 0
                                     or xs.max() == crop.shape[1] - 1
                                     or ys.max() == crop.shape[0] - 1),
                # 贴着扫描前沿 = 它很可能只被扫了一半,面积/长宽比是截断后的值。
                # **标记不过滤**:错的几何量也是上层需要知道的事实。
                "touches_unscanned": bool(n_scanned < ny_full
                                          and ys.max() == crop.shape[0] - 1),
                "size_plausible": bool(equiv_d_m != equiv_d_m
                                       or equiv_d_m <= _PLAUSIBLE_MAX_DIAMETER_M),
            })

        clusters.sort(key=lambda c: -c["area_px"])
        truncated = len(clusters) > max_clusters
        if truncated:
            clusters = clusters[:max_clusters]
        for rank, c in enumerate(clusters, 1):
            c["rank"] = rank

        return SkillResult(
            skill_name=name, success=True,
            data={
                "scan_path": scan_path,
                "frame": {
                    "cx_m": cx_m, "cy_m": cy_m, "w_m": w_m, "h_m": h_m,
                    "angle_deg": angle_deg,
                    # 角度不可知时坐标未经验证 —— 说出来,不要按 0° 悄悄算。
                    "angle_known": angle_known,
                    "rows_total": ny_full, "rows_scanned": n_scanned,
                    "cols": nx_full,
                },
                "polarity_used": chosen,
                "polarity_rule": rule,
                "polarity_evidence": {
                    s: {k: v for k, v in d.items() if k not in ("labelled",)}
                    for s, d in sides.items()},
                "threshold_mad": k_mad,
                # 报告是否预处理及其类型。绝对峰高等特征仅在相同预处理口径下可直接比较。
                "leveling_used": level,
                "tilt_pp_pm": tilt_pp_m * 1e12,
                "tilt_over_sigma": (tilt_pp_m / sigma) if sigma > 0 else None,
                "tilt_warning": tilt_warning,
                "min_area_px": min_area,
                "n_raw_components": n_raw,
                "n_clusters": len(clusters),
                "truncated": truncated,
                "clusters": clusters,
            },
        )

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _describe_side(mask, leveled, med, side, min_area, np, label):
        labelled, n_raw = label(mask)
        largest_area = largest_aspect = 0
        largest_peak = 0.0
        n_kept = 0
        if n_raw:
            sizes = np.bincount(labelled.ravel())
            sizes[0] = 0
            n_kept = int((sizes >= min_area).sum())
            top = int(sizes.argmax())
            if top:
                ys, xs = np.where(labelled == top)
                largest_area = int(ys.size)
                _, _, largest_aspect = ExtractClusters._axes(xs, ys, np)
                vals = leveled[labelled == top]
                largest_peak = float(np.nanmax(vals) - med) if side == "bright" \
                    else float(med - np.nanmin(vals))
        return {"labelled": labelled, "n_raw": int(n_raw),
                "n_above_min_area": n_kept,
                "largest_area_px": largest_area,
                "largest_aspect": float(largest_aspect),
                "largest_peak_pm": largest_peak * 1e12}

    @staticmethod
    def _plane_pp(arr, np):
        """整帧最小二乘平面的峰谷值(米)—— 「这帧调平了没有」的那个量。"""
        ny, nx = arr.shape
        Y, X = np.mgrid[0:ny, 0:nx].astype(np.float64)
        A = np.column_stack([X.ravel(), Y.ravel(), np.ones(arr.size)])
        try:
            co, *_ = np.linalg.lstsq(A, arr.ravel(), rcond=None)
        except Exception:  # noqa: BLE001
            return 0.0
        plane = co[0] * X + co[1] * Y + co[2]
        return float(plane.max() - plane.min())

    @staticmethod
    def _axes(xs, ys, np, sx: float = 1.0, sy: float = 1.0):
        """返回等效椭圆的长轴全长、短轴全长与轴比。
        均匀椭圆半轴 a 的二阶矩为 a²/4，因此 4σ = 2a，才是轴全长。
        直接返回 sqrt(eig) 只得到标准差，会把尺寸低报四倍。
        二阶矩使用全部像素，比只由极端像素决定的 Feret 径更不易受单个挂点影响。
        先将像素坐标按 x/y 各自尺度转换为物理坐标，再计算协方差。
        长轴方向未必沿 x，不能算完像素轴后直接把 x 尺度配给长轴。
        """
        if ys.size < 4:
            return 0.0, 0.0, 0.0
        X = (xs - xs.mean()) * float(sx)
        Y = (ys - ys.mean()) * float(sy)
        cov = np.cov(np.stack([X, Y]))
        eig = np.linalg.eigvalsh(cov)
        minor = 4.0 * float(np.sqrt(max(eig[0], 0.0)))
        major = 4.0 * float(np.sqrt(max(eig[1], 0.0)))
        return major, minor, (minor / major if major > 0 else 0.0)

    # ── `_perimeter` 已删除(2026-08-11)────────────────────────────────
    #
    # 它是 `circularity = 4πA/P²` 的唯一用途,而那个判据整条作废了:
    # 边计数口径下轴对齐正方形恒等于 π/4 = 0.785,而数字化圆盘的**上确界**只有
    # 4π²/64 = 0.617 —— 阈值 0.65 卡在两者之间,**圆的一律不合格、方的一律合格**。
    # 2026-08-10 那次修的是「饱和」,修对了,但修完暴露的是更深的一层:
    # 这个量量的是「边界有多贴合像素栅格」,不是「有多圆」。
    #
    # 不留这个 helper,是因为留着它下次就会被人接回去用
    # (它看起来只是一个无害的几何工具)。要看那段历史与数据,去
    # `tests/v2/vision/test_roundness.py::test_the_old_metric_ranked_a_square_above_a_circle`
    # —— **那条测试自带一份本地实现**,所以历史事实钉得住,而生产代码里没有它。
