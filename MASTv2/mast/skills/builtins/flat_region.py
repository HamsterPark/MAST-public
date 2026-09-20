"""Find the flattest sub-window of the latest STM scan.

Used by ``ShapeTipOnSurface`` to pick a clean spot for the tip-shaper plunge.
The .sxm topography is plane-subtracted (removes scan tilt), then a sliding
window scans for the minimum RMS roughness. Returns the centre of the best
window in *instrument coordinates* (metres) so the caller can re-centre the
scan frame and dive there.
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

#: ``same_terrace`` 判据:窗内属于同一台面层的像素比例下限。留 2% 余量给台面层
#: 标签在边界上的抖动(中值滤波之后仍会有几个像素摇摆),不给跨台阶的窗留活路。
_SAME_TERRACE_FRAC = 0.98

# 局部 RMS 上限是独立的绝对验收条件，不随整帧噪声自适应放宽。
# 否则整帧越差，可接受的局部残差反而越大。默认值是工程工作点，
# 需要按样品与测量条件验证，不能视为已经完成仪器标定。
USABLE_FLAT_RMS_M = 25e-12


# 机器可读标记区分“已搜索但没有可用平区”与“无法分析”。
# 前者可触发换区重扫，后者需声明未知并按工作流处理。
# 生产方与消费方共用此常量，避免拼写漂移使分支失效。
VERDICT_NO_USABLE_REGION = "no_usable_region"


def _local_plane_rms(patch, np) -> "float | None":
    """每个窗口独立拟合平面后的残差 RMS，衡量局部平坦度。
    整帧去平面后再求窗口 RMS，衡量的是窗口离全局平面多远；
    带台阶的帧中，单一台面可能远离全局平面，而跨台阶窗口反而更接近。
    因此必须在窗口内拟合，避免把整体高度偏移混同于局部起伏。
    平面拟合仍可能吸收部分跨台阶变化，解释结果时需结合几何条件。
    有效点不足时返回 None，不为不可判定的窗口编造分数。
    """
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    design = np.column_stack([xx.ravel(), yy.ravel(), np.ones(patch.size)])
    z = patch.ravel()
    ok = np.isfinite(z)
    # 3 个自由度,少于 12 点的拟合残差没有意义(窗口本来就有 50% NaN 上限)
    if int(ok.sum()) < 12:
        return None
    try:
        coeff, *_ = np.linalg.lstsq(design[ok], z[ok], rcond=None)
    except Exception:  # noqa: BLE001 — 奇异设计矩阵:这个窗判不了,跳过
        return None
    return float(np.std(z[ok] - design[ok] @ coeff))


class FindFlatRegion(BaseSkill):
    """Locate the lowest-RMS sub-window of an .sxm scan."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="FindFlatRegion",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在一张 .sxm 扫描图的形貌通道上滑窗,返回去平面之后 RMS 粗糙度"
                "最低的那一块窗。返回的是窗中心坐标,单位米(仪器坐标系)。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path",
                    type="str",
                    description="要分析的 .sxm 扫描文件路径。",
                    required=True,
                ),
                ParameterSpec(
                    name="window_fraction",
                    type="float",
                    description=(
                        "搜索窗的边长,以整幅扫描范围的**比例**表示。0.2 = 20%"
                        "(例如 100 nm 的图里开一个 20 nm 的窗)。常用 0.1–0.3。"
                    ),
                    required=False,
                    default=0.2,
                    min_value=0.05,
                    max_value=0.5,
                ),
                ParameterSpec(
                    name="stride_fraction",
                    type="float",
                    description=(
                        "相邻两次落窗之间的步长,以窗边长的比例表示。0.5 = 相邻"
                        "两个窗重叠 50%(推荐)。更小 = 搜得更密。"
                    ),
                    required=False,
                    default=0.5,
                    min_value=0.1,
                    max_value=1.0,
                ),
                # ── 2026-08-14:一次要**多个**平区 ────────────────────────
                #
                # 起因是扎针:``poke_phase`` 最多扎 30 次,而落点此前只保证
                # 「没被用过」(FindCleanSpot 是纯几何避让,不含任何地形信息)——
                # 也就是说**可能扎在台阶边缘上**,而那是结果最不可控的地方。
                #
                # 本技能内部本来就在遍历所有窗口(``_sweep``),只是最后只交出
                # 打分最好的那一个。把候选留下来即可,不需要第二个 skill ——
                # 「同一个动作 N 份实现」是本仓最贵的形状之一。
                #
                # ⚠️ ``count=1``(缺省)时**整条路径逐字不变**:仍然只看 ``best``,
                # 返回的键一个不多一个不少。这是这次改动的安全边界。
                ParameterSpec(
                    name="count",
                    type="int",
                    description=(
                        "要交出几块互不重叠的平区。1(缺省)与历史行为**逐字一致**:"
                        "只报打分最好的那一个窗。>1 时额外返回一个 `sites` 列表,"
                        "从好到差排序,彼此至少相隔 `min_site_spacing_m`。"
                    ),
                    required=False,
                    default=1,
                    min_value=1,
                    max_value=64,
                ),
                ParameterSpec(
                    # 2026-08-25 由 `min_separation_m` 改名。原来它与下面那个
                    # 「与被排除点的最小距离」**同名**,而 skill_adapter 把
                    # parameters 收成字典 —— 后者覆盖前者,于是这一条(连同它的
                    # 扎针避让指示与 max_value)**从上线起一次都没到达过模型**。
                    # 保留那一个的名字是因为活的调用方用的是它:
                    # data_processing.find_flat_region、test_tip_shaping_workflow、
                    # 以及 exclude_used_spots 自己的描述都按那个名字引用排除语义。
                    name="min_site_spacing_m",
                    type="float",
                    unit="m",
                    description=(
                        "交出的这些落点之间的最小中心距。只在 count > 1 时用得上。"
                        "省略则退回窗的边长(也就是窗与窗不重叠)。"
                        "**要在这些落点上扎针**的调用方,应当传两倍于自己修针避让"
                        "半径的值 —— 这些落点之间不能互相污染。"
                    ),
                    required=False,
                    min_value=1e-10,
                    max_value=1e-5,
                ),
                ParameterSpec(
                    name="channel",
                    type="str",
                    description=(
                        "要用的形貌通道名('Z' 是标准选择)。"
                        "文件里没有这个通道时,退回第一个可用通道。"
                    ),
                    required=False,
                    default="Z",
                ),
                ParameterSpec(
                    name="exclude_used_spots",
                    type="str",
                    description=(
                        "可选:分号分隔的一串**已经用过的** (x,y) 坐标,都是米量 —— "
                        "每个值都可以带 SI 前缀:'100n,-50n;1.2u,0'。"
                        "窗中心落在其中任何一个的 `min_separation_m` 之内的一律跳过,"
                        "这样重试时就不会再挑中同一个点。"
                    ),
                    required=False,
                    default="",
                ),
                ParameterSpec(
                    name="min_separation_m",
                    type="float",
                    description=(
                        "与被排除的那些点之间的最小距离,单位**米**。"
                        "缺省 = 窗的边长(与用过的坑不重叠)。"
                    ),
                    unit="m",
                    required=False,
                    default=0.0,
                    min_value=0.0,
                ),
                # ── 2026-07-30:三个补齐(设计文档 scan_intelligence_scripted_rfc) ──
                ParameterSpec(
                    name="min_window_m",
                    type="float",
                    description=(
                        "窗的最小**物理**边长,单位**米**。光靠 window_fraction "
                        "表达不了「至少要有 50 nm 的平坦表面」—— 50 nm 帧的 0.2 "
                        "是 10 nm,根本不满足那个要求。Auto-tilt 要 >= 50 nm"
                        "(最好 100 nm)。帧本身就比这个还小时,本技能**直接失败**,"
                        "而不是悄悄把窗缩小。"
                    ),
                    unit="m",
                    required=False,
                    default=0.0,
                    min_value=0.0,
                    max_value=1e-5,
                ),
                ParameterSpec(
                    name="usable_rms_m",
                    type="float",
                    unit="m",
                    description=(
                        "可用平区的局部残差上限。即使最平窗口也超过上限，本技能仍失败并说明没有可用平区；argmin 存在不代表满足验收条件。留空使用 25 pm 的工程工作点，使用前应按样品与判据验证；覆盖默认值时记录理由。"
                    ),
                    required=False,
                    min_value=0.0,
                    max_value=1e-8,
                ),
                ParameterSpec(
                    name="same_terrace",
                    type="bool",
                    description=(
                        "要求整个窗都落在**同一个**原子台面上。"
                        "只看最小 RMS 的话,可能挑中一个跨台阶、两半各自平坦的窗 —— "
                        "那对倾斜测量毫无用处,因为倾斜测量的要害正是「台面**就是**"
                        "晶面」。默认关(向后兼容)。"
                    ),
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=2,
            tags=["scan", "analysis", "flat", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        scan_path = params["scan_path"]
        window_frac = float(params.get("window_fraction", 0.2))
        stride_frac = float(params.get("stride_fraction", 0.5))
        channel_name = params.get("channel", "Z")
        excl_str = params.get("exclude_used_spots", "") or ""
        min_sep = float(params.get("min_separation_m", 0.0))
        min_window_m = float(params.get("min_window_m", 0.0) or 0.0)
        same_terrace = bool(params.get("same_terrace", False))
        # 2026-08-14:一次要多个平区(扎针落点)。1 = 历史行为,逐字不变。
        want_count = max(1, int(params.get("count", 1) or 1))
        # 落点间距 —— 与上面 `min_sep`(排除距离)是**两个**参数,
        # 2026-08-25 之前它们同名,这一行读到的其实是排除距离。
        min_sep_m = params.get("min_site_spacing_m")
        min_sep_m = float(min_sep_m) if min_sep_m else None
        # 绝对线可以被调用方**显式**覆写,但默认是常量。
        # 这与被禁掉的 `max(25 pm, k×噪声)` 不是一回事:那个是**帧自己**把线推上去
        # (脏图 ⇒ 线变松 ⇒ 越脏越容易过),而这个是调用方写出来的一个数,
        # 看得见、进得了版本历史、也进得了报告。同 `sharp_edge_nm` 的惯例。
        usable_rms_m = float(params.get("usable_rms_m") or USABLE_FLAT_RMS_M)

        # 参数校验排在**加载 .sxm 之前**：一个写错的坐标串不该等到分析完一整张图
        # 才被指出来，而且此刻还没有任何东西可以「凑合着用」。
        #
        # **拒绝，不要凑合。** 这个参数的意思是「别再选这几个点」，而看不懂的那几块
        # 正是要避开的位置。忽略它们等于把针尖送回刚才那个坏点，而调用方会以为回避
        # 生效了 —— 挑出来的点看起来和正常结果一模一样。（原实现是
        # `except ValueError: continue`，静默丢弃。）
        excluded, bad_excl = self._parse_excluded(excl_str)
        if bad_excl:
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error=(
                    "exclude_used_spots 里有解析不了的坐标: "
                    + "; ".join(bad_excl[:5])
                    + "。每一项写成 `x,y`,分号分隔,数值可带 SI 前缀"
                    "(如 '100n,-50n;1.2u,0')。**没有把这些点排除掉就选点是危险的**"
                    "—— 它们正是你要避开的位置。"
                ),
            )

        p = Path(scan_path)
        if not p.exists():
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error=f"scan_path not found: {scan_path}",
            )

        try:
            import numpy as np
            from mast.io.nanonis_files import read_sxm
            # 去趋势复用 mast.vision.tilt.plane_subtract 的稳健拟合。
            # 普通最小二乘可能被多级台阶带偏，使每个平台内部残留渐变。
            # 本模块各入口共用同一算法，避免形貌分析使用不同预处理。
            from mast.vision.tilt import plane_subtract
            from mast.vision.frame_validity import judge_frame
        except ImportError as e:
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error=f"missing dependency: {e}",
            )

        try:
            scan = read_sxm(scan_path)
        except Exception as e:
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error=f"failed to read .sxm: {e}",
            )

        # 通过 sxm_oriented_frames 统一处理扫描方向，再将窗口中心转为物理坐标。
        # 原始数组行序不直接代表扫描框的正向坐标。
        from mast.io.nanonis_files import sxm_oriented_frames
        oriented = sxm_oriented_frames(scan, channel_name)
        if oriented.get("forward") is None:
            channels = scan.get("channels", {}) or {}
            first = next(iter(channels), None)
            if first is None:
                return SkillResult(
                    skill_name="FindFlatRegion", success=False,
                    error=f"no usable channel in {scan_path}",
                )
            oriented = sxm_oriented_frames(scan, first)
        img = oriented.get("forward")
        if img is None:
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error="no forward/backward frame in selected channel",
            )

        img = np.asarray(img, dtype=np.float64)

        # 死平帧上的所有窗口都可能并列最小，不能因此报告完美平区。
        # 前置判据区分缺少信息与有效但平坦的形貌，复用 mast.vision.frame_validity。
        frame = judge_frame(img)
        if not frame.usable:
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error=frame.reason,
                data=frame.meta(scan_path=scan_path,
                                rows=int(img.shape[0]), cols=int(img.shape[1])),
            )

        # Remove tilt; any NaN holes propagate but are filtered below.
        try:
            leveled = plane_subtract(img)
        except Exception as e:
            logger.info("plane_subtract failed, using raw data: %s", e)
            leveled = img - np.nanmean(img)

        ny, nx = leveled.shape

        # Pull scan geometry (centre + extents + ROTATION in metres/degrees).
        # 用 mosaic.parse_xy_meta 而不是自己 split:它是仓库里唯一同时处理
        # scan_angle 的几何解析入口。之前这里只取 offset/range,扫描框一旦旋转,
        # 返回的坐标就整个错位 —— 而调用方拿这个坐标去移动针尖。
        header = scan.get("header", {})
        angle_deg = 0.0
        try:
            from mast.io.mosaic import parse_xy_meta
            meta = parse_xy_meta(header)
        except Exception:  # noqa: BLE001
            meta = None
        if meta:
            cx_m, cy_m = float(meta["cx"]), float(meta["cy"])
            w_m, h_m = float(meta["w"]), float(meta["h"])
            angle_deg = float(meta.get("angle") or 0.0)
        else:
            try:
                cx_m, cy_m = (float(v) for v in
                              str(header.get("scan_offset", "0 0")).split()[:2])
            except Exception:
                cx_m, cy_m = 0.0, 0.0
            try:
                w_m, h_m = (float(v) for v in
                            str(header.get("scan_range", "1e-7 1e-7")).split()[:2])
            except Exception:
                w_m, h_m = 1e-7, 1e-7

        # Convert window size from fraction → pixels, honouring the physical floor.
        win_px = max(8, int(min(nx, ny) * window_frac))
        if min_window_m > 0:
            frame_short_m = min(w_m, h_m)
            if frame_short_m < min_window_m:
                # 显式失败,不静默缩窗:调用方要的是「至少这么大一块平地」,
                # 给它一块更小的会让它以为条件满足了。
                return SkillResult(
                    skill_name="FindFlatRegion", success=False,
                    error=(
                        f"帧本身只有 {frame_short_m * 1e9:.1f} nm,小于要求的最小"
                        f"窗口 {min_window_m * 1e9:.1f} nm —— 换一张更大的图再找。"),
                    data={"frame_short_m": frame_short_m,
                          "min_window_m": min_window_m},
                )
            px_per_m = min(nx / w_m, ny / h_m)
            win_px = max(win_px, int(math.ceil(min_window_m * px_per_m)))
            win_px = min(win_px, min(nx, ny))
        stride_px = max(1, int(win_px * stride_frac))

        # 逐像素台面层标签(same_terrace 用)。分割不可用时降级为"不做同层约束"
        # 并在结果里说明 —— 一个分析组件缺失不该让整次搜索失败。
        layer_labels = None
        terrace_note = ""
        if same_terrace:
            try:
                from mast.vision.seg_scale_adaptive import kde_layers
                from mast.vision.tilt import noise_floor
                from mast.vision.tilt import plane_subtract as tilt_plane_subtract

                filled = np.nan_to_num(img, nan=float(np.nanmean(img)))
                # 逐行去趋势会削弱台阶与延展结构，不能用它制造看起来更平的窗口。
                # 局部平坦度保留几何语义，只在窗口内拟合全局平面。
                coarse = tilt_plane_subtract(filled)
                # 行内差分 MAD 对台阶更稳健；高估噪声带宽会合并 KDE 峰。
                sig_n = noise_floor(coarse)
                if sig_n <= 0:
                    sig_n = max(float(np.std(coarse)) * 1e-3, 1e-15)
                _peaks, layer_labels = kde_layers(coarse, sig_n)
                if layer_labels is not None and len(np.unique(layer_labels)) < 2:
                    layer_labels = None      # 单层 = 整帧同一台面,约束自动满足
            except Exception as exc:  # noqa: BLE001
                terrace_note = f"同层约束不可用(分割失败: {exc}),已按无约束搜索"
                logger.info("FindFlatRegion: %s", terrace_note)
                layer_labels = None

        if min_sep <= 0:
            # default: window-side worth of separation
            min_sep = (w_m + h_m) * 0.5 * window_frac

        # （excluded 已在 execute 开头解析并校验过）

        # 像素 → 米。**用共享的那一份**(2026-08-10 合并):约定和旋转都在
        # `mast.io.mosaic.px_to_m` 里,这里不再自己算 —— 这套换算一度有三份实现,
        # 而第三份(一次性脚本里内联的)是唯一没被测过的那份。
        from mast.io.mosaic import px_to_m as _px_to_m

        def px_to_m(px_x: float, px_y: float) -> tuple[float, float]:
            return _px_to_m(px_x, px_y, nx=nx, ny=ny, cx_m=cx_m, cy_m=cy_m,
                            w_m=w_m, h_m=h_m, angle_deg=angle_deg)

        best = {"rms": float("inf"), "cx_m": None, "cy_m": None, "px": None}
        windows_checked = 0
        windows_skipped = 0
        windows_cross_terrace = 0
        # 合格窗口的候选池(2026-08-14,给 ``count > 1`` 用)。
        # ``count == 1`` 时它被填但不被读 —— 一份列表的内存换掉一个第二实现。
        # 数量级:256 px 图 / 128 px 窗 / 细扫步长 16 ⇒ 81 个窗,可忽略。
        candidates: list[dict] = []

        def _sweep(step: int) -> None:
            """按 ``step`` 的步长扫一遍,把 ``best`` 与三个计数器就地更新。

            提成函数是为了**能用不同步长扫两遍**:粗步长便宜,够用来说「有」;
            但要说「**这里没有**」就必须细扫 —— 见下面调用处的理由。
            """
            nonlocal windows_checked, windows_skipped, windows_cross_terrace
            for iy in range(0, ny - win_px + 1, step):
                for ix in range(0, nx - win_px + 1, step):
                    patch = leveled[iy:iy + win_px, ix:ix + win_px]
                    # Skip windows with too many NaN holes.
                    valid = patch[~np.isnan(patch)]
                    if valid.size < patch.size * 0.5:
                        continue

                    # 整窗同层:最小 RMS 会选到「跨台阶但两半各自平坦」的窗,那对
                    # 倾斜测量毫无用处 —— 台面就是晶面,单一台面内测到的斜率才是
                    # 压电扫描平面与晶面的失配角。
                    if layer_labels is not None:
                        lab = layer_labels[iy:iy + win_px, ix:ix + win_px]
                        counts = np.bincount(lab.ravel())
                        if counts.max() < lab.size * _SAME_TERRACE_FRAC:
                            windows_cross_terrace += 1
                            continue
                    # 「这块自己有多平」——**窗内**再拟合一次平面取残差。
                    # 不是「离全帧平面多远」;两者在有台阶的表面上指向相反的窗,
                    # 几何定义见 `_local_plane_rms`。
                    rms = _local_plane_rms(patch, np)
                    if rms is None:
                        continue
                    # Centre of this window in metres.
                    cx, cy = px_to_m(ix + win_px / 2.0, iy + win_px / 2.0)

                    if any(
                        (cx - ex) ** 2 + (cy - ey) ** 2 < min_sep ** 2
                        for ex, ey in excluded
                    ):
                        windows_skipped += 1
                        continue

                    windows_checked += 1
                    candidates.append({"rms": rms, "cx_m": cx, "cy_m": cy,
                                       "px": (ix, iy)})
                    if rms < best["rms"]:
                        best.update({"rms": rms, "cx_m": cx, "cy_m": cy,
                                     "px": (ix, iy)})

        _sweep(stride_px)

        # 粗网格搜索可能漏过符合验收条件的小区域。
        # 仅在即将报告无可用区域时补充细网格搜索，减少假阴性；
        # 粗网格已经找到合格区域时保持原有成本。
        if best["rms"] > usable_rms_m:
            fine = max(1, win_px // 8)
            if fine < stride_px:
                _sweep(fine)

        if best["cx_m"] is None:
            hint = ("Try a smaller window_fraction or a different scan.")
            # 所有窗口都跨台阶时也应提供更小窗口的重试建议。
            # 否则调用方缺少新的尺寸，可能在相同条件下重复失败。
            smaller_cross: list = []
            if windows_cross_terrace and layer_labels is not None:
                for frac in (0.75, 0.5, 0.35, 0.25):
                    w2 = int(win_px * frac)
                    if w2 < 8:
                        continue
                    best_frac = 0.0
                    st2 = max(1, w2 // 4)
                    for jy in range(0, ny - w2 + 1, st2):
                        for jx in range(0, nx - w2 + 1, st2):
                            sub = layer_labels[jy:jy + w2, jx:jx + w2]
                            cc = np.bincount(sub.ravel())
                            best_frac = max(best_frac, cc.max() / sub.size)
                    side2 = (w_m / nx) * w2
                    smaller_cross.append({"side_m": side2,
                                          "same_terrace_frac": round(float(best_frac), 3)})
                    if best_frac >= _SAME_TERRACE_FRAC:
                        break

            if windows_cross_terrace:
                ok2 = [d2 for d2 in smaller_cross
                       if d2["same_terrace_frac"] >= _SAME_TERRACE_FRAC]
                if ok2:
                    hint = (f"{windows_cross_terrace} 个窗因跨台阶被排除,"
                            f"但**换小一档就装得下**:"
                            f"{ok2[0]['side_m'] * 1e9:.0f} nm 的窗能落在单个台面里。"
                            f"用 min_window_m={ok2[0]['side_m']:.3e} 重试。")
                else:
                    hint = (f"{windows_cross_terrace} 个窗因跨台阶被排除 —— "
                        "这块区域的台面比要求的窗口还窄,换一处更大的平台。")
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error=(
                    f"no valid windows found (checked {windows_checked}, "
                    f"skipped {windows_skipped}, cross-terrace "
                    f"{windows_cross_terrace}). {hint}"
                ),
                data={
                    "windows_checked": windows_checked,
                    "windows_skipped": windows_skipped,
                    "windows_cross_terrace": windows_cross_terrace,
                    # 全跨台阶时:更小的窗能不能装进同一个台面。
                    # ``side_m`` 升序,第一个 ``same_terrace_frac >= 0.98`` 的就是答案。
                    "smaller_windows_same_terrace": smaller_cross,
                },
            )

        # argmin 仅说明相对最平，不能证明达到绝对验收条件。
        # 最优窗口仍超标时必须报告没有可用区域，不能把它交给后续流程当作平坦表面。
        if best["rms"] > usable_rms_m:
            # 若当前视野内没有满足条件的窗口，可建议重新选择区域或调整视野。
            # 建议不代表缩小视野后必然通过，仍须满足相同的局部残差与几何约束。
            smaller = []
            for frac in (0.75, 0.5, 0.35):
                w2 = int(win_px * frac)
                if w2 < 8:
                    continue
                b2 = float("inf")
                st2 = max(1, w2 // 4)
                for jy in range(0, ny - w2 + 1, st2):
                    for jx in range(0, nx - w2 + 1, st2):
                        r2 = _local_plane_rms(leveled[jy:jy + w2, jx:jx + w2], np)
                        if r2 is not None and r2 < b2:
                            b2 = r2
                if b2 < float("inf"):
                    smaller.append(((w_m / nx) * w2, b2))
            ok_smaller = [(s, r) for s, r in smaller if r <= usable_rms_m]
            if ok_smaller:
                s, r = ok_smaller[0]
                tail = (f"但**换小一档就有**:{s * 1e9:.0f} nm 的窗在这张图上能到 "
                        f"{r * 1e12:.1f} pm。用 min_window_m={s:.3e} 重试,"
                        f"比换地方便宜。(该提示未套 same_terrace/排除点,是线索不是承诺。)")
            else:
                tail = ("而且**缩小窗口也没用**"
                        + (f"(试到 {smaller[-1][0] * 1e9:.0f} nm 仍有 "
                           f"{smaller[-1][1] * 1e12:.1f} pm)" if smaller else "")
                        + " —— 这一片确实该换地方。")
            return SkillResult(
                skill_name="FindFlatRegion", success=False,
                error=(f"这张图里没有可用的 {(w_m / nx) * win_px * 1e9:.0f} nm 平区:"
                       f"最平的一块局部残差 {best['rms'] * 1e12:.1f} pm,"
                       f"超过可用线 {usable_rms_m * 1e12:.0f} pm。{tail}"),
                data={
                    # 机器可读 no_usable_region 表示已完成搜索但没有合格区域，
                    # 与无法分析的失败分开，允许调用方选择换区重扫或处理未知条件。
                    "verdict": VERDICT_NO_USABLE_REGION,
                    "usable_rms_m": usable_rms_m,
                    "best_rms_m": best["rms"],
                    "best_center_x_m": best["cx_m"],
                    "best_center_y_m": best["cy_m"],
                    "window_side_m": (w_m / nx) * win_px,
                    "smaller_windows": [{"side_m": s, "best_rms_m": r}
                                        for s, r in smaller],
                    "windows_checked": windows_checked,
                    "windows_skipped": windows_skipped,
                    "windows_cross_terrace": windows_cross_terrace,
                },
            )

        window_side_m = (w_m / nx) * win_px
        data = {
            "scan_path": scan_path,
            "usable_rms_m": usable_rms_m,
            "center_x_m": best["cx_m"],
            "center_y_m": best["cy_m"],
            "window_side_m": window_side_m,
            "rms_m": best["rms"],
            "pixel_origin": list(best["px"]),
            "window_side_px": win_px,
            "windows_checked": windows_checked,
            "windows_skipped": windows_skipped,
            "windows_cross_terrace": windows_cross_terrace,
            "scan_center_x_m": cx_m,
            "scan_center_y_m": cy_m,
            "scan_width_m": w_m,
            "scan_height_m": h_m,
            "scan_angle_deg": angle_deg,
            "same_terrace_enforced": layer_labels is not None,
        }
        if terrace_note:
            data["note"] = terrace_note
        # 很小的搜索窗口容易产生选择偏倚，局部最小 RMS 不能代表整体平整度。
        # 报告窗口尺度、局部残差与整帧幅度的关系，使调用方知道该结论适用的空间范围。
        frame_rms_m = float(np.nanstd(leveled[np.isfinite(leveled)])) \
            if np.isfinite(leveled).any() else None
        data["window_side_px"] = win_px
        data["frame_rms_m"] = frame_rms_m
        if frame_rms_m and frame_rms_m > 0:
            data["rms_ratio_to_frame"] = best["rms"] / frame_rms_m
        if win_px < 24:
            data["scale_caveat"] = (
                f"窗口只有 {win_px} 像素（{window_side_m * 1e9:.1f} nm）。"
                f"**这个尺度上的残差不能与更大尺度比较，也不保证你去那里扫会看到同样的平整度** —— "
                f"小窗口在任何图上都挑得到看着平的一块（选择偏倚）。"
                f"要一块真的能用的平区，把 window_fraction 调大或改用 min_window_m 指定物理尺寸。"
            )
        # ── count > 1:再交出 N 个互不重叠的平区(2026-08-14)──────────────
        #
        # 用途是**扎针落点**:``poke_phase`` 最多扎 30 次,而此前落点只保证「没被
        # 用过」—— 可能扎在台阶边缘。这里交出来的每一个都过了和 ``best`` 同一套
        # 判据(同层 + 窗内局部平面残差 + 排除表),所以**不是「差不多的地方」**。
        #
        # ⚠️ 两条纪律:
        # ① 只收 ``rms <= usable_rms_m`` 的 —— 候选已按 rms 升序,第一个超线就
        #    break。**绝不为了凑满 count 而放行不合格的窗**:凑数会让调用方以为
        #    「找到了 N 个可用点」,而其中几个根本不可用。
        # ② 交出的数量少于 count 是**正常结果**,不是失败。报文里说清楚实到几个,
        #    调用方据此决定是继续扎还是换地方。
        if want_count > 1 and candidates:
            sep = min_sep_m if min_sep_m else (w_m / nx) * win_px
            sites: list[dict] = []
            for c in sorted(candidates, key=lambda d: d["rms"]):
                if c["rms"] > usable_rms_m:
                    break
                if all((c["cx_m"] - s["center_x_m"]) ** 2
                       + (c["cy_m"] - s["center_y_m"]) ** 2 >= sep * sep
                       for s in sites):
                    sites.append({"center_x_m": c["cx_m"],
                                  "center_y_m": c["cy_m"],
                                  "rms_m": c["rms"]})
                    if len(sites) >= want_count:
                        break
            data["sites"] = sites
            data["sites_requested"] = want_count
            data["sites_min_separation_m"] = sep
            if len(sites) < want_count:
                data["sites_note"] = (
                    f"要 {want_count} 个,这张图上只找到 {len(sites)} 个互相间隔 "
                    f"≥ {sep * 1e9:.0f} nm 的可用平区 —— **这不是失败**,"
                    "是这一片能给的就这么多。用完了换一块地方再扫一张。")
        return SkillResult(skill_name="FindFlatRegion", success=True, data=data)

    @staticmethod
    def _parse_excluded(s: str) -> "tuple[list[tuple[float, float]], list[str]]":
        """``"1n,2n;3n,-1n"`` → ``([(1e-9, 2e-9), (3e-9, -1e-9)], [])``。

        两处都是 2026-08-04 改的，起因是同一件事：

        1. **接受 SI 前缀。** 有量纲参数现在整体走字符串通道，模型到处被教「写
           '100n' 不要写 1e-7」—— 它多半会把这个习惯带到这个自由格式串里来。原来
           这里用的是裸 ``float()``，``'1n'`` 直接 ValueError。
        2. **解析不了要说出来,不能静默跳过。** 原来是 ``except ValueError: continue``
           —— 于是「排除这几个已用过的点」会悄悄变成「一个都不排除」，然后技能
           **高高兴兴地把针尖送回刚才那个坏点**。它不报错、不降级、没有任何痕迹。

        返回 ``(坐标, 看不懂的块)``；调用方负责把后者说给用户听。
        """
        from mast.core.si_quantity import SIParseError, parse_quantity

        out: list[tuple[float, float]] = []
        bad: list[str] = []
        for chunk in (s or "").split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                x_s, y_s = chunk.split(",")
                out.append((parse_quantity(x_s, strict=False, what="x"),
                            parse_quantity(y_s, strict=False, what="y")))
            except (ValueError, SIParseError):
                bad.append(chunk)
        return out, bad
