"""TiltProbeCircle —— 恒流内接圆测倾斜(Nanonis SmarTilt 做法的自研版)。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

Nanonis 界面上的 **SmarTilt** 按钮做的事是:反馈开着(恒流),让针尖在当前扫描
框的**内接圆**上跑一圈,由 Z(θ) 解出样品倾斜,结果写进 Piezo Calibration 的
sample tilt。手册(Scan Control / ROI SmarTilt)里有这个功能,但 **TCP 协议不
暴露它** —— ``nanonis_spm`` 里没有任何 SmarTilt / AutoTilt 命令(全仓库大小写
不敏感搜索确认)。所以要自动调平,这一圈必须自己跑。

**为什么是圆而不是拟合一帧图**:整圈几秒钟跑完,x 和 y 两个方向是在同一个时间
尺度上测的。而一帧 512 线 × 2 s/线的图要扫 34 分钟,沿慢扫轴图像顶部与底部相隔
半小时,那段时间的热漂移会原样表现为视在倾斜,与真实倾斜无法区分 —— 帧法**只有
快扫方向是准的**(见 :func:`mast.vision.tilt.estimate_tilt`)。

安全性:反馈必须开着(恒流跟随表面),半径受当前扫描框约束,每一步都是 Follow-Me
的小位移。这与 DANGEROUS 的 ``MoveProbeXY``(任意大位移)本质不同,定级 CONFIRM,
与 ``MoveToXY`` 一致。
"""

from __future__ import annotations

import logging
import math
import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.vision.tilt import CIRCLE_MIN_POINTS, fit_circle_tilt

_log = logging.getLogger(__name__)

#: 半径相对当前扫描框短边的默认比例。0.4 = 内接圆(0.5)留一点余量,别贴着框边 ——
#: 框边正是漂移和压电非线性最大的地方。
DEFAULT_RADIUS_FRAC = 0.4

#: 半径的绝对上下限(米)。下限保证圆足够大、倾斜产生的 Z 起伏高过噪声:
#: 0.1° 在 20 nm 半径上产生 35 pm,已是典型噪声底的两倍多。
MIN_RADIUS_M = 2e-9
MAX_RADIUS_M = 5e-7


def _first_float(record):
    """从 Nanonis 回包里取第一个浮点数;取不到返回 None。"""
    parsed = getattr(record, "return_value", None)
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    vals = parsed[2]
    if not isinstance(vals, (list, tuple)) or not vals:
        return None
    try:
        return float(vals[0])
    except (TypeError, ValueError):
        return None


def _two_floats(record):
    parsed = getattr(record, "return_value", None)
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    vals = parsed[2]
    if not isinstance(vals, (list, tuple)) or len(vals) < 2:
        return None
    try:
        return float(vals[0]), float(vals[1])
    except (TypeError, ValueError):
        return None


class TiltProbeCircle(BaseSkill):
    """Measure sample tilt by walking a constant-current circle (SmarTilt-style)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TiltProbeCircle",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在 Z 反馈**开启**（恒流）的状态下让针尖绕一个小圆走一圈，并拟合 Z(theta)，"
                "以此测量样品倾斜。这是 Nanonis SmarTilt 按钮的自建等价物（TCP 协议没有把它暴露出来）"
                "。与拟合一整幅扫描帧不同，两条轴是在同一个数秒量级的时间尺度上测出来的，因此谁都不会被热漂移污染。"
                "它需要一块**平坦**的位置 —— 一个跨过台阶的圆会被它的拟合残差挡掉。返回测得的倾斜；"
                "它**不会**改变任何硬件的倾斜设置（要闭环请用 AutoTilt）。"
            ),
            parameters=[
                ParameterSpec(
                    name="radius_m",
                    type="float",
                    description=(
                        "圆半径，单位**米**。不传则由当前扫描帧推出（取短边的 0.4 倍，也就是留了余量地落在内切圆里）"
                        "。"
                    ),
                    unit="m",
                    required=False,
                    min_value=MIN_RADIUS_M,
                    max_value=MAX_RADIUS_M,
                ),
                ParameterSpec(
                    name="n_points",
                    type="int",
                    description=(
                        "沿圆周取多少个采样点。点越多，噪声平均得越干净，但也越慢。"
                    ),
                    required=False,
                    default=24,
                    min_value=CIRCLE_MIN_POINTS,
                    max_value=180,
                ),
                ParameterSpec(
                    name="settle_s",
                    type="float",
                    description=(
                        "每个点上读 Z 之前的稳定时间，好让反馈在横向移动之后跟上来。"
                    ),
                    unit="s",
                    required=False,
                    default=0.05,
                    min_value=0.0,
                    max_value=5.0,
                ),
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description=(
                        "圆心 X。不传则用针尖当前位置。"
                    ),
                    unit="m",
                    required=False,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="圆心 Y。不传则用当前位置。",
                    unit="m",
                    required=False,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="noise_floor_m",
                    type="float",
                    description=(
                        "用来判断拟合残差的 Z 噪声本底（一个跨过台阶的圆会被挡掉）。不传则由起始点上的重复读数估出来。"
                    ),
                    unit="m",
                    required=False,
                    min_value=0.0,
                    max_value=1e-8,
                ),
            ],
            # 反馈必须开着:恒流下 Z 才跟随表面。反馈关着走这一圈 = 针尖以固定
            # 高度扫过一个倾斜的表面,轻则测不到东西,重则撞上去。
            preconditions=["z_controller_on", "scan_not_running"],
            estimated_duration_s=15.0,
            composition_level=2,
            tags=["tilt", "measure", "smartilt", "write"],
        )

    # ── 几何 ─────────────────────────────────────────────────────────────────

    def _resolve_geometry(self, context, params, calls):
        """返回 (cx, cy, radius, note)。任何一项拿不到就返回 None + 原因。"""
        cx = params.get("center_x_m")
        cy = params.get("center_y_m")
        radius = params.get("radius_m")
        note = ""

        if cx is None or cy is None:
            rec = context.safe_call("FolMe_XYPosGet", 1)
            calls.append(rec)
            pos = _two_floats(rec) if not rec.error else None
            if pos is None:
                return None, "读不到针尖当前位置(FolMe_XYPosGet)"
            cx, cy = pos if cx is None or cy is None else (cx, cy)

        if radius is None:
            rec = context.safe_call("Scan_FrameGet")
            calls.append(rec)
            frame = getattr(rec, "return_value", None)
            width = height = None
            if not rec.error and isinstance(frame, (list, tuple)) and len(frame) > 2:
                vals = frame[2]
                if isinstance(vals, (list, tuple)) and len(vals) >= 4:
                    try:
                        width, height = abs(float(vals[2])), abs(float(vals[3]))
                    except (TypeError, ValueError):
                        width = height = None
            if not width or not height:
                return None, (
                    "读不到当前扫描框尺寸,无法推导圆半径 —— "
                    "请显式给 radius_m"
                )
            radius = min(width, height) * DEFAULT_RADIUS_FRAC
            note = (f"半径由扫描框推导: min({width:.3g}, {height:.3g}) × "
                    f"{DEFAULT_RADIUS_FRAC} = {radius:.3g} m")

        radius = max(MIN_RADIUS_M, min(MAX_RADIUS_M, float(radius)))
        return (float(cx), float(cy), radius, note), ""

    def _estimate_noise(self, context, calls, n=8):
        """在起点重复读 Z 估噪声底(不动针尖,几十毫秒的事)。"""
        reads = []
        for _ in range(n):
            rec = context.safe_call("ZCtrl_ZPosGet")
            calls.append(rec)
            val = _first_float(rec) if not rec.error else None
            if val is not None:
                reads.append(val)
        if len(reads) < 3:
            return 0.0
        import numpy as np
        arr = np.asarray(reads, dtype=float)
        # 用相邻差分的 MAD:重复读之间若有慢漂移,直接取 std 会把漂移算进噪声。
        diffs = np.diff(arr)
        if diffs.size == 0:
            return 0.0
        mad = float(np.median(np.abs(diffs - np.median(diffs))))
        return mad * 1.4826 / math.sqrt(2.0)

    # ── 执行 ─────────────────────────────────────────────────────────────────

    def execute(self, context, params: dict) -> SkillResult:
        calls = []
        n_points = int(params.get("n_points") or 24)
        settle_s = float(params.get("settle_s") or 0.0)

        geom, err = self._resolve_geometry(context, params, calls)
        if geom is None:
            return SkillResult(skill_name="TiltProbeCircle", success=False,
                               error=err, nanonis_calls=calls)
        cx, cy, radius, geom_note = geom

        noise = params.get("noise_floor_m")
        noise = float(noise) if noise is not None else None

        # 记下起点,无论成败都要回来 —— 把针尖留在圆周上某个随机角度,会让调用方
        # 之后的一切位置推理都错位。
        start = None
        rec_start = context.safe_call("FolMe_XYPosGet", 1)
        calls.append(rec_start)
        if not rec_start.error:
            start = _two_floats(rec_start)

        angles: list[float] = []
        z_vals: list[float] = []
        times: list[float] = []
        t0 = time.monotonic()
        failures = 0

        try:
            # 先走到圆周起点并估噪声(在圆上估,和测量点同一工作条件)。
            for k in range(n_points):
                if context.check_abort():
                    return SkillResult(
                        skill_name="TiltProbeCircle", success=False,
                        error="用户中止 —— 圆周测量未完成",
                        data={"points_done": len(z_vals)},
                        nanonis_calls=calls,
                    )
                theta = 2.0 * math.pi * k / n_points
                x = cx + radius * math.cos(theta)
                y = cy + radius * math.sin(theta)

                rec_move = context.safe_call("FolMe_XYPosSet", x, y, 1)
                calls.append(rec_move)
                if rec_move.error:
                    failures += 1
                    continue
                if settle_s > 0:
                    time.sleep(settle_s)

                if k == 0 and noise is None:
                    noise = self._estimate_noise(context, calls)

                rec_z = context.safe_call("ZCtrl_ZPosGet")
                calls.append(rec_z)
                z = _first_float(rec_z) if not rec_z.error else None
                if z is None:
                    failures += 1
                    continue
                angles.append(theta)
                z_vals.append(z)
                times.append(time.monotonic() - t0)
        finally:
            # 回起点。放在 finally 里:异常、abort、硬件报错都不能把针尖丢在圆上。
            if start is not None:
                back = context.safe_call("FolMe_XYPosSet", start[0], start[1], 1)
                calls.append(back)

        if len(z_vals) < CIRCLE_MIN_POINTS:
            return SkillResult(
                skill_name="TiltProbeCircle", success=False,
                error=(f"圆周上只取到 {len(z_vals)} 个有效点"
                       f"(需要 ≥{CIRCLE_MIN_POINTS};{failures} 次读写失败)"),
                data={"points_done": len(z_vals), "failures": failures},
                nanonis_calls=calls,
            )

        fit = fit_circle_tilt(angles, z_vals, radius, times_s=times,
                              noise_floor_m=noise or 0.0)

        data = fit.as_dict()
        data.update({
            "center_x_m": cx,
            "center_y_m": cy,
            "radius_m": radius,
            "noise_floor_m": noise or 0.0,
            "failed_points": failures,
            "elapsed_s": time.monotonic() - t0,
        })
        if geom_note:
            data["geometry_note"] = geom_note

        if not fit.valid:
            # 测量失败是**显式失败**,不是「成功但数值可疑」。台阶穿圆时给一个
            # 数字,下游会拿它去调硬件。
            reason = {
                "residual_too_large": (
                    f"圆周上的 Z 起伏不符合单一倾斜平面"
                    f"(残差 {fit.residual_rms_m:.3g} m = 噪声底的 "
                    f"{fit.residual_ratio:.1f} 倍)—— 圆多半跨过了台阶或污染物。"
                    f"换一块更平的地方再测。"),
                "too_few_points": "有效点太少,无法拟合",
                "fit_failed": "正弦拟合失败",
                "bad_radius": "半径无效",
                "shape_mismatch": "角度与 Z 数组长度不一致",
            }.get(fit.invalid_reason, fit.invalid_reason)
            return SkillResult(
                skill_name="TiltProbeCircle", success=False,
                error=reason, data=data, nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="TiltProbeCircle",
            success=True,
            data=data,
            summary=(
                f"倾斜 {fit.slope_mag_deg:.4f}° "
                f"(x={fit.tilt_x_deg:+.4f}°, y={fit.tilt_y_deg:+.4f}°), "
                f"下坡方向 {fit.downhill_deg:.0f}°, "
                f"半径 {radius * 1e9:.1f} nm / {len(z_vals)} 点"
            ),
            nanonis_calls=calls,
        )
