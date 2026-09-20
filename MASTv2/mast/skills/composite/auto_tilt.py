"""自动调平 —— TiltCalibrate(一次性标定)+ AutoTilt(闭环)。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

Nanonis 界面上有 **SmarTilt** 按钮,但 **TCP 协议不暴露它**(``nanonis_spm`` 里
没有任何 SmarTilt / AutoTilt 命令,手册确认这是个界面功能)。所以自动调平要自己
做闭环:**测 → 算 → 设 → 复测 → 验收 / 回滚**。

## 为什么必须先标定

``Piezo_TiltSet(tilt_x, tilt_y)`` 与图像上测到的斜率之间,轴对应、符号、增益都
取决于仪器接线和 Nanonis 内部约定,**无法先验假定**。猜错符号 = 把倾斜往反方向
加倍。所以:

  * :class:`TiltCalibrate` 用 ±0.2° 的小步试探解出一个 2×2 响应矩阵 G,把
    「符号 + 轴交换 + 增益」一次吃掉,存进 instrument_profile;
  * **没有 G 时 AutoTilt 一律跳过**,不带着猜来的方向去动硬件。

## 为什么是 Python composite 而不是声明式 spec

迭代收敛判据(残差序列的比较)、2×2 矩阵乘法、反正切,全都超出声明式表达式的
能力(白名单里没有 ``atan`` / ``degrees``);位置决策树还需要动态步骤。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from mast.core import instrument_profile
from mast.core.si_quantity import format_si_readable
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.vision.tilt import circle_tilt_resolution_deg, z_span_for_frame

logger = logging.getLogger(__name__)


# ── 触发 / 验收判据(偏好级,可在设置里改) ──────────────────────────────────
#
# 统一物理量是「这一帧的斜坡吃掉多少 Z 量程」z_span = L·tan(θ)。用它而不是
# 「粗扫一个角度阈值、精扫另一个」:同样 0.3° 在 1 µm 帧上吃掉 5.2 nm 的 Z,
# 在 10 nm 帧上只吃 52 pm —— 粗扫敏感、精扫宽容自动成立,少一个自由度,而且
# 「防 Z 打满 / 防帧角撞边」这个物理动机直接可见。

#: 触发调平:帧内斜坡吃掉的 Z 量程超过总量程的这个比例。
DEFAULT_Z_BUDGET_FRAC = 0.05
#: 触发调平:斜坡高过表面自身起伏这个倍数时,形貌已被斜坡淹没。
DEFAULT_K_TOPO = 10.0
#: 硬红线:超过总量程这个比例,帧边缘有 rail / 撞边风险,必须调。
Z_SPAN_HARD_LIMIT_FRAC = 0.20
#: 验收阈 = 触发阈 × 这个系数(迟滞,防边界抖动反复触发)。
ACCEPT_FRAC_OF_TRIGGER = 0.5

#: 单次施加的最大倾斜增量(度)。tilt 阶跃会让扫描平面突转 → Z 瞬态;小步 +
#: 反馈开着是防撞针的硬要求,这是「试错 nudge」根本不知道的安全细节。
MAX_TILT_STEP_DEG = 1.0
#: 每小步之后的稳定时间。
TILT_STEP_SETTLE_S = 1.0
#: 闭环最多迭代几轮。
MAX_ITERATIONS = 3
#: 一轮之后残差没降到上一轮的这个比例以下 = 发散(标定失效 / 表面变了 / 针尖事件)。
CONVERGENCE_RATIO = 0.7

#: 标定用的试探步长(度)。小到无害,大到可测:0.2° 在 100 nm 跨度上产生 350 pm
#: 的斜坡,远高于典型噪声底。
CALIB_STEP_DEG = 0.2
#: 标定健全性:测到的响应幅度必须落在试探步长的这个倍数区间内。
CALIB_RESPONSE_MIN = 0.3
CALIB_RESPONSE_MAX = 3.0


def _read_tilt(context, calls) -> "tuple[float, float] | None":
    rec = context.safe_call("Piezo_TiltGet")
    calls.append(rec)
    if rec.error:
        return None
    parsed = getattr(rec, "return_value", None)
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    vals = parsed[2]
    if not isinstance(vals, (list, tuple)) or len(vals) < 2:
        return None
    try:
        return float(vals[0]), float(vals[1])
    except (TypeError, ValueError):
        return None


def _write_tilt(context, calls, tilt_x: float, tilt_y: float) -> str:
    rec = context.safe_call("Piezo_TiltSet", float(tilt_x), float(tilt_y))
    calls.append(rec)
    return rec.error or ""


def _operator_stopped(context) -> bool:
    """用户喊停了没有(缺陷⑬)。读不到 abort 通道就是 False —— 这里 fail-open。

    倾斜循环里的 ``time.sleep`` 加起来是十几秒量级,而中间只有 ``_measure`` 那一处
    能退出(TiltProbeCircle 自己查 abort)。十几秒对一个按了停止的人来说是很长的。
    """
    check = getattr(context, "check_abort", None)
    try:
        return bool(callable(check) and check())
    except Exception:  # noqa: BLE001 — 查停止本身不该把技能带走
        return False


def _stopped_by_operator(skill: str, calls, where: str) -> SkillResult:
    """软停的统一返回:**不是失败,不是超时,是有人喊停**。

    ``aborted_by_operator`` 是给下游用的机器可读位(与 graph_executor.abort_facts
    同名同义),``where`` 是给人看的「停在哪儿了」—— 停下来之后仪器是什么状态,
    这句话必须有人说,否则下一个动作是在一个没人描述过的状态上做的。
    """
    return SkillResult(
        skill_name=skill, success=False,
        error=f"aborted by user —— 用户要求停止。{where}",
        data={"aborted": True, "aborted_by_operator": True,
              "abort_reason": "aborted by user", "stopped_where": where},
        nanonis_calls=calls)


def _measure(context, params: dict) -> "tuple[dict | None, str]":
    """跑一次 TiltProbeCircle,返回 (data, error)。"""
    res = context.run("TiltProbeCircle", dict(params))
    if not getattr(res, "success", False):
        return None, getattr(res, "error", "TiltProbeCircle 失败")
    return dict(getattr(res, "data", {}) or {}), ""


class TiltCalibrate(BaseSkill):
    """Calibrate the piezo-tilt response matrix (one-off, operator present)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TiltCalibrate",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "一次性标定 Piezo_TiltSet 的两个轴与实测表面斜率之间的对应关系："
                "在每个轴上施加一个小的试探步，解出 2x2 响应矩阵"
                "（符号 + 轴交换 + 增益一次吃掉）。解得出来就把结果存进 instrument profile；**解不稳（响应矩阵条件数超过 TILT_CAL_MAX_COND）则整个技能失败、什么都不写** —— 那不是「存了一个差的」，是没有标定，AutoTilt 仍然会拒绝运行。"
                "**没有它 AutoTilt 一律拒绝运行** —— 轴对应与符号取决于这台装置的"
                "接线，猜错方向不是把倾斜去掉，而是把它**加倍**。"
                "每台仪器跑一次，跑在一块**平**的地方，而且要有用户在场。"
            ),
            parameters=[
                ParameterSpec(
                    name="step_deg", type="float",
                    description=(
                        "施加在每个倾斜轴上的试探步长。小到无害，"
                        "大到可测。"),
                    unit="deg", required=False, default=CALIB_STEP_DEG,
                    min_value=0.02, max_value=1.0,
                ),
                ParameterSpec(
                    name="radius_m", type="float",
                    description="测量圆的半径（见 TiltProbeCircle）。",
                    unit="m", required=False,
                    min_value=2e-9, max_value=5e-7,
                ),
                ParameterSpec(
                    name="n_points", type="int",
                    description="每个测量圆上取几个点。",
                    required=False, default=24, min_value=8, max_value=180,
                ),
            ],
            preconditions=["z_controller_on", "scan_not_running"],
            estimated_duration_s=90.0,
            composition_level=3,
            tags=["tilt", "calibration", "composite"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        step = float(params.get("step_deg") or CALIB_STEP_DEG)
        probe = {k: params[k] for k in ("radius_m", "n_points")
                 if params.get(k) is not None}

        orig = _read_tilt(context, calls)
        if orig is None:
            return SkillResult(
                skill_name="TiltCalibrate", success=False,
                error="读不到当前压电倾斜(Piezo_TiltGet)—— 无法标定,也无法回滚",
                nanonis_calls=calls)

        def restore():
            err = _write_tilt(context, calls, orig[0], orig[1])
            if err:
                logger.error("TiltCalibrate 恢复原倾斜失败: %s", err)

        try:
            base, err = _measure(context, probe)
            if base is None:
                return SkillResult(
                    skill_name="TiltCalibrate", success=False,
                    error=f"基线测量失败: {err}", nanonis_calls=calls)
            s0 = (base["tilt_x_deg"], base["tilt_y_deg"])

            responses = []
            for axis in (0, 1):
                if _operator_stopped(context):
                    # 软停:**只停,不动**。不在这里 restore —— 那是一次写类恢复,
                    # 会让压电再动一次;软停的语义是「停手」,把仪器搬回去是
                    # E_STOP / 用户的事(缺陷⑬ 要求四)。停在哪儿说清楚就是了。
                    return _stopped_by_operator(
                        "TiltCalibrate", calls,
                        f"压电倾斜停在试探步上(原值 {orig[0]:.4f}/{orig[1]:.4f}°,"
                        f"当前可能是轴 {axis} 上 +{step}° 的试探值)。"
                        "需要的话用 SetPiezoTilt 手动写回原值。")
                target = list(orig)
                target[axis] += step
                werr = _write_tilt(context, calls, target[0], target[1])
                if werr:
                    restore()
                    return SkillResult(
                        skill_name="TiltCalibrate", success=False,
                        error=f"施加试探步失败(轴 {axis}): {werr}",
                        nanonis_calls=calls)
                time.sleep(TILT_STEP_SETTLE_S)

                probed, err = _measure(context, probe)
                if probed is None:
                    restore()
                    return SkillResult(
                        skill_name="TiltCalibrate", success=False,
                        error=f"试探测量失败(轴 {axis}): {err}",
                        nanonis_calls=calls)
                # 响应是**二维向量**:x 轴的一步可能主要出现在测到的 y 上
                # (轴交换),这正是要把它解成矩阵而不是两个标量的原因。
                responses.append((
                    (probed["tilt_x_deg"] - s0[0]) / step,
                    (probed["tilt_y_deg"] - s0[1]) / step,
                ))
                restore()
                time.sleep(TILT_STEP_SETTLE_S)
        except Exception:
            # 任何意外都不能把仪器留在试探倾斜上 —— 那是个用户没同意过的状态。
            restore()
            raise

        # M 的列 = 每个 tilt 轴引起的测量斜率变化
        m = [[responses[0][0], responses[1][0]],
             [responses[0][1], responses[1][1]]]

        mag0 = math.hypot(responses[0][0], responses[0][1])
        mag1 = math.hypot(responses[1][0], responses[1][1])
        data: dict[str, Any] = {
            "step_deg": step,
            "response_x": list(responses[0]),
            "response_y": list(responses[1]),
            "response_mag": [mag0, mag1],
            "baseline_tilt_deg": list(s0),
            "original_tilt": list(orig),
        }

        for i, mag in enumerate((mag0, mag1)):
            if not (CALIB_RESPONSE_MIN <= mag <= CALIB_RESPONSE_MAX):
                return SkillResult(
                    skill_name="TiltCalibrate", success=False,
                    error=(
                        f"轴 {i} 的响应幅度 {mag:.3f} 不在合理区间 "
                        f"[{CALIB_RESPONSE_MIN}, {CALIB_RESPONSE_MAX}] —— "
                        "该轴可能没有响应,或响应异常。**未写入任何标定**;"
                        "先确认压电倾斜通道接线与当前位置是否够平。"),
                    data=data, nanonis_calls=calls)

        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        if abs(det) < 1e-9:
            return SkillResult(
                skill_name="TiltCalibrate", success=False,
                error="响应矩阵奇异(两轴响应共线),未写入标定",
                data=data, nanonis_calls=calls)

        # G = -M⁻¹:要抵消测到的斜率 s,施加 Δtilt = G·s。
        inv = [[m[1][1] / det, -m[0][1] / det],
               [-m[1][0] / det, m[0][0] / det]]
        g = [[-inv[0][0], -inv[0][1]], [-inv[1][0], -inv[1][1]]]

        # ⚠️ 算不出条件数就**让这个技能失败**,不要编一个数。
        # 这里以前兜底成 `cond = 1.0` —— 而 1.0 是条件数的**最优值**(完美条件),
        # 于是 `set_tilt_calibration` 里那道 `cond > TILT_CAL_MAX_COND` 的闸门
        # 被彻底解除:一次没能验证过的标定照样被持久化,而 `:276` 那句摘要还会把
        # 编造的「条件数 1.00」印给用户当测量值看。
        #
        # 兜底值不是中性的 —— 它恰好是最令人放心、也就是使检查失效的那个值。
        # 无法验证的标定不该被写进 profile:之后**每一次调平**都会用它。
        try:
            import numpy as np
            cond = float(np.linalg.cond(np.asarray(m, dtype=float)))
        except Exception as exc:  # noqa: BLE001
            data["matrix_m"] = m
            data["matrix_g"] = g
            return SkillResult(
                skill_name="TiltCalibrate", success=False,
                error=(f"无法计算响应矩阵的条件数({exc})—— 标定是否可靠**无从判断**,"
                       "拒绝写入。算不出条件数不是条件数良好的证据。"),
                data=data, nanonis_calls=calls)
        if not math.isfinite(cond):
            data["matrix_m"] = m
            data["matrix_g"] = g
            return SkillResult(
                skill_name="TiltCalibrate", success=False,
                error=(f"响应矩阵条件数非有限({cond})—— 两轴响应实际上共线,"
                       "解出来的矩阵不可靠。**未写入**。"),
                data=data, nanonis_calls=calls)
        data["matrix_m"] = m
        data["matrix_g"] = g
        data["cond"] = cond

        stored = instrument_profile.set_tilt_calibration(g, cond=cond)
        if stored is None:
            return SkillResult(
                skill_name="TiltCalibrate", success=False,
                error=(
                    f"标定被拒绝(条件数 {cond:.1f} 超过上限 "
                    f"{instrument_profile.TILT_CAL_MAX_COND})—— 两轴响应几乎"
                    "共线,解出来的矩阵不可靠。**未写入**。"),
                data=data, nanonis_calls=calls)

        data["stored"] = stored
        return SkillResult(
            skill_name="TiltCalibrate", success=True, data=data,
            summary=(f"倾斜响应标定完成:条件数 {cond:.2f},"
                     f"响应幅度 {mag0:.2f}/{mag1:.2f}"),
            nanonis_calls=calls)


class AutoTilt(BaseSkill):
    """Measure and compensate the sample tilt (closed loop, SmarTilt-style)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AutoTilt",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在一块平地上测出样品倾斜，用 piezo tilt 把它补偿掉，"
                "再复测一次验收。要不要补偿由它自己判（判据是：在你**接下来"
                "要扫的那一帧**上，这个斜坡吃掉多少 Z 量程），补偿分成限幅的小步"
                "施加，残差不收敛就**回滚**（ROLLS BACK）。前提是这台仪器上"
                "已经跑过 TiltCalibrate。"
                "图看起来是斜的、或者要换到更精细的尺度之前，就调它；"
                "你不需要给任何角度 —— 角度是它自己测的。"
            ),
            parameters=[
                ParameterSpec(
                    name="next_frame_m", type="float",
                    description=(
                        "这次调平要对多大的帧负责 —— 那一帧的边长。"
                        "触发判据是斜坡**在那一帧上**吃掉多少 Z 量程，"
                        "所以一张 1 um 的综览会比一张 10 nm 的特写"
                        "严得多。不填则用当前帧。"),
                    unit="m", required=False,
                    min_value=1e-10, max_value=1e-5,
                ),
                ParameterSpec(
                    name="radius_m", type="float",
                    description="测量圆的半径（见 TiltProbeCircle）。",
                    unit="m", required=False,
                    min_value=2e-9, max_value=5e-7,
                ),
                ParameterSpec(
                    name="n_points", type="int",
                    description="每个测量圆上取几个点。",
                    required=False, default=24, min_value=8, max_value=180,
                ),
                ParameterSpec(
                    name="max_iterations", type="int",
                    description="「补偿 + 验收」最多做几轮。",
                    required=False, default=MAX_ITERATIONS,
                    min_value=1, max_value=6,
                ),
                ParameterSpec(
                    name="surface_rms_m", type="float",
                    description=(
                        "表面自身高度起伏的 RMS，从一张扫描帧上量出来。"
                        "给了它之后，斜坡一旦把形貌淹没，也会触发一次"
                        "补偿（这正是「图明显看着是斜的」那种情况）。"
                        "没有帧可以拿来量的话就不填。"),
                    unit="m", required=False,
                    min_value=0.0, max_value=1e-6,
                ),
                ParameterSpec(
                    name="force", type="bool",
                    description=(
                        "即便实测倾斜对目标帧来说已经在预算之内，"
                        "也照样补偿。"),
                    required=False, default=False,
                ),
            ],
            preconditions=["z_controller_on", "scan_not_running"],
            estimated_duration_s=120.0,
            composition_level=3,
            tags=["tilt", "levelling", "composite"],
        )

    # ── 判据 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _thresholds(surface_rms_m: "float | None"):
        """(触发阈, 验收阈, 硬红线) —— 全部是「斜坡吃掉多少 Z」(米)。

        **有两个独立的调平理由,取「或」而不是「与」**:

          * **安全**:斜坡吃掉太多 Z 量程,帧角上有 rail / 撞边风险
            (``z_span > z_budget_frac × z_range``);
          * **数据质量**:斜坡把形貌淹没了 —— 这正是要求的「图明显是倾斜的」
            (``z_span > k_topo × 表面起伏``)。

        任一成立就该调平,所以触发阈取两者的**较小值**。之前这里写的是 max,
        等价于「两个都超才调」,而表面起伏那一项永远小得多 —— 于是「图明显倾斜」
        这个最主要的场景一次也触发不了。

        ``surface_rms_m`` 缺省(没有帧可参考,例如只做了一次圆测量)时只用安全
        判据。**不要把圆拟合的残差当表面起伏塞进来**:圆是特意跑在平地上的,
        它的残差按构造就是噪声,那样算出来的阈值是个噪声阈值,不是形貌阈值。
        """
        z_range = float(instrument_profile.get_config("z_range_m", 1.5e-6))
        safety = DEFAULT_Z_BUDGET_FRAC * z_range
        trigger = safety
        if surface_rms_m and float(surface_rms_m) > 0:
            trigger = min(trigger, DEFAULT_K_TOPO * float(surface_rms_m))
        return (trigger,
                trigger * ACCEPT_FRAC_OF_TRIGGER,
                Z_SPAN_HARD_LIMIT_FRAC * z_range)

    @staticmethod
    def _frame_diagonal(context, calls, params) -> float:
        explicit = params.get("next_frame_m")
        if explicit:
            side = float(explicit)
            return math.hypot(side, side)
        rec = context.safe_call("Scan_FrameGet")
        calls.append(rec)
        parsed = getattr(rec, "return_value", None)
        if not rec.error and isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 4:
                try:
                    return math.hypot(abs(float(vals[2])), abs(float(vals[3])))
                except (TypeError, ValueError):
                    pass
        return math.hypot(1e-7, 1e-7)      # 100 nm 兜底

    # ── 执行 ─────────────────────────────────────────────────────────────────

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        max_iter = int(params.get("max_iterations") or MAX_ITERATIONS)
        force = bool(params.get("force", False))
        probe = {k: params[k] for k in ("radius_m", "n_points")
                 if params.get(k) is not None}

        def report(outcome: str, reason: str, **extra) -> SkillResult:
            data = {"outcome": outcome, "reason": reason}
            data.update(extra)
            ok = outcome in ("applied", "no_action_needed")
            return SkillResult(
                skill_name="AutoTilt", success=ok,
                error="" if ok else f"{outcome}: {reason}",
                data=data, nanonis_calls=calls,
                summary=f"AutoTilt: {outcome}({reason})" if reason else
                        f"AutoTilt: {outcome}")

        # ① 标定 —— 没有它绝不动硬件
        calib = instrument_profile.get_tilt_calibration()
        if calib is None:
            return report(
                "skipped", "calibration_missing",
                next_action_hint="run_tilt_calibrate",
                detail=("这台仪器还没做过倾斜响应标定。Piezo_TiltSet 的轴对应与"
                        "符号取决于接线,猜错方向会把倾斜往反方向加倍 —— "
                        "先跑一次 TiltCalibrate。"))
        g = calib["g"]

        # ② 原始倾斜(回滚目标是**它**,不是 0)
        orig = _read_tilt(context, calls)
        if orig is None:
            return report("failed", "tilt_unreadable",
                          detail="读不到当前压电倾斜,无法安全回滚")

        diag = self._frame_diagonal(context, calls, params)

        # ③ 测量
        first, err = _measure(context, probe)
        if first is None:
            return report("skipped", "measure_failed", detail=err,
                          next_action_hint="survey_first")

        resolution = circle_tilt_resolution_deg(
            float(first.get("noise_floor_m") or 0.0),
            float(first.get("radius_m") or 1.0),
            int(first.get("n_points") or 1))

        trigger, accept, hard = self._thresholds(params.get("surface_rms_m"))
        span = z_span_for_frame(first["slope_mag_deg"], diag)
        z_range = float(instrument_profile.get_config("z_range_m", 1.5e-6))

        before = {
            "tilt_x_deg": orig[0], "tilt_y_deg": orig[1],
            "measured_slope_deg": first["slope_mag_deg"],
            "z_span_m": span,
            "z_span_frac": span / z_range if z_range > 0 else 0.0,
        }
        common = {
            # 用的是哪个矩阵,必须跟着每一份回包走 —— 否则拿到一句
            # 「diverged」的人无法判断是标定的问题还是控制律的问题,
            # 只能去查配置文件才能定位。
            "matrix_g": [list(row) for row in g],
            "matrix_g_cond": calib.get("cond"),
            "before": before, "frame_diagonal_m": diag,
            "trigger_z_span_m": trigger, "accept_z_span_m": accept,
            "hard_limit_z_span_m": hard,
            "measurement_resolution_deg": resolution,
            "site": {"x_m": first.get("center_x_m"),
                     "y_m": first.get("center_y_m"),
                     "radius_m": first.get("radius_m"),
                     "from": "current_position"},
        }

        # ④ 闸门
        if span <= trigger and not force:
            return report("no_action_needed",
                          "within_budget" if span < hard else "", **common)

        # 验收阈不能低于测量分辨率 —— 否则「残余倾斜没达标」只是在追噪声。
        min_accept_span = z_span_for_frame(resolution * 2.0, diag)
        if accept < min_accept_span:
            accept = min_accept_span
            common["accept_z_span_m"] = accept
            common["accept_raised_to_resolution"] = True

        # ⑤ 迭代:施加 → 复测 → 验收 / 收敛 / 回滚
        history = []
        current = list(orig)
        latest = first
        applied_any = False

        for i in range(max_iter):
            slope = (latest["tilt_x_deg"], latest["tilt_y_deg"])
            delta = (g[0][0] * slope[0] + g[0][1] * slope[1],
                     g[1][0] * slope[0] + g[1][1] * slope[1])

            # 限步:tilt 阶跃 → 扫描平面突转 → Z 瞬态。小步 + 反馈开着是防撞针的
            # 硬要求,拆成几步走比一次到位安全得多。
            mag = math.hypot(*delta)
            n_sub = max(1, math.ceil(mag / MAX_TILT_STEP_DEG))
            limit = float(instrument_profile.get_config("tilt_limit_deg", 5.0))

            truncated = False
            sub_step = (delta[0] / n_sub, delta[1] / n_sub)
            for _k in range(n_sub):
                if _operator_stopped(context):
                    # 软停:停在当前这一小步上,**不回滚**(见 _stopped_by_operator)。
                    # 小步本来就是为了「随时停下都还在安全范围内」而拆的。
                    return _stopped_by_operator(
                        "AutoTilt", calls,
                        f"倾斜停在 {current[0]:.4f}/{current[1]:.4f}°"
                        f"(第 {i + 1}/{max_iter} 轮的第 {_k}/{n_sub} 小步)。"
                        "每一小步都在限幅内,停在这里是安全的。")
                target = [current[0] + sub_step[0], current[1] + sub_step[1]]
                for axis in (0, 1):
                    if abs(target[axis]) > limit:
                        target[axis] = math.copysign(limit, target[axis])
                        truncated = True
                werr = _write_tilt(context, calls, target[0], target[1])
                if werr:
                    # 写失败 → 回到原始倾斜。注意这里**要**回滚:前面的小步已经
                    # 写进去了,把针尖留在一个走了一半的补偿量上比不补偿更糟。
                    _write_tilt(context, calls, orig[0], orig[1])
                    return report("failed", "hw_reject", detail=werr,
                                  history=history, **common)
                current = target
                applied_any = True
                time.sleep(TILT_STEP_SETTLE_S)

            verify, err = _measure(context, probe)
            if verify is None:
                _write_tilt(context, calls, orig[0], orig[1])
                return report("rolled_back", "verify_failed", detail=err,
                              history=history, **common)

            new_span = z_span_for_frame(verify["slope_mag_deg"], diag)
            # 逐轮必须留下**能重算这一步**的东西,不只是结果。
            #
            # 曾经的回包只有 `rolled_back(diverged)` 加一串**幅度**
            # (`residual_slope_deg` 是 `slope_mag_deg`,一个标量),
            # 于是「这一轮到底往哪个方向走了多少」在回包里**根本不存在** ——
            # 只能靠查配置文件拿矩阵、再反推谱半径来定位。
            #
            # **一个只报结论不报过程的判据,在它出错时无法被诊断。**
            # 下面这四项合起来足以离线重放一轮:slope → delta → 施加 → 复测。
            history.append({
                "iteration": i + 1,
                "slope_in_deg": list(slope),          # 这一轮读到的斜率**向量**
                "delta_tilt_deg": list(delta),        # G·slope 算出的增量
                "n_sub_steps": n_sub,
                "applied_tilt": list(current),
                "residual_slope_vec_deg": [verify.get("tilt_x_deg"),
                                           verify.get("tilt_y_deg")],
                "residual_slope_deg": verify["slope_mag_deg"],
                "residual_z_span_m": new_span,
                "truncated_at_limit": truncated,
            })

            if new_span <= accept:
                return report(
                    "applied", "",
                    after={"tilt_x_deg": current[0], "tilt_y_deg": current[1],
                           "measured_slope_deg": verify["slope_mag_deg"],
                           "z_span_m": new_span},
                    applied={"tilt_x_deg": current[0], "tilt_y_deg": current[1]},
                    iterations=i + 1, history=history, **common)

            if new_span > span * CONVERGENCE_RATIO:
                # 没在收敛 —— 标定失效 / 表面变了 / 针尖事件。回到**原始**倾斜。
                _write_tilt(context, calls, orig[0], orig[1])
                return report(
                    "rolled_back", "diverged",
                    # ``detail`` 是给**人**读的那句话（它会被旁白原样念出去），
                    # 所以数字走可读 SI，不是 `%g` —— 后者把 6.4 nm 印成
                    # `6.4e-09 m`，一个用户当场比不出大小的形状。
                    detail=(f"第 {i + 1} 轮后残余 Z 占用 "
                            f"{format_si_readable(new_span, 'm')},"
                            f"未降到上一轮 {format_si_readable(span, 'm')} 的 "
                            f"{CONVERGENCE_RATIO:.0%} 以下"),
                    iterations=i + 1, history=history, **common)

            span = new_span
            latest = verify

        # 迭代用尽仍未达标:保留已改善的结果,但如实说没达标。
        return report(
            "failed", "not_converged",
            detail=f"{max_iter} 轮后残余 Z 占用 "
                   f"{format_si_readable(span, 'm')} 仍高于验收阈 "
                   f"{format_si_readable(accept, 'm')}",
            after={"tilt_x_deg": current[0], "tilt_y_deg": current[1],
                   "z_span_m": span},
            applied={"tilt_x_deg": current[0], "tilt_y_deg": current[1]}
            if applied_any else None,
            iterations=max_iter, history=history, **common)
