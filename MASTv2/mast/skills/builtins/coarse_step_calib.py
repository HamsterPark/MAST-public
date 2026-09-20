# -*- coding: utf-8 -*-
"""CalibrateCoarseStep —— 粗动一步走多远？用有刻度的压电去量没刻度的马达。

## 为什么需要（本机的一个真实盲区）

粗动**没有位置反馈**：`Motor_GetPos` / `Motor_StepCounterGet` 在本机 PMD 上都回
「Cannot access Motor Control Module」（那是「本控制器不支持」，不是「模块没运行」）。
于是「走了多远」只能靠步数推算，而步数与距离的比例从来没人量过。

压电是标定过的。所以：**在表面上扎一个坑当基准，粗动 N 步，再扫同一个压电中心，
用整幅图的相位相关求位移** —— 位移 ÷ N 就是每步多远。

## 三条从失败里学到的设计（2026-08-28，三次都没跑完）

1. **不能用 `RelocateCoarseXY`。** 它的前置检查要求落点离已访问站点 ≥200 步，
   而标定要的正是小步长 ⇒ 直接被拒。那是「别重复访问」的效率约束，不是安全约束；
   安全那部分（降压、退针、**验电流真的归零**）本技能自己做。
2. **步数要很小。** 第一次用 20 步、800 nm 视野，什么都没看到。
   结合「y+ 走 50 步就撞到表面」反推：若样品倾斜约 1°，每步可能是 **100 nm 量级**，
   20 步 = 2 µm 早跑出视野。⇒ 默认 2–4 步、视野 1500 nm。
3. **动马达之前必须确认扫描真的结束了。** 曾经本地脚本已经停了，
   而仪器端的 `ScanAt` 还持有锁 363 秒 —— 期间针尖撞上东西、放大器锁死在 10 nA，
   而当时以为已经在撤针。**「命令被挡在门外」和「命令发出去但没效果」长得一样。**

## 判据

位移用相位相关的**峰值锐度**自证：峰/中位比太低说明两帧之间没有可对齐的共同特征
（坑跑出视野，或表面太平），那时**拒答**而不是报一个凑出来的数。
"""

from __future__ import annotations

import logging
import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 相位相关峰的锐度下限（峰/中位）。低于它说明两帧没有可对齐的共同特征。
_MIN_CORR_SNR = 12.0

#: 退针后判「针尖确实自由了」的电流上限（A）。
_CLEAR_CURRENT_A = 1e-12

#: 横移期间的安全偏压（V）。同 relocate：带成像偏压时几十 nm 就场发射。
_MOVE_BIAS_V = 0.5


def phase_shift(a, b):
    """b 相对 a 的整体平移（像素）+ 峰锐度。用相位相关，对亮度变化不敏感。"""
    import numpy as np

    A = np.nan_to_num(np.asarray(a, float) - np.nanmean(a))
    B = np.nan_to_num(np.asarray(b, float) - np.nanmean(b))
    if A.shape != B.shape or min(A.shape) < 16:
        return None, None, None
    w = np.hanning(A.shape[0])[:, None] * np.hanning(A.shape[1])[None, :]
    R = np.fft.fft2(A * w) * np.conj(np.fft.fft2(B * w))
    R /= np.maximum(np.abs(R), 1e-30)
    c = np.fft.fftshift(np.real(np.fft.ifft2(R)))
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    snr = float(c.max() / max(float(np.median(np.abs(c))), 1e-12))
    return int(ix - c.shape[1] // 2), int(iy - c.shape[0] // 2), snr


class CalibrateCoarseStep(BaseSkill):
    """Measure how far one coarse motor step travels, using the piezo as the ruler."""

    _settle_s = 1.5

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CalibrateCoarseStep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "量**粗动一步走多远**（x/y 分别量，两者可以不一样）。"
                "本机粗动没有位置反馈（PMD 不支持 `Motor_GetPos`/`StepCounterGet`），"
                "所以拿**有标定的压电当尺子**：扎个坑做基准 → 粗动 N 步 → 回扫同一个"
                "压电中心 → 整幅图相位相关求位移 → 位移÷N。"
                "\n\n**不要拿 `RelocateCoarseXY` 做这件事**：它要求落点离已访问站点 ≥200 步，"
                "而标定要的正是小步长。那是「别重复访问」的效率约束；安全那部分"
                "（降压、退针、**验电流真的归零**）本技能自己做并逐条报出来。"
                "\n\n**步数要小。** 2026-08-28 反推：若样品倾斜约 1°，每步可能是 100 nm 量级 —— "
                "20 步就是 2 µm，早跑出视野，怎么扫都看不到位移。默认 2–4 步 / 1500 nm 视野。"
                "\n\n**位移自证**：相位相关峰的锐度低于 %.0f 就**拒答**，"
                "而不是报一个凑出来的数 —— 那说明两帧之间没有可对齐的共同特征"
                "（坑跑出视野，或表面太平）。"
                "\n\n⚠️ 本技能的方法在 2026-08-28 夜三次尝试都被别的问题打断"
                "（站点检查、视野太小、仪器锁），**尚未在真机上跑完一次完整标定**。"
                % _MIN_CORR_SNR
            ),
            parameters=[
                ParameterSpec(
                    name="axis", type="str",
                    description="'x' 或 'y' —— 两个方向要分别量。",
                    required=True, allowed_values=["x", "y"]),
                ParameterSpec(
                    name="steps", type="str",
                    description=("要测的步数，逗号分隔。默认 '2,4' —— 两个不同步数"
                                 "才能自证线性（位移应当与步数成比例）。"),
                    required=False, default="2,4"),
                ParameterSpec(
                    name="size_m", type="float", unit="m",
                    description="扫描视野。要能容下预期位移的 2–3 倍。",
                    required=False, default=1.5e-6,
                    min_value=1e-7, max_value=2.5e-6),
                ParameterSpec(
                    name="make_pit", type="bool",
                    description=("先扎一个坑做基准特征。表面本来就有足够特征时可以关掉 —— "
                                 "但**平坦表面上相位相关立不住**，那正是坑的用处。"),
                    required=False, default=True),
            ],
            estimated_duration_s=2400.0,
            composition_level=2,
            tags=["coarse", "calibration", "motor", "粗动", "标定"],
        )

    # ── 步骤 ──────────────────────────────────────────────────────────
    def _lock_is_free(self, context):
        """动马达之前先确认没有别的链路握着仪器（08-28 撞针的直接原因）。"""
        res = context.run("MotorMove", {"direction": "z-retract", "steps": 0})
        err = str(getattr(res, "error", "") or "")
        return ("占用" not in err and "busy" not in err.lower()), err

    def _clear(self, context):
        """降压 → 退针 → **证明**电流归零。证不出来就不动马达。"""
        context.run("SetBias", {"bias_v": _MOVE_BIAS_V})
        time.sleep(0.5)
        context.run("WithdrawTip", {})
        time.sleep(self._settle_s)
        cur = (getattr(context.run("GetCurrent", {}), "data", None) or {}).get("current_a")
        ok = cur is not None and abs(float(cur)) <= _CLEAR_CURRENT_A
        return ok, cur

    def _scan(self, context, cx_m, cy_m, size_m):
        res = context.run("ScanAt", {
            "center_x_m": cx_m, "center_y_m": cy_m, "size_m": size_m,
            "pixels": 256, "line_time_s": 0.04, "purpose": "survey"})
        d = getattr(res, "data", None) or {}
        return d.get("scan_path") or d.get("path")

    @staticmethod
    def _load(path):
        import numpy as np

        from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
        from mast.vision.tilt import plane_subtract

        fr = sxm_oriented_frames(read_sxm(path), channel="Z")
        return np.asarray(plane_subtract(np.asarray(fr["forward"], float)), float)

    def execute(self, context, params: dict) -> SkillResult:
        axis = str(params.get("axis") or "").lower()
        if axis not in ("x", "y"):
            return SkillResult(skill_name="CalibrateCoarseStep", success=False,
                               error="axis 只能是 'x' 或 'y'（给了 %r）" % params.get("axis"))
        try:
            steps_list = [int(t) for t in str(params.get("steps") or "2,4").split(",") if t.strip()]
        except ValueError:
            return SkillResult(skill_name="CalibrateCoarseStep", success=False,
                               error="steps 解析不了：%r" % params.get("steps"))
        steps_list = [s for s in steps_list if s > 0]
        if len(steps_list) < 2:
            return SkillResult(
                skill_name="CalibrateCoarseStep", success=False,
                error=("至少要两个不同步数 —— 单个步数算得出一个数，"
                       "但**没有任何东西能证明它是线性的**（位移该与步数成比例）。"))
        size_m = float(params.get("size_m") or 1.5e-6)

        free, err = self._lock_is_free(context)
        if not free:
            return SkillResult(
                skill_name="CalibrateCoarseStep", success=False,
                error="仪器正被另一条链路占用，不能动马达：" + err[:200],
                data={"blocked_by_lock": True})

        fr = getattr(context.run("GetScanFrame", {}), "data", None) or {}
        cx_m = float(fr.get("center_x_m") or 0.0)
        cy_m = float(fr.get("center_y_m") or 0.0)
        nm_px = size_m * 1e9 / 256.0

        if params.get("make_pit", True):
            for _ in range(4):
                context.run("TipShape", {
                    "tip_lift_m": -10e-9, "lift_height_m": 10e-9,
                    "lift_time_1_s": 0.1, "lift_time_2_s": 0.1, "bias_lift_v": 0.02,
                    "change_bias": False, "bias_settling_s": 0.5,
                    "end_wait_s": 0.2, "restore_feedback": True})
                time.sleep(0.4)

        ref_path = self._scan(context, cx_m, cy_m, size_m)
        if not ref_path:
            return SkillResult(skill_name="CalibrateCoarseStep", success=False,
                               error="基准帧扫描失败。")
        prev = self._load(ref_path)

        runs = []
        for n in steps_list:
            ok, cur = self._clear(context)
            if not ok:
                runs.append({"steps": n, "abort": "清障未通过",
                             "current_a": cur})
                break
            res = context.run("MotorMove", {"direction": "%s+" % axis, "steps": int(n)})
            if not getattr(res, "success", False):
                runs.append({"steps": n, "abort": "MotorMove 失败",
                             "error": str(getattr(res, "error", ""))[:180]})
                break
            ap = context.run("ApproachTip", {})
            if not getattr(ap, "success", False):
                runs.append({"steps": n, "abort": "重新进针失败"})
                break
            path = self._scan(context, cx_m, cy_m, size_m)
            if not path:
                runs.append({"steps": n, "abort": "扫描失败"})
                break
            img = self._load(path)
            dx, dy, snr = phase_shift(prev, img)
            row = {"steps": n, "dx_nm": None if dx is None else dx * nm_px,
                   "dy_nm": None if dy is None else dy * nm_px, "corr_snr": snr}
            if snr is not None and snr < _MIN_CORR_SNR:
                row["refused"] = (
                    "相关峰锐度 %.1f < %.0f —— 两帧之间没有可对齐的共同特征，"
                    "**不报位移**。多半是位移超出视野（把 steps 调小或 size 调大），"
                    "或表面太平（开 make_pit）。" % (snr, _MIN_CORR_SNR))
            else:
                row["nm_per_step_x"] = row["dx_nm"] / n
                row["nm_per_step_y"] = row["dy_nm"] / n
            runs.append(row)
            prev = img

        good = [r for r in runs if r.get("nm_per_step_x") is not None]
        data = {"axis": axis, "size_m": size_m, "nm_per_px": nm_px, "runs": runs,
                "min_corr_snr": _MIN_CORR_SNR}
        if len(good) < 2:
            data["verdict"] = "undetermined"
            data["message"] = ("可用测量不足 2 个 —— **判不了**。单点算得出一个数，"
                               "但证不了线性；这里不给那个数。")
            return SkillResult(skill_name="CalibrateCoarseStep", success=True, data=data)

        # 线性自证：位移÷步数在不同步数下应当一致
        key = "nm_per_step_%s" % axis
        vals = [abs(r[key]) for r in good]
        spread = (max(vals) - min(vals)) / max(sum(vals) / len(vals), 1e-12)
        data["nm_per_step"] = sum(vals) / len(vals)
        data["spread_frac"] = spread
        data["verdict"] = "ok" if spread <= 0.35 else "nonlinear"
        data["message"] = (
            "%s 轴每步 ≈ %.1f nm（%d 个步数，彼此差 %.0f%%）。%s"
            % (axis, data["nm_per_step"], len(good), 100 * spread,
               "" if spread <= 0.35 else
               " **不同步数给出的每步距离差得太多** —— 可能有滑移或黏滑不均，"
               "这个平均值别当成刻度用。"))
        return SkillResult(skill_name="CalibrateCoarseStep", success=True, data=data)
