# -*- coding: utf-8 -*-
"""多扫描角采图后进行压电与漂移定标。

压电畸变与热漂移对扫描角的依赖不同，因此逐角采集并将合格帧交给 CalibratePiezoMultiAngle。
ConfigureScan 不设置速度；速度单独下发，角度和线时间均读回核对。"""
from __future__ import annotations

import logging
from typing import Iterator

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)

logger = logging.getLogger(__name__)

__all__ = ["AcquireAngleSeriesForCalibration"]

#: 角度张开度低于这个值时压电与漂移在方程里分不开，采了也白采 —— 所以在
#: **动仪器之前**就拒绝，而不是采完再报「解不出来」。
_MIN_SPREAD_DEG = 15.0


def _parse_angles(raw) -> list[float]:
    if raw is None or str(raw).strip() == "":
        return [0.0, 45.0, 90.0]
    out = []
    for part in str(raw).replace("；", ",").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


class AcquireAngleSeriesForCalibration(CompositeSkillGraph):
    """换几个扫描角各扫一帧，然后解出压电畸变与热漂移。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireAngleSeriesForCalibration",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "多扫描角采原子分辨图并解出压电 x/y 尺度、非正交与热漂移。"
                "换角度是把压电与漂移分开的唯一手段 —— 单帧原理上做不到。"
            ),
            parameters=[
                ParameterSpec(
                    name="angles_deg", type="str", required=False,
                    default="0,45,90",
                    description=("扫描角列表，逗号分隔。至少两个、张开 ≥15°；"
                                 "0/45/90 已经够用（cond≈3），六个角度只是更稳"),
                ),
                ParameterSpec(
                    name="center_x_m", type="float", unit="m", required=False,
                    default=None, description="扫描中心 X；不填用当前扫描框",
                ),
                ParameterSpec(
                    name="center_y_m", type="float", unit="m", required=False,
                    default=None, description="扫描中心 Y；不填用当前扫描框",
                ),
                ParameterSpec(
                    name="scan_size_m", type="float", unit="m", required=False,
                    default=None, min_value=1e-10, max_value=1e-7,
                    description="视场边长；不填用当前扫描框",
                ),
                ParameterSpec(
                    name="line_time_s", type="float", unit="s", required=False,
                    default=None, min_value=1e-3, max_value=60.0,
                    description=("每线时间；不填走出厂档位表（8 nm → atomic 档 "
                                 "1.2 s/线 ≈ 6.7 nm/s）。原子分辨要慢扫"),
                ),
                ParameterSpec(
                    name="surface", type="str", required=False, default="Au(111)",
                    description="表面（决定理论晶格常数）",
                ),
                ParameterSpec(
                    name="retries_per_angle", type="int", required=False,
                    default=1, min_value=1, max_value=5,
                    description=("每个角度最多扫几帧。原子分辨有随机性 ——"
                                 "「多扫一会儿有时候就出来了」"),
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=3600.0,
            composition_level=4,
            tags=["piezo", "calibration", "atomic", "multiframe", "composite"],
        )

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        angles = _parse_angles(params.get("angles_deg"))
        retries = int(params.get("retries_per_angle") or 1)
        surface = str(params.get("surface") or "Au(111)")

        if len(angles) < 2:
            executor.set_partial("refused", "need_two_angles")
            logger.error("AcquireAngleSeriesForCalibration: 至少要两个扫描角")
            return

        # **动仪器之前**就检查角度张不张得开。采完再说「解不出来」等于白扔
        # 一小时机时，而这个检查不需要任何硬件调用。
        try:
            from mast.vision.lattice_multiframe import angle_conditioning
            spread, cond = angle_conditioning(angles)
        except Exception:  # noqa: BLE001
            spread, cond = 999.0, 1.0
        if spread < _MIN_SPREAD_DEG:
            executor.set_partial("refused", "angles_too_close")
            executor.set_partial("angle_spread_deg", spread)
            logger.error(
                "AcquireAngleSeriesForCalibration: 这几个角度只张开 %.1f°，"
                "压电与漂移分不开 —— 采了也解不出来，所以现在就停。", spread)
            return
        executor.set_partial("angle_spread_deg", spread)
        executor.set_partial("condition_number", cond)

        cx = params.get("center_x_m")
        cy = params.get("center_y_m")
        size = params.get("scan_size_m")
        if cx is None or cy is None or size is None:
            fd = {}
            try:
                ctx = getattr(executor, "context", None)
                if ctx is not None:
                    fd = getattr(ctx.run("GetScanFrame", {}), "data", None) or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("读扫描框失败: %s", exc)
            cx = cx if cx is not None else fd.get("center_x_m")
            cy = cy if cy is not None else fd.get("center_y_m")
            size = size if size is not None else fd.get("width_m")
        if cx is None or cy is None or not size:
            executor.set_partial("refused", "no_scan_frame")
            logger.error("读不到扫描框且调用方没给 —— 不猜一个框去扫。")
            return
        cx, cy, size = float(cx), float(cy), float(size)

        # 线时间：不填就查出厂档位表（同一份解析器，三个调用点共用）
        lt = params.get("line_time_s")
        try:
            from mast.core.scan_policy import resolve_line_time
            line_time, lt_source = resolve_line_time(lt, size)
        except Exception:  # noqa: BLE001
            line_time, lt_source = (float(lt) if lt else 1.2), "fallback"
        executor.set_partial("line_time_s", line_time)
        executor.set_partial("line_time_source", lt_source)

        speed = size / line_time
        done_angles = list(executor.progress.partial_data.get("done_angles", []))
        frames = list(executor.progress.partial_data.get("frames", []))
        executor.set_total_steps(len(angles) * retries * 5)

        for a in angles:
            if a in done_angles:
                continue
            got = False
            for k in range(retries):
                tag = "a%g_%d" % (a, k + 1)

                yield CompositeStep(
                    step_id=f"{tag}:configure", skill_name="ConfigureScan",
                    params={"center_x_m": cx, "center_y_m": cy,
                            "width_m": size, "height_m": size,
                            "angle_deg": float(a),
                            # 关键：不让它顺手改速度（见模块注释第 1 条）
                            "set_scan_speed": False},
                    optional=True, checkpoint_after=False,
                    tags=("angle", f"angle={a}", "configure"),
                )
                if executor.progress.aborted:
                    return

                yield CompositeStep(
                    step_id=f"{tag}:speed", skill_name="SetScanSpeed",
                    # fwd_speed / bwd_speed 必填（模块注释第 2 条）
                    params={"fwd_speed": speed, "bwd_speed": speed,
                            "fwd_line_time": line_time,
                            "bwd_line_time": line_time,
                            "keep_const": 1, "speed_ratio": 1.0},
                    optional=True, checkpoint_after=False,
                    tags=("angle", f"angle={a}", "speed"),
                )
                if executor.progress.aborted:
                    return

                # 读回核对：角度与线时间都必须真的生效（模块注释第 3 条）
                bad = ""
                try:
                    ctx = getattr(executor, "context", None)
                    if ctx is not None:
                        fr = getattr(ctx.run("GetScanFrame", {}), "data", None) or {}
                        sp = getattr(ctx.run("GetScanSpeed", {}), "data", None) or {}
                        if abs(float(fr.get("angle_deg", -999)) - a) > 0.5:
                            bad = "角度没跟上：要 %.1f，读回 %s" % (a, fr.get("angle_deg"))
                        elif abs(float(sp.get("fwd_time_s", 0)) - line_time) \
                                > 0.05 * line_time:
                            bad = ("线时间没跟上：要 %.4f s，读回 %s"
                                   % (line_time, sp.get("fwd_time_s")))
                except Exception as exc:  # noqa: BLE001
                    bad = "读回失败: %s" % exc
                if bad:
                    logger.error("AcquireAngleSeriesForCalibration %s: %s", tag, bad)
                    frames.append({"angle_deg": a, "attempt": k + 1,
                                   "ok": False, "error": bad})
                    executor.set_partial("frames", list(frames))
                    continue

                yield CompositeStep(
                    step_id=f"{tag}:scan", skill_name="FullScan",
                    params={"center_x_m": cx, "center_y_m": cy,
                            "width_m": size, "height_m": size,
                            "line_time_s": line_time},
                    optional=True, checkpoint_after=False,
                    tags=("angle", f"angle={a}", "scan"),
                )
                if executor.progress.aborted:
                    return

                yield CompositeStep(
                    step_id=f"{tag}:save", skill_name="SaveScan", params={},
                    optional=True, checkpoint_after=True,
                    tags=("angle", f"angle={a}", "save"),
                )
                if executor.progress.aborted:
                    return

                sr = executor.sub_results.get(f"{tag}:save")
                path = (getattr(sr, "data", None) or {}).get("saved_path")
                if not path:
                    yield CompositeStep(
                        step_id=f"{tag}:latest", skill_name="GetLatestScanFile",
                        params={}, optional=True, checkpoint_after=False,
                        tags=("angle", f"angle={a}", "latest"),
                    )
                    lr = executor.sub_results.get(f"{tag}:latest")
                    path = (getattr(lr, "data", None) or {}).get("path")
                if not path:
                    frames.append({"angle_deg": a, "attempt": k + 1,
                                   "ok": False, "error": "没拿到文件路径"})
                    executor.set_partial("frames", list(frames))
                    continue

                frames.append({"angle_deg": a, "attempt": k + 1,
                               "ok": True, "path": str(path)})
                executor.set_partial("frames", list(frames))
                got = True
                break

            if got:
                done_angles.append(a)
                executor.set_partial("done_angles", list(done_angles))

        good = [f["path"] for f in frames if f.get("ok") and f.get("path")]
        if len(good) >= 2:
            yield CompositeStep(
                step_id="calibrate", skill_name="CalibratePiezoMultiAngle",
                params={"scan_paths": ",".join(good), "surface": surface},
                optional=True, checkpoint_after=True, tags=("calibrate",),
            )

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        frames = list(pd.get("frames", []))
        cal = sub_results.get("calibrate")
        cd = getattr(cal, "data", None) or {}
        out = {
            "refused": pd.get("refused"),
            "angle_spread_deg": pd.get("angle_spread_deg"),
            "condition_number": pd.get("condition_number"),
            "line_time_s": pd.get("line_time_s"),
            "line_time_source": pd.get("line_time_source"),
            "frames": frames,
            "n_frames_ok": sum(1 for f in frames if f.get("ok")),
            "calibration": cd or None,
        }
        if pd.get("refused") == "angles_too_close":
            out["advice"] = (
                "扫描角只张开 %.1f°，压电与漂移分不开 —— 一帧都没扫。"
                "把角度拉开到 ≥15°（0/45/90 就很好）。"
                % (pd.get("angle_spread_deg") or 0.0))
        elif out["n_frames_ok"] < 2:
            out["advice"] = (
                "只成功了 %d 帧，解不了。先用 ScanUntilAtomicResolution 确认"
                "这个位置能出原子分辨，再回来做定标。" % out["n_frames_ok"])
        elif cd.get("ok"):
            out["advice"] = (
                "把 X / Y 压电灵敏度分别乘以 %.4f / %.4f。**本技能不写仪器** —— "
                "改灵敏度要人来决定。剪切 %+.2f° 已与热漂移分离（漂移另报）。"
                % (cd.get("x_scale") or 1.0, cd.get("y_scale") or 1.0,
                   cd.get("piezo_shear_deg") or 0.0))
        return out
