# -*- coding: utf-8 -*-
"""在同一位置和设定电流下比较偏压序列的成像质量。

正负偏压交错排列，以减小缓慢漂变对极性比较的影响；末尾重复首个条件作为时间漂变对照。"""
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

__all__ = ["AcquireBiasImagingSeries"]


def _parse_biases(raw) -> list[float]:
    if raw is None or str(raw).strip() == "":
        return [0.02, -0.02, 0.1, -0.1]
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


def _interleave(vs: list[float]) -> list[float]:
    """按 |V| 升序、同一 |V| 正负相邻。"""
    seen, out = set(), []
    for v in sorted(vs, key=lambda x: (abs(x), -x)):
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


class AcquireBiasImagingSeries(CompositeSkillGraph):
    """变偏压扫一组图，逐帧判定原子分辨质量。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireBiasImagingSeries",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "同一位置变偏压扫一组图并逐帧判定成像质量。正负交错 + 末尾重复"
                "首条件，让偏压效应与针尖随时间的漂变可以分开。"
            ),
            parameters=[
                ParameterSpec(
                    name="biases_v", type="str", required=False,
                    default="0.02,-0.02,0.1,-0.1",
                    description="偏压列表，逗号分隔。会自动按 |V| 升序、正负交错",
                ),
                ParameterSpec(
                    name="setpoint_a", type="float", unit="A", required=False,
                    default=None, min_value=1e-12, max_value=5e-9,
                    description="全程固定的电流设定点；不填就用当前值",
                ),
                ParameterSpec(
                    name="repeat_first_at_end", type="bool", required=False,
                    default=True,
                    description=("末尾把第一个偏压再扫一遍。**默认 True** —— "
                                 "没有它，跨 |V| 的趋势与针尖随时间的劣化分不开"),
                ),
                ParameterSpec(
                    name="center_x_m", type="float", unit="m", required=False,
                    default=None, description="扫描中心 X；不填用当前扫描框"),
                ParameterSpec(
                    name="center_y_m", type="float", unit="m", required=False,
                    default=None, description="扫描中心 Y；不填用当前扫描框"),
                ParameterSpec(
                    name="scan_size_m", type="float", unit="m", required=False,
                    default=None, min_value=1e-10, max_value=1e-7,
                    description="视场边长；不填用当前扫描框"),
                ParameterSpec(
                    name="line_time_s", type="float", unit="s", required=False,
                    default=None, min_value=1e-3, max_value=60.0,
                    description="每线时间；不填走出厂档位表"),
                ParameterSpec(
                    name="settle_s", type="float", unit="s", required=False,
                    default=3.0, min_value=0.0, max_value=120.0,
                    description="改完偏压等多久再开扫"),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=3600.0,
            composition_level=4,
            tags=["bias", "atomic", "imaging", "series", "composite"],
        )

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        biases = _interleave(_parse_biases(params.get("biases_v")))
        if len(biases) < 2:
            executor.set_partial("refused", "need_two_biases")
            logger.error("AcquireBiasImagingSeries: 至少要两个偏压")
            return
        if bool(params.get("repeat_first_at_end", True)):
            biases = biases + [biases[0]]
        executor.set_partial("bias_order", list(biases))

        cx = params.get("center_x_m")
        cy = params.get("center_y_m")
        size = params.get("scan_size_m")
        sp = params.get("setpoint_a")
        ctx = getattr(executor, "context", None)
        if cx is None or cy is None or size is None:
            fd = {}
            try:
                if ctx is not None:
                    fd = getattr(ctx.run("GetScanFrame", {}), "data", None) or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("读扫描框失败: %s", exc)
            cx = cx if cx is not None else fd.get("center_x_m")
            cy = cy if cy is not None else fd.get("center_y_m")
            size = size if size is not None else fd.get("width_m")
        if sp is None:
            try:
                if ctx is not None:
                    sd = getattr(ctx.run("GetSetpoint", {}), "data", None) or {}
                    sp = sd.get("setpoint_a")
            except Exception:  # noqa: BLE001
                sp = None
        if cx is None or cy is None or not size:
            executor.set_partial("refused", "no_scan_frame")
            logger.error("读不到扫描框且调用方没给 —— 不猜一个框去扫。")
            return
        cx, cy, size = float(cx), float(cy), float(size)

        try:
            from mast.core.scan_policy import resolve_line_time
            line_time, lt_src = resolve_line_time(params.get("line_time_s"), size)
        except Exception:  # noqa: BLE001
            line_time, lt_src = 1.2, "fallback"
        speed = size / line_time
        settle = float(params.get("settle_s") or 3.0)
        executor.set_partial("line_time_s", line_time)
        executor.set_partial("line_time_source", lt_src)

        frames = list(executor.progress.partial_data.get("frames", []))
        executor.set_total_steps(len(biases) * 5)

        for i, bv in enumerate(biases):
            tag = "b%d" % (i + 1)

            yield CompositeStep(
                step_id=f"{tag}:bias", skill_name="SetBias",
                params={"bias_v": float(bv)}, optional=True,
                checkpoint_after=False, tags=("bias", f"bias={bv}", "set"),
            )
            if executor.progress.aborted:
                return
            # 整定等待走基类的 ``abortable_sleep``，**不是**一个叫 Wait 的技能 ——
            # 全仓没有那个技能，写成步骤的话每一轮都会被「未知技能」拒掉，
            # 而 optional=True 会让这件事悄悄过去（第一版就是这么写的）。
            if settle > 0 and ctx is not None:
                self.abortable_sleep(ctx, settle)

            # 偏压是这个实验唯一的自变量。它没跟上，这一帧就没有意义 ——
            # 而 SetBias 报 success 只说明命令发出去了。
            bad = ""
            try:
                if ctx is not None:
                    got = (getattr(ctx.run("GetBias", {}), "data", None) or {}).get("bias_v")
                    tol = max(1e-3, abs(bv) * 0.05)
                    if got is None or abs(float(got) - float(bv)) > tol:
                        bad = "偏压没跟上：要 %.4g，读回 %s" % (bv, got)
                    elif sp:
                        now = (getattr(ctx.run("GetSetpoint", {}), "data", None)
                               or {}).get("setpoint_a")
                        # setpoint 必须全程不变，否则「只变偏压」这句话不成立
                        if now and abs(abs(float(now)) - abs(float(sp))) / abs(float(sp)) > 0.05:
                            bad = ("setpoint 被带偏了：%.4g -> %.4g"
                                   % (float(sp), float(now)))
            except Exception as exc:  # noqa: BLE001
                bad = "读回失败: %s" % exc
            if bad:
                logger.error("AcquireBiasImagingSeries %s: %s", tag, bad)
                frames.append({"bias_v": bv, "index": i, "ok": False, "error": bad})
                executor.set_partial("frames", list(frames))
                continue

            yield CompositeStep(
                step_id=f"{tag}:configure", skill_name="ConfigureScan",
                params={"center_x_m": cx, "center_y_m": cy,
                        "width_m": size, "height_m": size,
                        "set_scan_speed": False},
                optional=True, checkpoint_after=False,
                tags=("bias", f"bias={bv}", "configure"),
            )
            yield CompositeStep(
                step_id=f"{tag}:speed", skill_name="SetScanSpeed",
                params={"fwd_speed": speed, "bwd_speed": speed,
                        "fwd_line_time": line_time, "bwd_line_time": line_time,
                        "keep_const": 1, "speed_ratio": 1.0},
                optional=True, checkpoint_after=False,
                tags=("bias", f"bias={bv}", "speed"),
            )
            yield CompositeStep(
                step_id=f"{tag}:scan", skill_name="FullScan",
                params={"center_x_m": cx, "center_y_m": cy,
                        "width_m": size, "height_m": size,
                        "line_time_s": line_time},
                optional=True, checkpoint_after=False,
                tags=("bias", f"bias={bv}", "scan"),
            )
            if executor.progress.aborted:
                return
            yield CompositeStep(
                step_id=f"{tag}:save", skill_name="SaveScan", params={},
                optional=True, checkpoint_after=True,
                tags=("bias", f"bias={bv}", "save"),
            )
            sr = executor.sub_results.get(f"{tag}:save")
            path = (getattr(sr, "data", None) or {}).get("saved_path")
            if not path:
                frames.append({"bias_v": bv, "index": i, "ok": False,
                               "error": "没拿到文件路径"})
                executor.set_partial("frames", list(frames))
                continue

            yield CompositeStep(
                step_id=f"{tag}:assess", skill_name="AssessAtomicResolution",
                params={"scan_path": str(path), "allow_reduced_scale": True},
                optional=True, checkpoint_after=False,
                tags=("bias", f"bias={bv}", "assess"),
            )
            ar = executor.sub_results.get(f"{tag}:assess")
            ad = getattr(ar, "data", None) or {}
            frames.append({
                "bias_v": bv, "index": i, "ok": True, "path": str(path),
                "verdict": ad.get("verdict"),
                "concentration": ad.get("angular_concentration"),
                "snr": ad.get("snr"),
                "period_nm": ad.get("period_fast_axis_nm"),
                "coverage": ad.get("coverage"),
            })
            executor.set_partial("frames", list(frames))
            logger.info("AcquireBiasImagingSeries %+.4g V -> %s (集中度 %s)",
                        bv, ad.get("verdict"), ad.get("angular_concentration"))

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        frames = list(pd.get("frames", []))
        order = list(pd.get("bias_order", []))
        good = [f for f in frames if f.get("ok")]
        out = {
            "refused": pd.get("refused"),
            "bias_order": order,
            "line_time_s": pd.get("line_time_s"),
            "line_time_source": pd.get("line_time_source"),
            "frames": frames,
            "n_ok": len(good),
        }
        if pd.get("refused"):
            out["advice"] = "被拒：%s —— 一帧都没扫。" % pd["refused"]
            return out

        # 首尾同条件的两帧之差 = 这段时间里与偏压无关的漂变，是跨 |V| 比较的刻度
        drift_note = None
        if len(order) >= 2 and order[0] == order[-1]:
            first = next((f for f in good if f.get("index") == 0), None)
            last = next((f for f in good if f.get("index") == len(order) - 1), None)
            if first and last and first.get("concentration") and last.get("concentration"):
                a, b = float(first["concentration"]), float(last["concentration"])
                ratio = b / a if a else float("nan")
                out["time_drift_ratio"] = ratio
                drift_note = (
                    "同一偏压 %+.4g V 在开头与结尾的角向集中度：%.0f → %.0f"
                    "（比值 %.2f）。**跨偏压的差异要大于这个比值才算数** —— "
                    "否则那只是针尖在这段时间里自己变了。" % (order[0], a, b, ratio))
                out["time_drift_note"] = drift_note

        if len(good) >= 2:
            best = max(good, key=lambda f: float(f.get("concentration") or 0))
            out["best_bias_v"] = best.get("bias_v")
            out["best_concentration"] = best.get("concentration")
            out["advice"] = (
                "成像最好的偏压是 %+.4g V（角向集中度 %.0f）。%s"
                % (best.get("bias_v"), float(best.get("concentration") or 0),
                   drift_note or "没有时间对照帧 —— 跨偏压的比较缺一把刻度，"
                                 "下次把 repeat_first_at_end 打开。"))
        else:
            out["advice"] = "成功的帧不足两张，比不了。"
        return out
