"""ExecuteScanPlan —— 执行一份多图计划,带帧间守卫。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

计划由 :mod:`mast.core.scan_planner` 确定性地排出来(扫哪几张、每张什么参数、什么
顺序),经 ``plan_scan_batch`` 工具发布到扫描地图上让用户先看一眼,再交给这里执行。

**计划是一份 JSON,是两半之间唯一的接口** —— 这不是多此一举:计划可检视意味着
「将要发生什么」在发生之前就能看到,而不是事后从日志里拼。

## 为什么不直接复用 BatchRegionsScan

它是均一参数的裸循环:没有逐帧参数(bias 系列每帧不同)、没有帧间守卫。而守卫正是
批量扫图与「连着扫 N 张」的全部区别:

  * 针尖事件(tip_change CRITICAL / 撞针)→ **中止整批**,不是跳过这一帧。针尖坏了
    之后每一帧都是废的,继续扫只是在浪费机时和针尖寿命;
  * 单帧质量差 → 重扫**一次**,再差就标记继续(行动预算,不无限重试);
  * 坏帧率超过阈值 → 提前中止。系统性问题(表面脏了 / 针尖钝了 / 参数不合适)
    不会因为多扫几张就自己好起来;
  * 大跳变后先等压电稳定(规划器已经标好了哪几帧需要)。
  * **坐标代次(coord_epoch)不对就整批拒绝** —— 见下。

## 坐标代次

计划里的每一帧都是一对米坐标,而米坐标只在**一个代次内**有意义:一次横向粗动
之后,同样的 (x, y) 指的是另一片表面。所以计划顶层带 ``coord_epoch`` 时,这里
在开跑前**和每一帧之前**都拿它跟权威代次核对一次(``core.coord_epoch``),对不上
就整批拒绝,剩余帧全拒同因。

**拒绝,不夹紧,不换算。** 跨代次坐标换算在本仓有意不存在(``io/coarse_map.py``
的「WHY STEPS, NOT METRES」:粗动步进开环,步长随温度能差五倍),一个换算出来的
坐标看上去和真坐标一模一样,而它是编的。

旧格式(没有 ``coord_epoch``)照常执行,只在 ``data["warnings"]`` 里记一句 ——
一份没盖章的计划不该因为强制点上线就突然跑不了,但也不该假装它受过保护。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from mast.core import coord_epoch
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 大跳变之后的压电稳定等待(s)。
MOVE_SETTLE_S = 5.0
#: 单帧最多重扫几次。
MAX_RESCANS_PER_FRAME = 1
#: 坏帧率超过这个比例就提前中止。
BAD_FRAME_RATE_ABORT = 0.30
#: 判坏帧率之前至少要扫过这么多帧。
#:
#: 5 而不是 3:n=3 时最小的非零坏帧率就是 33% —— 已经越过 30% 的线,于是「前三帧
#: 里坏了一张」必然中止整批。而单张坏帧通常是局部的(一粒脏东西、一次瞬时干扰),
#: 不是系统性问题。这个守卫要抓的是后者。
MIN_FRAMES_FOR_RATE = 5
#: 而且至少要有这么多坏帧。**一张坏帧永远不中止批次** —— 与 stop_on_bad_quality
#: 的说明一致:一张差图通常是局部的。
MIN_BAD_FOR_RATE = 2


def needs_bias_change(frame: dict, attempts: int) -> bool:
    """这一次尝试要不要先走 BiasSettleChange —— 答案与 ``attempts`` **无关**。

    这条规则单独拎出来命名,是因为它原本写的是 ``attempts == 0 and …``:
    偏压只在第一次尝试时设,重扫那一次不设,而 ``bias_v`` 又已经从下发参数里
    pop 掉了 ⇒ **重扫用的是上一次留下的偏压**。两次尝试之间会插事情(针尖复核、
    脉冲、任何改偏压的动作),于是偏压系列里的某一帧被悄悄换成另一个偏压的图 ——
    数据看着完好,标签是错的。

    ``attempts`` 仍然收在签名里:它是这条规则**刻意不看**的那个量,签名里留着它
    才能让「不看」这件事本身可测(``test_a_rescan_sets_the_bias_again``)。
    """
    return frame.get("bias_v") is not None


class ExecuteScanPlan(BaseSkill):
    """Run a planned multi-frame scan batch with inter-frame guards."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ExecuteScanPlan",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "执行一份由 plan_scan_batch 排出来的扫描计划:每一帧都按规划器为它"
                "定好的那套参数采集,帧与帧之间有守卫(针尖事件 -> 中止整批;"
                "质量差 -> 重扫一次;坏帧率高 -> 提前中止)。"
                "把计划 JSON **原样**传进来 —— 不要手工去改里面的帧;"
                "要改就重新排一次计划,好让用户看得见改了什么。"
            ),
            parameters=[
                ParameterSpec(
                    name="plan_json",
                    type="str",
                    description=(
                        "plan_scan_batch 返回的那份计划(JSON)。里面装着逐帧的中心、"
                        "尺寸与已经定好的参数,外加它是在哪一代 coord_epoch 下排的。"
                        "**原样**传进来:计划的 coord_epoch 落后于当前代次时一律直接"
                        "拒绝(一次横向粗动已经让那些坐标指向了另一片表面)—— "
                        "重新排计划,**绝不**去换算坐标。"
                    ),
                    required=True,
                ),
                ParameterSpec(
                    name="stop_on_bad_quality",
                    type="bool",
                    description=(
                        "某一帧重扫之后仍然差,就中止整批。默认关:一张坏帧通常是"
                        "局部现象,而系统性问题已经由坏帧率那道守卫兜住了。"
                    ),
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="save_each",
                    type="bool",
                    description="每采到一帧就保存。",
                    required=False,
                    default=True,
                ),
            ],
            preconditions=["z_controller_on", "scan_not_running"],
            estimated_duration_s=600.0,
            composition_level=4,
            tags=["scan", "batch", "plan", "composite"],
        )

    # ── 守卫 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _frame_verdict(scan_result) -> dict[str, Any]:
        """把一帧的结果读成 {ok, tip_event, crash, quality, reason}。

        判据取自帧结果里**已经存在**的字段,不在这里另起一套 —— 两套判据打架比
        没有判据更难查(见 RFC 的边界声明:TipHealthWatch 不重做)。
        """
        data = getattr(scan_result, "data", {}) or {}
        ok = bool(getattr(scan_result, "success", False))
        crash = bool(data.get("crash_indicator") or
                     data.get("crash_check") == "crash")
        tip_event = bool(data.get("tip_change_critical") or
                         data.get("tip_event"))
        quality = data.get("fft_quality")
        return {
            "ok": ok and not crash,
            "tip_event": tip_event,
            "crash": crash,
            "quality": quality,
            "reason": ("" if ok else str(getattr(scan_result, "error", ""))),
        }

    # ── 执行 ─────────────────────────────────────────────────────────────────

    def execute(self, context, params: dict) -> SkillResult:
        try:
            plan = json.loads(params["plan_json"])
        except (TypeError, ValueError, KeyError) as exc:
            return SkillResult(
                skill_name="ExecuteScanPlan", success=False,
                error=f"计划 JSON 解析失败: {exc}")

        frames = plan.get("frames") if isinstance(plan, dict) else None
        if not isinstance(frames, list) or not frames:
            return SkillResult(
                skill_name="ExecuteScanPlan", success=False,
                error="计划里没有任何帧")

        save_each = bool(params.get("save_each", True))
        stop_on_bad = bool(params.get("stop_on_bad_quality", False))

        done: list[dict] = []
        bad: list[dict] = []
        aborted_at: "int | None" = None
        abort_reason = ""
        warnings: list[str] = []
        stale_verdict: "coord_epoch.EpochVerdict | None" = None

        # ── 坐标代次:开跑前核对一次 ─────────────────────────────────────
        stamped = plan.get("coord_epoch")
        verdict0 = coord_epoch.verify(stamped, what="这份计划的坐标")
        if verdict0.stale:
            # 一帧都不扫。拒绝里**不带任何坐标** —— 既不夹紧也不换算,给出去的
            # 每一个数字都得是能负责的数字。
            return SkillResult(
                skill_name="ExecuteScanPlan", success=False,
                error=verdict0.message,
                data={"planned_frames": len(frames), "done": 0, "bad": 0,
                      "frames_done": [], "frames_bad": [], "scanned_paths": [],
                      "aborted_at": 0, "abort_reason": verdict0.message,
                      "refusal_code": coord_epoch.REFUSAL_CODE,
                      "plan_coord_epoch": verdict0.stamped,
                      "current_coord_epoch": verdict0.current,
                      "frames_refused": len(frames)},
                summary=(f"整批拒绝:计划是第 {verdict0.stamped} 代坐标,"
                         f"当前第 {verdict0.current} 代"))
        if verdict0.state != coord_epoch.MATCH:
            # 没盖章 / 查不到 —— 两种都不是「陈旧」,照常执行,但说出来。
            # 「读不到」被当成答案是本仓反复出事的那一类。
            warnings.append(verdict0.message)

        for i, frame in enumerate(frames):
            if context.check_abort():
                aborted_at, abort_reason = i, "用户中止"
                break

            # 每帧前再核对一次。批次中途本不该发生粗动,但绕行分支(detour)可能
            # 做,而一旦做了,剩下每一帧都会扫在错的地方。查询是 COUNT,廉价。
            #
            # 条件是「这份计划**盖过章**」而不是「开跑那次核对通过」:开跑时代次
            # 查不到、扫到一半又能查了 —— 那一刻正是最该核对的时候。
            if verdict0.stamped is not None:
                v = coord_epoch.verify(stamped, what="这份计划的坐标")
                if v.stale:
                    aborted_at = i
                    abort_reason = v.message
                    stale_verdict = v
                    break
                if v.state != coord_epoch.MATCH and v.message not in warnings:
                    warnings.append(v.message)

            try:
                center_x = float(frame["center_x_m"])
                center_y = float(frame["center_y_m"])
                size = float(frame["size_m"])
            except (KeyError, TypeError, ValueError):
                bad.append({"index": i, "reason": "帧几何非法", "frame": frame})
                continue

            # 规划器标记的大跳变:压电蠕变在大位移后要一段时间才安定,不等的话
            # 接下来一两帧会带上漂移。
            if frame.get("needs_settle"):
                time.sleep(MOVE_SETTLE_S)

            scan_params: dict[str, Any] = {
                "center_x_m": center_x, "center_y_m": center_y, "size_m": size,
            }
            # 逐帧参数:只把规划器真正定下来的值往下传。None 一律不传 ——
            # 那是「保持现值」,不是「设成 0」。
            for key in ("bias_v", "setpoint_a", "line_time_s", "pixels"):
                if frame.get(key) is not None:
                    scan_params[key] = frame[key]

            attempts = 0
            verdict = None
            while attempts <= MAX_RESCANS_PER_FRAME:
                # 偏压变更走安全通道(穿零会把针尖推向表面)。ScanAt 自己会设
                # bias,但那是直接设;系列里跨零的那一步必须先过这里。
                #
                # **每一次尝试都设**,不是只在第一次 —— 规则和它的来历都在
                # `needs_bias_change` 那里。BiasSettleChange 对「已经是这个值」
                # 是廉价的,重设的代价远小于一帧错标签。
                if needs_bias_change(frame, attempts):
                    res_bias = context.run(
                        "BiasSettleChange", {"bias_v": float(frame["bias_v"])})
                    if not getattr(res_bias, "success", False):
                        verdict = {"ok": False, "tip_event": False, "crash": False,
                                   "quality": None,
                                   "reason": f"偏压变更被拒: "
                                             f"{getattr(res_bias, 'error', '')}"}
                        break
                    scan_params.pop("bias_v", None)   # 已经设好了

                res = context.run("ScanAt", dict(scan_params))
                verdict = self._frame_verdict(res)
                verdict["saved_path"] = (getattr(res, "data", {}) or {}).get(
                    "saved_path")
                if verdict["tip_event"] or verdict["crash"]:
                    break
                if verdict["ok"]:
                    break
                attempts += 1

            if save_each and verdict and verdict["ok"]:
                context.run("SaveScan", {})

            record = {"index": i, "label": frame.get("label", ""),
                      "center_x_m": center_x, "center_y_m": center_y,
                      "size_m": size, "attempts": attempts + 1, **(verdict or {})}

            # 针尖事件 / 撞针 → 中止整批。针尖坏了之后每一帧都是废的。
            if verdict and (verdict["tip_event"] or verdict["crash"]):
                bad.append(record)
                aborted_at = i
                abort_reason = ("扫描中检测到针尖事件" if verdict["tip_event"]
                                else "检测到撞针")
                break

            if verdict and verdict["ok"]:
                done.append(record)
            else:
                bad.append(record)
                if stop_on_bad:
                    aborted_at, abort_reason = i, "帧质量不合格且已设为遇差即停"
                    break

            # 坏帧率:系统性问题不会因为多扫几张就自己好起来。
            total = len(done) + len(bad)
            if (total >= MIN_FRAMES_FOR_RATE
                    and len(bad) >= MIN_BAD_FOR_RATE
                    and len(bad) / total > BAD_FRAME_RATE_ABORT):
                aborted_at = i
                abort_reason = (f"坏帧率 {len(bad)}/{total} 超过 "
                                f"{BAD_FRAME_RATE_ABORT:.0%} —— 多半是表面 / 针尖 / "
                                f"参数的系统性问题,继续扫只是浪费机时")
                break

        planned = len(frames)
        data = {
            "planned_frames": planned,
            "done": len(done),
            "bad": len(bad),
            "aborted_at": aborted_at,
            "abort_reason": abort_reason,
            "frames_done": done,
            "frames_bad": bad,
            # 仅汇总成功采集且通过归属验证的帧路径。
            "scanned_paths": [r.get("saved_path") for r in done
                              if r.get("saved_path")],
        }
        if verdict0.stamped is not None:
            # 这批图属于哪一代 —— 存下来的 .sxm 自己不带代次(它与代次的唯一联系
            # 是写盘时刻),这一条是下游把文件对回表面的那根线。
            data["coord_epoch"] = verdict0.stamped
        if warnings:
            data["warnings"] = list(warnings)
        if stale_verdict is not None:
            # 中途粗动:剩余帧全拒同因,原因码与开跑时那条完全一样。代次取**当时
            # 判定用的那两个数**,不重新查一遍 —— 报出去的数必须是做决定的那个数。
            data["refusal_code"] = coord_epoch.REFUSAL_CODE
            data["plan_coord_epoch"] = stale_verdict.stamped
            data["current_coord_epoch"] = stale_verdict.current
            data["frames_refused"] = (planned - aborted_at
                                      if aborted_at is not None else planned)

        if aborted_at is not None:
            return SkillResult(
                skill_name="ExecuteScanPlan", success=False,
                error=(f"批次在第 {aborted_at + 1}/{planned} 帧中止:{abort_reason}。"
                       f"已完成 {len(done)} 帧,失败 {len(bad)} 帧。"),
                data=data,
                summary=f"批次中止 @{aborted_at + 1}/{planned}:{abort_reason}")

        if not done:
            return SkillResult(
                skill_name="ExecuteScanPlan", success=False,
                error=f"计划里 {planned} 帧一帧都没成功",
                data=data)

        # 部分成功 = 成功,但缺口必须写进 summary。把 fail_count 埋在 data 里
        # 没人读 —— 这是 2026-07-27 修过的同一个坑。
        summary = f"完成 {len(done)}/{planned} 帧"
        if bad:
            summary += f",{len(bad)} 帧未达标(见 frames_bad)"
        return SkillResult(skill_name="ExecuteScanPlan", success=True,
                           data=data, summary=summary)
