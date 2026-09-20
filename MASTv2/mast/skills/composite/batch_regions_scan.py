"""BatchRegionsScan — scan a list of operator-chosen regions one by one.

The "I'm going to bed — give me a tour of the surface" skill. The operator (via
the IC agent) names several regions; the agent records them as an array and this
composite scans each in turn, producing one .sxm per region plus a summary the
operator can review in the morning.

Difference vs SurveySurface_TileScan: that one tiles a SINGLE square region into
a regular N×N grid; this one scans an ARBITRARY, explicit list of regions —
different centers, sizes, even angles — wherever the operator pointed.

Plan per region (static — fully determined by the regions list):
  1. ConfigureScan(center, size, angle, channels)
  2. SetScanSpeed(line_time)          (optional — best-effort)
  3. StartScan
  4. WaitScanComplete (timeout per region)
  5. SaveScan                          (optional, gated by save_each → .sxm path)
  6. AssessImageQuality                (optional, gated by assess_quality)

Steps are optional=True so one bad region doesn't abort the whole night
(partial-success semantics, same as SurveySurface_TileScan / GridSTS). Run only
after the tip is approached + stable. CONFIRM-gated like every scan composite.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.agents._shared.skill_adapter import wrap_skill

logger = logging.getLogger(__name__)


def _resolve_region_line_time(explicit, region_size_m: float) -> float:
    """单个区域的每线时间:显式优先,否则按该区域自己的尺寸查档位表。"""
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass
    try:
        from mast.core.scan_policy import get_tier_for_size
        return float(get_tier_for_size(float(region_size_m))["line_time_s"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("BatchRegionsScan 档位表查询失败,退回 0.1 s: %s", exc)
        return 0.1

_MAX_REGIONS = 64
# Sanity bounds (METERS). Generous, but catch unit slips (e.g. "100" meaning
# 100 m instead of 100e-9 m). Size ceiling matches ConfigureScan's hard cap.
_MAX_ABS_CENTER = 1e-3   # ±1 mm stage ceiling
_MIN_SIZE = 1e-10        # 0.1 nm  (= ConfigureScan min)
_MAX_SIZE = 1e-5         # 10 µm   (= ConfigureScan max)


class BatchRegionsScan(CompositeSkillGraph):
    """Scan an explicit list of regions sequentially (one .sxm each)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="BatchRegionsScan",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把用户指定的一串区域一个接一个扫过去,每个区域产出一张 .sxm,"
                "外加一份汇总。适用场景:要求了几片要成像的区域(例如离开"
                "仪器过夜之前),想要一次无人值守的表面巡游。`regions` 是一个 JSON "
                "数组,每一项是 {center_x_m, center_y_m, width_m, height_m, "
                "[angle_deg], [label]} 对象,所有距离都以**米**计。只在针尖已经"
                "进针并稳定之后才跑。每个区域都是可选的 —— 一个失败不会中止其余的。"
            ),
            parameters=[
                ParameterSpec(
                    name="regions",
                    type="str",
                    description=(
                        "由区域对象组成的 JSON 数组,每一项都要有 center_x_m、"
                        "center_y_m、width_m、height_m,单位**米**"
                        "(angle_deg、label 可选)。例: "
                        '[{"center_x_m": 1e-7, "center_y_m": 0, "width_m": 5e-8, '
                        '"height_m": 5e-8, "label": "A"}, {"center_x_m": -2e-7, '
                        '"center_y_m": 1e-7, "width_m": 1e-7, "height_m": 1e-7}]。'
                        "最多 64 个区域。尺寸 100p–10u m。"
                    ),
                    required=True,
                ),
                ParameterSpec(
                    name="channels",
                    type="str",
                    description="每个区域要采的通道,逗号分隔。缺省 'Z,Current'。",
                    required=False, default="Z,Current",
                ),
                ParameterSpec(
                    name="line_time_s",
                    type="float",
                    description=(
                        "每个区域的正扫每线时间(s)。**除非要求说了一个数,"
                        "否则别填**:省略时,每个区域会按**它自己的**尺寸去查逐尺度"
                        "档位表拿线时间 —— 一个批次里的区域尺寸可以差很多,给它们"
                        "同一个速度,对其中大多数都是错的。"
                    ),
                    unit="s", required=False,
                    # None,不是 0.1(pydantic 会物化 default,写 0.1 就分不出
                    # 「没传」与「显式 0.1」)。
                    default=None,
                    min_value=0.01, max_value=10.0,
                ),
                ParameterSpec(
                    name="wait_timeout_s",
                    type="float",
                    description="每个区域的扫描最多等多久(s)。",
                    unit="s", required=False, default=180.0,
                    min_value=10.0, max_value=3600.0,
                ),
                ParameterSpec(
                    name="save_each",
                    type="bool",
                    description="每个区域扫完就 SaveScan(每个区域一张 .sxm)。",
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="assess_quality",
                    type="bool",
                    description="对每个区域跑一次 AssessImageQuality;并推荐最好的那一个。",
                    required=False, default=False,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=600.0,
            composition_level=3,
            tags=["scan", "batch", "regions", "survey", "overview", "composite"],
        )

    # ------------------------------------------------------------------
    # Region parsing / validation (shared by plan() + run_composite())
    # ------------------------------------------------------------------

    def _parse_regions(self, params: dict) -> tuple[list[dict], str]:
        raw = params.get("regions")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return [], "regions is required (a JSON array of region objects)."
        if isinstance(raw, (list, tuple)):
            items: Any = list(raw)
        else:
            try:
                items = json.loads(raw)
            except (ValueError, TypeError) as exc:
                return [], f"regions is not valid JSON: {exc}"
        if not isinstance(items, list):
            return [], "regions must be a JSON array (list) of region objects."
        if not items:
            return [], "regions array is empty."
        if len(items) > _MAX_REGIONS:
            return [], f"too many regions ({len(items)} > {_MAX_REGIONS})."
        out: list[dict] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                return [], f"region #{i} is not an object."
            try:
                cx = float(it["center_x_m"]); cy = float(it["center_y_m"])
                w = float(it["width_m"]); h = float(it["height_m"])
            except (KeyError, TypeError, ValueError):
                return [], (f"region #{i} needs numeric center_x_m, center_y_m, "
                            "width_m, height_m (in meters).")
            ang = float(it.get("angle_deg", 0.0) or 0.0)
            if not (abs(cx) <= _MAX_ABS_CENTER and abs(cy) <= _MAX_ABS_CENTER):
                return [], (f"region #{i} center out of range (|x|,|y| must be "
                            f"≤ {_MAX_ABS_CENTER:.0e} m); check meters vs nm.")
            if not (_MIN_SIZE <= w <= _MAX_SIZE and _MIN_SIZE <= h <= _MAX_SIZE):
                return [], (f"region #{i} size out of range ({_MIN_SIZE:.0e}–"
                            f"{_MAX_SIZE:.0e} m); check meters vs nm.")
            out.append({
                "center_x_m": cx, "center_y_m": cy,
                "width_m": w, "height_m": h, "angle_deg": ang,
                "label": str(it.get("label") or f"R{i + 1}"),
            })
        return out, ""

    # ------------------------------------------------------------------
    # Plan: one Configure+Speed+Start+Wait(+Save)(+Assess) per region
    # ------------------------------------------------------------------

    def plan(self, params: dict) -> list[CompositeStep]:
        regions, err = self._parse_regions(params)
        if err:
            return []   # run_composite's pre-flight surfaces the error
        channels = params.get("channels", "Z,Current")
        explicit_line_time = params.get("line_time_s")
        wait_timeout_ms = int(float(params.get("wait_timeout_s", 180.0)) * 1000)
        save_each = bool(params.get("save_each", True))
        assess = bool(params.get("assess_quality", False))

        steps: list[CompositeStep] = []
        for i, r in enumerate(regions):
            # **逐区**查档:一个批次里的区域尺寸可以差很多,给它们同一个速度
            # 对其中大多数都是错的。
            line_time = _resolve_region_line_time(
                explicit_line_time, max(r["width_m"], r["height_m"]))
            speed = (max(r["width_m"], r["height_m"]) / line_time
                     if line_time > 0 else 200e-9)
            tag = (f"region={i}", f"label={r['label']}")
            steps.append(CompositeStep(
                step_id=f"region_{i}:configure",
                skill_name="ConfigureScan",
                params={
                    "center_x_m": r["center_x_m"], "center_y_m": r["center_y_m"],
                    "width_m": r["width_m"], "height_m": r["height_m"],
                    "angle_deg": r["angle_deg"], "channels": channels,
                },
                optional=True, checkpoint_after=False, tags=("configure",) + tag,
            ))
            steps.append(CompositeStep(
                step_id=f"region_{i}:speed",
                skill_name="SetScanSpeed",
                params={
                    "fwd_speed": speed, "bwd_speed": speed,
                    "fwd_line_time": line_time, "bwd_line_time": line_time,
                    "keep_const": 0,
                },
                optional=True, checkpoint_after=False, tags=("speed",) + tag,
            ))
            steps.append(CompositeStep(
                step_id=f"region_{i}:start",
                skill_name="StartScan", params={},
                optional=True, checkpoint_after=False, tags=("start",) + tag,
            ))
            steps.append(CompositeStep(
                step_id=f"region_{i}:wait",
                skill_name="WaitScanComplete",
                params={"timeout_ms": wait_timeout_ms},
                optional=True, checkpoint_after=True, tags=("wait",) + tag,
            ))
            if save_each:
                steps.append(CompositeStep(
                    step_id=f"region_{i}:save",
                    skill_name="SaveScan", params={},
                    optional=True, checkpoint_after=False, tags=("save",) + tag,
                ))
            if assess:
                steps.append(CompositeStep(
                    step_id=f"region_{i}:assess",
                    skill_name="AssessImageQuality", params={},
                    optional=True, checkpoint_after=False, tags=("assess",) + tag,
                ))
        return steps

    # ------------------------------------------------------------------
    # Hooks: per-region record (center / path / quality / success)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_region_step(step_id: str) -> tuple[int, str] | None:
        if not step_id.startswith("region_"):
            return None
        head, _, phase = step_id.partition(":")
        try:
            return int(head.split("_")[1]), phase
        except (IndexError, ValueError):
            return None

    def _region_records(self) -> dict[str, dict[str, Any]]:
        return self._executor.progress.partial_data.setdefault(
            "region_records", {})

    def _record_for(self, i: int) -> dict[str, Any]:
        recs = self._region_records()
        rec = recs.get(str(i))
        if rec is None:
            regions = self._executor.progress.partial_data.get("regions", [])
            label = regions[i]["label"] if i < len(regions) else f"R{i + 1}"
            rec = {"index": i + 1, "label": label, "success": True}
            recs[str(i)] = rec
        return rec

    def on_step_result(self, step: CompositeStep, sub_result) -> None:
        parsed = self._parse_region_step(step.step_id)
        if parsed is None:
            return
        i, phase = parsed
        rec = self._record_for(i)
        data = getattr(sub_result, "data", None) or {}
        if phase == "configure":
            rec["center_x_m"] = step.params.get("center_x_m")
            rec["center_y_m"] = step.params.get("center_y_m")
            rec["width_m"] = step.params.get("width_m")
            rec["height_m"] = step.params.get("height_m")
        elif phase == "wait":
            # 两种「没扫完」都只判**这一个区域**失败,不中止整批 —— 这不是新决定,
            # 是跟着这一处 timed_out 已有的策略走(见 aggregate:失败区域被排除在
            # scanned_paths 之外,后面的区域照跑)。批量扫描的价值就在于一个区域
            # 出问题不影响其余区域,而「谁停了这一帧」对下一个区域没有影响。
            if data.get("timed_out", False):
                rec["success"] = False
                rec.setdefault("error", "scan timeout")
            elif data.get("stopped_early", False):
                # 与超时分开写,因为用户要做的事不同:超时调 timeout,
                # 中途停止要去查是谁停的。合成一句会把人送去调错东西。
                rec["success"] = False
                done = data.get("lines_done")
                total = data.get("lines_total")
                where = (f" ({done}/{total} 行)"
                         if done is not None and total else "")
                rec.setdefault("error", f"scan stopped early{where}")
        elif phase == "save":
            if data.get("saved_path"):
                rec["sxm_path"] = data["saved_path"]
        elif phase == "assess":
            if data:
                rec["quality"] = data.get("fft_quality")
                rec["assessment"] = {
                    k: v for k, v in data.items()
                    if k in ("fft_quality", "label", "confidence", "snr_db")
                }

    def on_step_failed(self, step: CompositeStep, msg: str) -> bool:
        parsed = self._parse_region_step(step.step_id)
        if parsed is None:
            return step.optional
        i, _phase = parsed
        rec = self._record_for(i)
        rec["success"] = False
        rec.setdefault("error", f"{_phase}: {msg}")
        return step.optional

    # ------------------------------------------------------------------
    # Aggregate: ordered region list + recommendation
    # ------------------------------------------------------------------

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        regions = pd.get("regions", [])
        recs: dict[str, dict] = pd.get("region_records", {})
        ordered: list[dict[str, Any]] = []
        for i in range(len(regions)):
            rec = recs.get(str(i))
            if rec is None:
                rec = {"index": i + 1,
                       "label": regions[i].get("label", f"R{i + 1}"),
                       "success": False, "error": "region not reached"}
            ordered.append(rec)
        successful = [r for r in ordered if r.get("success")]
        rated = [r for r in successful if r.get("quality") is not None]
        best = max(rated, key=lambda r: r["quality"]) if rated else None
        return {
            "region_count": len(regions),
            "regions": ordered,
            "success_count": len(successful),
            "fail_count": len(ordered) - len(successful),
            # Only SUCCESSFUL regions contribute to scanned_paths. A region whose
            # configure step was rejected by the safety gate could still carry an
            # sxm_path (the scan sub-skill had already reported a saved file from
            # an earlier region's state), and that path then flowed downstream as
            # if it were this region's data — 2026-07-27 field forensics, where a
            # batch with 3/4 regions rejected still published 4 paths.
            "scanned_paths": [r["sxm_path"] for r in successful if r.get("sxm_path")],
            "recommended_region": best,
        }

    # ------------------------------------------------------------------
    # run_composite: validate, run, partial-success
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        regions, err = self._parse_regions(params)
        if err:
            return self.fail(err)
        logger.info("BatchRegionsScan: %d regions", len(regions))
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        executor.set_partial("regions", regions)
        executor.set_partial_default("region_records", {})
        self._executor = executor

        executor.run_plan(iter(self.plan(params)))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        if executor.progress.aborted:
            return self.fail(
                executor.progress.aborted_reason or "batch scan aborted", **data)
        # A partial batch is a failure with completed/failed counts. Preserve
        # aggregate data for callers that can use partial results, but include
        # paths only for regions whose acquisition actually succeeded.
        fail_count = int(data.get("fail_count", 0) or 0)
        region_count = int(data.get("region_count", 0) or 0)
        success_count = int(data.get("success_count", 0) or 0)
        if fail_count > 0:
            logger.warning("BatchRegionsScan: %d/%d regions failed",
                           fail_count, region_count)
        # Zero completed regions is failure, even if the batch itself returned
        # normally. All safety-gate refusals must not be reported as a successful
        # scan merely because the caller ignores the separate fail_count.
        if region_count > 0 and success_count == 0:
            return self.fail(
                f"0/{region_count} regions scanned — every region failed", **data)
        # A genuine partial (some scanned, some not) stays a success: the scanned
        # regions are real data and the caller should keep them rather than
        # re-run the whole batch. But say so in the summary — fail_count alone
        # sits inside the data dict where no one looked.
        if fail_count > 0:
            res = self.ok(**data)
            # ok() only fills data; the shortfall has to reach SkillResult.summary,
            # which is what the caller and the records layer actually surface.
            res.summary = (
                f"部分完成：{success_count}/{region_count} 个区域实际扫描成功，"
                f"{fail_count} 个失败（详见 regions[].error）。")
            return res
        return self.ok(**data)


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(BatchRegionsScan, context_provider)
