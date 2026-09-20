"""SurveySurface_TileScan — multi-tile global STM overview (graph-shaped).

Scans an NxN grid of tiles to build a global understanding of a surface
before zooming in. Optionally annotates each tile with a quick vision
quality assessment so downstream agents (XD, IC re-planning) can pick
the best region to drill into.

Plan per tile (static — tile topology fully determined by params):
  1. ConfigureScan(center, size, channels)
  2. SetScanSpeed(line_time)         (optional — best-effort)
  3. StartScan
  4. WaitScanComplete (timeout per tile)
  5. AssessImageQuality              (optional, gated by assess_quality)

The skill itself does NOT run AutoApproach / ConditionTip — the operator
is expected to have approached and verified tip first. Use this AFTER
the tip is stable and BEFORE zooming into a specific feature.

Example: 200 nm × 200 nm overview via 4 × 50 nm tiles ::
    SurveySurface_TileScan(
        center_x_m=0.0, center_y_m=0.0,
        total_size_m=200e-9, tile_size_m=50e-9,
    )

Phase 7 migration: subclasses CompositeSkillGraph; plan() is static
(per-tile steps emitted up-front) with each tile contributing a
ConfigureScan + StartScan + WaitScanComplete (+ optional Assess). Steps
are marked ``optional=True`` so one bad tile doesn't abort the survey
— same partial-success semantics as GridSTS.
"""
from __future__ import annotations

import logging
import math
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


def _resolve_tile_line_time(explicit, tile_size_m: float) -> float:
    """每块 tile 的每线时间:显式优先,否则按 tile 尺寸查用户的档位表。

    换掉原来写死的 0.1 s —— 那个常数对 500 nm 的粗查 tile 和 5 nm 的精扫 tile
    给的是同一个速度。查表失败时仍退回 0.1,绝不因为查表挂掉而扫不了图。
    """
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass
    try:
        from mast.core.scan_policy import get_tier_for_size
        return float(get_tier_for_size(float(tile_size_m))["line_time_s"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("SurveySurface 档位表查询失败,退回 0.1 s: %s", exc)
        return 0.1


class SurveySurface_TileScan(CompositeSkillGraph):
    """Multi-tile global STM overview scan (graph-shaped).

    Divides a square region into N×N tiles and scans each in raster order
    (row-by-row, left-to-right top-to-bottom). Each tile produces one
    .sxm file; the skill returns the list of tile_path + tile_center for
    downstream agents to consume.

    Optional final step: run AssessImageQuality on each tile and return
    a quality score per tile, plus the recommended "best" tile.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SurveySurface_TileScan",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "用 NxN 个扫描瓦片巡查一块方形表面区域，拿到全局概貌。"
                "每个瓦片产出一个 .sxm 文件。"
                "**在**放大到具体特征**之前**用它。只在针尖已进针且稳定之后跑。"
                "可选地给每个瓦片的成像质量打分，并推荐最适合后续高分辨扫描的"
                "那一块。"
            ),
            parameters=[
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description=(
                        "巡查区中心 X，单位 METERS。常见 -1.5u 到 1.5u（±1.5 µm）。"
                        "例：0 ✓；1.5u ✓；1（= 1 meter！）✗。"
                    ),
                    unit="m",
                    required=False,
                    default=0.0,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="巡查区中心 Y，单位 METERS。常见 -1.5u 到 1.5u。",
                    unit="m",
                    required=False,
                    default=0.0,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="total_size_m",
                    type="float",
                    description=(
                        "被巡查的方形区总边长，单位 METERS。默认 200n"
                        "（200 nm）。硬上限 10u（10 µm）。例：200n = 200 nm；1u = 1 µm。"
                    ),
                    unit="m",
                    required=False,
                    default=200e-9,
                    min_value=10e-9,
                    max_value=1e-5,
                ),
                ParameterSpec(
                    name="tile_size_m",
                    type="float",
                    description=(
                        "单个扫描瓦片的边长，单位 METERS。默认 50n"
                        "（50 nm）。**必须能整除 total_size_m**，才凑得出整数的 NxN "
                        "栅格；除不尽的值会**向上取整**，保证巡查范围**至少**覆盖 "
                        "total_size_m。"
                    ),
                    unit="m",
                    required=False,
                    default=50e-9,
                    min_value=1e-9,
                    max_value=1e-6,
                ),
                ParameterSpec(
                    name="line_time_s",
                    type="float",
                    description=(
                        "每个瓦片的正扫行时间，单位秒。**除非用户点了名，否则留空** —— "
                        "留空时它按瓦片尺寸从用户的分尺度策略表里取。"
                        "越慢 = SNR 越好，但总时间越长。整轮巡查 "
                        "≈ (N×N tiles) × tile_lines × line_time。"
                    ),
                    unit="s",
                    required=False,
                    # None,不是 0.1 —— pydantic 会把 default 物化进参数,写着
                    # 0.1 就永远分不出「没传」与「显式 0.1」,档位表轮不到。
                    default=None,
                    min_value=0.01,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="channels",
                    type="str",
                    description="每个瓦片要采的通道，逗号分隔。默认 'Z,Current'。",
                    required=False,
                    default="Z,Current",
                ),
                ParameterSpec(
                    name="wait_timeout_s",
                    type="float",
                    description="单个瓦片扫描的等待上限。",
                    unit="s",
                    required=False,
                    default=180.0,
                    min_value=10.0,
                    max_value=3600.0,
                ),
                ParameterSpec(
                    name="assess_quality",
                    type="bool",
                    description=(
                        "每扫完一个瓦片就对它跑一次 AssessImageQuality。"
                        "每个瓦片多花 ~2-5 s，但这样这个技能才能推荐出最适合"
                        "后续高分辨扫描的那一块区域。"
                    ),
                    required=False,
                    default=True,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=600.0,
            composition_level=3,
            tags=["scan", "survey", "overview", "composite"],
        )

    # ------------------------------------------------------------------
    # Grid layout helper (pulled out so plan() + run_composite share it)
    # ------------------------------------------------------------------

    def _compute_grid(self, params: dict) -> tuple[int, float, float, float, float, float, str, float, bool, str]:
        """Validate params and compute grid layout.

        Returns ``(n, tile, line_time, cx, cy, total, channels,
        wait_timeout, assess_quality, error)``. ``error`` is empty iff
        the grid is valid.
        """
        cx = float(params.get("center_x_m", 0.0))
        cy = float(params.get("center_y_m", 0.0))
        total = float(params.get("total_size_m", 200e-9))
        tile = float(params.get("tile_size_m", 50e-9))
        # 每块 tile 的速度按 **tile 尺寸** 查档位表(不是按整片 survey 的尺寸):
        # 实际扫的是一块一块的 tile,该用哪个档由它决定。
        line_time = _resolve_tile_line_time(params.get("line_time_s"), tile)
        channels = params.get("channels", "Z,Current")
        wait_timeout = float(params.get("wait_timeout_s", 180.0))
        assess_quality = bool(params.get("assess_quality", True))

        if tile >= total:
            err = (
                f"tile_size_m ({tile:.2e}) >= total_size_m ({total:.2e}); "
                "use FullScan for a single tile instead."
            )
            return (0, tile, line_time, cx, cy, total, channels,
                    wait_timeout, assess_quality, err)

        n = int(math.ceil(total / tile))
        if n < 2:
            err = (
                f"computed grid is {n}x{n}; need >= 2x2 for a survey. "
                f"Increase total_size_m or decrease tile_size_m."
            )
            return (n, tile, line_time, cx, cy, total, channels,
                    wait_timeout, assess_quality, err)
        if n > 8:
            err = (
                f"computed grid is {n}x{n} = {n * n} tiles; too many. "
                f"Reduce total_size_m or increase tile_size_m (max 8x8)."
            )
            return (n, tile, line_time, cx, cy, total, channels,
                    wait_timeout, assess_quality, err)

        return (n, tile, line_time, cx, cy, total, channels,
                wait_timeout, assess_quality, "")

    # ------------------------------------------------------------------
    # Plan: static — tile topology determined entirely by params
    # ------------------------------------------------------------------

    def plan(self, params: dict) -> list[CompositeStep]:
        (n, tile, line_time, cx, cy, total, channels,
         wait_timeout, assess_quality, err) = self._compute_grid(params)
        if err:
            # plan() returning [] makes run_composite raise the grid error
            # via the explicit pre-flight check; never reached when params
            # are valid.
            return []

        # Grid origin = lower-left tile center
        x0 = cx - (n - 1) * tile / 2.0
        y0 = cy - (n - 1) * tile / 2.0
        speed = tile / line_time if line_time > 0 else 200e-9
        wait_timeout_ms = int(wait_timeout * 1000)

        steps: list[CompositeStep] = []
        # Raster order: bottom-to-top rows, left-to-right within each row
        for row in range(n):
            for col in range(n):
                tile_cx = x0 + col * tile
                tile_cy = y0 + row * tile
                tile_tag = f"row={row}", f"col={col}"

                steps.append(CompositeStep(
                    step_id=f"tile_{row}_{col}:configure",
                    skill_name="ConfigureScan",
                    params={
                        "center_x_m": tile_cx,
                        "center_y_m": tile_cy,
                        "width_m": tile,
                        "height_m": tile,
                        "channels": channels,
                    },
                    optional=True,
                    checkpoint_after=False,
                    tags=("configure",) + tile_tag,
                ))
                steps.append(CompositeStep(
                    step_id=f"tile_{row}_{col}:speed",
                    skill_name="SetScanSpeed",
                    params={
                        "fwd_speed": speed,
                        "bwd_speed": speed,
                        "fwd_line_time": line_time,
                        "bwd_line_time": line_time,
                        "keep_const": 0,
                    },
                    optional=True,           # best-effort, never fatal
                    checkpoint_after=False,
                    tags=("speed",) + tile_tag,
                ))
                steps.append(CompositeStep(
                    step_id=f"tile_{row}_{col}:start",
                    skill_name="StartScan",
                    params={},
                    optional=True,
                    checkpoint_after=False,
                    tags=("start",) + tile_tag,
                ))
                steps.append(CompositeStep(
                    step_id=f"tile_{row}_{col}:wait",
                    skill_name="WaitScanComplete",
                    params={"timeout_ms": wait_timeout_ms},
                    optional=True,
                    checkpoint_after=True,   # checkpoint per finished tile
                    tags=("wait",) + tile_tag,
                ))
                if assess_quality:
                    steps.append(CompositeStep(
                        step_id=f"tile_{row}_{col}:assess",
                        skill_name="AssessImageQuality",
                        params={},
                        optional=True,
                        checkpoint_after=False,
                        tags=("assess",) + tile_tag,
                    ))
        return steps

    # ------------------------------------------------------------------
    # Hooks: stash per-tile success + quality
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tile_step(step_id: str) -> tuple[int, int, str] | None:
        """Extract (row, col, phase) from 'tile_<row>_<col>:<phase>'."""
        if not step_id.startswith("tile_"):
            return None
        head, _, phase = step_id.partition(":")
        parts = head.split("_")
        if len(parts) != 3:
            return None
        try:
            return int(parts[1]), int(parts[2]), phase
        except ValueError:
            return None

    def _tile_records(self) -> dict[str, dict[str, Any]]:
        """Return the partial_data['tile_records'] dict (creating it)."""
        records = self._executor.progress.partial_data.setdefault(
            "tile_records", {},
        )
        return records

    def _record_for(self, row: int, col: int, n: int) -> dict[str, Any]:
        records = self._tile_records()
        key = f"{row}_{col}"
        rec = records.get(key)
        if rec is None:
            rec = {
                "index": row * n + col + 1,
                "row": row,
                "col": col,
                "success": True,           # provisional; flipped on failure
            }
            records[key] = rec
        return rec

    def on_step_result(self, step: CompositeStep, sub_result) -> None:
        parsed = self._parse_tile_step(step.step_id)
        if parsed is None:
            return
        row, col, phase = parsed
        n = int(self._executor.progress.partial_data.get("grid_n", 0))
        rec = self._record_for(row, col, n)

        if phase == "configure":
            params = step.params
            rec["center_x_m"] = params.get("center_x_m")
            rec["center_y_m"] = params.get("center_y_m")
        elif phase == "wait":
            # The wait skill reports success on every way a scan can end. Two of
            # them mean this tile did NOT finish.
            #
            # Both fail the TILE only, never the survey — following this call
            # site's existing timed_out policy rather than inventing one. A
            # survey exists to map many tiles; one truncated tile says nothing
            # about the next, and aborting the grid would throw away the tiles
            # that did scan.
            data = getattr(sub_result, "data", None) or {}
            if data.get("timed_out", False):
                rec["success"] = False
                rec["error"] = (
                    f"scan timeout after {self._executor.progress.partial_data.get('wait_timeout_s')}s"
                )
            elif data.get("stopped_early", False):
                # Kept distinct from the timeout: raising the timeout fixes one
                # and does nothing for the other.
                rec["success"] = False
                done = data.get("lines_done")
                total = data.get("lines_total")
                where = (f" ({done}/{total} lines)"
                         if done is not None and total else "")
                rec["error"] = f"scan stopped early{where}"
        elif phase == "assess":
            data = getattr(sub_result, "data", None) or {}
            if data:
                rec["quality"] = data.get("fft_quality", 0.0)
                rec["assessment"] = {
                    k: v for k, v in data.items()
                    if k in ("fft_quality", "label", "confidence", "snr_db")
                }
            else:
                rec["quality"] = None

    def on_step_failed(self, step: CompositeStep, msg: str) -> bool:
        parsed = self._parse_tile_step(step.step_id)
        if parsed is None:
            return step.optional
        row, col, phase = parsed
        n = int(self._executor.progress.partial_data.get("grid_n", 0))
        rec = self._record_for(row, col, n)
        rec["success"] = False
        # First failure wins for the error field; don't overwrite with later
        # phase failures of the same tile.
        rec.setdefault("error", f"{phase}: {msg}")
        return step.optional

    # ------------------------------------------------------------------
    # Aggregate: build the tile list + recommendation
    # ------------------------------------------------------------------

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        n = int(pd.get("grid_n", 0))
        tile_records: dict[str, dict[str, Any]] = pd.get("tile_records", {})

        # Emit in raster order (row, col) — matches v1 output.
        ordered: list[dict[str, Any]] = []
        for row in range(n):
            for col in range(n):
                rec = tile_records.get(f"{row}_{col}")
                if rec is None:
                    # No record means the tile's plan steps were never reached
                    # (e.g. aborted mid-way). Synthesize a placeholder.
                    rec = {
                        "index": row * n + col + 1,
                        "row": row,
                        "col": col,
                        "success": False,
                        "error": "tile not reached",
                    }
                ordered.append(rec)

        successful = [t for t in ordered if t.get("success")]
        fail_count = len(ordered) - len(successful)

        best_tile: dict[str, Any] | None = None
        if pd.get("assess_quality") and successful:
            rated = [t for t in successful if t.get("quality") is not None]
            if rated:
                best_tile = max(rated, key=lambda t: t["quality"])

        return {
            "grid_n": n,
            "tile_size_m": pd.get("tile_size_m"),
            "total_size_m": pd.get("total_size_m"),
            "effective_size_m": pd.get("effective_size_m"),
            "center": pd.get("center"),
            "tiles": ordered,
            "tile_count": n * n,
            "success_count": len(successful),
            "fail_count": fail_count,
            "recommended_tile": best_tile,
        }

    # ------------------------------------------------------------------
    # run_composite override: pre-flight grid validation + custom success
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        # Pre-flight grid validation — fail fast with the same error messages
        # v1 callers expect (test_total_must_exceed_tile_size etc.)
        (n, tile, line_time, cx, cy, total, channels,
         wait_timeout, assess_quality, err) = self._compute_grid(params)
        if err:
            return self.fail(err)

        effective_size = n * tile
        logger.info(
            "SurveySurface_TileScan: %dx%d grid, tile=%.2e m, total=%.2e m, "
            "effective=%.2e m, center=(%.2e, %.2e)",
            n, n, tile, total, effective_size, cx, cy,
        )

        # Set up executor + stash params for aggregate
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        executor.set_partial("grid_n", n)
        executor.set_partial("tile_size_m", tile)
        executor.set_partial("total_size_m", total)
        executor.set_partial("effective_size_m", effective_size)
        executor.set_partial("center", (cx, cy))
        executor.set_partial("assess_quality", assess_quality)
        executor.set_partial("wait_timeout_s", wait_timeout)
        # tile_records is resume-friendly: don't wipe prior state
        executor.set_partial_default("tile_records", {})
        self._executor = executor

        all_good = executor.run_plan(iter(self.plan(params)))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()

        # Partial-success semantics (matches v1 + GridSTS): we report ok as
        # long as we weren't externally aborted; per-tile failures live in
        # tile_records / fail_count for downstream consumers to inspect.
        # If the executor recorded an explicit abort (e.g. external abort
        # flag), fail with that reason.
        if executor.progress.aborted:
            return self.fail(
                executor.progress.aborted_reason or "survey aborted",
                **data,
            )
        if data.get("fail_count", 0) > 0:
            logger.warning(
                "Survey completed with %d/%d failed tiles",
                data["fail_count"], n * n,
            )
        return self.ok(**data)


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(SurveySurface_TileScan, context_provider)
