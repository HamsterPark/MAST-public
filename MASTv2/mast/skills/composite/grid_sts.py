"""GridSTS — sparse grid STS acquisition (Phase 7 graph-shaped composite)."""

from __future__ import annotations

import logging
from typing import Iterator

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

# Ceiling on the per-point bookkeeping that rides in partial_data (and therefore
# into every checkpoint) and on the marker list handed to the recorder. 20×20
# covers any grid anyone runs interactively; beyond it the per-point breakdown is
# dropped and the counters still tell the whole story.
_MAX_TRACKED_POINTS = 400


class GridSTS(CompositeSkillGraph):
    """Acquire STS spectra on a regular NxN grid.

    For each grid point: MoveToXY -> AcquireSTS.
    Inspired by the gpSTS sparse acquisition approach (fixed grid,
    without Gaussian process -- GP optimisation is a future extension).
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GridSTS",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在 NxN 栅格上采 STS 谱：逐点移动，每点采一条谱。"
            ),
            parameters=[
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description="栅格中心 X",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="栅格中心 Y",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="nx",
                    type="int",
                    description="X 方向的栅格点数",
                    required=False,
                    default=3,
                    min_value=1,
                    max_value=20,
                ),
                ParameterSpec(
                    name="ny",
                    type="int",
                    description="Y 方向的栅格点数",
                    required=False,
                    default=3,
                    min_value=1,
                    max_value=20,
                ),
                ParameterSpec(
                    name="spacing_m",
                    type="float",
                    description="栅格点间距",
                    unit="m",
                    required=True,
                    min_value=1e-10,
                ),
                ParameterSpec(
                    name="start_v",
                    type="float",
                    description="STS 起始偏压",
                    unit="V",
                    required=False,
                    default=-2.0,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="end_v",
                    type="float",
                    description="STS 终止偏压",
                    unit="V",
                    required=False,
                    default=2.0,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="num_points",
                    type="int",
                    description="谱的采样点数",
                    required=False,
                    default=40,
                    min_value=10,
                    max_value=10000,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=300.0,
            composition_level=3,
            tags=["spectroscopy", "grid", "sts", "composite"],
        )

    # --- Plan: static (params fully determine the grid) ---

    def plan(self, params: dict) -> list[CompositeStep]:
        cx = params["center_x_m"]
        cy = params["center_y_m"]
        nx = params.get("nx", 3)
        ny = params.get("ny", 3)
        spacing = params["spacing_m"]
        start_v = params.get("start_v", -2.0)
        end_v = params.get("end_v", 2.0)
        # Fallback MUST match the metadata() default for num_points (40);
        # an out-of-sync 200 here meant an omitted param silently acquired
        # 5x the declared points (single source of truth = metadata default).
        num_points = params.get("num_points", 40)

        steps: list[CompositeStep] = []
        steps.append(CompositeStep(
            step_id="configure",
            skill_name="ConfigureSTS",
            params={"start_v": start_v, "end_v": end_v, "num_points": num_points},
            optional=False,
            checkpoint_after=True,
            tags=("setup",),
        ))
        for iy in range(ny):
            for ix in range(nx):
                x = cx + (ix - (nx - 1) / 2.0) * spacing
                y = cy + (iy - (ny - 1) / 2.0) * spacing
                steps.append(CompositeStep(
                    step_id=f"move_{ix}_{iy}",
                    skill_name="MoveToXY",
                    params={"x_m": x, "y_m": y, "wait": True},
                    optional=True,   # one bad move ≠ whole grid abort
                    checkpoint_after=False,
                    tags=("move", f"ix={ix}", f"iy={iy}"),
                ))
                steps.append(CompositeStep(
                    step_id=f"sts_{ix}_{iy}",
                    skill_name="AcquireSTS",
                    params={},
                    optional=True,
                    checkpoint_after=True,    # checkpoint per acquired spectrum
                    tags=("sts", f"ix={ix}", f"iy={iy}"),
                ))
        return steps

    # --- Hooks ---

    def on_step_result(self, step: CompositeStep, sub_result) -> None:
        if step.step_id.startswith("sts_"):
            suffix = step.step_id[len("sts_"):]   # "ix_iy"
            failed_moves = self._executor.progress.partial_data.get(
                "failed_moves", [])
            if suffix in failed_moves:
                # The MoveToXY to this grid point FAILED, so this spectrum was
                # acquired at the WRONG (previous) position — do NOT count it as a
                # good data point. It used to be tallied as "succeeded", recording
                # mislocated spectra as valid grid data.
                suspect = self._executor.progress.partial_data.setdefault(
                    "suspect", 0)
                self._executor.set_partial("suspect", suspect + 1)
                self._note_point(suffix, "suspect")
                return
            sts = self._executor.progress.partial_data.setdefault("succeeded", 0)
            self._executor.set_partial("succeeded", sts + 1)
            self._note_point(suffix, "ok")

    def _note_point(self, suffix: str, status: str) -> None:
        """Record one grid point's outcome by its ``ix_iy`` suffix.

        A flat str→str map so it survives checkpointing, and so a resumed run
        keeps the outcomes it already established."""
        try:
            seen = dict(self._executor.progress.partial_data.get("point_status") or {})
            if len(seen) >= _MAX_TRACKED_POINTS and suffix not in seen:
                return
            seen[suffix] = status
            self._executor.set_partial("point_status", seen)
        except Exception:  # noqa: BLE001 — bookkeeping never breaks the grid
            pass

    def on_step_failed(self, step: CompositeStep, msg: str) -> bool:
        if step.step_id.startswith("move_"):
            # Remember which grid points we could NOT reach, so the spectrum at
            # that point isn't later counted as good data.
            suffix = step.step_id[len("move_"):]  # "ix_iy"
            fm = list(self._executor.progress.partial_data.get("failed_moves", []))
            if suffix not in fm:
                fm.append(suffix)
                self._executor.set_partial("failed_moves", fm)
        if step.step_id.startswith(("move_", "sts_")):
            failed = self._executor.progress.partial_data.setdefault("failed", 0)
            self._executor.set_partial("failed", failed + 1)
            self._note_point(step.step_id.split("_", 1)[1], "failed")
        # Respect step.optional (default behaviour)
        return step.optional

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        nx = max(1, int(progress.partial_data.get("nx", 0)))
        ny = max(1, int(progress.partial_data.get("ny", 0)))
        # ONE grid point costs TWO steps (move + sts), plus one configure step up
        # front: total_steps = 1 + 2·nx·ny. The old `total_steps - 1` therefore
        # reported 2·nx·ny — DOUBLE the real point count. A 3×3 grid narrated
        # itself as 18 points, and a total failure said "All 18 spectra failed"
        # for 9 real spectra. (The point count itself was suspected wrong.
        # It was — just not in the 0-vs-1 indexing that was first suspected.)
        has_dims = ("nx" in progress.partial_data and "ny" in progress.partial_data)
        total_points = (
            nx * ny if has_dims
            else max(0, (progress.total_steps - 1) // 2)  # aborted before configure
        )
        return {
            "total_points": total_points,
            "succeeded": int(progress.partial_data.get("succeeded", 0)),
            "failed": int(progress.partial_data.get("failed", 0)),
            # Spectra acquired after a FAILED move (wrong position) — not counted
            # as good data.
            "suspect": int(progress.partial_data.get("suspect", 0)),
            "nx": nx,
            "ny": ny,
            "spacing_m": progress.partial_data.get("spacing_m"),
            "points": self._grid_points(progress),
        }

    @staticmethod
    def _grid_points(progress: CompositeProgress) -> list[dict]:
        """Every grid point with its coordinate and outcome.

        The recorder turns this into one map marker per spectrum (it looks for a
        ``points`` list carrying x/y — same contract BatchRegionsScan uses for
        its regions). Coordinates are rebuilt with the SAME formula ``plan()``
        used, from the origin and spacing stashed in partial_data, so the map
        cannot disagree with where the tip was actually sent.

        Empty when the origin is unknown (a run aborted before it was recorded):
        the recorder then falls back to one marker, which is honest — we have
        nothing to place the individual points with."""
        cx = progress.partial_data.get("center_x_m")
        cy = progress.partial_data.get("center_y_m")
        spacing = progress.partial_data.get("spacing_m")
        if cx is None or cy is None or not spacing:
            return []
        nx = max(1, int(progress.partial_data.get("nx", 0) or 0))
        ny = max(1, int(progress.partial_data.get("ny", 0) or 0))
        if nx * ny > _MAX_TRACKED_POINTS:
            return []
        status = progress.partial_data.get("point_status") or {}
        out: list[dict] = []
        for iy in range(ny):
            for ix in range(nx):
                st = status.get(f"{ix}_{iy}")
                if st is None:
                    continue      # never reached (aborted / resumed past it)
                out.append({
                    "x_m": float(cx) + (ix - (nx - 1) / 2.0) * float(spacing),
                    "y_m": float(cy) + (iy - (ny - 1) / 2.0) * float(spacing),
                    "index": iy * nx + ix + 1,
                    "label": f"网格谱 ({ix},{iy})",
                    # A spectrum taken after a failed move sits at the WRONG
                    # place, so it is not a good point — same judgement the
                    # counters make.
                    "success": st == "ok",
                    "error": None if st == "ok" else (
                        "移动失败,谱点位置不可信" if st == "suspect" else "失败"),
                })
        return out

    def run_composite(self, context, params: dict) -> SkillResult:
        # Capture grid dims into partial_data BEFORE the executor starts
        # so the aggregate() helper can read them back without rederiving.
        nx = params.get("nx", 3)
        ny = params.get("ny", 3)

        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        executor.set_partial("nx", nx)
        executor.set_partial("ny", ny)
        executor.set_partial("spacing_m", params.get("spacing_m"))
        # Grid origin, so aggregate() can rebuild each point's coordinate with
        # the same formula plan() used. Without these the whole N×N grid reached
        # the scan map as a SINGLE marker at the last position visited — the map
        # is supposed to show where every spectrum was taken.
        executor.set_partial("center_x_m", params.get("center_x_m"))
        executor.set_partial("center_y_m", params.get("center_y_m"))
        # Accumulators: preserve across resume
        executor.set_partial_default("succeeded", 0)
        executor.set_partial_default("failed", 0)
        self._executor = executor

        all_good = executor.run_plan(iter(self.plan(params)))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()

        # This override bypasses _base._graph_execute, whose success path is
        # the ONLY place the step sidecar was cleared — so a finished grid's
        # sidecar lived forever and the NEXT same-size grid resume-skipped its
        # leading points ("第5个点扫描不到", 2026-07-10 #95). Any run that
        # reached its end (success or non-aborted partial) must never seed a
        # future one; only an aborted/interrupted run keeps its sidecar for
        # the legitimate crash-resume window.
        if not executor.progress.aborted:
            executor.clear_sidecar()

        # GridSTS succeeds if at least one STS succeeded (partial-success
        # semantics, same as v1).
        if data.get("succeeded", 0) > 0:
            return self.ok(**data)
        reason = executor.progress.aborted_reason or (
            f"All {data['total_points']} spectra failed"
        )
        return self.fail(reason, **data)


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(GridSTS, context_provider)
