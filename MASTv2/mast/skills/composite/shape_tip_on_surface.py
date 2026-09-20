"""ShapeTipOnSurface — find flat spot, plunge the tip from shallow to deep,
detect contact via real-time current, then check whether the crater is round.

Workflow (one attempt):
    1. Use a recent .sxm (or trigger a fresh wide scan if none provided).
    2. FindFlatRegion → (cx_m, cy_m).
    3. Re-centre the scan frame on (cx_m, cy_m), width = `cluster_window_m`.
    4. Progressive tip-shaper sequence with `n_depth_steps` increasing depths:
         a. Run TipShape at the current depth.
         b. MonitorCurrent for `monitor_after_s` seconds to catch contact.
         c. If contact_detected (|I| > contact_threshold_a) → break.
    5. If contact made, run a fresh small scan over the spot,
       call AssessClusterRoundness, decide accept-or-relocate.
    6. If not round, add this (cx,cy) to the excluded list and try again.

Up to `max_attempts` iterations. Returns success if a round cluster is found,
otherwise fails with diagnostics. Always restores feedback + scan frame at end.

Phase 7 migration: subclasses CompositeSkillGraph and uses ``plan_dynamic``
(a generator) because each attempt's step sequence depends on prior results
(stopping early on contact, deciding accept-or-relocate from roundness).
Step IDs are stable (``attempt_<n>:phase[:sub]``) so a partially-completed
shape session can be resumed.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)
from mast.agents._shared.skill_adapter import wrap_skill

logger = logging.getLogger(__name__)

#: `round_threshold` 作废(2026-08-11)的拒绝文案 —— **一份,两个入口共用**
#: (`validate_params` 走 agent 那条路,`plan_dynamic` 顶上那份走直接
#: `.execute()` 那条路)。写成常量是因为「两个地方各写一句」正是让其中一句
#: 悄悄过期的做法。
_ROUND_THRESHOLD_RETIRED = (
    "`round_threshold` 已作废(2026-08-11),请用 `min_axis_ratio`"
    "(等效轴比,完美圆 = 1.0;0.75 =「不比长短轴差 25% 的椭圆更不规则」)。"
    "**两个数不可换算**:旧的比的是 0.6*circularity+0.4*aspect,而那个 "
    "circularity = 4πA/P² 在像素化边界上的**上确界只有 0.617**(轴对齐正方形却是 "
    "0.785)—— 阈值 0.65 卡在两者之间,**圆的一律不合格、方的一律合格**。"
    "不做静默别名是因为两个阈值**方向相同** —— 一个漏改的 0.65 不会崩,"
    "只会悄悄把闸门放宽到「长短轴差 35%」。")


class ShapeTipOnSurface(CompositeSkillGraph):
    """Find flat spot → plunge tip → verify cluster roundness (graph-shaped)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ShapeTipOnSurface",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"tip_shaping"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在样品上端到端地整形针尖：先在一张大图上找一块平区，"
                "再用 TipShape 逐级往下扎、同时盯电流判接触，然后扫那个坑"
                "并评估圆度。簇不对称就换一个新点位重试。最后恢复反馈 + "
                "扫描框。"
            ),
            parameters=[
                ParameterSpec(
                    name="wide_scan_path",
                    type="str",
                    description=(
                        "用来挑平区的那张已有 .sxm 大图的路径。"
                        "留空则现扫一张新的大图。"
                    ),
                    required=False,
                    default="",
                ),
                ParameterSpec(
                    name="wide_scan_width_m",
                    type="float",
                    description=(
                        "没给 scan_path 时，那张搜索用大图的宽度，单位 METRES。"
                        "常见 50n–200n（50–200 nm）。"
                        "例：100n（= 100 nm）✓；1（= 1 m！）✗。"
                    ),
                    unit="m",
                    required=False,
                    default=1e-7,
                    min_value=1e-9,
                    max_value=1e-5,
                ),
                ParameterSpec(
                    name="cluster_window_m",
                    type="float",
                    description=(
                        "扎完之后那个小扫描窗的边长，单位 METRES。"
                        "常见 3n–10n（3–10 nm）。"
                    ),
                    unit="m",
                    required=False,
                    default=5e-9,
                    min_value=1e-9,
                    max_value=5e-8,
                ),
                ParameterSpec(
                    name="shallow_depth_m",
                    type="float",
                    description=(
                        "最浅的扎针深度（相对当前位置的负 Z），单位 METRES。"
                        "常见 -300p（= -0.3 nm）。"
                    ),
                    unit="m",
                    required=False,
                    default=-5e-10,
                    min_value=-1e-8,
                    max_value=-1e-11,
                ),
                ParameterSpec(
                    name="deep_depth_m",
                    type="float",
                    description=(
                        "最深的扎针深度（更负的 Z），单位 METRES。"
                        "常见 -3n to -5n（= -3 to -5 nm）。簇的大小随深度增长。"
                        "上限 ±1 µm。"
                    ),
                    unit="m",
                    required=False,
                    default=-3e-9,
                    min_value=-1e-6,
                    max_value=-1e-11,
                ),
                ParameterSpec(
                    name="n_depth_steps",
                    type="int",
                    description=(
                        "在 shallow_depth_m 与 deep_depth_m 之间分几档深度"
                        "（linspace）。"
                    ),
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=20,
                ),
                ParameterSpec(
                    name="contact_threshold_a",
                    type="float",
                    description=(
                        "MonitorCurrent 用来判定「已接触」的电流绝对值阈值（A）。"
                        "常见 50n（50 nA）。"
                    ),
                    unit="A",
                    required=False,
                    default=5e-8,
                    min_value=1e-11,
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="monitor_after_s",
                    type="float",
                    description=(
                        "每一步整形之后盯电流盯多久，单位 SECONDS。"
                        "常见 0.2–0.5。"
                    ),
                    unit="s",
                    required=False,
                    default=0.3,
                    min_value=0.05,
                    max_value=5.0,
                ),
                ParameterSpec(
                    name="min_axis_ratio",
                    type="float",
                    description=(
                        "坑的**等效轴比**达到这个值或更高就算通过："
                        "0.75 = 「不比一个两轴相差 25% 的椭圆更不规则」。"
                        "低于它 → 换一个新点位重试。"
                        "它**取代**了 round_threshold=0.65（2026-08-11）：那一个比的是 "
                        "0.6*circularity+0.4*aspect，在那把尺子上一个**完美圆盘**"
                        "最高只到 0.770，而一个轴对齐的**正方形**却能拿 0.871。"
                        "**这两个数不可互换。**"
                    ),
                    required=False,
                    default=0.75,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ParameterSpec(
                    name="max_attempts",
                    type="int",
                    description="跨不同点位的最大重试次数。",
                    required=False,
                    default=3,
                    min_value=1,
                    max_value=10,
                ),
                ParameterSpec(
                    name="allow_on_qplus",
                    type="bool",
                    description=(
                        "默认关 —— 把针尖往表面里扎几 nm **就是**整形针尖的"
                        "常规做法，不需要谁批准。"
                        "只有守卫被重新打开时（MAST_QPLUS_POKE_GUARD=1）这个"
                        "开关才有意义。真正保护音叉的是**深度包络**"
                        "（超限是 REFUSED，绝不夹紧）和扎之前**自动把偏压降到 "
                        "20 mV** —— 一次跳过了那道降压的扎针会把音叉激振，"
                        "**毁针尖的是那个，不是扎针本身。**"
                    ),
                    required=False,
                    default=False,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=180.0,
            composition_level=3,
            tags=["tip", "shaper", "cluster", "composite", "autonomous"],
        )

    # ------------------------------------------------------------------
    # Helpers — extract scalar fields off a step result (resume / dict)
    # ------------------------------------------------------------------

    @staticmethod
    def _result_data(executor: GraphExecutor, step_id: str) -> dict[str, Any]:
        """Return data dict for a completed step (handles missing/None)."""
        res = executor.sub_results.get(step_id)
        if res is None:
            return {}
        return getattr(res, "data", None) or {}

    @staticmethod
    def _result_path(data: dict[str, Any]) -> str:
        """Return the saved .sxm path from a GetLatestScanFile result."""
        return data.get("path", "") or data.get("file_path", "")

    @staticmethod
    def _depths(shallow: float, deep: float, n_steps: int) -> list[float]:
        if n_steps <= 1:
            return [deep]
        step_size = (deep - shallow) / (n_steps - 1)
        return [shallow + i * step_size for i in range(n_steps)]

    # ------------------------------------------------------------------
    # Dynamic plan — step sequence depends on prior step results
    # ------------------------------------------------------------------

    def validate_params(self, params: dict) -> list[str]:
        """标准校验 + 针尖安全包络(下压深度超上限拒绝,不夹紧)。"""
        errors = super().validate_params(params)
        from mast.skills.builtins._tip_policy import apply_tip_policy
        _, plan = apply_tip_policy(
            params, ("poke_shallow_depth_m", "poke_deep_depth_m"),
            {"poke_shallow_depth_m": "shallow_depth_m",
             "poke_deep_depth_m": "deep_depth_m"})
        if plan is not None and not plan.ok:
            errors.extend(plan.refusals)
        # ── `round_threshold` 作废 —— 两个入口都要拒 ──────────────────────
        #
        # 2026-08-11。这里是 **agent / LLM 那条路**(``skill_adapter`` 会调
        # ``validate_params``);``plan_dynamic`` 顶上还有一份,那是**直接
        # ``.execute()``** 那条路(``validate_params`` 不在其中)。同一条规则、
        # 同一段文案(``_ROUND_THRESHOLD_RETIRED``),两个入口 —— 与本文件里
        # ``qplus_gate`` 的做法一致。
        #
        # ⚠️ 第一版只写在 ``plan_dynamic`` 里,而且写成 ``return SkillResult(...)``
        # —— ``plan_dynamic`` 是**生成器**,``return`` 只结束迭代、返回值被丢掉。
        # 于是它**照跑不误**:用自己的 ``min_axis_ratio`` 默认值 0.75 扎完三轮,
        # 再报「找不到够圆的簇」。调用方以为自己设的是 0.65,实际跑的是 0.75 ——
        # **正是这条检查要挡的那种「能跑的错版本」,只是换成了本地自造的一份。**
        # 是测试里那句 `assert not ctx.run_log` 抓到的。
        if params.get("round_threshold") is not None:
            errors.append(_ROUND_THRESHOLD_RETIRED)
        return errors

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        # 这个技能把针尖**主动压进表面**做成簇 —— 对 qPlus 传感器是毁灭性的
        # (石英音叉损坏不可逆,要拆机重装并重新标定 f₀/Q)。默认拒绝,要做得签字。
        from mast.skills.builtins._tip_policy import (
            apply_tip_policy,
            policy_fields_for_result,
            qplus_gate,
        )
        # 作废的阈值:**在任何一步硬件动作之前**拦掉。这一份管的是直接
        # ``.execute()`` 的调用方(``validate_params`` 只在 agent 那条路上跑)。
        if params.get("round_threshold") is not None:
            executor.progress.aborted = True
            executor.progress.aborted_reason = _ROUND_THRESHOLD_RETIRED
            return
        gate = qplus_gate("ShapeTipOnSurface", params)
        if gate is not None:
            executor.progress.aborted = True
            executor.progress.aborted_reason = gate.error
            return
        params, tip_plan = apply_tip_policy(
            params, ("poke_shallow_depth_m", "poke_deep_depth_m", "poke_steps"),
            {"poke_shallow_depth_m": "shallow_depth_m",
             "poke_deep_depth_m": "deep_depth_m",
             "poke_steps": "n_depth_steps"})
        if tip_plan is not None and not tip_plan.ok:
            executor.progress.aborted = True
            executor.progress.aborted_reason = "；".join(tip_plan.refusals)
            return
        for _k, _v in policy_fields_for_result(tip_plan).items():
            executor.set_partial(_k, _v)

        wide_scan_path = (params.get("wide_scan_path") or "").strip()
        wide_w = float(params.get("wide_scan_width_m", 1e-7))
        cluster_w = float(params.get("cluster_window_m", 5e-9))
        shallow = float(params.get("shallow_depth_m", -5e-10))
        deep = float(params.get("deep_depth_m", -3e-9))
        n_steps = int(params.get("n_depth_steps", 5))
        contact_thr = float(params.get("contact_threshold_a", 5e-8))
        monitor_s = float(params.get("monitor_after_s", 0.3))
        # `round_threshold` 的作废拒绝在 `validate_params` 里(见那里的说明:
        # 这里是生成器,`return SkillResult(...)` 会被丢掉)。
        min_axis_ratio = float(params.get("min_axis_ratio", 0.75))
        max_attempts = int(params.get("max_attempts", 3))
        depths = self._depths(shallow, deep, n_steps)

        # ── Save original frame (so we can restore at finalize) ──────
        yield CompositeStep(
            step_id="get_original_frame",
            skill_name="GetScanFrame",
            params={},
            optional=True,
            checkpoint_after=False,
            tags=("setup",),
        )
        original_frame = self._result_data(executor, "get_original_frame")
        executor.set_partial("original_frame", dict(original_frame))

        excluded_spots: list[tuple[float, float]] = list(
            executor.progress.partial_data.get("excluded_spots", [])
        )
        attempt_log: list[dict[str, Any]] = list(
            executor.progress.partial_data.get("attempt_log", [])
        )

        accepted_attempt: int | None = None
        accept_data: dict[str, Any] | None = None

        for attempt in range(1, max_attempts + 1):
            prefix = f"attempt_{attempt}"
            logger.info("ShapeTipOnSurface attempt %d/%d", attempt, max_attempts)

            # ── 1. Acquire / locate wide scan path ───────────────────
            if attempt == 1 and wide_scan_path:
                scan_path = wide_scan_path
            else:
                # Trigger a fresh wide scan: ConfigureScan → StartScan →
                # WaitScanComplete → SaveScan → GetLatestScanFile
                cx0 = original_frame.get("center_x_m", 0.0)
                cy0 = original_frame.get("center_y_m", 0.0)
                yield CompositeStep(
                    step_id=f"{prefix}:wide_configure",
                    skill_name="ConfigureScan",
                    params={
                        "center_x_m": cx0,
                        "center_y_m": cy0,
                        "width_m": wide_w,
                        "height_m": wide_w,
                    },
                    optional=False,
                    checkpoint_after=False,
                    tags=("wide_scan", f"attempt={attempt}"),
                )
                yield CompositeStep(
                    step_id=f"{prefix}:wide_start",
                    skill_name="StartScan",
                    params={"direction": "up"},
                    optional=False,
                    checkpoint_after=False,
                    tags=("wide_scan", f"attempt={attempt}"),
                )
                yield CompositeStep(
                    step_id=f"{prefix}:wide_wait",
                    skill_name="WaitScanComplete",
                    params={"timeout_ms": 300_000},
                    optional=False,
                    checkpoint_after=False,
                    tags=("wide_scan", f"attempt={attempt}"),
                )
                yield CompositeStep(
                    step_id=f"{prefix}:wide_save",
                    skill_name="SaveScan",
                    params={},
                    optional=True,
                    checkpoint_after=False,
                    tags=("wide_scan", f"attempt={attempt}"),
                )
                yield CompositeStep(
                    step_id=f"{prefix}:wide_latest",
                    skill_name="GetLatestScanFile",
                    params={},
                    optional=False,
                    checkpoint_after=True,
                    tags=("wide_scan", f"attempt={attempt}"),
                )
                latest_data = self._result_data(executor, f"{prefix}:wide_latest")
                save_data = self._result_data(executor, f"{prefix}:wide_save")
                scan_path = self._result_path(latest_data) or self._result_path(save_data)
                if not scan_path:
                    attempt_log.append({
                        "attempt": attempt, "stage": "fresh_wide_scan",
                        "error": "no saved wide scan path",
                    })
                    executor.set_partial("attempt_log", attempt_log)
                    # No wide scan → can't continue this attempt; bail out
                    # of the whole composite (mirrors v1 hard-fail).
                    executor.set_partial("hard_fail",
                                         "Could not produce a wide scan for FindFlatRegion")
                    break

            # ── 2. FindFlatRegion (mandatory — bail on failure) ──────
            excl_str = ";".join(f"{x},{y}" for x, y in excluded_spots)
            yield CompositeStep(
                step_id=f"{prefix}:find_flat",
                skill_name="FindFlatRegion",
                params={
                    "scan_path": scan_path,
                    "exclude_used_spots": excl_str,
                    "min_separation_m": cluster_w,
                },
                optional=False,
                checkpoint_after=True,
                tags=("find_flat", f"attempt={attempt}"),
            )
            flat_data = self._result_data(executor, f"{prefix}:find_flat")
            cx, cy = flat_data.get("center_x_m"), flat_data.get("center_y_m")
            if cx is None or cy is None:
                # find_flat failure was caught by executor; we just stop.
                break
            logger.info(
                "  flat spot: (%.3e, %.3e) m  rms=%s",
                cx, cy, flat_data.get("rms_m"),
            )

            # ── 3. Recentre the scan frame ───────────────────────────
            yield CompositeStep(
                step_id=f"{prefix}:configure_cluster",
                skill_name="ConfigureScan",
                params={
                    "center_x_m": cx,
                    "center_y_m": cy,
                    "width_m": cluster_w,
                    "height_m": cluster_w,
                },
                optional=False,
                checkpoint_after=False,
                tags=("configure_cluster", f"attempt={attempt}"),
            )

            # ── 4. Progressive plunge with contact detection ─────────
            plunge_log: list[dict[str, Any]] = []
            contact_made = False
            contact_depth_m: float | None = None
            for i, d in enumerate(depths, start=1):
                # Use TipShapeWithReadback (wait=0 CONCURRENT current+Z sampling)
                # so the contact signal is captured DURING the plunge. The old
                # code ran a blocking TipShape (feedback restored) and only THEN
                # sampled current — by which point the Z controller had servo'd
                # the current back to the pA-nA setpoint, so the contact spike was
                # already gone and every spot read "no_contact".
                # Contact is now judged from the permanent Z change (z3−z1 indent
                # verdict) plus any in-process current jump.
                yield CompositeStep(
                    step_id=f"{prefix}:plunge_{i}:shape",
                    skill_name="TipShapeWithReadback",
                    params={
                        "tip_lift_m": d,
                        "lift_height_m": abs(d),
                        "change_bias": False,
                        "restore_feedback": True,
                        "lift_time_1_s": 0.1,
                        "lift_time_2_s": 0.1,
                        "end_wait_s": 0.05,
                        "timeout_ms": 30000,
                        "poll_hz": 2000.0,
                        "pre_roll_s": 0.05,
                        "post_roll_s": 0.05,
                    },
                    optional=True,           # one shape failure ≠ abort
                    checkpoint_after=True,
                    tags=("plunge", f"attempt={attempt}", f"step={i}"),
                )
                shape_data = self._result_data(executor, f"{prefix}:plunge_{i}:shape")
                shape_success = f"{prefix}:plunge_{i}:shape" in executor.sub_results

                indent = shape_data.get("indent") or {}
                verdict = indent.get("verdict")
                cur_jump = ((shape_data.get("jumps") or {}).get("current") or {}).get(
                    "max_abs_delta")
                # Contact = a permanent Z change (tip touched: cluster / pit /
                # tip-changed) OR a clear in-process current jump above threshold.
                contact_this = (
                    verdict in ("cluster", "tip_changed_or_pit")
                    or (isinstance(cur_jump, (int, float)) and cur_jump > contact_thr)
                )

                entry = {
                    "step": i,
                    "depth_m": d,
                    "shape_success": shape_success,
                    "indent_verdict": verdict,
                    "delta_m": indent.get("delta_m"),
                    "current_jump_a": cur_jump,
                    "contact_detected": bool(contact_this),
                }
                plunge_log.append(entry)

                if shape_success and contact_this:
                    contact_made = True
                    contact_depth_m = d
                    break

            if not contact_made:
                excluded_spots.append((cx, cy))
                attempt_log.append({
                    "attempt": attempt, "spot": [cx, cy],
                    "stage": "plunge", "outcome": "no_contact",
                    "plunge_log": plunge_log,
                })
                executor.set_partial("excluded_spots", list(excluded_spots))
                executor.set_partial("attempt_log", attempt_log)
                continue

            # ── 5. Small cluster scan ────────────────────────────────
            yield CompositeStep(
                step_id=f"{prefix}:cluster_configure",
                skill_name="ConfigureScan",
                params={
                    "center_x_m": cx,
                    "center_y_m": cy,
                    "width_m": cluster_w,
                    "height_m": cluster_w,
                },
                optional=False,
                checkpoint_after=False,
                tags=("cluster_scan", f"attempt={attempt}"),
            )
            yield CompositeStep(
                step_id=f"{prefix}:cluster_start",
                skill_name="StartScan",
                params={"direction": "up"},
                optional=True,
                checkpoint_after=False,
                tags=("cluster_scan", f"attempt={attempt}"),
            )
            cluster_started = f"{prefix}:cluster_start" in executor.sub_results
            if cluster_started:
                yield CompositeStep(
                    step_id=f"{prefix}:cluster_wait",
                    skill_name="WaitScanComplete",
                    params={"timeout_ms": 120_000},
                    optional=True,
                    checkpoint_after=False,
                    tags=("cluster_scan", f"attempt={attempt}"),
                )
                yield CompositeStep(
                    step_id=f"{prefix}:cluster_save",
                    skill_name="SaveScan",
                    params={},
                    optional=True,
                    checkpoint_after=False,
                    tags=("cluster_scan", f"attempt={attempt}"),
                )
            yield CompositeStep(
                step_id=f"{prefix}:cluster_latest",
                skill_name="GetLatestScanFile",
                params={},
                optional=True,
                checkpoint_after=True,
                tags=("cluster_scan", f"attempt={attempt}"),
            )
            latest_data = self._result_data(executor, f"{prefix}:cluster_latest")
            save_data = self._result_data(executor, f"{prefix}:cluster_save")
            small_scan_path = (
                self._result_path(latest_data) or self._result_path(save_data)
            )
            if not small_scan_path:
                excluded_spots.append((cx, cy))
                attempt_log.append({
                    "attempt": attempt, "spot": [cx, cy],
                    "stage": "cluster_scan", "outcome": "no_scan_file",
                })
                executor.set_partial("excluded_spots", list(excluded_spots))
                executor.set_partial("attempt_log", attempt_log)
                continue

            # ── 6. Roundness check ───────────────────────────────────
            yield CompositeStep(
                step_id=f"{prefix}:assess_roundness",
                skill_name="AssessClusterRoundness",
                params={
                    "scan_path": small_scan_path,
                    "min_axis_ratio": min_axis_ratio,
                    # 评估本次产生的凸起时显式选用 bright 极性。
                    "polarity": "bright",
                },
                optional=True,
                checkpoint_after=True,
                tags=("roundness", f"attempt={attempt}"),
            )
            assess_data = self._result_data(executor, f"{prefix}:assess_roundness")
            assess_success = f"{prefix}:assess_roundness" in executor.sub_results
            if not assess_success:
                excluded_spots.append((cx, cy))
                attempt_log.append({
                    "attempt": attempt, "spot": [cx, cy],
                    "stage": "AssessClusterRoundness", "error": "assess failed",
                })
                executor.set_partial("excluded_spots", list(excluded_spots))
                executor.set_partial("attempt_log", attempt_log)
                continue

            attempt_log.append({
                "attempt": attempt,
                "spot": [cx, cy],
                "contact_depth_m": contact_depth_m,
                # 等效轴比：「相当于一个短轴/长轴 = q 的椭圆」。
                # 用户的用法是**比较相继几次哪次更圆**,
                # 所以这个连续量比 is_round 更该进日志。
                "equivalent_axis_ratio": assess_data.get("equivalent_axis_ratio"),
                "is_round": assess_data.get("is_round"),
                # 三态：None = 团簇太小,**判不了**,不是「不圆」。
                # 两者都不接受(不能把没验证过的针尖当整好了),但日志里要分得开 ——
                # 「一直不圆」和「一直判不了」要采取的下一步动作完全不同。
                "roundness_undecidable": assess_data.get("roundness_undecidable"),
                "cluster_scan_path": small_scan_path,
            })
            executor.set_partial("attempt_log", attempt_log)

            if assess_data.get("is_round") is True:
                accepted_attempt = attempt
                accept_data = {
                    "success_attempt": attempt,
                    "spot_x_m": cx,
                    "spot_y_m": cy,
                    "contact_depth_m": contact_depth_m,
                    "cluster_scan_path": small_scan_path,
                    "equivalent_axis_ratio": assess_data.get("equivalent_axis_ratio"),
                }
                executor.set_partial("accept_data", accept_data)
                break

            excluded_spots.append((cx, cy))
            executor.set_partial("excluded_spots", list(excluded_spots))

        # ── Cleanup / finalize: always restore feedback + scan frame ──
        # These run regardless of success/failure (mirrors v1 `finally`).
        yield CompositeStep(
            step_id="finalize:restore_feedback",
            skill_name="ZControllerOnOff",
            # 参数名是 ``enable`` (zcontrol.ZControllerOnOff)。曾写成 ``on``:
            # validate_params 直接判失败,而这一步 optional=True 会把失败吞掉 ——
            # 反馈没恢复且无人知道,类文档「Always restores feedback」是假的。
            params={"enable": True},
            optional=True,
            checkpoint_after=False,
            tags=("finalize",),
        )
        if original_frame:
            yield CompositeStep(
                step_id="finalize:restore_frame",
                skill_name="ConfigureScan",
                params={
                    "center_x_m": original_frame.get("center_x_m", 0.0),
                    "center_y_m": original_frame.get("center_y_m", 0.0),
                    "width_m": original_frame.get("width_m", 1e-7),
                    "height_m": original_frame.get("height_m", 1e-7),
                },
                optional=True,
                checkpoint_after=True,
                tags=("finalize",),
            )

        # Record terminal disposition for aggregate()
        executor.set_partial(
            "outcome_summary",
            {
                "accepted_attempt": accepted_attempt,
                "max_attempts": max_attempts,
            },
        )

    # ------------------------------------------------------------------
    # Aggregate + run_composite override (custom success semantics)
    # ------------------------------------------------------------------

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        out: dict[str, Any] = {
            "attempts": pd.get("outcome_summary", {}).get("max_attempts", 0),
            "attempt_log": list(pd.get("attempt_log", [])),
        }
        accept = pd.get("accept_data")
        if accept:
            out.update(accept)
            # On success ``attempts`` should match the accepted-attempt index,
            # matching v1's return-value shape.
            out["attempts"] = accept.get("success_attempt", out["attempts"])
        out["points"] = self._plunge_points(out["attempt_log"])
        return out

    @staticmethod
    def _plunge_points(attempt_log: list) -> list[dict]:
        """Every place this composite drove the tip into the surface.

        Additive view over ``attempt_log`` — that list keeps its own key names,
        which other readers depend on. The recorder looks for a ``points`` list
        with ``x_m``/``y_m``, so without this translation a run that plunged at
        five different spots left ONE map marker and four invisible craters. Each
        one becomes its own avoidance zone, which is the whole point: debris from
        tip forming is why the usable area shrinks."""
        out: list[dict] = []
        for entry in attempt_log or []:
            if not isinstance(entry, dict):
                continue
            spot = entry.get("spot")
            if not (isinstance(spot, (list, tuple)) and len(spot) >= 2):
                continue
            try:
                x, y = float(spot[0]), float(spot[1])
            except (TypeError, ValueError):
                continue
            out.append({
                "x_m": x, "y_m": y,
                "index": entry.get("attempt"),
                "label": f"修针尖尝试#{entry.get('attempt')}",
                # The tip touched the surface here either way — a rejected
                # cluster is still a crater. ``success`` only says whether this
                # attempt produced the round apex we wanted.
                "success": entry.get("is_round") is True,
                "error": entry.get("outcome") or entry.get("error"),
            })
        return out

    def run_composite(self, context, params: dict) -> SkillResult:
        # ⑫ module preflight: this composite drives TipShape/TipShapeWithReadback,
        # which need the Nanonis Tip Shaper module running. Check it ONCE up front
        # (field trace s306 discovered it off only after several steps).
        # Fail-open — only a clearly "module not running" probe early-exits here.
        from mast.skills.composite._preflight import (
            PROBE_TIP_SHAPER,
            preflight_modules,
        )
        missing = preflight_modules(
            context, (PROBE_TIP_SHAPER,), accumulator=self._all_calls)
        if missing:
            return self.fail(missing)

        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        # Accumulators: preserve across resume
        executor.set_partial_default("attempt_log", [])
        executor.set_partial_default("excluded_spots", [])
        self._executor = executor

        all_good = executor.run_plan(self.plan_dynamic(params, executor))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()

        # Custom success semantics:
        #   • Success iff a round cluster was accepted (accept_data set).
        #   • If hard_fail was raised mid-plan (no wide scan), fail with that.
        #   • Otherwise fail with "no round cluster after N attempts" — same
        #     message shape v1 callers expect.
        pd = executor.progress.partial_data
        if pd.get("accept_data"):
            return self.ok(**data)
        hard_fail = pd.get("hard_fail")
        if hard_fail:
            return self.fail(hard_fail, **data)
        if executor.progress.aborted:
            return self.fail(
                executor.progress.aborted_reason or "shape aborted",
                **data,
            )
        max_attempts = int(params.get("max_attempts", 3))
        return self.fail(
            f"No round cluster found after {max_attempts} attempts",
            **data,
        )


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(ShapeTipOnSurface, context_provider)
