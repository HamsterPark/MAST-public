"""TipPulse — basic tip conditioning via bias pulse (Phase 7 graph-shaped composite).

Migration note (Phase 7):
  Original v1 implementation called Bias_Get + Bias_Set + sleep + Bias_Set
  in a software loop. The graph version replaces the per-iteration
  Set+sleep+Set with the hardware-timed ``BiasPulse`` builtin
  (``Bias_Pulse`` Nanonis TCP), which atomically pulses the bias to
  ``pulse_v`` for ``duration_s`` and restores the controller state. The
  outcome (bias spikes to pulse_v for duration_s then returns) is
  functionally equivalent. We still snapshot the bias up front via
  ``GetBias`` so callers can verify the restore in result.data.
"""

# K (Keep) — migrated to CompositeSkillGraph 2026-05-19

from __future__ import annotations

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


class TipPulse(CompositeSkillGraph):
    """Apply bias voltage pulses for basic tip conditioning.

    Snapshots current bias, applies *count* hardware-timed pulses at
    *pulse_v* for *duration_s*, and reports the original bias for
    downstream verification. Inspired by Scanbot's rule-based tip repair.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TipPulse",
            version="1.1.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"bias_pulse"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "施加偏压脉冲修针。先快照当前偏压，打脉冲，再恢复。"
            ),
            parameters=[
                ParameterSpec(
                    name="pulse_v",
                    type="float",
                    description=(
                        "脉冲偏压。**没有具体理由就别填** —— 留空时，它按已登记"
                        "针尖（材料 × 制法 × 形态）从针尖策略表里取：钨的电化学"
                        "腐蚀针和 PtIr 剪切针受不住同一个脉冲。你**填了**的值"
                        "会被采用，但超出该针尖的安全包络时是**拒绝**（不是夹紧）。"
                    ),
                    unit="V",
                    required=False,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="duration_s",
                    type="float",
                    description="每次脉冲的时长（留空 → 针尖策略表）",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.01,
                    max_value=5.0,
                ),
                ParameterSpec(
                    name="count",
                    type="int",
                    description="打几次脉冲（留空 → 针尖策略表）",
                    required=False,
                    default=1,
                    min_value=1,
                    max_value=50,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=5.0,
            composition_level=3,
            tags=["tip", "pulse", "conditioning", "composite"],
        )

    def validate_params(self, params: dict) -> list[str]:
        """标准校验 + 针尖安全包络。

        包络在这里而不是 execute:它必须在任何硬件调用之前拦住,而 validate 的
        错误会原样回给调用方(模型/用户看到的是「为什么被拒」而不是一次静默的
        no-op)。**超上限拒绝,不夹紧** —— 夹了调用方会以为自己用的是原来那个值。
        """
        errors = super().validate_params(params)
        from mast.skills.builtins._tip_policy import apply_tip_policy
        _, plan = apply_tip_policy(params, ("pulse_v", "pulse_count"),
                                   {"pulse_count": "count"})
        if plan is not None and not plan.ok:
            errors.extend(plan.refusals)
        return errors

    # --- Plan: static (count alone determines step list) ---

    def plan(self, params: dict) -> list[CompositeStep]:
        # 没给的参数按当前针尖查方案表 —— 模型不该替物理发明电压。
        from mast.skills.builtins._tip_policy import apply_tip_policy
        params, plan = apply_tip_policy(
            params, ("pulse_v", "pulse_duration_s", "pulse_count"),
            {"pulse_duration_s": "duration_s", "pulse_count": "count"})
        self._tip_plan = plan
        pulse_v = params.get("pulse_v")
        if pulse_v is None:
            # 方案表也没给(解析层不可用)——退到既有的保守默认,而不是 KeyError。
            pulse_v = 3.0
        duration_s = params.get("duration_s", 0.1)
        count = params.get("count", 1)

        steps: list[CompositeStep] = []
        # 1. Snapshot original bias so callers can see it in result.data
        steps.append(CompositeStep(
            step_id="snapshot_bias",
            skill_name="GetBias",
            params={},
            optional=False,
            checkpoint_after=False,
            tags=("snapshot",),
        ))
        # 2. Apply N hardware-timed pulses. Each BiasPulse pulses to
        #    pulse_v for duration_s and auto-restores via Nanonis hardware
        #    (Bias_Pulse(wait=1, width_s, bias_v, z_hold=1, absolute=True)).
        for i in range(1, count + 1):
            steps.append(CompositeStep(
                step_id=f"pulse_{i}",
                skill_name="BiasPulse",
                params={
                    "width_s": duration_s,
                    "bias_v": pulse_v,
                    # 1 = hold Z during the pulse (BiasPulse 自身默认也是 1)。
                    # 曾是 0 ("no change"):脉冲期间反馈仍在追电流,而几伏的脉冲
                    # 会让电流暴冲若干数量级 —— Z 会被一路压向表面。
                    "z_hold": 1,
                    "absolute": True,
                },
                optional=False,
                # Only checkpoint after the final pulse: tight inner loop
                # of N short pulses, flushing every one is overhead.
                checkpoint_after=(i == count),
                tags=("pulse", f"i={i}"),
            ))
        return steps

    # --- Hooks ---

    def on_step_result(self, step: CompositeStep, sub_result) -> None:
        # Capture original bias from the snapshot step so aggregate()
        # can emit it. GetBias returns data={"bias_v": float}.
        if step.step_id == "snapshot_bias":
            data = getattr(sub_result, "data", {}) or {}
            original = data.get("bias_v")
            if original is not None:
                self._executor.set_partial("original_bias_v", original)

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        out = {
            "pulse_v": progress.partial_data.get("pulse_v"),
            "duration_s": progress.partial_data.get("duration_s"),
            "count": progress.partial_data.get("count"),
            "original_bias_v": progress.partial_data.get("original_bias_v"),
        }
        # 参数来源痕迹:每个数字是调用方给的还是方案表给的,事后必须查得到。
        for key in ("tip_policy", "tip_policy_notes", "tip_registered", "tip_name"):
            if key in progress.partial_data:
                out[key] = progress.partial_data[key]
        return out

    # --- Custom run_composite to seed partial_data and force ok() semantics ---

    def run_composite(self, context, params: dict) -> SkillResult:
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        # 先按当前针尖补齐没给的参数,再种 partial_data —— 否则 aggregate 报出去
        # 的会是「模型给了什么」而不是「实际打了什么」。
        from mast.skills.builtins._tip_policy import (
            apply_tip_policy,
            policy_fields_for_result,
        )
        params, plan = apply_tip_policy(
            params, ("pulse_v", "pulse_duration_s", "pulse_count"),
            {"pulse_duration_s": "duration_s", "pulse_count": "count"})
        self._tip_plan = plan
        if plan is not None and not plan.ok:
            return self.fail("；".join(plan.refusals))

        # Stash params so aggregate() can read them without re-deriving
        executor.set_partial("pulse_v", params.get("pulse_v", 3.0))
        executor.set_partial("duration_s", params.get("duration_s", 0.1))
        executor.set_partial("count", params.get("count", 1))
        for key, val in policy_fields_for_result(plan).items():
            executor.set_partial(key, val)
        self._executor = executor

        all_good = executor.run_plan(iter(self.plan(params)))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()

        if all_good:
            return self.ok(**data)
        return self.fail(
            executor.progress.aborted_reason or "tip_pulse aborted",
            **data,
        )


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(TipPulse, context_provider)
