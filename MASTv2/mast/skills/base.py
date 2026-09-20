"""BaseSkill: abstract base class for all MAST skills.

vendored from v1 mast/skills/base.py 2026-04-23. Zero behavioural changes.
Required by every v1 skill (130 builtins + 5 composite + 21 paper).
mast.agents._shared.skill_adapter.wrap_skill is the SOLE point that imports
this in agent code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from mast.core.preconditions import (
    PRECONDITION_CHECKS as _PRECONDITION_CHECKS,
    check_state_preconditions as _check_state_preconditions,
    context_suffix as _precondition_context_suffix,
    precondition_recognized as _precondition_recognized,
)
from mast.core.types import (
    HardwareState,
    ParameterSpec,
    SkillMetadata,
    SkillResult,
)

# Module-level constants — avoid recreating per call
_VALIDATE_TYPE_MAP: dict[str, type] = {"float": float, "int": int, "str": str, "bool": bool}

# 审查 [#86]: the precondition vocabulary now lives in ONE place —
# mast.core.preconditions.PRECONDITION_CHECKS — the single source of truth shared
# with SafetyGuard / SafetyGate, so the exact-key (here) and substring (safety)
# layers can no longer drift. It still maps each precondition string to a
# (HardwareState attr, expected value) pair; re-exported under the historical
# _PRECONDITION_CHECKS name (tests/v2/unit/test_preconditions.py imports it).
#
# Earlier HIGH review (2026-05-30): motor.py (MotorMove / MotorMoveClosedLoop)
# and zcontrol.py (SetZPosition) declare "z_controller_off", which used to be
# UNKNOWN here → check_preconditions flagged it as unverifiable → those skills
# always failed, AND the "is the Z controller off before coarse motion?"
# tip-crash safety check never ran. That key (and the scan_not_running alias)
# is now part of the canonical table in mast.core.preconditions.


class BaseSkill(ABC):
    """Abstract base class for all MAST skills."""

    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """Return skill metadata: name, version, params, safety level, etc."""
        ...

    def validate_params(self, params: dict) -> list[str]:
        """Validate parameters against metadata specs.

        Returns list of error messages (empty list means valid).
        Default implementation checks required params exist and types/ranges match.
        """
        errors: list[str] = []
        meta = self.metadata()
        spec_by_name: dict[str, ParameterSpec] = {p.name: p for p in meta.parameters}

        # Check required params are present
        for spec in meta.parameters:
            if spec.required and spec.name not in params:
                errors.append(f"Missing required parameter: '{spec.name}'")

        # Validate each provided param
        for key, value in params.items():
            spec = spec_by_name.get(key)
            if spec is None:
                errors.append(f"Unknown parameter: '{key}'")
                continue

            # 审查 HIGH [#31]: the LangChain tool path
            # (skill_adapter._schema_from_metadata) builds the pydantic args
            # schema with `default = None` for any optional ParameterSpec whose
            # `default` is None. When the LLM omits such a param, pydantic
            # materialises it as None and passes it straight into _run kwargs →
            # validate_params. Previously `not isinstance(None, float)` raised a
            # spurious "must be float, got NoneType" error, which blocked core
            # write skills (e.g. SetBias when slew_rate_v_per_s is omitted) on
            # the agent path. A None value for an OPTIONAL param means "not
            # provided" — the execute() bodies already fall back via
            # params.get(name, <default>) — so skip type/range/allowed checks.
            # (Required params with a None value still flow through the type
            # check below and are correctly rejected.)
            if value is None and not spec.required:
                continue

            expected_type = _VALIDATE_TYPE_MAP.get(spec.type)
            if expected_type is not None:
                # Allow int where float is expected.
                # 审查 [#145]: previously this wrote the coerced
                # float back into the *caller's* dict (`params[key] = value`),
                # a hidden side-effect in a method whose contract is "return a
                # list of errors". Callers that validate the same dict against
                # several skills, log the original params, or re-validate after
                # a failure saw their ints silently mutated to floats. Coerce
                # only the LOCAL `value` for the range/allowed checks below and
                # leave the caller's dict untouched — execute() bodies already
                # accept an int where a float is expected (Nanonis packs both).
                if spec.type == "float" and isinstance(value, int):
                    value = float(value)
                elif not isinstance(value, expected_type):
                    errors.append(
                        f"Parameter '{key}' must be {spec.type}, got {type(value).__name__}"
                    )
                    continue

            if spec.min_value is not None and isinstance(value, (int, float)):
                if value < spec.min_value:
                    errors.append(
                        f"Parameter '{key}' = {value} is below minimum {spec.min_value}"
                    )
            if spec.max_value is not None and isinstance(value, (int, float)):
                if value > spec.max_value:
                    errors.append(
                        f"Parameter '{key}' = {value} is above maximum {spec.max_value}"
                    )
            if spec.allowed_values is not None and value not in spec.allowed_values:
                errors.append(
                    f"Parameter '{key}' = {value!r} not in {spec.allowed_values}"
                )

        return errors

    def check_preconditions(self, state: HardwareState) -> list[str]:
        """Check hardware preconditions against current state.

        Returns list of unmet condition descriptions (empty means all met).
        Default implementation checks metadata.preconditions against state fields.
        """
        unmet: list[str] = []
        meta = self.metadata()

        for condition in meta.preconditions:
            check = _PRECONDITION_CHECKS.get(condition)
            if check is None:
                # Not an exact-name key. It may still be a known substring-rule
                # precondition (e.g. 'bias_nonzero') enforced via
                # check_state_preconditions for the SafetyGate. Evaluate it there
                # so a SATISFIED precondition isn't mis-reported as "Cannot
                # verify" — the 2026-07-06 bug where bias_nonzero blocked EVERY
                # approach because it was absent from this exact-name dict.
                subs = _check_state_preconditions([condition], state)
                if subs:
                    unmet.extend(subs)
                elif not _precondition_recognized(condition):
                    unmet.append(f"Cannot verify precondition: '{condition}'")
                continue
            attr, expected = check
            actual = getattr(state, attr, None)
            if actual is None:
                unmet.append(f"Precondition '{condition}': state.{attr} is unknown")
            elif actual != expected:
                # 报告伴随状态，避免把多个不同控制器状态折叠成裸布尔。
                # 字段定义见 preconditions.PRECONDITION_CONTEXT_FIELDS。
                unmet.append(
                    f"Precondition '{condition}' not met: "
                    f"state.{attr} is {actual}, expected {expected}"
                    + _precondition_context_suffix(attr, state)
                )

        # WRITE DOWN WHY IT WAS REFUSED (2026-07-12).
        #
        # 「进针功能调用失败」  was reported with no cause, and we could not
        # supply one afterwards: an unmet precondition produced a ToolMessage to
        # the model and then vanished. The operator saw a failure; nothing on disk
        # said which condition, on which state field, with what actual value. This
        # line is the difference between "the approach failed" and "the approach
        # was refused because state.bias_v was 0.0 and it needs bias_nonzero".
        if unmet:
            try:
                from mast.core.diagnostics import record

                record(
                    "precondition_block", meta.name,
                    "; ".join(unmet)[:300],
                    unmet=unmet,
                    declared=list(meta.preconditions),
                    # the live state the check was made against — the other half
                    # of the answer, and the half nobody ever had
                    state={
                        k: getattr(state, k, None)
                        # ``z_controller_status`` 补于 2026-08-10:台账里只记布尔
                        # 时,事后复盘同样分不出 Hold / SafeTip / 真的 Off ——
                        # 而事后复盘正是这份台账存在的全部理由。
                        # ``withdrawn`` —— 这里原本写的是 ``tip_withdrawn``,
                        # 而 HardwareState 上**没有这个字段**,于是被下面那个
                        # ``hasattr`` 静默跳过:「针尖是不是退开的」从上线起
                        # 一次都没进过台账。`hasattr` 守卫把一个拼写错误变成了
                        # 一条永远缺席的记录 —— 同一形状本仓记过
                        # (silent_fallback_wrong_name)。
                        for k in ("z_controller_on", "z_controller_status",
                                  "scan_running", "bias_v",
                                  "current_a", "setpoint_a", "withdrawn")
                        if hasattr(state, k)
                    },
                )
            except Exception:  # noqa: BLE001 — diagnostics never break a skill
                pass

        return unmet

    @abstractmethod
    def execute(self, context: "ExecutionContext", params: dict) -> SkillResult:
        """Execute the skill. Must return SkillResult.

        Args:
            context: ExecutionContext providing safe_call() and sub-skill access.
            params: Validated parameter dict.
        """
        ...

    def abortable_sleep(
        self, context, seconds: float, poll_interval: float = 0.1,
    ) -> bool:
        """Sleep with periodic abort checks.

        Returns ``True`` if aborted (caller should exit), ``False`` if the full
        duration elapsed normally.

        为什么在 **builtin** 基类而不是 composite 基类：这件事和「是不是
        composite」无关，只和「这一步要等一段时间」有关。2026-08-22 的实证：
        新写的 builtin ``CharacteriseQuietDrift`` 默认要驻停 30 分钟，照着
        composite 的写法调了 ``self.abortable_sleep`` —— 而它继承的是本类，
        于是运行时抛 ``AttributeError``。**抛异常的 skill 到不了 agent 的错误
        处理路径**：没有 SkillResult、没有诊断记录、没有 HITL、没有恢复，
        只有一个死掉的回合（``test_all_skills_execute`` 的原话）。

        ``CompositeSkill`` 继承本类，所以它原来那一份已经删掉 —— 同一个动作
        两份实现，早晚只有一份是对的。
        """
        elapsed = 0.0
        while elapsed < seconds:
            chunk = min(poll_interval, seconds - elapsed)
            time.sleep(chunk)
            elapsed += chunk
            if hasattr(context, "check_abort") and context.check_abort():
                return True
        return False

    def rollback(self, context: "ExecutionContext", params: dict) -> None:
        """Optional rollback action. Default is no-op."""
        pass
