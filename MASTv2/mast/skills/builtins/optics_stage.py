"""Optical-bench stage skills — TERS/THz motion hardware (non-Nanonis).

7 skills: ListOpticalDevices, OpticalStageGetPos, OpticalStageMove,
          StopOpticalStage, HomeOpticalStage, DelayLineGetDelay,
          DelayLineMoveTo.

Hardware access goes through the module-level
:func:`mast.instruments.registry.get_instrument_registry` singleton (the
plan_overlay pattern) — NOT through ExecutionContext, which is the Nanonis
path. Soft travel limits are enforced inside the drivers (Layer-0), so
these skills only translate errors into SkillResults; they cannot bypass
the limits even with hallucinated parameters.

On machines without the optical bench every skill degrades to a failed
SkillResult with a clear message — never an exception (UI 绝不冻结 /
no-hardware-no-crash).
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.instruments.base import (
    InstrumentError,
    InstrumentUnavailable,
    MotionTimeout,
    TravelLimitError,
)
from mast.instruments.registry import get_instrument_registry
from mast.skills.base import BaseSkill

__all__ = [
    "ListOpticalDevices",
    "OpticalStageGetPos",
    "OpticalStageMove",
    "OpticalStageWiggle",
    "StopOpticalStage",
    "HomeOpticalStage",
    "DelayLineGetDelay",
    "DelayLineMoveTo",
]


def _fail(skill: str, exc: Exception) -> SkillResult:
    """Uniform failure translation for the driver exception family."""
    if isinstance(exc, InstrumentUnavailable):
        msg = f"optical hardware unavailable: {exc}"
    elif isinstance(exc, TravelLimitError):
        msg = f"refused by soft travel limit: {exc}"
    elif isinstance(exc, MotionTimeout):
        msg = f"motion timeout: {exc}"
    elif isinstance(exc, InstrumentError):
        msg = str(exc)
    else:
        msg = f"{type(exc).__name__}: {exc}"
    return SkillResult(skill_name=skill, success=False, error=msg)


class ListOpticalDevices(BaseSkill):
    """List configured optical-bench devices (stages, delay line)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ListOpticalDevices",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从 config/optical_instruments.json 列出光学平台上的运动设备："
                "id、类型、各轴（含行程限位与角色）、连接状态，以及（若已配置）泵浦-探测延时线的绑定关系。"
                "先从这里开始，才知道其余光学技能接受哪些 device_id/axis 取值。"
            ),
            estimated_duration_s=0.1,
            composition_level=0,
            tags=["optics", "inventory", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            reg = get_instrument_registry()
            devices = reg.list_devices()
            delay_cfg = None
            try:
                dl = reg.delay_line()
                lo, hi = dl.delay_range_ps
                delay_cfg = {
                    "device_id": dl.config.device_id,
                    "axis": dl.config.axis,
                    "ps_per_mm": dl.config.ps_per_mm,
                    "zero_offset_mm": dl.config.zero_offset_mm,
                    "delay_range_ps": [lo, hi],
                }
            except InstrumentError:
                pass  # unconfigured / unavailable — inventory still useful
            return SkillResult(
                skill_name="ListOpticalDevices",
                success=True,
                data={"devices": devices, "delay_line": delay_cfg},
                summary=(
                    f"{len(devices)} optical device(s) configured"
                    + (", delay line bound" if delay_cfg else ", no delay line")
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("ListOpticalDevices", exc)


class OpticalStageGetPos(BaseSkill):
    """Read the position of one optical stage axis."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="OpticalStageGetPos",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读一条光学平台台子轴的当前位置（原生单位；单位与限位见 ListOpticalDevices）"
                "。"
            ),
            parameters=[
                ParameterSpec(
                    name="device_id",
                    type="str",
                    description="设备 id，取自 ListOpticalDevices",
                    required=True,
                ),
                ParameterSpec(
                    name="axis",
                    type="str",
                    description="该设备上的轴名",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["optics", "stage", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            ax = get_instrument_registry().axis(params["device_id"], params["axis"])
            pos = ax.get_position()
            return SkillResult(
                skill_name="OpticalStageGetPos",
                success=True,
                data={
                    "device_id": params["device_id"],
                    "axis": params["axis"],
                    "position": pos,
                    "unit": ax.config.unit,
                    "limits": [ax.config.min_pos, ax.config.max_pos],
                },
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("OpticalStageGetPos", exc)


class OpticalStageMove(BaseSkill):
    """Move one optical stage axis (absolute or relative)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="OpticalStageMove",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # CONFIRM: optical alignment moves are recoverable (soft limits
            # bound the travel) but a wrong large move loses tip-focus
            # alignment, so keep an operator/LLM confirmation in the loop.
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一条光学平台台子轴移动到某个位置（原生单位）。软行程限位由驱动强制执行，超出范围的目标在任何运动发生之前就会被拒绝。"
                "要做相对位移就用 relative=true。"
            ),
            parameters=[
                ParameterSpec(
                    name="device_id",
                    type="str",
                    description="设备 id，取自 ListOpticalDevices",
                    required=True,
                ),
                ParameterSpec(
                    name="axis",
                    type="str",
                    description="该设备上的轴名",
                    required=True,
                ),
                ParameterSpec(
                    name="position",
                    type="float",
                    description=(
                        "目标位置（绝对）或位移量（相对），用该轴的原生单位"
                    ),
                    required=True,
                ),
                ParameterSpec(
                    name="relative",
                    type="bool",
                    description="True = 从当前位置按位移量移动",
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="wait",
                    type="bool",
                    description="阻塞直到到位（推荐）",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="timeout_s",
                    type="float",
                    description="运动时限，单位秒",
                    required=False,
                    default=60.0,
                    min_value=0.1,
                    max_value=600.0,
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["optics", "stage", "move", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            ax = get_instrument_registry().axis(params["device_id"], params["axis"])
            wait = params.get("wait", True)
            timeout = params.get("timeout_s", 60.0)
            if params.get("relative", False):
                status = ax.move_rel(params["position"], wait=wait, timeout=timeout)
            else:
                status = ax.move_abs(params["position"], wait=wait, timeout=timeout)
            return SkillResult(
                skill_name="OpticalStageMove",
                success=True,
                data={
                    "device_id": params["device_id"],
                    "axis": params["axis"],
                    "position": status.position,
                    "unit": ax.config.unit,
                    "on_target": status.on_target,
                    "moving": status.moving,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("OpticalStageMove", exc)


class OpticalStageWiggle(BaseSkill):
    """Bring-up self-test: nudge an axis a little and verify it moves.

    Answers the first question at the instrument: "is this axis wired,
    alive, and moving in the direction/scale I think?" — WITHOUT touching
    the STM. Reads the position, moves by a small delta, reads back, then
    returns to the start. Reports observed vs commanded displacement so a
    dead axis (no motion), a reversed encoder (wrong sign), or a wrong
    ``counts_per_unit`` scale (ratio far from 1) each show up on the spot.
    Near a soft limit it automatically wiggles the other way.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="OpticalStageWiggle",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # CONFIRM: it moves hardware. Small and self-reversing, but motion
            # is motion — the operator/LLM confirms. During bring-up the human
            # is at the rig, so one confirmation per test is the right friction.
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "对一条光学台轴做诊断性抖动：按一个小位移量走过去再走回来，然后报告它到底动没动、往哪个方向动、"
                "以及实测量／指令量之比（标度核查）。装机调试期间用它确认某条轴是活的、配置也对，再去信任更大的移动。"
                "在标度得到确认之前，先从很小的位移量起步。"
            ),
            parameters=[
                ParameterSpec(
                    name="device_id",
                    type="str",
                    description="设备 id，取自 ListOpticalDevices",
                    required=True,
                ),
                ParameterSpec(
                    name="axis",
                    type="str",
                    description="该设备上的轴名",
                    required=True,
                ),
                ParameterSpec(
                    name="delta",
                    type="float",
                    description=(
                        "轻推的幅度，用该轴的原生单位（标度确认之前请保持很小）"
                    ),
                    required=True,
                ),
                ParameterSpec(
                    name="tolerance",
                    type="float",
                    description=(
                        "算作「动了」所需的最小 |位移|；默认 = |delta| 的 20%"
                    ),
                    required=False,
                    default=0.0,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="return_to_start",
                    type="bool",
                    description="结束后移回起始位置",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="timeout_s",
                    type="float",
                    description="每次移动的时限，单位秒",
                    required=False,
                    default=30.0,
                    min_value=0.1,
                    max_value=600.0,
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["optics", "stage", "diagnostic", "selftest", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            ax = get_instrument_registry().axis(params["device_id"], params["axis"])
            delta = float(params["delta"])
            if delta == 0.0:
                return SkillResult(
                    skill_name="OpticalStageWiggle",
                    success=False,
                    error="delta must be non-zero",
                )
            tol = params.get("tolerance") or 0.2 * abs(delta)
            timeout = params.get("timeout_s", 30.0)

            start = ax.get_position()

            # Move forward by delta; if that would cross a soft limit (axis is
            # parked at/near an end), wiggle the other way instead.
            used = delta
            try:
                fwd = ax.move_abs(start + delta, wait=True, timeout=timeout).position
            except TravelLimitError:
                used = -delta
                try:
                    fwd = ax.move_abs(start - delta, wait=True, timeout=timeout).position
                except TravelLimitError:
                    return SkillResult(
                        skill_name="OpticalStageWiggle",
                        success=False,
                        error=(
                            f"axis at {start:g} {ax.config.unit} cannot move ±"
                            f"{abs(delta):g} without crossing its soft limits "
                            f"[{ax.config.min_pos}, {ax.config.max_pos}] — "
                            "use a smaller delta or move off the limit first"
                        ),
                    )

            observed = fwd - start
            moved = abs(observed) >= tol
            direction_ok = (observed > 0) == (used > 0) if moved else None
            scale_ratio = observed / used if used else float("nan")

            # Return to start (best-effort; a failed return leaves a known,
            # small offset which we report rather than hide).
            final = fwd
            returned_to_start: bool | None = None
            if params.get("return_to_start", True):
                try:
                    final = ax.move_abs(start, wait=True, timeout=timeout).position
                    returned_to_start = abs(final - start) <= max(tol, 1e-9)
                except Exception:  # noqa: BLE001
                    returned_to_start = False

            if moved and direction_ok:
                summary = (
                    f"{params['axis']} moved {observed:+.4g} {ax.config.unit} "
                    f"(commanded {used:+.4g}, ratio {scale_ratio:.3f}) — alive"
                )
            elif moved and not direction_ok:
                summary = (
                    f"{params['axis']} moved {observed:+.4g} {ax.config.unit} "
                    f"but OPPOSITE to commanded {used:+.4g} — encoder/sign reversed"
                )
            else:
                summary = (
                    f"{params['axis']} did NOT move (Δ={observed:+.4g} "
                    f"{ax.config.unit} < tol {tol:g}) — check wiring/power/config"
                )

            return SkillResult(
                skill_name="OpticalStageWiggle",
                success=moved,
                data={
                    "device_id": params["device_id"],
                    "axis": params["axis"],
                    "unit": ax.config.unit,
                    "commanded_delta": used,
                    "start_position": start,
                    "forward_position": fwd,
                    "observed_delta": observed,
                    "moved": moved,
                    "direction_ok": direction_ok,
                    "scale_ratio": scale_ratio,
                    "final_position": final,
                    "returned_to_start": returned_to_start,
                },
                error="" if moved else "axis did not move within tolerance",
                summary=summary,
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("OpticalStageWiggle", exc)


class StopOpticalStage(BaseSkill):
    """Panic-stop optical stage motion (one device or the whole bench)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopOpticalStage",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,  # a stop must never wait for approval
            description=(
                "立即停止光学台的运动。不传 device_id 则对每一台已连接的光学设备做急停。"
            ),
            parameters=[
                ParameterSpec(
                    name="device_id",
                    type="str",
                    description="要停的设备；不传则停**所有**设备",
                    required=False,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["optics", "stage", "stop", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            reg = get_instrument_registry()
            device_id = params.get("device_id")
            if device_id:
                reg.controller(device_id).stop_all()
                scope = device_id
            else:
                reg.stop_all()
                scope = "all"
            return SkillResult(
                skill_name="StopOpticalStage",
                success=True,
                data={"stopped": scope},
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("StopOpticalStage", exc)


class HomeOpticalStage(BaseSkill):
    """Home / find-zero one optical stage axis."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="HomeOpticalStage",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # CONFIRM: homing sweeps the full travel — never run it while an
            # experiment depends on the current optical alignment.
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "给一条光学台轴做参考（寻零）。台子会朝它的参考标记扫过去，途中会**丢掉当前位置** —— 请先确认没有任何测量依赖于它。"
                "带绝对位置传感器的压电轴（PI E-816）会报告不需要寻零。"
            ),
            parameters=[
                ParameterSpec(
                    name="device_id",
                    type="str",
                    description="设备 id，取自 ListOpticalDevices",
                    required=True,
                ),
                ParameterSpec(
                    name="axis",
                    type="str",
                    description="该设备上的轴名",
                    required=True,
                ),
                ParameterSpec(
                    name="timeout_s",
                    type="float",
                    description="寻零时限，单位秒",
                    required=False,
                    default=120.0,
                    min_value=1.0,
                    max_value=600.0,
                ),
            ],
            estimated_duration_s=30.0,
            composition_level=0,
            tags=["optics", "stage", "home", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            ax = get_instrument_registry().axis(params["device_id"], params["axis"])
            status = ax.home(wait=True, timeout=params.get("timeout_s", 120.0))
            return SkillResult(
                skill_name="HomeOpticalStage",
                success=True,
                data={
                    "device_id": params["device_id"],
                    "axis": params["axis"],
                    "position": status.position,
                    "unit": ax.config.unit,
                    "homed": status.homed,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("HomeOpticalStage", exc)


class DelayLineGetDelay(BaseSkill):
    """Read the pump-probe optical delay."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="DelayLineGetDelay",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读当前泵浦-探测光学延时，单位皮秒（并给出由台子行程决定的可达延时范围）。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["optics", "delay_line", "pump_probe", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            dl = get_instrument_registry().delay_line()
            delay = dl.get_delay_ps()
            lo, hi = dl.delay_range_ps
            return SkillResult(
                skill_name="DelayLineGetDelay",
                success=True,
                data={
                    "delay_ps": delay,
                    "delay_range_ps": [lo, hi],
                    "stage_position": dl.axis.get_position(),
                    "stage_unit": dl.axis.config.unit,
                },
                summary=f"delay = {delay:.3f} ps (range {lo:.1f} … {hi:.1f} ps)",
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("DelayLineGetDelay", exc)


class DelayLineMoveTo(BaseSkill):
    """Move the pump-probe delay line to an optical delay (ps)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="DelayLineMoveTo",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把泵浦-探测延时线移到某个目标光学延时，单位皮秒（经标定过的零点偏移换算成台子位置；台子的软限位由驱动强制执行）"
                "。"
            ),
            parameters=[
                ParameterSpec(
                    name="delay_ps",
                    type="float",
                    description="目标光学延时，单位皮秒",
                    unit="ps",
                    required=True,
                ),
                ParameterSpec(
                    name="wait",
                    type="bool",
                    description="阻塞直到到位（推荐）",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="timeout_s",
                    type="float",
                    description="运动时限，单位秒",
                    required=False,
                    default=60.0,
                    min_value=0.1,
                    max_value=600.0,
                ),
            ],
            estimated_duration_s=3.0,
            composition_level=0,
            tags=["optics", "delay_line", "pump_probe", "move", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            dl = get_instrument_registry().delay_line()
            lo, hi = dl.delay_range_ps
            target = params["delay_ps"]
            if not (lo <= target <= hi):
                return SkillResult(
                    skill_name="DelayLineMoveTo",
                    success=False,
                    error=(
                        f"delay {target} ps outside reachable range "
                        f"[{lo:.2f}, {hi:.2f}] ps (stage travel limits)"
                    ),
                )
            status = dl.move_to_delay_ps(
                target,
                wait=params.get("wait", True),
                timeout=params.get("timeout_s", 60.0),
            )
            actual = dl.position_to_delay(status.position)
            return SkillResult(
                skill_name="DelayLineMoveTo",
                success=True,
                data={
                    "delay_ps": actual,
                    "requested_ps": target,
                    "stage_position": status.position,
                    "stage_unit": dl.axis.config.unit,
                    "on_target": status.on_target,
                },
                summary=f"delay line at {actual:.3f} ps",
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("DelayLineMoveTo", exc)
