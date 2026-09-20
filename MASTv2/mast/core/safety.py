"""SafetyGuard: three-layer safety — parameter bounds, state guards, approval levels."""

from __future__ import annotations

import logging

from mast.config import SafetyLimits
from mast.core.si_quantity import parse_quantity as _parse_quantity
from mast.core.preconditions import (
    check_state_preconditions as _shared_check_state_preconditions,
)
from mast.core.types import (
    HardwareState,
    OperatingMode,
    SafetyLevel,
    SkillMetadata,
)

logger = logging.getLogger(__name__)

# Maps parameter names/units to global safety limit fields
_GLOBAL_CHECKS: list[tuple[str, str, str, str]] = [
    # (param_name_pattern, unit_pattern, limits_min_attr, limits_max_attr)
    # Both name AND unit must match to trigger a check. `pattern` matches as a
    # SUBSTRING of the (lower-cased) parameter name, so "start_v" also covers
    # "bias_start_v" / "sts_start_v".
    ("bias_v", "v", "bias_min_v", "bias_max_v"),
    # Spectroscopy / sweep bias endpoints. Before this, the admin-tunable global
    # bias envelope (±10 V) covered ONLY the DC "bias_v" param — every STS / bias
    # sweep / pulse endpoint (start_v/end_v/lower_v/upper_v/pulse_v/bias_lift_v)
    # slipped past it, so an LLM-hallucinated start_v=-50 reached the junction.
    ("start_v", "v", "bias_min_v", "bias_max_v"),   # start_v / bias_start_v / sts_start_v
    ("end_v", "v", "bias_min_v", "bias_max_v"),      # end_v / bias_end_v / sts_end_v
    ("lower_v", "v", "bias_min_v", "bias_max_v"),
    ("upper_v", "v", "bias_min_v", "bias_max_v"),
    ("pulse_v", "v", "bias_min_v", "bias_max_v"),
    ("lift_v", "v", "bias_min_v", "bias_max_v"),     # bias_lift_v (tip-shaper bias)
    ("z_pos", "m", "z_min_m", "z_max_m"),
    # z_offset is a SIGNED RELATIVE fine-Z displacement (STS retract is negative);
    # gate it against the symmetric relative envelope, NOT the absolute z floor
    # (z_min_m=0.0 would reject every legitimate negative retract).
    ("z_offset", "m", "z_offset_min_m", "z_offset_max_m"),
    # Tip-shaper plunge/lift excursions — cap the hallucinated half-micron plunge.
    ("tip_lift", "m", "tip_lift_min_m", "tip_lift_max_m"),
    ("lift_height", "m", "tip_lift_min_m", "tip_lift_max_m"),
    ("deep_depth", "m", "tip_lift_min_m", "tip_lift_max_m"),
    ("x_m", "m", "xy_min_m", "xy_max_m"),
    ("y_m", "m", "xy_min_m", "xy_max_m"),
    ("center_x", "m", "xy_min_m", "xy_max_m"),
    ("center_y", "m", "xy_min_m", "xy_max_m"),
    ("setpoint", "a", "setpoint_min_a", "setpoint_max_a"),
    # Scan extent: width_m / height_m. Without this, LLM can pass width_m=2
    # (2 meters!) and validator misses it. Cap at 10 µm.
    ("width_m", "m", "scan_size_min_m", "scan_size_max_m"),
    ("height_m", "m", "scan_size_min_m", "scan_size_max_m"),
]


# ── Physical-plausibility floor (#118 deepening) ────────────────────────────
# A HARD, rig-INDEPENDENT ceiling on the magnitude of a physical quantity —
# distinct from the admin-tunable SafetyLimits envelope. An admin may widen
# setpoint_max_a, but a tunnelling current of 1.5 A is not "out of the configured
# range", it is physically impossible for any STM (a real setpoint is fA–µA; 1.5 A
# would vaporise the junction). That is almost always an order-of-magnitude / unit
# slip (the model wrote 1.5 meaning 1.5 nA). Catching it as "物理荒谬" FIRST gives
# the crispest possible retry signal — a magnitude fix, not a "raise the limit"
# invitation — and it holds even if the per-skill / admin bound was loosened.
#
# Keyed by (name-substring, unit-substring, |ceiling|, human hint). The ceilings
# are deliberately generous so a legitimate high-bias / high-current rig is never
# rejected here — only the physically-impossible order of magnitude is.
#: relative slack on every bound comparison. '100n' parses to 1.0000000000000001e-07 and
#: was refused against a 1e-07 ceiling — the value the operator meant IS the ceiling
#: (2026-09-11, STM-Bench trials). One part in 1e9 is far below any physical resolution.
_FLOAT_SLACK = 1e-9

_PHYSICAL_ABSURD: list[tuple[str, str, float, str]] = [
    # Tunnelling-current setpoint / current. Real STM: ~1 fA … 100 µA. ≥ 1 mA at a
    # tunnel junction is impossible (1.5 A is ~1e10× a 100 pA setpoint).
    ("setpoint", "a", 1e-3, "STM 隧道电流约 1 fA–100 µA(典型 1 pA–100 nA)"),
    ("current", "a", 1e-3, "STM 隧道电流约 1 fA–100 µA(典型 1 pA–100 nA)"),
    # Bias / sweep / pulse voltages. STM bias is a ±10 V-class quantity; the
    # admin ±10 V envelope already handles ordinary over-range, so the physical
    # floor here is deliberately generous (≥ 10 kV) — it only catches a truly
    # insane hallucination (e.g. 1e6 V) and never a merely-unusual high-bias rig.
    ("bias_v", "v", 1e4, "STM 偏压约 ±10 V 量级"),
    ("start_v", "v", 1e4, "STM 偏压约 ±10 V 量级"),
    ("end_v", "v", 1e4, "STM 偏压约 ±10 V 量级"),
    ("lower_v", "v", 1e4, "STM 偏压约 ±10 V 量级"),
    ("upper_v", "v", 1e4, "STM 偏压约 ±10 V 量级"),
    ("pulse_v", "v", 1e4, "STM 偏压约 ±10 V 量级"),
    ("lift_v", "v", 1e4, "STM 偏压约 ±10 V 量级"),
    # Z-controller gains. The P gain is a LENGTH (metres) and the I gain a SPEED
    # (m/s): a piezo's ENTIRE range is ~µm, so a P gain of 3 is not "aggressive
    # tuning", it is three METRES — an exponent that fell off in transit. That
    # is exactly what reached the hardware on 2026-08-03 (3e-12 → 3, a 1e12×
    # error) with nothing in the system to stop it. The ceilings sit ~3 decades
    # above the per-skill max_value so the two layers never argue: the bounds
    # layer rejects "wrong for this rig", this one rejects "not a length at all".
    # Unit matching here is EXACT (see below), so the dimensionless p_gain/i_gain
    # of the Kelvin loop, the generic PI controllers and the PLL are untouched.
    ("p_gain", "m", 1e-3, "Z 控制器 P 增益是长度,典型 1e-13–1e-9 m(压电总程仅 ~µm)"),
    ("i_gain", "m/s", 1.0, "Z 控制器 I 增益是速度,典型 1e-9–1e-5 m/s"),
]


def physically_absurd_violations(meta: SkillMetadata, params: dict) -> list[str]:
    """Return physical-impossibility violations (empty = physically plausible).

    Runs on the RAW params (numeric-looking strings coerced) BEFORE the tunable
    global-bounds check. A hit means the value is off by orders of magnitude —
    a unit / exponent mistake — not merely outside the configured envelope, so
    the message tells the model to fix the magnitude and NOT resend the same
    number, rather than implying the limit should be raised.
    """
    violations: list[str] = []
    specs = {p.name: p for p in meta.parameters}
    for name, value in (params or {}).items():
        spec = specs.get(name)
        if spec is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            # Dimensioned parameters reach the model as STRINGS since
            # 2026-08-04 (a number-typed tool argument was corrupted 12/12 on
            # this provider), so "3p" and "3e-12" both have to be understood
            # here. A plain float() would fail on the prefix form and skip the
            # check — turning the string fix into a hole in this table.
            try:
                value = _parse_quantity(value, strict=False, what=name)
            except Exception:  # noqa: BLE001 — genuinely non-numeric arg
                continue
        name_l = name.lower()
        unit_l = (spec.unit or "").lower()
        for pat, unit_pat, ceiling, hint in _PHYSICAL_ABSURD:
            if pat in name_l and unit_pat == unit_l and abs(value) >= ceiling:
                violations.append(
                    f"物理荒谬值: '{name}' = {value} {spec.unit or ''} 在物理上不可能"
                    f"({hint})。这不是越界,是量级丢了——请把值写成**带 SI 前缀的"
                    f"字符串**(100 pA→'100p'、1 nA→'1n'、3 pm→'3p';50 mV 这类接近 1 的"
                    f"量写 '0.05' 即可)。切勿重试相同数值,也不要改用 '1e-10' 这种"
                    f"指数写法——量级极小的参数会直接拒绝它。"
                )
                break
    return violations


def implausible_reading_notes(readings: "dict[str, tuple[float, str]]") -> list[str]:
    """The same table, pointed at values we READ instead of values we write.

    Why a read needs this at all
    ============================
    2026-08-03, the second failure of the same day. A gain of 3 m had reached the
    hardware. The agent then read it back —

        'gains': [3.0, 1.666700005531311, 1.7999639511108398]

    — and reported: "p_gain = 3.0（= 3e-12）… 偏差都在 float32 表示精度范围内,
    数值通道工作正常." It had compared them item by item in its reasoning and
    still concluded they agreed. 3.0 and 3e-12 differ by a factor of 1e12.

    The write error and the read error are the SAME defect expressed twice, and
    they point the same way, so they cover for each other: "I sent 3e-12" and "the
    3.0 I read back IS 3e-12" corroborate into one self-consistent false report.
    Acting on that report means approaching with P = 3 metres.

    So a bare number in a tool result is not neutral — it is an invitation to be
    narrated as fine. Annotating the magnitude removes that: a reading carrying
    "≈ 1e12× the typical value" cannot be quietly described as float32 noise.
    (`GetZControllerState` already does this for the on/off disagreement — "surface
    the disagreement rather than leaving two raw numbers for the model to
    reconcile". This is that same doctrine, applied to magnitude.)

    Args:
        readings: ``{name: (value, unit)}`` — the unit is the physical unit of the
            reading, matched EXACTLY against the table (same rule as the write
            side), so a dimensionless same-named gain never trips.

    Returns:
        Human-readable warnings, one per implausible reading. Empty = nothing in
        the batch is physically impossible (which is NOT the same as "correct").
    """
    notes: list[str] = []
    for name, pair in (readings or {}).items():
        try:
            value, unit = pair
        except (TypeError, ValueError):  # pragma: no cover — defensive
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
        name_l = str(name).lower()
        unit_l = (unit or "").lower()
        for pat, unit_pat, ceiling, hint in _PHYSICAL_ABSURD:
            if pat in name_l and unit_pat == unit_l and abs(value) >= ceiling:
                notes.append(
                    f"⚠ 读回值 '{name}' = {value!r} {unit or ''} 在物理上不可能"
                    f"({hint})。这个读数**不是浮点精度误差**,它比典型值大若干个"
                    f"数量级 —— 硬件里现在很可能真的是一个错误的量级。"
                    f"不要据此判定「数值正常」,先在 Nanonis 面板上人工核对。"
                )
                break
    return notes


def _is_truthy_absolute(val: object) -> bool:
    """Fail-closed parse of an ``absolute`` flag from raw (pre-pydantic) args.

    The safety gate sees the LLM's raw JSON, where ``absolute`` may arrive as a
    bool, an int (0/1), or a string ("true"/"1"/"yes"/"on"). Returns True for
    any recognised truthy form; a real ``False`` / 0 / "false" / "" returns
    False; anything else unrecognised errs toward True (gate rather than skip).
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "on", "y", "t"):
            return True
        if s in ("false", "0", "no", "off", "n", "f", ""):
            return False
        return True  # unrecognised non-empty string → gate (fail-closed)
    if val is None:
        return False
    return True  # unknown type → gate


def is_coarse_sample_approach(skill_name: str, params: dict | None) -> bool:
    """True iff this call is a coarse Z motion that can crash the tip into the sample.

    This is the ONE physically-dangerous action class in MAST (2026-06-11
    safety re-scoping). A coarse Z step toward the sample (pan-type piezo
    stepper) has NO current-feedback stop, so too many / too large steps crash
    the tip into the sample and can wreck tip + sample. Everything else is
    bounded by Nanonis (bias/current/fine-Z ranges) or by the current-feedback
    AutoApproach module, so this is the SOLE action gated behind human approval
    (BLOCKED on the autonomous agent path, human-approval on manual/composite).

    Two coarse-motor shapes are recognised:

    * ``MotorMove`` open-loop step — keyed on the SEMANTIC
      ``direction='z-approach'`` (independent of the raw Nanonis Z code).
    * ``MotorMoveClosedLoop`` move with a Z component — closed-loop coarse moves
      use POSITIONAL feedback (servo to a target XYZ); they do NOT stop on tip
      contact, and the Z direction toward/away from the sample cannot be proven
      safe from the static target alone (the sign depends on the instrument's Z
      convention and, for relative moves, the current position). So be
      CONSERVATIVE: treat ANY closed-loop move that touches Z as a coarse sample
      approach. A pure-XY closed-loop move (no Z) is NOT flagged.
    """
    p = params or {}
    if skill_name == "MotorMove":
        direction = str(p.get("direction", "")).strip().lower()
        return direction in ("z-approach", "z_approach", "zapproach")
    if skill_name == "MotorMoveClosedLoop":
        target_z = p.get("target_z_m")
        if target_z is None:
            return False  # pure-XY closed-loop move — no Z component
        # Absolute positioning with a Z target present: gate even at 0.0, since
        # an absolute Z target's "safe side" cannot be proven statically.
        #
        # This gate runs in SafetyGateMiddleware on the RAW LLM JSON args, BEFORE
        # pydantic coercion (safety_mw parses the model's tool-call dict). A weak
        # model routinely emits absolute="true" / absolute=1 (we already coerce
        # bias_v="50.0" strings elsewhere), and motor.py executes with a truthy
        # test (`1 if params.get("absolute") else 0`). A bare `is True` here saw
        # those as False and dropped into the relative branch — where a 0.0
        # target then read as "no Z move" and skipped the human-approval gate
        # entirely. Parse the flag fail-closed: any recognised truthy form gates,
        # and anything unparseable also gates (matches the target_z fallback).
        if _is_truthy_absolute(p.get("absolute")):
            return True
        # Relative move: any non-zero Z step is a coarse Z motion → gate.
        # An unparseable Z target fails closed (treated as a Z move).
        try:
            return _parse_quantity(target_z, strict=False, what="target_z") != 0.0
        except Exception:  # noqa: BLE001
            return True
    return False


def _is_falsey(val: object, *, default_true: bool = True) -> bool:
    """True iff *val* denotes OFF/disable. Missing (None) → uses default_true.

    Mirrors _is_truthy_absolute but for enable/enabled flags whose SAFE default
    is ON: a missing flag is treated as enable (not a disable), so only an
    explicit off value trips the gate."""
    if val is None:
        return not default_true  # missing → default (enable) → not a disable
    return not _is_truthy_absolute(val)


def is_calibration_change(skill_name: str, params: dict | None) -> bool:
    """True iff this call rewrites the global bias/current CALIBRATION scale.

    SetBiasCalibration / SetCurrentCalibration multiply every subsequent bias or
    current value system-wide — a 1000× factor turns a "safe" 1 V request into
    1 kV at the junction, silently defeating the ±10 V bound. These are rare,
    deliberate rig-setup operations that must never run autonomously (they are
    CONFIRM for the manual/HITL pane, but CONFIRM ≡ auto on the agent path, so
    the agent path needs an explicit gate — 审查)."""
    return skill_name in ("SetBiasCalibration", "SetCurrentCalibration")


def is_coarse_drive_change(skill_name: str, params: dict | None) -> bool:
    """True iff this call rewrites the coarse stepper's DRIVE amplitude/frequency.

    The one instrument parameter whose wrong value is unrecoverable by software.
    A pan-type coarse stepper is driven by a few-hundred-volt sawtooth, and the
    voltage the CONTROLLER can output is not the voltage the piezo STACK
    survives: some Nanonis controllers go to 400 V while some stacks fail at 300.
    Nothing reads back which rig you are on, and getting it wrong destroys the
    stack — no retry, no rollback, no scan.

    So this is not a judgement a language model should be making, and its
    baseline ``safety_level`` cannot say so: ``SetMotorFreqAmp`` is CONFIRM, and
    CONFIRM on the autonomous path means the model approves itself.

    Value-independent, unlike ``is_coarse_sample_approach``: there is no
    amplitude that is safe to set without knowing this rig, and "this rig" is
    knowledge only the operator has. The per-rig ceiling lives in
    ``mast.core.coarse_drive``; this predicate just makes sure a human is the one
    consulting it.

    Frequency rides along in the same call and is gated with it. It is far less
    dangerous on its own — a wrong frequency changes how far a step travels, not
    whether the stack survives — but it is not a number the agent needs either,
    and splitting the gate would mean a second path into the same setter."""
    return skill_name in ("SetMotorFreqAmp",)


def is_unguarded_lateral_coarse_move(skill_name: str, params: dict | None) -> bool:
    """True iff this is a RAW lateral coarse step, bypassing the guarded path.

    ``MotorMove(direction='x±'|'y±')`` slides the sample stage with the tip
    hanging over it. Its only precondition is ``state.withdrawn``, which means
    "the fine-Z piezo is at its high limit" — one or two microns of clearance —
    and that check passes when the state is simply unknown. The clearance a
    lateral move actually needs comes from stepping the COARSE Z motor back,
    which nothing here does; nor is there a vacuum check, a drive-voltage
    readback, per-chunk current watching, or any record of where the stage has
    already been.

    ``RelocateCoarseXY`` does all of that and stays CONFIRM so it can run
    unattended. Gating the raw primitive is what makes that meaningful: leaving
    both autonomous would just mean the model keeps calling the cheaper one.
    So the rule is not "coarse moves need a human" — it is **the guarded path is
    the autonomous path, and the bare motor command is the one that needs a
    human**.

    Not flagged: Z moves (retract is safe, approach is already gated by
    ``is_coarse_sample_approach``) and the composite's own stepping, which goes
    through ``safe_call`` and never passes a skill name through this predicate."""
    if skill_name != "MotorMove":
        return False
    direction = str((params or {}).get("direction", "")).strip().lower()
    return direction in ("x+", "x-", "y+", "y-")


def is_protection_disable(skill_name: str, params: dict | None) -> bool:
    """True iff this call DISABLES a hardware protection layer.

    Turning OFF SafeTip (EnableSafeTip enable=False) or the Z soft-limits
    (SetZLimitsEnabled enabled=False) removes a hardware safety net. Enabling
    them is always safe (AUTO); DISABLING must not happen autonomously without a
    human in the loop (2026-07-03 review). Value-dependent, like
    is_coarse_sample_approach — the skill's baseline safety_level can't express
    "safe to turn on, gated to turn off"."""
    p = params or {}
    if skill_name == "EnableSafeTip":
        return _is_falsey(p.get("enable"))
    if skill_name == "SetZLimitsEnabled":
        return _is_falsey(p.get("enabled"))
    return False


# ── Operating-mode capability gates (added 2026-07-07) ──────────────────────
# Capability tags declared by skills in SkillMetadata.capabilities. Single
# source of truth for what "electrical pulse" / "tip shaping" mean to the
# global operating-mode gate (see mast.core.types.OperatingMode). These two
# predicates only IDENTIFY a call's tip-processing class; the SAFE/SEMI mode
# DECISION lives in the middlewares (SafetyGate / the pulse-HITL subclass).
CAP_BIAS_PULSE = "bias_pulse"
CAP_TIP_SHAPING = "tip_shaping"


def is_tip_shaping(
    skill_name: str,
    params: dict | None,
    capabilities: frozenset[str] | None = None,
) -> bool:
    """True iff this call mechanically shapes the tip (Z plunge / tip shaper).

    A capability-tag lookup — every tip-shaping skill (TipShape,
    TipShapeWithReadback, ShapeTipOnSurface) declares ``"tip_shaping"`` in its
    metadata. ``skill_name`` / ``params`` are accepted for signature symmetry
    with the other value-dependent gates (and possible future refinement)."""
    return CAP_TIP_SHAPING in (capabilities or frozenset())


def is_electrical_pulse(
    skill_name: str,
    params: dict | None,
    capabilities: frozenset[str] | None = None,
) -> bool:
    """True iff this call delivers an electrical (bias/voltage) pulse to the tip.

    Two shapes, mirroring the two physical mechanisms:

    * A skill tagged ``"bias_pulse"`` — BiasPulse / TipPulse / ConditionTip /
      ConditionTip_DQN. These exist to pulse the junction.
    * A ``"tip_shaping"`` skill that will put a voltage on the junction. The
      TipShaper has **two** bias fields and only ONE of them is conditional —
      vendor wording, same sentence, one with a condition and one without:

        ``Bias (V)`` … is the value applied to the Bias signal **if Change Bias
        is True**.
        ``Bias Lift (V)`` … is the Bias voltage applied **just after the first
        Z ramping**.

      So ``change_bias=False`` does **NOT** mean "no voltage": ``bias_lift_v``
      lands regardless. Until 2026-08-11 this function short-circuited to False
      the moment it saw ``change_bias=False`` and never looked at
      ``bias_lift_v`` — and ``bias_lift_v``'s own ParameterSpec default was
      **3.0 V**. Result: the gate cleared "pure-mechanical" pokes that were in
      fact putting 3 V on the tip, unconditionally and silently. (For a qPlus
      sensor the operator's rule is 20 mV; 3 V is 150× that and rings the fork.)

    Fail-closed on an unparseable OR OMITTED bias (treated as electrical) so a
    malformed — or merely under-specified — tool-call can never sneak a pulse
    past the SAFE/SEMI gate. Omitted is deliberately fail-closed: what the skill
    will resolve an omitted bias to is not visible from the args, and "I didn't
    say" must not read as "zero"."""
    caps = capabilities or frozenset()
    if CAP_BIAS_PULSE in caps:
        return True
    if CAP_TIP_SHAPING in caps:
        p = params or {}
        # (a) the CONDITIONAL half — Bias (V), applied only when Change Bias.
        # change_bias defaults to **False** in TipShape/TipShapeWithReadback
        # since 2026-08-11 (the workflow layer refuses the step-change path),
        # hence default_true=False: a MISSING flag means bias is NOT stepped.
        if not _is_falsey(p.get("change_bias"), default_true=False):
            if _bias_is_live(p.get("bias_v"), "bias_v"):
                return True
        # (b) the UNCONDITIONAL half — Bias Lift (V). No change_bias check here,
        # on purpose: that is exactly the guard that wasn't one.
        return _bias_is_live(p.get("bias_lift_v"), "bias_lift_v")
    return False


def _bias_is_live(value: object, what: str) -> bool:
    """True iff *value* will put a (possibly unknown) non-zero bias on the tip.

    ``None`` → True. An omitted bias is resolved later by the skill (it follows
    the current imaging bias) and the gate cannot see the outcome, so "not
    stated" is treated as "live" rather than as zero. Only an explicit,
    parseable 0 disarms it."""
    if value is None:
        return True
    try:
        return _parse_quantity(value, strict=False, what=what) != 0.0
    except Exception:  # noqa: BLE001
        return True  # unparseable → fail-closed (treat as electrical)


# Half-auto (SEMI) shallow-plunge cap. Mechanical tip shaping is allowed in
# semi-auto but only "shallow": a depth param (unit m, name ~ tip_lift /
# lift_height / depth) whose magnitude exceeds this is blocked with a
# shallow-only hint so the agent retries with a smaller plunge. Deliberately
# generous (5 nm) so the default ShapeTipOnSurface progression (shallow ≈0.5 nm,
# deep ≈3 nm) still runs, while an aggressive tens-of-nm poke is refused. Tune
# here (a future admin-tunable SafetyLimits field can supersede this constant).
SEMI_TIP_LIFT_MAX_M = 5e-9

_SEMI_DEPTH_NAME_PATTERNS = ("tip_lift", "lift_height", "depth")


def semi_tip_depth_violations(
    meta: SkillMetadata,
    params: dict | None,
    cap_m: float = SEMI_TIP_LIFT_MAX_M,
) -> list[str]:
    """Return depth params exceeding the SEMI shallow-plunge cap (empty = OK).

    Only meaningful for tip-shaping calls (the caller gates on is_tip_shaping).
    Scans the skill's own ParameterSpecs for metre-unit depth params and compares
    the supplied magnitude against *cap_m*. A non-numeric / unparseable value is
    skipped here (Layer-1 global bounds and per-skill validation still apply)."""
    out: list[str] = []
    specs = {p.name: p for p in meta.parameters}
    for name, value in (params or {}).items():
        spec = specs.get(name)
        if spec is None or (spec.unit or "").lower() != "m":
            continue
        nl = name.lower()
        if not any(pat in nl for pat in _SEMI_DEPTH_NAME_PATTERNS):
            continue
        try:
            # Raw model args — dimensioned ones are strings now (see
            # physically_absurd_violations). float() alone would fail on "5n"
            # and `continue` past the cap, i.e. stop capping exactly the plunge
            # depths this exists to cap.
            mag = abs(_parse_quantity(value, strict=False, what=name))
        except Exception:  # noqa: BLE001
            continue
        if mag > cap_m:
            out.append(
                f"'{name}' = {value} m exceeds the half-auto shallow-plunge cap "
                f"of {cap_m:g} m"
            )
    return out


# ── The operating-mode decision itself (2026-08-27) ─────────────────────
# Until now the two predicates above only IDENTIFIED a call's tip-processing
# class and every entry point made the SAFE/SEMI DECISION for itself. There were
# two such decisions in the tree (``safety_mw._mode_block`` for the agent's own
# tool calls, ``skill_forge_tools._mode_refusal`` for run_composite) and — the
# reason this function exists — **none at all** in ``ExecutionContext.run``,
# which is the one choke point every composite sub-step, every
# ``POST /api/skills/{name}/execute`` and every conduct step goes through. So a
# template containing TipPulse really did pulse in SAFE as long as it entered by
# one of those doors. See ``mast.core.types.OperatingMode``.
#
# The decision lives HERE, next to the predicates it is made of, and the callers
# only render it: three copies of "which capability tags mean tip processing" is
# exactly the shape this repo keeps paying for (one gets a new tag, the other two
# silently keep clearing it).
#
# ``mode`` is a PARAMETER, not a module-level read: the agent path already
# threads the operator's choice through an explicit ``get_mode`` callable, and
# taking it as an argument keeps this module free of
# :mod:`mast.core.operating_mode` (and therefore pure and testable).


def mode_refusal(
    tool_name: str,
    meta: "SkillMetadata | None",
    args: dict | None,
    mode: "OperatingMode | None",
) -> "str | None":
    """The SAFE/SEMI refusal for this call, or ``None`` to allow it.

    Returns the refusal SENTENCE without any caller prefix — each entry point
    adds its own (``[safety_gate] `` for the middleware, ``[<skill>] `` for the
    forge) so existing transcripts and the tests keyed on those prefixes keep
    reading the same. What must never drift is the leading KEY:
    ``safe_mode_tip_processing_blocked`` / ``semi_mode_shallow_only``.

    Three-way, and the middle one matters:

    * SAFE — refuse BOTH electrical pulses and mechanical shaping.
    * SEMI — refuse only a too-DEEP mechanical plunge. Deliberately NOT "refuse
      any shaping": a SEMI pulse simply runs and is
      announced in the diagnostics ledger, and making one entry point stricter
      than another is how "the same skill answers differently depending on which
      door it came through" gets built.
    * AUTO — allow.

    **Unknown mode allows.** ``None`` means nobody bound a mode source (tests,
    headless ``pipeline.main``, offline tools), and this layer is the "do not
    repair the tip" BEHAVIOUR gate, not a physical protection — the physical
    ones (hard gates, the numeric envelope, SafetyGuard) are unconditional and
    do not route through here. Same direction as
    ``operating_mode.safe_mode_active`` and ``_mode_block``: an unbound process
    behaves exactly as it did before any of this existed.

    A missing ``meta`` also allows: "I could not read the metadata" is answered
    by the caller's own metadata gate (``ExecutionContext.run`` already refuses
    outright there), and answering it a second time here would turn one honest
    error into a misleading mode message.
    """
    if meta is None:
        return None
    if mode is None or mode is OperatingMode.AUTO:
        return None
    caps = getattr(meta, "capabilities", None) or frozenset()
    params = args or {}
    try:
        pulse = is_electrical_pulse(tool_name, params, caps)
        shaping = is_tip_shaping(tool_name, params, caps)
    except Exception:  # noqa: BLE001 — a predicate crash must not wedge execution
        logger.warning("mode gate predicates failed for %s — allowing", tool_name)
        return None

    if mode is OperatingMode.SAFE:
        if pulse or shaping:
            what = "电脉冲" if pulse else "机械修针"
            return (
                f"safe_mode_tip_processing_blocked: 当前是 SAFE 模式，"
                f"'{tool_name}' 含{what}（能力标签 {sorted(caps)}）—— 拒绝执行。"
                "SAFE 的契约是「当前针尖视为良好、专注实验、不修针」，"
                "包一层组合技能、换一个入口都不会改变这一点。"
                "请直接用当前针尖继续实验，或换一个不动针尖的做法；"
                "确实需要修针时请用户切换模式。不要重试针尖处理。"
            )
        return None

    if mode is OperatingMode.SEMI and shaping:
        try:
            viol = semi_tip_depth_violations(meta, params)
        except Exception:  # noqa: BLE001
            logger.warning("mode gate depth check failed for %s — allowing", tool_name)
            return None
        if viol:
            return (
                "semi_mode_shallow_only: 半自动模式仅允许浅层机械修针。"
                + "；".join(viol)
                + "。请减小下压深度后重试。"
            )
    return None


def _valid_check_item(item: object) -> tuple[str, str, str, str] | None:
    """Validate one admin-supplied check entry.

    A corrupt/typo'd admin safety_checks.json (missing keys, or a min_attr /
    max_attr that is not a real SafetyLimits field) must NOT crash SafetyGuard
    construction — that would silently disable the whole skill executor. We
    drop the bad entry (with a logged warning) and keep the rest. Returns a
    normalized (pattern, unit, min_attr, max_attr) tuple, or None if invalid.
    """
    if not isinstance(item, dict):
        logger.warning("Dropping non-dict safety_checks entry: %r", item)
        return None
    try:
        pattern = item["pattern"]
        unit = item["unit"]
        min_attr = item["min_attr"]
        max_attr = item["max_attr"]
    except (KeyError, TypeError):
        logger.warning("Dropping safety_checks entry missing keys: %r", item)
        return None
    if not all(isinstance(x, str) for x in (pattern, unit, min_attr, max_attr)):
        logger.warning("Dropping safety_checks entry with non-str fields: %r", item)
        return None
    # min_attr / max_attr MUST name real SafetyLimits fields, or the eager
    # getattr in SafetyGuard.__init__ would raise AttributeError.
    valid_attrs = set(SafetyLimits.model_fields)
    if min_attr not in valid_attrs or max_attr not in valid_attrs:
        logger.warning(
            "Dropping safety_checks entry with unknown limit attr "
            "(min_attr=%r, max_attr=%r); valid: %s",
            min_attr, max_attr, sorted(valid_attrs),
        )
        return None
    return (pattern, unit, min_attr, max_attr)


def _get_effective_checks() -> list[tuple[str, str, str, str]]:
    """Return _GLOBAL_CHECKS merged with any admin overrides.

    Malformed admin override entries (bad/typo min_attr/max_attr, missing keys)
    are skipped with a warning rather than aborting — see _valid_check_item.
    """
    try:
        from mast.admin.override_store import ConfigOverrideRegistry
        ovr = ConfigOverrideRegistry.get().get_safety_checks()
        if not ovr:
            return list(_GLOBAL_CHECKS)
        checks = list(_GLOBAL_CHECKS)
        # Remove entries
        removals = set(ovr.get("removals", []))
        checks = [c for c in checks if c[0] not in removals]
        # Override existing (only with a validated replacement; a bad override
        # entry is dropped and the built-in check is left untouched).
        for item in ovr.get("overrides", []) or []:
            valid = _valid_check_item(item)
            if valid is None:
                continue
            checks = [valid if c[0] == valid[0] else c for c in checks]
        # Add new (validated)
        for item in ovr.get("additions", []) or []:
            valid = _valid_check_item(item)
            if valid is not None:
                checks.append(valid)
        return checks
    except Exception as exc:  # never let a bad override file disable safety
        logger.warning("Falling back to built-in safety checks: %s", exc)
        return list(_GLOBAL_CHECKS)


#: 由「已登记的仪器事实」推出、只允许**收紧**的那些上限。
#:
#: 每一项是 (SafetyLimits 字段, instrument_profile 键, 方向)。方向恒为 "max"
#: —— 收紧一条上限 = 取更小的那个。**这张表里不会有 "widen"**，那不是疏忽：
#: `skills/builtins/instrument_limits.py` 开头那句 "An agent that can widen its
#: own limits has, in the strict sense, no limits." 同样管着程序自己。
_INSTRUMENT_CLAMPS: tuple[tuple[str, str], ...] = (
    # 设定点上限必须受前放满量程限制。
    # 不可到达的设定点会使反馈持续推进 Z，不能允许配置保留这种无效目标。
    ("setpoint_max_a", "preamp_full_scale_a"),
)


def clamp_to_instrument_facts(
    limits: SafetyLimits,
) -> "tuple[SafetyLimits, tuple[str, ...]]":
    """按已登记的仪器事实收紧包络。**只收紧，绝不放宽。**

    返回 ``(limits, notes)``。``notes`` 是人话的「哪一条被收紧了、依据是什么」，
    空元组表示没有任何一条需要动。

    **这个函数是静默的** —— 不打日志、不写台账。原因很实际：
    ``ExecutionContext._safety_guard()`` 每建一个 context 就新造一个
    ``SafetyGuard``，在这里说话会把日志刷满，而刷满的日志等于没有日志。
    大声那一层是 :func:`report_instrument_clamps`，启动时跑一次。

    **为什么这次自动写，而 `envelope_reconcile` 只报告**：两者的输入不是一回事。
    对账比的是**硬件回读**（``Piezo_RangeGet``），那是一个可能因为模块没开、
    标定漂了、温度变了而不可靠的实时读数，拿它自动改安全线会让「限值是多少」
    随机器状态漂移。而这里用的是**用户在初始化页面上亲手登记的一个常数**
    ——「这台前放是 ±10 nA」。收紧到用户自己声明过的物理量程，不是程序在
    替他做决定，是把他已经说过的话执行到位。

    调用方有两个（活 guard 与报告用的合并值），必须**都**调，否则
    ``safety_view.pending_restart`` 会永远报 True —— 那正是 KNOWN_ISSUES §1.1
    记下的「只对了一半的『已生效』比诚实的『要重启』更危险」。
    """
    import math

    try:
        from mast.core import instrument_profile as _iprof
    except Exception:  # noqa: BLE001 — 缺 profile 绝不能让 guard 建不起来
        return limits, ()

    updates: dict[str, float] = {}
    notes: list[str] = []
    for field, profile_key in _INSTRUMENT_CLAMPS:
        try:
            raw = _iprof.get_config(profile_key)
            if raw is None:
                continue                      # 没登记 —— 没有依据，不动
            fact = float(raw)
            current = float(getattr(limits, field))
        except Exception:  # noqa: BLE001 — 坏值等同于没登记
            continue
        if not math.isfinite(fact) or fact <= 0:
            continue
        if fact >= current:
            continue                          # 事实比配置宽 —— **不放宽**，什么都不做
        # 收紧到「比下限还低」= 造出一个空包络（max ≤ min），那时**每一个**设定点
        # 都被拒。这不是保守，是坏掉 —— 而且它长得就像「安全系统坏了」，最可能的
        # 下一步是有人把整条检查关掉。
        #
        # 这一条不靠编造常数：判据取自包络自己的下限。空区间从来不是合法状态。
        # 触发它的真实路径是数据录入错误 —— `instrument_profile.sanitize()` 会把值
        # **夹进** spec 声明的范围 (1e-12, 1e-2)，所以误填的 0 或负数不会被拒绝，
        # 而是变成 1e-12（1 pA）：一个合法正数，一个荒唐的前放量程。
        floor_attr = field.replace("_max_", "_min_")
        floor = getattr(limits, floor_attr, None)
        if isinstance(floor, (int, float)) and fact <= float(floor):
            logger.warning(
                "仪器事实收紧被跳过：instrument_profile.%s = %.6g 不高于 %s = %.6g，"
                "照做会造出空包络。请核对这个登记值。",
                profile_key, fact, floor_attr, float(floor),
            )
            continue
        updates[field] = fact
        notes.append(
            f"{field}: {current:.6g} → {fact:.6g}"
            f"（依据 instrument_profile.{profile_key}，只收紧）"
        )
    if not updates:
        return limits, ()
    return limits.model_copy(update=updates), tuple(notes)


def report_instrument_clamps() -> "tuple[str, ...]":
    """启动时记录因仪器能力而发生的有效限值收紧，并返回说明。
    
    与对账流程共用日志和诊断台账；auto_applied=True 表示实际已经调整。
    比较合并覆写后的生效值，不与出厂默认值比较；只有确实发生变动时才报告。"""
    notes = clamp_to_instrument_facts(_merge_overrides(SafetyLimits()))[1]
    if not notes:
        return ()
    logger.warning(
        "安全包络已按已登记的仪器事实**收紧**（只收紧，不放宽）："
    )
    for n in notes:
        logger.warning("  %s", n)
    logger.warning(
        "  依据是用户在仪器档案里登记的常数，不是硬件回读。要改这条线，"
        "改档案里的那个数，别改包络。"
    )
    try:
        from mast.core.diagnostics import record as diag_record

        diag_record(
            "note", subject="safety_envelope_instrument_clamp",
            reason="；".join(notes),
            clamps=list(notes),
            # 对账那条是 False（只报告）。这条是 True —— 包络真的变了。
            auto_applied=True,
            direction="tighten_only",
        )
    except Exception:  # noqa: BLE001
        logger.debug("instrument clamp: diagnostics record failed", exc_info=True)
    return notes


def _merge_overrides(limits: SafetyLimits) -> SafetyLimits:
    """SafetyLimits + 磁盘上的管理员覆写。**只合并，不收紧。**

    单独抽出来是因为有两个调用方需要「合并后但未收紧」这个中间态：
    :func:`_get_effective_limits`（下一步就收紧）与 :func:`report_instrument_clamps`
    （要拿它当**基准**，才能说出「收紧到底改没改生效值」）。

    A missing admin module or a malformed override file must NOT crash
    SafetyGuard construction (that would silently disable the executor), so we
    fall back to the built-in limits. But the failure is logged at WARNING —
    the old ``except (ImportError, Exception): pass`` swallowed every override
    error silently, hiding a broken admin config (/ #114).
    """
    try:
        from mast.admin.override_store import ConfigOverrideRegistry
        ovr = ConfigOverrideRegistry.get().get_safety_limits()
        if ovr:
            limits = limits.model_copy(update=ovr)
    except Exception as exc:  # never let a bad override file disable safety
        logger.warning("Falling back to built-in safety limits: %s", exc)
    return limits


def _get_effective_limits(limits: SafetyLimits) -> SafetyLimits:
    """Return SafetyLimits merged with any admin overrides, then clamped."""
    limits = _merge_overrides(limits)
    # 仪器事实收紧排在覆写**之后** —— 顺序是判据的一部分：管理员覆写可以把设定点
    # 上限调到任何值，但调不到前放量程之上。反过来（先收紧再合并覆写）等于给
    # 「用覆写放宽到物理上达不到的设定点」留了一条路，而那条路的终点是撞针。
    limits, _notes = clamp_to_instrument_facts(limits)
    return limits


class SafetyGuard:
    """Three-layer safety: parameter bounds, state guards, approval levels."""

    def __init__(self, limits: SafetyLimits):
        # The UNMERGED limits, kept so ``reload`` can re-derive from code
        # defaults + the CURRENT override file. Re-merging on top of the already
        # merged ``_limits`` would be wrong in the dangerous direction: removing
        # a field from the override file would leave the old (possibly widened)
        # value stuck in place forever.
        self._raw_limits = limits
        self._limits: SafetyLimits
        self._checks: list[tuple[str, str, str, str]]
        self._resolved_checks: list[tuple[str, str, float, float]]
        self._apply_overrides()

    def _apply_overrides(self) -> None:
        """(Re)derive effective limits + resolved checks from raw + overrides.

        Everything is built into LOCALS first and published by three plain
        attribute assignments at the end. A concurrent
        ``check_parameter_bounds`` therefore sees either the whole old envelope
        or the whole new one — never a half-filled ``_resolved_checks``. (Name
        binding is atomic in CPython; that is the whole mechanism, and it is
        why no lock is taken on the hot path.)
        """
        limits = _get_effective_limits(self._raw_limits)
        checks = _get_effective_checks()
        # Pre-resolve limit values for each check to avoid repeated getattr.
        # Defence-in-depth: a single check whose attr is missing on the limits
        # object is dropped (with a warning) rather than aborting construction
        # of the entire SafetyGuard (which would silently disable the executor).
        resolved: list[tuple[str, str, float, float]] = []
        for pat, unit_pat, min_attr, max_attr in checks:
            try:
                gmin = getattr(limits, min_attr)
                gmax = getattr(limits, max_attr)
            except AttributeError:
                logger.warning(
                    "Skipping safety check %r: limit attr missing "
                    "(min_attr=%r, max_attr=%r)", pat, min_attr, max_attr,
                )
                continue
            resolved.append((pat, unit_pat, gmin, gmax))
        self._limits = limits
        self._checks = checks
        self._resolved_checks = resolved

    def reload(self) -> None:
        """Re-read the admin overrides into this live guard.

        Subscribed to ``ConfigOverrideRegistry.signal_reload`` by
        ``mast.admin.reload_wiring`` — before that, a saved override reached
        NOTHING until the process restarted, while the write endpoint reported
        ``reloaded=True`` (KNOWN_ISSUES §1.1).

        This covers the MANUAL execution path only (SkillExecutor). The agent
        path bakes the envelope into each tool's pydantic schema at graph-build
        time, so it needs a graph rebuild, not a re-merge — that half lives in
        ``reload_wiring`` too, and the honest status it produces is what
        ``restart_required`` reports.
        """
        self._apply_overrides()

    def check_parameter_bounds(self, skill_meta: SkillMetadata, params: dict) -> list[str]:
        """Check all parameters against their min/max bounds AND global safety limits.
        Returns list of violation messages (empty = safe)."""
        # #118 deepening: a physically-impossible magnitude (e.g. a 1.5 A tunnelling
        # setpoint) is reported FIRST, on its own — it is a unit/exponent slip, not
        # an envelope violation, and the crispest fix is a magnitude correction.
        absurd = physically_absurd_violations(skill_meta, params)
        if absurd:
            return absurd

        violations: list[str] = []

        param_specs = {p.name: p for p in skill_meta.parameters}

        for name, value in params.items():
            spec = param_specs.get(name)
            if spec is None:
                continue

            # allowed_values must be validated for NON-numeric params too. The
            # old code `continue`d on any non-(int/float) value BEFORE reaching
            # the allowed_values check, so every enum STRING param (MotorMove
            # direction, axis, …) skipped validation entirely — a typo'd
            # direction then fell through to a silent wrong-axis default
            # (motor.py dir_map.get(direction, 0) → X+). Check the enum here,
            # before the numeric-only bounds below (2026-07-03 review).
            if not isinstance(value, (int, float)):
                if (spec.allowed_values is not None
                        and value not in spec.allowed_values):
                    violations.append(
                        f"Parameter '{name}' = {value!r} not in allowed values "
                        f"{spec.allowed_values}"
                    )
                    continue
                # A dimensioned quantity written as text ("3p", "5e-8") is the
                # normal form on the agent path since 2026-08-04. Falling
                # through to `continue` here would skip its bounds entirely —
                # the same class of hole the enum check above was added to fix.
                if spec.allowed_values is None and (spec.unit or ""):
                    try:
                        value = _parse_quantity(value, strict=False, what=name)
                    except Exception:  # noqa: BLE001 — genuinely non-numeric
                        continue
                else:
                    continue

            # Check spec-level bounds (with _FLOAT_SLACK: a parsed '100n' must clear 1e-07)
            if spec.min_value is not None and value < spec.min_value - _FLOAT_SLACK * abs(spec.min_value):
                violations.append(
                    f"Parameter '{name}' = {value} below minimum {spec.min_value}"
                )
            if spec.max_value is not None and value > spec.max_value + _FLOAT_SLACK * abs(spec.max_value):
                violations.append(
                    f"Parameter '{name}' = {value} above maximum {spec.max_value}"
                )

            # Check allowed_values
            if spec.allowed_values is not None and value not in spec.allowed_values:
                violations.append(
                    f"Parameter '{name}' = {value} not in allowed values {spec.allowed_values}"
                )

            # Check global safety limits (pre-resolved values, no getattr per iteration)
            name_lower = name.lower()
            unit_lower = spec.unit.lower() if spec.unit else ""
            for pattern, unit_pat, global_min, global_max in self._resolved_checks:
                if pattern in name_lower and unit_pat in unit_lower:
                    if value < global_min - _FLOAT_SLACK * abs(global_min):
                        violations.append(
                            f"Parameter '{name}' = {value} violates global safety minimum {global_min}"
                        )
                    if value > global_max + _FLOAT_SLACK * abs(global_max):
                        violations.append(
                            f"Parameter '{name}' = {value} violates global safety maximum {global_max}"
                        )
                    break

        # Check required parameters are present
        for spec in skill_meta.parameters:
            if spec.required and spec.name not in params and spec.default is None:
                violations.append(f"Required parameter '{spec.name}' is missing")

        return violations

    def check_state_preconditions(
        self, skill_meta: SkillMetadata, state: HardwareState
    ) -> list[str]:
        """Check hardware state preconditions.
        Returns list of violation messages.

        Thin wrapper over the module-level :func:`check_state_preconditions` so
        the manual path and the agent path (``SafetyGate``) share ONE parser.
        They used to be two copies, and they had already drifted: the agent copy
        never grew the ``withdrawn`` branch, so a coarse motor move requested by
        the agent skipped the "is the tip actually clear" check that the manual
        path applied (found 2026-07-31). ``_GLOBAL_CHECKS`` was unified for the
        same reason after the same kind of drift."""
        return check_state_preconditions(skill_meta, state)

    def requires_approval(self, skill_meta: SkillMetadata) -> str:
        """Returns approval requirement: 'none', 'llm', 'human'."""
        if skill_meta.safety_level == SafetyLevel.AUTO:
            return "none"
        elif skill_meta.safety_level == SafetyLevel.CONFIRM:
            return "llm"
        elif skill_meta.safety_level == SafetyLevel.DANGEROUS:
            return "human"
        return "human"

    def validate_execution(
        self, skill_meta: SkillMetadata, params: dict, state: HardwareState
    ) -> tuple[bool, list[str]]:
        """Combined validation. Returns (is_safe, list_of_issues)."""
        issues: list[str] = []
        issues.extend(self.check_parameter_bounds(skill_meta, params))
        issues.extend(self.check_state_preconditions(skill_meta, state))
        return (len(issues) == 0, issues)


def check_state_preconditions(
    skill_meta: SkillMetadata, state: HardwareState
) -> list[str]:
    """Hardware-state preconditions -> violation messages (empty = all met).

    Delegates to :mod:`mast.core.preconditions`, which has been the DECLARED
    single source of the precondition vocabulary since -- but was not
    the only one. Three hand-maintained chains existed and all three had drifted:

      * this one, the only one that knew ``withdrawn``;
      * ``SafetyGate``'s copy, which never grew it, so that check silently passed
        for everything the AGENT ran -- the path with nobody watching;
      * ``core.preconditions`` itself, which ``BaseSkill.check_preconditions``
        reads, so a precondition living ONLY here came back as
        "Cannot verify precondition" at the wrap_skill layer and FAILED the call.

    That last one is the sharp edge, and it is not symmetrical with the others: a
    rule added here alone does not merely go unenforced elsewhere, it breaks every
    skill that declares it. Both missing rules (``withdrawn`` and the computed
    ``vacuum_ok_for_coarse``) now live in ``preconditions``; there is one chain
    (2026-07-31)."""
    return _shared_check_state_preconditions(
        list(skill_meta.preconditions), state)
