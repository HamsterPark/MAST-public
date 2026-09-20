"""Coarse-stepper drive amplitude / frequency — the operator's parameter, not the agent's.

WHY THIS IS ITS OWN MODULE
==========================
A pan-type coarse stepper is driven by a sawtooth of a few hundred volts. The
controller's ceiling and the STACK's ceiling are different numbers, and only the
second one matters:

    「有些 Nanonis 控制器支持 400V，但是有时候 300V 就烧坏了」

There is no reading, no status bit and no error that tells you which rig you are
on. The number lives in one place — the head of the person who built the rig —
so MAST's job is to hold that number, compare against it, and never guess.

``SetMotorFreqAmp`` previously took ``amplitude_v`` straight from its caller,
bounded only by a hardcoded 200 V ceiling written for "a generic piezo drive".
On the autonomous path that is no bound at all: CONFIRM is approved by the model
itself. Nothing about a language model's training makes it competent to pick a
drive voltage for a specific piezo stack, and the failure is not recoverable by
software — the stack is dead.

THE FOUR LOCKS
==============
1. **The agent cannot see the skill.** ``SetMotorFreqAmp`` sits behind an
   ``advanced_capabilities`` gate that is OFF by default; while it is off the
   tool is not wrapped at all, so it is absent from the tool list. Turning it on
   is an admin-PIN write (``admin_pin.GUARDED_KEYS``), fail-closed: with no PIN
   set, the key cannot be written at all.
2. **The agent cannot call it anyway.** ``safety.is_coarse_drive_change`` blocks
   it on the autonomous path and forces human approval on the manual path — the
   same Layer-0 shape as ``is_calibration_change`` / ``is_protection_disable``.
   Two independent locks because lock 1 is a switch a human could flip.
3. **The rig ceiling.** :func:`authorize` refuses anything above the declared
   ``max_amplitude_v`` — and refuses EVERYTHING when nothing has been declared.
   Silence is not consent: not knowing what this stack tolerates is a reason to
   do nothing, not a reason to fall back to a number someone picked for a
   different instrument.
4. **An absolute ceiling** no configuration can widen, mirroring
   ``safety._PHYSICAL_ABSURD``: 400 V is the most any coarse-motion controller
   in this class outputs, so a larger request is an error of kind, not of degree.

WHY NOT CLAMP
=============
An out-of-range request is REFUSED, never silently reduced. Clamping turns
"300 V would destroy this stack" into "ran at 300 V" while the caller believes it
asked for 400 — a wrong action reported as a success. The whole point of holding
a per-rig ceiling is to make the mismatch visible.

WHY THE READBACK CHECK EXISTS
=============================
The declared ceiling constrains what MAST writes. It says nothing about what the
drive is ACTUALLY set to — somebody may have changed it in the Nanonis UI, or a
previous session may have left it high. So every coarse move reads the drive back
(``Motor_FreqAmpGet``) and compares before it steps. :func:`readback_matches` is
that comparison, and **an unreadable drive is a refusal**, not a pass: the whole
value of the check is that it fails when we cannot see.

Storage: the ``coarse_drive`` settings key (a single top-level dict, like
``instrument_profile``), hydrated at startup and persisted through an injected
sink. Dependency-light: stdlib only, so both the skill layer and the safety layer
can import it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Settings key holding the declaration. Registered in
#: ``SettingsStore.KNOWN_KEYS`` AND ``admin_pin.GUARDED_KEYS``.
SETTINGS_KEY = "coarse_drive"

#: Hard ceilings no declaration may exceed. Same character as
#: ``safety._PHYSICAL_ABSURD``: rig-independent, not admin-widenable, and a
#: violation means "wrong kind of number" rather than "turn the limit up".
ABSOLUTE_MAX_AMPLITUDE_V = 400.0
ABSOLUTE_MAX_FREQUENCY_HZ = 20_000.0

#: Relative tolerance for the pre-move readback comparison. Drive electronics
#: quantise the amplitude DAC, so an exact match is not a reasonable demand; 2 %
#: is far tighter than any difference that would matter physically.
READBACK_REL_TOL = 0.02

#: Fields of the declaration. Unknown keys are dropped, exactly like
#: ``instrument_profile.sanitize``.
_SPEC: dict[str, tuple[str, str, type, tuple[float, float], Any]] = {
    "max_amplitude_v": (
        "本机粗动驱动幅度上限(用户声明)", "V", float,
        (0.0, ABSOLUTE_MAX_AMPLITUDE_V), None),
    "expected_frequency_hz": (
        "本机粗动驱动频率(用于移动前读回核对;留空=不核对频率)", "Hz", float,
        (0.0, ABSOLUTE_MAX_FREQUENCY_HZ), None),
}
_TEXT_KEYS: tuple[str, ...] = ("declared_by", "notes")
_STAMP_KEYS: tuple[str, ...] = ("declared_at",)

ALL_KEYS: tuple[str, ...] = tuple(_SPEC) + _TEXT_KEYS + _STAMP_KEYS

_lock = threading.RLock()
_decl: dict[str, Any] = {}
_persist_sink: Callable[[dict], None] | None = None


# ── holder ───────────────────────────────────────────────────────────────────

def sanitize(raw: Any) -> dict[str, Any]:
    """Clean a raw declaration dict. Never raises; returns {} for a non-dict.

    Out-of-range numbers are DROPPED, not clamped — a declaration is a claim
    about hardware, and a claim we had to correct is not a claim we should act
    on."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, (_label, _unit, kind, (lo, hi), _default) in _SPEC.items():
        if key not in raw or raw[key] is None or raw[key] == "":
            continue
        try:
            val = kind(raw[key])
        except (TypeError, ValueError):
            continue
        if val != val or val in (float("inf"), float("-inf")):
            continue
        if not (lo <= val <= hi):
            logger.warning("coarse_drive: 丢弃越界声明 %s=%r (允许 %s..%s)",
                           key, raw[key], lo, hi)
            continue
        out[key] = val
    for key in _TEXT_KEYS:
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            out[key] = v.strip()[:200]
    for key in _STAMP_KEYS:
        try:
            v = float(raw[key])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            continue
        if v == v:
            out[key] = v
    return out


def set_declaration(raw: Any) -> dict[str, Any]:
    """Install the declaration (from settings hydration or an admin write)."""
    global _decl
    clean = sanitize(raw)
    if "max_amplitude_v" in clean and "declared_at" not in clean:
        clean["declared_at"] = time.time()
    with _lock:
        _decl = clean
    if clean.get("max_amplitude_v") is not None:
        # Loud on purpose: this number is the only thing standing between a
        # language model's arithmetic and a piezo stack.
        logger.warning("coarse_drive: 本机粗动幅度上限已声明为 %.1f V",
                       float(clean["max_amplitude_v"]))
    else:
        logger.info("coarse_drive: 未声明本机粗动幅度上限 —— 一切写入将被拒绝")
    return dict(clean)


def get_declaration() -> dict[str, Any]:
    with _lock:
        return dict(_decl)


def set_persist_sink(sink: Callable[[dict], None] | None) -> None:
    global _persist_sink
    _persist_sink = sink


def _persist() -> None:
    sink = _persist_sink
    if callable(sink):
        try:
            sink(get_declaration())
        except Exception as exc:  # noqa: BLE001 — persistence must not break a write
            logger.debug("coarse_drive persist failed: %s", exc)


def declare(max_amplitude_v: float | None,
            expected_frequency_hz: float | None = None,
            *, declared_by: str = "", notes: str = "") -> dict[str, Any]:
    """Operator-facing writer (admin route). Returns the stored declaration."""
    payload: dict[str, Any] = {"declared_at": time.time()}
    if max_amplitude_v is not None:
        payload["max_amplitude_v"] = max_amplitude_v
    if expected_frequency_hz is not None:
        payload["expected_frequency_hz"] = expected_frequency_hz
    if declared_by:
        payload["declared_by"] = declared_by
    if notes:
        payload["notes"] = notes
    out = set_declaration(payload)
    _persist()
    return out


def max_amplitude_v() -> float | None:
    v = get_declaration().get("max_amplitude_v")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def expected_frequency_hz() -> float | None:
    v = get_declaration().get("expected_frequency_hz")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def is_declared() -> bool:
    return max_amplitude_v() is not None


# ── the gate ─────────────────────────────────────────────────────────────────

_UNDECLARED = (
    "本机粗动驱动幅度上限【尚未声明】,拒绝设置驱动电压。\n"
    "控制器支持的电压和这台机器的压电叠堆能承受的电压是两个数 —— 有些控制器支持 400 V,"
    "但有的叠堆 300 V 就烧了,而且没有任何读数会告诉你是哪一种。\n"
    "请用户在【高级】页填写本机上限(需要 admin PIN)。在那之前 MAST 不会替这台机器猜。"
)


def authorize(amplitude_v: Any, frequency_hz: Any = None,
              axis: str = "all") -> tuple[bool, str]:
    """May we write this drive setting? ``(ok, reason)``; ``reason`` is Chinese.

    Refuses — never clamps. A silently-reduced amplitude is a wrong action
    reported as a success, which is worse than a refusal the caller can see."""
    try:
        amp = float(amplitude_v)
    except (TypeError, ValueError):
        return False, f"驱动幅度不是一个数值:{amplitude_v!r}"
    if amp != amp or amp in (float("inf"), float("-inf")):
        return False, "驱动幅度不是一个有限数值"
    if amp < 0:
        return False, "驱动幅度不能为负"

    # Lock 4 — absolute, before anything configurable, so no declaration can
    # widen it and a fat-fingered ceiling cannot authorise an absurd request.
    if amp > ABSOLUTE_MAX_AMPLITUDE_V:
        return False, (
            f"驱动幅度 {amp:g} V 超过绝对上限 {ABSOLUTE_MAX_AMPLITUDE_V:g} V —— "
            "这已经不是「上限设低了」,而是数量级/单位错了。任何配置都不能放宽这一条。")

    if frequency_hz is not None:
        try:
            freq = float(frequency_hz)
        except (TypeError, ValueError):
            return False, f"驱动频率不是一个数值:{frequency_hz!r}"
        if freq != freq or freq < 0 or freq > ABSOLUTE_MAX_FREQUENCY_HZ:
            return False, (
                f"驱动频率 {frequency_hz!r} 超出 0..{ABSOLUTE_MAX_FREQUENCY_HZ:g} Hz")

    # Lock 3 — the rig ceiling.
    ceiling = max_amplitude_v()
    if ceiling is None:
        return False, _UNDECLARED
    if amp > ceiling:
        return False, (
            f"驱动幅度 {amp:g} V 超过本机声明上限 {ceiling:g} V —— 拒绝执行。\n"
            "【不会自动降到上限】:那样调用方会以为自己设的是 "
            f"{amp:g} V,而实际跑的是 {ceiling:g} V。要用更高的电压,"
            "必须由用户先修改本机声明。")
    return True, (f"{amp:g} V ≤ 本机声明上限 {ceiling:g} V（axis={axis}）")


def readback_matches(read_amplitude_v: Any,
                     read_frequency_hz: Any = None) -> tuple[bool, str]:
    """Is the drive CURRENTLY within the declared envelope? ``(ok, reason)``.

    Called before every coarse move. The declaration constrains what MAST writes;
    it says nothing about what the hardware is set to right now — the Nanonis UI
    is right there, and a previous session may have left the drive high.

    **Unreadable is a refusal.** A check whose failure mode is "pass" is not a
    check, and this one guards an irreversible outcome."""
    ceiling = max_amplitude_v()
    if ceiling is None:
        return False, _UNDECLARED
    try:
        amp = float(read_amplitude_v)
    except (TypeError, ValueError):
        return False, (
            "读不到当前粗动驱动幅度,拒绝粗动。"
            "读不到 ≠ 没问题 —— 电压可能是别人在 Nanonis 界面里改的。")
    if amp != amp or amp in (float("inf"), float("-inf")):
        return False, "粗动驱动幅度读数不是有限数值,拒绝粗动"
    if amp > ceiling * (1.0 + READBACK_REL_TOL):
        return False, (
            f"当前粗动驱动幅度 {amp:g} V 高于本机声明上限 {ceiling:g} V —— 拒绝粗动。"
            "请在 Nanonis 里把它调回来,或由用户更新本机声明。")

    expected = expected_frequency_hz()
    if expected is not None and read_frequency_hz is not None:
        try:
            freq = float(read_frequency_hz)
        except (TypeError, ValueError):
            freq = float("nan")
        if freq != freq:
            return False, "读不到当前粗动驱动频率,而本机声明了期望频率,拒绝粗动"
        if abs(freq - expected) > max(expected * 0.10, 1.0):
            # A warning, not a refusal: a wrong frequency changes how far a step
            # goes (bad for the odometer) but does not destroy the stack.
            return True, (
                f"幅度 {amp:g} V 合规,但驱动频率 {freq:g} Hz 与声明的 "
                f"{expected:g} Hz 不符 —— 每步走的距离会变,里程表会漂。")
    return True, f"当前驱动幅度 {amp:g} V ≤ 本机声明上限 {ceiling:g} V"


def format_block() -> str:
    """One short block for the system prompt. Never empty, never a number to act on.

    Deliberately does NOT invite the model to propose a value: the point of the
    block is that this is somebody else's parameter."""
    ceiling = max_amplitude_v()
    if ceiling is None:
        return ("【粗动驱动电压】本机上限未声明 —— 任何设置驱动电压的尝试都会被拒绝,"
                "粗动移动也会被拒绝。这是用户在【高级】页填写的参数,不要尝试自己设定。")
    freq = expected_frequency_hz()
    tail = f"、期望频率 {freq:g} Hz" if freq is not None else ""
    return (f"【粗动驱动电压】本机声明上限 {ceiling:g} V{tail}(用户设定,只读)。"
            "这是唯一能保护压电叠堆的数字,**不要尝试修改它,也不要尝试设置驱动电压** —— "
            "需要改就交给用户。")


__all__ = [
    "SETTINGS_KEY",
    "ABSOLUTE_MAX_AMPLITUDE_V",
    "ABSOLUTE_MAX_FREQUENCY_HZ",
    "READBACK_REL_TOL",
    "ALL_KEYS",
    "sanitize",
    "set_declaration",
    "get_declaration",
    "set_persist_sink",
    "declare",
    "max_amplitude_v",
    "expected_frequency_hz",
    "is_declared",
    "authorize",
    "readback_matches",
    "format_block",
]
