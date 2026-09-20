"""qPlus oscillation amplitude as an INDEPENDENT tip-crash indicator.

With a qPlus sensor present, the oscillation amplitude can serve as a
crash indicator independent of the tunnelling current: if the tip crashes
during approach, the amplitude drops to zero and only recovers once the
tip is withdrawn.

Why this is worth its own skill
-------------------------------
Every crash check MAST had reads the same physical quantity family — the
tunnelling current, or the variance of a scan built from it. A qPlus sensor's
oscillation amplitude is a *mechanically* independent witness: a tip in contact
is damped to a standstill regardless of what the current preamp reports. On a
rig that has one, it can catch a crash the current-based checks miss, and it
answers a question the current cannot — "is the tip still free to move?".

Design constraints, all of them safety-driven
---------------------------------------------
* **Additive, never subtractive.** Nothing here suppresses or overrides
  :class:`CheckScanForCrash` or the approach-time current checks. It adds a
  verdict; the existing ones keep theirs.
* **Never say "no crash" when the reading failed.** ``status`` is one of
  ``ok`` / ``crash`` / ``unavailable`` / ``no_baseline`` — the last two are not
  "fine", they are "cannot tell", and callers must treat them that way. A crash
  detector whose failure mode looks like success is worse than no detector.
* **Zero amplitude is only meaningful against a baseline.** An amplitude of
  0 V means nothing on its own: the oscillator may simply not be running, or
  the signal may be in different units on the instrument. So the crash verdict needs
  a baseline captured while the tip was known-free, and without one this
  reports ``no_baseline`` rather than guessing.
* **Absent hardware is not a fault.** A rig with no qPlus has no such signal;
  that is ``unavailable``, reported calmly, and it must not make an STM-only
  session look broken.

The amplitude is read through the generic signal table (``Signals_NamesGet`` /
``Signals_ValGet``) — verified against the installed ``nanonis_spm``: there is
no ``PLL_AmpGet``; ``PLL_AmpCtrlSetpntGet`` is the amplitude *setpoint*, not the
measurement, and using it would have read back a constant that never falls on a
crash.
"""
from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: Substrings that identify an oscillation-amplitude channel in the Nanonis
#: signal list. The operator's rig calls it "OCD1 Amplitude"; other setups name
#: the Oscillation Control module differently, so match on the amplitude word
#: plus an oscillation-control hint rather than on one exact string.
_AMP_HINTS: tuple[str, ...] = ("amplitude", "amp")
_OSC_HINTS: tuple[str, ...] = ("oc ", "ocd", "osc", "pll", "excitation")

#: Below this FRACTION of the free-oscillation baseline the tip is judged to be
#: in contact. A crashed qPlus does not droop — it stops — so the gap between
#: "free" and "crashed" is large and the threshold does not need to be tuned
#: per rig. Deliberately not 0: thermal drift and preamp offset leave a small
#: residual, and requiring exact zero would miss real crashes.
_CRASH_FRACTION = 0.10


def _signal_names(record) -> list[str]:
    """Channel names out of a ``Signals_NamesGet`` reply ([] when unreadable)."""
    rv = getattr(record, "return_value", None)
    if not (isinstance(rv, (list, tuple)) and len(rv) > 2):
        return []
    body = rv[2]
    if not isinstance(body, (list, tuple)):
        return []
    out: list[str] = []
    for el in body:
        if isinstance(el, str):
            out.append(el)
        elif isinstance(el, (list, tuple)):
            out.extend(str(x) for x in el if isinstance(x, str))
    return out


def find_amplitude_signal(context) -> tuple[int, str] | None:
    """``(index, name)`` of the oscillation-amplitude channel, or None.

    None means "this rig does not expose one" — an STM without a qPlus sensor,
    which is a normal configuration and not an error.

    Checks the remembered index first. That key
    (``qplus_amplitude_signal_index``) was written by ``_set_baseline`` and read
    by nobody — the only reader in the tree was a test — so every call re-fetched
    the whole signal table to re-derive an answer already on file. Honouring it
    also gives the operator a way to OVERRIDE the name match on a rig whose
    channel is called something the hints do not cover, which was the stated
    purpose of exposing the key at all.
    """
    remembered = _remembered_index()
    if remembered is not None:
        return remembered, ""
    rec = context.safe_call("Signals_NamesGet")
    if getattr(rec, "error", ""):
        logger.debug("qplus: Signals_NamesGet failed: %s", rec.error)
        return None
    for i, name in enumerate(_signal_names(rec)):
        low = " ".join(str(name).lower().split())
        if any(a in low for a in _AMP_HINTS) and any(o in low for o in _OSC_HINTS):
            return i, str(name)
    return None


class ReadTipOscillationAmplitude(BaseSkill):
    """Read the qPlus oscillation amplitude (and remember it as a baseline)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ReadTipOscillationAmplitude",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取 qPlus 振荡振幅（OCD1 Amplitude 等通道）。可作为**独立于电流**的"
                "针尖状态判据：针尖接触表面后振幅会被阻尼到接近 0，退针后才恢复。"
                "没有 qPlus 的机器上返回 status='unavailable'（这不是故障）。"
            ),
            parameters=[
                ParameterSpec(
                    name="signal_index",
                    type="int",
                    description=(
                        "显式指定振幅信号索引；-1 = 按通道名自动查找。"
                        "自动查找不到时返回 unavailable，不会猜一个索引。"
                    ),
                    required=False,
                    default=-1,
                ),
                ParameterSpec(
                    name="set_baseline",
                    type="bool",
                    description=(
                        "True = 把这次读数记为「自由振荡基线」。**只应在确认针尖未接触"
                        "表面时调用**（进针前、或退针之后）。基线是撞针判据的分母。"
                    ),
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=0.3,
            tags=["qplus", "afm", "read", "tip"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params.get("signal_index", -1))
        name = ""
        if idx < 0:
            found = find_amplitude_signal(context)
            if found is None:
                return SkillResult(
                    skill_name="ReadTipOscillationAmplitude", success=True,
                    data={"status": "unavailable", "amplitude": None,
                          "note": ("信号表里没有振荡振幅通道 —— 这台机器很可能没有 "
                                   "qPlus 传感器。这不是故障，只是这条判据用不上。")})
            idx, name = found
        rec = context.safe_call("Signals_ValGet", int(idx), 1)
        if getattr(rec, "error", ""):
            return SkillResult(
                skill_name="ReadTipOscillationAmplitude", success=False,
                error=f"读取振幅失败（signal {idx}）：{rec.error}",
                nanonis_calls=[rec])
        amp = _scalar(rec)
        if amp is None:
            return SkillResult(
                skill_name="ReadTipOscillationAmplitude", success=False,
                error=f"振幅读数无法解析（signal {idx}）",
                nanonis_calls=[rec])
        data = {"status": "ok", "amplitude": amp, "signal_index": int(idx),
                "signal_name": name}
        if params.get("set_baseline"):
            _set_baseline(context, amp, idx, name)
            data["baseline_set"] = True
        return SkillResult(skill_name="ReadTipOscillationAmplitude",
                           success=True, data=data, nanonis_calls=[rec])


class CheckTipCrashByAmplitude(BaseSkill):
    """Judge tip contact from the oscillation amplitude against its baseline.

    The skill itself SUCCEEDS whenever the analysis ran; the verdict is in
    ``data["status"]``. Callers must branch on that, and must treat
    ``unavailable`` / ``no_baseline`` as "cannot tell" — never as "no crash".
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CheckTipCrashByAmplitude",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "用 qPlus 振幅判断针尖是否已接触表面（撞针）。判据：振幅低于自由振荡"
                "基线的 10%。**这是对电流类判据的补充，不替代它们。**"
                "status ∈ ok|crash|unavailable|no_baseline —— 后两者是「判断不了」，"
                "不是「没撞」。"
            ),
            parameters=[
                ParameterSpec(
                    name="signal_index", type="int", required=False, default=-1,
                    description="显式振幅信号索引；-1 = 自动查找。"),
            ],
            estimated_duration_s=0.3,
            tags=["qplus", "afm", "crash", "tip", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # 未驱动音叉时，振幅读数不能与自由振荡基线比较来判定 crash。
        # 激励关闭返回 unavailable；激励状态不可读也返回 unavailable。
        # 未知既不是正常也不是碰撞，需要先获得有效的驱动状态。
        exc_on, exc_note = _excitation_state(context)
        if exc_on is not True:
            return SkillResult(
                skill_name="CheckTipCrashByAmplitude", success=True,
                data={"status": "unavailable", "crash_indicator": None,
                      "excitation_on": exc_on, "note": exc_note})
        read = ReadTipOscillationAmplitude().execute(
            context, {"signal_index": params.get("signal_index", -1)})
        if not read.success:
            # A failed READ is not a verdict. Report it as such.
            return SkillResult(
                skill_name="CheckTipCrashByAmplitude", success=True,
                data={"status": "unavailable", "crash_indicator": None,
                      "note": f"振幅读取失败，无法判断：{read.error}"},
                nanonis_calls=list(read.nanonis_calls))
        if read.data.get("status") != "ok":
            return SkillResult(
                skill_name="CheckTipCrashByAmplitude", success=True,
                data={"status": "unavailable", "crash_indicator": None,
                      "note": read.data.get("note", "没有可用的振幅通道")},
                nanonis_calls=list(read.nanonis_calls))

        amp = float(read.data["amplitude"])
        base = _get_baseline(context)
        if base is None or base <= 0:
            return SkillResult(
                skill_name="CheckTipCrashByAmplitude", success=True,
                data={"status": "no_baseline", "crash_indicator": None,
                      "amplitude": amp,
                      "note": ("没有自由振荡基线，无法判断振幅是否塌了。"
                               "请在针尖确认未接触时先调用 "
                               "ReadTipOscillationAmplitude(set_baseline=True)。")},
                nanonis_calls=list(read.nanonis_calls))

        frac = amp / base
        crashed = frac < _CRASH_FRACTION
        return SkillResult(
            skill_name="CheckTipCrashByAmplitude", success=True,
            data={"status": "crash" if crashed else "ok",
                  "crash_indicator": bool(crashed),
                  "amplitude": amp, "baseline": base,
                  "fraction_of_baseline": round(frac, 4),
                  "threshold_fraction": _CRASH_FRACTION,
                  "note": (f"振幅 {amp:.4g} 只有自由振荡基线 {base:.4g} 的 "
                           f"{frac:.1%}（阈值 {_CRASH_FRACTION:.0%}）—— "
                           "针尖很可能已接触表面。退针后振幅应恢复；"
                           "若退针后仍不恢复，则不是撞针而是振荡回路本身的问题。"
                           if crashed else
                           f"振幅为基线的 {frac:.1%}，针尖仍在自由振荡。"
                           "（这不排除电流类判据发现的问题。）")},
            nanonis_calls=list(read.nanonis_calls))


# ── baseline storage ─────────────────────────────────────────────────
#
# The InstrumentProfile — same store the dI/dV calibration lives in, persisted
# across runs by its sink, so a baseline taken this morning still means
# something this afternoon.
#
# Both keys are registered in ``instrument_profile._CONFIG_SPEC``. That is not
# optional: the profile ``sanitize()``s every write against that spec and
# SILENTLY DROPS unknown keys. An unregistered key would have written cleanly,
# read back None forever, and left this crash detector permanently in
# "no_baseline" — the exact silent-no-op shape this repo has been bitten by
# before (SettingsStore.KNOWN_KEYS, override_store._ALL_FILES).


def _set_baseline(context, amp: float, idx: int, name: str) -> None:
    """Persist the free-oscillation baseline. Best-effort; never raises."""
    try:
        from mast.core import instrument_profile as ip

        prof = ip.get_profile()
        prof["qplus_amplitude_baseline"] = float(amp)
        prof["qplus_amplitude_signal_index"] = int(idx)
        ip.set_profile(prof)
        sink = getattr(ip, "_persist_sink", None)
        if callable(sink):
            sink(ip.get_profile())
    except Exception as exc:  # noqa: BLE001 — persistence must not break a read
        logger.debug("qplus baseline persist failed: %s", exc)


def _remembered_index() -> int | None:
    """The amplitude signal index on file, or None to fall back to a name scan.

    -1 is the documented "auto-discover" sentinel and must NOT be treated as a
    real index — passing it to ``Signals_ValGet`` would read whatever the
    controller does with a negative channel."""
    try:
        from mast.core.instrument_profile import get_config

        v = get_config("qplus_amplitude_signal_index", None)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return int(v)
    except Exception as exc:  # noqa: BLE001
        logger.debug("qplus remembered index read failed: %s", exc)
    return None


def _excitation_state(context) -> "tuple[bool | None, str]":
    """返回音叉激励是否有效：(True/False/None, 说明)。
    同时要求 PLL_OutOnOffGet 显示输出开启，且 PLL_ExcitationGet 的激励幅度大于零。
    任一证据不可读返回 None；此时振幅读数不能支持基于自由振荡基线的判定。
    """
    on = None
    exc_v = None
    try:
        rec = context.safe_call("PLL_OutOnOffGet", 1)
        if not getattr(rec, "error", ""):
            raw = _first_scalar(getattr(rec, "return_value", None))
            if raw is not None:
                on = bool(int(raw))
    except Exception:  # noqa: BLE001 — 读不到就是读不到,不抛
        logger.debug("读 PLL 输出开关失败", exc_info=True)
    try:
        rec_v = context.safe_call("PLL_ExcitationGet", 1)
        if not getattr(rec_v, "error", ""):
            exc_v = _first_scalar(getattr(rec_v, "return_value", None))
    except Exception:  # noqa: BLE001
        logger.debug("读 PLL 激励幅度失败", exc_info=True)

    if on is None or exc_v is None:
        return None, ("读不到 PLL 激励状态 —— **判不了**是否撞针。"
                      "音叉有没有被驱动不知道的时候,振幅读数不代表任何东西。")
    if not on or float(exc_v) <= 0.0:
        return False, (f"PLL 激励关闭(输出 {'开' if on else '关'},"
                       f"激励 {float(exc_v):.4g} V)—— **振幅通道无判据能力**:"
                       "音叉没被驱动,读到的是未驱动解调器的噪声底,"
                       "拿它跟自由振荡基线比永远比出「塌了」。这不是撞针提示。")
    return True, ""


def _first_scalar(rv):
    """回包里的第一个标量(两种形态都接,§2.21)。"""
    if not isinstance(rv, (list, tuple)) or len(rv) <= 2:
        return None
    vals = rv[2]
    if isinstance(vals, (int, float)) and not isinstance(vals, bool):
        return float(vals)
    if isinstance(vals, (list, tuple)) and vals:
        item = vals[0]
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        try:
            return float(item)
        except (TypeError, ValueError):
            return None
    return None


def _get_baseline(context) -> float | None:
    try:
        from mast.core.instrument_profile import get_config

        v = get_config("qplus_amplitude_baseline", None)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v)
    except Exception as exc:  # noqa: BLE001
        logger.debug("qplus baseline read failed: %s", exc)
    return None


def _scalar(record) -> float | None:
    """First numeric value out of a Signals_ValGet reply."""
    rv = getattr(record, "return_value", None)
    if not (isinstance(rv, (list, tuple)) and len(rv) > 2):
        return None
    body = rv[2]
    if isinstance(body, (int, float)) and not isinstance(body, bool):
        return float(body)
    if isinstance(body, (list, tuple)):
        for el in body:
            if isinstance(el, (int, float)) and not isinstance(el, bool):
                return float(el)
    return None


__all__ = [
    "ReadTipOscillationAmplitude",
    "CheckTipCrashByAmplitude",
    "find_amplitude_signal",
]
