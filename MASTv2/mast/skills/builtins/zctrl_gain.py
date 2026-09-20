"""Z controller gain skills.

vendored from v1 mast/skills/builtins/zctrl_gain.py 2026-04-23.

2026-08-03 — the gains stopped being "just three numbers". On the rig, an agent
sent ``p_gain=3`` meaning ``3e-12`` (the exponent was lost between its own
reasoning and the tool call) and the write went through untouched: the parameters
carried ``min_value=0`` and no upper bound, and their ``unit`` was the empty
string, which made them invisible to every unit-matched safety table as well. The
Z loop was open at the time, so nothing moved — with it closed, P = 3 metres
drives the tip into the sample the moment feedback engages.

Three independent things changed here as a result, and it is worth knowing which
one does what, because only the first two defend against a wrong number ARRIVING:

  1. Real bounds and real units (``max_value`` + ``unit``) — this is what rejects
     ``p_gain=3``. It is also what lets ``core.safety`` see these parameters at
     all; unit matching there is exact, and "" matches nothing.
  2. ``_PHYSICAL_ABSURD`` entries in ``core.safety`` — the rig-independent floor
     under (1), phrased as "fix the magnitude", not "raise the limit".
  3. Write-then-read-back verification (below) — this does NOT catch a wrong
     number: if the model sends 3.0, the hardware takes 3.0 and reads back 3.0,
     which agrees. What it catches is the write not landing as issued. Keeping
     that distinction straight matters; a verification step credited with a job
     it cannot do is worse than none.

2 skills: SetZCtrlGain, GetZCtrlGain.
"""

from __future__ import annotations

from mast.core.safety import implausible_reading_notes
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill
from mast.skills.verify import values_match

#: Physical units of the three gains, in the order ``ZCtrl_GainSet`` takes them.
#: Kept in one place because both skills need it and because the unit strings are
#: load-bearing — the safety tables match on them exactly.
_GAIN_UNITS = {"p_gain": "m", "time_constant_s": "s", "i_gain": "m/s"}


def _parse_gains(record) -> "dict[str, float] | None":
    """``{p_gain, time_constant_s, i_gain}`` out of a ``ZCtrl_GainGet`` record.

    None means "could not read", which is deliberately distinct from any set of
    numbers: a caller must never be able to confuse an unreadable instrument with
    one that happens to answer zero.
    """
    if record is None or getattr(record, "error", None):
        return None
    parsed = getattr(record, "return_value", None)
    if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
        vals = parsed[2]
        if isinstance(vals, (list, tuple)) and len(vals) >= 3:
            try:
                return {
                    "p_gain": float(vals[0]),
                    "time_constant_s": float(vals[1]),
                    "i_gain": float(vals[2]),
                }
            except (TypeError, ValueError):  # pragma: no cover — defensive
                return None
    return None


def gain_reading_notes(gains: "dict[str, float] | None") -> list[str]:
    """Magnitude warnings for a set of gains we just READ."""
    if not gains:
        return []
    return implausible_reading_notes(
        {k: (v, _GAIN_UNITS.get(k, "")) for k, v in gains.items()}
    )


class SetZCtrlGain(BaseSkill):
    """Set Z controller gains."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZCtrlGain",
            version="1.1.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置 Z 控制器的 P/I 增益与时间常数。\n\n"
                "**优先使用 ApplyZCtrlPreset**(按参数组名应用,数值由代码从用户"
                "维护的存储里取)。只有当用户在本次对话里逐字念出了具体数值时,"
                "才直接调用本技能传裸数值。\n\n"
                "三个参数都是有量纲的物理量,且都是极小的数 —— **写成带 SI 前缀的"
                "字符串**(如 '3p'、'16.667u'、'180n')。⚠️ 前缀不可省略:"
                "**指数写法与裸数字都会被拒绝**。I = P / T。"
            ),
            parameters=[
                # Gains MUST be non-negative: a negative Z-controller gain inverts
                # the feedback loop (positive feedback), which drives the tip INTO
                # the surface instead of holding setpoint — an immediate crash.
                #
                # The upper bounds were missing until 2026-08-03, when an exponent
                # lost in transit put p_gain=3 (three METRES) on the instrument.
                # The note that used to sit here said the upper end was "left to
                # the instrument (units are rig-specific)"; that reasoning was
                # wrong twice over — the units are not rig-specific (they are
                # metres and m/s, printed on the Nanonis panel), and "rig-specific"
                # argues for a WIDE bound, not for none at all.
                #
                # These ceilings are ~5-7 decades above anything real (typical:
                # 3e-12 m, 1.7e-5 s, 1.8e-7 m/s), so no legitimate rig — qPlus,
                # odd calibration, room-temperature — is refused. What they stop is
                # the magnitude catastrophe. A rig that needs tighter values than
                # these uses the per-skill override, which is where "this machine"
                # limits belong.
                ParameterSpec(
                    name="p_gain",
                    type="float",
                    description=(
                        "比例增益 —— 它是一个**长度**，单位米 "
                        "(Nanonis 面板上的 'Proportional (m)')。"
                        "典型值 1p ~ 10p m(即 1–10 pm)。必须 >= 0。"
                    ),
                    unit="m",
                    required=True,
                    min_value=0.0,
                    max_value=1e-6,
                ),
                ParameterSpec(
                    name="time_constant_s",
                    type="float",
                    description=(
                        "时间常数，单位秒。典型值 10u ~ 100u s "
                        "(即 10–100 µs)。T = P / I。"
                    ),
                    unit="s",
                    required=True,
                    min_value=0.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="i_gain",
                    type="float",
                    description=(
                        "积分增益 —— 它是一个**速度**，单位米每秒 "
                        "(Nanonis 面板上的 'Integral (m/s)')。"
                        "典型值 10n ~ 1u m/s(即 10 nm/s – 1 µm/s)。"
                        "I = P / T,必须 >= 0。"
                    ),
                    unit="m/s",
                    required=True,
                    min_value=0.0,
                    max_value=1e-3,
                ),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["z", "gain", "write", "readback"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        p = params["p_gain"]
        tc = params["time_constant_s"]
        i = params["i_gain"]
        calls = []

        # 比例与积分**同时为零** = 控制器不产生任何输出，反馈环死掉 —— 而
        # `z_controller_on` 仍然读 True。每一条问「Z 控制器开着吗」的安全判据
        # （进针前置、扫描前置、粗动闸门）都会拿到一个误导性的「是」，然后放行。
        # 带着死环进针的意思是：针在下降，而没有任何东西会在建立隧道时停住它。
        #
        # 三个参数的 min_value 都是 0.0，所以 0/0/0 一路合法通过到硬件（用户
        # 2026-08-04 拍板堵掉）。**下限不改**：把 min_value 抬成某个非零数是在
        # 编造一个物理上并不存在的界，而且会让 schema 拒绝的理由变得无法解释。
        #
        # 只禁「两个都零」这一种。单独 p_gain=0 不禁 —— 纯积分控制是真实存在的
        # 控制方式，而这个技能把三个参数独立传给 `ZCtrl_GainSet`，Nanonis 收到
        # 不自洽三元组时怎么处理没有验证过。禁一个没把握的组合，会在某天挡住一次
        # 合法操作，而那时没人记得这里为什么这样写。
        # 先容错解析再比较。agent 路径上 `skill_adapter._coerce_si_params` 已经把
        # "3p"/"3e-12" 这类字符串转成了浮点（2026-08-04 起有量纲参数以字符串到达
        # 模型，那是数值损坏的修复），但手动执行路径没有验证过。**一条依赖上游
        # 一定转换过的安全守卫，等于一条会被字符串 "0" 绕过的守卫。**
        def _num(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            try:
                return float(v)          # "0" / "0.0" / "3e-12" —— 不需要任何依赖
            except (TypeError, ValueError):
                pass
            try:                          # "3p" 这类 SI 前缀才需要解析器
                from mast.core.si_quantity import parse_quantity

                return parse_quantity(v, strict=False, what="gain")
            except Exception:  # noqa: BLE001 — 解析不了就交给下游报错
                return None

        p_n, i_n = _num(p), _num(i)
        if p_n == 0 and i_n == 0:
            return SkillResult(
                skill_name="SetZCtrlGain",
                success=False,
                error=(
                    "拒绝：比例增益与积分增益同时为零。"
                    "这样写出去，控制器不产生任何输出——反馈环实际上是死的，"
                    "但 Z 控制器状态**仍然报 On**，所有「控制器开着吗」的前置检查"
                    "都会被这个假的「是」放行。\n\n"
                    "如果你要关掉反馈，用 `ZControllerOnOff(on=false)` —— "
                    "那条路会把状态如实报成 Off，下游判据才拿得到真相。\n\n"
                    "如果你是想设一组很小的增益，请检查是不是量级写丢了："
                    "本机典型值 p_gain='3p'、time_constant_s='16.667u'、"
                    "i_gain='180n'。这三个参数必须写成**带 SI 前缀的字符串**，"
                    "'3e-12' 与裸数字都会被拒。"
                ),
                data={"p_gain": p, "time_constant_s": tc, "i_gain": i,
                      "rejected_by": "zero_pi_guard"},
                nanonis_calls=calls,
            )

        # Read the CURRENT gains before writing. Two reasons, and the second is
        # the one that matters: (a) it is what we restore to if the write lands
        # wrong, and (b) it is read now rather than remembered, so it is a fact
        # about the instrument instead of a claim about what we think we set.
        prior_rec = context.safe_call("ZCtrl_GainGet")
        calls.append(prior_rec)
        prior = _parse_gains(prior_rec)

        # ZCtrl_GainSet(P_gain, Time_constant_s, I_gain)
        record = context.safe_call("ZCtrl_GainSet", p, tc, i)
        calls.append(record)
        if record.error:
            return SkillResult(
                skill_name="SetZCtrlGain",
                success=False,
                error=record.error,
                nanonis_calls=calls,
            )

        requested = {"p_gain": p, "time_constant_s": tc, "i_gain": i}
        back_rec = context.safe_call("ZCtrl_GainGet")
        calls.append(back_rec)
        got = _parse_gains(back_rec)

        if got is None:
            # The write DID happen and reported no error; only the confirmation is
            # missing. Reporting failure here would be a lie in the other
            # direction — and would invite a retry of a write that already landed.
            return SkillResult(
                skill_name="SetZCtrlGain",
                success=True,
                data={
                    **requested,
                    "readback_ok": None,
                    "readback_unavailable": True,
                    "note": (
                        "增益已写入,但回读失败,无法确认硬件真的接受了这组值。"
                        "进针或扫图前请在 Nanonis 面板上人工核对 Z-Controller 增益。"
                    ),
                },
                nanonis_calls=calls,
            )

        # The verdict is computed HERE. It is never phrased as two numbers for
        # something downstream to compare — see values_match's docstring for the
        # 2026-08-03 report that called 3.0 and 3e-12 "float32 精度范围内".
        mismatches: list[str] = []
        for key in ("p_gain", "time_constant_s", "i_gain"):
            ok, detail = values_match(requested[key], got[key])
            if not ok:
                mismatches.append(f"{key}: {detail}")

        if mismatches:
            restored = False
            restore_error = None
            if prior is not None:
                r = context.safe_call(
                    "ZCtrl_GainSet",
                    prior["p_gain"],
                    prior["time_constant_s"],
                    prior["i_gain"],
                )
                calls.append(r)
                restored = not r.error
                restore_error = r.error
            tail = (
                "已还原为写入前的值。"
                if restored
                else (
                    "无法还原(写入前的值也读不到)。"
                    if prior is None
                    else f"还原也失败了({restore_error})。"
                )
            )
            return SkillResult(
                skill_name="SetZCtrlGain",
                success=False,
                error=(
                    "写后回读不一致 —— 硬件里的增益不是刚才请求的值: "
                    + "; ".join(mismatches)
                    + f"。{tail}"
                    "**不要进针、不要扫图**,先在 Nanonis 面板上人工核对 "
                    "Z-Controller → Controller Adjustment。"
                ),
                data={
                    "requested": requested,
                    "readback": got,
                    "prior": prior,
                    "restored": restored,
                    "readback_ok": False,
                },
                nanonis_calls=calls,
            )

        data = {**requested, "readback": got, "readback_ok": True}
        notes = gain_reading_notes(got)
        if notes:
            # In range for the loop, in range for the bounds, and still absurd as
            # a physical quantity — say so rather than letting the numbers pass as
            # confirmation.
            data["warnings"] = notes
        return SkillResult(
            skill_name="SetZCtrlGain",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class GetZCtrlGain(BaseSkill):
    """Read Z controller gains."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZCtrlGain",
            version="1.1.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读回 Z 控制器当前的 P/I 增益与时间常数。"
                "P 的单位是米(m),I 的单位是米每秒(m/s),T 是秒。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "gain", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_GainGet")
        if record.error:
            return SkillResult(
                skill_name="GetZCtrlGain",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        gains = _parse_gains(record)
        if gains is None:
            data: dict = {"raw": decode_reply(record.return_value)}
        else:
            data = dict(gains)
            data["units"] = dict(_GAIN_UNITS)
            notes = gain_reading_notes(gains)
            if notes:
                # A bare number is not neutral: on 2026-08-03 a p_gain of 3.0 was
                # read back and narrated as "= 3e-12, 在 float32 精度范围内".
                # Annotated with its own magnitude, it cannot be read that way.
                data["warnings"] = notes
        return SkillResult(
            skill_name="GetZCtrlGain",
            success=True,
            data=data,
            nanonis_calls=[record],
        )
