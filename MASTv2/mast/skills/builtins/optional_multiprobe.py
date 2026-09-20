"""Multi-probe: N independent scanners, each with its own Z loop, bias and current.

**Optional hardware, OFF by default** (设置 → 硬件模块).

A 4-probe STM is four microscopes sharing a sample. Every ``MProbe*`` call takes a
``Scanner_Index`` as its FIRST argument, and every one of those probes is a tip
that can be crashed independently of the others.

TWO THINGS THAT WILL BITE
=========================
1. **The probe index is not optional and there is no sensible default.** MAST does
   NOT default ``probe`` to 0 — an agent that omits the index and gets probe 0 is
   an agent that crashed the wrong tip. It is a required parameter everywhere.

2. **The single-probe skills do not know about probes.** ``SetBias``,
   ``ZController``, ``Withdraw`` and friends act on the MAIN Nanonis channel. On a
   multi-probe rig that is one specific probe (whichever is the "active scanner"),
   not all of them. To act on probe N you must use the skills in this file. This is
   why ``WithdrawProbe`` exists alongside the plain ``Withdraw``: pressing 中止
   retracts via the main channel *and* — because ``MProbeZCtrl_Withdraw`` is on the
   post-abort allow-list — the agent can still retract each probe explicitly.
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

_MAX_PROBE = 7   # Nanonis multi-probe tops out well below this; a generous bound.


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """用 io.nanonis_files.decode_reply 提取回包内容，排除错误与原始字节信封。
    
    原始 bytes 不应进入需要 JSON 序列化的结果；所有调用方共用同一解码入口。"""
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


def _probe_spec() -> ParameterSpec:
    """The probe index. Required, always — see the module docstring."""
    return ParameterSpec(
        name="probe", type="int",
        description=("**哪一根探针**（扫描器序号，0 起算）。必填 —— 没有默认值，"
                     "因为默认下去会撞坏错的那根针尖。"),
        required=True, min_value=0, max_value=_MAX_PROBE,
    )


class SetProbeZController(BaseSkill):
    """One probe's Z feedback: on/off, setpoint, gains."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetProbeZController",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定**某一根**探针的 Z 控制器：开／关它，并设定它的设定值与增益。\n"
                "\n"
                "当心。把一根探针的 Z 环打开、而设定值又是它根本达不到的，会把那根探针**扎进样品**。"
                "把它关掉则会让探针停在当前 Z 上、没有任何东西托着它 —— 想让探针处于安全状态，请用 WithdrawProbe，"
                "而不是关掉。\n"
                "\n"
                "这个技能只作用于你点名的那根探针，别的一根都不动。普通的 ZController 技能作用于**主**通道，"
                "那是某一根特定的探针 —— 除非它恰好就是当前的扫描探针，否则不是这一根。"
            ),
            parameters=[
                _probe_spec(),
                ParameterSpec(name="on", type="bool",
                              description="闭合这根探针的 Z 反馈环",
                              required=True),
                ParameterSpec(name="setpoint", type="float",
                              description="Z 控制器设定值（电流，单位 A）—— 不传则保持不变",
                              unit="A", required=False, default=None),
                ParameterSpec(name="p_gain", type="float",
                              description="比例增益 —— 不传则保持不变",
                              required=False, default=None, min_value=0.0),
                ParameterSpec(name="i_gain", type="float",
                              description="积分增益 —— 不传则保持不变",
                              required=False, default=None, min_value=0.0),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["multiprobe", "zcontroller", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetProbeZController"
        probe = int(params["probe"])
        calls: list = []

        sp = params.get("setpoint")
        if sp is not None:
            rec = context.safe_call("MProbeZCtrl_SetpntSet", probe, float(sp))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"MProbeZCtrl_SetpntSet failed: {rec.error}", calls)

        pg, ig = params.get("p_gain"), params.get("i_gain")
        if pg is not None or ig is not None:
            # GainSet takes both. Read the current pair so an omitted one is preserved
            # rather than silently zeroed — a zeroed I gain is a dead loop.
            cur = context.safe_call("MProbeZCtrl_GainGet", probe)
            calls.append(cur)
            if cur.error:
                return _fail(name, f"MProbeZCtrl_GainGet failed: {cur.error}", calls)
            got = _rv(cur)
            cur_p, cur_i = (got[0], got[1]) if isinstance(got, (list, tuple)) and len(got) >= 2 \
                else (0.0, 0.0)
            rec = context.safe_call("MProbeZCtrl_GainSet", probe,
                                    float(pg) if pg is not None else float(cur_p),
                                    float(ig) if ig is not None else float(cur_i))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"MProbeZCtrl_GainSet failed: {rec.error}", calls)

        on = bool(params["on"])
        rec = context.safe_call("MProbeZCtrl_OnOffSet", probe, 1 if on else 0)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"MProbeZCtrl_OnOffSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"probe": probe, "controller_on": on},
            summary=f"探针 {probe} 的 Z 反馈已{'闭合' if on else '断开'}",
        )


class GetProbeZController(BaseSkill):
    """One probe's Z state."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetProbeZController",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读**某一根**探针的 Z 控制器：它是否开着、它的设定值、增益、Z 位置与 Z 限值。"
                "移动那根探针之前先读这个。"
            ),
            parameters=[_probe_spec()],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["multiprobe", "zcontroller", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        calls: list = []
        data: dict = {"probe": probe}
        # Literal verbs: every safety tool in this repo (abort-policy check,
        # security audit, API-coverage census) finds Nanonis calls by grepping
        # safe_call("…"). A verb behind a variable is invisible to all of them.
        for key, thunk in (
            ("controller_on", lambda: context.safe_call("MProbeZCtrl_OnOffGet", probe)),
            ("setpoint", lambda: context.safe_call("MProbeZCtrl_SetpntGet", probe)),
            ("gains", lambda: context.safe_call("MProbeZCtrl_GainGet", probe)),
            ("z_m", lambda: context.safe_call("MProbeZCtrl_ZPosGet", probe)),
            ("limits", lambda: context.safe_call("MProbeZCtrl_LimitsGet", probe)),
        ):
            rec = thunk()
            calls.append(rec)
            data[key] = None if rec.error else _rv(rec)
        return SkillResult(skill_name="GetProbeZController", success=True,
                           nanonis_calls=calls, data=data)


class WithdrawProbe(BaseSkill):
    """Retract one probe. The multi-probe analogue of Withdraw."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="WithdrawProbe",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "退回**某一根**探针：关掉它的 Z 控制器，并把它的 Z 驱到安全（完全退出）的那一端。"
                "\n"
                "\n"
                "这是一根探针的**安全**状态，而且永远允许 —— 包括在 abort 之后。拿不准某根探针的状态，"
                "就把它退回来。普通的 Withdraw 技能只作用于主通道；在多探针装置上，你进过针的每一根探针都必须各自退回。"
            ),
            parameters=[_probe_spec()],
            estimated_duration_s=3.0,
            composition_level=0,
            tags=["multiprobe", "withdraw", "safe", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        rec = context.safe_call("MProbeZCtrl_Withdraw", probe)
        if rec.error:
            return _fail("WithdrawProbe", f"MProbeZCtrl_Withdraw failed: {rec.error}", [rec])
        return SkillResult(skill_name="WithdrawProbe", success=True, nanonis_calls=[rec],
                           data={"probe": probe, "withdrawn": True},
                           summary=f"探针 {probe} 已退回")


class ConfigureProbeScanner(BaseSkill):
    """One probe's scanner: calibration and speed."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureProbeScanner",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定**某一根**探针的扫描器标定（X/Y/Z 系数）与移动速度。\n"
                "\n"
                "标定把下达的伏特换算成米。它不是一个装点门面的设置：X 系数错了，就意味着 MoveProbeXY 走的距离和你要的不一样（方向倒是一样）"
                "，而且不会报错 —— 在多探针装置上，这意味着把一根探针开进另一根里去。"
            ),
            parameters=[
                _probe_spec(),
                ParameterSpec(name="speed", type="float",
                              description="扫描器移动速度（m/s）—— 不传则保持不变",
                              unit="m/s", required=False, default=None, min_value=0.0),
                ParameterSpec(name="factor_x", type="float",
                              description="X 标定系数 —— 不传则保持不变",
                              required=False, default=None),
                ParameterSpec(name="factor_y", type="float",
                              description="Y 标定系数 —— 不传则保持不变",
                              required=False, default=None),
                ParameterSpec(name="factor_z", type="float",
                              description="Z 标定系数 —— 不传则保持不变",
                              required=False, default=None),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["multiprobe", "scanner", "calibration", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureProbeScanner"
        probe = int(params["probe"])
        calls: list = []

        fx, fy, fz = params.get("factor_x"), params.get("factor_y"), params.get("factor_z")
        if fx is not None or fy is not None or fz is not None:
            cur = context.safe_call("MProbeScanner_CalibrGet", probe)
            calls.append(cur)
            if cur.error:
                return _fail(name, f"MProbeScanner_CalibrGet failed: {cur.error}", calls)
            got = _rv(cur)
            c = list(got) if isinstance(got, (list, tuple)) and len(got) >= 3 else [1.0, 1.0, 1.0]
            rec = context.safe_call("MProbeScanner_CalibrSet", probe,
                                    float(fx) if fx is not None else float(c[0]),
                                    float(fy) if fy is not None else float(c[1]),
                                    float(fz) if fz is not None else float(c[2]))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"MProbeScanner_CalibrSet failed: {rec.error}", calls)

        sp = params.get("speed")
        if sp is not None:
            rec = context.safe_call("MProbeScanner_SpeedSet", probe, float(sp))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"MProbeScanner_SpeedSet failed: {rec.error}", calls)

        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"probe": probe}, summary=f"探针 {probe} 扫描器已配置")


class MoveProbeXY(BaseSkill):
    """Move one probe laterally."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MoveProbeXY",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "把**某一根**探针移动到一个绝对 (X, Y) 位置，单位**米**。\n"
                "\n"
                "DANGEROUS。这会真的驱动一根针尖横穿样品。在多探针装置上，各根探针共用同一个表面、"
                "彼此可能只隔几微米 —— 一次按错误探针的坐标系算出来的移动、或者用错扫描器标定的移动，"
                "会把一根针尖开进另一根里、或者开上一道台阶边。\n"
                "\n"
                "坐标单位是**米**：100 nm 是 100n，不是 100。先用 GetProbeZController 加扫描器自己的 XY 读取接口读一下当前位置，"
                "并且优先用小步的相对移动。StopProbeScanner 能中止一次正在进行的移动，按 中止 也能。"
            ),
            parameters=[
                _probe_spec(),
                ParameterSpec(name="x_m", type="float",
                              description="绝对 X，单位**米**（100 nm = 100n）",
                              unit="m", required=True,
                              min_value=-1e-3, max_value=1e-3),
                ParameterSpec(name="y_m", type="float",
                              description="绝对 Y，单位**米**（100 nm = 100n）",
                              unit="m", required=True,
                              min_value=-1e-3, max_value=1e-3),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["multiprobe", "scanner", "motion", "dangerous", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        x, y = float(params["x_m"]), float(params["y_m"])
        rec = context.safe_call("MProbeScanner_XYPosSet", probe, x, y)
        if rec.error:
            return _fail("MoveProbeXY", f"MProbeScanner_XYPosSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="MoveProbeXY", success=True, nanonis_calls=[rec],
                           data={"probe": probe, "x_m": x, "y_m": y},
                           summary=f"探针 {probe} 已移动到 ({x:.3e}, {y:.3e}) m")


class StopProbeScanner(BaseSkill):
    """Halt one probe's motion."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopProbeScanner",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "立即停止**某一根**探针的扫描器运动。永远允许，包括在 abort 之后。停止只是中止这次移动 —— 它**不会**把探针退回来；"
                "要退请用 WithdrawProbe。"
            ),
            parameters=[_probe_spec()],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["multiprobe", "scanner", "stop", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        rec = context.safe_call("MProbeScanner_Stop", probe)
        if rec.error:
            return _fail("StopProbeScanner", f"MProbeScanner_Stop failed: {rec.error}", [rec])
        return SkillResult(skill_name="StopProbeScanner", success=True, nanonis_calls=[rec],
                           data={"probe": probe}, summary=f"探针 {probe} 扫描器已停止")


class SetProbeBias(BaseSkill):
    """One probe's bias."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetProbeBias",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定**某一根**探针的偏压，单位**伏特**。\n"
                "\n"
                "每根探针都有自己的偏压。设 probe 1 的偏压不会改变 probe 0 的 —— 而普通的 SetBias 技能作用于主通道，"
                "那是某一根特定的探针，未必是你心里想的那一根。"
            ),
            parameters=[
                _probe_spec(),
                ParameterSpec(name="bias_v", type="float",
                              description="偏压，单位**伏特**",
                              unit="V", required=True, min_value=-10.0, max_value=10.0),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["multiprobe", "bias", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        v = float(params["bias_v"])
        rec = context.safe_call("MProbeBias_Set", probe, v)
        if rec.error:
            return _fail("SetProbeBias", f"MProbeBias_Set failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetProbeBias", success=True, nanonis_calls=[rec],
                           data={"probe": probe, "bias_v": v},
                           summary=f"探针 {probe} 偏压 = {v:g} V")


class PulseProbeBias(BaseSkill):
    """A bias pulse on one probe."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PulseProbeBias",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在**某一根**探针上施加一次偏压**脉冲**：跳到某个值、维持一段设定的宽度，然后回来。"
                "\n"
                "\n"
                "当心。偏压脉冲正是你用来**蓄意改造**针尖或表面的手段 —— 它是 TipPulse 的多探针版本。"
                "它会钝化针尖、挖出一个坑、或者挪动一个吸附物，而且是故意的。宽度单位是**秒**（1 ms = 0.001）"
                "。\n"
                "\n"
                "hold_z=true 会在脉冲期间冻结 Z 控制器（标准做法 —— 反馈环绝不能去追那个电流尖峰）"
                "。relative=true 让这个值成为相对当前偏压的偏移量，而不是绝对值。"
            ),
            parameters=[
                _probe_spec(),
                ParameterSpec(name="value_v", type="float",
                              description="脉冲偏压，单位**伏特**（绝对值；relative 时则为偏移量）",
                              unit="V", required=True, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="width_s", type="float",
                              description="脉冲宽度，单位**秒**（1 ms = 0.001）",
                              unit="s", required=True, min_value=1e-6, max_value=10.0),
                ParameterSpec(name="hold_z", type="bool",
                              description="脉冲期间冻结 Z 控制器（标准做法）",
                              required=False, default=True),
                ParameterSpec(name="relative", type="bool",
                              description="value_v 是相对当前偏压的偏移量",
                              required=False, default=False),
                ParameterSpec(name="wait", type="bool",
                              description="阻塞直到脉冲结束",
                              required=False, default=True),
            ],
            estimated_duration_s=3.0,
            composition_level=0,
            tags=["multiprobe", "bias", "pulse", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        # MProbeBias_Pulse(Scanner_Index, Wait_Until_Done, Width, Value, ZCtrl_Hold, Abs_Rel)
        rec = context.safe_call(
            "MProbeBias_Pulse", probe,
            1 if bool(params.get("wait", True)) else 0,
            float(params["width_s"]),
            float(params["value_v"]),
            1 if bool(params.get("hold_z", True)) else 0,
            1 if bool(params.get("relative", False)) else 0,
        )
        if rec.error:
            return _fail("PulseProbeBias", f"MProbeBias_Pulse failed: {rec.error}", [rec])
        return SkillResult(
            skill_name="PulseProbeBias", success=True, nanonis_calls=[rec],
            data={"probe": probe, "value_v": float(params["value_v"]),
                  "width_s": float(params["width_s"])},
            summary=(f"探针 {probe} 偏压脉冲：{float(params['value_v']):g} V × "
                     f"{float(params['width_s']):g} s"),
        )


class GetProbeBias(BaseSkill):
    """One probe's bias and range."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetProbeBias",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读**某一根**探针的偏压、它的量程设置与标定。",
            parameters=[_probe_spec()],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["multiprobe", "bias", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        calls: list = []
        data: dict = {"probe": probe}
        for key, thunk in (
            ("bias_v", lambda: context.safe_call("MProbeBias_Get", probe)),
            ("range", lambda: context.safe_call("MProbeBias_RangeGet", probe)),
            ("calibration", lambda: context.safe_call("MProbeBias_CalibrGet", probe)),
        ):
            rec = thunk()
            calls.append(rec)
            data[key] = None if rec.error else _rv(rec)
        return SkillResult(skill_name="GetProbeBias", success=True,
                           nanonis_calls=calls, data=data)


class GetProbeCurrent(BaseSkill):
    """One probe's tunnelling current."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetProbeCurrent",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读**某一根**探针的隧道电流（单位安培）及其可用的前放增益档。用这个逐探针的读数来判断那一根特定探针是否处在隧穿距离内 —— 主 Current 通道只报其中一根。"
            ),
            parameters=[_probe_spec()],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["multiprobe", "current", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        calls: list = []
        data: dict = {"probe": probe}
        for key, thunk in (
            ("current_a", lambda: context.safe_call("MProbeCurrent_Get", probe)),
            ("gains", lambda: context.safe_call("MProbeCurrent_GainsGet", probe)),
        ):
            rec = thunk()
            calls.append(rec)
            data[key] = None if rec.error else _rv(rec)
        return SkillResult(skill_name="GetProbeCurrent", success=True,
                           nanonis_calls=calls, data=data)


class ConfigureProbeCurrentGain(BaseSkill):
    """One probe's preamp gain + filter."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureProbeCurrentGain",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定**某一根**探针的电流前放增益与滤波器，用的是该探针自己那份增益表里的**序号**（表用 GetProbeCurrent 读）"
                "。\n"
                "\n"
                "改增益会改变同一个电流**读出来是多少**。要做就在 Z 控制器关闭、或探针已退回的状态下做："
                "环闭合时，一次增益改变在反馈看来就是电流突变，而环的回应是去移动这根探针。"
            ),
            parameters=[
                _probe_spec(),
                ParameterSpec(name="gain_index", type="int",
                              description="这根探针增益表里的序号（表来自 GetProbeCurrent）",
                              required=True, min_value=0, max_value=31),
                ParameterSpec(name="filter_index", type="int",
                              description="这根探针滤波器表里的序号",
                              required=False, default=0, min_value=0, max_value=31),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["multiprobe", "current", "gain", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        probe = int(params["probe"])
        rec = context.safe_call("MProbeCurrent_GainSet", probe,
                                int(params["gain_index"]),
                                int(params.get("filter_index", 0) or 0))
        if rec.error:
            return _fail("ConfigureProbeCurrentGain",
                         f"MProbeCurrent_GainSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="ConfigureProbeCurrentGain", success=True,
                           nanonis_calls=[rec],
                           data={"probe": probe, "gain_index": int(params["gain_index"])},
                           summary=f"探针 {probe} 前放增益索引 = {params['gain_index']}")
