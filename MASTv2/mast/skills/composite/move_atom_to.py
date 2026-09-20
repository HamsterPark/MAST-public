"""``MoveAtomTo`` —— 用针尖把一个吸附原子横向搬到指定位置。

Eigler 与 Schweizer 1990 年的做法,一步不多:把针尖降到原子后方,把结电阻降到原子肯跟着
走的程度,慢慢拖到目标,**先把电阻升回成像值再离开**,然后复扫确认。

几何约定(写死,免得 pull / push 的说法各理解各的):针尖从 ``atom − u·approach_offset``
出发(``u`` 是原子指向目标的单位向量),扫过原子把它带走,终点就是 ``target``。不提供
push 模式——模拟器不建排斥模型,同一次运行里换模型也无从验证。

**顺序是安全的一部分。** 进入操纵条件时先切电流量程再降偏压最后降设定点;离开时倒过来,
**先把设定点升回成像值**——不然抬着一条几十 kΩ 的结走开,会把刚放好的原子又拖回来。
中止(``ctx.abort``)之后还原步骤跑不了,仪器就停在操纵条件上,所以那种情况的错误信息里
必须点名说清楚。
"""

from __future__ import annotations

import logging
import math
from typing import Iterator

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeProgress, CompositeStep, GraphExecutor

logger = logging.getLogger(__name__)

#: 拖拽航点的最大步长:一步太长就等于让针尖瞬移过去,原子跟不上
MAX_WAYPOINT_M = 1e-10
#: 电流量程必须留出的余量:设定点顶到量程上沿,前放读不回来,Z 环会一路伸到撞针
GAIN_HEADROOM = 2.0
#: 重试时把设定点乘这个数(降电阻),并封顶在安全界内
RETRY_SETPOINT_FACTOR = 1.5
MAX_SETPOINT_A = 100e-9


class MoveAtomTo(CompositeSkillGraph):
    """把一个吸附原子横向搬到目标位置,并复扫确认。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MoveAtomTo",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一个吸附原子从当前位置横向搬到目标位置:降低结电阻让原子跟着针尖走,"
                "拖到目标后**先升回成像电阻再离开**,然后复扫一小帧确认。"
                "针尖从原子后方出发扫过它,终点就是目标位置。"
                "默认参数是文献量级的起点,不是这台机器上标定过的值:原子跟不动就把设定点"
                "调高(电阻调低)重试,跟得太狠会被针尖捡走。"
            ),
            parameters=[
                ParameterSpec(name="atom_x_m", type="float", unit="m",
                              description="原子现在的 x,例如 '12.5n'(SI 前缀必须写)。",
                              required=True, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(name="atom_y_m", type="float", unit="m",
                              description="原子现在的 y,例如 '-3n'。",
                              required=True, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(name="target_x_m", type="float", unit="m",
                              description="目标位置 x,例如 '16.5n'。",
                              required=True, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(name="target_y_m", type="float", unit="m",
                              description="目标位置 y,例如 '-3n'。",
                              required=True, min_value=-1.5e-6, max_value=1.5e-6),
                # MoveToXY 的语义:坐标来自存储过的地图/旧扫描才传,现读的就别传。
                ParameterSpec(name="coord_epoch", type="int",
                              description=("坐标的代次。坐标来自地图标记或旧扫描时传;"
                                           "刚扫出来的坐标不要传。"),
                              required=False, default=None, min_value=0, max_value=1000000),
                ParameterSpec(name="manip_bias_v", type="float",
                              description=("操纵偏压的**大小**,单位伏(普通数字,例如 "
                                           "0.01)。符号跟随当前成像偏压,不穿零。"),
                              required=False, default=0.01, min_value=0.001, max_value=0.5),
                ParameterSpec(name="manip_setpoint_a", type="float", unit="A",
                              description="操纵设定点,例如 '57n'。",
                              required=False, default=57e-9,
                              min_value=1e-9, max_value=MAX_SETPOINT_A),
                ParameterSpec(name="manip_speed_m_s", type="float", unit="m/s",
                              description="拖拽速度,例如 '500p'(0.5 nm/s)。",
                              required=False, default=5e-10, min_value=1e-11, max_value=1e-7),
                ParameterSpec(name="approach_offset_m", type="float", unit="m",
                              description="出发点落在原子后方多远,例如 '300p'。",
                              required=False, default=3e-10, min_value=0.0, max_value=1e-9),
                ParameterSpec(name="precision_m", type="float", unit="m",
                              description=("算「到位」的半径,例如 '150p'。不要小于晶格"
                                           "间距的一半——原子只能停在格点上。"),
                              required=False, default=1.5e-10, min_value=1e-11, max_value=2e-9),
                ParameterSpec(name="verify", type="bool",
                              description="搬完复扫一帧确认(默认开)。",
                              required=False, default=True),
                ParameterSpec(name="verify_size_m", type="float", unit="m",
                              description="确认帧的边长,例如 '6n'。",
                              required=False, default=6e-9, min_value=1e-9, max_value=5e-8),
                ParameterSpec(name="max_attempts", type="int",
                              description="最多尝试几次(失败一次就降电阻重来)。",
                              required=False, default=2, min_value=1, max_value=3),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=120.0,
            composition_level=3,
            tags=["composite", "manipulation", "atom", "tip", "folme", "write"],
        )

    # ── plan ──
    def plan_dynamic(self, params: dict, executor: GraphExecutor) -> Iterator[CompositeStep]:
        ax, ay = float(params["atom_x_m"]), float(params["atom_y_m"])
        tx, ty = float(params["target_x_m"]), float(params["target_y_m"])
        epoch = params.get("coord_epoch")
        max_attempts = int(params.get("max_attempts") or 2)
        setpoint = float(params.get("manip_setpoint_a") or 57e-9)

        yield CompositeStep(step_id="read:bias", skill_name="GetBias", params={},
                            tags=("readback",))
        yield CompositeStep(step_id="read:setpoint", skill_name="GetSetpoint", params={},
                            tags=("readback",))
        yield CompositeStep(step_id="read:speed", skill_name="GetTipSpeed", params={},
                            tags=("readback",))
        # the preamp range, without which the manipulation setpoint cannot be checked against
        # it. A setpoint above the range reads back saturated, the Z loop never sees it reach
        # the target, and the tip extends until it hits the surface.
        yield CompositeStep(step_id="read:gain", skill_name="GetCurrentGains", params={},
                            optional=True, tags=("readback", "preamp"))
        prior = self._prior(executor)
        executor.set_partial("prior", prior)

        for attempt in range(1, max_attempts + 1):
            pre = f"a{attempt}"
            ux, uy = self._unit(ax, ay, tx, ty)
            off = float(params.get("approach_offset_m") or 0.0)
            start = (ax - ux * off, ay - uy * off)
            move_params = {"wait": True}
            if epoch is not None:
                move_params["coord_epoch"] = int(epoch)
            yield CompositeStep(step_id=f"{pre}:pre_move", skill_name="MoveToXY",
                                params={"x_m": start[0], "y_m": start[1], **move_params},
                                tags=("move", "approach"))
            gain = self._gain_for(setpoint, prior)
            if gain is not None:
                yield CompositeStep(step_id=f"{pre}:manip_gain", skill_name="SetCurrentGain",
                                    params={"gain_index": gain}, optional=True,
                                    tags=("setup", "preamp"))
            bias = math.copysign(abs(float(params.get("manip_bias_v") or 0.01)),
                                 prior.get("bias_v") or 1.0)
            yield CompositeStep(step_id=f"{pre}:manip_bias", skill_name="SetBias",
                                params={"bias_v": bias}, tags=("setup",))
            yield CompositeStep(step_id=f"{pre}:manip_setpoint", skill_name="SetSetpoint",
                                params={"setpoint_a": setpoint}, tags=("setup",))
            speed = float(params.get("manip_speed_m_s") or 5e-10)
            yield CompositeStep(step_id=f"{pre}:manip_speed", skill_name="SetTipSpeed",
                                params={"speed_m_s": speed, "custom_speed": True},
                                tags=("setup",))
            for n, (wx, wy) in enumerate(self._waypoints(start, (tx, ty))):
                yield CompositeStep(step_id=f"{pre}:drag_{n:03d}", skill_name="MoveToXY",
                                    params={"x_m": wx, "y_m": wy, **move_params},
                                    optional=True, checkpoint_after=False,
                                    tags=("move", "drag"))
            # release: raise the resistance BEFORE the tip goes anywhere
            yield CompositeStep(step_id=f"{pre}:restore_setpoint", skill_name="SetSetpoint",
                                params={"setpoint_a": prior["setpoint_a"] or 50e-12},
                                optional=True,
                                tags=("restore",))
            yield CompositeStep(step_id=f"{pre}:restore_bias", skill_name="SetBias",
                                params={"bias_v": prior["bias_v"] or 0.1}, optional=True,
                                tags=("restore",))
            if gain is not None and prior.get("gain_index") is not None:
                yield CompositeStep(step_id=f"{pre}:restore_gain", skill_name="SetCurrentGain",
                                    params={"gain_index": int(prior["gain_index"])},
                                    optional=True, tags=("restore",))
            yield CompositeStep(step_id=f"{pre}:restore_speed", skill_name="SetTipSpeed",
                                params={"speed_m_s": prior["speed_m_s"],
                                        "custom_speed": bool(prior.get("custom_speed", True))},
                                optional=True, tags=("restore",))
            if not params.get("verify", True):
                executor.set_partial("moved", None)
                return
            yield CompositeStep(step_id=f"{pre}:verify_scan", skill_name="ScanAt",
                                params={"center_x_m": tx, "center_y_m": ty,
                                        "size_m": float(params.get("verify_size_m") or 6e-9)},
                                optional=True, tags=("verify", "scan"))
            scan_path = self._scan_path(executor, f"{pre}:verify_scan")
            if scan_path is None:
                executor.set_partial("moved", None)
                executor.set_partial("verify_verdict", "no_frame")
                return
            executor.set_partial("verify_scan_path", scan_path)
            yield CompositeStep(step_id=f"{pre}:verify", skill_name="VerifyAdatomAt",
                                params={"scan_path": scan_path, "target_x_m": tx,
                                        "target_y_m": ty,
                                        "tolerance_m": float(params.get("precision_m") or 1.5e-10)},
                                optional=True, tags=("verify",))
            res = executor.sub_results.get(f"{pre}:verify")
            verdict = (res.data or {}).get("verdict") if res is not None else None
            executor.set_partial("verify_verdict", verdict)
            executor.set_partial("attempts", attempt)
            if verdict == "at_target":
                executor.set_partial("moved", True)
                executor.set_partial("final_x_m", (res.data or {}).get("found_x_m"))
                executor.set_partial("final_y_m", (res.data or {}).get("found_y_m"))
                executor.set_partial("residual_m", (res.data or {}).get("residual_m"))
                executor.set_partial("bystanders", (res.data or {}).get("others"))
                return
            executor.set_partial("moved", False)
            executor.set_partial("residual_m", (res.data or {}).get("residual_m") if res else None)
            executor.set_partial("bystanders", (res.data or {}).get("others") if res else None)
            if verdict != "displaced" or attempt >= max_attempts:
                # not_found / ambiguous: we do not know where the atom is, so trying again
                # with a lower resistance would just be dragging something unidentified
                return
            ax = float((res.data or {}).get("found_x_m") or ax)
            ay = float((res.data or {}).get("found_y_m") or ay)
            setpoint = min(setpoint * RETRY_SETPOINT_FACTOR, MAX_SETPOINT_A)

    # ── helpers ──
    @staticmethod
    def _unit(ax, ay, tx, ty) -> tuple[float, float]:
        dx, dy = tx - ax, ty - ay
        n = math.hypot(dx, dy)
        return (1.0, 0.0) if n <= 0 else (dx / n, dy / n)

    @staticmethod
    def _waypoints(start, end) -> list[tuple[float, float]]:
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = math.hypot(dx, dy)
        n = max(1, int(math.ceil(dist / MAX_WAYPOINT_M)))
        n = min(n, 400)                       # a metre-long drag is not a manipulation
        return [(start[0] + dx * (i + 1) / n, start[1] + dy * (i + 1) / n) for i in range(n)]

    @staticmethod
    def _gain_for(setpoint_a: float, prior: dict) -> int | None:
        """A gain index whose full scale clears the manipulation setpoint with headroom.

        A setpoint above the preamp's range reads back as saturation, the Z loop never sees
        it reach the target, and it extends until the tip hits the surface."""
        full = prior.get("full_scale_a")
        if not full or float(full) >= setpoint_a * GAIN_HEADROOM:
            return None
        idx = prior.get("gain_index")
        return max(0, int(idx) - 1) if idx is not None else None

    def _prior(self, executor: GraphExecutor) -> dict:
        def data(step_id: str) -> dict:
            res = executor.sub_results.get(step_id)
            return (res.data or {}) if res is not None and isinstance(res.data, dict) else {}

        def pick(d: dict, *keys, default):
            """A key present with a None value is "not given", not "set to nothing".

            Reading it as a value puts None into the restore step, SetSetpoint refuses it, and
            the composite ends with the junction still at the manipulation resistance — which
            is the one failure this whole skill exists to avoid."""
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return v
            return default

        bias = data("read:bias")
        setp = data("read:setpoint")
        speed = data("read:speed")
        gain = data("read:gain")
        return {"bias_v": pick(bias, "bias_v", "bias", default=0.1),
                "setpoint_a": pick(setp, "setpoint_a", "setpoint", default=50e-12),
                "speed_m_s": pick(speed, "speed_m_s", "speed", default=293e-9),
                "custom_speed": speed.get("custom_speed", True),
                # from the gain readback, not from the speed one: reading them off the wrong
                # step is silent, and its only symptom is that the range is never widened
                "gain_index": gain.get("gain_index"),
                "full_scale_a": gain.get("full_scale_a")}

    @staticmethod
    def _scan_path(executor: GraphExecutor, step_id: str) -> str | None:
        res = executor.sub_results.get(step_id)
        if res is None or not isinstance(res.data, dict):
            return None
        for key in ("saved_path", "scan_path", "path", "product_path"):
            val = res.data.get(key)
            if isinstance(val, str) and val:
                return val
        return None

    # ── result ──
    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        p = progress.partial_data
        prior = p.get("prior") or {}
        restored = {
            "setpoint": self._ok(sub_results, "restore_setpoint"),
            "bias": self._ok(sub_results, "restore_bias"),
            "speed": self._ok(sub_results, "restore_speed"),
        }
        gain_steps = [k for k in sub_results if k.endswith("restore_gain")]
        if gain_steps:
            restored["gain"] = self._ok(sub_results, "restore_gain")
        instrument_restored = restored["setpoint"] and restored["bias"]
        r = p.get("residual_m")
        return {
            "moved": p.get("moved"),
            "verify_verdict": p.get("verify_verdict"),
            "final_x_m": p.get("final_x_m"), "final_y_m": p.get("final_y_m"),
            "residual_m": r, "residual_nm": (None if r is None else float(r) * 1e9),
            "attempts": p.get("attempts", 1),
            "steps": list(sub_results),
            "restored": restored, "instrument_restored": bool(instrument_restored),
            "prior_bias_v": prior.get("bias_v"), "prior_setpoint_a": prior.get("setpoint_a"),
            "junction_resistance_ohm": self._resistance(p),
            "bystanders": p.get("bystanders"),
            "verify_scan_path": p.get("verify_scan_path"),
        }

    @staticmethod
    def _ok(sub_results: dict, suffix: str) -> bool:
        hits = [r for k, r in sub_results.items() if k.endswith(suffix)]
        return bool(hits) and bool(hits[-1].success)

    @staticmethod
    def _resistance(partial: dict) -> float | None:
        b, i = partial.get("manip_bias_used_v"), partial.get("manip_setpoint_used_a")
        if b is None or not i:
            return None
        return abs(float(b)) / float(i)

    def run_composite(self, context, params: dict) -> SkillResult:
        executor = GraphExecutor(composite_name=self._skill_name(), context=context,
                                 on_step_result=self.on_step_result,
                                 on_step_failed=self.on_step_failed)
        executor.set_partial_default("attempts", 1)
        executor.set_partial("manip_bias_used_v", params.get("manip_bias_v"))
        executor.set_partial("manip_setpoint_used_a", params.get("manip_setpoint_a"))
        self._executor = executor
        executor.run_plan(self.plan_dynamic(params, executor))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        if not data["instrument_restored"]:
            return SkillResult(
                skill_name=self._skill_name(), success=False, data=data,
                error=(f"仪器仍停在操纵条件上(设定点 {params.get('manip_setpoint_a')} A、"
                       f"偏压 {params.get('manip_bias_v')} V):立刻 SetSetpoint / SetBias "
                       f"还原成像值,否则下一次扫描会把表面拖乱。"),
                summary="搬运中止,仪器未还原")
        if data["moved"] is False:
            r = data.get("residual_nm")
            return SkillResult(
                skill_name=self._skill_name(), success=False, data=data,
                error=(f"原子没到位({data.get('verify_verdict')}"
                       + (f",残差 {r:.2f} nm" if r is not None else "")
                       + f",试了 {data['attempts']} 次)"),
                summary="搬运未成功,仪器已还原")
        summary = ("原子已到位" if data["moved"] else "已搬运,未复扫确认")
        if data.get("residual_nm") is not None:
            summary += f",残差 {data['residual_nm'] * 1000:.0f} pm"
        return SkillResult(skill_name=self._skill_name(), success=True, data=data,
                           summary=summary)

