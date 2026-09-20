"""XY positioning skills via Follow Me.

vendored from v1 mast/skills/builtins/navigation.py 2026-04-23. Zero behavioural changes.
1 skill: MoveToXY.
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


#: 读不到 FolMe 速度时的兜底等待上限。**它是兜底,不是估算** ——
#: 超时文案会说清这一点,免得下一个人以为这个数是算出来的。
_FALLBACK_MOVE_TIMEOUT_S = 300.0


def _first_two_floats(return_value) -> "tuple[float, float] | None":
    """从一次回包里取头两个数(剥三段信封 + 解 1-元组包裹)。

    轮询和「量起点」用的是同一份解析 —— 写两遍就会有一天只改一遍。
    """
    body = return_value
    if isinstance(return_value, (list, tuple)) and len(return_value) > 2:
        body = return_value[2]
    try:
        vals = [float(v[0] if isinstance(v, (list, tuple)) and v else v)
                for v in (body if isinstance(body, (list, tuple)) else [])]
    except (TypeError, ValueError):
        return None
    return (vals[0], vals[1]) if len(vals) >= 2 else None


def _read_xy(context, calls: list) -> "tuple[float, float] | None":
    """当前 FolMe XY;读不到返回 None(**不是** (0,0))。"""
    rec = context.safe_call("FolMe_XYPosGet", 1)
    calls.append(rec)
    if rec.error:
        return None
    return _first_two_floats(rec.return_value)


class MoveToXY(BaseSkill):
    """Move tip to XY position using Follow Me."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MoveToXY",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="用 Follow Me 把针尖移到指定的 XY 位置。",
            parameters=[
                ParameterSpec(
                    name="x_m",
                    type="float",
                    description="目标 X 位置，单位米",
                    unit="m",
                    required=True,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="y_m",
                    type="float",
                    description="目标 Y 位置，单位米",
                    unit="m",
                    required=True,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="wait",
                    type="bool",
                    description="返回前先等待移动完成",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="coord_epoch",
                    type="int",
                    description=(
                        "目标 x/y 所属的坐标代次。**只有**当这对坐标来自某个"
                        "自带代次的存储源（地图 marker、扫描计划、已保存的位点）"
                        "时才传它。传了之后，若期间发生过横向粗动（lateral coarse "
                        "move），这次移动就会"
                        "被**拒绝** —— 那些米数如今指向的是另一块表面，而两个代次"
                        "之间没有任何换算关系。对于你刚从仪器回读出来的坐标请"
                        "**省略（OMIT）**它：它们按构造就是当前代次。"
                    ),
                    required=False,
                    # 刻意 None(同 full_scan.line_time_s / imaging.angle_deg 那两条
                    # 注释):wrap_skill 会把 ParameterSpec.default 物化进 pydantic
                    # 字段,写个具体数字就再也分不出「没传」和「显式传了它」——
                    # 而这里「没传」必须逐字节保持旧行为(修针 relocate 用的是动作
                    # 瞬间读回的坐标,天然新鲜,不能被误伤)。
                    default=None,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["navigation", "move", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        x_m = params["x_m"]
        y_m = params["y_m"]
        wait = params.get("wait", True)

        # ── 坐标代次核对(explicit-only)───────────────────────────────────
        #
        # 一次横向粗动之后,同样的 (x, y) 指的是另一片表面。调用方**说得出**这对
        # 坐标属于哪一代时(取自地图标记 / 计划 / 存下来的站点),就在这里核对;
        # 对不上 ⇒ 拒绝,**不换算**(跨代次换算在本仓有意不存在,见
        # ``io/coarse_map.py`` 的「WHY STEPS, NOT METRES」)。
        #
        # 不传 ⇒ 一个字节都不变。省略是常态而不是疏忽:修针 relocate 用的是动作
        # 瞬间从仪器读回的坐标,天然属于当前代次,拿一道查不到就报警的闸去卡它
        # 只会把好路径也堵上。
        if params.get("coord_epoch") is not None:
            from mast.core import coord_epoch as _ce
            v = _ce.verify(params["coord_epoch"], what="这对目标坐标")
            if v.stale:
                return SkillResult(
                    skill_name="MoveToXY", success=False,
                    error=v.message,
                    data={"refusal_code": _ce.REFUSAL_CODE,
                          "requested_coord_epoch": v.stamped,
                          "current_coord_epoch": v.current,
                          "moved": False},
                )

        # FolMe_XYPosSet(wait=1) 会在控制器端等待运动完成，低速位移可能超过 socket 超时。
        # 改用 wait=0 发起，然后通过 FolMe_XYPosGet 有界轮询，保持通信连接可用并报告进度。
        record = context.safe_call("FolMe_XYPosSet", x_m, y_m, 0)
        calls = [record]
        if record.error:
            return SkillResult(
                skill_name="MoveToXY",
                success=False,
                error=record.error,
                nanonis_calls=calls,
            )
        if not wait:
            return SkillResult(
                skill_name="MoveToXY", success=True,
                data={"x_m": x_m, "y_m": y_m, "wait": False,
                      "arrived": None,
                      "note": "已下发移动指令,未等待到位(wait=false)。"},
                nanonis_calls=calls,
            )

        # 轮询到位。容差取 0.5 nm:FolMe 的落点精度远好于此,而更严的容差会在
        # 压电蠕变上空转。超时不谎报成功,如实说「下发了但没看到它到」。
        import time as _t

        tol_m = 0.5e-9
        # 移动预算从当前距离和速度派生，并保留余量：距离/速度 × 1.5 + 10 s。
        # 固定短超时可能在低速移动尚未完成时到期，固定长超时又会延迟真正失败的反馈。
        # 速度读不到时使用有界兜底，并明确说明预算来源未知，不能声称已按实际速度计算。
        speed = None
        rec_sp = context.safe_call("FolMe_SpeedGet")
        calls.append(rec_sp)
        if not rec_sp.error:
            _sp = _first_two_floats(rec_sp.return_value)
            # FolMe_SpeedGet 回 (speed, custom) —— 头一个就是速度。
            if _sp and _sp[0] > 0:
                speed = _sp[0]

        start = _read_xy(context, calls)
        budget = _FALLBACK_MOVE_TIMEOUT_S
        derived = False
        if speed and start is not None:
            dist = ((float(x_m) - start[0]) ** 2 + (float(y_m) - start[1]) ** 2) ** 0.5
            budget = max(5.0, dist / speed * 1.5 + 10.0)
            derived = True
        deadline = _t.monotonic() + budget
        last: "tuple[float, float] | None" = None
        frozen = 0
        while _t.monotonic() < deadline:
            rec = context.safe_call("FolMe_XYPosGet", 1)
            calls.append(rec)
            if rec.error:
                # 位置读不到 ⇒ **不知道到没到**,不是「没到」。继续轮询;
                # 真断链的话下一拍的 safe_call 会带着断链错误回来。
                _t.sleep(0.2)
                continue
            pos = _first_two_floats(rec.return_value)
            if pos is not None:
                # 位置**一位都没变**的连续次数。压电走到限位就会停在一个
                # 逐位相同的读数上,而「还在走」的读数每拍都在变 ——
                # 这一个计数器就是两者唯一的区别。见下面的超时文案。
                if last is not None and pos[0] == last[0] and pos[1] == last[1]:
                    frozen += 1
                else:
                    frozen = 0
                last = pos
                if (abs(pos[0] - float(x_m)) <= tol_m
                        and abs(pos[1] - float(y_m)) <= tol_m):
                    return SkillResult(
                        skill_name="MoveToXY", success=True,
                        data={"x_m": pos[0], "y_m": pos[1], "wait": True,
                              "arrived": True,
                              "requested_x_m": x_m, "requested_y_m": y_m},
                        nanonis_calls=calls,
                    )
            _t.sleep(0.2)

        return SkillResult(
            skill_name="MoveToXY", success=False,
            error=(f"移动指令已下发,但 {budget:.0f} s 内没看到针尖到位"
                   + (f"(预算由距离 ÷ FolMe 速度 {speed * 1e9:.1f} nm/s 派生)"
                      if derived else
                      "(**读不到 FolMe 速度,这是兜底值不是算出来的**)")
                   + f"。目标 ({x_m:.3e}, {y_m:.3e}) m"
                   + (f",最后读到 ({last[0]:.3e}, {last[1]:.3e}) m" if last
                      else ",而且**一次位置都没读到**")
                   + (
                       # 连续多个采样点位置完全不变时，区分停滞与仍在移动。
                       # 目标不可达或输出受限可能导致停滞，不能继续声称针尖正在接近目标。
                       # 返回明确原因并让调用方核对当前范围，而不是等待不会发生的到达事件。
                       "。**针尖停住了,不是还在走**:位置连续 %d 拍逐位不变。"
                       "最可能是目标越过了压电范围 —— 先读 GetPiezoConfig 的 range"
                       "(半程 = range/2),确认目标在范围内;这台仪器的实际半程"
                       "可能小于 config 里的 xy_max_m。**重试不会有帮助。**" % frozen
                       if frozen >= 5 else
                       "。**这不是「没动」**:指令已经发出去了,针尖可能仍在移动中 ——"
                       "下一步之前先读一次位置,别假定它还在原处。"
                   )),
            data={"requested_x_m": x_m, "requested_y_m": y_m,
                  # 给**代码**看的那一半:光有文案,调用方没法分支。
                  "stalled": bool(frozen >= 5),
                  "stall_polls": int(frozen),
                  "last_x_m": last[0] if last else None,
                  "last_y_m": last[1] if last else None,
                  "arrived": False,
                  "budget_s": budget, "budget_derived": derived,
                  "folme_speed_m_s": speed},
            nanonis_calls=calls,
        )
