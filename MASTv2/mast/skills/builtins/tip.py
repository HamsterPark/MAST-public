"""Tip safety skills: retract and emergency retract.

vendored from v1 mast/skills/builtins/tip.py 2026-04-23.
2 skills: SafeRetract, EmergencyRetract.
"""

from __future__ import annotations

import time

from mast.core.tip_park import NOT_PARKED, PARKED, tip_parked
from mast.core.types import (
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill


class SafeRetract(BaseSkill):
    """Safely retract the tip (withdraw), then **confirm it actually parked**.

    在这之前这个技能是「发出即成功」::

        record = context.safe_call("ZCtrl_Withdraw", 0, 1)
        return SkillResult(..., data={"retracted": True})

    ``(0, 1)`` = 不等待、1 ms 超时 —— 命令一发出就报 ``retracted: True``,而压电
    这时还在往上爬。它验证的是**发过命令**,不是**动作生效**。而它恰恰是 planner
    提示词里「出事就退针」的首选技能。

    现在:下发 → 有界轮询 :func:`mast.core.tip_park.tip_parked`(判据在那里,不在
    这里)→ ``data["retracted"]`` **三态**:

    * ``True``  —— 确认到位(Z 反馈断开 + Z 停在收回端)。
    * ``False`` —— 预算内**读得到状态、状态说没到位**。这是一个确定的否定。
    * ``None``  —— 判不了(读不到 / 本机没声明 ``z_extend_sign``)。
      **「没查」永远不折叠成 ``True``。**

    ``success`` 的不对称,是有意的
    =============================
    * ``retracted is False`` ⇒ ``success=False``。读数是确定的:发了退针命令,
      预算走完针还没到收回端。「命令发出去了」不等于「达标」—— 这正是 2026-07-10
      现场那一族假成功。文案里两种可能(还在爬 / 根本没生效)一起说,**不替仪器
      断定是哪一种**,并附上实测差值。慢机器可以把 ``_confirm_budget_s`` 调大。
    * ``retracted is None`` ⇒ ``success=True`` + 一句大声的「已下发未确认」。
      「读不到」被当成「出故障」是本仓在案的旧错(``a_stop_with_no_release``:
      读不到被判成出故障,于是退了针)。判据链断了不构成「退针失败」这个断言,
      调用方要的区分在 ``retracted`` 里,不在 ``success`` 里。

    轮询**不看 abort**:退针本身在 abort 之后也是放行的(``execution_context``
    的白名单),而这几秒只读的确认恰恰是在确认那个 abort 想要的动作 —— 中途放弃
    只会丢掉证据,命令早已发出。预算有界(默认 5 s)是这条成立的前提。
    """

    #: 确认预算。Withdraw 是快动作;超时不代表失败,代表「没确认到」。
    #: 测试与慢机器可覆盖(类属性,不进参数表 —— 加参数是三处联动的事)。
    _confirm_budget_s = 5.0
    _confirm_poll_s = 0.25

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SafeRetract",
            version="2.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "安全退针（withdraw），随后**回读硬件确认它确实停在了收回位**"
                "（Z feedback 断开 + Z 停在收回端）。`retracted` 是三态："
                "True = 已确认，False = 回读到了、但仍未到位，"
                "None = 判不了。绝不臆断为 True。"
            ),
            # 下发 + 最多 _confirm_budget_s 的回读确认。常见情形 1–2 s,
            # 这里给的是有界的**上界**(以前的 2.0 是「只发命令」那个版本的数)。
            estimated_duration_s=7.0,
            composition_level=0,
            tags=["tip", "safety", "retract"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # ZCtrl_Withdraw(wait=0, timeout=1) —— 不阻塞地下发,确认交给下面的回读。
        # (``WithdrawTip`` 走的是另一半:``(1, -1)`` 阻塞到仪器说走完。两个形状
        # 都在用,该收敛成一个 —— 收敛前别再抄第三份。)
        record = context.safe_call("ZCtrl_Withdraw", 0, 1)
        if record.error:
            return SkillResult(
                skill_name="SafeRetract",
                success=False,
                error=record.error,
                data={"retracted": None,
                      "note": "退针命令没发出去(TCP 失败),更谈不上到位。"},
                nanonis_calls=[record],
            )

        t0 = time.monotonic()
        # do-while:预算为 0 也**至少读一次**。一次都不读就等于把「没查」当答案。
        while True:
            verdict = tip_parked(context)
            if verdict.state == PARKED:
                break
            # 配置缺口(如本机没声明 z_extend_sign)等多久都不会变 —— 立刻收工,
            # 免得报出一个看起来像「超时」其实是「没人填过」的结论。
            if not verdict.retry_useful:
                break
            if (time.monotonic() - t0) >= self._confirm_budget_s:
                break
            time.sleep(self._confirm_poll_s)

        waited_s = time.monotonic() - t0
        park = {
            "state": verdict.state,
            "reason": verdict.reason,
            "evidence": verdict.evidence(),
            "feedback_on": verdict.feedback_on,
            "module_status": verdict.module_status,
            "z_m": verdict.z_m,
            "rail_m": verdict.rail_m,
            "gap_m": verdict.gap_m,
            "tolerance_m": verdict.tolerance_m,
            "unreadable": list(verdict.unreadable),
            "undeclared": list(verdict.undeclared),
            "read_at": verdict.read_at,
        }

        if verdict.state == PARKED:
            return SkillResult(
                skill_name="SafeRetract",
                success=True,
                data={"retracted": True, "park": park,
                      "confirm_waited_s": waited_s},
                summary=f"已确认退针到位（{verdict.evidence()}）",
                nanonis_calls=[record],
            )

        if verdict.state == NOT_PARKED:
            msg = (f"退针已下发,但 {waited_s:.1f} s 内没有确认到位:{verdict.reason}"
                   f"（{verdict.evidence()}）。可能还在走,也可能这条命令没生效 —— "
                   f"两种都没被排除,请复核后再做任何依赖「针已退开」的动作。")
            return SkillResult(
                skill_name="SafeRetract",
                success=False,
                error=msg,
                data={"retracted": False, "park": park,
                      "confirm_waited_s": waited_s},
                summary=msg,
                nanonis_calls=[record],
            )

        msg = (f"退针**已下发未确认**:{verdict.reason}"
               f"（{verdict.evidence()}）。命令发出去了,但判不了它到没到位 —— "
               f"「判不了」既不是「退到了」也不是「没退到」。")
        return SkillResult(
            skill_name="SafeRetract",
            success=True,
            data={"retracted": None, "park": park,
                  "confirm_waited_s": waited_s},
            summary=msg,
            nanonis_calls=[record],
        )


class EmergencyRetract(BaseSkill):
    """Emergency tip retraction using the emergency port.

    Sequence:
    1. Ensure Z controller is ON
    2. Stop any running scan
    3. Withdraw tip
    All calls use the emergency connection role.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="EmergencyRetract",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # AUTO (2026-06-11): an emergency retract moves the tip AWAY from the
            # sample — the safe direction. Gating an emergency safety action
            # behind human approval was backwards (it could delay the very
            # retract that protects the tip). It must execute immediately.
            safety_level=SafetyLevel.AUTO,
            description="经专用应急端口紧急退针。",
            estimated_duration_s=3.0,
            composition_level=1,
            tags=["tip", "safety", "emergency", "retract"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []

        # 1. Ensure Z controller is ON
        rec_z_on = context.safe_call("ZCtrl_OnOffSet", 1, role="emergency")
        calls.append(rec_z_on)

        # 2. Stop any running scan
        rec_stop = context.safe_call("Scan_Action", 1, 0, role="emergency")
        calls.append(rec_stop)

        # 3. Withdraw tip
        rec_withdraw = context.safe_call("ZCtrl_Withdraw", 0, 1, role="emergency")
        calls.append(rec_withdraw)

        # Check for errors in any step
        errors = [r.error for r in calls if r.error]
        if errors:
            return SkillResult(
                skill_name="EmergencyRetract",
                success=False,
                error=f"Emergency retract errors: {'; '.join(errors)}",
                nanonis_calls=calls,
            )
        return SkillResult(
            skill_name="EmergencyRetract",
            success=True,
            data={"retracted": True, "emergency": True},
            nanonis_calls=calls,
        )
