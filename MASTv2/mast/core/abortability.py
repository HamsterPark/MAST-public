"""可中止性 —— **一个动作能不能被停下来，是那个动作的属性，不是一句形容词**。

## 为什么要有这个模块

「有风险，但可以随时中止」这种句式散在各处的 description 和 docstring 里，而它
在仓里至少有一处是**假话**：``WaitForScanEndBlocking`` 的 description 写着
「中止 still works（the emergency port is a separate socket）」—— 急停口确实是另一
条 socket，但**用户的软停今天不往那条 socket 上发任何东西**（``runtime.abort_run``
只 ``ev.set()``；只有 ``emergency_stop()`` 会发 ``Motor_StopMove`` / ``Scan_Action``
/ ``ZCtrl_Withdraw``）。于是那句话描述的是一个**没人走的**通道。

一句话散着写就会这样：它在写的时候是关于「理论上存在的能力」的，读的人当成了
「按下去就会发生的事」。所以这里把它变成**一张表**：一个动作属于哪一类、
它到底能被什么打断、以及那件事**今天有没有人在做**。

## 三类

``POLLED``
    动作自己在轮询停止事件。停止在下一次 poll 生效（延迟 = poll 间隔）。
    ``WaitScanComplete``、``AutoApproach`` 的等待相、``_z_settle`` 的稳定循环都是这类。

``BETWEEN_STEPS``
    动作本身不查，但它被切成了小步，执行器在**步与步之间**查。停止延迟 = 一步的
    时长。``relocate_coarse_xy`` 的分块粗动是这类（默认一块 50 步）。

``BLOCKING_HELD``
    调用线程整个停在**一条固件调用**里，它没有检查点，自己查不了任何东西。
    ⚠️ **这不等于「不可逆」** —— 见下。

## ``BLOCKING_HELD`` 的准确说法（这条曾经被说反）

说「``Motor_StartMove`` 发出去就在固件里跑完，内部没有检查点，所以无法中止」——
**前半句对，结论错**。nanonis 的文档写着 ``Motor_StartMove`` 的 wait 标志是
「returns when the motor reaches its destination **or the movement stops**」，而
``Motor_StopMove`` 存在、无参数、并且在 ``execution_context._ABORT_SAFE_WRITES``
里是**无条件放行**的。

准确的说法是：

  **调用线程**无法自查（它parked 在 ``pool.safe_call`` 里，还占着那个 role 的锁），
  但**动作本身**可以被**另一条 socket 上的停止动词**打断。急停口
  （``pool.urgent_call``）就是为这件事存在的，好让停止命令不用排在卡住的调用后面。

所以缺的不是「可中止性」，缺的是**有人去发那个动词**。这个区别很重要：
前者是物理约束（改不了），后者是我们没接线（能改）。把它们混成一句
「停不了」，就会让一件能修的事被当成不能修的事记下来。

⇒ 对这一类，正确的告知是**发起之前**说清楚两件事：
   ① 这一步一旦发出，**软停不会打断它**；
   ② 能打断它的是 ``interrupted_by(verb)`` 那个动词，而今天只有**紧急停止**会发。
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Abortability", "INTERRUPTED_BY", "interrupted_by", "describe",
           "SOFT_STOP_CAVEAT", "soft_stop_sends_hardware_verbs"]


class Abortability(str, Enum):
    POLLED = "polled"
    BETWEEN_STEPS = "between_steps"
    BLOCKING_HELD = "blocking_held"


# 阻塞型固件调用 → **能打断它的那个停止动词**。
#
# 这张表只收「调用线程会被 park 住、且确实存在一个停止动词」的那些。表里每一个
# value 都必须同时在 ``execution_context._ABORT_SAFE_WRITES`` 里（否则中止之后那个
# 停止动词自己会被 abort 闸门拒掉，收尾路径就把自己掐死了）——
# ``tests/v2/unit/core/test_abortability_table.py`` 钉着这条。
INTERRUPTED_BY: dict[str, str] = {
    "Motor_StartMove": "Motor_StopMove",
    "Scan_WaitEndOfScan": "Scan_Action",          # Scan_Action(1, 0) = stop
    "BiasSpectr_Start": "BiasSpectr_Stop",
    "ZSpectr_Start": "ZSpectr_Stop",
    "GenSwp_Start": "GenSwp_Stop",
    "HSSwp_Start": "HSSwp_Stop",
    "PLLFreqSwp_Start": "PLLFreqSwp_Stop",
    "PLLPhasSwp_Start": "PLLPhasSwp_Stop",
    "APRFGen_RFSwpStart": "APRFGen_SwpStop",
    "Pattern_ExpStart": "Pattern_ExpStop",
    "AutoApproach_Open": "AutoApproach_OnOffSet",  # (0) = off
}


def interrupted_by(verb: str) -> str | None:
    """打断 *verb* 的那个停止动词；``None`` = 这条不是阻塞型 / 没有停止动词。"""
    return INTERRUPTED_BY.get(str(verb or ""))


# 软停（用户点「停止」/ chat abort / abort_run）今天**只置位事件，不下发任何硬件
# 动词**。只有紧急停止会下发。这个常量是那句话的单一真源 —— 它同时被
# `/chat/abort` 的回包和技能说明消费，改了要一起改，而不是各写各的。
SOFT_STOP_CAVEAT = (
    "停止信号已送达：仪器写命令即刻被拒（只放行读取与停止/退针类动作），正在轮询的"
    "等待循环会在下一次 poll 退出。**但软停不会替你下发任何硬件停止动词** —— 已经"
    "发出、正阻塞在主 socket 上的那一条固件动作（如 Motor_StartMove 的当前一段、正在"
    "进行的一次 BiasSpectr 扫）要靠 Motor_StopMove / BiasSpectr_Stop 这类动词从急停"
    "socket 上打断，而今天只有**紧急停止**会发它们。需要立刻让针停下，请用紧急停止。"
)


def soft_stop_sends_hardware_verbs() -> bool:
    """软停会不会自己下发硬件停止动词。

    今天是 ``False``（``runtime.abort_run`` / chat abort 都只 ``ev.set()``）。
    如果哪天接上了，改这里 **一处**，所有告知文本跟着变 —— 而不是去 grep 十几处
    description 里那句「可以随时中止」。
    """
    return False


_KIND_TEXT = {
    Abortability.POLLED: (
        "**可中止**：这个动作自己在查停止事件，你按下停止后它在下一次 poll 就退出。"),
    Abortability.BETWEEN_STEPS: (
        "**只能在步与步之间中止**：动作被切成小步，停止在当前这一步跑完后生效 —— "
        "所以延迟等于一步的时长，不是零。"),
    Abortability.BLOCKING_HELD: (
        "⚠️ **一旦发出，软停打断不了它**：调用线程会整个停在这条固件调用里，"
        "它没有检查点，自己查不了停止事件。"),
}


def describe(kind: Abortability, *, verb: str | None = None) -> str:
    """给用户/调用方看的一句话。``BLOCKING_HELD`` 会带上那个能打断它的动词。

    **发起之前**说，不是失败之后说 —— 这一类的全部代价都在「发出去」那一刻确定。
    """
    text = _KIND_TEXT[Abortability(kind)]
    if Abortability(kind) is not Abortability.BLOCKING_HELD:
        return text
    stop_verb = interrupted_by(verb) if verb else None
    if stop_verb:
        text += (f"\n\n它**不是不可逆的**：`{stop_verb}` 能从急停 socket 上把它打断。"
                 "但软停不会替你发那个动词 —— 今天只有**紧急停止**会发。")
    else:
        text += "\n\n仓里没有登记能打断它的停止动词，请当作发出即跑完来规划。"
    return text
