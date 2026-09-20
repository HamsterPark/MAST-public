"""自主度：**谁能点这个头**，而不是**什么可以做**。

## 为什么需要这一层

长期目标是全自动实验室——科学目标自己生成、方案自己起草、执行自己看护。
在那条路上，「必须有一个人按下批准键」是一道**结构性的天花板**：它不是一条
安全约束，它是一条在场约束。夜里三点没有人在场，于是什么也做不成。

但直接删掉那道门是错的另一头。本仓真正拦住过事故的是**包络**，不是**在场**：
参数超界当场拒绝（拒绝不夹紧）、DANGEROUS 技能的 SafetyGate、速度闸、预算闸、
三态求值里那句「读不到 ≠ 通过」。这些在有人没人的时候同样有效。

所以这里把两件被捆在一起的事拆开：

* **什么可以做** —— 由包络决定，三档下**完全一样**，一个数都不放宽；
* **谁能点头** —— 由自主度决定，可配。

这条分界也是为什么本模块不做任何数值判断：它只回答「这个动作，此刻，由谁
点头算数」。任何形如「autonomous 档下把上限放宽一点」的改动都是在越过这条
分界——那种放宽属于包络，要去 validator 那边光明正大地改，并留下依据。

## 三档

``attended``    人批人 ack。**默认**，也是改名前的既有行为。
``supervised``  agent 可批可 ack，但点火**延迟一个撤销窗**（默认 10 分钟），
                窗口内任何 abort 意图都能把它撤回。人不必在场，但来得及后悔。
``autonomous``  包络内即批即跑，全程留痕可回放。

生效值 = ``min(全局设置, spec 声明的上限)``——**取更严的那个**。模板作者比
设置页更懂这份流程能放到多松：一份要在 4 K 下扎针的模板可以把自己钉死在
``attended``，而设置页开到 ``autonomous`` 也不能把它拉上去。

## 有意留在 attended 的两件事

1. **需要人动手的等待**（换样品、开某个阀门）。agent 说「我 ack 了」在物理上
   毫无意义——这一档由 :class:`~mast.conduct.spec.WaitSpec` 的
   ``agent_ackable`` 逐条声明，默认 False。
2. **越权的升级建议**。L2 诊断席给出的处置若不在模板声明的
   ``allowed_escalations`` 里，三档一律转人。包络之外没有自主可言。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AUTONOMY_LEVELS", "DEFAULT_LEVEL", "DEFAULT_IGNITION_DELAY_S",
    "rank", "stricter_of", "normalise", "from_code", "to_code",
    "ApprovalVerdict", "who_may_approve", "describe", "ignition_payload",
]

#: 三档，**从严到松**。顺序即是 :func:`rank` 的定义，别重排。
AUTONOMY_LEVELS: tuple[str, ...] = ("attended", "supervised", "autonomous")

#: 出厂默认。与改名之前逐字节等价的那一档 —— 一个新能力不该自带打开的开关。
DEFAULT_LEVEL = "attended"

#: ``supervised`` 档下，批准到点火之间的撤销窗（秒）。
#:
#: 十分钟是这样来的：它要长到人从别的房间走回来、或者看一眼手机上的通知还
#: 来得及说「等等」，又要短到不至于让「批了但没动」变成一种常态——后者会让
#: 用户学会忽略这条通知，而一条被忽略的通知等于没有。
DEFAULT_IGNITION_DELAY_S = 600.0


def from_code(code: float | int | None) -> str:
    """把设置里的数值档位翻成名字。

    设置块的契约是 ``dict[str, float]``（旋钮目录、边界表、前端渲染器全都建在
    这个形状上），所以自主度在**设置层**用 0/1/2 表示，在**代码层**一律用名字。
    翻译只有这一个函数，两边各说各的话是「一个记号承担两种结构」的开头。

    越界或读不到 ⇒ 最严档。理由同 :func:`rank`。
    """
    try:
        i = int(round(float(code)))
    except (TypeError, ValueError):
        return DEFAULT_LEVEL
    if 0 <= i < len(AUTONOMY_LEVELS):
        return AUTONOMY_LEVELS[i]
    return DEFAULT_LEVEL


def to_code(level: str | None) -> float:
    """名字 → 设置里的数值档位。"""
    return float(rank(normalise(level)))


def rank(level: str) -> int:
    """档位的严格程度序号（0 = 最严）。认不出来的名字按最严处理。

    「认不出来 ⇒ 最严」是这里唯一正确的方向：一个拼错的档位名如果被当成
    ``autonomous``，那台仪器就会在没人看着的时候按一个谁也没批准过的策略动。
    """
    try:
        return AUTONOMY_LEVELS.index(str(level))
    except ValueError:
        return 0


def normalise(level: str | None) -> str:
    """把外面传进来的东西收口成闭集里的一个名字（认不出来 ⇒ 最严档）。"""
    s = str(level or "").strip().lower()
    return s if s in AUTONOMY_LEVELS else DEFAULT_LEVEL


def stricter_of(a: str | None, b: str | None) -> str:
    """两个档位取更严的那个。"""
    na, nb = normalise(a), normalise(b)
    return na if rank(na) <= rank(nb) else nb


@dataclass(frozen=True)
class ApprovalVerdict:
    """「这次批准算不算数」的裁决 —— 连同为什么。

    ``allowed`` 为假时 ``reason`` 必须能直接念给调用方听：一个 agent 拿到
    「不行」而不知道为什么，只会换一种说法再试一次。
    """

    allowed: bool
    level: str
    #: 批准生效到实际点火之间的等待（秒）。0 = 立刻。
    ignition_delay_s: float = 0.0
    reason: str = ""

    @property
    def deferred(self) -> bool:
        return self.allowed and self.ignition_delay_s > 0.0


def who_may_approve(level: str | None, *, by: str,
                    ignition_delay_s: float | None = None) -> ApprovalVerdict:
    """在 *level* 档下，署名为 *by* 的这次批准算不算数。

    ``by`` 的形状约定：``agent:<name>`` 表示一次模型驱动的批准；其余一律当成
    人（用户在面板上按的、或者带着自己名字调 API 的）。**这里不去验证身份**
    ——认证是另一层的事，这一层只按署名分流，而署名会进审计流。
    """
    lv = normalise(level)
    is_agent = str(by or "").strip().lower().startswith("agent:")

    if not is_agent:
        # 人批准在任何一档都算数。自主度是往上放开的，不是往回收紧的：
        # autonomous 档下人依然可以随时自己按那个按钮。
        return ApprovalVerdict(True, lv, 0.0, "")

    if lv == "attended":
        return ApprovalVerdict(
            False, lv, 0.0,
            "当前自主度为 attended：批准要由人来点。"
            "要让 agent 批，把 conduct 自主度调到 supervised（批后有撤销窗）"
            "或 autonomous —— 这两档下包络一个数都不放宽，变的只是谁点头。")

    if lv == "supervised":
        delay = (DEFAULT_IGNITION_DELAY_S if ignition_delay_s is None
                 else max(0.0, float(ignition_delay_s)))
        return ApprovalVerdict(
            True, lv, delay,
            f"agent 批准已记下，{int(delay)} 秒后点火；这段时间里 abort 能把它撤回。")

    return ApprovalVerdict(True, lv, 0.0, "")


def ignition_payload(verdict: ApprovalVerdict, now_epoch: float) -> dict:
    """批准事件里那几个字段：谁批的、什么档、窗口多长、**什么时候可以点火**。

    2026-08-27 补。在它之前，:attr:`ApprovalVerdict.deferred` 与
    ``ignition_delay_s`` **只有生产方没有消费方**：两条 approve 路径都只读
    ``verdict.allowed``，Director 的 ``_dispatch`` 在 ``approved`` 分支无条件
    采纳。也就是说 ``supervised`` 在行为上等于 ``autonomous`` —— 而 403 的
    文案还在推荐它（「批后有撤销窗」）。撤销窗是这一档存在的全部理由。

    ``ignite_at`` 是**绝对时刻**，不是剩余秒数：进程在窗口中间重启是常态
    （一份 conduct 的时间尺度是 hour~day），剩余秒数会在每次重启时从头再数，
    于是一个十分钟的窗可以无限延长。绝对时刻跨重启自然延续，
    ``reconcile_after_restart`` 一个字都不用改。

    人批 / 窗口为 0 时 ``ignite_at == now_epoch`` —— 不是省略这个键：省略会让
    「立刻点火」和「旧格式的事件」长得一样，而后者要按立刻处理。
    """
    delay = max(0.0, float(getattr(verdict, "ignition_delay_s", 0.0) or 0.0))
    return {
        "autonomy": getattr(verdict, "level", "") or "",
        "ignition_delay_s": delay,
        "deferred": bool(getattr(verdict, "deferred", False)),
        "ignite_at": float(now_epoch) + delay,
    }


def describe(level: str | None) -> str:
    """给面板/日志的一句人话。"""
    lv = normalise(level)
    return {
        "attended": "有人值守：批准与 ack 都要人来点",
        "supervised": "半自主：agent 可批，点火前留一个撤销窗",
        "autonomous": "自主：包络内即批即跑，全程留痕",
    }[lv]
