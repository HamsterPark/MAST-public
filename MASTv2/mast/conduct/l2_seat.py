"""L2 **值守席** —— 一个阶段栽了之后，谁来看一眼。

设计：``campaign_director_design.md`` §3-6 的三级执行体第三级。L0 是脚本步，
L1 是闸门（:mod:`mast.conduct.llm_seat`），L2 是这里：**只在一个阶段已经失败
之后**被叫醒一次，看一眼现场，给一个闭集里的处置。

## 它与 L1 的区别不是「更聪明」，是**手里的东西不一样**

L1 闸门拿到的是一个证据包（几个标量），出的是 ``pass`` / ``fail`` / ``detour``
之一。它便宜、频繁、结构极窄。

L2 拿到的是一个**只读的诊断能力**：它可以自己去看几张图、读几个通道、查一下
账本，然后说「再试一次」还是「去修针」还是「跳过这一段」。代价是一整次 agent
run，所以它每个阶段每个证据代次**最多醒一次**（次数由 spec 说了算，不硬编码）。

## 「只读」是结构，不是嘱咐

这一席的技能面来自一个**过滤后的注册表视图**：只有 ``READ`` / ``ANALYSIS``
两类进得去。这一条要落在**注册表**上而不是提示词或工具清单上，因为
``ExecutionContext.run`` 每一个子步都按名字回注册表查类——所以即使模型凭空
说出一个写技能的名字，执行层也会「查无此技能」而拒绝。

工具清单决定模型**看得见**什么；注册表视图决定**能发生**什么。两者只有后者
是安全边界。

三层叠起来（每一层单独可测）：

1. **注册表视图** —— 只读类才在视图里；
2. **工具面** —— 自建最小工具集，不复用 IC 的（IC 的提示词教的是操作仪器，
   责任面错了），不给 handoff、不给 meta 工具（后者含写面）；
3. **仲裁面** —— READ/ANALYSIS 本来就不取仪器令牌，所以这一席在结构上不可能
   挡住主流程，也不可能与用户抢仪器。

## 它的「权」在哪里

不在于它能不能动手——它不动手。在于 ``StageSpec.allowed_escalations``：模板
作者写进那个集合的处置，L2 说了**算**，Director 直接执行，不再叫人。写不进去
的（比如一份从没声明过 ``abort`` 的模板）一律转人。

这就是「包络内自主」的形状：**放开的边界由模板显式声明，而不是由这一席自己
主张**。越权不是拒绝一次调用那么简单——它意味着模型正在提议一件没人授权过的
事，那件事值得一个人看一眼。

## 判不了的三种来源

与 L1 逐字相同（见 :mod:`mast.conduct.llm_seat` 的模块 docstring）：模型弃权
是一次判决；返回值不在闭集里、调用炸了、超时——都是**判决器坏了**，一律
:class:`~mast.conduct.llm_seat.SeatUnavailable`，Director 收到它就转人。
**不在这里兜一个默认处置**：一个「provider 500 于是跳过这一段」的兜底，会以
「诊断过了」的形式留在记录里。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from mast.conduct.llm_seat import (
    DEFAULT_DEADLINE_S,
    SeatUnavailable,
    _call_bounded,
    cost_source,
)

logger = logging.getLogger(__name__)

__all__ = [
    "L2_DEADLINE_S", "READONLY_CATEGORIES", "EscalationContext",
    "EscalationAdvice", "FilteredRegistry", "readonly_registry_view",
    "make_escalation_advisor",
]

#: L2 的墙钟上限。比 L1 的 60 s 宽得多：这一席要真的去看几张图、跑几个分析，
#: 而它发生在**阶段已经失败之后**——那时没有什么正在推进，多等几分钟不会
#: 拖住任何东西。上限仍然要有：一次不回来的诊断不该把 conduct 永远停在
#: 「正在诊断」上，那种停法比停下来问人更难被发现。
L2_DEADLINE_S = 300.0

#: 进得了只读视图的技能类别。与 ``instrument_lock.needs_token`` 免令牌的那两类
#: **刻意是同一个集合**：一个不改变仪器状态的技能，既不需要令牌，也不该被这一
#: 席之外的理由拦住。两处若哪天分叉，说明其中一处对「什么叫只读」改了主意，
#: 那值得当场发现。
READONLY_CATEGORIES: frozenset[str] = frozenset({"READ", "ANALYSIS"})


@dataclass(frozen=True)
class EscalationContext:
    """交给 L2 的现场。**全部是已经发生的事实**，没有一个待填的数。"""

    conduct_id: str
    stage_id: str
    #: 失败的那句话（Director 写的，不是模型写的）
    why: str
    #: 这一阶段这个证据代次里，之前已经试过几次
    attempts: int
    evidence_epoch: int
    #: 模板允许的处置。**这就是它的权限边界**，逐字交给模型看。
    allowed: tuple[str, ...] = ()
    #: 最近若干条事件（人读摘要），给模型一个时间感
    recent: tuple[str, ...] = ()
    #: 额外的现场标量（闸门证据包等）
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EscalationAdvice:
    """一次诊断的产出。``route`` 一定在闭集里，且一定在 ``allowed`` 里。"""

    route: str
    reason: str = ""
    #: 模型看过哪些东西（技能名），进审计流。
    looked_at: tuple[str, ...] = ()


class FilteredRegistry:
    """注册表的**只读视图**。

    包一层而不是复制一份：注册表在运行期可以热更新（技能覆盖层、composite 热
    注册），复制会让这一席看到一张过期的表，而「看到的和能跑的不是同一张表」
    正是本仓踩过的形状。

    这一层刻意**只实现读**：``register`` / ``unregister`` 在这里不存在，所以
    一个拿到视图的调用方连「往里塞一个写技能」这条路都没有。
    """

    def __init__(self, inner: Any,
                 categories: frozenset[str] = READONLY_CATEGORIES) -> None:
        self._inner = inner
        self._categories = frozenset(categories)

    # ── 判据 ─────────────────────────────────────────────────────
    def _is_readonly(self, meta: Any) -> bool:
        cat = getattr(meta, "category", None)
        name = getattr(cat, "name", None) or getattr(cat, "value", None) or cat
        return str(name).upper() in self._categories

    # ── 读接口（与 SkillRegistry 同名，鸭子替身可用）───────────────
    def get(self, name: str, version: str | None = None):
        """取技能类。**不是只读类 ⇒ 当作不存在。**

        「不存在」而不是「拒绝」是有意的：调用方拿到的是与「这台机器上没有这个
        技能」完全一样的结果，于是它不会去找一条绕过拒绝的路——没有什么可绕的。
        """
        try:
            cls = (self._inner.get(name, version) if version is not None
                   else self._inner.get(name))
        except TypeError:
            cls = self._inner.get(name)
        if cls is None:
            return None
        meta = getattr(cls, "metadata", None) or getattr(cls, "meta", None)
        if meta is None:
            # 读不到元数据 ⇒ 判不了它是不是只读 ⇒ 不给。
            # （「读不到」不是「安全」——这条在本仓记过一整天的账。）
            return None
        return cls if self._is_readonly(meta) else None

    def list_skills(self) -> list:
        out = []
        for item in self._inner.list_skills():
            meta = item if hasattr(item, "category") else getattr(item, "metadata", None)
            if meta is not None and self._is_readonly(meta):
                out.append(item)
        return out

    def __getattr__(self, item: str):
        # 显式黑名单：写接口一律不透传。别的读方法（describe / has / …）
        # 随内层演进，不必在这里逐个追。
        if item in {"register", "unregister", "discover", "clear",
                    "register_many", "replace"}:
            raise AttributeError(
                f"{item!r} 不在只读视图上 —— L2 诊断席不改注册表")
        return getattr(self._inner, item)


def readonly_registry_view(registry: Any) -> FilteredRegistry:
    """把一个注册表包成只读视图。"""
    return FilteredRegistry(registry)


def _closed_set_route(raw: Any, allowed: tuple[str, ...]) -> str:
    """把模型的回答收口成 ``allowed`` 里的一个名字，收不住就抛。

    收不住时抛 :class:`SeatUnavailable` 而不是回一个默认值：一个越权的建议
    被悄悄改成 ``wait_operator``，记录里会长得像「模型建议叫人」，而实际发生的
    是「模型建议了一件没人授权的事」。后者值得被看见。
    """
    route = str(raw or "").strip()
    if route not in allowed:
        raise SeatUnavailable(
            f"L2 给的处置 {route!r} 不在模板允许的 {allowed} 里 —— "
            f"这不是一次判决，是一个越权的提议，交给人看。")
    return route


def make_escalation_advisor(
    *,
    registry_provider: Callable[[], Any] | None = None,
    ask: Callable[[EscalationContext, Any], dict] | None = None,
    deadline_s: float = L2_DEADLINE_S,
) -> Callable[[EscalationContext], EscalationAdvice]:
    """建一个 L2 顾问。

    *ask* 是真正去问模型的那一步（默认实现见 :func:`_default_ask`）。把它做成
    参数是为了测试能注入一个不联网的替身——而不是让测试去 monkeypatch 一个
    模块级函数（那种替身会在真实调用路径变了之后继续绿着）。

    *registry_provider* 回一个**完整**注册表；这里自己包只读视图。让调用方
    传完整表、由这一层收窄，是因为「谁负责收窄」必须只有一个答案：如果调用方
    也可以传一个已经包好的视图，那么某一天有人传进来一个没包的，这里看不出
    区别。
    """
    _ask = ask or _default_ask

    def advise(ctx: EscalationContext) -> EscalationAdvice:
        if not ctx.allowed:
            raise SeatUnavailable(
                f"阶段 {ctx.stage_id} 没有声明任何 allowed_escalations —— "
                f"没有可授权的处置，L2 无事可做，交给人。")

        reg = None
        if registry_provider is not None:
            try:
                reg = readonly_registry_view(registry_provider())
            except Exception as exc:  # noqa: BLE001
                raise SeatUnavailable(f"取不到只读注册表: {exc}") from exc

        decision = _call_bounded(lambda: _ask(ctx, reg), deadline_s,
                                 f"L2 诊断（{ctx.stage_id}）")
        if not isinstance(decision, dict):
            raise SeatUnavailable(f"L2 返回的不是一个 dict: {type(decision)!r}")

        route = _closed_set_route(decision.get("route"), tuple(ctx.allowed))
        return EscalationAdvice(
            route=route,
            reason=str(decision.get("reason") or "")[:500],
            looked_at=tuple(str(x) for x in (decision.get("looked_at") or ()))[:20],
        )

    return advise


def _default_ask(ctx: EscalationContext, registry: Any) -> dict:
    """默认实现：起一次**只读**的 agent run，让它出一个闭集处置。

    这里刻意不复用 instrument_control 的图：IC 的系统提示词教的是怎么操作仪器，
    而这一席的责任是「看一眼、说一句」。把一个教人开车的提示词交给一个不该开车
    的席位，最好的情况是它一直在说自己做不到。
    """
    from mast.agents._shared.models import make_chat_model

    lines = [
        "你在给一台扫描隧道显微镜的自动实验流程做一次**只读**诊断。",
        f"阶段 {ctx.stage_id} 失败了：{ctx.why}",
        f"这一阶段在当前证据代次里已经试过 {ctx.attempts} 次。",
        "",
        "你能选的处置**只有**这些（其余一律不接受）：",
        *(f"  - {r}" for r in ctx.allowed),
        "",
        "你**不能**给任何数值（电压、电流、坐标、速度）——那些由模板与标定决定。",
        "你唯一的产出是上面列表里的一个名字，外加一句为什么。",
    ]
    if ctx.recent:
        lines += ["", "最近发生的事：", *(f"  - {r}" for r in ctx.recent[:10])]
    if ctx.evidence:
        lines += ["", f"现场读数：{ctx.evidence}"]
    lines += ["", '只回一个 JSON：{"route": "<上面之一>", "reason": "<一句话>"}']

    model = make_chat_model("orchestrator", max_tokens=1024, temperature=0.1,
                            usage_source=cost_source(ctx.conduct_id) + ":l2")
    resp = model.invoke("\n".join(lines))
    text = getattr(resp, "content", None) or str(resp)
    if isinstance(text, list):  # provider 回分段内容时
        text = " ".join(str(getattr(p, "text", p)) for p in text)

    import json
    import re

    m = re.search(r"\{.*\}", str(text), re.S)
    if not m:
        raise SeatUnavailable("L2 没有回出一个 JSON —— 判不了")
    try:
        obj = json.loads(m.group(0))
    except Exception as exc:  # noqa: BLE001
        raise SeatUnavailable(f"L2 的 JSON 解析不了: {exc}") from exc
    if not isinstance(obj, dict):
        raise SeatUnavailable("L2 回的 JSON 不是一个对象")
    return obj
