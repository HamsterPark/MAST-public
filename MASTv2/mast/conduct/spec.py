"""ConductSpec —— conduct 的**定义**(frozen dataclass,代码内模板)。

设计:``docs/v2/design/campaign_director_design.md`` §4.1-4.6。

## 定义 vs 进度

照 ``planning/plan_store.py`` 已经拆过一次的那条线:

* **Spec = 定义** —— 本模块。不可变、版本化,approve 时连 params 一起冻结。
* **State = 进度** —— :mod:`mast.conduct.store` 的 ``conducts`` 单行真源。
* **events = 审计流** —— 同上的 ``conduct_events``。

把这三样揉进一个结构,是 plan_store 修过的那个坑的入口(推进一个阶段就吃掉
上一份计划)。这里从第一行起就是分开的。

## 两道校验分居两处(刻意)

* **「这个值能不能存在」** —— 本模块的 ``__post_init__``。闭集成员、kind 与
  字段的搭配、not 只能有一个孩子……**模板 import 的那一刻就炸**,这是最早、
  最便宜的一道。
* **「这套装配立不立得住」** —— :mod:`mast.conduct.validator`。需要整份 spec
  甚至技能注册表才能判(bindings 命中上游 produces、wait 前必有确认式退针、
  DANGEROUS 步落在声明了 capability 的阶段),跑在 approve 之前。

分开的理由:第一道**没有语境也该拦**;第二道**没有语境就判不了**,而判不了
必须说出来(见 ``ValidationReport.checks_skipped``),不能默认放行。

## 数值从哪来

``params`` 里**只许有字面值或 bindings 引用,LLM 永不填数**(readiness survey
§4.2-6)。闸门的 llm 型也只选路不给数 —— 每条路由的动作参数在 spec 里预写好。
数值权限由结构约束：避免指数丢失、未经配置的工作点或模型生成坐标进入硬件动作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# ── 闭集 ──────────────────────────────────────────────────────────────────
#
# 每一个都是**闭集**:模板写错一个字,import 就炸,而不是运行到那一步才发现
# 一个谁也没定义过的分支被静默当成了默认。

#: 步的执行形态。
STEP_KINDS = ("skill", "composite", "analysis", "wait")

#: ``step_idx == -1`` = 停在阶段的 entry_gate 上,还没进第一步。
#:
#: 位置只有 ``(stage_idx, step_idx)`` 两个数是 §4.7 定下的,加一列要动 schema
#: 与恢复清算。定义住在这里而不是 ``director``,是因为 M4-a 之后**两个模块**
#: 要懂这个约定(director 推进位置,recovery 回退位置)—— 一个约定两份字面量
#: 是本仓「同一个动作 N 份实现」那族缺陷的入口。``director.AT_ENTRY_GATE``
#: 仍然导得出来,它 re-export 这一个。
AT_ENTRY_GATE = -1

#: 闸门形态。**能 rule 则 rule** —— llm 只在 rule 判不了时用。
GATE_KINDS = ("rule", "llm")

#: 闸门裁决(闭集)。``pass`` 是唯一「继续」,其余全是保守方向。
GATE_VERDICTS = ("pass", "fail", "detour", "wait_operator")

#: 保守裁决 = 除 ``pass`` 以外的全部。escape / 无人值守去向只能落在这里面 ——
#: 「spec 没写」≠「允许」,fail-open 方向在本设计里是被点名的陷阱(§10-5)。
CONSERVATIVE_VERDICTS = tuple(v for v in GATE_VERDICTS if v != "pass")

#: 等待形态。``both`` = 人的 ack **且** 物理条件(换样品就是它)。
WAIT_KINDS = ("operator", "condition", "both")

#: 阶段失败后的去向。
STAGE_FAIL_THEN = ("detour", "wait_operator", "abort", "skip", "escalate")

#: 绕道返回方式。
ON_RETURN = ("gate_recheck", "restart_stage", "resume_step")

#: 绕道触发源。
DETOUR_TRIGGERS = ("gate_verdict", "recovery_tip_fail", "escalation_approved")

#: L2 升级建议的闭集(允许集由 StageSpec 逐阶段声明,越权→等人)。
ESCALATIONS = ("continue_retry", "detour", "skip_stage", "abort", "wait_operator")

#: 证据来源。
EVIDENCE_SOURCES = ("verify_verdict", "frame_metrics", "monitor_events",
                    "temperature", "step_data")

#: 证据的代次要求。**只有一个合法值**,这是刻意的:跨代次证据在本仓一律作废
#: (``io/coarse_map.py`` 的「WHY STEPS, NOT METRES」+ 修复项 的强制点)。
#:
#: 畴指纹一类的**表面性质记录**是唯一已知的例外候选 —— 它是样品的事实,粗动
#: 不作废,但它的 verdict 需要在新 epoch 下重判(针尖变了,同一片表面测出的
#: 指纹可能不同)。想放宽这个集合的人,先答畴搜索设计的开放问题 3
#: (``docs/v2/design/`` 里的 S3 那份);在那之前多一个值就等于多一条把旧证据
#: 喂进闸门的路。
EVIDENCE_EPOCHS = ("current",)

#: 证据缺席的去向。**没有默认** —— 每个闸门必须自己声明缺证据时往哪走。
#: 「读不到」不是一个值,尤其不是「通过」。
EVIDENCE_MISSING = ("fail", "wait_operator")

#: rule 叶子的比较算子。
RULE_OPS = ("<=", ">=", "==", "in", "exists")

#: rule 内部节点的组合算子。
RULE_COMBINATORS = ("all", "any", "not")

#: 参数类型(params_schema 用)。
PARAM_TYPES = ("float", "int", "str", "bool")

#: 等待条件可以看的信号。温度是第一个(修复项 的公共只读口),扩展时**必须同时**
#: 给出「读不到怎么办」—— ConditionSpec.stale_after_s 是结构强制的一半。
CONDITION_SIGNALS = ("temperature_k",)

#: 条件比较算子。
CONDITION_OPS = ("<=", ">=")


def _tuple(value, what: str) -> tuple:
    """序列 → tuple。**不接受裸字符串**(``"S1"`` 静默变成五个字符是经典坑)。"""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{what} 需要一个序列,不是字符串: {value!r}")
    return tuple(value)


def _one_of(value, allowed: tuple, what: str):
    if value not in allowed:
        raise ValueError(f"{what} 必须是 {allowed} 之一,收到 {value!r}")
    return value


def _nonempty(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} 必填")
    return value


# ── 参数声明 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    """用户可填的一个参数(面板据此渲染表单)。

    ``min_value``/``max_value`` 是**包络**:超出 ⇒ 拒绝,**绝不夹紧**。夹紧会
    让一个越界的输入变成一次看起来正常的运行(针尖登记那条教训:超包络要拒绝
    不要夹紧)。校验实现见 :func:`mast.conduct.validator.check_params`。
    """

    name: str
    type: str
    unit: str = ""
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    help: str = ""
    #: 闭集参数的合法值。非空时 min/max 不参与判定。
    choices: tuple = ()

    def __post_init__(self) -> None:
        _nonempty(self.name, "ParamSpec.name")
        _one_of(self.type, PARAM_TYPES, "ParamSpec.type")
        object.__setattr__(self, "choices", _tuple(self.choices, "ParamSpec.choices"))
        if (self.min_value is not None and self.max_value is not None
                and self.min_value > self.max_value):
            raise ValueError(f"ParamSpec({self.name}) 的包络上下界反了")


# ── 判据树 ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RuleLeaf:
    """一次比较。``field`` 是证据包里的取值路径(点分)。

    求值是 Director(M1-b)的事,本模块只定义形状 —— 但**求值必须是三态**:
    字段缺席不是 False,那条路由由 ``GateSpec.evidence_missing`` 决定。
    在这里写下来,是免得 M1-b 顺手写成 ``evidence.get(field) <= value``。
    """

    field: str
    op: str
    value: Any = None

    def __post_init__(self) -> None:
        _nonempty(self.field, "RuleLeaf.field")
        _one_of(self.op, RULE_OPS, "RuleLeaf.op")
        if self.op == "in" and not isinstance(self.value, (tuple, list, frozenset, set)):
            raise ValueError(f"RuleLeaf({self.field}) 的 in 需要一个集合")
        if self.op == "in":
            object.__setattr__(self, "value", tuple(self.value))


@dataclass(frozen=True)
class RuleTree:
    """``all`` / ``any`` / ``not`` 组合。"""

    op: str
    children: tuple = ()

    def __post_init__(self) -> None:
        _one_of(self.op, RULE_COMBINATORS, "RuleTree.op")
        kids = _tuple(self.children, "RuleTree.children")
        object.__setattr__(self, "children", kids)
        if not kids:
            raise ValueError(f"RuleTree({self.op}) 至少要有一个子判据")
        if self.op == "not" and len(kids) != 1:
            raise ValueError("RuleTree(not) 只能有一个子判据")
        for k in kids:
            if not isinstance(k, (RuleLeaf, RuleTree)):
                raise TypeError(f"RuleTree 的子判据必须是 RuleLeaf/RuleTree,收到 {k!r}")


#: 判据树的两种节点。
RulePredicate = RuleLeaf | RuleTree


# ── 证据与闸门 ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvidenceSpec:
    """闸门要收哪一份证据。

    ``max_age_s`` 与 ``min_epoch`` 是**结构过滤**,不是建议:闸门收不到过期或
    跨代次的证据,所以「在自己刚炸出来的坑上判针尖」这件事在结构上做不到 ——
    而不是靠每个闸门作者自己记得去过滤。
    """

    source: str
    selector: str = ""
    max_age_s: float | None = None
    min_epoch: str = "current"
    #: **字段投影**:只把这几个字段放进证据包。空 = 整包(今天的默认行为)。
    #:
    #: 为什么需要它:一步的产出里带着 ``_progress.partial_data`` —— 逐次尝试的
    #: 原始账,以及**全部 params**(坐标、设定值、视野)。整包进 rule 闸门没关系
    #: (判据只读它点名的那个字段),但整包进 **llm 闸门的提示词**就不是没关系:
    #: 路由是闭集、模型填不了数,所以它不危险,**但多余字段会把判断拉偏**。
    #: 一道闸门问的是一个具体的问题,喂给它的应该正好是回答那个问题要的东西。
    #:
    #: ⚠️ **投影不到的字段与「这个字段没产出」是两句话。** 前者是 spec 写错了
    #: (校验器规则⑤会在 approve 时拦下来),后者是仪器/技能那一侧的事实。
    #: 运行时两者都走 ``missing``,但**措辞不同** —— 一句指向模板,一句指向机器,
    #: 而用户照着去查的地方完全不一样。
    fields: tuple = ()

    def __post_init__(self) -> None:
        _one_of(self.source, EVIDENCE_SOURCES, "EvidenceSpec.source")
        _one_of(self.min_epoch, EVIDENCE_EPOCHS, "EvidenceSpec.min_epoch")
        if self.max_age_s is not None and self.max_age_s <= 0:
            raise ValueError("EvidenceSpec.max_age_s 必须为正(或 None 表示不限)")
        object.__setattr__(self, "fields",
                           _tuple(self.fields, "EvidenceSpec.fields"))
        for name in self.fields:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("EvidenceSpec.fields 只收非空字段名")


@dataclass(frozen=True)
class GateOutcome:
    """一条路由的去向。"""

    verdict: str
    note: str = ""

    def __post_init__(self) -> None:
        _one_of(self.verdict, GATE_VERDICTS, "GateOutcome.verdict")


@dataclass(frozen=True)
class GateSpec:
    """阶段/步边界的判定。

    ``kind="llm"`` 时 ``llm_node`` **直接就是** ``llm_node.decide_route`` 吃的
    那个 node dict —— 不另造一套 schema。``decide_route`` 保证返回值必是该
    node 的某条路由(任何失败都走 escape),所以这里强制两条:

    1. node 的每条 route 都要在 ``routes`` 里有裁决 —— 少一条就是一个静默的洞:
       模型选了它,Director 拿不到 verdict,然后呢?
    2. escape 路由必须映射**保守**裁决。escape 是「判不了」的出口,把它接到
       ``pass`` 等于让「判不了」变成「通过」——本仓一天出现五次的那族错误。
    """

    gate_id: str
    kind: str
    evidence: tuple = ()
    rule: RulePredicate | None = None
    llm_node: dict | None = None
    routes: dict = field(default_factory=dict)
    unattended_escape: str = "wait_operator"
    evidence_missing: str = "wait_operator"

    def __post_init__(self) -> None:
        _nonempty(self.gate_id, "GateSpec.gate_id")
        _one_of(self.kind, GATE_KINDS, "GateSpec.kind")
        _one_of(self.unattended_escape, CONSERVATIVE_VERDICTS,
                f"GateSpec({self.gate_id}).unattended_escape")
        _one_of(self.evidence_missing, EVIDENCE_MISSING,
                f"GateSpec({self.gate_id}).evidence_missing")
        object.__setattr__(self, "evidence",
                           _tuple(self.evidence, "GateSpec.evidence"))
        for e in self.evidence:
            if not isinstance(e, EvidenceSpec):
                raise TypeError(f"GateSpec({self.gate_id}).evidence 只收 EvidenceSpec")
        object.__setattr__(self, "routes", dict(self.routes))
        if not self.routes:
            raise ValueError(f"GateSpec({self.gate_id}) 必须声明 routes")
        for name, out in self.routes.items():
            if not isinstance(out, GateOutcome):
                raise TypeError(
                    f"GateSpec({self.gate_id}).routes[{name!r}] 必须是 GateOutcome")

        if self.kind == "rule":
            if self.rule is None:
                raise ValueError(f"GateSpec({self.gate_id}) kind=rule 却没有 rule")
            if self.llm_node is not None:
                raise ValueError(f"GateSpec({self.gate_id}) kind=rule 不该带 llm_node")
        else:
            if not isinstance(self.llm_node, dict) or not self.llm_node:
                raise ValueError(f"GateSpec({self.gate_id}) kind=llm 却没有 llm_node")
            object.__setattr__(self, "llm_node", dict(self.llm_node))
            node_routes = list((self.llm_node.get("routes") or {}).keys())
            if not node_routes:
                raise ValueError(f"GateSpec({self.gate_id}) 的 llm_node 没有 routes")
            missing = [r for r in node_routes if r not in self.routes]
            if missing:
                raise ValueError(
                    f"GateSpec({self.gate_id}): llm_node 能返回 {missing} 但 routes "
                    f"里没有它们的裁决 —— 模型选中就没人接得住")
            # decide_route 的 escape 默认是最后一条路由(见其源码),这里照同一
            # 条规则推,免得两处对「escape 是谁」有两种理解。
            escape = self.llm_node.get("escape") or node_routes[-1]
            if escape not in self.routes:
                raise ValueError(
                    f"GateSpec({self.gate_id}): escape 路由 {escape!r} 无裁决")
            if self.routes[escape].verdict == "pass":
                raise ValueError(
                    f"GateSpec({self.gate_id}): escape 路由 {escape!r} 映射到 pass ——"
                    f"「判不了」不许变成「通过」,请映射到 {CONSERVATIVE_VERDICTS} 之一")


# ── 等待 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConditionSpec:
    """物理条件闸。

    ``stale_after_s`` **必填**,而且这是本设计最要紧的一个字段:读数过期意味着
    「读不到」,而「读不到」既不是「到了」也不是「没到」。没有时效限制，停止更新的温度链会让流程无限等待。
    数据不可用时必须报告原因，不能用陈旧读数作条件判决。

    ``hold_s`` 防回弹:降到位又升回去不算到位。

    ## 三个 ``*_ref``:用户填的数要真的**到达这里**

    ``ConditionSpec`` 是 frozen dataclass,写在模板里 —— 于是「等到 5 K」这个
    阈值在模板作者手上,而填参数的人以为是自己在决定。两边都不知道对方存在,
    结果是**填的人以为它生效了**,机器等的是另一个数,而且任何日志上都对不出来。

    所以数值可以声明成一条绑定:``value_ref="params.target_temperature_k"``。
    与 ``StepSpec.bindings`` 同一套命名空间、同一条纪律 —— **取不到就是失败,
    没有默认值兜底**(一个「取不到就用模板里那个数」的兜底,正是上面那句话的
    另一种写法)。给了 ref 的那一项,dataclass 上的字面值只是**占位**,
    :meth:`resolve` 之前谁都不该拿它去判定。
    """

    signal: str
    op: str
    value: float
    stale_after_s: float
    hold_s: float = 0.0
    #: 人读描述,面板直接显示(面板不该自己拼 "≤ 5.2 K" 这种话)。
    desc: str = ""
    #: ``params.<name>``;给了就以参数为准,取不到 = 步失败。
    value_ref: str = ""
    stale_after_ref: str = ""
    hold_ref: str = ""

    def __post_init__(self) -> None:
        _one_of(self.signal, CONDITION_SIGNALS, "ConditionSpec.signal")
        _one_of(self.op, CONDITION_OPS, "ConditionSpec.op")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise ValueError("ConditionSpec.value 必须是数值")
        if self.stale_after_s <= 0:
            raise ValueError("ConditionSpec.stale_after_s 必须为正 —— "
                             "没有它,「读不到」会被当成「还没到」永远等下去")
        if self.hold_s < 0:
            raise ValueError("ConditionSpec.hold_s 不能为负")
        for field_name in ("value_ref", "stale_after_ref", "hold_ref"):
            ref = getattr(self, field_name)
            if ref and not ref.startswith("params."):
                raise ValueError(
                    f"ConditionSpec.{field_name}={ref!r} —— 只能引用 "
                    f"'params.<name>'。等待条件的数只有两个合法来源:模板里的"
                    f"字面值,或用户填的参数;第三种来源都是发明数字。")

    @property
    def refs(self) -> "dict[str, str]":
        """``{目标字段: params.<name>}``,只含真的声明了的。"""
        out = {}
        for field_name, target in (("value_ref", "value"),
                                   ("stale_after_ref", "stale_after_s"),
                                   ("hold_ref", "hold_s")):
            ref = getattr(self, field_name)
            if ref:
                out[target] = ref
        return out

    def resolve(self, params: "Mapping[str, Any]") -> "dict[str, float]":
        """按 params 解出真正要用的三个数。取不到 **抛 KeyError**。

        返回 ``{value, stale_after_s, hold_s}`` —— 三个都给,调用方不必再判
        「这个是不是绑定过的」。
        """
        out = {"value": float(self.value),
               "stale_after_s": float(self.stale_after_s),
               "hold_s": float(self.hold_s)}
        for target, ref in self.refs.items():
            key = ref[len("params."):]
            if key not in params:
                raise KeyError(
                    f"等待条件的 {target} 绑定到 {ref},而 conduct 参数里没有 "
                    f"{key!r}(有 {sorted(params)}) —— 不兜底:拿模板里的占位值"
                    f"去等,等的就不是人填的那个条件了")
            val = params[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise KeyError(f"等待条件的 {target} 绑定到 {ref},而它是 "
                               f"{type(val).__name__}({val!r}),不是数值")
            out[target] = float(val)
        if out["stale_after_s"] <= 0:
            raise KeyError(f"等待条件解出的 stale_after_s={out['stale_after_s']} "
                           f"不为正 —— 「读不到」会被当成「还没到」永远等下去")
        if out["hold_s"] < 0:
            raise KeyError(f"等待条件解出的 hold_s={out['hold_s']} 为负")
        return out


@dataclass(frozen=True)
class WaitSpec:
    """一次等待。

    双闸(``kind="both"``)= 人的 ack **AND** 物理条件。两个证据回答两个问题,
    互不替代:人确认了不等于降到温,降到温不等于样品换好了。缺哪个,面板显示
    哪个。

    ``max_wait_s`` 到点只**升级通知,不放弃** —— 等人没有 fail-closed。
    HITL 那条 900 s fail-closed 正是 conduct 不能用它的原因。
    """

    kind: str
    message: str
    ack_required: bool | None = None
    condition: ConditionSpec | None = None
    renotify_every_s: float = 14400.0
    max_wait_s: float | None = None

    def __post_init__(self) -> None:
        _one_of(self.kind, WAIT_KINDS, "WaitSpec.kind")
        _nonempty(self.message, "WaitSpec.message")
        if self.ack_required is None:
            object.__setattr__(self, "ack_required", self.kind != "condition")
        if self.kind in ("condition", "both") and self.condition is None:
            raise ValueError(f"WaitSpec(kind={self.kind}) 必须带 condition")
        if self.kind == "operator" and self.condition is not None:
            raise ValueError("WaitSpec(kind=operator) 不该带 condition —— "
                             "要等条件请写 both")
        if self.kind in ("operator", "both") and not self.ack_required:
            raise ValueError(f"WaitSpec(kind={self.kind}) 的人闸不能关掉")
        if self.renotify_every_s <= 0:
            raise ValueError("WaitSpec.renotify_every_s 必须为正")


# ── 步 ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepSpec:
    """一步。

    ``bindings`` 只能命中上游步的 ``produces`` 白名单;解析失败 = 步失败,
    **没有默认值兜底**。一个「取不到就用 0」的绑定,会把一次读失败变成一次
    看起来正常的运行。

    ``timeout_s`` 是**软超时:不杀步**。卡死的 TCP 事务杀不得(force-kill 会
    永久损坏 Nanonis 端口),所以这个数只喂给 API 层的停滞告警阈值 ——
    我们不假装能解一个解不了的问题。
    """

    step_id: str
    kind: str
    skill: str = ""
    params: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    analysis_fn: str = ""
    wait: WaitSpec | None = None
    timeout_s: float | None = None
    retries: int = 0
    produces: tuple = ()
    title: str = ""
    #: 允许这步失败而不拖垮阶段(如逐点质量评估)。
    optional: bool = False
    #: 这一步**成功之后**立刻消费的闸门(2026-08-15 加)。
    #:
    #: 阶段只有首尾两个闸位,而真实流程会在中间需要判定 —— 典型是修针段:
    #: 修完没成就该停下,不该继续走完「换回样品 → 等人 → 进针 → 复验」再说不行,
    #: 那是白烧一次用户往返。以前的替代办法是把阶段拆两半,但绕道的返回时机
    #: 是按**阶段**算的,拆开之后第二半永远不跑。
    #:
    #: 求值、路由、证据的 epoch/超龄过滤与阶段闸门**完全同一套**
    #: (``entry_gate``/``exit_gate`` 只是「长在阶段边界上」的特例)。
    #: 判 ``pass`` 就继续下一步;其余裁决与阶段闸门同义(等人/绕道/阶段失败)。
    gate: GateSpec | None = None

    def __post_init__(self) -> None:
        _nonempty(self.step_id, "StepSpec.step_id")
        _one_of(self.kind, STEP_KINDS, "StepSpec.kind")
        # 参数与绑定各**拷一份**:模板里的字典若被下游改写,同一份 spec 的下一次
        # 运行就悄悄换了参数。拷贝断的是别名,不是修改本身 —— spec 仍应视为只读。
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "bindings", dict(self.bindings))
        object.__setattr__(self, "produces", _tuple(self.produces, "StepSpec.produces"))
        if self.retries < 0:
            raise ValueError(f"StepSpec({self.step_id}).retries 不能为负")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"StepSpec({self.step_id}).timeout_s 必须为正或 None")

        if self.kind in ("skill", "composite"):
            _nonempty(self.skill, f"StepSpec({self.step_id}).skill")
            if self.wait is not None or self.analysis_fn:
                raise ValueError(f"StepSpec({self.step_id}) kind={self.kind} "
                                 f"不该带 wait/analysis_fn")
        elif self.kind == "analysis":
            _nonempty(self.analysis_fn, f"StepSpec({self.step_id}).analysis_fn")
            if self.skill or self.wait is not None:
                raise ValueError(f"StepSpec({self.step_id}) kind=analysis "
                                 f"不该带 skill/wait")
        else:  # wait
            if not isinstance(self.wait, WaitSpec):
                raise ValueError(f"StepSpec({self.step_id}) kind=wait 必须带 WaitSpec")
            if self.skill or self.analysis_fn:
                raise ValueError(f"StepSpec({self.step_id}) kind=wait "
                                 f"不该带 skill/analysis_fn")
        for name, ref in self.bindings.items():
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"StepSpec({self.step_id}).bindings[{name!r}] "
                                 f"必须是一条引用路径字符串")
            if name in self.params:
                raise ValueError(
                    f"StepSpec({self.step_id}): 参数 {name!r} 同时被 params 写死和 "
                    f"bindings 绑定 —— 两个真源,运行时谁赢取决于实现细节")
        if self.gate is not None and not isinstance(self.gate, GateSpec):
            raise TypeError(f"StepSpec({self.step_id}).gate 必须是 GateSpec")

    @property
    def touches_hardware(self) -> bool:
        """这步会不会下发到仪器。``analysis``/``wait`` 不取仪器令牌。"""
        return self.kind in ("skill", "composite")


# ── 阶段 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageFailPolicy:
    max_retries: int = 0
    then: str = "wait_operator"

    def __post_init__(self) -> None:
        _one_of(self.then, STAGE_FAIL_THEN, "StageFailPolicy.then")
        if self.max_retries < 0:
            raise ValueError("StageFailPolicy.max_retries 不能为负")


@dataclass(frozen=True)
class StageSpec:
    """一个阶段。

    ``entry_actions`` 是**阶段入口重申设置**(idempotent):不信任跨阶段的仪器
    状态假设。手动接管、别的入口、上一次崩在半路 —— 任何一条都会让「上个阶段
    结束时是什么样」成为一个猜测。

    ``capabilities`` 声明本阶段允许动用的危险能力(值域同
    ``SkillMetadata.capabilities``,如 ``bias_pulse`` / ``tip_shaping``)。
    校验器据此跑 §4.3 规则③:DANGEROUS 步必须落在声明过对应 capability 的阶段。
    **不填 = 一条都不允许**,不是「随便」—— 一个不填 capability 就形同虚设的
    SAFE 模式,本仓已经有过一次。
    """

    stage_id: str
    title: str
    steps: tuple
    entry_actions: tuple = ()
    entry_gate: GateSpec | None = None
    exit_gate: GateSpec | None = None
    on_fail: StageFailPolicy = field(default_factory=StageFailPolicy)
    mandatory: bool = True
    allowed_escalations: frozenset = frozenset({"continue_retry", "wait_operator"})
    capabilities: frozenset = frozenset()
    #: **治疗段**:只有绕道进得来,正常流程一次都不该踏进去(2026-08-15 加)。
    #:
    #: 修针段是唯一的实例。在这个字段之前引擎表达不出「不进去」——
    #: ``_advance_stage`` 是无条件 ``si + 1``,而绕道又要按 ``stage_index``
    #: 找得到这个阶段,所以它必须待在 ``stages`` 里。于是只剩两个都荒谬的位置:
    #: 排最前 ⇒ 每次 conduct 开工先做一整轮换样品修针(两次人工换样品);
    #: 排最后 ⇒ 实验做完之后再修一次针。
    #:
    #: ``mandatory`` **不是**这个意思 —— 它管的是「L2 的 skip 建议是否可采纳」,
    #: 两件事别混。
    entered_only_by_detour: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.stage_id, "StageSpec.stage_id")
        _nonempty(self.title, f"StageSpec({self.stage_id}).title")
        object.__setattr__(self, "steps", _tuple(self.steps, "StageSpec.steps"))
        object.__setattr__(self, "entry_actions",
                           _tuple(self.entry_actions, "StageSpec.entry_actions"))
        object.__setattr__(self, "allowed_escalations",
                           frozenset(self.allowed_escalations))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        if not self.steps:
            raise ValueError(f"StageSpec({self.stage_id}) 一步都没有")
        for s in tuple(self.entry_actions) + tuple(self.steps):
            if not isinstance(s, StepSpec):
                raise TypeError(f"StageSpec({self.stage_id}) 的步必须是 StepSpec")
        for e in self.allowed_escalations:
            _one_of(e, ESCALATIONS, f"StageSpec({self.stage_id}).allowed_escalations")
        seen: set[str] = set()
        for s in tuple(self.entry_actions) + tuple(self.steps):
            if s.step_id in seen:
                raise ValueError(f"StageSpec({self.stage_id}) 里 step_id "
                                 f"{s.step_id!r} 重复 —— 事件与产出都按它分键")
            seen.add(s.step_id)

    @property
    def all_steps(self) -> tuple:
        """入口动作 + 主体,按执行顺序。"""
        return tuple(self.entry_actions) + tuple(self.steps)


# ── 绕道与预算 ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetourPolicy:
    """坏针绕道。

    ``max_detours_per_conduct`` 是熔断:修针 ping-pong(修完还是坏、再修)会
    烧一整夜机时和一根针。超过就停下来等人。

    ``target_stage=""`` = **这份 conduct 没有修针阶段**(例如只跑测量段的
    早期模板)。那样就必须 ``triggers`` 也为空,而且校验器会拒绝任何指向
    ``detour`` 的闸门路由或阶段失败策略 —— 一个触发得了、却无处可去的绕道,
    就是又一个「能挂不能解」。没有修针段时正确的去向是 ``wait_operator``:
    人来修针,而不是假装有个地方可去。
    """

    target_stage: str
    triggers: frozenset = frozenset(DETOUR_TRIGGERS)
    max_detours_per_conduct: int = 3
    on_return: str = "gate_recheck"

    def __post_init__(self) -> None:
        object.__setattr__(self, "triggers", frozenset(self.triggers))
        if not str(self.target_stage).strip() and self.triggers:
            raise ValueError(
                "DetourPolicy 没有 target_stage 却声明了 triggers —— "
                "触发得了却无处可去。没有修针段就把 triggers 留空,"
                "让闸门走 wait_operator")
        for t in self.triggers:
            _one_of(t, DETOUR_TRIGGERS, "DetourPolicy.triggers")
        _one_of(self.on_return, ON_RETURN, "DetourPolicy.on_return")
        if self.max_detours_per_conduct < 0:
            raise ValueError("DetourPolicy.max_detours_per_conduct 不能为负")


@dataclass(frozen=True)
class RecoveryPolicy:
    """恢复自检里那两项**只有 spec 说得出口**的东西(设计 §8 的 A2 / A3)。

    ## 不填 = 不知道,**不是通过**

    这里每一个字段的缺省都是「没声明」,而没声明的那一项在自检里报的是
    **「读不到」**,于是走 WAITING_OPERATOR。这不是保守派做法,是这张表的
    第四行:`任意 | 任一读不到 | WAITING_OPERATOR`。一个「没声明就当过了」的
    缺省会让**任何一份没接线的模板**在重启后自动续跑 —— 那正是本仓
    「producer wired, consumer absent」那族缺陷的温床。

    ## A3 为什么是**步**,不是一个回调

    设计 §6-4 逐字写着「自检项也是步,跨 tick;接触档经 ``executor.run`` 取锁」。
    针尖复验要进针、要移位、要扫一帧 —— 它必须走完整安全管道、必须受
    ``instrument_lock`` 仲裁、必须能被急停和 abort 停住。一个注入的
    ``recovery_probe(item) -> str`` 回调做不到其中任何一条,它只适合当替身。
    所以这里收的是一串 :class:`StepSpec`:与阶段里的步**同一套**执行体、
    同一套 ``bindings``、同一套 ``produces``。

    ## A3 的判据必须是三态

    ``tip_rule`` 走 ``rules.evaluate``:真 = 针可用、假 = 针坏、**判不了 = 判不了**。
    三个去向各不相同(续跑 / 停下问人 / 重试一次再停),把第三种折叠进第二种,
    就是把「这一帧没看清」说成「针坏了」—— 而那句话会把用户送去换样品。
    """

    #: A3 接触档按顺序跑的复验步。空 ⇒ A3 报「本 spec 没声明针尖复验」→ 读不到。
    tip_check: tuple = ()
    #: A3 的三态判据,读的是 ``tip_check`` 各步 ``produces`` 的**并集**。
    tip_rule: "RulePredicate | None" = None
    #: 判不了时整串重跑几次(设计 §8:「换位重试 1 次」——换不换位是复验步
    #: 自己的事,这里只决定再问一遍)。
    tip_retries: int = 1
    #: A2 的温度上限取自哪个 conduct 参数(``params.<name>``)。
    #: 空 ⇒ 从 spec 自己声明的温度等待条件推(见 :mod:`mast.conduct.recovery`);
    #: 推不出来就把「合理窗」如实报成**没检查**,而不是报成检查过了。
    temp_ceiling_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tip_check",
                           _tuple(self.tip_check, "RecoveryPolicy.tip_check"))
        for st in self.tip_check:
            if not isinstance(st, StepSpec):
                raise TypeError("RecoveryPolicy.tip_check 只收 StepSpec")
            if st.kind == "wait":
                # 自检里插一个等人步 = 把「重启后自检」变成「重启后叫人」,
                # 而叫人的决定应该由自检的**结论**做出,不是由自检的中途做出。
                raise ValueError(
                    f"RecoveryPolicy.tip_check[{st.step_id}] 不能是 wait 步 —— "
                    f"要人来看是自检的结论,不是自检的一步")
            if st.gate is not None:
                raise ValueError(
                    f"RecoveryPolicy.tip_check[{st.step_id}] 不能带 gate —— "
                    f"A3 的裁决只有 tip_rule 一个出口,两个出口会各走各的")
        if self.tip_retries < 0:
            raise ValueError("RecoveryPolicy.tip_retries 不能为负")
        if self.tip_check and self.tip_rule is None:
            # 跑了复验却没有判据 = 采了证据没人判,结论只能靠「没报错」——
            # 那正是「假成功」的形状。
            raise ValueError("RecoveryPolicy 声明了 tip_check 却没有 tip_rule:"
                             "采了证据没有判据,结论就只剩「没报错」")
        if self.tip_rule is not None and not self.tip_check:
            raise ValueError("RecoveryPolicy 声明了 tip_rule 却没有 tip_check:"
                             "判据没有证据可判")
        if self.temp_ceiling_ref and not self.temp_ceiling_ref.startswith("params."):
            raise ValueError(
                f"RecoveryPolicy.temp_ceiling_ref={self.temp_ceiling_ref!r} —— "
                f"只能引用 'params.<name>'。温度窗的数只有两个合法来源:"
                f"用户填的参数,或 spec 自己声明的等待条件;第三种都是发明数字。")

    @property
    def produces(self) -> tuple:
        """复验步 ``produces`` 的并集 —— ``tip_rule`` 能读到的全部字段名。"""
        out: list = []
        for st in self.tip_check:
            for name in st.produces:
                if name not in out:
                    out.append(name)
        return tuple(out)


@dataclass(frozen=True)
class ConductBudget:
    """预算与节奏。

    ``usd_max`` 超了**不硬停**,转 WAITING_OPERATOR + 通知 —— 一个跑了六小时的
    实验因为差几毛钱被砍掉,比超支更贵。花销以 ``api_cost_recorder`` 的实测为准,
    不自造汇率。
    """

    usd_max: float = 20.0
    llm_wakes_per_stage_max: int = 1
    tick_interval_s: float = 10.0
    wait_tick_interval_s: float = 60.0

    def __post_init__(self) -> None:
        if self.usd_max <= 0:
            raise ValueError("ConductBudget.usd_max 必须为正")
        if self.llm_wakes_per_stage_max < 0:
            raise ValueError("llm_wakes_per_stage_max 不能为负")
        if not 5.0 <= self.tick_interval_s <= 15.0:
            raise ValueError("tick_interval_s 设计区间是 5-15 s(§3-1)")
        if self.wait_tick_interval_s < self.tick_interval_s:
            raise ValueError("等待态的 tick 不该比执行态还密")


# ── 顶层 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConductSpec:
    """一份 conduct 模板。

    ``spec_version`` 改模板必 bump:恢复自检 A5 拿它对账,**不匹配就等人,
    绝不自动迁移**。一个跑到一半的 conduct 遇上换了定义的模板,续跑意味着
    前半段和后半段属于两个不同的实验。
    """

    spec_id: str
    spec_version: int
    title: str
    stages: tuple
    detour: DetourPolicy
    budgets: ConductBudget = field(default_factory=ConductBudget)
    attended_default: bool = True
    auto_resume_after_recovery: bool = False
    params_schema: tuple = ()
    #: 恢复自检里 A2/A3 需要 spec 说的那两句话(M4-a)。**默认什么都没声明**,
    #: 于是 A3 报「读不到」→ 等人 —— 一份没接线的模板重启后停下来问人,
    #: 而不是自己续跑。
    recovery: RecoveryPolicy = field(default_factory=lambda: RecoveryPolicy())
    #: 这份模板**最多**允许放到哪一档自主度(见 ``mast.conduct.autonomy``)。
    #:
    #: 生效值 = min(全局设置, 这个字段) —— **取更严的那个**。模板作者比设置页
    #: 更懂这份流程能放多松:一份要在 4 K 下扎针的模板可以把自己钉死在
    #: ``attended``,设置页开到 ``autonomous`` 也拉不上去。
    #:
    #: 默认 ``autonomous``(= 不额外限制),因为限制该由**说得出理由的人**显式
    #: 写下来。默认钉死在 attended 看起来更安全,实际后果是每份模板都带着一句
    #: 没人写过理由的限制,而没人写过理由的限制迟早会被整批删掉。
    max_autonomy: str = "autonomous"

    def __post_init__(self) -> None:
        _nonempty(self.spec_id, "ConductSpec.spec_id")
        _nonempty(self.title, "ConductSpec.title")
        if not isinstance(self.spec_version, int) or isinstance(self.spec_version, bool):
            raise ValueError("ConductSpec.spec_version 必须是整数")
        if self.spec_version < 1:
            raise ValueError("ConductSpec.spec_version 从 1 开始")
        object.__setattr__(self, "stages", _tuple(self.stages, "ConductSpec.stages"))
        object.__setattr__(self, "params_schema",
                           _tuple(self.params_schema, "ConductSpec.params_schema"))
        if not self.stages:
            raise ValueError(f"ConductSpec({self.spec_id}) 一个阶段都没有")
        for st in self.stages:
            if not isinstance(st, StageSpec):
                raise TypeError("ConductSpec.stages 只收 StageSpec")
        ids = [st.stage_id for st in self.stages]
        if len(set(ids)) != len(ids):
            raise ValueError(f"ConductSpec({self.spec_id}) 的 stage_id 有重复: {ids}")
        for p in self.params_schema:
            if not isinstance(p, ParamSpec):
                raise TypeError("ConductSpec.params_schema 只收 ParamSpec")
        pnames = [p.name for p in self.params_schema]
        if len(set(pnames)) != len(pnames):
            raise ValueError(f"ConductSpec({self.spec_id}) 的参数名有重复: {pnames}")
        if not isinstance(self.recovery, RecoveryPolicy):
            raise TypeError("ConductSpec.recovery 必须是 RecoveryPolicy")
        # 闭集当场收口:一个拼错的档位名如果活到运行期,会被 normalise 悄悄
        # 当成最严档 —— 那是安全的方向,但模板作者会以为自己设的是别的东西。
        from mast.conduct.autonomy import AUTONOMY_LEVELS

        if self.max_autonomy not in AUTONOMY_LEVELS:
            raise ValueError(
                f"ConductSpec({self.spec_id}).max_autonomy={self.max_autonomy!r} "
                f"不在闭集里: {AUTONOMY_LEVELS}")
        rec_ids = [st.step_id for st in self.recovery.tip_check]
        if len(set(rec_ids)) != len(rec_ids):
            raise ValueError(f"ConductSpec({self.spec_id}) 的恢复复验步 step_id "
                             f"有重复: {rec_ids}")
        # 复验步与流程步**不许同名**:两者的产出都进同一条 ``step_finished``
        # 审计流(``director._produced`` 按 step_id 建索引),重名会让一次自检
        # 的读数盖掉一个流程步的产出,而闸门照样按 selector 去读它。
        flow_ids = {st.step_id for stage in self.stages for st in stage.all_steps}
        clash = sorted(set(rec_ids) & flow_ids)
        if clash:
            raise ValueError(
                f"ConductSpec({self.spec_id}): 恢复复验步与流程步同名 {clash} —— "
                f"产出走同一条审计流,重名会让自检的读数盖掉流程步的产出")

    def stage(self, stage_id: str) -> StageSpec | None:
        for st in self.stages:
            if st.stage_id == stage_id:
                return st
        return None

    def stage_index(self, stage_id: str) -> int | None:
        for i, st in enumerate(self.stages):
            if st.stage_id == stage_id:
                return i
        return None

    @property
    def total_steps(self) -> int:
        return sum(len(st.all_steps) for st in self.stages)


__all__ = [
    "AT_ENTRY_GATE",
    "STEP_KINDS", "GATE_KINDS", "GATE_VERDICTS", "CONSERVATIVE_VERDICTS",
    "WAIT_KINDS", "STAGE_FAIL_THEN", "ON_RETURN", "DETOUR_TRIGGERS",
    "ESCALATIONS", "EVIDENCE_SOURCES", "EVIDENCE_EPOCHS", "EVIDENCE_MISSING",
    "RULE_OPS", "RULE_COMBINATORS", "PARAM_TYPES", "CONDITION_SIGNALS",
    "CONDITION_OPS",
    "ParamSpec", "RuleLeaf", "RuleTree", "RulePredicate", "EvidenceSpec",
    "GateOutcome", "GateSpec", "ConditionSpec", "WaitSpec", "StepSpec",
    "StageFailPolicy", "StageSpec", "DetourPolicy", "RecoveryPolicy",
    "ConductBudget",
    "ConductSpec",
]
